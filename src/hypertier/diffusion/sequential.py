# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the stages of a pipeline one after another, each one loaded only while it runs.

A video pipeline is a text encoder, a transformer and a VAE: three models that never have to
be in VRAM, or in page-locked RAM, at the same time. :class:`SequentialPipeline` is the cycle
*load, run, release* around each stage, and the VRAM accounting that goes with it:

* The budget of a stage is the VRAM that is really free when it starts: read from the driver
  after the previous stage was released and the allocator's cache emptied. A caller's
  ``vram_budget`` can only lower it.
* What a stage is expected to need, a peak measured beforehand, is checked against that
  budget less a margin **before** anything is loaded, so a configuration that does not fit
  fails at once, with what would fit, instead of running out of memory minutes later.
* Afterwards the time and the peaks of the stage are on record.

All VRAM figures are what the process holds as the driver counts it: that is what
``nvidia-smi`` shows and what a budget of "3.4 GB" means. A process holds more than PyTorch's
allocator: its CUDA context, which **grows while a stage runs**. CUDA loads every kernel the
first time it is used, so each new kind of operation, dtype and algorithm adds a little, and
the first stage adds most. Measured on the reference laptop (driver 595), in MiB outside the
allocator at the peak of each stage of one generation::

                                               start   text   transformer   VAE
    a process as a user runs it                   77    123           155   141
    as bench_video runs its workers               77    153           209   195

``bench_video`` sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` and warms the GPU up with a matmul in
float16: cuBLAS then keeps a larger workspace of its own, and more kernels are loaded. The
transformer stage holds 16 to 18 MiB more while it runs than after it: page-locked RAM costs
2 MiB of VRAM per GiB for as long as it is registered with the driver. A small matmul, a
small convolution and a small attention before the first stage load 20 MiB of all that, and
the text stage still adds 28 MiB after them: the context cannot be warmed up to its final
size cheaply, so its growth is budgeted for instead (:data:`CUDA_CONTEXT`).

Importing this module does not import PyTorch.
"""

from __future__ import annotations

import contextlib
import gc
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from hypertier.telemetry._common import MiB, format_bytes, require_cuda, resolve_device
from hypertier.telemetry.vram import process_vram

VRAM_MARGIN = 300 * MiB
"""VRAM left free above the expected peak of a stage. The peaks are measurements on one
machine, and a desktop may take some VRAM at any time."""

CUDA_CONTEXT = 210 * MiB
"""VRAM a process holds outside PyTorch's allocator, at the most: what the peak of a stage is
counted with. The most measured on the reference laptop is 209 MiB (see the module
docstring). What a process holds now is read from the driver; this number is for what cannot
be read: how far the context will have grown at the peak of a stage that has not run yet,
and what the process holds when the driver does not say (WSL2)."""


@dataclass
class StageReport:
    """What one stage was allowed, what it was expected to take and what it took.

    Attributes:
        name: the stage.
        budget_bytes: VRAM the process could hold while the stage ran.
        expected_peak_bytes: the measured peak the stage was admitted with, or None when
            nothing is known about it.
        free_before_bytes: the driver's free VRAM right before the stage.
        load_seconds: of ``seconds``, the part spent loading weights (set by the stage).
        seconds: wall time of the stage, loading and releasing included.
        peak_allocated_bytes: most VRAM in live tensors.
        peak_reserved_bytes: most VRAM held by PyTorch's allocator.
        context_bytes: VRAM the process held outside the allocator, read while the stage was
            still loaded: what the driver attributes to the process, less what the allocator
            held at that moment. :data:`CUDA_CONTEXT` when the driver does not say.
        context_measured: whether ``context_bytes`` was read from the driver.
        peak_process_bytes: ``peak_reserved_bytes`` plus ``context_bytes``: the most VRAM the
            process held, as the driver counts it.
        details: whatever else the stage wants on record.
    """

    name: str
    budget_bytes: int = 0
    expected_peak_bytes: int | None = None
    free_before_bytes: int = 0
    load_seconds: float = 0.0
    seconds: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    context_bytes: int | None = None
    context_measured: bool = False
    peak_process_bytes: int = 0
    details: dict[str, Any] = field(default_factory=dict)


class SequentialPipeline:
    """Stages run one at a time on one GPU, each within the VRAM that is free when it starts.

    A subclass runs each stage inside :meth:`stage`, loading the model in the block and
    releasing it before the block ends.
    """

    def __init__(
        self,
        device: Any = "cuda",
        *,
        vram_budget: int | None = None,
        margin: int = VRAM_MARGIN,
    ) -> None:
        """
        Args:
            device: the CUDA device every stage computes on.
            vram_budget: VRAM the process may hold, in bytes. None: whatever is free when a
                stage starts. A budget above that is lowered to it.
            margin: bytes to leave free above the expected peak of a stage.
        """
        if vram_budget is not None and vram_budget <= 0:
            raise ValueError(f"a VRAM budget of {vram_budget} bytes leaves nothing to run in")
        if margin < 0:
            raise ValueError(f"the margin cannot be negative: {margin}")
        self._torch = require_cuda()
        self.device = resolve_device(device)
        self.vram_budget = vram_budget
        self.margin = margin
        self.reports: list[StageReport] = []

    def free_vram(self) -> int:
        """The driver's free VRAM now, once garbage is collected and the cache emptied."""
        gc.collect()
        self._torch.cuda.empty_cache()
        return int(self._torch.cuda.mem_get_info(self.device)[0])

    def context(self) -> int | None:
        """VRAM this process holds outside PyTorch's allocator now; None if it is not known.

        What the driver attributes to the process
        (:func:`~hypertier.telemetry.vram.process_vram`), less what the allocator holds.
        """
        # A process is in the driver's list once it has a context on the device: asking for
        # the free VRAM creates it, if nothing has yet.
        self._torch.cuda.mem_get_info(self.device)
        held = process_vram(self.device)
        if held is None:
            return None
        return max(held - int(self._torch.cuda.memory_reserved(self.device)), 0)

    def available(self) -> int:
        """VRAM the process may hold from now on, its CUDA context included.

        What is free plus what the process already holds, or the caller's budget if that is
        less. Other processes may take VRAM later: that is what the margin is for.

        Expected peaks are counted with :data:`CUDA_CONTEXT`, as large as a context was seen
        to grow. A context that is smaller now will grow during the stage, out of the VRAM
        that is free: so the context counts as already held only for what it holds now, and
        never for more than the peaks count. Before the first stage that is 70 MiB of 210.
        Whatever it holds above :data:`CUDA_CONTEXT`, the peaks do not know of: that comes
        off the budget.
        """
        free = self.free_vram()
        context = self.context()
        if context is None:  # the driver does not say: as much as the peaks count
            context = CUDA_CONTEXT
        counted, beyond = min(context, CUDA_CONTEXT), max(context - CUDA_CONTEXT, 0)
        mine = free + int(self._torch.cuda.memory_reserved(self.device)) + counted
        return mine if self.vram_budget is None else min(mine, self.vram_budget - beyond)

    def read_context(self, report: StageReport) -> None:
        """Put on ``report`` what the process holds outside the allocator at this moment.

        For a stage to call while its model is still loaded: page-locked RAM costs VRAM for
        as long as it is registered, and that is gone once the stage has released it. A
        stage that does not call it is read when it ends.
        """
        context = self.context()
        report.context_measured = context is not None
        report.context_bytes = CUDA_CONTEXT if context is None else context

    def fits(self, expected_peak: int, available: int) -> bool:
        return expected_peak + self.margin <= available

    @contextlib.contextmanager
    def stage(
        self, name: str, expected_peak: int | None = None, advice: str = ""
    ) -> Iterator[StageReport]:
        """Run one stage: admit it, time it, and put its peaks on record.

        Args:
            name: how the report calls the stage.
            expected_peak: VRAM the process is expected to hold at the peak of the stage, in
                bytes, from measurements. None admits the stage unchecked.
            advice: what to try instead, added to the error when the stage does not fit.

        The block loads the stage's model, runs it and releases it. The report it is handed
        is complete once the block has ended; it is also appended to :attr:`reports`.
        """
        torch = self._torch
        available = self.available()
        report = StageReport(
            name,
            budget_bytes=available,
            expected_peak_bytes=expected_peak,
            free_before_bytes=int(torch.cuda.mem_get_info(self.device)[0]),
        )
        if expected_peak is not None and not self.fits(expected_peak, available):
            raise ValueError(self.shortfall(name, expected_peak, available, advice))
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        try:
            yield report
        finally:
            torch.cuda.synchronize(self.device)
            report.seconds = time.perf_counter() - started
            report.peak_allocated_bytes = int(torch.cuda.max_memory_allocated(self.device))
            report.peak_reserved_bytes = int(torch.cuda.max_memory_reserved(self.device))
            if report.context_bytes is None:
                self.read_context(report)
            report.peak_process_bytes = report.peak_reserved_bytes + report.context_bytes
            self.reports.append(report)

    def shortfall(self, name: str, expected_peak: int, available: int, advice: str = "") -> str:
        """The message of a stage that does not fit."""
        text = (
            f"the {name} stage peaks at {format_bytes(expected_peak)} of VRAM, and with "
            f"{format_bytes(self.margin)} to spare that does not fit in the "
            f"{format_bytes(available)} this process can hold"
        )
        return f"{text}. {advice}" if advice else text


__all__ = ["CUDA_CONTEXT", "VRAM_MARGIN", "SequentialPipeline", "StageReport"]
