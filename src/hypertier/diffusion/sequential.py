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

All VRAM figures are what the process holds as the driver counts it, its CUDA context
included: that is what ``nvidia-smi`` shows and what a budget of "3.4 GB" means.

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

VRAM_MARGIN = 300 * MiB
"""VRAM left free above the expected peak of a stage. The peaks are measurements on one
machine, and a desktop may take some VRAM at any time."""

CUDA_CONTEXT = 143 * MiB
"""What a process holds in VRAM outside PyTorch's allocator: its CUDA context and the
libraries loaded into it. 111 to 143 MiB measured on the reference laptop (driver 595),
depending on the stage; the largest is taken, so that peaks are not understated."""


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
        peak_process_bytes: ``peak_reserved_bytes`` plus :data:`CUDA_CONTEXT`.
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

    def available(self) -> int:
        """VRAM the process may hold from now on, its CUDA context included.

        What is free plus what the process already holds, or the caller's budget if that is
        less. Other processes may take VRAM later: that is what the margin is for.
        """
        free = self.free_vram()
        held = int(self._torch.cuda.memory_reserved(self.device)) + CUDA_CONTEXT
        mine = free + held
        return mine if self.vram_budget is None else min(mine, self.vram_budget)

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
            report.peak_process_bytes = report.peak_reserved_bytes + CUDA_CONTEXT
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
