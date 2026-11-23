# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""VRAM accounting for a block of GPU work.

Two numbers matter, and they differ. PyTorch's allocator peak (``max_memory_allocated``) is
what *our* tensors needed. The driver's free memory (``mem_get_info``) is what is really left
after the CUDA context, other processes and, on the reference laptop, the Windows desktop,
which runs on the same 4 GB GPU. The planner budgets against the second; kernels and
streaming slots are judged by the first.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

from hypertier.telemetry._common import MiB, format_bytes, require_cuda, resolve_device


@dataclass
class VramStats:
    """VRAM figures for one tracked block, in bytes (final once the block has exited)."""

    device: int
    total_bytes: int = 0
    free_before: int = 0
    free_after: int = 0
    allocated_before: int = 0
    allocated_after: int = 0
    peak_allocated: int = 0
    peak_reserved: int = 0
    samples: list[tuple[str, int]] = field(default_factory=list)  # (label, free bytes)

    @property
    def peak_delta(self) -> int:
        """Peak memory allocated inside the block on top of what was live at entry."""
        return max(0, self.peak_allocated - self.allocated_before)

    @property
    def min_free_observed(self) -> int:
        """Lowest driver-reported free VRAM among entry, ``sample()`` calls and exit."""
        return min([self.free_before, self.free_after, *(free for _, free in self.samples)])

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["peak_delta"] = self.peak_delta
        data["min_free_observed"] = self.min_free_observed
        return data

    def summary(self) -> str:
        return (
            f"peak +{format_bytes(self.peak_delta)} "
            f"(allocator peak {format_bytes(self.peak_allocated)}, "
            f"reserved {format_bytes(self.peak_reserved)}); "
            f"free {format_bytes(self.free_before)} -> {format_bytes(self.free_after)} "
            f"of {format_bytes(self.total_bytes)}, min {format_bytes(self.min_free_observed)}"
        )


class VramTracker:
    """Handle yielded by :func:`track_vram`."""

    def __init__(self, stats: VramStats, device: Any) -> None:
        self.stats = stats
        self._device = device

    def sample(self, label: str) -> int:
        """Record the driver's free VRAM now (e.g. after loading weights); return it."""
        torch = require_cuda()
        free, _ = torch.cuda.mem_get_info(self._device)
        self.stats.samples.append((label, free))
        return free


@contextmanager
def track_vram(
    device: Any = None, *, reset_peak: bool = True, synchronize: bool = True
) -> Iterator[VramTracker]:
    """Record peak allocated VRAM and real free VRAM around a block of work.

    ``reset_peak`` clears PyTorch's peak counter on entry so the peak belongs to this block;
    the counter is global per device, so nested trackers should pass ``reset_peak=False``.
    ``synchronize`` waits for queued kernels so their memory is counted. The stats are
    filled in even if the block raises, which helps when diagnosing an out-of-memory error.
    """
    torch = require_cuda()
    dev = resolve_device(device)
    if synchronize:
        torch.cuda.synchronize(dev)
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(dev)
    free, total = torch.cuda.mem_get_info(dev)
    stats = VramStats(
        device=dev.index,
        total_bytes=total,
        free_before=free,
        free_after=free,
        allocated_before=torch.cuda.memory_allocated(dev),
    )
    failed = False
    try:
        yield VramTracker(stats, dev)
    except BaseException:
        failed = True
        raise
    finally:
        try:
            if synchronize:
                torch.cuda.synchronize(dev)
            stats.free_after, _ = torch.cuda.mem_get_info(dev)
            stats.allocated_after = torch.cuda.memory_allocated(dev)
            stats.peak_allocated = torch.cuda.max_memory_allocated(dev)
            stats.peak_reserved = torch.cuda.max_memory_reserved(dev)
        except RuntimeError:
            if not failed:  # a broken CUDA context must not hide the block's own error
                raise


def process_vram(device: Any = None) -> int | None:
    """VRAM the driver attributes to this process on ``device``, in bytes; None if unknown.

    Everything the process holds: PyTorch's allocator, and what no counter of PyTorch shows,
    which is the CUDA context, the kernels loaded into it as they are first used, and the
    driver's bookkeeping for page-locked RAM.

    Read through NVML when its Python bindings are installed (``nvidia-ml-py``, the module
    ``pynvml``): a call into the driver's own library, 0.07 ms on the reference laptop once
    the library is loaded, which takes 8 ms the first time. Without them, or when NVML
    cannot be asked, through ``nvidia-smi``: no dependency, and a process to start every
    time (24 ms). Both read the same list and give the same figure.

    None when neither can be asked, when the driver keeps no figure per process (WSL2), and
    when this process is not in its list, as in a container with process IDs of its own.
    """
    torch = require_cuda()
    dev = resolve_device(device)
    uuid = getattr(torch.cuda.get_device_properties(dev), "uuid", None)
    gpu_uuid = None if uuid is None else f"GPU-{uuid}"
    held = _process_vram_nvml(os.getpid(), gpu_uuid)
    if held is _NOT_ASKED:
        held = _process_vram_smi(os.getpid(), gpu_uuid)
    return held


_NOT_ASKED: Any = object()
"""What :func:`_process_vram_nvml` returns when NVML could not be asked at all, which is not
the same as NVML answering that it does not know."""

_nvml_loaded: list[Any] = []
"""Empty until NVML was tried; then the module ``pynvml`` with NVML initialised, or None."""


def _nvml() -> Any:
    """The module ``pynvml`` with NVML initialised, or None when it cannot be used.

    Tried once and kept for the life of the process: the bindings are not installed, or the
    library of the driver is not found, for good. NVML is shut down when the interpreter
    exits.
    """
    if not _nvml_loaded:
        try:
            import pynvml
        except ImportError:
            _nvml_loaded.append(None)
            return None
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            _nvml_loaded.append(None)
            return None

        def shutdown() -> None:
            with contextlib.suppress(pynvml.NVMLError):
                pynvml.nvmlShutdown()

        atexit.register(shutdown)
        _nvml_loaded.append(pynvml)
    return _nvml_loaded[0]


def _process_vram_nvml(pid: int, gpu_uuid: str | None) -> int | None:
    """Bytes of ``pid`` on the GPU ``gpu_uuid`` as NVML lists it, or None if it does not say.

    :data:`_NOT_ASKED` when NVML cannot be used or a call fails: the caller then asks
    ``nvidia-smi``. Without a UUID the process has to be listed on one GPU only, as in
    :func:`parse_compute_apps`.
    """
    pynvml = _nvml()
    if pynvml is None:
        return _NOT_ASKED
    try:
        if gpu_uuid is not None:
            handles = [pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)]
        else:
            count = pynvml.nvmlDeviceGetCount()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(count)]
        found = [
            process.usedGpuMemory
            for handle in handles
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if process.pid == pid
        ]
    except (pynvml.NVMLError, TypeError):  # TypeError: bindings from before strings were taken
        return _NOT_ASKED
    if len(found) != 1 or found[0] is None:  # None: the driver does not say (WSL2)
        return None
    return int(found[0])


def _process_vram_smi(pid: int, gpu_uuid: str | None) -> int | None:
    """The same from the listing of ``nvidia-smi``; None also when it is missing or fails."""
    command = ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
               "--format=csv,noheader,nounits"]  # fmt: skip
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return parse_compute_apps(done.stdout, pid, gpu_uuid)


def parse_compute_apps(listing: str, pid: int, gpu_uuid: str | None) -> int | None:
    """Bytes of ``pid`` on the GPU ``gpu_uuid`` in a ``pid, gpu_uuid, MiB`` listing, or None.

    Without a UUID the process has to be listed once: on two GPUs it is not known which.
    """
    found = []
    for line in listing.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or fields[0] != str(pid):
            continue
        if gpu_uuid is not None and fields[1] != gpu_uuid:
            continue
        if not fields[2].isdigit():  # "[N/A]": the driver does not say
            return None
        found.append(int(fields[2]) * MiB)
    return found[0] if len(found) == 1 else None
