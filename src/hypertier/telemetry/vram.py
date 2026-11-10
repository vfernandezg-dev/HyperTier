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

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

from hypertier.telemetry._common import format_bytes, require_cuda, resolve_device


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
