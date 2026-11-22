# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Page-locked host memory reserved once and reused: the RAM tier that feeds VRAM.

Only page-locked ("pinned") memory can be copied to the GPU asynchronously, which is what lets
a transfer hide behind compute. Locking pages is slow, though: about 1-2 GB/s on the reference
laptop and 0.55 GB/s under WSL2, against 6.7 GB/s for the copy itself. A pool therefore pins
its memory once and hands the same pages out again, for every pass and, after
:meth:`PinnedPool.reset`, for the next model.

PyTorch is imported lazily, so importing this module never imports it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hypertier.telemetry._common import GiB, require_cuda
from hypertier.telemetry.bandwidth import release_pinned_cache

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self

ALIGN = 256
"""Alignment in bytes of every allocation, and of every parameter inside a streamed block.

It is a multiple of every dtype's element size, which viewing raw bytes as a typed tensor
requires, and it keeps weights on the same boundary in RAM and in every VRAM slot, so a kernel
that picks its code path from the pointer alignment picks the same one wherever a block lands.
"""


def align_up(nbytes: int, align: int = ALIGN) -> int:
    """Smallest multiple of ``align`` that is >= ``nbytes``."""
    return -(-nbytes // align) * align


def pow2_ceil(n: int) -> int:
    """Smallest power of two that is >= ``n`` (1 for ``n`` <= 1)."""
    return 1 << (n - 1).bit_length() if n > 1 else 1


@dataclass
class _Segment:
    tensor: Any  # pinned uint8 tensor, one allocation
    size: int
    used: int = 0


class PinnedPool:
    """Page-locked RAM handed out in aligned, contiguous pieces and reused after a reset.

    ``capacity_bytes`` is pinned up front (0: nothing; the pool grows on first use). When no
    segment has room for a request the pool grows instead of failing, by one segment of
    ``segment_bytes`` or, for a request larger than that, by one segment big enough to hold
    it: every piece stays contiguous, so a block of weights is one copy to VRAM. Whatever was
    pinned stays pinned until :meth:`close`, so the cost is paid once per high-water mark.

    Segments are powers of two because PyTorch's pinned allocator rounds every request up to
    one (measured: asking for 300 MiB locks 512 MiB). A segment sized like that rounding wastes
    nothing by itself; the waste is the tail of each segment that is too short for the next
    piece, visible as ``capacity_bytes - used_bytes``.

    Not thread-safe.
    """

    def __init__(self, capacity_bytes: int = 0, *, segment_bytes: int = GiB) -> None:
        if capacity_bytes < 0:
            raise ValueError(f"capacity_bytes must be >= 0, got {capacity_bytes}")
        if segment_bytes < ALIGN or segment_bytes & (segment_bytes - 1):
            raise ValueError(
                f"segment_bytes must be a power of two >= {ALIGN}, got {segment_bytes}"
            )
        self._torch = require_cuda()
        self.segment_bytes = segment_bytes
        self.pin_seconds = 0.0
        """Time spent locking pages so far: what reusing the pool saves the next model."""
        self._segments: list[_Segment] = []
        self._leases = 0
        self._closed = False
        remaining = capacity_bytes
        while remaining > 0:
            size = min(segment_bytes, pow2_ceil(remaining))
            self._grow(size)
            remaining -= size

    @property
    def capacity_bytes(self) -> int:
        """Bytes currently page-locked by the pool."""
        return sum(seg.size for seg in self._segments)

    @property
    def used_bytes(self) -> int:
        """Bytes handed out since the last reset, alignment padding included."""
        return sum(seg.used for seg in self._segments)

    def alloc(self, nbytes: int) -> torch.Tensor:
        """Contiguous pinned ``uint8`` view of ``nbytes``, starting on an ``ALIGN`` boundary.

        The memory is not zeroed. It stays valid until the pool is reset or closed.
        """
        if self._closed:
            raise RuntimeError("the pool is closed")
        if nbytes < 0:
            raise ValueError(f"nbytes must be >= 0, got {nbytes}")
        size = align_up(nbytes)
        for seg in self._segments:
            if seg.size - seg.used >= size:
                break
        else:
            if size == 0:
                return self._torch.empty(0, dtype=self._torch.uint8)
            seg = self._grow(max(self.segment_bytes, pow2_ceil(size)))
        view = seg.tensor[seg.used : seg.used + nbytes]
        seg.used += size
        return view

    def reset(self) -> None:
        """Make the whole pool available again without pinning anything.

        Every view handed out before becomes invalid: the same pages will hold the next
        model. Waits for queued GPU work first, because a copy still in flight would read
        pages that are about to be overwritten. Refuses while a streamer is attached.
        """
        self._check_idle("reset")
        self._drain()
        for seg in self._segments:
            seg.used = 0

    def close(self) -> None:
        """Release the pool's pages (idempotent).

        Pages still referenced by a tensor stay locked until that tensor dies; PyTorch then
        caches them, and ``hypertier.telemetry.bandwidth.release_pinned_cache()`` returns
        them to the OS.
        """
        if self._closed:
            return
        self._check_idle("close")
        self._drain()
        self._closed = True
        self._segments.clear()
        release_pinned_cache()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -----------------------------------------------------------------------------------------

    def _grow(self, size: int) -> _Segment:
        torch = self._torch
        t0 = time.perf_counter()
        # Outside inference mode: views of an inference tensor could not back a parameter.
        with torch.inference_mode(False):
            tensor = _pin(torch, size)
        self.pin_seconds += time.perf_counter() - t0
        if tensor.data_ptr() % ALIGN:
            raise RuntimeError(f"pinned segment is not {ALIGN}-byte aligned")
        seg = _Segment(tensor, size)
        self._segments.append(seg)
        return seg

    def _drain(self) -> None:
        """Wait for every CUDA stream, the streamers' copy streams included."""
        torch = self._torch
        if torch.cuda.is_initialized():
            for index in range(torch.cuda.device_count()):
                torch.cuda.synchronize(index)

    def _check_idle(self, action: str) -> None:
        if self._leases:
            raise RuntimeError(
                f"cannot {action} the pool: {self._leases} streamer(s) still attached; "
                "close them first"
            )

    def _attach(self) -> None:
        if self._closed:
            raise RuntimeError("the pool is closed")
        self._leases += 1

    def _detach(self) -> None:
        self._leases -= 1


def _pin(torch: Any, nbytes: int) -> Any:
    """The one place page-locked memory comes from; ``nbytes`` is a power of two."""
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
