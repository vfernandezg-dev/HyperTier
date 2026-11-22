# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Page-locked host memory reserved once and reused: the RAM tier that feeds VRAM.

Only page-locked ("pinned") memory can be copied to the GPU asynchronously, which is what lets
a transfer hide behind compute. Locking pages is slow, though: about 2 GB/s on the reference
laptop, against 6.7 GB/s for the copy itself. A pool therefore pins its memory once and hands
the same pages out again, for every pass and, after :meth:`PinnedPool.reset`, for the next
model.

The pool locks exactly the bytes it needs: anonymous ``mmap`` memory registered with CUDA
(``cudaHostRegister``). PyTorch's own pinned allocator rounds every request up to a power of
two (asking for 300 MiB locks 512 MiB), which for a model of many gigabytes is RAM the machine
does not have. It remains the fallback where registering is not possible.

PyTorch is imported lazily, so importing this module never imports it.
"""

from __future__ import annotations

import mmap
import time
import warnings
import weakref
from collections.abc import Callable
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


@dataclass
class _Segment:
    tensor: Any  # page-locked uint8 tensor, one allocation
    size: int
    unpin: Callable[[], None] | None  # unlocks the pages; None when PyTorch owns them
    used: int = 0


class PinnedPool:
    """Page-locked RAM handed out in aligned, contiguous pieces and reused after a reset.

    ``capacity_bytes`` is pinned up front (0: nothing; the pool grows on first use). When no
    segment has room for a request the pool grows instead of failing, by one segment of
    ``segment_bytes`` or, for a request larger than that, by one segment of exactly the
    request: every piece stays contiguous, so a block of weights is one copy to VRAM. Sizes
    are exact up to the page size. Whatever was pinned stays pinned until :meth:`close`, so
    the cost is paid once per high-water mark; ``capacity_bytes - used_bytes`` is what the
    current contents leave unused.

    Closing the pool unlocks its pages but never frees memory that a tensor still uses: views
    handed out before stay valid, as ordinary pageable memory, and each segment is returned to
    the OS when its last tensor dies. A pool that is garbage-collected without being closed
    unlocks its pages too, except at interpreter exit, where the OS does it.

    Not thread-safe.
    """

    def __init__(self, capacity_bytes: int = 0, *, segment_bytes: int = GiB) -> None:
        if capacity_bytes < 0:
            raise ValueError(f"capacity_bytes must be >= 0, got {capacity_bytes}")
        if segment_bytes < 1:
            raise ValueError(f"segment_bytes must be >= 1, got {segment_bytes}")
        self._torch = require_cuda()
        self.segment_bytes = segment_bytes
        self.pin_seconds = 0.0
        """Time spent locking pages so far: what reusing the pool saves the next model."""
        self._segments: list[_Segment] = []
        self._leases = 0
        self._closed = False
        # A plain function, not a method: the finalizer must not keep the pool alive.
        self._finalizer = weakref.finalize(self, _release, self._torch, self._segments)
        # Not at interpreter exit: the CUDA context may already be going away, so waiting for
        # its streams could fail there, and the OS unlocks and frees the pages anyway.
        self._finalizer.atexit = False
        remaining = capacity_bytes
        while remaining > 0:
            size = min(segment_bytes, remaining)
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

        The memory is not zeroed. Its contents are yours until the pool is reset.
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
            seg = self._grow(max(self.segment_bytes, size))
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
        _drain(self._torch)
        for seg in self._segments:
            seg.used = 0

    def close(self) -> None:
        """Unlock the pool's pages, after waiting for queued GPU work (idempotent).

        Views handed out before remain valid tensors in pageable memory.
        """
        if self._closed:
            return
        self._check_idle("close")
        self._closed = True
        self._finalizer()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -----------------------------------------------------------------------------------------

    def _grow(self, size: int) -> _Segment:
        torch = self._torch
        size = align_up(size, mmap.PAGESIZE)
        t0 = time.perf_counter()
        # Outside inference mode: views of an inference tensor could not back a parameter.
        with torch.inference_mode(False):
            tensor, unpin = _pin(torch, size)
        self.pin_seconds += time.perf_counter() - t0
        if tensor.data_ptr() % ALIGN:
            raise RuntimeError(f"pinned segment is not {ALIGN}-byte aligned")
        seg = _Segment(tensor, size, unpin)
        self._segments.append(seg)
        return seg

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


def _pin(torch: Any, nbytes: int) -> tuple[Any, Callable[[], None] | None]:
    """Page-lock exactly ``nbytes`` (a multiple of the page size); return (tensor, unpin).

    The one place page-locked memory comes from, and the only code that touches
    ``torch.cuda.cudart()``, an undocumented binding of the CUDA runtime. Anonymous shared
    ``mmap`` memory is registered with ``cudaHostRegister``. Shared rather than private, so a
    ``fork`` (data-loader workers) does not copy every locked page into the child. ``unpin``
    unregisters the pages and must run before the memory is freed, or they stay locked until
    the process exits; it never frees anything itself, because the mapping belongs to the
    tensor and is released with its last view.

    If registering fails for any reason (a PyTorch without the binding, a platform that
    refuses it), a warning says so and PyTorch's pinned allocator is used instead: correct,
    but it rounds the size up to a power of two. ``unpin`` is then None.
    """
    try:
        cudart = torch.cuda.cudart()
        tensor = torch.frombuffer(mmap.mmap(-1, nbytes), dtype=torch.uint8)
        address = tensor.data_ptr()
        status = cudart.cudaHostRegister(address, nbytes, 0)
        if int(status) != 0:
            raise RuntimeError(f"cudaHostRegister returned {status!r}")
        if not tensor.is_pinned():
            cudart.cudaHostUnregister(address)
            raise RuntimeError("registered memory is not reported as pinned")
    except Exception as exc:  # noqa: BLE001 - whatever went wrong, the fallback still works
        warnings.warn(
            f"could not page-lock {nbytes} bytes with cudaHostRegister "
            f"({type(exc).__name__}: {exc}); falling back to PyTorch's pinned allocator, which "
            "rounds every allocation up to a power of two and can use up to twice the RAM",
            stacklevel=2,
        )
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True), None

    def unpin() -> None:
        status = cudart.cudaHostUnregister(address)
        if int(status) != 0:
            raise RuntimeError(f"cudaHostUnregister returned {status!r}")

    return tensor, unpin


def _drain(torch: Any) -> None:
    """Wait for every CUDA stream, the streamers' copy streams included."""
    if torch.cuda.is_initialized():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


def _release(torch: Any, segments: list[_Segment]) -> None:
    """Unlock every segment once no queued copy reads it; runs on close and on collection."""
    if not segments:
        return
    _drain(torch)
    cached = False
    for seg in segments:
        if seg.unpin is None:
            cached = True
            continue
        try:
            seg.unpin()
        except Exception as exc:  # noqa: BLE001 - keep unlocking the other segments
            warnings.warn(
                f"could not unlock a {seg.size}-byte pinned segment ({exc}); its pages stay "
                "locked until the process exits",
                stacklevel=2,
            )
    segments.clear()
    if cached:
        # PyTorch keeps freed pinned blocks; hand back those no tensor uses any more.
        release_pinned_cache()
