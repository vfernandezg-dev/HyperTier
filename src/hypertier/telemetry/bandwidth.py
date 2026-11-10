# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Hardware probes: the measured inputs of HyperTier's performance law.

Every placement decision follows from one inequality::

    time_per_pass >= max(compute_time, non_resident_bytes / link_bandwidth)

This module measures its terms on the local machine:

* RAM <-> VRAM copy bandwidth, pageable and pinned, from 1 MiB to 1 GiB;
* whether a host-to-device copy on its own stream actually hides behind compute;
* how much pinned (page-locked) memory the OS lets one process hold;
* sequential NVMe reads straight into a pinned buffer, bypassing the page cache;
* VRAM bandwidth and dense fp16/bf16 matmul throughput.

GPU timings use CUDA events (the device timeline) after a warm-up, repeated and summarised as
median and p90. Bandwidth is decimal GB/s (1e9 B/s), the unit PCIe and NVMe datasheets use;
buffer sizes are binary (MiB, GiB). PyTorch is imported lazily, so the statistics, the file
I/O helpers and the pinned-probe algorithm work, and are tested, without it.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import gc
import math
import mmap
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import Self

from hypertier.telemetry._common import (
    GiB,
    MiB,
    cuda_available,
    format_bytes,
    import_torch,
    is_wsl,
    mem_available_bytes,
    read_meminfo,
    require_cuda,
    resolve_device,
)

DIRECT_IO_ALIGN = 4096
"""Alignment of O_DIRECT buffers, offsets and sizes; covers 512 B and 4 KiB sectors."""

# ---------------------------------------------------------------------------------------------
# Statistics


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile with linear interpolation between order statistics (numpy's default)."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be within [0, 100], got {q}")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass(frozen=True)
class TimingStats:
    """Repeated timings in milliseconds.

    ``p90_ms`` is the 90th percentile of *time*: 90 % of the runs were at least that fast.
    Rates derived from it are the conservative figures a planner should budget with; the
    median is the typical case.
    """

    n: int
    median_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, samples_ms: Sequence[float]) -> TimingStats:
        if not samples_ms:
            raise ValueError("no timing samples")
        return cls(
            n=len(samples_ms),
            median_ms=percentile(samples_ms, 50.0),
            p90_ms=percentile(samples_ms, 90.0),
            min_ms=min(samples_ms),
            max_ms=max(samples_ms),
        )


def gb_per_s(nbytes: float, ms: float) -> float:
    """Decimal GB/s for ``nbytes`` moved in ``ms`` milliseconds."""
    if ms <= 0:
        raise ValueError(f"duration must be positive, got {ms} ms")
    return nbytes / (ms * 1e6)


def size_sweep(min_bytes: int = MiB, max_bytes: int = GiB) -> list[int]:
    """Doubling sizes from ``min_bytes`` up to ``max_bytes`` (1 MiB .. 1 GiB: 11 points)."""
    if min_bytes <= 0 or max_bytes < min_bytes:
        raise ValueError(f"invalid size range [{min_bytes}, {max_bytes}]")
    sizes = []
    size = min_bytes
    while size <= max_bytes:
        sizes.append(size)
        size *= 2
    return sizes


def overlap_efficiency(copy_ms: float, compute_ms: float, both_ms: float) -> float:
    """Fraction of the shorter operation hidden behind the longer one.

    1.0: the pair took ``max(copy, compute)`` (perfect overlap); 0.0: it took
    ``copy + compute`` (serialised). Measurement noise can push it slightly outside [0, 1].
    """
    shorter = min(copy_ms, compute_ms)
    if shorter <= 0:
        raise ValueError("copy and compute times must be positive")
    return (copy_ms + compute_ms - both_ms) / shorter


def _median(values: Sequence[float]) -> float:
    return percentile(values, 50.0)


# ---------------------------------------------------------------------------------------------
# CUDA timing helpers


def _time_cuda(
    fn: Callable[[], object], *, warmup: int, repeats: int, stream: Any = None
) -> tuple[list[float], list[float]]:
    """Run ``fn`` ``repeats`` times after ``warmup`` runs; return (event_ms, wall_ms) samples.

    Events are recorded on ``stream`` (default: the current stream), so ``fn`` must enqueue
    its work there. Event time is the device timeline. Wall time adds launch and sync
    overhead; it is kept as a cross-check, notably for pageable copies, which the driver
    stages on the CPU.
    """
    torch = import_torch()
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    stream = torch.cuda.current_stream() if stream is None else stream
    for _ in range(warmup):
        fn()
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    event_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        start.record(stream)
        fn()
        end.record(stream)
        end.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1e3)
        event_ms.append(start.elapsed_time(end))
    return event_ms, wall_ms


def host_buffer(nbytes: int, *, pinned: bool) -> Any:
    """Host uint8 tensor with every page touched, so first-touch faults are never timed."""
    torch = import_torch()
    buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pinned)
    buf.fill_(1)
    return buf


def release_pinned_cache() -> None:
    """Hand cached pinned blocks back to the OS.

    PyTorch's pinned-memory allocator keeps freed blocks for reuse, so ``del tensor`` alone
    does not return page-locked RAM (measured on WSL2: MemAvailable only recovers after this
    call). ``torch._C._host_emptyCache`` is private API, hence the guard.
    """
    torch = import_torch()
    gc.collect()
    empty = getattr(torch._C, "_host_emptyCache", None)
    if empty is not None:
        empty()


def warm_up_gpu(seconds: float = 1.0, *, device: Any = None) -> None:
    """Keep the GPU and its PCIe link busy so neither is measured in an idle power state."""
    torch = require_cuda()
    dev = resolve_device(device)
    a = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
    host = host_buffer(64 * MiB, pinned=True)
    dst = torch.empty(64 * MiB, dtype=torch.uint8, device=dev)
    deadline = time.perf_counter() + seconds
    try:
        while time.perf_counter() < deadline:
            torch.matmul(a, a)
            dst.copy_(host, non_blocking=True)
            torch.cuda.synchronize(dev)
    finally:
        del a, host, dst
        release_pinned_cache()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------
# RAM <-> VRAM copies


@dataclass(frozen=True)
class CopyResult:
    """One point of the RAM <-> VRAM sweep."""

    direction: str  # "h2d" (RAM -> VRAM) or "d2h" (VRAM -> RAM)
    memory: str  # host side: "pinned" or "pageable"
    size_bytes: int
    timing: TimingStats  # CUDA events
    wall: TimingStats  # host clock, cross-check
    gbps_median: float
    gbps_p90: float


def _repeats_for(size: int, repeats: int, budget_bytes: int = 8 * GiB, floor: int = 5) -> int:
    """Fewer repeats for huge transfers, so the 1 GiB points do not dominate the run time."""
    return max(min(repeats, floor), min(repeats, budget_bytes // size))


def _check_choices(name: str, values: Sequence[str], allowed: Sequence[str]) -> None:
    if not values or any(v not in allowed for v in values):
        raise ValueError(f"{name} must be a non-empty subset of {tuple(allowed)}, got {values}")


def _copy_point(
    host: Any, dev_buf: Any, direction: str, memory: str, size: int, warmup: int, repeats: int
) -> CopyResult:
    src, dst = host[:size], dev_buf[:size]
    if direction == "d2h":
        src, dst = dst, src
    event_ms, wall_ms = _time_cuda(
        partial(dst.copy_, src, non_blocking=True),
        warmup=warmup,
        repeats=_repeats_for(size, repeats),
        stream=import_torch().cuda.current_stream(dev_buf.device),
    )
    timing = TimingStats.from_samples(event_ms)
    return CopyResult(
        direction=direction,
        memory=memory,
        size_bytes=size,
        timing=timing,
        wall=TimingStats.from_samples(wall_ms),
        gbps_median=gb_per_s(size, timing.median_ms),
        gbps_p90=gb_per_s(size, timing.p90_ms),
    )


def measure_copy_bandwidth(
    sizes: Iterable[int] | None = None,
    *,
    directions: Sequence[str] = ("h2d", "d2h"),
    memories: Sequence[str] = ("pinned", "pageable"),
    warmup: int = 3,
    repeats: int = 20,
    device: Any = None,
) -> list[CopyResult]:
    """RAM <-> VRAM bandwidth per transfer size (default: powers of two, 1 MiB .. 1 GiB).

    Pinned copies are DMA straight from page-locked memory. Pageable copies are staged by
    the driver through its own pinned bounce buffers, which is the cost HyperTier avoids by
    streaming from pinned memory. Points of 512 MiB and more use fewer repeats; ``n`` is
    recorded in every result.
    """
    torch = require_cuda()
    _check_choices("directions", directions, ("h2d", "d2h"))
    _check_choices("memories", memories, ("pinned", "pageable"))
    points = sorted(set(sizes)) if sizes is not None else size_sweep()
    if not points or points[0] <= 0:
        raise ValueError("sizes must be positive")
    dev = resolve_device(device)
    dev_buf = torch.empty(points[-1], dtype=torch.uint8, device=dev)
    results: list[CopyResult] = []
    try:
        for memory in memories:
            host = host_buffer(points[-1], pinned=memory == "pinned")
            try:
                for direction in directions:
                    for size in points:
                        results.append(
                            _copy_point(host, dev_buf, direction, memory, size, warmup, repeats)
                        )
            finally:
                del host
                if memory == "pinned":
                    release_pinned_cache()
    finally:
        del dev_buf
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------------------------
# Copy / compute overlap


@dataclass(frozen=True)
class OverlapResult:
    """Pinned H2D copy on a dedicated stream vs. matmuls on the default stream (medians)."""

    copy_bytes: int
    matmul_n: int
    matmul_dtype: str
    matmul_iters: int
    copy_ms: float  # copy alone
    compute_ms: float  # compute alone
    both_ms: float  # both enqueued together, from start until both finished
    copy_ms_overlapped: float  # how long the copy took while compute was running
    efficiency: float  # overlap_efficiency(): 1.0 = copy fully hidden
    speedup: float  # (copy_ms + compute_ms) / both_ms; 2.0 is ideal for equal halves
    copy_gbps: float  # bandwidth of the copy alone


def measure_copy_compute_overlap(
    copy_bytes: int = 256 * MiB,
    *,
    matmul_n: int = 4096,
    dtype: str = "float16",
    warmup: int = 2,
    repeats: int = 10,
    device: Any = None,
) -> OverlapResult:
    """Check that a pinned H2D copy on its own stream runs concurrently with compute.

    Layer streaming only pays off if the copy engine and the SMs work in parallel. The
    compute side is a loop of matmuls sized to last about as long as the copy, the hardest
    case to hide, since the efficiency is normalised by the shorter of the two.
    """
    torch = require_cuda()
    try:
        return _overlap(torch, resolve_device(device), copy_bytes, matmul_n, dtype, warmup, repeats)
    finally:
        # _overlap's buffers die with its frame; hand the pinned block back to the OS.
        release_pinned_cache()
        torch.cuda.empty_cache()


def _overlap(
    torch: Any, dev: Any, copy_bytes: int, matmul_n: int, dtype: str, warmup: int, repeats: int
) -> OverlapResult:
    host = host_buffer(copy_bytes, pinned=True)
    dst = torch.empty(copy_bytes, dtype=torch.uint8, device=dev)
    a = torch.randn(matmul_n, matmul_n, device=dev, dtype=getattr(torch, dtype))
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    copy_stream = torch.cuda.Stream(device=dev)
    compute_stream = torch.cuda.current_stream(dev)
    start = torch.cuda.Event(enable_timing=True)
    copied = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def copy() -> None:
        with torch.cuda.stream(copy_stream):
            dst.copy_(host, non_blocking=True)

    copy_ms = _median(_time_cuda(copy, warmup=warmup, repeats=repeats, stream=copy_stream)[0])
    one_matmul_ms = _median(
        _time_cuda(
            partial(torch.matmul, a, b, out=c),
            warmup=warmup,
            repeats=repeats,
            stream=compute_stream,
        )[0]
    )
    iters = max(1, round(copy_ms / one_matmul_ms))

    def compute() -> None:
        for _ in range(iters):
            torch.matmul(a, b, out=c)

    compute_ms = _median(
        _time_cuda(compute, warmup=warmup, repeats=repeats, stream=compute_stream)[0]
    )

    def both() -> tuple[float, float]:
        start.record(compute_stream)
        copy_stream.wait_event(start)
        with torch.cuda.stream(copy_stream):
            dst.copy_(host, non_blocking=True)
            copied.record(copy_stream)
        compute()
        compute_stream.wait_event(copied)
        end.record(compute_stream)
        end.synchronize()
        return start.elapsed_time(end), start.elapsed_time(copied)

    for _ in range(warmup):
        both()
    samples = [both() for _ in range(repeats)]

    both_ms = _median([total for total, _ in samples])
    return OverlapResult(
        copy_bytes=copy_bytes,
        matmul_n=matmul_n,
        matmul_dtype=dtype,
        matmul_iters=iters,
        copy_ms=copy_ms,
        compute_ms=compute_ms,
        both_ms=both_ms,
        copy_ms_overlapped=_median([copy_part for _, copy_part in samples]),
        efficiency=overlap_efficiency(copy_ms, compute_ms, both_ms),
        speedup=(copy_ms + compute_ms) / both_ms,
        copy_gbps=gb_per_s(copy_bytes, copy_ms),
    )


# ---------------------------------------------------------------------------------------------
# VRAM bandwidth and compute


@dataclass(frozen=True)
class DeviceCopyResult:
    """VRAM -> VRAM copy; bandwidth counts read plus write traffic (2 x size)."""

    size_bytes: int
    timing: TimingStats
    gbps_median: float
    gbps_p90: float


def measure_device_bandwidth(
    size_bytes: int = 256 * MiB,
    *,
    warmup: int = 3,
    repeats: int = 20,
    device: Any = None,
) -> DeviceCopyResult:
    """Effective VRAM bandwidth: the ceiling of memory-bound kernels such as LLM decode."""
    torch = require_cuda()
    dev = resolve_device(device)
    src = torch.ones(size_bytes, dtype=torch.uint8, device=dev)
    dst = torch.empty_like(src)
    try:
        event_ms, _ = _time_cuda(
            partial(dst.copy_, src),
            warmup=warmup,
            repeats=repeats,
            stream=torch.cuda.current_stream(dev),
        )
    finally:
        del src, dst
        torch.cuda.empty_cache()
    timing = TimingStats.from_samples(event_ms)
    traffic = 2 * size_bytes
    return DeviceCopyResult(
        size_bytes=size_bytes,
        timing=timing,
        gbps_median=gb_per_s(traffic, timing.median_ms),
        gbps_p90=gb_per_s(traffic, timing.p90_ms),
    )


@dataclass(frozen=True)
class MatmulResult:
    """Square ``n x n @ n x n`` product; 2 n^3 FLOPs each."""

    dtype: str
    n: int
    timing: TimingStats
    tflops_median: float
    tflops_p90: float


def measure_matmul_tflops(
    sizes: Sequence[int] = (1024, 2048, 4096, 8192),
    dtypes: Sequence[str] = ("float16", "bfloat16"),
    *,
    warmup: int = 3,
    repeats: int = 10,
    device: Any = None,
) -> list[MatmulResult]:
    """Effective dense matmul throughput: the compute term for prefill and diffusion.

    PyTorch accumulates fp16/bf16 products in fp32, which GeForce Ampere tensor cores run at
    half their fp16-accumulate rate, so expect well under the headline peak. Sizes that do
    not fit comfortably in free VRAM are skipped.
    """
    torch = require_cuda()
    dev = resolve_device(device)
    stream = torch.cuda.current_stream(dev)
    results: list[MatmulResult] = []
    for name in dtypes:
        dtype = getattr(torch, name, None)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError(f"unsupported dtype {name!r}")
        elem = torch.empty((), dtype=dtype).element_size()
        for n in sizes:
            free, _ = torch.cuda.mem_get_info(dev)
            if 3 * n * n * elem > 0.8 * free:
                continue
            a = torch.randn(n, n, device=dev, dtype=dtype)
            b = torch.randn_like(a)
            c = torch.empty_like(a)
            event_ms, _ = _time_cuda(
                partial(torch.matmul, a, b, out=c), warmup=warmup, repeats=repeats, stream=stream
            )
            del a, b, c
            timing = TimingStats.from_samples(event_ms)
            flops = 2 * n**3
            results.append(
                MatmulResult(
                    dtype=name,
                    n=n,
                    timing=timing,
                    tflops_median=flops / (timing.median_ms * 1e9),
                    tflops_p90=flops / (timing.p90_ms * 1e9),
                )
            )
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------------------------
# Largest pinned allocation


@dataclass(frozen=True)
class PinnedProbeResult:
    """Outcome of :func:`probe_max_pinned`."""

    max_pinned_bytes: int  # held at the same time when the probe stopped
    blocks: list[int]  # successful allocations, in order
    stop_reason: str  # "cap", "alloc_failed" or "ram_guard"
    cap_bytes: int
    reserve_bytes: int
    first_error: str | None
    pin_gbps: float | None  # how fast the driver pins memory (bytes / time spent allocating)
    trace: list[tuple[int, int | None]]  # (bytes pinned so far, MemAvailable afterwards)
    mem_available_before: int | None
    mem_available_after_release: int | None


def default_ram_reserve(mem_total: int | None = None) -> int:
    """RAM the pinned probe always leaves available: max(2 GiB, 15 % of total RAM)."""
    total = read_meminfo().get("MemTotal") if mem_total is None else mem_total
    return max(2 * GiB, int(0.15 * total)) if total else 2 * GiB


def _pow2_floor(n: int) -> int:
    return 1 << (n.bit_length() - 1) if n > 0 else 0


def _first_line(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}"


def _pin(nbytes: int) -> object:
    torch = import_torch()
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


def probe_max_pinned(
    cap_bytes: int = 16 * GiB,
    *,
    first_block: int = 256 * MiB,
    max_block: int = 2 * GiB,
    min_block: int = 64 * MiB,
    reserve_bytes: int | None = None,
    mem_total: int | None = None,
    allocate: Callable[[int], object] | None = None,
    release: Callable[[], None] | None = None,
    mem_available: Callable[[], int | None] = mem_available_bytes,
) -> PinnedProbeResult:
    """Pin growing blocks until allocation fails or a limit is hit, then free everything.

    Stops at the first of: ``cap_bytes``; an allocation failure (the driver/OS limit we are
    after; failing sizes are halved down to ``min_block`` to narrow it); or the RAM guard,
    which keeps MemAvailable above ``reserve_bytes`` and the total below MemTotal minus the
    reserve. Pinned pages cannot be swapped and, on WSL2, the VM's RAM is shared with
    Docker/Podman: an incomplete answer is better than waking the OOM killer. Blocks are
    powers of two because PyTorch's pinned allocator rounds requests up to one.

    ``allocate``, ``release`` and ``mem_available`` default to PyTorch pinned tensors and
    /proc/meminfo; tests inject fakes.
    """
    for label, value in (
        ("first_block", first_block),
        ("max_block", max_block),
        ("min_block", min_block),
    ):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{label} must be a positive power of two, got {value}")
    if not min_block <= first_block <= max_block:
        raise ValueError("expected min_block <= first_block <= max_block")
    if allocate is None:
        require_cuda()
        allocate = _pin
        release = release or release_pinned_cache

    total_ram = read_meminfo().get("MemTotal") if mem_total is None else mem_total
    reserve = default_ram_reserve(total_ram) if reserve_bytes is None else reserve_bytes
    ceiling = min(cap_bytes, max(0, total_ram - reserve)) if total_ram else cap_bytes

    held: list[object] = []
    blocks: list[int] = []
    trace: list[tuple[int, int | None]] = []
    total = 0
    block = first_block
    grow = True
    stop = "cap"
    first_error: str | None = None
    pin_seconds = 0.0
    before = mem_available()
    try:
        while True:
            size = min(block, _pow2_floor(ceiling - total))
            if size < min_block:
                stop = "cap" if ceiling == cap_bytes else "ram_guard"
                break
            available = mem_available()
            if available is not None and available - size < reserve:
                size = _pow2_floor(max(0, available - reserve))
                if size < min_block:
                    stop = "ram_guard"
                    break
                block, grow = size, False
            t0 = time.perf_counter()
            try:
                held.append(allocate(size))
            except (RuntimeError, MemoryError) as exc:
                first_error = first_error or _first_line(exc)
                grow = False
                if size // 2 < min_block:
                    stop = "alloc_failed"
                    break
                block = size // 2
                continue
            pin_seconds += time.perf_counter() - t0
            blocks.append(size)
            total += size
            trace.append((total, mem_available()))
            if grow:
                block = min(block * 2, max_block)
    finally:
        held.clear()
        gc.collect()
        if release is not None:
            release()
    return PinnedProbeResult(
        max_pinned_bytes=total,
        blocks=blocks,
        stop_reason=stop,
        cap_bytes=cap_bytes,
        reserve_bytes=reserve,
        first_error=first_error,
        pin_gbps=total / pin_seconds / 1e9 if pin_seconds > 0 else None,
        trace=trace,
        mem_available_before=before,
        mem_available_after_release=mem_available(),
    )


# ---------------------------------------------------------------------------------------------
# NVMe sequential reads


class AlignedBuffer:
    """Page-aligned writable host buffer usable with O_DIRECT; pinned when requested.

    Pinned buffers are PyTorch pinned tensors exposed as a ``memoryview`` through ctypes (no
    numpy), so ``readinto`` lands bytes directly in DMA-able memory: the NVMe -> pinned RAM
    leg of the streaming path. Unpinned buffers are anonymous mmaps, page-aligned by
    construction.
    """

    def __init__(self, nbytes: int, *, pinned: bool) -> None:
        if nbytes <= 0 or nbytes % DIRECT_IO_ALIGN:
            raise ValueError(f"size must be a positive multiple of {DIRECT_IO_ALIGN}")
        self.nbytes = nbytes
        self.pinned = pinned
        self._tensor: Any = None
        self._mmap: mmap.mmap | None = None
        if pinned:
            torch = require_cuda()
            self._tensor = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            address = self._tensor.data_ptr()
            if address % DIRECT_IO_ALIGN:
                raise RuntimeError("pinned buffer is not page-aligned; O_DIRECT would fail")
            raw = (ctypes.c_ubyte * nbytes).from_address(address)
            self.view = memoryview(raw).cast("B")
        else:
            self._mmap = mmap.mmap(-1, nbytes)
            self.view = memoryview(self._mmap)

    def close(self) -> None:
        self.view.release()
        if self._mmap is not None:
            with contextlib.suppress(BufferError):  # a stray slice still alive: GC frees it
                self._mmap.close()
        self._tensor = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _open(path: Path, flags: int, direct: bool | None) -> tuple[int, bool]:
    """Open ``path`` with O_DIRECT when requested (True) or possible (None)."""
    o_direct = getattr(os, "O_DIRECT", 0)
    if direct is not False:
        if o_direct:
            try:
                return os.open(path, flags | o_direct, 0o600), True
            except OSError as exc:
                # tmpfs and some FUSE/network filesystems refuse O_DIRECT with EINVAL.
                if direct or exc.errno not in (errno.EINVAL, errno.EOPNOTSUPP):
                    raise
        elif direct:
            raise OSError(errno.EINVAL, "O_DIRECT is not available on this platform")
    return os.open(path, flags, 0o600), False


def _drop_page_cache(fd: int) -> bool:
    """Evict this file's clean cached pages (unprivileged, unlike /proc/sys/vm/drop_caches)."""
    fadvise = getattr(os, "posix_fadvise", None)
    if fadvise is None:
        return False
    fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    return True


def write_file(
    path: str | os.PathLike[str],
    buffer: memoryview,
    total_bytes: int,
    *,
    direct: bool | None = None,
) -> tuple[float, str]:
    """Fill ``path`` with ``total_bytes`` by repeating ``buffer``; return (seconds, method).

    The time includes fsync, i.e. until the data reached the device. Without O_DIRECT the
    file's pages are dropped from the page cache afterwards, so later reads hit the disk.
    """
    chunk = len(buffer)
    if chunk == 0 or total_bytes <= 0 or total_bytes % chunk:
        raise ValueError("total_bytes must be a positive multiple of the buffer size")
    fd, used_direct = _open(Path(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, direct)
    try:
        t0 = time.perf_counter()
        for _ in range(total_bytes // chunk):
            pending = buffer
            while pending:
                pending = pending[os.write(fd, pending):]
        os.fsync(fd)
        seconds = time.perf_counter() - t0
        if not used_direct:
            _drop_page_cache(fd)
    finally:
        os.close(fd)
    return seconds, "o_direct" if used_direct else "buffered+fsync"


def read_file(
    path: str | os.PathLike[str],
    buffer: memoryview,
    chunk_bytes: int | None = None,
    *,
    direct: bool | None = None,
) -> tuple[int, float, str]:
    """Read the whole file sequentially into ``buffer``; return (bytes, seconds, method).

    ``buffer[:chunk_bytes]`` is reused for every request. With O_DIRECT the reads bypass the
    page cache and the device DMAs straight into ``buffer``. Otherwise the file's cached
    pages are evicted first with posix_fadvise(DONTNEED) ("fadvise_dontneed"); "buffered"
    means neither was possible and the result may be a cache measurement.
    """
    chunk = len(buffer) if chunk_bytes is None else chunk_bytes
    if not 0 < chunk <= len(buffer):
        raise ValueError("chunk_bytes must be in (0, len(buffer)]")
    fd, used_direct = _open(Path(path), os.O_RDONLY, direct)
    method = "o_direct"
    if not used_direct:
        method = "fadvise_dontneed" if _drop_page_cache(fd) else "buffered"
    target = buffer[:chunk]
    total = 0
    try:
        with open(fd, "rb", buffering=0, closefd=True) as f:  # FileIO: readinto == read(2)
            t0 = time.perf_counter()
            while n := f.readinto(target):
                total += n
            seconds = time.perf_counter() - t0
    finally:
        target.release()
    return total, seconds, method


def _is_under_mnt(path: Path) -> bool:
    return path.resolve().parts[1:2] == ("mnt",)


def filesystem_type(path: str | os.PathLike[str]) -> str | None:
    """Filesystem type of the mount holding ``path`` (Linux), e.g. ``ext4`` or ``9p``."""
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None
    target = str(Path(path).resolve())
    best, fstype = "", None
    for line in lines:
        left, _, right = line.partition(" - ")
        fields = left.split()
        if len(fields) < 5 or not right:
            continue
        mount_point = fields[4].replace("\\040", " ")
        inside = target == mount_point or target.startswith(mount_point.rstrip("/") + "/")
        if inside and len(mount_point) >= len(best):
            best, fstype = mount_point, right.split()[0]
    return fstype


@dataclass(frozen=True)
class NvmeRead:
    """One sequential pass over the test file."""

    chunk_bytes: int
    bytes_read: int
    seconds: float
    gbps: float
    method: str  # "o_direct", "fadvise_dontneed" or "buffered" (see read_file)
    page_cache_growth_bytes: int | None  # ~0 when the page cache was bypassed


@dataclass(frozen=True)
class NvmeReport:
    """Result of :func:`measure_nvme_read`."""

    directory: str
    filesystem: str | None
    file_bytes: int
    pinned_buffer: bool
    write_seconds: float
    write_gbps: float
    write_method: str
    reads: list[NvmeRead] = field(default_factory=list)


def measure_nvme_read(
    directory: str | os.PathLike[str] = "~/dev",
    *,
    file_bytes: int = 4 * GiB,
    chunk_sizes: Sequence[int] = (4 * MiB, 16 * MiB, 64 * MiB),
    pinned: bool | None = None,
    direct: bool | None = None,
    allow_windows_mount: bool = False,
) -> NvmeReport:
    """Sequential read throughput of the disk under ``directory``, into a pinned buffer.

    Writes a temporary file of random (incompressible) data, then reads it once per chunk
    size with ``readinto`` straight into one page-aligned buffer, pinned when CUDA is
    available: the NVMe -> pinned RAM leg the streamer will use.

    The page cache is bypassed with O_DIRECT; where a filesystem refuses O_DIRECT, the
    file's pages are evicted with posix_fadvise(DONTNEED) before each pass. The method and
    the page-cache growth are recorded so cache hits are visible. On WSL2 the disk is a VHDX
    file on the Windows host: the guest cannot see host-side caching, so a result above the
    drive's PCIe link limit would point to it. ``/mnt/<drive>`` paths are refused on WSL
    (they measure the 9P bridge, not the disk). The file is always deleted.
    """
    root = Path(directory).expanduser()
    if is_wsl() and _is_under_mnt(root) and not allow_windows_mount:
        raise ValueError(
            f"{root} is under /mnt (a Windows drive reached through 9P on WSL); "
            "use a Linux-filesystem path such as ~/dev"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"directory not found: {root}")
    chunks = sorted(set(chunk_sizes))
    if not chunks or any(c <= 0 or c % DIRECT_IO_ALIGN or file_bytes % c for c in chunks):
        raise ValueError(
            f"chunk sizes must be positive multiples of {DIRECT_IO_ALIGN} dividing file_bytes"
        )
    margin = 256 * MiB
    free = shutil.disk_usage(root).free
    if free < file_bytes + margin:
        raise OSError(
            errno.ENOSPC,
            f"need {format_bytes(file_bytes + margin)} free in {root}, have {format_bytes(free)}",
        )
    use_pinned = cuda_available() if pinned is None else pinned
    fd, name = tempfile.mkstemp(prefix="hypertier-nvme-", suffix=".bin", dir=root)
    os.close(fd)
    path = Path(name)
    reads: list[NvmeRead] = []
    try:
        with AlignedBuffer(chunks[-1], pinned=use_pinned) as buf:
            buf.view[:] = os.urandom(buf.nbytes)
            write_seconds, write_method = write_file(path, buf.view, file_bytes, direct=direct)
            for chunk in chunks:
                cached_before = read_meminfo().get("Cached")
                nread, seconds, method = read_file(path, buf.view, chunk, direct=direct)
                cached_after = read_meminfo().get("Cached")
                if nread != file_bytes:
                    raise OSError(f"short read: {nread} of {file_bytes} bytes")
                growth = (
                    None
                    if cached_before is None or cached_after is None
                    else cached_after - cached_before
                )
                seconds = max(seconds, 1e-9)
                reads.append(NvmeRead(chunk, nread, seconds, nread / seconds / 1e9, method, growth))
    finally:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
    write_seconds = max(write_seconds, 1e-9)
    return NvmeReport(
        directory=str(root.resolve()),
        filesystem=filesystem_type(root),
        file_bytes=file_bytes,
        pinned_buffer=use_pinned,
        write_seconds=write_seconds,
        write_gbps=file_bytes / write_seconds / 1e9,
        write_method=write_method,
        reads=reads,
    )
