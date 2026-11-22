# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.stream.pinned (the page-locked RAM pool)."""

import gc
import mmap
import subprocess
import sys

import pytest

from hypertier.stream.pinned import ALIGN, PinnedPool, align_up
from hypertier.telemetry import KiB, MiB, read_meminfo

PAGE = mmap.PAGESIZE


def ram_in_use() -> int:
    """Anonymous plus shared memory, system-wide; MemAvailable lags after memory is freed."""
    info = read_meminfo()
    return info["AnonPages"] + info["Shmem"]


def test_align_up():
    assert [align_up(n) for n in (0, 1, 255, 256, 257)] == [0, 256, 256, 256, 512]
    assert align_up(10, 4) == 12
    assert align_up(PAGE + 1, PAGE) == 2 * PAGE


def test_pool_needs_cuda(monkeypatch):
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        PinnedPool(MiB)


@pytest.mark.cuda
@pytest.mark.parametrize(("capacity", "segment"), [(-1, MiB), (0, 0), (0, -PAGE)])
def test_pool_validates_sizes(capacity, segment):
    with pytest.raises(ValueError):
        PinnedPool(capacity, segment_bytes=segment)


@pytest.mark.cuda
def test_alloc_returns_aligned_contiguous_pinned_views():
    with PinnedPool(0, segment_bytes=MiB) as pool:
        assert pool.capacity_bytes == 0  # nothing is pinned before the first request
        sizes = (1000, 1, 300 * KiB, 0, 4096)
        views = [pool.alloc(n) for n in sizes]
        assert pool.capacity_bytes == MiB
        assert pool.used_bytes == sum(align_up(n) for n in sizes)
        end = 0
        for view, n in zip(views, sizes):
            assert view.numel() == n and view.is_contiguous()
            if n:
                assert view.is_pinned()
                assert view.data_ptr() % ALIGN == 0
                assert view.data_ptr() >= end, "allocations overlap"
                end = view.data_ptr() + n


@pytest.mark.cuda
def test_alloc_can_start_on_a_page_for_direct_reads():
    with PinnedPool(0, segment_bytes=MiB) as pool:
        pool.alloc(100)
        page = pool.alloc(3 * PAGE, align=PAGE)  # O_DIRECT needs a block-aligned address
        assert page.data_ptr() % PAGE == 0 and page.is_pinned()
        assert pool.used_bytes == 4 * PAGE
        assert pool.alloc(10).data_ptr() == page.data_ptr() + 3 * PAGE
        for bad in (100, 768, 2 * PAGE):
            with pytest.raises(ValueError, match="align"):
                pool.alloc(16, align=bad)


@pytest.mark.cuda
def test_capacity_is_locked_at_its_exact_size():
    odd = MiB + 5000  # not a power of two, not even a whole number of pages
    with PinnedPool(2 * MiB + odd, segment_bytes=2 * MiB) as pool:
        assert pool.capacity_bytes == 2 * MiB + align_up(odd, PAGE)
        assert pool.pin_seconds > 0
        pinned = pool.pin_seconds
        pool.alloc(2 * MiB)
        pool.alloc(odd)
        assert pool.pin_seconds == pinned, "requests that fit must not pin anything"


@pytest.mark.cuda
def test_pool_grows_instead_of_failing():
    with PinnedPool(0, segment_bytes=MiB) as pool:
        big = pool.alloc(3 * MiB + 100)  # larger than a segment: gets one of exactly its size
        assert big.numel() == 3 * MiB + 100 and big.is_pinned()
        assert pool.capacity_bytes == 3 * MiB + PAGE
        pool.alloc(2 * KiB)  # fits in what the page rounding left over
        assert pool.capacity_bytes == 3 * MiB + PAGE
        pool.alloc(MiB)  # no room left anywhere: one more regular segment
        assert pool.capacity_bytes == 4 * MiB + PAGE


@pytest.mark.cuda
def test_reset_hands_out_the_same_pages_without_pinning_again():
    with PinnedPool(0, segment_bytes=MiB) as pool:
        first = pool.alloc(600 * KiB)
        address, pinned = first.data_ptr(), pool.pin_seconds
        del first
        pool.reset()
        assert pool.used_bytes == 0 and pool.capacity_bytes == MiB
        assert pool.alloc(600 * KiB).data_ptr() == address
        assert pool.pin_seconds == pinned


@pytest.mark.cuda
def test_reset_waits_for_copies_still_reading_the_pool():
    import torch

    size = 64 * MiB
    with PinnedPool(0, segment_bytes=size) as pool:
        src = pool.alloc(size)
        src.fill_(1)
        stream = torch.cuda.Stream()
        targets = [torch.empty(size, dtype=torch.uint8, device="cuda") for _ in range(4)]
        with torch.cuda.stream(stream):
            for dst in targets:  # about 40 ms of copies queued behind each other
                dst.copy_(src, non_blocking=True)
        pool.reset()
        pool.alloc(size).fill_(2)  # the next model overwrites the same pages
        torch.cuda.synchronize()
        assert all(int(dst.min()) == 1 and int(dst.max()) == 1 for dst in targets)


@pytest.mark.cuda
def test_close_unlocks_the_pages_and_views_stay_valid_until_dropped():
    size = 64 * MiB + 5 * PAGE
    PinnedPool(PAGE).close()  # CUDA and the registration path allocate on first use
    gc.collect()
    base = ram_in_use()
    pool = PinnedPool(size)
    view = pool.alloc(size)
    view.fill_(7)
    assert view.is_pinned()
    assert pool.capacity_bytes == size
    assert abs(ram_in_use() - base - size) <= 16 * MiB, "locked more (or less) than asked"

    pool.close()
    pool.close()  # idempotent
    assert pool.capacity_bytes == 0
    with pytest.raises(RuntimeError):
        pool.alloc(16)
    gc.collect()
    assert not view.is_pinned()
    assert int(view.min()) == 7 == int(view.max()), "the memory must outlive the pool"
    view.fill_(9)  # still writable, ordinary memory

    del view
    gc.collect()
    assert ram_in_use() - base <= 16 * MiB, "the mapping must go away with its last tensor"


@pytest.mark.cuda
def test_pool_dropped_without_close_unlocks_its_pages():
    pool = PinnedPool(0, segment_bytes=MiB)
    view = pool.alloc(1000)
    view.fill_(3)
    assert view.is_pinned()
    del pool
    gc.collect()
    assert not view.is_pinned()
    assert int(view.max()) == 3


@pytest.mark.cuda
@pytest.mark.parametrize("failure", ["binding missing", "registration refused"])
def test_falls_back_to_the_pytorch_pinned_allocator_with_a_warning(monkeypatch, failure):
    import torch

    class Refuses:
        def cudaHostRegister(self, *args):  # same names as the CUDA runtime binding
            return 2  # cudaErrorMemoryAllocation

        def cudaHostUnregister(self, *args):
            raise AssertionError("nothing was registered")

    def missing():
        raise AttributeError("this PyTorch has no cudart binding")

    monkeypatch.setattr(torch.cuda, "cudart", missing if failure == "binding missing" else Refuses)
    with pytest.warns(UserWarning, match="cudaHostRegister.*falling back"):
        pool = PinnedPool(MiB)
    with pool:
        view = pool.alloc(300 * KiB)
        view.fill_(5)
        assert view.is_pinned() and view.data_ptr() % ALIGN == 0
        dst = torch.empty(300 * KiB, dtype=torch.uint8, device="cuda")
        dst.copy_(view, non_blocking=True)
        assert int(dst.min()) == 5 == int(dst.max())
        with pytest.warns(UserWarning, match="falling back"):
            pool.alloc(2 * MiB)  # growing takes the same path
    assert int(view.max()) == 5, "views must survive the pool here too"


@pytest.mark.cuda
@pytest.mark.filterwarnings("ignore:could not page-lock")
def test_fallback_memory_that_starts_off_a_page_still_serves_page_aligned_requests(monkeypatch):
    import torch

    from hypertier.telemetry.bandwidth import release_pinned_cache

    def missing():
        raise AttributeError("this PyTorch has no cudart binding")

    monkeypatch.setattr(torch.cuda, "cudart", missing)
    release_pinned_cache()
    # PyTorch packs small pinned requests into one segment, 512 bytes apart, and makes some
    # itself (reading a scalar from the GPU): whatever is pinned next starts off a page.
    tiny = torch.empty(8, dtype=torch.uint8, pin_memory=True)
    with PinnedPool(0, segment_bytes=MiB) as pool:
        first = pool.alloc(100)
        assert first.data_ptr() % PAGE, "the premise: this pinned memory is not page-aligned"
        assert first.data_ptr() % ALIGN == 0
        page = pool.alloc(3 * PAGE, align=PAGE)
        assert page.data_ptr() % PAGE == 0 and page.is_pinned()
        whole = pool.alloc(MiB, align=PAGE)  # as large as a segment: needs room to align
        assert whole.data_ptr() % PAGE == 0 and whole.numel() == MiB
    del tiny


@pytest.mark.cuda
@pytest.mark.parametrize("cuda_torn_down", [False, True])
def test_exiting_without_close_is_silent(cuda_torn_down):
    code = [
        "import torch",
        "from hypertier.stream.pinned import PinnedPool",
        "pool = PinnedPool(0, segment_bytes=1 << 20)",
        "view = pool.alloc(1000)",
        "view.fill_(1)",
        "assert view.is_pinned()",
    ]
    if cuda_torn_down:
        # What a finalizer running at exit could meet: waiting for the streams fails.
        code += [
            "def gone(*args, **kwargs):",
            "    raise RuntimeError('the CUDA context is gone')",
            "torch.cuda.synchronize = gone",
        ]
    # PyTorch warns on import when NumPy is missing; that is not what this test is about.
    command = [sys.executable, "-W", "ignore:Failed to initialize NumPy", "-c", "\n".join(code)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
