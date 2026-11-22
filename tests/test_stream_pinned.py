# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.stream.pinned (the page-locked RAM pool)."""

import pytest

from hypertier.stream.pinned import ALIGN, PinnedPool, align_up, pow2_ceil
from hypertier.telemetry import KiB, MiB


def test_align_up_and_pow2_ceil():
    assert [align_up(n) for n in (0, 1, 255, 256, 257)] == [0, 256, 256, 256, 512]
    assert align_up(10, 4) == 12
    assert [pow2_ceil(n) for n in (0, 1, 2, 3, 4, 5, 1023, 1024, 1025)] == [
        1, 1, 2, 4, 4, 8, 1024, 1024, 2048,
    ]


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
@pytest.mark.parametrize(("capacity", "segment"), [(-1, MiB), (0, 3 * MiB), (0, 128)])
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
def test_initial_capacity_is_pinned_in_power_of_two_segments():
    with PinnedPool(3 * MiB, segment_bytes=2 * MiB) as pool:
        assert pool.capacity_bytes == 3 * MiB  # 2 MiB + 1 MiB
        assert pool.pin_seconds > 0
        pinned = pool.pin_seconds
        pool.alloc(2 * MiB)
        pool.alloc(MiB)
        assert pool.pin_seconds == pinned, "requests that fit must not pin anything"


@pytest.mark.cuda
def test_pool_grows_instead_of_failing():
    with PinnedPool(0, segment_bytes=MiB) as pool:
        big = pool.alloc(3 * MiB)  # larger than a segment: gets one big enough for it
        assert big.numel() == 3 * MiB and big.is_pinned()
        assert pool.capacity_bytes == 4 * MiB
        pool.alloc(MiB)  # fits in the tail of the big segment
        assert pool.capacity_bytes == 4 * MiB
        pool.alloc(MiB)  # no room left anywhere: one more regular segment
        assert pool.capacity_bytes == 5 * MiB


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
def test_closed_pool_refuses_requests():
    pool = PinnedPool(MiB, segment_bytes=MiB)
    pool.close()
    pool.close()  # idempotent
    assert pool.capacity_bytes == 0
    with pytest.raises(RuntimeError):
        pool.alloc(16)
