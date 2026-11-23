# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.telemetry.vram."""

import pytest

from hypertier.telemetry import GiB, MiB, vram
from hypertier.telemetry.vram import VramStats, track_vram


def test_vram_stats_derived_values():
    stats = VramStats(
        device=0,
        total_bytes=4 * GiB,
        free_before=3 * GiB,
        free_after=2 * GiB,
        allocated_before=100 * MiB,
        allocated_after=100 * MiB,
        peak_allocated=612 * MiB,
        peak_reserved=640 * MiB,
        samples=[("weights loaded", GiB + GiB // 2)],
    )
    assert stats.peak_delta == 512 * MiB
    assert stats.min_free_observed == GiB + GiB // 2
    assert "peak +512.0 MiB" in stats.summary()
    data = stats.as_dict()
    assert data["peak_delta"] == 512 * MiB
    assert data["samples"] == [("weights loaded", GiB + GiB // 2)]


def test_peak_delta_never_negative():
    stats = VramStats(device=0, allocated_before=200 * MiB, peak_allocated=100 * MiB)
    assert stats.peak_delta == 0


def test_track_vram_needs_cuda(monkeypatch):
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError), track_vram():
        pass


@pytest.mark.cuda
def test_track_vram_records_peak_and_free_memory():
    import torch

    with track_vram() as tracker:
        x = torch.empty(64 * MiB, dtype=torch.uint8, device="cuda")
        free_mid = tracker.sample("allocated")
        del x
    stats = tracker.stats
    assert stats.peak_delta >= 64 * MiB
    assert 0 < stats.free_before <= stats.total_bytes
    assert [label for label, _ in stats.samples] == ["allocated"]
    assert stats.min_free_observed <= free_mid


@pytest.mark.cuda
def test_track_vram_fills_stats_when_the_block_raises():
    import torch

    with pytest.raises(KeyError), track_vram() as tracker:
        y = torch.empty(32 * MiB, dtype=torch.uint8, device="cuda")
        raise KeyError("boom")
    assert tracker.stats.peak_delta >= 32 * MiB
    del y


def test_the_vram_of_a_process_is_read_from_the_listing_of_the_driver():
    listing = "100, GPU-aaaa, 278\n200, GPU-aaaa, 1024\n200, GPU-bbbb, 64\n"
    assert vram.parse_compute_apps(listing, 100, "GPU-aaaa") == 278 * MiB
    assert vram.parse_compute_apps(listing, 200, "GPU-bbbb") == 64 * MiB
    assert vram.parse_compute_apps(listing, 100, None) == 278 * MiB
    assert vram.parse_compute_apps(listing, 200, None) is None, "on two GPUs: which one?"
    assert vram.parse_compute_apps(listing, 300, "GPU-aaaa") is None, "not listed"
    assert vram.parse_compute_apps(listing, 100, "GPU-bbbb") is None
    assert vram.parse_compute_apps("100, GPU-aaaa, [N/A]\n", 100, "GPU-aaaa") is None  # WSL2
    assert vram.parse_compute_apps("", 100, None) is None
    assert vram.parse_compute_apps("No running processes found\n", 100, None) is None


@pytest.mark.cuda
def test_the_vram_of_this_process_is_its_allocator_and_its_context(monkeypatch):
    import torch

    block = torch.empty(256 * MiB, dtype=torch.uint8, device="cuda")
    held = vram.process_vram()
    if held is None:
        pytest.skip("the driver does not say what this process holds")
    outside = held - torch.cuda.memory_reserved()
    assert held >= block.numel() and 40 * MiB < outside < 400 * MiB, "a CUDA context"

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(vram.subprocess, "run", missing)
    assert vram.process_vram() is None
