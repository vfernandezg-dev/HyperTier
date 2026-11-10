# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.telemetry.vram."""

import pytest

from hypertier.telemetry import GiB, MiB
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
