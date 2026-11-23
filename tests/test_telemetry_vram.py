# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.telemetry.vram."""

import os
import sys
import types

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


def fake_nvml(devices, *, init_fails=False):
    """A stand-in for the module ``pynvml``: ``devices`` is {uuid: [(pid, bytes or None)]}."""
    module = types.ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    def init():
        module.inits += 1
        if init_fails:
            raise NVMLError("the library of the driver was not found")

    def by_uuid(uuid):
        if uuid not in devices:
            raise NVMLError("no such GPU")
        return uuid

    def processes(handle):
        return [types.SimpleNamespace(pid=pid, usedGpuMemory=used) for pid, used in devices[handle]]

    module.NVMLError, module.inits = NVMLError, 0
    module.nvmlInit, module.nvmlShutdown = init, lambda: None
    module.nvmlDeviceGetHandleByUUID = by_uuid
    module.nvmlDeviceGetCount = lambda: len(devices)
    module.nvmlDeviceGetHandleByIndex = lambda index: list(devices)[index]
    module.nvmlDeviceGetComputeRunningProcesses = processes
    return module


def install(monkeypatch, module):
    """Make ``import pynvml`` give ``module`` (None: not installed), not tried yet."""
    monkeypatch.setitem(sys.modules, "pynvml", module)
    monkeypatch.setattr(vram, "_nvml_loaded", [])
    monkeypatch.setattr(vram.atexit, "register", lambda function: function)
    return module


def test_the_vram_of_a_process_is_read_through_nvml_when_it_is_installed(monkeypatch):
    devices = {
        "GPU-aaaa": [(100, 278 * MiB), (200, 1024 * MiB)],
        "GPU-bbbb": [(200, 64 * MiB), (300, None)],
    }
    nvml = install(monkeypatch, fake_nvml(devices))
    ask = vram._process_vram_nvml
    assert ask(100, "GPU-aaaa") == 278 * MiB
    assert ask(200, "GPU-bbbb") == 64 * MiB
    assert ask(100, None) == 278 * MiB
    assert ask(200, None) is None, "on two GPUs: which one?"
    assert ask(400, "GPU-aaaa") is None, "not listed"
    assert ask(100, "GPU-bbbb") is None
    assert ask(300, "GPU-bbbb") is None, "the driver does not say (WSL2)"
    assert nvml.inits == 1, "NVML is initialised once and kept"
    # NVML that cannot be asked is not NVML that does not know: nvidia-smi is asked then.
    assert ask(100, "GPU-cccc") is vram._NOT_ASKED, "a call that fails"
    assert ask(100, "GPU-aaaa") == 278 * MiB, "and the next one is asked again"
    broken = install(monkeypatch, fake_nvml(devices, init_fails=True))
    assert ask(100, "GPU-aaaa") is vram._NOT_ASKED and ask(100, "GPU-aaaa") is vram._NOT_ASKED
    assert broken.inits == 1, "not tried again: the library will not appear"
    install(monkeypatch, None)
    assert ask(100, "GPU-aaaa") is vram._NOT_ASKED, "the bindings are not installed"


@pytest.mark.cuda
def test_nvidia_smi_is_asked_only_when_nvml_cannot_be(monkeypatch):
    import torch

    asked = []

    def smi(pid, gpu_uuid):
        asked.append((pid, gpu_uuid))
        return 7 * MiB

    monkeypatch.setattr(vram, "_process_vram_smi", smi)
    monkeypatch.setattr(vram, "_process_vram_nvml", lambda pid, gpu_uuid: 5 * MiB)
    assert vram.process_vram() == 5 * MiB and not asked
    monkeypatch.setattr(vram, "_process_vram_nvml", lambda pid, gpu_uuid: None)
    assert vram.process_vram() is None and not asked, "NVML answered: it does not know"
    monkeypatch.setattr(vram, "_process_vram_nvml", lambda pid, gpu_uuid: vram._NOT_ASKED)
    assert vram.process_vram() == 7 * MiB
    assert asked == [(os.getpid(), f"GPU-{torch.cuda.get_device_properties(0).uuid}")]


@pytest.mark.cuda
def test_the_vram_of_this_process_is_its_allocator_and_its_context(monkeypatch):
    import torch

    block = torch.empty(256 * MiB, dtype=torch.uint8, device="cuda")
    held = vram.process_vram()
    if held is None:
        pytest.skip("the driver does not say what this process holds")
    outside = held - torch.cuda.memory_reserved()
    assert held >= block.numel() and 40 * MiB < outside < 400 * MiB, "a CUDA context"
    # Both ways read the same list of the driver.
    me = (os.getpid(), f"GPU-{torch.cuda.get_device_properties(0).uuid}")
    by_smi = vram._process_vram_smi(*me)
    assert by_smi is not None and abs(by_smi - held) <= MiB
    by_nvml = vram._process_vram_nvml(*me)
    if by_nvml is vram._NOT_ASKED:
        print("\n[vram] NVML bindings not installed: read through nvidia-smi")
    else:
        assert by_nvml == held, "NVML first"

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    install(monkeypatch, None)
    monkeypatch.setattr(vram.subprocess, "run", missing)
    assert vram.process_vram() is None, "neither can be asked"
