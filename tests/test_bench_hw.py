# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmarks/bench_hw.py (metadata, summary and a quick end-to-end run)."""

import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_hw.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_hw", BENCH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_metadata_is_json_serializable_and_anonymous(bench):
    meta = bench.collect_metadata()
    text = json.dumps(meta)
    assert meta["python"] and meta["cpu"]["logical_cpus"]
    host = socket.gethostname()
    if len(host) >= 4:
        assert host not in text, "results may be published: no hostname"


def test_tilde_hides_the_home_directory(bench):
    home = str(Path.home())
    assert bench._tilde(home + "/dev") == "~/dev"
    assert bench._tilde("/srv/data") == "/srv/data"


def _fake_sysfs(root: Path, *, nvidia_boot_vga: str, outputs: dict[str, str]) -> None:
    """Hybrid-laptop layout: dGPU at 01:00.0 (card1), iGPU at 06:00.0 (card2)."""
    for address, card, boot in (("0000:01:00.0", "card1", nvidia_boot_vga),
                                ("0000:06:00.0", "card2", "1" if nvidia_boot_vga == "0" else "0")):
        dev = root / "bus" / "pci" / "devices" / address
        dev.mkdir(parents=True)
        (dev / "boot_vga").write_text(boot + "\n")
        drm = root / "class" / "drm" / card
        drm.mkdir(parents=True)
        (drm / "device").symlink_to(dev)
    drm = root / "class" / "drm"
    for name, status in outputs.items():
        (drm / name).mkdir()
        (drm / name / "status").write_text(status + "\n")


def test_sysfs_pci_address(bench):
    assert bench._sysfs_pci_address("00000000:01:00.0") == "0000:01:00.0"
    assert bench._sysfs_pci_address("0000000A:3B:00.1") == "000a:3b:00.1"
    assert bench._sysfs_pci_address("garbage") is None


def test_display_info_hybrid_with_external_monitor(bench, tmp_path):
    _fake_sysfs(tmp_path, nvidia_boot_vga="0", outputs={
        "card1-HDMI-A-1": "connected", "card1-DP-1": "disconnected", "card2-eDP-1": "connected",
    })
    info = bench.display_info("00000000:01:00.0", sysfs=tmp_path)
    assert info == {"boot_vga": False, "connected_outputs": ["HDMI-A-1"]}


def test_display_info_discrete_mode(bench, tmp_path):
    _fake_sysfs(tmp_path, nvidia_boot_vga="1", outputs={"card1-eDP-2": "connected"})
    info = bench.display_info("00000000:01:00.0", sysfs=tmp_path)
    assert info == {"boot_vga": True, "connected_outputs": ["eDP-2"]}


def test_display_info_without_sysfs(bench, tmp_path):
    assert bench.display_info("00000000:01:00.0", sysfs=tmp_path) is None
    assert bench.display_info(None, sysfs=tmp_path) is None


def test_summary_shows_display_line(bench):
    payload = {"metadata": {"gpu": {
        "name": "GPU", "nvidia_smi": {"display_active": "Disabled"},
        "display": {"boot_vga": False, "connected_outputs": []},
    }}, "results": {}}
    text = bench.format_summary(payload)
    assert "Display active Disabled, outputs: none; primary GPU: no" in text


def test_quick_config_and_overrides(bench):
    args = bench.parse_args(["--quick", "--nvme-file-gib", "0.1", "--pinned-cap-gib", "1"])
    cfg = bench.make_config(args)
    assert cfg.copy_max_bytes == 8 * 2**20
    assert cfg.nvme_file_bytes % max(cfg.nvme_chunks) == 0
    assert cfg.pinned_cap_bytes == 2**30


def test_summary_tolerates_partial_payloads(bench):
    payload = {
        "metadata": bench.collect_metadata(),
        "results": {},
        "errors": {"nvme": "OSError: no space"},
        "skipped": ["copy"],
    }
    text = bench.format_summary(payload)
    assert "nvme: OSError: no space" in text and "Skipped: copy" in text


@pytest.mark.cuda
def test_quick_run_writes_json(bench, tmp_path):
    out_dir = tmp_path / "results"
    nvme_dir = tmp_path / "nvme"
    nvme_dir.mkdir()
    code = bench.main(["--quick", "--out-dir", str(out_dir), "--nvme-dir", str(nvme_dir)])
    files = list(out_dir.glob("hw_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert code == 0, data["errors"]
    assert set(bench.PHASES) <= data["results"].keys()
    assert data["metadata"]["gpu"]["name"]
    assert list(nvme_dir.iterdir()) == []
