# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmarks/bench_streaming.py (config, summary and a quick end-to-end run)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_streaming.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_streaming", BENCH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_quick_config_and_overrides(bench):
    cfg = bench.make_config(bench.parse_args(["--quick", "--rows", "8,16", "--passes", "2"]))
    assert cfg.width == bench.QUICK.width and cfg.slots == bench.QUICK.slots
    assert cfg.rows == (8, 16) and cfg.passes == 2
    assert bench.make_config(bench.parse_args([])) == bench.Config()
    assert bench.Config().blocks % 2 == 1, "an odd block count is the case worth measuring"
    with pytest.raises(SystemExit):
        bench.parse_args(["--slots", "0"])


def test_summary_lists_every_case(bench):
    case = {
        "rows": 8192, "slots": 2, "resident_ms": 275.5, "streamed_ms": 279.1, "copy_ms": 70.7,
        "stall_ms": 0.02, "efficiency": 0.949, "bound_fraction": 0.987, "copy_gbps": 6.65,
        "misses_per_pass": 0, "bit_exact": True,
    }
    payload = {
        "metadata": {"timestamp_utc": "2026-11-22T00:00:00+00:00"},
        "results": {
            "model": {"blocks": 7, "block_bytes": 64 << 20, "weights_bytes": 448 << 20,
                      "dtype": "float16"},
            "builds": [{"slots": 2, "weights_vram_bytes": 128 << 20, "pinned_bytes": 896 << 20,
                        "pin_seconds": 0.5, "build_seconds": 0.8}],
            "cases": [case, {**case, "slots": 1, "misses_per_pass": 7.0, "bit_exact": False}],
        },
    }
    lines = bench.format_summary(payload).splitlines()
    assert any("7 blocks x 64.0 MiB" in line for line in lines)
    assert any("2 slot(s): 128.0 MiB of VRAM" in line for line in lines)
    rows = [line.split() for line in lines if line.strip().startswith("8192")]
    assert [row[1] for row in rows] == ["2", "1"]
    assert rows[0][2:7] == ["275.5", "70.7", "279.1", "0.0", "0.95"]
    assert rows[0][-2:] == ["0", "yes"] and rows[1][-2:] == ["7", "NO"]
    assert bench.format_summary({}).strip().startswith("HyperTier layer streaming")


def test_summary_lists_the_runs_from_files(bench):
    case = {
        "rows": 2048, "tier": "nvme", "slots": 2, "resident_ms": 70.0, "streamed_ms": 160.0,
        "copy_ms": 70.7, "read_ms": 155.0, "read_gbps": 2.9, "bound_fraction": 0.97,
        "enqueue_ms": 11.0, "enqueue_p99_ms": 22.5, "handoff_ms": 0.05, "handoff_p99_ms": 0.3,
        "bit_exact": True,
    }  # fmt: skip
    build = {"tier": "nvme", "tiers": "nnnnnnn", "slots": 2, "ram_budget": 128 << 20,
             "ram_bytes": 128 << 20, "build_seconds": 0.03}  # fmt: skip
    payload = {"results": {"from_file": {"builds": [build], "cases": [case]}}}
    lines = bench.format_summary(payload).splitlines()
    built = "nvme  2 slot(s): nnnnnnn, 128.0 MiB locked (budget 128.0 MiB)"
    assert any(built in line for line in lines)
    (row,) = [line.split() for line in lines if line.strip().startswith("2048")]
    assert row[:8] == ["2048", "nvme", "2", "70.0", "70.7", "155.0", "160.0", "0.97"]
    assert row[8] == "2.90" and row[-1] == "yes"

    serial = {**case, "tier": "ram", "read_ms": 0.0, "read_gbps": None, "serial_compute_ms": 55.0,
              "same_output": False}  # fmt: skip
    del serial["resident_ms"], serial["bit_exact"]
    ckpt = {"name": "Qwen3-8B", "blocks": 36, "block_bytes": 385892864,
            "weights_bytes": 36 * 385892864, "dtype": "bfloat16", "files": 5}  # fmt: skip
    long_build = {**build, "tier": "mixed", "tiers": "nr" * 18, "ram_budget": None}
    payload = {"results": {"checkpoint_run": {
        "checkpoint": ckpt, "builds": [long_build], "cases": [serial]}}}  # fmt: skip
    text = bench.format_summary(payload)
    assert "Checkpoint Qwen3-8B: 36 blocks of up to 368.0 MiB (bfloat16), 12.9 GiB in 5" in text
    assert "18 r + 18 n" in text and "1-slot compute" in text
    (row,) = [line.split() for line in text.splitlines() if line.strip().startswith("2048")]
    assert row[3] == "55.0" and row[8] == "-" and row[-1] == "NO"


@pytest.mark.cuda
def test_quick_run_writes_json(bench, tmp_path):
    out_dir = tmp_path / "results"
    code = bench.main(["--quick", "--out-dir", str(out_dir), "--model-dir", str(out_dir)])
    files = list(out_dir.glob("streaming_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    cases = data["results"]["cases"]
    assert code == 0, [c for c in cases if not c["bit_exact"]]
    assert data["schema"] == bench.SCHEMA and data["metadata"]["gpu"]["name"]
    assert len(cases) == len(bench.QUICK.rows) * len(bench.QUICK.slots)
    assert all(c["bit_exact"] for c in cases)
    # One slot cannot prefetch: every block but the one already loaded is a demand load.
    demand_loads = {1: bench.QUICK.blocks}
    assert all(c["misses_per_pass"] == demand_loads.get(c["slots"], 0) for c in cases)
    assert all(c["copies_per_pass"] == bench.QUICK.blocks for c in cases)
    from_file = data["results"]["from_file"]
    assert [b["tier"] for b in from_file["builds"]] == list(bench.QUICK.file_tiers)
    assert {b["tiers"] for b in from_file["builds"]} == {"rrrrrrr", "nrnrnrn", "nnnnnnn"}
    assert len(from_file["cases"]) == len(bench.QUICK.rows) * len(bench.QUICK.file_tiers)
    assert all(c["bit_exact"] for c in from_file["cases"])
    on_disk = {c["tier"]: c["bytes_read"] for c in from_file["cases"]}
    assert on_disk["ram"] == 0 < on_disk["mixed"] < on_disk["nvme"]
    assert list(out_dir.iterdir()) == files, "the temporary model file must be removed"


@pytest.mark.cuda
def test_quick_run_on_a_checkpoint_directory(bench, tmp_path):
    out_dir = tmp_path / "results"
    checkpoint = tmp_path / "toy"
    checkpoint.mkdir()
    bench.write_checkpoint(checkpoint, bench.build_model(bench.QUICK))
    code = bench.main(
        ["--quick", "--out-dir", str(out_dir), "--checkpoint", str(checkpoint),
         "--prefix", "blocks.{}."]
    )  # fmt: skip
    (path,) = out_dir.glob("streaming_*.json")
    run = json.loads(path.read_text())["results"]["checkpoint_run"]
    assert code == 0, run["cases"]
    assert run["checkpoint"]["blocks"] == bench.QUICK.blocks and run["checkpoint"]["files"] == 1
    assert [(b["tier"], b["slots"]) for b in run["builds"]] == [
        ("ram", 1), ("ram", 2), ("mixed", 2), ("nvme", 2),
    ]  # fmt: skip
    assert len(run["cases"]) == 4 * len(bench.QUICK.checkpoint_rows)
    assert all(c["same_output"] for c in run["cases"]), "every tier must compute the same"
    with pytest.raises(ValueError, match="no tensor"):
        bench.main(["--quick", "--out-dir", str(out_dir), "--checkpoint", str(checkpoint)])
