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
        "misses": 0, "bit_exact": True,
    }
    payload = {
        "metadata": {"timestamp_utc": "2026-11-22T00:00:00+00:00"},
        "results": {
            "model": {"blocks": 7, "block_bytes": 64 << 20, "weights_bytes": 448 << 20,
                      "dtype": "float16"},
            "builds": [{"slots": 2, "weights_vram_bytes": 128 << 20, "pinned_bytes": 896 << 20,
                        "pin_seconds": 0.5, "build_seconds": 0.8}],
            "cases": [case, {**case, "slots": 1, "bit_exact": False}],
        },
    }
    lines = bench.format_summary(payload).splitlines()
    assert any("7 blocks x 64.0 MiB" in line for line in lines)
    assert any("2 slot(s): 128.0 MiB of VRAM" in line for line in lines)
    rows = [line.split() for line in lines if line.strip().startswith("8192")]
    assert [row[1] for row in rows] == ["2", "1"]
    assert rows[0][2:7] == ["275.5", "70.7", "279.1", "0.0", "0.95"]
    assert rows[0][-1] == "yes" and rows[1][-1] == "NO"
    assert bench.format_summary({}).strip().startswith("HyperTier layer streaming")


@pytest.mark.cuda
def test_quick_run_writes_json(bench, tmp_path):
    out_dir = tmp_path / "results"
    code = bench.main(["--quick", "--out-dir", str(out_dir)])
    files = list(out_dir.glob("streaming_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    cases = data["results"]["cases"]
    assert code == 0, [c for c in cases if not c["bit_exact"]]
    assert data["schema"] == bench.SCHEMA and data["metadata"]["gpu"]["name"]
    assert len(cases) == len(bench.QUICK.rows) * len(bench.QUICK.slots)
    assert all(c["bit_exact"] and c["misses"] == 0 or c["slots"] == 1 for c in cases)
    assert all(c["copies_per_pass"] == bench.QUICK.blocks for c in cases)
