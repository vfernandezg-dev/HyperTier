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
