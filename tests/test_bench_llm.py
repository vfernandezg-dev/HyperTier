# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmarks/bench_llm.py (config, comparisons, summary and a quick end-to-end run)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from toys import tiny_causal_lm

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_llm.py"
GiB = 1 << 30


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_llm", BENCH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_quick_config_and_overrides(bench):
    args = ["--model", "m", "--quick", "--prompt-tokens", "8,16", "--references", "sync,resident"]
    cfg = bench.make_config(bench.parse_args(args))
    assert cfg.new_tokens == bench.QUICK.new_tokens and cfg.prompt_tokens == (8, 16)
    assert cfg.references == ("sync", "resident")
    assert bench.make_config(bench.parse_args(["--model", "m"])) == bench.Config()
    assert bench.Config().prompt_tokens == (512, 2048, 4096) and bench.Config().slots == 2
    budget = bench.make_config(bench.parse_args(["--model", "m", "--ram-budget-gib", "4"]))
    assert budget.ram_budget_gib == 4.0
    for bad in (["--model", "m", "--references", "gpu"], ["--prompt-tokens", "0"], []):
        with pytest.raises(SystemExit):
            bench.parse_args(bad)


def test_every_worker_gets_the_cublas_workspace_and_stays_offline(bench, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    env = bench.worker_env()
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and env["HF_HUB_OFFLINE"] == "1"


def test_references_that_can_run_are_picked(bench):
    cfg = bench.Config()
    small, large = 1 * GiB, 15 * GiB
    picked = bench.pick_references(cfg, small, 3 * GiB)
    assert picked[:2] == ["sync", "resident"] and picked[-1] == "cpu"
    assert "resident" not in bench.pick_references(cfg, large, 3 * GiB)
    assert "resident" not in bench.pick_references(cfg, 2 * GiB, 3 * GiB), "no room to run it"
    only = bench.dataclasses.replace(cfg, references=("resident", "sync"))
    assert bench.pick_references(only, large, 0) == ["sync", "resident"]
    none = bench.dataclasses.replace(cfg, references=("none",))
    assert bench.pick_references(none, small, 3 * GiB) == []
    # accelerate gets enough VRAM to place something beside the largest module it offloads
    # (with too little it runs on the CPU), and little enough that a small model must offload.
    assert bench._gpu_share(cfg, large, 4 * GiB) == int(2.8 * GiB)
    assert bench._gpu_share(cfg, small, 4 * GiB) == int(0.4 * GiB)
    fixed = bench.dataclasses.replace(cfg, accelerate_gpu_gib=2.0)
    assert bench._gpu_share(fixed, small, 4 * GiB) == 2 * GiB


def test_logits_are_compared_up_to_the_first_different_token(bench):
    torch = pytest.importorskip("torch")
    a = torch.arange(20.0).view(5, 4)
    ours = {"tokens": [7, 8, 9, 10, 11], "seconds_per_token": 1.0}

    same = bench.compare(ours, {"tokens": [7, 8, 9, 10, 11], "seconds_per_token": 3.0}, a, a.clone())
    assert same["identical_tokens"] and same["bit_exact_logits"]
    assert same["token_match_rate"] == 1.0 and same["first_difference"] is None
    assert same["max_logit_difference"] == 0.0 and same["seconds_per_token"] == 3.0

    b = a.clone()
    b[1, 2] += 0.25  # a last-bit difference that does not change the token ...
    b[2, 0] += 5.0  # ... and the step where the other model picks another one
    b[4] += 100.0  # after that the prompts differ: not compared
    other = {"tokens": [7, 8, 3, 10, 11], "seconds_per_token": 2.0}
    diff = bench.compare(ours, other, a, b)
    assert not diff["identical_tokens"] and not diff["bit_exact_logits"]
    assert diff["first_difference"] == 2 and diff["logit_steps_compared"] == 3
    assert diff["token_match_rate"] == 0.8 and diff["max_logit_difference"] == 5.0

    short = bench.compare(ours, {"tokens": [7, 8], "seconds_per_token": 2.0}, a, a[:2])
    assert not short["identical_tokens"], "one stopped early"
    assert short["token_match_rate"] == 1.0 and short["bit_exact_logits"]


def test_prompts_fall_back_to_random_ids_without_a_tokenizer(bench, tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    # Only a config: transformers still builds a tokenizer, with an empty vocabulary, that
    # turns any text into no tokens. That must not be mistaken for a prompt (nor loop).
    transformers.Qwen3Config(vocab_size=1500, hidden_size=64, num_hidden_layers=2).save_pretrained(
        tmp_path
    )
    cfg = bench.dataclasses.replace(bench.QUICK, prompt_tokens=(8, 40))
    prompts = bench.make_prompts(tmp_path, cfg, vocab=1500)
    assert prompts["source"] == "random ids"
    assert len(prompts["pool"]) == 40 and len(prompts["generate"]) == 16
    assert all(0 <= token < 1500 for token in prompts["pool"] + prompts["generate"])
    assert bench.make_prompts(tmp_path, cfg, vocab=1500) == prompts, "the same for every worker"


def _row(tokens, total, copy, compute, **extra):
    row = {
        "tokens": tokens, "wall_ms": total + 2, "total_ms": total, "copy_ms": copy,
        "compute_ms": compute, "stall_ms": 1.0, "read_ms": 0.0, "in_situ_efficiency": 0.9,
        "handoff_p99_ms": 0.3, "peak_vram_bytes": 900 << 20,
    }  # fmt: skip
    return {**row, **extra}


def test_prefill_efficiency_is_measured_against_the_one_slot_pass(bench):
    serial = [_row(512, 2900, 2300, 600), _row(2048, 4700, 2300, 2400), _row(4096, 0, 0, 0)]
    serial[2] = {"tokens": 4096, "error": "out of memory"}
    streamed = [_row(512, 2320, 2300, 650), _row(2048, 2700, 2300, 2500), _row(4096, 5200, 1, 1)]
    resident = [{"tokens": 512, "wall_ms": 580.0}]
    table = bench.prefill_table(serial, streamed, resident)

    first, second, third = table
    # Copy-bound: 580 of the 600 ms of compute were hidden behind the copies.
    assert first["efficiency"] == pytest.approx((2300 + 600 - 2320) / 600)
    assert first["bound_fraction"] == pytest.approx(2300 / 2320)
    assert first["serial_ms"] == 2900 and first["streamed_ms"] == 2320
    assert first["resident_ms"] == 580.0
    assert first["efficiency_vs_resident"] == pytest.approx((2300 + 580 - 2322) / 580)
    # Compute-bound: of the 2300 ms of copies, 2000 were hidden behind compute.
    assert second["efficiency"] == pytest.approx((2300 + 2400 - 2700) / 2300)
    assert "resident_ms" not in second
    assert third == {"tokens": 4096, "error": "one-slot pass: out of memory"}
    # Blocks read from disk: a third thing to overlap. The pass is bound by the reads, and
    # hid 2780 of the 2860 ms that are not the longest of the three.
    from_disk = bench.prefill_table(
        [_row(512, 3200, 2300, 560, read_ms=3240)], [_row(512, 3320, 2300, 600, read_ms=3240)]
    )[0]
    assert from_disk["efficiency"] == pytest.approx((2300 + 560 + 3240 - 3320) / (2300 + 560))
    assert from_disk["bound_fraction"] == pytest.approx(3240 / 3320)
    assert bench.hidden_share((5.0, 0.0), 5.0) is None, "nothing to hide"
    assert bench.prefill_table([], streamed[:1]) == [{"tokens": 512, "error": "one-slot pass: not run"}]


def test_the_criterion_needs_tokens_vram_and_overlap(bench):
    def results(tokens=True, vram=0.8e9, efficiencies=(0.95, 0.85)):
        return {
            "references": {"sync": {"identical_tokens": tokens}},
            "streamed": {"weights_vram_measured_bytes": vram},
            "prefill": [{"tokens": 1, "efficiency": e} for e in efficiencies],
        }

    good = bench.criterion(results())
    assert good["passed"] and good["prefill_efficiency_min"] == 0.85
    assert not bench.criterion(results(tokens=False))["passed"]
    assert not bench.criterion(results(vram=1.3e9))["weights_vram_ok"]
    low = bench.criterion(results(efficiencies=(0.95, 0.79)))
    assert not low["prefill_efficiency_ok"] and not low["passed"]
    # Nothing measured is not a pass: a size that ran out of memory has no efficiency.
    empty = bench.criterion({**results(), "prefill": [{"tokens": 1, "error": "out of memory"}]})
    assert empty["prefill_efficiency_ok"] is None and not empty["passed"]
    assert not bench.criterion({**results(), "references": {}})["passed"]


def test_summary_shows_every_reference_and_every_prefill_size(bench):
    table = bench.prefill_table(
        [_row(512, 2900, 2300, 600), _row(4096, 9, 9, 9)],
        [_row(512, 2320, 2300, 650), {"tokens": 4096, "error": "out of memory"}],
    )
    reference = {
        "identical_tokens": True, "token_match_rate": 1.0, "bit_exact_logits": True,
        "max_logit_difference": 0.0, "seconds_per_token": 6.5,
    }  # fmt: skip
    results = {
        "model": {"name": "Tiny-1B", "architecture": "TinyLM", "weights_bytes": 15 << 30,
                  "files": 5},
        "streamed": {
            "blocks": 40, "head_rows": [[0, 1]] * 4, "slots": 2, "slot_bytes": 368 << 20,
            "tiers": "r" * 40, "weights_vram_measured_bytes": 736 << 20,
            "weights_vram_bytes": 736 << 20, "ram_bytes": 14 << 30, "load_seconds": 14.4,
            "generation": {
                "tokens": [1] * 32, "decode_ms": 2335.0, "bytes_per_token": 14 << 30,
                "decode_copy_ms": 2280.0, "decode_compute_ms": 98.0, "peak_vram_bytes": 800 << 20,
                "seconds_per_token": 2.4, "decode_ms_max": 2390.0, "handoff_ms": 0.15,
                "handoff_p99_worst_ms": 55.8,
            },
        },
        "references": {
            "sync": reference,
            "accelerate": {**reference, "identical_tokens": False, "token_match_rate": 0.97,
                           "bit_exact_logits": False, "max_logit_difference": 0.125},
            "resident": {"error": "OutOfMemoryError: no"},
            "cpu": {**reference, "seconds_per_token": 0.46,
                    "prefill": [{"tokens": 512, "wall_ms": 91234.0, "peak_vram_bytes": 0}]},
        },
        "prefill": table,
    }  # fmt: skip
    results["criterion"] = bench.criterion(results)
    lines = bench.format_summary({"results": results}).splitlines()
    assert any("Tiny-1B (TinyLM): 15.0 GiB of weights in 5 file(s)" in line for line in lines)
    assert any("40 blocks (4 of them the output projection), 2 slot(s)" in line for line in lines)
    rows = {line.split()[0]: line.split() for line in lines if line.strip()}
    assert rows["sync"][1:5] == ["yes", "1.00", "yes", "0"]
    assert rows["accelerate"][1:5] == ["NO", "0.97", "NO", "0.125"]
    assert rows["resident"][1] == "failed:"
    assert rows["512"][1:6] == ["2300", "600", "0", "2320", "0.97"]
    assert rows["4096"][1:] == ["out", "of", "memory"]
    assert any("On the CPU, 512 tokens: 91234 ms" in line for line in lines)
    assert any("slowest token 2390 ms" in line for line in lines)
    assert any("0.15 ms median, worst p99 of a token 55.8 ms" in line for line in lines)
    assert any("weights in VRAM <= 1.2 GB" in line and "yes (0.77 GB)" in line for line in lines)
    assert any("efficiency >= 80%" in line and "yes (lowest 0.97)" in line for line in lines)
    assert bench.format_summary({}).strip().startswith("HyperTier language model benchmark")


@pytest.mark.cuda
def test_quick_run_end_to_end(bench, disk_tmp, tmp_path, capsys):
    pytest.importorskip("transformers")
    tiny_causal_lm(disk_tmp)  # no tokenizer in it: the prompts are random ids
    out_dir = tmp_path / "results"
    code = bench.main(
        ["--model", str(disk_tmp), "--quick", "--references", "sync", "--out-dir", str(out_dir)]
    )
    assert code == 0, "the tokens must equal those of the strict baseline"
    (path,) = out_dir.glob("llm_*.json")
    assert list(out_dir.iterdir()) == [path], "the scratch directory of the workers is removed"
    payload = json.loads(path.read_text())
    assert payload["schema"] == bench.SCHEMA
    meta = payload["metadata"]
    assert meta["worker_env"] == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    assert meta["transformers"] and meta["torch"]["version"] and meta["gpu"]["name"]
    res = payload["results"]
    assert res["model"]["name"] == disk_tmp.name and str(disk_tmp.parent) not in path.read_text()
    assert res["prompt"]["source"] == "random ids"
    streamed = res["streamed"]
    assert streamed["slots"] == 2 and res["serial"]["slots"] == 1
    assert streamed["blocks"] == 5 + len(streamed["head_rows"]) and len(streamed["head_rows"]) > 1
    assert set(streamed["tiers"]) == {"r"}
    assert streamed["weights_vram_measured_bytes"] <= streamed["weights_vram_bytes"] + 4096
    assert len(streamed["generation"]["tokens"]) == bench.QUICK.new_tokens
    strict = res["references"]["sync"]
    assert strict["identical_tokens"] and strict["bit_exact_logits"]
    assert strict["logit_steps_compared"] == bench.QUICK.new_tokens
    assert [row["tokens"] for row in res["prefill"]] == list(bench.QUICK.prompt_tokens)
    for row in res["prefill"]:
        assert row["serial_copy_ms"] > 0 and row["serial_compute_ms"] > 0 and row["streamed_ms"] > 0
        assert row["peak_vram_bytes"] > 0
    assert res["criterion"]["identical_tokens"] is True
    assert "F1 exit criterion" in capsys.readouterr().out
