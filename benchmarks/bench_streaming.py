# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure ping-pong layer streaming against the same model resident in VRAM (HyperTier F1).

A residual MLP small enough to fit in VRAM is run twice: fully resident, and with its blocks
streamed from page-locked RAM through ``LayerStreamer``. The batch size sets the compute per
block, so the sweep goes from transfer-bound passes (copies longer than compute) to
compute-bound ones, for each number of VRAM slots. Per case it reports the pass time, how much
of the shorter side was hidden (``overlap_efficiency`` against the resident pass, which also
counts any slowdown the copies cause), the distance to the bound ``max(compute, copy)``, the
VRAM held for weights, and whether the output is bit-exact.

Everything, plus the metadata needed to reproduce the run, goes to
``benchmarks/results/streaming_<UTC timestamp>.json``; a summary table is printed.

    python benchmarks/bench_streaming.py            # ~1 min, ~1 GiB of RAM, ~0.9 GiB of VRAM
    python benchmarks/bench_streaming.py --quick    # tiny sizes, smoke test
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hypertier.stream import LayerStreamer
from hypertier.telemetry import bandwidth as bw
from hypertier.telemetry import cuda_available, format_bytes
from hypertier.telemetry.vram import track_vram

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # tests load this file by path, without benchmarks/ on sys.path
    sys.path.insert(0, str(_HERE))
import bench_hw  # sibling script: metadata and result files are shared

REPO_ROOT = _HERE.parent
SCHEMA = "hypertier.bench_streaming/1"


@dataclasses.dataclass(frozen=True)
class Config:
    """Every knob of a run; stored in the JSON so results can be reproduced."""

    blocks: int = 7  # odd on purpose: the case where ``i % slots`` breaks
    width: int = 4096  # two width x width linears per block: 64 MiB in fp16
    dtype: str = "float16"
    rows: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)  # batch size: compute per block
    slots: tuple[int, ...] = (1, 2, 3)
    passes: int = 10
    warmup_passes: int = 2
    warmup_seconds: float = 1.0
    seed: int = 0


QUICK = Config(
    width=256, rows=(64, 512), slots=(1, 2), passes=3, warmup_passes=1, warmup_seconds=0.2
)


def _ints(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError(f"expected positive integers, got {text!r}")
    return values


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HyperTier layer streaming benchmark (F1).")
    parser.add_argument("--quick", action="store_true", help="tiny sizes: a fast smoke test")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "benchmarks" / "results",
        help="where the JSON goes (default: benchmarks/results/)",
    )
    parser.add_argument("--blocks", type=int, help="streamed blocks (default 7)")
    parser.add_argument("--width", type=int, help="hidden size; a block is 4 x width^2 bytes")
    parser.add_argument("--rows", type=_ints, help="batch sizes, comma separated")
    parser.add_argument("--slots", type=_ints, help="VRAM slot counts, comma separated")
    parser.add_argument("--passes", type=int, help="timed passes per case (default 10)")
    return parser.parse_args(argv)


def make_config(args: argparse.Namespace) -> Config:
    cfg = QUICK if args.quick else Config()
    names = ("blocks", "width", "rows", "slots", "passes")
    changes = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    return dataclasses.replace(cfg, **changes)


# ---------------------------------------------------------------------------------------------
# Model and measurements


def build_model(cfg: Config) -> Any:
    """Residual MLP on the CPU; ``model.blocks`` is what gets streamed."""
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(cfg.width, cfg.width)
            self.fc2 = nn.Linear(cfg.width, cfg.width)

        def forward(self, x: Any) -> Any:
            return x + 0.1 * self.fc2(torch.relu(self.fc1(x)))

    class Stack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(Block() for _ in range(cfg.blocks))

        def forward(self, x: Any) -> Any:
            for block in self.blocks:
                x = block(x)
            return x

    torch.manual_seed(cfg.seed)
    return Stack().to(getattr(torch, cfg.dtype)).eval()


def _median(values: Sequence[float]) -> float:
    return bw.percentile(values, 50.0)


def time_passes(model: Any, x: Any, passes: int, warmup: int) -> tuple[float, Any]:
    """Median wall time in ms of one pass, until the GPU is idle again; and the last output."""
    import torch

    times = []
    out = None
    with torch.no_grad():
        for index in range(warmup + passes):
            t0 = time.perf_counter()
            out = model(x)
            torch.cuda.synchronize()
            if index >= warmup:
                times.append((time.perf_counter() - t0) * 1e3)
    return _median(times), out


def measure_resident(model: Any, inputs: dict[int, Any], cfg: Config) -> dict[int, dict[str, Any]]:
    """Reference numbers and outputs with every weight in VRAM."""
    import torch

    reference = copy.deepcopy(model).cuda()
    weights = sum(p.numel() * p.element_size() for p in reference.parameters())
    results = {}
    for rows, x_cpu in inputs.items():
        x = x_cpu.cuda()
        with track_vram() as vram:  # weights were loaded before: this is the activations
            pass_ms, out = time_passes(reference, x, cfg.passes, cfg.warmup_passes)
            expected = out.cpu()
            del out
        if not bool(torch.isfinite(expected).all()):
            raise RuntimeError(f"the reference output is not finite with {rows} rows")
        results[rows] = {
            "rows": rows,
            "pass_ms": pass_ms,
            "weights_vram_bytes": weights,
            "activations_peak_bytes": vram.stats.peak_delta,
            "expected": expected,
        }
    del reference
    torch.cuda.empty_cache()
    return results


def measure_streamed(
    model: Any, inputs: dict[int, Any], resident: dict[int, dict[str, Any]], slots: int, cfg: Config
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Stream the blocks through ``slots`` VRAM slots; one result per batch size."""
    import torch

    t0 = time.perf_counter()
    streamer = LayerStreamer(model.blocks, slots=slots)
    build = {
        "slots": streamer.num_slots,
        "slot_bytes": streamer.slot_bytes,
        "streamed_bytes": streamer.streamed_bytes,
        "weights_vram_bytes": streamer.vram_bytes,
        "build_seconds": time.perf_counter() - t0,
        "pinned_bytes": streamer._pool.capacity_bytes,  # page-locked RAM holding the weights
        "pin_seconds": streamer._pool.pin_seconds,
    }
    cases = []
    with streamer:
        for rows, x_cpu in inputs.items():
            x = x_cpu.cuda()
            base = resident[rows]
            with track_vram() as vram:  # slots were reserved before: this is the activations
                pass_ms, out = time_passes(model, x, cfg.passes, cfg.warmup_passes)
                exact = torch.equal(out.cpu(), base["expected"])
                del out
            stats = streamer.pass_stats()[-cfg.passes :]
            copy_ms = _median([s.copy_ms for s in stats])
            in_situ = [s.efficiency for s in stats if s.efficiency is not None]
            bound_ms = max(base["pass_ms"], copy_ms)
            cases.append(
                {
                    "rows": rows,
                    "slots": streamer.num_slots,
                    "resident_ms": base["pass_ms"],
                    "streamed_ms": pass_ms,
                    "copy_ms": copy_ms,
                    "compute_ms_in_situ": _median([s.compute_ms for s in stats]),
                    "stall_ms": _median([s.stall_ms for s in stats]),
                    "copies_per_pass": _median([s.copies for s in stats]),
                    "bytes_per_pass": _median([s.bytes_copied for s in stats]),
                    "copy_gbps": bw.gb_per_s(_median([s.bytes_copied for s in stats]), copy_ms),
                    "misses_per_pass": _median([s.misses for s in stats]),
                    # Both against the resident pass: the copies' effect on compute counts.
                    "efficiency": bw.overlap_efficiency(copy_ms, base["pass_ms"], pass_ms),
                    "efficiency_in_situ": _median(in_situ) if in_situ else None,
                    "bound_ms": bound_ms,  # max(compute, copy): the law's lower bound
                    "bound_fraction": bound_ms / pass_ms,
                    "slowdown": pass_ms / base["pass_ms"],
                    "activations_peak_bytes": vram.stats.peak_delta,
                    "bit_exact": exact,
                }
            )
    return build, cases


# ---------------------------------------------------------------------------------------------
# Summary table


def format_summary(payload: dict[str, Any]) -> str:
    """Human-readable digest of a results payload."""
    meta = payload.get("metadata") or {}
    gpu = meta.get("gpu") or {}
    smi = gpu.get("nvidia_smi") or {}
    tmeta = meta.get("torch") or {}
    res = payload.get("results") or {}
    out = ["", f"HyperTier layer streaming  {meta.get('timestamp_utc', '')}"]
    if gpu:
        out.append(
            f"GPU    {gpu.get('name')} ({gpu.get('capability')}), driver "
            f"{smi.get('driver_version', '?')}, VRAM free "
            f"{format_bytes(gpu.get('free_memory_bytes_at_start', 0))} of "
            f"{format_bytes(gpu.get('total_memory_bytes', 0))}"
        )
    out.append(
        f"SW     torch {tmeta.get('version', '-')} (CUDA {tmeta.get('cuda', '-')}), "
        f"Python {meta.get('python', '-')}{', WSL2' if meta.get('wsl') else ''}"
    )
    model = res.get("model")
    if model:
        out.append(
            f"Model  {model['blocks']} blocks x {format_bytes(model['block_bytes'])} "
            f"({model['dtype']}), {format_bytes(model['weights_bytes'])} of weights; resident "
            f"model holds all of them in VRAM"
        )
    for build in res.get("builds") or []:
        out.append(
            f"       {build['slots']} slot(s): {format_bytes(build['weights_vram_bytes'])} of "
            f"VRAM for weights; {format_bytes(build['pinned_bytes'])} pinned in "
            f"{build['pin_seconds']:.2f} s, built in {build['build_seconds']:.2f} s"
        )
    cases = res.get("cases") or []
    if cases:
        out += [
            "",
            "Per pass, medians. eff = share of the shorter of (copy, resident compute) hidden;",
            "bound = max(resident compute, copy) / streamed; miss = blocks loaded on demand",
            "instead of prefetched. Times in ms.",
            (
                f"{'rows':>6}{'slots':>6}{'resident':>10}{'copy':>8}{'streamed':>10}{'stall':>8}"
                f"{'eff':>7}{'bound':>7}{'GB/s':>7}{'miss':>6}  exact"
            ),
        ]
        for c in cases:
            out.append(
                f"{c['rows']:>6}{c['slots']:>6}{c['resident_ms']:>10.1f}{c['copy_ms']:>8.1f}"
                f"{c['streamed_ms']:>10.1f}{c['stall_ms']:>8.1f}{c['efficiency']:>7.2f}"
                f"{c['bound_fraction']:>7.2f}{c['copy_gbps']:>7.2f}{c['misses_per_pass']:>6.0f}  "
                f"{'yes' if c['bit_exact'] else 'NO'}"
            )
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------


def run(cfg: Config) -> dict[str, Any]:
    import torch

    model = build_model(cfg)
    generator = torch.Generator().manual_seed(cfg.seed)
    inputs = {
        rows: torch.randn(rows, cfg.width, generator=generator).to(getattr(torch, cfg.dtype))
        for rows in cfg.rows
    }
    block_bytes = sum(p.numel() * p.element_size() for p in model.blocks[0].parameters())
    bw.warm_up_gpu(cfg.warmup_seconds)
    print("[bench_streaming] resident model ...", file=sys.stderr, flush=True)
    resident = measure_resident(model, inputs, cfg)
    builds, cases = [], []
    for slots in cfg.slots:
        print(f"[bench_streaming] {slots} slot(s) ...", file=sys.stderr, flush=True)
        build, slot_cases = measure_streamed(model, inputs, resident, slots, cfg)
        builds.append(build)
        cases += slot_cases
    for entry in resident.values():
        del entry["expected"]
    return {
        "model": {
            "blocks": cfg.blocks,
            "block_bytes": block_bytes,
            "weights_bytes": block_bytes * cfg.blocks,
            "dtype": cfg.dtype,
        },
        "resident": list(resident.values()),
        "builds": builds,
        "cases": cases,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = make_config(args)
    if not cuda_available():
        print("bench_streaming needs a CUDA GPU", file=sys.stderr)
        return 2
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "metadata": bench_hw.collect_metadata(),
        "config": dataclasses.asdict(cfg),
        "results": run(cfg),
    }
    path = bench_hw.write_results(payload, args.out_dir, prefix="streaming")
    print(format_summary(payload))
    print(f"\nResults written to {path}")
    return 0 if all(case["bit_exact"] for case in payload["results"]["cases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
