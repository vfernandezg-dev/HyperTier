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

The same model is then saved as a safetensors file and streamed from it, with its blocks in
page-locked RAM, on disk (read every pass through staging buffers) or half and half. That adds
the disk read rate, the distance to ``max(compute, copy, read)`` and how long a copy waits
between being asked for and being enqueued by the reader thread.

``--checkpoint DIR`` runs the file part on a real checkpoint instead, e.g. the 36 layers of
368 MiB of Qwen3-8B, which do not fit in VRAM: each block does one matmul per stored matrix,
the outputs must be identical in every tier, and the pass with one slot, where nothing
overlaps, gives the copy and compute times that the others are compared with.

Everything, plus the metadata needed to reproduce the run, goes to
``benchmarks/results/streaming_<UTC timestamp>.json``; a summary table is printed.

    python benchmarks/bench_streaming.py            # ~2 min, ~1.5 GiB of RAM, ~0.9 GiB of VRAM
    python benchmarks/bench_streaming.py --quick    # tiny sizes, smoke test
    python benchmarks/bench_streaming.py --checkpoint "$HYPERTIER_MODELS/Qwen3-8B"
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import dataclasses
import json
import re
import shutil
import struct
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hypertier.io import SafetensorsIndex
from hypertier.stream import FileWeights, LayerStreamer
from hypertier.telemetry import bandwidth as bw
from hypertier.telemetry import cuda_available, format_bytes
from hypertier.telemetry.vram import track_vram

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # tests load this file by path, without benchmarks/ on sys.path
    sys.path.insert(0, str(_HERE))
import bench_hw  # sibling script: metadata and result files are shared

REPO_ROOT = _HERE.parent
SCHEMA = "hypertier.bench_streaming/2"


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
    # Weights read from a safetensors file:
    file_tiers: tuple[str, ...] = ("ram", "mixed", "nvme")
    file_slots: int = 2
    num_stage: int = 2
    # --checkpoint: tokens per pass (compute per block) and timed passes
    checkpoint_rows: tuple[int, ...] = (256, 2048)
    checkpoint_passes: int = 3


QUICK = Config(
    width=256, rows=(64, 512), slots=(1, 2), passes=3, warmup_passes=1, warmup_seconds=0.2,
    checkpoint_rows=(8, 64), checkpoint_passes=2,
)  # fmt: skip


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
    parser.add_argument(
        "--model-dir", type=Path, default=REPO_ROOT / "benchmarks" / "results",
        help="disk to stream the saved model from; a temporary file is written there",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        help="stream the blocks of this safetensors checkpoint instead of the toy model",
    )
    parser.add_argument(
        "--prefix", default="model.layers.{}.",
        help="tensor name prefix of block {} in --checkpoint (default: model.layers.{}.)",
    )
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
# Weights read from files


def write_checkpoint(directory: Path, model: Any) -> Path:
    """Save ``model`` as one safetensors file, tensors sorted by name like the format does."""
    names = {"torch.float16": "F16", "torch.bfloat16": "BF16", "torch.float32": "F32"}
    header: dict[str, Any] = {}
    blobs = []
    offset = 0
    for name, value in sorted(model.state_dict().items()):
        nbytes = value.numel() * value.element_size()
        header[name] = {
            "dtype": names[str(value.dtype)],
            "shape": list(value.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        blobs.append(ctypes.string_at(value.contiguous().data_ptr(), nbytes))
        offset += nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-(8 + len(raw)) % 8)
    path = directory / "model.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        f.writelines(blobs)
    return path


def tier_budget(host_bytes: Sequence[int], tier: str, num_stage: int) -> int | None:
    """A RAM budget that keeps all, about half or none of the blocks in page-locked RAM."""
    staging = num_stage * max(host_bytes)
    if tier == "ram":
        return None
    return staging + (sum(host_bytes) // 2 if tier == "mixed" else 0)


def _file_case(stats: Sequence[Any], pass_ms: float) -> dict[str, Any]:
    """What the file-backed cases report on top of the times of a pass (medians)."""
    read_ms = _median([s.read_ms for s in stats])
    bytes_read = _median([s.bytes_read for s in stats])
    return {
        "streamed_ms": pass_ms,
        "copy_ms": _median([s.copy_ms for s in stats]),
        "bytes_per_pass": _median([s.bytes_copied for s in stats]),
        "read_ms": read_ms,
        "bytes_read": bytes_read,
        "read_gbps": bw.gb_per_s(bytes_read, read_ms) if read_ms > 0 else None,
        "read_stall_ms": _median([s.read_stall_ms for s in stats]),
        "stall_ms": _median([s.stall_ms for s in stats]),
        "misses_per_pass": _median([s.misses for s in stats]),
        # From asking for a copy to having it enqueued; "handoff" is the part of it spent
        # with the bytes already in RAM. p99: the worst per-pass p99 seen.
        "enqueue_ms": _median([s.enqueue_ms for s in stats]),
        "enqueue_p99_ms": max(s.enqueue_p99_ms for s in stats),
        "handoff_ms": _median([s.handoff_ms for s in stats]),
        "handoff_p99_ms": max(s.handoff_p99_ms for s in stats),
    }


def measure_from_file(
    model: Any,
    inputs: dict[int, Any],
    resident: dict[int, dict[str, Any]],
    directory: Path,
    tier: str,
    cfg: Config,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Stream the saved model from ``directory``; one result per batch size."""
    import torch

    blocks = copy.deepcopy(model).blocks  # attaching replaces the weights with placeholders
    prefixes = [f"blocks.{i}." for i in range(len(blocks))]
    plans = FileWeights(directory, prefixes).plan(torch, blocks)
    limit = tier_budget([plan.host_bytes for plan in plans], tier, cfg.num_stage)
    weights = FileWeights(directory, prefixes, ram_budget=limit, num_stage=cfg.num_stage)
    t0 = time.perf_counter()
    streamer = LayerStreamer(blocks, slots=cfg.file_slots, weights=weights)
    build = {
        "tier": tier,
        "tiers": "".join(t[0] for t in streamer.tiers),
        "slots": streamer.num_slots,
        "num_stage": cfg.num_stage,
        "ram_budget": limit,
        "ram_bytes": streamer.ram_bytes,
        "pinned_bytes": streamer._pool.capacity_bytes,
        "weights_vram_bytes": streamer.vram_bytes,
        "build_seconds": time.perf_counter() - t0,
    }

    def forward(x: Any) -> Any:
        for block in blocks:
            x = block(x)
        return x

    cases = []
    with streamer:
        for rows, x_cpu in inputs.items():
            x = x_cpu.cuda()
            base = resident[rows]
            pass_ms, out = time_passes(forward, x, cfg.passes, cfg.warmup_passes)
            exact = torch.equal(out.cpu(), base["expected"])
            del out
            case = _file_case(streamer.pass_stats()[-cfg.passes :], pass_ms)
            bound_ms = max(base["pass_ms"], case["copy_ms"], case["read_ms"])
            case.update(
                rows=rows, tier=tier, slots=streamer.num_slots, resident_ms=base["pass_ms"],
                bound_ms=bound_ms, bound_fraction=bound_ms / pass_ms, bit_exact=exact,
            )
            cases.append(case)
    return build, cases


def checkpoint_blocks(index: SafetensorsIndex, template: str, inputs: dict[int, Any]) -> Any:
    """Modules with exactly the tensors stored under each block prefix, as placeholders.

    A block's forward does one matmul per stored matrix on ``inputs[in_features]`` and adds
    the result to a running sum: about the compute of a prefill over that many tokens, and a
    single number that depends on every weight, to compare between tiers.
    """
    import torch
    from torch import nn

    from hypertier.stream.source import torch_dtype

    pattern = re.compile(re.escape(template).replace(r"\{\}", r"(\d+)"))
    groups: dict[int, list[Any]] = {}
    for name in index:
        match = pattern.match(name)
        if match:
            groups.setdefault(int(match.group(1)), []).append(index[name])
    if not groups:
        raise ValueError(f"no tensor of the checkpoint matches {template!r}")

    class Stored(nn.Module):
        def __init__(self, entries: list[Any], prefix: str) -> None:
            super().__init__()
            self.matrices: list[tuple[Any, str]] = []
            for entry in sorted(entries, key=lambda e: (str(e.path), e.offset)):
                *path, leaf = entry.name[len(prefix) :].split(".")
                owner = self
                for part in path:
                    if part not in owner._modules:
                        owner.add_module(part, nn.Module())
                    owner = owner._modules[part]
                empty = torch.empty(
                    entry.shape, dtype=torch_dtype(torch, entry.dtype), device="meta"
                )
                owner.register_parameter(leaf, nn.Parameter(empty, requires_grad=False))
                if len(entry.shape) == 2:
                    self.matrices.append((owner, leaf))

        def forward(self, total: Any) -> Any:
            for owner, leaf in self.matrices:
                weight = getattr(owner, leaf)
                total = total + (inputs[weight.shape[1]] @ weight.T).float().sum()
            return total

    numbers = sorted(groups)
    prefixes = [template.format(i) for i in numbers]
    blocks = nn.ModuleList(Stored(groups[i], prefix) for i, prefix in zip(numbers, prefixes))
    return blocks, prefixes


def run_checkpoint(cfg: Config, root: Path, template: str) -> dict[str, Any]:
    """Stream the blocks of a real checkpoint from disk in every tier."""
    import torch

    index = SafetensorsIndex(root)
    inputs: dict[int, Any] = {}
    blocks, prefixes = checkpoint_blocks(index, template, inputs)
    plans = FileWeights(index, prefixes).plan(torch, blocks)
    host = [plan.host_bytes for plan in plans]
    widths = sorted({p.shape[1] for p in blocks.parameters() if p.dim() == 2})
    dtype = next(blocks.parameters()).dtype
    bw.warm_up_gpu(cfg.warmup_seconds)

    def forward(_: Any) -> Any:
        total = torch.zeros((), device="cuda")
        for block in blocks:
            total = block(total)
        return total

    builds, cases = [], []
    # One slot first: copy and compute strictly alternate, so they are measured alone.
    for tier, slots in (("ram", 1), ("ram", 2), ("mixed", 2), ("nvme", 2)):
        note = f"[bench_streaming] checkpoint, {tier}, {slots} slot(s) ..."
        print(note, file=sys.stderr, flush=True)
        limit = tier_budget(host, tier, cfg.num_stage)
        weights = FileWeights(index, prefixes, ram_budget=limit, num_stage=cfg.num_stage)
        t0 = time.perf_counter()
        streamer = LayerStreamer(blocks, slots=slots, weights=weights)
        builds.append(
            {
                "tier": tier,
                "tiers": "".join(t[0] for t in streamer.tiers),
                "slots": streamer.num_slots,
                "num_stage": cfg.num_stage,
                "ram_budget": limit,
                "ram_bytes": streamer.ram_bytes,
                "pinned_bytes": streamer._pool.capacity_bytes,
                "weights_vram_bytes": streamer.vram_bytes,
                "build_seconds": time.perf_counter() - t0,
            }
        )
        with streamer:
            for rows in cfg.checkpoint_rows:
                generator = torch.Generator().manual_seed(cfg.seed + rows)
                inputs.clear()
                for width in widths:
                    x = torch.randn(rows, width, generator=generator) * 0.02
                    inputs[width] = x.to(dtype).cuda()
                pass_ms, out = time_passes(forward, None, cfg.checkpoint_passes, 1)
                stats = streamer.pass_stats()[-cfg.checkpoint_passes :]
                case = _file_case(stats, pass_ms)
                case.update(
                    rows=rows, tier=tier, slots=streamer.num_slots, output=float(out),
                    compute_ms_in_situ=_median([s.compute_ms for s in stats]),
                )
                cases.append(case)
        inputs.clear()
    for case in cases:
        # The one-slot pass of the same batch: nothing overlaps there.
        serial = next(c for c in cases if c["slots"] == 1 and c["rows"] == case["rows"])
        copy_ms, compute_ms = serial["copy_ms"], serial["compute_ms_in_situ"]
        bound_ms = max(compute_ms, copy_ms, case["read_ms"])
        case.update(
            serial_copy_ms=copy_ms, serial_compute_ms=compute_ms, bound_ms=bound_ms,
            bound_fraction=bound_ms / case["streamed_ms"],
            efficiency=bw.overlap_efficiency(copy_ms, compute_ms, case["streamed_ms"]),
            same_output=case["output"] == serial["output"],
        )
    return {
        "checkpoint": {
            "name": root.name,
            "blocks": len(blocks),
            "block_bytes": max(plan.size for plan in plans),
            "weights_bytes": sum(plan.size for plan in plans),
            "dtype": str(dtype).removeprefix("torch."),
            "files": len(index.files),
        },
        "builds": builds,
        "cases": cases,
    }


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
    from_file = res.get("from_file") or res.get("checkpoint_run")
    if from_file:
        out += _file_summary(from_file)
    return "\n".join(out)


def _file_summary(part: dict[str, Any]) -> list[str]:
    """The table of a run that streams from files (the saved toy model or a checkpoint)."""
    out = [""]
    ckpt = part.get("checkpoint")
    if ckpt:
        out.append(
            f"Checkpoint {ckpt['name']}: {ckpt['blocks']} blocks of up to "
            f"{format_bytes(ckpt['block_bytes'])} ({ckpt['dtype']}), "
            f"{format_bytes(ckpt['weights_bytes'])} in {ckpt['files']} file(s)"
        )
    else:
        out.append("The same model streamed from a safetensors file on disk")
    out.append("(per block: r = kept in page-locked RAM, n = read from disk on every pass):")
    for build in part.get("builds") or []:
        budget = build["ram_budget"]
        blocks = build["tiers"]
        if len(blocks) > 12:
            blocks = f"{blocks.count('r')} r + {blocks.count('n')} n"
        out.append(
            f"       {build['tier']:<5} {build['slots']} slot(s): {blocks}, "
            f"{format_bytes(build['ram_bytes'])} locked"
            f"{'' if budget is None else ' (budget ' + format_bytes(budget) + ')'}, "
            f"built in {build['build_seconds']:.1f} s"
        )
    cases = part.get("cases") or []
    if not cases:
        return out
    reference = "1-slot compute" if ckpt else "resident"
    out += [
        "",
        f"Per pass, medians. bound = max({reference}, copy, read) / streamed; enq = from asking",
        "for a copy to having it enqueued, median / worst p99; handoff = the part of it with the",
        "bytes already in RAM. Times in ms.",
        (
            f"{'rows':>6} {'tier':<5}{'slots':>6}{'compute':>9}{'copy':>8}{'read':>8}"
            f"{'streamed':>10}{'bound':>7}{'rd GB/s':>9}{'enq':>14}{'handoff':>14}  ok"
        ),
    ]
    for c in cases:
        compute = c.get("resident_ms", c.get("serial_compute_ms"))
        read = c["read_gbps"]
        ok = c.get("bit_exact", c.get("same_output"))
        out.append(
            f"{c['rows']:>6} {c['tier']:<5}{c['slots']:>6}{compute:>9.1f}{c['copy_ms']:>8.1f}"
            f"{c['read_ms']:>8.1f}{c['streamed_ms']:>10.1f}{c['bound_fraction']:>7.2f}"
            f"{(f'{read:.2f}' if read else '-'):>9}"
            f"{c['enqueue_ms']:>7.2f} /{c['enqueue_p99_ms']:>5.1f}"
            f"{c['handoff_ms']:>7.2f} /{c['handoff_p99_ms']:>5.1f}  {'yes' if ok else 'NO'}"
        )
    return out


# ---------------------------------------------------------------------------------------------


def run(cfg: Config, model_dir: Path) -> dict[str, Any]:
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
    file_builds, file_cases = [], []
    model_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="streaming-model-", dir=model_dir))
    try:
        write_checkpoint(scratch, model)
        for tier in cfg.file_tiers:
            print(f"[bench_streaming] from file, {tier} ...", file=sys.stderr, flush=True)
            build, tier_cases = measure_from_file(model, inputs, resident, scratch, tier, cfg)
            file_builds.append(build)
            file_cases += tier_cases
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
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
        "from_file": {"builds": file_builds, "cases": file_cases},
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
    }
    if args.checkpoint is not None:
        results = {"checkpoint_run": run_checkpoint(cfg, args.checkpoint, args.prefix)}
        checked = [case["same_output"] for case in results["checkpoint_run"]["cases"]]
    else:
        results = run(cfg, args.model_dir)
        checked = [case["bit_exact"] for case in results["cases"] + results["from_file"]["cases"]]
    payload["results"] = results
    path = bench_hw.write_results(payload, args.out_dir, prefix="streaming")
    print(format_summary(payload))
    print(f"\nResults written to {path}")
    return 0 if all(checked) else 1


if __name__ == "__main__":
    sys.exit(main())
