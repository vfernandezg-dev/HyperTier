# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure a streamed causal language model against its baselines (HyperTier F1).

The model is loaded with ``hypertier.adapters.load_causal_lm`` and checked on the three
things F1 has to deliver:

* **The same tokens.** Greedy generation is compared with a *strict baseline*: the stock
  ``transformers`` model doing the same operations, each layer copied to the GPU
  synchronously right before it runs. No slots, events, page-locked pool or reader thread.
  The logits of every step should be the same bits. Other references are compared by token
  match rate and largest logit difference, since they do not do the same arithmetic: the
  stock model resident in VRAM (when it fits), the stock model with ``accelerate``
  offloading, and the stock model on the CPU. The last one is also the speed to beat: a CPU
  reads weights from RAM several times faster than they cross PCIe.
* **Little VRAM.** Bytes of weights held in VRAM, computed and measured.
* **Overlap in the prefill.** A prompt of N tokens is run with ``logits_to_keep=1``. The run
  with one slot, where no copy can overlap compute, gives the copy and compute times; the
  run with the requested slots is compared with them:
  ``efficiency = (copy + compute - streamed) / min(copy, compute)``. With blocks read from
  disk there is a third thing to overlap, and the same ratio is taken over the three: of
  the time that is not the longest of them, the share that the pass hid.

Every measurement runs in its own process, with ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in all of
them, so nothing shares VRAM, page-locked RAM or allocator state with anything else.
Everything, plus the metadata needed to reproduce the run, goes to
``benchmarks/results/llm_<UTC timestamp>.json``; a summary is printed.

    python benchmarks/bench_llm.py --model "$HYPERTIER_MODELS/Qwen3-8B"     # ~10 min
    python benchmarks/bench_llm.py --model "$HYPERTIER_MODELS/Qwen3-0.6B"   # ~1 min
    python benchmarks/bench_llm.py --model DIR --quick                      # smoke test
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hypertier.telemetry import GiB, MiB, cuda_available, format_bytes
from hypertier.telemetry import bandwidth as bw

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # tests load this file by path, without benchmarks/ on sys.path
    sys.path.insert(0, str(_HERE))
import bench_hw  # sibling script: metadata and result files are shared

REPO_ROOT = _HERE.parent
SCHEMA = "hypertier.bench_llm/1"
CUBLAS_WORKSPACE = ":4096:8"
REFERENCES = ("sync", "resident", "accelerate", "cpu")

# F1 exit criterion (docs: roadmap). The numbers are checked, the verdict is only printed:
# a model whose output projection is a large share of its bytes (Qwen3-0.6B: 26 %) cannot
# hide those copies behind compute with two slots, and that is not a defect of the run.
MAX_WEIGHTS_VRAM = 1.2e9
MIN_PREFILL_EFFICIENCY = 0.8

PROMPT = "Explain, step by step, how a computer moves data between fast and slow memory."
FILLER = (
    "A tiered memory system keeps the data that is needed next in the fastest tier and moves "
    "everything else to slower and larger tiers. The cost of a pass is set by how many bytes "
    "have to cross the slowest link, not by how much memory the fastest tier has. "
)


@dataclasses.dataclass(frozen=True)
class Config:
    """Every knob of a run; stored in the JSON so results can be reproduced."""

    slots: int = 2
    num_stage: int = 2
    ram_budget_gib: float | None = None  # None: every streamed block stays in page-locked RAM
    new_tokens: int = 32  # greedy tokens compared with the baselines
    prompt_tokens: tuple[int, ...] = (512, 2048, 4096)  # prefill sizes
    prefill_passes: int = 3
    warmup_passes: int = 1
    references: tuple[str, ...] = ("auto",)
    accelerate_gpu_gib: float | None = None  # VRAM accelerate may fill; None: see _gpu_share
    cpu_prefill: bool = True  # also time the smallest prompt on the CPU (minutes for 8B)
    seed: int = 0


QUICK = Config(new_tokens=6, prompt_tokens=(48, 96), prefill_passes=2)


def _ints(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError(f"expected positive integers, got {text!r}")
    return values


def _names(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = sorted(set(values) - {*REFERENCES, "auto", "none"})
    if unknown or not values:
        raise argparse.ArgumentTypeError(
            f"unknown reference(s) {unknown}; choose from auto, none, {', '.join(REFERENCES)}"
        )
    return values


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HyperTier language model benchmark (F1).")
    parser.add_argument("--model", type=Path, help="model directory (config + safetensors)")
    parser.add_argument("--quick", action="store_true", help="tiny sizes: a fast smoke test")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "benchmarks" / "results",
        help="where the JSON goes (default: benchmarks/results/)",
    )  # fmt: skip
    parser.add_argument("--slots", type=int, help="VRAM slots (default 2)")
    parser.add_argument("--num-stage", type=int, help="staging buffers (default 2)")
    parser.add_argument(
        "--ram-budget-gib", type=float,
        help="page-locked RAM for the weights; less than the model forces the NVMe tier",
    )  # fmt: skip
    parser.add_argument("--new-tokens", type=int, help="greedy tokens to compare (default 32)")
    parser.add_argument("--prompt-tokens", type=_ints, help="prefill sizes, comma separated")
    parser.add_argument("--prefill-passes", type=int, help="timed passes per size (default 3)")
    parser.add_argument(
        "--references", type=_names,
        help="baselines to run, comma separated: sync, resident, accelerate, cpu, auto "
        "(those that can run here; the default) or none",
    )  # fmt: skip
    parser.add_argument("--accelerate-gpu-gib", type=float, help="VRAM accelerate may fill")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)  # a job file: see run_job
    args = parser.parse_args(argv)
    if args.worker is None and args.model is None:
        parser.error("--model is required")
    return args


def make_config(args: argparse.Namespace) -> Config:
    cfg = QUICK if args.quick else Config()
    names = (
        "slots", "num_stage", "ram_budget_gib", "new_tokens", "prompt_tokens",
        "prefill_passes", "references", "accelerate_gpu_gib",
    )  # fmt: skip
    changes = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    return dataclasses.replace(cfg, **changes)


# ---------------------------------------------------------------------------------------------
# Workers: each runs in its own process (see run_job)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _generate(torch: Any, model: Any, ids: Sequence[int], new_tokens: int) -> dict[str, Any]:
    """Greedy tokens and the raw logits of every step, as ``generate()`` computes them."""
    x = torch.tensor([list(ids)])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=new_tokens,
            do_sample=False, temperature=None, top_p=None, top_k=None,
            output_logits=True, return_dict_in_generate=True,
        )  # fmt: skip
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    tokens = out.sequences[0, len(ids) :].tolist()
    return {
        "tokens": tokens,
        "logits": torch.stack([step[0] for step in out.logits]).float().cpu(),
        "seconds": seconds,
        "seconds_per_token": seconds / max(len(tokens), 1),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }


def _prefill(
    torch: Any, model: Any, prompts: dict[str, list[int]], passes: int, warmup: int,
    streamer: Any = None, device: str = "cuda",
) -> list[dict[str, Any]]:  # fmt: skip
    """Time ``model(prompt, logits_to_keep=1)`` for every prompt; medians over the passes."""
    rows = []
    for size, ids in prompts.items():
        x = torch.tensor([ids], device=device)
        walls: list[float] = []
        stats: list[Any] = []
        peak = 0
        try:
            for number in range(warmup + passes):
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    out = model(input_ids=x, logits_to_keep=1, use_cache=True)
                torch.cuda.synchronize()
                wall = (time.perf_counter() - t0) * 1e3
                del out
                last = None if streamer is None else streamer.pass_stats()[-1]
                if number >= warmup:
                    walls.append(wall)
                    stats.append(last)
                    peak = max(peak, torch.cuda.max_memory_allocated())
        except torch.OutOfMemoryError:
            rows.append({"tokens": int(size), "error": "out of memory"})
            torch.cuda.empty_cache()
            continue
        row: dict[str, Any] = {
            "tokens": int(size), "wall_ms": _median(walls), "peak_vram_bytes": peak,
        }  # fmt: skip
        if streamer is not None:
            for name in (
                "total_ms", "copy_ms", "compute_ms", "stall_ms", "read_ms", "read_stall_ms",
                "enqueue_ms", "enqueue_p99_ms", "handoff_ms", "handoff_p99_ms",
            ):  # fmt: skip
                row[name] = _median([getattr(s, name) for s in stats])
            row["bytes_copied"] = _median([s.bytes_copied for s in stats])
            row["misses"] = _median([s.misses for s in stats])
            row["in_situ_efficiency"] = _median([s.efficiency or 0.0 for s in stats])
        rows.append(row)
        torch.cuda.empty_cache()
    return rows


def worker_streamed(job: dict[str, Any], torch: Any) -> dict[str, Any]:
    """The model under test: weights streamed through ``job["slots"]`` VRAM slots."""
    from hypertier.adapters import load_causal_lm
    from hypertier.telemetry import read_meminfo

    before = torch.cuda.memory_allocated()
    locked = read_meminfo().get("Shmem", 0)
    t0 = time.perf_counter()
    streamed = load_causal_lm(
        job["model"], slots=job["slots"], ram_budget=job["ram_budget"], num_stage=job["num_stage"]
    )
    streamer = streamed.streamer
    tiers = "".join(tier[0] for tier in streamer.tiers)
    result: dict[str, Any] = {
        "load_seconds": time.perf_counter() - t0,
        "slots": streamer.num_slots,
        "blocks": streamer.num_blocks,
        "slot_bytes": streamer.slot_bytes,
        "streamed_bytes": streamer.streamed_bytes,
        "tiers": tiers,
        "head_rows": [list(rows) for rows in streamed.head_rows],
        "ram_bytes": streamed.ram_bytes,
        "pinned_bytes": read_meminfo().get("Shmem", 0) - locked,
        "embedding_bytes": streamed.embedding_bytes,
        "resident_bytes": streamed.resident_bytes,
        "weights_vram_bytes": streamed.vram_bytes,
        "weights_vram_measured_bytes": torch.cuda.memory_allocated() - before,
    }
    tensors = {}
    try:
        if job.get("generate"):
            generation = _generate(torch, streamed.model, **job["generate"])
            tensors["logits"] = generation.pop("logits")
            passes = streamer.pass_stats()
            decode = passes[1:] or passes  # the first pass is the prompt
            generation["decode_passes"] = len(decode)
            generation["decode_ms"] = _median([s.total_ms for s in decode])
            generation["decode_ms_max"] = max(s.total_ms for s in decode)
            generation["decode_copy_ms"] = _median([s.copy_ms for s in decode])
            generation["decode_compute_ms"] = _median([s.compute_ms for s in decode])
            generation["decode_read_ms"] = _median([s.read_ms for s in decode])
            generation["bytes_per_token"] = _median([s.bytes_copied for s in decode])
            generation["handoff_ms"] = _median([s.handoff_ms for s in decode])
            generation["handoff_p99_worst_ms"] = max(s.handoff_p99_ms for s in decode)
            # Per forward of the model, the prompt first: where a slow token or copy was.
            generation["pass_ms"] = [s.total_ms for s in passes]
            generation["pass_handoff_p99_ms"] = [s.handoff_p99_ms for s in passes]
            result["generation"] = generation
        if job.get("prefill"):
            result["prefill"] = _prefill(torch, streamed.model, streamer=streamer, **job["prefill"])
    finally:
        streamed.close()
    return {"result": result, "tensors": tensors}


def _stock(torch: Any, job: dict[str, Any], **kwargs: Any) -> Any:
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(job["model"], dtype="auto", **kwargs)
    return model.eval()


def worker_resident(job: dict[str, Any], torch: Any) -> dict[str, Any]:
    """The stock model with every weight in VRAM: what streaming is meant to equal."""
    t0 = time.perf_counter()
    model = _stock(torch, job).to("cuda")
    result: dict[str, Any] = {
        "load_seconds": time.perf_counter() - t0,
        "weights_vram_measured_bytes": torch.cuda.memory_allocated(),
    }
    tensors = {}
    if job.get("generate"):
        result["generation"] = _generate(torch, model, **job["generate"])
        tensors["logits"] = result["generation"].pop("logits")
    if job.get("prefill"):
        result["prefill"] = _prefill(torch, model, **job["prefill"])
    return {"result": result, "tensors": tensors}


def worker_accelerate(job: dict[str, Any], torch: Any) -> dict[str, Any]:
    """The stock model, offloaded to RAM by ``accelerate`` beyond ``job["gpu_bytes"]``."""
    import accelerate

    t0 = time.perf_counter()
    limits = {0: job["gpu_bytes"], "cpu": 1 << 40}
    model = _stock(torch, job, device_map="auto", max_memory=limits)
    placed = [str(device) for device in getattr(model, "hf_device_map", {}).values()]
    on_gpu = sum(device.isdigit() for device in placed)
    if not on_gpu:
        # accelerate keeps room on the GPU for the largest module it offloads. When nothing
        # else fits beside that, it leaves the whole model on the CPU and runs it there.
        raise RuntimeError(
            f"accelerate put no module on the GPU with {format_bytes(job['gpu_bytes'])} of "
            "VRAM (--accelerate-gpu-gib): the run would be on the CPU"
        )
    result: dict[str, Any] = {
        "accelerate": accelerate.__version__,
        "load_seconds": time.perf_counter() - t0,
        "gpu_bytes": job["gpu_bytes"],
        "modules_on_gpu": on_gpu,
        "modules_offloaded": len(placed) - on_gpu,
        "weights_vram_measured_bytes": torch.cuda.memory_allocated(),
    }
    tensors = {}
    if job.get("generate"):
        result["generation"] = _generate(torch, model, **job["generate"])
        tensors["logits"] = result["generation"].pop("logits")
    return {"result": result, "tensors": tensors}


def worker_cpu(job: dict[str, Any], torch: Any) -> dict[str, Any]:
    """The stock model on the CPU, weights in RAM: no GPU and no copies at all."""
    t0 = time.perf_counter()
    model = _stock(torch, job)
    result: dict[str, Any] = {
        "load_seconds": time.perf_counter() - t0,
        "threads": torch.get_num_threads(),
        "weights_vram_measured_bytes": 0,
    }
    tensors = {}
    if job.get("generate"):
        result["generation"] = _generate(torch, model, **job["generate"])
        tensors["logits"] = result["generation"].pop("logits")
    if job.get("prefill"):
        result["prefill"] = _prefill(torch, model, device="cpu", **job["prefill"])
    return {"result": result, "tensors": tensors}


def load_sync(torch: Any, model: Any, head_rows: Sequence[Sequence[int]], device: Any) -> Any:
    """Turn a stock model held in RAM into the strict baseline, in place.

    Same modules, same operations and same order as the streamed model, with the simplest
    possible way of getting the weights there: before a layer runs, its parameters are
    copied to the GPU with a blocking ``.to()``; afterwards they point at RAM again. The
    output projection is multiplied in the same vocabulary ranges, and the embedding is
    looked up on the CPU. Nothing of the streaming runtime is involved.
    """
    from hypertier.stream import find_blocks

    nn = torch.nn
    layers = find_blocks(model)
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()

    class Head(nn.Module):
        def forward(self, x: Any) -> Any:
            pieces = []
            for start, stop in head_rows:
                weight = head.weight[start:stop].to(device)
                bias = None if head.bias is None else head.bias[start:stop].to(device)
                pieces.append(nn.functional.linear(x, weight, bias))
            return torch.cat(pieces, dim=-1)

    def attach(layer: Any) -> None:
        home = [(param, param.data) for param in layer.parameters()]

        def load(module: Any, args: Any) -> None:
            for param, data in home:
                param.data = data.to(device)

        def unload(module: Any, args: Any, output: Any) -> None:
            for param, data in home:
                param.data = data

        layer.register_forward_pre_hook(load)
        layer.register_forward_hook(unload, always_call=True)

    for layer in layers:
        attach(layer)
    skip = {id(m) for m in (*layers.modules(), *embedding.modules(), *head.modules())}
    for module in model.modules():
        if id(module) in skip:
            continue
        for param in module._parameters.values():
            if param is not None:
                param.data = param.data.to(device)
        for key, buf in module._buffers.items():
            if buf is not None:
                module._buffers[key] = buf.to(device)
    embedding.register_forward_pre_hook(lambda module, args: tuple(a.to("cpu") for a in args))
    embedding.register_forward_hook(lambda module, args, output: output.to(device))
    model.set_output_embeddings(Head())
    # generate() sends its inputs to model.device: the first parameter, on the CPU.
    cls = type(model)
    model.__class__ = type(cls.__name__, (cls,), {"device": property(lambda self: device)})
    return model


def worker_sync(job: dict[str, Any], torch: Any) -> dict[str, Any]:
    """The strict baseline (see :func:`load_sync`)."""
    t0 = time.perf_counter()
    model = load_sync(torch, _stock(torch, job), job["head_rows"], torch.device("cuda", 0))
    result: dict[str, Any] = {
        "load_seconds": time.perf_counter() - t0,
        "weights_vram_measured_bytes": torch.cuda.memory_allocated(),
    }
    tensors = {}
    if job.get("generate"):
        result["generation"] = _generate(torch, model, **job["generate"])
        tensors["logits"] = result["generation"].pop("logits")
    return {"result": result, "tensors": tensors}


def worker_metadata(job: dict[str, Any], torch: Any) -> dict[str, Any]:
    """Hardware and versions, gathered here so that the parent never opens a CUDA context."""
    return {"result": bench_hw.collect_metadata(), "tensors": {}}


WORKERS = {
    "streamed": worker_streamed, "sync": worker_sync, "resident": worker_resident,
    "accelerate": worker_accelerate, "cpu": worker_cpu, "metadata": worker_metadata,
}  # fmt: skip


def run_job(path: Path) -> int:
    """Entry point of a worker process: run the job in ``path`` and write what it asks for."""
    import torch

    job = json.loads(path.read_text())
    try:
        import transformers

        transformers.utils.logging.disable_progress_bar()
        output = WORKERS[job["mode"]](job, torch)
    except Exception as exc:  # reported to the parent, which decides what it means
        Path(job["out"]).write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        raise
    if output["tensors"]:
        torch.save(output["tensors"], job["tensors"])
    Path(job["out"]).write_text(json.dumps(output["result"]))
    return 0


# ---------------------------------------------------------------------------------------------
# Orchestration: the parent never opens a CUDA context, which would take VRAM from the workers


def worker_env() -> dict[str, str]:
    env = dict(os.environ)
    env["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
    env["HF_HUB_OFFLINE"] = "1"  # local directories only: nothing is looked up online
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def launch(job: dict[str, Any], scratch: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run ``job`` in a new process; return its result and its tensors."""
    import torch

    name = job["name"]
    job = {**job, "out": str(scratch / f"{name}.json"), "tensors": str(scratch / f"{name}.pt")}
    path = scratch / f"{name}.job.json"
    path.write_text(json.dumps(job))
    print(f"[bench_llm] {name} ...", file=sys.stderr, flush=True)
    done = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(path)],
        env=worker_env(), check=False,
    )  # fmt: skip
    out = Path(job["out"])
    result = json.loads(out.read_text()) if out.exists() else {}
    if done.returncode != 0 and "error" not in result:
        result = {"error": f"worker exited with code {done.returncode}"}
    tensors = Path(job["tensors"])
    return result, (torch.load(tensors) if tensors.exists() else {})


def make_prompts(model_dir: Path, cfg: Config, vocab: int) -> dict[str, Any]:
    """Token ids for generation and for every prefill size, the same for every worker.

    With a tokenizer in the model directory they come from text. Without one (a bare
    checkpoint, a test model) they are random ids: enough to time and to compare outputs.
    """
    import torch
    import transformers

    longest = max(cfg.prompt_tokens)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
        prompt = tokenizer(PROMPT).input_ids
        # A directory without tokenizer files can still give a tokenizer, one with an empty
        # vocabulary that turns any text into no tokens: count before trusting it.
        per_copy = len(tokenizer(FILLER).input_ids)
        pool = tokenizer(FILLER * (longest // per_copy + 2)).input_ids if per_copy else []
    except Exception:  # noqa: BLE001 - whatever stops a tokenizer means "there is none"
        prompt, pool = [], []
    if prompt and len(pool) >= longest:
        return {"source": "text", "generate": prompt, "pool": pool[:longest]}
    generator = torch.Generator().manual_seed(cfg.seed)
    ids = torch.randint(0, vocab, (longest + 16,), generator=generator).tolist()
    return {"source": "random ids", "generate": ids[longest:], "pool": ids[:longest]}


def compare(ours: dict[str, Any], theirs: dict[str, Any], a: Any, b: Any) -> dict[str, Any]:
    """Tokens and logits of two greedy generations from the same prompt.

    Logits are compared only up to the first step where the tokens differ: from there on the
    two models are answering different prompts.
    """
    import torch

    mine, other = ours["tokens"], theirs["tokens"]
    common = min(len(mine), len(other))
    agree = [x == y for x, y in zip(mine, other)]
    first = agree.index(False) if False in agree else common
    steps = min(first + 1, common, len(a), len(b))
    diff = (a[:steps] - b[:steps]).abs()
    return {
        "tokens_compared": common,
        "identical_tokens": len(mine) == len(other) and all(agree),
        "token_match_rate": sum(agree) / common if common else 0.0,
        "first_difference": None if first == common else first,
        "logit_steps_compared": steps,
        "bit_exact_logits": bool(steps) and bool(torch.equal(a[:steps], b[:steps])),
        "max_logit_difference": float(diff.max()) if steps else None,
        "seconds_per_token": theirs["seconds_per_token"],
    }


def inventory(model_dir: Path) -> dict[str, Any]:
    """Sizes read from the checkpoint headers and the config: no weight is loaded."""
    from hypertier.io import SafetensorsIndex

    index = SafetensorsIndex(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    return {
        "name": model_dir.resolve().name,  # not the path: results may be shared
        "architecture": (config.get("architectures") or [config.get("model_type")])[0],
        "vocab_size": config.get("vocab_size"),
        "weights_bytes": index.nbytes,
        "tensors": len(index),
        "files": len(index.files),
    }


def _gpu_share(cfg: Config, weights_bytes: int, free_vram: int) -> int:
    """VRAM handed to accelerate: enough to work, little enough that it has to offload.

    It needs room for what it places on the GPU plus the largest module it offloads, so too
    small a share leaves everything on the CPU (the worker refuses to measure that).
    """
    if cfg.accelerate_gpu_gib is not None:
        return int(cfg.accelerate_gpu_gib * GiB)
    return int(min(0.7 * free_vram, 0.4 * weights_bytes))


def pick_references(cfg: Config, weights_bytes: int, free_vram: int) -> list[str]:
    if "none" in cfg.references:
        return []
    if "auto" not in cfg.references:
        return [name for name in REFERENCES if name in cfg.references]
    names = ["sync"]
    if 1.3 * weights_bytes + GiB // 2 < free_vram:  # the weights, activations and a margin
        names.append("resident")
    try:
        import accelerate  # noqa: F401
    except ImportError:
        pass
    else:
        names.append("accelerate")
    return [*names, "cpu"]


def hidden_share(parts: Sequence[float], total: float) -> float | None:
    """Of the time that overlapping could hide, the share that a pass of ``total`` hid.

    ``parts`` are the times of the things that can run at once (copies, compute, disk
    reads). One after the other they take their sum; perfectly overlapped, the longest. For
    two parts this is ``telemetry.bandwidth.overlap_efficiency``.
    """
    serial, longest = sum(parts), max(parts)
    return (serial - total) / (serial - longest) if serial > longest else None


def prefill_table(
    serial: Sequence[dict], streamed: Sequence[dict], resident: Sequence[dict] = ()
) -> list[dict[str, Any]]:
    """Per prompt size, the streamed pass against the one-slot pass, where nothing overlaps.

    When the model also ran resident in VRAM, its pass is the compute time nobody slowed
    down, and ``efficiency_vs_resident`` uses it instead of the one-slot compute.
    """
    by_size = {row["tokens"]: row for row in serial}
    in_vram = {row["tokens"]: row["wall_ms"] for row in resident if "error" not in row}
    table = []
    for row in streamed:
        base = by_size.get(row["tokens"])
        problem = row.get("error")
        if problem is None and (base is None or "error" in base):
            problem = "one-slot pass: " + (base or {}).get("error", "not run")
        if problem is not None:
            table.append({"tokens": row["tokens"], "error": problem})
            continue
        copy_ms, compute_ms = base["copy_ms"], base["compute_ms"]
        read_ms = row.get("read_ms", 0.0)
        bound = max(copy_ms, compute_ms, read_ms)
        table.append({
            "tokens": row["tokens"],
            "serial_ms": base["total_ms"],
            "serial_copy_ms": copy_ms,
            "serial_compute_ms": compute_ms,
            "streamed_ms": row["total_ms"],
            "wall_ms": row["wall_ms"],
            "read_ms": read_ms,
            "stall_ms": row["stall_ms"],
            "efficiency": hidden_share((copy_ms, compute_ms, read_ms), row["total_ms"]),
            "in_situ_efficiency": row["in_situ_efficiency"],
            "bound_fraction": bound / row["total_ms"] if row["total_ms"] else None,
            "handoff_p99_ms": row["handoff_p99_ms"],
            "peak_vram_bytes": row["peak_vram_bytes"],
        })  # fmt: skip
        if row["tokens"] in in_vram:
            whole = in_vram[row["tokens"]]
            table[-1]["resident_ms"] = whole
            table[-1]["efficiency_vs_resident"] = bw.overlap_efficiency(
                copy_ms, whole, row["wall_ms"]
            )
    return table


def criterion(results: dict[str, Any]) -> dict[str, Any]:
    """The three checks of the F1 exit criterion, for whatever model was run."""
    strict = results["references"].get("sync", {})
    vram = results["streamed"].get("weights_vram_measured_bytes")
    measured = [row["efficiency"] for row in results["prefill"] if row.get("efficiency") is not None]
    lowest = min(measured) if measured else None
    checks = {
        "identical_tokens": strict.get("identical_tokens"),
        "weights_vram_bytes": vram,
        "weights_vram_ok": None if vram is None else vram <= MAX_WEIGHTS_VRAM,
        "prefill_efficiency_min": lowest,
        "prefill_efficiency_ok": None if lowest is None else lowest >= MIN_PREFILL_EFFICIENCY,
    }
    verdicts = ("identical_tokens", "weights_vram_ok", "prefill_efficiency_ok")
    checks["passed"] = all(checks[name] is True for name in verdicts)
    return checks


def run(cfg: Config, model_dir: Path, scratch: Path, free_vram: int) -> dict[str, Any]:
    model = inventory(model_dir)
    prompts = make_prompts(model_dir, cfg, model["vocab_size"] or 2)
    generate = {"ids": prompts["generate"], "new_tokens": cfg.new_tokens}
    prefill = {
        "prompts": {str(size): prompts["pool"][:size] for size in cfg.prompt_tokens},
        "passes": cfg.prefill_passes,
        "warmup": cfg.warmup_passes,
    }
    budget = None if cfg.ram_budget_gib is None else int(cfg.ram_budget_gib * GiB)
    base = {"model": str(model_dir), "num_stage": cfg.num_stage, "ram_budget": budget}

    streamed, logits = launch(
        {**base, "name": "streamed", "mode": "streamed", "slots": cfg.slots,
         "generate": generate, "prefill": prefill}, scratch,
    )  # fmt: skip
    if "error" in streamed:
        raise RuntimeError(f"the streamed model failed: {streamed['error']}")
    serial, _ = launch(
        {**base, "name": "serial", "mode": "streamed", "slots": 1, "prefill": prefill}, scratch
    )
    references: dict[str, Any] = {}
    resident_prefill = None
    for name in pick_references(cfg, model["weights_bytes"], free_vram):
        job = {**base, "name": name, "mode": name, "generate": generate}
        if name == "sync":
            job["head_rows"] = streamed["head_rows"]
        elif name == "accelerate":
            job["gpu_bytes"] = _gpu_share(cfg, model["weights_bytes"], free_vram)
        elif name == "resident":
            job["prefill"] = prefill
        elif name == "cpu" and cfg.cpu_prefill:
            smallest = str(min(cfg.prompt_tokens))  # once: it takes minutes on a large model
            job["prefill"] = {
                "prompts": {smallest: prefill["prompts"][smallest]}, "passes": 1, "warmup": 0,
            }  # fmt: skip
        other, other_logits = launch(job, scratch)
        if "error" in other:
            references[name] = other
            continue
        entry = compare(
            streamed["generation"], other["generation"], logits["logits"], other_logits["logits"]
        )
        entry["load_seconds"] = other["load_seconds"]
        entry["peak_vram_bytes"] = other["generation"]["peak_vram_bytes"]
        for key in ("accelerate", "gpu_bytes", "modules_on_gpu", "modules_offloaded", "threads"):
            if key in other:
                entry[key] = other[key]
        if name == "resident":
            resident_prefill = other.get("prefill")
        elif "prefill" in other:
            entry["prefill"] = other["prefill"]
        references[name] = entry
    results: dict[str, Any] = {
        "model": model,
        "prompt": {
            "source": prompts["source"],
            "generate_tokens": len(prompts["generate"]),
            "text": PROMPT if prompts["source"] == "text" else None,
        },
        "streamed": streamed,
        "serial": serial,
        "references": references,
        "prefill": prefill_table(
            serial.get("prefill", []), streamed.get("prefill", []), resident_prefill or ()
        ),
    }
    results["criterion"] = criterion(results)
    return results


# ---------------------------------------------------------------------------------------------
# Summary


def _yes(value: Any) -> str:
    return "-" if value is None else "yes" if value else "NO"


def format_summary(payload: dict[str, Any]) -> str:
    res = payload.get("results") or {}
    out = ["HyperTier language model benchmark", "=" * 34]
    if not res:
        return "\n".join(out)
    model, streamed = res["model"], res["streamed"]
    gen = streamed.get("generation", {})
    out += [
        (
            f"{model['name']} ({model['architecture']}): {format_bytes(model['weights_bytes'])} "
            f"of weights in {model['files']} file(s)"
        ),
        (
            f"streamed: {streamed['blocks']} blocks ({len(streamed['head_rows'])} of them the "
            f"output projection), {streamed['slots']} slot(s) of "
            f"{format_bytes(streamed['slot_bytes'])}, tiers {streamed['tiers']}"
        ),
        (
            f"weights in VRAM: {format_bytes(streamed['weights_vram_measured_bytes'])} measured, "
            f"{format_bytes(streamed['weights_vram_bytes'])} computed; page-locked RAM "
            f"{format_bytes(streamed['ram_bytes'])}; loaded in {streamed['load_seconds']:.1f} s"
        ),
    ]
    if gen:
        out += [
            "",
            (
                f"Generation, {len(gen['tokens'])} greedy tokens: {gen['decode_ms']:.0f} ms per "
                f"token ({format_bytes(gen['bytes_per_token'])} copied in "
                f"{gen['decode_copy_ms']:.0f} ms, compute {gen['decode_compute_ms']:.0f} ms; "
                f"slowest token {gen['decode_ms_max']:.0f} ms), "
                f"peak VRAM {format_bytes(gen['peak_vram_bytes'])}"
            ),
            (
                f"From asking for a copy to having it enqueued: {gen['handoff_ms']:.2f} ms "
                f"median, worst p99 of a token {gen['handoff_p99_worst_ms']:.1f} ms"
            ),
            (
                f"{'reference':<12}{'same tokens':>12}{'match':>7}{'bit-exact':>11}"
                f"{'max |dlogit|':>14}{'s/token':>9}"
            ),
        ]
        for name, ref in res["references"].items():
            if "error" in ref:
                out.append(f"{name:<12}failed: {ref['error']}")
                continue
            worst = ref["max_logit_difference"]
            out.append(
                f"{name:<12}{_yes(ref['identical_tokens']):>12}{ref['token_match_rate']:>7.2f}"
                f"{_yes(ref['bit_exact_logits']):>11}"
                f"{('-' if worst is None else f'{worst:.4g}'):>14}{ref['seconds_per_token']:>9.2f}"
            )
        out.append(f"{'streamed':<12}{'':>44}{gen['seconds_per_token']:>9.2f}")
    if res.get("prefill"):
        out += [
            "",
            "Prefill with logits_to_keep=1, medians, ms. copy and compute are those of the pass",
            "with one slot, where they do not overlap. eff = (copy + compute - streamed) /",
            "min(copy, compute), over copy, compute and read when blocks come from disk.",
            "bound = max(copy, compute, read) / streamed.",
            (
                f"{'tokens':>7}{'copy':>8}{'compute':>9}{'read':>7}{'streamed':>10}{'eff':>6}"
                f"{'bound':>7}{'stall':>7}{'handoff p99':>13}{'peak VRAM':>11}"
                f"{'resident':>10}{'eff':>6}"
            ),
        ]
        for row in res["prefill"]:
            if "error" in row:
                out.append(f"{row['tokens']:>7}  {row['error']}")
                continue
            eff = row["efficiency"]
            line = (
                f"{row['tokens']:>7}{row['serial_copy_ms']:>8.0f}{row['serial_compute_ms']:>9.0f}"
                f"{row['read_ms']:>7.0f}{row['streamed_ms']:>10.0f}"
                f"{('-' if eff is None else f'{eff:.2f}'):>6}"
                f"{row['bound_fraction']:>7.2f}{row['stall_ms']:>7.0f}"
                f"{row['handoff_p99_ms']:>13.2f}{row['peak_vram_bytes'] / MiB:>8.0f} MiB"
            )
            if "resident_ms" in row:
                line += f"{row['resident_ms']:>10.0f}{row['efficiency_vs_resident']:>6.2f}"
            out.append(line)
        for row in (res["references"].get("cpu") or {}).get("prefill", []):
            if "error" not in row:
                out.append(f"On the CPU, {row['tokens']} tokens: {row['wall_ms']:.0f} ms")
    check = res.get("criterion") or {}
    if check:
        vram, lowest = check["weights_vram_bytes"], check["prefill_efficiency_min"]
        out += [
            "",
            "F1 exit criterion (set for a 7-8B model in bf16):",
            f"  same tokens as the strict baseline      {_yes(check['identical_tokens'])}",
            (
                f"  weights in VRAM <= {MAX_WEIGHTS_VRAM / 1e9:.1f} GB              "
                f"{_yes(check['weights_vram_ok'])}"
                + ("" if vram is None else f" ({vram / 1e9:.2f} GB)")
            ),
            (
                f"  prefill overlap efficiency >= {MIN_PREFILL_EFFICIENCY:.0%}       "
                f"{_yes(check['prefill_efficiency_ok'])}"
                + ("" if lowest is None else f" (lowest {lowest:.2f})")
            ),
        ]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker is not None:
        return run_job(args.worker)
    cfg = make_config(args)
    if not cuda_available():
        print("bench_llm needs a CUDA GPU", file=sys.stderr)
        return 2
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the tokenizer is read from the directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="llm-", dir=args.out_dir))
    try:
        metadata, _ = launch({"name": "metadata", "mode": "metadata"}, scratch)
        if "error" in metadata:
            raise RuntimeError(f"could not read the hardware metadata: {metadata['error']}")
        metadata["worker_env"] = {"CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE}
        metadata["transformers"] = _package_version("transformers")
        metadata["accelerate"] = _package_version("accelerate")
        payload: dict[str, Any] = {
            "schema": SCHEMA, "metadata": metadata, "config": dataclasses.asdict(cfg),
        }  # fmt: skip
        free_vram = metadata["gpu"]["free_memory_bytes_at_start"]
        payload["results"] = run(cfg, args.model, scratch, free_vram)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    path = bench_hw.write_results(payload, args.out_dir, prefix="llm")
    print(format_summary(payload))
    print(f"\nResults written to {path}")
    strict = payload["results"]["references"].get("sync")
    return 0 if strict is None or strict.get("identical_tokens") else 1


def _package_version(name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    sys.exit(main())
