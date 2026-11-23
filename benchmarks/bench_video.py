# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a video with a streamed Wan pipeline and hold it against the F3 criterion.

The pipeline is ``hypertier.adapters.load_wan_pipeline``: text encoder, transformer and VAE
run one after another, each loaded only while it runs. Three things are checked, each run in
a process of its own:

* **It fits.** The generation runs under ``--vram-budget-gb`` (3.4 by default, the budget of
  the criterion), and the VRAM the process holds is sampled from the driver all the way
  through: the peak has to stay under that budget. Per stage: time, the peaks of the
  allocator, and for the transformer how well the copies hid behind compute.
* **The same latents.** The first ``--check-steps`` steps are sampled twice more from the
  same text embeddings: by the streamed transformer, and by a *strict baseline*, the
  transformer as ``diffusers`` loads it with every block copied to the GPU, blocking, right
  before it runs. The two have to be equal bit for bit. In bfloat16 nothing less means
  anything: one unit in the last place becomes another video after a few steps.
* **The frames, within the tiling's PSNR.** The first ``--psnr-frames`` frames are decoded
  again without tiles, in float32, on the CPU (the only place it fits), and the PSNR of the
  tiled decode against them has to reach the threshold of the VAE configuration in use.

Timings are of a warm GPU: it is kept busy for ``--warm-up`` seconds first, and the
temperature goes into the result. Everything, with the metadata needed to reproduce the run,
goes to ``benchmarks/results/video_<UTC timestamp>.json``; a summary is printed. The video
itself is written next to it under the same name: numbers do not show a broken picture. It
is an ``.mp4`` when ``diffusers`` can encode one (``imageio`` with ffmpeg, or OpenCV), and
otherwise an animated ``.webp``, which Pillow always writes and a browser plays.

Run it from a checkout that nothing edits meanwhile (``git worktree add`` at a commit): every
measurement is a new process that imports the code as it is on disk at that moment.

    python benchmarks/bench_video.py --model "$HYPERTIER_MODELS/Wan2.2-TI2V-5B-Diffusers"
    python benchmarks/bench_video.py --model DIR --height 480 --width 832 --frames 121
    python benchmarks/bench_video.py --model DIR --quick        # 480p x 17 frames, 4 steps
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
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hypertier.telemetry import MiB, cuda_available

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # tests load this file by path, without benchmarks/ on sys.path
    sys.path.insert(0, str(_HERE))
import bench_hw  # sibling script: metadata and result files are shared

if TYPE_CHECKING:
    from typing_extensions import Self

SCHEMA = "hypertier.bench_video/1"
CUBLAS_WORKSPACE = ":4096:8"
GB = 1000**3
PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a "
    "spotlighted stage."
)
NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


@dataclasses.dataclass(frozen=True)
class Config:
    height: int = 704
    width: int = 1280
    frames: int = 49
    steps: int = 50
    guidance: float = 5.0
    seed: int = 0
    slots: int = 2
    prompt: str = PROMPT
    negative: str = NEGATIVE
    vram_budget_gb: float = 3.4
    vae_config: str | None = None
    check_steps: int = 2
    psnr_frames: int = 9
    warm_up: float = 60.0
    fps: int = 24
    save_video: bool = True


QUICK = {"height": 480, "width": 832, "frames": 17, "steps": 4, "check_steps": 1,
         "psnr_frames": 5, "warm_up": 5.0}  # fmt: skip


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, help="a diffusers WanPipeline directory")
    parser.add_argument("--quick", action="store_true", help="a short video, for a smoke test")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--slots", type=int)
    parser.add_argument("--prompt")
    parser.add_argument("--negative")
    parser.add_argument("--vram-budget-gb", type=float, help="0: whatever is free")
    parser.add_argument("--vae-config", help="a VAE configuration by name, instead of the best")
    parser.add_argument("--check-steps", type=int, help="steps compared with the baseline; 0: none")
    parser.add_argument("--psnr-frames", type=int, help="frames compared untiled; 0: none")
    parser.add_argument("--warm-up", type=float, help="seconds of GPU load before timing")
    parser.add_argument("--fps", type=int, help="frames per second of the video file")
    parser.add_argument("--no-video", dest="save_video", action="store_false", default=None,
                        help="do not write the video file")  # fmt: skip
    parser.add_argument("--out-dir", type=Path, default=_HERE / "results")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker is None and args.model is None:
        parser.error("--model is required")
    return args


def make_config(args: argparse.Namespace) -> Config:
    values = dict(QUICK) if args.quick else {}
    for field in dataclasses.fields(Config):
        given = getattr(args, field.name)
        if given is not None:
            values[field.name] = given
    cfg = Config(**values)
    if cfg.steps < 1 or cfg.check_steps < 0 or cfg.psnr_frames < 0:
        raise SystemExit("steps must be positive, and the checks cannot be negative")
    if cfg.psnr_frames and (cfg.psnr_frames - 1) % 4:
        raise SystemExit("--psnr-frames is one frame plus groups of four: 1, 5, 9...")
    return cfg


# ---------------------------------------------------------------------------------------------
# Workers: one process each


class DriverPeak:
    """Lowest free VRAM the driver reports while the block runs, sampled every 10 ms."""

    def __init__(self, torch: Any) -> None:
        self._torch = torch
        self.total = int(torch.cuda.mem_get_info()[1])
        self.min_free = int(torch.cuda.mem_get_info()[0])
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.01):
            self.min_free = min(self.min_free, int(self._torch.cuda.mem_get_info()[0]))

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join()
        self.min_free = min(self.min_free, int(self._torch.cuda.mem_get_info()[0]))


def _temperature() -> int | None:
    found = bench_hw.nvidia_smi(["temperature.gpu"])
    return int(found["temperature.gpu"]) if found else None


def _used_now() -> int | None:
    found = bench_hw.nvidia_smi(["memory.used"])
    return int(found["memory.used"]) * MiB if found else None


def save_video(frames: Any, path: Path, fps: int) -> Path:
    """Write ``uint8`` frames ``[frames, height, width, 3]`` as a video; returns the file.

    ``path`` is taken without its suffix: ``.mp4`` when ``diffusers`` has an encoder for it,
    an animated ``.webp`` written by Pillow otherwise.
    """
    from PIL import Image

    images = [Image.fromarray(frame) for frame in frames.numpy()]
    try:
        from diffusers.utils import export_to_video

        return Path(export_to_video(images, str(path.with_suffix(".mp4")), fps=fps))
    except (ImportError, ValueError):  # no imageio with ffmpeg, and no OpenCV
        target = path.with_suffix(".webp")
        images[0].save(
            target, save_all=True, append_images=images[1:], duration=round(1000 / fps),
            loop=0, quality=90, method=4,
        )  # fmt: skip
        return target


def worker_generate(job: dict[str, Any], torch: Any) -> dict[str, Any]:
    """The whole generation, as a user runs it."""
    from hypertier.adapters import load_wan_pipeline
    from hypertier.telemetry.bandwidth import warm_up_gpu

    cfg = job["config"]
    idle = job["idle_used_bytes"]
    budget = int(cfg["vram_budget_gb"] * GB) or None
    pipeline = load_wan_pipeline(
        job["model"], slots=cfg["slots"], vae_config=cfg["vae_config"], vram_budget=budget
    )
    warm_up_gpu(cfg["warm_up"], device=pipeline.device)
    size = {"height": cfg["height"], "width": cfg["width"], "num_frames": cfg["frames"]}
    result: dict[str, Any] = {"temperature_c": [_temperature()], "idle_used_bytes": idle}
    result["expected_peaks_bytes"] = pipeline.check(**size)
    started = time.perf_counter()
    with DriverPeak(torch) as peak:
        guided = cfg["guidance"] > 1.0
        embeds = pipeline.encode([cfg["prompt"], cfg["negative"]] if guided else [cfg["prompt"]])
        latents = pipeline.denoise(
            embeds[:1], embeds[1:] if guided else None, **size,
            num_inference_steps=cfg["steps"], guidance_scale=cfg["guidance"], seed=cfg["seed"],
        )  # fmt: skip
        result["temperature_c"].append(_temperature())
        frames = pipeline.decode(latents)
        torch.cuda.synchronize()
    result["seconds"] = time.perf_counter() - started
    result["temperature_c"].append(_temperature())
    used = peak.total - peak.min_free
    result["peak_device_used_bytes"] = used
    result["peak_process_bytes"] = None if idle is None else used - idle
    result["stages"] = {r.name: dataclasses.asdict(r) for r in pipeline.reports}
    result["vae_config"] = pipeline.reports[-1].details["config"]
    result["frames_shape"] = list(frames.shape)
    result["measured_model"] = pipeline.measured
    if cfg["save_video"]:
        written = save_video(frames, Path(job["video"]), cfg["fps"])
        result["video"] = {"format": written.suffix[1:], "fps": cfg["fps"], "file": written.name}
    keep = min(cfg["psnr_frames"], frames.shape[0])
    tensors = {"embeds": embeds, "latents": latents, "frames": frames[:keep].clone()}
    return {"result": result, "tensors": tensors}


def load_strict(torch: Any, directory: Path) -> Any:
    """The strict baseline: the transformer of ``diffusers``, a blocking copy per block.

    The same operations as the streamed model, one row of the time embedding included, with
    none of its machinery: no slots, no second stream, no conversion of ours.
    """
    import diffusers

    from hypertier.adapters.hf_diffusers import _OneRow

    model = diffusers.WanTransformer3DModel.from_pretrained(directory, torch_dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    for name, child in model.named_children():
        if name != "blocks":
            child.to("cuda")
    model.scale_shift_table.data = model.scale_shift_table.data.to("cuda")
    for block in model.blocks:
        home = [(param, param.data) for param in block.parameters()]

        def load(module: Any, args: Any, home: Any = home) -> None:
            for param, data in home:
                param.data = data.to("cuda")

        def unload(module: Any, args: Any, output: Any, home: Any = home) -> None:
            for param, data in home:
                param.data = data

        block.register_forward_pre_hook(load)
        block.register_forward_hook(unload, always_call=True)
    one_row = _OneRow()
    model.condition_embedder.register_forward_pre_hook(one_row.before, with_kwargs=True)
    model.condition_embedder.register_forward_hook(one_row.after, with_kwargs=True)
    return model


def worker_steps(job: dict[str, Any], torch: Any) -> dict[str, Any]:
    """The first steps again, streamed or strict, from the embeddings of the generation."""
    from hypertier.adapters import load_wan_pipeline
    from hypertier.adapters.hf_diffusers import denoise, load_wan_transformer

    cfg = job["config"]
    root = Path(job["model"])
    pipeline = load_wan_pipeline(root, slots=cfg["slots"])
    shape = pipeline.latent_shape(cfg["height"], cfg["width"], cfg["frames"])
    embeds = torch.load(job["inputs"])["embeds"]
    guided = cfg["guidance"] > 1.0
    started = time.perf_counter()

    def sample(model: Any) -> Any:
        return denoise(
            model, pipeline.scheduler_class.from_pretrained(root / "scheduler"), embeds[:1],
            embeds[1:] if guided else None, latent_shape=shape,
            expand_timesteps=pipeline.expand_timesteps, num_inference_steps=cfg["check_steps"],
            guidance_scale=cfg["guidance"],
            generator=torch.Generator("cpu").manual_seed(cfg["seed"]),
        ).cpu()  # fmt: skip

    if job["mode"] == "strict":
        latents = sample(load_strict(torch, root / "transformer"))
    else:
        with load_wan_transformer(root / "transformer", slots=cfg["slots"]) as streamed:
            latents = sample(streamed.model)
    result = {"seconds": time.perf_counter() - started, "steps": cfg["check_steps"]}
    return {"result": result, "tensors": {"latents": latents}}


def worker_reference(job: dict[str, Any], torch: Any) -> dict[str, Any]:
    """The first frames untiled, in float32, on the CPU: what tiling is measured against."""
    import diffusers

    cfg = job["config"]
    vae = diffusers.AutoencoderKLWan.from_pretrained(
        Path(job["model"]) / "vae", torch_dtype=torch.float32
    ).eval()
    latents = torch.load(job["inputs"])["latents"].float()
    latents = latents[:, :, : (cfg["psnr_frames"] - 1) // 4 + 1]
    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean).view(shape)
    std = torch.tensor(vae.config.latents_std).view(shape)
    started = time.perf_counter()
    with torch.inference_mode():
        video = vae.decode(latents / (1.0 / std) + mean, return_dict=False)[0][0]
        frames = video.div_(2).add_(0.5).clamp_(0, 1).mul_(255).round_().to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).contiguous()
    result = {"seconds": time.perf_counter() - started, "frames": int(frames.shape[0])}
    return {"result": result, "tensors": {"frames": frames}}


def worker_metadata(job: dict[str, Any], torch: Any) -> dict[str, Any]:
    """Hardware and versions, gathered here so that the parent never opens a CUDA context."""
    return {"result": bench_hw.collect_metadata(), "tensors": {}}


WORKERS = {
    "generate": worker_generate, "streamed": worker_steps, "strict": worker_steps,
    "reference": worker_reference, "metadata": worker_metadata,
}  # fmt: skip


def run_job(path: Path) -> int:
    """Entry point of a worker process: run the job in ``path`` and write what it asks for."""
    job = json.loads(path.read_text())
    if job["mode"] == "generate":  # before this process has a CUDA context of its own
        job["idle_used_bytes"] = _used_now()
    import torch

    try:
        import diffusers
        import transformers

        diffusers.utils.logging.disable_progress_bar()
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
    print(f"[bench_video] {name} ...", file=sys.stderr, flush=True)
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


def psnr(frames: Any, reference: Any) -> float:
    """Peak signal-to-noise ratio of two uint8 videos, in dB (infinite when they are equal)."""
    import torch

    error = (frames.double() - reference.double()).pow(2).mean()
    return float("inf") if float(error) == 0 else float(10 * torch.log10(255.0**2 / error))


def run(cfg: Config, model: Path, scratch: Path) -> dict[str, Any]:
    import torch

    base = {"model": str(model), "config": dataclasses.asdict(cfg)}
    job = {**base, "name": "generate", "mode": "generate", "video": str(scratch / "video")}
    generated, tensors = launch(job, scratch)
    results: dict[str, Any] = {"model": model.name, "generate": generated, "checks": {}}
    if "error" in generated:
        return results
    checks = results["checks"]
    budget = int(cfg.vram_budget_gb * GB)
    peak = generated["peak_process_bytes"]
    if budget and peak is not None:
        checks["within_vram_budget"] = peak <= budget
    inputs = str(scratch / "generate.pt")
    if cfg.check_steps:
        runs = {}
        for mode in ("streamed", "strict"):
            job = {**base, "name": f"steps_{mode}", "mode": mode, "inputs": inputs}
            runs[mode] = launch(job, scratch)
        results["steps"] = {mode: result for mode, (result, _) in runs.items()}
        if all("latents" in kept for _, kept in runs.values()):
            ours, strict = (runs[mode][1]["latents"] for mode in ("streamed", "strict"))
            checks["latents_equal_strict_baseline"] = bool(torch.equal(ours, strict))
    if cfg.psnr_frames:
        job = {**base, "name": "reference", "mode": "reference", "inputs": inputs}
        reference, kept = launch(job, scratch)
        results["reference"] = reference
        if "frames" in kept:
            count = min(kept["frames"].shape[0], tensors["frames"].shape[0])
            measured = psnr(tensors["frames"][:count], kept["frames"][:count])
            threshold = generated["stages"]["vae"]["details"]["min_psnr_db"]
            results["psnr"] = {"db": measured, "threshold_db": threshold, "frames": count}
            checks["psnr_above_threshold"] = measured >= threshold
    return results


# ---------------------------------------------------------------------------------------------
# Summary


def _yes(value: Any) -> str:
    return "not checked" if value is None else ("yes" if value else "NO")


def format_summary(payload: dict[str, Any]) -> str:
    cfg, results = payload["config"], payload["results"]
    generated = results["generate"]
    title = (
        f"bench_video: {cfg['height']} x {cfg['width']} px, {cfg['frames']} frames, "
        f"{cfg['steps']} steps, guidance {cfg['guidance']:g}, {cfg['slots']} slots"
    )
    lines = [title]
    if "error" in generated:
        return "\n".join([*lines, f"  the generation failed: {generated['error']}"])
    temperature = generated["temperature_c"]
    lines.append(
        f"  whole generation    {generated['seconds']:8.1f} s   GPU at {temperature[0]} °C "
        f"at the start, {temperature[-1]} °C at the end"
    )
    for name, stage in generated["stages"].items():
        expected = stage["expected_peak_bytes"]
        promise = "" if expected is None else f" (expected {expected / MiB:.0f})"
        lines.append(
            f"  {name:<19} {stage['seconds']:8.1f} s   load {stage['load_seconds']:5.1f} s   "
            f"peak {stage['peak_process_bytes'] / MiB:5.0f} MiB{promise}"
        )
    passes = generated["stages"].get("transformer", {}).get("details", {})
    if passes.get("passes"):
        lines.append(
            f"  transformer pass    {passes['seconds_per_pass']:8.2f} s   copies "
            f"{passes['copy_seconds_per_pass']:.2f} s, overlap efficiency "
            f"{passes['overlap_efficiency_median']:.2f} (lowest {passes['overlap_efficiency_min']:.2f})"
        )
    vae = generated["stages"]["vae"]["details"]
    lines.append(f"  VAE configuration   {vae['config']}, {vae['tiles']} tiles")
    peak = generated["peak_process_bytes"]
    if peak is not None:
        lines.append(
            f"  VRAM held at the peak, as the driver counts it   {peak / MiB:.0f} MiB "
            f"({peak / GB:.2f} GB) of a budget of {cfg['vram_budget_gb']:g} GB"
        )
    checks = results["checks"]
    lines.append("criterion:")
    lines.append(f"  within the VRAM budget                 {_yes(checks.get('within_vram_budget'))}")
    steps = f" ({cfg['check_steps']} steps)" if cfg["check_steps"] else ""
    same = _yes(checks.get("latents_equal_strict_baseline"))
    lines.append(f"  same latents as the strict baseline    {same}{steps}")
    measured = results.get("psnr")
    detail = "" if measured is None else (
        f" ({measured['db']:.1f} dB over {measured['frames']} frames, "
        f"at least {measured['threshold_db']:.1f})"
    )  # fmt: skip
    lines.append(f"  PSNR of the tiled decode               {_yes(checks.get('psnr_above_threshold'))}{detail}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker is not None:
        return run_job(args.worker)
    cfg = make_config(args)
    if not cuda_available():
        print("bench_video needs a CUDA GPU", file=sys.stderr)
        return 2
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="video-", dir=args.out_dir))
    try:
        metadata, _ = launch({"name": "metadata", "mode": "metadata"}, scratch)
        if "error" in metadata:
            raise RuntimeError(f"could not read the hardware metadata: {metadata['error']}")
        metadata["worker_env"] = {"CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE}
        for package in ("diffusers", "transformers"):
            metadata[package] = _package_version(package)
        payload: dict[str, Any] = {
            "schema": SCHEMA, "metadata": metadata, "config": dataclasses.asdict(cfg),
        }  # fmt: skip
        payload["results"] = run(cfg, args.model, scratch)
        path = bench_hw.write_results(payload, args.out_dir, prefix="video")
        video = payload["results"]["generate"].get("video")
        if video is not None:  # next to the results, under the same name
            kept = path.with_suffix("." + video["format"])
            shutil.move(scratch / video.pop("file"), kept)
            path.write_text(json.dumps(payload, indent=2) + "\n")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(format_summary(payload))
    print(f"\nResults written to {path}")
    if video is not None:
        print(f"Video written to {kept}")
    checks = payload["results"]["checks"]
    failed = "error" in payload["results"]["generate"] or not all(checks.values())
    return 1 if failed else 0


def _package_version(name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    sys.exit(main())
