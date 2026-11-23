# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmarks/bench_video.py (config, summary and a quick end-to-end run)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from toys import tiny_wan_pipeline

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_video.py"
MiB = 1 << 20


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_video", BENCH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_the_default_is_the_main_size_of_the_criterion_and_quick_is_short(bench):
    cfg = bench.make_config(bench.parse_args(["--model", "m"]))
    assert (cfg.height, cfg.width, cfg.frames, cfg.steps) == (704, 1280, 49, 50)
    assert cfg.vram_budget_gb == 3.4 and cfg.check_steps == 2 and cfg.psnr_frames == 9
    quick = bench.make_config(bench.parse_args(["--model", "m", "--quick", "--steps", "6"]))
    assert (quick.height, quick.width, quick.frames, quick.steps) == (480, 832, 17, 6)
    assert quick.check_steps == 1 and quick.psnr_frames == 5
    assert quick.fps == 24 and quick.save_video
    silent = bench.make_config(bench.parse_args(["--model", "m", "--no-video", "--fps", "16"]))
    assert not silent.save_video and silent.fps == 16
    long = ["--model", "m", "--height", "480", "--width", "832", "--frames", "121"]
    assert bench.make_config(bench.parse_args(long)).frames == 121
    with pytest.raises(SystemExit, match="one frame plus groups of four"):
        bench.make_config(bench.parse_args(["--model", "m", "--psnr-frames", "8"]))
    with pytest.raises(SystemExit):
        bench.parse_args([])


def test_every_worker_gets_the_cublas_workspace_and_stays_offline(bench, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    env = bench.worker_env()
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and env["HF_HUB_OFFLINE"] == "1"


def test_psnr_of_equal_and_of_different_frames(bench):
    torch = pytest.importorskip("torch")
    frames = torch.zeros(2, 4, 4, 3, dtype=torch.uint8)
    assert bench.psnr(frames, frames) == float("inf")
    assert bench.psnr(frames + 255, frames) == pytest.approx(0.0)
    assert bench.psnr(frames + 1, frames) == pytest.approx(48.13, abs=0.01)


def _frames(torch, count=5, height=32, width=48):
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (count, height, width, 3), dtype=torch.uint8, generator=generator)


def test_without_the_video_extra_the_video_is_a_webp_and_says_how_to_get_an_mp4(
    bench, tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    image = pytest.importorskip("PIL.Image")
    monkeypatch.setattr(bench, "mp4_encoder", lambda: False)
    written, why = bench.save_video(_frames(torch), tmp_path / "video", fps=24)
    assert written == tmp_path / "video.webp" and why == bench.MP4_HINT
    assert "pip install hypertier[video]" in why
    assert image.open(written).n_frames == 5 and image.open(written).size == (48, 32)


def test_with_the_video_extra_the_video_is_an_mp4(bench, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("imageio_ffmpeg")
    imageio = pytest.importorskip("imageio.v2")
    assert bench.mp4_encoder()
    written, why = bench.save_video(_frames(torch), tmp_path / "video", fps=24)
    assert written == tmp_path / "video.mp4" and why is None
    with imageio.get_reader(written) as reader:
        meta = reader.get_meta_data()
        assert tuple(meta["size"]) == (48, 32) and meta["fps"] == 24 and meta["codec"] == "h264"
        assert reader.count_frames() == 5

    def broken(*args, **kwargs):
        raise OSError("no encoder")

    monkeypatch.setattr(imageio, "get_writer", broken)
    written, why = bench.save_video(_frames(torch), tmp_path / "again", fps=24)
    assert written == tmp_path / "again.webp", "the measurement is not lost for its video"
    assert "could not be written (OSError: no encoder)" in why
    assert not (tmp_path / "again.mp4").exists()


def _stage(seconds, peak, expected=None, **details):
    return {
        "seconds": seconds, "load_seconds": 1.0, "peak_process_bytes": peak * MiB,
        "expected_peak_bytes": None if expected is None else expected * MiB, "details": details,
    }  # fmt: skip


def test_the_summary_shows_every_stage_and_every_check(bench):
    cfg = bench.dataclasses.asdict(bench.Config())
    generated = {
        "seconds": 1400.0, "temperature_c": [63, 85, 76], "peak_process_bytes": 2900 * MiB,
        "stages": {
            "text": _stage(5.0, 1200, 1300),
            "transformer": _stage(1280.0, 2500, 2600, passes=100, seconds_per_pass=12.8,
                                  copy_seconds_per_pass=1.48, overlap_efficiency_median=1.0,
                                  overlap_efficiency_min=0.97),
            "vae": _stage(128.0, 2760, 2815, config="bf16-resident", tiles=16),
        },
    }  # fmt: skip
    generated["stages"]["vae"].update(context_bytes=141 * MiB, context_measured=True)
    generated["stages"]["text"].update(context_bytes=210 * MiB, context_measured=False)
    results = {
        "generate": generated,
        "checks": {"within_vram_budget": True, "latents_equal_strict_baseline": True,
                   "psnr_above_threshold": False},
        "psnr": {"db": 37.9, "threshold_db": 38.4, "frames": 9},
    }  # fmt: skip
    text = bench.format_summary({"config": cfg, "results": results})
    assert "704 x 1280 px, 49 frames, 50 steps" in text
    assert "GPU at 63 °C at the start, 76 °C at the end" in text
    for line in ("text", "transformer", "vae"):
        assert f"\n  {line}" in text
    assert "peak  2500 MiB (expected 2600)\n" in text
    assert "peak  2760 MiB (expected 2815), 141 outside the allocator" in text
    assert "peak  1200 MiB (expected 1300)\n" in text, "not read from the driver: not shown"
    assert "copies 1.48 s, overlap efficiency 1.00 (lowest 0.97)" in text
    assert "bf16-resident, 16 tiles" in text
    assert "2900 MiB (3.04 GB) of a budget of 3.4 GB" in text
    assert "within the VRAM budget                 yes" in text
    assert "same latents as the strict baseline    yes (2 steps)" in text
    assert "NO (37.9 dB over 9 frames, at least 38.4)" in text
    unchecked = {"generate": generated, "checks": {}}
    assert bench.format_summary({"config": cfg, "results": unchecked}).count("not checked") == 3
    failed = {"generate": {"error": "ValueError: does not fit"}, "checks": {}}
    assert "the generation failed: ValueError" in bench.format_summary(
        {"config": cfg, "results": failed}
    )


@pytest.mark.cuda
def test_quick_run_end_to_end(bench, disk_tmp, tmp_path, capsys):
    pytest.importorskip("diffusers")
    tiny_wan_pipeline(disk_tmp)
    out_dir = tmp_path / "results"
    code = bench.main(
        ["--model", str(disk_tmp), "--height", "96", "--width", "160", "--frames", "9",
         "--steps", "3", "--check-steps", "2", "--psnr-frames", "5", "--warm-up", "0",
         "--vram-budget-gb", "0", "--vae-config", "fp32-streamed", "--out-dir", str(out_dir)]
    )  # fmt: skip
    assert code == 0, "equal to the strict baseline, and one tile in float32 is the reference"
    (path,) = out_dir.glob("video_*.json")
    mp4 = bench.mp4_encoder()  # the optional extra: hypertier[video]
    video = path.with_suffix(".mp4" if mp4 else ".webp")
    assert sorted(out_dir.iterdir()) == sorted([path, video]), "the scratch directory is removed"
    assert video.stat().st_size > 1000
    payload = json.loads(path.read_text())
    assert payload["schema"] == bench.SCHEMA
    meta = payload["metadata"]
    assert meta["worker_env"] == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    assert meta["diffusers"] and meta["torch"]["version"] and meta["gpu"]["name"]
    res = payload["results"]
    assert res["model"] == disk_tmp.name and str(disk_tmp.parent) not in path.read_text()
    generated = res["generate"]
    assert list(generated["stages"]) == ["text", "transformer", "vae"]
    assert generated["frames_shape"] == [9, 96, 160, 3] and not generated["measured_model"]
    assert generated["vae_config"] == "fp32-streamed"
    assert generated["stages"]["transformer"]["details"]["passes"] == 6
    assert len(generated["temperature_c"]) == 3
    said = capsys.readouterr().out
    if mp4:
        assert generated["video"] == {"format": "mp4", "fps": 24}
        assert "pip install hypertier[video]" not in said
    else:
        from PIL import Image

        assert generated["video"] == {"format": "webp", "fps": 24, "not_mp4": bench.MP4_HINT}
        assert "An animated WebP: for an MP4, pip install hypertier[video]" in said
        assert Image.open(video).n_frames == 9 and Image.open(video).size == (160, 96)
    assert generated["peak_process_bytes"] > generated["stages"]["vae"]["peak_reserved_bytes"]
    assert res["checks"] == {"latents_equal_strict_baseline": True, "psnr_above_threshold": True}
    assert res["psnr"]["frames"] == 5 and res["psnr"]["db"] > 45
    assert res["steps"]["strict"]["steps"] == 2
    assert "same latents as the strict baseline    yes (2 steps)" in said
