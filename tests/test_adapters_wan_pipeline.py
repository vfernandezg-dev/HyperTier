# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.adapters.wan_pipeline.

A tiny random Wan 2.2 pipeline (tokenizer, umT5, transformer, VAE, scheduler) is saved as
``WanPipeline.save_pretrained`` saves one, and a generation through its stages is compared
with ``WanPipeline`` of ``diffusers`` on the same directory: the latents bit for bit, and the
frames too when the VAE decodes the frame in one tile in float32, as ``diffusers`` does.

What is checked before a generation (sizes, the VRAM each stage is expected to need) is
arithmetic on measured peaks: it is tested with the free VRAM replaced by a number.

With ``HYPERTIER_MODELS`` and ``-m slow``, on Wan 2.2 TI2V-5B: real prompts against
``WanPipeline``, and the two sizes of the F3 criterion within 3.4 GB of VRAM.
"""

import gc
import json
import os
import threading
from pathlib import Path

import pytest
from toys import tiny_wan_pipeline

from hypertier.adapters import load_wan_pipeline
from hypertier.adapters import wan_pipeline as wan
from hypertier.adapters.hf_diffusers import VAE_CONFIGS, VaeConfig
from hypertier.diffusion.sequential import CUDA_CONTEXT, VRAM_MARGIN

MiB = 1 << 20
GB = 1000**3
SIZE = {"height": 96, "width": 160, "num_frames": 9}  # 6 x 10 latent cells, 3 latent frames
WHOLE = VaeConfig("whole-fp32", "float32", False, 600, ((1, 0),), 0.0)  # one tile, as diffusers
PROMPT = "a red fox runs over fresh snow"
NEGATIVE = "blurry, static"


@pytest.fixture
def diffusers():
    return pytest.importorskip("diffusers")


def stock_pipeline(directory):
    """``WanPipeline`` as its documentation loads it: the VAE in float32, the rest in bf16."""
    import diffusers
    import torch

    vae = diffusers.AutoencoderKLWan.from_pretrained(
        directory, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = diffusers.WanPipeline.from_pretrained(directory, vae=vae, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def stock(pipe, prompt, negative, *, guidance, seed, steps, output):
    import torch

    return pipe(
        prompt=prompt, negative_prompt=negative, **SIZE, num_inference_steps=steps,
        guidance_scale=guidance, generator=torch.Generator("cpu").manual_seed(seed),
        output_type=output,
    ).frames  # fmt: skip


def as_frames(images):
    """The PIL frames of ``diffusers`` as one uint8 tensor ``[frames, height, width, 3]``."""
    import numpy as np
    import torch

    return torch.from_numpy(np.stack([np.asarray(image) for image in images]))


def reader_threads():
    return [thread for thread in threading.enumerate() if thread.name == "hypertier-reader"]


def test_importing_the_pipeline_imports_neither_torch_nor_diffusers():
    import subprocess
    import sys

    code = (
        "import sys, hypertier.adapters, hypertier.adapters.wan_pipeline, "
        "hypertier.diffusion.sequential; "
        "sys.exit(('torch' in sys.modules) + 2 * ('diffusers' in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


# ---------------------------------------------------------------------------------------------
# A generation


@pytest.mark.cuda
def test_a_generation_gives_the_latents_and_the_frames_of_wanpipeline(diffusers, disk_tmp):
    import torch

    tiny_wan_pipeline(disk_tmp)
    theirs = stock_pipeline(disk_tmp)
    cases = [
        (PROMPT, NEGATIVE, 5.0, 0, 4),
        (PROMPT, NEGATIVE, 5.0, 7, 4),
        ("two cats boxing, bright gloves", "", 3.5, 1, 5),
        (PROMPT, NEGATIVE, 1.0, 0, 4),  # no guidance: the negative prompt is not encoded
        ("  A   red fox\n", NEGATIVE, 5.0, 0, 3),  # cleaned as the pipeline cleans it
    ]
    with load_wan_pipeline(disk_tmp, vae_config=WHOLE) as ours:
        assert ours.expand_timesteps and not ours.measured
        for prompt, negative, guidance, seed, steps in cases:
            options = {"guidance": guidance, "seed": seed, "steps": steps}
            latents = stock(theirs, prompt, negative, output="latent", **options)
            frames = as_frames(stock(theirs, prompt, negative, output="pil", **options)[0])
            assert bool(torch.isfinite(latents).all()) and float(frames.float().std()) > 5
            for one_pass in (False, True):
                result = ours(
                    prompt, negative, **SIZE, num_inference_steps=steps,
                    guidance_scale=guidance, seed=seed, one_pass=one_pass,
                )  # fmt: skip
                assert result.latents.device.type == "cpu"
                assert torch.equal(result.latents, latents.cpu()), (prompt, guidance, one_pass)
                assert result.frames.dtype == torch.uint8
                assert torch.equal(result.frames, frames), (prompt, guidance, one_pass)
                assert result.vae_config == "whole-fp32"


@pytest.mark.cuda
def test_every_stage_is_on_record_and_nothing_stays_loaded(diffusers, disk_tmp):
    import torch

    tiny_wan_pipeline(disk_tmp)
    gc.collect()
    before = torch.cuda.memory_allocated()
    with load_wan_pipeline(disk_tmp, vae_config=WHOLE) as ours:
        result = ours(PROMPT, NEGATIVE, **SIZE, num_inference_steps=3)
        assert list(result.stages) == ["text", "transformer", "vae"]
        text, transformer, vae = result.stages.values()
        assert text.details["prompts"] == 2 and text.details["tokens"] == [31, 15]
        assert transformer.details["tokens"] == 45 and transformer.details["passes"] == 6
        assert transformer.details["slots"] == 2 and transformer.details["steps"] == 3
        assert vae.details["config"] == "whole-fp32" and vae.details["tiles"] == 1
        for report in result.stages.values():
            assert report.expected_peak_bytes is None, "nothing was measured for a tiny model"
            assert 0 < report.load_seconds < report.seconds
            assert 0 < report.peak_allocated_bytes <= report.peak_reserved_bytes
            assert report.peak_process_bytes == report.peak_reserved_bytes + report.context_bytes
            assert 0 < report.context_bytes <= CUDA_CONTEXT
            assert report.budget_bytes >= report.free_before_bytes > 0
        del result, text, transformer, vae, report
        gc.collect()
        assert torch.cuda.memory_allocated() == before and not reader_threads()
        again = ours(PROMPT, NEGATIVE, **SIZE, num_inference_steps=2, seed=3)
        assert list(again.stages) == ["transformer", "vae"], "both prompts were in memory"


@pytest.mark.cuda
def test_the_stages_can_be_called_one_by_one(diffusers, disk_tmp):
    import torch

    tiny_wan_pipeline(disk_tmp)
    theirs = stock_pipeline(disk_tmp)
    expected = stock(theirs, PROMPT, NEGATIVE, guidance=5.0, seed=2, steps=4, output="latent")
    with load_wan_pipeline(disk_tmp) as ours:
        embeds = ours.encode([PROMPT, NEGATIVE, PROMPT])
        assert embeds.shape == (3, wan.MAX_SEQUENCE_LENGTH, 64) and embeds.dtype == torch.bfloat16
        assert torch.equal(embeds[0], embeds[2]) and not torch.equal(embeds[0], embeds[1])
        assert not embeds[0, 31:].any() and embeds[0, 30].any(), "zeros after the last token"
        latents = ours.denoise(
            embeds[:1], embeds[1:2], **SIZE, num_inference_steps=4, seed=2
        )
        assert torch.equal(latents, expected.cpu())
        frames = ours.decode(latents)
        assert frames.shape == (9, 96, 160, 3) and frames.dtype == torch.uint8
        assert [report.name for report in ours.reports] == ["text", "transformer", "vae"]


@pytest.mark.cuda
def test_prompts_are_kept_by_their_cleaned_text(diffusers, disk_tmp):
    import torch

    tiny_wan_pipeline(disk_tmp)
    with load_wan_pipeline(disk_tmp) as ours:
        first = ours.encode([PROMPT])
        assert len(ours.reports) == 1
        assert torch.equal(ours.encode(["  a red  fox runs over\tfresh snow "]), first)
        assert len(ours.reports) == 1, "the same prompt once cleaned: not encoded again"
        key = next(iter(ours._prompts))
        assert key[0] == PROMPT and key[1].endswith("text_encoder")
        assert key[2:] == (wan.MAX_SEQUENCE_LENGTH, "torch.bfloat16")
        ours.encode([f"prompt number {n}" for n in range(wan.CACHED_PROMPTS)])
        assert len(ours._prompts) == wan.CACHED_PROMPTS and key not in ours._prompts
        assert ours.reports[-1].details["prompts"] == wan.CACHED_PROMPTS
        ours.close()
        assert not ours._prompts


# ---------------------------------------------------------------------------------------------
# What is refused before anything is loaded


@pytest.mark.cuda
def test_sizes_the_model_cannot_draw_are_refused_with_the_nearest_that_it_can(
    diffusers, disk_tmp
):
    tiny_wan_pipeline(disk_tmp)
    with load_wan_pipeline(disk_tmp) as ours:
        assert ours.latent_shape(704, 1280, 49) == (13, 44, 80)
        assert ours.tokens(704, 1280, 49) == 11_440 and ours.tokens(480, 832, 121) == 12_090
        with pytest.raises(ValueError, match="the nearest size below is 96 x 160"):
            ours(PROMPT, height=100, width=170, num_frames=9)
        with pytest.raises(ValueError, match="the nearest size below is 32 x 160"):
            ours.latent_shape(16, 160, 9)
        with pytest.raises(ValueError, match="the nearest length below is 9"):
            ours(PROMPT, height=96, width=160, num_frames=12)
        assert not ours.reports, "refused before any stage ran"


@pytest.fixture
def measured(diffusers, disk_tmp, monkeypatch):
    """A pipeline that believes it is Wan 2.2 TI2V-5B, with the free VRAM set by the test."""
    tiny_wan_pipeline(disk_tmp)
    ours = load_wan_pipeline(disk_tmp)
    ours.measured = True
    ours._capability = (8, 6)

    def set_available(nbytes):
        monkeypatch.setattr(ours, "available", lambda: nbytes)

    ours.set_available = set_available
    return ours


@pytest.mark.cuda
def test_the_criterion_sizes_fit_in_the_budget_of_f3(measured):
    resident = VAE_CONFIGS[1]
    measured.set_available(int(3.4 * GB))
    for height, width, frames in [(704, 1280, 49), (480, 832, 121)]:
        peaks = measured.check(height=height, width=width, num_frames=frames)
        assert set(peaks) == {"text", "transformer", "vae"}
        assert peaks["vae"] == resident.peak_bytes(frames)
        assert max(peaks.values()) + VRAM_MARGIN <= int(3.4 * GB)
        tokens = measured.tokens(height, width, frames)
        assert peaks["transformer"] == measured.transformer_peak(tokens, slots=2)
    assert not measured.reports, "checking loads nothing"


@pytest.mark.cuda
def test_a_generation_that_does_not_fit_is_refused_with_what_would(measured):
    tokens = measured.tokens(704, 1280, 49)
    two, one = (measured.transformer_peak(tokens, slots=n) for n in (2, 1))
    assert one < two < measured.transformer_peak(tokens, slots=2, one_pass=True)
    size = {"height": 704, "width": 1280, "num_frames": 49}

    measured.set_available(two + VRAM_MARGIN - 1)
    with pytest.raises(ValueError) as refused:
        measured(PROMPT, **size)
    message = str(refused.value)
    assert "the transformer (11440 tokens, 2 slots) stage peaks at" in message
    assert "slots=1 fits" in message and "frames fit at 704 x 1280" in message
    shorter = int(message.split(" frames fit")[0].rsplit(" ", 1)[1])
    assert shorter < 49 and (shorter - 1) % 4 == 0
    measured.check(height=704, width=1280, num_frames=shorter)  # what it advised does fit
    assert not measured.reports, "refused before any stage ran"

    measured.set_available(one + VRAM_MARGIN - 1)
    with pytest.raises(ValueError) as refused:
        measured.check(**size)
    assert "slots=1 fits" not in str(refused.value) and "frames fit" in str(refused.value)
    assert f"ffn_chunk={wan.FFN_CHUNK} fits, with latents that differ" in str(refused.value)

    measured.set_available(900 * MiB)
    with pytest.raises(ValueError, match="Nothing shorter fits either"):
        measured.check(**size)

    measured.slots = 1
    measured.set_available(one + VRAM_MARGIN)
    assert measured.check(**size)["transformer"] == one


@pytest.mark.cuda
def test_the_peak_of_the_transformer_is_read_between_what_was_measured(measured):
    def expected(mib):
        return mib * MiB + CUDA_CONTEXT

    for (slots, way), points in wan.TRANSFORMER_PEAKS.items():
        chunk = wan.FFN_CHUNK if way == "chunked" else None

        def peak(tokens, slots=slots, way=way, chunk=chunk):
            return measured.transformer_peak(
                tokens, slots=slots, one_pass=way == "one pass", ffn_chunk=chunk
            )

        assert [tokens for tokens, _ in points] == sorted({tokens for tokens, _ in points})
        points = [point for point in points if chunk is None or point[0] > chunk]
        for tokens, mib in points:
            assert peak(tokens) == expected(mib), (slots, way, tokens)
        (fewer, low), (more, high) = points[:2]
        assert peak((fewer + more) // 2) == pytest.approx(expected((low + high) / 2), abs=MiB)
        most = points[-1][0]
        message = f"measured up to {most} tokens: {most + 880} tokens is outside"
        with pytest.raises(ValueError, match=message):
            peak(most + 880)

    whole = measured.transformer_peak(12_090, slots=2)
    assert measured.transformer_peak(500, slots=2) == measured.transformer_peak(1_950, slots=2)
    assert measured.transformer_peak(12_090, slots=2, ffn_chunk=wan.FFN_CHUNK) < whole
    assert measured.transformer_peak(12_090, slots=2, ffn_chunk=12_090) == whole, "one chunk"
    assert measured.transformer_peak(12_090, slots=2, ffn_chunk=64) < whole, "smaller chunks"
    with pytest.raises(ValueError, match="was not measured with 3 slots: it is refused"):
        measured.transformer_peak(12_090, slots=3)
    with pytest.raises(ValueError, match="not measured with 2 slots, both branches .* at a time"):
        measured.transformer_peak(12_090, slots=2, one_pass=True, ffn_chunk=wan.FFN_CHUNK)
    with pytest.raises(ValueError, match=f"networks {wan.FFN_CHUNK + 1} tokens at a time"):
        measured.transformer_peak(12_090, slots=2, ffn_chunk=wan.FFN_CHUNK + 1)
    measured.measured = False
    assert measured.transformer_peak(10**6, slots=7) is None, "another model: nothing is known"


@pytest.mark.cuda
def test_a_video_of_more_tokens_than_were_measured_is_refused_not_guessed(measured):
    import torch

    size = {"height": 704, "width": 1280, "num_frames": 121}  # 27,280 tokens
    most = wan.TRANSFORMER_PEAKS[(2, "whole")][-1][0]
    assert most < 27_280 == wan.TRANSFORMER_PEAKS[(2, "chunked")][-1][0]
    measured.set_available(1 << 40)
    with pytest.raises(ValueError) as refused:
        measured(PROMPT, **size)
    message = str(refused.value)
    assert f"with 2 slots was measured up to {most} tokens: 27280 tokens is outside" in message
    assert f"ffn_chunk={wan.FFN_CHUNK} fits, with latents that differ by rounding" in message
    shorter = int(message.split(" frames fit at 704 x 1280")[0].rsplit(" ", 1)[1])
    assert measured.tokens(704, 1280, shorter) <= most < measured.tokens(704, 1280, shorter + 4)
    with pytest.raises(ValueError, match=f"measured up to {most} tokens"):
        measured.denoise(torch.zeros(1, 512, 64), **size, guidance_scale=1.0)
    assert not measured.reports, "refused before any stage ran"

    measured.ffn_chunk = wan.FFN_CHUNK
    chunked = measured.transformer_peak(27_280, slots=2, ffn_chunk=wan.FFN_CHUNK)
    assert measured.check(**size)["transformer"] == chunked
    measured.set_available(chunked + VRAM_MARGIN - 1)
    with pytest.raises(ValueError) as refused:
        measured.check(**size)
    message = str(refused.value)
    assert "(27280 tokens, 2 slots, the feed-forward networks 2048 tokens at a time)" in message
    assert "slots=1 fits" in message and "ffn_chunk" not in message.split("Try:")[1]


@pytest.mark.cuda
def test_the_feed_forward_networks_can_run_in_chunks(diffusers, disk_tmp, monkeypatch):
    import torch

    tiny_wan_pipeline(disk_tmp)
    asked = []
    load = wan.load_wan_transformer

    def spy(*args, **kwargs):
        asked.append(kwargs["ffn_chunk"])
        return load(*args, **kwargs)

    monkeypatch.setattr(wan, "load_wan_transformer", spy)
    run = {**SIZE, "num_inference_steps": 3}
    with load_wan_pipeline(disk_tmp, vae_config=WHOLE) as whole:
        expected = whole(PROMPT, NEGATIVE, **run).latents
    with load_wan_pipeline(disk_tmp, vae_config=WHOLE, ffn_chunk=45) as one_chunk:
        assert torch.equal(one_chunk(PROMPT, NEGATIVE, **run).latents, expected), "45 tokens"
    with load_wan_pipeline(disk_tmp, vae_config=WHOLE, ffn_chunk=16) as ours:
        result = ours(PROMPT, NEGATIVE, **run)
        assert result.stages["transformer"].details["ffn_chunk"] == 16
        error = (result.latents.float() - expected.float()).norm() / expected.float().norm()
        assert float(error) < 0.05, "rounding, not another video"
    assert asked == [None, 45, 16]
    with pytest.raises(ValueError, match="cannot run 0 tokens at a time"):
        load_wan_pipeline(disk_tmp, ffn_chunk=0)


@pytest.mark.cuda
def test_a_length_outside_what_was_measured_is_refused_not_guessed(measured):
    import torch

    measured.set_available(1 << 40)
    for frames in (1, 13, 125, 241):
        with pytest.raises(ValueError, match=f"17 to 121 frames: {frames} frames is outside"):
            measured(PROMPT, height=480, width=832, num_frames=frames)
        with pytest.raises(ValueError, match="measured for videos of 17 to 121 frames"):
            measured.decode(torch.zeros(1, 48, (frames - 1) // 4 + 1, 30, 52))
    assert measured.check(height=480, width=832, num_frames=17)["vae"] > 0
    assert measured.check(height=480, width=832, num_frames=121)["vae"] > 0
    assert not measured.reports, "refused before any stage ran"


@pytest.mark.cuda
def test_the_vae_is_chosen_for_the_gpu_and_can_be_fixed_by_the_caller(measured, disk_tmp):
    streamed, resident, small, wide = VAE_CONFIGS
    measured.set_available(int(3.4 * GB))
    assert measured._vae_config(int(3.4 * GB), 49) is resident
    assert measured._vae_config(resident.peak_bytes(49) + VRAM_MARGIN - 1, 49) is small
    assert measured._vae_config(1 << 40, 49) is streamed
    measured._capability = (7, 5)  # a GTX 1650: no native bfloat16
    assert measured._vae_config(1 << 40, 49) is wide

    fixed = load_wan_pipeline(disk_tmp, vae_config="bf16-streamed")
    fixed.measured, fixed._capability = True, (8, 6)
    assert fixed._vae_config(1 << 40, 49) is streamed
    fixed.available = lambda: int(3.4 * GB)
    with pytest.raises(ValueError, match=r"the VAE \(49 frames\) stage peaks at"):
        fixed.check(height=704, width=1280, num_frames=49)
    with pytest.raises(ValueError, match="no VAE configuration is called 'fp16'"):
        load_wan_pipeline(disk_tmp, vae_config="fp16")


@pytest.mark.cuda
def test_a_directory_that_is_not_a_wan_pipeline_with_one_transformer_is_refused(
    diffusers, disk_tmp, tmp_path
):
    tiny_wan_pipeline(disk_tmp)
    index = json.loads((disk_tmp / "model_index.json").read_text())
    with pytest.raises(ValueError, match="model_index.json is missing"):
        load_wan_pipeline(tmp_path)
    (disk_tmp / "model_index.json").write_text(
        json.dumps({**index, "transformer_2": ["diffusers", "WanTransformer3DModel"]})
    )
    with pytest.raises(ValueError, match="samples with two transformers"):
        load_wan_pipeline(disk_tmp)
    (disk_tmp / "model_index.json").write_text(json.dumps({**index, "_class_name": "Other"}))
    with pytest.raises(ValueError, match="holds a Other"):
        load_wan_pipeline(disk_tmp)


# ---------------------------------------------------------------------------------------------
# The real model: Wan 2.2 TI2V-5B (needs HYPERTIER_MODELS and -m slow)

REAL = "Wan2.2-TI2V-5B-Diffusers"
SMALL = {"height": 480, "width": 832, "num_frames": 17}  # 1,950 tokens: diffusers can run it
REAL_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
REAL_CASES = [  # prompt, negative prompt, guidance, seed
    ("Two anthropomorphic cats in comfy boxing gear fight on a spotlighted stage.",
     REAL_NEGATIVE, 5.0, 0),
    ("A hummingbird hovering in front of a red flower, slow motion.", REAL_NEGATIVE, 5.0, 3),
    ("Aerial shot of an autumn forest with a winding river.", "", 1.0, 1),
]  # fmt: skip
REAL_STEPS = 3


@pytest.fixture(scope="module")
def real_dir():
    root = os.environ.get("HYPERTIER_MODELS")
    if not root or not (Path(root) / REAL / "model_index.json").is_file():
        pytest.skip(f"set HYPERTIER_MODELS to a directory that holds {REAL}")
    pytest.importorskip("diffusers")
    return Path(root) / REAL


@pytest.fixture(scope="module")
def real_expected(real_dir):
    """Latents of ``WanPipeline`` itself for every case, before our pipeline loads anything.

    The stock pipeline, text encoder and transformer offloaded to the CPU module by module
    as ``enable_sequential_cpu_offload`` does: about 21 GiB of RAM. It is loaded without its
    VAE, which the latents do not need; it then takes 8 px per latent cell instead of 16, so
    it is asked for half the size to get the same latent.
    """
    import diffusers
    import torch

    pipe = diffusers.WanPipeline.from_pretrained(real_dir, vae=None, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_sequential_cpu_offload()
    expected = []
    for prompt, negative, guidance, seed in REAL_CASES:
        latents = pipe(
            prompt=prompt, negative_prompt=negative, height=SMALL["height"] // 2,
            width=SMALL["width"] // 2, num_frames=SMALL["num_frames"],
            num_inference_steps=REAL_STEPS, guidance_scale=guidance,
            generator=torch.Generator("cpu").manual_seed(seed), output_type="latent",
        ).frames  # fmt: skip
        expected.append(latents.cpu())
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return expected


@pytest.mark.slow
@pytest.mark.cuda
def test_real_prompts_give_the_latents_of_wanpipeline(real_dir, real_expected):
    """Criterion (a) end to end: text encoder and transformer, on real prompts."""
    import torch

    with load_wan_pipeline(real_dir) as ours:
        assert ours.measured and ours.expand_timesteps
        for (prompt, negative, guidance, seed), expected in zip(REAL_CASES, real_expected):
            assert expected.shape == (1, 48, 5, 30, 52) and float(expected.std()) > 0.1
            guided = guidance > 1.0
            embeds = ours.encode([prompt, negative] if guided else [prompt])
            for one_pass in (False, True) if guided else (False,):
                latents = ours.denoise(
                    embeds[:1], embeds[1:] if guided else None, **SMALL,
                    num_inference_steps=REAL_STEPS, guidance_scale=guidance, seed=seed,
                    one_pass=one_pass,
                )  # fmt: skip
                assert torch.equal(latents, expected), (prompt, guidance, one_pass)
        texts = [report for report in ours.reports if report.name == "text"]
        assert [report.details["prompts"] for report in texts] == [2, 1, 1], "negative cached"


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("height", "width", "num_frames"), [(704, 1280, 49), (480, 832, 121)]
)
def test_the_sizes_of_the_f3_criterion_run_within_3_4_gb(real_dir, height, width, num_frames):
    """Two steps of each size under the budget of the criterion, passed explicitly."""
    import torch

    budget = int(3.4 * GB)
    prompt, negative, _, seed = REAL_CASES[0]
    with load_wan_pipeline(real_dir, vram_budget=budget) as ours:
        result = ours(
            prompt, negative, height=height, width=width, num_frames=num_frames,
            num_inference_steps=2, seed=seed,
        )  # fmt: skip
    assert result.frames.shape == (num_frames, height, width, 3)
    assert result.frames.dtype == torch.uint8 and float(result.frames.float().std()) > 10
    assert result.vae_config == "bf16-resident"
    for name, report in result.stages.items():
        print(
            f"\n[pipeline] {height}x{width}x{num_frames} {name}: {report.seconds:.1f} s, "
            f"expected {report.expected_peak_bytes / MiB:.0f} MiB, held "
            f"{report.peak_process_bytes / MiB:.0f} MiB of {report.budget_bytes / MiB:.0f}"
        )
        assert report.budget_bytes <= budget
        assert report.peak_process_bytes <= budget, name
        assert report.peak_process_bytes <= report.expected_peak_bytes, "the table holds"
        if report.context_measured:
            print(f"[pipeline] {name}: {report.context_bytes / MiB:.0f} MiB outside the allocator")
            assert report.context_bytes <= CUDA_CONTEXT, "no context above the most measured"
    assert result.stages["transformer"].details["overlap_efficiency_median"] > 0.9


@pytest.mark.slow
@pytest.mark.cuda
def test_the_longest_video_measured_runs_in_what_the_table_says(real_dir):
    """720p x 121 frames, 27,280 tokens: one slot and the feed-forward networks in chunks."""
    import torch

    size = {"height": 704, "width": 1280, "num_frames": 121}
    with load_wan_pipeline(real_dir, slots=1, ffn_chunk=wan.FFN_CHUNK) as ours:
        expected = ours.check(**size)["transformer"]
        embeds = ours.encode([REAL_CASES[0][0]])
        latents = ours.denoise(embeds, **size, num_inference_steps=1, guidance_scale=1.0)
        report = ours.reports[-1]
    assert latents.shape == (1, 48, 31, 44, 80) and bool(torch.isfinite(latents).all())
    assert report.details["tokens"] == 27_280 and report.details["ffn_chunk"] == wan.FFN_CHUNK
    measured = dict(wan.TRANSFORMER_PEAKS[(1, "chunked")])[27_280]
    print(
        f"\n[pipeline] 27,280 tokens, 1 slot, ffn in chunks: {report.seconds:.1f} s, "
        f"{report.peak_reserved_bytes / MiB:.0f} MiB reserved (measured {measured}), "
        f"{report.context_bytes / MiB:.0f} MiB outside the allocator"
    )
    assert report.peak_reserved_bytes <= measured * MiB, "the table holds"
    assert report.peak_process_bytes <= report.expected_peak_bytes == expected
