# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the VAE stage of hypertier.adapters.hf_diffusers.

A tiny random Wan VAE of the 2.2 kind (16 px per latent cell) is saved as a normal checkpoint
and compared with the same checkpoint loaded by ``diffusers``, resident in VRAM. Two things
are exact and compared bit for bit: a frame that fits in one tile against the whole decode of
``diffusers``, and every tile against the same tile through ``diffusers``. Tiling itself is
not: on the real VAE it is judged by PSNR against the untiled float32 decode (``-m slow``),
on a clip sampled by the model in ten steps and on one sampled by FastWan in three.
"""

import gc
import os
from pathlib import Path

import pytest

from hypertier.adapters.hf_diffusers import (
    VAE_CONFIGS,
    VaeConfig,
    choose_vae_config,
    denoise,
    denoise_dmd,
    load_wan_transformer,
    load_wan_vae,
    usable_vae_configs,
)
from hypertier.diffusion.sequential import VRAM_MARGIN
from hypertier.diffusion.tiling import Canvas

MiB = 1 << 20
Z = 8
FRAME = (6, 10)  # latent cells: 96 x 160 px
LATENT_FRAMES = 3  # 9 frames


@pytest.fixture
def diffusers():
    return pytest.importorskip("diffusers")


def tiny_wan_vae(directory, *, seed=0):
    """Write a random tiny Wan 2.2 VAE as a normal ``diffusers`` checkpoint (float32)."""
    import torch
    from diffusers import AutoencoderKLWan

    torch.manual_seed(seed)
    vae = AutoencoderKLWan(
        base_dim=16, decoder_base_dim=16, z_dim=Z, dim_mult=[1, 2, 4, 4], num_res_blocks=1,
        attn_scales=[], temperal_downsample=[False, True, True], in_channels=12, out_channels=12,
        patch_size=2, scale_factor_spatial=16, scale_factor_temporal=4, is_residual=True,
        latents_mean=[0.1 * i for i in range(Z)], latents_std=[1.0 + 0.1 * i for i in range(Z)],
    )  # fmt: skip
    vae.save_pretrained(directory)
    return directory


def reference(directory, dtype):
    import torch
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(directory, torch_dtype=getattr(torch, dtype))
    return vae.to("cuda").eval()


def latents(seed=1):
    import torch

    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, Z, LATENT_FRAMES, *FRAME, generator=gen)


def denormalised(vae, sample, dtype):
    """What ``WanPipeline`` hands to ``vae.decode``."""
    import torch

    shape = (1, Z, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean).view(shape)
    std = torch.tensor(vae.config.latents_std).view(shape)
    return (sample / (1.0 / std) + mean).to("cuda", getattr(torch, dtype))


def to_frames(video):
    """``[3, frames, height, width]`` in [-1, 1] to uint8 frames, as ``diffusers`` rounds."""
    import numpy as np
    import torch
    from diffusers.video_processor import VideoProcessor

    processor = VideoProcessor(vae_scale_factor=16)
    images = processor.postprocess_video(video[None].float(), output_type="pil")[0]
    return torch.from_numpy(np.stack([np.asarray(image) for image in images]))


def config(dtype, streamed, max_tile_cells):
    return VaeConfig("test", dtype, streamed, max_tile_cells, ((1, 0),), 0.0)


# ---------------------------------------------------------------------------------------------
# Choosing a configuration (no GPU)


AMPERE, TURING = (8, 6), (7, 5)  # compute capabilities: RTX 30 series, GTX 16 series


def test_the_configurations_are_ordered_best_first_and_chosen_by_budget_and_length():
    streamed, resident, small, wide = VAE_CONFIGS
    assert [c.name for c in VAE_CONFIGS] == [
        "bf16-streamed", "bf16-resident", "bf16-streamed-small", "fp32-streamed"
    ]  # fmt: skip
    assert streamed.max_tile_cells > resident.max_tile_cells > small.max_tile_cells
    assert small.max_tile_cells == wide.max_tile_cells and VRAM_MARGIN == 300 * MiB
    for frames in (17, 33, 49, 121):
        peaks = [config.peak_bytes(frames) for config in VAE_CONFIGS]
        assert peaks[0] > peaks[1] > peaks[3] > peaks[2], "small tiles in bf16 need the least"

        def choose(budget, frames=frames, **options):
            return choose_vae_config(budget, frames, capability=AMPERE, **options)

        assert choose(1 << 40) is streamed
        assert choose(peaks[0] + VRAM_MARGIN) is streamed
        assert choose(peaks[0] + VRAM_MARGIN - 1) is resident
        assert choose(peaks[1] + VRAM_MARGIN - 1) is small, "before float32: faster, and less"
        assert choose(peaks[2], margin=0) is small
        with pytest.raises(ValueError, match=f"bf16-streamed-small, peaks at .* for {frames} fr"):
            choose(peaks[2] + VRAM_MARGIN - 1)
    for known in VAE_CONFIGS:
        assert known.measured_frames == (17, 121), "what was measured"
        by_length = [known.peak_bytes(frames) for frames in (17, 33, 49, 85, 121)]
        assert by_length == sorted(by_length), "never less for a longer video"
        assert by_length[::2] == [nbytes for _, nbytes in known.peaks]
    # Between two measured lengths: on the line between them.
    (_, low), (_, mid), (_, high) = resident.peaks
    assert resident.peak_bytes(33) == (low + mid) // 2
    assert resident.peak_bytes(85) == (mid + high) // 2
    # The reference laptop under the F3 criterion: 3.4 GB of VRAM for the whole process.
    assert choose_vae_config(3_400_000_000, 49, capability=AMPERE) is resident
    assert choose_vae_config(3_400_000_000, 121, capability=AMPERE) is resident
    only = (config("float32", True, 9),)
    assert choose_vae_config(100, 1, configs=only, margin=10, capability=AMPERE).name == "test"


def test_a_length_that_was_not_measured_is_refused_not_extrapolated():
    for frames in (1, 13, 16, 122, 125, 241):
        for known in VAE_CONFIGS:
            with pytest.raises(ValueError, match=f"17 to 121 frames: nothing is known about {frames}"):
                known.peak_bytes(frames)
        with pytest.raises(ValueError, match="measured for videos of 17 to 121 frames"):
            choose_vae_config(1 << 40, frames, capability=AMPERE)
        with pytest.raises(ValueError, match="fp32-streamed was measured for videos of 17 to"):
            choose_vae_config(1 << 40, frames, capability=TURING)


def test_a_gpu_without_native_bfloat16_gets_the_float32_configuration():
    wide = VAE_CONFIGS[-1]
    assert wide.dtype == "float32"
    assert usable_vae_configs(VAE_CONFIGS, TURING) == (wide,)
    assert usable_vae_configs(VAE_CONFIGS, (6, 1)) == (wide,)
    assert usable_vae_configs(VAE_CONFIGS, (8, 0)) == VAE_CONFIGS
    assert usable_vae_configs(VAE_CONFIGS, (12, 0)) == VAE_CONFIGS
    assert choose_vae_config(1 << 40, 49, capability=TURING) is wide
    with pytest.raises(ValueError, match=r"compute capability 7\.5: the bfloat16 config"):
        choose_vae_config(wide.peak_bytes(49) + VRAM_MARGIN - 1, 49, capability=TURING)
    with pytest.raises(ValueError, match="no VAE configuration can run on this GPU"):
        choose_vae_config(1 << 40, 49, capability=TURING, configs=VAE_CONFIGS[:3])


# ---------------------------------------------------------------------------------------------
# Decoding


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("dtype", "streamed"),
    [("bfloat16", False), ("bfloat16", True), ("float32", True), ("float32", False)],
)
def test_a_frame_that_fits_in_one_tile_is_decoded_as_diffusers_decodes_it(
    diffusers, disk_tmp, dtype, streamed
):
    import torch

    tiny_wan_vae(disk_tmp)
    plain = reference(disk_tmp, dtype)
    sample = latents()
    with torch.no_grad():
        whole = plain.decode(denormalised(plain, sample, dtype), return_dict=False)[0][0]
    expected = to_frames(whole.cpu())
    assert expected.shape == (9, 96, 160, 3) and float(expected.float().std()) > 10
    with load_wan_vae(disk_tmp, config(dtype, streamed, 60)) as ours:
        assert ours.plan(*FRAME).count == 1
        frames = ours.decode(sample)
        assert frames.dtype == torch.uint8 and frames.device.type == "cpu"
        assert torch.equal(frames, expected)
        assert torch.equal(ours.decode(sample.to("cuda", torch.float64)), expected)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("dtype", "streamed", "slots", "max_tile_cells", "margin"),
    [
        ("bfloat16", False, 2, 24, 1),
        ("bfloat16", True, 2, 30, 1),
        ("bfloat16", True, 1, 30, 0),
        ("float32", True, 3, 16, 1),
    ],
)
def test_tiles_are_decoded_as_diffusers_decodes_them(
    diffusers, disk_tmp, dtype, streamed, slots, max_tile_cells, margin
):
    """Weights resident or streamed, converted or not: each tile has the bits of ``diffusers``."""
    import torch

    tiny_wan_vae(disk_tmp)
    plain = reference(disk_tmp, dtype)
    sample = latents()
    with load_wan_vae(disk_tmp, config(dtype, streamed, max_tile_cells), slots=slots) as ours:
        plan = ours.plan(*FRAME, min_overlap=2)
        assert plan.count > 1
        frames = ours.decode(sample, min_overlap=2, margin=margin)
        if streamed:  # the decoder runs once per latent frame of every tile
            assert len(ours.streamer.pass_stats()) == plan.count * LATENT_FRAMES
    z = denormalised(plain, sample, dtype)
    canvas = Canvas(torch, plan, FRAME, scale=16, margin=margin)
    rows, columns = plan.tile
    with torch.no_grad():
        for i, top in enumerate(plan.rows):
            for j, left in enumerate(plan.columns):
                tile = z[:, :, :, top : top + rows, left : left + columns]
                canvas.add(i, j, plain.decode(tile, return_dict=False)[0][0])
    assert torch.equal(frames, to_frames(canvas.result()))


@pytest.mark.cuda
def test_cudnn_benchmark_is_off_while_decoding_and_put_back(diffusers, disk_tmp):
    import torch

    tiny_wan_vae(disk_tmp)
    sample = latents()
    seen = []
    with load_wan_vae(disk_tmp, config("bfloat16", False, 60)) as ours:
        expected = ours.decode(sample)
        spy = ours.vae.decoder.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(torch.backends.cudnn.benchmark),
            with_kwargs=True,
        )
        torch.backends.cudnn.benchmark = True
        try:
            frames = ours.decode(sample)
            assert torch.backends.cudnn.benchmark is True
        finally:
            torch.backends.cudnn.benchmark = False
        spy.remove()
    assert seen == [False] * LATENT_FRAMES
    assert torch.equal(frames, expected)


# ---------------------------------------------------------------------------------------------
# Where the weights are, and what is refused


@pytest.mark.cuda
@pytest.mark.parametrize("streamed", [False, True])
def test_only_the_decoder_is_loaded_and_close_gives_the_skeleton_back(
    diffusers, disk_tmp, streamed
):
    import torch

    tiny_wan_vae(disk_tmp)
    gc.collect()
    before = torch.cuda.memory_allocated()
    ours = load_wan_vae(disk_tmp, config("bfloat16", streamed, 60))
    vae = ours.vae
    unused = [*vae.encoder.parameters(), *vae.quant_conv.parameters()]
    used = [*vae.decoder.parameters(), *vae.post_quant_conv.parameters()]
    assert unused and all(param.device.type == "meta" for param in unused)
    assert all(param.dtype == torch.bfloat16 and not param.requires_grad for param in used)
    on_gpu = sum(p.numel() * p.element_size() for p in used if p.device.type == "cuda")
    waiting = sum(p.numel() * p.element_size() for p in used if p.device.type == "meta")
    assert ours.resident_bytes == on_gpu > 0
    if streamed:
        assert waiting > on_gpu, "the blocks hold most of the decoder"
        assert ours.vram_bytes == 2 * ours.streamer.slot_bytes + on_gpu
        assert ours.ram_bytes == ours.streamer.ram_bytes >= waiting
    else:
        assert waiting == 0 and ours.streamer is None
        assert ours.vram_bytes == on_gpu and ours.ram_bytes == 0
    assert ours.decode(latents()).shape == (9, 96, 160, 3)
    del used, unused  # the last references of the test to the weights on the GPU
    ours.close()
    ours.close()
    assert torch.cuda.memory_allocated() == before
    assert all(param.device.type == "meta" for param in vae.parameters())
    assert not [m for m in vae.modules() if m._forward_hooks or m._forward_pre_hooks]


@pytest.mark.cuda
def test_what_cannot_be_loaded_or_decoded_is_refused(diffusers, disk_tmp):
    import json

    import torch

    tiny_wan_vae(disk_tmp)
    gc.collect()
    before = torch.cuda.memory_allocated()
    with pytest.raises(ValueError, match="no VAE configuration is called 'fp16'"):
        load_wan_vae(disk_tmp, "fp16")
    with load_wan_vae(disk_tmp, config("bfloat16", False, 60)) as ours:
        with pytest.raises(ValueError, match="this decodes one video"):
            ours.decode(torch.zeros(2, Z, 1, 6, 10))
        with pytest.raises(ValueError, match="this decodes one video"):
            ours.decode(torch.zeros(1, Z + 1, 1, 6, 10))
    small = load_wan_vae(disk_tmp, config("bfloat16", False, 20))
    with small, pytest.raises(ValueError, match="no tile of at most 20 latent cells"):
        small.decode(latents())
    del small
    stored = json.loads((disk_tmp / "config.json").read_text())
    (disk_tmp / "config.json").write_text(json.dumps({**stored, "_class_name": "Other"}))
    with pytest.raises(ValueError, match="holds a Other"):
        load_wan_vae(disk_tmp)
    assert torch.cuda.memory_allocated() == before


# ---------------------------------------------------------------------------------------------
# The real VAE: Wan 2.2 TI2V-5B (needs HYPERTIER_MODELS and -m slow)

REAL = "Wan2.2-TI2V-5B-Diffusers"
FAST = "FastWan2.2-TI2V-5B-FullAttn-Diffusers"  # its transformer, distilled for three steps
FAST_STEPS = (1000, 757, 522)
CLIP = (3, 30, 52)  # 480p x 9 frames: short to sample, and as wide as the long mode
COMPARED = 2  # latent frames decoded and compared: the first 5 frames
PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a "
    "spotlighted stage."
)
NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def psnr(frames, expected):
    import torch

    error = (frames.double() - expected.double()).pow(2).mean()
    return float(10 * torch.log10(255.0**2 / error))


@pytest.fixture(scope="module")
def real_dir():
    root = os.environ.get("HYPERTIER_MODELS")
    if not root or not (Path(root) / REAL / "vae").is_dir():
        pytest.skip(f"set HYPERTIER_MODELS to a directory that holds {REAL}")
    pytest.importorskip("diffusers")
    return Path(root) / REAL


@pytest.fixture(scope="module")
def real_embeds(real_dir):
    """The prompt and the negative prompt through the real text encoder."""
    import torch
    import transformers
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    from hypertier.adapters import load_text_encoder

    tokenizer = transformers.AutoTokenizer.from_pretrained(real_dir / "tokenizer")
    batch = tokenizer(
        [prompt_clean(PROMPT), prompt_clean(NEGATIVE)], padding="max_length", max_length=512,
        truncation=True, return_attention_mask=True, return_tensors="pt",
    )  # fmt: skip
    with load_text_encoder(real_dir / "text_encoder") as encoder, torch.no_grad():
        hidden = encoder.model(batch.input_ids.cuda(), batch.attention_mask.cuda())
        hidden = hidden.last_hidden_state.cpu()
    for row, used in enumerate(batch.attention_mask.sum(1).tolist()):
        hidden[row, used:] = 0  # as WanPipeline: the tokens of the prompt, then zeros
    gc.collect()
    torch.cuda.empty_cache()
    return hidden


@pytest.fixture(scope="module", params=["wan", "fastwan"])
def real_clip(request, real_dir, real_embeds):
    """Latents of a real video, sampled here: ten steps of the model, or three of FastWan.

    Tiling is judged on what a model draws: random latents decode to noise, where the
    borders of a tile matter in another way. Both samplers, because the thresholds were
    measured on the first and a video in few steps is what most users will decode.
    """
    import diffusers
    import torch

    seeded = torch.Generator("cpu").manual_seed(0)
    if request.param == "fastwan":
        fast = real_dir.parent / FAST / "transformer"
        if not fast.is_dir():
            pytest.skip(f"set HYPERTIER_MODELS to a directory that holds {FAST}")
        with load_wan_transformer(fast) as transformer:
            clip = denoise_dmd(
                transformer.model, real_embeds[:1], latent_shape=CLIP, timesteps=FAST_STEPS,
                generator=seeded,
            ).cpu()  # fmt: skip
    else:
        with load_wan_transformer(real_dir / "transformer") as transformer:
            clip = denoise(
                transformer.model,
                diffusers.UniPCMultistepScheduler.from_pretrained(real_dir / "scheduler"),
                real_embeds[:1], real_embeds[1:], latent_shape=CLIP, expand_timesteps=True,
                num_inference_steps=10, generator=seeded,
            ).cpu()  # fmt: skip
    gc.collect()
    torch.cuda.empty_cache()
    return clip[:, :, :COMPARED].float()


@pytest.fixture(scope="module")
def real_reference(real_dir, real_clip):
    """The untiled float32 decode, which only fits on the CPU: about two minutes."""
    import diffusers
    import torch

    vae = diffusers.AutoencoderKLWan.from_pretrained(real_dir / "vae", torch_dtype=torch.float32)
    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean).view(shape)
    std = torch.tensor(vae.config.latents_std).view(shape)
    with torch.inference_mode():
        video = vae.eval().decode(real_clip / (1.0 / std) + mean, return_dict=False)[0][0]
        frames = video.div_(2).add_(0.5).clamp_(0, 1).mul_(255).round_().to(torch.uint8)
    return frames.permute(1, 2, 3, 0).contiguous()


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.parametrize("name", [known.name for known in VAE_CONFIGS])
def test_the_real_vae_reaches_the_psnr_of_its_configuration(
    request, real_dir, real_clip, real_reference, name
):
    import torch

    sampler = request.node.callspec.params["real_clip"]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with load_wan_vae(real_dir / "vae", name) as ours:
        chosen = ours.config
        plan = ours.plan(*CLIP[1:])
        frames = ours.decode(real_clip)
        reserved = torch.cuda.max_memory_reserved()
    quality = psnr(frames, real_reference)
    print(
        f"\n[vae] {sampler}, {name}: {plan.count} tiles of {plan.tile[0]} x {plan.tile[1]} cells, "
        f"{quality:.1f} dB against the untiled float32 decode (at least "
        f"{chosen.min_psnr_db:.1f}), {reserved / MiB:.0f} MiB reserved at the peak"
    )
    assert frames.shape == real_reference.shape == (4 * COMPARED - 3, 480, 832, 3)
    assert float(real_reference.float().std()) > 20, "the clip is a picture, not a flat frame"
    assert quality >= chosen.min_psnr_db
    promised = chosen.peaks[0][1]  # of the shortest video measured: this clip is shorter
    assert reserved <= promised, "under the peak the table promises to the planner"
