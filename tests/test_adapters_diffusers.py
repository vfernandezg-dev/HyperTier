# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.adapters.hf_diffusers.

They need a GPU and ``diffusers``. A tiny random Wan transformer is saved as a normal
checkpoint and compared with the same checkpoint loaded by ``diffusers``, fully resident in
VRAM: one pass against its ``forward``, and the sampling loop against ``WanPipeline``.

Everything exact is compared bit for bit. In bfloat16 a sampler turns a difference of one
unit in the last place into another video, so a tolerance at the end of the loop would accept
a wrong loop; what cannot be exact is judged by the error of one pass.

With ``HYPERTIER_MODELS`` and ``-m slow`` the same is checked on Wan 2.2 TI2V-5B (about five
minutes): against ``WanPipeline`` at a size it can run on 4 GB, and against the strict baseline
at full size.
"""

import gc
import json
import os
import sys
from pathlib import Path

import pytest

from hypertier.adapters import hf_diffusers
from hypertier.adapters.hf_diffusers import (
    denoise,
    denoise_dmd,
    dmd_sigmas,
    load_wan_transformer,
)
from hypertier.diffusion import chunk_tokens, runs_one_sample_at_a_time

MiB = 1 << 20
LATENT = (3, 8, 8)  # frames, height, width in latent cells: 48 tokens with 1 x 2 x 2 patches
TOKENS = 48
CHANNELS = 16
TEXT_DIM = 32
KEPT_IN_FP32 = {"rope", "time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"}

CHUNKED_OVER_NOISE = 1.5
"""How far a chunked feed-forward network may move one pass, in units of the rounding noise
of that pass: its distance to the same pass run in the next wider dtype. Chunking changes
which kernel rounds, so its error has to be of the size of rounding, not a number picked by
hand. Measured on Wan 2.2 TI2V-5B at 1,950 tokens: a bfloat16 pass is 1.1-1.9 % from float32,
and chunking moves it by 0.43-0.74 of that."""


@pytest.fixture
def diffusers():
    return pytest.importorskip("diffusers")


def tiny_wan_transformer(directory, *, stored="float32", seed=0):
    """Write a random tiny Wan transformer as a normal ``diffusers`` checkpoint.

    Three blocks, so that slots are reused in another order on every pass. Returns
    ``directory``.
    """
    import torch
    from diffusers import WanTransformer3DModel

    torch.manual_seed(seed)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
        in_channels=CHANNELS, out_channels=CHANNELS, text_dim=TEXT_DIM, freq_dim=32, ffn_dim=48,
        num_layers=3, cross_attn_norm=True, qk_norm="rms_norm_across_heads", rope_max_seq_len=32,
    )  # fmt: skip
    with torch.no_grad():
        for param in model.parameters():
            if not param.any():  # the output projection starts at zero: every output equal
                param.normal_(0, 0.5)
        for param in model.parameters():  # all of it, as a distilled checkpoint stores it
            param.data = param.data.to(getattr(torch, stored))
    model.save_pretrained(directory)
    return directory


def reference(directory, dtype):
    """The checkpoint as ``diffusers`` loads it, resident in VRAM."""
    import torch
    from diffusers import WanTransformer3DModel

    model = WanTransformer3DModel.from_pretrained(directory, torch_dtype=getattr(torch, dtype))
    return model.to("cuda").eval()


def inputs(dtype, *, batch=1, seed=1, text_tokens=20):
    """A latent and a text conditioning per sample, on the GPU."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    latent = torch.randn(batch, CHANNELS, *LATENT, generator=gen)
    text = torch.randn(batch, text_tokens, TEXT_DIM, generator=gen)
    return latent.to("cuda", getattr(torch, dtype)), text.to("cuda", getattr(torch, dtype))


def per_token(*timesteps):
    """One timestep per token, the same for every token of a sample: ``[batch, tokens]``."""
    import torch

    return torch.tensor(timesteps, dtype=torch.float32, device="cuda")[:, None].expand(-1, TOKENS)


def run(model, latent, timestep, text):
    return model(
        hidden_states=latent, timestep=timestep, encoder_hidden_states=text, return_dict=False
    )[0]


def one_by_one(model, latent, timestep, text):
    """A batch through ``model`` as one pass per sample: what a batch has to equal."""
    import torch

    rows = range(latent.shape[0])
    return torch.cat([run(model, latent[i : i + 1], timestep[i : i + 1], text[i : i + 1]) for i in rows])


def relative_error(got, expected):
    """Root mean square of the difference, over that of ``expected``."""
    got, expected = got.double().cpu(), expected.double().cpu()
    return float((got - expected).pow(2).mean().sqrt() / expected.pow(2).mean().sqrt())


def make_scheduler(diffusers, kind):
    if kind == "euler":
        return diffusers.FlowMatchEulerDiscreteScheduler(shift=5.0)
    return diffusers.UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5.0
    )


def pipeline_latents(diffusers, model, scheduler, cond, uncond, *, expand, steps, guidance, seed):
    """Latents of ``WanPipeline`` around ``model``; without a VAE it assumes 8 px per cell."""
    import torch

    pipe = diffusers.WanPipeline(
        tokenizer=None, text_encoder=None, vae=None, scheduler=scheduler, transformer=model,
        expand_timesteps=expand,
    )  # fmt: skip
    pipe.set_progress_bar_config(disable=True)
    frames, height, width = LATENT
    return pipe(
        prompt_embeds=cond, negative_prompt_embeds=uncond, height=8 * height, width=8 * width,
        num_frames=4 * (frames - 1) + 1, num_inference_steps=steps, guidance_scale=guidance,
        generator=torch.Generator("cpu").manual_seed(seed), output_type="latent",
    ).frames  # fmt: skip


def test_importing_the_adapter_imports_neither_torch_nor_diffusers():
    import subprocess

    code = (
        "import sys, hypertier.adapters.hf_diffusers, hypertier.diffusion; "
        "sys.exit(('torch' in sys.modules) + 2 * ('diffusers' in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


# ---------------------------------------------------------------------------------------------
# One pass


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("stored", "dtype", "slots"),
    [
        ("float32", "bfloat16", 2),  # the 5B checkpoint: stored in float32, run in bfloat16
        ("float32", "bfloat16", 1),
        ("float32", "bfloat16", 3),
        ("float32", "float16", 2),
        ("float32", "float32", 2),
        ("bfloat16", "bfloat16", 2),  # a distilled checkpoint: the float32 modules come back
        ("bfloat16", "float32", 2),
    ],
)
def test_passes_give_the_bits_of_diffusers(diffusers, disk_tmp, stored, dtype, slots):
    import torch

    tiny_wan_transformer(disk_tmp, stored=stored)
    want = reference(disk_tmp, dtype)
    latent, text = inputs(dtype)
    with load_wan_transformer(disk_tmp, dtype=dtype, slots=slots) as ours, torch.no_grad():
        assert ours.streamer.tiers == ("ram",) * 3
        for step, t in enumerate([999.0, 757.0, 301.0, 12.0]):
            # Per token as Wan 2.2 TI2V takes it, then per sample as the other models do.
            timestep = per_token(t) if step % 2 == 0 else torch.tensor([t], device="cuda")
            expected = run(want, latent, timestep, text)
            assert bool(torch.isfinite(expected).all()) and float(expected.float().std()) > 0.1
            assert torch.equal(run(ours.model, latent, timestep, text), expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_a_batch_gives_the_bits_of_its_samples_run_alone(diffusers, disk_tmp, dtype):
    import torch

    tiny_wan_transformer(disk_tmp)
    want = reference(disk_tmp, dtype)
    latent, text = inputs(dtype, batch=3)
    with load_wan_transformer(disk_tmp, dtype=dtype) as ours, torch.no_grad():
        for timestep in (per_token(757.0, 301.0, 12.0), torch.tensor([757.0, 301.0, 12.0])):
            timestep = timestep.to("cuda")
            got = run(ours.model, latent, timestep, text)
            assert torch.equal(got, one_by_one(want, latent, timestep, text))
        assert len(ours.streamer.pass_stats()) == 2, "the weights go up once per batch"


@pytest.mark.cuda
def test_one_row_of_the_time_embedding_is_kept_when_the_tokens_share_a_timestep(
    diffusers, disk_tmp
):
    import torch

    tiny_wan_transformer(disk_tmp)
    want = reference(disk_tmp, "bfloat16")
    latent, text = inputs("bfloat16", batch=2)
    seen = []
    mixed = per_token(757.0).clone()
    mixed[:, :16] = 0.0  # image-to-video: the tokens of the frames that are given
    with load_wan_transformer(disk_tmp) as ours, torch.no_grad():
        spy = ours.model.blocks[0].register_forward_pre_hook(
            lambda module, args: seen.append(tuple(args[2].shape))
        )
        cases = [
            (latent[:1], per_token(757.0), text[:1]),
            (latent[:1], mixed, text[:1]),
            (latent, per_token(757.0, 301.0), text),  # each sample has its own
            (latent, torch.cat([per_token(757.0), mixed]), text),  # one sample is not uniform
        ]
        for case in cases:
            assert torch.equal(run(ours.model, *case), one_by_one(want, *case))
        spy.remove()
    dim = 24
    assert seen == [(1, 1, 6, dim), (1, TOKENS, 6, dim), (2, 1, 6, dim), (2, TOKENS, 6, dim)]


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("dtype", "wider"),
    [("bfloat16", "float32"), ("float16", "float32"), ("float32", "float64")],
)
def test_a_chunked_ffn_is_judged_by_the_error_of_one_pass(diffusers, disk_tmp, dtype, wider):
    """Not exact, and not asked to be: its error is measured against the rounding of the pass."""
    import torch

    tiny_wan_transformer(disk_tmp)
    latent, text = inputs(dtype)
    timestep = per_token(757.0)
    calls = []
    with torch.no_grad():
        whole = run(reference(disk_tmp, dtype), latent, timestep, text)
        wide = getattr(torch, wider)
        exact = run(reference(disk_tmp, wider), latent.to(wide), timestep, text.to(wide))
        with load_wan_transformer(disk_tmp, dtype=dtype, ffn_chunk=20) as ours:
            inner = ours.model.blocks[0].ffn.net[0]
            spy = inner.register_forward_pre_hook(
                lambda module, args: calls.append(args[0].shape[1])
            )
            chunked = run(ours.model, latent, timestep, text)
            spy.remove()
    noise, error = relative_error(whole, exact), relative_error(chunked, whole)
    print(
        f"\n[wan] {dtype}: one pass is {noise:.2e} from {wider}; the ffn in chunks of 20 of "
        f"{TOKENS} tokens moves it {error:.2e}, {error / noise:.2f} of that"
    )
    assert calls == [20, 20, 8]
    assert error <= CHUNKED_OVER_NOISE * noise


# ---------------------------------------------------------------------------------------------
# The sampling loop


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("dtype", "kind", "expand"),
    [
        ("bfloat16", "unipc", True),  # Wan 2.2 TI2V-5B as its pipeline runs it
        ("bfloat16", "euler", True),
        ("bfloat16", "unipc", False),  # the models with one timestep per sample
        ("float32", "unipc", True),
    ],
)
def test_denoise_returns_the_latents_of_wanpipeline(diffusers, disk_tmp, dtype, kind, expand):
    import torch

    tiny_wan_transformer(disk_tmp)
    want = reference(disk_tmp, dtype)
    shape = (1, CHANNELS, *LATENT)
    with load_wan_transformer(disk_tmp, dtype=dtype) as ours:
        for seed, text_tokens, guidance, steps in [
            (0, 20, 5.0, 6), (7, 20, 5.0, 6), (7, 9, 5.0, 6), (3, 33, 3.5, 4),
            (0, 20, 1.0, 6), (7, 9, 1.0, 5), (5, 20, 0.0, 4),
        ]:  # fmt: skip
            text = inputs(dtype, batch=2, seed=100 + seed, text_tokens=text_tokens)[1]
            cond, uncond = text[:1], text[1:]
            expected = pipeline_latents(
                diffusers, want, make_scheduler(diffusers, kind), cond, uncond,
                expand=expand, steps=steps, guidance=guidance, seed=seed,
            )  # fmt: skip
            assert tuple(expected.shape) == shape
            assert bool(torch.isfinite(expected).all()) and float(expected.float().std()) > 0.1
            for one_pass in (False, True):
                got = denoise(
                    ours.model, make_scheduler(diffusers, kind), cond,
                    uncond if guidance > 1 else None, latent_shape=LATENT,
                    expand_timesteps=expand, num_inference_steps=steps,
                    guidance_scale=guidance, one_pass=one_pass,
                    generator=torch.Generator("cpu").manual_seed(seed),
                )  # fmt: skip
                # float32 from UniPC; the Euler scheduler returns the dtype of the model.
                assert got.dtype == expected.dtype and got.device.type == "cuda"
                assert torch.equal(got, expected), (seed, text_tokens, guidance, one_pass)


@pytest.mark.cuda
def test_one_pass_of_guidance_copies_the_weights_once_per_step(diffusers, disk_tmp):
    import torch

    tiny_wan_transformer(disk_tmp)
    text = inputs("bfloat16", batch=2)[1]
    noise = torch.randn(1, CHANNELS, *LATENT, generator=torch.Generator().manual_seed(4))
    with load_wan_transformer(disk_tmp) as ours:
        passes = []
        results = []
        for one_pass in (False, True):
            before = len(ours.streamer.pass_stats())
            latents = denoise(
                ours.model, make_scheduler(diffusers, "unipc"), text[:1], text[1:],
                latent_shape=LATENT, expand_timesteps=True, num_inference_steps=5,
                latents=noise, one_pass=one_pass,
            )  # fmt: skip
            results.append(latents)
            passes.append(len(ours.streamer.pass_stats()) - before)
    assert passes == [10, 5]
    assert torch.equal(results[0], results[1])


# ---------------------------------------------------------------------------------------------
# A few steps, with a distilled transformer

FEW_STEPS = (1000, 757, 522)  # the timesteps FastWan is distilled for


def test_the_sigmas_of_a_few_steps_are_read_on_a_schedule_with_shift_8():
    torch = pytest.importorskip("torch")
    sigmas = dmd_sigmas(torch, FEW_STEPS)
    assert sigmas.dtype == torch.float32 and sigmas.device.type == "cpu"
    # 1, 28/37 and 12/23 as the float32 arithmetic of the schedule leaves them.
    assert [value.item().hex() for value in sigmas] == [
        "0x1.0000000000000p+0", "0x1.83759e0000000p-1", "0x1.0b21640000000p-1",
    ]  # fmt: skip
    assert sigmas.tolist() == pytest.approx([1.0, 28 / 37, 12 / 23], abs=1e-7)
    # The nearest timestep of the table, which has 756.76 and 757.67 but none at 757.
    assert dmd_sigmas(torch, (758,)).tolist() == pytest.approx([8 * 0.281 / (1 + 7 * 0.281)])
    assert not torch.equal(dmd_sigmas(torch, (758,)), sigmas[1:2])
    # The shift of the scheduler in the checkpoint is not the one of the sampler.
    assert dmd_sigmas(torch, FEW_STEPS, shift=5.0).tolist() == pytest.approx(
        [1.0, 0.7570978, 0.5215617], abs=1e-7
    )
    assert dmd_sigmas(torch, (1,), shift=1.0).tolist() == pytest.approx([0.001])


def fastvideo_loop(model, cond, steps, generator):
    """The few-step loop as FastVideo runs it, operation by operation.

    Transcribed from ``DmdDenoisingStage``, ``pred_noise_to_pred_video`` and the ``add_noise``
    of its flow-matching scheduler (github.com/hao-ai-lab/FastVideo, Apache-2.0), around a
    ``WanTransformer3DModel`` of ``diffusers``: latents as ``[batch, frames, channels, height,
    width]``, every frame a row with a sigma of its own looked up in the table.

    The sources, at commit ``a7eb3401de33d61530c09bcdb6ced12de322b719``, to read this function
    against line by line. The loop, ``DmdDenoisingStage.forward``:

        https://github.com/hao-ai-lab/FastVideo/blob/a7eb3401de33d61530c09bcdb6ced12de322b719/fastvideo/pipelines/basic/wan/stages/dmd.py#L95-L168

    The clean video from the prediction, ``pred_noise_to_pred_video``:

        https://github.com/hao-ai-lab/FastVideo/blob/a7eb3401de33d61530c09bcdb6ced12de322b719/fastvideo/models/utils.py#L170-L208

    The table of sigmas and timesteps, and ``add_noise``, of ``FlowMatchEulerDiscreteScheduler``:

        https://github.com/hao-ai-lab/FastVideo/blob/a7eb3401de33d61530c09bcdb6ced12de322b719/fastvideo/models/schedulers/scheduling_flow_match_euler_discrete.py#L141-L157
        https://github.com/hao-ai-lab/FastVideo/blob/a7eb3401de33d61530c09bcdb6ced12de322b719/fastvideo/models/schedulers/scheduling_flow_match_euler_discrete.py#L601-L635

    The shift of that table in this stage, ``DMD_TRAINING_NOISE_SHIFT`` = 8.0, and the
    timesteps of FastWan 2.2 TI2V-5B, 1000, 757 and 522:

        https://github.com/hao-ai-lab/FastVideo/blob/a7eb3401de33d61530c09bcdb6ced12de322b719/fastvideo/pipelines/basic/wan/wan_dmd_pipeline.py#L61
        https://github.com/hao-ai-lab/FastVideo/blob/a7eb3401de33d61530c09bcdb6ced12de322b719/fastvideo/models/wan/definition.py#L16
        https://github.com/hao-ai-lab/FastVideo/blob/a7eb3401de33d61530c09bcdb6ced12de322b719/fastvideo/models/wan/pipeline_config.py#L278-L280

    What is not theirs here: the transformer, which is the one of ``diffusers`` called through
    ``run``, without their autocast, attention backends or image conditioning.
    """
    import numpy as np
    import torch

    timesteps = np.linspace(1, 1000, 1000, dtype=np.float32)[::-1].copy()
    sigmas = torch.from_numpy(timesteps).to(torch.float32) / 1000
    sigmas = 8.0 * sigmas / (1 + (8.0 - 1) * sigmas)
    table = (sigmas * 1000).to("cuda")
    sigmas = sigmas.to("cuda")
    frames, height, width = LATENT
    shape = (1, frames, CHANNELS, height, width)
    latents = torch.randn(shape, generator=generator, dtype=torch.float32).to("cuda")
    steps = torch.tensor(steps, dtype=torch.long, device="cuda")
    for i, t in enumerate(steps):
        noise_latents = latents.clone()
        t_expand = t.repeat(latents.shape[0])
        with torch.no_grad():
            pred_noise = run(
                model, latents.to(model.dtype).permute(0, 2, 1, 3, 4), t_expand, cond
            ).permute(0, 2, 1, 3, 4)
        # pred_noise_to_pred_video: float64, the sigma of the nearest timestep of the table
        rows = pred_noise.flatten(0, 1)
        timestep = t_expand.expand(rows.shape[0])
        nearest = torch.argmin((table.double().unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas.double()[nearest].reshape(-1, 1, 1, 1)
        pred_video = noise_latents.flatten(0, 1).double() - sigma_t * rows.double()
        pred_video = pred_video.to(rows.dtype).unflatten(0, pred_noise.shape[:2])
        if i < len(steps) - 1:
            next_timestep = steps[i + 1] * torch.ones([1], dtype=torch.long, device="cuda")
            noise = torch.randn(shape, dtype=pred_video.dtype, generator=generator).to("cuda")
            # add_noise: float32
            clean, noise_rows = pred_video.flatten(0, 1), noise.flatten(0, 1)
            timestep = next_timestep.expand(clean.shape[0])
            nearest = torch.argmin((table.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
            sigma = sigmas[nearest].reshape(-1, 1, 1, 1)
            sample = (1 - sigma) * clean + sigma * noise_rows
            latents = sample.type_as(noise_rows).unflatten(0, pred_video.shape[:2])
        else:
            latents = pred_video
    return latents.permute(0, 2, 1, 3, 4)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_a_few_steps_give_the_latents_of_the_loop_of_fastvideo(diffusers, disk_tmp, dtype):
    import torch

    tiny_wan_transformer(disk_tmp)
    want = reference(disk_tmp, dtype)
    with load_wan_transformer(disk_tmp, dtype=dtype) as ours:
        for seed, text_tokens, steps in [
            (0, 20, FEW_STEPS), (1024, 20, FEW_STEPS), (7, 9, FEW_STEPS), (3, 33, (1000, 500)),
            (5, 20, (1000,)), (2, 20, (999, 800, 600, 400, 200)),
        ]:  # fmt: skip
            cond = inputs(dtype, seed=100 + seed, text_tokens=text_tokens)[1]
            expected = fastvideo_loop(want, cond, steps, torch.Generator("cpu").manual_seed(seed))
            assert bool(torch.isfinite(expected).all()) and float(expected.float().std()) > 0.1
            for model in (ours.model, want):  # streamed, and the same loop on the resident one
                got = denoise_dmd(
                    model, cond, latent_shape=LATENT, timesteps=steps,
                    generator=torch.Generator("cpu").manual_seed(seed),
                )  # fmt: skip
                assert got.shape == (1, CHANNELS, *LATENT) and got.device.type == "cuda"
                assert got.dtype == expected.dtype == getattr(torch, dtype)
                assert torch.equal(got, expected), (seed, text_tokens, steps)


@pytest.mark.cuda
def test_a_few_steps_are_one_pass_each_with_one_timestep_per_sample(diffusers, disk_tmp):
    import torch

    tiny_wan_transformer(disk_tmp)
    cond = inputs("bfloat16", text_tokens=20)[1]
    frames, height, width = LATENT
    with load_wan_transformer(disk_tmp) as ours:
        seen = []
        spy = ours.model.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs["timestep"].clone()), with_kwargs=True
        )
        generator = torch.Generator("cpu").manual_seed(11)
        drawn = denoise_dmd(
            ours.model, cond, latent_shape=LATENT, timesteps=FEW_STEPS, generator=generator
        )
        assert len(ours.streamer.pass_stats()) == 3, "no guidance: one pass per step"
        assert [tuple(t.shape) for t in seen] == [(1,)] * 3, "not one timestep per token"
        assert all(t.dtype == torch.long for t in seen)
        assert [int(t) for t in seen] == list(FEW_STEPS), "the nominal ones, not the table's"
        # The initial noise is the first draw of the generator, as [1, frames, channels, ...].
        generator = torch.Generator("cpu").manual_seed(11)
        noise = torch.randn(1, frames, CHANNELS, height, width, generator=generator)
        given = denoise_dmd(
            ours.model, cond, latent_shape=LATENT, timesteps=FEW_STEPS, generator=generator,
            latents=noise.permute(0, 2, 1, 3, 4),
        )  # fmt: skip
        assert torch.equal(given, drawn)
        other = denoise_dmd(
            ours.model, cond, latent_shape=LATENT, timesteps=FEW_STEPS,
            generator=torch.Generator("cpu").manual_seed(12), latents=noise.permute(0, 2, 1, 3, 4),
        )  # fmt: skip
        assert not torch.equal(other, drawn), "the noise added back comes from the generator"
        spy.remove()


@pytest.mark.cuda
def test_a_few_steps_refuse_what_they_would_get_wrong(diffusers, disk_tmp):
    import torch

    tiny_wan_transformer(disk_tmp)
    plain = reference(disk_tmp, "bfloat16")
    text = inputs("bfloat16", batch=2)[1]

    def sample(embeds=text[:1], **changes):
        return denoise_dmd(plain, embeds, **{"latent_shape": LATENT, "timesteps": FEW_STEPS, **changes})

    with pytest.raises(ValueError, match="samples one video"):
        sample(text)
    for wrong in ((), (522, 757, 1000), (1000, 757, 757)):
        with pytest.raises(ValueError, match="go down"):
            sample(timesteps=wrong)
    with pytest.raises(ValueError, match="drawn on the CPU"):
        sample(generator=torch.Generator("cuda").manual_seed(0))
    with pytest.raises(ValueError, match="does not divide into patches"):
        sample(latent_shape=(3, 7, 8))
    with pytest.raises(ValueError, match="latent_shape asks for"):
        sample(latents=torch.zeros(1, CHANNELS, 3, 8, 4))
    assert sample(timesteps=(1000,)).shape == (1, CHANNELS, *LATENT)


# ---------------------------------------------------------------------------------------------
# Where the weights are, and what is refused


@pytest.mark.cuda
def test_weights_are_placed_by_how_they_are_used(diffusers, disk_tmp):
    import torch

    tiny_wan_transformer(disk_tmp)
    with load_wan_transformer(disk_tmp, slots=2) as ours:
        model = ours.model
        resident = 0
        for name, param in model.named_parameters():
            kept = bool(KEPT_IN_FP32.intersection(name.split(".")))
            assert param.dtype == (torch.float32 if kept else torch.bfloat16), name
            assert not param.requires_grad
            streamed = name.startswith("blocks.")
            assert param.device.type == ("meta" if streamed else "cuda"), name
            resident += 0 if streamed else param.numel() * param.element_size()
        resident += sum(b.numel() * b.element_size() for b in model.buffers())
        assert all(buffer.device.type == "cuda" for buffer in model.buffers())
        assert any(p.dtype == torch.float32 for p in model.blocks[0].parameters())
        assert ours.resident_bytes == resident
        assert ours.vram_bytes == 2 * ours.streamer.slot_bytes + resident
        assert ours.ram_bytes == ours.streamer.ram_bytes > 0
        assert model.device.type == "cuda" and model.dtype == torch.bfloat16
        assert all(runs_one_sample_at_a_time(block) for block in model.blocks)


@pytest.mark.cuda
def test_close_gives_the_skeleton_back(diffusers, disk_tmp):
    import torch

    tiny_wan_transformer(disk_tmp)
    latent, text = inputs("bfloat16")
    gc.collect()  # pipelines of earlier tests are freed by the collector, whenever it runs
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    ours = load_wan_transformer(disk_tmp, ffn_chunk=16)
    model = ours.model
    with torch.no_grad():
        run(model, latent, per_token(757.0), text)
    assert torch.cuda.memory_allocated() > before
    ours.close()
    ours.close()
    assert torch.cuda.memory_allocated() == before
    assert all(param.device.type == "meta" for param in model.parameters())
    wrapped = [module for module in model.modules() if "forward" in vars(module)]
    hooked = [m for m in model.modules() if m._forward_hooks or m._forward_pre_hooks]
    assert not wrapped and not hooked


@pytest.mark.cuda
def test_what_cannot_be_loaded_is_refused(diffusers, disk_tmp):
    import torch

    tiny_wan_transformer(disk_tmp)
    gc.collect()
    before = torch.cuda.memory_allocated()
    with pytest.raises(ValueError, match="tokens at a time"):
        load_wan_transformer(disk_tmp, ffn_chunk=0)
    with pytest.raises(ValueError, match="cannot run in torch.int32"):
        load_wan_transformer(disk_tmp, dtype="int32")
    with load_wan_transformer(disk_tmp) as whole:
        in_ram = whole.ram_bytes
    with pytest.raises(ValueError, match="converted block"):  # they cannot be left on disk
        load_wan_transformer(disk_tmp, ram_budget=in_ram - 1)
    config = json.loads((disk_tmp / "config.json").read_text())
    (disk_tmp / "config.json").write_text(json.dumps({**config, "_class_name": "Other"}))
    with pytest.raises(ValueError, match="holds a Other"):
        load_wan_transformer(disk_tmp)
    assert torch.cuda.memory_allocated() == before, "a failed load leaves nothing behind"


@pytest.mark.cuda
def test_denoise_refuses_what_it_would_get_wrong(diffusers, disk_tmp):
    import torch

    tiny_wan_transformer(disk_tmp)
    plain = reference(disk_tmp, "bfloat16")
    text = inputs("bfloat16", batch=2)[1]
    common = {"latent_shape": LATENT, "expand_timesteps": True, "num_inference_steps": 2}

    def sample(model, *embeds, **changes):
        return denoise(model, make_scheduler(diffusers, "unipc"), *embeds, **{**common, **changes})

    with pytest.raises(ValueError, match="needs negative_prompt_embeds"):
        sample(plain, text[:1])
    with pytest.raises(ValueError, match="samples one video"):
        sample(plain, text, text)
    with pytest.raises(ValueError, match="pad both to the same length"):
        sample(plain, text[:1], text[1:, :5])
    with pytest.raises(ValueError, match="does not divide into patches"):
        sample(plain, text[:1], text[1:], latent_shape=(3, 7, 8))
    with pytest.raises(ValueError, match="latent_shape asks for"):
        sample(plain, text[:1], text[1:], latents=torch.zeros(1, CHANNELS, 3, 8, 4))
    with pytest.raises(ValueError, match="one_pass needs a transformer from load_wan_transformer"):
        sample(plain, text[:1], text[1:], one_pass=True)
    assert sample(plain, text[:1], guidance_scale=1.0).shape == (1, CHANNELS, *LATENT)


# ---------------------------------------------------------------------------------------------
# The real model: Wan 2.2 TI2V-5B (needs HYPERTIER_MODELS)

REAL = "Wan2.2-TI2V-5B-Diffusers"
SMALL = (5, 30, 52)  # 480p x 17 frames, 1,950 tokens: the per-token path of diffusers fits
FULL = {"720p x 49 frames": (13, 44, 80), "480p x 121 frames": (31, 30, 52)}  # ~12k tokens
SMALL_CASES = [(0, 27, 5.0), (1, 300, 5.0), (2, 60, 1.0)]  # seed, prompt tokens, guidance
SMALL_STEPS = 3


def real_embeds(seed, prompt_tokens):
    """Conditioning shaped like the text encoder's: the tokens of the prompt, then zeros.

    Random, with the spread of real umT5 outputs. Equality with ``diffusers`` on real prompts
    is the business of the whole pipeline; a pass is the same function of any input.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    embeds = torch.zeros(2, 512, 4096)
    embeds[0, :prompt_tokens] = 0.08 * torch.randn(prompt_tokens, 4096, generator=gen)
    embeds[1, :126] = 0.08 * torch.randn(126, 4096, generator=gen)  # the negative prompt
    return embeds.to("cuda", torch.bfloat16)


def strict_baseline(directory):
    """The transformer as ``diffusers`` loads it, each block copied to the GPU before it runs.

    Blocking copies of pageable memory and nothing else: no slots, no second stream, no
    conversion of ours.
    """
    import torch
    from diffusers import WanTransformer3DModel

    model = WanTransformer3DModel.from_pretrained(directory, torch_dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    for name, child in model.named_children():
        if name != "blocks":
            child.to("cuda")
    model.scale_shift_table.data = model.scale_shift_table.data.to("cuda")
    for block in model.blocks:
        home = [(param, param.data) for param in block.parameters()]

        def load(module, args, home=home):
            for param, data in home:
                param.data = data.to("cuda")

        def unload(module, args, output, home=home):
            for param, data in home:
                param.data = data

        block.register_forward_pre_hook(load)
        block.register_forward_hook(unload, always_call=True)
    return model


def peak(torch):
    return f"{torch.cuda.max_memory_allocated() / MiB:.0f} MiB allocated"


@pytest.fixture(scope="module")
def real_dir():
    root = os.environ.get("HYPERTIER_MODELS")
    if not root or not (Path(root) / REAL / "transformer").is_dir():
        pytest.skip(f"set HYPERTIER_MODELS to a directory that holds {REAL}")
    pytest.importorskip("diffusers")
    return Path(root) / REAL


@pytest.fixture(scope="module")
def real_expected(real_dir):
    """What ``diffusers`` gives for every case below, computed before our model is loaded.

    Two copies of the transformer in RAM are 18.5 GiB: one after the other instead.
    """
    import diffusers
    import torch

    model = strict_baseline(real_dir / "transformer")
    expected = {}
    pipe = diffusers.WanPipeline(
        tokenizer=None, text_encoder=None, vae=None, transformer=model, expand_timesteps=True,
        scheduler=diffusers.UniPCMultistepScheduler.from_pretrained(real_dir / "scheduler"),
    )  # fmt: skip
    pipe.set_progress_bar_config(disable=True)
    frames, height, width = SMALL
    for seed, prompt_tokens, guidance in SMALL_CASES:
        embeds = real_embeds(seed, prompt_tokens)
        expected[seed] = pipe(
            prompt_embeds=embeds[:1], negative_prompt_embeds=embeds[1:], height=8 * height,
            width=8 * width, num_frames=4 * (frames - 1) + 1, guidance_scale=guidance,
            num_inference_steps=SMALL_STEPS, output_type="latent",
            generator=torch.Generator("cpu").manual_seed(seed),
        ).frames.cpu()  # fmt: skip
    # At full size the per-token path of diffusers needs 1.3 GiB more than there is. The
    # strict baseline does what we do, one row of the time embedding, in the plainest way.
    one_row = hf_diffusers._OneRow()
    model.condition_embedder.register_forward_pre_hook(one_row.before, with_kwargs=True)
    model.condition_embedder.register_forward_hook(one_row.after, with_kwargs=True)
    embeds = real_embeds(0, 27)
    for name, shape in FULL.items():
        scheduler = diffusers.UniPCMultistepScheduler.from_pretrained(real_dir / "scheduler")
        expected[name] = denoise(
            model, scheduler, embeds[:1], embeds[1:], latent_shape=shape, expand_timesteps=True,
            num_inference_steps=1, generator=torch.Generator("cpu").manual_seed(0),
        ).cpu()  # fmt: skip
    del pipe, model
    gc.collect()
    torch.cuda.empty_cache()
    return expected


def real_pass():
    """The inputs of one pass at 1,950 tokens, already rounded to bfloat16."""
    import torch

    latent = torch.randn(1, 48, *SMALL, generator=torch.Generator().manual_seed(1))
    timestep = per_token(757.0)[:, :1].expand(1, 1950)
    return latent.to("cuda", torch.bfloat16), timestep, real_embeds(3, 60)[:1]


@pytest.fixture(scope="module")
def real_pass_in_float32(real_dir, real_expected):
    """That pass with the model run in float32, which fits at this size: what bfloat16 rounds.

    18.3 GiB of page-locked RAM, so it is released before the bfloat16 model is loaded.
    """
    import torch

    latent, timestep, text = real_pass()
    with load_wan_transformer(real_dir / "transformer", dtype="float32") as wide, torch.no_grad():
        exact = run(wide.model, latent.float(), timestep, text.float()).cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return exact


@pytest.fixture(scope="module")
def real_wan(real_dir, real_expected, real_pass_in_float32):
    with load_wan_transformer(real_dir / "transformer") as ours:
        yield ours


@pytest.mark.slow
@pytest.mark.cuda
def test_the_real_model_samples_the_latents_of_wanpipeline(real_dir, real_expected, real_wan):
    import diffusers
    import torch

    for seed, prompt_tokens, guidance in SMALL_CASES:
        embeds = real_embeds(seed, prompt_tokens)
        expected = real_expected[seed]
        assert bool(torch.isfinite(expected).all()) and float(expected.std()) > 0.1
        for one_pass in (False, True) if guidance > 1 else (False,):
            got = denoise(
                real_wan.model,
                diffusers.UniPCMultistepScheduler.from_pretrained(real_dir / "scheduler"),
                embeds[:1], embeds[1:] if guidance > 1 else None, latent_shape=SMALL,
                expand_timesteps=True, num_inference_steps=SMALL_STEPS, guidance_scale=guidance,
                one_pass=one_pass, generator=torch.Generator("cpu").manual_seed(seed),
            )  # fmt: skip
            assert torch.equal(got.cpu(), expected), (seed, prompt_tokens, guidance, one_pass)


@pytest.mark.slow
@pytest.mark.cuda
def test_the_real_model_at_full_size_gives_the_latents_of_the_strict_baseline(
    real_dir, real_expected, real_wan
):
    import diffusers
    import torch

    embeds = real_embeds(0, 27)
    for name, shape in FULL.items():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        got = denoise(
            real_wan.model,
            diffusers.UniPCMultistepScheduler.from_pretrained(real_dir / "scheduler"),
            embeds[:1], embeds[1:], latent_shape=shape, expand_timesteps=True,
            num_inference_steps=1, generator=torch.Generator("cpu").manual_seed(0),
        )  # fmt: skip
        tokens = shape[0] * shape[1] * shape[2] // 4
        print(f"\n[wan] {name}, {tokens} tokens, 2 slots: peak {peak(torch)}")
        assert bool(torch.isfinite(got).all())
        assert torch.equal(got.cpu(), real_expected[name]), name


@pytest.mark.slow
@pytest.mark.cuda
def test_the_real_chunked_ffn_stays_within_the_rounding_of_one_pass(
    real_wan, real_pass_in_float32
):
    import torch

    model = real_wan.model
    arguments = real_pass()
    with torch.no_grad():
        whole = run(model, *arguments).cpu()
        undo = [chunk_tokens(block.ffn, 512) for block in model.blocks]
        try:
            chunked = run(model, *arguments).cpu()
        finally:
            for restore in undo:
                restore()
        assert torch.equal(run(model, *arguments).cpu(), whole), "undone: exact again"
    noise = relative_error(whole, real_pass_in_float32)
    error = relative_error(chunked, whole)
    print(
        f"\n[wan] 1950 tokens: one bfloat16 pass is {noise:.2e} from float32; the ffn in "
        f"chunks of 512 moves it {error:.2e}, {error / noise:.2f} of that"
    )
    assert error <= CHUNKED_OVER_NOISE * noise



# ---------------------------------------------------------------------------------------------
# A real distilled model: FastWan 2.2 TI2V-5B (needs HYPERTIER_MODELS)

FAST = "FastWan2.2-TI2V-5B-FullAttn-Diffusers"


@pytest.fixture(scope="module")
def fast_dir():
    root = os.environ.get("HYPERTIER_MODELS")
    if not root or not (Path(root) / FAST / "transformer").is_dir():
        pytest.skip(f"set HYPERTIER_MODELS to a directory that holds {FAST}")
    pytest.importorskip("diffusers")
    return Path(root) / FAST / "transformer"


@pytest.mark.slow
@pytest.mark.cuda
def test_the_real_distilled_model_gives_the_latents_of_the_strict_baseline(fast_dir):
    """Three steps at 480p x 17 frames, streamed, against a blocking copy per block."""
    import torch

    embeds = real_embeds(0, 27)[:1]

    def sample(model):
        return denoise_dmd(
            model, embeds, latent_shape=SMALL, timesteps=FEW_STEPS,
            generator=torch.Generator("cpu").manual_seed(1024),
        ).cpu()  # fmt: skip

    expected = sample(strict_baseline(fast_dir))
    gc.collect()
    torch.cuda.empty_cache()
    assert expected.shape == (1, 48, *SMALL) and expected.dtype == torch.bfloat16
    assert bool(torch.isfinite(expected).all()) and 0.5 < float(expected.float().std()) < 2
    torch.cuda.reset_peak_memory_stats()
    with load_wan_transformer(fast_dir) as ours:
        got = sample(ours.model)
        stats = ours.streamer.pass_stats()
    print(f"\n[fastwan] 1950 tokens, 3 steps: peak {peak(torch)}, {len(stats)} passes")
    assert len(stats) == 3 and torch.equal(got, expected)
