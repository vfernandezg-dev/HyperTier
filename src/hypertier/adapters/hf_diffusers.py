# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the transformer of a Wan video model of ``diffusers`` that does not fit in VRAM.

:func:`load_wan_transformer` returns the ordinary ``WanTransformer3DModel``, with its weights
placed by how they are used::

    transformer blocks  streamed through the VRAM slots (``LayerStreamer``), one block each,
                        converted to the dtype they run in while they are loaded
    everything else     (patch and condition embedders, output head, tables) resident on
                        the GPU: about 170 MiB for the 5B model

:func:`denoise` is the sampling loop of ``WanPipeline`` for text-to-video, written here so
that it decides how the two branches of classifier-free guidance share a pass over the
weights. On the same inputs it returns the latents of ``WanPipeline`` bit for bit, and that is
how it is tested: in bfloat16 a difference of one unit in the last place in one pass becomes
another video after a few steps, so nothing short of equality says the loop is right.

Three things make that equality hold, and each is what ``diffusers`` does:

* Parameters under the modules the class lists in ``_keep_in_fp32_modules`` (norms,
  modulation tables, the time embedder) stay in float32; the rest run in ``dtype``.
* Models that take one timestep per token (Wan 2.2 TI2V) get one per token: the condition
  embedder computes a row per token. When every token has the same timestep the rows are
  equal, so one is kept after the embedder and the blocks broadcast it, instead of
  materialising them all in float32 (+1.3 GiB at 12,090 tokens). Computing a single row gives
  other bits (0.6-1.2 % per pass, measured), because another shape picks another kernel.
* A batch is run one sample at a time (:func:`~hypertier.diffusion.one_sample_at_a_time`).

The model classes are not rewritten. Needs the optional extra: ``pip install
hypertier[diffusers]``. Importing this module imports neither PyTorch nor ``diffusers``.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hypertier.adapters import _skeleton as skeleton
from hypertier.adapters.hf_transformers import StreamedModel
from hypertier.diffusion.chunking import (
    chunk_tokens,
    one_sample_at_a_time,
    runs_one_sample_at_a_time,
)
from hypertier.io.safetensors_index import SafetensorsIndex
from hypertier.stream.pingpong import LayerStreamer
from hypertier.stream.source import FileWeights
from hypertier.telemetry._common import require_cuda, resolve_device

if TYPE_CHECKING:
    import torch

MIN_DIFFUSERS = (0, 43)
"""Oldest ``diffusers`` this adapter is tested with; the ``[diffusers]`` extra pins the same."""

_MODEL_CLASS = "WanTransformer3DModel"


def load_wan_transformer(
    model_dir: str | PathLike[str],
    *,
    dtype: torch.dtype | str = "bfloat16",
    slots: int = 2,
    ram_budget: int | None = None,
    num_stage: int = 2,
    ffn_chunk: int | None = None,
    device: torch.device | str = "cuda",
) -> StreamedModel:
    """Load a Wan transformer from a local directory, streamed through ``slots`` slots.

    Args:
        model_dir: the transformer directory of a ``diffusers`` checkpoint (``config.json``
            and ``*.safetensors``). Nothing is downloaded.
        dtype: the dtype the model runs in, as ``torch_dtype`` in ``diffusers``. Tensors
            stored in another one are converted once, on the CPU, while they are loaded.
        slots: VRAM buffers, each as large as the largest block (see
            :class:`~hypertier.stream.LayerStreamer`).
        ram_budget: page-locked RAM the blocks may take. None keeps them all in RAM. Blocks
            that are converted have to fit: only a checkpoint stored as it runs can leave
            blocks on disk (see :class:`~hypertier.stream.FileWeights`).
        num_stage: staging buffers for the blocks read from disk.
        ffn_chunk: run the feed-forward network of every block over this many tokens at a
            time, for token counts whose activations do not fit otherwise. **Not exact**: the
            output differs by rounding (see :func:`~hypertier.diffusion.chunk_tokens`).
        device: the CUDA device that computes.

    Returns the same holder as the language-model adapter: ``model`` is the
    ``WanTransformer3DModel``, to be called as usual or driven by :func:`denoise`.

    Bytes moved per forward of the model: every block from RAM to VRAM once, in the dtype it
    runs in (9.15 GiB for the 5B model in bfloat16). Neither the number of tokens nor the
    samples in the batch change it.

    Limits:

    * Building the model sends every ``Parameter`` registered in the process to ``meta``
      while it lasts: do not build other modules in another thread meanwhile.
    * The limits of :class:`~hypertier.stream.LayerStreamer` apply: inference only, one
      thread, no ``torch.compile``, no ``model.to(...)``.
    """
    torch = require_cuda()
    diffusers = _import_diffusers()
    dev = resolve_device(device)
    root = Path(model_dir)
    index = SafetensorsIndex(root)
    cls = getattr(diffusers, _MODEL_CLASS)
    config = cls.load_config(root)
    if config.get("_class_name") != _MODEL_CLASS:
        raise ValueError(
            f"{root} holds a {config.get('_class_name')}; this adapter loads {_MODEL_CLASS}"
        )
    run_in = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    if not run_in.is_floating_point:
        raise ValueError(f"the model cannot run in {run_in}")
    if ffn_chunk is not None and ffn_chunk < 1:
        raise ValueError(f"cannot run {ffn_chunk} tokens at a time")

    with skeleton.parameters_on_meta(torch):
        model = cls.from_config(config)
    model.eval()
    _set_dtypes(torch, model, run_in, cls._keep_in_fp32_modules or ())

    blocks = model.blocks
    embedder = model.condition_embedder

    undo: list[Any] = []
    streamer = None
    try:
        skip = {id(module) for module in blocks.modules()}
        resident_bytes = skeleton.make_resident(torch, model, index, dev, skip, undo, cast=True)
        weights = FileWeights(
            index,
            [f"blocks.{i}." for i in range(len(blocks))],
            ram_budget=ram_budget,
            num_stage=num_stage,
            cast=True,
        )
        streamer = LayerStreamer(list(blocks), slots=slots, device=dev, weights=weights)

        one_row = _OneRow()
        before = embedder.register_forward_pre_hook(one_row.before, with_kwargs=True)
        after = embedder.register_forward_hook(one_row.after, with_kwargs=True)
        undo += [before.remove, after.remove]

        # Everything that computes on activations, so that a batch gives the bits of its
        # samples run alone. What is left in the model's forward is elementwise or a reshape.
        per_sample = [
            model.patch_embedding,
            embedder.time_embedder,
            embedder.time_proj,
            embedder.text_embedder,
            getattr(embedder, "image_embedder", None),
            *blocks,
            model.norm_out,
            model.proj_out,
        ]
        undo += [one_sample_at_a_time(module) for module in per_sample if module is not None]
        if ffn_chunk is not None:
            undo += [chunk_tokens(block.ffn, ffn_chunk) for block in blocks]
    except BaseException:
        if streamer is not None:
            streamer.close()
        while undo:
            undo.pop()()
        raise
    return StreamedModel(model, streamer, dev, [], resident_bytes, 0, undo)


def denoise(
    transformer: Any,
    scheduler: Any,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None = None,
    *,
    latent_shape: tuple[int, int, int],
    expand_timesteps: bool,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    generator: torch.Generator | None = None,
    latents: torch.Tensor | None = None,
    one_pass: bool = False,
) -> torch.Tensor:
    """Sample the latents of one video: the text-to-video loop of ``WanPipeline``.

    Args:
        transformer: a ``WanTransformer3DModel``; the ``model`` of
            :func:`load_wan_transformer`, or any other (a baseline to compare with).
        scheduler: a ``diffusers`` scheduler, as the pipeline would hold it. It is driven
            through ``set_timesteps`` and ``step``, and keeps the state of this run.
        prompt_embeds: ``[1, tokens, dim]``, the output of the text encoder for the prompt.
        negative_prompt_embeds: the same for the negative prompt. Needed when
            ``guidance_scale`` is above 1, which is when guidance is applied.
        latent_shape: ``(frames, height, width)`` of the latent video, in latent cells.
        expand_timesteps: whether the model takes one timestep per token: the field of that
            name in the ``model_index.json`` of the pipeline (true for Wan 2.2 TI2V).
        num_inference_steps: steps of the scheduler.
        guidance_scale: scale of classifier-free guidance.
        generator: source of the initial noise. On the CPU, as ``diffusers`` recommends, the
            noise does not depend on the GPU.
        latents: initial noise ``[1, channels, frames, height, width]``, instead of drawing it.
        one_pass: run both branches of the guidance in one pass over the weights, a sample at
            a time inside each block. Same bits as two passes; needs a transformer from
            :func:`load_wan_transformer`.

    Returns the latents on the device of the transformer, as ``WanPipeline`` does with
    ``output_type="latent"``: in the dtype the scheduler leaves them (float32 with UniPC).
    Denormalising and decoding them is the VAE's stage.

    Bytes moved per step: with guidance, every block twice (once per branch); with
    ``one_pass`` or without guidance, once. One pass costs the activations of one sample plus
    the hidden states of the other between blocks, and wins time only while the copy of a
    block takes longer than its compute, i.e. with few tokens.
    """
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    device, dtype = transformer.device, transformer.dtype
    guided = guidance_scale > 1.0
    if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
        raise ValueError(
            f"prompt_embeds is {list(prompt_embeds.shape)}: this loop samples one video, so it "
            "takes [1, tokens, dim]"
        )
    if guided and negative_prompt_embeds is None:
        raise ValueError(
            f"guidance_scale {guidance_scale} applies guidance, which needs "
            "negative_prompt_embeds; pass them, or a scale of 1 or less to sample without it"
        )
    if guided and negative_prompt_embeds.shape != prompt_embeds.shape:
        raise ValueError(
            f"negative_prompt_embeds is {list(negative_prompt_embeds.shape)} and prompt_embeds "
            f"{list(prompt_embeds.shape)}: pad both to the same length"
        )
    if one_pass and not all(runs_one_sample_at_a_time(block) for block in transformer.blocks):
        raise ValueError(
            "one_pass needs a transformer from load_wan_transformer, whose blocks run a batch "
            "one sample at a time: on any other it would be a real batch, with other bits"
        )
    frames, height, width = latent_shape
    p_t, p_h, p_w = transformer.config.patch_size
    if min(frames, height, width) < 1 or frames % p_t or height % p_h or width % p_w:
        raise ValueError(
            f"a latent of {frames} x {height} x {width} cells does not divide into patches of "
            f"{p_t} x {p_h} x {p_w}"
        )
    shape = (1, transformer.config.in_channels, frames, height, width)
    if latents is not None and tuple(latents.shape) != shape:
        raise ValueError(f"latents is {list(latents.shape)}, but latent_shape asks for {list(shape)}")

    with torch.no_grad():
        cond = prompt_embeds.to(device=device, dtype=dtype)
        uncond = negative_prompt_embeds.to(device=device, dtype=dtype) if guided else None
        scheduler.set_timesteps(num_inference_steps, device=device)
        scheduler.set_begin_index(0)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        else:
            latents = latents.to(device=device, dtype=torch.float32)
        tokens = (frames // p_t) * (height // p_h) * (width // p_w)
        ones = torch.ones(tokens, dtype=torch.float32, device=device)
        both = torch.cat([cond, uncond]) if guided and one_pass else None

        for t in scheduler.timesteps:
            x = latents.to(dtype)
            timestep = (ones * t).unsqueeze(0) if expand_timesteps else t.expand(1)
            if both is not None:
                # Views: the latent and the timestep are the same for both branches.
                pair = transformer(
                    hidden_states=x.expand(2, *x.shape[1:]),
                    timestep=timestep.expand(2, *timestep.shape[1:]),
                    encoder_hidden_states=both,
                    return_dict=False,
                )[0]
                noise, noise_uncond = pair[0:1], pair[1:2]
            else:
                noise = transformer(
                    hidden_states=x,
                    timestep=timestep,
                    encoder_hidden_states=cond,
                    return_dict=False,
                )[0]
                if guided:
                    noise_uncond = transformer(
                        hidden_states=x,
                        timestep=timestep,
                        encoder_hidden_states=uncond,
                        return_dict=False,
                    )[0]
            if guided:
                noise = noise_uncond + guidance_scale * (noise - noise_uncond)
            latents = scheduler.step(noise, t, latents, return_dict=False)[0]
    return latents


# ---------------------------------------------------------------------------------------------


def _import_diffusers() -> Any:
    try:
        import diffusers
    except ImportError as exc:
        raise RuntimeError(
            "this adapter needs Hugging Face diffusers: pip install hypertier[diffusers]"
        ) from exc
    found = skeleton.version_tuple(diffusers.__version__)
    if found < MIN_DIFFUSERS:
        wanted = ".".join(map(str, MIN_DIFFUSERS))
        raise RuntimeError(
            f"this adapter needs diffusers >= {wanted}, found {diffusers.__version__}: "
            "pip install -U hypertier[diffusers]"
        )
    return diffusers


def _set_dtypes(torch: Any, model: Any, run_in: Any, keep_in_fp32: Any) -> None:
    """Give every floating-point parameter of the skeleton the dtype ``diffusers`` loads it in.

    ``from_pretrained(torch_dtype=...)`` converts a parameter to float32 when a component of
    its dotted name is in ``_keep_in_fp32_modules``, and to ``torch_dtype`` otherwise. The
    parameters are placeholders on ``meta`` here: only their dtype changes.
    """
    keep = set(keep_in_fp32)
    for prefix, module in model.named_modules():
        for key, param in list(module._parameters.items()):
            if param is None or not param.dtype.is_floating_point:
                continue
            name = f"{prefix}.{key}" if prefix else key
            wanted = torch.float32 if keep.intersection(name.split(".")) else run_in
            if param.dtype != wanted:
                empty = torch.empty(param.shape, dtype=wanted, device="meta")
                module._parameters[key] = skeleton.frozen(torch, empty)


class _OneRow:
    """Hooks of the condition embedder: one row of the time embedding when all are equal.

    The embedder has computed a row per token, as ``diffusers`` does. When every token of a
    sample has the same timestep those rows are equal, and the blocks give the same bits
    whether they read N equal rows or broadcast one. When the timesteps differ between tokens
    (image-to-video fixes the frames it is given) every row is kept: the whole per-token path.
    """

    def __init__(self) -> None:
        self._uniform = False

    def before(self, module: Any, args: Any, kwargs: Any) -> None:
        timestep = args[0] if args else kwargs.get("timestep")
        tokens = kwargs.get("timestep_seq_len", args[3] if len(args) > 3 else None)
        self._uniform = False
        if timestep is not None and tokens is not None and tokens > 1:
            rows = timestep.reshape(-1, tokens)
            # One small blocking read per pass, before any block runs.
            self._uniform = bool((rows == rows[:, :1]).all())

    def after(self, module: Any, args: Any, kwargs: Any, output: Any) -> Any:
        if not self._uniform:
            return None
        temb, timestep_proj, *rest = output
        # Copies, so that the N rows are freed instead of living on behind a view.
        return (temb[:, :1].clone(), timestep_proj[:, :1].clone(), *rest)


__all__ = ["MIN_DIFFUSERS", "denoise", "load_wan_transformer"]
