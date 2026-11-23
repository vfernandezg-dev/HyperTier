# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the stages of a Wan video model of ``diffusers`` that do not fit in VRAM.

The transformer is streamed and gives the latents of ``WanPipeline`` bit for bit; the VAE
decodes them in tiles (:func:`load_wan_vae`), which is approximate and judged by PSNR.

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

import warnings
from dataclasses import dataclass
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
from hypertier.diffusion.tiling import Canvas, TilePlan, plan_tiles
from hypertier.io.safetensors_index import SafetensorsIndex
from hypertier.stream.pingpong import LayerStreamer
from hypertier.stream.source import FileWeights
from hypertier.telemetry._common import MiB, format_bytes, require_cuda, resolve_device

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self

MIN_DIFFUSERS = (0, 43)
"""Oldest ``diffusers`` this adapter is tested with; the ``[diffusers]`` extra pins the same."""

_MODEL_CLASS = "WanTransformer3DModel"
_VAE_CLASS = "AutoencoderKLWan"


@dataclass(frozen=True)
class VaeConfig:
    """One way to decode with the VAE on a small GPU, and what it was measured to cost.

    Attributes:
        name: how :func:`load_wan_vae` and the logs call it.
        dtype: the dtype the decoder runs in.
        streamed: whether the decoder's weights go through VRAM slots, which leaves room for
            larger tiles, or are resident.
        max_tile_cells: the largest area of a tile, in latent cells.
        peak_bytes: VRAM the process holds at the peak of a decode, as the driver counts it
            (the CUDA context included).
        min_psnr_db: the PSNR against the untiled float32 decode that a decode has to reach:
            the lowest measured, less 1 dB.
    """

    name: str
    dtype: str
    streamed: bool
    max_tile_cells: int
    peak_bytes: int
    min_psnr_db: float


VAE_CONFIGS: tuple[VaeConfig, ...] = (
    VaeConfig("bf16-streamed", "bfloat16", True, 600, 3085 * MiB, 41.4),
    VaeConfig("bf16-resident", "bfloat16", False, 340, 2815 * MiB, 38.4),
    VaeConfig("fp32-streamed", "float32", True, 256, 2445 * MiB, 37.2),
)
"""The configurations of the Wan 2.2 VAE (16 px per latent cell), best first.

Measured on the reference laptop at the two sizes of the F3 criterion, each run in a process
of its own with the GPU warmed up (63 to 77 °C). Time and peak are of one whole video; PSNR is
against the untiled float32 decode, over the first 9 frames of five videos per size::

                   720p x 49 frames              480p x 121 frames
                   tiles      time    peak       tiles      time    peak       PSNR, dB
    bf16-streamed   8 (24x23) 106 s   2871 MiB    3 (30x20) 106 s   3085 MiB   42.4 - 54.4
    bf16-resident  16 (14x23) 128 s   2763 MiB    6 (17x20) 124 s   2815 MiB   39.4 - 53.6
    fp32-streamed  24 (14x17) 293 s   2445 MiB   10 (17x14) 297 s   2445 MiB   38.2 - 55.2

The PSNR follows the picture far more than the configuration: 15 dB between a blurred
background and a street of neon signs. Then the tile: with the same tiles float32 is 0.3 to
0.8 dB above bfloat16, while tiles of 340 cells instead of 256 are 1.2 to 2.7 dB above. That
is why float32 comes last: its weights leave room for small tiles only, and every block is
copied to VRAM for every latent frame of every tile (104 s of its 293 s are copies).
"""

VAE_MARGIN = 300 * MiB
"""VRAM left free above the measured peak when a configuration is chosen: the peaks are
measurements on one machine, and the desktop may take some at any time."""


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
        # The arguments that carry the batch are named: the rotary table of a block is shared.
        first = (0,)
        in_block = ("hidden_states", "encoder_hidden_states", "temb")
        per_sample = [
            (model.patch_embedding, first),
            (embedder.time_embedder, first),
            (embedder.time_proj, first),
            (embedder.text_embedder, first),
            (getattr(embedder, "image_embedder", None), first),
            *[(block, in_block) for block in blocks],
            (model.norm_out, first),
            (model.proj_out, first),
        ]
        undo += [
            one_sample_at_a_time(module, batched)
            for module, batched in per_sample
            if module is not None
        ]
        if ffn_chunk is not None:
            undo += [chunk_tokens(block.ffn, ffn_chunk) for block in blocks]
    except BaseException:
        if streamer is not None:
            streamer.close()
        while undo:
            undo.pop()()
        raise
    return StreamedModel(model, streamer, dev, [], resident_bytes, 0, undo)


def choose_vae_config(
    vram_budget: int,
    *,
    margin: int = VAE_MARGIN,
    configs: tuple[VaeConfig, ...] = VAE_CONFIGS,
) -> VaeConfig:
    """The best configuration whose measured peak leaves ``margin`` of ``vram_budget``.

    Args:
        vram_budget: VRAM the process may hold, in bytes, as the driver counts it.
        margin: bytes to leave free above the measured peak.
        configs: what to choose from, best first.
    """
    for config in configs:
        if config.peak_bytes + margin <= vram_budget:
            return config
    least = min(configs, key=lambda config: config.peak_bytes)
    raise ValueError(
        f"no VAE configuration fits {format_bytes(vram_budget)} of VRAM with "
        f"{format_bytes(margin)} to spare: the smallest, {least.name}, peaks at "
        f"{format_bytes(least.peak_bytes)}. Free VRAM, or pass a smaller margin"
    )


class WanVae:
    """The decoder of a Wan VAE on the GPU, and how it decodes.

    Attributes:
        vae: the ``AutoencoderKLWan``. Only its decoder has weights: the encoder is an empty
            skeleton, since text-to-video never encodes.
        config: the :class:`VaeConfig` it was loaded with.
        streamer: the :class:`~hypertier.stream.LayerStreamer` behind the decoder's blocks,
            or None when the weights are resident.
        device: the GPU that computes.
        resident_bytes: weights kept on the GPU outside the slots.
    """

    def __init__(
        self,
        vae: Any,
        config: VaeConfig,
        streamer: LayerStreamer | None,
        device: Any,
        resident_bytes: int,
        undo: list[Any],
    ) -> None:
        self.vae = vae
        self.config = config
        self.streamer = streamer
        self.device = device
        self.resident_bytes = resident_bytes
        self._undo = undo
        self._closed = False

    @property
    def vram_bytes(self) -> int:
        """VRAM held by weights. The activations of a tile are extra, and far larger."""
        slots = 0 if self.streamer is None else self.streamer.vram_bytes
        return slots + self.resident_bytes

    @property
    def ram_bytes(self) -> int:
        """Page-locked RAM held by weights: none when they are resident."""
        return 0 if self.streamer is None else self.streamer.ram_bytes

    def plan(self, latent_height: int, latent_width: int, *, min_overlap: int = 4) -> TilePlan:
        """How a latent frame of this size is cut into tiles under this configuration."""
        return plan_tiles(
            latent_height, latent_width, self.config.max_tile_cells, min_overlap=min_overlap
        )

    def decode(self, latents: torch.Tensor, *, min_overlap: int = 4, margin: int = 1) -> torch.Tensor:
        """Decode the latents of one video into frames.

        Args:
            latents: ``[1, channels, frames, height, width]`` as the sampling loop returns
                them, normalised. Any device, any floating dtype.
            min_overlap: latent cells shared by neighbouring tiles, at least.
            margin: latent cells at the interior edges of a tile left out of the blend.

        Returns ``uint8`` frames ``[frames, height, width, 3]`` in host RAM, rounded as the
        video processor of ``diffusers`` rounds them.

        Every tile goes through ``vae.decode``, the whole length of the video at once, and is
        blended on the CPU: VRAM holds the activations of one tile. Not exact: see
        :mod:`hypertier.diffusion.tiling`.

        Bytes moved: none with resident weights. With streamed weights every block of the
        decoder goes to VRAM once per latent frame of every tile, because the decoder runs
        frame by frame.
        """
        import torch

        vae = self.vae
        if latents.ndim != 5 or latents.shape[0] != 1 or latents.shape[1] != vae.config.z_dim:
            raise ValueError(
                f"latents is {list(latents.shape)}: this decodes one video, "
                f"[1, {vae.config.z_dim}, frames, height, width]"
            )
        dtype = getattr(torch, self.config.dtype)
        scale = vae.spatial_compression_ratio
        height, width = latents.shape[-2:]
        plan = self.plan(height, width, min_overlap=min_overlap)
        canvas = Canvas(torch, plan, (height, width), scale=scale, margin=margin)
        # cudnn.benchmark is 8-15 % slower here, measured, and its choice of algorithm makes
        # two runs differ: off while this decodes, whatever the process has set.
        benchmark, torch.backends.cudnn.benchmark = torch.backends.cudnn.benchmark, False
        try:
            with torch.inference_mode():
                # The arithmetic of WanPipeline, in float32 whatever the decoder runs in.
                shape = (1, vae.config.z_dim, 1, 1, 1)
                mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(shape)
                std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(shape)
                z = latents.to(device="cpu", dtype=torch.float32) / (1.0 / std) + mean
                z = z.to(device=self.device, dtype=dtype)
                rows, columns = plan.tile
                for i, top in enumerate(plan.rows):
                    for j, left in enumerate(plan.columns):
                        tile = z[:, :, :, top : top + rows, left : left + columns]
                        pixels = vae.decode(tile, return_dict=False)[0]
                        canvas.add(i, j, pixels[0])
                        del pixels
                video = canvas.result()  # [3, frames, height, width] in [-1, 1]
                video = video.div_(2).add_(0.5).clamp_(0, 1).mul_(255).round_().to(torch.uint8)
        finally:
            torch.backends.cudnn.benchmark = benchmark
        return video.permute(1, 2, 3, 0).contiguous()

    def close(self) -> None:
        """Free the VRAM and the page-locked RAM (idempotent); the VAE is left a skeleton."""
        if self._closed:
            return
        self._closed = True
        if self.streamer is not None:
            self.streamer.close()
        while self._undo:
            self._undo.pop()()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def load_wan_vae(
    model_dir: str | PathLike[str],
    config: VaeConfig | str = "bf16-resident",
    *,
    slots: int = 2,
    device: torch.device | str = "cuda",
) -> WanVae:
    """Load the decoder of a Wan VAE from a local directory.

    Args:
        model_dir: the VAE directory of a ``diffusers`` checkpoint. Nothing is downloaded.
        config: a :class:`VaeConfig`, or the name of one of :data:`VAE_CONFIGS`;
            :func:`choose_vae_config` picks one for a VRAM budget.
        slots: VRAM buffers for a streamed decoder, each as large as its largest block.
        device: the CUDA device that computes.

    Only the decoder is read from the checkpoint, converted to the dtype it runs in. The
    encoder stays an empty skeleton. With streamed weights the blocks of the decoder wait in
    page-locked RAM and the rest of it (first and last convolutions, the output norm) is
    resident.

    The measured peaks and PSNR thresholds of :data:`VAE_CONFIGS` hold for the Wan 2.2 VAE
    (16 px per latent cell); another VAE loads and decodes, but with numbers of its own.
    """
    torch = require_cuda()
    diffusers = _import_diffusers()
    dev = resolve_device(device)
    if isinstance(config, str):
        named = {known.name: known for known in VAE_CONFIGS}
        if config not in named:
            raise ValueError(f"no VAE configuration is called {config!r}: {sorted(named)}")
        config = named[config]
    root = Path(model_dir)
    index = SafetensorsIndex(root)
    cls = getattr(diffusers, _VAE_CLASS)
    stored = cls.load_config(root)
    if stored.get("_class_name") != _VAE_CLASS:
        raise ValueError(f"{root} holds a {stored.get('_class_name')}; this loads {_VAE_CLASS}")
    with skeleton.parameters_on_meta(torch):
        vae = cls.from_config(stored)
    vae.eval()
    _set_dtypes(torch, vae, getattr(torch, config.dtype), cls._keep_in_fp32_modules or ())

    names = {id(module): name for name, module in vae.named_modules()}
    blocks = _decoder_blocks(vae.decoder) if config.streamed else []
    unused = [module for module in (vae.encoder, vae.quant_conv) if module is not None]
    skip = {id(inner) for module in (*blocks, *unused) for inner in module.modules()}
    undo: list[Any] = []
    streamer = None
    try:
        resident_bytes = skeleton.make_resident(torch, vae, index, dev, skip, undo, cast=True)
        if blocks:
            weights = FileWeights(index, [f"{names[id(block)]}." for block in blocks], cast=True)
            with warnings.catch_warnings():
                # The blocks of a decoder grow as the channels shrink: unequal by design.
                warnings.filterwarnings("ignore", message="the largest streamed block")
                streamer = LayerStreamer(blocks, slots=slots, device=dev, weights=weights)
    except BaseException:
        if streamer is not None:
            streamer.close()
        while undo:
            undo.pop()()
        raise
    return WanVae(vae, config, streamer, dev, resident_bytes, undo)


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


def _decoder_blocks(decoder: Any) -> list[Any]:
    """The modules of a Wan decoder that hold its weights, in the order they run."""
    middle = decoder.mid_block
    blocks = [middle.resnets[0]]
    for attention, resnet in zip(middle.attentions, middle.resnets[1:]):
        blocks += [attention, resnet]
    for up in decoder.up_blocks:
        blocks += list(up.resnets)
        blocks += list(getattr(up, "upsamplers", None) or [])  # Wan 2.1
        blocks.append(getattr(up, "upsampler", None))  # Wan 2.2
    return [block for block in blocks if block is not None and any(True for _ in block.parameters())]


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


__all__ = [
    "MIN_DIFFUSERS",
    "VAE_CONFIGS",
    "VAE_MARGIN",
    "VaeConfig",
    "WanVae",
    "choose_vae_config",
    "denoise",
    "load_wan_transformer",
    "load_wan_vae",
]
