# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Text to video with a Wan pipeline of ``diffusers`` on a GPU that holds none of its models.

:func:`load_wan_pipeline` reads a local ``WanPipeline`` directory and returns an object that
runs its three stages one after another, each loaded only while it runs
(:class:`~hypertier.diffusion.sequential.SequentialPipeline`)::

    text         umT5 encoder, blocks read from disk on every call   (load_text_encoder)
    transformer  blocks streamed from page-locked RAM                (load_wan_transformer)
    vae          decoder in tiles, in the best configuration that fits (load_wan_vae)

The latents are those of ``WanPipeline`` bit for bit, because every stage does what
``diffusers`` does: prompts are cleaned and tokenized the same way and encoded one at a time,
and the sampling loop is :func:`~hypertier.adapters.hf_diffusers.denoise`. The frames are
not: the VAE decodes in tiles, which is approximate (see
:mod:`hypertier.diffusion.tiling`).

A transformer distilled for a few steps (FastWan) takes the place of the pipeline's with
``transformer=`` and ``dmd_timesteps=``: three passes instead of a hundred, sampled by
:func:`~hypertier.adapters.hf_diffusers.denoise_dmd`. ``diffusers`` has no pipeline to compare
that with; it is the loop of FastVideo.

Before anything is loaded, a generation is checked against the VRAM that is free, with the
peaks measured for Wan 2.2 TI2V-5B: what does not fit is refused with what would, and so is
what was not measured. Nothing is extrapolated.

Needs the optional extra: ``pip install hypertier[diffusers]``. Importing this module imports
neither PyTorch nor ``diffusers``.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

from hypertier.adapters import _skeleton as skeleton
from hypertier.adapters.hf_diffusers import (
    VAE_CONFIGS,
    VaeConfig,
    _import_diffusers,
    choose_vae_config,
    denoise,
    denoise_dmd,
    load_wan_transformer,
    load_wan_vae,
    usable_vae_configs,
    vae_config_named,
)
from hypertier.adapters.hf_transformers import load_text_encoder
from hypertier.diffusion.sequential import (
    CUDA_CONTEXT,
    VRAM_MARGIN,
    SequentialPipeline,
    StageReport,
)
from hypertier.io.safetensors_index import SafetensorsIndex
from hypertier.telemetry._common import MiB

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self

MAX_SEQUENCE_LENGTH = 512
"""Tokens a prompt is padded or cut to, as in ``WanPipeline``."""

CACHED_PROMPTS = 8
"""Embeddings kept in memory, 4 MiB each for umT5-XXL: the negative prompt is the same in
every call, and a prompt sampled again with another seed need not be encoded again."""

TEXT_PEAK = 912 * MiB + CUDA_CONTEXT
"""VRAM the process holds at the peak of the text stage: umT5-XXL through two slots, two
prompts of 512 tokens (880 MiB of allocator for one prompt), and what a process holds outside
the allocator."""

FFN_CHUNK = 2048
"""Tokens the feed-forward networks ran over at a time in the peaks measured "in chunks".
Chunks of 512 and of 4,096 tokens gave the same peak at 12,090 tokens: once the networks are
in chunks, the peak is somewhere else."""

_WHOLE, _CHUNKED, _ONE_PASS = "whole", "chunked", "one pass"

TRANSFORMER_PEAKS: dict[tuple[int, str], tuple[tuple[int, int], ...]] = {
    (2, _WHOLE): (
        (1_950, 1_098), (5_070, 1_502), (11_440, 2_304), (12_090, 2_402), (14_960, 2_752),
        (16_720, 2_972), (18_480, 3_218), (20_240, 3_438),
    ),
    (1, _WHOLE): (
        (1_950, 784), (5_070, 1_188), (11_440, 1_990), (12_090, 2_088), (14_960, 2_438),
        (16_720, 2_658), (18_480, 2_904), (20_240, 3_124), (22_000, 3_338), (23_760, 3_572),
    ),
    (2, _CHUNKED): (
        (1_950, 1_098), (5_070, 1_282), (11_440, 1_812), (12_090, 1_880), (14_960, 2_108),
        (16_720, 2_252), (22_000, 2_706), (27_280, 3_158),
    ),
    (1, _CHUNKED): (
        (1_950, 784), (5_070, 968), (11_440, 1_498), (12_090, 1_566), (14_960, 1_794),
        (16_720, 1_938), (22_000, 2_392), (27_280, 2_844),
    ),
    (2, _ONE_PASS): ((1_950, 1_310), (5_070, 2_034), (11_440, 3_522), (12_090, 3_236)),
    (1, _ONE_PASS): ((1_950, 996), (5_070, 1_720), (11_440, 3_208), (12_090, 3_348)),
}  # fmt: skip
"""The peak of the transformer stage of Wan 2.2 TI2V-5B by the number of tokens: ``(tokens,
MiB)`` pairs as measured, fewest tokens first, for every way of running it that was measured:
the number of slots, and the feed-forward networks whole, or in chunks of :data:`FFN_CHUNK`
tokens, or both branches of the guidance in one pass over the weights.

A peak is what PyTorch's allocator reserved in a run of two guided steps, in a process of its
own on the reference laptop, with 3,652 MiB of VRAM free. Between two measured numbers of
tokens it is read off the line through them; above the most, nothing is known. The sizes are
480p x 17, 49 and 121 frames (1,950, 5,070 and 12,090 tokens) and 720p x 49 to 121 (880 tokens
per four frames: 11,440 to 27,280).

Whole, the peak grows by 135 KB per token; in chunks by 85 KB, which is what lets 720p x 121
run in 4 GB. A slot is 312 MiB. The longest video measured for a way is the longest that ran:
the next length up was not tried where the last one left under 150 MiB free.

One pass holds the time embedding of both branches at once before one row of it is kept,
which is most of what it adds. Its two largest sizes ran with the VRAM almost full, where the
allocator gives back what it has cached before it fails: the figure for two slots at 12,090
tokens is lower than the one below it for that reason, not because it needs less.
"""


def _between(points: Sequence[tuple[int, int]], at: int) -> float:
    """The value at ``at`` of the line through the two measured ``points`` around it.

    Before the first point it is the value of the first, and after the last that of the last:
    no line is continued past what was measured.
    """
    x, y = points[0]
    if at <= x:
        return y
    for further, more in points[1:]:
        if at <= further:
            return y + (more - y) * (at - x) / (further - x)
        x, y = further, more
    return y


@dataclass(frozen=True)
class VideoResult:
    """What a generation gives back.

    Attributes:
        frames: ``uint8`` frames ``[frames, height, width, 3]`` in host RAM.
        latents: what the sampling loop returned, on the CPU: the input of the VAE stage.
        stages: the :class:`~hypertier.diffusion.sequential.StageReport` of every stage that
            ran, by name. The text stage is missing when every prompt was cached.
        vae_config: the name of the VAE configuration that was chosen.
    """

    frames: torch.Tensor
    latents: torch.Tensor
    stages: dict[str, StageReport]
    vae_config: str


class WanVideoPipeline(SequentialPipeline):
    """A ``WanPipeline`` directory, run a stage at a time. See :func:`load_wan_pipeline`."""

    def __init__(
        self,
        model_dir: str | PathLike[str],
        *,
        transformer: str | PathLike[str] | None = None,
        dmd_timesteps: Sequence[int] | None = None,
        slots: int = 2,
        ffn_chunk: int | None = None,
        vae_config: VaeConfig | str | None = None,
        vram_budget: int | None = None,
        margin: int = VRAM_MARGIN,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__(device, vram_budget=vram_budget, margin=margin)
        self._diffusers = _import_diffusers()
        self.root = Path(model_dir)
        self.transformer_dir = self.root / "transformer" if transformer is None else Path(transformer)
        self.dmd_timesteps = None if dmd_timesteps is None else tuple(int(t) for t in dmd_timesteps)
        if self.dmd_timesteps is not None and not self.dmd_timesteps:
            raise ValueError("dmd_timesteps is empty: a sampler needs at least one step")
        self.slots = slots
        if ffn_chunk is not None and ffn_chunk < 1:
            raise ValueError(f"cannot run {ffn_chunk} tokens at a time")
        self.ffn_chunk = ffn_chunk
        if isinstance(vae_config, str):
            vae_config = vae_config_named(vae_config)
        self.vae_config = vae_config
        index = _read_json(self.root / "model_index.json")
        if index.get("_class_name") != "WanPipeline":
            raise ValueError(
                f"{self.root} holds a {index.get('_class_name')}; this runs a WanPipeline"
            )
        if (index.get("transformer_2") or [None])[0] is not None:
            raise ValueError(
                f"{self.root} samples with two transformers, one after the other; only "
                "pipelines with one are supported"
            )
        transformer = _read_json(self.transformer_dir / "config.json")
        vae = _read_json(self.root / "vae" / "config.json")
        self.expand_timesteps = bool(index.get("expand_timesteps", False))
        self.patch_size = tuple(transformer["patch_size"])
        self.spatial = int(vae["scale_factor_spatial"])
        self.temporal = int(vae["scale_factor_temporal"])
        self.scheduler_class = getattr(self._diffusers, index["scheduler"][1])
        # The peaks were measured for one model: for another nothing is promised up front.
        self.measured = _is_wan22_ti2v_5b(transformer, vae)
        self._capability = tuple(self._torch.cuda.get_device_capability(self.device))
        self._prompts: OrderedDict[tuple[str, str, int, str], Any] = OrderedDict()
        self._text_dtype: str | None = None

    # -----------------------------------------------------------------------------------------
    # What a generation needs

    def latent_shape(self, height: int, width: int, num_frames: int) -> tuple[int, int, int]:
        """Frames, rows and columns of the latent video of a ``height`` x ``width`` video."""
        _, p_h, p_w = self.patch_size
        rows, columns = self.spatial * p_h, self.spatial * p_w
        if height < rows or width < columns or height % rows or width % columns:
            fit = (max(height // rows, 1) * rows, max(width // columns, 1) * columns)
            raise ValueError(
                f"{height} x {width} px does not divide into patches of {rows} x {columns} px: "
                f"the nearest size below is {fit[0]} x {fit[1]}"
            )
        if num_frames < 1 or (num_frames - 1) % self.temporal:
            fit = max((num_frames - 1) // self.temporal, 0) * self.temporal + 1
            raise ValueError(
                f"{num_frames} frames is not one frame plus groups of {self.temporal}: the "
                f"nearest length below is {fit}"
            )
        frames = (num_frames - 1) // self.temporal + 1
        return frames, height // self.spatial, width // self.spatial

    def tokens(self, height: int, width: int, num_frames: int) -> int:
        """Tokens the transformer attends over for a video of this size."""
        frames, rows, columns = self.latent_shape(height, width, num_frames)
        p_t, p_h, p_w = self.patch_size
        return (frames // p_t) * (rows // p_h) * (columns // p_w)

    def transformer_peak(
        self,
        tokens: int,
        *,
        slots: int,
        one_pass: bool = False,
        ffn_chunk: int | None = None,
    ) -> int | None:
        """VRAM the process holds at the peak of the transformer stage, from measurements.

        None for a model whose peaks were not measured. For Wan 2.2 TI2V-5B the peak is
        interpolated between the numbers of tokens it was measured with
        (:data:`TRANSFORMER_PEAKS`). With fewer tokens than the fewest measured it is the peak
        of those, which it cannot exceed. With more than the most, or run in a way that was
        not measured, nothing is known and nothing is made up: a ``ValueError``.
        """
        if not self.measured:
            return None
        chunked = ffn_chunk is not None and tokens > ffn_chunk
        way = _ONE_PASS if one_pass else _CHUNKED if chunked else _WHOLE
        how = _how(slots, one_pass, ffn_chunk if chunked else None)
        points = TRANSFORMER_PEAKS.get((slots, way))
        if points is None or (chunked and (one_pass or ffn_chunk > FFN_CHUNK)):
            raise ValueError(
                f"the VRAM the transformer needs was not measured with {how}: it is refused "
                "instead of guessed"
            )
        most = points[-1][0]
        if tokens > most:
            raise ValueError(
                f"the VRAM the transformer needs with {how} was measured up to {most} tokens: "
                f"{tokens} tokens is outside what can be checked, and is refused instead of "
                "guessed"
            )
        return round(_between(points, tokens) * MiB) + CUDA_CONTEXT

    def check(
        self, *, height: int, width: int, num_frames: int, one_pass: bool = False
    ) -> dict[str, int]:
        """Refuse now, with what would fit, a generation that is not expected to fit.

        Every stage is checked against the VRAM that is free at this moment; each is checked
        again, against what is free then, when it starts. Returns the expected peak of each
        stage in bytes: empty for a model whose peaks were not measured.

        What is compared with what: the expected peaks are what PyTorch's allocator reserved
        in measured runs, plus the CUDA context of a process; the budget is the VRAM the
        driver reports free after ``torch.cuda.empty_cache()``, plus what this process holds
        (:meth:`~hypertier.diffusion.sequential.SequentialPipeline.available`), or the
        caller's ``vram_budget`` if that is less. The F3 criterion (3.4 GB) is the peak of the
        process as the driver counts it, which ``bench_video`` samples while a generation
        runs: see :class:`~hypertier.adapters.hf_diffusers.VaeConfig`.

        Nothing is extrapolated. The peaks of the VAE were measured for videos of 17 to 121
        frames, and those of the transformer up to a number of tokens that depends on how it
        runs (:data:`TRANSFORMER_PEAKS`): a video outside what was measured is refused.
        """
        if not self.measured:
            return {}
        shortest, longest = self._vae_candidates()[0].measured_frames
        if not shortest <= num_frames <= longest:
            raise ValueError(
                f"the VRAM this model needs was measured for videos of {shortest} to {longest} "
                f"frames: {num_frames} frames is outside what can be checked, and is refused "
                "instead of guessed"
            )
        available = self.available()
        one_pass = one_pass and self.dmd_timesteps is None  # a distilled model has no guidance

        def peak(frames: int, slots: int, ffn_chunk: int | None) -> int:
            tokens = self.tokens(height, width, frames)
            found = self.transformer_peak(
                tokens, slots=slots, one_pass=one_pass, ffn_chunk=ffn_chunk
            )
            assert found is not None  # the model is the measured one
            return found

        def fits(frames: int, slots: int, ffn_chunk: int | None) -> bool:
            try:
                self._vae_config(available, frames)
                return self.fits(peak(frames, slots, ffn_chunk), available)
            except ValueError:  # too large for every VAE configuration, or not measured
                return False

        tokens = self.tokens(height, width, num_frames)
        if not fits(num_frames, self.slots, self.ffn_chunk):
            advice = []
            if self.slots > 1 and fits(num_frames, 1, self.ffn_chunk):
                advice.append("slots=1 fits, at the price of copies that no longer overlap")
            if self.ffn_chunk is None:
                for slots in dict.fromkeys((self.slots, 1)):
                    if fits(num_frames, slots, FFN_CHUNK):
                        also = "" if slots == self.slots else " with slots=1"
                        advice.append(
                            f"ffn_chunk={FFN_CHUNK}{also} fits, with latents that differ by "
                            "rounding"
                        )
                        break
            shorter = next(
                (n for n in range(num_frames - self.temporal, shortest - 1, -self.temporal)
                 if fits(n, self.slots, self.ffn_chunk)),
                None,
            )  # fmt: skip
            if shorter is not None:
                advice.append(f"{shorter} frames fit at {height} x {width}")
            hint = "Try: " + "; or ".join(advice) if advice else "Nothing shorter fits either"
            try:
                expected = peak(num_frames, self.slots, self.ffn_chunk)
            except ValueError as not_measured:
                raise ValueError(f"{not_measured}. {hint}") from None
            if not self.fits(expected, available):
                chunk = self.ffn_chunk if tokens > (self.ffn_chunk or tokens) else None
                name = f"transformer ({tokens} tokens, {_how(self.slots, one_pass, chunk)})"
                raise ValueError(self.shortfall(name, expected, available, hint))
            least = min(c.peak_bytes(num_frames) for c in self._vae_candidates())
            raise ValueError(
                self.shortfall(f"VAE ({num_frames} frames)", least, available, hint)
            )
        if not self.fits(TEXT_PEAK, available):
            raise ValueError(self.shortfall("text", TEXT_PEAK, available))
        chosen = self._vae_config(available, num_frames)
        return {
            "text": TEXT_PEAK,
            "transformer": peak(num_frames, self.slots, self.ffn_chunk),
            "vae": chosen.peak_bytes(num_frames),
        }

    # -----------------------------------------------------------------------------------------
    # The stages

    def encode(self, prompts: Sequence[str]) -> torch.Tensor:
        """The embeddings of ``prompts`` as ``WanPipeline`` computes them: ``[n, 512, dim]``.

        Each prompt is cleaned, cut or padded to 512 tokens, encoded with the bits it gets
        alone, and zeroed after its last token. Prompts seen before come from memory; when
        all of them do, the text encoder is not loaded at all.

        Bytes moved: every block of the encoder from disk to VRAM once, whatever the number
        of prompts; nothing when they are all cached.
        """
        torch = self._torch
        cleaned = [self._clean(prompt) for prompt in prompts]
        directory = self.root / "text_encoder"
        if self._text_dtype is None:
            dtype = skeleton.checkpoint_dtype(torch, SafetensorsIndex(directory))
            self._text_dtype = str(dtype)
        keys = [
            (text, str(directory.resolve()), MAX_SEQUENCE_LENGTH, self._text_dtype)
            for text in cleaned
        ]
        missing = list(dict.fromkeys(key for key in keys if key not in self._prompts))
        if missing:
            import transformers

            tokenizer = transformers.AutoTokenizer.from_pretrained(self.root / "tokenizer")
            batch = tokenizer(
                [key[0] for key in missing],
                padding="max_length",
                max_length=MAX_SEQUENCE_LENGTH,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            used = batch.attention_mask.gt(0).sum(dim=1).tolist()
            with self.stage("text", TEXT_PEAK if self.measured else None) as report:
                started = time.perf_counter()
                with load_text_encoder(directory, device=self.device) as encoder:
                    report.load_seconds = time.perf_counter() - started
                    with torch.no_grad():
                        hidden = encoder.model(
                            batch.input_ids.to(self.device), batch.attention_mask.to(self.device)
                        ).last_hidden_state.cpu()
                    report.details.update(prompts=len(missing), tokens=used)
                    self.read_context(report)
            for row, (key, length) in enumerate(zip(missing, used)):
                embedding = hidden[row].clone()
                embedding[length:] = 0  # as WanPipeline: the tokens of the prompt, then zeros
                self._remember(key, embedding)
        for key in keys:
            self._prompts.move_to_end(key)
        return torch.stack([self._prompts[key] for key in keys])

    def denoise(
        self,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None = None,
        *,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: int = 0,
        one_pass: bool = False,
    ) -> torch.Tensor:
        """Sample the latents of one video; returns them on the CPU.

        The loop is :func:`~hypertier.adapters.hf_diffusers.denoise` with the scheduler of
        the pipeline and noise drawn on the CPU from ``seed``. With ``dmd_timesteps`` it is
        :func:`~hypertier.adapters.hf_diffusers.denoise_dmd`: one pass per timestep, and
        ``negative_prompt_embeds``, ``num_inference_steps``, ``guidance_scale`` and
        ``one_pass`` are not used.
        """
        torch = self._torch
        shape = self.latent_shape(height, width, num_frames)
        tokens = self.tokens(height, width, num_frames)
        few = self.dmd_timesteps
        guided = guidance_scale > 1.0 and few is None
        expected = self.transformer_peak(
            tokens, slots=self.slots, one_pass=one_pass and guided, ffn_chunk=self.ffn_chunk
        )
        generator = torch.Generator("cpu").manual_seed(seed)
        with self.stage("transformer", expected) as report:
            started = time.perf_counter()
            with load_wan_transformer(
                self.transformer_dir,
                slots=self.slots,
                ffn_chunk=self.ffn_chunk,
                device=self.device,
            ) as streamed:
                report.load_seconds = time.perf_counter() - started
                if few is not None:
                    latents = denoise_dmd(
                        streamed.model,
                        prompt_embeds,
                        latent_shape=shape,
                        timesteps=few,
                        generator=generator,
                    ).cpu()
                else:
                    latents = denoise(
                        streamed.model,
                        self.scheduler_class.from_pretrained(self.root / "scheduler"),
                        prompt_embeds,
                        negative_prompt_embeds,
                        latent_shape=shape,
                        expand_timesteps=self.expand_timesteps,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        one_pass=one_pass and guided,
                    ).cpu()
                report.details.update(
                    tokens=tokens,
                    slots=self.slots,
                    ffn_chunk=self.ffn_chunk,
                    steps=num_inference_steps if few is None else len(few),
                    dmd_timesteps=None if few is None else list(few),
                    weights_vram_bytes=streamed.vram_bytes,
                    pinned_ram_bytes=streamed.ram_bytes,
                    **_pass_summary(streamed.streamer.pass_stats()),
                )
                self.read_context(report)  # with the page-locked RAM still registered
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents into ``uint8`` frames ``[frames, height, width, 3]`` in host RAM.

        The configuration is the best one whose measured peak fits in the VRAM that is free
        now, on this GPU: see :func:`~hypertier.adapters.hf_diffusers.choose_vae_config`.
        """
        num_frames = (latents.shape[2] - 1) * self.temporal + 1
        config = self._vae_config(self.available(), num_frames)
        expected = config.peak_bytes(num_frames) if self.measured else None
        with self.stage("vae", expected) as report:
            started = time.perf_counter()
            with load_wan_vae(self.root / "vae", config, device=self.device) as vae:
                report.load_seconds = time.perf_counter() - started
                plan = vae.plan(*latents.shape[-2:])
                frames = vae.decode(latents)
                report.details.update(
                    config=config.name,
                    tiles=plan.count,
                    tile_cells=list(plan.tile),
                    min_psnr_db=config.min_psnr_db,
                    weights_vram_bytes=vae.vram_bytes,
                    pinned_ram_bytes=vae.ram_bytes,
                )
                self.read_context(report)
        return frames

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: int = 0,
        one_pass: bool = False,
    ) -> VideoResult:
        """Generate one video from a prompt.

        Args:
            prompt: what to draw.
            negative_prompt: what to steer away from. Used when ``guidance_scale`` is above 1.
            height, width: of the video, in pixels: multiples of 32 for Wan 2.2 TI2V.
            num_frames: one frame plus groups of four.
            num_inference_steps: steps of the scheduler.
            guidance_scale: scale of classifier-free guidance; 1 or less samples without it.
            seed: of the initial noise, drawn on the CPU.
            one_pass: both branches of the guidance in one pass over the weights. Same bits
                as two passes; it only wins time with few tokens, and costs VRAM.

        A pipeline with ``dmd_timesteps`` samples at those timesteps without guidance:
        ``negative_prompt``, ``num_inference_steps``, ``guidance_scale`` and ``one_pass`` are
        not used, and the negative prompt is not encoded.

        Bytes moved: the text encoder from disk once; every transformer block from RAM to
        VRAM once per pass, two passes per guided step; the VAE's weights once.
        """
        first = len(self.reports)
        self.latent_shape(height, width, num_frames)  # a size it cannot draw: refused now
        self.check(height=height, width=width, num_frames=num_frames, one_pass=one_pass)
        guided = guidance_scale > 1.0 and self.dmd_timesteps is None
        embeds = self.encode([prompt, negative_prompt] if guided else [prompt])
        latents = self.denoise(
            embeds[:1],
            embeds[1:] if guided else None,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            one_pass=one_pass,
        )
        frames = self.decode(latents)
        stages = {report.name: report for report in self.reports[first:]}
        return VideoResult(frames, latents, stages, stages["vae"].details["config"])

    def close(self) -> None:
        """Forget the cached prompts. No stage outlives the call that ran it."""
        self._prompts.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -----------------------------------------------------------------------------------------

    def _clean(self, prompt: str) -> str:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        from diffusers.utils import is_ftfy_available

        if not is_ftfy_available():
            raise RuntimeError(
                "WanPipeline cleans prompts with ftfy, and without it the same prompt gives "
                "other tokens: pip install hypertier[diffusers]"
            )
        return prompt_clean(prompt)

    def _remember(self, key: tuple[str, str, int, str], embedding: Any) -> None:
        self._prompts[key] = embedding
        while len(self._prompts) > CACHED_PROMPTS:
            self._prompts.popitem(last=False)

    def _vae_candidates(self) -> tuple[VaeConfig, ...]:
        """What the VAE stage may run in: the caller's choice, or what this GPU can run."""
        if self.vae_config is not None:
            return (self.vae_config,)
        return usable_vae_configs(VAE_CONFIGS, self._capability)

    def _vae_config(self, available: int, num_frames: int) -> VaeConfig:
        candidates = self._vae_candidates()
        if not self.measured and candidates:
            return candidates[0]  # no peaks to choose by: the best one, unchecked
        return choose_vae_config(
            available, num_frames, margin=self.margin, capability=self._capability,
            configs=candidates,
        )  # fmt: skip


def load_wan_pipeline(
    model_dir: str | PathLike[str],
    *,
    transformer: str | PathLike[str] | None = None,
    dmd_timesteps: Sequence[int] | None = None,
    slots: int = 2,
    ffn_chunk: int | None = None,
    vae_config: VaeConfig | str | None = None,
    vram_budget: int | None = None,
    margin: int = VRAM_MARGIN,
    device: torch.device | str = "cuda",
) -> WanVideoPipeline:
    """Open a local ``WanPipeline`` directory to generate video a stage at a time.

    Args:
        model_dir: a ``diffusers`` pipeline directory (``model_index.json`` and one directory
            per component). Nothing is downloaded, and nothing is loaded yet.
        transformer: the directory of another transformer to sample with, instead of the
            pipeline's: a distilled one, which ships without the text encoder and the VAE it
            shares with its base model.
        dmd_timesteps: the timesteps of a transformer distilled by distribution matching
            (1000, 757, 522 for FastWan 2.2 TI2V-5B): the pipeline then samples at those,
            one pass each, without guidance. None: the scheduler of the pipeline.
        slots: VRAM buffers the transformer's blocks are streamed through.
        ffn_chunk: run the feed-forward network of every transformer block over this many
            tokens at a time: about a third less VRAM per token, for videos that do not fit
            otherwise. **Not exact**: the latents differ by rounding from those of
            ``WanPipeline`` (see :func:`~hypertier.diffusion.chunk_tokens`). None: whole.
        vae_config: how the VAE decodes: a
            :class:`~hypertier.adapters.hf_diffusers.VaeConfig` or the name of one. None
            takes the best one that this GPU can run and that fits when the stage starts.
        vram_budget: VRAM the process may hold, in bytes, as the driver counts it. None: what
            is free when each stage starts, measured after the previous stage is released
            and the allocator's cache emptied.
        margin: bytes to leave free above the expected peak of a stage.
        device: the CUDA device every stage computes on.

    Call the result with a prompt, or call its stages one by one (``encode``, ``denoise``,
    ``decode``). Each call loads a stage, runs it and releases it: nothing stays in VRAM or
    in page-locked RAM between calls, and the transformer is read again every time (14-17 s
    for the 5B model).
    """
    return WanVideoPipeline(
        model_dir,
        transformer=transformer,
        dmd_timesteps=dmd_timesteps,
        slots=slots,
        ffn_chunk=ffn_chunk,
        vae_config=vae_config,
        vram_budget=vram_budget,
        margin=margin,
        device=device,
    )


# ---------------------------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise ValueError(f"{path} is missing: this needs a diffusers pipeline directory") from None


def _is_wan22_ti2v_5b(transformer: dict[str, Any], vae: dict[str, Any]) -> bool:
    """Whether the measured peaks are of this model: Wan 2.2 TI2V-5B and its VAE."""
    width = transformer.get("num_attention_heads", 0) * transformer.get("attention_head_dim", 0)
    return (
        width == 3072
        and transformer.get("num_layers") == 30
        and transformer.get("in_channels") == 48
        and vae.get("scale_factor_spatial") == 16
        and vae.get("z_dim") == 48
    )


def _how(slots: int, one_pass: bool, ffn_chunk: int | None) -> str:
    """A way of running the transformer, as the messages name it."""
    text = f"{slots} slot" + ("" if slots == 1 else "s")
    if one_pass:
        text += ", both branches of the guidance in one pass"
    if ffn_chunk is not None:
        text += f", the feed-forward networks {ffn_chunk} tokens at a time"
    return text


def _pass_summary(stats: Sequence[Any]) -> dict[str, Any]:
    """What the passes of the transformer cost, in a few numbers."""
    if not stats:
        return {"passes": 0}
    hidden = [s.efficiency for s in stats if s.efficiency is not None]
    return {
        "passes": len(stats),
        "seconds_per_pass": median(s.total_ms for s in stats) / 1e3,
        "copy_seconds_per_pass": median(s.copy_ms for s in stats) / 1e3,
        "stall_seconds": sum(s.stall_ms for s in stats) / 1e3,
        "overlap_efficiency_min": min(hidden) if hidden else None,
        "overlap_efficiency_median": median(hidden) if hidden else None,
    }


__all__ = [
    "MAX_SEQUENCE_LENGTH",
    "VideoResult",
    "WanVideoPipeline",
    "load_wan_pipeline",
]
