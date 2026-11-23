# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Hugging Face ``transformers`` model that does not fit in VRAM.

:func:`load_causal_lm` returns the ordinary ``transformers`` model, to be called or driven by
``generate()`` as usual, with its weights placed by size and by how they are used::

    token embedding   stays on the CPU, memory-mapped from the checkpoint: a lookup touches
                      one row per token, so only the activations go up to the GPU
    decoder layers    streamed through the VRAM slots (``LayerStreamer``), one block each
    output projection streamed too, split by vocabulary rows into pieces no larger than a
                      layer; whole, it would make every slot three times as big
    everything else   (final norm, rotary tables...) resident on the GPU: kilobytes

:func:`load_text_encoder` does the same for an encoder-only model (T5, umT5), the text stage
of a diffusion pipeline: embedding on the CPU, encoder blocks streamed, the rest resident.

The model classes are not rewritten. The model is built by its own ``__init__`` with every
parameter sent to the ``meta`` device as it is registered, so no weight is ever allocated in
RAM, while the buffers that ``__init__`` computes (rotary frequencies) are computed on the CPU
exactly as in a normal load. ``model.device`` answers with the GPU, since ``generate()`` moves
its inputs there.

Needs the optional extra: ``pip install hypertier[hf]``. Importing this module imports
neither PyTorch nor ``transformers``.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from itertools import pairwise
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hypertier.adapters import _skeleton as skeleton
from hypertier.diffusion.chunking import one_sample_at_a_time
from hypertier.io.safetensors_index import SafetensorsIndex
from hypertier.stream.pingpong import LayerStreamer, find_blocks
from hypertier.stream.pinned import ALIGN, align_up
from hypertier.stream.source import FileWeights
from hypertier.telemetry._common import format_bytes, require_cuda, resolve_device

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self

MIN_TRANSFORMERS = (5, 24)
"""Oldest ``transformers`` this adapter is tested with; the ``[hf]`` extra pins the same."""

_BATCHED_IN_A_BLOCK = ("hidden_states", "attention_mask", "position_bias")
"""The arguments of an encoder block that have a row per prompt, in T5 and umT5."""


def head_rows(rows: int, row_bytes: int, chunk_bytes: int) -> list[tuple[int, int]]:
    """Split ``rows`` rows into the fewest even ranges of at most ``chunk_bytes`` each.

    This is how the output projection (vocabulary x hidden) is cut into blocks that fit the
    VRAM slots: ``chunk_bytes`` is the size of the largest decoder layer.
    """
    if rows < 0 or row_bytes < 1:
        raise ValueError(f"cannot split {rows} rows of {row_bytes} bytes")
    per_chunk = chunk_bytes // row_bytes
    if per_chunk < 1:
        raise ValueError(
            f"one row of the output projection ({format_bytes(row_bytes)}) is larger than a "
            f"block ({format_bytes(chunk_bytes)})"
        )
    parts = max(1, -(-rows // per_chunk))
    bounds = [rows * k // parts for k in range(parts + 1)]
    return list(pairwise(bounds))


class StreamedModel:
    """A ``transformers`` model whose weights are streamed, and what it holds where.

    Attributes:
        model: the ``transformers`` model. Call it or use ``model.generate()`` as usual,
            under ``torch.no_grad()`` or ``torch.inference_mode()``.
        streamer: the :class:`~hypertier.stream.LayerStreamer` behind it, for
            ``pass_stats()``: a pass is one forward of the model, i.e. the prompt or one
            generated token.
        device: the GPU that computes.
        head_rows: the vocabulary ranges the output projection is streamed in; empty for a
            model without one (a text encoder).
        resident_bytes: weights and buffers kept on the GPU outside the slots.
        embedding_bytes: size of the token embedding, which stays in the file (page cache).
    """

    def __init__(
        self,
        model: Any,
        streamer: LayerStreamer,
        device: Any,
        rows: list[tuple[int, int]],
        resident_bytes: int,
        embedding_bytes: int,
        undo: list[Any],
    ) -> None:
        self.model = model
        self.streamer = streamer
        self.device = device
        self.head_rows = tuple(rows)
        self.resident_bytes = resident_bytes
        self.embedding_bytes = embedding_bytes
        self._undo = undo
        self._closed = False

    @property
    def vram_bytes(self) -> int:
        """VRAM held by weights: the slots plus what is resident. Activations are extra."""
        return self.streamer.vram_bytes + self.resident_bytes

    @property
    def ram_bytes(self) -> int:
        """Page-locked RAM held by weights: RAM-tier blocks plus staging buffers."""
        return self.streamer.ram_bytes

    def close(self) -> None:
        """Stop streaming and free the VRAM and the page-locked RAM (idempotent).

        The model is left as an empty skeleton: its weights were never in memory, they are in
        the checkpoint. Load it again to use it again.
        """
        if self._closed:
            return
        self._closed = True
        self.streamer.close()
        while self._undo:
            self._undo.pop()()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def load_causal_lm(
    model_dir: str | PathLike[str],
    *,
    slots: int = 2,
    ram_budget: int | None = None,
    num_stage: int = 2,
    device: torch.device | str = "cuda",
) -> StreamedModel:
    """Load a causal language model from a local directory, streamed through ``slots`` slots.

    Args:
        model_dir: a ``transformers`` model directory with safetensors weights
            (``config.json`` and ``*.safetensors``). Nothing is downloaded.
        slots: VRAM buffers, each as large as the largest decoder layer (see
            :class:`~hypertier.stream.LayerStreamer`).
        ram_budget: page-locked RAM the streamed weights may take. None keeps them all in
            RAM; with less, the blocks that do not fit are read from disk on every pass (see
            :class:`~hypertier.stream.FileWeights`). The token embedding is not counted: it
            is page cache, which the kernel reclaims.
        num_stage: staging buffers for the blocks read from disk.
        device: the CUDA device that computes.

    Bytes moved per forward of the model (the prompt, or one generated token): every decoder
    layer and the whole output projection from RAM to VRAM, plus the activations of the
    embedding. The size of the prompt does not change it.

    Limits:

    * Decoder-only models whose checkpoint names its tensors as the modules do, with an
      ``nn.Embedding`` going in and an ``nn.Linear`` coming out. Anything else raises.
    * Building the model sends every ``Parameter`` registered in the process to ``meta``
      while it lasts: do not build other modules in another thread meanwhile.
    * The limits of :class:`~hypertier.stream.LayerStreamer` apply: inference only, one
      thread, no ``torch.compile``, no ``model.to(...)``.
    """
    torch = require_cuda()
    transformers = _import_transformers()
    dev = resolve_device(device)
    root = Path(model_dir)
    index = SafetensorsIndex(root)
    config = transformers.AutoConfig.from_pretrained(root)
    with skeleton.parameters_on_meta(torch):
        model = transformers.AutoModelForCausalLM.from_config(
            config, dtype=skeleton.checkpoint_dtype(torch, index)
        )
    model.eval()
    with contextlib.suppress(OSError):  # a checkpoint without one keeps the defaults
        model.generation_config = transformers.GenerationConfig.from_pretrained(root)

    names = {id(module): name for name, module in model.named_modules()}
    layers = find_blocks(model)
    layers_name = names[id(layers)]
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if not isinstance(embedding, torch.nn.Embedding) or not isinstance(head, torch.nn.Linear):
        raise TypeError(
            f"{type(model).__name__} goes from {type(embedding).__name__} to "
            f"{type(head).__name__}; this adapter needs nn.Embedding and nn.Linear"
        )
    embedding_name, head_name = names[id(embedding)], names[id(head)]
    inside = layers_name + "."
    if embedding_name.startswith(inside) or head_name.startswith(inside):
        raise ValueError("the embedding or the output projection is inside the decoder layers")

    embedding_entry = skeleton.require_entry(index, f"{embedding_name}.weight")
    tied = head.weight is embedding.weight
    head_entry = index.get(f"{head_name}.weight")
    if head_entry is None:
        if not tied:
            skeleton.require_entry(index, f"{head_name}.weight")  # raises, naming the tensor
        head_entry = embedding_entry
    bias_entry = (
        None if head.bias is None else skeleton.require_entry(index, f"{head_name}.bias")
    )
    if head_entry.shape != tuple(head.weight.shape):
        raise ValueError(
            f"{head_entry.name!r} is {list(head_entry.shape)} in the checkpoint, but the "
            f"output projection is {list(head.weight.shape)}"
        )

    layer_bytes = max(
        sum(align_up(p.numel() * p.element_size()) for p in layer.parameters())
        for layer in layers
    )
    vocab = head_entry.shape[0]
    pieces = [entry for entry in (head_entry, bias_entry) if entry is not None]
    row_bytes = sum(entry.nbytes // max(vocab, 1) for entry in pieces)
    # Each piece is padded to the slot alignment: leave room for that inside a layer's size.
    rows = head_rows(vocab, max(row_bytes, 1), layer_bytes - ALIGN * len(pieces))
    chunked = _chunked_linear_type(torch)(
        head.in_features, tuple(rows), head.bias is not None, head.weight.dtype
    )
    sources: list[Any] = [f"{layers_name}.{i}." for i in range(len(layers))]
    for start, stop in rows:
        piece = {"weight": head_entry.rows(start, stop)}
        if bias_entry is not None:
            piece["bias"] = bias_entry.rows(start, stop)
        sources.append(piece)

    undo: list[Any] = []
    streamer = None
    try:
        table = skeleton.checked_map(torch, embedding_entry, embedding.weight)
        _look_up_on_cpu(embedding, skeleton.frozen(torch, table), dev, undo)

        parent = model.get_submodule(head_name.rpartition(".")[0])
        attribute = head_name.rpartition(".")[2]
        setattr(parent, attribute, chunked)
        undo.append(functools.partial(setattr, parent, attribute, head))

        skip = {id(m) for m in (*layers.modules(), *embedding.modules(), *chunked.modules())}
        resident_bytes = skeleton.make_resident(torch, model, index, dev, skip, undo)

        weights = FileWeights(index, sources, ram_budget=ram_budget, num_stage=num_stage)
        streamer = LayerStreamer(
            [*layers, *chunked.chunks], slots=slots, device=dev, weights=weights
        )
        # generate() sends its inputs to model.device, which would otherwise be the device
        # of the first parameter: the embedding, on the CPU.
        original = type(model)
        model.__class__ = skeleton.on_device_type(original, dev)
        undo.append(functools.partial(setattr, model, "__class__", original))
    except BaseException:
        if streamer is not None:
            streamer.close()
        while undo:
            undo.pop()()
        raise
    return StreamedModel(
        model, streamer, dev, rows, resident_bytes, embedding_entry.nbytes, undo
    )


def load_text_encoder(
    model_dir: str | PathLike[str],
    *,
    slots: int = 2,
    keep_in_ram: bool = False,
    num_stage: int = 2,
    device: torch.device | str = "cuda",
) -> StreamedModel:
    """Load an encoder-only text model (T5, umT5) from a local directory, streamed.

    This is the text stage of a diffusion pipeline. Call ``model(input_ids, attention_mask)``
    with both on ``device``, under ``torch.no_grad()`` or ``torch.inference_mode()``.

    Args:
        model_dir: a ``transformers`` model directory with safetensors weights. Nothing is
            downloaded.
        slots: VRAM buffers, each as large as the largest encoder block.
        keep_in_ram: where the blocks wait between calls. False reads every block from disk
            on each call, with only the staging buffers page-locked: right for an encoder
            that runs once per prompt, since nothing is read or locked that is not then used
            once. True keeps them all in page-locked RAM, which pays when many prompts are
            encoded. Measured on the umT5-XXL of Wan 2.2 (8.63 GiB of blocks, two prompts of
            512 tokens, reference laptop): from disk, 1.7 s to load and 3.4 s per call with
            0.72 GiB locked; in RAM, 9.8 s to load and 1.4-1.8 s per call with 8.63 GiB.
        num_stage: staging buffers for the blocks read from disk.
        device: the CUDA device that computes.

    Bytes moved per call: every encoder block to VRAM once, from disk or from RAM, plus the
    activations of the embedding. The length of the prompts does not change it, and neither
    does their number, so encoding prompts together costs one pass instead of one each.

    The prompts of a batch run one at a time inside each block, while its weights are in the
    slot (:func:`~hypertier.diffusion.one_sample_at_a_time`). A real batch does not round like
    a batch of one, so this is what gives each prompt the bits it gets alone, which are those
    of a pipeline that encodes its prompts one by one, as ``diffusers`` does.

    Limits:

    * Encoder-only models whose checkpoint names its tensors as the modules do, with an
      ``nn.Embedding`` going in. Anything else raises.
    * As for :func:`load_causal_lm`: the model is built with every ``Parameter`` of the
      process sent to ``meta`` meanwhile, and the limits of
      :class:`~hypertier.stream.LayerStreamer` apply.
    """
    torch = require_cuda()
    transformers = _import_transformers()
    dev = resolve_device(device)
    root = Path(model_dir)
    index = SafetensorsIndex(root)
    config = transformers.AutoConfig.from_pretrained(root)
    with skeleton.parameters_on_meta(torch):
        model = transformers.AutoModelForTextEncoding.from_config(
            config, dtype=skeleton.checkpoint_dtype(torch, index)
        )
    model.eval()

    names = {id(module): name for name, module in model.named_modules()}
    layers = find_blocks(model)
    layers_name = names[id(layers)]
    embedding = model.get_input_embeddings()
    # T5 looks tokens up through a second module that holds the same parameter: every module
    # that holds it gets the same table, and the checkpoint stores it under one of the names.
    holders = [
        module
        for module in model.modules()
        if any(param is embedding.weight for param in module._parameters.values())
    ]
    if not all(isinstance(module, torch.nn.Embedding) for module in holders):
        kinds = sorted({type(module).__name__ for module in holders})
        raise TypeError(
            f"the token embedding of {type(model).__name__} is shared with {kinds}; this "
            "adapter needs an encoder whose embedding is only looked up"
        )
    if any(names[id(module)].startswith(layers_name + ".") for module in holders):
        raise ValueError("the embedding is inside the encoder blocks")
    stored = [index.get(f"{names[id(module)]}.weight") for module in holders]
    embedding_entry = next((entry for entry in stored if entry is not None), None)
    if embedding_entry is None:
        skeleton.require_entry(index, f"{names[id(embedding)]}.weight")  # raises, naming it

    sources = [f"{layers_name}.{i}." for i in range(len(layers))]
    ram_budget = None
    if not keep_in_ram:
        # Exactly the staging buffers: no block fits beside them, so all come from disk.
        plans = FileWeights(index, sources).plan(torch, layers)
        ram_budget = num_stage * max(plan.host_bytes for plan in plans)

    undo: list[Any] = []
    streamer = None
    try:
        table = skeleton.frozen(
            torch, skeleton.checked_map(torch, embedding_entry, embedding.weight)
        )
        for module in holders:
            _look_up_on_cpu(module, table, dev, undo)
        skip = {id(m) for m in (*layers.modules(), *holders)}
        resident_bytes = skeleton.make_resident(torch, model, index, dev, skip, undo)
        weights = FileWeights(index, sources, ram_budget=ram_budget, num_stage=num_stage)
        streamer = LayerStreamer(layers, slots=slots, device=dev, weights=weights)
        # Whatever computes on activations: the blocks, and the modules with resident weights
        # (the final norm). T5 hands the position bias of the first block, with the mask of
        # each prompt in it, on to the others: it is cut per sample like the hidden states.
        accepted = inspect.signature(layers[0].forward).parameters
        in_block = [name for name in _BATCHED_IN_A_BLOCK if name in accepted]
        resident = [
            module
            for module in model.modules()
            if id(module) not in skip and any(p is not None for p in module._parameters.values())
        ]
        undo += [one_sample_at_a_time(layer, in_block) for layer in layers]
        undo += [one_sample_at_a_time(module, (0,)) for module in resident]
        original = type(model)
        model.__class__ = skeleton.on_device_type(original, dev)
        undo.append(functools.partial(setattr, model, "__class__", original))
    except BaseException:
        if streamer is not None:
            streamer.close()
        while undo:
            undo.pop()()
        raise
    return StreamedModel(model, streamer, dev, [], resident_bytes, embedding_entry.nbytes, undo)


# ---------------------------------------------------------------------------------------------


def _look_up_on_cpu(embedding: Any, table: Any, device: Any, undo: list[Any]) -> None:
    """Make ``embedding`` read its rows from ``table`` on the CPU and send them to ``device``.

    ``table`` is the embedding memory-mapped from the file: ids come down to it, activations
    go up, and the table itself never takes VRAM or page-locked RAM.
    """
    undo.append(functools.partial(skeleton.set_attribute, embedding, "weight", embedding.weight))
    skeleton.set_attribute(embedding, "weight", table)
    down = embedding.register_forward_pre_hook(skeleton.inputs_to_cpu, with_kwargs=True)
    up = embedding.register_forward_hook(functools.partial(skeleton.output_to, device))
    undo += [down.remove, up.remove]


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "this adapter needs Hugging Face transformers: pip install hypertier[hf]"
        ) from exc
    found = skeleton.version_tuple(transformers.__version__)
    if found < MIN_TRANSFORMERS:
        wanted = ".".join(map(str, MIN_TRANSFORMERS))
        raise RuntimeError(
            f"this adapter needs transformers >= {wanted}, found {transformers.__version__}: "
            "pip install -U hypertier[hf]"
        )
    return transformers


@functools.cache
def _chunked_linear_type(torch: Any) -> Any:
    nn = torch.nn

    class ChunkedLinear(nn.Module):
        """An ``nn.Linear`` cut along its output features into pieces that are streamed.

        Each piece is a block of the streamer: its weights are in VRAM only while it runs.
        """

        def __init__(
            self, in_features: int, rows: tuple[tuple[int, int], ...], bias: bool, dtype: Any
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = rows[-1][1] if rows else 0
            self.chunks = nn.ModuleList(
                nn.Linear(in_features, stop - start, bias=bias, device="meta", dtype=dtype)
                for start, stop in rows
            )
            self.requires_grad_(False)

        def forward(self, x: Any) -> Any:
            return torch.cat([chunk(x) for chunk in self.chunks], dim=-1)

    return ChunkedLinear


__all__ = [
    "MIN_TRANSFORMERS", "StreamedModel", "head_rows", "load_causal_lm", "load_text_encoder",
]  # fmt: skip
