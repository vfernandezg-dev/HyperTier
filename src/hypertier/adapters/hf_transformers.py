# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Hugging Face ``transformers`` causal language model that does not fit in VRAM.

:func:`load_causal_lm` returns the ordinary ``transformers`` model, to be called or driven by
``generate()`` as usual, with its weights placed by size and by how they are used::

    token embedding   stays on the CPU, memory-mapped from the checkpoint: a lookup touches
                      one row per token, so only the activations go up to the GPU
    decoder layers    streamed through the VRAM slots (``LayerStreamer``), one block each
    output projection streamed too, split by vocabulary rows into pieces no larger than a
                      layer; whole, it would make every slot three times as big
    everything else   (final norm, rotary tables...) resident on the GPU: kilobytes

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
import mmap
from collections.abc import Iterator
from itertools import pairwise
from math import prod
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hypertier.io.safetensors_index import SafetensorsIndex, TensorEntry
from hypertier.stream.pingpong import LayerStreamer, find_blocks
from hypertier.stream.pinned import ALIGN, align_up
from hypertier.stream.source import FileWeights, read_tensor, torch_dtype
from hypertier.telemetry._common import format_bytes, require_cuda, resolve_device

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self

MIN_TRANSFORMERS = (5, 24)
"""Oldest ``transformers`` this adapter is tested with; the ``[hf]`` extra pins the same."""

_FLOATING = frozenset({"BF16", "F16", "F32", "F64"})


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
        head_rows: the vocabulary ranges the output projection is streamed in.
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
    with _parameters_on_meta(torch):
        model = transformers.AutoModelForCausalLM.from_config(
            config, dtype=_checkpoint_dtype(torch, index)
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

    embedding_entry = _entry(index, f"{embedding_name}.weight")
    tied = head.weight is embedding.weight
    head_entry = index.get(f"{head_name}.weight")
    if head_entry is None:
        if not tied:
            _entry(index, f"{head_name}.weight")  # raises, naming the tensor
        head_entry = embedding_entry
    bias_entry = None if head.bias is None else _entry(index, f"{head_name}.bias")
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
        # The embedding reads its rows from the file; ids come down to it, activations go up.
        undo.append(functools.partial(_set, embedding, "weight", embedding.weight))
        table = _checked_map(torch, embedding_entry, embedding.weight)
        _set(embedding, "weight", _frozen(torch, table))
        down = embedding.register_forward_pre_hook(_inputs_to_cpu, with_kwargs=True)
        up = embedding.register_forward_hook(functools.partial(_output_to, dev))
        undo += [down.remove, up.remove]

        parent = model.get_submodule(head_name.rpartition(".")[0])
        attribute = head_name.rpartition(".")[2]
        setattr(parent, attribute, chunked)
        undo.append(functools.partial(setattr, parent, attribute, head))

        skip = {id(m) for m in (*layers.modules(), *embedding.modules(), *chunked.modules())}
        resident_bytes = _make_resident(torch, model, index, dev, skip, undo)

        weights = FileWeights(index, sources, ram_budget=ram_budget, num_stage=num_stage)
        streamer = LayerStreamer(
            [*layers, *chunked.chunks], slots=slots, device=dev, weights=weights
        )
        # generate() sends its inputs to model.device, which would otherwise be the device
        # of the first parameter: the embedding, on the CPU.
        original = type(model)
        model.__class__ = _on_device_type(original, dev)
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


# ---------------------------------------------------------------------------------------------


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "this adapter needs Hugging Face transformers: pip install hypertier[hf]"
        ) from exc
    found = _version(transformers.__version__)
    if found < MIN_TRANSFORMERS:
        wanted = ".".join(map(str, MIN_TRANSFORMERS))
        raise RuntimeError(
            f"this adapter needs transformers >= {wanted}, found {transformers.__version__}: "
            "pip install -U hypertier[hf]"
        )
    return transformers


def _version(text: str) -> tuple[int, ...]:
    """The leading numbers of a version string: ``"5.24.0.dev0"`` -> ``(5, 24, 0)``."""
    numbers = []
    for part in text.split("."):
        if not part.isdigit():
            break
        numbers.append(int(part))
    return tuple(numbers)


@contextlib.contextmanager
def _parameters_on_meta(torch: Any) -> Iterator[None]:
    """Send every ``Parameter`` registered while this lasts to ``meta``: shape, no memory.

    Unlike building under ``torch.device("meta")``, buffers are left alone, so the tables a
    model computes in ``__init__`` hold real values, computed on the CPU as in a normal load
    (computed on the GPU they differ in the last bits). Weight initialisation runs on ``meta``
    tensors, where it costs nothing.
    """

    def to_meta(module: Any, name: str, param: Any) -> Any:
        if param is None or param.device.type == "meta":
            return None  # includes tied weights: they stay one object
        empty = torch.empty(param.shape, dtype=param.dtype, device="meta")
        return torch.nn.Parameter(empty, requires_grad=False)

    handle = torch.nn.modules.module.register_module_parameter_registration_hook(to_meta)
    try:
        yield
    finally:
        handle.remove()


def _checkpoint_dtype(torch: Any, index: SafetensorsIndex) -> Any:
    """The floating dtype that holds most bytes of the checkpoint: what the model is built in."""
    nbytes: dict[str, int] = {}
    for entry in index.values():
        if entry.dtype in _FLOATING:
            nbytes[entry.dtype] = nbytes.get(entry.dtype, 0) + entry.nbytes
    if not nbytes:
        raise ValueError("the checkpoint has no floating-point tensors")
    return torch_dtype(torch, max(nbytes, key=nbytes.__getitem__))


def _entry(index: SafetensorsIndex, name: str) -> TensorEntry:
    entry = index.get(name)
    if entry is None:
        raise ValueError(
            f"tensor {name!r} is not in the checkpoint; this adapter needs a checkpoint that "
            "names its tensors as the model's modules do"
        )
    return entry


def _frozen(torch: Any, tensor: Any) -> Any:
    return torch.nn.Parameter(tensor, requires_grad=False)


def _set(module: Any, name: str, value: Any) -> None:
    setattr(module, name, value)


def _mapped_tensor(torch: Any, entry: TensorEntry) -> Any:
    """A CPU tensor over the bytes of ``entry`` in its file; pages are read when touched.

    The mapping is private and writable, because PyTorch has no read-only tensors: a stray
    write would change this process's copy of a page, never the file.
    """
    dtype = torch_dtype(torch, entry.dtype)
    if entry.nbytes == 0:
        return torch.empty(entry.shape, dtype=dtype)
    start = entry.offset - entry.offset % mmap.ALLOCATIONGRANULARITY
    with open(entry.path, "rb") as f:
        mapping = mmap.mmap(
            f.fileno(), entry.offset - start + entry.nbytes, access=mmap.ACCESS_COPY, offset=start
        )
    # The tensor keeps the mapping alive; it is never closed by hand.
    flat = torch.frombuffer(
        mapping, dtype=dtype, count=prod(entry.shape), offset=entry.offset - start
    )
    return flat.view(entry.shape)


def _inputs_to_cpu(module: Any, args: Any, kwargs: Any) -> Any:
    def down(value: Any) -> Any:
        return value.to("cpu") if hasattr(value, "device") else value

    return tuple(down(a) for a in args), {key: down(value) for key, value in kwargs.items()}


def _output_to(device: Any, module: Any, args: Any, output: Any) -> Any:
    return output.to(device)


def _make_resident(
    torch: Any, model: Any, index: SafetensorsIndex, device: Any, skip: set[int], undo: list[Any]
) -> int:
    """Put on the GPU every parameter and buffer outside the modules in ``skip``.

    Parameters come from the checkpoint. So do the buffers stored in it; the others keep the
    value the model's ``__init__`` computed on the CPU. Returns the bytes now on the device.
    """
    nbytes = 0
    for prefix, module in model.named_modules():
        if id(module) in skip:
            continue
        dot = prefix + "." if prefix else ""
        for key, param in list(module._parameters.items()):
            if param is None:
                continue
            entry = _entry(index, dot + key)
            value = _checked(torch, entry, param).to(device)
            _set(module, key, _frozen(torch, value))
            undo.append(functools.partial(_set, module, key, param))
            nbytes += entry.nbytes
        for key, buf in list(module._buffers.items()):
            if buf is None:
                continue
            entry = None if key in module._non_persistent_buffers_set else index.get(dot + key)
            value = buf if entry is None else _checked(torch, entry, buf)
            module._buffers[key] = value.to(device)
            undo.append(functools.partial(module._buffers.__setitem__, key, buf))
            nbytes += value.numel() * value.element_size()
    return nbytes


def _check(torch: Any, entry: TensorEntry, like: Any) -> None:
    if entry.shape != tuple(like.shape) or torch_dtype(torch, entry.dtype) != like.dtype:
        raise ValueError(
            f"{entry.name!r} is {entry.dtype} {list(entry.shape)} in the checkpoint, but the "
            f"model expects {like.dtype} {list(like.shape)}"
        )


def _checked(torch: Any, entry: TensorEntry, like: Any) -> Any:
    """The tensor of ``entry`` read into memory, after checking it is what the model expects."""
    _check(torch, entry, like)
    return read_tensor(torch, entry)


def _checked_map(torch: Any, entry: TensorEntry, like: Any) -> Any:
    _check(torch, entry, like)
    return _mapped_tensor(torch, entry)


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


@functools.cache
def _on_device_type(cls: type, device: Any) -> type:
    """``cls`` with ``device`` fixed: a subclass made for the instances that are streamed."""
    body = {
        "device": property(lambda self: device),
        "__module__": cls.__module__,
        "__qualname__": cls.__qualname__,
        "__doc__": cls.__doc__,
    }
    return type(cls.__name__, (cls,), body)


__all__ = ["MIN_TRANSFORMERS", "StreamedModel", "head_rows", "load_causal_lm"]
