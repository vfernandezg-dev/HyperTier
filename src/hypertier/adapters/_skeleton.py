# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""What every adapter does the same way, whatever library the model comes from.

An adapter builds the model with its own classes and no weight in memory (a *skeleton*), finds
each tensor in the checkpoint by the name the modules give it, and then places the weights by
how they are used: streamed, resident on the GPU, or left in the file. The pieces that do not
depend on the model family live here, so the language-model adapter and the diffusion one
cannot drift apart on them.

Importing this module imports neither PyTorch nor any model library.
"""

from __future__ import annotations

import contextlib
import functools
import mmap
from collections.abc import Iterator
from math import prod
from typing import Any

from hypertier.io.safetensors_index import SafetensorsIndex, TensorEntry
from hypertier.stream.source import read_tensor, torch_dtype

_FLOATING = frozenset({"BF16", "F16", "F32", "F64"})


def version_tuple(text: str) -> tuple[int, ...]:
    """The leading numbers of a version string: ``"5.24.0.dev0"`` -> ``(5, 24, 0)``."""
    numbers = []
    for part in text.split("."):
        if not part.isdigit():
            break
        numbers.append(int(part))
    return tuple(numbers)


@contextlib.contextmanager
def parameters_on_meta(torch: Any) -> Iterator[None]:
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


def checkpoint_dtype(torch: Any, index: SafetensorsIndex) -> Any:
    """The floating dtype that holds most bytes of the checkpoint: what the model is built in."""
    nbytes: dict[str, int] = {}
    for entry in index.values():
        if entry.dtype in _FLOATING:
            nbytes[entry.dtype] = nbytes.get(entry.dtype, 0) + entry.nbytes
    if not nbytes:
        raise ValueError("the checkpoint has no floating-point tensors")
    return torch_dtype(torch, max(nbytes, key=nbytes.__getitem__))


def require_entry(index: SafetensorsIndex, name: str) -> TensorEntry:
    """The tensor called ``name``, or an error that says what the adapters expect."""
    entry = index.get(name)
    if entry is None:
        raise ValueError(
            f"tensor {name!r} is not in the checkpoint; this adapter needs a checkpoint that "
            "names its tensors as the model's modules do"
        )
    return entry


def frozen(torch: Any, tensor: Any) -> Any:
    return torch.nn.Parameter(tensor, requires_grad=False)


def set_attribute(module: Any, name: str, value: Any) -> None:
    """``setattr`` with a signature ``functools.partial`` can undo later."""
    setattr(module, name, value)


def mapped_tensor(torch: Any, entry: TensorEntry) -> Any:
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


def inputs_to_cpu(module: Any, args: Any, kwargs: Any) -> Any:
    """Forward pre-hook (``with_kwargs=True``) of a module whose weights stay on the CPU."""

    def down(value: Any) -> Any:
        return value.to("cpu") if hasattr(value, "device") else value

    return tuple(down(a) for a in args), {key: down(value) for key, value in kwargs.items()}


def output_to(device: Any, module: Any, args: Any, output: Any) -> Any:
    """Forward hook that sends the output of a CPU module up to ``device``."""
    return output.to(device)


def make_resident(
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
            entry = require_entry(index, dot + key)
            value = checked(torch, entry, param).to(device)
            set_attribute(module, key, frozen(torch, value))
            undo.append(functools.partial(set_attribute, module, key, param))
            nbytes += entry.nbytes
        for key, buf in list(module._buffers.items()):
            if buf is None:
                continue
            entry = None if key in module._non_persistent_buffers_set else index.get(dot + key)
            value = buf if entry is None else checked(torch, entry, buf)
            module._buffers[key] = value.to(device)
            undo.append(functools.partial(module._buffers.__setitem__, key, buf))
            nbytes += value.numel() * value.element_size()
    return nbytes


def check(torch: Any, entry: TensorEntry, like: Any) -> None:
    if entry.shape != tuple(like.shape) or torch_dtype(torch, entry.dtype) != like.dtype:
        raise ValueError(
            f"{entry.name!r} is {entry.dtype} {list(entry.shape)} in the checkpoint, but the "
            f"model expects {like.dtype} {list(like.shape)}"
        )


def checked(torch: Any, entry: TensorEntry, like: Any) -> Any:
    """The tensor of ``entry`` read into memory, after checking it is what the model expects."""
    check(torch, entry, like)
    return read_tensor(torch, entry)


def checked_map(torch: Any, entry: TensorEntry, like: Any) -> Any:
    """Like :func:`checked`, but memory-mapped: for a table too large to read up front."""
    check(torch, entry, like)
    return mapped_tensor(torch, entry)


@functools.cache
def on_device_type(cls: type, device: Any) -> type:
    """``cls`` with ``device`` fixed: a subclass made for the instances that are streamed.

    A model reports the device of its first parameter, which for a streamed model is a
    placeholder or a table kept on the CPU, while callers move their inputs to
    ``model.device``.
    """
    body = {
        "device": property(lambda self: device),
        "__module__": cls.__module__,
        "__qualname__": cls.__qualname__,
        "__doc__": cls.__doc__,
    }
    return type(cls.__name__, (cls,), body)
