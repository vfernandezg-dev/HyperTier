# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Bound the activations of a module by running its work in pieces.

Activations are never compressed in the compute path; their peak is bounded by splitting the
work, and the two ways to split are not equally exact:

* :func:`one_sample_at_a_time` runs a batch as one call per sample. Every call has the shapes
  of a batch of one, so it picks the kernels of a batch of one and gives its bits: a batch
  run this way **equals** one pass per sample, bit for bit.
* :func:`chunk_tokens` runs a per-token module over ranges of tokens. Other shapes pick other
  kernels, so the values are the same **up to rounding** (1 ulp of bfloat16 measured). A
  sampler amplifies that over its steps: judge it by the error of one pass, never by the
  image at the end.

Both replace ``forward`` on the instance, not on the class, and return what undoes it (undo
several on one module in reverse order). The weights stay where they are meanwhile: a block of a :class:`~hypertier.stream.LayerStreamer`
holds its slot for the whole call, so its weights are copied once however many pieces run.

Importing this module does not import PyTorch.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

_MARK = "one_sample_at_a_time"


def one_sample_at_a_time(module: Any) -> Callable[[], None]:
    """Make ``module`` run a batch as one call per sample; returns what undoes it.

    The batch is the first dimension of the first tensor the module is called with. Every
    tensor argument with that first dimension is cut, also inside tuples and lists; anything
    else (a table shared by the batch, a flag) is passed whole to every call. Outputs are
    tensors, or tuples and lists of them, that keep the batch dimension: they are joined along
    it. A batch of one is passed through untouched.

    Bytes moved per pass: unchanged. This is what lets one pass over the weights serve
    several samples (both branches of classifier-free guidance, two prompts) with the
    activations of one.
    """
    inner = module.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        size = _batch_size((*args, *kwargs.values()))
        if size is None or size <= 1:
            return inner(*args, **kwargs)
        joined = piece = None
        for number in range(size):
            piece = inner(
                *_sample(args, number, size),
                **{key: _sample(value, number, size) for key, value in kwargs.items()},
            )
            if joined is None:
                joined = _allocate(piece, size, module)
            _store(joined, piece, number)
        return _finish(joined, piece)

    setattr(forward, _MARK, True)
    return _replace_forward(module, forward)


def runs_one_sample_at_a_time(module: Any) -> bool:
    """Whether :func:`one_sample_at_a_time` is in effect on ``module``."""
    return bool(getattr(vars(module).get("forward"), _MARK, False))


def chunk_tokens(module: Any, tokens: int) -> Callable[[], None]:
    """Make ``module`` run over at most ``tokens`` tokens at a time; returns what undoes it.

    For a module that treats every token on its own (a feed-forward network): its first
    argument is ``[batch, tokens, ...]`` and so is its output. The other arguments are passed
    whole to every call. Not bit for bit the unchunked module: see the module docstring.

    Bytes moved per pass: unchanged.
    """
    if tokens < 1:
        raise ValueError(f"cannot run {tokens} tokens at a time")
    inner = module.forward

    def forward(x: Any, *args: Any, **kwargs: Any) -> Any:
        total = x.shape[1]
        if total <= tokens:
            return inner(x, *args, **kwargs)
        out = None
        for start in range(0, total, tokens):
            piece = inner(x[:, start : start + tokens], *args, **kwargs)
            if out is None:
                out = piece.new_empty((piece.shape[0], total, *piece.shape[2:]))
            out[:, start : start + tokens] = piece
        return out

    return _replace_forward(module, forward)


# ---------------------------------------------------------------------------------------------


def _replace_forward(module: Any, forward: Callable[..., Any]) -> Callable[[], None]:
    previous = vars(module).get("forward")  # None: the method of the class
    module.forward = forward
    return functools.partial(_restore_forward, module, forward, previous, [])


def _restore_forward(module: Any, forward: Any, previous: Any, done: list[bool]) -> None:
    if done:
        return
    if vars(module).get("forward") is not forward:
        raise RuntimeError(
            f"the forward of {type(module).__name__} was replaced again after this: undo the "
            "replacements in reverse order"
        )
    if previous is None:
        del module.forward
    else:
        module.forward = previous
    done.append(True)


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "new_empty") and hasattr(value, "shape")


def _batch_size(values: Any) -> int | None:
    for value in values:
        if _is_tensor(value):
            if value.ndim:
                return int(value.shape[0])
        elif type(value) in (tuple, list):
            found = _batch_size(value)
            if found is not None:
                return found
    return None


def _sample(value: Any, number: int, size: int) -> Any:
    if _is_tensor(value):
        whole = not value.ndim or value.shape[0] != size
        return value if whole else value[number : number + 1]
    if type(value) in (tuple, list):
        return type(value)(_sample(item, number, size) for item in value)
    return value


def _allocate(piece: Any, size: int, module: Any) -> Any:
    if _is_tensor(piece):
        if not piece.ndim or piece.shape[0] != 1:
            raise ValueError(
                f"{type(module).__name__} returned {list(piece.shape)} for one sample: to run "
                "one sample at a time, its outputs must keep the batch dimension"
            )
        return piece.new_empty((size, *piece.shape[1:]))
    if type(piece) in (tuple, list):
        return [_allocate(item, size, module) for item in piece]
    return None


def _store(joined: Any, piece: Any, number: int) -> None:
    if _is_tensor(piece):
        joined[number : number + 1] = piece
    elif type(piece) in (tuple, list):
        for target, item in zip(joined, piece):
            _store(target, item, number)


def _finish(joined: Any, piece: Any) -> Any:
    if _is_tensor(piece):
        return joined
    if type(piece) in (tuple, list):
        return type(piece)(_finish(target, item) for target, item in zip(joined, piece))
    return piece  # None, a number: the same for every sample


__all__ = ["chunk_tokens", "one_sample_at_a_time", "runs_one_sample_at_a_time"]
