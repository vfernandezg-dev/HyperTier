# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Where every tensor of a safetensors checkpoint lives on disk, read from the headers only.

A safetensors file is an 8-byte little-endian header length, a JSON header and then the raw
tensor bytes. The header says which bytes belong to which tensor, so a streamer can read just
the block it needs, straight into page-locked memory, without ever loading the model: this
module is the inventory behind that (bytes per block, and which contiguous byte ranges to
read). It is plain Python, with no PyTorch and no ``safetensors`` dependency.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import pairwise
from math import prod
from os import PathLike
from pathlib import Path

MAX_HEADER_BYTES = 100 * 1024 * 1024
"""Largest header accepted, the limit of the reference implementation: a guard against
reading garbage as a length."""

DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8, "C64": 8,
}  # fmt: skip
"""Bytes per element of every safetensors dtype."""

SHARD_INDEX = "model.safetensors.index.json"


@dataclass(frozen=True)
class TensorEntry:
    """One tensor: its type and the bytes that hold it."""

    name: str
    dtype: str  # safetensors name, e.g. "BF16"
    shape: tuple[int, ...]
    path: Path
    offset: int  # of its first byte, from the start of ``path``
    nbytes: int

    def rows(self, start: int, stop: int) -> TensorEntry:
        """Rows ``start`` to ``stop`` (exclusive) along the first dimension, as an entry.

        Tensors are stored row-major, so a range of rows is a contiguous range of bytes. This
        is how a tensor larger than a VRAM slot is streamed in pieces, e.g. an output
        projection of vocabulary x hidden split by vocabulary.
        """
        if not self.shape:
            raise ValueError(f"tensor {self.name!r} is a scalar: it has no rows")
        if not 0 <= start <= stop <= self.shape[0]:
            raise ValueError(
                f"rows {start}:{stop} are outside tensor {self.name!r} of {self.shape[0]} rows"
            )
        row_bytes = self.nbytes // self.shape[0] if self.shape[0] else 0
        return TensorEntry(
            f"{self.name}[{start}:{stop}]",
            self.dtype,
            (stop - start, *self.shape[1:]),
            self.path,
            self.offset + start * row_bytes,
            (stop - start) * row_bytes,
        )


@dataclass(frozen=True)
class Run:
    """Tensors that sit next to each other in one file: a single read covers them all."""

    path: Path
    offset: int
    nbytes: int
    entries: tuple[TensorEntry, ...]


class SafetensorsIndex(Mapping[str, TensorEntry]):
    """Tensor name -> :class:`TensorEntry` for one file or a directory of shards.

    A directory is read through its ``model.safetensors.index.json`` when there is one (the
    Hugging Face layout), otherwise through every ``*.safetensors`` file in it. Only the
    headers are read. They are validated like the reference implementation does, because an
    offset taken on trust becomes a read of the wrong bytes into a model.
    """

    def __init__(self, source: str | PathLike[str]) -> None:
        root = Path(source)
        self.files = _shards(root)
        self._entries: dict[str, TensorEntry] = {}
        for path in self.files:
            for entry in _read_header(path):
                if entry.name in self._entries:
                    other = self._entries[entry.name].path.name
                    raise ValueError(f"tensor {entry.name!r} is in both {other} and {path.name}")
                self._entries[entry.name] = entry

    def __getitem__(self, name: str) -> TensorEntry:
        return self._entries[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def nbytes(self) -> int:
        """Bytes of tensor data in the checkpoint."""
        return sum(entry.nbytes for entry in self._entries.values())

    def runs(self, names: Iterable[str], *, max_gap: int = 0) -> list[Run]:
        """The fewest contiguous reads that cover ``names``, in file order.

        Tensors of one block are normally adjacent in the file, because safetensors stores
        them sorted by name, so a block is one run (two when it straddles shards). Bytes
        between two tensors, up to ``max_gap``, are read along rather than paying for another
        request.
        """
        return runs_of((self._entries[name] for name in names), max_gap=max_gap)


def runs_of(entries: Iterable[TensorEntry], *, max_gap: int = 0) -> list[Run]:
    """:meth:`SafetensorsIndex.runs` for entries, which may be row ranges of tensors."""
    if max_gap < 0:
        raise ValueError(f"max_gap must be >= 0, got {max_gap}")
    wanted = sorted(set(entries), key=_position)
    runs: list[Run] = []
    group: list[TensorEntry] = []
    end = 0
    for entry in wanted:
        if group and (entry.path != group[0].path or entry.offset - end > max_gap):
            runs.append(Run(group[0].path, group[0].offset, end - group[0].offset, tuple(group)))
            group = []
        end = max(end, entry.offset + entry.nbytes) if group else entry.offset + entry.nbytes
        group.append(entry)
    if group:
        runs.append(Run(group[0].path, group[0].offset, end - group[0].offset, tuple(group)))
    return runs


def _position(entry: TensorEntry) -> tuple[str, int, int]:
    return str(entry.path), entry.offset, entry.nbytes


def _shards(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"no safetensors file or directory at {root}")
    listing = root / SHARD_INDEX
    if listing.is_file():
        weight_map = json.loads(listing.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{listing} has no weight_map")
        files = sorted({root / str(name) for name in weight_map.values()})
    else:
        files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files in {root}")
    return files


def _read_header(path: Path) -> list[TensorEntry]:
    size = path.stat().st_size
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) < 8:
            raise ValueError(f"{path}: too small to be a safetensors file")
        (length,) = struct.unpack("<Q", prefix)
        if length > MAX_HEADER_BYTES or 8 + length > size:
            raise ValueError(f"{path}: invalid header length {length} for a file of {size} bytes")
        raw = f.read(length)
    try:
        header = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: the header is not valid JSON") from exc
    if not isinstance(header, dict):
        # ValueError like every other malformed file: it is the content that is wrong.
        raise ValueError(f"{path}: the header is not a JSON object")  # noqa: TRY004
    data_start = 8 + length
    entries = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        entries.append(_entry(path, name, info, data_start, size))
    entries.sort(key=lambda entry: entry.offset)
    for before, after in pairwise(entries):
        if before.offset + before.nbytes > after.offset:
            raise ValueError(f"{path}: tensors {before.name!r} and {after.name!r} overlap")
    return entries


def _entry(path: Path, name: str, info: object, data_start: int, size: int) -> TensorEntry:
    try:
        dtype, shape = info["dtype"], info["shape"]  # type: ignore[index]
        begin, end = info["data_offsets"]  # type: ignore[index]
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError(f"{path}: malformed entry for tensor {name!r}") from exc
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"{path}: tensor {name!r} has unknown dtype {dtype!r}")
    ints = (begin, end, *shape) if isinstance(shape, list) else (None,)
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in ints):
        raise ValueError(f"{path}: tensor {name!r} has an invalid shape or offsets")
    nbytes = end - begin
    if nbytes != prod(shape) * DTYPE_BYTES[dtype]:
        raise ValueError(
            f"{path}: tensor {name!r} spans {nbytes} bytes, but {dtype} {shape} needs "
            f"{prod(shape) * DTYPE_BYTES[dtype]}"
        )
    if data_start + end > size:
        raise ValueError(f"{path}: tensor {name!r} ends past the end of the file (truncated?)")
    return TensorEntry(name, dtype, tuple(shape), path, data_start + begin, nbytes)
