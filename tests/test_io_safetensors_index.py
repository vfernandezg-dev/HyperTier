# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.io.safetensors_index (plain Python: they run everywhere)."""

import json
import struct

import pytest

from hypertier.io import Run, SafetensorsIndex, TensorEntry, runs_of
from hypertier.io.safetensors_index import MAX_HEADER_BYTES

MiB = 1 << 20


def blob(seed: int, nbytes: int) -> bytes:
    return bytes((seed + i) % 251 for i in range(nbytes))


def layer(index: int) -> dict:
    """Two tensors named like a transformer layer, 16 and 6 bytes."""
    return {
        f"layers.{index}.a.weight": ("F32", (2, 2), blob(index, 16)),
        f"layers.{index}.b.weight": ("BF16", (3,), blob(index + 100, 6)),
    }


def raw_file(path, header, data: bytes = b""):
    """A file with exactly this header: the way to write invalid ones."""
    raw = header if isinstance(header, bytes) else json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    return path


def test_entries_point_at_the_bytes_of_each_tensor(tmp_path, safetensors_writer):
    tensors = {**layer(0), "norm.weight": ("F16", (4,), blob(7, 8)), **layer(1)}
    path = safetensors_writer(tmp_path / "model.safetensors", tensors, metadata={"format": "pt"})
    index = SafetensorsIndex(path)
    assert index.files == [path]
    assert list(index) == list(tensors) and len(index) == 5
    assert "__metadata__" not in index
    assert index.nbytes == sum(len(data) for _, _, data in tensors.values())
    content = path.read_bytes()
    for name, (dtype, shape, data) in tensors.items():
        entry = index[name]
        assert entry == TensorEntry(name, dtype, shape, path, entry.offset, len(data))
        assert content[entry.offset : entry.offset + entry.nbytes] == data
    # Data starts on a multiple of 8, as the reference writer pads it; nothing more is promised.
    assert index["layers.0.a.weight"].offset % 8 == 0


def test_runs_merge_adjacent_tensors_and_split_at_gaps(tmp_path, safetensors_writer):
    tensors = {**layer(0), "norm.weight": ("F16", (4,), blob(7, 8)), **layer(1)}
    index = SafetensorsIndex(safetensors_writer(tmp_path / "m.safetensors", tensors))
    names = list(tensors)
    first = index[names[0]]

    (whole,) = index.runs(names)
    assert whole == Run(first.path, first.offset, index.nbytes, tuple(index[n] for n in names))
    (block,) = index.runs(n for n in reversed(names) if n.startswith("layers.1."))
    assert [e.name for e in block.entries] == ["layers.1.a.weight", "layers.1.b.weight"]
    assert block.nbytes == 22 and block.offset == index["layers.1.a.weight"].offset

    apart = [n for n in names if n != "norm.weight"]  # the 8 bytes of the norm are in between
    assert [run.nbytes for run in index.runs(apart)] == [22, 22]
    assert [run.nbytes for run in index.runs(apart, max_gap=7)] == [22, 22]
    (joined,) = index.runs(apart, max_gap=8)
    assert joined.nbytes == 22 + 8 + 22 and len(joined.entries) == 4

    assert index.runs([]) == []
    assert len(index.runs([names[0], names[0]])[0].entries) == 1
    with pytest.raises(KeyError):
        index.runs(["missing"])
    with pytest.raises(ValueError):
        index.runs(names, max_gap=-1)


def test_rows_of_a_tensor_are_a_contiguous_range_of_its_bytes(tmp_path, safetensors_writer):
    tensors = {
        "bias": ("F32", (5,), blob(1, 20)),
        "table": ("BF16", (10, 3, 2), blob(2, 120)),  # 12 bytes per row
        "scalar": ("F32", (), blob(3, 4)),
        "empty": ("F32", (0, 4), b""),
    }
    path = safetensors_writer(tmp_path / "m.safetensors", tensors)
    index = SafetensorsIndex(path)
    content = path.read_bytes()
    table = index["table"]

    part = table.rows(2, 7)
    assert part == TensorEntry("table[2:7]", "BF16", (5, 3, 2), path, table.offset + 24, 60)
    assert content[part.offset : part.offset + part.nbytes] == tensors["table"][2][24:84]
    assert table.rows(0, 10).nbytes == table.nbytes and table.rows(4, 4).nbytes == 0
    assert index["bias"].rows(1, 3).shape == (2,) and index["bias"].rows(1, 3).nbytes == 8
    assert index["empty"].rows(0, 0).nbytes == 0
    assert "table[2:7]" not in index, "a range of rows is not added to the index"
    for start, stop in [(-1, 2), (3, 2), (0, 11)]:
        with pytest.raises(ValueError, match="outside tensor"):
            table.rows(start, stop)
    with pytest.raises(ValueError, match="scalar"):
        index["scalar"].rows(0, 1)

    # Neighbouring ranges are one read; a range and the tensor it comes from, too.
    pieces = [table.rows(0, 4), table.rows(4, 10)]
    (run,) = runs_of(reversed(pieces))
    assert (run.offset, run.nbytes, run.entries) == (table.offset, 120, tuple(pieces))
    (run,) = runs_of([table.rows(1, 3), table])
    assert (run.offset, run.nbytes) == (table.offset, 120) and len(run.entries) == 2
    assert [r.nbytes for r in runs_of([table.rows(0, 2), table.rows(5, 6)])] == [24, 12]
    assert [r.nbytes for r in runs_of([table.rows(0, 2), table.rows(5, 6)], max_gap=36)] == [72]


def test_shards_are_found_through_the_weight_map(tmp_path, safetensors_writer):
    one = safetensors_writer(tmp_path / "model-00001-of-00002.safetensors", layer(0))
    two = safetensors_writer(
        tmp_path / "model-00002-of-00002.safetensors", {**layer(1), **layer(2)}
    )
    safetensors_writer(tmp_path / "adapter.safetensors", layer(0))  # not part of the model
    weight_map = {name: one.name for name in layer(0)}
    weight_map.update({name: two.name for name in {**layer(1), **layer(2)}})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    index = SafetensorsIndex(tmp_path)
    assert index.files == [one, two] and len(index) == 6
    assert index["layers.2.b.weight"].path == two
    # A block that straddles two shards needs one read per file.
    straddling = ["layers.0.b.weight", "layers.1.a.weight"]
    assert [(run.path, run.nbytes) for run in index.runs(straddling)] == [(one, 6), (two, 16)]


def test_directory_without_a_weight_map_uses_every_shard(tmp_path, safetensors_writer):
    safetensors_writer(tmp_path / "b.safetensors", layer(1))
    safetensors_writer(tmp_path / "a.safetensors", layer(0))
    index = SafetensorsIndex(tmp_path)
    assert [path.name for path in index.files] == ["a.safetensors", "b.safetensors"]
    assert len(index) == 4

    safetensors_writer(tmp_path / "c.safetensors", layer(0))
    with pytest.raises(ValueError, match="in both"):
        SafetensorsIndex(tmp_path)


def test_missing_sources_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        SafetensorsIndex(tmp_path / "nothing.safetensors")
    with pytest.raises(FileNotFoundError, match=r"no \*\.safetensors"):
        SafetensorsIndex(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    with pytest.raises(ValueError, match="weight_map"):
        SafetensorsIndex(tmp_path)


GOOD = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}


@pytest.mark.parametrize(
    ("header", "data", "message"),
    [
        (b"{not json", b"", "not valid JSON"),
        ([1, 2], b"", "not a JSON object"),
        ({"t": {**GOOD, "dtype": "F128"}}, bytes(8), "unknown dtype"),
        ({"t": {**GOOD, "shape": [3]}}, bytes(8), "needs 12"),
        ({"t": {**GOOD, "shape": [-2]}}, bytes(8), "invalid shape or offsets"),
        ({"t": {**GOOD, "shape": [True, 2]}}, bytes(8), "invalid shape or offsets"),
        ({"t": {**GOOD, "shape": "2"}}, bytes(8), "invalid shape or offsets"),
        ({"t": {**GOOD, "data_offsets": [8, 0]}}, bytes(8), "spans -8 bytes"),
        ({"t": {"dtype": "F32", "shape": [2]}}, bytes(8), "malformed entry"),
        ({"t": "F32"}, bytes(8), "malformed entry"),
        ({"t": GOOD}, bytes(7), "past the end of the file"),
        ({"t": GOOD, "u": {**GOOD, "data_offsets": [4, 12]}}, bytes(12), "overlap"),
    ],
)
def test_invalid_headers_are_rejected(tmp_path, header, data, message):
    path = raw_file(tmp_path / "bad.safetensors", header, data)
    with pytest.raises(ValueError, match=message):
        SafetensorsIndex(path)


def test_invalid_header_lengths_are_rejected(tmp_path):
    short = tmp_path / "short.safetensors"
    short.write_bytes(b"\x00" * 7)
    with pytest.raises(ValueError, match="too small"):
        SafetensorsIndex(short)
    for length in (MAX_HEADER_BYTES + 1, 64):  # absurd, and longer than the file itself
        path = tmp_path / f"len{length}.safetensors"
        path.write_bytes(struct.pack("<Q", length) + b"{}")
        with pytest.raises(ValueError, match="invalid header length"):
            SafetensorsIndex(path)


def test_holes_between_tensors_are_allowed(tmp_path):
    header = {"a": GOOD, "b": {**GOOD, "data_offsets": [16, 24]}}
    index = SafetensorsIndex(raw_file(tmp_path / "holes.safetensors", header, bytes(24)))
    assert index["b"].offset - index["a"].offset == 16
    assert len(index.runs(["a", "b"])) == 2


def test_real_checkpoint_layers_are_single_runs(models_dir):
    root = models_dir / "Qwen3-8B"
    if not root.is_dir():
        pytest.skip("Qwen3-8B is not downloaded")
    index = SafetensorsIndex(root)
    assert len(index) == 399 and len(index.files) == 5
    layers = [[n for n in index if n.startswith(f"model.layers.{i}.")] for i in range(36)]
    assert all(len(names) == 11 for names in layers)
    sizes = {sum(index[n].nbytes for n in names) for names in layers}
    assert sizes == {385_892_864}  # 368.02 MiB of bf16 per layer
    split = [i for i, names in enumerate(layers) if len(index.runs(names)) > 1]
    assert split == [7, 17, 27]  # the layers that straddle two shards
    assert all(index[n].nbytes % 256 == 0 for n in layers[0])
    assert index["lm_head.weight"].shape == (151936, 4096)
    assert {index[n].offset % 512 for n in layers[0]} != {0}, "offsets are not block-aligned"
