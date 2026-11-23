# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Skip tests marked ``cuda`` when no GPU is usable (CI, CPU-only boxes, missing PyTorch).

Also the fixtures shared by the weight-file tests: a minimal safetensors writer (no dependency
on the ``safetensors`` package) and the directory of real checkpoints, when there is one.
"""

import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _cuda_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)


def write_safetensors(path: Path, tensors: dict, *, pad_to: int = 8, metadata=None) -> Path:
    """Write ``tensors`` (name -> (dtype, shape, bytes)) in the order given, like safetensors.

    The header is padded with spaces so the data starts on a multiple of ``pad_to``, which is
    what the reference writer does with 8; tensor offsets then land wherever they land.
    """
    header: dict = {}
    blobs = []
    offset = 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]
        }
        blobs.append(bytes(data))
        offset += len(data)
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-(8 + len(raw)) % pad_to)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    return path


@pytest.fixture
def safetensors_writer():
    return write_safetensors


@pytest.fixture
def disk_tmp(request, tmp_path):
    """A scratch directory on the repository's disk.

    ``tmp_path`` is often tmpfs, where ``O_DIRECT`` means nothing and reads never touch a
    drive; pytest's cache directory lives next to the code instead.
    """
    cache = getattr(request.config, "cache", None)
    if cache is None:
        yield tmp_path
        return
    path = Path(tempfile.mkdtemp(dir=cache.mkdir("hypertier-disk")))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def models_dir() -> Path:
    """Directory of real checkpoints (``HYPERTIER_MODELS``); tests that need it skip without.

    Such tests are marked ``slow``, which leaves them out of a plain ``pytest``.
    """
    root = os.environ.get("HYPERTIER_MODELS")
    if not root or not Path(root).is_dir():
        pytest.skip("set HYPERTIER_MODELS to a directory of downloaded models")
    return Path(root)
