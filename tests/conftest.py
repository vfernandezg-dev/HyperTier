# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Skip tests marked ``cuda`` when no GPU is usable (CI, CPU-only boxes, missing PyTorch)."""

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
