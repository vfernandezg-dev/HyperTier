# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Reading weights from disk: what is where (``safetensors_index``) and how to fetch it
without the page cache (``direct``). Importing this package never imports PyTorch.
"""

from hypertier.io.direct import DirectReader, aligned_span
from hypertier.io.safetensors_index import Run, SafetensorsIndex, TensorEntry

__all__ = ["DirectReader", "Run", "SafetensorsIndex", "TensorEntry", "aligned_span"]
