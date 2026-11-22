# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Just-in-time movement of weights between memory tiers, overlapped with compute.

``pingpong`` streams a stack of blocks through a few VRAM slots; ``pinned`` is the page-locked
RAM the weights are copied from; ``source`` reads them from safetensors files, keeping in RAM
what fits and streaming the rest from disk. Importing this package never imports PyTorch.
"""

from hypertier.stream.pingpong import LayerStreamer, PassStats, find_blocks
from hypertier.stream.pinned import PinnedPool
from hypertier.stream.source import FileWeights

__all__ = ["FileWeights", "LayerStreamer", "PassStats", "PinnedPool", "find_blocks"]
