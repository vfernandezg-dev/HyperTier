# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Just-in-time movement of weights between memory tiers, overlapped with compute.

``pingpong`` streams a stack of blocks through a few VRAM slots; ``pinned`` is the page-locked
RAM the weights are copied from. Importing this package never imports PyTorch.
"""

from hypertier.stream.pingpong import LayerStreamer, PassStats, find_blocks
from hypertier.stream.pinned import PinnedPool

__all__ = ["LayerStreamer", "PassStats", "PinnedPool", "find_blocks"]
