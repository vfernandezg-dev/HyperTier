# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Just-in-time movement of weights between memory tiers, overlapped with compute.

``pinned`` is the page-locked RAM the weights are copied to VRAM from. Importing this package
never imports PyTorch.
"""

from hypertier.stream.pinned import PinnedPool

__all__ = ["PinnedPool"]
