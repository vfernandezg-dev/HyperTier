# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Measurements of the machine HyperTier runs on.

``bandwidth`` profiles the links between memory tiers (VRAM, pinned RAM, NVMe) and the GPU's
compute throughput: the measured inputs of the planner. ``vram`` accounts for the VRAM a
block of work uses. Importing this package never imports PyTorch.
"""

from hypertier.telemetry._common import (
    GiB,
    KiB,
    MiB,
    cuda_available,
    format_bytes,
    is_wsl,
    read_meminfo,
)

__all__ = ["GiB", "KiB", "MiB", "cuda_available", "format_bytes", "is_wsl", "read_meminfo"]
