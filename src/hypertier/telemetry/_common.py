# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the telemetry modules: units, lazy PyTorch import and /proc readers.

PyTorch is imported on first use, never at module import, so CI and CPU-only machines can
import and test everything that does not touch a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

KiB = 1 << 10
MiB = 1 << 20
GiB = 1 << 30


def format_bytes(n: float) -> str:
    """Binary-unit size for humans, e.g. ``1.5 GiB``."""
    for unit, scale in (("GiB", GiB), ("MiB", MiB), ("KiB", KiB)):
        if abs(n) >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{int(n)} B"


def import_torch() -> Any:
    """Return the ``torch`` module, with an actionable error when it is missing."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("this measurement needs PyTorch: pip install torch") from exc
    return torch


def cuda_available() -> bool:
    """True when PyTorch is installed and can see a CUDA device."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def require_cuda() -> Any:
    """Return ``torch``, or raise when no CUDA device is usable."""
    torch = import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available")
    return torch


def resolve_device(device: Any = None) -> Any:
    """``torch.device`` with an explicit index (default: the current CUDA device)."""
    torch = import_torch()
    dev = torch.device("cuda") if device is None else torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def read_meminfo() -> dict[str, int]:
    """``/proc/meminfo`` converted to bytes; empty where it does not exist (non-Linux)."""
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return {}
    info: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        fields = rest.split()
        if fields and fields[0].isdigit():
            scale = 1024 if len(fields) > 1 and fields[1] == "kB" else 1
            info[key] = int(fields[0]) * scale
    return info


def mem_available_bytes() -> int | None:
    """RAM the kernel can hand out without swapping (``MemAvailable``), if known."""
    return read_meminfo().get("MemAvailable")


def is_wsl() -> bool:
    """True inside Windows Subsystem for Linux."""
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False
