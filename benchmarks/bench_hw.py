# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Profile this machine's memory tiers and GPU compute (HyperTier F0 hardware benchmark).

Measures RAM <-> VRAM bandwidth (pageable and pinned), copy/compute overlap, VRAM bandwidth,
fp16/bf16 matmul throughput, NVMe sequential reads into pinned memory and the largest pinned
allocation the OS allows. Everything, plus the metadata needed to reproduce the run, goes to
``benchmarks/results/hw_<UTC timestamp>.json``; a summary table is printed.

    python benchmarks/bench_hw.py            # full profile: ~2 min, ~4.3 GiB free in ~/dev
    python benchmarks/bench_hw.py --quick    # tiny sizes, smoke test
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import hypertier
from hypertier.telemetry import GiB, MiB, cuda_available, format_bytes, is_wsl, read_meminfo
from hypertier.telemetry import bandwidth as bw
from hypertier.telemetry.vram import track_vram

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hypertier.bench_hw/1"
# Order matters: the pinned probe runs last because it temporarily locks most of the RAM.
PHASES = ("matmul", "vram_bandwidth", "copy", "overlap", "nvme", "pinned_max")
GPU_PHASES = {"matmul", "vram_bandwidth", "copy", "overlap", "pinned_max"}
NVSMI_FIELDS = (
    "driver_version",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.gen.gpumax",
    "pcie.link.gen.hostmax",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "clocks.max.sm",
    "clocks.max.memory",
    "power.limit",
    "memory.total",
    "memory.used",
)


@dataclasses.dataclass(frozen=True)
class Config:
    """Every knob of a run; stored in the JSON so results can be reproduced."""

    copy_min_bytes: int = MiB
    copy_max_bytes: int = GiB
    copy_repeats: int = 20
    overlap_copy_bytes: int = 256 * MiB
    overlap_matmul_n: int = 4096
    matmul_sizes: tuple[int, ...] = (1024, 2048, 4096, 8192)
    matmul_repeats: int = 10
    device_copy_bytes: int = 256 * MiB
    nvme_dir: str = "~/dev"
    nvme_file_bytes: int = 4 * GiB
    nvme_chunks: tuple[int, ...] = (4 * MiB, 16 * MiB, 64 * MiB)
    pinned_cap_bytes: int = 16 * GiB
    pinned_first_block: int = 256 * MiB
    pinned_max_block: int = 2 * GiB
    pinned_min_block: int = 64 * MiB
    pinned_reserve_bytes: int | None = None
    warmup_seconds: float = 1.0


QUICK = Config(
    copy_max_bytes=8 * MiB,
    copy_repeats=5,
    overlap_copy_bytes=32 * MiB,
    overlap_matmul_n=1024,
    matmul_sizes=(512, 1024),
    matmul_repeats=5,
    device_copy_bytes=32 * MiB,
    nvme_file_bytes=64 * MiB,
    nvme_chunks=(4 * MiB, 16 * MiB),
    pinned_cap_bytes=512 * MiB,
    pinned_first_block=64 * MiB,
    pinned_max_block=256 * MiB,
    pinned_min_block=16 * MiB,
    warmup_seconds=0.3,
)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HyperTier hardware benchmark (F0).")
    parser.add_argument("--quick", action="store_true", help="tiny sizes: a fast smoke test")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "benchmarks" / "results",
        help="where the JSON goes (default: benchmarks/results/)",
    )
    parser.add_argument(
        "--nvme-dir", help="directory on the disk to test; must be a Linux path, not /mnt/ "
        "(default: ~/dev)",
    )
    parser.add_argument("--nvme-file-gib", type=float, help="NVMe test file size (default 4)")
    parser.add_argument("--max-copy-mib", type=int, help="largest RAM<->VRAM copy (default 1024)")
    parser.add_argument("--pinned-cap-gib", type=float, help="pinned probe cap (default 16)")
    parser.add_argument(
        "--pinned-reserve-gib", type=float,
        help="RAM the pinned probe leaves available (default: max(2 GiB, 15%% of RAM))",
    )
    parser.add_argument(
        "--skip", action="append", default=[], choices=PHASES, help="skip a phase (repeatable)"
    )
    return parser.parse_args(argv)


def make_config(args: argparse.Namespace) -> Config:
    cfg = QUICK if args.quick else Config()
    changes: dict[str, Any] = {}
    if args.nvme_dir:
        changes["nvme_dir"] = args.nvme_dir
    if args.nvme_file_gib is not None:
        step = max(cfg.nvme_chunks)
        changes["nvme_file_bytes"] = max(step, int(args.nvme_file_gib * GiB) // step * step)
    if args.max_copy_mib is not None:
        changes["copy_max_bytes"] = args.max_copy_mib * MiB
    if args.pinned_cap_gib is not None:
        changes["pinned_cap_bytes"] = int(args.pinned_cap_gib * GiB)
    if args.pinned_reserve_gib is not None:
        changes["pinned_reserve_bytes"] = int(args.pinned_reserve_gib * GiB)
    return dataclasses.replace(cfg, **changes)


# ---------------------------------------------------------------------------------------------
# Metadata


def _run(cmd: Sequence[str], timeout: float = 15.0) -> str | None:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def nvidia_smi(fields: Sequence[str] = NVSMI_FIELDS, index: int = 0) -> dict[str, str] | None:
    """Query nvidia-smi; fields it does not know are retried one by one and left out."""

    def query(names: Sequence[str]) -> list[str] | None:
        out = _run(
            ["nvidia-smi", f"--id={index}", f"--query-gpu={','.join(names)}",
             "--format=csv,noheader,nounits"]
        )
        if not out:
            return None
        values = [v.strip() for v in out.splitlines()[0].split(",")]
        return values if len(values) == len(names) else None

    values = query(fields)
    if values is not None:
        return dict(zip(fields, values))
    single = {name: query([name]) for name in fields}
    found = {name: v[0] for name, v in single.items() if v}
    return found or None


def _git_info() -> dict[str, Any] | None:
    commit = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    if commit is None:
        return None
    status = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"])
    return {"commit": commit, "dirty": bool(status)}


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _os_release() -> str | None:
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return None


def _triton_version() -> str | None:
    try:
        import triton
    except Exception:  # noqa: BLE001 - any import problem just means "not available"
        return None
    return triton.__version__


def collect_metadata() -> dict[str, Any]:
    """Hardware, driver and software versions needed to reproduce (or cite) a run.

    The hostname and GPU UUID are left out on purpose: results may be shared publicly.
    """
    meminfo = read_meminfo()
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypertier_version": hypertier.__version__,
        "git": _git_info(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "os_release": _os_release(),
        "wsl": is_wsl(),
        "cpu": {"model": _cpu_model(), "logical_cpus": os.cpu_count()},
        "ram": {
            "total_bytes": meminfo.get("MemTotal"),
            "available_bytes": meminfo.get("MemAvailable"),
            "swap_total_bytes": meminfo.get("SwapTotal"),
        },
        "env": {
            name: os.environ.get(name)
            for name in ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_DEVICE_ORDER")
        },
        "torch": None,
        "triton": None,
        "gpu": None,
    }
    try:
        import torch
    except ImportError:
        return meta
    matmul = torch.backends.cuda.matmul
    meta["torch"] = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "matmul_allow_tf32": matmul.allow_tf32,
        "matmul_fp16_reduced_precision_reduction": matmul.allow_fp16_reduced_precision_reduction,
        "matmul_bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
    }
    meta["triton"] = _triton_version()
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        meta["gpu"] = {
            "name": props.name,
            "capability": f"sm_{props.major}{props.minor}",
            "multiprocessors": props.multi_processor_count,
            "total_memory_bytes": total,
            "free_memory_bytes_at_start": free,
            "nvidia_smi": nvidia_smi(index=index),
        }
    return meta


def pcie_link_under_load(seconds: float = 1.0) -> dict[str, str] | None:
    """Read the PCIe link state while copies keep it busy (idle links train down)."""
    import torch

    found: dict[str, Any] = {}

    def query() -> None:
        time.sleep(seconds / 2)
        found["link"] = nvidia_smi(("pcie.link.gen.current", "pcie.link.width.current"))

    host = bw.host_buffer(64 * MiB, pinned=True)
    dst = torch.empty(64 * MiB, dtype=torch.uint8, device="cuda")
    thread = threading.Thread(target=query, daemon=True)
    thread.start()
    deadline = time.perf_counter() + seconds
    try:
        while thread.is_alive() or time.perf_counter() < deadline:
            dst.copy_(host, non_blocking=True)
            torch.cuda.synchronize()
    finally:
        thread.join()
        del host, dst
        bw.release_pinned_cache()
    return found.get("link")


# ---------------------------------------------------------------------------------------------
# Running phases


def _tilde(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home):] if path == home or path.startswith(home + os.sep) else path


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(item) for item in obj]
    return obj


def build_phases(cfg: Config) -> dict[str, Callable[[], Any]]:
    return {
        "matmul": partial(
            bw.measure_matmul_tflops, cfg.matmul_sizes, repeats=cfg.matmul_repeats
        ),
        "vram_bandwidth": partial(bw.measure_device_bandwidth, cfg.device_copy_bytes),
        "copy": partial(
            bw.measure_copy_bandwidth,
            bw.size_sweep(cfg.copy_min_bytes, cfg.copy_max_bytes),
            repeats=cfg.copy_repeats,
        ),
        "overlap": partial(
            bw.measure_copy_compute_overlap,
            cfg.overlap_copy_bytes,
            matmul_n=cfg.overlap_matmul_n,
        ),
        "nvme": partial(
            bw.measure_nvme_read,
            cfg.nvme_dir,
            file_bytes=cfg.nvme_file_bytes,
            chunk_sizes=cfg.nvme_chunks,
        ),
        "pinned_max": partial(
            bw.probe_max_pinned,
            cfg.pinned_cap_bytes,
            first_block=cfg.pinned_first_block,
            max_block=cfg.pinned_max_block,
            min_block=cfg.pinned_min_block,
            reserve_bytes=cfg.pinned_reserve_bytes,
        ),
    }


def run_phase(name: str, fn: Callable[[], Any], payload: dict[str, Any], gpu: bool) -> None:
    """Run one phase; a failure is recorded and does not lose the other phases."""
    print(f"[bench_hw] {name} ...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    try:
        if gpu:
            with track_vram() as tracker:
                result = fn()
            payload["vram"][name] = tracker.stats.as_dict()
        else:
            result = fn()
        payload["results"][name] = _jsonable(result)
    except Exception as exc:  # noqa: BLE001 - reported in the JSON and the summary
        payload["errors"][name] = f"{type(exc).__name__}: {exc}"
        print(f"[bench_hw] {name} FAILED: {payload['errors'][name]}", file=sys.stderr)
    payload["durations_s"][name] = round(time.perf_counter() - t0, 3)


def write_results(payload: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"hw_{stamp}.json"
    suffix = 1
    while path.exists():
        path = out_dir / f"hw_{stamp}_{suffix}.json"
        suffix += 1
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------------------------
# Summary table


def _fmt(value: Any, spec: str = ".2f") -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "-"


def format_summary(payload: dict[str, Any]) -> str:
    """Human-readable digest of a results payload; tolerates missing phases."""
    meta = payload.get("metadata") or {}
    res = payload.get("results") or {}
    gpu = meta.get("gpu") or {}
    smi = gpu.get("nvidia_smi") or {}
    tmeta = meta.get("torch") or {}
    ram = meta.get("ram") or {}
    cpu = meta.get("cpu") or {}
    out = ["", f"HyperTier hardware profile  {meta.get('timestamp_utc', '')}"]
    if gpu:
        out.append(
            f"GPU    {gpu.get('name')} ({gpu.get('capability')}, {gpu.get('multiprocessors')} SMs)"
            f", driver {smi.get('driver_version', '?')}, VRAM free "
            f"{format_bytes(gpu.get('free_memory_bytes_at_start', 0))} of "
            f"{format_bytes(gpu.get('total_memory_bytes', 0))}"
        )
        load = meta.get("pcie_link_under_load") or {}
        gen = load.get("pcie.link.gen.current", smi.get("pcie.link.gen.current", "?"))
        width = load.get("pcie.link.width.current", smi.get("pcie.link.width.current", "?"))
        out.append(
            f"PCIe   gen {gen} x{width} under load (GPU supports gen "
            f"{smi.get('pcie.link.gen.gpumax', '?')}, host/CPU gen "
            f"{smi.get('pcie.link.gen.hostmax', '?')})"
        )
    out.append(
        f"SW     torch {tmeta.get('version', '-')} (CUDA {tmeta.get('cuda', '-')}), "
        f"triton {meta.get('triton') or '-'}, Python {meta.get('python', '-')}"
    )
    ram_text = format_bytes(ram["total_bytes"]) if ram.get("total_bytes") else "?"
    out.append(
        f"Host   {cpu.get('model', '?')}, RAM {ram_text}{' (WSL2 VM)' if meta.get('wsl') else ''}"
    )

    copy = res.get("copy")
    if copy:
        columns = [("h2d", "pinned"), ("h2d", "pageable"), ("d2h", "pinned"), ("d2h", "pageable")]
        out += ["", "RAM <-> VRAM copies, GB/s, median / p90 (CUDA events)"]
        out.append(f"{'size':>10}" + "".join(f"{d.upper() + ' ' + m:>19}" for d, m in columns))
        points = {(r["direction"], r["memory"], r["size_bytes"]): r for r in copy}
        for size in sorted({r["size_bytes"] for r in copy}):
            cells = []
            for d, m in columns:
                r = points.get((d, m, size))
                cells.append(f"{_fmt(r['gbps_median'])} / {_fmt(r['gbps_p90'])}" if r else "-")
            out.append(f"{format_bytes(size):>10}" + "".join(f"{c:>19}" for c in cells))

    ov = res.get("overlap")
    if ov:
        out += [
            "",
            (
                f"Overlap  H2D {format_bytes(ov['copy_bytes'])} on its own stream "
                f"({_fmt(ov['copy_ms'], '.1f')} ms) || {ov['matmul_iters']} x matmul "
                f"{ov['matmul_n']} {ov['matmul_dtype']} ({_fmt(ov['compute_ms'], '.1f')} ms)"
                f" -> together {_fmt(ov['both_ms'], '.1f')} ms: efficiency "
                f"{_fmt(100 * ov['efficiency'], '.0f')} %, speedup {_fmt(ov['speedup'])}x"
            ),
        ]

    vbw = res.get("vram_bandwidth")
    if vbw:
        out.append(
            f"VRAM     {_fmt(vbw['gbps_median'], '.1f')} GB/s median, "
            f"{_fmt(vbw['gbps_p90'], '.1f')} p90 (device copy, read + write)"
        )

    mm = res.get("matmul")
    if mm:
        sizes = sorted({r["n"] for r in mm})
        out += ["", "Matmul TFLOPS, median (p90)" + "".join(f"{f'n={n}':>16}" for n in sizes)]
        for dtype in sorted({r["dtype"] for r in mm}):
            row = {r["n"]: r for r in mm if r["dtype"] == dtype}
            cells = [
                f"{_fmt(row[n]['tflops_median'])} ({_fmt(row[n]['tflops_p90'])})"
                if n in row
                else "-"
                for n in sizes
            ]
            out.append(f"{dtype:>27}" + "".join(f"{c:>16}" for c in cells))

    nv = res.get("nvme")
    if nv:
        reads = " | ".join(
            f"{format_bytes(r['chunk_bytes'])}: {_fmt(r['gbps'])} GB/s" for r in nv["reads"]
        )
        methods = sorted({r["method"] for r in nv["reads"]})
        out += [
            "",
            (
                f"NVMe     {nv['directory']} ({nv['filesystem']}), "
                f"{format_bytes(nv['file_bytes'])} file, read via {'/'.join(methods)} into "
                f"{'pinned' if nv['pinned_buffer'] else 'pageable'} buffer:"
            ),
            f"         {reads}",
            f"         write {_fmt(nv['write_gbps'])} GB/s ({nv['write_method']})",
        ]

    pin = res.get("pinned_max")
    if pin:
        avail = [
            format_bytes(v) if v is not None else "?"
            for v in (pin["mem_available_before"], pin["mem_available_after_release"])
        ]
        out += [
            "",
            (
                f"Pinned   max held {format_bytes(pin['max_pinned_bytes'])} "
                f"(stop: {pin['stop_reason']}; cap {format_bytes(pin['cap_bytes'])}, "
                f"reserve {format_bytes(pin['reserve_bytes'])}), pinning "
                f"{_fmt(pin['pin_gbps'])} GB/s; MemAvailable {avail[0]} -> {avail[1]} "
                "after release"
            ),
        ]
        if pin.get("first_error"):
            out.append(f"         first error: {pin['first_error']}")

    vram = payload.get("vram") or {}
    if vram:
        peaks = ", ".join(f"{k} +{format_bytes(v['peak_delta'])}" for k, v in vram.items())
        lowest = min(v["min_free_observed"] for v in vram.values())
        out += ["", f"VRAM peak per phase: {peaks}; lowest free seen {format_bytes(lowest)}"]

    errors = payload.get("errors") or {}
    if errors:
        out += ["", "Errors:"] + [f"  {k}: {v}" for k, v in errors.items()]
    skipped = payload.get("skipped") or []
    if skipped:
        out.append(f"Skipped: {', '.join(skipped)}")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = make_config(args)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "metadata": collect_metadata(),
        "config": dataclasses.asdict(cfg),
        "results": {},
        "vram": {},
        "errors": {},
        "skipped": [],
        "durations_s": {},
    }
    gpu = cuda_available()
    if gpu:
        bw.warm_up_gpu(cfg.warmup_seconds)
        payload["metadata"]["pcie_link_under_load"] = pcie_link_under_load()
    phases = build_phases(cfg)
    for name in PHASES:
        if name in args.skip or (name in GPU_PHASES and not gpu):
            payload["skipped"].append(name)
            continue
        run_phase(name, phases[name], payload, gpu)
    nvme = payload["results"].get("nvme")
    if nvme:  # keep the user name out of results that may be published
        nvme["directory"] = _tilde(nvme["directory"])
    path = write_results(payload, args.out_dir)
    print(format_summary(payload))
    print(f"\nResults written to {path}")
    return 1 if payload["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
