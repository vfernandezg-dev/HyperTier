# CLAUDE.md — HyperTier

Guidance for Claude Code and other AI coding agents working in this repository.
Keep this file short and current: it is loaded into every agent session.

## Project in one paragraph

HyperTier is an open-source **tiered-memory inference runtime** (Python + PyTorch + OpenAI Triton).
Goal: run very large models (15 GB → 400 GB+) on consumer GPUs with **4 GB of VRAM** by treating
VRAM as the fastest tier of `VRAM ◄ pinned RAM ◄ NVMe` and moving weights, experts and caches just
in time, overlapped with compute. Modality-agnostic: LLMs/MoE, video & image diffusion (DiT), multimodal.
Reference hardware: **RTX 3050 Laptop (4 GB, Ampere sm_86) + 32 GB RAM + NVMe** (Lenovo Legion 5,
Ryzen 7 5800H, PCIe Gen3 x8), native Linux: Ubuntu 26.04, NVIDIA driver 595, system Python 3.14.
BIOS: hybrid GPU mode (desktop on the Radeon iGPU, PRIME on-demand) with a 1 GiB iGPU frame buffer,
so the RTX 3050 is free for compute. If VRAM free drops to ~2.8 GiB, the MUX was switched back to
discrete (e.g. by Lenovo Vantage in the Windows dual boot); fix it before benchmarking.
Measured there (`bench_hw`): VRAM free 3.57 GiB, H2D/D2H pinned ≈ 6.7 GB/s, NVMe O_DIRECT read
≈ 3.0 GB/s, pinned RAM up to 20.8 GiB, copy/compute overlap ≈ 1.0.

## Status

- Version `0.0.1` (placeholder). Current phase: **F0 — Foundations** (environment, hardware bandwidth
  probe, VRAM tracker, CI). No public API yet — do not invent one beyond the roadmap.
- Roadmap: F0 foundations → F1 ping-pong layer streaming + NVMe tier → F2 TurboQuant KV cache +
  Triton kernels → F3 video diffusion (sequential pipeline, chunking, VAE tiling) → F4 MoE expert
  cache + planner → F5 v0.1.0 release.
- Design docs live in `docs/` (Spanish, **local only, gitignored**): `arquitectura-y-roadmap.md`,
  `estado-del-arte-2025-2026.md`. Read them before architectural work; never commit them.

## Commands

```bash
# one-time setup on Linux (apt deps, .venv, torch+triton, verifies GPU, runs tests; detects WSL2)
bash scripts/setup_linux.sh
source .venv/bin/activate

pytest -q                       # tests
TRITON_INTERPRET=1 pytest -q    # run Triton kernels on CPU (no GPU needed)
python benchmarks/bench_hw.py   # hardware profile -> benchmarks/results/hw_*.json (--quick: smoke)
ruff check .                    # lint (line length 100, target py310)
python -m build                 # build sdist + wheel
```

No CUDA Toolkit is needed: PyTorch wheels ship their own CUDA libraries; only the NVIDIA driver is.

## Repository layout (target)

```
src/hypertier/
  api.py        load / load_pipeline / wrap  (F5)
  planner/      hardware profile, model inventory, placement, throughput predictor
  stream/       LayerStreamer (ping-pong), ExpertCache (MoE)
  io/           safetensors index, readinto → pinned buffers, async I/O
  quant/        Lloyd-Max codebooks, rotations, bit-plane packing, TurboQuant, weight formats
  kv/           TurboQuantKVCache
  diffusion/    sequential pipeline, token chunking, VAE tiling
  kernels/      triton/ + torch/ fallbacks
  backend/      triton | torch | torch.compile dispatch
  adapters/     Hugging Face transformers, diffusers
  telemetry/    VRAM, PCIe/NVMe bandwidth, overlap efficiency
tests/  benchmarks/  examples/  scripts/
```

Create modules only when the current phase needs them.

## Architecture decisions (do not silently reverse)

1. **The bottleneck is bandwidth, not VRAM.** `time ≥ max(compute, non_resident_bytes / link_bw)`.
   Every feature must state its effect on bytes moved per pass. The planner models a pass as
   `t ≈ max(c, x) + (1 − e)·min(c, x)` (`c` compute, `x` transfer, `e` copy/compute overlap
   efficiency). Take `e` from `bench_hw` on the target machine, never hard-code it:
   ≈ 0.95 on native Linux, ≈ 0.7 on WSL2 for copies ≥ 64 MiB (issue #1).
2. **KV cache = TurboQuant-MSE** (random rotation + Lloyd-Max codebook per coordinate, calibration-free).
   **QJL residual is OFF by default** (hurts softmax attention at equal bit budget; arXiv:2605.08114).
   Give keys more bits than values. Keep boundary layers and a recent-token window in fp16.
3. **Bit-plane packing** for any bit-width; attention scores computed on packed keys without
   de-rotating (`<q,k> = ‖k‖·<Rq, c[idx]>`); values accumulated in rotated basis, de-rotated once.
4. **Model-agnostic streaming via forward hooks** that repoint `param.data` to VRAM slot views.
   Do not rewrite model classes. Slots are assigned **LRU, never `i % num_slots`** (breaks with odd
   block counts + cyclic prefetch). CUDA events: `slot_ready` (compute waits copy),
   `slot_free` (copy waits compute).
5. **Weights stream as 4-bit / fp8 / lossless (DFloat11-style)**, never "2.5-bit lossless".
   Do not compress activations in the compute path; use chunking/tiling (exact) instead.
6. **Triton first, PyTorch fallback.** Kernels must also run under `TRITON_INTERPRET=1` and be
   tested bit-exact (or within tolerance) against a pure-PyTorch reference.
7. **Native Linux first.** Development and benchmarks run on native Linux only; WSL2 is no longer
   used for either. WSL2 stays **supported for users**: `setup_linux.sh` and telemetry detect it
   (`is_wsl()`), and `bench_hw` labels its results. Native Windows only via the PyTorch fallback.

## Coding conventions

- Python ≥ 3.10, type hints on public functions, docstrings explain *why*, not *what*.
- Every source file starts with:
  ```python
  # Copyright 2026 The HyperTier Authors
  # SPDX-License-Identifier: Apache-2.0
  ```
- No new runtime dependency without a reason in the PR; keep the placeholder install light.
- Each kernel or algorithm ships with: reference implementation, test against it, and a benchmark.
- Benchmarks must record hardware, driver, CUDA/torch/triton versions and config, and write
  results under `benchmarks/results/` (gitignored). Reproducibility matters: results may feed a paper.

## Git and contributions

- **Every commit is signed off (DCO):** `git commit -s -m "type: summary"`.
  Conventional prefixes: `feat`, `fix`, `perf`, `docs`, `test`, `chore`, `refactor`.
- Do not push, tag, publish to PyPI or force-push unless the maintainer asks explicitly.
- Never commit: model weights (`*.safetensors`, `*.gguf`, …), `docs/`, `.env`, `.pypirc`, tokens.
- Do not edit `LICENSE` (canonical Apache-2.0 text). Attribution goes in `NOTICE` / `AUTHORS`.
- Code copied or adapted from other projects must have a compatible license (MIT/BSD/Apache-2.0)
  and keep its attribution. Never redistribute model weights; users download them themselves.
- Use third-party names (NVIDIA, CUDA, Triton, Llama, Wan…) only to describe compatibility.

## Working with agents

- Before large changes: read the relevant design doc, propose a short plan, then implement.
- Prefer small, verifiable steps: implement → test → measure. Report numbers, not impressions.
- When a claim depends on hardware (bandwidth, VRAM, speed), measure it on the reference GPU or
  label it as an estimate.
- Use subagents for broad codebase searches or literature lookups; keep design decisions here.
- The maintainer communicates in **Spanish**; reply in Spanish. Code, comments, commit messages
  and public docs stay in **English**.
