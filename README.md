# HyperTier

**Tiered-memory inference runtime: run 400 GB AI models on 4 GB GPUs — layer streaming, KV-cache compression, Triton kernels.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-early%20development-orange.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

> ⚠️ **Early development.** The public API does not exist yet. Follow the roadmap below or watch the repo for the first release (v0.1.0).

---

## What is HyperTier?

HyperTier is an open-source, modality-agnostic inference engine and **VRAM virtualizer** written in Python, PyTorch and OpenAI Triton. Its goal is to run very large models — from 15 GB to 400 GB+ — on consumer GPUs with as little as **4 GB of VRAM** (reference hardware: NVIDIA RTX 3050 Laptop, 32 GB RAM, NVMe SSD), cutting the *active* VRAM footprint by 95–99 % without destroying model accuracy.

It treats GPU memory as the fastest tier of a hierarchy:

```
VRAM (fast, tiny)  ◄──  pinned RAM (medium)  ◄──  NVMe SSD (large, slow)
```

and moves weights, experts and caches between tiers just in time, overlapping every transfer with computation.

### Target workloads

1. **LLMs and giant MoE models** — text, code, reasoning and decision-making agents.
2. **Video and image diffusion / DiT models** — Wan 2.x, HunyuanVideo, LTX-Video, Flux.
3. **Multimodal and task-execution models.**

## The four pillars

| Pillar | What it does |
|---|---|
| **1. Online KV-cache compression** | TurboQuant-style vector quantization (random rotation + Lloyd-Max codebooks) to ~2.5–3.5 bits per value, calibration-free, with fused Triton kernels that compute attention directly on compressed data. |
| **2. Asynchronous layer & expert streaming** | Ping-pong double buffering over dedicated CUDA streams: block *i* computes while block *i+1* is copied from pinned RAM or NVMe (`safetensors`), plus an LRU expert cache for MoE models. |
| **3. Sequential slicing & VAE tiling** | Text encoder → DiT → VAE run as separate stages; token-chunked FFN/attention and spatio-temporal VAE tiling remove out-of-memory peaks in video generation. |
| **4. Portability** | `@triton.jit` kernels for NVIDIA / AMD / Intel GPUs with no C++ build step, and a vectorized PyTorch (`torch.compile`) fallback everywhere else. |

A **bandwidth-aware planner** decides what lives in each tier, based on the measured PCIe/NVMe bandwidth of the machine, and predicts throughput before running.

## Honest expectations

Streaming makes models *fit*; the speed is then bounded by the slowest link the active weights must cross:

```
time_per_pass ≥ max(compute, non_resident_active_bytes / link_bandwidth)
```

- **Diffusion / video DiT:** best case — compute per block hides the transfers.
- **MoE models:** practical, thanks to expert caching (only active experts move).
- **Dense 400 GB models:** they run, but at minutes per token. Suitable for offline/batch agents, not interactive chat.

## Roadmap (v0.1.0 MVP)

| Phase | Deliverable | Status |
|---|---|---|
| F0 | Foundations: CI (CPU + Triton interpreter), hardware bandwidth probe, VRAM tracker | done |
| F1 | Ping-pong layer streaming + NVMe tier + Hugging Face adapter | done |
| F2 | TurboQuant KV cache + Triton kernels | third: after F4 |
| F3 | Video diffusion: sequential pipeline, chunking, VAE tiling (Wan 2.2 TI2V-5B) | first: in progress |
| F4 | MoE expert cache + planner v0 | second: after F3 |
| F5 | `pip install hypertier` v0.1.0 with reproducible examples on a 4 GB GPU | last |

The phase numbers are names, not the order. After F1 the order is **F3 → F4 → F2**, then F5:

- **F3 first.** Video diffusion uses no KV cache, and it is where streaming pays most: a
  denoising step runs thousands of tokens through every block, so the copies hide behind
  compute.
- **F4 next.** The planner decides on which device each block is computed in each phase:
  resident on the GPU, on the CPU, or streamed.
- **F2 last.** Where the KV cache lives depends on where attention is computed, and that is
  what F4 decides.

Design documents (architecture, bandwidth model and research survey): **coming soon.**

## Results so far

F0 and F1 are done. Qwen3-8B in bf16 (15.3 GiB of weights) on the reference laptop (RTX 3050
Laptop with 4 GB of VRAM, 32 GB of RAM), measured with `benchmarks/bench_llm.py`: two VRAM
slots, streamed weights in page-locked RAM, each run in its own process.

| | HyperTier, streamed | `accelerate` offloading | CPU only |
|---|---|---|---|
| 32 greedy tokens | identical to the baseline, logits bit for bit | the same tokens | differ from the first one |
| Peak VRAM while generating | 775 MiB (736 MiB of weights) | 2415 MiB | none |
| Decoding, seconds per token | 2.28 | 2.88 | 0.45 |
| Prefill of 512 tokens | 2.3 s | not measured | 81 s |
| Prefill of 4096 tokens | 4.9 s, 1.7 GiB of VRAM at its peak | not measured | not measured |

The baseline is the stock `transformers` model doing the same operations, with each layer
copied to the GPU right before it runs.

Decoding a dense model token by token is bound by the link, as the expectations above say:
every token moves 14.1 GiB over PCIe, while the CPU reads the same weights from RAM several
times faster. Streaming pays when many tokens share a pass. Choosing where each block is
computed, per phase, is the planner's job (F4).

## Installation

```bash
pip install hypertier   # placeholder release, no functionality yet
```

From source:

```bash
git clone https://github.com/vfernandezg-dev/HyperTier.git
cd HyperTier
pip install -e ".[dev]"
pytest
```

## Contributing

Contributions, ideas and benchmarks on low-VRAM hardware are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

Please report vulnerabilities privately — see [SECURITY.md](SECURITY.md). Do not open public issues for security problems.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

HyperTier builds on published research (TurboQuant, PolarQuant, and others listed in the docs). NVIDIA, CUDA, Triton and other names are trademarks of their respective owners and are used here only to describe compatibility.
