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

- Version `0.0.1` (placeholder). **F0 and F1 are done.** F1 landed `stream/pingpong.py`
  (`LayerStreamer` core, LRU `SlotTable`), `stream/pinned.py` (`PinnedPool`), `io/` (safetensors
  index, `O_DIRECT` reads), `stream/source.py` (`FileWeights`: RAM tier + NVMe tier),
  `adapters/hf_transformers.py` (`load_causal_lm`) and `bench_llm`.
- **Order from here: F3 → F4 → F2**, then F5. Not the numeric order, for these reasons:
  - F3 (video diffusion) first: it uses no KV cache, and it is where streaming wins most
    clearly: thousands of tokens per step, far above the crossover of ≈ 13 tokens per pass
    (decision 8).
  - F4 (MoE + planner) next: it decides on which device each block is computed in each phase
    (resident on the GPU, on the CPU, or streamed).
  - F2 (KV-cache compression) last: where the KV cache lives depends on where attention is
    computed, and that is what F4 decides.
- **F3 is in progress.** Landed: dtype conversion while loading (`FileWeights(cast=True)`) and
  the Wan transformer with its sampling loop (`adapters/hf_diffusers.py`,
  `diffusion/chunking.py`). Exit criterion: 720p x 49 frames, and 480p x 121 as the long mode
  (the model does not guarantee quality at 480p). Measured on Wan 2.2 TI2V-5B with 2 slots:
  latents equal to `WanPipeline` bit for bit, 0.80 GiB of weights in VRAM, 1.8 GiB allocated at
  the peak and 12.8-13.9 s per pass at about 12,000 tokens, with the GPU at 85 °C.
- F1 exit criterion, measured with `bench_llm` on Qwen3-8B (2 slots, all blocks in RAM): same
  tokens as the strict baseline, 0.77 GB of weights in VRAM, prefill overlap efficiency
  0.90-1.00; 2.28 s/token against 2.88 with `accelerate` offloading, at a third of its peak VRAM.
- Known limitation: Qwen3-0.6B does not reach 80 % prefill efficiency with long prompts (about
  0.7 with 2048 and 4096 tokens): `lm_head` is 26 % of its bytes and has no compute to hide it.
- Left open by F1, now under milestone F4: issue #4 (a copy waits behind an unrelated disk read
  when some blocks are in RAM and some on disk).
- No stable public API yet — do not invent one beyond the roadmap.
- Phases (the numbers are names; the order is the one above): F0 foundations; F1 ping-pong
  layer streaming + NVMe tier; F2 TurboQuant KV cache + Triton kernels; F3 video diffusion
  (sequential pipeline, chunking, VAE tiling); F4 MoE expert cache + planner; F5 v0.1.0 release.
- Design docs live in `docs/` (Spanish, **local only, gitignored**): `arquitectura-y-roadmap.md`,
  `estado-del-arte-2025-2026.md`. Read them before architectural work; never commit them.

## Commands

```bash
# one-time setup on Linux (apt deps, .venv, torch+triton, ".[dev,test]", verifies GPU, runs tests;
# detects WSL2). The "test" extra is transformers, accelerate, diffusers and ftfy: the model
# tests and bench_llm need it.
bash scripts/setup_linux.sh
source .venv/bin/activate

pytest -q                       # tests, without the slow ones
pytest -q -m slow -rs           # the slow ones: real checkpoints (HYPERTIER_MODELS), ~6 minutes
TRITON_INTERPRET=1 pytest -q    # run Triton kernels on CPU (no GPU needed)
python benchmarks/bench_hw.py   # hardware profile -> benchmarks/results/hw_*.json (--quick: smoke)
python benchmarks/bench_streaming.py  # streaming vs resident model -> results/streaming_*.json
python benchmarks/bench_streaming.py --checkpoint "$HYPERTIER_MODELS/Qwen3-8B"  # real layers
python benchmarks/bench_llm.py --model "$HYPERTIER_MODELS/Qwen3-8B"  # tokens, VRAM, prefill overlap
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
   RAM → CPU is a link too, and the fastest one: see decision 8.
2. **KV cache = TurboQuant-MSE** (random rotation + Lloyd-Max codebook per coordinate, calibration-free).
   **QJL residual is OFF by default** (hurts softmax attention at equal bit budget; arXiv:2605.08114).
   Give keys more bits than values. Keep boundary layers and a recent-token window in fp16.
3. **Bit-plane packing** for any bit-width; attention scores computed on packed keys without
   de-rotating (`<q,k> = ‖k‖·<Rq, c[idx]>`); values accumulated in rotated basis, de-rotated once.
4. **Model-agnostic streaming via forward hooks** that swap each block's `Parameter` objects
   for ones viewing its VRAM slot, and swap the originals back afterwards. Not `param.data`: it
   cannot switch to or from a `meta` placeholder, which blocks read from disk use while they
   are not loaded. Do not rewrite model classes. Slots are assigned **LRU, never `i % num_slots`** (breaks with odd
   block counts + cyclic prefetch), and LRU never evicts a block of the current prefetch window
   (plain LRU drops pending prefetches from 4 slots up). CUDA events: `slot_ready` (compute waits
   copy), `slot_free` (copy waits compute). Every parameter is aligned to 256 B in RAM and VRAM.
   Weights wait in a `PinnedPool`: `mmap` memory locked with `cudaHostRegister` at its exact size
   (≈ 1.8 GB/s; 20 GiB verified, `RLIMIT_MEMLOCK` and `/dev/shm` do not limit it). PyTorch's
   pinned allocator rounds up to a power of two, packs small requests 512 B apart (so its
   memory is not page-aligned) and is only the fallback.
   Weights read from files (`FileWeights`): blocks that fit the RAM budget stay pinned, the
   rest are read every pass with `O_DIRECT` in 64 MiB requests (≈ 2.9 GB/s) into staging
   buffers. A checkpoint stored in another floating dtype than it runs in (a float32 DiT run
   in bf16) is converted once, on the CPU, while it is read into the RAM tier (`cast=True`);
   a converted block cannot use the NVMe tier, which needs it converted on disk (issue #9).
   When a block comes from disk, one reader thread reads and enqueues every copy;
   with every block in RAM there is no thread and the main thread enqueues them, as with
   weights in memory (issue #4). The host order between the two
   threads is kept by **permits**: only the main thread assigns slots, it issues a permit
   after `slot_free` is recorded, and it waits for the reader's notice on the host before
   `wait_event(slot_ready)`. The reader never allocates GPU memory. Recording a *timing*
   CUDA event can block while another thread is in a blocking read (`.item()`, `.to("cpu")`);
   plain events and copies do not, so only telemetry is exposed to it.
   A slot is as large as the biggest block, so oversized blocks waste VRAM in every slot
   (`LayerStreamer` warns above 1.5x the median).
   The Hugging Face adapter (`adapters/hf_transformers.py`, `load_causal_lm`) builds the model
   with its own `__init__` while a registration hook sends every `Parameter` to `meta`:
   buffers are then computed on the CPU as in a normal load (`inv_freq` computed on the GPU
   differs in the last bits). The token embedding stays on the CPU as a private `mmap` of the
   checkpoint, `lm_head` is streamed as ranges of vocabulary rows no larger than a layer
   (`TensorEntry.rows`), and `model.device` answers with the GPU through a per-instance
   subclass, because `generate()` moves its inputs there. With two slots the pieces of
   `lm_head` after the first have no compute to hide behind.
   `load_text_encoder` is the same for an encoder-only model (T5, umT5), the text stage of a
   diffusion pipeline. T5 looks tokens up through a second module that holds the embedding's
   parameter, so every module that holds it gets the same table. Its blocks are read from
   disk on every call by default, with only the staging buffers pinned: the stage runs once
   per prompt (umT5-XXL of Wan 2.2: 1.7 s to load and 3.4 s per call from disk, 9.8 s and
   1.4-1.8 s with 8.63 GiB pinned). A real batch of two prompts does not round like two
   batches of one, and `diffusers` encodes each prompt in a pass of its own. So every block
   runs the batch one prompt at a time while its weights are in the slot: the bits of a pass
   per prompt, with the weights read once (3.3 s for two prompts instead of 6.7 s).
5. **Weights stream as 4-bit / fp8 / lossless (DFloat11-style)**, never "2.5-bit lossless".
   Do not compress activations in the compute path; bound their peak by splitting the work.
   Splitting is not all exact. Chunking a module over tokens or over the batch computes the
   same values up to rounding: 1 ulp of bf16 measured, because another shape picks another
   kernel. Spatial VAE tiling is **approximate**: 40-43 dB of PSNR against the untiled
   decode, measured on the Wan 2.2 VAE at 480p with 256 px tiles overlapping 32-64 px.
6. **Triton first, PyTorch fallback.** Kernels must also run under `TRITON_INTERPRET=1` and be
   tested bit-exact (or within tolerance) against a pure-PyTorch reference.
7. **Native Linux first.** Development and benchmarks run on native Linux only; WSL2 is no longer
   used for either. WSL2 stays **supported for users**: `setup_linux.sh` and telemetry detect it
   (`is_wsl()`), and `bench_hw` labels its results. Native Windows only via the PyTorch fallback.

8. **The CPU is a compute tier.** A block whose weights are in RAM can be computed where they
   are: the CPU reads them at ≈ 36 GB/s on the reference laptop, against 6.7 GB/s over PCIe.
   Measured with `bench_llm` on Qwen3-8B: one token per pass takes 0.45 s on the CPU and 2.28 s
   streamed (nothing hides the copies); 512 tokens take 81 s on the CPU and 2.3 s streamed.
   The crossover is at ≈ 13 tokens per pass, an estimate interpolated between those two CPU
   measurements. Below it, compute on the CPU the blocks that are in RAM; above it, stream
   them to the GPU. The F4 planner will choose the compute device per block and per phase
   (prefill, decoding). First case: `lm_head` on the CPU while decoding. Far above the
   crossover the GPU wins by a wide margin: two prompts of 512 tokens through the umT5-XXL
   of Wan 2.2 take 125 s on the CPU in bf16 and 3.4 s streamed from disk.

9. **Diffusion is compared bit for bit.** In bfloat16 any change of shape in a kernel (a batch
   of two, tokens in chunks, one row instead of N) changes about one unit in the last place.
   The 30 blocks of a pass turn that into about 1 %, and the sampler into another video: 15 dB
   of PSNR after 16 steps with the timestep collapsed to one per sample; the same scene with
   other details. So equivalence means equal bits, and a threshold at the end of sampling
   says nothing. What cannot be exact (the FFN in token chunks, a real batch) is judged by the
   error of **one** pass, in units of the rounding noise of that pass: its distance to the
   same pass run in float32. On Wan 2.2 TI2V-5B at 1,950 tokens a bf16 pass is 1.1-1.9 % from
   float32, and chunking the FFN moves it by 0.43-0.74 of that; the tests allow 1.5.
   The Wan adapter (`load_wan_transformer`, `denoise`) equals `WanPipeline` by doing what
   `diffusers` does. Modules in `_keep_in_fp32_modules` stay in float32. The time embedding is
   computed per token, and one row is kept after the embedder when every token has the same
   timestep (the rows are equal; kept whole they cost 1.3 GiB more at 12,000 tokens); other
   timesteps take the whole per-token path. A batch runs one sample at a time inside each
   module (`one_sample_at_a_time`), so both branches of the guidance can share one pass over
   the weights with the same bits. Two passes stay the default: one pass only wins while the
   copy of a block outlasts its compute, below about 2,000 tokens on the reference laptop
   (at 1,950 they tie: 3.58 s per step against 3.64, with 0.2 GiB more).

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
- Measure host RAM as `AnonPages + Shmem` from `read_meminfo()` (pinned memory shows as `Shmem`),
  not `MemAvailable`: freed pages return to it with a lag of hundreds of MiB.
- A benchmark that compares runs gives each its own process (`bench_llm`): VRAM, page-locked
  RAM and allocator state are not shared, and the parent never opens a CUDA context.
- Report timings in thermal steady state: warm the GPU up first and record its temperature
  with the result. On the reference laptop a pass takes about 6 % longer once the GPU is hot.
- Tests and benchmarks that need real checkpoints read their directory from `HYPERTIER_MODELS`
  (set it in the git-ignored `.env`) and skip without it. Never write that path in the repo.
- Those tests are marked `slow`, and a plain `pytest` leaves them out. A PR that touches
  `stream/`, `adapters/` or `diffusion/` also runs `pytest -m slow -rs` and says so in its
  description, with the tests that were skipped: a skipped test did not run.

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
