# Contributing to HyperTier

Thanks for your interest! HyperTier is in early development, so the most valuable contributions right now are:

- **Benchmarks on low-VRAM hardware** — PCIe/NVMe bandwidth and VRAM numbers from your GPU (4–8 GB cards especially).
- **Design feedback** on the architecture (design documents are coming soon; meanwhile, open an issue).
- **Research pointers** to papers or techniques we are missing.

## Development setup

HyperTier targets native Linux with an NVIDIA GPU (only the driver is needed, not the CUDA Toolkit).

```bash
git clone https://github.com/vfernandezg-dev/HyperTier.git
cd HyperTier
bash scripts/setup_linux.sh            # apt deps, .venv, torch + triton, GPU check, tests
source .venv/bin/activate
pytest
ruff check .
```

Triton kernels can be tested without a GPU by setting `TRITON_INTERPRET=1`.

Without the script, `pip install -e ".[dev]"` is enough for what CI runs. The tests that load
Hugging Face models and `benchmarks/bench_llm.py` also need a GPU, PyTorch and
`pip install -e ".[test]"`; they are skipped otherwise.

`benchmarks/bench_video.py` writes the video it generates as an animated WebP. With
`pip install -e ".[video]"` (`imageio` and `imageio-ffmpeg`) it writes an MP4 instead. Both
packages are BSD-2-Clause, but the wheel of `imageio-ffmpeg` carries an `ffmpeg` executable
under the GPL, version 3 or later. It runs as a program of its own, and nothing in HyperTier
links to it or ships it; that is why it is an extra of its own and not part of `test`.

Tests on real checkpoints are marked `slow`, and a plain `pytest` leaves them out. Run them
with `pytest -m slow -rs` (about half an hour on a 4 GB laptop GPU). They read the directory of
the checkpoints from the `HYPERTIER_MODELS` environment variable and are skipped without it, or
without the checkpoint they need: `-rs` lists what was skipped, and a skipped test did not run.

## Pull requests

1. Open an issue first for anything larger than a small fix.
2. Keep PRs focused; include tests for new behavior.
3. For performance changes, include before/after numbers and the hardware used.
4. Run `pytest` and `ruff check .` before submitting.
   If the PR touches `src/hypertier/stream/`, `src/hypertier/adapters/` or
   `src/hypertier/diffusion/`, also run `pytest -m slow -rs`, and say in the description that
   you did and which tests were skipped. If you cannot run them (no GPU, or no checkpoints),
   say that instead: a maintainer then has to run them before the PR is merged.
5. Sign off every commit (see below).

## Developer Certificate of Origin (DCO)

HyperTier uses the [Developer Certificate of Origin 1.1](https://developercertificate.org/) instead of a CLA.
By signing off a commit you certify that you wrote the change, or otherwise have the right to submit it
under the project's open-source license.

Add the sign-off with the `-s` flag:

```bash
git commit -s -m "Describe your change"
```

This appends a line like the following to the commit message, using your Git name and email:

```
Signed-off-by: Jane Doe <jane.doe@example.com>
```

- The name and email must match your Git configuration (`git config user.name` / `git config user.email`).
  You may use your GitHub no-reply address if you prefer to keep your email private.
- Forgot to sign off? Fix the last commit with `git commit --amend -s --no-edit`, or all commits in your
  branch with `git rebase --signoff main`, then force-push.
- Pull requests with unsigned commits cannot be merged.

## License

By contributing you agree that your contributions are licensed under the [Apache License 2.0](LICENSE),
and you will be credited in the [AUTHORS](AUTHORS) file once your first contribution is merged.
