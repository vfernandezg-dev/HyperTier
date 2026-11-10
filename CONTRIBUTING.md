# Contributing to HyperTier

Thanks for your interest! HyperTier is in early development, so the most valuable contributions right now are:

- **Benchmarks on low-VRAM hardware** — PCIe/NVMe bandwidth and VRAM numbers from your GPU (4–8 GB cards especially).
- **Design feedback** on the architecture (design documents are coming soon; meanwhile, open an issue).
- **Research pointers** to papers or techniques we are missing.

## Development setup

```bash
git clone https://github.com/vfernandezg-dev/HyperTier.git
cd HyperTier
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
ruff check .
```

Triton kernels can be tested without a GPU by setting `TRITON_INTERPRET=1`.

## Pull requests

1. Open an issue first for anything larger than a small fix.
2. Keep PRs focused; include tests for new behavior.
3. For performance changes, include before/after numbers and the hardware used.
4. Run `pytest` and `ruff check .` before submitting.

By contributing you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
