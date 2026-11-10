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
