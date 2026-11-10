# Security Policy

## Supported versions

HyperTier is in early development. Security fixes are applied to the latest release on the
`main` branch only.

| Version | Supported |
|---|---|
| latest `main` / most recent release | ✅ |
| older releases | ❌ |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues, discussions or pull requests.**

Report them privately by either:

- Email: **security@vfernandezg.dev**
- GitHub: [private vulnerability reporting](https://github.com/vfernandezg-dev/HyperTier/security/advisories/new)

Please include, as far as possible:

- a description of the issue and its impact;
- the affected version or commit, and your environment (OS, GPU, driver, Python/PyTorch/Triton versions);
- steps or a minimal proof of concept to reproduce it;
- any suggested fix or mitigation.

## What to expect

HyperTier is maintained by a small team, so timelines are best effort:

- acknowledgement of your report within **7 days**;
- an initial assessment within **14 days**;
- a fix or mitigation plan as soon as practical, coordinated with you before public disclosure.

We follow coordinated disclosure and are happy to credit reporters in the release notes
unless you prefer to remain anonymous.

## Scope notes

HyperTier loads model weights and runs GPU kernels on the user's machine. Issues of particular
interest include:

- code execution through model files or checkpoints (HyperTier is designed to load
  `safetensors` and never unpickle untrusted data);
- path traversal or unsafe file handling when reading or caching weights from disk;
- memory-safety problems in custom kernels that can be triggered by crafted inputs;
- supply-chain issues in the packaging or release process.

Vulnerabilities in third-party dependencies (PyTorch, Triton, CUDA, model repositories) should
be reported to those projects directly; let us know too if HyperTier is affected.
