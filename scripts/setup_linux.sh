#!/usr/bin/env bash
# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0
#
# HyperTier — one-shot environment setup for Linux (Ubuntu/Debian) + NVIDIA GPU.
#
# Usage (from anywhere inside the repository):
#   bash scripts/setup_linux.sh
#
# Works on the repository the script lives in (no cloning), creates .venv with the system
# python3, installs PyTorch + Triton + dev tools, and verifies that CUDA, pinned memory and a
# Triton kernel actually work on the GPU. Safe to re-run.
#
# Native Linux is the development and benchmark platform. WSL2 is still supported for users:
# it is detected and gets WSL-specific advice (driver lives on Windows, RAM limit, /mnt I/O).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m[x] %s\033[0m\n' "$*"; exit 1; }

# ---------------------------------------------------------------------------
step "1/6  Checking OS, NVIDIA driver and memory"
[ "$(uname -s)" = "Linux" ] || fail "HyperTier needs Linux (native or WSL2); this is $(uname -s)."
IS_WSL=0
if grep -qi microsoft /proc/version; then
  IS_WSL=1
  warn "WSL2 detected. Supported, but copy/compute overlap is lower than on native Linux
    (~0.7 vs ~0.95, measured by benchmarks/bench_hw.py); expect slower streaming."
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  if [ "$IS_WSL" -eq 1 ]; then
    fail "nvidia-smi not found inside WSL.
    Fix: update the NVIDIA driver on *Windows* (do NOT install any NVIDIA driver inside WSL),
    then run 'wsl --shutdown' in PowerShell and open the distro again."
  fi
  fail "nvidia-smi not found: the NVIDIA driver is missing.
    Fix (Ubuntu): sudo ubuntu-drivers install   then reboot and re-run this script.
    No CUDA Toolkit is needed; PyTorch wheels ship their own CUDA libraries."
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
MEM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
echo "System RAM: ${MEM_GB} GB"
if [ "$MEM_GB" -lt 20 ] && [ "$IS_WSL" -eq 1 ]; then
  warn "WSL only sees ${MEM_GB} GB of RAM. Raise it in C:\\Users\\<you>\\.wslconfig:
        [wsl2]
        memory=24GB
    then run 'wsl --shutdown' in PowerShell."
elif [ "$MEM_GB" -lt 16 ]; then
  warn "Only ${MEM_GB} GB of RAM; the pinned-RAM tier will be small."
fi

# ---------------------------------------------------------------------------
step "2/6  Checking system packages"
# python3-dev + build-essential: Triton compiles a small C launcher at runtime.
PKGS=(git python3-venv python3-pip python3-dev build-essential)
MISSING=()
for p in "${PKGS[@]}"; do
  dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || MISSING+=("$p")
done
if [ ${#MISSING[@]} -eq 0 ]; then
  echo "All system packages already installed; skipping apt (no sudo needed)."
else
  echo "Missing: ${MISSING[*]}  -> installing (asks for your password)"
  sudo apt-get update -y
  sudo apt-get install -y "${MISSING[@]}"
fi
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || fail "python3 is $(python3 --version 2>&1); HyperTier needs Python >= 3.10."
echo "Using $(python3 --version) from $(command -v python3)"

# ---------------------------------------------------------------------------
step "3/6  Repository: $REPO_ROOT"
if [ "$IS_WSL" -eq 1 ] && [[ "$REPO_ROOT" == /mnt/* ]]; then
  warn "Repo is on a Windows drive (/mnt/...): disk I/O through 9P is much slower and distorts
    NVMe benchmarks. Clone it under ~ instead."
fi
cd "$REPO_ROOT"
[ -f pyproject.toml ] || fail "pyproject.toml not found in $REPO_ROOT; run the script from the repo."

# ---------------------------------------------------------------------------
step "4/6  Creating virtualenv and installing PyTorch + Triton (large download)"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch triton
python -m pip install -e ".[dev]"

# ---------------------------------------------------------------------------
step "5/6  Verifying CUDA, pinned memory and a Triton kernel on the GPU"
# Triton's @jit reads the kernel source with inspect.getsourcelines(), which fails for code
# piped through stdin; run the check from a real file in a private temp dir instead.
CHECK_DIR="$(mktemp -d)"
trap 'rm -rf -- "$CHECK_DIR"' EXIT
cat > "$CHECK_DIR/check_gpu.py" <<'PY'
import sys
import torch

print(f"Python  {sys.version.split()[0]}")
print(f"PyTorch {torch.__version__}  (built for CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    sys.exit("[x] torch.cuda.is_available() is False. Check 'nvidia-smi'; this PyTorch build may "
             "need a newer NVIDIA driver (native: sudo ubuntu-drivers install, then reboot; "
             "WSL2: update the driver on Windows, then 'wsl --shutdown').")
props = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info()
print(f"GPU     {props.name}  (sm_{props.major}{props.minor}, {props.multi_processor_count} SMs)")
print(f"VRAM    {free/2**30:.2f} GiB free / {total/2**30:.2f} GiB total")

# Pinned (page-locked) host memory: HyperTier's RAM tier depends on it (WSL2 can restrict it).
try:
    buf = torch.empty(512 * 2**20, dtype=torch.uint8).pin_memory()
    print(f"Pinned  512 MiB allocation OK (is_pinned={buf.is_pinned()})")
    del buf
except RuntimeError as e:
    print(f"[!] Pinned 512 MiB allocation FAILED: {e}")

import triton
import triton.language as tl
print(f"Triton  {triton.__version__}")

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

n = 1_000_003
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")
out = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
torch.cuda.synchronize()
assert torch.allclose(out, x + y), "Triton kernel produced wrong results"
print("Triton  vector-add kernel ran on the GPU and matches PyTorch  ✔")
PY
python "$CHECK_DIR/check_gpu.py"

# ---------------------------------------------------------------------------
step "6/6  Running the test suite"
python -m pytest -q

printf '\n\033[1;32mEnvironment ready.\033[0m\n'
echo "Next time, activate it with:"
echo "    cd $REPO_ROOT && source .venv/bin/activate"
