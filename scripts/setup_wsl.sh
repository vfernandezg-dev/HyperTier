#!/usr/bin/env bash
# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0
#
# HyperTier — one-shot development environment setup for WSL2 (Ubuntu) + NVIDIA GPU.
#
# Usage (inside WSL):
#   bash /mnt/d/Github/HyperTier/scripts/setup_wsl.sh
#
# Clones the repo into the Linux filesystem (~/dev/HyperTier by default; fast I/O),
# creates a virtualenv, installs PyTorch + Triton and verifies that CUDA and a
# Triton kernel actually run on the GPU. Safe to re-run.
set -euo pipefail

REPO_URL="https://github.com/vfernandezg-dev/HyperTier.git"
DEST="${HYPERTIER_HOME:-$HOME/dev/HyperTier}"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m[x] %s\033[0m\n' "$*"; exit 1; }

# ---------------------------------------------------------------------------
step "1/6  Checking WSL2, GPU and memory"
grep -qi microsoft /proc/version || warn "This does not look like WSL; continuing anyway."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found inside WSL.
    Fix: update the NVIDIA driver on *Windows* (do NOT install any NVIDIA driver inside WSL),
    then run 'wsl --shutdown' in PowerShell and open Ubuntu again."
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
MEM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
echo "RAM visible to WSL: ${MEM_GB} GB"
if [ "$MEM_GB" -lt 20 ]; then
  warn "WSL only sees ${MEM_GB} GB of RAM. HyperTier needs more for its RAM tier.
    Create C:\\Users\\<you>\\.wslconfig with:
        [wsl2]
        memory=24GB
        swap=8GB
    then run 'wsl --shutdown' in PowerShell. (Setup continues meanwhile.)"
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
  echo "Missing: ${MISSING[*]}  -> installing (asks for your Ubuntu password)"
  sudo apt-get update -y
  sudo apt-get install -y "${MISSING[@]}"
fi

# ---------------------------------------------------------------------------
step "3/6  Getting the repository into the Linux filesystem: $DEST"
case "$DEST" in /mnt/*) warn "DEST is on a Windows drive; disk I/O will be much slower." ;; esac
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" pull --ff-only || warn "Could not fast-forward; leaving your local copy as is."
else
  mkdir -p "$(dirname "$DEST")"
  git clone "$REPO_URL" "$DEST"
fi
cd "$DEST"

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
    sys.exit("[x] torch.cuda.is_available() is False. Update the NVIDIA driver on Windows "
             "(this PyTorch build needs a recent one), run 'wsl --shutdown', and retry.")
props = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info()
print(f"GPU     {props.name}  (sm_{props.major}{props.minor}, {props.multi_processor_count} SMs)")
print(f"VRAM    {free/2**30:.2f} GiB free / {total/2**30:.2f} GiB total")

# Pinned (page-locked) host memory: WSL2 can restrict it, and HyperTier depends on it.
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
echo "    cd $DEST && source .venv/bin/activate"
