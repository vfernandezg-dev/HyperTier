# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Toy models and measuring helpers shared by the streaming tests.

PyTorch is imported inside the functions: CI has none, and importing this must not need it.
"""

import copy
import ctypes
import threading
import time

from hypertier.telemetry import read_meminfo


def toy_mlp(num_blocks: int, width: int, *, bias: bool = True, dtype: str = "float32"):
    """Residual MLP; with width 4096 in fp16 a block is 64 MiB and one pass is compute-bound."""
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(width, width, bias=bias)
            self.fc2 = nn.Linear(width, width, bias=bias)
            self.fail = False

        def forward(self, x):
            if self.fail:
                raise ArithmeticError("boom")
            return x + 0.1 * self.fc2(torch.relu(self.fc1(x)))

    class Stack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(Block() for _ in range(num_blocks))

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    torch.manual_seed(0)
    return Stack().to(getattr(torch, dtype)).eval()


def toy_transformer(num_blocks: int, *, d_model: int = 100, seed: int = 0):
    """Pre-norm transformer encoder with odd sizes, so parameters need alignment padding."""
    import torch
    from torch import nn

    class Stack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                nn.TransformerEncoderLayer(
                    d_model, 4, dim_feedforward=333, dropout=0.0, batch_first=True, norm_first=True
                )
                for _ in range(num_blocks)
            )
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.norm(x)

    torch.manual_seed(seed)
    return Stack().eval()


def resident_outputs(model, x, passes: int, mode: str = "no_grad"):
    """Outputs of ``passes`` chained passes of a copy of ``model`` that lives fully in VRAM."""
    import torch

    reference = copy.deepcopy(model).cuda()
    outputs = []
    with getattr(torch, mode)():
        for _ in range(passes):
            x = reference(x)
            outputs.append(x)
    assert all(bool(torch.isfinite(out).all()) for out in outputs)
    return outputs


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def timed_passes(model, x, passes: int) -> float:
    """Median wall time in ms of a full pass, until the GPU is idle again."""
    import torch

    times = []
    with torch.no_grad():
        for _ in range(passes):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
    return median(times)


def ram_in_use() -> int:
    """Anonymous plus shared memory, system-wide: pageable tensors plus page-locked ones.

    Not ``MemAvailable``: freed pages come back to it with a lag (measured: still 250-350 MiB
    short 1.5 s after pinning and releasing 256 MiB), which would be charged to the build.
    On Linux with the NVIDIA driver, memory locked by the pinned allocator shows as ``Shmem``.
    """
    info = read_meminfo()
    return info["AnonPages"] + info["Shmem"]


class RamSampler:
    """Highest :func:`ram_in_use` seen while the block runs, sampled every millisecond."""

    def __enter__(self):
        self.start = self.highest = ram_in_use()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(0.001):
            self.highest = max(self.highest, ram_in_use())

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join()
        self.highest = max(self.highest, ram_in_use())


SAFETENSORS_DTYPES = {"torch.float32": "F32", "torch.float16": "F16", "torch.bfloat16": "BF16",
                      "torch.float64": "F64", "torch.int64": "I64", "torch.uint8": "U8"}  # fmt: skip


def tensor_bytes(tensor) -> bytes:
    """The raw bytes of a CPU tensor (no NumPy needed)."""
    tensor = tensor.detach().contiguous()
    return ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size())


def checkpoint_tensors(model) -> dict:
    """``model.state_dict()`` as the writer fixture wants it, sorted by name like safetensors."""
    return {
        name: (SAFETENSORS_DTYPES[str(value.dtype)], tuple(value.shape), tensor_bytes(value))
        for name, value in sorted(model.state_dict().items())
    }


def tiny_causal_lm(directory, family="qwen3", *, tied=False, dtype="bfloat16", seed=0):
    """Write a random tiny language model as a normal ``transformers`` checkpoint.

    Five layers of width 64 and a vocabulary of 1500, so the output projection is larger than
    a layer and has to be split. Returns ``directory``.
    """
    import torch
    import transformers

    common = {"vocab_size": 1500, "tie_word_embeddings": tied, "bos_token_id": 1,
              "eos_token_id": 2, "pad_token_id": 0}  # fmt: skip
    if family == "gpt2":
        config = transformers.GPT2Config(n_embd=64, n_layer=5, n_head=4, n_positions=128, **common)
    else:
        cls = {"qwen3": transformers.Qwen3Config, "llama": transformers.LlamaConfig}[family]
        extra = {"head_dim": 16} if family == "qwen3" else {}
        config = cls(
            hidden_size=64, intermediate_size=128, num_hidden_layers=5, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=128, **extra, **common,
        )  # fmt: skip
    torch.manual_seed(seed)
    model = transformers.AutoModelForCausalLM.from_config(config)
    with torch.no_grad():
        for param in model.parameters():  # larger than the default init: logits far from ties
            param.mul_(4.0)
    model.to(getattr(torch, dtype)).save_pretrained(directory)
    return directory
