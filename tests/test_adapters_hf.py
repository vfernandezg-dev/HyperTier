# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.adapters.hf_transformers.

How the output projection is split is plain Python and runs everywhere. The rest needs a GPU
and ``transformers``: tiny models are made from a config, saved as a normal checkpoint and
compared with the same checkpoint loaded the normal way, fully resident in VRAM.
"""

import sys
from itertools import pairwise

import pytest
from toys import tiny_causal_lm

from hypertier.adapters import hf_transformers as hf
from hypertier.adapters import load_causal_lm

MiB = 1 << 20

# ---------------------------------------------------------------------------------------------
# Splitting the output projection (no PyTorch)


def test_head_rows_are_even_ranges_no_larger_than_a_chunk():
    assert hf.head_rows(10, 4, 40) == [(0, 10)]
    assert hf.head_rows(10, 4, 39) == [(0, 5), (5, 10)]
    assert hf.head_rows(10, 4, 12) == [(0, 2), (2, 5), (5, 7), (7, 10)]
    assert hf.head_rows(7, 1, 1) == [(i, i + 1) for i in range(7)]
    assert hf.head_rows(0, 4, 40) == [(0, 0)]
    for rows, row_bytes, chunk in [(151_936, 8192, 385_892_864), (1000, 130, 4096), (17, 3, 10)]:
        ranges = hf.head_rows(rows, row_bytes, chunk)
        assert ranges[0][0] == 0 and ranges[-1][1] == rows
        assert all(a[1] == b[0] for a, b in pairwise(ranges)), "no gap, no overlap"
        sizes = [stop - start for start, stop in ranges]
        assert max(sizes) * row_bytes <= chunk and max(sizes) - min(sizes) <= 1
        assert len(ranges) == -(-rows // (chunk // row_bytes)), "the fewest that fit"
    with pytest.raises(ValueError, match="larger than a block"):
        hf.head_rows(10, 100, 99)


def test_head_rows_of_the_reference_models():
    # Qwen3-8B: 151936 x 4096 in bf16 against layers of 368 MiB -> 4 pieces of 296.75 MiB.
    assert hf.head_rows(151_936, 8192, 385_892_864) == [
        (0, 37_984), (37_984, 75_968), (75_968, 113_952), (113_952, 151_936)
    ]  # fmt: skip
    # Qwen3-0.6B: 151936 x 1024 against layers of 30 MiB -> 10 pieces.
    assert len(hf.head_rows(151_936, 2048, 31_457_280)) == 10


def test_importing_the_adapter_imports_neither_torch_nor_transformers():
    import subprocess

    code = (
        "import sys, hypertier.adapters; "
        "sys.exit(('torch' in sys.modules) + 2 * ('transformers' in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_versions_are_compared_by_their_leading_numbers():
    assert hf._version("5.24.0") == (5, 24, 0)
    assert hf._version("5.25.0.dev0") == (5, 25, 0)
    assert hf._version("4.57.6") < hf.MIN_TRANSFORMERS <= hf._version("5.24.0")


# ---------------------------------------------------------------------------------------------
# Tiny models


def load_resident(directory, rows=None):
    """The checkpoint loaded the normal way, in VRAM; ``rows`` splits its head the same way."""
    import torch
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(directory, dtype="auto").cuda()
    model.eval()
    if rows is not None:
        head = model.get_output_embeddings()
        chunked = hf._chunked_linear_type(torch)(
            head.in_features, tuple(rows), head.bias is not None, head.weight.dtype
        )
        for chunk, (start, stop) in zip(chunked.chunks, rows):
            chunk.weight = torch.nn.Parameter(head.weight[start:stop].detach().clone(), False)
            if head.bias is not None:
                chunk.bias = torch.nn.Parameter(head.bias[start:stop].detach().clone(), False)
        model.set_output_embeddings(chunked)
    return model


def generate(model, ids, new_tokens=12):
    import torch

    with torch.inference_mode():
        out = model.generate(
            input_ids=ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
            do_sample=False, output_logits=True, return_dict_in_generate=True,
        )  # fmt: skip
    return out.sequences.cpu(), torch.stack(out.logits).cpu()


def prompt(length=9, seed=1):
    import torch

    return torch.randint(3, 1500, (1, length), generator=torch.Generator().manual_seed(seed))


@pytest.fixture
def transformers():
    return pytest.importorskip("transformers")


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("family", "tied", "dtype", "slots", "tier"),
    [
        ("qwen3", False, "bfloat16", 2, "ram"), ("qwen3", True, "bfloat16", 2, "ram"),
        ("qwen3", False, "float32", 1, "ram"), ("qwen3", True, "float32", 3, "nvme"),
        ("qwen3", False, "bfloat16", 2, "nvme"), ("llama", False, "float16", 2, "ram"),
        ("llama", True, "bfloat16", 2, "nvme"), ("gpt2", True, "float32", 2, "ram"),
    ],
)  # fmt: skip
def test_generate_gives_the_tokens_and_logits_of_the_resident_model(
    transformers, disk_tmp, family, tied, dtype, slots, tier
):
    import torch

    tiny_causal_lm(disk_tmp, family, tied=tied, dtype=dtype)
    ids = prompt()
    budget = None
    if tier == "nvme":
        with load_causal_lm(disk_tmp) as probe:
            budget = 2 * (probe.streamer.slot_bytes + 8192)  # the two staging buffers only
    with load_causal_lm(disk_tmp, slots=slots, ram_budget=budget) as streamed:
        assert set(streamed.streamer.tiers) == {tier}
        assert len(streamed.head_rows) > 1, "the test model must need a split head"
        tokens, logits = generate(streamed.model, ids)
        again, _ = generate(streamed.model, ids.cuda())  # ids already on the GPU work too
        rows = list(streamed.head_rows)
    # Same arithmetic, weights resident: every logit of every step is the same bits.
    want_tokens, want_logits = generate(load_resident(disk_tmp, rows), ids.cuda())
    assert torch.equal(tokens, want_tokens) and torch.equal(again, want_tokens)
    assert torch.equal(logits, want_logits)
    # The stock model multiplies by the whole projection at once: last-bit differences only.
    stock_tokens, stock_logits = generate(load_resident(disk_tmp), ids.cuda())
    tolerance = 0.05 if dtype == "float32" else 0.5
    assert torch.allclose(logits.float(), stock_logits.float(), atol=tolerance, rtol=0.02)
    assert (tokens == stock_tokens).float().mean() > 0.9


@pytest.mark.cuda
def test_weights_are_placed_by_how_they_are_used(transformers, disk_tmp):
    import torch

    tiny_causal_lm(disk_tmp)
    before = torch.cuda.memory_allocated()
    with load_causal_lm(disk_tmp) as streamed:
        model, streamer = streamed.model, streamed.streamer
        assert model.device == streamed.device == torch.device("cuda", 0)
        assert isinstance(model, transformers.Qwen3ForCausalLM)
        assert type(model).__name__ == "Qwen3ForCausalLM"
        # Embedding: on the CPU, from the file. Layers and head: placeholders until they run.
        assert model.get_input_embeddings().weight.device.type == "cpu"
        assert all(p.device.type == "meta" for p in model.model.layers.parameters())
        assert all(p.device.type == "meta" for p in model.lm_head.parameters())
        assert model.model.norm.weight.device.type == "cuda"
        assert model.model.rotary_emb.inv_freq.device.type == "cuda"
        assert not any(p.requires_grad for p in model.parameters())
        assert streamer.num_blocks == 5 + len(streamed.head_rows)
        assert streamed.head_rows[0][0] == 0 and streamed.head_rows[-1][1] == 1500
        # VRAM for weights: the slots and a few hundred bytes of norm and rotary tables.
        assert streamed.resident_bytes == 64 * 2 + 2 * 8 * 4
        assert streamed.vram_bytes == 2 * streamer.slot_bytes + streamed.resident_bytes
        assert torch.cuda.memory_allocated() - before <= streamed.vram_bytes + 4096
        assert streamed.embedding_bytes == 1500 * 64 * 2
        with torch.inference_mode():
            logits = model(prompt().cuda(), logits_to_keep=1).logits
        assert logits.shape == (1, 1, 1500) and logits.device.type == "cuda"
        stats = streamer.pass_stats()
        assert len(stats) == 1 and stats[0].blocks == streamer.num_blocks
    assert torch.cuda.memory_allocated() - before <= logits.numel() * 2 + 8192


@pytest.mark.cuda
def test_buffers_computed_by_init_equal_those_of_a_normal_load(transformers, disk_tmp):
    import torch

    tiny_causal_lm(disk_tmp)
    resident = load_resident(disk_tmp)
    with load_causal_lm(disk_tmp) as streamed:
        ours = dict(streamed.model.named_buffers())
        theirs = dict(resident.named_buffers())
        assert ours.keys() == theirs.keys() and ours
        for name, buf in ours.items():
            assert buf.dtype == theirs[name].dtype and torch.equal(buf, theirs[name]), name
        assert streamed.model.generation_config.eos_token_id == 2


@pytest.mark.cuda
def test_close_undoes_everything_and_frees_the_gpu(transformers, disk_tmp):
    import torch

    tiny_causal_lm(disk_tmp)
    before = torch.cuda.memory_allocated()
    streamed = load_causal_lm(disk_tmp)
    model = streamed.model
    generate(model, prompt(), new_tokens=3)
    streamed.close()
    streamed.close()  # idempotent
    assert type(model) is transformers.Qwen3ForCausalLM
    assert isinstance(model.lm_head, torch.nn.Linear)
    assert all(p.device.type == "meta" for p in model.parameters())
    assert all(b.device.type == "cpu" for b in model.buffers())
    assert not model.get_input_embeddings()._forward_hooks
    assert not model.get_input_embeddings()._forward_pre_hooks
    assert torch.cuda.memory_allocated() == before
    assert not [t for t in __import__("threading").enumerate() if t.name == "hypertier-reader"]


@pytest.mark.cuda
def test_a_checkpoint_that_does_not_match_the_model_is_refused(transformers, disk_tmp, tmp_path):
    import json

    import torch

    tiny_causal_lm(disk_tmp)
    before = torch.cuda.memory_allocated()
    config = json.loads((disk_tmp / "config.json").read_text())
    config.pop("layer_types", None)  # derived from the number of layers, which changes below

    (disk_tmp / "config.json").write_text(json.dumps({**config, "num_hidden_layers": 6}))
    with pytest.raises(ValueError, match=r"'model\.layers\.5\..*' is not in the checkpoint"):
        load_causal_lm(disk_tmp)
    (disk_tmp / "config.json").write_text(json.dumps({**config, "vocab_size": 1400}))
    with pytest.raises(ValueError, match=r"the output projection is \[1400, 64\]"):
        load_causal_lm(disk_tmp)
    (disk_tmp / "config.json").write_text(json.dumps({**config, "intermediate_size": 96}))
    with pytest.raises(ValueError, match=r"the module expects torch.bfloat16 \[96, 64\]"):
        load_causal_lm(disk_tmp)
    assert torch.cuda.memory_allocated() == before, "a failed load leaves nothing on the GPU"
    empty = tmp_path / "empty"  # not tmp_path itself: without a cache it is disk_tmp
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_causal_lm(empty)


@pytest.mark.cuda
def test_an_old_transformers_is_reported(transformers, disk_tmp, monkeypatch):
    monkeypatch.setattr(transformers, "__version__", "4.57.6")
    with pytest.raises(RuntimeError, match=r"needs transformers >= 5\.24, found 4\.57\.6"):
        load_causal_lm(disk_tmp)


@pytest.mark.cuda
def test_real_checkpoint_generates_the_tokens_of_the_resident_model(transformers, models_dir):
    import torch

    root = models_dir / "Qwen3-0.6B"
    if not root.is_dir():
        pytest.skip("Qwen3-0.6B is not downloaded")
    tokenizer = transformers.AutoTokenizer.from_pretrained(root)
    ids = tokenizer("The three primary colours are", return_tensors="pt").input_ids
    before = torch.cuda.memory_allocated()
    with load_causal_lm(root) as streamed:
        assert streamed.vram_bytes <= 61 * MiB  # two slots of 30 MiB against 1.1 GiB of weights
        assert torch.cuda.memory_allocated() - before <= 61 * MiB
        assert len(streamed.head_rows) == 10 and streamed.streamer.num_blocks == 38
        tokens, logits = generate(streamed.model, ids, new_tokens=16)
    want_tokens, want_logits = generate(load_resident(root), ids.cuda(), new_tokens=16)
    assert torch.equal(tokens, want_tokens)
    # One position per step: on the reference GPU even the stock head gives the same bits.
    assert (logits.float() - want_logits.float()).abs().max() <= 0.25
