# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.stream.pingpong.

The slot policy is plain Python and runs everywhere, CI included. Everything that touches
weights needs the GPU: toy models are compared bit for bit with the same model resident in
VRAM, and VRAM, RAM and overlap are measured on the device.
"""

import copy
import gc
import mmap
import subprocess
import sys
import threading
import time
import warnings

import pytest

from hypertier.stream import LayerStreamer, PassStats, PinnedPool, find_blocks
from hypertier.stream.pingpong import Load, SlotTable
from hypertier.stream.pinned import ALIGN
from hypertier.telemetry import MiB, read_meminfo

# ---------------------------------------------------------------------------------------------
# Slot policy: no PyTorch needed

BLOCK_COUNTS = (1, 2, 3, 7)
SLOT_COUNTS = (1, 2, 3, 4)


def drive(table: SlotTable, order, content=None):
    """Run ``order`` through ``table`` like the streamer does, checking every load.

    Returns the slot content and, per entered block, ``(slot, loads)``.
    """
    content = {} if content is None else content
    steps = []
    for block in order:
        slot, loads = table.enter(block)
        for load in loads:
            if load.block != block:
                assert load.slot != slot, f"prefetch of {load.block} overwrites running {block}"
            content[load.slot] = load.block
        assert content[slot] == block, f"block {block} runs on the weights of {content[slot]}"
        assert table.slot_of(block) == slot and table.block_in(slot) == block
        steps.append((slot, loads))
        assert table.exit(block) == slot
    return content, steps


def primed(num_blocks: int, num_slots: int):
    table = SlotTable(num_blocks, num_slots)
    content = {load.slot: load.block for load in table.prime()}
    return table, content


def test_importing_the_package_does_not_import_torch():
    code = "import sys, hypertier.stream; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


@pytest.mark.parametrize("num_slots", SLOT_COUNTS)
@pytest.mark.parametrize("num_blocks", BLOCK_COUNTS)
def test_prime_loads_the_first_blocks_in_order(num_blocks, num_slots):
    table = SlotTable(num_blocks, num_slots)
    held = min(num_blocks, num_slots)
    assert table.num_slots == held
    assert table.prime() == [Load(block=i, slot=i, demand=False) for i in range(held)]
    assert table.prime() == []  # already loaded


@pytest.mark.parametrize("num_slots", SLOT_COUNTS)
@pytest.mark.parametrize("num_blocks", BLOCK_COUNTS)
def test_cyclic_passes_prefetch_every_block_and_never_touch_a_running_slot(num_blocks, num_slots):
    table, content = primed(num_blocks, num_slots)
    passes = 5
    _, steps = drive(table, list(range(num_blocks)) * passes, content)
    loads = [load for _, step_loads in steps for load in step_loads]
    if num_slots == 1 and num_blocks > 1:
        # A single slot cannot prefetch: every block is loaded when it is needed.
        assert all(load.demand for load in loads)
        assert len(loads) == passes * num_blocks - 1  # block 0 was primed
        return
    assert not any(load.demand for load in loads), "a prefetched block was evicted before its use"
    if num_blocks <= num_slots:
        assert loads == []  # everything fits: nothing moves after priming
    else:
        steady = [len(step_loads) for _, step_loads in steps[num_blocks:]]
        assert steady == [1] * len(steady)  # one copy per block, all the weights once per pass


def test_window_wraps_to_block_zero():
    assert SlotTable(7, 2).window(6) == [0]
    assert SlotTable(7, 4).window(5) == [6, 0, 1]
    assert SlotTable(1, 2).window(0) == []
    assert SlotTable(2, 4).window(1) == [0]


def test_odd_block_count_alternates_slots_where_modulo_would_collide():
    # Three blocks, two slots. With ``slot = block % 2``, block 2 runs in slot 0 and the
    # prefetch that wraps to block 0 targets slot 0 too: the slot being computed.
    table, content = primed(3, 2)
    _, steps = drive(table, [0, 1, 2, 0, 1, 2], content)
    assert [slot for slot, _ in steps] == [0, 1, 0, 1, 0, 1]
    slot_of_block_2, loads_during_block_2 = steps[2]
    assert loads_during_block_2 == [Load(block=0, slot=1, demand=False)]
    assert slot_of_block_2 == 2 % 2 == 0 % 2  # the collision ``i % slots`` would produce


def test_four_slots_evict_the_finished_block_not_a_pending_prefetch():
    # Plain LRU fails here: the block prefetched two steps ago is older than the one that
    # just finished, yet it is the next to run.
    table, content = primed(7, 4)
    order = list(range(7)) * 3
    evicted = []
    for block in order:
        _, loads = table.enter(block)
        for load in loads:
            evicted.append((block, content.get(load.slot)))
            content[load.slot] = load.block
        table.exit(block)
    assert evicted == [(block, (block - 1) % 7) for block in order[1:]]


def test_out_of_order_block_is_a_demand_load_and_prefetch_recovers():
    table, content = primed(7, 2)
    drive(table, [0, 1, 2], content)
    slot, loads = table.enter(5)  # jump: 3 was prefetched, 5 is not loaded
    assert [(load.block, load.demand) for load in loads] == [(5, True), (6, False)]
    assert loads[0].slot == slot != loads[1].slot
    for load in loads:
        content[load.slot] = load.block
    table.exit(5)
    _, steps = drive(table, [6, 0, 1, 2], content)
    assert not any(load.demand for _, step_loads in steps for load in step_loads)


def test_running_blocks_are_never_evicted():
    table = SlotTable(3, 2)
    table.prime()
    slot0, _ = table.enter(0)
    slot1, loads = table.enter(1)  # nested: both slots are running
    assert slot0 != slot1 and loads == []
    with pytest.raises(RuntimeError, match="no VRAM slot"):
        table.enter(2)
    assert table.exit(1) == slot1
    slot2, loads = table.enter(2)
    assert slot2 == slot1 and loads == [Load(block=2, slot=slot1, demand=True)]


def test_reentrant_block_keeps_its_slot_until_the_last_exit():
    table = SlotTable(3, 2)
    slot, loads = table.enter(0)
    assert loads[0] == Load(block=0, slot=slot, demand=True)
    assert table.enter(0) == (slot, [])
    assert table.exit(0) is None
    assert table.exit(0) == slot
    assert table.exit(0) is None  # not running any more


@pytest.mark.parametrize(("num_blocks", "num_slots"), [(0, 2), (3, 0)])
def test_slot_table_validates_sizes(num_blocks, num_slots):
    with pytest.raises(ValueError):
        SlotTable(num_blocks, num_slots)


def test_slot_table_rejects_unknown_blocks():
    table = SlotTable(3, 2)
    for block in (-1, 3):
        with pytest.raises(IndexError):
            table.enter(block)


# ---------------------------------------------------------------------------------------------
# Toy models (PyTorch imported lazily: CI has none)


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


# ---------------------------------------------------------------------------------------------
# find_blocks (PyTorch, no GPU)


def test_find_blocks_picks_the_largest_module_list():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = nn.ModuleList(nn.Linear(32, 32) for _ in range(2))  # not a rival

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(10, 32)
            self.adapters = nn.ModuleList(nn.Linear(4, 4) for _ in range(3))
            self.layers = nn.ModuleList(Block() for _ in range(5))

    model = Model()
    assert find_blocks(model) is model.layers
    assert find_blocks(model.layers) is model.layers  # the root itself can be the list


def test_find_blocks_refuses_to_guess():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class TwoStacks(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.double_blocks = nn.ModuleList(nn.Linear(32, 32) for _ in range(4))
            self.single_blocks = nn.ModuleList(nn.Linear(32, 32) for _ in range(3))

    with pytest.raises(ValueError, match="double_blocks.*single_blocks"):
        find_blocks(TwoStacks())
    with pytest.raises(ValueError, match="no ModuleList"):
        find_blocks(nn.Sequential(nn.Linear(4, 4), nn.ReLU()))


# ---------------------------------------------------------------------------------------------
# Streaming on the GPU


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
@pytest.mark.parametrize(
    ("num_blocks", "slots"), [(7, 2), (7, 3), (7, 4), (8, 2), (7, 1), (3, 2), (2, 2), (1, 2)]
)
def test_streamed_transformer_is_bit_exact(num_blocks, slots, mode):
    import torch

    model = toy_transformer(num_blocks)
    x = torch.randn(4, 33, 100, device="cuda")
    expected = resident_outputs(model, x, passes=3, mode=mode)
    model.norm.cuda()
    with LayerStreamer(model.blocks, slots=slots) as streamer, getattr(torch, mode)():
        assert streamer.num_slots == min(slots, num_blocks)
        y = x
        for want in expected:
            y = model(y)
            assert torch.equal(y, want)


@pytest.mark.cuda
def test_streamer_built_inside_inference_mode_serves_normal_forwards():
    import torch

    model = toy_transformer(7)
    x = torch.randn(2, 9, 100, device="cuda")
    (expected,) = resident_outputs(model, x, passes=1)
    model.norm.cuda()
    with torch.inference_mode():
        streamer = LayerStreamer(model.blocks)
    with streamer:
        assert not any(p.is_inference() for p in model.blocks.parameters())
        with torch.no_grad():
            assert torch.equal(model(x), expected)
        with torch.inference_mode():
            assert torch.equal(model(x), expected)


@pytest.mark.cuda
def test_parameters_live_in_pinned_ram_and_run_from_aligned_slot_views():
    import torch

    model = toy_transformer(7)
    params = list(model.blocks.parameters())
    seen = []

    def inspect(module, args):  # registered after the streamer: runs once weights are in place
        seen.append([(p.device.type, p.data_ptr() % ALIGN) for p in module.parameters()])

    with LayerStreamer(model.blocks) as streamer:
        assert streamer.slot_bytes % ALIGN == 0
        assert streamer.vram_bytes == 2 * streamer.slot_bytes
        assert streamer.streamed_bytes == 7 * streamer.slot_bytes
        assert streamer.slot_bytes >= sum(p.numel() * 4 for p in model.blocks[0].parameters())
        for p in params:
            assert p.device.type == "cpu" and p.is_pinned() and not p.requires_grad
            assert p.data_ptr() % ALIGN == 0
        handles = [block.register_forward_pre_hook(inspect) for block in model.blocks]
        model.norm.cuda()
        with torch.no_grad():
            model(torch.randn(2, 9, 100, device="cuda"))
        for handle in handles:
            handle.remove()
        assert len(seen) == 7
        assert all(where == ("cuda", 0) for block in seen for where in block)
        assert all(p.device.type == "cpu" for p in params), "weights must point back to RAM"


@pytest.mark.cuda
def test_buffers_become_resident_and_return_on_close():
    import torch
    from torch import nn

    blocks = nn.ModuleList(nn.BatchNorm1d(16) for _ in range(3)).eval()
    reference = copy.deepcopy(blocks).cuda()
    x = torch.randn(8, 16, device="cuda")
    with LayerStreamer(blocks) as streamer, torch.no_grad():
        assert all(b.device.type == "cuda" for b in blocks.buffers())
        buffer_bytes = sum(b.numel() * b.element_size() for b in blocks.buffers())
        assert streamer.vram_bytes == 2 * streamer.slot_bytes + buffer_bytes
        y, want = x, x
        for block, ref in zip(blocks, reference):
            y, want = block(y), ref(want)
        assert torch.equal(y, want)
    assert all(b.device.type == "cpu" for b in blocks.buffers())


@pytest.mark.cuda
def test_mlp_is_bit_exact_within_two_slots_of_vram():
    import torch

    from hypertier.telemetry.vram import track_vram

    model = toy_mlp(7, 4096, dtype="float16")
    x = torch.randn(8192, 4096, device="cuda", dtype=torch.float16)
    reference = copy.deepcopy(model).cuda()
    with torch.no_grad(), track_vram() as resident:  # weights already loaded: activations only
        expected = reference(x)
    assert bool(torch.isfinite(expected).all())
    activations = resident.stats.peak_delta
    weights = sum(p.numel() * p.element_size() for p in reference.parameters())
    del reference
    torch.cuda.empty_cache()

    with track_vram() as streamed:
        streamer = LayerStreamer(model.blocks)
        with torch.no_grad():
            out = model(x)
        exact = torch.equal(out, expected)
    streamer.close()
    peak = streamed.stats.peak_delta
    budget = 2 * streamer.slot_bytes + activations + 16 * MiB
    print(
        f"\n[stream] 7 blocks x {streamer.slot_bytes / MiB:.2f} MiB ({weights / MiB:.0f} MiB of "
        f"weights): bit-exact={exact}; VRAM peak {peak / MiB:.0f} MiB = 2 slots "
        f"({2 * streamer.slot_bytes / MiB:.0f} MiB) + {(peak - 2 * streamer.slot_bytes) / MiB:.0f}"
        f" MiB; activations alone {activations / MiB:.0f} MiB; budget {budget / MiB:.0f} MiB; "
        f"resident model peaks at {(weights + activations) / MiB:.0f} MiB"
    )
    assert exact
    assert peak <= budget


@pytest.mark.cuda
def test_copies_hide_behind_compute_when_blocks_are_compute_bound():
    import torch

    from hypertier.telemetry.bandwidth import overlap_efficiency, warm_up_gpu

    model = toy_mlp(7, 4096, dtype="float16")
    x = torch.randn(8192, 4096, device="cuda", dtype=torch.float16)
    reference = copy.deepcopy(model).cuda()
    warm_up_gpu(1.0)
    timed_passes(reference, x, 2)
    resident_ms = timed_passes(reference, x, 7)
    del reference
    torch.cuda.empty_cache()

    with LayerStreamer(model.blocks) as streamer:
        timed_passes(model, x, 2)  # first pass starts cold; steady state from the second
        streamed_ms = timed_passes(model, x, 7)
        stats = streamer.pass_stats()[2:]
    copy_ms = median([s.copy_ms for s in stats])
    in_situ = median([s.efficiency for s in stats])
    efficiency = overlap_efficiency(copy_ms, resident_ms, streamed_ms)
    print(
        f"\n[stream] pass: resident {resident_ms:.1f} ms, streamed {streamed_ms:.1f} ms, copies "
        f"{copy_ms:.1f} ms ({stats[-1].bytes_copied / MiB:.0f} MiB at "
        f"{stats[-1].bytes_copied / copy_ms / 1e6:.2f} GB/s); efficiency vs resident model "
        f"{efficiency:.3f}; telemetry efficiency {in_situ:.3f}, stall "
        f"{median([s.stall_ms for s in stats]):.2f} ms"
    )
    assert all(s.copies == 7 and s.misses == 0 and s.blocks == 7 for s in stats)
    assert resident_ms > 2 * copy_ms, "the test needs compute per block well above the copy"
    assert efficiency >= 0.8
    assert in_situ >= 0.8


@pytest.mark.cuda
def test_build_keeps_ram_at_the_model_plus_one_block():
    import torch

    # CUDA, the copy stream and page-locking allocate host memory on first use.
    with LayerStreamer(toy_mlp(2, 64).blocks):
        pass
    # Realistic blocks: biases and a width that is no power of two (61.05 MiB each).
    model = toy_mlp(7, 4000, dtype="float16")
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    corner = model.blocks[6].fc2.weight[:8, :8].clone()
    gc.collect()
    with RamSampler() as ram:  # the model is already in RAM: this measures the extra
        streamer = LayerStreamer(model.blocks)
    extra = ram.highest - ram.start
    block = streamer.slot_bytes
    pinned = streamer._pool.capacity_bytes
    print(
        f"\n[stream] build of 7 x {block / MiB:.2f} MiB: RAM in use peaked {extra / MiB:.1f} MiB "
        f"above the model and ended {(ram_in_use() - ram.start) / MiB:+.1f} MiB; pool "
        f"{pinned / MiB:.2f} MiB locked for {weights / MiB:.2f} MiB of weights in "
        f"{streamer._pool.pin_seconds:.2f} s"
    )
    assert block % (1 << 20) != 0 and block > weights // 7
    assert torch.equal(model.blocks[6].fc2.weight[:8, :8], corner)
    assert 0 <= pinned - streamer.streamed_bytes < 7 * mmap.PAGESIZE, "locked more than needed"
    streamer.close()
    assert extra <= block + 16 * MiB  # one block, plus slack for what else the system does


@pytest.mark.cuda
def test_close_leaves_a_valid_cpu_model():
    import torch

    model = toy_transformer(7)
    model.blocks[3].linear1.weight.requires_grad_(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    grad_flags = [p.requires_grad for p in model.parameters()]
    x = torch.randn(2, 9, 100, device="cuda")
    (expected,) = resident_outputs(model, x, passes=1)
    model.norm.cuda()

    streamer = LayerStreamer(model.blocks)
    with torch.no_grad():
        out = model(x)
    attached = model.state_dict()  # correct while attached too: weights point to RAM
    assert all(torch.equal(attached[name].cpu(), value) for name, value in before.items())
    streamer.close()  # right after the pass, with its kernels still queued
    streamer.close()  # idempotent
    assert torch.equal(out, expected)

    gc.collect()  # the private pool is gone: only the parameters keep its memory alive
    assert not any(p.is_pinned() for p in model.blocks.parameters()), "pages must be unlocked"
    after = model.state_dict()
    assert after.keys() == before.keys()
    assert all(torch.equal(after[name].cpu(), value) for name, value in before.items())
    assert all(p.device.type == "cpu" for p in model.blocks.parameters())
    assert [p.requires_grad for p in model.parameters()] == grad_flags
    assert all(not b._forward_hooks and not b._forward_pre_hooks for b in model.blocks)
    assert len(streamer.pass_stats()) == 1  # telemetry survives the close
    with torch.no_grad():  # and the model simply runs on the CPU again
        assert bool(torch.isfinite(model.cpu()(x.cpu())).all())


@pytest.mark.cuda
def test_pool_is_reused_by_a_second_model():
    import torch

    first, second = toy_transformer(7), toy_transformer(5, seed=1)
    first_weights = {name: value.clone() for name, value in first.state_dict().items()}
    x = torch.randn(2, 9, 100, device="cuda")
    (expected_first,) = resident_outputs(first, x, passes=1)
    (expected_second,) = resident_outputs(second, x, passes=1)
    first.norm.cuda()
    second.norm.cuda()

    with PinnedPool(0, segment_bytes=4 * MiB) as pool:
        streamer = LayerStreamer(first.blocks, pool=pool)
        with torch.no_grad():
            assert torch.equal(first(x), expected_first)
        with pytest.raises(RuntimeError, match="still attached"):
            pool.reset()
        streamer.close()
        assert not any(p.is_pinned() for p in first.blocks.parameters()), "weights left the pool"
        capacity, pin_seconds = pool.capacity_bytes, pool.pin_seconds

        pool.reset()
        assert pool.used_bytes == 0
        with LayerStreamer(second.blocks, pool=pool), torch.no_grad():
            assert torch.equal(second(x), expected_second)
        assert pool.capacity_bytes == capacity, "the second model must reuse the pinned pages"
        assert pool.pin_seconds == pin_seconds

    state = first.state_dict()  # not overwritten by the model that took over the pool
    assert all(torch.equal(state[name].cpu(), value) for name, value in first_weights.items())


@pytest.mark.cuda
def test_streaming_works_on_the_pinned_allocator_fallback(monkeypatch):
    import torch

    def missing():
        raise AttributeError("this PyTorch has no cudart binding")

    model = toy_transformer(7)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    x = torch.randn(2, 9, 100, device="cuda")
    expected = resident_outputs(model, x, passes=2)
    model.norm.cuda()
    monkeypatch.setattr(torch.cuda, "cudart", missing)
    with pytest.warns(UserWarning, match="cudaHostRegister.*falling back"):
        streamer = LayerStreamer(model.blocks)
    with streamer, torch.no_grad():
        assert all(p.is_pinned() for p in model.blocks.parameters())
        y = x
        for want in expected:
            y = model(y)
            assert torch.equal(y, want)
    after = model.state_dict()
    assert all(torch.equal(after[name].cpu(), value) for name, value in before.items())


@pytest.mark.cuda
def test_out_of_order_and_failing_blocks_stay_correct():
    import torch

    model = toy_mlp(7, 64)
    reference = copy.deepcopy(model).cuda()
    x = torch.randn(5, 64, device="cuda")
    with LayerStreamer(model.blocks) as streamer, torch.no_grad():
        y, want = x, x
        for index in (0, 1, 2, 5, 6, 3, 3, 0):
            y, want = model.blocks[index](y), reference.blocks[index](want)
        assert torch.equal(y, want)

        model.blocks[2].fail = True
        with pytest.raises(ArithmeticError):
            model(x)
        assert all(p.device.type == "cpu" for p in model.blocks.parameters())
        model.blocks[2].fail = False
        assert torch.equal(model(x), reference(x))
        assert sum(s.misses for s in streamer.pass_stats()) > 0  # the jumps were demand loads


@pytest.mark.cuda
def test_pass_stats_count_copies_and_bytes():
    import torch

    model = toy_transformer(7)
    model.norm.cuda()
    x = torch.randn(2, 9, 100, device="cuda")
    with LayerStreamer(model.blocks) as streamer, torch.no_grad():
        for _ in range(3):
            model(x)
        stats = streamer.pass_stats()
        assert stats == streamer.pass_stats()  # reading does not start a new pass
    assert all(isinstance(s, PassStats) for s in stats)
    # Blocks 0 and 1 are loaded at construction, so the first pass copies one block less.
    assert [s.copies for s in stats] == [6, 7, 7]
    assert [s.bytes_copied for s in stats] == [s.copies * streamer.slot_bytes for s in stats]
    for s in stats:
        assert s.blocks == 7 and s.misses == 0
        assert s.copy_ms > 0 and s.compute_ms > 0 and s.stall_ms >= 0
        assert s.total_ms >= s.compute_ms - 1e-3
        assert s.efficiency is not None


@pytest.mark.cuda
def test_pass_stats_without_copies_or_telemetry():
    import torch

    x = torch.randn(2, 9, 100, device="cuda")
    model = toy_transformer(2)  # both blocks fit in the two slots: nothing is streamed
    model.norm.cuda()
    with LayerStreamer(model.blocks) as streamer, torch.no_grad():
        model(x)
        model(x)
        stats = streamer.pass_stats()
    assert [(s.copies, s.bytes_copied, s.efficiency) for s in stats] == [(0, 0, None)] * 2

    model = toy_transformer(3)
    model.norm.cuda()
    with LayerStreamer(model.blocks, telemetry=False) as streamer, torch.no_grad():
        model(x)
        assert streamer.pass_stats() == []


@pytest.mark.cuda
def test_oversized_block_warns():
    from torch import nn

    with pytest.warns(UserWarning, match="largest streamed block"):
        LayerStreamer([nn.Linear(64, 64), nn.Linear(64, 64), nn.Linear(64, 512)]).close()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*largest streamed block.*")
        LayerStreamer([nn.Linear(64, 64), nn.Linear(64, 64), nn.Linear(64, 90)]).close()


@pytest.mark.cuda
def test_rejects_blocks_it_cannot_stream_and_leaves_them_untouched():
    from torch import nn

    block = nn.Linear(8, 8)
    twin = nn.Linear(8, 8)
    twin.weight = block.weight
    with pytest.raises(ValueError, match="empty"):
        LayerStreamer([])
    with pytest.raises(TypeError):
        LayerStreamer([block, "not a module"])
    with pytest.raises(ValueError, match="distinct"):
        LayerStreamer([block, block])
    with pytest.raises(ValueError, match="share parameters"):
        LayerStreamer([block, twin])
    with pytest.raises(ValueError, match="slots"):
        LayerStreamer([block], slots=0)
    with pytest.raises(ValueError, match="CUDA device"):
        LayerStreamer([block], device="cpu")
    assert block.weight.requires_grad and not block.weight.is_pinned()
    assert not block._forward_hooks and not block._forward_pre_hooks
