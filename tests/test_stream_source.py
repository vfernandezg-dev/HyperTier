# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.stream.source: weights read from safetensors files.

The planning and the reader's protocol are plain Python: they run in CI against fake streams
and events that check the order of every recording and wait. Reading real files into real
slots needs the GPU.
"""

import copy
import random
import threading
import time

import pytest
from toys import checkpoint_tensors, resident_outputs, toy_mlp, toy_transformer

from hypertier.io import SafetensorsIndex
from hypertier.stream import FileWeights, LayerStreamer
from hypertier.stream.pingpong import SlotTable
from hypertier.stream.pinned import ALIGN
from hypertier.stream.source import Permit, Reader, merge_spans, plan_block, plan_tiers
from hypertier.telemetry import MiB

PAGE = 4096

# ---------------------------------------------------------------------------------------------
# Planning (no PyTorch)


def test_everything_stays_in_ram_when_it_fits():
    assert plan_tiers([10, 20, 30], 30, 2, None) == [True, True, True]
    assert plan_tiers([10, 20, 30], 30, 2, 60) == [True, True, True]


def test_a_partial_budget_pays_staging_first_and_spreads_the_rest():
    sizes = [10] * 7
    # 2 staging buffers of 10, and room for 3 more blocks: every other one, not the first 3.
    assert plan_tiers(sizes, 10, 2, 50) == [False, True, False, True, False, True, False]
    assert plan_tiers(sizes, 10, 2, 30) == [False, False, False, True, False, False, False]
    assert plan_tiers(sizes, 10, 2, 20) == [False] * 7  # staging only: all from disk
    assert plan_tiers(sizes, 10, 1, 69) == [True, False, True, True, True, False, True]
    assert plan_tiers([10, 50, 10, 50, 10], 50, 1, 75) == [False, False, True, False, False]
    with pytest.raises(ValueError, match="staging"):
        plan_tiers(sizes, 10, 2, 19)


def test_spans_merge_only_what_is_adjacent_on_both_sides():
    assert merge_spans([(0, 100, 0, 256), (0, 356, 256, 512)]) == [(0, 100, 0, 768)]
    # A tensor whose size is no multiple of the slot alignment leaves a gap in the slot.
    assert merge_spans([(0, 0, 0, 300), (0, 300, 512, 256)]) == [(0, 0, 0, 300), (0, 300, 512, 256)]
    assert merge_spans([(0, 0, 0, 256), (0, 300, 256, 256)]) == [(0, 0, 0, 256), (0, 300, 256, 256)]
    assert merge_spans([(0, 0, 0, 256), (1, 256, 256, 256)]) == [(0, 0, 0, 256), (1, 256, 256, 256)]
    assert merge_spans([(0, 0, 0, 0), (0, 0, 0, 256)]) == [(0, 0, 0, 256)]


def blob(seed: int, nbytes: int) -> bytes:
    return bytes((seed * 7 + i) % 251 for i in range(nbytes))


def assemble(plan) -> bytearray:
    """Do what the streamer does with a plan, in plain Python: read, then copy to a slot."""
    host = bytearray(plan.host_bytes)
    for run, base, length in plan.reads:
        start = run.offset // PAGE * PAGE
        assert base % PAGE == 0 and length % PAGE == 0 and start <= run.offset
        with open(run.path, "rb") as f:
            f.seek(start)
            data = f.read(length)
        host[base : base + len(data)] = data
    slot = bytearray(plan.size)
    for src, dst, nbytes in plan.spans:
        slot[dst : dst + nbytes] = host[src : src + nbytes]
    return slot


@pytest.mark.parametrize("sizes", [(512, 256, 1024), (300, 256, 1000, 8), (4, 4, 4)])
def test_a_planned_block_puts_every_tensor_where_the_slot_expects_it(
    tmp_path, safetensors_writer, sizes
):
    other = {"other.weight": ("U8", (700,), blob(99, 700))}
    block = {f"b.{i}": ("U8", (n,), blob(i, n)) for i, n in enumerate(sizes)}
    path = safetensors_writer(tmp_path / "m.safetensors", {**other, **block})
    plan = plan_block(SafetensorsIndex(path), list(block))

    assert [entry.name for entry, _ in plan.entries] == list(block)
    assert all(offset % ALIGN == 0 for _, offset in plan.entries)
    assert len(plan.reads) == 1 and plan.host_bytes % PAGE == 0
    aligned = all(n % ALIGN == 0 for n in sizes)
    assert (len(plan.spans) == 1) == aligned, "one copy only when nothing needs padding"
    slot = assemble(plan)
    for (entry, offset), (_, _, data) in zip(plan.entries, block.values()):
        assert bytes(slot[offset : offset + entry.nbytes]) == data


def test_a_block_split_over_two_shards_is_two_reads(tmp_path, safetensors_writer):
    first = {"b.0": ("U8", (512,), blob(1, 512)), "b.1": ("U8", (256,), blob(2, 256))}
    second = {"b.2": ("U8", (768,), blob(3, 768))}
    safetensors_writer(tmp_path / "a.safetensors", first)
    safetensors_writer(tmp_path / "b.safetensors", second)
    plan = plan_block(SafetensorsIndex(tmp_path), ["b.0", "b.1", "b.2"])
    assert len(plan.reads) == 2 and len(plan.spans) == 2
    assert plan.reads[1][1] == plan.reads[0][2], "the second range starts on the next page"
    slot = assemble(plan)
    assert bytes(slot) == b"".join(data for _, _, data in {**first, **second}.values())
    assert plan_block(SafetensorsIndex(tmp_path), []).host_bytes == 0


# ---------------------------------------------------------------------------------------------
# The reader's protocol against fake streams and events (no PyTorch)


class FakeEvent:
    """stage_free: true once the copy that read the staging buffer would have finished."""

    def __init__(self) -> None:
        self.done_at = 0.0

    def query(self) -> bool:
        return time.perf_counter() >= self.done_at


class FakeDevice:
    """Slots, staging buffers and disk as plain lists; every call checks the host order."""

    def __init__(self, num_slots: int, rng: random.Random, pace: float) -> None:
        self.free_gen = [0] * num_slots  # recorded by the main thread
        self.ready_gen = [0] * num_slots  # recorded by the reader
        self.in_slot: list = [None] * num_slots
        self.in_stage: dict = {}
        self.busy: dict = {}
        self.reads: list = []
        self.copies: list = []
        self.threads: set = set()
        self.fail_read_of = None
        self._rng = rng
        self._pace = pace

    def nap(self, scale: float = 1.0) -> None:
        time.sleep(self._rng.random() * self._pace * scale)

    def enter_thread(self) -> None:
        self.threads.add(threading.get_ident())

    def exit_thread(self) -> None:
        pass

    def new_event(self) -> FakeEvent:
        return FakeEvent()

    def read(self, block, stage, between):
        if block == self.fail_read_of:
            raise OSError(5, "simulated I/O error")
        event = self.busy.get(stage)
        assert event is None or event.query(), "overwrote staging while a copy still read it"
        self.in_stage[stage] = None  # being overwritten
        for _ in range(3):
            self.nap()
            between()
        self.in_stage[stage] = block
        self.reads.append(block)
        return 1.0, 100

    def copy(self, permit, stage, stage_free):
        assert threading.get_ident() in self.threads
        assert self.free_gen[permit.slot] == permit.free_gen, "waited for a stale slot_free"
        if stage is not None:
            assert self.in_stage[stage] == permit.block, "copied the wrong staging buffer"
            stage_free.done_at = time.perf_counter() + self._rng.random() * self._pace * 4
            self.busy[stage] = stage_free
        self.nap(0.3)
        self.in_slot[permit.slot] = permit.block
        self.ready_gen[permit.slot] += 1
        permit.ready_gen = self.ready_gen[permit.slot]
        self.copies.append((permit.block, permit.slot))


class Model:
    """The main thread's side of the protocol, as the streamer's hooks perform it."""

    def __init__(self, num_blocks, slots, num_stage, nvme, *, seed=0, pace=0.0002, start=True):
        self.rng = random.Random(seed)
        self.table = SlotTable(num_blocks, slots)
        self.device = FakeDevice(self.table.num_slots, self.rng, pace)
        self.reader = Reader(self.device, num_blocks, nvme, num_stage, tick=self.tick)
        self.permit_of: list = [None] * self.table.num_slots
        if start:
            self.reader.start()
        for load in self.table.prime():
            self.issue(load)

    def tick(self, name: str) -> None:
        if self.rng.random() < 0.3:
            self.device.nap()

    def issue(self, load) -> Permit:
        permit = Permit(
            load.block, load.slot, load.demand, 1, self.device.free_gen[load.slot],
            time.perf_counter(), done=threading.Event(),
        )
        self.permit_of[load.slot] = permit
        self.reader.issue(permit)
        return permit

    def run_block(self, block: int) -> None:
        slot, loads = self.table.enter(block)
        for load in loads:
            self.issue(load)
        self.tick("pre_hook")
        permit = self.permit_of[slot]
        assert permit.done.wait(5), f"deadlock: block {block} was never copied"
        self.reader.wait(permit)  # raises what the reader raised
        assert permit.ready_gen == self.device.ready_gen[slot], "would wait for a stale slot_ready"
        assert self.device.in_slot[slot] == block, "the slot holds another block's weights"
        self.device.nap(2.0)  # the block computes
        assert self.device.in_slot[slot] == block, "the slot was overwritten while it ran"
        self.device.free_gen[slot] += 1  # slot_free.record: only now may the slot be reassigned
        self.tick("post_hook")
        assert self.table.exit(block) == slot


TIERS = {
    "nvme": lambda n: list(range(n)),
    "mixed": lambda n: list(range(0, n, 2)),
    "ram": lambda n: [],
}


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("num_stage", [1, 2])
@pytest.mark.parametrize("slots", [1, 2, 3])
@pytest.mark.parametrize("num_blocks", [1, 3, 7])
def test_protocol_keeps_every_wait_behind_its_recording(num_blocks, slots, num_stage, tier):
    nvme = TIERS[tier](num_blocks)
    seed = num_blocks * 100 + slots * 10 + num_stage
    model = Model(num_blocks, slots, num_stage, nvme, seed=seed)
    try:
        for _ in range(12):
            for block in range(num_blocks):
                model.run_block(block)
    finally:
        model.reader.stop()
    assert not model.reader.alive
    assert len(model.device.threads) == 1, "one thread enqueues every copy"
    if num_blocks > model.table.num_slots:
        on_disk = set(nvme)
        assert {block for block in model.device.reads} == on_disk
        # Every block is copied once per pass; nothing in RAM is ever read from disk.
        assert len(model.device.copies) >= 12 * num_blocks - model.table.num_slots


@pytest.mark.parametrize("seed", range(6))
def test_protocol_survives_jumps_and_repeats(seed):
    nvme = TIERS["mixed" if seed % 2 else "nvme"](7)
    model = Model(7, 2 + seed % 2, 1 + seed % 3, nvme, seed=seed)
    rng = random.Random(seed)
    try:
        for _ in range(150):
            block = rng.randrange(7) if rng.random() < 0.3 else (model.table._tick + 1) % 7
            model.run_block(block)
    finally:
        model.reader.stop()


def test_the_block_the_model_waits_for_is_read_before_any_read_ahead():
    model = Model(7, 2, 2, list(range(7)), start=False)  # permits for blocks 0 and 1 are queued
    queued = list(model.permit_of)
    model.reader.start()
    assert all(permit.done.wait(5) for permit in queued)
    model.reader.stop()
    # The window after the last permit is blocks 2 and 3, yet 0 and 1 were read first.
    assert model.device.reads[:2] == [0, 1], "demand reads come first, in permit order"


def test_a_permit_is_served_while_a_staging_buffer_is_still_in_flight():
    # Blocks 0 and 2 come from disk through a single staging buffer; block 1 is in RAM.
    model = Model(3, 1, 1, [0, 2], start=False, pace=0.0)
    model.reader.start()
    model.run_block(0)  # copied from staging: the buffer is in flight from here on
    event = model.device.busy[0]
    event.done_at = time.perf_counter() + 0.4  # a long copy
    t0 = time.perf_counter()
    model.run_block(1)  # RAM tier: must not wait for the staging buffer
    assert time.perf_counter() - t0 < 0.2, "the reader blocked on stage_free"
    model.run_block(2)  # needs the buffer: this one does wait for the copy to finish
    assert time.perf_counter() >= event.done_at
    model.reader.stop()


def test_a_newer_permit_for_the_same_slot_cancels_the_older_one():
    model = Model(7, 1, 2, [], start=False)  # one slot; the permit for block 0 is queued
    stale = model.permit_of[0]
    _, loads = model.table.enter(3)  # the model jumps before the reader ever ran
    newer = model.issue(loads[0])
    assert stale.cancelled and not newer.cancelled
    model.reader.start()
    assert newer.done.wait(5) and stale.done.is_set()
    model.reader.stop()
    assert model.device.copies == [(3, 0)], "the superseded copy was never made"


def test_a_read_error_reaches_the_main_thread_and_later_permits_fail_fast():
    model = Model(7, 2, 2, list(range(7)), pace=0.0)
    model.device.fail_read_of = 4
    with pytest.raises(RuntimeError, match="weight reader failed") as caught:
        for block in range(7):
            model.run_block(block)
    assert isinstance(caught.value.__cause__, OSError)
    assert not model.reader.alive
    late = model.issue(model.table.enter(6)[1][0]) if model.table.slot_of(6) is None else None
    if late is not None:
        assert late.done.is_set() and isinstance(late.error, OSError)
    model.reader.stop()  # still fine to stop


def test_stop_abandons_a_read_and_releases_whoever_waits():
    model = Model(7, 2, 2, list(range(7)), pace=0.01)
    model.run_block(0)
    waiting = Permit(5, 0, True, 1, 0, time.perf_counter(), done=threading.Event())
    model.reader.issue(waiting)
    model.reader.stop()
    assert not model.reader.alive and waiting.done.is_set()
    with pytest.raises(ValueError):
        Reader(model.device, 7, [0], 0)


# ---------------------------------------------------------------------------------------------
# Real files, real slots


def save(directory, model, writer, *, shards: int = 1):
    """Write ``model`` as a checkpoint of ``shards`` files and return its directory."""
    tensors = list(checkpoint_tensors(model).items())
    per_shard = -(-len(tensors) // shards)
    for number in range(shards):
        part = dict(tensors[number * per_shard : (number + 1) * per_shard])
        writer(directory / f"model-{number:05d}.safetensors", part)
    return directory


def block_prefixes(model) -> list[str]:
    return [f"blocks.{i}." for i in range(len(model.blocks))]


def budget(directory, model, tier: str, num_stage: int) -> int | None:
    """A RAM budget that puts the checkpoint in the given tier."""
    plans = FileWeights(directory, block_prefixes(model)).plan(_torch(), model.blocks)
    host = max(plan.host_bytes for plan in plans)
    return {"ram": None, "nvme": num_stage * host, "mixed": (num_stage + 3) * host}[tier]


def _torch():
    import torch

    return torch


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
@pytest.mark.parametrize(
    ("tier", "num_stage", "slots", "num_blocks", "shards"),
    [
        ("ram", 2, 2, 7, 1), ("mixed", 2, 2, 7, 1), ("nvme", 2, 2, 7, 1), ("nvme", 1, 2, 7, 1),
        ("nvme", 2, 1, 7, 1), ("nvme", 3, 3, 7, 3), ("mixed", 1, 3, 8, 2), ("nvme", 2, 2, 3, 1),
        ("nvme", 2, 2, 2, 1), ("nvme", 1, 2, 1, 1),
    ],
)  # fmt: skip
def test_weights_from_files_are_bit_exact_in_every_tier(
    disk_tmp, safetensors_writer, tier, num_stage, slots, num_blocks, shards, mode
):
    import torch

    model = toy_transformer(num_blocks)  # odd sizes: blocks need several copies each
    x = torch.randn(4, 33, 100, device="cuda")
    expected = resident_outputs(model, x, passes=3, mode=mode)
    save(disk_tmp, model, safetensors_writer, shards=shards)
    model.norm.cuda()
    limit = budget(disk_tmp, model, tier, num_stage)
    weights = FileWeights(disk_tmp, block_prefixes(model), ram_budget=limit, num_stage=num_stage)
    with LayerStreamer(model.blocks, slots=slots, weights=weights) as streamer:
        if num_blocks > 3:
            assert set(streamer.tiers) == {"ram", "nvme"} if tier == "mixed" else {tier}
        if limit is not None:
            assert streamer.ram_bytes <= limit
            assert streamer._pool.capacity_bytes <= limit + MiB  # the pool's own granularity
        with getattr(torch, mode)():
            y = x
            for want in expected:
                y = model(y)
                assert torch.equal(y, want)


@pytest.mark.cuda
def test_parameters_are_meta_placeholders_outside_their_block(disk_tmp, safetensors_writer):
    import torch

    model = toy_transformer(7)
    shapes = [(tuple(p.shape), p.dtype) for p in model.blocks.parameters()]
    x = torch.randn(2, 9, 100, device="cuda")
    (expected,) = resident_outputs(model, x, passes=1)
    save(disk_tmp, model, safetensors_writer)
    model.norm.cuda()
    seen = []

    def inspect(module, args):
        seen.append([(p.device.type, p.data_ptr() % ALIGN) for p in module.parameters()])

    weights = FileWeights(disk_tmp, block_prefixes(model), ram_budget=None)
    with LayerStreamer(model.blocks, weights=weights) as streamer:
        at_rest = list(model.blocks.parameters())
        assert all(p.device.type == "meta" and not p.requires_grad for p in at_rest)
        assert [(tuple(p.shape), p.dtype) for p in at_rest] == shapes
        handles = [block.register_forward_pre_hook(inspect) for block in model.blocks]
        with torch.no_grad():
            assert torch.equal(model(x), expected)
        for handle in handles:
            handle.remove()
        assert all(where == ("cuda", 0) for block in seen for where in block)
        assert all(a is b for a, b in zip(at_rest, model.blocks.parameters()))
        assert streamer.streamed_bytes == sum(streamer._sizes)
    # Closed: the model keeps its structure and no weights; they are still in the file.
    assert all(p.device.type == "meta" for p in model.blocks.parameters())
    assert all(not b._forward_hooks and not b._forward_pre_hooks for b in model.blocks)


@pytest.mark.cuda
def test_the_reader_thread_never_allocates_gpu_memory(disk_tmp, safetensors_writer):
    import torch

    model = toy_mlp(7, 256)
    x = torch.randn(64, 256, device="cuda")
    save(disk_tmp, model, safetensors_writer)
    limit = budget(disk_tmp, model, "nvme", 2)
    weights = FileWeights(disk_tmp, block_prefixes(model), ram_budget=limit)
    with LayerStreamer(model.blocks, weights=weights) as streamer, torch.no_grad():
        assert set(streamer.tiers) == {"nvme"}
        model(x)  # the model's own activations are cached by the allocator after one pass
        torch.cuda.synchronize()
        reserved, allocated = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
        for _ in range(30):
            out = model(x)
            del out
            assert torch.cuda.memory_reserved() == reserved
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        names = [thread.name for thread in threading.enumerate()]
        assert names.count("hypertier-reader") == 1
    assert "hypertier-reader" not in [thread.name for thread in threading.enumerate()]


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("tier", "num_stage", "slots"),
    [
        ("nvme", 1, 1), ("nvme", 1, 2), ("nvme", 2, 2), ("nvme", 3, 3),
        ("mixed", 2, 2), ("ram", 2, 2),
    ],
)
def test_random_delays_in_both_threads_never_change_a_bit(
    disk_tmp, safetensors_writer, tier, num_stage, slots
):
    import torch

    rng = random.Random(f"{tier}{num_stage}{slots}")
    model = toy_mlp(7, 64)
    reference = copy.deepcopy(model).cuda()
    x = torch.randn(5, 64, device="cuda")
    save(disk_tmp, model, safetensors_writer)

    def nap(*_):
        if rng.random() < 0.25:
            time.sleep(rng.random() * 0.002)

    limit = budget(disk_tmp, model, tier, num_stage)
    weights = FileWeights(disk_tmp, block_prefixes(model), ram_budget=limit, num_stage=num_stage)
    with LayerStreamer(model.blocks, slots=slots, weights=weights) as streamer, torch.no_grad():
        want = reference(x)
        streamer._reader.tick = nap  # before a read, between requests, around every copy
        for block in model.blocks:
            block.register_forward_pre_hook(nap, prepend=True)  # before the streamer's hook
            block.register_forward_hook(nap)  # and after it
        for number in range(300):
            if number % 25 == 24:  # now and then, out of order
                order = [rng.randrange(7) for _ in range(5)]
                y, target = x, x
                for index in order:
                    y, target = model.blocks[index](y), reference.blocks[index](target)
                assert torch.equal(y, target), f"pass {number}, order {order}"
            else:
                assert torch.equal(model(x), want), f"pass {number}"
        stats = streamer.pass_stats()
    assert len(stats) >= 300 and sum(s.copies for s in stats) > 0


@pytest.mark.cuda
def test_pass_stats_account_for_reads_and_hand_off(disk_tmp, safetensors_writer):
    import torch

    model = toy_mlp(7, 256)
    x = torch.randn(64, 256, device="cuda")
    save(disk_tmp, model, safetensors_writer)
    on_disk = sum(p.numel() * p.element_size() for p in model.blocks[0].parameters())
    for tier in ("nvme", "ram"):
        limit = budget(disk_tmp, model, tier, 2)
        weights = FileWeights(disk_tmp, block_prefixes(model), ram_budget=limit)
        with LayerStreamer(copy.deepcopy(model).blocks, weights=weights) as streamer:
            blocks = streamer._blocks
            with torch.no_grad():
                for _ in range(4):
                    y = x
                    for block in blocks:
                        y = block(y)
            stats = streamer.pass_stats()[1:]
        assert all(s.copies == 7 and s.misses == 0 for s in stats)
        for s in stats:
            assert 0 <= s.handoff_ms <= s.enqueue_ms <= s.enqueue_p99_ms
            assert s.handoff_p99_ms <= s.enqueue_p99_ms
            if tier == "nvme":
                assert s.bytes_read == 7 * on_disk and s.read_ms > 0
            else:
                assert s.bytes_read == 0 and s.read_ms == 0 and s.read_stall_ms < 50


@pytest.mark.cuda
def test_a_failing_read_surfaces_in_the_forward_and_close_still_works(
    disk_tmp, safetensors_writer, monkeypatch
):
    import torch

    from hypertier.io import DirectReader

    model = toy_mlp(7, 64)
    x = torch.randn(5, 64, device="cuda")
    save(disk_tmp, model, safetensors_writer)
    limit = budget(disk_tmp, model, "nvme", 2)
    weights = FileWeights(disk_tmp, block_prefixes(model), ram_budget=limit)
    streamer = LayerStreamer(model.blocks, weights=weights)
    real = DirectReader.read_into
    calls = []

    def failing(self, *args, **kwargs):
        calls.append(1)
        if len(calls) > 3:
            raise OSError(5, "simulated I/O error")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(DirectReader, "read_into", failing)
    with torch.no_grad(), pytest.raises(RuntimeError, match="weight reader failed") as caught:
        for _ in range(3):
            model(x)
    assert isinstance(caught.value.__cause__, OSError)
    assert all(p.device.type == "meta" for p in model.blocks.parameters())
    streamer.close()
    streamer.close()
    assert "hypertier-reader" not in [thread.name for thread in threading.enumerate()]


@pytest.mark.cuda
def test_persistent_buffers_are_taken_from_the_checkpoint(disk_tmp, safetensors_writer):
    import torch
    from torch import nn

    class Stack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(nn.BatchNorm1d(16) for _ in range(3))

    torch.manual_seed(0)
    trained = Stack().eval()
    for block in trained.blocks:
        block.running_mean.normal_()
        block.running_var.uniform_(0.5, 2.0)
        block.weight.data.normal_()
    save(disk_tmp, trained, safetensors_writer)
    reference = copy.deepcopy(trained).cuda()
    skeleton = Stack().eval()  # default statistics: the real ones are only in the file
    x = torch.randn(8, 16, device="cuda")
    weights = FileWeights(disk_tmp, block_prefixes(skeleton))
    with LayerStreamer(skeleton.blocks, weights=weights), torch.no_grad():
        y, want = x, x
        for block, ref in zip(skeleton.blocks, reference.blocks):
            y, want = block(y), ref(want)
        assert torch.equal(y, want)


@pytest.mark.cuda
def test_checkpoints_that_do_not_match_the_blocks_are_rejected(disk_tmp, safetensors_writer):
    from torch import nn

    model = toy_mlp(3, 64)
    save(disk_tmp, model, safetensors_writer)
    prefixes = block_prefixes(model)
    originals = list(model.blocks.parameters())

    def attach(blocks, names, **options):
        return LayerStreamer(blocks, weights=FileWeights(disk_tmp, names, **options))

    with pytest.raises(ValueError, match="2 prefixes for 3 blocks"):
        attach(model.blocks, prefixes[:2])
    with pytest.raises(ValueError, match="not in the checkpoint"):
        attach(model.blocks, ["layers.0.", "layers.1.", "layers.2."])
    wider = nn.ModuleList(copy.deepcopy(block) for block in model.blocks)
    wider[1].fc1 = nn.Linear(64, 32)
    with pytest.raises(ValueError, match="the module expects"):
        attach(wider, prefixes)
    with pytest.raises(ValueError, match="the module expects"):
        attach(copy.deepcopy(model.blocks).half(), prefixes)
    with pytest.raises(ValueError, match="staging"):
        attach(model.blocks, prefixes, ram_budget=PAGE)
    with pytest.raises(ValueError, match="num_stage"):
        FileWeights(disk_tmp, prefixes, num_stage=0)
    # Nothing was attached: the modules still hold their own parameters, and no thread runs.
    assert all(a is b for a, b in zip(originals, model.blocks.parameters()))
    assert "hypertier-reader" not in [thread.name for thread in threading.enumerate()]
