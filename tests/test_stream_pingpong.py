# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.stream.pingpong: the slot policy, plain Python that runs everywhere."""

import subprocess
import sys

import pytest

from hypertier.stream.pingpong import Load, SlotTable

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
