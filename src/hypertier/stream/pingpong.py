# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Ping-pong layer streaming: run a stack of blocks with only a few of them in VRAM.

The weights of every block stay in page-locked RAM. VRAM holds ``slots`` buffers, each as large
as the biggest block, and while block ``i`` computes in one slot the next block is copied into
another on a dedicated CUDA stream. A pass over the model therefore costs about
``max(compute, bytes / link_bandwidth)`` instead of their sum, with ``slots x block`` of VRAM.

Two CUDA events per slot order the two streams on the device, so the host thread only enqueues
work and never blocks::

    compute stream   | block 0 @A | block 1 @B | block 2 @A | block 0 @B | block 1 @A | ...
    copy stream    0->A 1->B      |    2->A    |    0->B    |    1->A    |    2->B    | ...

    slot_ready[s]  recorded on the copy stream after the copy into slot s;
                   the compute stream waits for it before running the block in s.
    slot_free[s]   recorded on the compute stream after the block in s;
                   the copy stream waits for it before overwriting s.

The example has three blocks and two slots: the block-to-slot mapping is not fixed (it repeats
every six blocks), which is why slots are assigned by LRU and never by ``i % slots``. With an
odd block count the prefetch that wraps from the last block to block 0 would land on the slot
being computed.

The slot policy (:class:`SlotTable`) is plain Python and importing this module never imports
PyTorch, so the policy is tested on machines without it.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass

# ---------------------------------------------------------------------------------------------
# Slot policy (no PyTorch)


@dataclass(frozen=True)
class Load:
    """One host-to-device copy to enqueue: ``block`` goes into VRAM slot ``slot``."""

    block: int
    slot: int
    demand: bool  # True: needed right now (a miss); False: prefetched ahead of its use


class SlotTable:
    """Which block is in which VRAM slot, and which slot the next block may take.

    Blocks are expected in cyclic order (0 .. N-1, then 0 again: autoregressive decoding and
    diffusion steps walk the stack over and over), so entering block ``i`` asks for the next
    ``num_slots - 1`` blocks to be loaded behind it. A new block takes an empty slot or else
    the least recently used one, never a slot whose block is running and never one holding a
    block of the current prefetch window. That last rule matters from four slots up: plain LRU
    would evict a block prefetched earlier, before its first use, in favour of the block that
    just finished.

    Any other order still works: a block that is not loaded is a *demand* load (a miss), and
    stale prefetches lose their protection as soon as the window moves.
    """

    def __init__(self, num_blocks: int, num_slots: int) -> None:
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        self.num_blocks = num_blocks
        self.num_slots = min(num_slots, num_blocks)
        self._block: list[int | None] = [None] * self.num_slots
        self._slot: dict[int, int] = {}
        self._active = [0] * self.num_slots
        self._used = [0] * self.num_slots
        self._tick = 0

    def slot_of(self, block: int) -> int | None:
        """Slot holding ``block`` (loaded or with its copy enqueued), if any."""
        return self._slot.get(block)

    def block_in(self, slot: int) -> int | None:
        return self._block[slot]

    def window(self, block: int) -> list[int]:
        """Blocks to keep loaded while ``block`` runs: the next ones in cyclic order."""
        return [(block + d) % self.num_blocks for d in range(1, self.num_slots)]

    def prime(self, block: int = 0) -> list[Load]:
        """Loads that fill the slots before the first pass, starting at ``block``."""
        self._check(block)
        targets = [block, *self.window(block)]
        return self._prefetch(targets, keep=targets)

    def enter(self, block: int) -> tuple[int, list[Load]]:
        """``block`` is about to run: return its slot and the copies to enqueue, in order."""
        self._check(block)
        window = self.window(block)
        loads: list[Load] = []
        slot = self._slot.get(block)
        if slot is None:
            slot = self._victim(window)
            if slot is None:  # the window fills every idle slot: give up one prefetch
                slot = self._victim(())
            if slot is None:
                raise RuntimeError(
                    f"no VRAM slot for block {block}: all {self.num_slots} are running a block"
                )
            self._place(block, slot)
            loads.append(Load(block, slot, demand=True))
        else:
            self._touch(slot)
        self._active[slot] += 1
        loads += self._prefetch(window, keep=window)
        return slot, loads

    def exit(self, block: int) -> int | None:
        """``block`` finished: return its slot once no call of it is running any more."""
        slot = self._slot.get(block)
        if slot is None or not self._active[slot]:
            return None
        self._active[slot] -= 1
        return None if self._active[slot] else slot

    # -----------------------------------------------------------------------------------------

    def _check(self, block: int) -> None:
        if not 0 <= block < self.num_blocks:
            raise IndexError(f"block {block} out of range [0, {self.num_blocks})")

    def _touch(self, slot: int) -> None:
        self._tick += 1
        self._used[slot] = self._tick

    def _place(self, block: int, slot: int) -> None:
        previous = self._block[slot]
        if previous is not None:
            del self._slot[previous]
        self._block[slot] = block
        self._slot[block] = slot
        self._touch(slot)

    def _victim(self, keep: Collection[int]) -> int | None:
        """An empty slot, else the least recently used one that is idle and not in ``keep``."""
        best: int | None = None
        for slot in range(self.num_slots):
            if self._active[slot]:
                continue
            block = self._block[slot]
            if block is None:
                return slot
            if block in keep:
                continue
            if best is None or self._used[slot] < self._used[best]:
                best = slot
        return best

    def _prefetch(self, targets: Iterable[int], keep: Collection[int]) -> list[Load]:
        loads = []
        for block in targets:
            if block in self._slot:
                continue
            slot = self._victim(keep)
            if slot is None:
                break
            self._place(block, slot)
            loads.append(Load(block, slot, demand=False))
        return loads
