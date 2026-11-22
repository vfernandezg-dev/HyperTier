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

import statistics
import threading
import time
import warnings
from collections import deque
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from hypertier.stream.pinned import ALIGN, PinnedPool, align_up
from hypertier.stream.source import FileBackend, FileWeights, Permit
from hypertier.telemetry._common import (
    MiB,
    format_bytes,
    import_torch,
    require_cuda,
    resolve_device,
)
from hypertier.telemetry.bandwidth import overlap_efficiency, percentile

if TYPE_CHECKING:
    import torch
    from torch import nn
    from typing_extensions import Self

_OVERSIZE_RATIO = 1.5  # warn when the largest block exceeds this many times the median
_MIN_PRIVATE_SEGMENT = MiB  # tiny models: do not lock one segment per tiny block
_MAX_PENDING_PASSES = 64  # telemetry of passes the GPU has not finished yet
_HISTORY_PASSES = 1024

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


# ---------------------------------------------------------------------------------------------
# Telemetry


@dataclass(frozen=True)
class PassStats:
    """One pass over the blocks, timed with CUDA events on the device timeline.

    ``efficiency`` is an upper bound. The times are taken while copies and compute overlap,
    so when a copy in flight slows the kernels down (a few percent on native Linux, far more
    on WSL2: issue #1), that slowdown is inside ``compute_ms`` and the figure cannot see it:
    it only says how much of the shorter side the pipeline scheduled behind the longer one.
    The metric to trust is the one ``benchmarks/bench_streaming.py`` reports, which compares
    the streamed pass with the same pass on a model that is fully resident in VRAM.
    """

    blocks: int  # block forwards in the pass
    copies: int  # host-to-device copies enqueued during the pass
    misses: int  # of those, blocks loaded on demand instead of prefetched
    bytes_copied: int
    copy_ms: float  # sum of the copies (copy stream)
    compute_ms: float  # sum of the block forwards (compute stream)
    stall_ms: float  # time the compute stream waited for a slot to be ready
    total_ms: float  # first block reached -> last block or last copy finished
    # overlap_efficiency(copy, compute, total): an upper bound (see above); None without copies
    efficiency: float | None
    # Weights read from files (zero when they were already in memory):
    read_ms: float = 0.0  # disk reads that fed the copies of this pass
    bytes_read: int = 0
    read_stall_ms: float = 0.0  # host time the model waited for the reader thread
    # From asking for a copy to having it enqueued, per block. With a reader thread this
    # includes waiting for a disk read that is not finished, which nothing can shorten, and
    # the rare wait of a timing event behind a blocking read (see ``_TimingEvents``).
    enqueue_ms: float = 0.0  # median
    enqueue_p99_ms: float = 0.0
    # The part of it spent with the bytes already in RAM. With a reader thread, it was busy
    # with something else (a read request for another block); without one, this is just the
    # time the main thread takes to enqueue a copy.
    handoff_ms: float = 0.0  # median
    handoff_p99_ms: float = 0.0


@dataclass
class _PassRecord:
    """Events of a pass that the GPU may still be executing."""

    compute: list[list[Any]] = field(default_factory=list)  # [reached, started, finished]
    copies: list[Permit] = field(default_factory=list)  # asked for during the pass
    read_stall_ms: float = 0.0


class _TimingEvents:
    """Recycles timing events: a pass needs five per block, every pass.

    Shared with the reader thread, which times the copies it enqueues. Recording a timing
    event can wait for as long as another thread stays in a blocking read of a GPU tensor
    (``.item()``, ``.to("cpu")``); events without timing and the copies themselves do not.
    So with telemetry on, a copy asked for while the model's caller waits for a result (the
    next token, in ``generate()``) may be enqueued only when that result arrives. Measured on
    Qwen3-8B: it happened in 2 of 31 generated tokens (hand-off p99 of those passes 681 ms)
    and neither took longer for it, since with two slots that copy could not have started
    before the pass ended anyway.
    """

    def __init__(self, torch: Any) -> None:
        self._torch = torch
        self._idle: list[Any] = []
        self._lock = threading.Lock()

    def record(self, stream: Any) -> Any:
        with self._lock:
            event = self._idle.pop() if self._idle else None
        if event is None:
            event = self._torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event

    def recycle(self, events: Iterable[Any]) -> None:
        with self._lock:
            self._idle.extend(events)


# ---------------------------------------------------------------------------------------------
# Streamer


def find_blocks(model: nn.Module) -> nn.ModuleList:
    """The model's stack of blocks: its largest ``ModuleList`` by parameter bytes.

    Raises ``ValueError`` when there is none, or when a second, unrelated list is at least
    half as large (e.g. the double- and single-stream stacks of some diffusion transformers):
    guessing wrong would leave gigabytes of weights off the GPU or out of the stream. Pass the
    blocks explicitly then, concatenating the lists in the order they run.
    """
    torch = import_torch()
    found = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) >= 2:
            nbytes = sum(p.numel() * p.element_size() for p in module.parameters())
            if nbytes:
                found.append((nbytes, name, module))
    if not found:
        raise ValueError("no ModuleList with parameters found; pass the blocks explicitly")
    found.sort(key=lambda item: item[0], reverse=True)
    nbytes, name, best = found[0]
    inside = name + "." if name else ""
    rivals = [c for c in found[1:] if not c[1].startswith(inside) and 2 * c[0] >= nbytes]
    if rivals:
        listing = ", ".join(
            f"{n or '<root>'} ({len(m)} blocks, {format_bytes(b)})"
            for b, n, m in [found[0], *rivals]
        )
        raise ValueError(
            f"ambiguous block list: {listing}; pass the blocks explicitly, in the order they run"
        )
    return best


class LayerStreamer:
    """Stream the weights of ``blocks`` through a few VRAM slots while the model runs.

    Construction moves the blocks' parameters into page-locked RAM (one block at a time, so
    RAM peaks near the model size plus one block), reserves ``slots`` VRAM buffers of the size
    of the largest block and installs forward hooks. The model itself is not rewritten: call
    it as usual, under ``torch.no_grad()`` or ``torch.inference_mode()``, with its inputs and
    its non-streamed modules on ``device``. Before a block runs, each of its parameters is
    swapped for a ``Parameter`` that views the slot holding its weights; afterwards the
    original one, whose data lives in RAM, is put back, so ``state_dict()`` is always correct.
    See the module docstring for the streams and events that order copies against compute.

    Bytes moved per pass: every block that is not already in a slot, i.e. all the streamed
    weights once per pass when there are more blocks than slots, and nothing otherwise.

    Args:
        blocks: the modules to stream, in the order they run (see :func:`find_blocks`).
        slots: VRAM buffers. The prefetch runs ``slots - 1`` blocks ahead; two is the classic
            ping-pong. Capped at the number of blocks.
        device: the CUDA device that computes.
        pool: page-locked RAM to take the weights from. Default: a private pool that grows as
            blocks move in. Pass one to pay the pinning cost once across several models.
        telemetry: time every pass with CUDA events (see :meth:`pass_stats`).
        weights: read the weights from safetensors files instead of taking them from the
            modules (:class:`~hypertier.stream.source.FileWeights`). Blocks that do not fit in
            its RAM budget are read from disk on every pass by a reader thread, which exists
            only then, and a block's parameters are ``meta`` placeholders outside its forward.

    While attached, streamed parameters have ``requires_grad=False`` (a slot is overwritten
    right after its block ran, so gradients could not be computed), and the blocks' buffers
    live on ``device``: they are small and a model may write to them.

    Limits of this core:

    * Inference only, and one host thread drives the model.
    * Do not move or convert the streamed blocks while attached (``model.to(...)``,
      ``.half()``): that replaces the tensors the slots are fed from, and ``.to("cuda")`` would
      load every block into VRAM at once. Place the rest of the model on ``device`` instead.
    * Not compatible with CUDA Graphs: a captured graph replays kernels with the slot
      addresses of the capture and skips the Python hooks, so nothing repoints parameters or
      enqueues the copies and event waits.
    * Not compatible with ``torch.compile``: compiled code assumes that parameter storage does
      not change between calls, and the hooks change it on every block.
    * Blocks must be distinct modules that share no parameters with each other, and a block
      must not return a view of its own weights (the slot is reused after it finishes).
    * A module must reach its weights through its attributes (``self.weight``), as modules
      do. One that keeps its own references to the ``Parameter`` objects would compute with
      the copies in RAM; ``torch.nn`` RNNs do keep such a list, but refresh it when an
      attribute is assigned, which is how their parameters are swapped.
    * A largest block far above the median wastes VRAM in every slot; a warning says so.
    """

    def __init__(
        self,
        blocks: Iterable[nn.Module],
        *,
        slots: int = 2,
        device: torch.device | str = "cuda",
        pool: PinnedPool | None = None,
        telemetry: bool = True,
        weights: FileWeights | None = None,
    ) -> None:
        torch = require_cuda()
        modules = list(blocks)
        if not modules:
            raise ValueError("blocks is empty")
        if any(not isinstance(m, torch.nn.Module) for m in modules):
            raise TypeError("blocks must be torch.nn.Module instances")
        if len({id(m) for m in modules}) != len(modules):
            raise ValueError("blocks must be distinct modules; one appears twice")
        if slots < 1:
            raise ValueError(f"slots must be >= 1, got {slots}")
        dev = resolve_device(device)
        if dev.type != "cuda":
            raise ValueError(f"device must be a CUDA device, got {dev}")

        self.num_blocks = len(modules)
        self.num_slots = 0
        self.slot_bytes = 0
        """Size of one VRAM slot: the largest block, parameters aligned to 256 bytes."""
        self.vram_bytes = 0
        """VRAM held while attached: the slots plus the blocks' buffers."""
        self.streamed_bytes = 0
        """Bytes of weights streamed through the slots on demand."""
        self.ram_bytes = 0
        """Page-locked RAM the weights take: RAM-tier blocks plus staging buffers."""
        self.tiers: tuple[str, ...] = ()
        """Per block, ``"ram"`` (kept in page-locked RAM) or ``"nvme"`` (read every pass)."""

        self._torch = torch
        self._device = dev
        self._blocks = modules
        self._telemetry = telemetry
        self._closed = False
        self._pool: PinnedPool | None = None
        self._owns_pool = pool is None
        self._leased = False
        self._sizes: list[int] = []
        self._params: list[list[Any]] = []
        self._host_views: list[list[Any]] = []
        self._regions: list[Any] = []
        self._src: list[list[Any]] = []  # per block, what the main thread copies from
        self._weights = weights
        self._backend: FileBackend | None = None
        self._reader: Any = None  # the thread that enqueues copies, when blocks come from disk
        self._free_gen: list[int] = []  # how many times slot_free / slot_ready were recorded
        self._ready_gen: list[int] = []
        self._slot_permit: list[Permit | None] = []  # the copy last asked for, per slot
        self._grad_flags: list[tuple[Any, bool]] = []
        self._moved_buffers: list[tuple[Any, str, Any]] = []
        self._slots: list[Any] = []
        self._bind: list[list[_Swap]] = []  # per slot, per block: its slot parameters
        self._unbind: list[_Swap] = []  # per block: the parameters that live in RAM
        self._slot_dst: list[list[Any]] = []
        self._copy_stream: Any = None
        self._ready: list[Any] = []
        self._free: list[Any] = []
        self._hooks: list[Any] = []
        self._table: SlotTable | None = None
        self._running: dict[int, list[Any]] = {}
        self._events = _TimingEvents(torch)
        self._pass: _PassRecord | None = None
        self._last_block = -1
        self._pending: deque[_PassRecord] = deque()
        self._history: deque[PassStats] = deque(maxlen=_HISTORY_PASSES)
        try:
            # Outside inference mode, whatever the caller's mode: an inference tensor could
            # not back a parameter that is later used in a normal forward.
            with torch.inference_mode(False), torch.no_grad():
                self._build(slots, pool, weights)
        except BaseException:
            self._teardown()
            raise

    # -----------------------------------------------------------------------------------------
    # Public

    def pass_stats(self) -> list[PassStats]:
        """Timings of the passes run so far, oldest first (the last 1024 are kept).

        A pass ends when the model wraps around to an earlier block, or here if no block is
        running. Waits for the GPU to finish the work already enqueued, since the events of a
        pass can only be read once they have happened. Empty with ``telemetry=False``.
        """
        if self._pass is not None and not self._running:
            self._pending.append(self._pass)
            self._pass = None
            self._last_block = -1
        self._harvest(wait=True)
        return list(self._history)

    def close(self) -> None:
        """Detach from the model and free the VRAM slots (idempotent).

        Waits for the copy stream and for the kernels still reading the slots before anything
        is released. Afterwards the parameters are ordinary CPU tensors with their original
        ``requires_grad``, and buffers are back on the device they came from. With a private
        pool the parameters stay where they are, without a copy: the pool is closed, which
        unlocks its pages, and the memory is freed with the last parameter that uses it. With
        a caller's pool they are copied to pageable memory, so the pool can be reset for the
        next model.
        """
        if self._closed:
            return
        if self._running:
            raise RuntimeError("close() called while a streamed block is running")
        self._teardown()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -----------------------------------------------------------------------------------------
    # Construction and teardown

    def _build(self, slots: int, pool: PinnedPool | None, weights: FileWeights | None) -> None:
        torch = self._torch
        dev = self._device
        plans = None if weights is None else weights.plan(torch, self._blocks)
        if plans is None:
            layouts, sizes = _plan_layout(torch, self._blocks)
        else:
            layouts, sizes = _file_layout(self._blocks, plans)
        largest = max(sizes)
        median = statistics.median(sizes)
        if median > 0 and largest > _OVERSIZE_RATIO * median:
            warnings.warn(
                f"the largest streamed block ({format_bytes(largest)}) is "
                f"{largest / median:.1f}x the median ({format_bytes(median)}); every VRAM slot "
                f"is sized for it, so each one wastes about {format_bytes(largest - median)} "
                "most of the time. Split that block or keep it out of the streamed list.",
                stacklevel=3,
            )
        table = SlotTable(len(sizes), slots)
        self._table = table
        self._sizes = sizes
        self.num_slots = table.num_slots
        self.slot_bytes = largest
        self.streamed_bytes = sum(sizes)

        if pool is None:
            # Grows by one block at a time, each segment exactly as large as its block.
            pool = PinnedPool(0, segment_bytes=_MIN_PRIVATE_SEGMENT)
        pool._attach()
        self._pool = pool
        self._leased = True

        where = [_locations(block) for block in self._blocks]
        if plans is None:
            self._pin_modules(layouts, sizes, pool)
            self.tiers = ("ram",) * len(sizes)
            self.ram_bytes = sum(sizes)
            # At rest a block keeps its own Parameter objects, now backed by the pool.
            at_rest = [[param for param, _, _ in entries] for entries in layouts]
        else:
            # Reads the RAM tier into the pool and reserves the staging buffers.
            self._backend = FileBackend(self, weights, plans, pool)
            self.tiers, self.ram_bytes = self._backend.tiers, self._backend.ram_bytes
            if self._backend.reader is None:
                # Nothing comes from disk: this thread enqueues every copy, as it does with
                # weights in memory, and no copy waits to be handed to another thread.
                self._src = [self._backend.ram_sources(i) for i in range(len(sizes))]
            # At rest a block has no memory at all: meta tensors of the real shape and dtype.
            at_rest = [[_placeholder(torch, p) for p, _, _ in entries] for entries in layouts]
            self._load_buffers(weights.prefixes)
        self._unbind = [
            _swap(torch, [(where[i][id(p)], rest) for (p, _, _), rest in zip(entries, at_rest[i])])
            for i, entries in enumerate(layouts)
        ]
        for swap in self._unbind:
            _apply(swap)  # with weights from files, this drops whatever the modules held

        resident = self._move_buffers()

        self._slots = [
            torch.empty(largest, dtype=torch.uint8, device=dev) for _ in range(table.num_slots)
        ]
        if largest and any(slot.data_ptr() % ALIGN for slot in self._slots):
            raise RuntimeError(f"VRAM slot is not {ALIGN}-byte aligned")
        # Every (slot, block) parameter is made here, once: the hooks only swap them in, and
        # the reader thread only copies into the views below. Neither allocates on the GPU.
        self._bind = [
            [
                _swap(
                    torch,
                    [
                        (where[i][id(p)], _slot_parameter(torch, slot, off, n, rest))
                        for (p, off, n), rest in zip(entries, at_rest[i])
                    ],
                )
                for i, entries in enumerate(layouts)
            ]
            for slot in self._slots
        ]
        if plans is None:
            self._slot_dst = [[[slot[:size]] for size in sizes] for slot in self._slots]
        else:
            self._slot_dst = [
                [[slot[dst : dst + n] for _, dst, n in plan.spans] for plan in plans]
                for slot in self._slots
            ]
        self.vram_bytes = table.num_slots * largest + resident

        self._copy_stream = torch.cuda.Stream(device=dev)
        # Without timing: these only order the two streams, which is cheaper to wait on.
        self._ready = [torch.cuda.Event(enable_timing=False) for _ in self._slots]
        self._free = [torch.cuda.Event(enable_timing=False) for _ in self._slots]
        self._free_gen = [0] * table.num_slots
        self._ready_gen = [0] * table.num_slots
        self._slot_permit = [None] * table.num_slots

        for index, block in enumerate(self._blocks):
            self._hooks.append(
                block.register_forward_pre_hook(partial(self._enter, index), prepend=True)
            )
            # always_call: the slot must be released even when the block raises.
            self._hooks.append(
                block.register_forward_hook(partial(self._exit, index), always_call=True)
            )
        if self._backend is not None and self._backend.reader is not None:
            self._reader = self._backend.reader
            self._reader.start()
        for load in table.prime():
            self._issue(load, None)

    def _pin_modules(self, layouts: list[Any], sizes: list[int], pool: PinnedPool) -> None:
        """Move the modules' own weights into the pool, laid out like a slot."""
        # One block at a time: assigning ``param.data`` drops the last reference to the
        # original storage, so RAM never holds much more than the model plus one block.
        for entries, size in zip(layouts, sizes):
            region = pool.alloc(size)
            params: list[Any] = []
            views: list[Any] = []
            self._regions.append(region)
            self._src.append([region])
            self._params.append(params)
            self._host_views.append(views)
            for param, offset, nbytes in entries:
                view = _typed_view(region, offset, nbytes, param)
                view.copy_(param.data)
                self._grad_flags.append((param, param.requires_grad))
                param.requires_grad_(False)
                param.data = view
                params.append(param)
                views.append(view)

    def _load_buffers(self, prefixes: Any) -> None:
        """Take the blocks' persistent buffers from the checkpoint; they stay on the device."""
        for number, (block, prefix) in enumerate(zip(self._blocks, prefixes)):
            if not isinstance(prefix, str):
                continue  # tensors given one by one: buffers have no name to look up
            for path, module in block.named_modules():
                for key, buf in module._buffers.items():
                    name = f"{prefix}{path}.{key}" if path else prefix + key
                    stored = None if buf is None else self._backend.read_buffer(name)
                    if stored is None:
                        continue
                    if stored.shape != buf.shape or stored.dtype != buf.dtype:
                        raise ValueError(
                            f"block {number}: buffer {name!r} is {stored.dtype} "
                            f"{list(stored.shape)} in the checkpoint, but the module expects "
                            f"{buf.dtype} {list(buf.shape)}"
                        )
                    self._moved_buffers.append((module, key, buf.device))
                    module._buffers[key] = stored.to(self._device)

    def _move_buffers(self) -> int:
        """Make the blocks' buffers resident on the device; return their size in bytes."""
        dev = self._device
        moved: dict[int, tuple[Any, Any]] = {}  # id -> (original kept alive, device copy)
        resident: dict[int, int] = {}
        for block in self._blocks:
            for module in block.modules():
                for key, buf in module._buffers.items():
                    if buf is None:
                        continue
                    if buf.device != dev:
                        if id(buf) not in moved:
                            moved[id(buf)] = (buf, buf.to(dev))
                        self._moved_buffers.append((module, key, buf.device))
                        buf = moved[id(buf)][1]
                        module._buffers[key] = buf
                    resident[id(buf)] = buf.numel() * buf.element_size()
        return sum(resident.values())

    def _teardown(self) -> None:
        """Undo whatever ``_build`` did; also the path taken when construction fails."""
        torch = self._torch
        self._closed = True
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        if self._backend is not None:
            self._backend.stop()  # no copy is enqueued after this
        if self._copy_stream is not None:
            # A queued copy still reads the pool and writes a slot, and queued kernels still
            # read the slots: nothing below may be released before both streams are idle.
            self._copy_stream.synchronize()
            torch.cuda.synchronize(self._device)
            if self._pass is not None:
                self._pending.append(self._pass)
                self._pass = None
            self._harvest(wait=True)
        if not self._owns_pool:
            # The caller may reset the pool for another model: take the weights out of it.
            with torch.inference_mode(False), torch.no_grad():
                for params, views in zip(self._params, self._host_views):
                    for param, view in zip(params, views):
                        param.data = view.clone()
        for param, flag in self._grad_flags:
            param.requires_grad_(flag)
        for module, key, device in self._moved_buffers:
            buf = module._buffers.get(key)
            if buf is not None:
                module._buffers[key] = buf.to(device)
        for state in (
            self._grad_flags, self._moved_buffers, self._params, self._host_views,
            self._regions, self._slots, self._bind, self._unbind, self._slot_dst, self._ready,
            self._free, self._src, self._slot_permit,
        ):
            state.clear()
        if self._backend is not None:
            self._backend.close()  # after the streams are idle: copies read its buffers
            self._backend = None
        self._reader = None
        self._copy_stream = None
        if self._leased and self._pool is not None:
            self._pool._detach()
            self._leased = False
            if self._owns_pool:
                # Unlocks the pages; the parameters keep the memory itself alive.
                self._pool.close()
        self._pool = None

    # -----------------------------------------------------------------------------------------
    # Hot path: these run around every block and only enqueue work

    def _enter(self, index: int, module: Any, args: Any) -> None:
        run = self._running.get(index)
        if run is not None:  # the block calls itself: its weights are already in place
            run[0] += 1
            return
        slot, loads = self._table.enter(index)
        compute = self._torch.cuda.current_stream(self._device)
        record = None
        if self._telemetry:
            if self._pass is None or index <= self._last_block:
                self._roll_pass()
            self._last_block = index
            record = self._pass
        for load in loads:
            self._issue(load, record)
        permit = self._slot_permit[slot]
        if permit.done is not None:
            # Host order first: the reader must have recorded slot_ready before compute is
            # told to wait for it, or it would wait for the slot's previous copy.
            late = not permit.done.is_set()
            t0 = time.perf_counter() if late else 0.0
            self._reader.wait(permit)
            if late and record is not None:
                record.read_stall_ms += (time.perf_counter() - t0) * 1e3
        if permit.ready_gen != self._ready_gen[slot]:
            raise RuntimeError(
                f"slot {slot}: slot_ready is at generation {self._ready_gen[slot]}, but the "
                f"copy of block {index} recorded {permit.ready_gen}"
            )
        timing = None
        if record is not None:
            reached = self._events.record(compute)
            compute.wait_event(self._ready[slot])  # compute waits for the copy
            timing = [reached, self._events.record(compute), None]
            record.compute.append(timing)
        else:
            compute.wait_event(self._ready[slot])
        _apply(self._bind[slot][index])
        self._running[index] = [1, slot, timing]

    def _exit(self, index: int, module: Any, args: Any, output: Any) -> None:
        run = self._running.get(index)
        if run is None:  # the pre-hook raised before the block got a slot
            return
        run[0] -= 1
        if run[0]:
            return
        del self._running[index]
        _, slot, timing = run
        compute = self._torch.cuda.current_stream(self._device)
        if timing is not None:
            timing[2] = self._events.record(compute)
        # Every kernel of the block is enqueued by now: the slot is free once they finish.
        self._free[slot].record(compute)
        self._free_gen[slot] += 1  # before the table can hand the slot to another block
        _apply(self._unbind[index])
        self._table.exit(index)

    def _issue(self, load: Load, record: _PassRecord | None) -> None:
        """Ask for the copy of a block into its slot, whose last user has recorded slot_free.

        With every block in RAM (weights in memory, or read from files into the RAM tier) the
        main thread enqueues it here. When some block is read from disk it becomes a permit
        for the reader thread, which enqueues it once the bytes are in RAM.
        """
        slot, nbytes = load.slot, self._sizes[load.block]
        permit = Permit(
            load.block, slot, load.demand, nbytes, self._free_gen[slot], time.perf_counter(),
            timed=record is not None and nbytes > 0,
        )
        self._slot_permit[slot] = permit
        if record is not None:
            record.copies.append(permit)
        if self._reader is not None:
            permit.done = threading.Event()
            self._reader.issue(permit)
            return
        stream = self._copy_stream
        stream.wait_event(self._free[slot])  # the copy waits for compute
        if permit.timed:
            permit.started = self._events.record(stream)
        if nbytes:
            with self._torch.cuda.stream(stream):
                for dst, src in zip(self._slot_dst[slot][load.block], self._src[load.block]):
                    dst.copy_(src, non_blocking=True)
        self._ready[slot].record(stream)
        self._ready_gen[slot] += 1
        permit.ready_gen = self._ready_gen[slot]
        if permit.timed:
            permit.finished = self._events.record(stream)
        permit.available = permit.issued  # the bytes were in RAM all along
        permit.enqueued = time.perf_counter()

    # -----------------------------------------------------------------------------------------
    # Telemetry

    def _roll_pass(self) -> None:
        if self._pass is not None:
            self._pending.append(self._pass)
        self._pass = _PassRecord()
        self._harvest(wait=False)

    def _harvest(self, *, wait: bool) -> None:
        """Turn finished passes into stats; without ``wait`` only those the GPU completed."""
        pending = self._pending
        while pending:
            # The host can enqueue many passes ahead of the GPU; past the cap, wait for the
            # oldest instead of holding its events forever.
            if not self._finish(pending[0], wait or len(pending) > _MAX_PENDING_PASSES):
                break
            pending.popleft()

    def _finish(self, record: _PassRecord, wait: bool) -> bool:
        # A copy asked for at the end of a pass may not have been enqueued by the reader yet.
        unserved = [c for c in record.copies if c.done is not None and not c.done.is_set()]
        if unserved and not wait:
            return False
        for permit in unserved:
            self._reader.settle(permit)
        copies = [c for c in record.copies if c.finished is not None]
        compute = [t for t in record.compute if t[2] is not None]
        tails = [t[2] for t in compute[-1:]]
        if copies:
            tails.append(max(copies, key=lambda c: c.enqueued).finished)
        if wait:
            for event in tails:
                event.synchronize()
        elif not all(event.query() for event in tails):
            return False
        if compute:
            start = compute[0][0]
            copy_ms = sum(c.started.elapsed_time(c.finished) for c in copies)
            compute_ms = sum(started.elapsed_time(done) for _, started, done in compute)
            total_ms = max(start.elapsed_time(event) for event in tails)
            efficiency = None
            if copy_ms > 0 and compute_ms > 0:
                efficiency = overlap_efficiency(copy_ms, compute_ms, total_ms)
            latency = [(c.enqueued - c.issued) * 1e3 for c in copies]
            handoff = [(c.enqueued - c.available) * 1e3 for c in copies]
            self._history.append(
                PassStats(
                    blocks=len(compute),
                    copies=len(copies),
                    misses=sum(c.demand for c in copies),
                    bytes_copied=sum(c.nbytes for c in copies),
                    copy_ms=copy_ms,
                    compute_ms=compute_ms,
                    stall_ms=sum(reached.elapsed_time(started) for reached, started, _ in compute),
                    total_ms=total_ms,
                    efficiency=efficiency,
                    read_ms=sum(c.read_ms for c in copies),
                    bytes_read=sum(c.bytes_read for c in copies),
                    read_stall_ms=record.read_stall_ms,
                    enqueue_ms=percentile(latency, 50.0) if latency else 0.0,
                    enqueue_p99_ms=percentile(latency, 99.0) if latency else 0.0,
                    handoff_ms=percentile(handoff, 50.0) if handoff else 0.0,
                    handoff_p99_ms=percentile(handoff, 99.0) if handoff else 0.0,
                )
            )
        self._events.recycle(e for t in record.compute for e in t if e is not None)
        self._events.recycle(e for c in copies for e in (c.started, c.finished))
        return True


def _plan_layout(
    torch: Any, modules: list[Any]
) -> tuple[list[list[tuple[Any, int, int]]], list[int]]:
    """Per block, ``(parameter, byte offset, byte size)`` and the block's total size.

    The same layout is used in RAM and in every slot, so a block is a single contiguous copy.
    """
    owner: dict[int, int] = {}
    layouts: list[list[tuple[Any, int, int]]] = []
    sizes: list[int] = []
    for index, block in enumerate(modules):
        entries: list[tuple[Any, int, int]] = []
        offset = 0
        for name, param in block.named_parameters():
            if id(param) in owner:
                raise ValueError(
                    f"parameter {name!r} of block {index} also belongs to block "
                    f"{owner[id(param)]}; streamed blocks cannot share parameters"
                )
            owner[id(param)] = index
            if param.layout is not torch.strided:
                raise TypeError(f"parameter {name!r} of block {index} is not a dense tensor")
            nbytes = param.numel() * param.element_size()
            entries.append((param, offset, nbytes))
            offset = align_up(offset + nbytes)
        layouts.append(entries)
        sizes.append(offset)
    return layouts, sizes


def _file_layout(
    modules: list[Any], plans: Any
) -> tuple[list[list[tuple[Any, int, int]]], list[int]]:
    """The layout of :func:`_plan_layout`, in the order the tensors have in their files."""
    owner: dict[int, int] = {}
    layouts: list[list[tuple[Any, int, int]]] = []
    for index, (block, plan) in enumerate(zip(modules, plans)):
        params = dict(block.named_parameters())
        entries = []
        for name, (tensor, offset) in zip(plan.names, plan.entries):
            param = params[name]
            if owner.setdefault(id(param), index) != index:
                raise ValueError(
                    f"parameter {tensor.name!r} of block {index} also belongs to block "
                    f"{owner[id(param)]}; streamed blocks cannot share parameters"
                )
            entries.append((param, offset, tensor.nbytes))
        layouts.append(entries)
    return layouts, [plan.size for plan in plans]


def _placeholder(torch: Any, param: Any) -> Any:
    """What a file-backed parameter is while its block is not loaded: shape, dtype, no memory."""
    if param.device.type == "meta":
        return param
    empty = torch.empty(param.shape, dtype=param.dtype, device="meta")
    return torch.nn.Parameter(empty, requires_grad=False)


def _typed_view(buffer: Any, offset: int, nbytes: int, like: Any) -> Any:
    """``nbytes`` of a uint8 buffer seen as a tensor with ``like``'s dtype and shape."""
    return buffer[offset : offset + nbytes].view(like.dtype).view(like.shape)


def _slot_parameter(torch: Any, slot: Any, offset: int, nbytes: int, like: Any) -> Any:
    """A frozen Parameter over the bytes of a VRAM slot where ``like``'s weights land."""
    return torch.nn.Parameter(_typed_view(slot, offset, nbytes, like), requires_grad=False)


# Binding a block's weights means putting other Parameter objects in its modules, not changing
# ``param.data``: a parameter can then be anything, including a ``meta`` placeholder with no
# memory, which ``param.data`` cannot switch to or from.
_Swap = tuple[list[tuple[dict[str, Any], str, Any]], list[tuple[Any, str, Any]]]


def _locations(block: Any) -> dict[int, list[tuple[Any, str]]]:
    """For every parameter of ``block``, each (module, attribute name) that holds it."""
    found: dict[int, list[tuple[Any, str]]] = {}
    for module in block.modules():
        for name, param in module._parameters.items():
            if param is not None:
                found.setdefault(id(param), []).append((module, name))
    return found


def _swap(torch: Any, placements: list[tuple[list[tuple[Any, str]], Any]]) -> _Swap:
    """Precompute how to install ``param`` at every place of ``placements``.

    Most modules take a direct write to ``_parameters``. A class that overrides
    ``__setattr__`` does so to track its parameters (``torch.nn`` RNNs keep a flat list of
    them), so those are assigned through it.
    """
    direct: list[tuple[dict[str, Any], str, Any]] = []
    through_setattr: list[tuple[Any, str, Any]] = []
    for places, param in placements:
        for module, name in places:
            if type(module).__setattr__ is torch.nn.Module.__setattr__:
                direct.append((module._parameters, name, param))
            else:
                through_setattr.append((module, name, param))
    return direct, through_setattr


def _apply(swap: _Swap) -> None:
    for parameters, name, param in swap[0]:
        parameters[name] = param
    for module, name, param in swap[1]:
        setattr(module, name, param)
