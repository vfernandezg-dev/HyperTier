# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Weights that come from safetensors files: the RAM tier, the NVMe tier and their thread.

A model larger than page-locked RAM cannot keep all its blocks there. Those that fit stay
(the *RAM tier*, read once); the rest (the *NVMe tier*) are read from disk on every pass into
a few page-locked *staging* buffers, ahead of their use, and copied to VRAM from there::

    NVMe --pread, O_DIRECT, 64 MiB requests--> staging --H2D--> VRAM slot --> block
                           RAM tier: pinned for good ----H2D-->

One thread, the reader, does the disk reads and enqueues every host-to-device copy, so a copy
can start the moment its bytes are in RAM, while the main thread is busy running the model.
Two threads on the same CUDA events need an order that holds on the host, not only on the
device, and that order is the *permit*:

* The main thread alone assigns VRAM slots. It issues a permit for (block, slot) only after
  ``slot_free`` of the slot's previous user was recorded.
* The reader serves permits first in, first out: it waits for ``slot_free``, copies, records
  ``slot_ready`` (and ``stage_free`` for the staging buffer it read from) and only then
  tells the main thread, which waits for that notice *on the host* before making compute
  wait for ``slot_ready`` on the device.
* Each event is recorded by one thread only, and a generation counter per event turns any
  wait on a stale recording into an exception instead of silently corrupt weights.

The reader never allocates GPU memory: it copies into slots reserved by the main thread.

The engine (:class:`Reader`) and the planning helpers are plain Python; PyTorch is imported
lazily, so the protocol is tested with fake streams and events on machines without a GPU.
"""

from __future__ import annotations

import bisect
import ctypes
import mmap
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from os import PathLike
from typing import TYPE_CHECKING, Any, Protocol

from hypertier.io.direct import DirectReader, aligned_span
from hypertier.io.safetensors_index import Run, SafetensorsIndex, TensorEntry
from hypertier.stream.pinned import PinnedPool, align_up
from hypertier.telemetry._common import MiB, format_bytes

if TYPE_CHECKING:
    from torch import nn

# ---------------------------------------------------------------------------------------------
# The permit and the reader thread (no PyTorch)


@dataclass
class Permit:
    """One host-to-device copy: ``block`` into VRAM slot ``slot``.

    Created by the main thread once the slot is free to be overwritten on the host timeline.
    With a reader thread it is a permit in the literal sense (``done`` is the notice that
    comes back); without one the main thread enqueues the copy itself and ``done`` is None.
    """

    block: int
    slot: int
    demand: bool  # needed right now (a miss) rather than prefetched
    nbytes: int
    free_gen: int  # generation of ``slot_free`` this copy has to wait for
    issued: float  # perf_counter() when the main thread created it
    timed: bool = False  # record timing events for telemetry
    done: threading.Event | None = None
    cancelled: bool = False  # a newer permit took the slot before this one was served
    error: BaseException | None = None
    ready_gen: int = -1  # generation of ``slot_ready`` recorded by this copy
    available: float = 0.0  # perf_counter() when the permit existed and its bytes were in RAM
    enqueued: float = 0.0  # perf_counter() once the copy was enqueued
    started: Any = None  # timing events on the copy stream
    finished: Any = None
    read_ms: float = 0.0  # disk read that fed this copy
    bytes_read: int = 0


class Driver(Protocol):
    """What the reader needs from the device and the disk; faked in the protocol tests."""

    def enter_thread(self) -> None: ...

    def exit_thread(self) -> None: ...

    def new_event(self) -> Any:
        """An event with ``query()``: has the copy that last read a staging buffer finished?"""

    def read(self, block: int, stage: int, between: Callable[[], None]) -> tuple[float, int]:
        """Read ``block`` from disk into staging buffer ``stage``; return (ms, bytes)."""

    def copy(self, permit: Permit, stage: int | None, stage_free: Any) -> None:
        """Enqueue the copy for ``permit`` (from ``stage``, or from the RAM tier if None)."""


_EMPTY, _READING, _STAGED, _COPIED = range(4)


class _Stage:
    """A staging buffer: which block it holds and whether a copy is still reading it."""

    def __init__(self, free: Any) -> None:
        self.free = free  # stage_free: recorded after the copy that read the buffer
        self.state = _EMPTY
        self.block = -1
        self.read_ms = 0.0
        self.bytes_read = 0
        self.staged_at = 0.0  # perf_counter() when its read finished


class _Stopped(Exception):
    """Raised between two read requests to abandon a read when the reader is told to stop."""


class Reader:
    """The thread that feeds the VRAM slots: reads ahead from disk and serves permits.

    In priority order it (1) serves the permit at the head of the queue when its bytes are in
    RAM, (2) reads the block that permit asks for when they are not, (3) reads ahead the next
    NVMe-tier blocks, up to ``num_stage`` of them. A long read is cut into requests and
    permits are served between them. A staging buffer is reused only once ``stage_free`` says
    the copy that read it has finished; the reader polls that instead of blocking on it, so a
    permit never waits behind a device event.

    ``tick`` is called at every step with its name; tests use it to inject delays.
    """

    def __init__(
        self,
        driver: Driver,
        num_blocks: int,
        nvme_blocks: Sequence[int],
        num_stage: int,
        *,
        poll_seconds: float = 0.0005,
        tick: Callable[[str], None] | None = None,
    ) -> None:
        if num_stage < 1:
            raise ValueError(f"num_stage must be >= 1, got {num_stage}")
        self._driver = driver
        self._num_blocks = num_blocks
        self._nvme = sorted(nvme_blocks)
        self._on_nvme = frozenset(self._nvme)
        self._stages = [_Stage(driver.new_event()) for _ in range(num_stage if self._nvme else 0)]
        self._staged: dict[int, int] = {}  # block -> staging buffer holding it, not copied yet
        self._cv = threading.Condition()
        self._permits: deque[Permit] = deque()
        self._frontier = -1  # the last block a permit was issued for
        self._stop = False
        self._error: BaseException | None = None
        self._serving: Permit | None = None
        self._device_wait = False
        self._poll = poll_seconds
        self.tick = tick or _no_tick
        self._thread = threading.Thread(target=self._run, name="hypertier-reader", daemon=True)

    # ---- called by the main thread

    def start(self) -> None:
        self._thread.start()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def issue(self, permit: Permit) -> None:
        """Hand a permit to the reader. Call only after ``slot_free`` was recorded."""
        with self._cv:
            if self._error is not None or self._stop:
                permit.error = self._error or RuntimeError("the weight reader is stopped")
                permit.done.set()
                return
            for queued in self._permits:
                if queued.slot == permit.slot:
                    queued.cancelled = True  # its slot goes to the newer block instead
            self._permits.append(permit)
            self._frontier = permit.block
            self._cv.notify()

    def wait(self, permit: Permit) -> None:
        """Block until the copy for ``permit`` is enqueued; raise what the reader raised."""
        self.settle(permit)
        if permit.error is not None:
            raise RuntimeError(
                f"the weight reader failed while loading block {permit.block}"
            ) from permit.error
        if not permit.done.is_set():
            raise RuntimeError("the weight reader thread died")

    def settle(self, permit: Permit) -> None:
        """Wait until ``permit`` is served, fails or can no longer be served; never raises."""
        while not permit.done.wait(0.25):
            if not self._thread.is_alive():
                return

    def stop(self) -> None:
        """Stop reading, end the thread and release whoever waits on a queued permit."""
        with self._cv:
            self._stop = True
            self._cv.notify()
        if self._thread.ident is not None:
            self._thread.join()
        with self._cv:
            abandoned = list(self._permits)
            self._permits.clear()
        for permit in abandoned:
            permit.cancelled = True
            permit.done.set()

    # ---- the thread

    def _run(self) -> None:
        try:
            self._driver.enter_thread()
            try:
                while True:
                    with self._cv:
                        while True:
                            if self._stop:
                                return
                            job = self._pick()
                            if job is not None:
                                break
                            self._cv.wait(self._poll if self._device_wait else None)
                    job()
            finally:
                self._driver.exit_thread()
        except _Stopped:
            pass
        except BaseException as exc:  # noqa: BLE001 - handed to the main thread
            with self._cv:
                self._error = exc
                failed = list(self._permits)
                self._permits.clear()
            if self._serving is not None:
                failed.append(self._serving)
            for permit in failed:
                permit.error = exc
                permit.done.set()

    def _pick(self) -> Callable[[], None] | None:
        """The next thing to do, chosen with the lock held; None when there is nothing."""
        self._device_wait = False
        head = self._head()
        if head is not None:
            if head.block not in self._on_nvme:
                return self._take(head, None)
            stage = self._staged.get(head.block)
            if stage is not None:
                return self._take(head, stage)
            # The block the model waits for is read before anything is read ahead.
            stage = self._stage_for(demand=True)
            return None if stage is None else partial(self._read, head.block, stage)
        for block in self._window():
            if block not in self._staged:
                stage = self._stage_for(demand=False)
                return None if stage is None else partial(self._read, block, stage)
        return None

    def _head(self) -> Permit | None:
        permits = self._permits
        while permits and permits[0].cancelled:
            permits.popleft().done.set()
        return permits[0] if permits else None

    def _take(self, permit: Permit, stage: int | None) -> Callable[[], None]:
        self._permits.popleft()
        self._serving = permit
        return partial(self._serve, permit, stage)

    def _window(self) -> list[int]:
        """The NVMe-tier blocks to have staged: the next ones after the last permit."""
        nvme = self._nvme
        if not nvme:
            return []
        at = bisect.bisect_right(nvme, self._frontier)
        return [nvme[(at + k) % len(nvme)] for k in range(min(len(self._stages), len(nvme)))]

    def _stage_for(self, *, demand: bool) -> int | None:
        """A staging buffer that may be overwritten now, or None (lock held)."""
        in_flight = False
        for index, stage in enumerate(self._stages):
            if stage.state == _COPIED:
                if stage.free.query():
                    stage.state = _EMPTY
                else:
                    in_flight = True
            if stage.state == _EMPTY:
                return index
        queued = {permit.block for permit in self._permits}
        keep = queued.union(self._window())
        staged = [i for i, stage in enumerate(self._stages) if stage.state == _STAGED]
        stale = [i for i in staged if self._stages[i].block not in keep]
        if not stale and demand:
            # Nothing is free and the model waits: give up a block read ahead, preferably
            # one nobody asked for yet. It is read again when its permit reaches the head.
            stale = [i for i in staged if self._stages[i].block not in queued] or staged
        if stale:
            index = max(stale, key=self._distance)
            del self._staged[self._stages[index].block]
            self._stages[index].state = _EMPTY
            return index
        self._device_wait = in_flight
        return None

    def _distance(self, stage: int) -> int:
        return (self._stages[stage].block - self._frontier) % self._num_blocks

    def _read(self, block: int, index: int) -> None:
        stage = self._stages[index]
        stage.block, stage.state = block, _READING
        self.tick("read")
        stage.read_ms, stage.bytes_read = self._driver.read(block, index, self._between)
        stage.staged_at = time.perf_counter()
        stage.state = _STAGED
        self._staged[block] = index

    def _between(self) -> None:
        """Between two requests of a read: serve the permits that can be served."""
        self.tick("chunk")
        while True:
            with self._cv:
                if self._stop:
                    raise _Stopped
                head = self._head()
                if head is None:
                    return
                stage = self._staged.get(head.block) if head.block in self._on_nvme else None
                if stage is None and head.block in self._on_nvme:
                    return
                self._permits.popleft()
                self._serving = head
            self._serve(head, stage)

    def _serve(self, permit: Permit, index: int | None) -> None:
        self.tick("serve")
        stage = None if index is None else self._stages[index]
        self._driver.copy(permit, index, None if stage is None else stage.free)
        permit.available = permit.issued
        if stage is not None:
            permit.available = max(permit.issued, stage.staged_at)
            permit.read_ms, permit.bytes_read = stage.read_ms, stage.bytes_read
            stage.read_ms, stage.bytes_read = 0.0, 0
            stage.state = _COPIED
            del self._staged[permit.block]
        permit.enqueued = time.perf_counter()
        self._serving = None
        self.tick("served")
        permit.done.set()  # the notice: slot_ready is recorded, the main thread may go on


def _no_tick(name: str) -> None:
    pass


# ---------------------------------------------------------------------------------------------
# Planning (no PyTorch)


def plan_tiers(
    ram_bytes: Sequence[int], stage_bytes: int, num_stage: int, ram_budget: int | None
) -> list[bool]:
    """Which blocks stay in page-locked RAM (True) and which are read from disk every pass.

    Everything stays in RAM when it fits in ``ram_budget``. Otherwise the staging buffers are
    paid first and the rest of the budget holds as many blocks as fit, evenly spread over the
    stack: every block runs once per pass whichever is chosen, and spreading them lets disk
    reads overlap the copies of the blocks in between. Provisional until the planner (F4).
    """
    count = len(ram_bytes)
    if ram_budget is None or sum(ram_bytes) <= ram_budget:
        return [True] * count
    room = ram_budget - num_stage * stage_bytes
    if room < 0:
        raise ValueError(
            f"ram_budget of {format_bytes(ram_budget)} is below the {num_stage} staging "
            f"buffer(s) of {format_bytes(stage_bytes)} that streaming from disk needs"
        )
    for keep in range(count - 1, 0, -1):
        chosen = {(2 * k + 1) * count // (2 * keep) for k in range(keep)}
        if sum(ram_bytes[i] for i in chosen) <= room:
            return [i in chosen for i in range(count)]
    return [False] * count


def merge_spans(spans: Sequence[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Join ``(run, source offset, slot offset, nbytes)`` pieces that are adjacent in both.

    A tensor is one piece. Neighbours in the file are neighbours in the slot whenever their
    sizes are multiples of the slot alignment, as in transformer layers, and then a whole
    block is a single copy.
    """
    merged: list[tuple[int, int, int, int]] = []
    for run, src, dst, nbytes in spans:
        if nbytes == 0:
            continue
        if merged:
            last_run, last_src, last_dst, last_n = merged[-1]
            if last_run == run and last_src + last_n == src and last_dst + last_n == dst:
                merged[-1] = (run, last_src, last_dst, last_n + nbytes)
                continue
        merged.append((run, src, dst, nbytes))
    return merged


@dataclass(frozen=True)
class BlockPlan:
    """Where a block's bytes are on disk, in RAM and in a slot."""

    entries: tuple[tuple[TensorEntry, int], ...]  # (tensor, slot offset), in file order
    size: int  # bytes of a slot the block uses
    reads: tuple[tuple[Run, int, int], ...]  # (run, where its aligned range goes, its length)
    spans: tuple[tuple[int, int, int], ...]  # copies: (host offset, slot offset, nbytes)
    host_bytes: int  # page-locked bytes the block takes, in RAM or in a staging buffer


def plan_block(index: SafetensorsIndex, names: Sequence[str]) -> BlockPlan:
    """Lay ``names`` out in a slot in file order and work out the reads and copies.

    Each run is read through its block-aligned range; the ranges of a block sit one after
    the other on page boundaries, in a RAM-tier region and in a staging buffer alike.
    """
    runs = index.runs(names)
    entries: list[tuple[TensorEntry, int]] = []
    reads: list[tuple[Run, int, int]] = []
    pieces: list[tuple[int, int, int, int]] = []
    slot_offset = 0
    host_offset = 0
    for number, run in enumerate(runs):
        start, length = aligned_span(run.offset, run.nbytes)
        reads.append((run, host_offset, length))
        for entry in run.entries:
            entries.append((entry, slot_offset))
            source = host_offset + entry.offset - start
            pieces.append((number, source, slot_offset, entry.nbytes))
            slot_offset = align_up(slot_offset + entry.nbytes)
        host_offset += length
    spans = tuple((src, dst, n) for _, src, dst, n in merge_spans(pieces))
    return BlockPlan(tuple(entries), slot_offset, tuple(reads), spans, host_offset)


# ---------------------------------------------------------------------------------------------
# The file-backed source


class FileWeights:
    """Take the weights of the streamed blocks from safetensors files.

    Pass it to ``LayerStreamer(blocks, weights=...)``. The modules only give the structure:
    a parameter named ``fc.weight`` in block ``i`` is the tensor ``prefixes[i] + "fc.weight"``
    of the checkpoint, and must have its shape and dtype. Whatever data the modules held is
    dropped. Outside a block's forward its parameters are ``meta`` tensors (real shape and
    dtype, no memory), so ``state_dict()`` does not return weights: they are in the file.

    Args:
        index: the checkpoint, or the path of its file or directory of shards.
        prefixes: per block, the prefix of its tensor names, e.g. ``"model.layers.3."``.
        ram_budget: page-locked RAM the weights may take, staging included. None: all the
            blocks stay in RAM (the RAM tier). With less than that, the blocks that do not
            fit are read from disk on every pass (the NVMe tier).
        num_stage: staging buffers, i.e. how many blocks the reader may be ahead. Two let a
            disk read overlap the copy of the previous block.
        chunk_bytes: size of each read request (see ``DirectReader``).
        direct: force (True) or forbid (False) ``O_DIRECT``; None uses it where possible.

    Bytes moved per pass: as with weights in memory for the copies to VRAM, plus one disk
    read of every NVMe-tier block.
    """

    def __init__(
        self,
        index: SafetensorsIndex | str | PathLike[str],
        prefixes: Sequence[str],
        *,
        ram_budget: int | None = None,
        num_stage: int = 2,
        chunk_bytes: int = 64 * MiB,
        direct: bool | None = None,
    ) -> None:
        if num_stage < 1:
            raise ValueError(f"num_stage must be >= 1, got {num_stage}")
        if ram_budget is not None and ram_budget < 0:
            raise ValueError(f"ram_budget must be >= 0, got {ram_budget}")
        self.index = index if isinstance(index, SafetensorsIndex) else SafetensorsIndex(index)
        self.prefixes = list(prefixes)
        self.ram_budget = ram_budget
        self.num_stage = num_stage
        self.chunk_bytes = chunk_bytes
        self.direct = direct

    def plan(self, torch: Any, blocks: Sequence[nn.Module]) -> list[BlockPlan]:
        """Match every block parameter with its tensor and lay the blocks out."""
        if len(self.prefixes) != len(blocks):
            raise ValueError(f"{len(self.prefixes)} prefixes for {len(blocks)} blocks")
        plans = []
        for number, (block, prefix) in enumerate(zip(blocks, self.prefixes)):
            names = []
            for local, param in block.named_parameters():
                name = prefix + local
                entry = self.index.get(name)
                if entry is None:
                    raise ValueError(f"block {number}: tensor {name!r} is not in the checkpoint")
                dtype = torch_dtype(torch, entry.dtype)
                if tuple(param.shape) != entry.shape or param.dtype != dtype:
                    raise ValueError(
                        f"block {number}: {name!r} is {entry.dtype} {list(entry.shape)} in the "
                        f"checkpoint, but the module expects {param.dtype} {list(param.shape)}"
                    )
                names.append(name)
            plans.append(plan_block(self.index, names))
        return plans


def torch_dtype(torch: Any, name: str) -> Any:
    """The torch dtype of a safetensors dtype name."""
    names = {
        "BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64",
        "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8",
        "BOOL": "bool",
    }  # fmt: skip
    if name not in names:
        raise ValueError(f"safetensors dtype {name} cannot be streamed")
    return getattr(torch, names[name])


class FileBackend:
    """Runtime of a :class:`FileWeights`: pinned regions, staging buffers and the reader.

    Built by ``LayerStreamer`` on the main thread, which also owns the slots, the events and
    their generation counters (``owner``); this object is the reader's :class:`Driver`.
    """

    def __init__(
        self, owner: Any, weights: FileWeights, plans: Sequence[BlockPlan], pool: PinnedPool
    ) -> None:
        torch = owner._torch
        self._owner = owner
        self._torch = torch
        self._plans = plans
        self._index = weights.index
        self._io = DirectReader(chunk_bytes=weights.chunk_bytes, direct=weights.direct)
        self._thread_stream: Any = None
        self._keep: list[Any] = []  # the pinned tensors behind the memoryviews
        host = [plan.host_bytes for plan in plans]
        in_ram = plan_tiers(host, max(host), weights.num_stage, weights.ram_budget)
        self.tiers = tuple("ram" if keep else "nvme" for keep in in_ram)
        nvme = [i for i, keep in enumerate(in_ram) if not keep]
        stage_bytes = max((host[i] for i in nvme), default=0)
        self.ram_bytes = sum(host[i] for i, keep in enumerate(in_ram) if keep)
        self.ram_bytes += weights.num_stage * stage_bytes if nvme else 0

        # RAM tier: read once, straight into page-locked memory, and kept.
        self._ram_src: dict[int, list[Any]] = {}
        for i, keep in enumerate(in_ram):
            if keep:
                region = pool.alloc(host[i], align=mmap.PAGESIZE)
                self._fill(i, self._bytes_of(region))
                self._ram_src[i] = [region[src : src + n] for src, _, n in plans[i].spans]
        # NVMe tier: every view of every staging buffer is made here, on the main thread.
        self._stage_bytes: list[memoryview] = []
        self._stage_src: list[dict[int, list[Any]]] = []
        for _ in range(weights.num_stage if nvme else 0):
            buffer = pool.alloc(stage_bytes, align=mmap.PAGESIZE)
            self._stage_bytes.append(self._bytes_of(buffer))
            self._stage_src.append(
                {i: [buffer[src : src + n] for src, _, n in plans[i].spans] for i in nvme}
            )
        self.reader = Reader(self, len(plans), nvme, weights.num_stage)

    # ---- Driver: these run on the reader thread

    def enter_thread(self) -> None:
        torch = self._torch
        torch.cuda.set_device(self._owner._device)
        # Every CUDA call of this thread happens on the copy stream.
        self._thread_stream = torch.cuda.stream(self._owner._copy_stream)
        self._thread_stream.__enter__()

    def exit_thread(self) -> None:
        if self._thread_stream is not None:
            self._thread_stream.__exit__(None, None, None)
            self._thread_stream = None

    def new_event(self) -> Any:
        return self._torch.cuda.Event(enable_timing=False)

    def read(self, block: int, stage: int, between: Callable[[], None]) -> tuple[float, int]:
        t0 = time.perf_counter()
        nbytes = self._fill(block, self._stage_bytes[stage], between)
        return (time.perf_counter() - t0) * 1e3, nbytes

    def copy(self, permit: Permit, stage: int | None, stage_free: Any) -> None:
        owner = self._owner
        slot, block = permit.slot, permit.block
        if owner._free_gen[slot] != permit.free_gen:
            raise RuntimeError(
                f"slot {slot}: slot_free is at generation {owner._free_gen[slot]}, but the "
                f"permit for block {block} was issued at {permit.free_gen}"
            )
        stream = owner._copy_stream
        stream.wait_event(owner._free[slot])  # the copy waits for compute
        if permit.timed:
            permit.started = owner._events.record(stream)
        sources = self._ram_src[block] if stage is None else self._stage_src[stage][block]
        for dst, src in zip(owner._slot_dst[slot][block], sources):
            dst.copy_(src, non_blocking=True)
        owner._ready[slot].record(stream)
        owner._ready_gen[slot] += 1
        permit.ready_gen = owner._ready_gen[slot]
        if stage_free is not None:
            stage_free.record(stream)
        if permit.timed:
            permit.finished = owner._events.record(stream)

    # ---- main thread

    def close(self) -> None:
        self.reader.stop()
        self._io.close()
        for view in self._stage_bytes:
            view.release()
        self._stage_bytes.clear()
        self._stage_src.clear()
        self._ram_src.clear()
        self._keep.clear()

    def read_buffer(self, name: str) -> Any | None:
        """A persistent buffer of the checkpoint as a CPU tensor, or None if it has none."""
        entry = self._index.get(name)
        if entry is None:
            return None
        with open(entry.path, "rb") as f:
            f.seek(entry.offset)
            data = bytearray(f.read(entry.nbytes))
        torch = self._torch
        dtype = torch_dtype(torch, entry.dtype)
        if not data:
            return torch.empty(entry.shape, dtype=dtype)
        return torch.frombuffer(data, dtype=dtype).view(entry.shape).clone()

    def _fill(
        self, block: int, target: memoryview, between: Callable[[], None] | None = None
    ) -> int:
        nbytes = 0
        for run, base, length in self._plans[block].reads:
            piece = target[base : base + length]
            try:
                self._io.read_into(piece, run.path, run.offset, run.nbytes, between=between)
            finally:
                piece.release()
            nbytes += run.nbytes
        return nbytes

    def _bytes_of(self, tensor: Any) -> memoryview:
        """A writable memoryview over a pinned tensor, which a file can be read into."""
        self._keep.append(tensor)
        if tensor.numel() == 0:
            return memoryview(bytearray())
        raw = (ctypes.c_ubyte * tensor.numel()).from_address(tensor.data_ptr())
        return memoryview(raw).cast("B")


__all__ = ["BlockPlan", "FileBackend", "FileWeights", "Permit", "Reader", "plan_tiers"]
