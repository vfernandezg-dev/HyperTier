# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.io.direct (plain Python and an anonymous mmap: they run everywhere)."""

import mmap
import os
import random

import pytest

from hypertier.io import DirectReader, aligned_span
from hypertier.telemetry.bandwidth import filesystem_type

ALIGN = 4096


@pytest.fixture
def weights(disk_tmp):
    """A file of 1 MiB + 1234 bytes on a real disk: not a multiple of the block size."""
    data = os.urandom((1 << 20) + 1234)
    path = disk_tmp / "weights.bin"
    path.write_bytes(data)
    return path, data


def page_buffer(nbytes: int) -> memoryview:
    """Writable, page-aligned memory, like a pinned staging buffer."""
    return memoryview(mmap.mmap(-1, max(nbytes, ALIGN)))


def test_aligned_span():
    assert aligned_span(0, 0) == (0, 0)
    assert aligned_span(0, 1) == (0, ALIGN)
    assert aligned_span(ALIGN, ALIGN) == (ALIGN, ALIGN)
    assert aligned_span(ALIGN - 1, 2) == (0, 2 * ALIGN)
    assert aligned_span(120, 385_892_864) == (0, 385_896_448)  # a Qwen3-8B layer at +120
    assert aligned_span(1000, 24, align=512) == (512, 512)
    with pytest.raises(ValueError):
        aligned_span(-1, 4)


def test_aligned_span_is_the_smallest_aligned_cover():
    rng = random.Random(0)
    for _ in range(2000):
        align = rng.choice((512, 4096))
        offset, nbytes = rng.randrange(10**7), rng.randrange(1, 10**6)
        start, length = aligned_span(offset, nbytes, align)
        assert start % align == 0 and length % align == 0
        assert start <= offset and offset + nbytes <= start + length
        assert offset - start < align and start + length - (offset + nbytes) < align


@pytest.mark.parametrize("direct", [None, False])
@pytest.mark.parametrize(
    ("offset", "nbytes"), [(0, 4096), (1, 1), (8 + 77_777 * 2 + 1, 300_000), (4096, 8192)]
)
def test_unaligned_ranges_land_at_the_returned_shift(weights, direct, offset, nbytes):
    path, data = weights
    start, length = aligned_span(offset, nbytes)
    buffer = page_buffer(length)
    with DirectReader(direct=direct) as reader:
        shift = reader.read_into(buffer, path, offset, nbytes)
        assert shift == offset - start
        assert bytes(buffer[shift : shift + nbytes]) == data[offset : offset + nbytes]
        assert reader.bytes_read >= shift + nbytes and reader.seconds > 0
        if direct is False:
            assert not reader.is_direct(path)
        elif filesystem_type(path) in ("ext4", "xfs", "btrfs"):
            assert reader.is_direct(path), "these filesystems support O_DIRECT"


def test_the_tail_of_the_file_is_read_through_a_range_that_passes_its_end(weights):
    path, data = weights
    offset, nbytes = len(data) - 5000, 5000
    buffer = page_buffer(aligned_span(offset, nbytes)[1])
    with DirectReader() as reader:
        shift = reader.read_into(buffer, path, offset, nbytes)
        assert bytes(buffer[shift : shift + nbytes]) == data[-5000:]
        with pytest.raises(EOFError):
            reader.read_into(buffer, path, offset, nbytes + 1)


def test_long_reads_are_split_into_requests_with_a_callback_between_them(weights):
    path, data = weights
    offset, nbytes = 100, 10 * ALIGN
    buffer = page_buffer(aligned_span(offset, nbytes)[1])
    calls = []
    with DirectReader(chunk_bytes=ALIGN) as reader:
        shift = reader.read_into(buffer, path, offset, nbytes, between=lambda: calls.append(1))
    assert bytes(buffer[shift : shift + nbytes]) == data[offset : offset + nbytes]
    assert len(calls) == 11  # the eleven blocks that cover 100 .. 100 + 40960

    def stop():
        raise KeyboardInterrupt

    with DirectReader(chunk_bytes=ALIGN) as reader, pytest.raises(KeyboardInterrupt):
        reader.read_into(buffer, path, offset, nbytes, between=stop)  # how a reader is stopped


def test_bad_buffers_and_sizes_are_rejected(weights):
    path, _ = weights
    with pytest.raises(ValueError, match="chunk_bytes"):
        DirectReader(chunk_bytes=1000)
    with DirectReader() as reader:
        assert reader.read_into(page_buffer(ALIGN)[:0], path, 4096 + 7, 0) == 7  # nothing to read
        with pytest.raises(ValueError, match="cannot hold"):
            reader.read_into(page_buffer(ALIGN), path, 10, ALIGN)
        with pytest.raises(ValueError, match="read-only"):
            reader.read_into(memoryview(bytes(ALIGN)), path, 0, 16)
        if reader.is_direct(path):
            with pytest.raises(ValueError, match="not aligned"):
                reader.read_into(page_buffer(3 * ALIGN)[8:], path, 0, ALIGN)
    reader.close()  # idempotent


def test_forcing_direct_io_fails_where_it_is_unsupported(weights, monkeypatch):
    path, _ = weights
    monkeypatch.delattr(os, "O_DIRECT", raising=False)
    with DirectReader(direct=True) as reader, pytest.raises(OSError):
        reader.read_into(page_buffer(ALIGN), path, 0, 16)
    with DirectReader() as reader:  # without forcing it, the page cache is used
        assert reader.read_into(page_buffer(ALIGN), path, 3, 16) == 3
        assert not reader.is_direct(path)
