# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.telemetry.bandwidth.

Pure logic (statistics, file I/O, the pinned-probe algorithm) runs everywhere, including CI
without PyTorch. Tests marked ``cuda`` run the real probes at tiny sizes on a GPU.
"""

import math
import os
import sys

import pytest

from hypertier.telemetry import GiB, KiB, MiB
from hypertier.telemetry import bandwidth as bw

# --- statistics ---------------------------------------------------------------------------


def test_percentile_interpolates_like_numpy():
    values = [4.0, 1.0, 3.0, 2.0]
    assert bw.percentile(values, 0) == 1.0
    assert bw.percentile(values, 50) == pytest.approx(2.5)
    assert bw.percentile(values, 90) == pytest.approx(3.7)
    assert bw.percentile(values, 100) == 4.0
    assert bw.percentile([7.0], 90) == 7.0


@pytest.mark.parametrize(("values", "q"), [([], 50), ([1.0], -1), ([1.0], 101)])
def test_percentile_rejects_bad_input(values, q):
    with pytest.raises(ValueError):
        bw.percentile(values, q)


def test_timing_stats_and_rates():
    stats = bw.TimingStats.from_samples([10.0, 20.0, 30.0, 40.0, 50.0])
    assert (stats.n, stats.median_ms, stats.min_ms, stats.max_ms) == (5, 30.0, 10.0, 50.0)
    assert stats.p90_ms == pytest.approx(46.0)
    assert bw.gb_per_s(10**9, 1000.0) == pytest.approx(1.0)
    assert bw.gb_per_s(GiB, 100.0) == pytest.approx(10.737418)
    with pytest.raises(ValueError):
        bw.gb_per_s(1, 0.0)
    with pytest.raises(ValueError):
        bw.TimingStats.from_samples([])


def test_size_sweep_doubles_from_1mib_to_1gib():
    assert bw.size_sweep(MiB, 8 * MiB) == [MiB, 2 * MiB, 4 * MiB, 8 * MiB]
    default = bw.size_sweep()
    assert (default[0], default[-1], len(default)) == (MiB, GiB, 11)
    with pytest.raises(ValueError):
        bw.size_sweep(0, MiB)
    with pytest.raises(ValueError):
        bw.size_sweep(2 * MiB, MiB)


def test_repeats_shrink_only_for_huge_transfers():
    assert bw._repeats_for(MiB, 20) == 20
    assert bw._repeats_for(GiB, 20) == 8
    assert bw._repeats_for(4 * GiB, 20) == 5
    assert bw._repeats_for(GiB, 3) == 3


def test_overlap_efficiency():
    assert bw.overlap_efficiency(10.0, 10.0, 10.0) == pytest.approx(1.0)  # fully hidden
    assert bw.overlap_efficiency(10.0, 10.0, 20.0) == pytest.approx(0.0)  # serialised
    assert bw.overlap_efficiency(10.0, 30.0, 35.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        bw.overlap_efficiency(0.0, 1.0, 1.0)


# --- pinned-memory probe (algorithm, with a fake allocator) -------------------------------


class FakeAllocator:
    """Stands in for pinned allocations: fails once more than ``limit`` bytes are held."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.held = 0
        self.releases = 0

    def allocate(self, nbytes: int) -> object:
        if self.held + nbytes > self.limit:
            raise RuntimeError("CUDA error: out of memory\nCompile with TORCH_USE_CUDA_DSA")
        self.held += nbytes
        return _Block(self, nbytes)

    def release(self) -> None:
        self.releases += 1


class _Block:
    def __init__(self, owner: FakeAllocator, nbytes: int) -> None:
        self.owner, self.nbytes = owner, nbytes

    def __del__(self) -> None:
        self.owner.held -= self.nbytes


def _probe(fake: FakeAllocator, **kwargs):
    params = {
        "first_block": 256 * MiB,
        "max_block": 2 * GiB,
        "min_block": 64 * MiB,
        "reserve_bytes": 0,
        "mem_total": 64 * GiB,
        "mem_available": lambda: None,
    }
    params.update(kwargs)
    return bw.probe_max_pinned(allocate=fake.allocate, release=fake.release, **params)


def test_pinned_probe_narrows_down_to_the_allocation_limit():
    fake = FakeAllocator(limit=GiB)
    result = _probe(fake, cap_bytes=4 * GiB)
    assert result.stop_reason == "alloc_failed"
    assert result.max_pinned_bytes == GiB
    assert result.blocks == [256 * MiB, 512 * MiB, 256 * MiB]
    assert result.first_error.startswith("RuntimeError: CUDA error: out of memory")
    assert fake.held == 0, "everything must be freed"
    assert fake.releases == 1


def test_pinned_probe_stops_at_the_cap():
    fake = FakeAllocator(limit=64 * GiB)
    result = _probe(fake, cap_bytes=GiB)
    assert (result.stop_reason, result.max_pinned_bytes) == ("cap", GiB)
    assert result.blocks == [256 * MiB, 512 * MiB, 256 * MiB]
    assert result.first_error is None
    assert fake.held == 0


def test_pinned_probe_keeps_mem_available_above_the_reserve():
    fake = FakeAllocator(limit=64 * GiB)
    result = _probe(
        fake,
        cap_bytes=16 * GiB,
        reserve_bytes=3 * GiB,
        mem_available=lambda: 4 * GiB - fake.held,
    )
    assert (result.stop_reason, result.max_pinned_bytes) == ("ram_guard", GiB)
    assert result.trace[-1] == (GiB, 3 * GiB)
    assert fake.held == 0


def test_pinned_probe_never_exceeds_total_ram_minus_reserve():
    fake = FakeAllocator(limit=64 * GiB)
    result = _probe(fake, cap_bytes=16 * GiB, reserve_bytes=GiB, mem_total=2 * GiB)
    assert (result.stop_reason, result.max_pinned_bytes) == ("ram_guard", GiB)


def test_pinned_probe_validates_block_sizes():
    fake = FakeAllocator(limit=GiB)
    with pytest.raises(ValueError):
        _probe(fake, cap_bytes=GiB, first_block=300 * MiB)
    with pytest.raises(ValueError):
        _probe(fake, cap_bytes=GiB, min_block=512 * MiB)


def test_default_ram_reserve():
    assert bw.default_ram_reserve(8 * GiB) == 2 * GiB
    assert bw.default_ram_reserve(32 * GiB) == int(0.15 * 32 * GiB)


# --- file I/O and the NVMe probe (small files, no GPU) ------------------------------------


def test_aligned_buffer_requires_page_multiple():
    with pytest.raises(ValueError):
        bw.AlignedBuffer(1000, pinned=False)


@pytest.mark.parametrize("direct", [False, None])
def test_write_then_read_roundtrip(tmp_path, direct):
    path = tmp_path / "data.bin"
    with bw.AlignedBuffer(256 * KiB, pinned=False) as src, bw.AlignedBuffer(
        256 * KiB, pinned=False
    ) as dst:
        src.view[:] = os.urandom(src.nbytes)
        seconds, write_method = bw.write_file(path, src.view, MiB, direct=direct)
        assert path.stat().st_size == MiB and seconds >= 0

        nread, _, read_method = bw.read_file(path, dst.view, direct=direct)
        assert nread == MiB
        assert bytes(dst.view) == bytes(src.view)  # last chunk == the repeated pattern

        half = 128 * KiB
        nread, _, _ = bw.read_file(path, dst.view, half, direct=direct)
        assert nread == MiB
        assert bytes(dst.view[:half]) == bytes(src.view[half:])  # last half-chunk read
    if direct is False:
        assert write_method == "buffered+fsync"
        assert read_method in {"fadvise_dontneed", "buffered"}
    else:
        assert read_method in {"o_direct", "fadvise_dontneed", "buffered"}


def test_read_file_rejects_oversized_chunk(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"\0" * 4096)
    with bw.AlignedBuffer(4096, pinned=False) as buf, pytest.raises(ValueError):
        bw.read_file(path, buf.view, 8192)


def test_measure_nvme_read_small_file(tmp_path):
    report = bw.measure_nvme_read(
        tmp_path, file_bytes=2 * MiB, chunk_sizes=(256 * KiB, MiB), pinned=False
    )
    assert report.file_bytes == 2 * MiB and not report.pinned_buffer
    assert [r.chunk_bytes for r in report.reads] == [256 * KiB, MiB]
    assert all(r.bytes_read == 2 * MiB and r.gbps > 0 for r in report.reads)
    assert report.write_gbps > 0
    assert list(tmp_path.iterdir()) == [], "the temporary file must be deleted"


def test_measure_nvme_read_validates_sizes(tmp_path):
    with pytest.raises(ValueError):
        bw.measure_nvme_read(tmp_path, file_bytes=3 * MiB, chunk_sizes=(2 * MiB,))
    with pytest.raises(ValueError):
        bw.measure_nvme_read(tmp_path, file_bytes=MiB, chunk_sizes=(1000,))


def test_measure_nvme_read_refuses_windows_drives_on_wsl(monkeypatch):
    monkeypatch.setattr(bw, "is_wsl", lambda: True)
    with pytest.raises(ValueError, match="/mnt"):
        bw.measure_nvme_read("/mnt/c/Users", file_bytes=MiB, chunk_sizes=(MiB,))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/mountinfo")
def test_filesystem_type_of_root():
    assert bw.filesystem_type("/")


# --- real probes on the GPU, tiny sizes ---------------------------------------------------


@pytest.mark.cuda
def test_copy_bandwidth_small_sweep():
    results = bw.measure_copy_bandwidth([MiB, 2 * MiB], warmup=1, repeats=3)
    combos = {(r.direction, r.memory) for r in results}
    assert combos == {(d, m) for d in ("h2d", "d2h") for m in ("pinned", "pageable")}
    assert len(results) == 8
    assert all(r.gbps_median > 0 and r.timing.n == 3 and r.wall.n == 3 for r in results)


@pytest.mark.cuda
def test_copy_compute_overlap_small():
    r = bw.measure_copy_compute_overlap(8 * MiB, matmul_n=512, warmup=1, repeats=3)
    assert min(r.copy_ms, r.compute_ms, r.both_ms, r.copy_gbps) > 0
    assert r.matmul_iters >= 1 and math.isfinite(r.efficiency)


@pytest.mark.cuda
def test_device_bandwidth_and_matmul_small():
    vram = bw.measure_device_bandwidth(16 * MiB, warmup=1, repeats=3)
    assert vram.gbps_median > 0
    mm = bw.measure_matmul_tflops(sizes=(256,), warmup=1, repeats=3)
    assert [r.dtype for r in mm] == ["float16", "bfloat16"]
    assert all(r.tflops_median > 0 for r in mm)
    with pytest.raises(ValueError):
        bw.measure_matmul_tflops(sizes=(256,), dtypes=("int8",))


@pytest.mark.cuda
def test_probe_max_pinned_small_cap_releases_memory():
    result = bw.probe_max_pinned(
        128 * MiB, first_block=32 * MiB, max_block=64 * MiB, min_block=16 * MiB
    )
    assert (result.stop_reason, result.max_pinned_bytes) == ("cap", 128 * MiB)
    assert result.pin_gbps and result.pin_gbps > 0


@pytest.mark.cuda
def test_nvme_read_into_pinned_buffer(tmp_path):
    report = bw.measure_nvme_read(tmp_path, file_bytes=4 * MiB, chunk_sizes=(MiB,))
    assert report.pinned_buffer
    assert report.reads[0].bytes_read == 4 * MiB
    assert list(tmp_path.iterdir()) == []
