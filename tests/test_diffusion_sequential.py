# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.diffusion.sequential: the budget and the record of a stage (GPU)."""

import pytest

from hypertier.diffusion.sequential import (
    CUDA_CONTEXT,
    VRAM_MARGIN,
    SequentialPipeline,
)

MiB = 1 << 20


@pytest.mark.cuda
def test_the_budget_is_what_is_free_unless_the_caller_gives_less():
    import torch

    free = SequentialPipeline()
    held = torch.empty(64 * MiB, dtype=torch.uint8, device="cuda")
    whole = free.available()
    driver_free = torch.cuda.mem_get_info()[0]
    context = free.context()
    if context is None:
        pytest.skip("the driver does not say what this process holds")
    # What is free, plus what this process holds: its tensors and its CUDA context.
    assert 40 * MiB < context < CUDA_CONTEXT, "a context that has run next to nothing"
    assert whole == pytest.approx(driver_free + torch.cuda.memory_reserved() + context, abs=MiB)
    assert whole >= driver_free + held.numel() + context
    del held
    assert free.available() == pytest.approx(whole, abs=MiB), "released: free, not counted twice"
    assert SequentialPipeline(vram_budget=whole // 2).available() == whole // 2
    assert SequentialPipeline(vram_budget=1 << 50).available() == pytest.approx(whole, abs=MiB)
    assert free.margin == VRAM_MARGIN == 300 * MiB
    with pytest.raises(ValueError, match="leaves nothing to run in"):
        SequentialPipeline(vram_budget=0)
    with pytest.raises(ValueError, match="margin cannot be negative"):
        SequentialPipeline(margin=-1)


@pytest.mark.cuda
def test_the_context_counts_as_held_for_what_it_is_now_up_to_what_the_peaks_count(monkeypatch):
    import torch

    pipeline = SequentialPipeline()
    monkeypatch.setattr(pipeline, "free_vram", lambda: 3000 * MiB)
    reserved = torch.cuda.memory_reserved()

    def available(context, budget=None):
        monkeypatch.setattr(pipeline, "context", lambda: context)
        pipeline.vram_budget = budget
        return pipeline.available() - reserved

    # A young context will grow during the stage, out of what is free: only what it holds.
    assert available(70 * MiB) == 3070 * MiB
    assert available(CUDA_CONTEXT) == 3000 * MiB + CUDA_CONTEXT
    # Above what the peaks count, the rest is not free for the stage: no more than that.
    assert available(CUDA_CONTEXT + 50 * MiB) == 3000 * MiB + CUDA_CONTEXT
    # The driver does not say: what the peaks count.
    assert available(None) == 3000 * MiB + CUDA_CONTEXT
    # A caller's budget is of the whole process: a context above the peaks' comes off it.
    assert available(70 * MiB, budget=2000 * MiB) == 2000 * MiB - reserved
    assert available(CUDA_CONTEXT + 50 * MiB, budget=2000 * MiB) == 1950 * MiB - reserved


@pytest.mark.cuda
def test_a_stage_is_admitted_by_its_expected_peak_and_leaves_a_report():
    import torch

    pipeline = SequentialPipeline(vram_budget=1024 * MiB, margin=100 * MiB)
    with pipeline.stage("first", expected_peak=924 * MiB) as report:
        work = torch.empty(200 * MiB, dtype=torch.uint8, device="cuda")
        report.load_seconds = 0.001
        del work
    assert pipeline.reports == [report] and report.name == "first"
    assert report.budget_bytes == 1024 * MiB and report.expected_peak_bytes == 924 * MiB
    assert report.peak_allocated_bytes >= 200 * MiB
    assert report.peak_reserved_bytes >= report.peak_allocated_bytes
    assert report.peak_process_bytes == report.peak_reserved_bytes + report.context_bytes
    if report.context_measured:  # read from the driver when the stage ended
        assert 40 * MiB < report.context_bytes < CUDA_CONTEXT
    else:
        assert report.context_bytes == CUDA_CONTEXT
    assert report.seconds > 0 and report.free_before_bytes > 0

    with pytest.raises(ValueError) as refused, pipeline.stage("second", 925 * MiB, "Try less."):
        raise AssertionError("a stage that does not fit must not run")
    message = str(refused.value)
    assert "the second stage peaks at 925" in message and "100" in message
    assert message.endswith("Try less.") and len(pipeline.reports) == 1

    with pipeline.stage("unknown") as report:  # nothing measured: admitted unchecked
        pass
    assert report.expected_peak_bytes is None and len(pipeline.reports) == 2


@pytest.mark.cuda
def test_the_context_of_a_stage_is_read_from_the_driver_or_is_what_the_peaks_count(monkeypatch):
    pipeline = SequentialPipeline()
    monkeypatch.setattr(pipeline, "context", lambda: 123 * MiB)
    with pipeline.stage("read while loaded") as report:
        pipeline.read_context(report)
        monkeypatch.setattr(pipeline, "context", lambda: 99 * MiB)  # after it was released
    assert report.context_bytes == 123 * MiB and report.context_measured
    assert report.peak_process_bytes == report.peak_reserved_bytes + 123 * MiB
    with pipeline.stage("read at the end") as report:
        pass
    assert report.context_bytes == 99 * MiB and report.context_measured
    monkeypatch.setattr(pipeline, "context", lambda: None)
    with pipeline.stage("the driver does not say") as report:
        pass
    assert report.context_bytes == CUDA_CONTEXT and not report.context_measured
    assert report.peak_process_bytes == report.peak_reserved_bytes + CUDA_CONTEXT


@pytest.mark.cuda
def test_a_stage_that_fails_is_still_on_record():
    pipeline = SequentialPipeline()
    with pytest.raises(RuntimeError, match="boom"), pipeline.stage("broken"):
        raise RuntimeError("boom")
    assert [report.name for report in pipeline.reports] == ["broken"]
    assert pipeline.reports[0].seconds > 0
