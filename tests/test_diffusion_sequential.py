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
    # What is free, plus what this process holds: its tensors and its CUDA context.
    assert whole == driver_free + torch.cuda.memory_reserved() + CUDA_CONTEXT
    assert whole >= driver_free + held.numel() + CUDA_CONTEXT
    del held
    assert free.available() == whole, "released memory is free again, not counted twice"
    assert SequentialPipeline(vram_budget=whole // 2).available() == whole // 2
    assert SequentialPipeline(vram_budget=1 << 50).available() == whole, "no more than there is"
    assert free.margin == VRAM_MARGIN == 300 * MiB
    with pytest.raises(ValueError, match="leaves nothing to run in"):
        SequentialPipeline(vram_budget=0)
    with pytest.raises(ValueError, match="margin cannot be negative"):
        SequentialPipeline(margin=-1)


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
    assert report.peak_process_bytes == report.peak_reserved_bytes + CUDA_CONTEXT
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
def test_a_stage_that_fails_is_still_on_record():
    pipeline = SequentialPipeline()
    with pytest.raises(RuntimeError, match="boom"), pipeline.stage("broken"):
        raise RuntimeError("boom")
    assert [report.name for report in pipeline.reports] == ["broken"]
    assert pipeline.reports[0].seconds > 0
