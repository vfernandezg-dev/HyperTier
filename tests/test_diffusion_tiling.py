# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.diffusion.tiling.

The plan and the blend weights are plain Python and run everywhere; the canvas needs PyTorch
on the CPU.
"""

import random
from itertools import pairwise

import pytest

from hypertier.diffusion.tiling import Canvas, TilePlan, blend_weights, plan_tiles, spread


def overlaps(starts, tile):
    return [a + tile - b for a, b in pairwise(starts)]


def test_spread_covers_with_the_fewest_evenly_spaced_tiles():
    assert spread(30, 30, 4) == [0]
    assert spread(30, 40, 4) == [0], "a tile larger than the frame is one tile"
    assert spread(30, 17, 4) == [0, 13]
    assert spread(52, 20, 4) == [0, 16, 32]
    assert spread(44, 14, 4) == [0, 10, 20, 30]
    rng = random.Random(0)
    for _ in range(300):
        size, tile, least = rng.randint(1, 120), rng.randint(2, 60), rng.randint(0, 8)
        if tile <= least:
            continue
        starts = spread(size, tile, least)
        if tile >= size:
            assert starts == [0]
            continue
        assert starts[0] == 0 and starts[-1] == size - tile, "from edge to edge"
        shared = overlaps(starts, tile)
        assert min(shared) >= least and max(shared) - min(shared) <= 1, "even, and enough"
        fewer = len(starts) - 1
        assert fewer * tile - (fewer - 1) * least < size, "one tile fewer would not cover"


def test_spread_refuses_what_cannot_advance():
    with pytest.raises(ValueError, match="cannot overlap its neighbour by 4 and advance"):
        spread(30, 4, 4)
    with pytest.raises(ValueError, match="cannot cover 0 cells"):
        spread(0, 4, 1)


def test_the_plans_of_the_reference_frames():
    # 480p is 30 x 52 latent cells and 720p is 44 x 80, with 16 px per cell.
    assert plan_tiles(30, 52, 340) == TilePlan((17, 20), (0, 13), (0, 16, 32))
    assert plan_tiles(44, 80, 340) == TilePlan((14, 23), (0, 10, 20, 30), (0, 19, 38, 57))
    assert plan_tiles(30, 52, 600) == TilePlan((30, 20), (0,), (0, 16, 32))
    assert plan_tiles(44, 80, 600) == TilePlan((24, 23), (0, 20), (0, 19, 38, 57))
    assert plan_tiles(30, 52, 256) == TilePlan((17, 14), (0, 13), (0, 10, 19, 28, 38))
    assert plan_tiles(44, 80, 256).tile == (14, 17)
    assert [plan_tiles(30, 52, cells).count for cells in (340, 600, 256)] == [6, 3, 10]
    assert [plan_tiles(44, 80, cells).count for cells in (340, 600, 256)] == [16, 8, 24]
    assert plan_tiles(30, 52, 340).decoded_cells == 2040
    whole = plan_tiles(30, 52, 30 * 52)
    assert whole == TilePlan((30, 52), (0,), (0,)) and whole.count == 1


def cheapest_by_search(height, width, max_cells, least):
    """Every tile shape tried: the fewest decoded cells, then the fewest tiles."""
    best = None
    for rows in range(1, height + 1):
        for columns in range(1, width + 1):
            if rows * columns > max_cells:
                continue
            if (rows < height and rows <= least) or (columns < width and columns <= least):
                continue
            count = len(spread(height, rows, least)) * len(spread(width, columns, least))
            cost = (count * rows * columns, count)
            best = cost if best is None or cost < best else best
    return best


def test_a_plan_fits_the_tile_and_decodes_the_fewest_cells():
    rng = random.Random(1)
    checked = 0
    while checked < 60:
        height, width = rng.randint(1, 26), rng.randint(1, 30)
        least = rng.randint(0, 4)
        max_cells = rng.randint(1, 200)
        expected = cheapest_by_search(height, width, max_cells, least)
        if expected is None:
            with pytest.raises(ValueError, match="no tile of at most"):
                plan_tiles(height, width, max_cells, min_overlap=least)
            continue
        plan = plan_tiles(height, width, max_cells, min_overlap=least)
        rows, columns = plan.tile
        assert rows * columns <= max_cells and rows <= height and columns <= width
        for starts, tile, size in ((plan.rows, rows, height), (plan.columns, columns, width)):
            assert starts[0] == 0 and starts[-1] + tile == size
            assert all(shared >= least for shared in overlaps(starts, tile))
        assert (plan.decoded_cells, plan.count) == expected
        checked += 1


def test_blend_weights_add_up_to_one_across_an_overlap():
    # Two tiles of 12 px that share 8: 2 px of margin on each side, a ramp over the 4 between.
    ends = blend_weights(12, 0, 8, margin=2, ramp=1)
    starts = blend_weights(12, 8, 0, margin=2, ramp=1)
    assert ends[:6] == [1.0] * 6 and ends[10:] == [0.0, 0.0], "nothing from its outer margin"
    assert starts[:2] == [0.0, 0.0] and starts[6:] == [1.0] * 6
    assert ends[6:10] == [0.875, 0.625, 0.375, 0.125]
    assert [a + b for a, b in zip(ends[4:], starts[:8])] == [1.0] * 8
    assert blend_weights(5, 0, 0, margin=2, ramp=1) == [1.0] * 5, "a frame edge is not blended"
    rng = random.Random(2)
    for _ in range(200):
        shared, margin, ramp = rng.randint(1, 40), rng.randint(0, 12), rng.randint(1, 8)
        length = shared + rng.randint(0, 30)
        ends = blend_weights(length, 0, shared, margin, ramp)
        starts = blend_weights(length, shared, 0, margin, ramp)
        total = [a + b for a, b in zip(ends[length - shared :], starts[:shared])]
        assert all(abs(value - 1.0) < 1e-12 for value in total)
        assert all(0.0 <= value <= 1.0 for value in ends + starts)
        sloped = sum(0.0 < value < 1.0 for value in starts)
        assert sloped >= min(ramp, shared), "the margin gives way before the ramp does"
    with pytest.raises(ValueError, match="cannot share 9 px"):
        blend_weights(8, 9, 0, margin=1, ramp=1)


def test_tiles_cut_from_one_video_blend_back_into_it():
    torch = pytest.importorskip("torch")
    scale = 4
    for height, width, max_cells, least, margin in [
        (30, 52, 340, 4, 1), (44, 80, 256, 4, 1), (9, 7, 20, 2, 0), (12, 12, 49, 5, 2), (6, 10, 60, 4, 1),
    ]:  # fmt: skip
        plan = plan_tiles(height, width, max_cells, min_overlap=least)
        video = torch.rand(3, 2, height * scale, width * scale, generator=torch.Generator().manual_seed(0))
        canvas = Canvas(torch, plan, (height, width), scale=scale, margin=margin)
        rows, columns = (cells * scale for cells in plan.tile)
        for i, top in enumerate(plan.rows):
            for j, left in enumerate(plan.columns):
                tile = video[:, :, top * scale : top * scale + rows, left * scale : left * scale + columns]
                canvas.add(i, j, tile.to(torch.bfloat16).to(torch.float32) if i == j == 0 else tile)
        blended = canvas.result()
        assert blended.shape == video.shape and blended.dtype == torch.float32
        assert torch.allclose(blended, video, atol=5e-3), "a weighted mean of equal values"
        difference = (blended - video).abs()
        difference[..., :rows, :columns] = 0  # the first tile went through bfloat16
        assert float(difference.max()) < 1e-6
        with pytest.raises(ValueError, match="no tile was added"):
            canvas.result()


def test_a_tile_of_another_size_is_refused():
    torch = pytest.importorskip("torch")
    plan = plan_tiles(9, 7, 20, min_overlap=2)
    canvas = Canvas(torch, plan, (9, 7), scale=4)
    with pytest.raises(ValueError, match=r"the tile is \[8, 8\] px, but the plan cuts"):
        canvas.add(0, 0, torch.zeros(3, 1, 8, 8))


def test_a_frame_too_small_or_a_tile_too_small_is_refused():
    with pytest.raises(ValueError, match="no frame of 0 x 5"):
        plan_tiles(0, 5, 100)
    with pytest.raises(ValueError, match="no tile of at most 20 latent cells covers 6 x 10"):
        plan_tiles(6, 10, 20)
