# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode a frame that does not fit in VRAM as overlapping tiles, blended in host RAM.

A video VAE turns every latent cell into a block of pixels, and its activations at full
resolution are what does not fit: a tile bounds them by its area. Tiles are cut from the
latent, decoded one at a time, and blended on the CPU, so VRAM never holds more than one tile.

**This is approximate**, unlike streaming the weights. A decoder is convolutional: the pixels
near the border of a tile are computed without the neighbours a whole frame would give them.
Two things keep the error small and out of sight:

* neighbouring tiles overlap, and the overlap is blended with a linear ramp;
* the outermost latent cells of an interior edge, the pixels with the least context, are
  left out of the blend altogether (``margin``): 1.5 dB of PSNR for free, measured.

Measured on the Wan 2.2 VAE against the untiled float32 decode: 38 to 55 dB of PSNR. Most of
that range is the picture (fine texture costs the most), then the size of the tile; the
differences are in fine texture near the borders of the tiles, and no seam shows.

The plan is plain Python: a tile is as large as VRAM allows (``max_cells``), and its shape is
fitted to the frame so that the fewest latent cells are decoded in all, since decoding time
follows them. Importing this module does not import PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TilePlan:
    """How a latent frame is cut into tiles of one shape.

    Attributes:
        tile: rows and columns of a tile, in latent cells.
        rows: the first latent row of each row of tiles.
        columns: the first latent column of each column of tiles.
    """

    tile: tuple[int, int]
    rows: tuple[int, ...]
    columns: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.rows) * len(self.columns)

    @property
    def decoded_cells(self) -> int:
        """Latent cells decoded in all, overlaps counted twice: what the time follows."""
        return self.count * self.tile[0] * self.tile[1]


def spread(size: int, tile: int, min_overlap: int) -> list[int]:
    """Evenly spaced starts of the fewest tiles of ``tile`` cells that cover ``size`` cells
    with neighbours overlapping by at least ``min_overlap``."""
    if size < 1 or tile < 1 or min_overlap < 0:
        raise ValueError(f"cannot cover {size} cells with tiles of {tile} overlapping {min_overlap}")
    if tile >= size:
        return [0]
    if tile <= min_overlap:
        raise ValueError(
            f"a tile of {tile} cells cannot overlap its neighbour by {min_overlap} and advance"
        )
    count = -(-(size - min_overlap) // (tile - min_overlap))
    return [round(k * (size - tile) / (count - 1)) for k in range(count)]


def plan_tiles(height: int, width: int, max_cells: int, *, min_overlap: int = 4) -> TilePlan:
    """The tiling of a ``height`` x ``width`` latent frame that decodes the fewest cells.

    Args:
        height, width: the latent frame, in cells.
        max_cells: the largest area of a tile, in latent cells: what the VRAM left for
            activations allows.
        min_overlap: latent cells shared by neighbouring tiles, at least.

    Among the tilings with as many rows and columns of tiles as you like, each with the
    smallest tile that still overlaps enough, this picks the one with the fewest cells decoded
    in all, and among equals the one with the fewest tiles. A frame that fits is one tile.
    """
    if height < 1 or width < 1:
        raise ValueError(f"there is no frame of {height} x {width} latent cells")
    best: tuple[int, int, int, int] | None = None
    for across in range(1, height + 1):
        rows = _smallest_tile(height, across, min_overlap)
        if rows is None:
            break
        for along in range(1, width + 1):
            columns = _smallest_tile(width, along, min_overlap)
            if columns is None:
                break
            if rows * columns > max_cells:
                continue
            cost = (across * along * rows * columns, across * along, rows, columns)
            if best is None or cost < best:
                best = cost
            break  # more columns of tiles only decode more
    if best is None:
        smallest = (min_overlap + 1) ** 2
        raise ValueError(
            f"no tile of at most {max_cells} latent cells covers {height} x {width} with an "
            f"overlap of {min_overlap}: the smallest that advances has {smallest}"
        )
    _, _, rows, columns = best
    return TilePlan(
        (rows, columns),
        tuple(spread(height, rows, min_overlap)),
        tuple(spread(width, columns, min_overlap)),
    )


def blend_weights(length: int, before: int, after: int, margin: int, ramp: int) -> list[float]:
    """Weights of a tile along one axis, per pixel.

    Args:
        length: pixels of the tile along the axis.
        before, after: pixels it shares with the previous and the next tile (0: a frame edge).
        margin: pixels at each interior edge that take no part in the blend.
        ramp: the shortest linear ramp to keep; the margin gives way to it in a small overlap.

    Across an overlap the weights of the two tiles add up to one: the tile that ends there
    falls from 1 to 0 as the one that starts rises, and each is 0 in its own outer margin.
    """
    weights = [1.0] * length
    for shared, leading in ((before, True), (after, False)):
        if not shared:
            continue
        if shared > length:
            raise ValueError(f"a tile of {length} px cannot share {shared} px with a neighbour")
        skip = min(margin, max((shared - ramp) // 2, 0))
        slope = shared - 2 * skip
        rising = [0.0] * skip + [(i + 0.5) / slope for i in range(slope)]
        if leading:
            weights[: shared - skip] = rising
        else:
            weights[length - shared + skip :] = rising[::-1]
    return weights


class Canvas:
    """A video in host RAM that tiles are blended into, one at a time.

    The sum of the weights is kept per pixel and divided out at the end, so a pixel covered
    by three tiles, or by one, comes out right as well.
    """

    def __init__(
        self,
        torch: Any,
        plan: TilePlan,
        frame: tuple[int, int],
        *,
        scale: int,
        margin: int = 1,
        ramp: int = 1,
    ) -> None:
        """
        Args:
            plan: the tiling of the latent frame.
            frame: rows and columns of the latent frame, in cells.
            scale: pixels per latent cell along each axis.
            margin: latent cells left out of the blend at interior edges.
            ramp: the shortest ramp to keep, in latent cells.
        """
        self._torch = torch
        self._plan = plan
        self._scale = scale
        self._size = (frame[0] * scale, frame[1] * scale)
        self._weights = (
            self._axis(plan.rows, plan.tile[0], margin, ramp),
            self._axis(plan.columns, plan.tile[1], margin, ramp),
        )
        self._sum: Any = None
        self._weight = torch.zeros(self._size)

    def _axis(self, starts: tuple[int, ...], tile: int, margin: int, ramp: int) -> list[Any]:
        torch, scale = self._torch, self._scale
        weights = []
        for k, start in enumerate(starts):
            before = (starts[k - 1] + tile - start) * scale if k else 0
            after = (start + tile - starts[k + 1]) * scale if k + 1 < len(starts) else 0
            weights.append(
                torch.tensor(blend_weights(tile * scale, before, after, margin * scale, ramp * scale))
            )
        return weights

    def add(self, row: int, column: int, pixels: Any) -> None:
        """Blend in the tile at ``row``, ``column`` of the plan: ``[channels, frames, h, w]``."""
        torch, scale = self._torch, self._scale
        height, width = (cells * scale for cells in self._plan.tile)
        if tuple(pixels.shape[-2:]) != (height, width):
            raise ValueError(
                f"the tile is {list(pixels.shape[-2:])} px, but the plan cuts {[height, width]}"
            )
        if self._sum is None:
            self._sum = torch.zeros((*pixels.shape[:-2], *self._size))
        top, left = self._plan.rows[row] * scale, self._plan.columns[column] * scale
        weight = self._weights[0][row][:, None] * self._weights[1][column][None, :]
        tall, wide = slice(top, top + height), slice(left, left + width)
        # To the CPU as it is, and widened there: converted first, a second copy of the tile
        # in float32 would sit in VRAM next to the activations.
        self._sum[..., tall, wide] += pixels.to("cpu").to(torch.float32) * weight
        self._weight[tall, wide] += weight

    def result(self) -> Any:
        """The blended video, float32 in host RAM. The canvas is used up."""
        if self._sum is None:
            raise ValueError("no tile was added")
        blended, self._sum = self._sum.div_(self._weight), None
        return blended


def _smallest_tile(size: int, count: int, min_overlap: int) -> int | None:
    """Cells of the smallest tile such that ``count`` of them cover ``size`` overlapping by
    ``min_overlap``; None when that many tiles could not advance."""
    if count == 1:
        return size
    tile = -(-(size + (count - 1) * min_overlap) // count)
    return tile if tile > min_overlap else None


__all__ = ["Canvas", "TilePlan", "blend_weights", "plan_tiles", "spread"]
