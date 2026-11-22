# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Read byte ranges of a file straight into a caller's buffer, bypassing the page cache.

The NVMe tier reads weights into page-locked staging memory that is then copied to VRAM. With
``O_DIRECT`` the drive writes into that memory itself: no second copy through the page cache,
and no RAM spent caching a model that does not fit in RAM anyway.

``O_DIRECT`` only accepts offsets, sizes and buffer addresses that are multiples of the
device's logical block (512 B or 4 KiB), and tensors in a safetensors file start wherever the
header ends. So a read covers the aligned range around the bytes that are wanted, and the
caller takes its view at the returned shift.
"""

from __future__ import annotations

import ctypes
import errno
import os
import time
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from hypertier.telemetry._common import MiB
from hypertier.telemetry.bandwidth import DIRECT_IO_ALIGN

if TYPE_CHECKING:
    from typing_extensions import Self


def aligned_span(offset: int, nbytes: int, align: int = DIRECT_IO_ALIGN) -> tuple[int, int]:
    """``(start, length)`` of the smallest ``align``-ed range covering ``nbytes`` at ``offset``.

    The wanted bytes begin ``offset - start`` bytes into what is read.
    """
    if offset < 0 or nbytes < 0:
        raise ValueError(f"offset and nbytes must be >= 0, got {offset} and {nbytes}")
    start = offset // align * align
    end = -(-(offset + nbytes) // align) * align
    return start, end - start


class DirectReader:
    """Positional reads into caller buffers, with ``O_DIRECT`` where the filesystem allows it.

    One request of ``chunk_bytes`` at a time: measured on the reference NVMe, 64 MiB requests
    read at 2.9 GB/s, a single 368 MiB request at 2.2 GB/s and 4 MiB requests at 2.5 GB/s.
    Reading from several threads was slower still, so one reader thread is the design.

    Files stay open until :meth:`close`. Where ``O_DIRECT`` is refused (tmpfs, some network
    filesystems) the read goes through the page cache instead; ``direct`` forces either way.
    """

    def __init__(self, *, chunk_bytes: int = 64 * MiB, direct: bool | None = None) -> None:
        if chunk_bytes <= 0 or chunk_bytes % DIRECT_IO_ALIGN:
            raise ValueError(f"chunk_bytes must be a positive multiple of {DIRECT_IO_ALIGN}")
        self.chunk_bytes = chunk_bytes
        self._direct = direct
        self._files: dict[Path, tuple[int, bool]] = {}
        self.bytes_read = 0
        self.seconds = 0.0

    def read_into(
        self,
        buffer: memoryview,
        path: str | PathLike[str],
        offset: int,
        nbytes: int,
        *,
        between: Callable[[], None] | None = None,
    ) -> int:
        """Fill ``buffer`` with the aligned range around ``nbytes`` at ``offset``.

        Returns the shift: the wanted bytes are ``buffer[shift : shift + nbytes]``. The
        buffer must hold the whole range (``aligned_span``) and, for ``O_DIRECT``, start on
        an aligned address, as page-locked pool memory does. ``between`` runs after every
        request, which is how a reader thread stays responsive during a long read.
        """
        start, length = aligned_span(offset, nbytes)
        shift = offset - start
        if nbytes == 0:
            return shift
        if len(buffer) < length:
            raise ValueError(f"buffer of {len(buffer)} bytes cannot hold the {length} to read")
        if buffer.readonly:
            raise ValueError("buffer is read-only")
        key = Path(path)
        fd, direct = self._open(key)
        if direct and _address(buffer) % DIRECT_IO_ALIGN:
            raise ValueError(f"buffer is not aligned to {DIRECT_IO_ALIGN} bytes")
        wanted = shift + nbytes  # the file may end before the aligned range does
        done = 0
        t0 = time.perf_counter()
        while done < wanted:
            request = buffer[done : min(done + self.chunk_bytes, length)]
            try:
                count = os.preadv(fd, [request], start + done)
            except OSError as exc:
                if not direct or exc.errno != errno.EINVAL or self._direct:
                    raise
                # Some filesystems accept the O_DIRECT open and refuse the read.
                fd, direct = self._reopen_buffered(key)
                continue
            finally:
                request.release()
            if count == 0:
                raise EOFError(f"{key}: file ends {wanted - done} bytes before the range read")
            done += count
            if between is not None:
                between()
        self.bytes_read += done
        self.seconds += time.perf_counter() - t0
        return shift

    def is_direct(self, path: str | PathLike[str]) -> bool:
        """Whether reads of ``path`` bypass the page cache (opens the file if needed)."""
        return self._open(Path(path))[1]

    def close(self) -> None:
        for fd, _ in self._files.values():
            os.close(fd)
        self._files.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -----------------------------------------------------------------------------------------

    def _open(self, path: Path) -> tuple[int, bool]:
        opened = self._files.get(path)
        if opened is not None:
            return opened
        o_direct = getattr(os, "O_DIRECT", 0)
        if self._direct is not False:
            if o_direct:
                try:
                    opened = os.open(path, os.O_RDONLY | o_direct), True
                except OSError as exc:
                    if self._direct or exc.errno not in (errno.EINVAL, errno.EOPNOTSUPP):
                        raise
            elif self._direct:
                raise OSError(errno.EINVAL, "O_DIRECT is not available on this platform")
        if opened is None:
            opened = os.open(path, os.O_RDONLY), False
        self._files[path] = opened
        return opened

    def _reopen_buffered(self, path: Path) -> tuple[int, bool]:
        os.close(self._files.pop(path)[0])
        opened = os.open(path, os.O_RDONLY), False
        self._files[path] = opened
        return opened


def _address(buffer: memoryview) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(buffer))
