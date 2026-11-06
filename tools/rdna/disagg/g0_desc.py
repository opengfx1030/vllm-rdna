# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL strided-descriptor arithmetic shared by the G0a spike modules.

An Nx5 descriptor list has rows ``(addr, len, device, stride, count)``. Row
``r`` stands for ``count`` entries of ``len`` bytes, entry ``k`` starting at
``addr + k * stride``; entries are numbered consecutively over all rows. This is
the numbering ``make_prepped_xfer`` takes its indices in.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def digest(buf: Any) -> str:
    """128-bit blake2b of a contiguous byte buffer."""
    return hashlib.blake2b(memoryview(buf), digest_size=16).hexdigest()


def row_starts(rows: np.ndarray) -> np.ndarray:
    """Number of the first entry of every row."""
    counts = rows[:, 4].astype(np.int64)
    return np.cumsum(counts) - counts


def num_entries(rows: np.ndarray) -> int:
    return int(rows[:, 4].astype(np.int64).sum()) if len(rows) else 0


def resolve(
    rows: np.ndarray, starts: np.ndarray, ids: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entries ``ids`` -> (address, length, row index)."""
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    total = num_entries(rows)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= total):
        raise IndexError(f"descriptor id out of range [0, {total})")
    row = np.searchsorted(starts, ids, side="right") - 1
    k = ids - starts[row]
    addr = rows[row, 0].astype(np.int64) + k * rows[row, 3].astype(np.int64)
    return addr, rows[row, 1].astype(np.int64), row
