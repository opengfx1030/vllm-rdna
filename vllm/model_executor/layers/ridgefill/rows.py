# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row split and cache-write bookkeeping for the RidgeFill draft.

Pure Python: no ``torch`` at import. The session builds one :class:`RowSplit`
per prefill forward and every late layer reuses it. The split is the contract
the cache-write-only path rests on:

* ``filled`` and ``exact`` partition this forward's packed rows, so each row's
  K/V and indexer key is written once and no row is dropped;
* each row keeps its packed order inside its sequence, so the GDN recurrence
  and the QSA compressor ring see the same token order as the full path;
* the cut can be aligned to the indexer's compression ratio so a compressed
  group never straddles the split. A straddling group would be pooled from a
  prefix in the cache-only pass and a suffix in the exact pass; the ring makes
  that work, but aligning removes the dependency.

Nothing here touches a device. ``runtime`` uploads the index tensors once per
forward and the model uses them for every layer.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass


class RowSplitError(ValueError):
    """The metadata cannot describe a split of this forward's rows."""


@dataclass(frozen=True)
class RowSplit:
    """Filled and exact rows of one forward, in packed row order.

    ``filled_idx`` and ``exact_idx`` are ascending tuples, so a caller can
    ``index_select`` with them without rebuilding them per layer. ``per_seq_*``
    counts keep the sequence grouping that the recurrent kernels need.
    """

    num_tokens: int
    filled_idx: tuple[int, ...]
    exact_idx: tuple[int, ...]
    filled_ranges: tuple[tuple[int, int], ...]
    exact_ranges: tuple[tuple[int, int], ...]
    per_seq_filled: tuple[int, ...]
    per_seq_exact: tuple[int, ...]

    @property
    def active(self) -> bool:
        return bool(self.filled_idx)

    @property
    def filled_count(self) -> int:
        return len(self.filled_idx)

    @property
    def exact_count(self) -> int:
        return len(self.exact_idx)

    def filled_mask(self) -> list[bool]:
        """Host bool list of length ``num_tokens``; True for filled rows."""
        mask = [False] * self.num_tokens
        for row in self.filled_idx:
            mask[row] = True
        return mask

    def kind(self, row: int) -> str:
        """``filled`` or ``exact`` for one packed row, in log time."""
        starts = [start for start, _ in self.filled_ranges]
        pos = bisect.bisect_right(starts, row) - 1
        if pos >= 0 and row < self.filled_ranges[pos][1]:
            return "filled"
        return "exact"


def _ranges(idx: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Collapse ascending indices into half-open ranges."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    prev: int | None = None
    for row in idx:
        if start is None or prev is None:
            start = prev = row
            continue
        if row == prev + 1:
            prev = row
            continue
        out.append((start, prev + 1))
        start = prev = row
    if start is not None and prev is not None:
        out.append((start, prev + 1))
    return tuple(out)


def _check_metadata(seq_lens: Sequence[int], query_start_loc: Sequence[int]) -> int:
    if len(query_start_loc) != len(seq_lens) + 1:
        raise RowSplitError(
            "query_start_loc must have one more entry than seq_lens, got "
            f"{len(query_start_loc)} and {len(seq_lens)}"
        )
    if not query_start_loc or query_start_loc[0] != 0:
        raise RowSplitError("query_start_loc must start at 0")
    for start, end in zip(query_start_loc, query_start_loc[1:]):
        if end < start:
            raise RowSplitError("query_start_loc must be non-decreasing")
    num_tokens = int(query_start_loc[-1])
    if num_tokens < 0:
        raise RowSplitError("query_start_loc must be non-negative")
    return num_tokens


def _split_from_flags(
    num_tokens: int,
    flags: Sequence[bool],
    per_seq_counts: Sequence[tuple[int, int]],
) -> RowSplit:
    filled = tuple(row for row, flag in enumerate(flags) if flag)
    exact = tuple(row for row, flag in enumerate(flags) if not flag)
    return RowSplit(
        num_tokens=num_tokens,
        filled_idx=filled,
        exact_idx=exact,
        filled_ranges=_ranges(filled),
        exact_ranges=_ranges(exact),
        per_seq_filled=tuple(f for f, _ in per_seq_counts),
        per_seq_exact=tuple(e for _, e in per_seq_counts),
    )


def build_row_split(
    *,
    seq_lens: Sequence[int],
    query_start_loc: Sequence[int],
    cuts: Sequence[int],
) -> RowSplit:
    """Split packed rows by absolute position, without reading ``positions``.

    ``seq_lens[s]`` is sequence ``s``'s full length after this chunk and
    ``query_start_loc[s]`` its first packed row. Row ``r`` of sequence ``s``
    sits at absolute position ``seq_lens[s] - (loc[s+1] - loc[s]) + r``, so the
    split needs only the scheduler's CPU metadata. ``cuts[s]`` is the first
    exact position of sequence ``s`` (``filled_end`` for its full length).
    """
    num_tokens = _check_metadata(seq_lens, query_start_loc)
    if len(cuts) != len(seq_lens):
        raise RowSplitError("cuts must have one entry per sequence")
    flags = [False] * num_tokens
    counts: list[tuple[int, int]] = []
    for seq, length in enumerate(seq_lens):
        start = int(query_start_loc[seq])
        end = int(query_start_loc[seq + 1])
        query_len = end - start
        if length < query_len:
            raise RowSplitError(
                f"sequence {seq} has length {length} shorter than its "
                f"{query_len} query rows"
            )
        first_position = int(length) - query_len
        filled_here = 0
        for row in range(start, end):
            if first_position + (row - start) < int(cuts[seq]):
                flags[row] = True
                filled_here += 1
        counts.append((filled_here, query_len - filled_here))
    return _split_from_flags(num_tokens, flags, counts)


def build_row_split_from_positions(
    *,
    seq_lens: Sequence[int],
    query_start_loc: Sequence[int],
    cuts: Sequence[int],
    positions: Sequence[int],
) -> RowSplit:
    """The positions-authoritative variant. Must agree with the arithmetic.

    The model already receives absolute positions (M-RoPE sends one row per
    axis; text positions repeat, so the first axis is enough). This builder
    keeps the old behaviour as the authority and is what the CPU proof
    compares the arithmetic builder against.
    """
    num_tokens = _check_metadata(seq_lens, query_start_loc)
    if len(cuts) != len(seq_lens):
        raise RowSplitError("cuts must have one entry per sequence")
    if len(positions) != num_tokens:
        raise RowSplitError(
            f"positions has {len(positions)} rows, forward has {num_tokens}"
        )
    flags = [False] * num_tokens
    counts: list[tuple[int, int]] = []
    for seq in range(len(seq_lens)):
        start = int(query_start_loc[seq])
        end = int(query_start_loc[seq + 1])
        filled_here = 0
        for row in range(start, end):
            if int(positions[row]) < int(cuts[seq]):
                flags[row] = True
                filled_here += 1
        counts.append((filled_here, (end - start) - filled_here))
    return _split_from_flags(num_tokens, flags, counts)


def prefix_split(num_tokens: int, filled_end: int) -> RowSplit:
    """Split without metadata: ``[0, filled_end)`` filled, the rest exact.

    Exact only when the forward is the whole prompt. A decode forward has
    ``filled_end == 0`` and is entirely exact.
    """
    num_tokens = int(num_tokens)
    end = max(0, min(int(filled_end), num_tokens))
    flags = [row < end for row in range(num_tokens)]
    counts = [(end, num_tokens - end)] if num_tokens else []
    return _split_from_flags(num_tokens, flags, counts)


def cuts_from_lengths(
    lengths: Sequence[int],
    *,
    tail: int,
    block_size: int = 0,
    align: bool = False,
    compress_ratio: int = 0,
) -> tuple[int, ...]:
    """First exact position of each sequence's full length.

    With ``align`` the cut rounds down to the indexer's group unit (block size
    times the compression ratio when both are known), so a compressed group
    never straddles the filled/exact split. Rounding down only lengthens the
    exact tail.
    """
    raw = [max(0, int(length) - int(tail)) for length in lengths]
    if not align or block_size <= 1:
        return tuple(raw)
    return tuple(
        align_cut(cut, block_size=block_size, compress_ratio=compress_ratio)
        for cut in raw
    )


def align_cut(cut: int, *, block_size: int, compress_ratio: int = 0) -> int:
    """Round a cut down so no compressed group straddles it."""
    unit = int(block_size) if block_size and block_size > 1 else 1
    if compress_ratio and compress_ratio > 1:
        unit = unit * int(compress_ratio)
    if unit <= 1:
        return max(0, int(cut))
    return (max(0, int(cut)) // unit) * unit


def straddling_groups(
    split: RowSplit,
    *,
    seq_lens: Sequence[int],
    query_start_loc: Sequence[int],
    compress_ratio: int,
) -> tuple[tuple[int, int], ...]:
    """``(seq, group_start_position)`` pairs whose tokens land in both sets.

    Empty when the cut is aligned to ``compress_ratio``. Reported by the CPU
    proof, never read on the hot path.
    """
    if compress_ratio <= 1 or not split.active:
        return ()
    ratio = int(compress_ratio)
    out: list[tuple[int, int]] = []
    filled = set(split.filled_idx)
    for seq in range(len(seq_lens)):
        start = int(query_start_loc[seq])
        end = int(query_start_loc[seq + 1])
        first_position = int(seq_lens[seq]) - (end - start)
        groups: dict[int, set[bool]] = {}
        for row in range(start, end):
            group = (first_position + (row - start)) // ratio
            groups.setdefault(group, set()).add(row in filled)
        for group, sides in groups.items():
            if len(sides) > 1:
                out.append((seq, group * ratio))
    return tuple(out)


def subset_query_start_loc(
    query_start_loc: Sequence[int], idx: Sequence[int]
) -> tuple[int, ...]:
    """Cumulative row counts per sequence after gathering ``idx``.

    Rows keep packed order inside their sequence, so the compacted batch is
    still sequence-contiguous and needs no block-table rebuild.
    """
    if len(query_start_loc) < 2 or query_start_loc[0] != 0:
        raise RowSplitError("query_start_loc must start at 0 and hold a sequence")
    num_tokens = int(query_start_loc[-1])
    selected = sorted(int(row) for row in idx)
    if selected and (selected[0] < 0 or selected[-1] >= num_tokens):
        raise RowSplitError("gathered rows fall outside the forward")
    counts: list[int] = [0]
    cursor = 0
    for seq in range(len(query_start_loc) - 1):
        start = int(query_start_loc[seq])
        end = int(query_start_loc[seq + 1])
        while cursor < len(selected) and selected[cursor] < start:
            cursor += 1
        n = 0
        while cursor < len(selected) and selected[cursor] < end:
            cursor += 1
            n += 1
        counts.append(counts[-1] + n)
    return tuple(counts)


@dataclass(frozen=True)
class CacheRow:
    """One row a cache stream must write, with its physical destination."""

    row: int
    slot: int
    position: int


@dataclass(frozen=True)
class CacheStream:
    """The rows one cache stream writes, split by which pass writes them.

    ``filled`` is written by the cache-write-only pass and ``exact`` by the
    layer's normal attention. ``ring`` marks a compressed stream whose store
    is completed by whichever pass holds the group's last token.
    """

    name: str
    filled: tuple[CacheRow, ...]
    exact: tuple[CacheRow, ...]
    ring: bool = False

    @property
    def writes(self) -> tuple[CacheRow, ...]:
        return self.filled + self.exact

    def overlapping(self) -> tuple[CacheRow, ...]:
        """Rows claimed by both passes. Must be empty."""
        exact_rows = {write.row for write in self.exact}
        return tuple(write for write in self.filled if write.row in exact_rows)


def plan_cache_stream(
    name: str,
    split: RowSplit,
    *,
    slots: Sequence[int],
    positions: Sequence[int],
    ring: bool = False,
) -> CacheStream:
    """Assign each row's cache write to the pass that will run it."""
    if len(slots) != split.num_tokens or len(positions) != split.num_tokens:
        raise RowSplitError(f"{name}: slots/positions must have one entry per row")
    filled = tuple(
        CacheRow(row, int(slots[row]), int(positions[row])) for row in split.filled_idx
    )
    exact = tuple(
        CacheRow(row, int(slots[row]), int(positions[row])) for row in split.exact_idx
    )
    return CacheStream(name=name, filled=filled, exact=exact, ring=ring)


def ledger(
    streams: Sequence[CacheStream],
) -> dict[str, list[tuple[int, int, int]]]:
    """``(row, slot, position)`` writes per stream, for the CPU proofs."""
    return {
        stream.name: [(w.row, w.slot, w.position) for w in stream.writes]
        for stream in streams
    }


def assert_partition(split: RowSplit) -> None:
    """Raise unless the two row sets cover every row exactly once."""
    total = split.filled_count + split.exact_count
    if total != split.num_tokens:
        raise RowSplitError(
            f"row sets cover {total} rows, forward has {split.num_tokens}"
        )
    if set(split.filled_idx) & set(split.exact_idx):
        raise RowSplitError("a row is in both the filled and exact set")
    for seq_f, seq_e in zip(split.per_seq_filled, split.per_seq_exact):
        if seq_f < 0 or seq_e < 0:
            raise RowSplitError("negative per-sequence row count")


__all__ = [
    "CacheRow",
    "CacheStream",
    "RowSplit",
    "RowSplitError",
    "align_cut",
    "assert_partition",
    "build_row_split",
    "build_row_split_from_positions",
    "cuts_from_lengths",
    "ledger",
    "plan_cache_stream",
    "prefix_split",
    "straddling_groups",
    "subset_query_start_loc",
]
