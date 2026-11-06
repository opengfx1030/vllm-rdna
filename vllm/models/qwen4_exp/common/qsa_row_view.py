# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row subsets of QSA attention metadata.

The RidgeFill cache-only path calls a QSA layer on one row set at a time: the
filled rows first, so their K/V lands in the cache before the exact rows attend
over it. Each call needs metadata whose row-indexed fields describe only its
rows. This module gathers those fields and recomputes ``query_start_loc`` on
device, so a layer can build the view without a host read.

It imports torch only. That keeps it usable from a CPU test and keeps the
indexer from importing the QSA layer module (which imports the indexer).
"""

from __future__ import annotations

from typing import Any, cast

import torch

# Row-indexed fields: one entry per packed token. Gathering by row keeps each
# row's slot, request and position.
_ROW_FIELDS = ("slot_mapping", "token_to_req", "logical_positions", "visible_blocks")

# Request-indexed fields stay as they are: ``block_table``, ``seq_lens`` and
# ``k_work_metadata`` address requests or work tiles, not tokens. A row subset
# never drops a request, because the cut always leaves a tail per sequence.

_ROW_VIEW_ATTR = "_ridgefill_row_views"


def _subset_query_start_loc(query_start_loc: torch.Tensor, index: torch.Tensor):
    """Cumulative row counts per sequence for a gathered row set, on device.

    ``searchsorted`` maps each kept row to its sequence, then ``index_add``
    counts rows per sequence. No host read, so a layer can build the view
    without a sync. Sequence order is preserved, so ``block_table`` still
    addresses the same requests.
    """
    if query_start_loc.numel() < 2:
        return query_start_loc[:1].zero_()
    seq = torch.searchsorted(query_start_loc, index, right=True) - 1
    counts = torch.zeros(
        query_start_loc.numel() - 1,
        dtype=query_start_loc.dtype,
        device=query_start_loc.device,
    )
    counts.index_add_(0, seq, torch.ones_like(index, dtype=counts.dtype))
    out = torch.empty_like(query_start_loc)
    out[0] = 0
    out[1:] = counts.cumsum(0)
    return out


def row_view(metadata, index: torch.Tensor | None):
    """Return ``metadata`` restricted to ``index`` rows, or metadata unchanged.

    ``index`` is ascending and keeps packed order. Row-indexed tensors gather;
    request-indexed ones stay. The result is a dataclass copy, so the original
    metadata the builder owns is never mutated.
    """
    if index is None:
        return metadata
    # Cache on the metadata object itself: the QSA layers of one forward share
    # it, and it dies with the forward. A global keyed by ``id`` could return a
    # view built for a freed object.
    views = getattr(metadata, _ROW_VIEW_ATTR, None)
    if views is None:
        views = {}
        try:
            setattr(metadata, _ROW_VIEW_ATTR, views)
        except (AttributeError, TypeError):
            views = None
    key = index.data_ptr()
    if views is not None and key in views:
        return views[key]
    from dataclasses import is_dataclass, replace

    if not is_dataclass(metadata):
        raise TypeError("row_view needs a dataclass attention metadata")
    changes: dict[str, Any] = {}
    for name in _ROW_FIELDS:
        value = getattr(metadata, name, None)
        if value is not None:
            changes[name] = value.index_select(0, index)
    rows = int(index.numel())
    if hasattr(metadata, "num_actual_tokens"):
        changes["num_actual_tokens"] = rows
    # Token-count fields are pure counts of the rows present. Recompute the
    # prefill count when the view holds only prefill rows; a decode row in the
    # view leaves the split counts alone.
    if getattr(metadata, "num_decode_tokens", 0) == 0 and hasattr(
        metadata, "num_prefill_tokens"
    ):
        changes["num_prefill_tokens"] = rows
    query_start_loc = getattr(metadata, "query_start_loc", None)
    if query_start_loc is not None:
        changes["query_start_loc"] = _subset_query_start_loc(query_start_loc, index)
    view = cast(Any, replace)(metadata, **changes)
    if views is not None:
        views[key] = view
    return view


__all__ = ["row_view"]
