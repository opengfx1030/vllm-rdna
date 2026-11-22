# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused launches of the HIP serving path of Inkling-2-Mini that remove graph nodes of a
decode step without changing a bit of its result
(``docs/rdna2/inkling2-fuse-attn-chain.md``).

``VLLM_INKLING2_FUSE_ATTN_CHAIN`` (default on, ``0`` or empty restores the separate
launches) switches them:

* the message ``attn_res`` of every layer of a loop pass is one launch
  (``inkling2_fuse_msgs_fill``): the messages of a pass depend only on the states of the
  previous pass, which are fixed while the pass runs.  Batches of at most
  ``MSGS_BATCH_TOKENS`` tokens (decode, verification) take it; larger chunks keep one
  launch per layer.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm import _custom_ops as ops

#: ``VLLM_INKLING2_FUSE_ATTN_CHAIN`` (default on): see the module docstring.
FUSE_ATTN_CHAIN = os.environ.get("VLLM_INKLING2_FUSE_ATTN_CHAIN", "1") not in ("", "0")

#: Rows of the persistent buffer of the batched message launch: batches of at most this
#: many tokens (decode, verification) take it, larger chunks one launch per layer.
MSGS_BATCH_TOKENS = 16


class MessageBatch:
    """The messages of every layer of a loop pass from one launch.

    ``fill`` writes ``[rows, L, 4, H]`` fp32 (one ``attn_res`` per layer and token,
    grid ``(T, L)``); ``select`` points ``v.msgs_qkv`` / ``v.msgs_mlp`` (the views the
    message gates read) at the rows of one layer."""

    def __init__(self, ws: Any, queries: torch.Tensor) -> None:
        layers, four, hidden = queries.shape
        assert four == 4
        self.queries = queries.data
        self.rows = min(ws.T, MSGS_BATCH_TOKENS)
        self.buf = torch.empty(
            self.rows, layers, 4, hidden, dtype=torch.float32, device=ws.device
        )
        self._views: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def usable(self, tokens: int) -> bool:
        return tokens <= self.rows

    def fill(self, v: Any, n_prev: int, eps: float) -> None:
        ops.inkling2_fuse_msgs_fill(
            v.prev_blk,
            n_prev,
            v.prev_x0,
            v.prev_x1,
            self.queries,
            self.buf[: v.T],
            eps,
        )

    def select(self, v: Any, layer: int) -> None:
        views = self._views.get((v.T, layer))
        if views is None:
            m = self.buf[: v.T, layer]
            views = self._views[(v.T, layer)] = (m[:, :3], m[:, 3:4])
        v.msgs_qkv, v.msgs_mlp = views
