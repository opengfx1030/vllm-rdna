# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split a layer, optionally defer the cold receive for ping-pong."""

from dataclasses import dataclass

import torch

from vllm.distributed.hetero_moe.capture import assert_outside_capture
from vllm.distributed.hetero_moe.combine import combine_rows
from vllm.distributed.hetero_moe.partition import partition_pairs
from vllm.distributed.hetero_moe.transport import ColdPayload


@dataclass
class _Pending:
    hot_out: torch.Tensor
    transport: object
    deferred: bool


def begin_moe(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hot: set[int],
    expert_fn,
    transport,
    *,
    capturing: bool | None = None,
) -> _Pending:
    """Run hot experts and issue the cold send.

    Args:
        hidden: ``[tokens, hidden]`` activations.
        topk_ids: ``[tokens, top_k]``.
        topk_weights: ``[tokens, top_k]``.
        hot: Expert ids resident on the fast tier.
        expert_fn: Local and, for in-process transports, cold apply.
        transport: Object with ``send`` / ``recv``.
        capturing: Test override for the capture guard.

    Returns:
        A pending combine. The cold receive has not run.
    """
    assert_outside_capture(capturing)
    split = partition_pairs(topk_ids, topk_weights, hot)
    hot_out = expert_fn(hidden, split.hot_ids, split.hot_weights)
    if split.cold_count == 0:
        return _Pending(hot_out=hot_out, transport=transport, deferred=False)
    rows = hidden.index_select(0, split.token_index.to(hidden.device))
    payload = ColdPayload(
        hidden=rows,
        expert_ids=split.expert_ids,
        router_weights=split.router_weights,
        token_index=split.token_index,
    )
    transport.send(payload)
    return _Pending(hot_out=hot_out, transport=transport, deferred=True)


def end_moe(pending: _Pending) -> torch.Tensor:
    """Receive weighted cold outputs and add them to the hot output.

    Args:
        pending: Value returned by :func:`begin_moe`.

    Returns:
        Combined layer output in float64 when ``expert_fn`` returns
        float64, otherwise the combined dtype.
    """
    if not pending.deferred:
        return pending.hot_out
    cold_out, token_index = pending.transport.recv()
    return combine_rows(pending.hot_out, cold_out, token_index)


def run_split_moe(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hot: set[int],
    expert_fn,
    transport,
    *,
    capturing: bool | None = None,
) -> torch.Tensor:
    """Begin and end one layer. Empty cold sets do not send.

    Args:
        hidden: Activations.
        topk_ids: Router ids.
        topk_weights: Router weights.
        hot: Fast-tier expert ids.
        expert_fn: Apply used for the hot rows and, in-process, the cold.
        transport: Cold transport.
        capturing: Forwarded to the capture guard.

    Returns:
        Combined expert output.
    """
    pending = begin_moe(
        hidden,
        topk_ids,
        topk_weights,
        hot,
        expert_fn,
        transport,
        capturing=capturing,
    )
    return end_moe(pending)
