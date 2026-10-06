# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE-layer hook used only when ``VLLM_HETERO_MOE=1``.

Hot pairs stay on the layer's existing quant method (Triton W4A16 on
gfx1100). Cold pairs are one fp16 row each, plus the expert id and the
router weight. Shared experts are applied once, on the hot call.
"""

import re

import torch

from vllm.distributed.hetero_moe.capture import assert_outside_capture
from vllm.distributed.hetero_moe.combine import combine_rows
from vllm.distributed.hetero_moe.partition import partition_pairs
from vllm.distributed.hetero_moe.runtime import get_runtime
from vllm.distributed.hetero_moe.transport import ColdPayload

_LAYER_INDEX = re.compile(r"layers\.(\d+)")


def layer_index_of(layer) -> int:
    """Parse ``layers.{i}`` from the layer name, else ``layer_index``."""
    name = str(getattr(layer, "layer_name", "") or "")
    match = _LAYER_INDEX.search(name)
    if match is not None:
        return int(match.group(1))
    return int(getattr(layer, "layer_index", 0))


def split_routed_forward(
    layer,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts,
    shared_experts_input,
) -> torch.Tensor:
    """Run hot experts locally and cold experts through the transport.

    The receive completes before this returns, so one forward is a
    correct combine. Overlap across micro-batches is ``run_ping_pong``,
    which the model runner does not call yet.

    Args:
        layer: ``RoutedExperts`` on the fast tier.
        x: Hidden states.
        topk_weights: Router weights.
        topk_ids: Router ids.
        shared_experts: Fast-tier shared experts, applied once.
        shared_experts_input: Input forwarded to the local apply.

    Returns:
        Combined routed output, plus the shared-expert contribution
        from the single local apply, in ``x``'s dtype when the local
        apply preserves it.

    Raises:
        RuntimeError: The current stream is capturing a graph.
    """
    assert_outside_capture()
    runtime = get_runtime()
    declared = getattr(layer, "global_num_experts", None)
    if declared:
        width = int(declared)
    else:
        # Outside capture. The layer normally carries the expert count,
        # so this host read is only the fallback.
        width = int(topk_ids.detach().to(device="cpu").max()) + 1
    runtime.bind_experts(width)
    index = layer_index_of(layer)
    runtime.note_routing(index, topk_ids)
    hot = runtime.hot_experts(index)
    split = partition_pairs(topk_ids, topk_weights, hot)
    hot_ids = split.hot_ids.to(device=topk_ids.device, dtype=topk_ids.dtype)
    hot_weights = split.hot_weights.to(
        device=topk_weights.device,
        dtype=topk_weights.dtype,
    )
    hot_out = layer.quant_method.apply(
        layer=layer,
        x=x,
        topk_weights=hot_weights,
        topk_ids=hot_ids,
        shared_experts=shared_experts,
        shared_experts_input=shared_experts_input,
    )
    if split.cold_count == 0:
        return hot_out
    rows = x.index_select(0, split.token_index.to(device=x.device))
    payload = ColdPayload(
        hidden=rows.to(torch.float16).contiguous(),
        expert_ids=split.expert_ids,
        router_weights=split.router_weights,
        token_index=split.token_index,
    )
    runtime.transport.send(payload)
    cold_out, token_index = runtime.transport.recv()
    combined = combine_rows(hot_out, cold_out, token_index)
    return combined.to(dtype=hot_out.dtype)
