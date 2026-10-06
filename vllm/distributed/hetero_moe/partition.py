# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split a MoE layer's (token, expert) pairs into hot and cold."""

from dataclasses import dataclass

import torch


@dataclass
class PairSplit:
    """Hot view of top-k plus the cold pairs to send.

    Attributes:
        hot_ids: Top-k ids with cold and padding positions set to -1.
        hot_weights: Router weights with those positions set to 0.
        token_index: Token row of each cold pair, shape ``[P]``.
        expert_ids: Cold expert ids, shape ``[P]``.
        router_weights: Cold router weights, shape ``[P]``.
    """

    hot_ids: torch.Tensor
    hot_weights: torch.Tensor
    token_index: torch.Tensor
    expert_ids: torch.Tensor
    router_weights: torch.Tensor

    @property
    def cold_count(self) -> int:
        return int(self.token_index.numel())


def partition_pairs(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hot: set[int],
) -> PairSplit:
    """Partition one layer.

    Args:
        topk_ids: ``[tokens, top_k]`` expert ids. Negatives are padding.
        topk_weights: ``[tokens, top_k]`` router weights.
        hot: Expert ids that stay on the fast tier.

    Returns:
        Hot tensors the local kernel can run, and one record per cold pair.
    """
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk ids and weights must share a shape")
    if topk_ids.ndim != 2:
        raise ValueError("topk ids must be [tokens, top_k]")
    ids = topk_ids.to(dtype=torch.long, device="cpu")
    weights = topk_weights.detach().to(device="cpu")
    valid = ids >= 0
    if hot:
        hot_ids_t = torch.tensor(sorted(hot), dtype=torch.long)
        is_hot = (ids.unsqueeze(-1) == hot_ids_t.view(1, 1, -1)).any(dim=-1)
        is_hot = is_hot & valid
    else:
        is_hot = torch.zeros_like(valid)
    neg = torch.full_like(ids, -1)
    zero = torch.zeros_like(weights)
    hot_ids = torch.where(is_hot, ids, neg)
    hot_weights = torch.where(is_hot, weights, zero)
    token_index, _slot = torch.nonzero(valid & ~is_hot, as_tuple=True)
    return PairSplit(
        hot_ids=hot_ids,
        hot_weights=hot_weights,
        token_index=token_index.to(torch.long),
        expert_ids=ids[token_index, _slot],
        router_weights=weights[token_index, _slot],
    )


def wire_hidden(hidden: torch.Tensor, token_index: torch.Tensor) -> torch.Tensor:
    """fp16 rows for the cold pairs, one row per pair.

    Args:
        hidden: ``[tokens, hidden_size]`` fast-tier activations.
        token_index: Cold-pair token positions.

    Returns:
        ``[cold_pairs, hidden_size]`` in fp16.
    """
    if token_index.numel() == 0:
        return hidden.new_empty((0, hidden.shape[-1])).to(torch.float16)
    rows = hidden.index_select(0, token_index.to(hidden.device))
    return rows.to(torch.float16).contiguous()
