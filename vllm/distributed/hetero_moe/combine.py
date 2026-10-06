# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Add weighted cold-expert outputs back onto the hot-expert output."""

import torch


def combine_rows(
    hot_out: torch.Tensor,
    cold_out: torch.Tensor,
    token_index: torch.Tensor,
) -> torch.Tensor:
    """Sum per-pair cold outputs onto the matching token rows.

    Several cold pairs may land on one token. The sum is in float64 and
    then cast back to ``hot_out.dtype``, so the split matches an
    all-local sum that used the same per-pair products.

    Args:
        hot_out: ``[tokens, hidden]`` local hot contribution.
        cold_out: ``[cold_pairs, hidden]`` weighted remote contribution.
        token_index: Token row of each cold pair.

    Returns:
        Combined tensor with ``hot_out``'s dtype and shape.
    """
    if token_index.numel() == 0:
        return hot_out
    if cold_out.shape[0] != token_index.shape[0]:
        raise ValueError("cold outputs and token index differ in length")
    if cold_out.shape[-1] != hot_out.shape[-1]:
        raise ValueError("cold outputs and hot outputs differ in width")
    acc = hot_out.to(torch.float64).clone()
    acc.index_add_(
        0,
        token_index.to(dtype=torch.long, device=acc.device),
        cold_out.to(dtype=torch.float64, device=acc.device),
    )
    return acc.to(dtype=hot_out.dtype)
