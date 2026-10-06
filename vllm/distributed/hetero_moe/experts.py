# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference expert apply used by loopback tests and the CPU policy path.

The device path does not use this. gfx1100 hot experts stay on the
existing Triton W4A16 MoE kernel. gfx1030 cold experts stay on
``RDNA2W4A16MoEExperts`` / ``moe_gptq_gemm_rdna2``.
"""

import torch


def apply_expert_tables(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    tables: torch.Tensor,
) -> torch.Tensor:
    """Weighted sum of ``hidden @ table[expert]``.

    Args:
        hidden: ``[N, H]`` or, for a rectangular top-k, ``[N, H]`` with
            ``expert_ids`` of shape ``[N, K]``.
        expert_ids: Expert id per row, or per top-k slot. ``-1`` skips.
        router_weights: Same shape as ``expert_ids``.
        tables: ``[E, H, H]`` expert matrices.

    Returns:
        ``[N, H]`` float64 weighted outputs.
    """
    if expert_ids.shape != router_weights.shape:
        raise ValueError("ids and weights must share a shape")
    if expert_ids.ndim == 2:
        acc = torch.zeros(
            hidden.shape[0],
            hidden.shape[1],
            dtype=torch.float64,
        )
        for slot in range(expert_ids.shape[1]):
            acc = acc + apply_expert_tables(
                hidden,
                expert_ids[:, slot],
                router_weights[:, slot],
                tables,
            )
        return acc
    if expert_ids.ndim != 1:
        raise ValueError("expert ids must be rank 1 or 2")
    if hidden.shape[0] != expert_ids.shape[0]:
        raise ValueError("hidden rows and expert ids differ")
    out = torch.zeros(hidden.shape[0], hidden.shape[1], dtype=torch.float64)
    hidden64 = hidden.detach().to(dtype=torch.float64, device="cpu")
    weights64 = router_weights.detach().to(dtype=torch.float64, device="cpu")
    table64 = tables.detach().to(dtype=torch.float64, device="cpu")
    ids = expert_ids.detach().to(dtype=torch.long, device="cpu")
    for row in range(hidden64.shape[0]):
        expert = int(ids[row])
        if expert < 0:
            continue
        out[row] = weights64[row] * (hidden64[row] @ table64[expert])
    return out
