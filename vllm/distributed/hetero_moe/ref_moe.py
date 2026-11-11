# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU reference of the W4A16 MoE (tests and the CPU stand-in of the servers).

Checkpoint layout (compressed-tensors pack-quantized, symmetric, group size g,
offset-binary nibbles): per expert ``w13`` is ``[2I, H]`` (gate rows then up
rows) and ``w2`` is ``[H, I]``; ``packed`` holds 8 nibbles per int32 along K,
low nibble first, value ``q`` stands for ``(q - 8) * scale`` with one fp16
scale per ``g`` consecutive K elements.

The reference dequantises to fp32, applies ``silu(gate) * up`` and ``w2``, and
accumulates the weighted expert outputs per token in float64 in k order, so
the result does not depend on how the pairs are split over devices; callers
round once to fp16.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

PACK = 8
ZERO_POINT = 8


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """``q`` int ``[N, K]`` in 0..15 -> int32 ``[N, K // 8]``."""
    n, k = q.shape
    assert k % PACK == 0
    nib = q.to(torch.int64).reshape(n, k // PACK, PACK)
    shifts = torch.arange(PACK, dtype=torch.int64) * 4
    acc = (nib << shifts).sum(dim=-1)
    acc = torch.where(acc >= 2**31, acc - 2**32, acc)
    return acc.to(torch.int32)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """int32 ``[N, K // 8]`` -> int32 ``[N, K]`` in 0..15."""
    shifts = torch.arange(PACK, dtype=torch.int32) * 4
    nib = (packed.unsqueeze(-1) >> shifts) & 0xF
    return nib.reshape(packed.shape[0], -1)


def quantize_w4a16(
    w: torch.Tensor, group: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric g-grouped quantisation of ``w [N, K]``.

    Returns ``(packed int32 [N, K/8], scale fp16 [N, K/g], dequantised fp32)``."""
    n, k = w.shape
    assert k % group == 0 and k % PACK == 0
    wg = w.float().reshape(n, k // group, group)
    scale = (wg.abs().amax(dim=-1) / 7.0).clamp(min=1e-6).to(torch.float16)
    q = torch.round(wg / scale.float().unsqueeze(-1)).clamp(-8, 7) + ZERO_POINT
    deq = ((q - ZERO_POINT) * scale.float().unsqueeze(-1)).reshape(n, k)
    return pack_int4(q.reshape(n, k)), scale, deq


def dequantize_w4a16(
    packed: torch.Tensor, scale: torch.Tensor, group: int
) -> torch.Tensor:
    q = unpack_int4(packed).float() - ZERO_POINT
    n, k = q.shape
    return (q.reshape(n, k // group, group) * scale.float().unsqueeze(-1)).reshape(n, k)


@dataclass
class Expert:
    """One expert in checkpoint (packed) form plus its fp32 dequantisation."""

    w13_packed: torch.Tensor  # int32 [2I, H/8]
    w13_scale: torch.Tensor  # fp16 [2I, H/g]
    w2_packed: torch.Tensor  # int32 [H, I/8]
    w2_scale: torch.Tensor  # fp16 [H, I/g]
    w13: torch.Tensor  # fp32 [2I, H]
    w2: torch.Tensor  # fp32 [H, I]


def make_expert(
    generator: torch.Generator, hidden: int, inter: int, group: int, std: float = 0.05
) -> Expert:
    w13 = torch.randn(2 * inter, hidden, generator=generator) * std
    w2 = torch.randn(hidden, inter, generator=generator) * std
    p13, s13, d13 = quantize_w4a16(w13, group)
    p2, s2, d2 = quantize_w4a16(w2, group)
    return Expert(p13, s13, p2, s2, d13, d2)


def make_experts(
    num: int, hidden: int, inter: int, group: int, seed: int = 0
) -> dict[int, Expert]:
    gen = torch.Generator().manual_seed(seed)
    return {e: make_expert(gen, hidden, inter, group) for e in range(num)}


def expert_forward(x: torch.Tensor, expert: Expert) -> torch.Tensor:
    """``x [R, H]`` fp32 -> ``[R, H]`` fp32 (silu-and-mul MLP)."""
    gate_up = x @ expert.w13.t()
    gate, up = gate_up.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ expert.w2.t()


def moe_partial(
    x: torch.Tensor,
    topk_w: torch.Tensor,
    topk_ids: torch.Tensor,
    experts: Mapping[int, Expert],
    *,
    slot_map: torch.Tensor | None = None,
    slot_experts: list[Expert] | None = None,
) -> torch.Tensor:
    """Weighted sum over the pairs whose expert is available, float64 ``[T, H]``.

    Experts come from ``experts`` (by global id) or, with ``slot_map``, from the
    slot pool ``slot_experts`` (slot = ``slot_map[id]``). Ids that are invalid
    (-1 or beyond the table) or not available are skipped, as the kernels do."""
    t_n, k_n = topk_ids.shape
    out = torch.zeros(t_n, x.shape[1], dtype=torch.float64)
    xf = x.float()
    ids = topk_ids.tolist()
    for t in range(t_n):
        for j in range(k_n):
            e = ids[t][j]
            if e < 0:
                continue
            if slot_map is not None:
                if e >= slot_map.shape[0] or slot_map[e] < 0:
                    continue
                assert slot_experts is not None
                expert = slot_experts[int(slot_map[e])]
            else:
                found = experts.get(e)
                if found is None:
                    continue
                expert = found
            y = expert_forward(xf[t : t + 1], expert)[0].double()
            out[t] += float(topk_w[t, j]) * y
    return out
