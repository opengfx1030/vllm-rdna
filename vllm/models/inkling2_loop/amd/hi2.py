# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HI2 (``rdna_hint_flex`` v3.2) routed experts for Inkling-2-Mini, torch only.

The quantizer writes each layer's routed experts as two *stacked* HI2 groups
(``hi2/stack.py`` of the quantizer)::

    model.language_model.layers.N.mlp.experts.gate_up.hi2_<field>
    model.language_model.layers.N.mlp.experts.down.hi2_<field>

with ``gate_up`` parts in row order ``(gate_proj, up_proj)``, one transform per
expert (``had [E, K/16]``, ``col_scale [E, K]``) shared by the parts, and
``part_off [E * P + 1]`` giving the rows of every (expert, part).  The flat row
buffers are the spec 2.4 container (see ``rdna_hint_flex_utils`` in the fork).

This module is deliberately free of ``vllm`` imports so the model modules can be
loaded by file path in the CPU tests (``moe.py`` keeps the same contract).  It
owns three things:

* validation of a stack as the kernels read it (offsets derived from
  ``row_nbits`` / ``row_block``, no offset trusted before ``validate_stack``);
* tensor-parallel slicing: rows for the fused gate/up (K is the hidden size,
  not split), 64-aligned columns for the down projection;
* the exact pure-torch decode of the stored ladder (float64, one rounding to
  fp16 per element, as ``hi2.format.dequantize_stored``) and the routed MoE
  reference in fp32 used as the correctness oracle for the kernels.

The stored basis is ``H (W * s)`` and the runtime activation is ``H (x / s)``
(spec 2.3); the reference below dequantizes back to the original basis, so it
does not need the activation transform.
"""

from __future__ import annotations

import math
from typing import Any

import regex as re
import torch

__all__ = [
    "FIELDS",
    "GROUP",
    "K_MULTIPLE",
    "OPTIONAL_FIELDS",
    "STACK_FIELDS",
    "STACK_GROUPS",
    "canonicalize_stack",
    "decode_part",
    "dequantize_stored_rows",
    "expert_weights",
    "part_rows",
    "reference_moe",
    "slice_stack",
    "softcapped_swiglu",
    "parse_tensor_name",
    "validate_stack",
]

GROUP = 16
BLOCK_SIZES = (16, 32, 64)
NBITS = (2, 4, 8)
K_MULTIPLE = 64

#: spec 2.4 fields of one flat container (``had``/``col_scale`` are per expert in
#: a stack, ``part_off`` only exists on a stack)
FIELDS = (
    "codes",
    "code_off",
    "blk_params",
    "param_off",
    "row_consts",
    "row_nbits",
    "row_block",
    "had",
)
OPTIONAL_FIELDS = ("col_scale",)
STACK_FIELDS = FIELDS + OPTIONAL_FIELDS + ("part_off",)

#: quantizer stack group -> (fork stage, parts).  Field names are the same.
STACK_GROUPS: dict[str, tuple[str, int]] = {
    "gate_up": ("w13", 2),
    "down": ("w2", 1),
}


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------

_HI2_STACK_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_up|down)\.hi2_(\w+)$"
)


def parse_tensor_name(name: str) -> tuple[int, str, str] | None:
    """``(layer, group, field)`` of a stacked expert tensor, else ``None``."""
    m = _HI2_STACK_RE.match(name)
    if m is None:
        return None
    return int(m.group(1)), m.group(2), m.group(3)


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------


def canonicalize_stack(t: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Dtypes and shapes the kernels read; one copy (the source views are mmap)."""
    out: dict[str, torch.Tensor] = {
        "codes": t["codes"].reshape(-1).to(torch.int32),
        "code_off": t["code_off"].reshape(-1).to(torch.int64).to(torch.int32),
        "blk_params": t["blk_params"].reshape(-1).to(torch.uint8),
        "param_off": t["param_off"].reshape(-1).to(torch.int64).to(torch.int32),
        "row_consts": t["row_consts"].reshape(-1, 3).to(torch.float16),
        "row_nbits": t["row_nbits"].reshape(-1).to(torch.uint8),
        "row_block": t["row_block"].reshape(-1).to(torch.uint8),
        "had": t["had"].reshape(t["had"].shape[0], -1).to(torch.uint8),
        "part_off": t["part_off"].reshape(-1).to(torch.int64).to(torch.int32),
    }
    if t.get("col_scale") is not None:
        out["col_scale"] = (
            t["col_scale"].reshape(t["col_scale"].shape[0], -1).to(torch.float16)
        )
    return out


def num_experts(t: dict[str, torch.Tensor]) -> int:
    return int(t["had"].shape[0])


def num_cols(t: dict[str, torch.Tensor]) -> int:
    return int(t["had"].shape[1]) * GROUP


def part_rows(t: dict[str, torch.Tensor], part: int) -> tuple[int, int]:
    """Row range ``[r0, r1)`` of stack part ``part`` (expert-major, part order)."""
    po = t["part_off"].to(torch.int64)
    return int(po[part]), int(po[part + 1])


def validate_stack(
    t: dict[str, torch.Tensor],
    parts: int,
    k: int | None = None,
    experts: int | None = None,
) -> None:
    """Refuse a stack the kernels would misread.  The row offsets are rederived
    from ``row_nbits`` / ``row_block``; a checkpoint offset is never trusted."""
    e = num_experts(t)
    kk = num_cols(t)
    if k is not None and kk != k:
        raise ValueError(f"hi2 stack has K={kk}, the layer expects K={k}")
    if experts is not None and e != experts:
        raise ValueError(f"hi2 stack has {e} experts, the layer expects {experts}")
    if kk % K_MULTIPLE:
        raise ValueError(f"hi2 stack K={kk} is not a multiple of {K_MULTIPLE}")
    r = int(t["row_nbits"].numel())
    if (
        t["row_block"].numel() != r
        or tuple(t["row_consts"].shape) != (r, 3)
        or t["code_off"].numel() != r + 1
        or t["param_off"].numel() != r + 1
    ):
        raise ValueError("hi2 stack row buffers do not have R rows")
    if t.get("col_scale") is not None and tuple(t["col_scale"].shape) != (e, kk):
        raise ValueError(f"hi2 stack col_scale {tuple(t['col_scale'].shape)}")
    nb = t["row_nbits"].to(torch.int64)
    bl = t["row_block"].to(torch.int64)
    if not bool(torch.isin(nb, torch.tensor(NBITS)).all()):
        raise ValueError("hi2 stack row_nbits outside (2, 4, 8)")
    if not bool(torch.isin(bl, torch.tensor(BLOCK_SIZES)).all()):
        raise ValueError("hi2 stack row_block outside (16, 32, 64)")
    co = t["code_off"].to(torch.int64)
    po = t["param_off"].to(torch.int64)
    if int(co[0]) != 0 or int(po[0]) != 0:
        raise ValueError("hi2 stack offsets must start at 0")
    if not torch.equal(co[1:] - co[:-1], kk * nb // 32):
        raise ValueError("hi2 stack code_off does not match row_nbits")
    if not torch.equal(po[1:] - po[:-1], kk // bl):
        raise ValueError("hi2 stack param_off does not match row_block")
    if int(co[-1]) != t["codes"].numel() or int(po[-1]) * 2 != t["blk_params"].numel():
        raise ValueError("hi2 stack offsets do not cover the buffers")
    part_off = t["part_off"].to(torch.int64)
    if part_off.numel() != e * parts + 1 or int(part_off[0]) != 0:
        raise ValueError(
            f"hi2 stack part_off has {part_off.numel()} entries, "
            f"expected {e * parts + 1}"
        )
    if int(part_off[-1]) != r or bool((part_off[1:] < part_off[:-1]).any()):
        raise ValueError("hi2 stack part_off does not cover the rows")
    counts = (part_off[1:] - part_off[:-1]).reshape(e, parts)
    if counts.numel() > 1 and bool((counts != counts[0, 0]).any()):
        raise ValueError("hi2 stack expert parts have different row counts")


# ---------------------------------------------------------------------------
# tensor-parallel slicing
# ---------------------------------------------------------------------------


def _take_rows(t: dict[str, torch.Tensor], r0: int, r1: int) -> dict[str, torch.Tensor]:
    """Rows ``[r0, r1)`` of a canonical stack as a flat container."""
    co = t["code_off"].to(torch.int64)
    po = t["param_off"].to(torch.int64)
    out = {
        "codes": t["codes"][int(co[r0]) : int(co[r1])],
        "code_off": (co[r0 : r1 + 1] - co[r0]).to(torch.int32),
        "blk_params": t["blk_params"][2 * int(po[r0]) : 2 * int(po[r1])],
        "param_off": (po[r0 : r1 + 1] - po[r0]).to(torch.int32),
        "row_consts": t["row_consts"][r0:r1],
        "row_nbits": t["row_nbits"][r0:r1],
        "row_block": t["row_block"][r0:r1],
    }
    return out


def _select_cols(
    part: dict[str, torch.Tensor], k0: int, k1: int
) -> dict[str, torch.Tensor]:
    """Columns ``[k0, k1)`` of every row of a flat container; both bounds must be
    multiples of 64 so no word, block or Hadamard group is cut."""
    if k0 % K_MULTIPLE or k1 % K_MULTIPLE or not 0 <= k0 < k1:
        raise ValueError(f"hi2 column slice [{k0}, {k1}) is not 64-aligned")
    r = int(part["row_nbits"].numel())
    nb = part["row_nbits"].to(torch.int64)
    bl = part["row_block"].to(torch.int64)
    co = part["code_off"].to(torch.int64)
    po = part["param_off"].to(torch.int64)
    w_lo, w_n = k0 * nb // 32, (k1 - k0) * nb // 32
    b_lo, b_n = k0 // bl, (k1 - k0) // bl
    new_co = torch.zeros(r + 1, dtype=torch.int64)
    new_co[1:] = torch.cumsum(w_n, 0)
    new_po = torch.zeros(r + 1, dtype=torch.int64)
    new_po[1:] = torch.cumsum(b_n, 0)
    codes = torch.empty(int(new_co[-1]), dtype=part["codes"].dtype)
    params = torch.empty(2 * int(new_po[-1]), dtype=part["blk_params"].dtype)
    pv = part["blk_params"].view(-1, 2)
    nv = params.view(-1, 2)
    for nbits in sorted(set(nb.tolist())):
        for bs in sorted(set(bl.tolist())):
            rows = ((nb == nbits) & (bl == bs)).nonzero().flatten()
            if rows.numel() == 0:
                continue
            wn, bn = int(w_n[rows[0]]), int(b_n[rows[0]])
            src = (co[rows] + w_lo[rows]).unsqueeze(1) + torch.arange(wn)
            dst = new_co[rows].unsqueeze(1) + torch.arange(wn)
            codes[dst.reshape(-1)] = part["codes"][src.reshape(-1)]
            src = (po[rows] + b_lo[rows]).unsqueeze(1) + torch.arange(bn)
            dst = new_po[rows].unsqueeze(1) + torch.arange(bn)
            nv[dst.reshape(-1)] = pv[src.reshape(-1)]
    out = dict(part)
    out["codes"] = codes
    out["code_off"] = new_co.to(torch.int32)
    out["blk_params"] = params
    out["param_off"] = new_po.to(torch.int32)
    return out


def _concat_rows(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack flat containers of equal K along R, rebasing their offsets."""
    first = parts[0]
    out = dict(first)
    for f, off in (("codes", "code_off"), ("blk_params", "param_off")):
        bases, total = [], 0
        for p in parts:
            bases.append(p[off].to(torch.int64)[:-1] + total)
            total += int(p[off][-1])
        out[off] = torch.cat(bases + [torch.tensor([total])]).to(torch.int32)
        out[f] = torch.cat([p[f] for p in parts])
    for f in ("row_consts", "row_nbits", "row_block"):
        out[f] = torch.cat([p[f] for p in parts])
    return out


def slice_stack(
    t: dict[str, torch.Tensor],
    stage: str,
    *,
    parts: int,
    rows_per_part: int,
    k: int,
    lo: int,
    hi: int,
    experts: int,
) -> dict[str, torch.Tensor]:
    """This rank's share of a validated stack.

    ``stage`` is ``"w13"`` (``parts == 2``: gate then up rows of every expert,
    K = hidden size, not split) or ``"w2"`` (``parts == 1``: every row, K of the
    down projection split over the intermediate units).  ``rows_per_part`` is the
    *full* (TP=1) number of rows of one part; ``[lo, hi)`` is this rank's
    intermediate-unit span (rows of w13, columns of w2).

    The rank's stack keeps the expert-major whole-part order, so the kernels'
    block-64 grouping works unchanged.  ``had``/``col_scale`` are per expert and
    follow the slice of the columns they describe.
    """
    validate_stack(t, parts, k, experts)
    e = num_experts(t)
    if stage == "w13":
        if parts != 2:
            raise ValueError("hi2: the w13 stack has two parts")
        pieces = []
        for x in range(e):
            for p in range(2):
                r0, r1 = part_rows(t, x * 2 + p)
                if r1 - r0 != rows_per_part:
                    raise ValueError(
                        f"hi2: w13 expert {x} part {p} has {r1 - r0} rows, "
                        f"expected {rows_per_part}"
                    )
                pieces.append(_take_rows(t, r0 + lo, r0 + hi))
        out = _concat_rows(pieces)
        out["had"] = t["had"]
        if t.get("col_scale") is not None:
            out["col_scale"] = t["col_scale"]
        nrows = hi - lo
        out["part_off"] = torch.cumsum(
            torch.tensor([0] + [nrows] * (2 * e), dtype=torch.int64), 0
        ).to(torch.int32)
        return out
    if stage != "w2":
        raise ValueError(f"hi2: unknown stage {stage!r}")
    if parts != 1:
        raise ValueError("hi2: the w2 stack has one part")
    pieces = []
    had = []
    col_scale = []
    for x in range(e):
        r0, r1 = part_rows(t, x)
        if r1 - r0 != rows_per_part:
            raise ValueError(
                f"hi2: w2 expert {x} has {r1 - r0} rows, expected {rows_per_part}"
            )
        piece = _select_cols(_take_rows(t, r0, r1), lo, hi)
        pieces.append(piece)
        had.append(t["had"][x, lo // GROUP : hi // GROUP])
        if t.get("col_scale") is not None:
            col_scale.append(t["col_scale"][x, lo:hi])
    out = _concat_rows(pieces)
    out["had"] = torch.stack(had).to(torch.uint8)
    if col_scale:
        out["col_scale"] = torch.stack(col_scale).to(torch.float16)
    # every row is kept; only the columns (K) shrink
    out["part_off"] = (torch.arange(0, e + 1, dtype=torch.int64) * rows_per_part).to(
        torch.int32
    )
    validate_stack(out, 1, hi - lo, e)
    return out


# ---------------------------------------------------------------------------
# decode (spec 2.2 / 2.3), float64, one fp16 rounding per element
# ---------------------------------------------------------------------------


def _hadamard16() -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < GROUP:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(GROUP)


def _unpack_codes(words: torch.Tensor, nbits: int) -> torch.Tensor:
    per_word = 32 // nbits
    w = words.to(torch.int64) & 0xFFFFFFFF
    mask = (1 << nbits) - 1
    c = torch.stack([(w >> (nbits * i)) & mask for i in range(per_word)], dim=-1)
    return c.reshape(*words.shape[:-1], -1)


def dequantize_stored_rows(
    t: dict[str, torch.Tensor], r0: int, r1: int
) -> torch.Tensor:
    """float64 ``[r1 - r0, K]`` reconstruction in the stored (rotated) basis."""
    k = num_cols(t)
    nb = t["row_nbits"][r0:r1].to(torch.int64)
    bl = t["row_block"][r0:r1].to(torch.int64)
    co = t["code_off"].to(torch.int64)
    po = t["param_off"].to(torch.int64)
    pv = t["blk_params"].view(-1, 2)
    out = torch.zeros(r1 - r0, k, dtype=torch.float64)
    for nbits in sorted(set(nb.tolist())):
        for bs in sorted(set(bl.tolist())):
            rows = ((nb == nbits) & (bl == bs)).nonzero().flatten()
            if rows.numel() == 0:
                continue
            gr = rows + r0
            wr, nblk = k * nbits // 32, k // bs
            words = t["codes"][co[gr].unsqueeze(1) + torch.arange(wr)]
            codes = _unpack_codes(words, nbits).reshape(rows.numel(), nblk, bs)
            p = pv[po[gr].unsqueeze(1) + torch.arange(nblk)]  # [r, nblk, 2]
            rc = t["row_consts"][gr].double().unsqueeze(1)  # [r, 1, 3]
            lo = (
                (p[..., 0].double() * rc[..., 1] + rc[..., 0]).float().to(torch.float16)
            )
            step = (p[..., 1].float() * rc[..., 2].float()).to(torch.float16)
            w = lo.double().unsqueeze(-1) + codes.double() * step.double().unsqueeze(-1)
            out[rows] = w.to(torch.float16).double().reshape(-1, k)
    return out


def _fwht_groups(x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    """Transform the flagged 16-column groups of ``x[..., K]`` (float64 in/out)."""
    k = x.shape[-1]
    xg = x.reshape(*x.shape[:-1], k // GROUP, GROUP)
    rot = xg @ _hadamard16()
    out = torch.where(flags.bool().reshape(-1, 1), rot, xg)
    return out.reshape(x.shape)


def decode_part(
    t: dict[str, torch.Tensor], expert: int, part: int, parts: int
) -> torch.Tensor:
    """float64 ``[R, K]`` of one expert's part in the original basis."""
    r0, r1 = part_rows(t, expert * parts + part)
    w = _fwht_groups(dequantize_stored_rows(t, r0, r1), t["had"][expert])
    scale = t.get("col_scale")
    if scale is not None and scale.numel():  # empty = the checkpoint has none
        w = w / scale[expert].double()
    return w


def expert_weights(
    w13: dict[str, torch.Tensor], w2: dict[str, torch.Tensor], expert: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(gate, up, down)`` float64 original-basis weights of one expert."""
    gate = decode_part(w13, expert, 0, 2)
    up = decode_part(w13, expert, 1, 2)
    down = decode_part(w2, expert, 0, 1)
    return gate, up, down


# ---------------------------------------------------------------------------
# routed MoE reference
# ---------------------------------------------------------------------------


def softcapped_swiglu(
    gate: torch.Tensor, up: torch.Tensor, gate_softcap: float, up_softcap: float
) -> torch.Tensor:
    """``softcap(gate) * sigmoid(gate) * softcap(up)`` in fp32, cast back."""
    out_dtype = gate.dtype
    gate, up = gate.float(), up.float()
    out = (
        torch.tanh(gate / gate_softcap)
        * gate_softcap
        * torch.sigmoid(gate)
        * (torch.tanh(up / up_softcap) * up_softcap)
    )
    return out.to(out_dtype)


def reference_moe(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w13: dict[str, torch.Tensor],
    w2: dict[str, torch.Tensor],
    gate_softcap: float,
    up_softcap: float,
    *,
    experts: int | None = None,
    weights_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Routed MoE forward of the reference model in fp32.

    ``hidden_states [T, H]`` (the model dtype), ``top_k_index [T, k]`` int,
    ``top_k_weights [T, k]`` fp32 (already scaled by ``route_scale``).  Every
    expert is dequantized once, the gate/up products and the down projection are
    fp32 ``F.linear``, the top-k outputs are weighted and accumulated in fp32 (as
    the reference class does), and the result is cast to the input dtype.
    """
    e_total = num_experts(w13) if experts is None else experts
    final = torch.zeros_like(hidden_states, dtype=torch.float32)
    x = hidden_states.float()
    for expert in torch.unique(top_k_index).tolist():
        if expert < 0 or expert >= e_total:
            continue
        top_k_pos, token_idx = torch.where((top_k_index == expert).t())
        if token_idx.numel() == 0:
            continue
        gate, up, down = expert_weights(w13, w2, int(expert))
        state = x[token_idx]
        cur = softcapped_swiglu(
            state @ gate.to(weights_dtype).T,
            state @ up.to(weights_dtype).T,
            gate_softcap,
            up_softcap,
        )
        cur = cur @ down.to(weights_dtype).T
        cur = cur * top_k_weights[token_idx, top_k_pos, None].float()
        final.index_add_(0, token_idx, cur)
    return final.to(hidden_states.dtype)
