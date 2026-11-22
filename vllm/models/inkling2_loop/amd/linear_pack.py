# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight packing and call helpers of the Inkling-2 ``inkling2_linear_*`` kernels.

Layouts, activation codes and the kernel contracts are documented in
``docs/rdna2/inkling2-linear-kernels.md``. Everything here runs once at load
time (packing) or only forwards to the native ops (``__call__``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import torch

from vllm import _custom_ops as ops

ACT_NONE = 0
ACT_SIGMOID = 1
ACT_TANH = 2
ACT_SOFTCAP = 3

# The grouped op reads consecutive row blocks of one weight, so the packed row
# order below is the segment order of every call.
ATTN_IN_NAMES = ("q", "gate", "rel", "sink", "k", "dyn_k", "v", "dyn_v")


@dataclasses.dataclass(frozen=True)
class LinearPack:
    """One packed weight [sum(seg_n), K] and the segments it serves.

    Segment ``i`` multiplies ``xs[seg_x[i]]`` with its ``seg_n[i]`` rows of
    ``weight``, applies ``seg_act[i]`` (``seg_p[i]`` is the softcap) and writes
    ``out_dtypes[i]`` values to ``outs[i]`` ([M, seg_n[i]]).
    """

    weight: torch.Tensor
    names: tuple[str, ...]
    seg_n: tuple[int, ...]
    seg_x: tuple[int, ...]
    out_dtypes: tuple[torch.dtype, ...]
    seg_act: tuple[int, ...] = ()
    seg_p: tuple[float, ...] = ()
    seg_valid: tuple[int, ...] = ()

    @property
    def num_inputs(self) -> int:
        return max(self.seg_x) + 1

    def alloc_outputs(
        self, rows: int, device: torch.device | str | None = None
    ) -> list[torch.Tensor]:
        """Output tensors for ``rows`` tokens, one per segment.

        Under CUDA graphs allocate once for the largest batch and pass the row
        slices ``out[:m]`` (contiguous, so every row stride stays valid).
        """
        device = self.weight.device if device is None else device
        return [
            torch.empty(rows, n, dtype=dt, device=device)
            for n, dt in zip(self.seg_n, self.out_dtypes)
        ]

    def __call__(
        self,
        xs: Sequence[torch.Tensor],
        outs: Sequence[torch.Tensor],
        variant: int = -1,
    ) -> None:
        if self.weight.dtype == torch.float32:
            ops.inkling2_linear_f32w(
                list(xs),
                self.weight,
                list(outs),
                list(self.seg_x),
                list(self.seg_n),
                list(self.seg_act),
                variant,
            )
        else:
            ops.inkling2_linear_grouped(
                list(xs),
                self.weight,
                list(outs),
                list(self.seg_x),
                list(self.seg_n),
                list(self.seg_act),
                list(self.seg_p),
                list(self.seg_valid),
                variant,
            )


def _pack(
    parts: Sequence[tuple[str, torch.Tensor, int, torch.dtype, int, float]],
    dtype: torch.dtype,
) -> LinearPack:
    names, weights, seg_x, out_dtypes, acts, ps = zip(*parts)
    ks = {w.shape[1] for w in weights}
    if len(ks) != 1 or any(w.dim() != 2 for w in weights):
        raise ValueError(f"packed weights need one K, got {sorted(ks)}")
    for name, w in zip(names, weights):
        if w.dtype != dtype:
            raise ValueError(f"{name}: expected {dtype}, got {w.dtype}")
    seg_n = tuple(int(w.shape[0]) for w in weights)
    return LinearPack(
        weight=torch.cat([w.detach() for w in weights], dim=0).contiguous(),
        names=tuple(names),
        seg_n=seg_n,
        seg_x=tuple(seg_x),
        out_dtypes=tuple(out_dtypes),
        seg_act=tuple(acts),
        seg_p=tuple(ps),
        seg_valid=seg_n,
    )


def inkling2_linear_pack_attn_in(
    q: torch.Tensor,
    gate: torch.Tensor,
    rel: torch.Tensor,
    sink: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dyn_k: torch.Tensor | None = None,
    dyn_v: torch.Tensor | None = None,
) -> LinearPack:
    """Attention input projections of one layer in ONE launch.

    Row order: ``q | gate | rel | sink | k [| dyn_k] | v [| dyn_v]``. The first
    four read ``xs[0]`` (hq), ``k`` / ``dyn_k`` read ``xs[1]`` (hk), ``v`` /
    ``dyn_v`` read ``xs[2]`` (hv); pass the same tensor three times when the
    layer has no per-stream messages. All outputs are fp16: the layouts that
    ``inkling2_attention`` / ``inkling2_kv_prep`` consume (gate stays the raw
    logit, the attention epilogue applies the sigmoid).

    Args:
        q: ``[nh * D, H]`` query projection.
        gate: ``[nh * D, H]`` output-gate projection (raw logit).
        rel: ``[nh * d_rel, H]`` relative-position projection.
        sink: ``[nh, H]`` attention-sink projection.
        k: ``[nkv * D, H]`` key projection.
        v: ``[nkv * D, H]`` value projection.
        dyn_k: ``[nkv * ksz, H]`` dynamic-conv taps of the key stream
            (full-attention layers only).
        dyn_v: ``[nkv * ksz, H]`` dynamic-conv taps of the value stream.

    """
    if (dyn_k is None) != (dyn_v is None):
        raise ValueError("dyn_k and dyn_v go together")
    f16 = torch.float16
    parts = [
        ("q", q, 0, f16, ACT_NONE, 1.0),
        ("gate", gate, 0, f16, ACT_NONE, 1.0),
        ("rel", rel, 0, f16, ACT_NONE, 1.0),
        ("sink", sink, 0, f16, ACT_NONE, 1.0),
        ("k", k, 1, f16, ACT_NONE, 1.0),
    ]
    if dyn_k is not None:
        parts.append(("dyn_k", dyn_k, 1, f16, ACT_NONE, 1.0))
    parts.append(("v", v, 2, f16, ACT_NONE, 1.0))
    if dyn_v is not None:
        parts.append(("dyn_v", dyn_v, 2, f16, ACT_NONE, 1.0))
    return _pack(parts, torch.float16)


def inkling2_linear_pack_o_proj(
    weight: torch.Tensor, out_dtype: torch.dtype = torch.float16
) -> LinearPack:
    """``o_proj.linear`` ``[H, nh * D]`` (K-sharded under TP: ``[H, nh_local * D]``).

    ``out_dtype=torch.float32`` gives the fp32 partial that a tensor-parallel
    reduction (or the fp32 post-attention short conv) consumes.
    """
    return _pack([("o", weight, 0, out_dtype, ACT_NONE, 1.0)], torch.float16)


def inkling2_linear_pack_down(weight: torch.Tensor) -> LinearPack:
    """Controller ``down_proj`` ``[rank, 2H]`` fp16 -> fp16 ``[M, rank]``."""
    return _pack([("down", weight, 0, torch.float16, ACT_NONE, 1.0)], torch.float16)


def inkling2_linear_pack_router(weight: torch.Tensor) -> LinearPack:
    """MoE router ``[E, H]`` fp32 weights, fp16 input, fp32 logits ``[M, E]``."""
    return _pack([("router", weight, 0, torch.float32, ACT_NONE, 1.0)], torch.float32)


def inkling2_linear_pack_controller(
    forget: torch.Tensor, gain: torch.Tensor
) -> LinearPack:
    """Controller ``forget_gate_proj | message_gate_proj`` (fp32 ``[H, rank]``).

    One launch, fp16 ``gate_hidden`` ``[M, rank]`` in, fp32 ``tanh`` gates out:
    ``outs[0]`` = forget gate, ``outs[1]`` = message gate (both ``[M, H]``).
    """
    return _pack(
        [
            ("forget", forget, 0, torch.float32, ACT_TANH, 1.0),
            ("gain", gain, 0, torch.float32, ACT_TANH, 1.0),
        ],
        torch.float32,
    )


@dataclasses.dataclass(frozen=True)
class LMHeadPack:
    """The ``lm_head`` rows of one tensor-parallel rank.

    ``weight`` is ``[rows, K]`` fp16; row ``r`` is vocabulary id ``col_base + r``
    and rows ``>= n_valid`` are padding (their logits are written as ``-inf``).
    """

    weight: torch.Tensor
    n_valid: int
    col_base: int

    @property
    def rows(self) -> int:
        return int(self.weight.shape[0])

    def __call__(
        self,
        x: torch.Tensor,
        logits: torch.Tensor | None,
        part_val: torch.Tensor | None = None,
        part_idx: torch.Tensor | None = None,
        softcap: float = 30.0,
        variant: int = -1,
    ) -> None:
        ops.inkling2_linear_lm_head(
            x,
            self.weight,
            logits,
            part_val,
            part_idx,
            softcap,
            self.n_valid,
            self.col_base,
            variant,
        )


def inkling2_linear_pack_lm_head(
    weight: torch.Tensor, vocab_size: int, tp_size: int = 1, tp_rank: int = 0
) -> LMHeadPack:
    """Rank ``tp_rank`` of a vocab-parallel ``lm_head`` (fp16 ``[V_ckpt, K]``).

    TP=1 keeps exactly ``vocab_size`` rows (fp32 logits ``[M, vocab_size]``, the
    layout the V2 sampler reads). TP>1 follows ``VocabParallelEmbedding``: the
    vocabulary is padded to a multiple of 64, then split evenly, so every rank
    writes ``[M, padded // tp]`` logits whose rows past ``vocab_size`` are
    ``-inf``.
    """
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank {tp_rank} outside [0, {tp_size})")
    if weight.shape[0] < vocab_size:
        raise ValueError(f"lm_head has {weight.shape[0]} rows < vocab {vocab_size}")
    if tp_size == 1:
        return LMHeadPack(weight[:vocab_size].contiguous(), vocab_size, 0)
    padded = -(-vocab_size // 64) * 64
    if padded % tp_size:
        raise ValueError(f"padded vocab {padded} not divisible by tp {tp_size}")
    per = padded // tp_size
    lo, hi = tp_rank * per, (tp_rank + 1) * per
    shard = weight.new_zeros(per, weight.shape[1])
    take = min(hi, vocab_size) - lo
    if take > 0:
        shard[:take] = weight[lo : lo + take]
    return LMHeadPack(shard, max(take, 0), lo)
