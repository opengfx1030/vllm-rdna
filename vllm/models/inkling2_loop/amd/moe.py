# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling-2-Mini (``inkling2_loop``) sparse MoE block, standalone PyTorch.

Mirrors ``Inkling2LoopTopKRouter`` / ``Inkling2LoopExperts`` /
``Inkling2LoopSparseMoeBlock`` of the transformers reference.  The router and the
block expose the same state-dict keys as the reference (``gate.weight``,
``gate.e_score_correction_bias``); the experts storage is pluggable:

* :class:`ExpertsReference` -- dense ``gate_up_proj [E, 2I, H]`` /
  ``down_proj [E, H, I]`` Parameters, identical to the reference experts.
* :class:`ExpertsNvfp4` / :class:`ExpertsMxfp8` -- the checkpoint's block-scaled
  weights, run by the gfx1030 kernels ``torch.ops._rocm_C.moe_nvfp4_gemm_rdna2``
  / ``moe_mxfp8_gemm_rdna2`` (fp16 activations, fp32 activation function).

Weight layout (model == reference ``gate_up_proj``):
    ``w13[e]`` is ``[2I, H]`` with rows ``[0, I)`` = **gate** and rows ``[I, 2I)`` =
    **up** (the reference does ``gate_up.chunk(2, dim=-1)`` -> ``gate, up``).  The
    kernels' N axis is that same row order, so gate_up output columns ``[:I]`` are
    gate and ``[I:]`` are up.  The *checkpoint* stores the rows interleaved
    (``[g0, u0, g1, u1, ...]``, verified on the real Inkling-2-Mini weights: rows
    ``2i`` / ``2i + 1`` are correlated, rows ``i`` / ``i + I`` are not); the
    transformers conversion ``Interleave(dim=1)`` de-interleaves them, and so does
    ``Inkling2LoopForConditionalGeneration.load_weights``
    (``copy_gate_up_deinterleaved``).  :class:`Inkling2LoopMoE` and the helpers
    below take the model layout, not the checkpoint layout.

Activation: ``softcap(gate, 4) * sigmoid(gate) * softcap(up, 25)`` in fp32 with
``softcap(x, c) = tanh(x / c) * c`` (the sigmoid sees the *raw* gate).

Kernel path details:
    * ``block_size_m`` (1/2/4/8) is the number of routed ROWS per expert block
      after ``moe_align_block_size``, not a token-count limit, so any number of
      tokens works in one call.  Tokens are nevertheless processed in chunks of
      ``max_tokens_per_call`` to bound the ``[M * top_k, 2I]`` fp16 scratch.
    * Both GEMMs accumulate into zero-initialised fp16 outputs with atomics.  The
      second GEMM fuses the router weight and the sum over the top-k experts
      (``mul_topk_weight=True``, ``output_topk=top_k``), so the weighted
      expert sum is accumulated in fp16 (the reference accumulates in fp32).

This module only needs torch; the two vLLM helpers used by the kernel path
(``moe_align_block_size``) are imported lazily.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture

from .glue import FUSE_GLUE

__all__ = [
    "ExpertsMxfp8",
    "ExpertsNvfp4",
    "ExpertsReference",
    "Inkling2LoopExperts",
    "Inkling2LoopMoE",
    "Inkling2LoopSparseMoeBlock",
    "Inkling2LoopTopKRouter",
    "moe_offload_config",
    "moe_offload_prefetch_config",
    "repack_mxfp8",
    "repack_nvfp4",
    "softcapped_swiglu",
]

_NVFP4_PAYLOAD_NAMES = (
    "w13_qweight",
    "w13_scales",
    "w13_globals",
    "w2_qweight",
    "w2_scales",
    "w2_globals",
)
_MXFP8_PAYLOAD_NAMES = (
    "w13_qweight",
    "w13_scales",
    "w2_qweight",
    "w2_scales",
)


def moe_offload_config(
    env: Any = None,
) -> tuple[int, str, str, int]:
    """Read Inkling-2 MoE expert-offload settings from the environment.

    Returns ``(slots, copy_mode, policy, decay_interval)``.  ``slots == 0``
    (default) disables offloading completely.
    """
    src = os.environ if env is None else env
    raw_slots = src.get("VLLM_INKLING2_MOE_OFFLOAD_SLOTS", "0").strip()
    slots = int(raw_slots) if raw_slots else 0
    if slots < 0:
        raise ValueError(f"VLLM_INKLING2_MOE_OFFLOAD_SLOTS must be >= 0, got {slots}")
    copy_mode = src.get("VLLM_INKLING2_MOE_OFFLOAD_COPY", "tensor").strip() or "tensor"
    if copy_mode not in ("tensor", "native"):
        raise ValueError(
            f"VLLM_INKLING2_MOE_OFFLOAD_COPY must be 'tensor' or 'native', "
            f"got {copy_mode!r}"
        )
    policy = src.get("VLLM_INKLING2_MOE_OFFLOAD_POLICY", "lfu_lru").strip() or "lfu_lru"
    if policy not in ("lfu_lru", "lru"):
        raise ValueError(
            f"VLLM_INKLING2_MOE_OFFLOAD_POLICY must be 'lfu_lru' or 'lru', "
            f"got {policy!r}"
        )
    raw_decay = src.get("VLLM_INKLING2_MOE_OFFLOAD_DECAY", "128").strip()
    decay = int(raw_decay) if raw_decay else 128
    if decay <= 0:
        raise ValueError(f"VLLM_INKLING2_MOE_OFFLOAD_DECAY must be > 0, got {decay}")
    return slots, copy_mode, policy, decay


def moe_offload_prefetch_config(env: Any = None) -> tuple[bool, int, int]:
    """Read Inkling-2 MoE layer-ahead async prefetch settings from the environment.

    Returns ``(enabled, topk_limit, max_experts)``.  ``topk_limit == 0`` uses
    all routed top-k experts per token; ``max_experts`` caps the number of
    missing experts queued onto ``copy_stream`` per layer visit.
    """
    src = os.environ if env is None else env
    raw_en = src.get("VLLM_INKLING2_MOE_OFFLOAD_PREFETCH", "1").strip()
    enabled = raw_en not in ("", "0")
    raw_topk = src.get("VLLM_INKLING2_MOE_OFFLOAD_PREFETCH_TOPK", "4").strip()
    topk_limit = int(raw_topk) if raw_topk else 4
    if topk_limit < 0:
        raise ValueError(
            f"VLLM_INKLING2_MOE_OFFLOAD_PREFETCH_TOPK must be >= 0, got {topk_limit}"
        )
    raw_max = src.get("VLLM_INKLING2_MOE_OFFLOAD_PREFETCH_MAX", "4").strip()
    max_experts = int(raw_max) if raw_max else 4
    if max_experts <= 0:
        raise ValueError(
            f"VLLM_INKLING2_MOE_OFFLOAD_PREFETCH_MAX must be > 0, got {max_experts}"
        )
    return enabled, topk_limit, max_experts


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class Inkling2LoopTopKRouter(nn.Module):
    """Sigmoid router; ``e_score_correction_bias`` only influences selection."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        if config.scoring_func != "sigmoid":
            raise ValueError(f"unsupported scoring_func {config.scoring_func!r}")
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.routed_scaling_factor = config.route_scale
        self.weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts, dtype=torch.float32),
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(logits [T, E], weights [T, k], indices [T, k])`` in fp32."""
        flat = hidden_states.float().reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = torch.sigmoid(logits)
        indices = torch.topk(
            scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False
        ).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def softcapped_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_softcap: float,
    up_softcap: float,
) -> torch.Tensor:
    """``softcap(gate) * sigmoid(gate) * softcap(up)`` in fp32, cast back to
    ``gate.dtype``."""
    out_dtype = gate.dtype
    gate, up = gate.float(), up.float()
    out = (
        torch.tanh(gate / gate_softcap)
        * gate_softcap
        * torch.sigmoid(gate)
        * (torch.tanh(up / up_softcap) * up_softcap)
    )
    return out.to(out_dtype)


def _require_gfx10x(where: str) -> None:
    """The HI2 expert GEMMs are gfx1030 kernels (``rdna_hint_flex_rdna2.cu``).

    vLLM registers an empty stub on other architectures; refuse them here instead
    of returning zeros from a model that looks healthy.
    """
    from vllm.platforms.rocm import on_gfx10x

    if not on_gfx10x():
        raise RuntimeError(f"{where}: the HI2 expert GEMMs need gfx1030 (RDNA2)")


def _sum_topk_rows(rows: torch.Tensor, topk_w: torch.Tensor, out: torch.Tensor) -> None:
    """``out[t] = sum_k rows[t * K + k] * topk_w[t, k]``, in place.

    ``rows`` is the 2-D ``[T * K, H]`` w2 output view of the workspace and
    ``topk_w`` is the workspace's fp32 ``[T, K]`` view; ``out`` is ``[T, H]``.
    The w2 GEMM writes the per-expert rows in that 2-D view, so the K axis is
    only a view here.
    """
    t, k = topk_w.shape
    y = rows.view(t, k, rows.shape[-1])
    y.mul_(topk_w.unsqueeze(-1))
    torch.sum(y, dim=1, out=out)


def _sum_topk_rows_f32(
    rows: torch.Tensor, topk_w: torch.Tensor, out: torch.Tensor
) -> None:
    """``out[t] = sum_k rows[t * K + k] * topk_w[t, k]`` in fp32, in ascending k.

    The deterministic replacement for the decode kernels' fp16 top-k epilogue
    (packed-fp16 CAS atomics: the sum rounds in arrival order, which changes
    between runs).  Element-wise multiplies and additions only -- no atomics,
    no reduction kernel -- so the result of a given set of routed rows does not
    depend on how ``sorted_ids`` ordered them.  ``rows`` is fp16 (the BM w2
    kernel's plain stores), the arithmetic is fp32, ``out`` is the layer's
    (fp16) output.
    """
    t, k = topk_w.shape
    y = rows.view(t, k, rows.shape[-1]).float()
    y.mul_(topk_w.unsqueeze(-1))
    acc = y[:, 0]
    for j in range(1, k):
        acc = acc + y[:, j]
    out.copy_(acc)


def _det_routed_rows(
    ws: Any, tokens: int, top_k: int, hidden: int, device: torch.device
) -> torch.Tensor:
    """``[tokens * top_k, hidden]`` fp16 scratch for the deterministic w2 stage.

    Cached on the workspace (one allocation per process / graph pool, like the
    HI2 path's ``moe_hidden_view``) so the serving forward does not allocate
    per step.  The BM kernels write one plain-store row per routed id, so this
    buffer needs no zeroing.
    """
    n = tokens * top_k
    device = torch.device(device)
    buf = getattr(ws, "_moe_det_rows", None)
    if (
        buf is None
        or buf.shape[0] < n
        or buf.shape[1] != hidden
        or buf.device != device
    ):
        cap = max(int(getattr(ws, "moe_chunk", tokens)), tokens) * top_k
        buf = torch.empty(cap, hidden, dtype=torch.float16, device=device)
        ws._moe_det_rows = buf
    return buf[:n]


# ---------------------------------------------------------------------------
# Experts
# ---------------------------------------------------------------------------


class Inkling2LoopExperts(nn.Module):
    """Interface: ``forward(hidden [T, H], top_k_index [T, k], top_k_weights
    [T, k]) -> [T, H]`` (dtype of ``hidden``)."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_softcap = float(config.mlp_gate_softcap)
        self.up_softcap = float(config.mlp_up_softcap)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return softcapped_swiglu(gate, up, self.gate_softcap, self.up_softcap)


class ExpertsReference(Inkling2LoopExperts):
    """Dense experts, same parameters and numerics as the reference class."""

    def __init__(
        self,
        config: Any,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(config)
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                2 * self.intermediate_dim,
                self.hidden_dim,
                dtype=dtype,
                device=device,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_dim,
                self.intermediate_dim,
                dtype=dtype,
                device=device,
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # accumulate the top-k expert outputs in fp32, as in the reference
        final = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert in torch.unique(top_k_index).tolist():
            top_k_pos, token_idx = torch.where((top_k_index == expert).t())
            state = hidden_states[token_idx]
            cur = self._apply_gate(F.linear(state, self.gate_up_proj[expert]))
            cur = F.linear(cur, self.down_proj[expert])
            cur = cur * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, cur.to(final.dtype))
        return final.to(hidden_states.dtype)


def _block_size_m(num_tokens: int, top_k: int, num_experts: int) -> int:
    """Routed rows per expert block (kernel supports 1, 2, 4, 8): the average
    number of rows an expert receives, rounded up to the next supported size,
    because every expert's row group is padded to a multiple of it."""
    avg = -(-num_tokens * top_k // num_experts)
    if avg <= 1:
        return 1
    if avg <= 2:
        return 2
    if avg <= 4:
        return 4
    return 8


class _ExpertsKernel(Inkling2LoopExperts):
    """Shared driver for the two block-scaled kernel formats."""

    #: tokens per kernel call (bounds the [M * top_k, 2I] fp16 scratch)
    max_tokens_per_call: int = 1024

    def _gemm(
        self,
        stage: str,
        a: torch.Tensor,
        c: torch.Tensor,
        topk_weights: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        top_k: int,
        block_size_m: int,
        mul_topk_weight: bool,
        output_topk: int,
    ) -> None:
        raise NotImplementedError

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        block_size_m: int | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        out = torch.empty_like(hidden_states)
        step = self.max_tokens_per_call
        for start in range(0, num_tokens, step):
            end = min(start + step, num_tokens)
            out[start:end] = self._forward_chunk(
                hidden_states[start:end],
                top_k_index[start:end],
                top_k_weights[start:end],
                block_size_m,
            )
        return out

    def _default_block_size_m(self, num_tokens: int, top_k: int) -> int:
        return _block_size_m(num_tokens, top_k, self.num_experts)

    def _forward_chunk(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        block_size_m: int | None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        num_tokens, top_k = top_k_index.shape
        device = hidden_states.device
        x = hidden_states.to(torch.float16).contiguous()
        ids = top_k_index.to(torch.int32).contiguous()
        if block_size_m is None:
            block_size_m = self._default_block_size_m(num_tokens, top_k)
        sorted_ids, expert_ids, ntpp = moe_align_block_size(
            ids, block_size_m, self.num_experts
        )
        no_weights = torch.empty(0, dtype=torch.float32, device=device)

        # GEMM 1: rows of the output are token * top_k + k; the kernel reads
        # activation row ``routed_id // top_k``.  Output must start at zero.
        gate_up = torch.zeros(
            num_tokens * top_k,
            2 * self.intermediate_dim,
            dtype=torch.float16,
            device=device,
        )
        out = torch.zeros(
            num_tokens, self.hidden_dim, dtype=torch.float16, device=device
        )
        topk_w_flat = top_k_weights.to(torch.float32).reshape(-1).contiguous()
        cache = getattr(self, "expert_cache", None)
        if cache is None:
            self._gemm(
                "w13",
                x,
                gate_up,
                no_weights,
                sorted_ids,
                expert_ids,
                ntpp,
                top_k,
                block_size_m,
                False,
                0,
            )
            act = self._apply_gate(gate_up).contiguous()  # fp32 math, fp16 result

            # GEMM 2: one activation row per routed id (top_k=1); the epilogue
            # scales by the router weight and sums the k rows of each token.
            self._gemm(
                "w2",
                act,
                out,
                topk_w_flat,
                sorted_ids,
                expert_ids,
                ntpp,
                1,
                block_size_m,
                True,
                top_k,
            )
            return out.to(hidden_states.dtype)

        local_ids = ids.cpu().reshape(-1).tolist()
        cache.policy.observe(local_ids)
        for pass_experts in cache.policy.plan_passes(local_ids):
            lease = cache.ensure(pass_experts)
            try:
                slot_expert_ids = cache.remap(expert_ids, lease)
                self._gemm(
                    "w13",
                    x,
                    gate_up,
                    no_weights,
                    sorted_ids,
                    slot_expert_ids,
                    ntpp,
                    top_k,
                    block_size_m,
                    False,
                    0,
                )
                act = self._apply_gate(gate_up).contiguous()
                self._gemm(
                    "w2",
                    act,
                    out,
                    topk_w_flat,
                    sorted_ids,
                    slot_expert_ids,
                    ntpp,
                    1,
                    block_size_m,
                    True,
                    top_k,
                )
            finally:
                cache.release(lease)
        return out.to(hidden_states.dtype)


def _pack_k_words(w: torch.Tensor) -> torch.Tensor:
    """[E, N, Kb] bytes -> [E, Kb/4, N] int32 (4 consecutive K bytes per
    little-endian word, K-first)."""
    num_e, n, kb = w.shape
    if kb % 4:
        raise ValueError(f"byte K dim {kb} must be a multiple of 4")
    w = w.contiguous().view(torch.uint8)
    out = torch.empty(num_e, kb // 4, n, dtype=torch.int32, device=w.device)
    for e in range(num_e):
        out[e] = w[e].view(torch.int32).t()
    return out


def _transpose_scales(s: torch.Tensor) -> torch.Tensor:
    """[E, N, K/block] one-byte scales -> contiguous uint8 [E, K/block, N]."""
    return s.contiguous().view(torch.uint8).transpose(1, 2).contiguous()


def repack_nvfp4(
    weight_u8: torch.Tensor,
    scale_e4m3: torch.Tensor,
    amax: torch.Tensor,
    denominator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Checkpoint NVFP4 -> kernel layout.

    ``weight_u8 [E, N, K/2]`` (two E2M1 per byte, low nibble = even k),
    ``scale_e4m3 [E, N, K/16]`` (uint8 or float8_e4m3fn), ``amax [E]``,
    ``denominator [1]`` (2688).  Returns ``b_q_weight [E, K/8, N]`` uint32 (word
    ``w`` holds k = 8w..8w+7, nibble i = k 8w+i), ``b_scales [E, K/16, N]``
    uint8 (raw e4m3 bytes) and ``b_globals [E]`` fp32 = ``amax / denominator``.
    """
    num_e, n, kd2 = weight_u8.shape
    k = kd2 * 2
    if k % 32 or n % 8:
        raise ValueError(f"NVFP4 kernel needs K % 32 == 0 and N % 8 == 0, got {k}, {n}")
    if tuple(scale_e4m3.shape) != (num_e, n, k // 16):
        raise ValueError(
            f"scale shape {tuple(scale_e4m3.shape)} != {(num_e, n, k // 16)}"
        )
    b_q = _pack_k_words(weight_u8).view(torch.uint32)
    b_s = _transpose_scales(scale_e4m3)
    b_g = amax.to(torch.float32).reshape(num_e) / denominator.to(torch.float32).reshape(
        ()
    )
    return b_q, b_s, b_g.contiguous()


#: UE8M0 scale bytes the MXFP8 kernel supports: its fp16 factor 2^(scale - 119) is a
#: fp16 number for 95..134 (the checkpoint's scales are 113..118).
MXFP8_SCALE_RANGE = (95, 134)


def repack_mxfp8(
    weight_e4m3: torch.Tensor, scale_ue8m0: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Checkpoint MXFP8 -> kernel layout.

    ``weight_e4m3 [E, N, K]`` (float8_e4m3fn or uint8), ``scale_ue8m0
    [E, N, K/32]``.  Returns ``[E, K/4, N]`` uint32 (4 consecutive K bytes per
    word, byte 0 = lowest k) and ``[E, K/32, N]`` uint8.
    """
    num_e, n, k = weight_e4m3.shape
    if k % 32 or n % 8:
        raise ValueError(f"MXFP8 kernel needs K % 32 == 0 and N % 8 == 0, got {k}, {n}")
    if tuple(scale_ue8m0.shape) != (num_e, n, k // 32):
        raise ValueError(
            f"scale shape {tuple(scale_ue8m0.shape)} != {(num_e, n, k // 32)}"
        )
    scales = _transpose_scales(scale_ue8m0)
    if not scales.is_meta:  # the meta-device header checks have no values to read
        low, high = int(scales.min()), int(scales.max())
        if low < MXFP8_SCALE_RANGE[0] or high > MXFP8_SCALE_RANGE[1]:
            raise ValueError(
                f"MXFP8 scale bytes span {low}..{high}; the kernel dequantizes with "
                f"the fp16 factor 2^(scale - 119), exact for "
                f"{MXFP8_SCALE_RANGE[0]}..{MXFP8_SCALE_RANGE[1]}"
            )
    return _pack_k_words(weight_e4m3).view(torch.uint32), scales


class ExpertsNvfp4(_ExpertsKernel):
    """NVFP4 experts (checkpoint layers 0-19).

    Buffers (kernel layout, filled by :meth:`load_checkpoint` or directly):
    ``w13_qweight [E, H/8, 2I]``, ``w13_scales [E, H/16, 2I]``, ``w13_globals
    [E]``, ``w2_qweight [E, I/8, H]``, ``w2_scales [E, I/16, H]``, ``w2_globals
    [E]``.  Packed words are stored as int32 (bit-identical to uint32).
    """

    def __init__(self, config: Any, device: torch.device | str | None = None) -> None:
        super().__init__(config)
        e, h, i = self.num_experts, self.hidden_dim, self.intermediate_dim
        for name, k, n in (("w13", h, 2 * i), ("w2", i, h)):
            self.register_buffer(
                f"{name}_qweight",
                torch.empty(e, k // 8, n, dtype=torch.int32, device=device),
            )
            self.register_buffer(
                f"{name}_scales",
                torch.empty(e, k // 16, n, dtype=torch.uint8, device=device),
            )
            self.register_buffer(
                f"{name}_globals", torch.empty(e, dtype=torch.float32, device=device)
            )

    @torch.no_grad()
    def load_checkpoint(
        self,
        w13_weight: torch.Tensor,
        w13_scale: torch.Tensor,
        w13_amax: torch.Tensor,
        w13_denominator: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_scale: torch.Tensor,
        w2_amax: torch.Tensor,
        w2_denominator: torch.Tensor,
    ) -> None:
        """Repack checkpoint tensors (``w13_weight [E, 2I, H/2]`` ...) in place."""
        for name, args in (
            ("w13", (w13_weight, w13_scale, w13_amax, w13_denominator)),
            ("w2", (w2_weight, w2_scale, w2_amax, w2_denominator)),
        ):
            q, s, g = repack_nvfp4(*args)
            getattr(self, f"{name}_qweight").copy_(q.view(torch.int32))
            getattr(self, f"{name}_scales").copy_(s)
            getattr(self, f"{name}_globals").copy_(g)

    def _gemm(
        self,
        stage,
        a,
        c,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_size_m,
        mul_topk_weight,
        output_topk,
    ) -> None:
        torch.ops._rocm_C.moe_nvfp4_gemm_rdna2(
            a,
            c,
            getattr(self, f"{stage}_qweight"),
            getattr(self, f"{stage}_scales"),
            getattr(self, f"{stage}_globals"),
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            block_size_m,
            mul_topk_weight,
            output_topk,
        )


class ExpertsMxfp8(_ExpertsKernel):
    """MXFP8 experts (checkpoint layers 20-23).

    Buffers: ``w13_qweight [E, H/4, 2I]``, ``w13_scales [E, H/32, 2I]``,
    ``w2_qweight [E, I/4, H]``, ``w2_scales [E, I/32, H]`` (int32 words, uint8
    UE8M0 scales).
    """

    def __init__(self, config: Any, device: torch.device | str | None = None) -> None:
        super().__init__(config)
        e, h, i = self.num_experts, self.hidden_dim, self.intermediate_dim
        for name, k, n in (("w13", h, 2 * i), ("w2", i, h)):
            self.register_buffer(
                f"{name}_qweight",
                torch.empty(e, k // 4, n, dtype=torch.int32, device=device),
            )
            self.register_buffer(
                f"{name}_scales",
                torch.empty(e, k // 32, n, dtype=torch.uint8, device=device),
            )

    @torch.no_grad()
    def load_checkpoint(
        self,
        w13_weight: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        """Repack checkpoint tensors (``w13_weight [E, 2I, H]`` ...) in place."""
        for name, args in (
            ("w13", (w13_weight, w13_scale)),
            ("w2", (w2_weight, w2_scale)),
        ):
            q, s = repack_mxfp8(*args)
            getattr(self, f"{name}_qweight").copy_(q.view(torch.int32))
            getattr(self, f"{name}_scales").copy_(s)

    def _gemm(
        self,
        stage,
        a,
        c,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_size_m,
        mul_topk_weight,
        output_topk,
    ) -> None:
        torch.ops._rocm_C.moe_mxfp8_gemm_rdna2(
            a,
            c,
            getattr(self, f"{stage}_qweight"),
            getattr(self, f"{stage}_scales"),
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            block_size_m,
            mul_topk_weight,
            output_topk,
        )


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


class Inkling2LoopSparseMoeBlock(nn.Module):
    """Router + experts.  ``experts_cls`` selects the experts storage."""

    def __init__(
        self,
        config: Any,
        experts_cls: type[Inkling2LoopExperts] = ExpertsReference,
        **experts_kwargs: Any,
    ) -> None:
        super().__init__()
        self.experts = experts_cls(config, **experts_kwargs)
        self.gate = Inkling2LoopTopKRouter(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``[B, S, H]`` or ``[T, H]`` in, same shape/dtype out."""
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        _, routing_weights, selected = self.gate(flat)
        return self.experts(flat, selected, routing_weights).reshape(shape)


# ---------------------------------------------------------------------------
# Serving module (one layer; under tensor parallelism one rank's shard of it)
# ---------------------------------------------------------------------------


class _MoEGate(Inkling2LoopTopKRouter):
    """Router whose selection bias is a (frozen) Parameter, so a loader that
    walks ``named_parameters()`` sees both ``weight`` and
    ``e_score_correction_bias`` (checkpoint key ``mlp.gate.bias``)."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        bias = self._buffers.pop("e_score_correction_bias")
        self.e_score_correction_bias = nn.Parameter(bias, requires_grad=False)


#: Routed rows per expert block of the serving expert GEMMs by token count, per expert
#: format: ``((largest token count, rows), ...)``, ``_BLOCK_M_WIDE`` above the last.
#: A block holds the rows of ONE expert and 8 * T routed rows spread over 128 experts,
#: so up to a few dozen tokens nearly every block holds a single row and a wider block
#: only adds padded rows of dot2 work; from ~64 tokens experts get several rows and
#: wide blocks stream each expert tile once.  Measured per launch on the V620 at the
#: real layer shapes (``tools/rdna/inkling2/moe_rows_bench.py``, see
#: ``docs/rdna2/inkling2-port-spec.md``): the width chosen is the fastest of the four
#: at every measured token count or within 3% of it, and 5 tokens in 8-row blocks cost
#: 387 us per NVFP4 layer (w13 + w2) against 241 us in 2-row blocks.
_BLOCK_M_BY_TOKENS = {
    "nvfp4": ((4, 1), (12, 2), (64, 4)),
    "mxfp8": ((4, 1), (48, 2)),
}
_BLOCK_M_WIDE = 8

#: ``VLLM_INKLING2_FUSE_GLUE``: with the real routing of speculative verify steps
#: (adjacent rows share ~44% of their experts: 12.4 / 16.3 / 19.7 / 22.3 distinct
#: experts for 2 / 3 / 4 / 5 rows, against 15.5 / 22.5 / 29.1 / 35.3 for uniform
#: routing) two-row blocks stream a shared expert once and are faster than one-row
#: blocks from 2 rows on (NVFP4 w13 + w2 per layer visit,
#: ``docs/rdna2/inkling2-decode-glue.md``: 2 rows 174 vs 181 us, 3 rows 225 vs
#: 238 us, 4 rows 268 vs 302 us, 5 rows 313 vs 362 us); a single row stays in
#: one-row blocks.
_BLOCK_M_BY_TOKENS_SPEC = {
    "nvfp4": ((1, 1), (12, 2), (64, 4)),
    "mxfp8": ((1, 1), (12, 2), (48, 4)),
}

#: Optional knob ``VLLM_INKLING2_SPEC_BM4_MIN`` (default ``0`` = off): when ``> 0``
#: (e.g. ``8``), raises ``block_size_m`` to at least ``4`` from that token count up
#: under ``FUSE_GLUE``, allowing A/B sweeps of correlated verify routing without
#: rebuilding.
_SPEC_BM4_MIN = int(os.environ.get("VLLM_INKLING2_SPEC_BM4_MIN", "0"))

#: ``VLLM_INKLING2_FUSE_GLUE``: chunks of at most this many tokens run the SwiGLU
#: inside the w2 GEMM (``moe_*_gemm_swiglu_rdna2``).  The fused load recomputes the
#: activation once per N tile, which saves a launch and an activation round trip at
#: decode / verify sizes (-4.1 us per layer visit at 1 row) but costs +1.1% at 64
#: tokens, +5.5% at 256 and +7.1% at 1000 tokens per NVFP4 w2 launch
#: (``moe_lab_big``).
_FUSE_SWIGLU_MAX_TOKENS = 16


#: Prefill expert GEMMs (``csrc/rocm/moe_prefill_bm_rdna2.cu``): 32-row work blocks
#: grouped from the router's 8-padded layout, whole K, one dequantisation of every
#: weight word per block (docs/rdna2/inkling2-moe-prefill-bm.md).  ON by default for
#: chunks of at least ``VLLM_INKLING2_PREFILL_BM_MIN_TOKENS`` tokens (smaller ones keep
#: the decode-shaped kernels above); ``VLLM_INKLING2_PREFILL_BM=0`` restores the
#: decode-shaped kernels at every chunk size (A/B and rollback knob).
#: ``VLLM_INKLING2_PREFILL_INT8=1`` (OFF by default, only where the dot2 prefill path is
#: eligible) runs the NVFP4 layers as W4A8: int8 activations (one scale per row and
#: 32 K), V_DOT4_I32_I8; the MXFP8 layers keep the fp16 dot2 kernel.  The numerics
#: change: see the NLL table of the doc.
def _prefill_switches(env) -> tuple[bool, bool, int]:
    """(dot2 prefill kernels on, W4A8 on, token gate) from the environment."""
    bm = env.get("VLLM_INKLING2_PREFILL_BM", "1") == "1"
    int8 = bm and env.get("VLLM_INKLING2_PREFILL_INT8", "0") == "1"
    return bm, int8, int(env.get("VLLM_INKLING2_PREFILL_BM_MIN_TOKENS", "128"))


_PREFILL_BM, _PREFILL_INT8, _PREFILL_BM_MIN_TOKENS = _prefill_switches(os.environ)


_ANNOUNCED_MESSAGES: set[tuple[str, tuple[Any, ...]]] = set()


def _deterministic_switch(env) -> bool:
    """``VLLM_RDNA2_MOE_DETERMINISTIC``: the routed experts must be run-to-run
    bit-identical.  Same variable, same truth values and same "resolved once at
    import" semantics as the WNA16 compressed-tensors MoE
    (``compressed_tensors_moe_wna16_rdna2.py`` / ``vllm.envs``); the stock
    Inkling-2 experts are NVFP4 / MXFP8, whose decode kernels reduce the K
    slices and the top-k rows with packed-fp16 CAS atomics."""
    return env.get("VLLM_RDNA2_MOE_DETERMINISTIC", "0").lower() in ("true", "1")


_DETERMINISTIC = _deterministic_switch(os.environ)


def _announce(message: str, *args: Any) -> None:
    """One INFO line per process through the vLLM logger (imported lazily: this module
    only needs torch)."""
    key = (message, args)
    if key in _ANNOUNCED_MESSAGES:
        return
    _ANNOUNCED_MESSAGES.add(key)
    from vllm.logger import init_logger

    init_logger(__name__).info_once(message, *args)


class Inkling2LoopMoE(_ExpertsKernel):
    """One Inkling-2-Mini MoE layer: fp32 sigmoid router + NVFP4 / MXFP8
    experts on the gfx1030 kernels.  ``forward`` takes flat ``[T, H]`` fp16.

    Under tensor parallelism (``tp_size > 1``) the router is replicated and every
    expert holds this rank's ``intermediate_size / tp_size`` units of the fused gate /
    up (``[gate; up]`` rows) and of the down projection's K; ``forward`` /
    ``forward_hip`` return this rank's *partial* output, which the caller all-reduces
    (the sum over ranks of the expert outputs is the TP=1 output).  Which slice a rank
    holds is decided by the loader (``loader.local_slice``); the module only needs the
    size.

    Lifecycle: the constructor creates the parameters in checkpoint layout
    (N-first).  Fill them with :meth:`load_expert_group` / :meth:`load_gate_bias`
    (or any ``param.data.copy_``), then call :meth:`process_weights_after_loading`,
    which repacks the experts into the kernel's K-major layout, frees the
    checkpoint tensors and registers the kernel tensors as non-persistent
    buffers.
    """

    #: experts repacked per step (bounds the temporary memory of the repack)
    repack_chunk_experts: int = 16

    def __init__(
        self,
        config: Any,
        quant_format: Literal["nvfp4", "mxfp8"],
        prefix: str = "",
        tp_size: int = 1,
        *,
        offload_slots: int | None = None,
        offload_copy: str | None = None,
        offload_policy: str | None = None,
        offload_decay: int | None = None,
        offload_prefetch: bool | None = None,
        offload_prefetch_topk: int | None = None,
        offload_prefetch_max: int | None = None,
    ) -> None:
        if quant_format not in ("nvfp4", "mxfp8"):
            raise ValueError(f"unsupported quant_format {quant_format!r}")
        super().__init__(config)
        if config.intermediate_size % tp_size:
            raise ValueError(
                f"intermediate_size {config.intermediate_size} is not divisible by "
                f"tensor parallel size {tp_size}"
            )
        env_slots, env_copy, env_policy, env_decay = moe_offload_config()
        env_pref, env_pref_topk, env_pref_max = moe_offload_prefetch_config()
        self.offload_slots = env_slots if offload_slots is None else int(offload_slots)
        self.offload_copy = env_copy if offload_copy is None else offload_copy
        self.offload_policy = env_policy if offload_policy is None else offload_policy
        self.offload_decay = env_decay if offload_decay is None else int(offload_decay)
        self.offload_prefetch = (
            env_pref if offload_prefetch is None else bool(offload_prefetch)
        )
        self.offload_prefetch_topk = (
            env_pref_topk
            if offload_prefetch_topk is None
            else int(offload_prefetch_topk)
        )
        self.offload_prefetch_max = (
            env_pref_max if offload_prefetch_max is None else int(offload_prefetch_max)
        )
        if self.offload_slots < 0:
            raise ValueError(f"offload_slots must be >= 0, got {self.offload_slots}")
        if self.offload_slots > 0:
            if tp_size != 1:
                raise ValueError(
                    "Inkling-2 MoE expert offloading currently supports TP=1 only "
                    f"(got tp_size={tp_size})"
                )
            if self.offload_copy not in ("tensor", "native"):
                raise ValueError(
                    f"offload_copy must be 'tensor' or 'native', got "
                    f"{self.offload_copy!r}"
                )
            if self.offload_policy not in ("lfu_lru", "lru"):
                raise ValueError(
                    f"offload_policy must be 'lfu_lru' or 'lru', got "
                    f"{self.offload_policy!r}"
                )
            if self.offload_decay <= 0:
                raise ValueError(f"offload_decay must be > 0, got {self.offload_decay}")
        self.tp_size = tp_size
        self.intermediate_dim = config.intermediate_size // tp_size
        self.quant_format = quant_format
        self.prefix = prefix
        self.top_k = config.num_experts_per_tok
        self.gate = _MoEGate(config)
        self.expert_cache: Any = None
        self._visit_demands: list[int] = [0, 0]
        self._visit_misses: list[int] = [0, 0]

        num_e, hidden, inter = self.num_experts, self.hidden_dim, self.intermediate_dim
        fp8 = torch.float8_e4m3fn
        # When offloading is enabled, stage the checkpoint-layout experts on CPU so
        # a 16 GiB GPU never allocates all 14.86 GiB of experts on device at once.
        expert_dev_kwargs: dict[str, Any] = (
            {"device": "cpu"} if self.offload_slots > 0 else {}
        )
        for stage, n, k in (("w13", 2 * inter, hidden), ("w2", hidden, inter)):
            if quant_format == "nvfp4":
                tensors = {
                    "weight": torch.empty(
                        num_e, n, k // 2, dtype=torch.uint8, **expert_dev_kwargs
                    ),
                    "weight_scale": torch.empty(
                        num_e, n, k // 16, dtype=fp8, **expert_dev_kwargs
                    ),
                    "weight_amax": torch.empty(
                        num_e, dtype=torch.float32, **expert_dev_kwargs
                    ),
                    "weight_outer_scale_denominator": torch.empty(
                        1, dtype=torch.float32, **expert_dev_kwargs
                    ),
                }
            else:
                tensors = {
                    "weight": torch.empty(num_e, n, k, dtype=fp8, **expert_dev_kwargs),
                    "weight_scale": torch.empty(
                        num_e, n, k // 32, dtype=torch.uint8, **expert_dev_kwargs
                    ),
                }
            for tail, tensor in tensors.items():
                setattr(
                    self, f"{stage}_{tail}", nn.Parameter(tensor, requires_grad=False)
                )
        self._expert_names = tuple(
            name for name, _ in self.named_parameters() if name.startswith("w")
        )
        self._processed = False

    # -- loading ----------------------------------------------------------

    @torch.no_grad()
    def load_expert_group(self, group: dict[str, torch.Tensor]) -> None:
        """Copy one layer's checkpoint expert tensors (keys are the tails after
        ``mlp.experts.``, e.g. ``w13_weight_scale``) into the parameters."""
        if set(group) != set(self._expert_names):
            raise ValueError(
                f"{self.prefix}: expert tensors {sorted(group)} != "
                f"{sorted(self._expert_names)}"
            )
        for name, tensor in group.items():
            param = getattr(self, name)
            if tensor.dtype != param.dtype or tensor.shape != param.shape:
                raise ValueError(
                    f"{self.prefix}.{name}: got {tensor.dtype} {tuple(tensor.shape)}, "
                    f"want {param.dtype} {tuple(param.shape)}"
                )
            param.data.copy_(tensor)

    @torch.no_grad()
    def load_gate_bias(self, bias: torch.Tensor) -> None:
        """Checkpoint ``mlp.gate.bias`` -> ``gate.e_score_correction_bias``."""
        self.gate.e_score_correction_bias.data.copy_(bias)

    @torch.no_grad()
    def process_weights_after_loading(self) -> None:
        if self._processed:
            return
        device = self.gate.weight.device
        step = self.repack_chunk_experts
        nvfp4 = self.quant_format == "nvfp4"
        if self.offload_slots > 0:
            self._process_weights_offload(device, step, nvfp4)
            return
        for stage in ("w13", "w2"):
            weight = getattr(self, f"{stage}_weight").data
            scale = getattr(self, f"{stage}_weight_scale").data
            if nvfp4:
                amax = getattr(self, f"{stage}_weight_amax").data
                den = getattr(self, f"{stage}_weight_outer_scale_denominator").data
            packed = scales = globals_ = None
            for lo in range(0, self.num_experts, step):
                hi = min(lo + step, self.num_experts)
                w_c, s_c = weight[lo:hi].to(device), scale[lo:hi].to(device)
                if nvfp4:
                    q_c, s_c, g_c = repack_nvfp4(
                        w_c, s_c, amax[lo:hi].to(device), den.to(device)
                    )
                else:
                    q_c, s_c = repack_mxfp8(w_c, s_c)
                q_c = q_c.view(torch.int32)
                if packed is None:
                    packed = torch.empty(
                        (self.num_experts, *q_c.shape[1:]),
                        dtype=torch.int32,
                        device=device,
                    )
                    scales = torch.empty(
                        (self.num_experts, *s_c.shape[1:]),
                        dtype=torch.uint8,
                        device=device,
                    )
                    if nvfp4:
                        globals_ = torch.empty(
                            self.num_experts, dtype=torch.float32, device=device
                        )
                assert packed is not None and scales is not None
                packed[lo:hi] = q_c
                scales[lo:hi] = s_c
                if nvfp4:
                    assert globals_ is not None
                    globals_[lo:hi] = g_c
            self.register_buffer(f"{stage}_qweight", packed, persistent=False)
            self.register_buffer(f"{stage}_scales", scales, persistent=False)
            if nvfp4:
                self.register_buffer(f"{stage}_globals", globals_, persistent=False)
        for name in self._expert_names:
            delattr(self, name)
        self._processed = True
        _announce(
            "Inkling-2 MoE prefill kernels: dot2 32-row %s, W4A8 %s, chunks of >= %d "
            "tokens (VLLM_INKLING2_PREFILL_BM / _INT8 / _BM_MIN_TOKENS)",
            "on" if _PREFILL_BM else "off",
            "on" if _PREFILL_INT8 else "off",
            _PREFILL_BM_MIN_TOKENS,
        )

    @torch.no_grad()
    def _process_weights_offload(
        self, device: torch.device, step: int, nvfp4: bool
    ) -> None:
        """Streaming repack into a pinned CPU bank and bounded GPU slot pool."""
        from vllm.model_executor.layers.fused_moe.rdna_expert_cache import (
            PackedExpertCache,
        )

        pin = device.type == "cuda"
        host_payload: dict[str, torch.Tensor] = {}
        for stage in ("w13", "w2"):
            weight = getattr(self, f"{stage}_weight").data
            scale = getattr(self, f"{stage}_weight_scale").data
            if nvfp4:
                amax = getattr(self, f"{stage}_weight_amax").data
                den = getattr(self, f"{stage}_weight_outer_scale_denominator").data
            packed = scales = globals_ = None
            for lo in range(0, self.num_experts, step):
                hi = min(lo + step, self.num_experts)
                w_c, s_c = weight[lo:hi].to(device), scale[lo:hi].to(device)
                if nvfp4:
                    q_c, s_c, g_c = repack_nvfp4(
                        w_c, s_c, amax[lo:hi].to(device), den.to(device)
                    )
                else:
                    q_c, s_c = repack_mxfp8(w_c, s_c)
                q_c = q_c.view(torch.int32)
                if packed is None:
                    packed = torch.empty(
                        (self.num_experts, *q_c.shape[1:]),
                        dtype=torch.int32,
                        device="cpu",
                        pin_memory=pin,
                    )
                    scales = torch.empty(
                        (self.num_experts, *s_c.shape[1:]),
                        dtype=torch.uint8,
                        device="cpu",
                        pin_memory=pin,
                    )
                    if nvfp4:
                        globals_ = torch.empty(
                            self.num_experts,
                            dtype=torch.float32,
                            device="cpu",
                            pin_memory=pin,
                        )
                assert packed is not None and scales is not None
                packed[lo:hi].copy_(q_c)
                scales[lo:hi].copy_(s_c)
                if nvfp4:
                    assert globals_ is not None
                    globals_[lo:hi].copy_(g_c)
                del w_c, s_c, q_c
                if nvfp4:
                    del g_c
            assert packed is not None and scales is not None
            host_payload[f"{stage}_qweight"] = packed
            host_payload[f"{stage}_scales"] = scales
            if nvfp4:
                assert globals_ is not None
                host_payload[f"{stage}_globals"] = globals_
            # Free this stage's raw checkpoint tensors before repacking the next
            # stage so host RAM does not double-buffer both stages simultaneously.
            del weight, scale
            if nvfp4:
                del amax, den
            for name in tuple(
                n for n in self._expert_names if n.startswith(f"{stage}_")
            ):
                if hasattr(self, name):
                    delattr(self, name)

        payload_names = _NVFP4_PAYLOAD_NAMES if nvfp4 else _MXFP8_PAYLOAD_NAMES
        self.expert_cache = PackedExpertCache(
            host_payload,
            slot_count=self.offload_slots,
            device=device,
            policy=self.offload_policy,
            decay_interval=self.offload_decay,
            copy_mode=self.offload_copy,
            payload_names=payload_names,
            allocate_dummy_slot=False,
        )
        # Bind the non-persistent buffers directly to the GPU slot pool so
        # _gemm / _gemm_swiglu require zero weight-pointer indirection changes.
        for name in payload_names:
            self.register_buffer(
                name, self.expert_cache.payload[name], persistent=False
            )
        self._processed = True
        _announce(
            "Inkling-2 MoE prefill kernels: dot2 32-row %s, W4A8 %s, chunks of >= %d "
            "tokens (VLLM_INKLING2_PREFILL_BM / _INT8 / _BM_MIN_TOKENS)",
            "on" if _PREFILL_BM else "off",
            "on" if _PREFILL_INT8 else "off",
            _PREFILL_BM_MIN_TOKENS,
        )

    def offload_stats(self) -> dict[str, Any] | None:
        """Return cache hit/miss and transfer counters, or None when resident."""
        cache = self.expert_cache
        if cache is None:
            return None
        demands = cache.expert_demands
        misses = cache.expert_misses
        return {
            "slots": cache.slot_count,
            "logical_experts": cache.logical_num_experts,
            "demands": demands,
            "misses": misses,
            "hit_rate": (demands - misses) / demands if demands else 1.0,
            "copied_bytes": cache.copied_bytes,
            "committed_passes": cache.committed_passes,
            "native_copy_calls": cache.native_copy_calls,
            "batch_copy_calls": cache.batch_copy_calls,
            "prefetch_calls": cache.prefetch_calls,
            "prefetch_experts": cache.prefetch_experts,
            "prefetch_hits": cache.prefetch_hits,
            "host_bytes": cache.host_bytes,
            "device_bytes": cache.device_bytes,
            "visit_demands": tuple(self._visit_demands),
            "visit_misses": tuple(self._visit_misses),
        }

    # -- compute ----------------------------------------------------------

    def _default_block_size_m(
        self, num_tokens: int, top_k: int, max_query_len: int = 0
    ) -> int:
        table = _BLOCK_M_BY_TOKENS_SPEC if FUSE_GLUE else _BLOCK_M_BY_TOKENS
        rows = _BLOCK_M_WIDE
        for limit, r in table[self.quant_format]:
            if num_tokens <= limit:
                rows = r
                break
        if FUSE_GLUE:
            if _SPEC_BM4_MIN > 0 and num_tokens >= _SPEC_BM4_MIN:
                return max(rows, 4)
            # Correlated multi-token spec verify (q_len >= 6, e.g. K=7 -> T>=8):
            # on NVFP4, bm=4 cuts expert blocks from ~45 to ~31 (344 -> 250 us/visit
            # on V620), while MXFP8 stays optimal at bm=2 (698 vs 722 us).
            if self.quant_format == "nvfp4" and max_query_len >= 6 and num_tokens >= 8:
                return max(rows, 4)
        return rows

    def _prefill_supported(self) -> bool:
        """Eligibility of the prefill kernels: K of both stages (hidden, this rank's
        intermediate size) must be a multiple of 32 (both NVFP4 and MXFP8 kernels
        handle K tails in 32-K steps, covering TP = 1, 2 and 4)."""
        return self.hidden_dim % 32 == 0 and self.intermediate_dim % 32 == 0

    def _gemm(
        self,
        stage,
        a,
        c,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_size_m,
        mul_topk_weight,
        output_topk,
        cta_info=None,
    ) -> None:
        from vllm import _custom_ops as ops

        qweight = getattr(self, f"{stage}_qweight")
        scales = getattr(self, f"{stage}_scales")
        if cta_info is not None:
            # prefill kernels: the work list replaces expert_ids and the padded count
            if self.quant_format == "nvfp4" and _PREFILL_INT8:
                a8, sa = ops.moe_prefill_act_quant_rdna2(a)
                ops.moe_nvfp4_bm_i8_rdna2(
                    a8,
                    sa,
                    c,
                    qweight,
                    scales,
                    getattr(self, f"{stage}_globals"),
                    topk_weights,
                    sorted_token_ids,
                    cta_info,
                    top_k,
                    mul_topk_weight,
                    output_topk,
                )
            elif self.quant_format == "nvfp4":
                ops.moe_nvfp4_bm_rdna2(
                    a,
                    c,
                    qweight,
                    scales,
                    getattr(self, f"{stage}_globals"),
                    topk_weights,
                    sorted_token_ids,
                    cta_info,
                    top_k,
                    mul_topk_weight,
                    output_topk,
                )
            else:
                ops.moe_mxfp8_bm_rdna2(
                    a,
                    c,
                    qweight,
                    scales,
                    topk_weights,
                    sorted_token_ids,
                    cta_info,
                    top_k,
                    mul_topk_weight,
                    output_topk,
                )
            return
        if self.quant_format == "nvfp4":
            ops.moe_nvfp4_gemm_rdna2(
                a,
                c,
                qweight,
                scales,
                getattr(self, f"{stage}_globals"),
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k,
                block_size_m,
                mul_topk_weight,
                output_topk,
            )
        else:
            ops.moe_mxfp8_gemm_rdna2(
                a,
                c,
                qweight,
                scales,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k,
                block_size_m,
                mul_topk_weight,
                output_topk,
            )

    def _gemm_swiglu(
        self,
        gate_up,
        c,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_size_m,
        mul_topk_weight,
        output_topk,
    ) -> None:
        """The w2 GEMM on ``swiglu(gate_up)`` (``gate_up``: ``[rows, 2 * I]``, the
        ``[gate | up]`` output of the w13 GEMM): the softcapped SwiGLU is applied in
        the kernel's activation load, the ``[rows, I]`` activation matrix is never
        written."""
        from vllm import _custom_ops as ops

        if self.quant_format == "nvfp4":
            ops.moe_nvfp4_gemm_swiglu_rdna2(
                gate_up,
                c,
                self.w2_qweight,
                self.w2_scales,
                self.w2_globals,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k,
                block_size_m,
                mul_topk_weight,
                output_topk,
                self.gate_softcap,
                self.up_softcap,
            )
        else:
            ops.moe_mxfp8_gemm_swiglu_rdna2(
                gate_up,
                c,
                self.w2_qweight,
                self.w2_scales,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k,
                block_size_m,
                mul_topk_weight,
                output_topk,
                self.gate_softcap,
                self.up_softcap,
            )

    def forward(  # type: ignore[override]
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """``[T, H]`` fp16 -> ``[T, H]`` fp16."""
        if not self._processed:
            raise RuntimeError(
                f"{self.prefix}: call process_weights_after_loading() before forward"
            )
        _, weights, ids = self.gate(hidden_states)
        return super().forward(hidden_states, ids, weights)

    # -- HIP serving path -------------------------------------------------

    @torch.no_grad()
    def prepare_hip(self) -> None:
        """Router weights of ``inkling2_glue_moe_route``; frees the fp32 parameter.

        The V_DOT2 router takes fp16 weights, which is exact when the checkpoint's bf16
        values convert to fp16 without loss; otherwise the fp32 weights stay (FMA
        path), so the router logits never deviate from the checkpoint's values.
        """
        if not self._processed:
            raise RuntimeError(
                f"{self.prefix}: call process_weights_after_loading() before "
                "prepare_hip()"
            )
        weight = self.gate.weight.detach()
        w16 = weight.to(torch.float16)
        self.router_fp16 = bool(torch.equal(w16.float(), weight))
        self.router_w = (w16 if self.router_fp16 else weight).contiguous()
        del self.gate.weight

    def forward_hip(
        self,
        x16: torch.Tensor,
        ws: Any,
        out_all: torch.Tensor,
        max_query_len: int = 0,
        visit: int = 0,
        next_mlp: Inkling2LoopMoE | None = None,
    ) -> torch.Tensor:
        """Router, expert GEMMs and swiglu on the HIP kernels (no allocation).

        ``x16``: fp16 ``[T, H]`` (the MoE input), ``out_all``: fp16 ``[T, H]``
        buffer that receives the output.  Chunks of at most ``ws.moe_chunk`` tokens
        share the scratch of ``ws`` (``Inkling2GlueWorkspace``): the router launch
        zero-fills ``gate_up`` (not on the prefill path, whose w13 kernel stores every
        routed row) and the chunk's output, the GEMMs accumulate into them.
        """
        if self.expert_cache is not None:
            return _forward_hip_offload(
                self, x16, ws, out_all, max_query_len, visit, next_mlp
            )

        from vllm import _custom_ops as ops

        T, K = x16.shape[0], self.top_k
        step = min(self.max_tokens_per_call, ws.moe_chunk)
        scale = self.gate.routed_scaling_factor
        bias = self.gate.e_score_correction_bias
        for s in range(0, T, step):
            e = min(s + step, T)
            tc = e - s
            det = _DETERMINISTIC
            prefill = (
                _PREFILL_BM
                and (tc >= _PREFILL_BM_MIN_TOKENS or det)
                and self._prefill_supported()
            )
            if det and not prefill:
                raise RuntimeError(
                    f"{self.prefix}: VLLM_RDNA2_MOE_DETERMINISTIC=1 needs the "
                    f"prefill-BM expert GEMMs (hidden and intermediate multiples of "
                    f"32, VLLM_INKLING2_PREFILL_BM on), got hidden "
                    f"{self.hidden_dim}, intermediate {self.intermediate_dim}"
                )
            # the prefill kernels group the router's 8-row blocks
            bm = (
                _BLOCK_M_WIDE
                if prefill
                else self._default_block_size_m(tc, K, max_query_len=max_query_len)
            )
            m = ws.moe_views(tc, bm)
            x = x16 if tc == T else x16[s:e]
            out = out_all if tc == T else out_all[s:e]
            ops.inkling2_glue_moe_route(
                x,
                self.router_w,
                bias,
                m.logits,
                m.topk_w,
                m.topk_ids,
                m.sorted_ids,
                m.expert_ids,
                ws.post_pad,
                ws.route_counters,
                bm,
                scale,
                # the prefill w13 kernel writes every routed row (plain stores)
                zero1=None if prefill else m.gu,
                zero2=out,
            )
            cta = None
            if prefill:
                _announce(
                    "Inkling-2 MoE prefill kernels ACTIVE: %s on a chunk of at least "
                    "%d tokens",
                    "W4A8 int8 NVFP4 + fp16 MXFP8"
                    if _PREFILL_INT8
                    else "fp16 dot2 32-row",
                    _PREFILL_BM_MIN_TOKENS,
                )
                cta = ops.moe_prefill_cta_rdna2(
                    m.expert_ids, ws.post_pad, self.num_experts, bm
                )
            self._gemm(
                "w13",
                x,
                m.gu,
                ws.no_weights,
                m.sorted_ids,
                m.expert_ids,
                ws.post_pad,
                K,
                bm,
                False,
                0,
                cta_info=cta,
            )
            # the prefill (BM) kernels run both GEMMs themselves: the fused w2 is the
            # decode-shaped kernel only
            if (
                FUSE_GLUE and not prefill and tc <= _FUSE_SWIGLU_MAX_TOKENS
            ):  # the SwiGLU runs in the w2 GEMM's activation load
                self._gemm_swiglu(
                    m.gu,
                    out,
                    m.topk_w_flat,
                    m.sorted_ids,
                    m.expert_ids,
                    ws.post_pad,
                    1,
                    bm,
                    True,
                    K,
                )
                continue
            ops.inkling2_glue_swiglu(m.gu, m.act, self.gate_softcap, self.up_softcap)
            if det:
                # Deterministic expert path (VLLM_RDNA2_MOE_DETERMINISTIC=1).
                # The decode kernels sum the K slices and the top-k rows of a
                # token with packed-fp16 CAS atomics, so their result rounds in
                # arrival order and changes between runs (the 48-layer model
                # amplifies that).  The prefill-BM w13 stores ONE fp32-chained
                # accumulator per (routed row, column) with plain stores, and
                # the BM w2 does the same when output_topk == 0; the weighted
                # top-k sum then runs in fp32 in ascending k, element-wise.
                _announce(
                    "Inkling-2 MoE deterministic expert path ACTIVE "
                    "(VLLM_RDNA2_MOE_DETERMINISTIC=1): prefill-BM kernels with "
                    "plain stores, fp32 top-k sum"
                )
                rows = _det_routed_rows(ws, tc, K, self.hidden_dim, out.device)
                self._gemm(
                    "w2",
                    m.act,
                    rows,
                    ws.no_weights,
                    m.sorted_ids,
                    m.expert_ids,
                    ws.post_pad,
                    1,
                    bm,
                    False,
                    0,
                    cta_info=cta,
                )
                _sum_topk_rows_f32(rows, m.topk_w, out)
                continue
            self._gemm(
                "w2",
                m.act,
                out,
                m.topk_w_flat,
                m.sorted_ids,
                m.expert_ids,
                ws.post_pad,
                1,
                bm,
                True,
                K,
                cta_info=cta,
            )
        return out_all


@eager_break_during_capture
def _forward_hip_offload(
    mlp: Inkling2LoopMoE,
    x16: torch.Tensor,
    ws: Any,
    out_all: torch.Tensor,
    max_query_len: int,
    visit: int,
    next_mlp: Inkling2LoopMoE | None = None,
) -> torch.Tensor:
    """Offloaded MoE execution outside stream capture (breakable HIP graphs).

    Runs the router, syncs active logical expert IDs to a reusable pinned host
    buffer, plans bounded passes (handling ``unique_experts > slot_count``
    overflow transparently), uploads misses via ``PackedExpertCache.ensure()``,
    remaps ``expert_ids`` / ``cta_info`` to GPU cache slots, and invokes the
    unchanged NVFP4/MXFP8 decode and prefill GEMM kernels into ``out_all``.

    When ``mlp.offload_prefetch`` is enabled and ``next_mlp`` has an offloaded
    ``expert_cache``, speculatively routes ``next_mlp`` on the current layer's
    MoE input ``x`` (piggybacked onto the same ``route_event`` sync with zero
    extra host waits) and queues its predicted misses onto ``copy_stream``
    immediately after launching this layer's GEMMs so H2D DMA overlaps this
    layer's MoE compute and the next layer's attention block.
    """
    from vllm import _custom_ops as ops

    cache = mlp.expert_cache
    assert cache is not None
    T, K = x16.shape[0], mlp.top_k
    step = min(mlp.max_tokens_per_call, ws.moe_chunk)
    scale = mlp.gate.routed_scaling_factor
    bias = mlp.gate.e_score_correction_bias
    visit_bucket = 0 if visit <= 0 else 1
    do_prefetch = (
        mlp.offload_prefetch
        and next_mlp is not None
        and next_mlp.expert_cache is not None
        and hasattr(next_mlp, "router_w")
    )

    for s in range(0, T, step):
        e = min(s + step, T)
        tc = e - s
        prefill = (
            _PREFILL_BM and tc >= _PREFILL_BM_MIN_TOKENS and mlp._prefill_supported()
        )
        bm = (
            _BLOCK_M_WIDE
            if prefill
            else mlp._default_block_size_m(tc, K, max_query_len=max_query_len)
        )
        m = ws.moe_views(tc, bm)
        x = x16 if tc == T else x16[s:e]
        out = out_all if tc == T else out_all[s:e]

        host_pref_ids: torch.Tensor | None = None
        if do_prefetch and e == T and not prefill and tc <= _FUSE_SWIGLU_MAX_TOKENS:
            assert next_mlp is not None
            ops.inkling2_glue_moe_route(
                x,
                next_mlp.router_w,
                next_mlp.gate.e_score_correction_bias,
                m.logits,
                m.topk_w,
                m.topk_ids,
                m.sorted_ids,
                m.expert_ids,
                ws.post_pad,
                ws.route_counters,
                bm,
                next_mlp.gate.routed_scaling_factor,
                zero1=None,
                zero2=None,
            )
            host_pref_ids = ws.offload_prefetch_scratch(tc)
            host_pref_ids.copy_(m.topk_ids, non_blocking=True)

        ops.inkling2_glue_moe_route(
            x,
            mlp.router_w,
            bias,
            m.logits,
            m.topk_w,
            m.topk_ids,
            m.sorted_ids,
            m.expert_ids,
            ws.post_pad,
            ws.route_counters,
            bm,
            scale,
            zero1=None if prefill else m.gu,
            zero2=out,
        )
        host_ids, route_event = ws.offload_route_scratch(tc)
        host_ids.copy_(m.topk_ids, non_blocking=True)
        if route_event is not None:
            route_event.record()
            route_event.synchronize()
        local_ids = host_ids.reshape(-1).tolist()
        cache.policy.observe(local_ids)
        passes = cache.policy.plan_passes(local_ids)

        cta = None
        if prefill:
            _announce(
                "Inkling-2 MoE prefill kernels ACTIVE: %s on a chunk of at least "
                "%d tokens",
                "W4A8 int8 NVFP4 + fp16 MXFP8" if _PREFILL_INT8 else "fp16 dot2 32-row",
                _PREFILL_BM_MIN_TOKENS,
            )
            cta = ops.moe_prefill_cta_rdna2(
                m.expert_ids, ws.post_pad, mlp.num_experts, bm
            )

        demands_before = cache.expert_demands
        misses_before = cache.expert_misses
        for pass_experts in passes:
            lease = cache.ensure(pass_experts)
            try:
                if cta is not None:
                    slot_col = cache.remap(cta[:, 0], lease)
                    cta_pass = cta.clone()
                    cta_pass[:, 0] = slot_col.clamp(min=0)
                    cta_pass[:, 2] = torch.where(slot_col >= 0, cta[:, 2], 0)
                    mlp._gemm(
                        "w13",
                        x,
                        m.gu,
                        ws.no_weights,
                        m.sorted_ids,
                        m.expert_ids,
                        ws.post_pad,
                        K,
                        bm,
                        False,
                        0,
                        cta_info=cta_pass,
                    )
                    ops.inkling2_glue_swiglu(
                        m.gu, m.act, mlp.gate_softcap, mlp.up_softcap
                    )
                    mlp._gemm(
                        "w2",
                        m.act,
                        out,
                        m.topk_w_flat,
                        m.sorted_ids,
                        m.expert_ids,
                        ws.post_pad,
                        1,
                        bm,
                        True,
                        K,
                        cta_info=cta_pass,
                    )
                else:
                    slot_expert_ids = cache.remap(m.expert_ids, lease)
                    mlp._gemm(
                        "w13",
                        x,
                        m.gu,
                        ws.no_weights,
                        m.sorted_ids,
                        slot_expert_ids,
                        ws.post_pad,
                        K,
                        bm,
                        False,
                        0,
                    )
                    if FUSE_GLUE and tc <= _FUSE_SWIGLU_MAX_TOKENS:
                        mlp._gemm_swiglu(
                            m.gu,
                            out,
                            m.topk_w_flat,
                            m.sorted_ids,
                            slot_expert_ids,
                            ws.post_pad,
                            1,
                            bm,
                            True,
                            K,
                        )
                    else:
                        ops.inkling2_glue_swiglu(
                            m.gu, m.act, mlp.gate_softcap, mlp.up_softcap
                        )
                        mlp._gemm(
                            "w2",
                            m.act,
                            out,
                            m.topk_w_flat,
                            m.sorted_ids,
                            slot_expert_ids,
                            ws.post_pad,
                            1,
                            bm,
                            True,
                            K,
                        )
            finally:
                cache.release(lease)
        mlp._visit_demands[visit_bucket] += cache.expert_demands - demands_before
        mlp._visit_misses[visit_bucket] += cache.expert_misses - misses_before
        if host_pref_ids is not None:
            assert next_mlp is not None and next_mlp.expert_cache is not None
            k_pre = mlp.offload_prefetch_topk
            pref_slice = host_pref_ids[:, :k_pre] if 0 < k_pre < K else host_pref_ids
            # Rank-major order prioritizes top-1/top-2 predicted experts if capped.
            pref_ids = pref_slice.t().reshape(-1).tolist()
            next_mlp.expert_cache.prefetch(
                pref_ids,
                copy_stream=ws.offload_copy_stream(),
                max_experts=mlp.offload_prefetch_max,
            )
    return out_all
