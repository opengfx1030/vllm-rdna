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

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "ExpertsMxfp8",
    "ExpertsNvfp4",
    "ExpertsReference",
    "Inkling2LoopExperts",
    "Inkling2LoopMoE",
    "Inkling2LoopSparseMoeBlock",
    "Inkling2LoopTopKRouter",
    "repack_mxfp8",
    "repack_nvfp4",
    "softcapped_swiglu",
]


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
        out = torch.zeros(
            num_tokens, self.hidden_dim, dtype=torch.float16, device=device
        )
        self._gemm(
            "w2",
            act,
            out,
            top_k_weights.to(torch.float32).reshape(-1).contiguous(),
            sorted_ids,
            expert_ids,
            ntpp,
            1,
            block_size_m,
            True,
            top_k,
        )
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
    return (
        _pack_k_words(weight_e4m3).view(torch.uint32),
        _transpose_scales(scale_ue8m0),
    )


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
# Serving module (one layer, TP=1)
# ---------------------------------------------------------------------------


class _MoEGate(Inkling2LoopTopKRouter):
    """Router whose selection bias is a (frozen) Parameter, so a loader that
    walks ``named_parameters()`` sees both ``weight`` and
    ``e_score_correction_bias`` (checkpoint key ``mlp.gate.bias``)."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        bias = self._buffers.pop("e_score_correction_bias")
        self.e_score_correction_bias = nn.Parameter(bias, requires_grad=False)


def _assert_tp1() -> None:
    from vllm.distributed import parallel_state

    if parallel_state.model_parallel_is_initialized():
        tp = parallel_state.get_tensor_model_parallel_world_size()
        assert tp == 1, f"Inkling2LoopMoE supports tensor parallel size 1, got {tp}"


class Inkling2LoopMoE(_ExpertsKernel):
    """One Inkling-2-Mini MoE layer: fp32 sigmoid router + NVFP4 / MXFP8
    experts on the gfx1030 kernels.  ``forward`` takes flat ``[T, H]`` fp16.

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
    ) -> None:
        if quant_format not in ("nvfp4", "mxfp8"):
            raise ValueError(f"unsupported quant_format {quant_format!r}")
        _assert_tp1()
        super().__init__(config)
        self.quant_format = quant_format
        self.prefix = prefix
        self.top_k = config.num_experts_per_tok
        self.gate = _MoEGate(config)

        num_e, hidden, inter = self.num_experts, self.hidden_dim, self.intermediate_dim
        fp8 = torch.float8_e4m3fn
        for stage, n, k in (("w13", 2 * inter, hidden), ("w2", hidden, inter)):
            if quant_format == "nvfp4":
                tensors = {
                    "weight": torch.empty(num_e, n, k // 2, dtype=torch.uint8),
                    "weight_scale": torch.empty(num_e, n, k // 16, dtype=fp8),
                    "weight_amax": torch.empty(num_e, dtype=torch.float32),
                    "weight_outer_scale_denominator": torch.empty(
                        1, dtype=torch.float32
                    ),
                }
            else:
                tensors = {
                    "weight": torch.empty(num_e, n, k, dtype=fp8),
                    "weight_scale": torch.empty(num_e, n, k // 32, dtype=torch.uint8),
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

    # -- compute ----------------------------------------------------------

    def _default_block_size_m(self, num_tokens: int, top_k: int) -> int:
        # the kernels support 1/2/4/8 routed rows per expert block
        if num_tokens <= 1:
            return 1
        if num_tokens <= 2:
            return 2
        if num_tokens <= 4:
            return 4
        return 8

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
        from vllm import _custom_ops as ops

        qweight = getattr(self, f"{stage}_qweight")
        scales = getattr(self, f"{stage}_scales")
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
