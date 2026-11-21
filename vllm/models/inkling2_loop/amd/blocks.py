# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stateless building blocks of the Inkling-2-Mini (``inkling2_loop``) backbone.

Plain-torch ports of the small reference modules in
``transformers.models.inkling2_loop.modeling_inkling2_loop``.  Parameter and
buffer names, shapes and dtypes are identical to the reference classes so that
``ours.load_state_dict(ref.state_dict())`` works and the checkpoint loader can
reuse the checkpoint -> reference key mapping.  The math (including every fp32
upcast) mirrors the reference; these modules hold no per-sequence state and
treat all leading dimensions of their inputs as opaque batch dimensions.

Config objects are only read by attribute (``hidden_size``, ``rms_norm_eps``,
``loop_gate_rank``, ...), so either our ``Inkling2LoopTextConfig`` or the
reference one can be passed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(
            hidden_states.square().mean(-1, keepdim=True) + self.eps
        )
        return hidden_states.to(input_dtype)


class Inkling2RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MessageGate(nn.Module):
    """Mixes hidden states and message as
    ``(1 - forget_gate) * hidden_states + message_gate * message``."""

    def __init__(
        self,
        config,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.norm = UnweightedRMSNorm(eps=config.rms_norm_eps)
        # ``down_proj`` follows the model dtype; the gate projections are fp32.
        self.down_proj = nn.Linear(
            2 * config.hidden_size,
            config.loop_gate_rank,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.forget_gate_proj = nn.Linear(
            config.loop_gate_rank,
            config.hidden_size,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        self.message_gate_proj = nn.Linear(
            config.loop_gate_rank,
            config.hidden_size,
            bias=False,
            dtype=torch.float32,
            device=device,
        )

    def forward(
        self, hidden_states: torch.Tensor, message: torch.Tensor
    ) -> torch.Tensor:
        hidden_states, message = hidden_states.float(), message.float()
        gate_hidden_states = torch.cat(
            [self.norm(hidden_states), self.norm(message)], dim=-1
        )
        gate_hidden_states = self.down_proj(
            gate_hidden_states.to(self.down_proj.weight.dtype)
        ).float()

        forget_gate = torch.tanh(self.forget_gate_proj(gate_hidden_states))
        message_gate = torch.tanh(self.message_gate_proj(gate_hidden_states))
        return (1 - forget_gate) * hidden_states + message_gate * message


def _attend_over_stack(
    query: torch.Tensor, stacked_states: torch.Tensor, scaling: float
) -> torch.Tensor:
    """Non-causal single-"head" attention over the stacked-block axis.

    Equivalent to the reference ``eager_attention_forward`` for this call
    (no mask, no bias, no sinks, no dropout, one kv group): keys and values are
    both ``stacked_states``.

    query:          ``(Q, H)`` (shared by every position) or ``(..., Q, H)``
    stacked_states: ``(..., n, H)``
    returns:        ``(..., Q, H)``
    """
    attn_weights = torch.matmul(query, stacked_states.transpose(-2, -1)) * scaling
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    return torch.matmul(attn_weights, stacked_states)


class BlockAttentionResidual(nn.Module):
    """Retrieves ``num_queries`` messages from the stacked block states with
    learned (position independent) queries."""

    def __init__(
        self, config, num_queries: int = 1, *, device: torch.device | str | None = None
    ):
        super().__init__()
        shape = (
            (config.hidden_size,)
            if num_queries == 1
            else (num_queries, config.hidden_size)
        )
        self.query = nn.Parameter(
            torch.zeros(*shape, dtype=torch.float32, device=device)
        )
        self.norm = UnweightedRMSNorm()
        self.num_key_value_groups = 1
        self.is_causal = False

    def forward(
        self, block_states: torch.Tensor, partial_block_states: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """``block_states``: ``(..., n, H)`` (``n`` may be 0), ``partial_block_states``:
        ``(..., H)``.  Returns ``num_queries`` fp32 tensors of shape ``(..., H)``."""
        stacked_states = torch.cat(
            [block_states, partial_block_states.unsqueeze(-2)], dim=-2
        ).float()
        stacked_states = self.norm(stacked_states)

        hidden_size = stacked_states.shape[-1]
        query_states = self.query.float().view(-1, hidden_size)
        attn_output = _attend_over_stack(
            query_states, stacked_states, hidden_size**-0.5
        )
        return attn_output.unbind(-2)


class MessageAttentionResidual(nn.Module):
    """Retrieves messages from the block states of the previous block using
    learned queries."""

    def __init__(self, config, *, device: torch.device | str | None = None):
        super().__init__()
        # one query per message (query, key, value and mlp) for each loop layer
        shape = (
            config.loop_layer_index_end - config.loop_layer_index_start,
            4,
            config.hidden_size,
        )
        self.query = nn.Parameter(
            torch.zeros(*shape, dtype=torch.float32, device=device)
        )
        self.norm = UnweightedRMSNorm()
        self.num_key_value_groups = 1
        self.is_causal = False

    def forward(
        self,
        block_states: torch.Tensor,
        partial_block_states: torch.Tensor,
        loop_layer_idx: int,
    ) -> torch.Tensor:
        """``loop_layer_idx`` is relative to ``loop_layer_index_start``.
        Returns the fp32 ``(..., 4, H)`` messages (query, key, value, mlp)."""
        stacked_states = torch.cat(
            [block_states, partial_block_states.unsqueeze(-2)], dim=-2
        ).float()
        stacked_states = self.norm(stacked_states)

        hidden_size = stacked_states.shape[-1]
        query_states = self.query[loop_layer_idx].float()
        return _attend_over_stack(query_states, stacked_states, hidden_size**-0.5)


class Softcap(nn.Module):
    def __init__(self, softcap: float):
        super().__init__()
        self.softcap = softcap

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(hidden_states / self.softcap) * self.softcap


def softcapped_swiglu(
    gate: torch.Tensor, up: torch.Tensor, gate_cap: float, up_cap: float
) -> torch.Tensor:
    """``softcap(gate) * sigmoid(gate) * softcap(up)`` in fp32 (the sigmoid sees
    the raw gate).  Returns **fp32**; callers cast to the activation dtype."""
    gate, up = gate.float(), up.float()
    return (
        (torch.tanh(gate / gate_cap) * gate_cap)
        * torch.sigmoid(gate)
        * (torch.tanh(up / up_cap) * up_cap)
    )


class SoftcappedSwiGLU(nn.Module):
    """SwiGLU where the sigmoid sees the raw gate and the softcaps only bound
    the linear gate factor and ``up``; result in ``gate.dtype``."""

    def __init__(self, config):
        super().__init__()
        self.gate_softcap = config.mlp_gate_softcap
        self.up_softcap = config.mlp_up_softcap

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return softcapped_swiglu(gate, up, self.gate_softcap, self.up_softcap).to(
            gate.dtype
        )


def l2_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Scale each ``x[..., :]`` vector down to L2 norm ``threshold`` when longer;
    shorter vectors are unchanged.  fp32 math, result in ``x.dtype``."""
    out_dtype = x.dtype
    return _l2_clip_fp32(x.float(), threshold).to(out_dtype)


def _l2_clip_fp32(x: torch.Tensor, threshold: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (threshold / (norm + 1e-6)).clamp(max=1.0)


def scaled_l2_clip(
    x: torch.Tensor, scale: torch.Tensor, threshold: float
) -> torch.Tensor:
    """``l2_clip`` applied after the learned ``scale`` (``[heads, dim]``, broadcast
    against ``x`` ``[T, heads, dim]``)."""
    out_dtype = x.dtype
    return _l2_clip_fp32(x.float() * scale.float(), threshold).to(out_dtype)


def log_scale(
    x: torch.Tensor, positions: torch.Tensor, alpha: float, n_floor: int
) -> torch.Tensor:
    """Scale ``x`` ``[T, heads, dim]`` by
    ``1 + alpha * log(max((position + 1) / n_floor, 1))`` per token
    (``positions`` ``[T]`` int64)."""
    out_dtype = x.dtype
    tau = 1.0 + alpha * torch.log(((positions + 1).float() / n_floor).clamp(min=1.0))
    return (x.float() * tau[..., None, None]).to(out_dtype)


def log_scaled_l2_clip(
    x: torch.Tensor,
    scale: torch.Tensor,
    positions: torch.Tensor,
    threshold: float,
    alpha: float,
    n_floor: int,
) -> torch.Tensor:
    """``scaled_l2_clip`` (result rounded to ``x.dtype``) followed by ``log_scale``."""
    clipped = scaled_l2_clip(x, scale, threshold)
    return log_scale(clipped, positions, alpha, n_floor)


class L2Clip(nn.Module):
    """Scales each head vector down to ``threshold`` when its L2 norm is larger;
    shorter vectors are unchanged."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return l2_clip(hidden_states, self.threshold)


class ScaledL2Clip(nn.Module):
    """``L2Clip`` applied after a learned per-head, per-channel scale."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        threshold: float,
        *,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.threshold = threshold
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim, device=device))

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        return scaled_l2_clip(hidden_states, self.scale, self.threshold)


class LogScale(nn.Module):
    """Scales the per-head states by
    ``tau = 1 + log_scaling_alpha * log(max((position + 1) / log_scaling_n_floor, 1))``.

    ``hidden_states``: ``(..., num_heads, head_dim)``; ``position_ids``: ``(...)``.
    """

    def __init__(self, config):
        super().__init__()
        self.log_scaling_alpha = config.log_scaling_alpha
        self.log_scaling_n_floor = config.log_scaling_n_floor

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        return log_scale(
            hidden_states,
            position_ids,
            self.log_scaling_alpha,
            self.log_scaling_n_floor,
        )


class LogScaledL2Clip(nn.Module):
    """``ScaledL2Clip`` followed by ``LogScale``."""

    def __init__(
        self,
        config,
        num_heads: int,
        head_dim: int,
        threshold: float,
        *,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim, device=device))
        self.threshold = threshold
        self.log_scaling_alpha = config.log_scaling_alpha
        self.log_scaling_n_floor = config.log_scaling_n_floor

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        return log_scaled_l2_clip(
            hidden_states,
            self.scale,
            position_ids,
            self.threshold,
            self.log_scaling_alpha,
            self.log_scaling_n_floor,
        )


class NormedEmbedding(nn.Embedding):
    """Token embedding followed by an ``Inkling2RMSNorm`` (``embed_norm``)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None,
        norm_eps: float,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__(
            num_embeddings, embedding_dim, padding_idx, dtype=dtype, device=device
        )
        self.embed_norm = Inkling2RMSNorm(
            embedding_dim, eps=norm_eps, dtype=dtype, device=device
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(super().forward(input_ids))


def relative_distance(
    q_positions: torch.Tensor, kv_positions: torch.Tensor
) -> torch.Tensor:
    """Backward distance ``q_positions[:, None] - kv_positions[None, :]`` of shape
    ``(q_len, kv_len)``.

    This is the pure core of the reference ``get_relative_distance``; the
    reference derives the position ranges (``arange(q_len) + q_offset`` and
    ``arange(kv_len) + kv_offset``) from a transformers ``Cache``, which has no
    equivalent here, so callers pass the absolute positions directly.
    """
    return q_positions[:, None] - kv_positions[None, :]


def route_topk(
    logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int,
    route_scale: float,
    scoring: str = "sigmoid",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k expert routing from fp32 router logits ``[T, E]``.

    Mirrors the math of ``Inkling2LoopTopKRouter.forward``: the correction
    bias only influences the selection, the weights are the unbiased scores of
    the selected experts, normalised to sum to one and then multiplied by
    ``route_scale``.  Returns ``(weights [T, k] fp32, ids [T, k] int32)``; the
    order of the ``k`` entries is unspecified (as in the reference).
    """
    if scoring != "sigmoid":
        raise ValueError(f"unsupported scoring function {scoring!r}")
    if logits.dtype != torch.float32:
        raise TypeError(f"router logits must be fp32, got {logits.dtype}")
    scores = torch.sigmoid(logits)
    ids = torch.topk(
        scores + e_score_correction_bias, top_k, dim=-1, sorted=False
    ).indices
    weights = scores.gather(1, ids)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * route_scale, ids.to(torch.int32)
