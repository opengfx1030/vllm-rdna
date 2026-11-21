# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone attention stack of thinkingmachines/Inkling-2-Mini (``inkling2_loop``).

Plain-torch port of the reference ``modeling_inkling2_loop`` attention classes
(``SlidingAttention``, ``FullAttention`` and their projections / short
convolutions) with *dense per-sequence state* instead of HF ``Cache`` objects.
Parameter names, shapes and dtypes are identical to the reference modules, so
``ours.load_state_dict(ref.state_dict())`` works and the existing
checkpoint -> reference key mapping can be reused.

Inputs are one sequence at a time: ``hidden_states`` is ``[S, hidden]`` (or
``[1, S, hidden]``) and ``positions`` is ``[S]``.  ``positions`` only drive the
log scaling of the full layers; the relative-position bias and the masks use
token *indices* (``state.num_tokens + arange(S)``), exactly like the reference,
which derives them from the cache length and never from ``position_ids``.

Differences from the reference are limited to state handling: the keys/values
(post short-conv, post-clip), the short-conv input tails and the token count
live in :class:`DenseAttnState`, updated in place by ``forward``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from vllm.models.inkling2_loop.configs import Inkling2LoopTextConfig


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


@dataclass
class DenseConvState:
    """Last ``kernel_size - 1`` conv inputs of one causal short convolution.

    ``tail`` is ``[1, channels, kernel_size - 1]`` fp32 (the convs run in fp32);
    all zeros means "no previous tokens", which equals the reference's zero
    left padding.
    """

    tail: torch.Tensor

    @classmethod
    def empty(
        cls,
        channels: int,
        kernel_size: int,
        device: torch.device | str | None = None,
    ) -> DenseConvState:
        return cls(
            torch.zeros(
                1, channels, kernel_size - 1, dtype=torch.float32, device=device
            )
        )


@dataclass
class DenseAttnState:
    """Per-sequence state of one attention layer.

    ``k`` / ``v``: ``[1, num_kv_heads, T, head_dim]`` post short-conv / post
    norm tensors of the retained tokens (all tokens for full layers, the last
    ``sliding_window - 1`` for sliding layers).  ``k_conv`` / ``v_conv``: conv
    tails of the key / value short convolutions.  ``num_tokens``: tokens seen so
    far (the next token has index ``num_tokens``).
    """

    k: torch.Tensor
    v: torch.Tensor
    k_conv: DenseConvState
    v_conv: DenseConvState
    num_tokens: int = 0

    @classmethod
    def empty(
        cls,
        num_kv_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> DenseAttnState:
        kv = torch.zeros(1, num_kv_heads, 0, head_dim, dtype=dtype, device=device)
        channels = num_kv_heads * head_dim
        return cls(
            k=kv,
            v=kv.clone(),
            k_conv=DenseConvState.empty(channels, conv_kernel_size, device),
            v_conv=DenseConvState.empty(channels, conv_kernel_size, device),
            num_tokens=0,
        )


def _prepend_tail(
    x: torch.Tensor, state: DenseConvState | None, kernel_size: int
) -> torch.Tensor:
    """``x`` is ``[B, C, S]`` fp32.  Returns ``[B, C, S + K - 1]`` (state tail, or
    zeros without a state, in front) and stores the new tail in ``state``."""
    if state is None:
        return F.pad(x, (kernel_size - 1, 0))
    context = torch.cat([state.tail, x], dim=-1)
    state.tail.copy_(context[..., context.shape[-1] - (kernel_size - 1) :])
    return context


# --------------------------------------------------------------------------
# Norm / clip / scale helpers
# --------------------------------------------------------------------------


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


class L2Clip(nn.Module):
    """Scales each head vector down to ``threshold`` when its L2 norm is larger."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        norm = torch.linalg.vector_norm(hidden_states, dim=-1, keepdim=True)
        return (hidden_states * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(
            input_dtype
        )


class ScaledL2Clip(nn.Module):
    """``L2Clip`` applied after a learned scale."""

    def __init__(self, num_heads: int, head_dim: int, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float() * self.scale.float()
        norm = torch.linalg.vector_norm(hidden_states, dim=-1, keepdim=True)
        return (hidden_states * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(
            input_dtype
        )


class LogScale(nn.Module):
    """``tau = 1 + alpha * log(max((position + 1) / n_floor, 1))`` per token."""

    def __init__(self, config: Inkling2LoopTextConfig):
        super().__init__()
        self.log_scaling_alpha = config.log_scaling_alpha
        self.log_scaling_n_floor = config.log_scaling_n_floor

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        scale = 1.0 + self.log_scaling_alpha * torch.log(
            ((position_ids + 1).float() / self.log_scaling_n_floor).clamp(min=1.0)
        )
        return (hidden_states.float() * scale[..., None, None]).to(input_dtype)


class LogScaledL2Clip(nn.Module):
    """``ScaledL2Clip`` followed by ``LogScale``."""

    def __init__(
        self,
        config: Inkling2LoopTextConfig,
        num_heads: int,
        head_dim: int,
        threshold: float,
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim))
        self.l2_clip = L2Clip(threshold)
        self.log_scale = LogScale(config)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = self.l2_clip(hidden_states.float() * self.scale.float()).to(
            input_dtype
        )
        return self.log_scale(hidden_states, position_ids)


# --------------------------------------------------------------------------
# Relative position bias
# --------------------------------------------------------------------------


class RelativePositionBias(nn.Module):
    """Hidden-states conditioned relative position bias (zero outside
    ``0 <= distance < rel_extent``)."""

    def __init__(
        self,
        config: Inkling2LoopTextConfig,
        num_heads: int,
        rel_extent: int,
        log_scaling: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.rel_extent = rel_extent
        self.linear = nn.Linear(
            config.hidden_size, num_heads * config.d_rel, bias=False
        )
        self.norm = (
            LogScaledL2Clip(
                config, num_heads, config.d_rel, config.rel_l2_clip_threshold
            )
            if log_scaling
            else ScaledL2Clip(num_heads, config.d_rel, config.rel_l2_clip_threshold)
        )
        self.proj = nn.Parameter(torch.zeros(config.d_rel, rel_extent))

    def forward(
        self,
        hidden_states: torch.Tensor,
        relative_distance: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        relative_states = self.linear(hidden_states).view(
            *hidden_states.shape[:-1], self.num_heads, -1
        )
        relative_states = self.norm(relative_states, position_ids)

        # relative_states: [batch, q_len, num_heads, d_rel] -> bias:
        # [batch, num_heads, q_len, kv_len]
        rel_logits = (relative_states @ self.proj).transpose(1, 2)
        gather_index = relative_distance.clamp(0, self.rel_extent - 1).expand(
            *rel_logits.shape[:2], -1, -1
        )
        position_bias = rel_logits.gather(-1, gather_index)
        return position_bias.masked_fill(
            (relative_distance < 0) | (relative_distance >= self.rel_extent), 0.0
        )


# --------------------------------------------------------------------------
# Short convolutions
# --------------------------------------------------------------------------


class ShortConvolution(nn.Module):
    """Depthwise causal short conv + residual, ``(conv(x) + x) / sqrt(K + 1)``, in fp32.

    Also used standalone for ``post_attention_convolution`` /
    ``post_mlp_convolution``.  ``hidden_states`` is ``[S, C]`` or ``[B, S, C]``
    (a ``state`` requires batch 1); the output has the input's shape and dtype.
    """

    def __init__(self, hidden_size: int, conv_kernel_size: int):
        super().__init__()
        self.conv_kernel_size = conv_kernel_size
        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=conv_kernel_size,
            groups=hidden_size,
            padding=conv_kernel_size - 1,
            bias=False,
        ).to(dtype=torch.float32)

    def new_state(self, device: torch.device | str | None = None) -> DenseConvState:
        return DenseConvState.empty(
            self.conv1d.in_channels, self.conv_kernel_size, device
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        dynamic_input: torch.Tensor | None = None,
        state: DenseConvState | None = None,
    ) -> torch.Tensor:
        """``dynamic_input`` is ignored (interface parity with the dynamic conv)."""
        squeeze = hidden_states.dim() == 2
        if squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        residual = hidden_states

        context = _prepend_tail(
            hidden_states.transpose(1, 2), state, self.conv_kernel_size
        )
        weight = self.conv1d.weight
        out = F.conv1d(
            context.to(weight.dtype),
            weight=weight,
            bias=self.conv1d.bias,
            padding=0,
            groups=weight.shape[0],
        )
        out = out.transpose(1, 2)
        out = (out + residual) / math.sqrt(self.conv_kernel_size + 1)
        out = out.to(dtype=input_dtype)
        return out.squeeze(0) if squeeze else out


class DynamicShortConvolution(nn.Module):
    """Short causal convolution whose taps are static per-channel weights plus
    per-token weights projected from ``dynamic_input``, shared within each
    key/value head.  Tap ``j`` is the backward distance (``static_weight[0]``
    weights the current token)."""

    def __init__(
        self, hidden_size: int, input_size: int, num_heads: int, conv_kernel_size: int
    ):
        super().__init__()
        self.conv_kernel_size = conv_kernel_size
        self.num_heads = num_heads
        self.static_weight = nn.Parameter(torch.zeros(conv_kernel_size, hidden_size))
        self.weight_proj = nn.Linear(
            input_size, num_heads * conv_kernel_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        dynamic_input: torch.Tensor,
        state: DenseConvState | None = None,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        batch_size, seq_len, hidden_size = hidden_states.shape
        dynamic_weights = self.weight_proj(dynamic_input).float()
        dynamic_weights = dynamic_weights.view(
            batch_size, seq_len, self.num_heads, 1, self.conv_kernel_size
        )
        weights = (
            self.static_weight.float().T.view(self.num_heads, -1, self.conv_kernel_size)
            + dynamic_weights
        )

        hidden_states = hidden_states.float()
        residual = hidden_states
        context = _prepend_tail(
            hidden_states.transpose(1, 2), state, self.conv_kernel_size
        )
        # windows[..., lag] is the token `lag` steps back
        windows = context.unfold(-1, self.conv_kernel_size, 1).flip(-1).transpose(1, 2)
        windows = windows.reshape(
            batch_size, seq_len, self.num_heads, -1, self.conv_kernel_size
        )
        hidden_states = (
            (windows * weights).sum(-1).reshape(batch_size, seq_len, hidden_size)
        )
        hidden_states = (hidden_states + residual) / math.sqrt(
            2 * self.conv_kernel_size + 1
        )
        return hidden_states.to(input_dtype)


# --------------------------------------------------------------------------
# Projections
# --------------------------------------------------------------------------


class QueryProjection(nn.Module):
    """Query states ``[B, num_heads, S, head_dim]``."""

    def __init__(
        self,
        config: Inkling2LoopTextConfig,
        num_heads: int,
        head_dim: int,
        log_scaling: bool,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.linear = nn.Linear(config.hidden_size, num_heads * head_dim, bias=False)
        self.norm = (
            LogScaledL2Clip(config, num_heads, head_dim, config.qk_l2_clip_threshold)
            if log_scaling
            else ScaledL2Clip(num_heads, head_dim, config.qk_l2_clip_threshold)
        )

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        states = self.linear(hidden_states)
        states = states.view(*states.shape[:-1], -1, self.head_dim)
        return self.norm(states, position_ids).transpose(1, 2)


class KeyValueProjection(nn.Module):
    """Key or value states ``[B, num_kv_heads, S, head_dim]``."""

    def __init__(
        self,
        config: Inkling2LoopTextConfig,
        num_heads: int,
        head_dim: int,
        is_key: bool,
        dynamic: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.linear = nn.Linear(config.hidden_size, num_heads * head_dim, bias=False)
        self.sconv = (
            DynamicShortConvolution(
                num_heads * head_dim,
                config.hidden_size,
                num_heads,
                config.sconv_kernel_size,
            )
            if dynamic
            else ShortConvolution(num_heads * head_dim, config.sconv_kernel_size)
        )
        self.norm = (
            L2Clip(config.qk_l2_clip_threshold)
            if is_key
            else UnweightedRMSNorm(eps=config.rms_norm_eps)
        )

    def forward(
        self, hidden_states: torch.Tensor, conv_state: DenseConvState | None = None
    ) -> torch.Tensor:
        states = self.linear(hidden_states)
        states = self.sconv(states, hidden_states, state=conv_state)
        states = states.view(*states.shape[:-1], -1, self.head_dim)
        return self.norm(states).transpose(1, 2)


class GatedOutputProjection(nn.Module):
    """Output projection applied after gating the attention output with a
    sigmoid of the layer input."""

    def __init__(self, config: Inkling2LoopTextConfig, num_heads: int, head_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, num_heads * head_dim, bias=False)
        self.linear = nn.Linear(num_heads * head_dim, config.hidden_size, bias=False)

    def forward(
        self, attn_output: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(hidden_states).float())
        attn_output = (attn_output.float() * gate).to(attn_output.dtype)
        return self.linear(attn_output)


# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``[B, kv_heads, S, D]`` -> ``[B, kv_heads * n_rep, S, D]``
    (repeat_interleave on dim 1)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class _DenseAttention(nn.Module):
    """Shared body of the sliding and full attention layers."""

    is_sliding: bool

    def __init__(
        self, config: Inkling2LoopTextConfig, layer_idx: int, is_sliding: bool
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = is_sliding
        self.head_dim = config.swa_head_dim if is_sliding else config.head_dim
        self.num_heads = (
            config.swa_num_attention_heads if is_sliding else config.num_attention_heads
        )
        self.num_key_value_heads = (
            config.swa_num_key_value_heads if is_sliding else config.num_key_value_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.sliding_window = config.sliding_window_size if is_sliding else None
        self.rel_extent = (
            config.sliding_window_size if is_sliding else config.rel_extent
        )
        self.conv_kernel_size = config.sconv_kernel_size
        self.scaling = self.head_dim**-0.5
        full = not is_sliding
        self.q_proj = QueryProjection(
            config, self.num_heads, self.head_dim, log_scaling=full
        )
        self.k_proj = KeyValueProjection(
            config, self.num_key_value_heads, self.head_dim, is_key=True, dynamic=full
        )
        self.v_proj = KeyValueProjection(
            config, self.num_key_value_heads, self.head_dim, is_key=False, dynamic=full
        )
        self.o_proj = GatedOutputProjection(config, self.num_heads, self.head_dim)
        self.sink_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.relative_position_bias = RelativePositionBias(
            config, self.num_heads, self.rel_extent, log_scaling=full
        )

    def new_state(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> DenseAttnState:
        return DenseAttnState.empty(
            self.num_key_value_heads,
            self.head_dim,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        state: DenseAttnState | None = None,
        key_hidden_states: torch.Tensor | None = None,
        value_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``hidden_states`` ``[S, hidden]`` or ``[1, S, hidden]``; ``positions``
        ``[S]`` (or ``[1, S]``).  ``key_hidden_states`` / ``value_hidden_states``
        default to ``hidden_states``.  ``state`` is updated in place (``None``:
        stateless prefill of a whole sequence).  Returns the input's shape."""
        squeeze = hidden_states.dim() == 2
        if squeeze:
            hidden_states = hidden_states.unsqueeze(0)
            if key_hidden_states is not None:
                key_hidden_states = key_hidden_states.unsqueeze(0)
            if value_hidden_states is not None:
                value_hidden_states = value_hidden_states.unsqueeze(0)
        position_ids = positions.reshape(1, -1)
        key_hidden_states = (
            hidden_states if key_hidden_states is None else key_hidden_states
        )
        value_hidden_states = (
            hidden_states if value_hidden_states is None else value_hidden_states
        )
        batch_size, q_len = hidden_states.shape[:2]
        if batch_size != 1:
            raise ValueError("dense attention state supports batch size 1")
        if state is None:
            state = self.new_state(hidden_states.device)

        query_states = self.q_proj(hidden_states, position_ids)
        key_states = self.k_proj(key_hidden_states, state.k_conv)
        value_states = self.v_proj(value_hidden_states, state.v_conv)

        prev_len = state.k.shape[2]
        key_states = torch.cat([state.k.to(key_states.dtype), key_states], dim=2)
        value_states = torch.cat([state.v.to(value_states.dtype), value_states], dim=2)
        kv_len = key_states.shape[2]
        # Token indices of the queries / retained keys (state.num_tokens counts the
        # tokens before this call; the retained keys are the last `prev_len` of them).
        q_index = torch.arange(q_len, device=hidden_states.device) + state.num_tokens
        kv_index = torch.arange(kv_len, device=hidden_states.device) + (
            state.num_tokens - prev_len
        )
        relative_distance = q_index[:, None] - kv_index[None, :]

        position_bias = self.relative_position_bias(
            hidden_states, relative_distance, position_ids
        )
        sinks = self.sink_proj(hidden_states).transpose(1, 2).unsqueeze(-1)

        attn_weights = (
            torch.matmul(
                query_states,
                repeat_kv(key_states, self.num_key_value_groups).transpose(2, 3),
            )
            * self.scaling
        )
        attn_weights = attn_weights + position_bias
        invalid = relative_distance < 0
        if self.sliding_window is not None:
            invalid = invalid | (relative_distance >= self.sliding_window)
        attn_weights = attn_weights.masked_fill(invalid, float("-inf"))
        # The sink is one extra logit per query with a zero value.
        combined_logits = torch.cat([attn_weights, sinks], dim=-1).float()
        attn_weights = F.softmax(combined_logits, dim=-1)[..., :-1].to(
            query_states.dtype
        )
        attn_output = torch.matmul(
            attn_weights, repeat_kv(value_states, self.num_key_value_groups)
        )
        attn_output = (
            attn_output.transpose(1, 2).contiguous().reshape(batch_size, q_len, -1)
        )
        output = self.o_proj(attn_output, hidden_states)

        if self.sliding_window is not None:
            keep = min(kv_len, self.sliding_window - 1)
            key_states = key_states[:, :, kv_len - keep :].contiguous()
            value_states = value_states[:, :, kv_len - keep :].contiguous()
        state.k = key_states
        state.v = value_states
        state.num_tokens += q_len
        return output.squeeze(0) if squeeze else output


class SlidingAttention(_DenseAttention):
    """Sliding-window layer: static short conv on k/v, window =
    ``sliding_window_size``."""

    def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int = 0):
        super().__init__(config, layer_idx, is_sliding=True)


class FullAttention(_DenseAttention):
    """Full-attention layer: dynamic short conv on k/v, log-scaled q / relative
    states, relative bias over ``rel_extent``."""

    def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int = 0):
        super().__init__(config, layer_idx, is_sliding=False)
