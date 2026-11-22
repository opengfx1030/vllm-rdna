# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling-2 attention (TP=1, eager torch core).

Mirrors ``Inkling2LoopSlidingAttention`` / ``Inkling2LoopFullAttention`` of the
transformers reference: no RoPE, hidden-conditioned additive relative bias,
attention sink, sigmoid-gated output, depthwise k/v short convs (static on
sliding layers, dynamic on full layers), L2-clipped q/k and RMS-normed v.

The score matmul is plain torch over the paged cache, one request at a time,
in query chunks: correctness first, the Triton/HIP kernel replaces ``_attend``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_piecewise_capture
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available

from ..configs import Inkling2LoopTextConfig
from . import blocks
from .state import (
    STREAM_K,
    STREAM_V,
    Inkling2ConvOwner,
    Inkling2KVOwner,
    Inkling2Metadata,
    conv_taps,
    kv_read,
    kv_write,
)

if TYPE_CHECKING:
    from .glue import Inkling2GlueWorkspace

_QUERY_CHUNK = 256


class _Linear(nn.Module):
    """Bias-free linear holding ``weight`` (checkpoint layout ``[out, in]``)."""

    def __init__(self, in_features: int, out_features: int, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


class StaticSconv(nn.Module):
    """Depthwise causal conv; ``conv1d.weight`` is ``[C, 1, k]`` (fp32).

    Tap ``k-1`` multiplies the current token (cross-correlation, as ``F.conv1d``).
    Output is ``(conv(x) + x) / sqrt(k + 1)``.
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = nn.Module()
        self.conv1d.weight = nn.Parameter(
            torch.empty(channels, 1, kernel_size, dtype=torch.float32),
            requires_grad=False,
        )

    def forward_taps(self, taps: torch.Tensor) -> torch.Tensor:
        # taps: [k, T, C] with taps[lag] = input lag tokens back
        k = self.kernel_size
        w = self.conv1d.weight.squeeze(1)  # [C, k]
        out = torch.zeros_like(taps[0])
        for lag in range(k):
            out = out + taps[lag] * w[:, k - 1 - lag]
        return (out + taps[0]) / math.sqrt(k + 1)


class DynamicSconv(nn.Module):
    """k/v conv of the full layers: static taps + taps projected from the input.

    ``static_weight`` is ``[k, C]`` with tap 0 on the current token;
    ``weight_proj`` maps the layer input to ``heads * k`` per-token taps shared
    by the ``C / heads`` channels of a head.  Output ``(conv + x) / sqrt(2k+1)``.
    """

    def __init__(
        self, channels: int, input_size: int, num_heads: int, kernel_size: int
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.static_weight = nn.Parameter(
            torch.zeros(kernel_size, channels, dtype=torch.float32),
            requires_grad=False,
        )
        self.weight_proj = _Linear(input_size, num_heads * kernel_size)

    def forward_taps(self, taps: torch.Tensor, dyn_input: torch.Tensor) -> torch.Tensor:
        k, T, C = taps.shape
        nh = self.num_heads
        dyn = self.weight_proj(dyn_input).float().view(T, nh, 1, k)
        w = self.static_weight.float().T.reshape(nh, C // nh, k) + dyn  # [T,nh,C/nh,k]
        out = torch.zeros(T, nh, C // nh, device=taps.device, dtype=torch.float32)
        for lag in range(k):
            out = out + taps[lag].view(T, nh, C // nh) * w[..., lag]
        return (out.reshape(T, C) + taps[0]) / math.sqrt(2 * k + 1)


class _QueryProj(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.linear = _Linear(hidden, heads * head_dim)
        self.norm = nn.Module()
        self.norm.scale = nn.Parameter(
            torch.ones(heads, head_dim, dtype=torch.float32), requires_grad=False
        )


class _KVProj(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int, sconv: nn.Module):
        super().__init__()
        self.linear = _Linear(hidden, heads * head_dim)
        self.sconv = sconv


class _OutProj(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.gate_proj = _Linear(hidden, heads * head_dim)
        self.linear = _Linear(heads * head_dim, hidden)


class _RelBias(nn.Module):
    def __init__(self, hidden: int, heads: int, d_rel: int, rel_extent: int) -> None:
        super().__init__()
        self.linear = _Linear(hidden, heads * d_rel)
        self.norm = nn.Module()
        self.norm.scale = nn.Parameter(
            torch.ones(heads, d_rel, dtype=torch.float32), requires_grad=False
        )
        self.proj = nn.Parameter(torch.empty(d_rel, rel_extent), requires_grad=False)


class Inkling2Attention(nn.Module):
    def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = config.is_sliding_layer(layer_idx)
        self.num_heads = (
            config.swa_num_attention_heads
            if self.is_sliding
            else (config.num_attention_heads)
        )
        self.num_kv_heads = (
            config.swa_num_key_value_heads
            if self.is_sliding
            else (config.num_key_value_heads)
        )
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.groups = self.num_heads // self.num_kv_heads
        self.window = config.sliding_window_size if self.is_sliding else None
        self.rel_extent = (
            config.sliding_window_size if self.is_sliding else config.rel_extent
        )
        self.scaling = self.head_dim**-0.5
        H = config.hidden_size
        nh, nkv, D = self.num_heads, self.num_kv_heads, self.head_dim
        self.q_proj = _QueryProj(H, nh, D)

        def make_sconv() -> nn.Module:
            if self.is_sliding:
                return StaticSconv(nkv * D, config.sconv_kernel_size)
            return DynamicSconv(nkv * D, H, nkv, config.sconv_kernel_size)

        self.k_proj = _KVProj(H, nkv, D, make_sconv())
        self.v_proj = _KVProj(H, nkv, D, make_sconv())
        self.o_proj = _OutProj(H, nh, D)
        self.sink_proj = _Linear(H, nh)
        self.relative_position_bias = _RelBias(H, nh, config.d_rel, self.rel_extent)

    # -- pieces -----------------------------------------------------------
    def _norm_q(self, x: torch.Tensor, scale: torch.Tensor, positions) -> torch.Tensor:
        thr = self.config.qk_l2_clip_threshold
        return self._scaled_clip(x, scale, thr, positions)

    def _scaled_clip(self, x, scale, thr, positions) -> torch.Tensor:
        in_dtype = x.dtype
        y = blocks.l2_clip(x.float() * scale.float(), thr).to(in_dtype)
        if not self.is_sliding:
            y = blocks.log_scale(
                y,
                positions,
                self.config.log_scaling_alpha,
                self.config.log_scaling_n_floor,
            )
        return y

    def _kv_stream(
        self,
        proj: _KVProj,
        h: torch.Tensor,
        stream: int,
        positions: torch.Tensor,
        conv_owner: Inkling2ConvOwner,
        md: Inkling2Metadata,
    ) -> torch.Tensor:
        x = proj.linear(h).float()  # [T, nkv*D] raw conv input
        taps = conv_taps(conv_owner, stream, x, positions, md)
        if self.is_sliding:
            y = proj.sconv.forward_taps(taps)
        else:
            y = proj.sconv.forward_taps(taps, h)
        return y.view(-1, self.num_kv_heads, self.head_dim)

    def forward(
        self,
        hq: torch.Tensor,
        hk: torch.Tensor,
        hv: torch.Tensor,
        positions: torch.Tensor,
        kv_owner: Inkling2KVOwner,
        conv_owner: Inkling2ConvOwner,
        kv_md: Inkling2Metadata,
        conv_md: Inkling2Metadata,
    ) -> torch.Tensor:
        T = hq.shape[0]
        nh, D = self.num_heads, self.head_dim
        q = self.q_proj.linear(hq).view(T, nh, D)
        q = self._scaled_clip(
            q, self.q_proj.norm.scale, self.config.qk_l2_clip_threshold, positions
        )

        k = self._kv_stream(self.k_proj, hk, STREAM_K, positions, conv_owner, conv_md)
        k = blocks.l2_clip(k, self.config.qk_l2_clip_threshold)
        v = self._kv_stream(self.v_proj, hv, STREAM_V, positions, conv_owner, conv_md)
        v = v * torch.rsqrt(
            v.square().mean(-1, keepdim=True) + self.config.rms_norm_eps
        )
        kv_write(kv_owner, k, v, kv_md)

        rb = self.relative_position_bias
        rel = rb.linear(hq).view(T, nh, self.config.d_rel)
        rel = self._scaled_clip(
            rel, rb.norm.scale, self.config.rel_l2_clip_threshold, positions
        )
        rel_logits = torch.einsum("thd,dr->thr", rel.float(), rb.proj.float())
        sinks = self.sink_proj(hq).float()  # [T, nh]

        attn = self._attend(q, rel_logits, sinks, positions, kv_owner, kv_md)

        gate = torch.sigmoid(self.o_proj.gate_proj(hq).float()).view(T, nh, D)
        attn = (attn.float() * gate).to(hq.dtype).reshape(T, nh * D)
        return self.o_proj.linear(attn)

    # -- attention core ---------------------------------------------------
    def _attend(
        self,
        q: torch.Tensor,
        rel_logits: torch.Tensor,
        sinks: torch.Tensor,
        positions: torch.Tensor,
        kv_owner: Inkling2KVOwner,
        md: Inkling2Metadata,
    ) -> torch.Tensor:
        T, nh, D = q.shape
        nkv, g = self.num_kv_heads, self.groups
        out = torch.empty(T, nh, D, dtype=torch.float32, device=q.device)
        qsl = md.query_start_loc.tolist()
        seq_lens = md.seq_lens.tolist()
        R = self.rel_extent
        for r in range(md.num_reqs):
            s, e = qsl[r], qsl[r + 1]
            if e == s:
                continue
            total = seq_lens[r]
            p0 = total - (e - s)
            lo = max(0, p0 - self.window + 1) if self.window else 0
            kk, vv = kv_read(kv_owner, md, r, lo, total)  # [n, nkv, D]
            kk = kk.float()
            vv = vv.float()
            pos_k = torch.arange(lo, total, device=q.device)
            for a in range(s, e, _QUERY_CHUNK):
                b = min(a + _QUERY_CHUNK, e)
                c = b - a
                qc = q[a:b].float().view(c, nkv, g, D)
                sc = torch.einsum("chgd,nhd->chgn", qc, kk) * self.scaling
                sc = sc.reshape(c, nh, -1)
                dist = positions[a:b, None] - pos_k[None, :]  # [c, n]
                idx = dist.clamp(0, R - 1)[:, None, :].expand(c, nh, -1)
                bias = torch.gather(rel_logits[a:b], 2, idx)
                bias = bias.masked_fill(((dist < 0) | (dist >= R))[:, None, :], 0.0)
                sc = sc + bias
                vis = dist >= 0
                if self.window:
                    vis = vis & (dist < self.window)
                sc = sc.masked_fill(~vis[:, None, :], float("-inf"))
                logits = torch.cat([sc, sinks[a:b, :, None]], dim=-1)
                p = torch.softmax(logits, dim=-1)[..., :-1]
                o = torch.einsum("chgn,nhd->chgd", p.view(c, nkv, g, -1), vv)
                out[a:b] = o.reshape(c, nh, D)
        return out

    # -- HIP serving path --------------------------------------------------
    @torch.no_grad()
    def prepare_hip(self, ws: Inkling2GlueWorkspace) -> None:
        """Pack the dense weights for the native kernels and free the originals.

        The pack owns the only copy of the six (eight with the dynamic-conv taps)
        projections, so the serving model holds the weights once.  Everything the
        kernels read per call is prepared here: fp32 scales, the transposed fp16
        relative-position basis, the conv weights and the shared output buffers.
        """
        from . import linear_pack as lp

        cfg = self.config
        q, rb = self.q_proj, self.relative_position_bias
        dyn: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)
        if not self.is_sliding:
            dyn = (
                self.k_proj.sconv.weight_proj.weight,
                self.v_proj.sconv.weight_proj.weight,
            )
        self._attn_in = lp.inkling2_linear_pack_attn_in(
            q.linear.weight,
            self.o_proj.gate_proj.weight,
            rb.linear.weight,
            self.sink_proj.weight,
            self.k_proj.linear.weight,
            self.v_proj.linear.weight,
            *dyn,
        )
        self._o_pack = lp.inkling2_linear_pack_o_proj(self.o_proj.linear.weight)
        self._rel_proj_t = rb.proj.detach().t().contiguous().half()
        self._q_scale = q.norm.scale.detach().float().contiguous()
        self._rel_scale = rb.norm.scale.detach().float().contiguous()
        if self.is_sliding:
            self._k_weight = self.k_proj.sconv.conv1d.weight.detach()
            self._v_weight = self.v_proj.sconv.conv1d.weight.detach()
        else:
            self._k_weight = self.k_proj.sconv.static_weight.detach()
            self._v_weight = self.v_proj.sconv.static_weight.detach()
        for module in (
            q.linear,
            self.o_proj.gate_proj,
            rb.linear,
            self.sink_proj,
            self.k_proj.linear,
            self.v_proj.linear,
            self.o_proj.linear,
        ):
            del module.weight
        if not self.is_sliding:
            del self.k_proj.sconv.weight_proj.weight
            del self.v_proj.sconv.weight_proj.weight
        del rb.proj
        self._segments = [
            ws.segment(name, n)
            for name, n in zip(self._attn_in.names, self._attn_in.seg_n)
        ]
        self._attn_out = ws.attn_out(self.num_heads * self.head_dim)
        self._attn_ws = ws.attn_ws
        self._window = self.window or 0
        self._log_scale = not self.is_sliding
        self._log_alpha = float(cfg.log_scaling_alpha)
        self._log_n_floor = float(cfg.log_scaling_n_floor)
        self._q_clip = float(cfg.qk_l2_clip_threshold)
        self._rel_clip = float(cfg.rel_l2_clip_threshold)
        self._rms_eps = float(cfg.rms_norm_eps)

    def forward_hip(
        self,
        hq: torch.Tensor,
        hk: torch.Tensor,
        hv: torch.Tensor,
        positions: torch.Tensor,
        kv_owner: Inkling2KVOwner,
        conv_owner: Inkling2ConvOwner,
        kv_md: Inkling2Metadata,
        conv_md: Inkling2Metadata,
        o_out: torch.Tensor,
    ) -> torch.Tensor:
        """Fused input projections, k / v short convs + KV write, attention, o_proj.

        ``hq`` / ``hk`` / ``hv`` are the fp16 ``[T, H]`` streams, ``o_out`` the fp16
        ``[T, H]`` buffer that receives the o_proj output (the input of the
        post-attention short conv).
        """
        from vllm import _custom_ops as ops

        T = hq.shape[0]
        outs = [seg[:T] for seg in self._segments]
        self._attn_in([hq, hk, hv], outs)
        named = dict(zip(self._attn_in.names, outs))
        k_off = conv_owner.stream_ranges[STREAM_K][0]
        v_off = conv_owner.stream_ranges[STREAM_V][0]
        for _ in range(conv_passes(conv_md)):
            ops.inkling2_kv_prep(
                named["k"],
                named["v"],
                self._k_weight,
                self._v_weight,
                named.get("dyn_k"),
                named.get("dyn_v"),
                conv_owner.kv_cache,
                conv_md.block_table,
                conv_md.slot_mapping,
                conv_md.seq_idx,
                conv_md.query_start_loc,
                conv_md.seq_lens,
                kv_owner.kv_cache,
                kv_md.slot_mapping,
                k_off,
                v_off,
                self._q_clip,
                self._rms_eps,
            )
        attn = self._attn_out[:T]
        attend_hip(self, named, positions, kv_owner, attn, kv_md)
        self._o_pack([attn], [o_out])
        return o_out


def conv_passes(md: Inkling2Metadata) -> int:
    """Launches of each op that reads conv history from the cache and writes this step's
    conv inputs into it in one kernel (``inkling2_sconv_static``, ``inkling2_kv_prep``).

    The first ``k - 1`` tokens of a request chunk take their history from the cache.
    The scheduler registers the blocks of the tokens it schedules in the prefix cache
    before they are computed, so a prefix-cache hit of the same batch can read history
    that another request only writes in this very step.  The reference path writes all
    inputs before it reads any (``conv_taps``); one launch cannot order the CTAs, a
    second launch can: the conv inputs the first one wrote do not depend on history, so
    the second finds them all in the cache and recomputes what the first derived from
    stale history.  A batch of single-token chunks reads the cache only at positions the
    previous step wrote, except when a request with a one-token prefix-cache tail
    shares a step with the owner that completes that block with its own decode token:
    that corner needs a write-phase / read-phase split inside the kernels
    (``docs/rdna2/inkling2-port-spec.md``, integration part A).

    This is the only place that decides the count: a launch count is frozen into a CUDA
    graph, so a graph has to take the count of the *widest batch it replays*.  A FULL
    graph replays batches of the query length it was captured with (``md`` is the
    capture-time metadata: 1 for decode, ``1 + k`` for verification).  A PIECEWISE graph
    replays any mixed / prefill batch, whatever the dummy batch it was captured with.
    """
    return 2 if md.max_query_len > 1 or _tracing_piecewise_graph() else 1


def _tracing_piecewise_graph() -> bool:
    return (
        is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
    )


@eager_break_during_piecewise_capture
def attend_hip(
    layer: Inkling2Attention,
    named: dict[str, torch.Tensor],
    positions: torch.Tensor,
    kv_owner: Inkling2KVOwner,
    out: torch.Tensor,
    captured_md: Inkling2Metadata,
) -> None:
    """Paged attention of one layer visit into ``out`` (the input of ``o_proj``).

    The only op of the serving forward whose launch parameters depend on the batch
    (request count, longest query and the split-K factors derived from them), so it is
    the one break point of the PIECEWISE graphs: it runs eagerly between two graph
    segments, and when a segment list is replayed ``captured_md`` is the dummy batch
    the graph was captured with, so the metadata of the step that replays it is looked
    up in the forward context.  Inside a FULL graph the op is recorded inline: the
    graph was captured for the padded request count, the uniform query length and
    ``max_seq_len = max_model_len``, which bound every batch it replays
    (``docs/rdna2/inkling2-attn-kernels.md``).  Rows of ``out`` past the last request
    (padded tokens) are not written.
    """
    from vllm import _custom_ops as ops

    kv_md = live_metadata(kv_owner.prefix, captured_md)
    key = (layer.num_kv_heads, layer._window)
    splits = kv_md.splits.get(key)
    if splits is None:
        splits = kv_md.splits[key] = ops.inkling2_attention_splits(
            kv_md.num_reqs,
            layer.num_kv_heads,
            layer._window,
            kv_md.max_seq_len,
            kv_md.max_query_len,
        )
    ops.inkling2_attention(
        named["q"],
        named["rel"],
        named["sink"],
        named["gate"],
        layer._q_scale,
        layer._rel_scale,
        layer._rel_proj_t,
        positions,
        kv_owner.kv_cache,
        kv_md.block_table,
        kv_md.seq_lens,
        kv_md.query_start_loc,
        out,
        layer._attn_ws,
        layer._window,
        layer.rel_extent,
        layer._log_scale,
        layer._log_alpha,
        layer._log_n_floor,
        layer._q_clip,
        layer._rel_clip,
        kv_md.max_query_len,
        splits[0],
        splits[1],
    )


def static_conv_hip(
    conv: StaticSconv,
    x: torch.Tensor,
    out: torch.Tensor,
    owner: Inkling2ConvOwner,
    md: Inkling2Metadata,
    stream: int,
) -> None:
    """Static depthwise short conv of one stream: fp16 ``x [T, C]`` -> fp32 ``out``."""
    from vllm import _custom_ops as ops

    offset = owner.stream_ranges[stream][0]
    for _ in range(conv_passes(md)):
        ops.inkling2_sconv_static(
            x,
            conv.conv1d.weight,
            out,
            owner.kv_cache,
            md.block_table,
            md.slot_mapping,
            md.seq_idx,
            md.query_start_loc,
            md.seq_lens,
            offset,
            owner.num_kv_heads,
        )


def get_metadata(prefix: str) -> Inkling2Metadata | None:
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return None
    md = attn_metadata[prefix]
    assert isinstance(md, Inkling2Metadata)
    return md


def live_metadata(prefix: str, captured: Inkling2Metadata) -> Inkling2Metadata:
    """The metadata of the step that is running an op: the forward context's when there
    is one, else ``captured`` (the metadata the caller already holds; outside the
    engine nothing sets a context)."""
    if is_forward_context_available():
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            md = attn_metadata[prefix]
            assert isinstance(md, Inkling2Metadata)
            return md
    return captured
