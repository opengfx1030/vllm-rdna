# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent buffers of the HIP serving path of Inkling-2.

Every kernel of ``csrc/rocm/inkling2_*_rdna2.cu`` runs on caller-provided memory
(no allocation, no host sync, shapes and pointers fixed), so the serving forward is
HIP-graph capturable.  This module owns that memory: one :class:`Inkling2GlueWorkspace`
per model, sized once for the scheduler's ``max_num_batched_tokens``, with per-step
contiguous views from :meth:`Inkling2GlueWorkspace.bind`.  Layers run one after the
other on one stream, so one set of per-layer buffers (the attention input projections,
the MoE scratch, the residual stream) serves all 24 layers and all 48 layer visits.

Layout and lifetimes follow ``docs/rdna2/inkling2-glue-kernels.md`` section 4.2 / 5.1.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import torch

from vllm import _custom_ops as ops

from ..configs import Inkling2LoopTextConfig
from .loopcd import LOOPCD_MAX_WEAK_ROWS

#: tokens per MoE router / expert-GEMM call (bounds the ``[M * top_k, 2I]`` scratch and
#: the ``T * K <= 8192`` limit of ``inkling2_glue_moe_route``)
MAX_MOE_TOKENS = 1024

#: ``VLLM_INKLING2_FUSE_GLUE`` (default on, ``0`` restores the shipped kernels and
#: tables): the decode-latency variants of the glue kernels
#: (``docs/rdna2/inkling2-decode-glue.md``): the SwiGLU fused into the second expert
#: GEMM (bit-identical to the two-launch path), and (same variable, read once by the
#: kernel host code) the R-split ``gate_mix`` for a few tokens with fp32 gate
#: weights (only the fp32 summation order of the rank sum changes).
FUSE_GLUE = os.environ.get("VLLM_INKLING2_FUSE_GLUE", "1") not in ("", "0")

#: ``VLLM_INKLING2_FUSE_SCONV_ATTN_RES`` (default on, ``0`` restores standalone
#: ``inkling2_sconv_static`` launches): fuses the post-attention and post-MLP
#: 4-tap static depthwise short convolutions into the prologue of the consuming
#: ``inkling2_glue_attn_res_sconv`` launch at TP=1 for decode / verify batches.
FUSE_SCONV_ATTN_RES = os.environ.get("VLLM_INKLING2_FUSE_SCONV_ATTN_RES", "1") not in (
    "",
    "0",
)

#: ``inkling2_attention`` keeps split-K partial rows only for the requests / tokens
#: whose split factor is > 1.  With ``splits_decode > 1`` the batch has at most
#: ``127 / H_kv`` requests (``ceil(256 / (R * H_kv)) >= 2``), which gives at most
#: ``383 / H_kv`` rows; with ``splits_tile > 1`` the tile grid has fewer than 144 CTAs
#: (16 tokens per CTA), which gives at most ``4592 / H_kv`` token rows.  Derived from
#: ``ops.inkling2_attention_splits`` (the sweep in
#: ``tests/models/inkling2_loop/test_serving_path_cpu.py``); the slack covers the
#: rounding.
_ATTN_WS_ROWS_NUM = 4975
_ATTN_WS_ROWS_SLACK = 16
_ATTN_PARTIAL_COLS = 128 + 2


def attention_workspace_numel(num_q_heads: int, num_kv_heads: int) -> int:
    """fp32 elements of a split-K workspace that serves every call whatever the batch
    (any number of requests / tokens, any context length)."""
    rows = -(-_ATTN_WS_ROWS_NUM // num_kv_heads) + _ATTN_WS_ROWS_SLACK
    return rows * num_q_heads * _ATTN_PARTIAL_COLS


def check_kernel_shapes(cfg: Inkling2LoopTextConfig) -> None:
    """The native kernels are specialised for the shapes of Inkling-2; refuse anything
    else up front (there is no torch fallback in the serving path)."""
    problems: list[str] = []
    for name, value in (("head_dim", cfg.head_dim), ("swa_head_dim", cfg.swa_head_dim)):
        if value != 128:
            problems.append(f"{name}={value} (the attention kernels need 128)")
    if cfg.d_rel != 16:
        problems.append(f"d_rel={cfg.d_rel} (the attention kernels need 16)")
    if cfg.sconv_kernel_size != 4:
        problems.append(
            f"sconv_kernel_size={cfg.sconv_kernel_size} (the kernels need 4)"
        )
    if not cfg.is_sliding_layer(0):
        problems.append(
            "layer 0 is a full-attention layer (the embedding feeds one input norm)"
        )
    for name, nh, nkv in (
        ("swa", cfg.swa_num_attention_heads, cfg.swa_num_key_value_heads),
        ("full", cfg.num_attention_heads, cfg.num_key_value_heads),
    ):
        if nh % nkv or not 1 <= nh // nkv <= 4 or nh % 2:
            problems.append(f"{name} heads {nh}/{nkv} (GQA group 1..4, even q heads)")
    if cfg.hidden_size % 8 or cfg.hidden_size > 4096:
        problems.append(f"hidden_size={cfg.hidden_size} (multiple of 8, at most 4096)")
    if cfg.loop_gate_rank % 2:
        problems.append(f"loop_gate_rank={cfg.loop_gate_rank} (even)")
    if cfg.n_routed_experts > 255 or cfg.num_experts_per_tok > 16:
        problems.append(
            f"{cfg.n_routed_experts} experts / top-{cfg.num_experts_per_tok} "
            "(the router kernel takes at most 255 / 16)"
        )
    if cfg.num_experts_per_tok * MAX_MOE_TOKENS > 8192:
        problems.append(
            f"top-{cfg.num_experts_per_tok} x {MAX_MOE_TOKENS} tokens exceeds the "
            "8192 routed rows of one router call"
        )
    if problems:
        raise ValueError(
            "Inkling-2 HIP serving path does not support this config: "
            + "; ".join(problems)
            + ". VLLM_INKLING2_PATH=reference runs the torch reference instead."
        )


class Inkling2GlueWorkspace:
    """Static buffers of the HIP path for up to ``max_tokens`` tokens per step.

    ``tp_size > 1`` sizes the per-rank buffers for the rank's share: the intermediate
    dimension of the experts, the attention workspace of the local heads, and an fp32
    ``o_out`` that receives the ``o_proj`` partial the all-reduce sums.
    """

    def __init__(
        self,
        cfg: Inkling2LoopTextConfig,
        max_tokens: int,
        device: torch.device | str,
        tp_size: int = 1,
        hi2: bool = False,
        tap_dtype: torch.dtype = torch.float32,
    ) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if tap_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"tap_dtype must be float32 or bfloat16, got {tap_dtype}")
        self.cfg = cfg
        self.device = torch.device(device)
        self.tap_dtype = tap_dtype
        T, H, R = max_tokens, cfg.hidden_size, cfg.loop_gate_rank
        E, K = cfg.n_routed_experts, cfg.num_experts_per_tok
        if cfg.intermediate_size % tp_size:
            raise ValueError(
                f"intermediate_size {cfg.intermediate_size} is not divisible by "
                f"tensor parallel size {tp_size}"
            )
        inter = cfg.intermediate_size // tp_size  # this rank's share of every expert
        # block states: one per layer whose index is a multiple of the attn-res block
        S = -(-cfg.num_hidden_layers // cfg.n_layers_per_attn_res_block)
        C = -(-H // 128)
        self.T, self.H, self.R, self.S = T, H, R, S
        self.E, self.K, self.I = E, K, inter
        self._moe_views: dict[tuple[int, int], SimpleNamespace] = {}
        self.has_loop = cfg.loop_num_passes > 1
        chunk = min(T, MAX_MOE_TOKENS)
        self.moe_chunk = chunk
        f32 = {"dtype": torch.float32, "device": self.device}
        f16 = {"dtype": torch.float16, "device": self.device}
        i32 = {"dtype": torch.int32, "device": self.device}

        self.blk = torch.zeros(T, S, H, **f32)  # persistent block states
        sizes = {  # flat fp32 buffers, bound per call by ``bind``
            "hid": T * H,  # the embedding, then the partial block at a layer's entry
            "res": T * H,
            "attn_c": T * H,
            "mlp_c": T * H,
        }
        sizes16 = {"x16": 3 * T * H, "moe_out": T * H}
        if self.has_loop:
            self.prev_blk = torch.zeros(T, max(S - 1, 1), H, **f32)
            sizes.update(
                pre_x0=T * H,
                pre_x1=T * H,
                prev_x0=T * H,
                prev_x1=T * H,
                msgs=T * 4 * H,
                raw=T * 3 * H,
                g_scratch=3 * T * H,
                g_part=3 * T * C,
            )
            sizes16["d"] = 3 * T * R
            self.gate_counters = torch.zeros(3 * T, **i32)  # zero ONCE
            self.loopcd_omega: torch.Tensor | None = torch.zeros(T, **f32)
            self.loopcd_omega_blk: torch.Tensor | None = torch.zeros(T, **f32)
            self.loopcd_logits_flag: torch.Tensor | None = torch.zeros(1, **i32)
            self.loopcd_weak_h: torch.Tensor | None = torch.zeros(
                LOOPCD_MAX_WEAK_ROWS, H, **f16
            )
        else:
            self.prev_blk = None
            self.gate_counters = None
            self.loopcd_omega = None
            self.loopcd_omega_blk = None
            self.loopcd_logits_flag = None
            self.loopcd_weak_h = None
        self.loopcd_hidden_active: bool = False
        self.loopcd_logits_active: bool = False
        self.loopcd_weak_pass: int = -1
        self.loopcd_logits_indices: torch.Tensor | None = None
        self.loopcd_weak_is_compact: bool = False
        self.f32 = {k: torch.empty(n, **f32) for k, n in sizes.items()}
        self.f16 = {k: torch.empty(n, **f16) for k, n in sizes16.items()}

        # MoE scratch of one chunk (consumed by the chunk's GEMMs before the next one)
        self.f32["logits"] = torch.empty(chunk * E, **f32)
        self.f32["topk_w"] = torch.empty(chunk * K, **f32)
        self.f16["gu"] = torch.empty(chunk * K * 2 * inter, **f16)
        # ``g_scratch`` ([3, T, H] fp32) is temporary scratch inside ``msg_gate_mix``
        # and is never live during ``mlp.forward_hip``; alias ``act`` into it when it
        # fits (always for Inkling-2-Mini: 8 * 1152 * 2 <= 3 * 2304 * 4 per token),
        # saving 18 MiB at chunk=1024 while keeping ``act`` valid for tc > 16.
        act_numel = chunk * K * inter
        if self.has_loop and self.f32["g_scratch"].numel() * 2 >= act_numel:
            self.f16["act"] = self.f32["g_scratch"].view(torch.float16)[:act_numel]
            self._act_aliased = True
        else:
            self.f16["act"] = torch.empty(act_numel, **f16)
            self._act_aliased = False
        n_pad, n_blk, _ = ops.inkling2_glue_moe_route_sizes(chunk, E, K, 8)
        self.sorted_ids = torch.empty(n_pad, **i32)
        self.expert_ids = torch.empty(n_blk, **i32)
        self.topk_ids = torch.empty(chunk, K, **i32)
        self.post_pad = torch.empty(1, **i32)
        self.route_counters = torch.zeros(chunk + 1, **i32)  # zero ONCE
        self.no_weights = torch.empty(0, **f32)

        # HI2 experts (``rdna_hint_flex_moe_*``): a block-64 routing layout, the
        # per-pair transformed activations and a w2 output buffer of its own (at TP>1
        # the rank's intermediate size is not half the hidden size, so the gate/up
        # scratch cannot hold the down output).  The block-64 buffer is rounded up to
        # the kernel's TORCH_CHECK: ``sorted_ids.numel() % 64 == 0``.
        self.hi2 = hi2
        self.sorted_ids64: torch.Tensor | None = None
        self.expert_ids64: torch.Tensor | None = None
        self.post_pad64: torch.Tensor | None = None
        if hi2:
            self.f16["moe_x"] = torch.empty(chunk * K * H, **f16)
            self.f16["moe_w2"] = torch.empty(chunk * K * H, **f16)
            n_pad64, n_blk64, _ = ops.inkling2_glue_moe_route_sizes(chunk, E, K, 64)
            self.sorted_ids64 = torch.empty(-(-n_pad64 // 64) * 64, **i32)
            self.expert_ids64 = torch.empty(n_blk64, **i32)
            self.post_pad64 = torch.empty(1, **i32)

        # attention: the fused input projections write into per-segment buffers shared
        # by every layer; the output / o_proj buffers and the split-K workspace too
        self._segments: dict[tuple[str, int], torch.Tensor] = {}
        self._attn_out: dict[int, torch.Tensor] = {}
        heads = {  # this rank's heads (attention is sharded by KV-head group)
            (
                cfg.swa_num_attention_heads // tp_size,
                cfg.swa_num_key_value_heads // tp_size,
            ),
            (cfg.num_attention_heads // tp_size, cfg.num_key_value_heads // tp_size),
        }
        numel = max(attention_workspace_numel(nh, nkv) for nh, nkv in heads)
        self.attn_ws = torch.empty(numel, **f32)
        self.o_out = torch.empty(T, H, **(f32 if tp_size > 1 else f16))

        # DSpark taps (``ensure_taps``): fp32 copies of the partial block at the
        # entry of the selected layer visits, plus the final fp32 normalised output
        self.taps: list[torch.Tensor] = []
        self.final32: torch.Tensor | None = None

    def tensors(self) -> list[torch.Tensor]:
        """Every persistent device buffer of the workspace (each once)."""
        f16_vals = (
            (t for k, t in self.f16.items() if k != "act")
            if self._act_aliased
            else self.f16.values()
        )
        held: list[torch.Tensor | None] = [
            self.blk,
            self.prev_blk,
            self.gate_counters,
            self.loopcd_omega,
            self.loopcd_omega_blk,
            self.loopcd_logits_flag,
            self.loopcd_weak_h,
            self.sorted_ids,
            self.expert_ids,
            self.topk_ids,
            self.post_pad,
            self.route_counters,
            self.sorted_ids64,
            self.expert_ids64,
            self.post_pad64,
            self.attn_ws,
            self.o_out,
            self.final32,
            *self.f32.values(),
            *f16_vals,
            *self._segments.values(),
            *self._attn_out.values(),
            *self.taps,
        ]
        return [t for t in held if t is not None]

    def nbytes(self) -> int:
        """Device bytes held by the workspace."""
        return sum(t.numel() * t.element_size() for t in self.tensors())

    # -- attention buffers --------------------------------------------------------
    def segment(self, name: str, n: int) -> torch.Tensor:
        """fp16 ``[max_tokens, n]`` output buffer of one fused-projection segment."""
        key = (name, n)
        buf = self._segments.get(key)
        if buf is None:
            buf = torch.empty(self.T, n, dtype=torch.float16, device=self.device)
            self._segments[key] = buf
        return buf

    def attn_out(self, width: int) -> torch.Tensor:
        """fp16 ``[max_tokens, width]`` attention output (input of ``o_proj``).

        Zero-initialised: the attention kernel writes the rows of the tokens that
        belong to a request only, so the padded rows of a CUDA graph keep what the
        buffer held (zeros, or finite attention outputs of an earlier step), never
        uninitialised memory."""
        buf = self._attn_out.get(width)
        if buf is None:
            buf = torch.zeros(self.T, width, dtype=torch.float16, device=self.device)
            self._attn_out[width] = buf
        return buf

    # -- DSpark taps ----------------------------------------------------------------
    def ensure_taps(self, count: int) -> None:
        """Allocate ``count`` partial-block taps and the final-output tap (once)."""
        if len(self.taps) < count:
            self.taps += [
                torch.empty(self.T, self.H, dtype=self.tap_dtype, device=self.device)
                for _ in range(count - len(self.taps))
            ]
        if count and self.final32 is None:
            self.final32 = torch.empty(
                self.T, self.H, dtype=self.tap_dtype, device=self.device
            )

    def final_tap(self, tokens: int) -> torch.Tensor:
        """``[tokens, H]`` view (``tap_dtype``: bf16 under DSpark, fp32 otherwise) of
        the final-output tap."""
        if self.final32 is None:
            raise RuntimeError("the DSpark taps were never allocated (ensure_taps)")
        return self.final32[:tokens]

    # -- MoE chunk views --------------------------------------------------------------
    def moe_views(self, tokens: int, block_size_m: int) -> SimpleNamespace:
        """Views of the per-chunk MoE scratch for ``tokens`` routed tokens (memoised:
        the same few shapes recur every step, and a view costs a Python call)."""
        key = (tokens, block_size_m)
        views = self._moe_views.get(key)
        if views is None:
            E, K, inter = self.E, self.K, self.I
            n_pad, n_blk, _ = ops.inkling2_glue_moe_route_sizes(
                tokens, E, K, block_size_m
            )
            tw = self.f32["topk_w"][: tokens * K]
            views = SimpleNamespace(
                sorted_ids=self.sorted_ids[:n_pad],
                expert_ids=self.expert_ids[:n_blk],
                gu=self.f16["gu"][: tokens * K * 2 * inter].view(tokens * K, 2 * inter),
                act=self.f16["act"][: tokens * K * inter].view(tokens * K, inter),
                topk_w=tw.view(tokens, K),
                topk_w_flat=tw,
                topk_ids=self.topk_ids[:tokens],
                logits=self.f32["logits"][: tokens * E].view(tokens, E),
            )
            self._moe_views[key] = views
        return views

    def moe64_views(
        self, tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(sorted_ids, expert_ids, ntpp)`` of the block-64 HI2 GEMM layout."""
        if not self.hi2:
            raise RuntimeError("moe64_views needs a workspace built with hi2=True")
        assert self.sorted_ids64 is not None
        assert self.expert_ids64 is not None
        assert self.post_pad64 is not None
        n_pad, n_blk, _ = ops.inkling2_glue_moe_route_sizes(tokens, self.E, self.K, 64)
        # the HI2 GEMM TORCH_CHECK requires ``sorted_ids.numel() % 64 == 0``; the
        # extra rows are pad ids the GEMM never reads (it stops at ``ntpp``)
        n_pad64 = -(-n_pad // 64) * 64
        return (
            self.sorted_ids64[:n_pad64],
            self.expert_ids64[:n_blk],
            self.post_pad64,
        )

    def moe_hidden_view(self, tokens: int) -> torch.Tensor:
        """``[tokens * top_k, H]`` w2 output scratch (fp16, preallocated)."""
        return self.f16["moe_w2"][: tokens * self.K * self.H].view(
            tokens * self.K, self.H
        )

    def moe_x_view(self, tokens: int, width: int) -> torch.Tensor:
        """``[tokens * top_k, width]`` row-major view of the per-pair scratch."""
        return self.f16["moe_x"][: tokens * self.K * width].view(tokens * self.K, width)

    # -- per-step views ---------------------------------------------------------------
    def bind(self, T: int) -> SimpleNamespace:
        """Contiguous per-call views (the ops need contiguous ``d`` / ``out`` / ...)."""
        if T > self.T:
            raise ValueError(
                f"{T} tokens exceed the {self.T} the workspace was sized for "
                "(scheduler max_num_batched_tokens)"
            )
        H, R = self.H, self.R
        f, h = self.f32, self.f16

        def v(buf: torch.Tensor, *shape: int) -> torch.Tensor:
            return buf[: math.prod(shape)].view(*shape)

        x16 = v(h["x16"], 3, T, H)
        ns = SimpleNamespace(
            T=T,
            hid=v(f["hid"], T, H),
            res=v(f["res"], T, H),
            attn_c=v(f["attn_c"], T, H),
            mlp_c=v(f["mlp_c"], T, H),
            moe_out=v(h["moe_out"], T, H),
            o_out=self.o_out[:T],
            blk=self.blk[:T],
            # the three fp16 gate / norm outputs and the layouts the ops take them in
            hq=x16[0],
            hk=x16[1],
            hv=x16[2],
            x16=x16,  # [3, T, H] (msg_gate_mix out)
            x16_g1=x16[:1],  # [1, T, H]
            norm3=x16.permute(1, 0, 2),  # [T, 3, H] (attn_res out, three layernorms)
            norm1=x16[:1].permute(1, 0, 2),  # [T, 1, H]
        )
        if self.has_loop:
            C = -(-H // 128)
            raw = v(f["raw"], T, 3, H)
            msgs = v(f["msgs"], T, 4, H)
            d = v(h["d"], 3, T, R)
            ns.pre_x0 = v(f["pre_x0"], T, H)
            ns.pre_x1 = v(f["pre_x1"], T, H)
            ns.prev_x0 = v(f["prev_x0"], T, H)
            ns.prev_x1 = v(f["prev_x1"], T, H)
            ns.msgs = msgs
            ns.msgs_qkv = msgs[:, :3]
            ns.msgs_mlp = msgs[:, 3:4]
            ns.raw = raw
            ns.raw3 = raw[:, :3]
            ns.raw1 = raw[:, :1]
            ns.g_scratch = v(f["g_scratch"], 3, T, H)
            ns.g_part = v(f["g_part"], 3, T, C)
            ns.d = d
            ns.d1 = d[:1]
            ns.prev_blk = self.prev_blk[:T]
        return ns
