#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp64 oracle of the Inkling-2 DSpark draft, and the checker of the HIP draft.

The oracle is the draft of ``mtp.safetensors`` as the upstream reference
implementation defines it (vllm PR #69028 / #67325 at ``b426b46e``, files
``nvidia_dspark.py``, ``nvidia_ops_dspark.py``, ``speculator.py``), written as plain
torch on the CPU in float64.  It has no rounding of its own: where the reference
rounds to bf16 (GEMM outputs, per-head norms, the residual stream, the recurrent gate,
the logits) the oracle keeps full precision, so it bounds the error of every
implementation, the reference's included.

Semantics (names are the tensors of ``mtp.safetensors``):

* ``alpha = softmax(fuse_weight, -1)`` (``[layers + 1, taps]``).  Draft layer ``l``
  reads the context ``mix_rms(taps, alpha[l], layers.l.dflare_norm)``, i.e. the RMSNorm
  of ``sum_t alpha[l, t] * tap_t``; the recurrent head reads ``f_rnn =
  mix_rms(taps of the last committed token, alpha[-1], markov_head.dflare_norm)``.
* Context K is ``rope(k_norm_ctx(x_ctx @ k_proj_ctx.T))`` (per-head RMSNorm), context V
  is ``x_ctx @ v_proj_ctx.T`` (no norm, no rope).
* The block input is ``embed_norm(embed(anchor))`` in row 0 and ``mask_embed`` in rows
  ``1..B-1``.  Layers are pre-norm: ``residual += attn(norm_in(x))``;
  ``residual += mlp(norm_post(residual))``; the stack ends in ``norm``.
* Block attention is bidirectional over the block's own keys and the context keys; a
  local layer sees the ``sliding_window_size`` context positions before the block
  start ``P`` (positions ``[P - W, P)``), the global layer all of them.  Scale
  ``head_dim ** -0.5``, GQA ``num_heads / num_kv_heads`` query heads per KV head, neox
  RoPE over the whole head with ``theta = rope_theta``.
* The Markov head (``rnn_hidden``): per draft step ``i`` (rows ``1..B-1``)::

      ctx_i = rnn_proj.bias + rnn_proj.weight[:, 2R:] @ [h_i ; f_rnn]
      raw   = ctx_i + rnn_proj.weight[:, :2R] @ [state ; markov_w1[prev]]
      gate, cand, out = raw.chunk(3)
      state = sigmoid(gate) * state + (1 - sigmoid(gate)) * tanh(cand)
      latent = tanh(out)
      logits = cap * tanh((lm_head @ h_i + lm_head @ hidden_out(latent)) / cap)

  with ``state = 0`` and ``prev = anchor`` at step 0 (``prev`` is the token chosen at
  the previous step).  The serving kernels fold ``hidden_out`` into the LM head once;
  the oracle keeps the unfolded definition, which checks the fold.

Run ``python dspark_check.py --help`` for the checker, which compares a draft built by
``vllm.models.inkling2_loop.amd.dspark`` against this oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

LAYER_TENSORS = (
    "dflare_norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.k_proj_ctx.weight",
    "self_attn.v_proj_ctx.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_norm_ctx.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
MARKOV_TENSORS = (
    "markov_head.dflare_norm.weight",
    "markov_head.hidden_out.weight",
    "markov_head.hidden_out.bias",
    "markov_head.markov_w1.weight",
    "markov_head.rnn_proj.weight",
    "markov_head.rnn_proj.bias",
)
GLOBAL_TENSORS = ("fuse_weight", "mask_embed", "norm.weight")


@dataclass
class DraftConfig:
    hidden_size: int = 2304
    num_layers: int = 3
    num_heads: int = 24
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9216
    markov_rank: int = 256
    block_size: int = 8
    window: int = 1024
    local_layer_ids: tuple[int, ...] = (0, 2)
    rope_theta: float = 1e6
    eps: float = 1e-6
    softcap: float = 30.0
    num_taps: int = 4

    @classmethod
    def from_mtp_config(cls, mtp: dict, softcap: float = 30.0) -> DraftConfig:
        return cls(
            hidden_size=mtp["hidden_size"],
            num_layers=mtp["num_hidden_layers"],
            num_heads=mtp["num_attention_heads"],
            num_kv_heads=mtp["num_key_value_heads"],
            head_dim=mtp["head_dim"],
            intermediate_size=mtp["intermediate_size"],
            markov_rank=mtp["markov_rank"],
            block_size=mtp["block_size"],
            window=mtp["sliding_window_size"],
            local_layer_ids=tuple(mtp["local_layer_ids"]),
            rope_theta=float(mtp["rope_theta"]),
            eps=float(mtp["rms_norm_eps"]),
            softcap=softcap,
            num_taps=len(mtp["target_layer_ids"]),
        )

    @property
    def num_steps(self) -> int:
        return self.block_size - 1


@dataclass
class Draft:
    """The tensors of one ``mtp.safetensors`` in the oracle's dtype."""

    cfg: DraftConfig
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    def layer(self, i: int, name: str) -> torch.Tensor:
        return self.tensors[f"layers.{i}.{name}"]

    @property
    def alpha(self) -> torch.Tensor:
        return torch.softmax(self["fuse_weight"], dim=-1)


def expected_names(num_layers: int) -> list[str]:
    names = list(GLOBAL_TENSORS) + list(MARKOV_TENSORS)
    for i in range(num_layers):
        names += [f"layers.{i}.{n}" for n in LAYER_TENSORS]
    return sorted(names)


def load_draft(
    path: str | Path, cfg: DraftConfig, dtype: torch.dtype = torch.float64
) -> Draft:
    """Read ``mtp.safetensors`` (strict names) into ``dtype``."""
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        names = sorted(f.keys())
        expected = expected_names(cfg.num_layers)
        if names != expected:
            raise ValueError(
                f"unexpected tensors: {sorted(set(names) ^ set(expected))}"
            )
        for n in names:
            tensors[n] = f.get_tensor(n).to(dtype)
    return Draft(cfg, tensors)


def random_draft(
    cfg: DraftConfig, vocab: int, seed: int = 0, dtype: torch.dtype = torch.float64
) -> Draft:
    """Random tensors of the checkpoint's names and shapes (CPU tests).

    Magnitudes follow a trained network's: matrices ``~ 1 / sqrt(fan_in)``, norm
    gains around one, so no op saturates.
    """
    g = torch.Generator().manual_seed(seed)
    H, D, R = cfg.hidden_size, cfg.head_dim, cfg.markov_rank
    nq, nkv, inter = cfg.num_heads, cfg.num_kv_heads, cfg.intermediate_size

    def mat(rows: int, cols: int, gain: float = 1.0) -> torch.Tensor:
        return (torch.randn(rows, cols, generator=g) * gain / math.sqrt(cols)).to(dtype)

    def gain(n: int) -> torch.Tensor:
        return (1.0 + 0.2 * torch.randn(n, generator=g)).to(dtype)

    t: dict[str, torch.Tensor] = {
        "fuse_weight": torch.randn(
            cfg.num_layers + 1, cfg.num_taps, generator=g, dtype=dtype
        ),
        "mask_embed": torch.randn(H, generator=g).to(dtype),
        "norm.weight": gain(H),
        "markov_head.dflare_norm.weight": gain(H),
        "markov_head.hidden_out.weight": mat(H, R, 1.0),
        "markov_head.hidden_out.bias": (0.1 * torch.randn(H, generator=g)).to(dtype),
        "markov_head.markov_w1.weight": torch.randn(vocab, R, generator=g).to(dtype),
        "markov_head.rnn_proj.weight": mat(3 * R, 2 * R + 2 * H, 2.0),
        "markov_head.rnn_proj.bias": (0.1 * torch.randn(3 * R, generator=g)).to(dtype),
    }
    for i in range(cfg.num_layers):
        p = f"layers.{i}."
        t[p + "dflare_norm.weight"] = gain(H)
        t[p + "input_layernorm.weight"] = gain(H)
        t[p + "post_attention_layernorm.weight"] = gain(H)
        t[p + "self_attn.q_proj.weight"] = mat(nq * D, H)
        t[p + "self_attn.k_proj.weight"] = mat(nkv * D, H)
        t[p + "self_attn.v_proj.weight"] = mat(nkv * D, H)
        t[p + "self_attn.k_proj_ctx.weight"] = mat(nkv * D, H)
        t[p + "self_attn.v_proj_ctx.weight"] = mat(nkv * D, H)
        t[p + "self_attn.o_proj.weight"] = mat(H, nq * D)
        t[p + "self_attn.q_norm.weight"] = gain(D)
        t[p + "self_attn.k_norm.weight"] = gain(D)
        t[p + "self_attn.k_norm_ctx.weight"] = gain(D)
        t[p + "mlp.gate_proj.weight"] = mat(inter, H)
        t[p + "mlp.up_proj.weight"] = mat(inter, H)
        t[p + "mlp.down_proj.weight"] = mat(H, inter)
    return Draft(cfg, t)


def synthetic_taps(
    n: int, cfg: DraftConfig, seed: int = 0, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """``[n, taps, hidden]`` taps of mixed scale with a few outlier channels,
    already rounded to bf16 (the precision the draft stores them in)."""
    g = torch.Generator().manual_seed(seed)
    scale = torch.tensor([1.0, 3.0, 8.0, 2.0][: cfg.num_taps], dtype=torch.float32)
    x = torch.randn(n, cfg.num_taps, cfg.hidden_size, generator=g)
    x = x * scale[None, :, None]
    outliers = torch.randperm(cfg.hidden_size, generator=g)[:4]
    x[:, :, outliers] *= 40.0
    return x.bfloat16().to(dtype)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w


def mix_rms(
    taps: torch.Tensor, alpha: torch.Tensor, w: torch.Tensor, eps: float
) -> torch.Tensor:
    """RMSNorm of ``sum_t alpha[t] * taps[..., t, :]``."""
    mixed = (taps * alpha[:, None]).sum(-2)
    return rms_norm(mixed, w, eps)


def rope_tables(
    pos: torch.Tensor, head_dim: int, theta: float, table: str = "exact"
) -> tuple[torch.Tensor, torch.Tensor]:
    """``cos, sin [T, head_dim / 2]`` of the neox RoPE.

    ``exact``: float64 angles.  ``fp32``: the float32 ``pos * inv_freq`` product the
    reference's cos/sin cache stores (it loses ~``pos * 6e-8`` rad at long context).
    """
    j = torch.arange(head_dim // 2, dtype=torch.float64)
    inv_freq = theta ** (-2.0 * j / head_dim)
    if table == "fp32":
        ang = pos.float()[:, None] * inv_freq.float()[None, :]
        return ang.cos().double(), ang.sin().double()
    ang = pos.double()[:, None] * inv_freq[None, :]
    return ang.cos(), ang.sin()


def rope(
    x: torch.Tensor, pos: torch.Tensor, theta: float, table: str = "exact"
) -> torch.Tensor:
    """Neox RoPE of ``x [T, heads, D]`` at positions ``pos [T]``."""
    cos, sin = rope_tables(pos, x.shape[-1], theta, table)
    cos, sin = cos[:, None, :].to(x.dtype), sin[:, None, :].to(x.dtype)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def head_norm(x: torch.Tensor, w: torch.Tensor, heads: int, eps: float) -> torch.Tensor:
    """Per-head RMSNorm of ``x [T, heads * D]`` -> ``[T, heads, D]``."""
    return rms_norm(x.view(x.shape[0], heads, -1), w, eps)


def attend(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """Bidirectional GQA attention: ``q [Q, nq, D]``, ``k, v [N, nkv, D]``."""
    group = q.shape[1] // k.shape[1]
    kk, vv = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
    s = torch.einsum("qhd,nhd->hqn", q, kk) * scale
    return torch.einsum("hqn,nhd->qhd", torch.softmax(s, dim=-1), vv)


def context_kv(
    d: Draft, taps: torch.Tensor, pos: torch.Tensor, rope_table: str = "exact"
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per draft layer ``(K [N, nkv, D], V [N, nkv, D])`` of ``taps [N, taps, H]``."""
    c = d.cfg
    out = []
    for i in range(c.num_layers):
        x = mix_rms(taps, d.alpha[i], d.layer(i, "dflare_norm.weight"), c.eps)
        k = x @ d.layer(i, "self_attn.k_proj_ctx.weight").T
        v = x @ d.layer(i, "self_attn.v_proj_ctx.weight").T
        k = head_norm(
            k, d.layer(i, "self_attn.k_norm_ctx.weight"), c.num_kv_heads, c.eps
        )
        k = rope(k, pos, c.rope_theta, rope_table)
        out.append((k, v.view(x.shape[0], c.num_kv_heads, c.head_dim)))
    return out


def block_input(d: Draft, anchor_embedding: torch.Tensor) -> torch.Tensor:
    """``[B, H]``: the normalized anchor embedding, then ``mask_embed`` rows."""
    x = d["mask_embed"].expand(d.cfg.block_size, -1).clone()
    x[0] = anchor_embedding
    return x


def block_hidden(
    d: Draft,
    ctx: list[tuple[torch.Tensor, torch.Tensor]],
    ctx_pos: torch.Tensor,
    x0: torch.Tensor,
    start: int,
    rope_table: str = "exact",
) -> torch.Tensor:
    """Final-norm output ``[B, H]`` of the block starting at position ``start``."""
    c = d.cfg
    B = c.block_size
    pos = torch.arange(start, start + B)
    resid = x0.clone()
    xn = rms_norm(resid, d.layer(0, "input_layernorm.weight"), c.eps)
    for i in range(c.num_layers):
        q = xn @ d.layer(i, "self_attn.q_proj.weight").T
        k = xn @ d.layer(i, "self_attn.k_proj.weight").T
        v = (xn @ d.layer(i, "self_attn.v_proj.weight").T).view(
            B, c.num_kv_heads, c.head_dim
        )
        q = head_norm(q, d.layer(i, "self_attn.q_norm.weight"), c.num_heads, c.eps)
        k = head_norm(k, d.layer(i, "self_attn.k_norm.weight"), c.num_kv_heads, c.eps)
        q, k = (
            rope(q, pos, c.rope_theta, rope_table),
            rope(k, pos, c.rope_theta, rope_table),
        )
        kc, vc = ctx[i]
        if i in c.local_layer_ids:
            keep = ctx_pos >= start - c.window
            kc, vc = kc[keep], vc[keep]
        o = attend(q, torch.cat([kc, k]), torch.cat([vc, v]), c.head_dim**-0.5)
        resid = resid + o.reshape(B, -1) @ d.layer(i, "self_attn.o_proj.weight").T
        xn = rms_norm(resid, d.layer(i, "post_attention_layernorm.weight"), c.eps)
        gate = xn @ d.layer(i, "mlp.gate_proj.weight").T
        up = xn @ d.layer(i, "mlp.up_proj.weight").T
        resid = (
            resid
            + (torch.nn.functional.silu(gate) * up)
            @ d.layer(i, "mlp.down_proj.weight").T
        )
        nxt = (
            d.layer(i + 1, "input_layernorm.weight")
            if i + 1 < c.num_layers
            else d["norm.weight"]
        )
        xn = rms_norm(resid, nxt, c.eps)
    return xn


def f_rnn(d: Draft, last_taps: torch.Tensor) -> torch.Tensor:
    """``[H]`` Markov-head context of the last committed token's taps ``[taps, H]``."""
    return mix_rms(
        last_taps, d.alpha[-1], d["markov_head.dflare_norm.weight"], d.cfg.eps
    )


@dataclass
class MarkovTrace:
    logits: torch.Tensor  # [steps, V]
    tokens: torch.Tensor  # [steps] int64 (the teacher tokens or the greedy chain)
    states: torch.Tensor  # [steps, R] state after each step
    latents: torch.Tensor  # [steps, R]


def markov_chain(
    d: Draft,
    h: torch.Tensor,
    f: torch.Tensor,
    anchor: int,
    lm_head: torch.Tensor,
    tokens: torch.Tensor | None = None,
    bf16_gate: bool = False,
) -> MarkovTrace:
    """The Markov head over ``h [steps, H]`` (backbone rows ``1..B-1``).

    ``tokens``: teacher-forced previous tokens (``tokens[i]`` is chosen at step ``i``);
    ``None`` follows the greedy argmax chain.  ``bf16_gate`` rounds the gate's
    intermediates to bf16 like the reference does (information only).
    """
    c = d.cfg
    R = c.markov_rank
    W = d["markov_head.rnn_proj.weight"]
    bias = d["markov_head.rnn_proj.bias"]
    ctx = bias + torch.cat([h, f.expand_as(h)], dim=-1) @ W[:, 2 * R :].T
    state = torch.zeros(R, dtype=h.dtype)
    prev = anchor
    base = h @ lm_head.T  # [steps, V]
    logits, chosen, states, latents = [], [], [], []

    def r16(x: torch.Tensor) -> torch.Tensor:
        return x.float().bfloat16().to(x.dtype) if bf16_gate else x

    for i in range(c.num_steps):
        inp = torch.cat([state, d["markov_head.markov_w1.weight"][prev]])
        raw = ctx[i] + W[:, : 2 * R] @ inp
        gate_raw, cand_raw, out_raw = r16(raw).chunk(3)
        gate = r16(torch.sigmoid(gate_raw))
        keep = r16(gate * state)
        update = r16(r16(1.0 - gate) * r16(torch.tanh(cand_raw)))
        state = r16(keep + update)
        latent = r16(torch.tanh(out_raw))
        hid = (
            d["markov_head.hidden_out.weight"] @ latent
            + d["markov_head.hidden_out.bias"]
        )
        full = base[i] + lm_head @ hid
        lg = c.softcap * torch.tanh(full / c.softcap)
        logits.append(lg)
        prev = int(tokens[i]) if tokens is not None else int(lg.argmax())
        chosen.append(prev)
        states.append(state)
        latents.append(latent)
    return MarkovTrace(
        torch.stack(logits),
        torch.tensor(chosen, dtype=torch.int64),
        torch.stack(states),
        torch.stack(latents),
    )


def propose(
    d: Draft,
    taps: torch.Tensor,
    ctx_pos: torch.Tensor,
    anchor_embedding: torch.Tensor,
    anchor: int,
    lm_head: torch.Tensor,
    tokens: torch.Tensor | None = None,
    start: int | None = None,
    rope_table: str = "exact",
) -> tuple[torch.Tensor, MarkovTrace]:
    """One draft proposal of a request.

    ``taps [N, taps, H]`` are the taps of the context positions ``ctx_pos [N]`` (rows
    after the last committed position are not given).  The block starts at ``start``
    (default: one past the last context position).  Returns the backbone hidden rows
    ``[B, H]`` and the Markov trace.
    """
    start = int(ctx_pos[-1]) + 1 if start is None else start
    ctx = context_kv(d, taps, ctx_pos, rope_table)
    h = block_hidden(
        d, ctx, ctx_pos, block_input(d, anchor_embedding), start, rope_table
    )
    trace = markov_chain(d, h[1:], f_rnn(d, taps[-1]), anchor, lm_head, tokens)
    return h, trace


def fold_head(d: Draft, lm_head: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``lm_head @ hidden_out`` and ``lm_head @ hidden_out.bias`` in float64 (the
    reference's fold, before its 16-bit rounding)."""
    w = d["markov_head.hidden_out.weight"].double()
    b = d["markov_head.hidden_out.bias"].double()
    h = lm_head.double()
    return h @ w, h @ b


def read_target_head(
    ckpt: str | Path, vocab: int, rows: list[int] | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(embed rows, embed_norm.weight, unembed [vocab, H])`` of the target checkpoint
    (bf16 on disk, returned as float64).  ``rows``: the embedding rows to read."""
    from safetensors import safe_open

    index = json.loads((Path(ckpt) / "model.safetensors.index.json").read_text())
    shard = Path(ckpt) / index["weight_map"]["model.llm.unembed.weight"]
    with safe_open(str(shard), framework="pt") as f:
        norm = f.get_tensor("model.llm.embed_norm.weight").double()
        sl = f.get_slice("model.llm.embed.weight")
        emb = (
            torch.stack([sl[r : r + 1][0] for r in rows]).double()
            if rows is not None
            else sl[:].double()
        )
        unembed = f.get_slice("model.llm.unembed.weight")[:vocab].double()
    return emb, norm, unembed


def fp16_weight_report(d: Draft) -> dict[str, float]:
    """How the bf16 -> fp16 weight conversion changes the draft: the largest
    magnitude, the number of values that overflow or flush to zero, and the largest
    relative rounding."""
    worst, ovf, flush, rel = 0.0, 0, 0, 0.0
    for t in d.tensors.values():
        a = t.abs()
        worst = max(worst, float(a.max()))
        h = t.float().half()
        ovf += int(torch.isinf(h).sum())
        flush += int(((h == 0) & (t != 0)).sum())
        nz = a > 6.2e-5  # fp16 normal range
        if nz.any():
            err = ((h.double() - t).abs() / a.clamp_min(1e-300))[nz]
            rel = max(rel, float(err.max()))
    return {"max_abs": worst, "overflow": ovf, "flushed_to_zero": flush, "max_rel": rel}


# ---------------------------------------------------------------------------
# The HIP draft under test (needs the compiled ops and a gfx1030 device)
# ---------------------------------------------------------------------------
def round_to_bf16(d: Draft) -> Draft:
    """``d`` with every tensor rounded to bf16, the values a checkpoint stores."""
    return Draft(d.cfg, {k: v.bfloat16().to(v.dtype) for k, v in d.tensors.items()})


@dataclass
class Context:
    """One request: the taps ``[N, taps, H]`` of its committed rows (row ``j`` is
    position ``j``) and the anchor token (the last committed one)."""

    taps: torch.Tensor
    anchor: int


@dataclass
class World:
    """Engine state of one proposal batch: pages, tables, buffers, metadata.

    ``pad`` requests follow the real ones, as the runner pads a batch up to a graph
    size: sequence length 0, null block table, slot mapping -1.
    """

    contexts: list[Context]
    pad: int
    stored: torch.Tensor  # [T, taps * H] bf16: the taps as the draft stores them
    used_taps: list[torch.Tensor]  # per request [N, taps, H] float64 (``stored``)
    ctx_positions: torch.Tensor
    ctx_slots: torch.Tensor
    query_start_loc: torch.Tensor
    num_rejected: torch.Tensor
    block_table: torch.Tensor  # [R + pad, max_blocks] int32
    input_ids: torch.Tensor
    positions: torch.Tensor
    spec: object
    seq_lens: torch.Tensor

    @property
    def num_reqs(self) -> int:
        return len(self.contexts) + self.pad


@dataclass
class HipResult:
    hidden: torch.Tensor  # [R, B, H] float32: the draft's final-norm rows
    tokens: torch.Tensor  # [R, steps] int64: the greedy chain
    logits: torch.Tensor | None  # [R, steps, V] float32
    f_rnn: torch.Tensor  # [R, H] float32


def draft_hf_config(d: Draft, vocab: int, padded_vocab: int):
    """The ``Inkling2DSparkConfig`` that matches the oracle's ``d.cfg``."""
    from vllm.models.inkling2_loop.configs import Inkling2DSparkConfig

    c = d.cfg
    return Inkling2DSparkConfig(
        block_size=c.block_size,
        target_layer_ids=[5, 23, 41, 48][: c.num_taps],
        hidden_size=c.hidden_size,
        num_hidden_layers=c.num_layers,
        num_attention_heads=c.num_heads,
        num_key_value_heads=c.num_kv_heads,
        head_dim=c.head_dim,
        intermediate_size=c.intermediate_size,
        rms_norm_eps=c.eps,
        rope_theta=c.rope_theta,
        local_layer_ids=list(c.local_layer_ids),
        sliding_window_size=c.window,
        markov_rank=c.markov_rank,
        vocab_size=vocab,
        padded_vocab_size=padded_vocab,
        final_logit_softcapping=c.softcap,
    )


def fake_vllm_config(hf_config, num_steps: int, max_reqs: int, max_model_len: int):
    """The slice of ``VllmConfig`` the draft model and its KV owners read."""
    import types

    return types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(
            draft_model_config=types.SimpleNamespace(hf_config=hf_config),
            num_speculative_tokens=num_steps,
            draft_sample_method="greedy",
        ),
        model_config=types.SimpleNamespace(
            dtype=torch.float16, max_model_len=max_model_len
        ),
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_reqs),
        compilation_config=types.SimpleNamespace(static_forward_context={}),
        cache_config=types.SimpleNamespace(block_size=16),
    )


def build_model(
    d: Draft,
    vocab: int,
    padded_vocab: int,
    max_reqs: int,
    max_model_len: int,
    device: str | torch.device,
    load: bool = True,
    num_steps: int | None = None,
    tp=None,
):
    """``Inkling2DSparkModel`` of ``d``'s shapes on ``device``; with ``load`` the
    weights of ``d`` (as bf16) go through the model's own loader.  ``num_steps`` is
    the speculative token count of the (fake) config, the block's by default; ``tp``
    (a ``parallel.TensorParallel``) is the tensor-parallel group the model finds
    itself in, the single rank by default."""
    from vllm.models.inkling2_loop.amd import dspark as mod
    from vllm.models.inkling2_loop.amd import state as state_mod

    hf = draft_hf_config(d, vocab, padded_vocab)
    steps = d.cfg.num_steps if num_steps is None else num_steps
    fake = fake_vllm_config(hf, steps, max_reqs, max_model_len)
    old, old_tp = state_mod.get_current_vllm_config, mod.current_tensor_parallel
    state_mod.get_current_vllm_config = lambda: fake
    if tp is not None:
        mod.current_tensor_parallel = lambda: tp
    try:
        with torch.device(device):
            model = mod.Inkling2DSparkModel(vllm_config=fake)
    finally:
        state_mod.get_current_vllm_config = old
        mod.current_tensor_parallel = old_tp
    if load:
        model.load_weights((k, v.bfloat16()) for k, v in d.tensors.items())
    return model


class HipDraft:
    """The serving draft model driven the way the V2 runner drives it.

    ``lm_head`` / ``embed_rows`` / ``embed_norm`` stand for the target's tied
    weights (they enter the draft in fp16).  Everything the oracle needs of them is
    kept in float64 as the draft sees it (``lm64``, ``embed64``, ``norm64``).  With
    ``tp`` (a ``parallel.TensorParallel`` of size > 1) this is one rank of a tensor-
    parallel group: the model holds the vocabulary shard of the head and of the
    embedding table that the target's rank would, the oracle still sees all of it.
    """

    def __init__(
        self,
        d: Draft,
        lm_head: torch.Tensor,
        embed_rows: dict[int, torch.Tensor],
        embed_norm: torch.Tensor,
        *,
        padded_vocab: int,
        max_reqs: int = 4,
        page_size: int = 16,
        max_model_len: int = 8192,
        device: str = "cuda",
        tp=None,
    ) -> None:
        import types

        from vllm.models.inkling2_loop.amd import dspark as mod
        from vllm.models.inkling2_loop.amd.parallel import NO_TP, lm_head_shard

        c = d.cfg
        self.d, self.c, self.mod = d, c, mod
        self.device = torch.device(device)
        self.max_reqs, self.page_size = max_reqs, page_size
        tp = NO_TP if tp is None else tp
        self.tp = tp
        self.model = build_model(
            d,
            lm_head.shape[0],
            padded_vocab,
            max_reqs,
            max_model_len,
            self.device,
            tp=tp,
        )
        dev = self.device
        table = torch.zeros(padded_vocab, c.hidden_size, dtype=torch.float16)
        for tok, row in embed_rows.items():
            table[tok] = row.half()
        norm16 = embed_norm.half().to(dev)
        lm16 = lm_head.half().to(dev)
        shard, head = table, lm16
        if tp.enabled:  # this rank's rows, zero where the last shard is padded
            lo, rows, take = lm_head_shard(padded_vocab, tp.size, tp.rank)
            shard = torch.zeros(rows, c.hidden_size, dtype=torch.float16)
            shard[:take] = table[lo : lo + take]
            lo, rows, take = lm_head_shard(lm_head.shape[0], tp.size, tp.rank)
            head = torch.zeros(rows, c.hidden_size, dtype=torch.float16, device=dev)
            head[:take] = lm16[lo : lo + take]
        self.model.model.embed_tokens = types.SimpleNamespace(weight=shard.to(dev))
        self.model.model.embed_norm = types.SimpleNamespace(
            weight=norm16, variance_epsilon=c.eps
        )
        self.model.lm_head = types.SimpleNamespace(weight=head)
        self.model.refresh_derived_weights()
        self.lm64 = lm16.double().cpu()
        self.norm64 = norm16.double().cpu()
        self.embed64 = {t: table[t].double() for t in embed_rows}
        self.gen = torch.Generator().manual_seed(1234)
        self.cache_pages = 0

    # -- engine state -----------------------------------------------------
    def _store_taps(self, contexts: list[Context], stored: torch.Tensor):
        """The draft's tap store on the taps of ``contexts`` (the target hands over
        its three fp32 partial blocks and the fp16 final output); returns the taps
        the draft then holds (bf16 values) per request, float64."""
        from vllm import _custom_ops as ops

        c, dev = self.c, self.device
        taps = torch.cat([x.taps for x in contexts])  # [T, taps, H]
        total = taps.shape[0]
        srcs = [taps[:, t].float().contiguous() for t in range(c.num_taps - 1)]
        srcs.append(taps[:, c.num_taps - 1].half().contiguous())
        ops.inkling2_dspark_store_taps([s.to(dev) for s in srcs], stored, total)
        used = stored[:total].cpu().view(total, c.num_taps, c.hidden_size).double()
        return list(used.split([x.taps.shape[0] for x in contexts]))

    def prepare(self, contexts: list[Context], pad: int = 0) -> World:
        """Pages (shuffled, page 0 left as the null block), block table, slot
        mappings, input buffers and metadata of a batch of requests."""
        import types

        mod, c, ps, dev = self.mod, self.c, self.page_size, self.device
        block = mod.BLOCK
        reqs = len(contexts)
        num_reqs = reqs + pad
        assert reqs > 0 and num_reqs <= self.max_reqs
        lens = [x.taps.shape[0] for x in contexts]
        need = [-(-(n + block) // ps) for n in lens]
        pool = (torch.randperm(sum(need), generator=self.gen) + 1).tolist()
        for layer in self.model.model.layers:
            layer.kv_owner.kv_cache = torch.zeros(
                sum(need) + 1,
                c.num_kv_heads,
                ps,
                2 * c.head_dim,
                dtype=torch.float16,
                device=dev,
            )
        table = torch.zeros(num_reqs, max(need), dtype=torch.int32)
        ctx_slots, blk_slots, positions = [], [], []
        cursor = 0
        for r, (n, k) in enumerate(zip(lens, need)):
            pages = torch.tensor(pool[cursor : cursor + k], dtype=torch.int64)
            cursor += k
            table[r, :k] = pages.int()
            pos = torch.arange(n + block)
            slot = pages[pos // ps] * ps + pos % ps
            ctx_slots.append(slot[:n])
            blk_slots.append(slot[n:])
            positions.append(pos[n:])
        for _ in range(pad):
            blk_slots.append(torch.full((block,), -1, dtype=torch.int64))
            positions.append(torch.zeros(block, dtype=torch.int64))
        stored = torch.zeros(
            sum(lens), c.num_taps * c.hidden_size, dtype=torch.bfloat16
        )
        stored = stored.to(dev)
        used = self._store_taps(contexts, stored)
        ids = torch.zeros(self.max_reqs * block, dtype=torch.int32)
        for r, x in enumerate(contexts):
            ids[r * block] = x.anchor
        steps = c.num_steps
        rows = torch.arange(num_reqs)[:, None] * block + 1 + torch.arange(steps)[None]
        sample_pos = torch.stack([p[1:] for p in positions]).flatten()
        spec = types.SimpleNamespace(
            num_speculative_steps=steps,
            num_query_per_req=block,
            draft_logits=None,
            sample_indices=rows.flatten().to(dev),
            sample_idx_mapping=torch.arange(num_reqs)
            .repeat_interleave(steps)
            .int()
            .to(dev),
            sample_pos=sample_pos.to(dev),
            input_buffers=types.SimpleNamespace(input_ids=ids.to(dev)),
            draft_tokens=torch.zeros(
                self.max_reqs, steps, dtype=torch.int64, device=dev
            ),
            f_rnn=torch.zeros(
                self.max_reqs, c.hidden_size, dtype=torch.float16, device=dev
            ),
        )
        seq_lens = torch.tensor(
            [n + block for n in lens] + [0] * pad, dtype=torch.int32, device=dev
        )
        table = table.to(dev)
        md = mod.Inkling2DraftMetadata(
            block_table=table,
            slot_mapping=torch.cat(blk_slots).to(dev),
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=num_reqs * block,
        )
        self.model._layer_metadata = lambda: [md] * len(self.model.model.layers)
        qsl = torch.tensor(
            [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32
        )
        return World(
            contexts=contexts,
            pad=pad,
            stored=stored,
            used_taps=used,
            ctx_positions=torch.cat([torch.arange(n) for n in lens]).to(dev),
            ctx_slots=torch.cat(ctx_slots).to(dev),
            query_start_loc=qsl.to(dev),
            num_rejected=torch.zeros(reqs, dtype=torch.int32, device=dev),
            block_table=table,
            input_ids=spec.input_buffers.input_ids,
            positions=torch.cat(positions).to(dev),
            spec=spec,
            seq_lens=seq_lens,
        )

    def refill(self, w: World, contexts: list[Context]) -> None:
        """New taps and anchors for the same request shapes, written into the same
        buffers: the state a CUDA graph captured on ``w`` sees when replayed."""
        shapes = [x.taps.shape[0] for x in contexts]
        assert shapes == [x.taps.shape[0] for x in w.contexts]
        w.used_taps = self._store_taps(contexts, w.stored)
        for r, x in enumerate(contexts):
            w.input_ids[r * self.mod.BLOCK] = x.anchor
        w.contexts = contexts

    # -- passes -----------------------------------------------------------
    @torch.no_grad()
    def context_pass(self, w: World) -> None:
        """``f_rnn`` of the last committed row and the context K/V of every layer
        (what the runner does with the target's taps before the draft block)."""
        self.model.compute_f_rnn(
            w.stored, w.query_start_loc, w.num_rejected, w.spec.f_rnn, len(w.contexts)
        )
        self.model.precompute_and_store_context_kv(
            w.stored, w.ctx_positions, w.ctx_slots
        )

    @torch.no_grad()
    def block_pass(self, w: World) -> torch.Tensor:
        """Backbone over the blocks, then the Markov head's sequential sampling."""
        rows = w.num_reqs * self.mod.BLOCK
        hidden = self.model.forward(w.input_ids[:rows], w.positions)
        assert self.model.sample_sequential(w.spec, w.num_reqs, hidden)
        return hidden

    @torch.no_grad()
    def propose(
        self, contexts: list[Context], capture_logits: bool = True, pad: int = 0
    ) -> HipResult:
        """Context pass then block pass for ``contexts`` (and ``pad`` padding
        requests).  ``capture_logits`` also returns the logits every step sampled
        from (host copies: not for timing)."""
        from vllm import _custom_ops as ops

        w = self.prepare(contexts, pad)
        self.context_pass(w)
        captured: list[torch.Tensor] = []
        real = ops.inkling2_dspark_argmax
        if capture_logits:

            def spy(logits, out, num_rows, vocab):
                captured.append(logits[:num_rows, :vocab].float().cpu())
                real(logits, out, num_rows, vocab)

            ops.inkling2_dspark_argmax = spy
        try:
            hidden = self.block_pass(w)
            torch.accelerator.synchronize()
        finally:
            ops.inkling2_dspark_argmax = real
        reqs, block, steps = len(contexts), self.mod.BLOCK, self.c.num_steps
        logits = None
        if capture_logits:  # one argmax call per (head chunk, step)
            chunks = -(-w.num_reqs // self.mod._HEAD_REQS)
            logits = torch.cat(
                [
                    torch.stack(captured[i * steps : (i + 1) * steps], 1)
                    for i in range(chunks)
                ]
            )[:reqs]
        self.world = w
        return HipResult(
            hidden=hidden[: reqs * block].float().cpu().view(reqs, block, -1),
            tokens=w.spec.draft_tokens[:reqs].cpu().clone(),
            logits=logits,
            f_rnn=w.spec.f_rnn[:reqs].float().cpu(),
        )

    def bench(self, w: World, iters: int = 20) -> dict[str, float]:
        """Mean milliseconds of the context pass and of the block pass (backbone +
        Markov head) of the batch ``w``, measured with HIP events."""
        out: dict[str, float] = {}
        for name, fn in (("context", self.context_pass), ("block", self.block_pass)):
            for _ in range(3):
                fn(w)
            start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(iters):
                fn(w)
            stop.record()
            stop.synchronize()
            out[name + "_ms"] = start.elapsed_time(stop) / iters
        return out


def cache_error(hip: HipDraft, w: World) -> dict[str, float]:
    """Largest |draft cache - oracle| of the context K and V over layers and
    requests, and the largest oracle magnitude (to scale it)."""
    c = hip.c
    worst_k = worst_v = scale = 0.0
    for r, ctx in enumerate(w.contexts):
        n = ctx.taps.shape[0]
        ref = context_kv(hip.d, w.used_taps[r], torch.arange(n))
        pos = torch.arange(n)
        pages = w.block_table[r].long().cpu()[pos // hip.page_size].to(hip.device)
        offs = (pos % hip.page_size).to(hip.device)
        for layer, (k_ref, v_ref) in zip(hip.model.model.layers, ref):
            got = layer.kv_owner.kv_cache[pages, :, offs, :].double().cpu()
            k, v = got[..., : c.head_dim], got[..., c.head_dim :]
            worst_k = max(worst_k, float((k - k_ref).abs().max()))
            worst_v = max(worst_v, float((v - v_ref).abs().max()))
            scale = max(scale, float(k_ref.abs().max()), float(v_ref.abs().max()))
    return {"k_max_abs": worst_k, "v_max_abs": worst_v, "kv_max_magnitude": scale}


def compare(hip: HipDraft, res: HipResult) -> dict[str, float]:
    """The HIP proposal against the oracle, request by request.

    The oracle follows the HIP draft's own tokens (teacher forcing) so every step's
    logits are compared on identical Markov inputs.  ``token_gap`` is how far below
    the oracle's best logit the HIP token's logit lies (zero when it is the argmax).
    """
    c, w = hip.c, hip.world
    out = dict.fromkeys(
        ("hidden_max_abs", "hidden_rel_l2", "f_rnn_max_abs", "logit_max_abs"), 0.0
    )
    out.update(logit_mean_abs=0.0, token_gap=0.0, token_mismatch=0, steps=0)
    mean_acc = 0.0
    for r, ctx in enumerate(w.contexts):
        n = ctx.taps.shape[0]
        row = hip.embed64[ctx.anchor]
        x = rms_norm(row, hip.norm64, c.eps)
        taps = w.used_taps[r]
        h, trace = propose(
            hip.d, taps, torch.arange(n), x, ctx.anchor, hip.lm64, tokens=res.tokens[r]
        )
        err = res.hidden[r].double() - h
        out["hidden_max_abs"] = max(out["hidden_max_abs"], float(err.abs().max()))
        out["hidden_rel_l2"] = max(out["hidden_rel_l2"], float(err.norm() / h.norm()))
        f_err = res.f_rnn[r].double() - f_rnn(hip.d, taps[-1])
        out["f_rnn_max_abs"] = max(out["f_rnn_max_abs"], float(f_err.abs().max()))
        if res.logits is not None:
            diff = (res.logits[r].double() - trace.logits).abs()
            out["logit_max_abs"] = max(out["logit_max_abs"], float(diff.max()))
            mean_acc += float(diff.mean())
        best = trace.logits.max(-1).values
        mine = trace.logits.gather(1, res.tokens[r][:, None])[:, 0]
        out["token_gap"] = max(out["token_gap"], float((best - mine).max()))
        out["token_mismatch"] += int((trace.logits.argmax(-1) != res.tokens[r]).sum())
        out["steps"] += c.num_steps
    out["logit_mean_abs"] = mean_acc / len(w.contexts)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=str(Path("~/hfcache/inkling2-mini").expanduser()))
    ap.add_argument(
        "--context", type=int, nargs="+", default=[300], help="context rows per request"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hip", action="store_true", help="also run the HIP draft")
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--bench", type=int, default=0, help="time this many proposals")
    ap.add_argument(
        "--bench-only", action="store_true", help="time the HIP draft, skip the oracle"
    )
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    cfg_json = json.loads((ckpt / "config.json").read_text())
    text = cfg_json["text_config"]
    cfg = DraftConfig.from_mtp_config(
        cfg_json["mtp_config"], text.get("final_logit_softcapping", 30.0)
    )
    t0 = time.time()
    d = load_draft(ckpt / "mtp.safetensors", cfg)
    print(
        f"loaded draft in {time.time() - t0:.1f}s; fp16 report:", fp16_weight_report(d)
    )
    vocab = text["unpadded_vocab_size"]
    anchors = [1234 + 17 * i for i in range(len(args.context))]
    emb, norm, unembed = read_target_head(ckpt, vocab, anchors)
    contexts = [
        Context(synthetic_taps(n, cfg, args.seed + i), a)
        for i, (n, a) in enumerate(zip(args.context, anchors))
    ]
    if not args.hip:
        for i, ctx in enumerate(contexts):
            n = ctx.taps.shape[0]
            x = rms_norm(emb[i], norm, cfg.eps)
            t0 = time.time()
            _, trace = propose(d, ctx.taps, torch.arange(n), x, ctx.anchor, unembed)
            print(
                f"oracle proposal {i} in {time.time() - t0:.1f}s; greedy chain "
                f"{trace.tokens.tolist()}"
            )
        return 0

    d = round_to_bf16(d)
    padded = d["markov_head.markov_w1.weight"].shape[0]
    hip = HipDraft(
        d,
        unembed,
        dict(zip(anchors, emb)),
        norm,
        padded_vocab=padded,
        max_reqs=max(4, len(contexts)),
        page_size=args.page_size,
    )
    if args.bench_only:
        w = hip.prepare(contexts)
        print("contexts", args.context, "timing:", hip.bench(w, args.bench or 20))
        return 0
    res = hip.propose(contexts)
    print("hip chains:", res.tokens.tolist())
    print("context cache vs oracle:", cache_error(hip, hip.world))
    print("draft vs oracle:", compare(hip, res))
    if args.bench:
        w = hip.prepare(contexts)
        print("timing:", hip.bench(w, args.bench))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
