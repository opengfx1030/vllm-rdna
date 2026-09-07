# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA2 (gfx1030) int8 flash-decode attention kernel.

Derived in spirit from the vllm-rdna2-recipe `fd_rdna2` plugin (Aron Hsiao,
GPL-3.0-or-later, 2026); re-derived here in original Triton against this
tree's per-token-head int8 KV layout (separate K and V split views over the
packed cache, per-token-head scales held in dedicated float32 cache tensors).

Layout notes (see TritonAttentionImpl._pth_key_value_caches /
_ensure_scale_caches in vllm/v1/attention/backends/triton_attn.py):
  - packed cache: (num_blocks, nkv, block_size, 2 * padded_hs) int8,
    padded_hs = hs + 4 (sizeof(float32) worth of scale bytes)
  - k view: (num_blocks, block_size, nkv, padded_hs) — K bytes at [0, hs),
    storage_offset 0
  - v view: same shape — V bytes at [0, hs) of the view, i.e. storage_offset
    padded_hs elements into the packed row
  - k/v_scale_cache: float32 (num_blocks, block_size, nkv) strided views over
    the inline scale bytes

Because the split views carry their own storage offsets, the kernel's int32
views of K and V both start at the first data byte of their respective
sections — no cross-section offset arithmetic inside the kernel.

Only the single-query decode path is implemented (q == 1). Speculative
verification (q > 1) delegates to the stock kernel until the batched
multi-query kernel lands.

Public entry point: :func:`fd_decode` (called from the FD_RDNA2 plugin).
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _fd_phase1(
    q_ptr,                # (KVH, 4, 64, PAD) fp16, byte-permuted Q
    k_i32_ptr,            # int32 view over K's storage (row start = K byte 0)
    v_i32_ptr,            # int32 view over V's storage (row start = V byte 0)
    k_scale_ptr,          # float32 K scales, (num_blocks, block_size, nkv)
    v_scale_ptr,          # float32 V scales
    bt_ptr,               # int32 block_table row for this sequence
    seq_ptr,              # int32 seqused_k — number of valid keys
    m_ptr, l_ptr, acc_ptr,    # fp32 partials: (chunks, KVH, PAD) / (chunks, KVH, PAD, 256)
    s_k_blk, s_k_kvh, s_k_tok,   # K strides in int32 units
    s_v_blk, s_v_kvh, s_v_tok,   # V strides in int32 units
    s_ks_blk, s_ks_tok, s_ks_kvh,  # k_scale strides in float32 elements
    BS: tl.constexpr, CHUNK: tl.constexpr, TILE: tl.constexpr,
    GQA: tl.constexpr, KVH: tl.constexpr, PAD: tl.constexpr, HS4: tl.constexpr,
):
    chunk = tl.program_id(0)
    kvh = tl.program_id(1)
    n = tl.load(seq_ptr)
    start = chunk * CHUNK
    if start >= n:
        return

    offs_w = tl.arange(0, 64)               # int32 words within a row (64 = 256B)
    offs_q = tl.arange(0, PAD)              # query columns (GQA padded to pow2)
    qbase = q_ptr + kvh * (4 * 64 * PAD)
    Q0 = tl.load(qbase + 0 * 64 * PAD + offs_w[:, None] * PAD + offs_q[None, :])
    Q1 = tl.load(qbase + 1 * 64 * PAD + offs_w[:, None] * PAD + offs_q[None, :])
    Q2 = tl.load(qbase + 2 * 64 * PAD + offs_w[:, None] * PAD + offs_q[None, :])
    Q3 = tl.load(qbase + 3 * 64 * PAD + offs_w[:, None] * PAD + offs_q[None, :])

    m_i = tl.full((PAD,), float("-inf"), tl.float32)
    l_i = tl.zeros((PAD,), tl.float32)
    a0 = tl.zeros((PAD, 64), tl.float32)
    a1 = tl.zeros((PAD, 64), tl.float32)
    a2 = tl.zeros((PAD, 64), tl.float32)
    a3 = tl.zeros((PAD, 64), tl.float32)

    for t in range(0, CHUNK, TILE):
        pos = start + t + tl.arange(0, TILE)
        tmask = pos < n
        blk = tl.load(bt_ptr + pos // BS, mask=tmask, other=0)

        k_row = blk * s_k_blk + kvh * s_k_kvh + (pos % BS) * s_k_tok
        v_row = blk * s_v_blk + kvh * s_v_kvh + (pos % BS) * s_v_tok
        ks_row = blk * s_ks_blk + (pos % BS) * s_ks_tok + kvh * s_ks_kvh

        Ki = tl.load(k_i32_ptr + k_row[:, None] + offs_w[None, :],
                     mask=tmask[:, None], other=0)          # (TILE, 64) i32
        kscale = tl.load(k_scale_ptr + ks_row, mask=tmask, other=1.0)

        # Byte b of each int32 word: b=0 is the LSB on little-endian.
        K0 = ((Ki << 24) >> 24).to(tl.float16)
        K1 = ((Ki << 16) >> 24).to(tl.float16)
        K2 = ((Ki << 8) >> 24).to(tl.float16)
        K3 = (Ki >> 24).to(tl.float16)

        S = tl.dot(K0, Q0) + tl.dot(K1, Q1) + tl.dot(K2, Q2) + tl.dot(K3, Q3)
        S = S * kscale[:, None]
        S = tl.where(tmask[:, None], S, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(S, axis=0))
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        P = tl.exp(S - m_new[None, :])
        P = tl.where(tmask[:, None], P, 0.0)
        l_i = l_i * alpha + tl.sum(P, axis=0)
        a0 = a0 * alpha[:, None]
        a1 = a1 * alpha[:, None]
        a2 = a2 * alpha[:, None]
        a3 = a3 * alpha[:, None]

        Vi = tl.load(v_i32_ptr + v_row[:, None] + offs_w[None, :],
                     mask=tmask[:, None], other=0)
        vscale = tl.load(v_scale_ptr + ks_row, mask=tmask, other=1.0)

        V0 = ((Vi << 24) >> 24).to(tl.float16)
        V1 = ((Vi << 16) >> 24).to(tl.float16)
        V2 = ((Vi << 8) >> 24).to(tl.float16)
        V3 = (Vi >> 24).to(tl.float16)

        Pv = (P * vscale[:, None]).to(tl.float16)
        Pt = tl.trans(Pv)                                   # (PAD, TILE)

        a0 += tl.dot(Pt, V0)
        a1 += tl.dot(Pt, V1)
        a2 += tl.dot(Pt, V2)
        a3 += tl.dot(Pt, V3)
        m_i = m_new

    base = (chunk * KVH + kvh) * PAD
    tl.store(m_ptr + base + offs_q, m_i)
    tl.store(l_ptr + base + offs_q, l_i)
    ap = acc_ptr + (base + offs_q)[:, None] * 256
    tl.store(ap + 0 * 64 + offs_w[None, :], a0)
    tl.store(ap + 1 * 64 + offs_w[None, :], a1)
    tl.store(ap + 2 * 64 + offs_w[None, :], a2)
    tl.store(ap + 3 * 64 + offs_w[None, :], a3)


@triton.jit
def _fd_combine(
    m_ptr, l_ptr, acc_ptr, out_ptr,
    seq_ptr,
    GQA: tl.constexpr, KVH: tl.constexpr,
    CHUNK: tl.constexpr, PAD: tl.constexpr,
):
    qh = tl.program_id(0)
    kvh = qh // GQA
    qi = qh % GQA
    offs_j = tl.arange(0, 256)                  # permuted index j = b*64 + w
    d_out = (offs_j % 64) * 4 + offs_j // 64    # un-permute: d = 4w + b

    nc = (tl.load(seq_ptr) + CHUNK - 1) // CHUNK

    m_g = float("-inf")
    for c in range(0, nc):
        m_g = tl.maximum(m_g, tl.load(m_ptr + (c * KVH + kvh) * PAD + qi))
    l_g = 0.0
    o = tl.zeros((256,), tl.float32)
    for c in range(0, nc):
        idx = (c * KVH + kvh) * PAD + qi
        w = tl.exp(tl.load(m_ptr + idx) - m_g)
        l_g += tl.load(l_ptr + idx) * w
        o += tl.load(acc_ptr + idx * 256 + offs_j) * w
    tl.store(out_ptr + qh * 256 + d_out, (o / l_g).to(tl.float16))


def _as_i32_view(t: torch.Tensor) -> torch.Tensor:
    """int32 view over an int8 tensor's storage, preserving its offset.

    ``t`` must have 4-byte-aligned storage offset and 4-byte-aligned strides
    along every dim used for row addressing (the packed cache guarantees this:
    every row is padded_hs = 260 bytes and all higher strides are multiples
    of it).
    """
    assert t.dtype == torch.int8
    st = t.untyped_storage()
    f8 = torch.empty(0, dtype=torch.int8, device=t.device)
    f8.set_(st, 0, (st.nbytes(),))
    off32 = t.storage_offset() // 4
    i32 = f8.view(torch.int32)[off32:]
    return i32


# Persistent per-(gchunks, kvh) workspaces: (m, l, acc, out, qtile).
_ws_cache: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}


def _pow2_pad(gqa: int) -> int:
    pad = 1
    while pad < gqa:
        pad *= 2
    return min(pad, 16)


def fd_decode(q, k, v, k_scale_cache, v_scale_cache, block_table, seqused_k,
              softmax_scale, BS=16, CHUNK=512, TILE=32, num_warps=8):
    """Single-query decode fast path. Returns fp16 (kvh * gqa, 256).

    Args:
        q: (1, kvh * gqa, 256) fp16 — single-token query (unscaled).
        k: int8 (num_blocks, block_size, n_kv, padded_hs) — K split view.
        v: int8 (num_blocks, block_size, n_kv, padded_hs) — V split view.
        k_scale_cache / v_scale_cache: float32 (num_blocks, block_size, n_kv).
        block_table: int32 (1, max_blocks).
        seqused_k: int32 tensor — number of valid keys for sequence 0.
        softmax_scale: float — 1/sqrt(head_dim) from the attention layer.
    """
    dev = q.device
    kvh = k.shape[2]
    gqa = q.shape[1] // kvh
    assert q.shape[0] == 1, "fd_decode is q==1 only; MTP verify uses stock"
    pad = _pow2_pad(gqa)

    k_i32 = _as_i32_view(k)
    v_i32 = _as_i32_view(v)

    # Strides in int32 units (elements // 4). The packed cache layout makes
    # every row-addressing stride a multiple of 4 bytes.
    s_k = (k.stride(0) // 4, k.stride(2) // 4, k.stride(1) // 4)
    s_v = (v.stride(0) // 4, v.stride(2) // 4, v.stride(1) // 4)
    s_ks = (k_scale_cache.stride(0), k_scale_cache.stride(1),
            k_scale_cache.stride(2))

    # Byte-permuted Q tile: qtile[kvh, b, w, qi] = q_scaled[kvh*gqa+qi, 4w+b].
    qs = (q[0].float() * softmax_scale).to(torch.float16)
    qv = qs.view(kvh, gqa, 64, 4)                    # d = 4w + b

    bt = block_table[0]
    max_ctx = bt.shape[-1] * BS
    gchunks = (max_ctx + CHUNK - 1) // CHUNK

    ws = _ws_cache.get((gchunks, kvh))
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "fd_decode workspace missing during graph capture; "
                "warm-up must run outside capture first")
        m = torch.empty(gchunks * kvh * pad, dtype=torch.float32, device=dev)
        l = torch.empty_like(m)
        acc = torch.empty(gchunks * kvh * pad * 256, dtype=torch.float32,
                          device=dev)
        out = torch.empty(kvh * gqa, 256, dtype=torch.float16, device=dev)
        qtile = torch.zeros(kvh, 4, 64, pad, dtype=torch.float16, device=dev)
        ws = (m, l, acc, out, qtile)
        _ws_cache[(gchunks, kvh)] = ws
    m, l, acc, out, qtile = ws
    qtile[:, :, :, :gqa] = qv.permute(0, 3, 2, 1)   # (KVH, 4, 64, GQA)

    _fd_phase1[(gchunks, kvh)](
        qtile, k_i32, v_i32, k_scale_cache, v_scale_cache,
        bt, seqused_k, m, l, acc,
        *s_k, *s_v, *s_ks,
        BS=BS, CHUNK=CHUNK, TILE=TILE, GQA=gqa, KVH=kvh, PAD=pad, HS4=64,
        num_warps=num_warps,
    )
    _fd_combine[(kvh * gqa,)](
        m, l, acc, out, seqused_k,
        GQA=gqa, KVH=kvh, CHUNK=CHUNK, PAD=pad,
    )
    return out


__all__ = ["fd_decode"]
