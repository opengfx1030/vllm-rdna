# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 RDNA attention ops vs torch references (fp16 and bf16).

Covers the per-op pieces DeepseekV4RDNAAttention relies on:
- paged_mqa_logits_decode_rdna2 (lightning indexer decode), both page layouts
- sparse_mla_prefill_rdna2 (ragged indices + attention sink)
- inverse GPT-J RoPE rows in the activation dtype
- fused indexer q RoPE + fp8 quant with fp16 input
- wo_a weight cache in the activation dtype
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA HIP kernels", allow_module_level=True)

from vllm.models.deepseek_v4.amd.rdna import ops as rdna_ops  # noqa: E402

DEV = "cuda"
FP8 = torch.float8_e4m3fn


def _cos_sin(max_pos: int, rope_dim: int, base: float = 10000.0) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    f = torch.outer(torch.arange(max_pos).float(), inv)
    return torch.cat([f.cos(), f.sin()], -1).contiguous()


# ── paged MQA logits ───────────────────────────────────────────────────────
def _make_indexer_cache(num_pages, block_size, d, block_flat, g):
    vals = (
        (torch.randn(num_pages, block_size, d, generator=g) * 150)
        .clamp(-448, 448)
        .to(FP8)
    )
    scales = torch.rand(num_pages, block_size, generator=g) * 0.5 + 0.25
    cache = torch.empty(num_pages, block_size * (d + 4), dtype=torch.uint8)
    if block_flat:
        cache[:, : block_size * d] = vals.view(torch.uint8).reshape(num_pages, -1)
        cache[:, block_size * d :] = scales.view(torch.uint8).reshape(num_pages, -1)
    else:
        per = cache.view(num_pages, block_size, d + 4)
        per[..., :d] = vals.view(torch.uint8)
        per[..., d:] = scales.unsqueeze(-1).view(torch.uint8)
    return cache.view(num_pages, block_size, 1, d + 4), vals.float(), scales


@pytest.mark.skipif(not rdna_ops.has_paged_mqa_logits(), reason="op not built")
@pytest.mark.parametrize("block_flat", [True, False])
@pytest.mark.parametrize("batch,max_len", [(1, 300), (4, 1000)])
def test_paged_mqa_logits(block_flat, batch, max_len):
    g = torch.Generator().manual_seed(batch)
    h, d, block_size = 64, 128, 64
    max_blocks = (max_len + block_size - 1) // block_size
    num_pages = batch * max_blocks + 2
    cache, vals, scales = _make_indexer_cache(num_pages, block_size, d, block_flat, g)
    # Up to the E4M3 max (448): the indexer quantizes q so amax maps there.
    q = (torch.randn(batch, 1, h, d, generator=g) * 150).clamp(-448, 448).to(FP8)
    w = torch.randn(batch, h, generator=g)
    ctx = torch.randint(1, max_len + 1, (batch,), generator=g, dtype=torch.int32)
    bt = torch.randperm(num_pages, generator=g)[: batch * max_blocks]
    bt = bt.view(batch, max_blocks).to(torch.int32)

    out = rdna_ops.paged_mqa_logits_decode(
        q.to(DEV),
        cache.to(DEV),
        w.to(DEV),
        ctx.to(DEV),
        bt.to(DEV),
        max_len,
        block_flat,
    ).cpu()

    ref = torch.full((batch, max_len), float("-inf"))
    for b in range(batch):
        n = int(ctx[b])
        pos = torch.arange(n)
        pages, slots = bt[b, pos // block_size].long(), pos % block_size
        k = vals[pages, slots] * scales[pages, slots].unsqueeze(-1)  # [n, d]
        s = torch.relu(q[b, 0].float() @ k.T) * w[b].unsqueeze(-1)  # [h, n]
        ref[b, :n] = s.sum(0)
    torch.testing.assert_close(out, ref, atol=1.0, rtol=2e-3)


# ── sparse MLA prefill ─────────────────────────────────────────────────────
@pytest.mark.skipif(not rdna_ops.has_sparse_mla_prefill(), reason="op not built")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_sink", [True, False])
def test_sparse_mla_prefill(dtype, with_sink):
    g = torch.Generator().manual_seed(0)
    t, h, d, num_kv = 9, 16, 512, 300
    q = (torch.randn(t, h, d, generator=g) * 0.1).to(dtype)
    kv = torch.randn(num_kv, d, generator=g).to(dtype)
    lens = torch.randint(0, 80, (t,), generator=g)
    lens[0] = 0  # empty row -> zeros (or sink-only)
    rows = [torch.randperm(num_kv, generator=g)[: int(n)] for n in lens]
    rows[1][0] = -1  # invalid entry is skipped
    indices = torch.cat(rows).to(torch.int32)
    indptr = torch.zeros(t + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(lens, 0)
    sink = torch.randn(h, generator=g) if with_sink else torch.empty(0)
    scale = d**-0.5

    out = torch.empty(t, h, d, dtype=dtype, device=DEV)
    rdna_ops.sparse_mla_prefill(
        q.to(DEV),
        kv.to(DEV),
        indices.to(DEV),
        indptr.to(DEV),
        num_kv,
        scale,
        sink.to(DEV),
        out,
    )
    ref = torch.zeros(t, h, d)
    for i in range(t):
        idx = rows[i][rows[i] >= 0].long()
        k = kv[idx].float()
        s = q[i].float() @ k.T * scale  # [h, n]
        if with_sink:
            s = torch.cat([s, sink.unsqueeze(-1)], -1)
            p = torch.softmax(s, -1)[:, :-1]
        elif idx.numel() == 0:
            continue
        else:
            p = torch.softmax(s, -1)
        ref[i] = p @ k
    torch.testing.assert_close(out.cpu().float(), ref, atol=2e-2, rtol=2e-2)


# ── inverse RoPE rows ──────────────────────────────────────────────────────
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_inverse_rope_rows_keeps_dtype(dtype):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inverse_rope_rows_

    g = torch.Generator().manual_seed(1)
    t, h, d, rope = 7, 16, 512, 64
    o = torch.randn(t, h, d, generator=g).to(dtype)
    pos = torch.randint(0, 2048, (t,), generator=g)
    cs = _cos_sin(2048, rope)
    out = o.to(DEV)
    rocm_inverse_rope_rows_(out, pos.to(DEV), cs.to(DEV), rope)
    assert out.dtype == dtype

    ref = o.float().clone()
    c, s = cs[pos, : rope // 2].unsqueeze(1), cs[pos, rope // 2 :].unsqueeze(1)
    a, b = ref[..., d - rope :: 2].clone(), ref[..., d - rope + 1 :: 2].clone()
    ref[..., d - rope :: 2] = a * c + b * s
    ref[..., d - rope + 1 :: 2] = b * c - a * s
    tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(out.cpu().float(), ref, atol=tol, rtol=tol)


# ── fused indexer q RoPE + fp8 quant ───────────────────────────────────────
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_indexer_q_rope_quant(dtype):
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
        fused_indexer_q_rope_quant,
    )

    g = torch.Generator().manual_seed(2)
    t, h, d, rope = 5, 64, 128, 64
    q = torch.randn(t, h, d, generator=g).to(dtype)
    w = torch.randn(t, h, generator=g).to(dtype)
    pos = torch.randint(0, 4096, (t,), generator=g)
    cs = _cos_sin(4096, rope)
    sm, hs = d**-0.5, h**-0.5
    q_fp8, w_out = fused_indexer_q_rope_quant(
        pos.to(DEV), q.to(DEV), cs.to(DEV), w.to(DEV), sm, hs
    )

    x = q.float()
    c, s = cs[pos, : rope // 2].unsqueeze(1), cs[pos, rope // 2 :].unsqueeze(1)
    e, o = x[..., d - rope :: 2].clone(), x[..., d - rope + 1 :: 2].clone()
    x[..., d - rope :: 2] = (e * c - o * s).to(dtype).float()
    x[..., d - rope + 1 :: 2] = (o * c + e * s).to(dtype).float()
    amax = x.abs().amax(-1).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    ref_q = x / scale.unsqueeze(-1)
    ref_w = w.float() * scale * sm * hs

    torch.testing.assert_close(w_out.cpu(), ref_w, atol=1e-5, rtol=1e-3)
    # fp8 quantization error dominates; compare in the dequantized domain.
    torch.testing.assert_close(
        q_fp8.cpu().float(), ref_q.to(FP8).float(), atol=0.07, rtol=0.07
    )


# ── wo_a cache in the activation dtype ─────────────────────────────────────
def test_wo_a_cache_fp16():
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _get_cached_wo_a_bf16

    groups, rank, hidden = 2, 128, 256
    wo_a = torch.nn.Module()
    wo_a.weight = torch.nn.Parameter(
        torch.randn(groups * rank, hidden), requires_grad=False
    )
    w16 = _get_cached_wo_a_bf16(wo_a, groups, rank, hidden, dtype=torch.float16)
    wbf = _get_cached_wo_a_bf16(wo_a, groups, rank, hidden)
    assert w16.dtype == torch.float16 and wbf.dtype == torch.bfloat16
    assert _get_cached_wo_a_bf16(wo_a, groups, rank, hidden, dtype=torch.float16) is w16
    torch.testing.assert_close(w16.float(), wbf.float(), atol=2e-2, rtol=1e-2)


# ── prefill MQA logits (memory-bounded head chunks) ────────────────────────
@pytest.mark.parametrize("chunk_bytes", [1 << 30, 64 * 1024])
def test_rdna_fp8_mqa_logits_matches_torch(chunk_bytes):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rdna_fp8_mqa_logits,
        fp8_mqa_logits_torch,
    )

    g = torch.Generator().manual_seed(4)
    m, h, d, n = 37, 64, 128, 300
    q = torch.randn(m, h, d, generator=g).to(FP8).to(DEV)
    k = torch.randn(n, d, generator=g).to(FP8).to(DEV)
    scale = (torch.rand(n, 1, generator=g) + 0.5).to(DEV)
    w = torch.randn(m, h, generator=g).to(DEV)
    ks = torch.randint(0, 50, (m,), generator=g, dtype=torch.int32).to(DEV)
    ke = (
        ks + torch.randint(1, 250, (m,), generator=g, dtype=torch.int32).to(DEV)
    ).clamp(max=n)
    out = _rdna_fp8_mqa_logits(q, (k, scale), w, ks, ke, max_chunk_bytes=chunk_bytes)
    # fp32 reference (the torch fallback rounds scores through bf16).
    score = torch.einsum("mhd,nd->hmn", q.float().cpu(), k.float().cpu())
    score = (score * scale.cpu().reshape(-1)).relu()
    ref = (score * w.cpu().t().unsqueeze(-1)).sum(0)
    pos = torch.arange(n)[None, :]
    mask = (pos >= ks.cpu()[:, None]) & (pos < ke.cpu()[:, None])
    ref = ref.masked_fill(~mask, float("-inf"))
    finite = torch.isfinite(ref)
    assert torch.equal(finite, torch.isfinite(out.cpu()))
    torch.testing.assert_close(out.cpu()[finite], ref[finite], atol=1e-2, rtol=1e-4)
    # And it agrees with the generic torch fallback to bf16 rounding.
    alt = fp8_mqa_logits_torch(q, (k, scale), w, ks, ke).cpu()
    torch.testing.assert_close(out.cpu()[finite], alt[finite], atol=1.0, rtol=5e-2)


# ── mHC pre / post vs the torch reference ──────────────────────────────────
@pytest.mark.skipif(not rdna_ops.has_mhc(), reason="op not built")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("t", [1, 5, 300])
def test_mhc_pre_post_match_torch(dtype, t):
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch

    g = torch.Generator().manual_seed(t)
    hc, hidden = 4, 4096
    res = (torch.randn(t, hc, hidden, generator=g) * 3).to(dtype).to(DEV)
    fn = (torch.randn(2 * hc + hc * hc, hc * hidden, generator=g) * 0.02).to(DEV)
    scale = (torch.rand(3, generator=g) + 0.5).to(DEV)
    base = (torch.randn(2 * hc + hc * hc, generator=g) * 0.3).to(DEV)
    args = (1e-6, 1e-6, 1e-6, 2.0, 20)
    post, comb, li = rdna_ops.mhc_pre(res, fn, scale, base, *args)
    rpost, rcomb, rli = mhc_pre_torch(res, fn, scale, base, *args)
    torch.testing.assert_close(post, rpost, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(comb, rcomb, atol=1e-5, rtol=1e-4)
    tol = 2e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(li.float(), rli.float(), atol=tol, rtol=tol)

    x = torch.randn(t, hidden, generator=g).to(dtype).to(DEV)
    out = rdna_ops.mhc_post(x, res, post, comb)
    ref = mhc_post_torch(x, res, post, comb)
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=tol)
