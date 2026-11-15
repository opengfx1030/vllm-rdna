# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unfused (GEMM + softmax) paged prefill attention of RDNA_ATTN.

CPU tests (float32, run everywhere) check the algorithm against a float64
reference: causal mask with a query offset into the paged cache (prefix hits),
GQA groups of the 27B at TP=2 (12q/2kv) and TP=1 (24q/4kv), head_dim 256 and 128,
partial last pages, shuffled block tables, several query blocks per sequence,
the scratch planner, and the K/V gather against the cache layouts the backend
uses (K packed by x, V unpacked).

GPU tests (gfx103x only, skipped elsewhere) compare with the production flash
kernel ``fa_rdna2_prefill_paged_varlen_gqa`` in fp16 and tabulate the error of
both against float64 (``-s`` prints the table), and run the backend routing
(``RdnaAttentionImpl._forward_prefill``) on a mixed batch.

    pytest -s tests/kernels/attention/test_fa_rdna2_unfused_prefill.py
"""

import math

import pytest
import torch

from vllm.v1.attention.ops import rdna_unfused_prefill as up

X = 8


def _on_gfx103() -> bool:
    try:
        return torch.cuda.is_available() and "gfx103" in (
            torch.cuda.get_device_properties(0).gcnArchName
        )
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _on_gfx103(), reason="Requires AMD RDNA2 (gfx103x)")


def _reference(q, k, v, scale):
    """float64 causal attention; q rows sit at the end of the k/v sequence."""
    n, h_q, _ = q.shape
    s_len, h_kv, _ = k.shape
    g = h_q // h_kv
    qd, kd, vd = q.double(), k.double(), v.double()
    pos = torch.arange(n, device=q.device)[:, None] + (s_len - n)
    mask = torch.arange(s_len, device=q.device)[None, :] <= pos
    out = torch.empty(n, h_q, q.shape[2], dtype=torch.float64, device=q.device)
    for h in range(h_q):
        s = (qd[:, h] @ kd[:, h // g].T) * scale
        s = s.masked_fill(~mask, float("-inf"))
        out[:, h] = torch.softmax(s, -1) @ vd[:, h // g]
    return out


def _v5d(value_cache4, x=X):
    nb, h, d, bs = value_cache4.shape
    return value_cache4.view(nb, h, d // x, x, bs).permute(0, 1, 2, 4, 3)


def _paged(ks, vs, h_kv, d, bs, dtype, device, seed):
    """Dense per-sequence K/V -> shuffled paged caches (K packed, V unpacked)."""
    g = torch.Generator().manual_seed(seed)
    need = [math.ceil(k.shape[0] / bs) for k in ks]
    nb = sum(need) + 3
    perm = torch.randperm(nb, generator=g).tolist()
    kc = torch.randn(nb, h_kv, d // X, bs, X, generator=g).to(dtype)  # stale slots
    vc4 = torch.randn(nb, h_kv, d, bs, generator=g).to(dtype)
    tables, nxt = [], 0
    for k, v in zip(ks, vs):
        blocks = perm[nxt : nxt + math.ceil(k.shape[0] / bs)]
        nxt += len(blocks)
        for i, b in enumerate(blocks):
            t0, t1 = i * bs, min((i + 1) * bs, k.shape[0])
            kc[b, :, :, : t1 - t0, :] = (
                k[t0:t1].view(t1 - t0, h_kv, d // X, X).permute(1, 2, 0, 3).to(dtype)
            )
            vc4[b, :, :, : t1 - t0] = v[t0:t1].permute(1, 2, 0).to(dtype)
        tables.append(blocks)
    width = max(len(t) for t in tables)
    table = torch.zeros(len(tables), width, dtype=torch.int32)
    for i, t in enumerate(tables):
        table[i, : len(t)] = torch.tensor(t, dtype=torch.int32)
    return kc.to(device), _v5d(vc4.to(device)), table.to(device)


def _dense(s_len, h_kv, d, seed, dtype, device, spread=1.0):
    g = torch.Generator().manual_seed(seed)
    k = (torch.randn(s_len, h_kv, d, generator=g) * spread).to(dtype).to(device)
    v = torch.randn(s_len, h_kv, d, generator=g).to(dtype).to(device)
    return k, v


GEOMETRIES = [  # (H_q, H_kv, D): 27B per rank at TP2 and TP1, D=128 GQA
    (12, 2, 256),
    (24, 4, 256),
    (16, 4, 128),
]


@pytest.mark.parametrize("h_q,h_kv,d", GEOMETRIES)
@pytest.mark.parametrize(
    "n,ctx,bs,qblock",
    [
        (70, 0, 16, 32),  # first chunk, several blocks, partial last block
        (130, 1000, 64, 64),  # prefix-cache hit: ctx > 0 at the chunk start
        (257, 5, 16, 128),  # tiny context, uneven last block
        (96, 4096 - 96, 1024, 96),  # one block per call, kv_len = 4 pages
    ],
)
def test_matches_float64_cpu(h_q, h_kv, d, n, ctx, bs, qblock):
    scale = d**-0.5
    s_len = n + ctx
    k, v = _dense(s_len, h_kv, d, 1, torch.float32, "cpu")
    q = torch.randn(n, h_q, d, generator=torch.Generator().manual_seed(2))
    kc, vc, table = _paged([k], [v], h_kv, d, bs, torch.float32, "cpu", 3)
    out = torch.empty_like(q)
    up.unfused_paged_prefill(q, kc, vc, table[0], s_len, scale, out, qblock)
    ref = _reference(q, k, v, scale)
    assert (out.double() - ref).abs().max().item() < 2e-5


def test_gather_matches_dense_layouts_cpu():
    h_kv, d, bs = 2, 256, 16
    k, v = _dense(70, h_kv, d, 4, torch.float32, "cpu")
    kc, vc, table = _paged([k], [v], h_kv, d, bs, torch.float32, "cpu", 5)
    gk = up.gather_kv(kc, table[0], 70)
    gv = up.gather_kv(vc, table[0], 70)
    assert torch.equal(gk, k.permute(1, 0, 2))
    assert torch.equal(gv, v.permute(1, 0, 2))


def test_softmax_in_place_matches_cpu():
    for width in (7, 64, 1000, 1025, 4097):
        s = torch.randn(48, width)
        ref = torch.softmax(s, -1)
        torch.ops.aten._softmax.out(s, -1, False, out=s)
        assert torch.equal(s, ref)


@pytest.mark.parametrize("kv_len", [64, 4096, 32768, 131072, 262144])
@pytest.mark.parametrize("h_kv,group,d", [(2, 6, 256), (4, 6, 256)])
def test_planner_bounds_scratch(kv_len, h_kv, group, d):
    budget = 768 << 20
    n = min(2048, kv_len)
    qb = up.plan_query_block(n, kv_len, group, h_kv, d, budget)
    if qb:
        assert qb == min(n, 512) or qb % 32 == 0
        assert qb >= 32
        assert up.scratch_bytes_required(n, kv_len, qb, group, h_kv, d) <= budget
        # the next 32-row step would not fit unless the cap was reached
        if qb < min(n, 512):
            assert (
                up.scratch_bytes_required(n, kv_len, qb + 32, group, h_kv, d) > budget
            )
    else:  # K and V alone (or a 32-row tile) exceed the budget
        assert up.scratch_bytes_required(n, kv_len, 32, group, h_kv, d) > budget
    assert up.plan_query_block(n, kv_len, group, h_kv, d, 1 << 20) == 0 or kv_len < 4096


def test_planner_rejects_short_kv():
    assert up.plan_query_block(100, 50, 6, 2, 256, 1 << 30) == 0


def test_config_env(monkeypatch):
    monkeypatch.delenv("VLLM_FA_RDNA2_UNFUSED_PREFILL", raising=False)
    assert up.config_from_env() is None
    monkeypatch.setenv("VLLM_FA_RDNA2_UNFUSED_PREFILL", "1")
    cfg = up.config_from_env()
    assert cfg is not None and cfg.min_q == 64 and cfg.scratch_bytes == 768 << 20


# --------------------------------------------------------------------------- GPU
GPU_CASES = [  # (H_q, H_kv, D, [(n, ctx)...], bs, spread)
    (12, 2, 256, [(300, 0)], 16, 1.0),
    (12, 2, 256, [(512, 1536)], 1024, 1.0),
    (12, 2, 256, [(300, 0), (200, 1500)], 16, 1.0),  # varlen, prefix hit
    (24, 4, 256, [(257, 700)], 16, 1.0),  # TP1 geometry, partial pages
    (12, 2, 256, [(256, 2048)], 16, 4.0),  # large logits (|score| up to ~30)
]


def _flash(q, kc, vc, table, lens, bs, scale):
    from vllm.v1.attention.ops import fa_rdna2_backend as fa

    cu = torch.tensor(
        [0] + [sum(n for n, _ in lens[: i + 1]) for i in range(len(lens))],
        dtype=torch.int32,
        device=q.device,
    )
    seq = torch.tensor([n + c for n, c in lens], dtype=torch.int32, device=q.device)
    out = torch.empty_like(q)
    fa.fa_rdna2_prefill_paged_varlen_gqa(
        q,
        kc,
        vc,
        table,
        cu,
        seq,
        bs,
        causal=True,
        sliding_window=0,
        scale=scale,
        out=out,
    )
    return out, cu, seq


@needs_gpu
@pytest.mark.parametrize("h_q,h_kv,d,lens,bs,spread", GPU_CASES)
def test_gpu_matches_flash_and_float64(h_q, h_kv, d, lens, bs, spread):
    dev, dt = "cuda", torch.float16
    scale = d**-0.5
    ks, vs, qs = [], [], []
    for i, (n, ctx) in enumerate(lens):
        k, v = _dense(n + ctx, h_kv, d, 10 + i, dt, dev, spread)
        ks.append(k)
        vs.append(v)
        qs.append(
            (
                torch.randn(n, h_q, d, generator=torch.Generator().manual_seed(20 + i))
                * spread
            )
            .to(dt)
            .to(dev)
        )
    q = torch.cat(qs)
    kc, vc, table = _paged(ks, vs, h_kv, d, bs, dt, dev, 7)
    flash, _, _ = _flash(q, kc, vc, table, lens, bs, scale)
    out = torch.empty_like(q)
    tok = 0
    for i, (n, ctx) in enumerate(lens):
        plan = up.plan_query_block(n, n + ctx, h_q // h_kv, h_kv, d, 768 << 20)
        assert plan > 0
        up.unfused_paged_prefill(
            q[tok : tok + n], kc, vc, table[i], n + ctx, scale, out[tok : tok + n], plan
        )
        tok += n
    tok = 0
    for i, (n, ctx) in enumerate(lens):
        ref = _reference(q[tok : tok + n], ks[i], vs[i], scale)
        e_f = (flash[tok : tok + n].double() - ref).abs().max().item()
        e_u = (out[tok : tok + n].double() - ref).abs().max().item()
        r_f = ((flash[tok : tok + n].double() - ref).norm() / ref.norm()).item()
        r_u = ((out[tok : tok + n].double() - ref).norm() / ref.norm()).item()
        print(
            f"H_q/H_kv/D {h_q}/{h_kv}/{d} n={n} ctx={ctx} bs={bs} spread={spread}: "
            f"max|err| flash {e_f:.2e} unfused {e_u:.2e}; rel L2 flash {r_f:.2e} "
            f"unfused {r_u:.2e}"
        )
        # fp16 outputs: the unfused scores are rounded once to fp16, so allow a
        # few times the flash kernel's own error plus the output rounding.
        assert r_u < max(4 * r_f, 2e-3) * (1.0 if spread == 1.0 else 5.0)
        tok += n


@needs_gpu
def test_gpu_softmax_in_place_bit_identical():
    for width in (64, 1000, 1024, 1025, 4096, 16384):
        s = torch.randn(3072, width, device="cuda", dtype=torch.float16)
        ref = torch.softmax(s, -1)
        torch.ops.aten._softmax.out(s, -1, False, out=s)
        assert torch.equal(s, ref)


@needs_gpu
def test_gpu_backend_routes_mixed_batch(monkeypatch):
    """Rows >= min_q take the GEMM path, short rows the flash kernel, in order."""
    monkeypatch.setenv("VLLM_FA_RDNA2_UNFUSED_PREFILL", "1")
    from vllm.v1.attention.backends.rdna_attn import RdnaAttentionImpl

    h_q, h_kv, d, bs = 12, 2, 256, 16
    scale = d**-0.5
    lens = [(3, 40), (300, 0), (1, 90), (130, 700), (2, 11)]  # (n, ctx)
    dev, dt = "cuda", torch.float16
    ks, vs, qs = [], [], []
    for i, (n, ctx) in enumerate(lens):
        k, v = _dense(n + ctx, h_kv, d, 30 + i, dt, dev)
        ks.append(k)
        vs.append(v)
        qs.append(
            torch.randn(n, h_q, d, generator=torch.Generator().manual_seed(40 + i))
            .to(dt)
            .to(dev)
        )
    q = torch.cat(qs)
    kc, vc, table = _paged(ks, vs, h_kv, d, bs, dt, dev, 9)
    flash, cu, seq = _flash(q, kc, vc, table, lens, bs, scale)
    impl = RdnaAttentionImpl(h_q, d, scale, h_kv, None, None, "auto")
    assert impl._unfused is not None
    dst = torch.zeros_like(q)
    host = ([n for n, _ in lens], [n + c for n, c in lens])
    impl._forward_prefill(
        q, kc, vc, table, cu, seq, max(host[1]), bs, 0, dst, host_rows=host
    )
    tok = 0
    for i, (n, ctx) in enumerate(lens):
        ref = _reference(q[tok : tok + n], ks[i], vs[i], scale)
        err = ((dst[tok : tok + n].double() - ref).norm() / ref.norm()).item()
        assert err < 2e-3, (i, n, ctx, err)
        if n < 64:  # short rows went through the flash kernel: identical output
            assert torch.equal(dst[tok : tok + n], flash[tok : tok + n])
        tok += n
