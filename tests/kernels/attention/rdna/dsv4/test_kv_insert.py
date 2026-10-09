# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""_rocm_C::dsv4_qnorm_rope_kv_insert_rdna vs a pure-torch reference.

The reference mirrors upstream's
``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`` math for the
fp8_ds_mla V4 row and runs on the CPU (IEEE fp32, RNE casts). The cache must
match byte for byte; q_out within fp16/bf16 rounding of the RMSNorm.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA HIP kernel", allow_module_level=True)

from vllm.models.deepseek_v4.amd.rdna import ops as rdna_ops  # noqa: E402

if not rdna_ops.has_qnorm_rope_kv_insert():
    pytest.skip(
        "_rocm_C::dsv4_qnorm_rope_kv_insert_rdna not built", allow_module_level=True
    )

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
QUANT_BLOCK = 64
TOKEN_DATA_BYTES = NOPE_DIM + ROPE_DIM * 2  # 576
SCALE_BYTES = NOPE_DIM // QUANT_BLOCK + 1  # 8
ROW_BYTES = TOKEN_DATA_BYTES + SCALE_BYTES  # 584
FP8_MAX = 448.0
EPS = 1e-6
MAX_POS = 4096


def _cos_sin_cache(max_pos: int = MAX_POS, base: float = 10000.0) -> torch.Tensor:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM)
    )
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def _rope_gptj(x: torch.Tensor, pos: torch.Tensor, cos_sin: torch.Tensor):
    """GPT-J RoPE on dims [448, 512) of fp32 x [..., 512]; pos broadcasts."""
    out = x.clone()
    cs = cos_sin[pos]  # [N, 64]
    cos, sin = cs[:, : ROPE_DIM // 2], cs[:, ROPE_DIM // 2 :]
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    xe = x[..., NOPE_DIM::2]
    xo = x[..., NOPE_DIM + 1 :: 2]
    out[..., NOPE_DIM::2] = xe * cos - xo * sin
    out[..., NOPE_DIM + 1 :: 2] = xe * sin + xo * cos
    return out


def _ceil_log2(x: torch.Tensor) -> torch.Tensor:
    mant, exp = torch.frexp(x)  # x = mant * 2^exp, mant in [0.5, 1)
    return torch.where(mant == 0.5, exp - 1, exp)


def _reference(
    q, kv, cache, slots, positions, cos_sin, padded, block_size, norm, q_rope
):
    """CPU reference. Returns (q_out, cache_after)."""
    dtype = q.dtype
    q, kv, cache = q.cpu(), kv.cpu(), cache.cpu().clone()
    slots, positions, cos_sin = slots.cpu(), positions.cpu(), cos_sin.cpu()
    n, hq, _ = q.shape

    qf = q.float()
    if norm:
        qf = qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + EPS)
    if q_rope:
        qf = _rope_gptj(qf, positions, cos_sin)
    q_out = torch.zeros(n, padded, HEAD_DIM, dtype=dtype)
    if padded:
        q_out[:, :hq] = qf.to(dtype)
    else:
        q_out = torch.empty(0, dtype=dtype)

    kf = _rope_gptj(kv.float(), positions, cos_sin)
    nope = kf[:, :NOPE_DIM].to(dtype).float().view(n, -1, QUANT_BLOCK)
    absmax = nope.abs().amax(-1).clamp_min(1e-4)
    exponent = _ceil_log2(absmax / FP8_MAX)  # [n, 7] int
    scaled = nope * torch.pow(2.0, -exponent.float()).unsqueeze(-1)
    fp8 = scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    fp8_bytes = fp8.view(torch.uint8).view(n, NOPE_DIM)
    rope_bytes = kf[:, NOPE_DIM:].to(torch.bfloat16).view(torch.uint8)
    scale_bytes = (exponent + 127).clamp(0, 255).to(torch.uint8)

    flat = cache.view(cache.shape[0], -1)
    for t in range(slots.numel()):
        s = int(slots[t])
        if s < 0:
            continue
        blk, off = divmod(s, block_size)
        row = flat[blk]
        base = off * TOKEN_DATA_BYTES
        row[base : base + NOPE_DIM] = fp8_bytes[t]
        row[base + NOPE_DIM : base + TOKEN_DATA_BYTES] = rope_bytes[t]
        sbase = block_size * TOKEN_DATA_BYTES + off * SCALE_BYTES
        row[sbase : sbase + SCALE_BYTES - 1] = scale_bytes[t]
        row[sbase + SCALE_BYTES - 1] = 0
    return q_out, cache


def _make_inputs(
    dtype, n, hq, block_size, num_blocks, n_insert, cache_3d, generic, seed
):
    g = torch.Generator().manual_seed(seed)
    dev = "cuda"
    if generic:  # arbitrary values of the activation dtype
        q = (torch.randn(n, hq, HEAD_DIM, generator=g) * 2).to(dtype)
        kv = (torch.randn(n, HEAD_DIM, generator=g) * 2).to(dtype)
    else:  # bf16-representable (and fp16-representable at this scale)
        q = (torch.randn(n, hq, HEAD_DIM, generator=g) * 2).to(torch.bfloat16)
        kv = (torch.randn(n, HEAD_DIM, generator=g) * 2).to(torch.bfloat16)
        q, kv = q.to(dtype), kv.to(dtype)
    # Spread magnitudes across quant blocks to exercise many UE8M0 exponents.
    kv[:, :NOPE_DIM] *= (
        torch.logspace(-3, 3, NOPE_DIM // QUANT_BLOCK)
        .repeat_interleave(QUANT_BLOCK)
        .to(dtype)
    )
    positions = torch.randint(0, MAX_POS, (n,), generator=g, dtype=torch.int64)
    total = num_blocks * block_size
    slots = torch.randperm(total, generator=g)[:n_insert].to(torch.int64)
    if n_insert > 2:
        slots[1] = -1  # skipped (padding / unscheduled)
    shape = (
        (num_blocks, block_size, ROW_BYTES)
        if cache_3d
        else (num_blocks, block_size * ROW_BYTES)
    )
    cache = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
    return (
        q.to(dev),
        kv.to(dev),
        cache.to(dev),
        slots.to(dev),
        positions.to(dev),
        _cos_sin_cache().to(dev),
    )


def _q_tol(dtype):
    return (
        dict(atol=2e-2, rtol=1.6e-2)
        if dtype == torch.bfloat16
        else dict(atol=4e-3, rtol=2e-3)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "n,hq,padded,n_insert",
    [
        (1, 16, 64, 1),  # decode, DSV4-Flash TP=4 (16 live heads -> 64)
        (37, 16, 64, 37),
        (130, 8, 8, 130),  # no padding heads
        (64, 16, 32, 50),  # DP padding: fewer slots than q rows
        (19, 16, 0, 19),  # KV insert only
    ],
)
@pytest.mark.parametrize("block_size", [64, 16])
@pytest.mark.parametrize("cache_3d", [False, True])
@pytest.mark.parametrize("generic", [False, True])
def test_kv_insert_matches_reference(
    dtype, n, hq, padded, n_insert, block_size, cache_3d, generic
):
    num_blocks = (n_insert + block_size - 1) // block_size + 3
    q, kv, cache, slots, positions, cos_sin = _make_inputs(
        dtype, n, hq, block_size, num_blocks, n_insert, cache_3d, generic, seed=n
    )
    q_ref, cache_ref = _reference(
        q, kv, cache, slots, positions, cos_sin, padded, block_size, True, True
    )
    q_out = rdna_ops.qnorm_rope_kv_insert(
        q, kv, cache, slots, positions, cos_sin, padded, EPS, block_size
    )
    torch.accelerator.synchronize()

    mismatch = (cache.cpu() != cache_ref).nonzero()
    assert mismatch.numel() == 0, f"{mismatch.shape[0]} cache bytes differ"
    assert q_out.dtype == dtype
    if padded == 0:
        assert q_out.numel() == 0
    else:
        assert q_out.shape == (n, padded, HEAD_DIM)
        torch.testing.assert_close(q_out.cpu().float(), q_ref.float(), **_q_tol(dtype))
        assert torch.all(q_out[:, hq:] == 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("norm,q_rope", [(False, True), (True, False), (False, False)])
def test_kv_insert_q_flags(dtype, norm, q_rope):
    n, hq, padded, block_size = 9, 16, 64, 64
    q, kv, cache, slots, positions, cos_sin = _make_inputs(
        dtype, n, hq, block_size, 2, n, False, True, seed=7
    )
    q_ref, cache_ref = _reference(
        q, kv, cache, slots, positions, cos_sin, padded, block_size, norm, q_rope
    )
    q_out = rdna_ops.qnorm_rope_kv_insert(
        q, kv, cache, slots, positions, cos_sin, padded, EPS, block_size, norm, q_rope
    )
    torch.accelerator.synchronize()
    assert torch.equal(cache.cpu(), cache_ref)
    torch.testing.assert_close(q_out.cpu().float(), q_ref.float(), **_q_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kv_insert_rope_stored_as_bf16(dtype):
    """The RoPE half of the row decodes as bf16 for fp16 inputs too."""
    n, block_size = 4, 16
    q, kv, cache, slots, positions, cos_sin = _make_inputs(
        dtype, n, 16, block_size, 1, n, True, True, seed=3
    )
    rdna_ops.qnorm_rope_kv_insert(
        q, kv, cache, slots, positions, cos_sin, 0, EPS, block_size
    )
    k_rope = _rope_gptj(kv.float().cpu(), positions.cpu(), cos_sin.cpu())[:, NOPE_DIM:]
    for t in range(n):
        if int(slots[t]) < 0:
            continue
        blk, off = divmod(int(slots[t]), block_size)
        flat = cache[blk].reshape(-1).cpu()
        base = off * TOKEN_DATA_BYTES
        rope = flat[base + NOPE_DIM : base + TOKEN_DATA_BYTES].view(torch.bfloat16)
        torch.testing.assert_close(rope.float(), k_rope[t], atol=0, rtol=2**-8)


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"),
    reason="upstream _C op not built",
)
def test_kv_insert_matches_upstream_bf16():
    """bf16 inputs: same cache row as upstream's fused op on this device."""
    dtype, n, hq, padded, block_size = torch.bfloat16, 33, 16, 64, 64
    q, kv, cache, slots, positions, cos_sin = _make_inputs(
        dtype, n, hq, block_size, 2, n, False, False, seed=5
    )
    ours = cache.clone()
    q_ours = rdna_ops.qnorm_rope_kv_insert(
        q, kv, ours, slots, positions, cos_sin, padded, EPS, block_size
    )
    q_up = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q, kv, cache, slots, positions, cos_sin, padded, EPS, block_size
    )
    torch.accelerator.synchronize()
    assert torch.equal(ours, cache)
    torch.testing.assert_close(q_ours, q_up, atol=1e-2, rtol=1e-2)


def test_kv_insert_cuda_graph_capture():
    """Captured replay writes the same rows as eager (no host syncs/allocs)."""
    dtype, n, hq, padded, block_size = torch.float16, 8, 16, 64, 64
    q, kv, cache, slots, positions, cos_sin = _make_inputs(
        dtype, n, hq, block_size, 2, n, False, True, seed=11
    )
    eager_cache = cache.clone()
    q_eager = rdna_ops.qnorm_rope_kv_insert(
        q, kv, eager_cache, slots, positions, cos_sin, padded, EPS, block_size
    )
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # warm-up outside capture
        rdna_ops.qnorm_rope_kv_insert(
            q, kv, cache.clone(), slots, positions, cos_sin, padded, EPS, block_size
        )
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        q_graph = rdna_ops.qnorm_rope_kv_insert(
            q, kv, cache, slots, positions, cos_sin, padded, EPS, block_size
        )
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(cache, eager_cache)
    assert torch.equal(q_graph, q_eager)


def test_kv_insert_fake_shapes():
    q = torch.empty(5, 16, HEAD_DIM, dtype=torch.float16, device="meta")
    kv = torch.empty(5, HEAD_DIM, dtype=torch.float16, device="meta")
    cache = torch.empty(2, 64 * ROW_BYTES, dtype=torch.uint8, device="meta")
    idx = torch.empty(5, dtype=torch.int64, device="meta")
    cs = torch.empty(MAX_POS, ROPE_DIM, dtype=torch.float32, device="meta")
    out = torch.ops._rocm_C.dsv4_qnorm_rope_kv_insert_rdna(
        q, kv, cache, idx, idx, cs, 64, EPS, 64
    )
    assert out.shape == (5, 64, HEAD_DIM) and out.dtype == torch.float16


def _dequant_rows(cache: torch.Tensor, slots: torch.Tensor, block_size: int):
    """Decode fp8_ds_mla V4 rows back to fp32 [len(slots), 512] (CPU)."""
    flat = cache.cpu().view(cache.shape[0], -1)
    rows = []
    for s in slots.cpu().tolist():
        blk, off = divmod(s, block_size)
        row = flat[blk]
        base = off * TOKEN_DATA_BYTES
        nope = row[base : base + NOPE_DIM].view(torch.float8_e4m3fn).float()
        sbase = block_size * TOKEN_DATA_BYTES + off * SCALE_BYTES
        exps = row[sbase : sbase + NOPE_DIM // QUANT_BLOCK].float() - 127.0
        nope = nope.view(-1, QUANT_BLOCK) * torch.pow(2.0, exps).unsqueeze(-1)
        rope = row[base + NOPE_DIM : base + TOKEN_DATA_BYTES].view(torch.bfloat16)
        rows.append(torch.cat([nope.reshape(-1), rope.float()]))
    return torch.stack(rows)


@pytest.mark.skipif(not rdna_ops.has_sparse_mla_decode(), reason="decode op not built")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_sink", [False, True])
def test_kv_insert_feeds_sparse_mla_decode(dtype, with_sink):
    """Rows written here decode correctly in sparse_mla_decode_rdna2."""
    n, hq, block_size = 48, 16, 64
    q, kv, cache, slots, positions, cos_sin = _make_inputs(
        dtype, n, hq, block_size, 2, n, True, True, seed=13
    )
    # Every token gets a real (distinct) slot here.
    gen = torch.Generator().manual_seed(13)
    slots = torch.randperm(2 * block_size, generator=gen)[:n].to("cuda")
    kv[:, :NOPE_DIM] = kv[:, :NOPE_DIM].clamp(-8, 8)  # keep softmax non-degenerate
    q_out = rdna_ops.qnorm_rope_kv_insert(
        q, kv, cache, slots, positions, cos_sin, hq, EPS, block_size
    )
    # Two queries: the last token attends to all rows, the first to 7 rows.
    num_rows = cache.shape[0] * block_size
    sel = [slots, slots[:7]]
    indices = torch.cat(sel).to(torch.int32)
    indptr = torch.tensor([0, n, n + 7], dtype=torch.int32, device="cuda")
    qd = q_out[[n - 1, 0]].contiguous()
    out = torch.empty_like(qd)
    empty_u8 = torch.empty(0, dtype=torch.uint8, device="cuda")
    empty_i32 = torch.empty(0, dtype=torch.int32, device="cuda")
    scale = HEAD_DIM**-0.5
    sink = torch.randn(hq, generator=torch.Generator().manual_seed(1))
    rdna_ops.sparse_mla_decode(
        qd,
        cache,
        indices,
        indptr,
        empty_u8,
        empty_i32,
        torch.zeros(3, dtype=torch.int32, device="cuda"),
        block_size,
        num_rows,
        0,
        0,
        scale,
        sink.cuda() if with_sink else torch.empty(0, device="cuda"),
        out,
    )
    torch.accelerator.synchronize()
    for b, rows in enumerate(sel):
        k = _dequant_rows(cache, rows, block_size)  # [L, 512]
        qb = qd[b].float().cpu()  # [H, 512]
        logits = qb @ k.T * scale
        if with_sink:
            logits = torch.cat([logits, sink.unsqueeze(-1)], dim=-1)
            p = torch.softmax(logits, dim=-1)[:, :-1]
        else:
            p = torch.softmax(logits, dim=-1)
        ref = p @ k
        torch.testing.assert_close(out[b].float().cpu(), ref, atol=3e-2, rtol=3e-2)
