# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression: fa_rdna2 kernels must read V from the physical layout the
production KV-cache writer (reshape_and_cache) actually writes.

Background: reshape_and_cache writes K PACKED ([nb, h, D/x, bs, x],
x-innermost) but V UNPACKED ([nb, h, D, bs], slot-innermost). The fa_rdna2
kernels historically indexed V with K's stride set (via a plain .view() in
_maybe_reinterp_v_to_5d that produced packed strides over unpacked data),
permuting every V vector and producing deterministic garbage end-to-end
(e.g. Qwen3.8-27B-AWQ answered every short prompt with the same canned
phrase).

The earlier shape-sweep test (test_fa_rdna2_shape_sweep.py) could not catch
this because it fills V directly in packed layout instead of going through
the production writer. This test populates the cache through the real
production path:

    PagedAttention.split_kv_cache          (production views)
    ops.reshape_and_cache                  (production writer)
    rdna_attn._reinterpret_v_to_5d         (production V re-view)

and compares kernel output against an fp32 reference computed from the raw
K/V inputs (never touching cache layout).

Requires gfx1030. Run:
    python -m pytest tests/kernels/attention/test_fa_rdna2_writer_layout.py
or directly:
    python tests/kernels/attention/test_fa_rdna2_writer_layout.py
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and "gfx103" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="Requires AMD RDNA2 (gfx1030) GPU",
)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.v1.attention.backends.rdna_attn import (  # noqa: E402
    _reinterpret_v_to_5d,
)
from vllm.v1.attention.ops import fa_rdna2_backend as fa  # noqa: E402
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (  # noqa: E402
    has_native_kv_cache_layout,
)
from vllm.v1.attention.ops.paged_attn import PagedAttention  # noqa: E402
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (  # noqa: E402
    triton_reshape_and_cache_flash,
)


def _fill_cache(
    seq_lens,
    H_kv,
    D,
    bs,
    seed,
    layout="dense",
    raw=False,
    writer="auto",
    dtype=torch.float16,
):
    """Allocate the production [2, nb, H_kv, D, bs] cache and fill it via
    the real writer. Returns (key_cache5d, value_cache5d, block_table,
    per_seq_kv) where per_seq_kv[s] = (K[sl, H_kv, D], V[sl, H_kv, D])
    raw fp16 inputs for the reference; raw=True prepends the cache itself.

    layout="dense": contiguous [2, nb, ...] (dense-model allocation).
    layout="interleaved": [nb, 2, ...] contiguous permuted to [2, nb, ...]
        (hybrid/mamba-aligned allocation, block stride 2x dense) — the
        Qwen3.8 hybrid server layout.
    layout="block_major": [nb, 3 layers, 2 K/V slots, ...] with a per-layer
        view. The block stride spans all three layers, as in BLHNC QSA packing.
    """
    total = sum(seq_lens)
    blocks_per_seq = [(sl + bs - 1) // bs for sl in seq_lens]
    nb = sum(blocks_per_seq) + 2  # 2 spare blocks, never referenced
    if layout == "block_major":
        buf = torch.zeros(nb, 3, 2, H_kv, D, bs, dtype=dtype, device="cuda")
        kv_cache = buf[:, 1].permute(1, 0, 2, 3, 4)
    elif layout == "interleaved":
        buf = torch.zeros(nb, 2, H_kv, D, bs, dtype=dtype, device="cuda")
        kv_cache = buf.permute(1, 0, 2, 3, 4)
    else:
        kv_cache = torch.zeros(2, nb, H_kv, D, bs, dtype=dtype, device="cuda")
    key_cache, value_cache = PagedAttention.split_kv_cache(kv_cache, H_kv, D)

    g = torch.Generator(device="cuda").manual_seed(seed)
    K = torch.randn(total, H_kv, D, dtype=dtype, device="cuda", generator=g)
    V = torch.randn(total, H_kv, D, dtype=dtype, device="cuda", generator=g)

    # Scrambled physical block order to exercise block_table indirection.
    perm = torch.randperm(nb, generator=torch.Generator().manual_seed(seed))
    max_blocks = max(blocks_per_seq)
    block_table = torch.zeros(len(seq_lens), max_blocks, dtype=torch.int32)
    slots: list[int] = []
    per_seq_kv = []
    blk = 0
    off = 0
    for si, sl in enumerate(seq_lens):
        nb_s = blocks_per_seq[si]
        blocks = [int(perm[blk + j]) for j in range(nb_s)]
        blk += nb_s
        block_table[si, :nb_s] = torch.tensor(blocks, dtype=torch.int32)
        slots.extend(b * bs + (j % bs) for j in range(sl) for b in [blocks[j // bs]])
        per_seq_kv.append((K[off : off + sl], V[off : off + sl]))
        off += sl
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device="cuda")

    ones = torch.ones(1, dtype=torch.float32, device="cuda")
    # Mirror the production writer selection (rdna_attn.do_kv_cache_update).
    if writer == "hip":
        torch.ops._rocm_C.reshape_and_cache_flash_rdna2(
            K, V, key_cache, value_cache, slot_mapping
        )
    elif bs in (16, 32) and has_native_kv_cache_layout(key_cache, value_cache):
        ops.reshape_and_cache(
            K, V, key_cache, value_cache, slot_mapping, "auto", ones, ones
        )
    else:
        triton_reshape_and_cache_flash(
            K, V, key_cache, value_cache, slot_mapping, "auto", ones, ones
        )

    value_cache5 = _reinterpret_v_to_5d(key_cache, value_cache, D)
    if raw:
        return kv_cache, key_cache, value_cache5, block_table.cuda(), per_seq_kv
    return key_cache, value_cache5, block_table.cuda(), per_seq_kv


@pytest.mark.parametrize("D", [128, 256])
def test_block_major_hip_writer_prefill_and_decode(D):
    """FA-RDNA2 must read HIP-written pages with a padded BLHNC block stride."""
    sl, bs, H_q, H_kv = 1100, 1024, 8, 2
    kc, vc, bt, per_seq_kv = _fill_cache(
        [sl], H_kv, D, bs, seed=D, layout="block_major", writer="hip"
    )
    assert kc.stride(0) > kc[0].numel()
    torch.manual_seed(D)
    q_prefill = torch.randn(sl, H_q, D, dtype=torch.float16, device="cuda")
    cu = torch.tensor([0, sl], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([sl], dtype=torch.int32, device="cuda")
    out_prefill = fa.fa_rdna2_prefill_paged_varlen(
        q_prefill, kc, vc, bt, cu, seq_lens, bs, 1, 0
    )
    ref_prefill = _ref_attention(q_prefill, per_seq_kv, [0, sl], H_kv, causal=True)
    torch.testing.assert_close(out_prefill, ref_prefill, rtol=2e-2, atol=2e-2)

    q_decode = q_prefill[-1:]
    out_decode = fa.fa_rdna2_decode_paged(q_decode, kc, vc, bt, seq_lens, bs, 8, 0)
    ref_decode = _ref_attention(q_decode, per_seq_kv, [0, 1], H_kv, causal=False)
    torch.testing.assert_close(out_decode, ref_decode, rtol=2e-2, atol=2e-2)


def _ref_attention(Q, per_seq_kv, cu_q, H_kv, causal, sliding_window=0, scale=None):
    """fp32 reference from raw K/V inputs. Q: [total_q, H_q, D]. For full
    prefill nq == sl per seq; for decode nq == 1 (query is the last token).
    A sliding window keeps keys with q - k < sliding_window (vLLM semantics).
    """
    H_q, D = Q.shape[1], Q.shape[2]
    group = H_q // H_kv
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    o_ref = torch.zeros(Q.shape, dtype=torch.float32, device=Q.device)
    cu = [int(c) for c in cu_q]
    for s, (K, V) in enumerate(per_seq_kv):
        q0, q1 = cu[s], cu[s + 1]
        nq = q1 - q0
        sl = K.shape[0]
        Qf = Q[q0:q1].float()
        Kf = K.float().repeat_interleave(group, dim=1)
        Vf = V.float().repeat_interleave(group, dim=1)
        for c0 in range(0, nq, 256):
            c1 = min(c0 + 256, nq)
            sc = torch.einsum("qhd,khd->qhk", Qf[c0:c1], Kf) * scale
            qi = torch.arange(c0, c1, device=Q.device) + (sl - nq)
            ki = torch.arange(sl, device=Q.device)
            mask = torch.zeros(c1 - c0, sl, dtype=torch.bool, device=Q.device)
            if causal:
                mask |= ki[None, :] > qi[:, None]
            if sliding_window > 0:
                mask |= qi[:, None] - ki[None, :] >= sliding_window
                if not causal:
                    mask |= ki[None, :] - qi[:, None] >= sliding_window
            sc = sc.masked_fill(mask[:, None, :], float("-inf"))
            p = sc.softmax(dim=-1)
            o_ref[q0 + c0 : q0 + c1] = torch.einsum("qhk,khd->qhd", p, Vf)
    return o_ref.to(Q.dtype)


def _max_rel_err(out, ref):
    diff = (out.float() - ref.float()).abs()
    return (diff / (ref.float().abs() + 1e-3)).max().item()


# Qwen3.8-27B-AWQ hybrid full-attention shape: H_q=24, H_kv=4, D=256,
# mamba-aligned block_size=784. Covers the general varlen kernel (short and
# 1k) and the splitk kernel (5k, kv_splits=5) — the production dispatch for
# this model. "interleaved" replicates the hybrid [nb, 2, ...] allocation.
@pytest.mark.parametrize("layout", ["dense", "interleaved"])
@pytest.mark.parametrize("sl", [26, 1024, 5000])
def test_prefill_varlen_d256_writer_layout(sl, layout):
    H_q, H_kv, D, bs = 24, 4, 256, 784
    kc, vc, bt, per_seq_kv = _fill_cache([sl], H_kv, D, bs, seed=sl, layout=layout)
    torch.manual_seed(sl)
    Q = torch.randn(sl, H_q, D, dtype=torch.float16, device="cuda")
    cu = torch.tensor([0, sl], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([sl], dtype=torch.int32, device="cuda")
    kv_splits = min(8, (sl + 1023) // 1024)
    if kv_splits >= 2:
        out = fa.fa_rdna2_prefill_paged_varlen_splitk(
            Q, kc, vc, bt, cu, seq_lens, bs, causal=True, kv_splits=kv_splits
        )
    else:
        out = fa.fa_rdna2_prefill_paged_varlen(Q, kc, vc, bt, cu, seq_lens, bs, 1, 0)
    ref = _ref_attention(Q, per_seq_kv, [0, sl], H_kv, causal=True)
    err = _max_rel_err(out, ref)
    assert err < 5e-3, f"prefill D=256 sl={sl} {layout}: max_rel_err={err}"


# D=128 paths: the sub-4096 "short" kernel (vectorized half2 K/V loads) and
# the general varlen kernel.
@pytest.mark.parametrize("kernel", ["short", "general"])
def test_prefill_d128_writer_layout(kernel):
    H_q, H_kv, D, bs = 16, 4, 128, 16
    sl = 512
    kc, vc, bt, per_seq_kv = _fill_cache([sl], H_kv, D, bs, seed=sl)
    torch.manual_seed(sl)
    Q = torch.randn(sl, H_q, D, dtype=torch.float16, device="cuda")
    cu = torch.tensor([0, sl], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([sl], dtype=torch.int32, device="cuda")
    if kernel == "short":
        out = fa.fa_rdna2_prefill_paged_varlen_short(
            Q, kc, vc, bt, cu, seq_lens, bs, 1, 0
        )
    else:
        out = fa.fa_rdna2_prefill_paged_varlen(Q, kc, vc, bt, cu, seq_lens, bs, 1, 0)
    ref = _ref_attention(Q, per_seq_kv, [0, sl], H_kv, causal=True)
    err = _max_rel_err(out, ref)
    assert err < 5e-3, f"prefill {kernel} D=128 sl={sl}: max_rel_err={err}"


# Decode kernel (kv_splits=8, the production value) at short and long KV.
# D=256 also runs the interleaved hybrid layout.
@pytest.mark.parametrize(
    "D,H_q,H_kv,bs,layout",
    [
        (256, 24, 4, 784, "dense"),
        (256, 24, 4, 784, "interleaved"),
        (128, 16, 4, 16, "dense"),
    ],
)
@pytest.mark.parametrize("sl", [26, 1024, 5000])
def test_decode_writer_layout(D, H_q, H_kv, bs, layout, sl):
    kc, vc, bt, per_seq_kv = _fill_cache([sl], H_kv, D, bs, seed=sl, layout=layout)
    torch.manual_seed(sl + 1)
    Q = torch.randn(1, H_q, D, dtype=torch.float16, device="cuda")
    seq_lens = torch.tensor([sl], dtype=torch.int32, device="cuda")
    out = fa.fa_rdna2_decode_paged(Q, kc, vc, bt, seq_lens, bs, 8, 0)
    ref = _ref_attention(Q, per_seq_kv, [0, 1], H_kv, causal=False)
    err = _max_rel_err(out, ref)
    assert err < 5e-3, f"decode D={D} sl={sl}: max_rel_err={err}"


# Multi-sequence varlen prefill in one launch (mixed short + 1k + 5k).
def test_prefill_varlen_d256_multiseq():
    H_q, H_kv, D, bs = 24, 4, 256, 784
    seq_lens_l = [37, 1000, 5000]
    kc, vc, bt, per_seq_kv = _fill_cache(seq_lens_l, H_kv, D, bs, seed=7)
    total = sum(seq_lens_l)
    torch.manual_seed(7)
    Q = torch.randn(total, H_q, D, dtype=torch.float16, device="cuda")
    cu = torch.tensor([0, 37, 1037, 6037], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor(seq_lens_l, dtype=torch.int32, device="cuda")
    out = fa.fa_rdna2_prefill_paged_varlen(Q, kc, vc, bt, cu, seq_lens, bs, 1, 0)
    ref = _ref_attention(Q, per_seq_kv, [0, 37, 1037, 6037], H_kv, causal=True)
    err = _max_rel_err(out, ref)
    assert err < 5e-3, f"prefill multiseq D=256: max_rel_err={err}"


def _run(fn, *args):
    try:
        fn(*args)
        return "PASS"
    except AssertionError as e:
        return f"FAIL ({e})"


# Chunked-prefill / prefix-cache path: the query tensor holds only a
# mid-sequence chunk (nq < seq_len). The causal mask must use the absolute
# query position (kv_offset + chunk-local index), not the chunk-local one.
@pytest.mark.parametrize("kernel", ["general", "splitk", "short", "gqa", "gqa128"])
def test_prefill_chunked_offset(kernel):
    if kernel in ("short", "gqa128"):
        H_q, H_kv, D, bs = 16, 4, 128, 16
        full, nq = 512, 256
    else:
        H_q, H_kv, D, bs = 24, 4, 256, 784
        full, nq = 2000, 1000
    kc, vc, bt, per_seq_kv = _fill_cache([full], H_kv, D, bs, seed=full)
    torch.manual_seed(full)
    Q = torch.randn(nq, H_q, D, dtype=torch.float16, device="cuda")
    cu = torch.tensor([0, nq], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([full], dtype=torch.int32, device="cuda")
    if kernel == "splitk":
        out = fa.fa_rdna2_prefill_paged_varlen_splitk(
            Q, kc, vc, bt, cu, seq_lens, bs, causal=True, kv_splits=4
        )
    elif kernel == "short":
        out = fa.fa_rdna2_prefill_paged_varlen_short(
            Q, kc, vc, bt, cu, seq_lens, bs, 1, 0
        )
    elif kernel.startswith("gqa"):
        out = fa.fa_rdna2_prefill_paged_varlen_gqa(
            Q, kc, vc, bt, cu, seq_lens, bs, 1, 0
        )
    else:
        out = fa.fa_rdna2_prefill_paged_varlen(Q, kc, vc, bt, cu, seq_lens, bs, 1, 0)
    ref = _ref_attention(Q, per_seq_kv, [0, nq], H_kv, causal=True)
    err = _max_rel_err(out, ref)
    assert err < 5e-3, f"chunked {kernel} D={D}: max_rel_err={err}"


def _run_prefill(kernel, Q, kc, vc, bt, cu, seq_lens, bs, **kw):
    if kernel == "short":
        return fa.fa_rdna2_prefill_paged_varlen_short(
            Q, kc, vc, bt, cu, seq_lens, bs, **kw
        )
    if kernel == "splitk":
        return fa.fa_rdna2_prefill_paged_varlen_splitk(
            Q, kc, vc, bt, cu, seq_lens, bs, kv_splits=4, **kw
        )
    if kernel.startswith("gqa"):
        return fa.fa_rdna2_prefill_paged_varlen_gqa(
            Q, kc, vc, bt, cu, seq_lens, bs, **kw
        )
    return fa.fa_rdna2_prefill_paged_varlen(Q, kc, vc, bt, cu, seq_lens, bs, **kw)


# The GQA prefill kernel on a multi-sequence batch (fresh prompts plus a chunk
# behind a prefix) for each instantiation: 2 q-heads of a group per CTA for
# even groups (the production D=256 path), 1 head for odd groups and MHA.
@pytest.mark.parametrize(
    "D,H_q,H_kv,bs",
    [
        (256, 24, 4, 784),
        (256, 12, 4, 784),
        (128, 16, 4, 16),
        (128, 28, 4, 16),
        (128, 8, 8, 16),
    ],
)
@pytest.mark.parametrize("layout", ["dense", "interleaved"])
def test_prefill_gqa_multiseq(D, H_q, H_kv, bs, layout):
    seq_lens_l, q_lens = [37, 1000, 2000], [37, 1000, 500]
    kc, vc, bt, per_seq_kv = _fill_cache(
        seq_lens_l, H_kv, D, bs, seed=11, layout=layout
    )
    cu_l = [0]
    for n in q_lens:
        cu_l.append(cu_l[-1] + n)
    torch.manual_seed(11)
    Q = torch.randn(cu_l[-1], H_q, D, dtype=torch.float16, device="cuda")
    cu = torch.tensor(cu_l, dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor(seq_lens_l, dtype=torch.int32, device="cuda")
    out = fa.fa_rdna2_prefill_paged_varlen_gqa(Q, kc, vc, bt, cu, seq_lens, bs, 1, 0)
    ref = _ref_attention(Q, per_seq_kv, cu_l, H_kv, causal=True)
    err = _max_rel_err(out, ref)
    assert err < 5e-3, f"gqa multiseq D={D} {H_q}/{H_kv} {layout}: {err}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("q_len", [1, 4])
@pytest.mark.parametrize("full,bs,window", [(96, 32, 32), (2080, 1024, 2048)])
def test_prefill_gqa_dflash_window(dtype, causal, q_len, full, bs, window):
    """DFlash2 geometry: GQA, BF16 cache, short noncausal draft steps."""
    H_q, H_kv, D = 8, 2, 128
    kc, vc, bt, per_seq_kv = _fill_cache([full], H_kv, D, bs, seed=38, dtype=dtype)
    Q = torch.randn(q_len, H_q, D, dtype=dtype, device="cuda")
    cu = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([full], dtype=torch.int32, device="cuda")
    out = fa.fa_rdna2_prefill_paged_varlen_gqa(
        Q,
        kc,
        vc,
        bt,
        cu,
        seq_lens,
        bs,
        causal=causal,
        sliding_window=window,
    )
    ref = _ref_attention(
        Q,
        per_seq_kv,
        [0, q_len],
        H_kv,
        causal=causal,
        sliding_window=window,
    )
    torch.testing.assert_close(out, ref, rtol=0.04, atol=0.04)


@pytest.mark.parametrize("q_len", [1, 4])
@pytest.mark.parametrize("full,bs,window", [(96, 32, 32), (2080, 1024, 2048)])
def test_prefill_gqa_bf16_vector_scalar_parity(q_len, full, bs, window):
    """Packed BF16 K/V loads must match the scalar cache-stride fallback."""
    H_q, H_kv, D = 8, 2, 128
    kc, vc, bt, _ = _fill_cache([full], H_kv, D, bs, seed=39, dtype=torch.bfloat16)
    # Change only the physical strides; logical K/V and block table are equal.
    k_storage = torch.empty(
        (*kc.shape[:-1], kc.shape[-1] * 2), dtype=kc.dtype, device=kc.device
    )
    k_scalar = k_storage[..., ::2]
    k_scalar.copy_(kc)
    v_storage = torch.empty(
        (*vc.shape[:-2], vc.shape[-2] * 2, vc.shape[-1]),
        dtype=vc.dtype,
        device=vc.device,
    )
    v_scalar = v_storage[..., ::2, :]
    v_scalar.copy_(vc)
    assert kc.stride(4) == 1 and vc.stride(3) == 1
    assert k_scalar.stride(4) != 1 and v_scalar.stride(3) != 1

    Q = torch.randn(q_len, H_q, D, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([full], dtype=torch.int32, device="cuda")
    args = (bt, cu, seq_lens, bs)
    packed = fa.fa_rdna2_prefill_paged_varlen_gqa(
        Q, kc, vc, *args, causal=False, sliding_window=window
    )
    scalar = fa.fa_rdna2_prefill_paged_varlen_gqa(
        Q, k_scalar, v_scalar, *args, causal=False, sliding_window=window
    )
    torch.testing.assert_close(packed, scalar, rtol=0, atol=0)


def test_prefill_gqa_bf16_unaligned_cache_base():
    """A BF16 cache view offset by one element must use the scalar path."""
    H_q, H_kv, D, bs, full = 8, 2, 128, 32, 96
    kc, vc, bt, _ = _fill_cache([full], H_kv, D, bs, seed=40, dtype=torch.bfloat16)
    k_storage = torch.empty(kc.numel() + 1, dtype=kc.dtype, device=kc.device)
    k_unaligned = k_storage[1:].view(kc.shape)
    k_unaligned.copy_(kc)
    v_extent = 1 + sum(
        (size - 1) * stride for size, stride in zip(vc.shape, vc.stride())
    )
    v_storage = torch.empty(v_extent + 1, dtype=vc.dtype, device=vc.device)
    v_unaligned = v_storage.as_strided(vc.shape, vc.stride(), storage_offset=1)
    v_unaligned.copy_(vc)
    assert k_unaligned.data_ptr() % 16 != 0
    assert v_unaligned.data_ptr() % 16 != 0
    assert k_unaligned.stride(4) == 1 and v_unaligned.stride(3) == 1

    Q = torch.randn(1, H_q, D, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([full], dtype=torch.int32, device="cuda")
    args = (bt, cu, seq_lens, bs)
    packed = fa.fa_rdna2_prefill_paged_varlen_gqa(
        Q, kc, vc, *args, causal=False, sliding_window=32
    )
    scalar = fa.fa_rdna2_prefill_paged_varlen_gqa(
        Q, k_unaligned, v_unaligned, *args, causal=False, sliding_window=32
    )
    torch.testing.assert_close(packed, scalar, rtol=0, atol=0)


@pytest.mark.parametrize("kv_splits", [4, 8, 16, 32])
@pytest.mark.parametrize(
    "q_len,full,bs,window,causal,layout",
    [
        (1, 96, 32, 32, False, "dense"),
        (3, 2080, 1024, 2048, False, "dense"),
        (4, 2080, 1024, 2048, True, "dense"),
        (3, 2080, 1024, 2048, False, "block_major"),
    ],
)
def test_prefill_gqa_bf16_splitk_window(
    kv_splits, q_len, full, bs, window, causal, layout
):
    """Split-K must merge masked and empty BF16 window partitions."""
    H_q, H_kv, D = 8, 2, 128
    kc, vc, bt, per_seq_kv = _fill_cache(
        [full], H_kv, D, bs, seed=41, dtype=torch.bfloat16, layout=layout
    )
    Q = torch.randn(q_len, H_q, D, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([full], dtype=torch.int32, device="cuda")
    args = (bt, cu, seq_lens, bs)
    split = fa.fa_rdna2_prefill_paged_varlen_gqa(
        Q,
        kc,
        vc,
        *args,
        causal=causal,
        sliding_window=window,
        kv_splits=kv_splits,
    )
    unsplit = fa.fa_rdna2_prefill_paged_varlen_gqa(
        Q,
        kc,
        vc,
        *args,
        causal=causal,
        sliding_window=window,
    )
    ref = _ref_attention(
        Q,
        per_seq_kv,
        [0, q_len],
        H_kv,
        causal=causal,
        sliding_window=window,
    )
    torch.testing.assert_close(split, unsplit, rtol=0.01, atol=0.001)
    torch.testing.assert_close(split, ref, rtol=0.04, atol=0.04)


@pytest.mark.parametrize("kv_splits", [8, 32])
def test_prefill_gqa_bf16_splitk_varlen_multiseq(kv_splits):
    """Empty sequences and unequal query lengths must not leave stale partials."""
    H_q, H_kv, D, bs = 8, 2, 128, 1024
    lengths = [0, 96, 2080]
    kc, vc, bt, per_seq_kv = _fill_cache(
        lengths, H_kv, D, bs, seed=42, dtype=torch.bfloat16
    )
    Q = torch.randn(4, H_q, D, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, 0, 1, 4], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    split = fa.fa_rdna2_prefill_paged_varlen_gqa(
        Q,
        kc,
        vc,
        bt,
        cu,
        seq_lens,
        bs,
        causal=False,
        sliding_window=2048,
        kv_splits=kv_splits,
    )
    unsplit = fa.fa_rdna2_prefill_paged_varlen_gqa(
        Q, kc, vc, bt, cu, seq_lens, bs, causal=False, sliding_window=2048
    )
    ref = _ref_attention(
        Q,
        per_seq_kv,
        [0, 0, 1, 4],
        H_kv,
        causal=False,
        sliding_window=2048,
    )
    torch.testing.assert_close(split, unsplit, rtol=0.01, atol=0.001)
    torch.testing.assert_close(split, ref, rtol=0.04, atol=0.04)


def test_prefill_gqa_bf16_splitk_empty_sequence_with_query():
    """A graph-padding query with no KV must merge neutral split results."""
    H_q, H_kv, D, bs = 8, 2, 128, 32
    kc, vc, bt, per_seq_kv = _fill_cache(
        [0, 96], H_kv, D, bs, seed=43, dtype=torch.bfloat16
    )
    Q = torch.randn(2, H_q, D, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([0, 96], dtype=torch.int32, device="cuda")
    out = fa.fa_rdna2_prefill_paged_varlen_gqa(
        Q,
        kc,
        vc,
        bt,
        cu,
        seq_lens,
        bs,
        causal=False,
        sliding_window=32,
        kv_splits=32,
    )
    assert torch.isfinite(out).all()
    assert (out[0] == 0).all()
    ref = _ref_attention(
        Q[1:],
        per_seq_kv[1:],
        [0, 1],
        H_kv,
        causal=False,
        sliding_window=32,
    )
    torch.testing.assert_close(out[1:], ref, rtol=0.04, atol=0.04)


def test_prefill_gqa_bf16_splitk_graph_replay():
    """Captured split-K must ignore padded rows as live query count changes."""
    H_q, H_kv, D, bs, full = 8, 2, 128, 32, 96
    kc, vc, bt, _ = _fill_cache([full], H_kv, D, bs, seed=44, dtype=torch.bfloat16)
    Q = torch.randn(4, H_q, D, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([full], dtype=torch.int32, device="cuda")
    out = torch.empty_like(Q)

    def run_split():
        fa.fa_rdna2_prefill_paged_varlen_gqa(
            Q,
            kc,
            vc,
            bt,
            cu,
            seq_lens,
            bs,
            causal=False,
            sliding_window=32,
            kv_splits=8,
            out=out,
        )

    run_split()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_split()
    for live_rows in (1, 2, 1):
        cu[1] = live_rows
        Q.copy_(torch.randn_like(Q))
        expected = fa.fa_rdna2_prefill_paged_varlen_gqa(
            Q,
            kc,
            vc,
            bt,
            cu,
            seq_lens,
            bs,
            causal=False,
            sliding_window=32,
        )
        graph.replay()
        torch.testing.assert_close(
            out[:live_rows], expected[:live_rows], rtol=0.01, atol=0.001
        )
        assert (out[live_rows:] == 0).all()


@pytest.mark.parametrize(
    "overrides,expected_splits",
    [
        ({}, 16),
        ({"max_seqlen_q": 4}, 16),
        ({"max_seqlen_q": 1024}, 1),
        ({"max_seqlen_q": None}, 1),
        ({"causal": True}, 1),
        ({"dtype": torch.float16}, 1),
        ({"head_size": 256}, 1),
        ({"num_heads": 16, "num_kv_heads": 4}, 1),
        ({"paged_block_size": 16}, 1),
        ({"sliding_window": 32}, 1),
        ({"num_seqs": 16}, 1),
        ({"max_seqlen_k": 512}, 1),
    ],
)
def test_gqa_split_dispatch_preserves_unmeasured_paths(
    monkeypatch, overrides, expected_splits
):
    from types import SimpleNamespace

    from vllm.v1.attention.backends import rdna_attn

    settings = (
        dict(
            dtype=torch.bfloat16,
            head_size=128,
            num_heads=8,
            num_kv_heads=2,
            paged_block_size=1024,
            sliding_window=2048,
            max_seqlen_q=3,
            max_seqlen_k=16384,
            causal=False,
            num_seqs=8,
        )
        | overrides
    )
    calls = []
    monkeypatch.setattr(rdna_attn, "_gqa_mode", lambda: "subgroup")
    monkeypatch.setattr(
        rdna_attn,
        "_get_fa_rdna2_module",
        lambda: SimpleNamespace(
            fa_rdna2_prefill_paged_varlen_gqa=lambda *a, **kw: calls.append(kw)
        ),
    )
    impl = SimpleNamespace(
        num_heads=settings["num_heads"],
        num_kv_heads=settings["num_kv_heads"],
        head_size=settings["head_size"],
        scale=settings["head_size"] ** -0.5,
    )
    q = torch.empty(1, impl.num_heads, impl.head_size, dtype=settings["dtype"])
    rdna_attn.RdnaAttentionImpl._forward_prefill(
        impl,
        q,
        None,
        None,
        None,
        None,
        torch.empty(settings["num_seqs"], dtype=torch.int32),
        settings["max_seqlen_k"],
        settings["paged_block_size"],
        settings["sliding_window"],
        torch.empty_like(q),
        causal=settings["causal"],
        max_seqlen_q=settings["max_seqlen_q"],
    )
    assert len(calls) == 1
    assert calls[0]["kv_splits"] == expected_splits


# Sliding window: query q attends keys k with q - k < window, in every
# prefill kernel (they used to keep window + 1 keys).
@pytest.mark.parametrize("kernel", ["general", "splitk", "short", "gqa", "gqa128"])
def test_prefill_sliding_window(kernel):
    if kernel in ("short", "gqa128"):
        H_q, H_kv, D, bs = 16, 4, 128, 16
    else:
        H_q, H_kv, D, bs = 24, 4, 256, 784
    full, nq, window = 700, 300, 100
    kc, vc, bt, per_seq_kv = _fill_cache([full], H_kv, D, bs, seed=5)
    torch.manual_seed(5)
    Q = torch.randn(nq, H_q, D, dtype=torch.float16, device="cuda")
    cu = torch.tensor([0, nq], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([full], dtype=torch.int32, device="cuda")
    out = _run_prefill(kernel, Q, kc, vc, bt, cu, seq_lens, bs, sliding_window=window)
    ref = _ref_attention(
        Q, per_seq_kv, [0, nq], H_kv, causal=True, sliding_window=window
    )
    err = _max_rel_err(out, ref)
    assert err < 5e-3, f"window {kernel}: max_rel_err={err}"


# The kernels use the caller's softmax scale (not a hardcoded 1/sqrt(D)) and
# write into the caller's `out` buffer.
@pytest.mark.parametrize("kernel", ["decode", "gqa", "short"])
def test_custom_scale_into_out(kernel):
    D = 128 if kernel == "short" else 256
    H_q, H_kv, bs = (16, 4, 16) if D == 128 else (24, 4, 784)
    sl = 600
    nq = 1 if kernel == "decode" else sl
    kc, vc, bt, per_seq_kv = _fill_cache([sl], H_kv, D, bs, seed=3)
    torch.manual_seed(3)
    Q = torch.randn(nq, H_q, D, dtype=torch.float16, device="cuda")
    seq_lens = torch.tensor([sl], dtype=torch.int32, device="cuda")
    out = torch.empty_like(Q)
    scale = 0.05
    if kernel == "decode":
        res = fa.fa_rdna2_decode_paged(
            Q, kc, vc, bt, seq_lens, bs, 16, 0, scale=scale, out=out
        )
    else:
        cu = torch.tensor([0, nq], dtype=torch.int32, device="cuda")
        res = _run_prefill(
            kernel, Q, kc, vc, bt, cu, seq_lens, bs, scale=scale, out=out
        )
    assert res.data_ptr() == out.data_ptr()
    ref = _ref_attention(
        Q, per_seq_kv, [0, nq], H_kv, causal=kernel != "decode", scale=scale
    )
    err = _max_rel_err(out, ref)
    assert err < 5e-3, f"scale {kernel}: max_rel_err={err}"


# Rows with seq_len 0 (CUDA-graph padding) must come out as zeros, not NaN.
def test_decode_empty_rows_are_zero():
    H_q, H_kv, D, bs = 24, 4, 256, 784
    kc, vc, bt, _ = _fill_cache([300, 0], H_kv, D, bs, seed=9)
    Q = torch.randn(2, H_q, D, dtype=torch.float16, device="cuda")
    seq_lens = torch.tensor([300, 0], dtype=torch.int32, device="cuda")
    out = fa.fa_rdna2_decode_paged(Q, kc, vc, bt, seq_lens, bs, 16, 0)
    assert torch.isfinite(out).all()
    assert (out[1] == 0).all()


# Spec-decode verify rows and short extends through the decode kernel: with
# cu_query_lens, request s owns queries cu[s]..cu[s+1]-1 at its last
# positions (causal). A zero-length padded request and tokens past cu[-1]
# (graph padding) must come out as zeros.
@pytest.mark.parametrize("D,H_q,H_kv,bs", [(256, 24, 4, 784), (128, 16, 4, 16)])
@pytest.mark.parametrize("window", [0, 50])
def test_decode_multi_token_queries(D, H_q, H_kv, bs, window):
    seq_lens_l, q_lens = [700, 1000, 33, 3, 2600, 0], [1, 3, 2, 3, 4, 0]
    kc, vc, bt, per_seq_kv = _fill_cache(seq_lens_l, H_kv, D, bs, seed=13)
    cu_l = [0]
    for n in q_lens:
        cu_l.append(cu_l[-1] + n)
    torch.manual_seed(13)
    Q = torch.randn(cu_l[-1] + 2, H_q, D, dtype=torch.float16, device="cuda")
    cu = torch.tensor(cu_l, dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor(seq_lens_l, dtype=torch.int32, device="cuda")
    out = fa.fa_rdna2_decode_paged(
        Q, kc, vc, bt, seq_lens, bs, 16, window, cu_query_lens=cu
    )
    ref = _ref_attention(Q, per_seq_kv, cu_l, H_kv, causal=True, sliding_window=window)
    err = _max_rel_err(out, ref)
    assert err < 5e-3, f"multi-token decode D={D}: max_rel_err={err}"
    assert (out[cu_l[-1] :] == 0).all()


def _common_metadata(q_lens, seq_lens, block_table, device="cuda"):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    cu = [0]
    for n in q_lens:
        cu.append(cu[-1] + n)
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor(cu, dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor(cu, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        num_reqs=len(q_lens),
        num_actual_tokens=cu[-1],
        max_query_len=max(q_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table,
        slot_mapping=torch.zeros(cu[-1], dtype=torch.int64, device=device),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
    )


def _builder(reorder_batch_threshold):
    from vllm.v1.attention.backends.rdna_attn import (
        RdnaAttentionMetadataBuilder,
    )

    builder = RdnaAttentionMetadataBuilder.__new__(RdnaAttentionMetadataBuilder)
    builder.reorder_batch_threshold = reorder_batch_threshold
    return builder


# Decode-first split: requests up to the reorder threshold (verify rows, short
# extends) go to the decode kernel, the rest to a prefill kernel whose
# query_start_loc is rebased to its first token.
def test_builder_splits_decode_first():
    cm = _common_metadata(
        [1, 3, 2, 50, 7],
        [9000, 900, 40, 60, 3000],
        torch.zeros(5, 1, dtype=torch.int32),
        device="cpu",
    )
    meta = _builder(3).build(0, cm)
    assert (meta.num_decodes, meta.num_decode_tokens) == (3, 6)
    assert meta.decode_query_start_loc.tolist() == [0, 1, 4, 6]
    assert meta.prefill_query_start_loc.tolist() == [0, 50, 57]
    assert meta.max_prefill_seq_len == 3000
    # No reorder threshold (split disabled): the whole batch is prefill.
    meta = _builder(None).build(0, cm)
    assert meta.num_decode_tokens == 0
    assert meta.prefill_query_start_loc is cm.query_start_loc


# RDNA_ATTN forward on a decode-first batch, split on (threshold 3, i.e. two
# speculative tokens) and off: a mixed batch (decode, verify, short extend,
# chunk behind a prefix, fresh prompt) and a verify-only batch.
@pytest.mark.parametrize("D,H_q,H_kv,bs", [(256, 24, 4, 784), (128, 16, 4, 16)])
@pytest.mark.parametrize("batch", ["mixed", "verify"])
@pytest.mark.parametrize("threshold", [None, 3])
def test_forward_split_decode(D, H_q, H_kv, bs, batch, threshold):
    from vllm.v1.attention.backends.rdna_attn import RdnaAttentionImpl

    if batch == "mixed":
        seq_lens_l, q_lens = [900, 1500, 40, 300, 64], [1, 3, 2, 100, 64]
    else:
        seq_lens_l, q_lens = [900, 1500, 40, 3000], [3, 3, 3, 3]
    kv_cache, _, _, bt, per_seq_kv = _fill_cache(
        seq_lens_l, H_kv, D, bs, seed=21, raw=True
    )
    cm = _common_metadata(q_lens, seq_lens_l, bt)
    meta = _builder(threshold).build(0, cm)
    impl = RdnaAttentionImpl(H_q, D, D**-0.5, H_kv, None, None, "auto")
    torch.manual_seed(21)
    Q = torch.randn(cm.num_actual_tokens, H_q, D, dtype=torch.float16, device="cuda")
    out = torch.empty_like(Q)
    impl.forward(None, Q, None, None, kv_cache.transpose(0, 1), meta, out)
    cu_l = cm.query_start_loc_cpu.tolist()
    ref = _ref_attention(Q, per_seq_kv, cu_l, H_kv, causal=True)
    err = _max_rel_err(out, ref)
    assert err < 5e-3, f"forward {batch} D={D} thr={threshold}: {err}"


def test_metadata_carries_causal():
    """RdnaAttentionMetadata must propagate CommonAttentionMetadata.causal.

    Regression: from_common used to drop the field, so
    getattr(attn_metadata, "causal", False) in forward() was always False
    and every fa_rdna2 prefill ran non-causal — deterministic garbage
    end-to-end despite correct kernels.
    """
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.rdna_attn import (
        RdnaAttentionMetadataBuilder,
    )

    cm = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=2,
        block_table_tensor=torch.zeros(1, 1, dtype=torch.int32),
        slot_mapping=torch.zeros(2, dtype=torch.int64),
    )
    builder = RdnaAttentionMetadataBuilder.__new__(RdnaAttentionMetadataBuilder)
    meta = builder.build(0, cm)
    assert meta.causal is True


if __name__ == "__main__":
    import os

    report_only = os.environ.get("FA_RDNA2_TEST_REPORT_ONLY") == "1"
    results = []
    for layout in ("dense", "interleaved"):
        for sl in (26, 1024, 5000):
            results.append(
                (
                    f"prefill  D=256 sl={sl} {layout}",
                    _run(test_prefill_varlen_d256_writer_layout, sl, layout),
                )
            )
    for kernel in ("short", "general"):
        results.append(
            (f"prefill  D=128 {kernel}", _run(test_prefill_d128_writer_layout, kernel))
        )
    for D, H_q, H_kv, bs, layout in (
        (256, 24, 4, 784, "dense"),
        (256, 24, 4, 784, "interleaved"),
        (128, 16, 4, 16, "dense"),
    ):
        for sl in (26, 1024, 5000):
            results.append(
                (
                    f"decode   D={D} sl={sl} {layout}",
                    _run(test_decode_writer_layout, D, H_q, H_kv, bs, layout, sl),
                )
            )
    results.append(("prefill  D=256 multiseq", _run(test_prefill_varlen_d256_multiseq)))
    results.append(("metadata causal", _run(test_metadata_carries_causal)))
    for kernel in ("general", "splitk", "short", "gqa", "gqa128"):
        results.append(
            (
                f"prefill  chunked-offset {kernel}",
                _run(test_prefill_chunked_offset, kernel),
            )
        )
        results.append(
            (
                f"prefill  sliding-window {kernel}",
                _run(test_prefill_sliding_window, kernel),
            )
        )
    for D, H_q, H_kv, bs in (
        (256, 24, 4, 784),
        (256, 12, 4, 784),
        (128, 16, 4, 16),
        (128, 28, 4, 16),
        (128, 8, 8, 16),
    ):
        for layout in ("dense", "interleaved"):
            results.append(
                (
                    f"prefill  gqa multiseq D={D} {H_q}/{H_kv} {layout}",
                    _run(test_prefill_gqa_multiseq, D, H_q, H_kv, bs, layout),
                )
            )
    for kernel in ("decode", "gqa", "short"):
        results.append(
            (f"scale+out {kernel}", _run(test_custom_scale_into_out, kernel))
        )
    results.append(("decode   empty rows", _run(test_decode_empty_rows_are_zero)))
    for D, H_q, H_kv, bs in ((256, 24, 4, 784), (128, 16, 4, 16)):
        for window in (0, 50):
            results.append(
                (
                    f"decode   multi-token D={D} window={window}",
                    _run(test_decode_multi_token_queries, D, H_q, H_kv, bs, window),
                )
            )
        for batch in ("mixed", "verify"):
            for threshold in (None, 3):
                results.append(
                    (
                        f"forward  {batch} D={D} thr={threshold}",
                        _run(
                            test_forward_split_decode,
                            D,
                            H_q,
                            H_kv,
                            bs,
                            batch,
                            threshold,
                        ),
                    )
                )
    results.append(
        ("builder  decode-first split", _run(test_builder_splits_decode_first))
    )
    for name, res in results:
        print(f"{name}: {res}", flush=True)
    if any(r != "PASS" for _, r in results):
        if not report_only:
            raise SystemExit(1)
    else:
        print("ALL PASS")
