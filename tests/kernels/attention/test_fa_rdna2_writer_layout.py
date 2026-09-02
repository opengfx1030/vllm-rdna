# SPDX-License-Identifier: Apache-2.0
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
    not (torch.cuda.is_available()
         and "gfx103" in torch.cuda.get_device_properties(0).gcnArchName),
    reason="Requires AMD RDNA2 (gfx1030) GPU",
)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.v1.attention.ops.paged_attn import PagedAttention  # noqa: E402
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (  # noqa: E402
    has_native_kv_cache_layout,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (  # noqa: E402
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.backends.rdna_attn import (  # noqa: E402
    _reinterpret_v_to_5d,
)
from vllm.v1.attention.ops import fa_rdna2_backend as fa  # noqa: E402


def _fill_cache(seq_lens, H_kv, D, bs, seed, layout="dense"):
    """Allocate the production [2, nb, H_kv, D, bs] cache and fill it via
    the real writer. Returns (key_cache5d, value_cache5d, block_table,
    per_seq_kv) where per_seq_kv[s] = (K[sl, H_kv, D], V[sl, H_kv, D])
    raw fp16 inputs for the reference.

    layout="dense": contiguous [2, nb, ...] (dense-model allocation).
    layout="interleaved": [nb, 2, ...] contiguous permuted to [2, nb, ...]
        (hybrid/mamba-aligned allocation, block stride 2x dense) — the
        Qwen3.8 hybrid server layout.
    """
    total = sum(seq_lens)
    blocks_per_seq = [(sl + bs - 1) // bs for sl in seq_lens]
    nb = sum(blocks_per_seq) + 2  # 2 spare blocks, never referenced
    if layout == "interleaved":
        buf = torch.zeros(nb, 2, H_kv, D, bs, dtype=torch.float16,
                          device="cuda")
        kv_cache = buf.permute(1, 0, 2, 3, 4)
    else:
        kv_cache = torch.zeros(2, nb, H_kv, D, bs, dtype=torch.float16,
                               device="cuda")
    key_cache, value_cache = PagedAttention.split_kv_cache(kv_cache, H_kv, D)

    g = torch.Generator(device="cuda").manual_seed(seed)
    K = torch.randn(total, H_kv, D, dtype=torch.float16, device="cuda",
                    generator=g)
    V = torch.randn(total, H_kv, D, dtype=torch.float16, device="cuda",
                    generator=g)

    # Scrambled physical block order to exercise block_table indirection.
    perm = torch.randperm(nb, generator=torch.Generator().manual_seed(seed))
    max_blocks = max(blocks_per_seq)
    block_table = torch.zeros(len(seq_lens), max_blocks, dtype=torch.int32)
    slots = []
    per_seq_kv = []
    blk = 0
    off = 0
    for si, sl in enumerate(seq_lens):
        nb_s = blocks_per_seq[si]
        blocks = [int(perm[blk + j]) for j in range(nb_s)]
        blk += nb_s
        block_table[si, :nb_s] = torch.tensor(blocks, dtype=torch.int32)
        slots.extend(b * bs + (j % bs) for j in range(sl)
                     for b in [blocks[j // bs]])
        per_seq_kv.append((K[off:off + sl], V[off:off + sl]))
        off += sl
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device="cuda")

    ones = torch.ones(1, dtype=torch.float32, device="cuda")
    # Mirror the production writer selection (rdna_attn.do_kv_cache_update).
    if bs in (16, 32) and has_native_kv_cache_layout(key_cache, value_cache):
        ops.reshape_and_cache(K, V, key_cache, value_cache, slot_mapping,
                              "auto", ones, ones)
    else:
        triton_reshape_and_cache_flash(K, V, key_cache, value_cache,
                                       slot_mapping, "auto", ones, ones)

    value_cache5 = _reinterpret_v_to_5d(key_cache, value_cache, D)
    return key_cache, value_cache5, block_table.cuda(), per_seq_kv


def _ref_attention(Q, per_seq_kv, cu_q, H_kv, causal):
    """fp32 reference from raw K/V inputs. Q: [total_q, H_q, D]. For full
    prefill nq == sl per seq; for decode nq == 1 (query is the last token).
    """
    H_q, D = Q.shape[1], Q.shape[2]
    group = H_q // H_kv
    O = torch.zeros(Q.shape, dtype=torch.float32, device=Q.device)
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
            sc = torch.einsum("qhd,khd->qhk", Qf[c0:c1], Kf) / math.sqrt(D)
            if causal:
                qi = torch.arange(c0, c1, device=Q.device) + (sl - nq)
                ki = torch.arange(sl, device=Q.device)
                mask = ki[None, :] > qi[:, None]
                sc = sc.masked_fill(mask[:, None, :], float("-inf"))
            p = sc.softmax(dim=-1)
            O[q0 + c0:q0 + c1] = torch.einsum("qhk,khd->qhd", p, Vf)
    return O.half()


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
    kc, vc, bt, per_seq_kv = _fill_cache([sl], H_kv, D, bs, seed=sl,
                                         layout=layout)
    torch.manual_seed(sl)
    Q = torch.randn(sl, H_q, D, dtype=torch.float16, device="cuda")
    cu = torch.tensor([0, sl], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([sl], dtype=torch.int32, device="cuda")
    kv_splits = min(8, (sl + 1023) // 1024)
    if kv_splits >= 2:
        out = fa.fa_rdna2_prefill_paged_varlen_splitk(
            Q, kc, vc, bt, cu, seq_lens, bs, 1, 0, kv_splits)
    else:
        out = fa.fa_rdna2_prefill_paged_varlen(
            Q, kc, vc, bt, cu, seq_lens, bs, 1, 0)
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
            Q, kc, vc, bt, cu, seq_lens, bs, 1, 0)
    else:
        out = fa.fa_rdna2_prefill_paged_varlen(
            Q, kc, vc, bt, cu, seq_lens, bs, 1, 0)
    ref = _ref_attention(Q, per_seq_kv, [0, sl], H_kv, causal=True)
    err = _max_rel_err(out, ref)
    assert err < 5e-3, f"prefill {kernel} D=128 sl={sl}: max_rel_err={err}"


# Decode kernel (kv_splits=8, the production value) at short and long KV.
# D=256 also runs the interleaved hybrid layout.
@pytest.mark.parametrize("D,H_q,H_kv,bs,layout", [
    (256, 24, 4, 784, "dense"), (256, 24, 4, 784, "interleaved"),
    (128, 16, 4, 16, "dense"),
])
@pytest.mark.parametrize("sl", [26, 1024, 5000])
def test_decode_writer_layout(D, H_q, H_kv, bs, layout, sl):
    kc, vc, bt, per_seq_kv = _fill_cache([sl], H_kv, D, bs, seed=sl,
                                         layout=layout)
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
    out = fa.fa_rdna2_prefill_paged_varlen(
        Q, kc, vc, bt, cu, seq_lens, bs, 1, 0)
    ref = _ref_attention(Q, per_seq_kv, [0, 37, 1037, 6037], H_kv,
                         causal=True)
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
@pytest.mark.parametrize("kernel", ["general", "splitk", "short"])
def test_prefill_chunked_offset(kernel):
    if kernel == "short":
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
            Q, kc, vc, bt, cu, seq_lens, bs, 1, 0, 4)
    elif kernel == "short":
        out = fa.fa_rdna2_prefill_paged_varlen_short(
            Q, kc, vc, bt, cu, seq_lens, bs, 1, 0)
    else:
        out = fa.fa_rdna2_prefill_paged_varlen(
            Q, kc, vc, bt, cu, seq_lens, bs, 1, 0)
    ref = _ref_attention(Q, per_seq_kv, [0, nq], H_kv, causal=True)
    err = _max_rel_err(out, ref)
    assert err < 5e-3, f"chunked {kernel} D={D}: max_rel_err={err}"


def test_metadata_carries_causal():
    """RdnaAttentionMetadata must propagate CommonAttentionMetadata.causal.

    Regression: from_common used to drop the field, so
    getattr(attn_metadata, "causal", False) in forward() was always False
    and every fa_rdna2 prefill ran non-causal — deterministic garbage
    end-to-end despite correct kernels.
    """
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.rdna_attn import RdnaAttentionMetadata

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
    meta = RdnaAttentionMetadata.from_common(cm)
    assert getattr(meta, "causal", None) is True


if __name__ == "__main__":
    import os
    report_only = os.environ.get("FA_RDNA2_TEST_REPORT_ONLY") == "1"
    results = []
    for layout in ("dense", "interleaved"):
        for sl in (26, 1024, 5000):
            results.append((f"prefill  D=256 sl={sl} {layout}",
                            _run(test_prefill_varlen_d256_writer_layout, sl,
                                 layout)))
    for kernel in ("short", "general"):
        results.append((f"prefill  D=128 {kernel}",
                        _run(test_prefill_d128_writer_layout, kernel)))
    for (D, H_q, H_kv, bs, layout) in ((256, 24, 4, 784, "dense"),
                                       (256, 24, 4, 784, "interleaved"),
                                       (128, 16, 4, 16, "dense")):
        for sl in (26, 1024, 5000):
            results.append((f"decode   D={D} sl={sl} {layout}",
                            _run(test_decode_writer_layout, D, H_q, H_kv, bs,
                                 layout, sl)))
    results.append(("prefill  D=256 multiseq",
                    _run(test_prefill_varlen_d256_multiseq)))
    results.append(("metadata causal", _run(test_metadata_carries_causal)))
    for kernel in ("general", "splitk", "short"):
        results.append((f"prefill  chunked-offset {kernel}",
                        _run(test_prefill_chunked_offset, kernel)))
    for name, res in results:
        print(f"{name}: {res}", flush=True)
    if any(r != "PASS" for _, r in results):
        if not report_only:
            raise SystemExit(1)
    else:
        print("ALL PASS")
