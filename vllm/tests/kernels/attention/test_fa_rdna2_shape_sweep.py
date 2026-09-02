# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Comprehensive shape sweep correctness harness for fa_rdna2 kernels.

Tests each public kernel wrapper from vllm.v1.attention.ops.fa_rdna2_backend
against a streaming fp32 reference across:
- HEAD_DIM = 128 and 256
- causal / noncausal
- sliding window (where supported)

Run via:
    HIP_VISIBLE_DEVICES=2,4 python tests/kernels/attention/test_fa_rdna2_shape_sweep.py
"""
import pathlib

import pytest
import torch
from torch.utils.cpp_extension import load_inline

# These tests only run on AMD RDNA2 hardware (gfx1030).
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and "gfx103" in torch.cuda.get_device_properties(0).gcnArchName),
    reason="Requires AMD RDNA2 (gfx1030) GPU",
)

_CU_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "csrc" / "rocm" / "fa_rdna2.cu"
)
assert _CU_PATH.is_file(), f"fa_rdna2.cu not found at {_CU_PATH}"
_CUDA_SRC = _CU_PATH.read_text()

_CPP_SRC = """
torch::Tensor fa_rdna2_decode_paged(torch::Tensor Q, torch::Tensor key_cache,
                                    torch::Tensor value_cache,
                                    torch::Tensor block_table,
                                    torch::Tensor seq_lens,
                                    int64_t block_size, int64_t kv_splits,
                                    int64_t sliding_window);
torch::Tensor fa_rdna2_prefill_paged_varlen(torch::Tensor Q, torch::Tensor key_cache,
                                            torch::Tensor value_cache,
                                            torch::Tensor block_table,
                                            torch::Tensor cu_query_lens,
                                            torch::Tensor seq_lens,
                                            int64_t block_size,
                                            int64_t causal,
                                            int64_t sliding_window);
torch::Tensor fa_rdna2_prefill_paged_varlen_short(torch::Tensor Q,
                                                  torch::Tensor key_cache,
                                                  torch::Tensor value_cache,
                                                  torch::Tensor block_table,
                                                  torch::Tensor cu_query_lens,
                                                  torch::Tensor seq_lens,
                                                  int64_t block_size,
                                                  int64_t causal,
                                                  int64_t sliding_window);
torch::Tensor fa_rdna2_prefill_paged_varlen_splitk(torch::Tensor Q,
                                                  torch::Tensor key_cache,
                                                  torch::Tensor value_cache,
                                                  torch::Tensor block_table,
                                                  torch::Tensor cu_query_lens,
                                                  torch::Tensor seq_lens,
                                                  int64_t block_size,
                                                  int64_t causal,
                                                  int64_t kv_splits,
                                                  int64_t sliding_window);
"""


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="test_fa_rdna2_shape_sweep",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=[
                "fa_rdna2_decode_paged",
                "fa_rdna2_prefill_paged_varlen",
                "fa_rdna2_prefill_paged_varlen_short",
                "fa_rdna2_prefill_paged_varlen_splitk",
            ],
            extra_cuda_cflags=["-O3", "-std=c++17", "--offload-arch=gfx1030"],
            verbose=False,
        )
    return _ext


def _make_paged_kv(num_blocks, H_kv, D, block_size, x_dim, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kc = torch.randn(num_blocks, H_kv, D // x_dim, block_size, x_dim,
                     device="cuda", dtype=torch.float16, generator=g)
    vc = torch.randn(num_blocks, H_kv, D // x_dim, block_size, x_dim,
                     device="cuda", dtype=torch.float16, generator=g)
    return kc, vc


def _reference_paged(Q, kc, vc, block_table, seq_lens, cu_query_lens, causal=False):
    """Streaming fp32 reference for varlen paged attention."""
    H_q = Q.shape[1]
    D = Q.shape[2]
    num_seqs = block_table.shape[0]
    H_kv = kc.shape[1]
    block_size = kc.shape[3]
    kv_group = H_q // H_kv
    O = torch.zeros_like(Q, dtype=torch.float32)
    for s in range(num_seqs):
        sl = int(seq_lens[s].item())
        max_blocks = block_table.shape[1]
        blocks = block_table[s, :max_blocks].tolist()
        K_rows = [kc[blocks[kv_pos // block_size], :, :,
                       kv_pos % block_size, :].permute(0, 1, 2).reshape(H_kv, D)
                  for kv_pos in range(sl)]
        V_rows = [vc[blocks[kv_pos // block_size], :, :,
                       kv_pos % block_size, :].permute(0, 1, 2).reshape(H_kv, D)
                  for kv_pos in range(sl)]
        K_s = torch.stack(K_rows, dim=0)
        V_s = torch.stack(V_rows, dim=0)
        q_start = int(cu_query_lens[s].item())
        q_end = int(cu_query_lens[s + 1].item())
        Qf = Q[q_start:q_end].float()
        Kf = K_s.repeat_interleave(kv_group, dim=1).float()
        Vf = V_s.repeat_interleave(kv_group, dim=1).float()
        scores = torch.einsum("qhd,khd->qhk", Qf, Kf) / (D ** 0.5)
        if causal:
            qi = torch.arange(q_end - q_start, device=Q.device)
            ki = torch.arange(sl, device=Q.device)
            qi_full = qi + q_start
            mask = ki.unsqueeze(0) <= qi_full.unsqueeze(1)
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        probs = scores.softmax(dim=-1)
        O[q_start:q_end] = torch.einsum("qhk,khd->qhd", probs, Vf)
    return O.half()


def _max_rel_err(out, ref):
    diff = (out.float() - ref.float()).abs()
    return (diff / (ref.float().abs() + 1e-3)).max().item()


# ===== Test cases =====



@pytest.mark.parametrize("N", [2048, 4096, 8192])
def test_prefill_varlen_short(N):
    """fa_rdna2_prefill_paged_varlen_short (HEAD=128): N<4096 path with swizzle."""
    ext = _get_ext()
    torch.manual_seed(0)
    H_q, H_kv, D, block_size, x_dim = 16, 4, 128, 16, 8
    max_blocks = (N + block_size - 1) // block_size
    kc, vc = _make_paged_kv(max_blocks, H_kv, D, block_size, x_dim, seed=N)
    block_table = torch.arange(0, max_blocks, device="cuda", dtype=torch.int32).view(1, max_blocks)
    seq_lens = torch.tensor([N], device="cuda", dtype=torch.int32)
    cu_query_lens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
    Q = torch.randn(N, H_q, D, device="cuda", dtype=torch.float16)
    out = ext.fa_rdna2_prefill_paged_varlen_short(
        Q, kc, vc, block_table, cu_query_lens, seq_lens, block_size, 0, 0)
    ref = _reference_paged(Q, kc, vc, block_table, seq_lens, cu_query_lens)
    assert _max_rel_err(out, ref) < 5e-3


@pytest.mark.parametrize("N", [4096, 8192])
def test_prefill_varlen_splitk(N):
    """fa_rdna2_prefill_paged_varlen_splitk (HEAD=128): N>=4096 path."""
    ext = _get_ext()
    torch.manual_seed(0)
    H_q, H_kv, D, block_size, x_dim = 16, 4, 128, 16, 8
    kv_splits = min(8, (N + 1023) // 1024)
    max_blocks = (N + block_size - 1) // block_size
    kc, vc = _make_paged_kv(max_blocks, H_kv, D, block_size, x_dim, seed=N)
    block_table = torch.arange(0, max_blocks, device="cuda", dtype=torch.int32).view(1, max_blocks)
    seq_lens = torch.tensor([N], device="cuda", dtype=torch.int32)
    cu_query_lens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
    Q = torch.randn(N, H_q, D, device="cuda", dtype=torch.float16)
    out = ext.fa_rdna2_prefill_paged_varlen_splitk(
        Q, kc, vc, block_table, cu_query_lens, seq_lens, block_size, 0, kv_splits, 0)
    ref = _reference_paged(Q, kc, vc, block_table, seq_lens, cu_query_lens)
    assert _max_rel_err(out, ref) < 5e-3


@pytest.mark.parametrize("N,causal", [(1024, True), (2048, True), (4096, True)])
def test_prefill_varlen_causal(N, causal):
    """Causal masking correctness for varlen prefill kernels."""
    ext = _get_ext()
    torch.manual_seed(0)
    H_q, H_kv, D, block_size, x_dim = 16, 4, 128, 16, 8
    max_blocks = (N + block_size - 1) // block_size
    kc, vc = _make_paged_kv(max_blocks, H_kv, D, block_size, x_dim, seed=N)
    block_table = torch.arange(0, max_blocks, device="cuda", dtype=torch.int32).view(1, max_blocks)
    seq_lens = torch.tensor([N], device="cuda", dtype=torch.int32)
    cu_query_lens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
    Q = torch.randn(N, H_q, D, device="cuda", dtype=torch.float16)
    if N < 4096:
        out = ext.fa_rdna2_prefill_paged_varlen_short(
            Q, kc, vc, block_table, cu_query_lens, seq_lens, block_size, 1, 0)
    else:
        kv_splits = min(8, (N + 1023) // 1024)
        out = ext.fa_rdna2_prefill_paged_varlen_splitk(
            Q, kc, vc, block_table, cu_query_lens, seq_lens, block_size, 1, kv_splits, 0)
    ref = _reference_paged(Q, kc, vc, block_table, seq_lens, cu_query_lens, causal=True)
    assert _max_rel_err(out, ref) < 5e-3


@pytest.mark.parametrize("N", [2048, 4096])
def test_prefill_varlen_head_256(N):
    """HEAD=256 varlen (Qwen3.5 wizardeur shape: H_q=24, H_kv=4)."""
    ext = _get_ext()
    torch.manual_seed(0)
    H_q, H_kv, D, block_size, x_dim = 24, 4, 256, 16, 8
    max_blocks = (N + block_size - 1) // block_size
    kc, vc = _make_paged_kv(max_blocks, H_kv, D, block_size, x_dim, seed=N)
    block_table = torch.arange(0, max_blocks, device="cuda", dtype=torch.int32).view(1, max_blocks)
    seq_lens = torch.tensor([N], device="cuda", dtype=torch.int32)
    cu_query_lens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
    Q = torch.randn(N, H_q, D, device="cuda", dtype=torch.float16)
    out = ext.fa_rdna2_prefill_paged_varlen(
        Q, kc, vc, block_table, cu_query_lens, seq_lens, block_size, 0, 0)
    ref = _reference_paged(Q, kc, vc, block_table, seq_lens, cu_query_lens)
    assert _max_rel_err(out, ref) < 5e-3


# ===== Sliding-window regression =====
#
# Bug: with sliding_window > 0, fa_rdna2_decode_paged returned NaN whenever
# some K/V splits were entirely outside the window. The all-masked split
# produced m_new = block_reduce_max(-INFINITY) = -INFINITY, then
# expf(-INFINITY - (-INFINITY)) = expf(NaN) = NaN propagated through the
# rest of the kernel. Fix: skip the online-softmax update when the entire
# block is masked (matches the prefill kernel's guard).
#
# These cases are sized so that with kv_splits splits, at least one full
# split is fully outside the window.


def _reference_paged_windowed(Q, kc, vc, block_table, seq_lens,
                              cu_query_lens, sliding_window):
    """Streaming fp32 reference that applies a sliding-window mask.

    Window semantics match fa_rdna2: for each query at position q, only
    K/V positions >= seq_len - sliding_window are kept. (Decode queries
    are at the END of the sequence, so this is the last `sliding_window`
    K/V positions.)
    """
    H_q = Q.shape[1]
    D = Q.shape[2]
    num_seqs = block_table.shape[0]
    H_kv = kc.shape[1]
    block_size = kc.shape[3]
    kv_group = H_q // H_kv
    O = torch.zeros_like(Q, dtype=torch.float32)
    for s in range(num_seqs):
        sl = int(seq_lens[s].item())
        blocks = block_table[s, :block_table.shape[1]].tolist()
        K_rows = [kc[blocks[kv_pos // block_size], :, :,
                       kv_pos % block_size, :].permute(0, 1, 2).reshape(H_kv, D)
                  for kv_pos in range(sl)]
        V_rows = [vc[blocks[kv_pos // block_size], :, :,
                       kv_pos % block_size, :].permute(0, 1, 2).reshape(H_kv, D)
                  for kv_pos in range(sl)]
        K_s = torch.stack(K_rows, dim=0)
        V_s = torch.stack(V_rows, dim=0)
        q_start = int(cu_query_lens[s].item())
        q_end = int(cu_query_lens[s + 1].item())
        Qf = Q[q_start:q_end].float()
        Kf = K_s.repeat_interleave(kv_group, dim=1).float()
        Vf = V_s.repeat_interleave(kv_group, dim=1).float()
        scores = torch.einsum("qhd,khd->qhk", Qf, Kf) / (D ** 0.5)
        if sliding_window > 0:
            ki = torch.arange(sl, device=Q.device)
            mask = ki >= (sl - sliding_window)
            scores = scores.masked_fill(~mask.view(1, 1, sl), float("-inf"))
        # If every position is masked for a query, softmax would NaN.
        # Guard the reference: clamp row max to a finite value if all -inf.
        all_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_masked, 0.0)
        probs = scores.softmax(dim=-1)
        O[q_start:q_end] = torch.einsum("qhk,khd->qhd", probs, Vf)
    return O.half()


@pytest.mark.parametrize("N,sliding_window", [
    (512, 64),
    (512, 128),
    (1024, 256),
])
@pytest.mark.parametrize("H_q,H_kv", [(4, 4), (40, 8)])
def test_decode_paged_sliding_window(N, sliding_window, H_q, H_kv):
    """Regression: sliding_window > 0 must not produce NaN in paged decode.

    Before the fix this returned NaN for every (N, sw) where at least one
    K/V split was fully outside the window.
    """
    ext = _get_ext()
    torch.manual_seed(0)
    D = 128
    block_size, x_dim, kv_splits = 16, 8, 8
    max_blocks = (N + block_size - 1) // block_size
    kc, vc = _make_paged_kv(max_blocks, H_kv, D, block_size, x_dim, seed=N)
    block_table = torch.arange(0, max_blocks, device="cuda", dtype=torch.int32).view(1, max_blocks)
    seq_lens = torch.tensor([N], device="cuda", dtype=torch.int32)
    Q = torch.randn(1, H_q, D, device="cuda", dtype=torch.float16)
    out = ext.fa_rdna2_decode_paged(
        Q, kc, vc, block_table, seq_lens, block_size, kv_splits, sliding_window)
    # The bug produced all-NaN output; assert no NaN, no Inf.
    assert not torch.isnan(out).any(), "decode_paged returned NaN with sliding_window > 0"
    assert not torch.isinf(out).any(), "decode_paged returned Inf with sliding_window > 0"
    # Correctness vs windowed fp32 reference.
    cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    ref = _reference_paged_windowed(Q, kc, vc, block_table, seq_lens, cu,
                                    sliding_window)
    # The last (only) query attends only to the last `sliding_window` K/V.
    assert _max_rel_err(out, ref) < 5e-3, (
        f"max_rel_err={_max_rel_err(out, ref)} for "
        f"N={N} sw={sliding_window} H_q={H_q} H_kv={H_kv}")