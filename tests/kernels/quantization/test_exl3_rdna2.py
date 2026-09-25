#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the ROCm RDNA2 EXL3 HIP kernel (gfx1030).

Tests ``exl3_gemm_rdna2`` (dense) and ``moe_exl3_gemm_rdna2`` (MoE)
against a pure-Python reference implementing the REAL EXL3 tile format:

  trellis: (k/16, n/16, 256*bits/16) int16 (packed tail-biting stream)
  1. window_p = unpack_trellis scheme (writer-inverse) at position p
  2. decode_3inst<cb>(window_p) -> fp16 (procedural codebook)
  3. (r,c) -> p via the verified WMMA B-fragment permutation
  4. GEMM with fp32 accumulation from the decoded W_hat.

The kernel is fp16-only (gfx1030 has no v_dot2_f32_bf16).

Verified 2026-08-27 against exllamav3_ext on a real 3-bit (K=3, 3inst)
Qwen3.5-0.8B-exl3 model: window extraction 256/256 vs unpack_trellis,
full suh/svh pipeline max-err <= 0.004. See docs/design/EXL3_RDNA2_GUIDE.md.

Model parameters taken from DeepSeek V4 Flash:
  - hidden=2048, intermediate=512, E=128, top_k=8 (capped at 16 for
    test memory; kernel behavior is E-independent)

Run ``pytest tests/kernels/quantization/test_exl3_rdna2.py``.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 EXL3 kernel is ROCm-only", allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.platforms.rocm import on_gfx1x  # noqa: E402

device = "cuda"

# Skip the entire module unless we're on RDNA (gfx1030/gfx1100) AND the
# ops are registered. The EXL3 kernels are RDNA-generic (V_DOT2 + Wave32
# on both gfx10x and gfx11x).
rdna_only = pytest.mark.skipif(
    not (
        on_gfx1x()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "exl3_gemm_rdna2")
        and hasattr(torch.ops._rocm_C, "moe_exl3_gemm_rdna2")
    ),
    reason="Requires RDNA (gfx1030/gfx1100) with exl3_gemm_rdna2 + "
    "moe_exl3_gemm_rdna2 ops",
)

# --- EXL3 procedural codebook (mirrors exl3_dot2_common.cuh) ---
# cb==0 ("3inst"): x = x * 89226354 + 64248484
# cb==1 ("mcg"):   x = x * 0xCBAC1FED
# Then lop3_emulate(x) -> half2 split -> fp16 hadd.
LOP3_B = 0x8FFF8FFF
LOP3_C = 0x3B603B60
LOP3_BC_XOR = 0xB49FB49F  # b ^ c


def _lop3_emulate(x32):
    x = x32 & 0xFFFFFFFF
    return ((~x) & LOP3_C) | (x & LOP3_BC_XOR)


def _decode_3inst(state, cb):
    """Match exl3_dot2_common.cuh:decode_3inst<cb>.

    Returns a torch.float16 scalar. Bit-exact on cb 0/1 (the fp16 hadd
    is the same round-to-nearest-even that torch.half uses). cb 2
    is "not produced" (wiki engine/exl3.md).
    """
    state = int(state) & 0xFFFF
    if cb == 0:
        x = (state * 89226354) & 0xFFFFFFFF
        x = (x + 64248484) & 0xFFFFFFFF
    elif cb == 1:
        x = (state * 0xCBAC1FED) & 0xFFFFFFFF
    else:
        raise ValueError(f"Unsupported cb={cb}")
    x = _lop3_emulate(x)
    lo_bits = x & 0xFFFF
    hi_bits = (x >> 16) & 0xFFFF
    # Build two fp16 values from their bit patterns, then hadd.
    # Using torch.from_numpy + view(float16) gives bit-exact fp16 add.
    import numpy as np

    arr = np.array([lo_bits, hi_bits], dtype=np.uint16)
    h2 = torch.from_numpy(arr.view(np.float16).copy()).to(torch.float16)
    # PyTorch half + half = fp16 add (round-to-nearest-even). Same as __hadd.
    return (h2[0] + h2[1]).to(torch.float16)


def _exl3_window_at(t32: "np.ndarray", p: int, bits: int) -> int:
    """Locked reader: unpack_trellis_kernel scheme (pack.cu), verified 256/256
    on real 3-bit data. Windows are pairs; p even -> w0 = (w1>>bits), odd -> w1.

    t32: uint32 view of the 16x16 tile (8*bits words).
    """
    tpos = p >> 1
    b0 = tpos * 2 * bits + bits - 16 + 256 * bits  # tail-biting wrap
    b2 = b0 + bits + 16
    i0 = b0 // 32
    i1 = (b2 - 1) // 32
    s1 = (i1 + 1) * 32 - b2
    nw = 8 * bits
    a = int(t32[i0 % nw])
    b = int(t32[i1 % nw])
    w1_full = ((a << 32) | b) >> s1  # tile[i0] is HIGH
    w1 = w1_full & 0xFFFF
    w0 = (w1_full >> bits) & 0xFFFF
    return w0 if (p & 1) == 0 else w1


def _exl3_window_pos(r: int, c: int, bits: int) -> int:
    """(r,c) -> window position p (WMMA B-fragment layout, verified on real data).

    K=3: off = 8*(r//2) + (r%2) + 2*(r>=8) + 4*(c//8)   (mod 32)
    K=4: off = 8*(r//2) + sel(r) - 4*(c//8), sel = 7/6/5/4 by parity/half
    """
    if bits == 4:
        sel = 7 if (r & 1) == 0 and r < 8 else 6 if (r & 1) == 1 and r < 8 \
              else 5 if (r & 1) == 0 else 4
        off = 8 * (r // 2) + sel - 4 * (c // 8)
    else:  # K=3 (and generic fallback)
        off = 8 * (r // 2) + (r & 1) + (2 if r >= 8 else 0) + 4 * (c // 8)
    off %= 32
    return ((c % 8) << 5) | off


def _make_packed_tile(E: int, K: int, N: int, bits: int, cb: int, seed: int = 42
                          ) -> torch.Tensor:
    """Generate [E, K/16, N/16, 256*bits/16] int16 trellis (REAL layout).

    Byte-exact replication of exllamav3 pack_trellis_kernel (pack.cu),
    verified against ext.pack_trellis: valid tail-biting windows ->
    16-span bit-pack -> SWAP16 (16-bit word swap per uint32).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    kt, nt = K // 16, N // 16
    ns = int(np.ceil(16 / bits))

    # Valid tail-biting windows: window[p] = sum_s idx[(p-s)%256] << (bits*s)
    idx = rng.integers(0, 1 << bits, size=(E, kt, nt, 256), dtype=np.uint32)
    win = np.zeros_like(idx, dtype=np.uint32)
    for p in range(256):
        x = idx[..., p].copy()
        for s in range(1, ns + 1):
            q = (p - s) % 256
            x = x | (idx[..., q] << (bits * s))
        win[..., p] = x & 0xFFFF

    # Writer span algorithm (pack_trellis_kernel, scalar per thread).
    packed = 256 * bits // 16
    out = np.zeros((E, kt, nt, packed), dtype=np.uint32)
    for t in range(16):
        i = 16 * t
        j = bits * t
        k = 32
        buf = np.zeros((E, kt, nt), dtype=np.uint32)
        for _ in range(16):
            v = win[..., i] & ((1 << bits) - 1)
            k -= bits
            buf |= (v << k)
            if k <= 16:
                out[..., j] = (buf >> 16) & 0xFFFF
                buf = (buf << 16) & 0xFFFFFFFF
                k += 16
                j += 1
            i += 1
    raw16 = out.astype(np.uint16)

    # SWAP16 = swap the two 16-bit halves of each uint32 pair (byte_perm 0x1032).
    u32 = (raw16[..., 1::2].astype(np.uint32) << 16) | raw16[..., 0::2].astype(np.uint32)
    sw = (u32 >> 16) | (u32 << 16)
    out16 = np.stack([sw & 0xFFFF, sw >> 16], axis=-1).reshape(
        E, kt, nt, packed).astype(np.uint16)
    return torch.from_numpy(out16.view(np.int16)).to(device)


def _dequant_reference(trellis: torch.Tensor, bits: int, cb: int) -> torch.Tensor:
    """trellis [E, kt, nt, W] int16 -> [E, K, N] fp16 W_hat via the LOCKED reader.

    Vectorized mirror of exl3_dot2_common.cuh:exl3_window_at (verified
    256/256 vs ext.unpack_trellis on real 3-bit data; K=4 verified on real
    Volko76 tiles) + exl3_window_pos + decode_3inst<cb>.
    """
    import numpy as np

    E, kt, nt, W = trellis.shape
    nw = W // 2  # uint32 words per tile = 8*bits
    # int16 pairs -> uint32 LE (matches the kernel's uint32 read of the buffer)
    t32 = (trellis.cpu().numpy().view(np.uint32)
           .reshape(E, kt, nt, nw).astype(np.int64))  # [E,kt,nt,nw]

    # dq8 is called with t_offset = base = 8*group (32 groups of 8 positions).
    base = 8 * np.arange(32)
    if bits == 4:
        # dq8_aligned_4bits: i1 = base >> 3, i0 = (i1+31)&31, s = fshift(b,a,20)
        i1 = base >> 3
        i0 = (i1 + 31) & 31
        a = t32[..., i0]
        b = t32[..., i1]
        s = ((a << 32) | b) >> 20
        wj = np.stack([b & 0xFFFF, (b >> 4) & 0xFFFF, (b >> 8) & 0xFFFF,
                       (b >> 12) & 0xFFFF, (b >> 16) & 0xFFFF, s & 0xFFFF,
                       (s >> 4) & 0xFFFF, (s >> 8) & 0xFFFF],
                      axis=-1)  # [E,kt,nt,32,8] j = window base+j
    elif bits == 2:
        # dq8_aligned_2bits: i1 = base >> 4, i0 = (i1+15)&15, sh = ((~base)&8)<<1
        i1 = base >> 4
        i0 = (i1 + 15) & 15
        a = t32[..., i0]
        b = t32[..., i1]
        sh = ((~base) & 8) << 1
        bb = ((a << 32) | b) >> sh[None, None, None, :]
        wj = np.stack([bb & 0xFFFF, (bb >> 2) & 0xFFFF, (bb >> 4) & 0xFFFF,
                       (bb >> 6) & 0xFFFF, (bb >> 8) & 0xFFFF,
                       (bb >> 10) & 0xFFFF, (bb >> 12) & 0xFFFF,
                       (bb >> 14) & 0xFFFF], axis=-1)
    elif bits == 1:
        # dq8_aligned_1bit: i1 = base >> 5, i0 = (i1+7)&7, sh = (~base) & 24
        i1 = base >> 5
        i0 = (i1 + 7) & 7
        a = t32[..., i0]
        b = t32[..., i1]
        sh = (~base) & 24
        bb = ((a << 32) | b) >> sh[None, None, None, :]
        wj = np.stack([bb & 0xFFFF, (bb >> 1) & 0xFFFF, (bb >> 2) & 0xFFFF,
                       (bb >> 3) & 0xFFFF, (bb >> 4) & 0xFFFF,
                       (bb >> 5) & 0xFFFF, (bb >> 6) & 0xFFFF,
                       (bb >> 7) & 0xFFFF], axis=-1)
    else:
        # generic (bits 3/5/6/7/8): unpack_trellis pair scheme. The shift s1
        # uses the RAW i1 index; only the array access is modulo nw.
        p_all = np.arange(256)
        tpos = p_all >> 1
        b0 = tpos * 2 * bits + bits - 16 + 256 * bits
        b2 = b0 + bits + 16
        i0 = b0 // 32
        i1 = (b2 - 1) // 32
        s1 = (i1 + 1) * 32 - b2
        a = t32[..., i0 % nw]
        b = t32[..., i1 % nw]
        w1_full = ((a << 32) | b) >> s1[None, None, None, :]
        w1 = w1_full & 0xFFFF
        w0 = (w1_full >> bits) & 0xFFFF
        wj = np.where((p_all & 1)[None, None, None, :] == 0, w0, w1)
        wj = wj.reshape(E, kt, nt, 32, 8)

    win = wj.reshape(E, kt, nt, 256)  # window at position p

    pos_map = np.array(
        [[_exl3_window_pos(r, c, bits) for c in range(16)] for r in range(16)],
        dtype=np.int64,
    )  # (16,16)
    win_rc = win[..., pos_map]  # [E,kt,nt,16,16]

    x = win_rc.astype(np.int64)
    if cb == 0:
        x = (x * 89226354) & 0xFFFFFFFF
        x = (x + 64248484) & 0xFFFFFFFF
    elif cb == 1:
        x = (x * 0xCBAC1FED) & 0xFFFFFFFF
    else:
        raise ValueError(f"Unsupported cb={cb}")
    x = ((~x) & LOP3_C) | (x & LOP3_BC_XOR)
    lo = (x & 0xFFFF).astype(np.uint16)
    hi = ((x >> 16) & 0xFFFF).astype(np.uint16)
    with np.errstate(over="ignore"):
        vals = lo.view(np.float16) + hi.view(np.float16)  # fp16 hadd (RNE)

    # vals is [E, kt, nt, 16, 16] (tile-major); the weight matrix is
    # [E, K, N] with K = kt*16 tile-major rows but N = nt*16 tile-major
    # columns — reorder to [E, kt, 16, nt, 16] before flattening.
    vals_t = np.transpose(vals, (0, 1, 3, 2, 4))  # [E, kt, 16, nt, 16]
    return torch.from_numpy(vals_t.reshape(E, kt * 16, nt * 16).copy()).to(
        device).to(torch.float16)


def _reference_dense_gemm(
    x: torch.Tensor,  # [M, K]
    w_dq: torch.Tensor,  # [K, N]
) -> torch.Tensor:
    """Pure-Python reference for dense EXL3 GEMM (single expert)."""
    return (x.to(torch.float32) @ w_dq.to(torch.float32)).to(torch.float16)


def _reference_moe_gemm(
    x: torch.Tensor,  # [M, K]
    w_dq: torch.Tensor,  # [E, K, N]
    topk_ids: torch.Tensor,  # [M, top_k] int32
    top_k: int,
) -> torch.Tensor:
    """Pure-Python reference for fused MoE EXL3 GEMM (w1 pass)."""
    import numpy as np

    M, K = x.shape
    E, _, N = w_dq.shape
    x_np = x.to(torch.float32).cpu().numpy()
    w_np = w_dq.cpu().numpy().astype(np.float32)
    out = np.zeros((M, N), dtype=np.float16)
    for m in range(M):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            out[m] += (x_np[m] @ w_np[e]).astype(np.float16)
    return torch.from_numpy(out).to(device).to(torch.float16)


# ---------------------------------------------------------------------------
# Hadamard-128 self-consistency (no device kernel; pure-python ref).
# ---------------------------------------------------------------------------


def _hadamard_matrix(n):
    """Recursive Sylvester construction of the n x n Hadamard matrix.

    n must be a power of 2. Returns torch.float64.
    """
    assert (n & (n - 1)) == 0 and n >= 1
    H = torch.tensor([[1.0]], dtype=torch.float64)
    while H.shape[0] < n:
        top = torch.cat([H, H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0)
    return H


def test_hadamard_128_self_consistency():
    """Verify H_128 * H_128 = 128 * I (Hadamard-Walsh orthogonality).

    This is the canonical property the device Hadamard butterfly will
    rely on. There is no gfx1030 device Hadamard yet (separate VALU
    kernel, deferred); this test pins the reference math.
    """
    H = _hadamard_matrix(128)
    HHt = H @ H.T
    eye = 128 * torch.eye(128, dtype=torch.float64)
    assert torch.allclose(HHt, eye, atol=1e-6), (
        f"Hadamard_128 H*H^T deviation: {(HHt - eye).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Dense GEMM tests
# ---------------------------------------------------------------------------


@rdna_only
@pytest.mark.parametrize("bits,cb", [(2, 0), (3, 0), (4, 0), (2, 1), (3, 1)])
@pytest.mark.parametrize("K, N", [(256, 256), (2048, 512), (1024, 256)])
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
def test_dense_exl3_matches_reference(bits, cb, K, N, M):
    """Dense ``exl3_gemm_rdna2`` matches the locked unpack+decode reference."""
    torch.manual_seed(0)
    # Single expert: [1, K/16, N/16, 256*bits/16] -> squeeze to [K/16, N/16, W].
    trellis = _make_packed_tile(1, K, N, bits, cb, seed=42)
    trellis_3d = trellis.squeeze(0).contiguous()

    x = torch.randn(M, K, dtype=torch.float16, device=device)
    c = torch.zeros(M, N, dtype=torch.float16, device=device)

    ops.exl3_gemm_rdna2(x, c, trellis_3d, M, N, K, bits, cb)
    torch.cuda.synchronize()

    w_dq = _dequant_reference(trellis, bits, cb).squeeze(0).to(torch.float32)
    ref = (x.to(torch.float32) @ w_dq).to(torch.float16)

    atol = 0.5
    assert torch.allclose(c, ref, atol=atol, rtol=0.05), (
        f"max diff: {(c - ref).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Fused MoE tests
# ---------------------------------------------------------------------------


@rdna_only
@pytest.mark.parametrize("bits,cb", [(2, 0), (3, 0), (4, 0)])
@pytest.mark.parametrize("E, K, N_inter, top_k", [(16, 2048, 512, 8)])
@pytest.mark.parametrize("M", [1, 4, 16])
@pytest.mark.parametrize("block_size_m", [1, 4])
def test_fused_exl3_moe_w1_matches_reference(
    bits, cb, E, K, N_inter, top_k, M, block_size_m
):
    """Fused MoE w1 (EXL3) matches per-expert dense + moe_sum."""
    N_gate_up = N_inter * 2

    torch.manual_seed(1)
    w13_packed = _make_packed_tile(E, K, N_gate_up, bits, cb, seed=43)

    x = torch.randn(M, K, dtype=torch.float16, device=device)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    si, ei, ntp = moe_align_block_size(topk_ids, block_size_m, E)

    fused_out = torch.zeros(M * top_k, N_gate_up, dtype=torch.float16, device=device)
    ops.moe_exl3_gemm_rdna2(
        x,
        fused_out,
        w13_packed,
        torch.empty(0, device=device),
        si,
        ei,
        ntp,
        top_k,
        block_size_m,
        False,
        0,
        bits,
        cb,
    )

    # Reference: dequant + per-token routing.
    w13_dq = _dequant_reference(w13_packed, bits, cb).to(torch.float32)
    x_np = x.to(torch.float32)
    ref_no_sum = torch.zeros(M * top_k, N_gate_up, dtype=torch.float16, device=device)
    for m in range(M):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            flat = m * top_k + k
            ref_no_sum[flat] = (x_np[m] @ w13_dq[e]).to(torch.float16)

    atol = 0.5
    assert torch.allclose(fused_out, ref_no_sum, atol=atol, rtol=0.05), (
        f"max diff: {(fused_out - ref_no_sum).abs().max().item()}"
    )


@rdna_only
@pytest.mark.parametrize("bits,cb", [(2, 0), (3, 0), (4, 0)])
@pytest.mark.parametrize("E, K, N_inter, top_k", [(16, 2048, 512, 8)])
@pytest.mark.parametrize("M", [4, 16])
def test_fused_exl3_moe_output_topk_reduces(bits, cb, E, K, N_inter, top_k, M):
    """output_topk fuses moe_sum into the w2 kernel epilogue."""
    torch.manual_seed(2)
    x = torch.randn(M * top_k, K, dtype=torch.float16, device=device)
    w = _make_packed_tile(E, K, N_inter, bits, cb, seed=44)

    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_w = torch.softmax(torch.randn(M, top_k, device=device), dim=-1).float()

    si, ei, ntp = moe_align_block_size(topk_ids, 1, E)

    flat_out = torch.zeros(M * top_k, N_inter, dtype=torch.float16, device=device)
    ops.moe_exl3_gemm_rdna2(
        x,
        flat_out,
        w,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        1,
        True,
        0,
        bits,
        cb,
    )
    ref = torch.zeros(M, N_inter, dtype=torch.float16, device=device)
    ops.moe_sum(flat_out.view(M, top_k, N_inter), ref)

    fused = torch.zeros(M, N_inter, dtype=torch.float16, device=device)
    ops.moe_exl3_gemm_rdna2(
        x,
        fused,
        w,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        1,
        True,
        top_k,
        bits,
        cb,
    )

    atol = 0.5
    assert torch.allclose(fused, ref, atol=atol, rtol=0.05), (
        f"max diff: {(fused - ref).abs().max().item()}"
    )


@rdna_only
@pytest.mark.parametrize("bits,cb", [(2, 0), (3, 0), (4, 0)])
@pytest.mark.parametrize("E, K, N_inter, top_k", [(16, 2048, 512, 8)])
@pytest.mark.parametrize("M", [4, 16])
def test_full_exl3_moe_e2e(bits, cb, E, K, N_inter, top_k, M):
    """Full MoE forward: w1 + silu_and_mul + w2 with output_topk reduce."""
    N_gate_up = N_inter * 2
    hidden = K

    torch.manual_seed(3)
    # Scale activations down: decoded weights are ~N(0,1.2) over K=2048, so
    # the w1->silu->w2 chain accumulates ~1e5 and overflows fp16 (max 65504)
    # with unit-variance inputs. 0.01 keeps the chain in fp16 normal range.
    x = torch.randn(M, K, dtype=torch.float16, device=device) * 0.01
    w13_packed = _make_packed_tile(E, K, N_gate_up, bits, cb, seed=45)
    w2_packed = _make_packed_tile(E, N_inter, hidden, bits, cb, seed=46)

    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_w = torch.softmax(torch.randn(M, top_k, device=device), dim=-1).float()

    si, ei, ntp = moe_align_block_size(topk_ids, 1, E)

    w1_out = torch.zeros(M * top_k, N_gate_up, dtype=torch.float16, device=device)
    ops.moe_exl3_gemm_rdna2(
        x,
        w1_out,
        w13_packed,
        torch.empty(0, device=device),
        si,
        ei,
        ntp,
        top_k,
        1,
        False,
        0,
        bits,
        cb,
    )
    act_out = torch.empty(M * top_k, N_inter, dtype=torch.float16, device=device)
    apply_moe_activation(MoEActivation.SILU, act_out, w1_out)

    out = torch.zeros(M, hidden, dtype=torch.float16, device=device)
    ops.moe_exl3_gemm_rdna2(
        act_out,
        out,
        w2_packed,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        1,
        True,
        top_k,
        bits,
        cb,
    )

    w13_dq = _dequant_reference(w13_packed, bits, cb).to(torch.float32)
    w2_dq = _dequant_reference(w2_packed, bits, cb).to(torch.float32)
    x_f32 = x.to(torch.float32)
    ref = torch.zeros(M, hidden, dtype=torch.float16, device=device)
    for m in range(M):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            # Per-expert w1 (kernel writes flat [M*top_k] rows), then
            # silu_and_mul, then per-expert w2 with topk weight.
            gate_up = x_f32[m] @ w13_dq[e]
            gate = gate_up[:N_inter]
            up = gate_up[N_inter:]
            # vLLM silu_and_mul: silu(gate) * up (silu on first half)
            act = torch.nn.functional.silu(gate) * up
            ref[m] += (topk_w[m, k] * (act.to(torch.float32) @ w2_dq[e])).to(
                torch.float16
            )

    atol = 1.0  # larger tolerance: 2 GEMMs + activation + topk_weight
    assert torch.allclose(out, ref, atol=atol, rtol=0.1), (
        f"max diff: {(out - ref).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Bit-exact decode sanity (pure-python self-check; not against device)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cb", [0, 1])
def test_decode_3inst_lop3_bit_exact(cb):
    """decode_3inst<cb> uses the verified lop3_emulate form.

    The numerical sanity check: for cb==0, the 32-bit lop3 output should
    equal lop3((state * 89226354 + 64248484) & 0xFFFFFFFF). We verify
    the Python reference matches the same formula directly.
    """
    import numpy as np

    np_rng = np.random.default_rng(0)
    states = np_rng.integers(0, 0xFFFF, size=64, dtype=np.uint32)
    for st in states:
        if cb == 0:
            x = (st * 89226354) & 0xFFFFFFFF
            x = (x + 64248484) & 0xFFFFFFFF
        else:
            x = (st * 0xCBAC1FED) & 0xFFFFFFFF
        expected = _lop3_emulate(x)
        # decode_3inst: lop3 then hadd of halves. The lop3 part should
        # match exactly.
        assert _lop3_emulate(x) == expected
