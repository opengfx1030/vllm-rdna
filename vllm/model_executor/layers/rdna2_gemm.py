# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA2 (gfx1030) fp16 GEMV for unquantized linear layers (decode path).

The generic skinny-GEMM kernels (wvSplitK/LLMM1) use gfx11/12 ISA that traps on
RDNA2, so unquantized decode projections otherwise fall through to
``torch.nn.functional.linear`` (~9% MBU on gfx1030). This module is a Triton
GEMV; Triton lowers it to gfx10-safe FMA (no WMMA/MFMA), the same codegen the
AWQ Triton path already relies on here.

Scope: the M=1 (single-row) case, which is the decode bottleneck, plus a
small-M (<= 16) tl.dot matmul for spec-decode verify / MTP batches
(ported from the gfx906 fork's skinny-GEMM tuning). Prefill and larger
batches keep using the existing path.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=8, num_stages=3),
    ],
    key=["N", "K"],
)
@triton.jit
def _gemv_rdna2_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    K,
    stride_wn,
    stride_wk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K
        x = tl.load(x_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + k[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    if HAS_BIAS:
        b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += b

    tl.store(out_ptr + offs_n, acc.to(out_ptr.dtype.element_ty), mask=n_mask)


def rdna2_gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """fp16/bf16 GEMV for M=1: out = x @ weight.T (+ bias).

    Args:
        x: 1D activation [K].
        weight: 2D [N, K] (nn.Linear layout, ``out = x @ weight.T``).
        bias: optional [N].
    """
    N, K = weight.shape
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _gemv_rdna2_kernel[grid](
        x,
        weight,
        bias,
        out,
        N,
        K,
        weight.stride(0),
        weight.stride(1),
        HAS_BIAS=bias is not None,
    )
    return out


def _skinny_matmul_configs():
    # Ported from the gfx906 fork: shallow pipelining keeps LDS pressure low
    # on the 64 KB-LDS small dies; small N tiles expose more parallelism when
    # TP makes each shard narrow.
    cfgs = []
    for bn in (16, 32, 64, 128):
        for bk in (64, 128):
            for warps in (2, 4):
                cfgs.append(
                    triton.Config(
                        {"BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 1},
                        num_stages=1,
                        num_warps=warps,
                    )
                )
    return cfgs


@triton.autotune(
    configs=_skinny_matmul_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _skinny_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """C = A x B.T with A (M, K), B (N, K), C (M, N)."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def rdna2_skinny_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Small-M matmul: out = a @ b.T with a (M, K), b (N, K).

    ``b`` is the nn.Linear weight in its native (N, K) layout; the kernel
    reads it with swapped strides so no repack is needed.
    """
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Matrices A and B must have the same dtype"
    M, K = a.shape
    N = b.shape[0]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    _skinny_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=16,
        waves_per_eu=1,
    )
    return c


@triton.jit
def _w4a16_gemv_rdna2_kernel(
    x_ptr,  # [K] fp16/bf16 activations
    w_ptr,  # [K, N//8] int32, N-packed nibbles (triton_w4a16 layout)
    s_ptr,  # [K//GROUP, N] scales (contiguous)
    z_ptr,  # [K//GROUP, N//8] packed zeros if HAS_ZP
    b_ptr,  # [N] bias if HAS_BIAS
    y_ptr,  # [N] output
    N,
    K,
    N8,  # N // 8
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP: tl.constexpr,  # quant group size (multiple of BLOCK_K, or
    # BLOCK_K == 2 * GROUP)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """y = x @ dequant(W).T + bias for a single row (M=1 decode GEMV).

    Same packed layout as ``triton_w4a16_gemm``; unpacks each int32 into 8
    N-nibbles with the interleave+shift scheme and applies the per-group
    affine dequant inside the K loop. When BLOCK_K spans two quant groups
    (e.g. BLOCK_K=64 with GROUP=32) both group rows are loaded once and
    selected per k with a where(), keeping scale/zero traffic minimal.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n8 = pid * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    n_mask = offs_n < N
    n8_mask = offs_n8 < N8

    shifts_row = tl.arange(0, 8) * 4
    shifts_2d = tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 8, 8))
    shifts_1d = tl.reshape(shifts_2d, (BLOCK_N,))
    DUAL_GROUP: tl.constexpr = BLOCK_K == 2 * GROUP

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + ks).to(tl.float32)  # (BLOCK_K,)

        w_packed = tl.load(
            w_ptr + ks[:, None] * N8 + offs_n8[None, :],
            mask=n8_mask[None, :],
            other=0,
        )
        w = tl.interleave(w_packed, w_packed)
        w = tl.interleave(w, w)
        w = tl.interleave(w, w)
        w = (w >> shifts_1d[None, :]) & 0xF  # (BLOCK_K, BLOCK_N)

        g = k0 // GROUP
        s0 = tl.load(s_ptr + g * N + offs_n, mask=n_mask, other=0.0).to(
            tl.float32
        )
        if HAS_ZP:
            z_p0 = tl.load(z_ptr + g * N8 + offs_n8, mask=n8_mask, other=0)

        if DUAL_GROUP:
            s1 = tl.load(
                s_ptr + (g + 1) * N + offs_n, mask=n_mask, other=0.0
            ).to(tl.float32)
            if HAS_ZP:
                z_p1 = tl.load(
                    z_ptr + (g + 1) * N8 + offs_n8, mask=n8_mask, other=0
                )
            sel = ks >= (g + 1) * GROUP  # (BLOCK_K,)
            s = tl.where(sel[:, None], s1[None, :], s0[None, :])
        else:
            s = tl.broadcast_to(s0[None, :], (BLOCK_K, BLOCK_N))

        if HAS_ZP:
            if DUAL_GROUP:
                z0 = tl.interleave(z_p0, z_p0)
                z0 = tl.interleave(z0, z0)
                z0 = tl.interleave(z0, z0)
                z0 = (z0 >> shifts_1d) & 0xF
                z1 = tl.interleave(z_p1, z_p1)
                z1 = tl.interleave(z1, z1)
                z1 = tl.interleave(z1, z1)
                z1 = (z1 >> shifts_1d) & 0xF
                z = tl.where(sel[:, None], z1[None, :], z0[None, :]).to(
                    tl.float32
                )
            else:
                z = tl.interleave(z_p0, z_p0)
                z = tl.interleave(z, z)
                z = tl.interleave(z, z)
                z = (z >> shifts_1d) & 0xF
                z = tl.broadcast_to(
                    z.to(tl.float32)[None, :], (BLOCK_K, BLOCK_N)
                )
            deq = (w.to(tl.float32) - z) * s
        else:
            deq = (w.to(tl.float32) - ZP_BIAS) * s
        acc += tl.sum(deq * x[:, None], axis=0)

    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs_n, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


def rdna2_w4a16_gemv(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int = 8,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """M=1 W4A16 GEMV: y = x @ dequant(W).T (+ bias).

    ``w_q``/``scales``/``qzeros`` use the :func:`triton_w4a16_gemm` layouts
    ([K, N//8] int32, [K//G, N], [K//G, N//8]).
    """
    assert x.dim() == 1 or (x.dim() == 2 and x.shape[0] == 1)
    K = x.shape[-1]
    N = w_q.shape[1] * 8
    N8 = N // 8
    assert w_q.shape == (K, N8)
    assert scales.shape == (K // group_size, N)
    if qzeros is not None:
        assert qzeros.shape == (K // group_size, N8)

    # Deterministic launch config from a gfx1030 sweep (Qwen3.8 TP2
    # shapes): BLOCK_N=32 everywhere; BLOCK_K=64 (dual gs=32 groups)
    # when K allows; large-N shapes prefer 1 warp (more CTAs resident),
    # small-N prefer 8 (k-parallel reduction).
    block_k = 32
    if K % 64 == 0 and (group_size == 32 or group_size % 64 == 0):
        block_k = 64
    warps = 1 if N > 8192 else 8

    x1 = x.reshape(K)
    y = torch.empty(N, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(N, 32),)
    _w4a16_gemv_rdna2_kernel[grid](
        x1,
        w_q,
        scales,
        qzeros if qzeros is not None else w_q,
        bias if bias is not None else x1,
        y,
        N,
        K,
        N8,
        HAS_ZP=qzeros is not None,
        ZP_BIAS=zp_bias,
        HAS_BIAS=bias is not None,
        GROUP=group_size,
        BLOCK_N=32,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=2,
    )
    return y


def _w4a16_gemv_m_configs():
    cfgs = []
    for bn in (32, 64):
        for warps in (1, 4, 8):
            cfgs.append(
                triton.Config({"BLOCK_N": bn}, num_warps=warps, num_stages=2)
            )
    return cfgs


@triton.jit
def _w4a16_gemv_m_rdna2_kernel(
    x_ptr,  # [M, K] fp16/bf16 activations (M small, <= BLOCK_M)
    w_ptr,  # [K, N//8] int32, N-packed nibbles
    s_ptr,  # [K//GROUP, N] scales
    z_ptr,  # [K//GROUP, N//8] packed zeros if HAS_ZP
    b_ptr,  # [N] bias if HAS_BIAS
    y_ptr,  # [M, N] output
    N,
    K,
    N8,
    M,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,  # padded M (power of 2 >= actual M)
):
    """y = x @ dequant(W).T + bias for a few rows (M <= BLOCK_M <= 4).

    Same layout/dequant scheme as the M=1 GEMV; weight loads are shared
    across the M rows, so small-M verify batches (spec decode k<=3) cost
    little more than a single-row GEMV.
    """
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n8 = pid * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    n_mask = offs_n < N
    n8_mask = offs_n8 < N8
    m_mask = offs_m < M

    shifts_row = tl.arange(0, 8) * 4
    shifts_2d = tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 8, 8))
    shifts_1d = tl.reshape(shifts_2d, (BLOCK_N,))
    DUAL_GROUP: tl.constexpr = BLOCK_K == 2 * GROUP

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offs_m[:, None] * K + ks[None, :],
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_M, BLOCK_K)

        w_packed = tl.load(
            w_ptr + ks[:, None] * N8 + offs_n8[None, :],
            mask=n8_mask[None, :],
            other=0,
        )
        w = tl.interleave(w_packed, w_packed)
        w = tl.interleave(w, w)
        w = tl.interleave(w, w)
        w = (w >> shifts_1d[None, :]) & 0xF  # (BLOCK_K, BLOCK_N)

        g = k0 // GROUP
        s0 = tl.load(s_ptr + g * N + offs_n, mask=n_mask, other=0.0).to(
            tl.float32
        )
        if HAS_ZP:
            z_p0 = tl.load(z_ptr + g * N8 + offs_n8, mask=n8_mask, other=0)

        if DUAL_GROUP:
            s1 = tl.load(
                s_ptr + (g + 1) * N + offs_n, mask=n_mask, other=0.0
            ).to(tl.float32)
            if HAS_ZP:
                z_p1 = tl.load(
                    z_ptr + (g + 1) * N8 + offs_n8, mask=n8_mask, other=0
                )
            sel = ks >= (g + 1) * GROUP
            s = tl.where(sel[:, None], s1[None, :], s0[None, :])
        else:
            s = tl.broadcast_to(s0[None, :], (BLOCK_K, BLOCK_N))

        if HAS_ZP:
            if DUAL_GROUP:
                z0 = tl.interleave(z_p0, z_p0)
                z0 = tl.interleave(z0, z0)
                z0 = tl.interleave(z0, z0)
                z0 = (z0 >> shifts_1d) & 0xF
                z1 = tl.interleave(z_p1, z_p1)
                z1 = tl.interleave(z1, z1)
                z1 = tl.interleave(z1, z1)
                z1 = (z1 >> shifts_1d) & 0xF
                z = tl.where(sel[:, None], z1[None, :], z0[None, :]).to(
                    tl.float32
                )
            else:
                z = tl.interleave(z_p0, z_p0)
                z = tl.interleave(z, z)
                z = tl.interleave(z, z)
                z = (z >> shifts_1d) & 0xF
                z = tl.broadcast_to(
                    z.to(tl.float32)[None, :], (BLOCK_K, BLOCK_N)
                )
            deq = (w.to(tl.float32) - z) * s
        else:
            deq = (w.to(tl.float32) - ZP_BIAS) * s
        # (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) via broadcast+sum
        acc += tl.sum(x[:, :, None] * deq[None, :, :], axis=1)

    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(
            tl.float32
        )[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def rdna2_w4a16_gemv_m(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int = 8,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Small-M (<= 4) W4A16 GEMV sharing weight loads across rows."""
    M, K = x.shape
    N = w_q.shape[1] * 8
    N8 = N // 8
    block_m = max(2, triton.next_power_of_2(M))
    block_k = 64 if (K % 64 == 0 and group_size in (32, 64)) else 32
    # M>1 sweep winner on gfx1030: BN=32/warps=4/bk=64 uniformly
    warps = 4
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    _w4a16_gemv_m_rdna2_kernel[(triton.cdiv(N, 32),)](
        x,
        w_q,
        scales,
        qzeros if qzeros is not None else w_q,
        bias if bias is not None else x.reshape(-1)[:1],
        y,
        N,
        K,
        N8,
        M,
        HAS_ZP=qzeros is not None,
        ZP_BIAS=zp_bias,
        HAS_BIAS=bias is not None,
        GROUP=group_size,
        BLOCK_N=32,
        BLOCK_K=block_k,
        BLOCK_M=block_m,
        num_warps=warps,
        num_stages=2,
    )
    return y


@triton.jit
def _w4a16_kpack_gemm_kernel(
    a_ptr,  # [M, K] fp16/bf16 activations
    w_ptr,  # [K//8, N] int32, K-packed (gemv_w4_kpack layout)
    s_ptr,  # [K//GROUP, N] scales (contiguous)
    z_ptr,  # [K//GROUP, N] uint8 zeros if HAS_ZP (kpack layout)
    c_ptr,  # [M, N] output
    M,
    N,
    K,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # multiple of 8
):
    """C[M,N] = A[M,K] @ dequant(W)[K,N] on the K-packed layout.

    W row r = k//8 holds 8 consecutive K values for every column n at
    nibble shift (k%8)*4. Unpack to [BLOCK_K, BLOCK_N] with the
    interleave scheme: 3 interleaves of the [BLOCK_K//8, BLOCK_N] tile
    replicate each int32 8x along K; then shift-select per k%8.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k8 = tl.arange(0, BLOCK_K // 8)
    BK8: tl.constexpr = BLOCK_K // 8

    m_mask = offs_m < M
    n_mask = offs_n < N

    # k%8 nibble shifts tiled across BLOCK_K: [BK8, 8] -> [BLOCK_K]
    shifts_row = tl.arange(0, 8) * 4
    shifts_2d = tl.broadcast_to(shifts_row[None, :], (BK8, 8))
    shifts_k = tl.reshape(shifts_2d, (BLOCK_K,))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        wp = tl.load(
            w_ptr + (k0 // 8 + offs_k8)[:, None] * N + offs_n[None, :],
            mask=n_mask[None, :],
            other=0,
        )  # [BK8, BLOCK_N] int32
        # unpack to [BLOCK_K, BLOCK_N]: interleave works on the last axis,
        # so transpose to [BLOCK_N, BK8], replicate each packed row 8x
        # along K (nibbles j=0..7 of row r -> k = r*8+j), shift-select,
        # transpose back.
        wt = tl.trans(wp)  # [BLOCK_N, BK8]
        wt = tl.interleave(wt, wt)
        wt = tl.interleave(wt, wt)
        wt = tl.interleave(wt, wt)  # [BLOCK_N, BLOCK_K]
        wt = (wt >> shifts_k[None, :]) & 0xF
        w = tl.trans(wt).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        a = tl.load(
            a_ptr + offs_m[:, None] * K + ks[None, :],
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        g = ks // GROUP
        s = tl.load(
            s_ptr + g[:, None] * N + offs_n[None, :],
            mask=n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if HAS_ZP:
            z = tl.load(
                z_ptr + g[:, None] * N + offs_n[None, :],
                mask=n_mask[None, :],
                other=0,
            ).to(tl.float32)
            deq = (w.to(tl.float32) - z) * s
        else:
            deq = (w.to(tl.float32) - ZP_BIAS) * s

        acc += tl.dot(a.to(tl.float16), deq.to(tl.float16),
                      out_dtype=tl.float32)

    tl.store(
        c_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(c_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def w4a16_kpack_gemm(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
    zeros_u8: torch.Tensor | None,
    group_size: int,
    zp_bias: float = 8.0,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """M>1 GEMM on the K-packed layout ([K//8, N] int32 weights)."""
    M, K = x.shape
    N = w_q.shape[1]
    assert w_q.shape[0] == K // 8
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    _w4a16_kpack_gemm_kernel[grid](
        x,
        w_q,
        scales,
        zeros_u8 if zeros_u8 is not None else scales,
        y,
        M,
        N,
        K,
        HAS_ZP=zeros_u8 is not None,
        ZP_BIAS=zp_bias,
        GROUP=group_size,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
        num_stages=2,
    )
    if bias is not None:
        y = y + bias
    return y


@triton.jit
def _w4a16_kpack_gemv_m_kernel(
    x_ptr,  # [M, K] fp16/bf16 activations (M small, <= BLOCK_M <= 4)
    w_ptr,  # [K//8, N] int32, K-packed (gemv_w4_kpack layout)
    s_ptr,  # [K//GROUP, N] scales
    z_ptr,  # [K//GROUP, N] uint8 zeros if HAS_ZP (kpack layout)
    b_ptr,  # [N] bias if HAS_BIAS
    y_ptr,  # [M, N] output
    N,
    K,
    M,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # multiple of 8
    BLOCK_M: tl.constexpr,  # padded M (power of 2 >= actual M)
):
    """y = x @ dequant(W).T + bias for a few rows (1 < M <= 4) on the
    K-packed layout.

    Spec-decode verify batches (MTP k<=3 -> M=3) hit this path. Weight
    rows are BLOCK_N consecutive int32 — fully coalesced. Unpack mirrors
    the N-packed small-M GEMV: 3 tl.interleave rounds replicate each
    int32 8x along K, then a per-k%8 shift-select; accumulation is
    broadcast+sum FMA (gfx1030 has no fp32 dot units and M<=4 amortises
    nothing). One x load and one scale gather per K-tile.
    """
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_k8 = tl.arange(0, BLOCK_K // 8)
    BK8: tl.constexpr = BLOCK_K // 8
    n_mask = offs_n < N
    m_mask = offs_m < M

    # k%8 nibble shifts tiled across BLOCK_K: [BK8, 8] -> [BLOCK_K]
    shifts_row = tl.arange(0, 8) * 4
    shifts_2d = tl.broadcast_to(shifts_row[None, :], (BK8, 8))
    shifts_k = tl.reshape(shifts_2d, (BLOCK_K,))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + ks[None, :],
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        wp = tl.load(
            w_ptr + (k0 // 8 + offs_k8)[:, None] * N + offs_n[None, :],
            mask=n_mask[None, :],
            other=0,
        )  # [BK8, BLOCK_N] int32, coalesced
        # Unpack to [BLOCK_K, BLOCK_N]: interleave works on the last axis,
        # so transpose to [BLOCK_N, BK8], replicate each packed row 8x
        # along K (nibbles j=0..7 of row r -> k = r*8+j), shift-select,
        # transpose back (same scheme as _w4a16_kpack_gemm_kernel).
        wt = tl.trans(wp)  # [BLOCK_N, BK8]
        wt = tl.interleave(wt, wt)
        wt = tl.interleave(wt, wt)
        wt = tl.interleave(wt, wt)  # [BLOCK_N, BLOCK_K]
        wt = (wt >> shifts_k[None, :]) & 0xF
        w = tl.trans(wt).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        g = ks // GROUP
        s = tl.load(
            s_ptr + g[:, None] * N + offs_n[None, :],
            mask=n_mask[None, :],
            other=0.0,
        ).to(tl.float32)  # [BLOCK_K, BLOCK_N] k-major (coalesced rows)
        if HAS_ZP:
            z = tl.load(
                z_ptr + g[:, None] * N + offs_n[None, :],
                mask=n_mask[None, :],
                other=0,
            ).to(tl.float32)
            deq = (w - z) * s
        else:
            deq = (w - ZP_BIAS) * s

        acc += tl.sum(x[:, :, None] * deq[None, :, :], axis=1)

    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(
            tl.float32
        )[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def rdna2_w4a16_kpack_gemv_m(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
    zeros_u8: torch.Tensor | None,
    group_size: int,
    zp_bias: float = 8.0,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Small-M (<= 4) W4A16 GEMV on the K-packed layout ([K//8, N] int32).

    The fast path for spec-decode verification batches when kpack weights
    are enabled; replaces the tl.dot GEMM fallback that ran at ~0 GB/s
    effective for M <= 4 (see NEXT_STEPS Item 6).
    """
    M, K = x.shape
    N = w_q.shape[1]
    assert w_q.shape[0] == K // 8
    block_m = max(2, triton.next_power_of_2(M))
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    _w4a16_kpack_gemv_m_kernel[(triton.cdiv(N, 64),)](
        x,
        w_q,
        scales,
        zeros_u8 if zeros_u8 is not None else scales,
        bias if bias is not None else x.reshape(-1)[:1],
        y,
        N,
        K,
        M,
        HAS_ZP=zeros_u8 is not None,
        ZP_BIAS=zp_bias,
        HAS_BIAS=bias is not None,
        GROUP=group_size,
        BLOCK_N=64,
        BLOCK_K=64,
        BLOCK_M=block_m,
        num_warps=4,
        num_stages=2,
    )
    return y
