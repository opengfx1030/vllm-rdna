#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical parity tests for the ROCm RDNA2 GDN prefill delta-h kernel.

Compares ``torch.ops._rocm_C.gdn_prefill_delta_h_rdna2``
(csrc/rocm/gdn_prefill_delta_h_rdna2.cu) against the Triton reference
``chunk_gated_delta_rule_fwd_h``
(vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py).

The kernel is the serial inter-chunk recurrence for Qwen3.5/3.6 GDN
prefill. The multi-chunk cases are the core risk (the serial
h-dependency chain makes any order-of-operations mistake amplify over
chunks), so chunk counts, chunk-boundary tails, varlen multi-seq, and
H/HV configs are all parametrized.

Build/registration is gfx1030-only (orchestrator-side); tests skip on
other architectures.

Run `pytest tests/kernels/quantization/test_gdn_prefill_delta_h_rdna2.py`.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "RDNA2 GDN prefill delta-h kernel is ROCm-only", allow_module_level=True
    )

from vllm.platforms.rocm import on_gfx10x  # noqa: E402

if not on_gfx10x():
    pytest.skip(
        "RDNA2 GDN prefill delta-h kernel is gfx1030-only", allow_module_level=True
    )

# Triggers registration of torch.ops._rocm_C.*
import vllm._rocm_C  # noqa: F401,E402

from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (  # noqa: E402
    chunk_gated_delta_rule_fwd_h,
)

device = "cuda"

K = 128  # head_k_dim
V = 128  # head_v_dim
BT = 64  # FLA_CHUNK_SIZE


def _make_inputs(seq_lens, H, Hg, seed, with_initial_state):
    """Build synthetic k/w/u/g (+ optional initial_state) for a varlen batch.

    ``seq_lens`` is a list of per-sequence token counts (flattened B=1
    layout, which is the standard FLA varlen convention). g is the
    CHUNK-LOCAL cumulative log-gate (fp32) that ``chunk_local_cumsum``
    produces upstream: the cumsum restarts every BT tokens.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    N = len(seq_lens)
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    cu_seqlens = torch.tensor(cu, device=device, dtype=torch.int32)
    T_total = cu[-1]

    # l2-normalised k/w. With unnormalised N(0,1) keys and weights the
    # inter-chunk operator (I - k^T w) has norm ~O(100), so |h| grows ~30x per
    # chunk (22 -> 1350 -> 1.6e4 -> fp16 inf/NaN by chunk 5) and the Triton
    # reference itself returns NaN from 4 chunks on, which makes the parity
    # comparison meaningless. Normalised k/w keep the recurrence bounded and
    # match the magnitudes the production kernel actually processes (cf. the
    # _l2norm helper in test_gdn_prefill_o_rdna2.py).
    k = F.normalize(
        torch.randn(1, T_total, Hg, K, device=device, generator=gen).float(),
        dim=-1,
    ).to(torch.float16)
    w = F.normalize(
        torch.randn(1, T_total, H, K, device=device, generator=gen).float(),
        dim=-1,
    ).to(torch.float16)
    u = torch.randn(1, T_total, H, V, device=device, dtype=torch.float16, generator=gen)
    # Chunk-local cumulative log-gate: monotonically decreasing within each
    # chunk, restarting at 0 for every chunk (chunk_local_cumsum's contract).
    # A cumsum running across the whole sequence would make exp(g_last) the
    # decay accumulated since token 0 rather than across the current chunk,
    # breaking the recurrence the kernel implements.
    pad = (-T_total) % BT
    increments = torch.rand(1, T_total + pad, H, device=device, generator=gen) * 0.05
    g = (
        (-increments)
        .view(1, -1, BT, H)
        .cumsum(dim=2)
        .view(1, -1, H)[:, :T_total]
        .contiguous()
    )

    initial_state = None
    if with_initial_state:
        initial_state = torch.randn(
            N, H, V, K, device=device, dtype=torch.float32, generator=gen
        )
    return k, w, u, g, cu_seqlens, initial_state


def _run_ref(k, w, u, g, cu_seqlens, initial_state, output_final_state):
    # The reference is annotated as returning two tensors but returns three.
    h_ref, v_new_ref, final_ref = chunk_gated_delta_rule_fwd_h(  # type: ignore[misc]
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=BT,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
    )
    return h_ref, v_new_ref, final_ref


def _run_hip(k, w, u, g, cu_seqlens, initial_state, output_final_state):
    B, T, Hg, _ = k.shape
    H = u.shape[2]
    N = len(cu_seqlens) - 1
    # Per-seq chunk counts, then the flat-per-seq-begin chunk_offsets.
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    nt_per_seq = [(s + BT - 1) // BT for s in lens]
    nt_cumsum = [0]
    for n in nt_per_seq:
        nt_cumsum.append(nt_cumsum[-1] + n)
    chunk_offsets = torch.tensor(nt_cumsum, device=device, dtype=torch.int32)
    NT_total = nt_cumsum[-1]

    h = torch.empty(B, NT_total, H, V, K, device=device, dtype=torch.float16)
    v_new = torch.empty_like(u)
    # The state slot follows the cache dtype: fp32 by default, fp16 for an
    # --mamba-ssm-cache-dtype float16 cache. The kernel is templated on it and
    # requires the initial and final slots to agree.
    final_state = (
        torch.empty(
            N,
            H,
            V,
            K,
            device=device,
            dtype=initial_state.dtype if initial_state is not None else torch.float32,
        )
        if output_final_state
        else None
    )
    torch.ops._rocm_C.gdn_prefill_delta_h_rdna2(
        k, u, w, g, h, v_new, initial_state, final_state, cu_seqlens, chunk_offsets, BT
    )
    return h, v_new, final_state


@pytest.mark.parametrize("H,Hg", [(12, 4), (4, 4)])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [64],  # exactly 1 chunk
        [192],  # 3 chunks (NT=3 serial loop)
        [640],  # many chunks (NT=10) — core serial-loop stress
        [100],  # chunk-boundary tail (64 + 36, not 16-aligned)
        [137],  # tail not multiple-of-16 nor of BT
        [63],  # tail where T_seq < BT (NT=1, t_len=63)
    ],
)
@pytest.mark.parametrize("with_initial_state", [False, True])
@pytest.mark.parametrize("output_final_state", [True])
def test_gdn_prefill_delta_h_rdna2_parity(
    seq_lens, H, Hg, with_initial_state, output_final_state
):
    torch.manual_seed(0)
    k, w, u, g, cu_seqlens, initial_state = _make_inputs(
        seq_lens, H, Hg, seed=42, with_initial_state=with_initial_state
    )

    h_ref, v_new_ref, final_ref = _run_ref(
        k, w, u, g, cu_seqlens, initial_state, output_final_state
    )
    h_hip, v_new_hip, final_hip = _run_hip(
        k, w, u, g, cu_seqlens, initial_state, output_final_state
    )

    assert h_hip.shape == h_ref.shape, (h_hip.shape, h_ref.shape)
    assert v_new_hip.shape == v_new_ref.shape

    # fp16 outputs: rtol ~1e-3 (V_DOT2 accumulation-order differences
    # are bounded and bounded by fp16 precision anyway; accumulators
    # agree with the Triton reference modulo low-bit rounding).
    torch.testing.assert_close(
        v_new_hip.float(), v_new_ref.float(), rtol=1e-3, atol=1e-3
    )
    # Per-chunk h states (fp16).
    torch.testing.assert_close(h_hip.float(), h_ref.float(), rtol=1e-3, atol=1e-3)
    # fp32 final state. NOTE the slot is fp32, but its VALUE is not fp32-clean:
    # the reference rounds the gated b_v to fp16 inside chunk_delta_h.py, so the
    # state carries fp16 quantization. Measured HIP-vs-reference gaps are <= 1
    # fp16 ulp (typically 1.5e-4 .. 8.8e-4; 1.6e-2 at |h|~31, where one fp16 ulp
    # IS 1.6e-2). Against an fp64 oracle of the same algorithm HIP is the MORE
    # accurate side (6.8e-4 vs Triton's 2.5e-3), and with g == 0 the two agree
    # bit-exactly. So use the same tolerance as h/v_new, not a tighter one.
    if output_final_state:
        torch.testing.assert_close(final_hip, final_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("H,Hg", [(12, 4), (4, 4)])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [64],  # exactly 1 chunk
        [192],  # 3 chunks (NT=3 serial loop)
        [100],  # chunk-boundary tail (64 + 36, not 16-aligned)
        [137],  # tail not multiple-of-16 nor of BT
        [63],  # tail where T_seq < BT (NT=1, t_len=63)
    ],
)
def test_gdn_prefill_delta_h_rdna2_fp16_state_parity(seq_lens, H, Hg):
    """An fp16 SSM state must behave like the fp32 path fed the same values.

    The recurrence runs in fp32 registers either way; the fp16 instantiation
    only rounds the h0 load and the final-state store. So the per-chunk h and
    the v_new must equal the fp32 run bit for bit, and the final state must
    match it to within the fp16 rounding of the value it stores.
    """
    torch.manual_seed(0)
    k, w, u, g, cu_seqlens, initial_state = _make_inputs(
        seq_lens, H, Hg, seed=42, with_initial_state=True
    )
    state16 = initial_state.to(torch.float16)

    # The fp32 instantiation fed the widened fp16 state: same recurrence.
    h32, v_new32, final32 = _run_hip(k, w, u, g, cu_seqlens, state16.float(), True)
    # The fp16 instantiation: same recurrence, fp16 state slot.
    h16, v_new16, final16 = _run_hip(k, w, u, g, cu_seqlens, state16, True)

    assert final16.dtype == torch.float16
    torch.testing.assert_close(h16.float(), h32.float(), rtol=0, atol=0)
    torch.testing.assert_close(v_new16.float(), v_new32.float(), rtol=0, atol=0)
    # The final state is the fp32 value rounded to fp16 (0.5 ulp of fp16).
    torch.testing.assert_close(final16.float(), final32.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("H,Hg", [(12, 4)])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [64, 64],  # equal-length varlen, 1 chunk each
        [64, 192, 128],  # mixed chunk counts across seqs
        [100, 37, 250],  # tails + mixed, varlen multi-seq
        [10, 10, 10],  # tiny seqs (NT=1 each, t_len < BT)
    ],
)
def test_gdn_prefill_delta_h_rdna2_varlen_multi_seq(seq_lens, H, Hg):
    """Varlen multi-sequence: chunks from different sequences must not
    cross-contaminate the h register state.
    """
    torch.manual_seed(0)
    k, w, u, g, cu_seqlens, initial_state = _make_inputs(
        seq_lens, H, Hg, seed=7, with_initial_state=True
    )

    h_ref, v_new_ref, final_ref = _run_ref(k, w, u, g, cu_seqlens, initial_state, True)
    h_hip, v_new_hip, final_hip = _run_hip(k, w, u, g, cu_seqlens, initial_state, True)

    # v_new is fp16 too; same fp16-ulp bound as h/final_state below.
    _ulpv = 2.0 ** (math.floor(math.log2(max(v_new_ref.abs().max().item(), 1.0))) - 10)
    torch.testing.assert_close(
        v_new_hip.float(), v_new_ref.float(), rtol=1e-3, atol=2.0 * _ulpv
    )
    # h is fp16, so a flat atol=1e-3 cannot hold once |h| grows: at |h|~31 one
    # fp16 ulp IS 1.56e-2, and this varlen case measures exactly that. Bound the
    # comparison by the fp16 ulp of the largest |h| (2 ulps), same as final_state.
    _ulp = 2.0 ** (math.floor(math.log2(max(h_ref.abs().max().item(), 1.0))) - 10)
    torch.testing.assert_close(h_hip.float(), h_ref.float(), rtol=1e-3, atol=2.0 * _ulp)
    # See the fp16-ulp note above: the fp32 slot's value still carries fp16
    # quantization from the reference's gated b_v.
    # fs is fp32, but its VALUE passed through the reference's fp16-quantised
    # gated b_v, so agreement is bounded by the fp16 ulp of the largest |h| --
    # NOT by an fp32 bound. At |h|~31 that ulp is 2**-6 = 1.56e-2, which is
    # exactly what the [100,37,250] varlen case measures. Bound it at 2 ulps.
    _ulp = 2.0 ** (math.floor(math.log2(max(final_ref.abs().max().item(), 1.0))) - 10)
    torch.testing.assert_close(final_hip, final_ref, rtol=1e-3, atol=2.0 * _ulp)


def test_gdn_prefill_delta_h_rdna2_no_initial_state_equals_zeros():
    """initial_state=None must behave identically to an explicit
    all-zero initial_state of the same shape.
    """
    torch.manual_seed(0)
    seq_lens = [192]
    H, Hg = 12, 4
    k, w, u, g, cu_seqlens, _ = _make_inputs(
        seq_lens, H, Hg, seed=11, with_initial_state=False
    )
    N = len(seq_lens)
    zero_state = torch.zeros(N, H, V, K, device=device, dtype=torch.float32)

    h_none, vn_none, fs_none = _run_hip(k, w, u, g, cu_seqlens, None, True)
    h_zero, vn_zero, fs_zero = _run_hip(k, w, u, g, cu_seqlens, zero_state, True)

    torch.testing.assert_close(vn_none, vn_zero)
    torch.testing.assert_close(h_none, h_zero)
    torch.testing.assert_close(fs_none, fs_zero)


def test_gdn_prefill_delta_h_rdna2_no_g_equal_no_g():
    """If g is the all-zero tensor (degenerate: decay == 1 everywhere),
    the recurrence must reduce to the no-gating form. Sanity check that
    the USE_G gating pathway is consistent across implementations by
    running the no-g path against the WITH-g path with zero g.
    """
    torch.manual_seed(0)
    seq_lens = [128]
    H, Hg = 4, 4
    k, w, u, _, cu_seqlens, h0 = _make_inputs(
        seq_lens, H, Hg, seed=13, with_initial_state=True
    )

    # Run with g=0; effect: v_new unchanged, h unchanged.
    g_zero = torch.zeros(
        1, cu_seqlens[-1].item(), H, device=device, dtype=torch.float32
    )

    h_hip, vn_hip, fs_hip = _run_hip(k, w, u, g_zero, cu_seqlens, h0, True)
    h_ref, vn_ref, fs_ref = _run_ref(k, w, u, g_zero, cu_seqlens, h0, True)

    torch.testing.assert_close(vn_hip.float(), vn_ref.float(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(h_hip.float(), h_ref.float(), rtol=1e-3, atol=1e-3)
    # Same fp16-ulp bound as the other final-state checks: fs is an fp32 slot
    # holding a value the reference quantised to fp16.
    _ulp = 2.0 ** (math.floor(math.log2(max(fs_ref.abs().max().item(), 1.0))) - 10)
    torch.testing.assert_close(fs_hip, fs_ref, rtol=1e-3, atol=2.0 * _ulp)


def _bits(x):
    return x.view(torch.int16 if x.dtype == torch.float16 else torch.int32)


def _scaled(k, w):
    # Scale k and w down so the long recurrences stay finite (the stock
    # synthetic inputs overflow after ~12 chunks, and NaN payloads are not
    # comparable bit for bit).
    return (k.float() * 0.05).half(), (w.float() * 0.05).half()


@pytest.mark.parametrize("variant", ["1", "2", "3", "4"])
@pytest.mark.parametrize("H,Hg", [(12, 4), (4, 4)])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [2048],  # 32 chunks, the production single-sequence step
        [64],
        [63],
        [100],
        [137],
        [640],
        [1],  # one valid row in a 64-row chunk
        [300, 1000, 748],  # varlen, no sequence starts on a chunk boundary
        [64] * 8,
        [5, 7, 2036],
        [65, 129, 193],
    ],
)
@pytest.mark.parametrize("with_initial_state", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16])
def test_gdn_prefill_delta_h_rdna2_pipe_bitwise(
    monkeypatch, variant, H, Hg, seq_lens, with_initial_state, state_dtype
):
    """VLLM_RDNA_GDN_DH_PIPE=1..4 (16-byte staging, prefetch, DPP reduction,
    LDS-staged v_new, optional pair-interleaved k) must reproduce the default
    kernel bit for bit."""
    if not with_initial_state and state_dtype == torch.float16:
        pytest.skip("the state dtype only matters with an initial state")
    torch.manual_seed(0)
    k, w, u, g, cu_seqlens, initial_state = _make_inputs(
        seq_lens, H, Hg, seed=sum(seq_lens) + H, with_initial_state=with_initial_state
    )
    k, w = _scaled(k, w)
    if initial_state is not None:
        initial_state = initial_state.to(state_dtype)

    monkeypatch.setenv("VLLM_RDNA_GDN_DH_PIPE", "0")
    ref = _run_hip(k, w, u, g, cu_seqlens, initial_state, True)
    monkeypatch.setenv("VLLM_RDNA_GDN_DH_PIPE", variant)
    out = _run_hip(k, w, u, g, cu_seqlens, initial_state, True)

    for name, r, o in zip(("h", "v_new", "final_state"), ref, out):
        assert torch.isfinite(r.float()).all(), name
        assert torch.equal(_bits(r), _bits(o)), name


def test_gdn_prefill_delta_h_rdna2_pipe_unaligned_inputs_fall_back(monkeypatch):
    """The pipelined kernels need 16-byte aligned k/u/w/v_new; anything else
    must silently run the default kernel and still give the same result."""
    torch.manual_seed(0)
    H, Hg = 12, 4
    k, w, u, g, cu_seqlens, initial_state = _make_inputs(
        [200], H, Hg, seed=3, with_initial_state=True
    )
    k, w = _scaled(k, w)
    flat = torch.empty(k.numel() + 1, device=device, dtype=torch.float16)
    flat[1:].copy_(k.flatten())
    k_off = flat[1:].view_as(k)
    assert k_off.data_ptr() % 16 != 0

    monkeypatch.setenv("VLLM_RDNA_GDN_DH_PIPE", "0")
    ref = _run_hip(k, w, u, g, cu_seqlens, initial_state, True)
    for variant in ("1", "2", "3", "4"):
        monkeypatch.setenv("VLLM_RDNA_GDN_DH_PIPE", variant)
        out = _run_hip(k_off, w, u, g, cu_seqlens, initial_state, True)
        for r, o in zip(ref, out):
            assert torch.equal(_bits(r), _bits(o))
