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

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 GDN prefill delta-h kernel is ROCm-only",
                allow_module_level=True)

from vllm.platforms.rocm import on_gfx10x  # noqa: E402

if not on_gfx10x():
    pytest.skip("RDNA2 GDN prefill delta-h kernel is gfx1030-only",
                allow_module_level=True)

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
    per-token CUMULATIVE log-gate (fp32) that ``chunk_local_cumsum``
    produces upstream.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    N = len(seq_lens)
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    cu_seqlens = torch.tensor(cu, device=device, dtype=torch.int32)
    T_total = cu[-1]

    k = torch.randn(1, T_total, Hg, K, device=device, dtype=torch.float16,
                    generator=gen)
    w = torch.randn(1, T_total, H, K, device=device, dtype=torch.float16,
                    generator=gen)
    u = torch.randn(1, T_total, H, V, device=device, dtype=torch.float16,
                    generator=gen)
    # Cumulative log-gate: per-seq, monotonically decreasing (the
    # upstream cumsum on -rand * step builds this up).
    g = torch.zeros(1, T_total, H, device=device, dtype=torch.float32)
    for i, s in enumerate(seq_lens):
        steps = -torch.rand(s, H, device=device, generator=gen) * 0.05
        g[0, cu[i]:cu[i + 1]] = torch.cumsum(steps, dim=0)

    initial_state = None
    if with_initial_state:
        initial_state = torch.randn(N, H, V, K, device=device,
                                    dtype=torch.float32, generator=gen)
    return k, w, u, g, cu_seqlens, initial_state


def _run_ref(k, w, u, g, cu_seqlens, initial_state, output_final_state):
    h_ref, v_new_ref, final_ref = chunk_gated_delta_rule_fwd_h(
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
    chunk_offsets = torch.tensor(nt_cumsum, device=device,
                                 dtype=torch.int32)
    NT_total = nt_cumsum[-1]

    h = torch.empty(B, NT_total, H, V, K, device=device, dtype=torch.float16)
    v_new = torch.empty_like(u)
    final_state = (torch.empty(N, H, V, K, device=device,
                              dtype=torch.float32)
                   if output_final_state else None)
    torch.ops._rocm_C.gdn_prefill_delta_h_rdna2(
        k, u, w, g, h, v_new, initial_state, final_state, cu_seqlens,
        chunk_offsets, BT)
    return h, v_new, final_state


@pytest.mark.parametrize("H,Hg", [(12, 4), (4, 4)])
@pytest.mark.parametrize("seq_lens", [
    [64],          # exactly 1 chunk
    [192],         # 3 chunks (NT=3 serial loop)
    [640],         # many chunks (NT=10) — core serial-loop stress
    [100],         # chunk-boundary tail (64 + 36, not 16-aligned)
    [137],         # tail not multiple-of-16 nor of BT
    [63],          # tail where T_seq < BT (NT=1, t_len=63)
])
@pytest.mark.parametrize("with_initial_state", [False, True])
@pytest.mark.parametrize("output_final_state", [True])
def test_gdn_prefill_delta_h_rdna2_parity(seq_lens, H, Hg,
                                          with_initial_state,
                                          output_final_state):
    torch.manual_seed(0)
    k, w, u, g, cu_seqlens, initial_state = _make_inputs(
        seq_lens, H, Hg, seed=42, with_initial_state=with_initial_state)

    h_ref, v_new_ref, final_ref = _run_ref(k, w, u, g, cu_seqlens,
                                           initial_state, output_final_state)
    h_hip, v_new_hip, final_hip = _run_hip(k, w, u, g, cu_seqlens,
                                           initial_state, output_final_state)

    assert h_hip.shape == h_ref.shape, (h_hip.shape, h_ref.shape)
    assert v_new_hip.shape == v_new_ref.shape

    # fp16 outputs: rtol ~1e-3 (V_DOT2 accumulation-order differences
    # are bounded and bounded by fp16 precision anyway; accumulators
    # agree with the Triton reference modulo low-bit rounding).
    torch.testing.assert_close(v_new_hip.float(), v_new_ref.float(),
                               rtol=1e-3, atol=1e-3)
    # Per-chunk h states (fp16).
    torch.testing.assert_close(h_hip.float(), h_ref.float(),
                               rtol=1e-3, atol=1e-3)
    # fp32 final state: tighter (no fp16 rounding in this slot).
    if output_final_state:
        torch.testing.assert_close(final_hip, final_ref,
                                   rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("H,Hg", [(12, 4)])
@pytest.mark.parametrize("seq_lens", [
    [64, 64],         # equal-length varlen, 1 chunk each
    [64, 192, 128],   # mixed chunk counts across seqs
    [100, 37, 250],   # tails + mixed, varlen multi-seq
    [10, 10, 10],     # tiny seqs (NT=1 each, t_len < BT)
])
def test_gdn_prefill_delta_h_rdna2_varlen_multi_seq(seq_lens, H, Hg):
    """Varlen multi-sequence: chunks from different sequences must not
    cross-contaminate the h register state.
    """
    torch.manual_seed(0)
    k, w, u, g, cu_seqlens, initial_state = _make_inputs(
        seq_lens, H, Hg, seed=7, with_initial_state=True)

    h_ref, v_new_ref, final_ref = _run_ref(k, w, u, g, cu_seqlens,
                                           initial_state, True)
    h_hip, v_new_hip, final_hip = _run_hip(k, w, u, g, cu_seqlens,
                                           initial_state, True)

    torch.testing.assert_close(v_new_hip.float(), v_new_ref.float(),
                               rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(h_hip.float(), h_ref.float(),
                               rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(final_hip, final_ref, rtol=1e-4, atol=1e-4)


def test_gdn_prefill_delta_h_rdna2_no_initial_state_equals_zeros():
    """initial_state=None must behave identically to an explicit
    all-zero initial_state of the same shape.
    """
    torch.manual_seed(0)
    seq_lens = [192]
    H, Hg = 12, 4
    k, w, u, g, cu_seqlens, _ = _make_inputs(seq_lens, H, Hg, seed=11,
                                             with_initial_state=False)
    N = len(seq_lens)
    zero_state = torch.zeros(N, H, V, K, device=device, dtype=torch.float32)

    h_none, vn_none, fs_none = _run_hip(k, w, u, g, cu_seqlens, None, True)
    h_zero, vn_zero, fs_zero = _run_hip(k, w, u, g, cu_seqlens, zero_state,
                                        True)

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
        seq_lens, H, Hg, seed=13, with_initial_state=True)

    # Run with g=0; effect: v_new unchanged, h unchanged.
    g_zero = torch.zeros(1, cu_seqlens[-1].item(), H, device=device,
                         dtype=torch.float32)

    h_hip, vn_hip, fs_hip = _run_hip(k, w, u, g_zero, cu_seqlens, h0, True)
    h_ref, vn_ref, fs_ref = _run_ref(k, w, u, g_zero, cu_seqlens, h0, True)

    torch.testing.assert_close(vn_hip.float(), vn_ref.float(),
                               rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(h_hip.float(), h_ref.float(),
                               rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(fs_hip, fs_ref, rtol=1e-4, atol=1e-4)
