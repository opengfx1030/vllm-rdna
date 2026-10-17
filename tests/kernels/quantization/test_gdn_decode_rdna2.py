#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical parity tests for the ROCm RDNA2 GDN packed decode kernel.

Compares ``torch.ops.vllm.gdn_decode_rdna2`` (csrc/rocm/gdn_decode_rdna2.cu)
against the Triton reference
``fused_recurrent_gated_delta_rule_packed_decode`` on random tensors,
including the out-of-range ``ssm_state_indices`` zero-output guard.

The kernel is only built/registered for gfx1030; tests skip elsewhere.

Run `pytest tests/kernels/quantization/test_gdn_decode_rdna2.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 GDN decode kernel is ROCm-only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx10x  # noqa: E402

if not on_gfx10x():
    pytest.skip("RDNA2 GDN decode kernel is gfx1030-only", allow_module_level=True)

from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_packed_decode,
)

device = "cuda"

K = 128  # head_k_dim; the HIP kernel requires NK == 1
V = 128
SOFTPLUS_THRESHOLD = 20.0


def _make_inputs(
    B: int,
    H: int,
    HV: int,
    num_blocks: int,
    seed: int,
    a_log_dtype: torch.dtype = torch.float32,
    dt_bias_dtype: torch.dtype = torch.float32,
):
    gen = torch.Generator(device=device).manual_seed(seed)
    qkv_dim = 2 * H * K + HV * V
    mixed_qkv = torch.randn(
        B, qkv_dim, device=device, dtype=torch.float16, generator=gen
    )
    a = torch.randn(B, HV, device=device, dtype=torch.float16, generator=gen)
    b = torch.randn(B, HV, device=device, dtype=torch.float16, generator=gen)
    # vLLM casts dt_bias to the model dtype on load while A_log stays
    # fp32, so all dtype combinations must work.
    A_log = torch.randn(HV, device=device, dtype=torch.float32, generator=gen).to(
        a_log_dtype
    )
    dt_bias = torch.randn(HV, device=device, dtype=torch.float32, generator=gen).to(
        dt_bias_dtype
    )
    state0 = torch.randn(
        num_blocks, HV, V, K, device=device, dtype=torch.float32, generator=gen
    )
    idx = torch.randperm(num_blocks, device=device, generator=gen)[:B].to(torch.int32)
    return mixed_qkv, a, b, A_log, dt_bias, state0, idx


def _run_ref(mixed_qkv, a, b, A_log, dt_bias, state0, idx, scale):
    state = state0.clone()
    out = torch.zeros(
        mixed_qkv.shape[0], 1, state0.shape[1], V, device=device, dtype=torch.float16
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state,
        out=out,
        ssm_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
    )
    return out, state


def _run_hip(mixed_qkv, a, b, A_log, dt_bias, state0, idx, scale):
    state = state0.clone()
    out = torch.zeros(
        mixed_qkv.shape[0], 1, state0.shape[1], V, device=device, dtype=torch.float16
    )
    torch.ops._rocm_C.gdn_decode_rdna2(
        mixed_qkv, a, b, A_log, dt_bias, out, state, idx, scale, True
    )
    return out, state


@pytest.mark.parametrize("B", [1, 4, 16])
@pytest.mark.parametrize("H,HV", [(4, 12), (2, 4)])
@pytest.mark.parametrize(
    "a_log_dtype,dt_bias_dtype",
    [
        (torch.float32, torch.float32),
        (torch.float16, torch.float16),
        (torch.float32, torch.float16),
    ],
)
def test_gdn_decode_rdna2_parity(B, H, HV, a_log_dtype, dt_bias_dtype):
    torch.manual_seed(0)
    num_blocks = max(2 * B, 8)
    mixed_qkv, a, b, A_log, dt_bias, state0, idx = _make_inputs(
        B,
        H,
        HV,
        num_blocks,
        seed=42,
        a_log_dtype=a_log_dtype,
        dt_bias_dtype=dt_bias_dtype,
    )
    scale = K**-0.5

    out_ref, state_ref = _run_ref(mixed_qkv, a, b, A_log, dt_bias, state0, idx, scale)
    out_hip, state_hip = _run_hip(mixed_qkv, a, b, A_log, dt_bias, state0, idx, scale)

    out_err = (out_hip.float() - out_ref.float()).abs().max().item()
    used = state_ref[idx.long()]
    state_err = (state_hip[idx.long()] - used).abs().max().item()
    scale_ref = out_ref.float().abs().max().item()
    assert out_err < 2e-2 * max(scale_ref, 1.0), (
        f"output mismatch: max abs err {out_err} (ref max {scale_ref})"
    )
    assert state_err < 2e-2, f"state mismatch: max abs err {state_err}"


def test_gdn_decode_rdna2_invalid_state_idx_zeroes_output():
    # state_idx outside [0, num_blocks) must zero the output and leave
    # the state untouched (NULL_BLOCK_ID collision guard).
    B, H, HV, num_blocks = 2, 2, 4, 4
    mixed_qkv, a, b, A_log, dt_bias, state0, idx = _make_inputs(
        B, H, HV, num_blocks, seed=7
    )
    idx[0] = -1
    idx[1] = num_blocks  # one past the end
    scale = K**-0.5

    out_hip, state_hip = _run_hip(mixed_qkv, a, b, A_log, dt_bias, state0, idx, scale)
    assert out_hip.abs().max().item() == 0.0
    torch.testing.assert_close(state_hip, state0)


# =============================================================================
# GDN speculative-decoding verify (torch.ops._rocm_C.gdn_spec_verify_rdna2).
#
# The MTP spec-verify step of fused_sigmoid_gating_delta_rule_update_kernel:
# varlen + continuous batching + spec decoding + in-place final state. The HIP
# kernel walks every token of a sequence in registers and stores the state to
# slot t of the request's index row.
#
# Tolerance: the fp32 state agrees to <= 1 ulp (the two kernels reduce the
# 128-wide K axis in different orders -- Triton uses sizePerThread=[1,4] with a
# DPP and shared-memory tree, the HIP kernel 8 lanes x 16 elements with a
# shuffle butterfly). Every other op matches bit-for-bit: exp/log go through
# the same llvm.exp2.f32 / llvm.log.f32 lowering, rsqrt is __frsqrt_rn
# (llvm.amdgcn.rsq.f32) and the divisions are IEEE. The fp16 output is
# bit-identical except ~1 element in 2000, by 1 fp16 ulp.
# =============================================================================

SPEC_H, SPEC_HV = 4, 12  # TP4: 16/4 key heads, 48/4 value heads


def _make_spec(n_req, lens, accepted, blocks=None, seed=0, a_dtype=torch.float32,
               d_dtype=torch.float32, state_dtype=torch.float32, a_3d=False):
    g = torch.Generator(device=device).manual_seed(seed)
    T = sum(lens)
    q = torch.randn(1, T, SPEC_H, K, device=device, dtype=torch.float16, generator=g)
    k = torch.randn(1, T, SPEC_H, K, device=device, dtype=torch.float16, generator=g)
    v = torch.randn(1, T, SPEC_HV, V, device=device, dtype=torch.float16, generator=g)
    # The layer hands the wrapper flattened [T, HV] rows (a_spec = a[:n]
    # .index_select(0, spec_token_indx)); 3-D [1, T, HV] is also accepted.
    gate_shape = (1, T, SPEC_HV) if a_3d else (T, SPEC_HV)
    a = torch.randn(*gate_shape, device=device, dtype=torch.float16, generator=g)
    b = torch.randn(*gate_shape, device=device, dtype=torch.float16, generator=g)
    A_log = torch.randn(SPEC_HV, device=device, dtype=torch.float32, generator=g).to(
        a_dtype
    )
    dt_bias = torch.randn(SPEC_HV, device=device, dtype=torch.float32, generator=g).to(
        d_dtype
    )
    blocks = blocks or (1 + n_req * 3)
    state = torch.randn(
        blocks, SPEC_HV, V, K, device=device, dtype=state_dtype, generator=g
    )
    cu = torch.zeros(n_req + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.tensor(lens, device=device, dtype=torch.int32)
    cu = cu.cumsum(0).to(torch.int32)
    # [n_req, 1 + num_spec]: the model passes the whole row for the state and
    # column 0 for the indices, so both stride patterns are covered.
    idx = (
        torch.arange(n_req * 3, device=device, dtype=torch.int32).reshape(n_req, 3) + 1
    )
    acc = (
        None
        if accepted is None
        else torch.tensor(accepted, device=device, dtype=torch.int32)
    )
    return q, k, v, a, b, A_log, dt_bias, state, cu, idx, acc


def _run_spec(parts, monkeypatch, hip, scale):
    q, k, v, a, b, A_log, dt_bias, state, cu, idx, acc = parts
    monkeypatch.setenv("VLLM_GDN_DECODE_RDNA2", "1" if hip else "0")
    s = state.clone()
    out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=s,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=idx,
        num_accepted_tokens=acc,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )
    return out, s


def _assert_spec_parity(parts, monkeypatch, state_atol=1e-5):
    scale = K**-0.5
    out_hip, state_hip = _run_spec(parts, monkeypatch, True, scale)
    out_tri, state_tri = _run_spec(parts, monkeypatch, False, scale)
    torch.accelerator.synchronize()

    used = parts[9].reshape(-1).unique().long()
    err_out = (out_hip.float() - out_tri.float()).abs()
    err_state = (state_hip[used].float() - state_tri[used].float()).abs()
    # <= 1 fp16 ulp on a handful of elements, bit-identical on the rest
    # (1 fp16 ulp is 6.1e-5 at |x| = 0.125 and 3.1e-5 at |x| = 0.03).
    assert err_out.max().item() <= 1.25e-4, f"out err {err_out.max().item()}"
    assert (err_out > 0).sum().item() < 0.005 * err_out.numel()
    # state: 1 ulp of the reduction order, absolutely bounded (an fp16 state
    # quantises ~1e-3, an fp32 state ~5e-7).
    assert err_state.max().item() <= state_atol, f"state err {err_state.max().item()}"
    torch.testing.assert_close(
        state_hip[used], state_tri[used], atol=state_atol, rtol=1e-6
    )
    # the untouched slots must stay untouched
    keep = torch.ones(state_hip.size(0), dtype=torch.bool, device=device)
    keep[used] = False
    assert torch.equal(state_hip[keep], state_tri[keep])


@pytest.mark.parametrize("n_req,accepted", [(1, [3]), (2, [3, 3]), (8, [3] * 8)])
def test_gdn_spec_verify_rdna2_parity(n_req, accepted, monkeypatch):
    parts = _make_spec(n_req, [3] * n_req, accepted)
    _assert_spec_parity(parts, monkeypatch)


@pytest.mark.parametrize("accepted", [[1] * 8, [2] * 8, [1, 2, 3, 1, 2, 3, 1, 2]])
def test_gdn_spec_verify_rdna2_partial_acceptance(accepted, monkeypatch):
    """num_accepted_tokens - 1 selects the state slot to start from, so a wrong
    offset silently reads the wrong history."""
    _assert_spec_parity(_make_spec(8, [3] * 8, accepted), monkeypatch)


@pytest.mark.parametrize("lens", [[3, 1, 2, 3], [1] * 4])
def test_gdn_spec_verify_rdna2_ragged(lens, monkeypatch):
    accepted = [min(ln, 3) for ln in lens]
    _assert_spec_parity(_make_spec(len(lens), lens, accepted), monkeypatch)


def test_gdn_spec_verify_rdna2_non_spec_varlen(monkeypatch):
    """IS_VARLEN with no num_accepted_tokens: one token per request, offset 0."""
    _assert_spec_parity(_make_spec(8, [1] * 8, None), monkeypatch)


def test_gdn_spec_verify_rdna2_fp16_gates_and_fp16_state(monkeypatch):
    """dt_bias is cast to the model dtype on load while A_log stays fp32, and
    the mamba state can be fp16."""
    _assert_spec_parity(
        _make_spec(4, [3] * 4, [3] * 4, a_dtype=torch.float16, d_dtype=torch.float16),
        monkeypatch,
    )
    _assert_spec_parity(
        _make_spec(4, [3] * 4, [3] * 4, state_dtype=torch.float16),
        monkeypatch,
        state_atol=2e-3,
    )


def test_gdn_spec_verify_rdna2_gate_layout_matches_call_site(monkeypatch):
    """Regression: the layer passes a/b as flattened [T, HV] rows -- see
    qwen_gdn_linear_attn.py ``a_spec = a.index_select(0, spec_token_indx)``
    on a 2-D ``a``. The op used to require 3-D [B, T, HV] and rejected the real
    engine call during warmup."""
    parts = _make_spec(8, [3] * 8, [3] * 8)
    assert parts[3].shape == (24, SPEC_HV) and parts[3].dim() == 2
    assert parts[4].shape == (24, SPEC_HV) and parts[4].dim() == 2
    _assert_spec_parity(parts, monkeypatch)


def test_gdn_spec_verify_rdna2_3d_gates(monkeypatch):
    """The 3-D [1, T, HV] spelling of the same token layout stays supported."""
    parts = _make_spec(4, [3] * 4, [3] * 4, a_3d=True)
    assert parts[3].dim() == 3
    _assert_spec_parity(parts, monkeypatch)


def test_gdn_spec_verify_rdna2_invalid_slot_zeroes_output(monkeypatch):
    """state_idx <= 0 (NULL_BLOCK_ID) must not touch the state and must zero the
    output instead of leaving the caller's buffer undefined."""
    parts = list(_make_spec(2, [3, 3], [3, 3]))
    idx = parts[9].clone()
    idx[0, 2] = 0  # NULL_BLOCK_ID on the slot the initial-state load reads
    parts[9] = idx
    monkeypatch.setenv("VLLM_GDN_DECODE_RDNA2", "1")
    out, state = _run_spec(tuple(parts), monkeypatch, True, K**-0.5)
    torch.accelerator.synchronize()
    assert out[0, :3].abs().max().item() == 0.0
    assert torch.equal(state[idx[0, 0]], parts[7][idx[0, 0]])
    # request 1 is a normal sequence and must be unaffected
    assert out[0, 3:6].abs().max().item() > 0.0


class _NoLaunchSpec:
    def __getitem__(self, _key):
        raise AssertionError("Triton kernel launched on the RDNA2 path")


def test_gdn_spec_verify_rdna2_does_not_launch_triton(monkeypatch):
    import vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating as fsg

    monkeypatch.setattr(
        fsg, "fused_sigmoid_gating_delta_rule_update_kernel", _NoLaunchSpec()
    )
    parts = _make_spec(8, [3] * 8, [3] * 8)
    out, state = _run_spec(parts, monkeypatch, True, K**-0.5)  # env var set to "1"
    monkeypatch.setenv("VLLM_GDN_DECODE_RDNA2", "1")
    torch.accelerator.synchronize()
    assert torch.isfinite(out).all()
    assert torch.isfinite(state).all()


def test_gdn_spec_verify_rdna2_model_shaped_arguments(monkeypatch):
    """Mirror exactly what GatedDeltaNet.forward passes on the MTP-2 spec-verify
    step (index_select slices for a/b, a sliced spec_query_start_loc, the 2-D
    state-index tensor and a sliced num_accepted_tokens) with the Triton kernel
    poisoned: the HIP op must be the one that runs."""
    import vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating as fsg

    monkeypatch.setattr(
        fsg, "fused_sigmoid_gating_delta_rule_update_kernel", _NoLaunchSpec()
    )
    monkeypatch.setenv("VLLM_GDN_DECODE_RDNA2", "1")

    n_req = 8
    parts = _make_spec(n_req, [3] * n_req, [3] * n_req)
    q, k, v, a, b, A_log, dt_bias, state, cu, idx, acc = parts
    # a/b are index_select'ed out of a wider token buffer by the layer
    tok_sel = torch.arange(n_req * 3, device=device, dtype=torch.int64)
    a_sel = a.reshape(-1, SPEC_HV).index_select(0, tok_sel)
    b_sel = b.reshape(-1, SPEC_HV).index_select(0, tok_sel)
    assert a_sel.is_contiguous() and b_sel.is_contiguous()
    # spec_query_start_loc is a slice of a larger per-step buffer
    qsl_buf = torch.zeros(n_req + 3, device=device, dtype=torch.int32)
    qsl_buf[: n_req + 1] = cu
    cu_slice = qsl_buf[: n_req + 1]
    acc_slice = acc.clone()
    out, st = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a_sel,
        b=b_sel,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu_slice,
        ssm_state_indices=idx,
        num_accepted_tokens=acc_slice,
        use_qk_l2norm_in_kernel=True,
    )
    torch.accelerator.synchronize()
    assert torch.isfinite(out).all()
    written = idx.reshape(-1).unique().long()
    assert torch.isfinite(state[written]).all()
