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
    pytest.skip("RDNA2 GDN decode kernel is gfx1030-only",
                allow_module_level=True)

from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_packed_decode,
)

device = "cuda"

K = 128  # head_k_dim; the HIP kernel requires NK == 1
V = 128
SOFTPLUS_THRESHOLD = 20.0


def _make_inputs(B: int, H: int, HV: int, num_blocks: int, seed: int,
                 a_log_dtype: torch.dtype = torch.float32,
                 dt_bias_dtype: torch.dtype = torch.float32):
    gen = torch.Generator(device=device).manual_seed(seed)
    qkv_dim = 2 * H * K + HV * V
    mixed_qkv = torch.randn(B, qkv_dim, device=device,
                            dtype=torch.float16, generator=gen)
    a = torch.randn(B, HV, device=device, dtype=torch.float16,
                    generator=gen)
    b = torch.randn(B, HV, device=device, dtype=torch.float16,
                    generator=gen)
    # vLLM casts dt_bias to the model dtype on load while A_log stays
    # fp32, so all dtype combinations must work.
    A_log = torch.randn(HV, device=device, dtype=torch.float32,
                        generator=gen).to(a_log_dtype)
    dt_bias = torch.randn(HV, device=device, dtype=torch.float32,
                          generator=gen).to(dt_bias_dtype)
    state0 = torch.randn(num_blocks, HV, V, K, device=device,
                         dtype=torch.float32, generator=gen)
    idx = torch.randperm(num_blocks, device=device,
                         generator=gen)[:B].to(torch.int32)
    return mixed_qkv, a, b, A_log, dt_bias, state0, idx


def _run_ref(mixed_qkv, a, b, A_log, dt_bias, state0, idx, scale):
    state = state0.clone()
    out = torch.zeros(mixed_qkv.shape[0], 1, state0.shape[1], V,
                      device=device, dtype=torch.float16)
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
    out = torch.zeros(mixed_qkv.shape[0], 1, state0.shape[1], V,
                      device=device, dtype=torch.float16)
    torch.ops._rocm_C.gdn_decode_rdna2(mixed_qkv, a, b, A_log, dt_bias, out,
                                       state, idx, scale, True)
    return out, state


@pytest.mark.parametrize("B", [1, 4, 16])
@pytest.mark.parametrize("H,HV", [(4, 12), (2, 4)])
@pytest.mark.parametrize(
    "a_log_dtype,dt_bias_dtype",
    [(torch.float32, torch.float32), (torch.float16, torch.float16),
     (torch.float32, torch.float16)],
)
def test_gdn_decode_rdna2_parity(B, H, HV, a_log_dtype, dt_bias_dtype):
    torch.manual_seed(0)
    num_blocks = max(2 * B, 8)
    mixed_qkv, a, b, A_log, dt_bias, state0, idx = _make_inputs(
        B, H, HV, num_blocks, seed=42, a_log_dtype=a_log_dtype,
        dt_bias_dtype=dt_bias_dtype)
    scale = K**-0.5

    out_ref, state_ref = _run_ref(mixed_qkv, a, b, A_log, dt_bias, state0,
                                  idx, scale)
    out_hip, state_hip = _run_hip(mixed_qkv, a, b, A_log, dt_bias, state0,
                                  idx, scale)

    out_err = (out_hip.float() - out_ref.float()).abs().max().item()
    used = state_ref[idx.long()]
    state_err = (state_hip[idx.long()] - used).abs().max().item()
    scale_ref = out_ref.float().abs().max().item()
    assert out_err < 2e-2 * max(scale_ref, 1.0), (
        f"output mismatch: max abs err {out_err} (ref max {scale_ref})")
    assert state_err < 2e-2, f"state mismatch: max abs err {state_err}"


def test_gdn_decode_rdna2_invalid_state_idx_zeroes_output():
    # state_idx outside [0, num_blocks) must zero the output and leave
    # the state untouched (NULL_BLOCK_ID collision guard).
    B, H, HV, num_blocks = 2, 2, 4, 4
    mixed_qkv, a, b, A_log, dt_bias, state0, idx = _make_inputs(
        B, H, HV, num_blocks, seed=7)
    idx[0] = -1
    idx[1] = num_blocks  # one past the end
    scale = K**-0.5

    out_hip, state_hip = _run_hip(mixed_qkv, a, b, A_log, dt_bias, state0,
                                  idx, scale)
    assert out_hip.abs().max().item() == 0.0
    torch.testing.assert_close(state_hip, state0)
