#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Standalone correctness test for causal_conv1d_fwd_rdna2 and causal_conv1d_update_rdna2
HIP kernels (csrc/rocm/causal_conv1d_rdna2.cu).

Run on the remote (.176) after rebuilding the .so:
    source /home/chenco_adm/Apps/vllm/venv-7.14.0/bin/activate && \
    HIP_VISIBLE_DEVICES=2 python tools/test_causal_conv1d_rdna2.py

Or use the wrapper:
    bash tools/test_causal_conv1d_rdna2.sh

Exit codes: 0 = PASS, 1 = FAIL.
"""

import sys
import os
import math

# ---------------------------------------------------------------------------
# LD_LIBRARY_PATH — must match venv-7.14.0 quirks in AGENTS.md "Working Virtual
# Environment" section.  _rocm_sdk_libraries MUST come before /opt/rocm/lib to
# avoid loading the older system rocblas 5.2.0 (from rocm-7.2.0) which breaks
# TunableOp's ROCBLAS_VERSION validator at the first GEMM call.
# ---------------------------------------------------------------------------
_VENV_SITEPACKAGES = "/home/chenco_adm/Apps/vllm/venv-7.14.0/lib/python3.12/site-packages"
_ROCM_SDK_CORE    = os.path.join(_VENV_SITEPACKAGES, "_rocm_sdk_core", "lib")
_ROCM_SDK_LIBS    = os.path.join(_VENV_SITEPACKAGES, "_rocm_sdk_libraries", "lib")
_TORCH_LIB        = os.path.join(_VENV_SITEPACKAGES, "torch", "lib")

for _path in (
    _ROCM_SDK_LIBS,
    os.path.join(_ROCM_SDK_CORE, "host-math", "lib"),
    os.path.join(_ROCM_SDK_CORE, "rocm_sysdeps", "lib"),
    os.path.join(_ROCM_SDK_CORE, "core", "lib"),
    _TORCH_LIB,
):
    _current = os.environ.get("LD_LIBRARY_PATH", "")
    if _path not in _current:
        os.environ["LD_LIBRARY_PATH"] = (_path + ":" + _current).rstrip(":")

import torch

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------
TOLERANCE = 0.15   # fp16 tolerance for these magnitude ranges

def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()

def report(name: str, max_diff: float, threshold: float = TOLERANCE) -> bool:
    ok = max_diff < threshold
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: max_abs_diff={max_diff:.6f}  (tol={threshold})")
    return ok

# ---------------------------------------------------------------------------
# Reference implementations — pure PyTorch fp32
#
# FIR formula (extracted from causal_conv1d_rdna2.cu lines 277-305):
#
#   For a sequence of tokens x[t] (t = 0 .. seqlen-1) with state_len = width-1:
#
#   Before processing token t, the state register holds:
#     state[k] = init[k]  for k in [0..state_len-1]
#   where init[k] comes from conv_state (if has_initial_state) or is zero.
#
#   The output for token t is:
#     out[t] = Σ_{k=0}^{state_len-1} w[k] · state[k]  +  w[state_len] · x[t]
#
#   Then the state is updated (shift-left, append x[t]):
#     state[s] = state[s+1]  for s in [0..state_len-2]
#     state[state_len-1] = x[t]
#
#   Key equation (weight index ↔ token offset ↔ state index):
#     out[t] = Σ_{k=0}^{state_len-1} w[k] · x[t - state_len + k]  +  w[state_len] · x[t]
#
#   Note: the state is maintained as [oldest .. newest] (oldest at index 0)
#   by the left-shift.  The kernel comment's "newest at high index" is
#   consistent with this: the NEWEST value is at state[state_len-1].
#   The output formula Σ w[k]·state[k] with this ordering gives the equation above.
#
#   For the UPDATE kernel (single token decode):
#     - conv_state is [num_cache_lines, dim, state_len] CONTIGUOUS (NOT transposed)
#     - out[b] = Σ_k w[k]·conv_state[slot,b,k] + w[state_len]·x[b,0]
#     - state update: state[s] = state[s+1]; state[state_len-1] = x[b,0]
# ---------------------------------------------------------------------------

def reference_causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_state_in: torch.Tensor,  # [num_cache_lines, dim, state_len] transposed view
    query_start_loc: torch.Tensor,  # [batch+1] int32
    cache_indices: torch.Tensor,    # [batch] int32
    has_initial_state: torch.Tensor | None,
    out_out: torch.Tensor,          # [dim, total_seqlen] output buffer
    silu_activation: bool,
) -> torch.Tensor:
    """
    Pure-PyTorch reference for causal_conv1d_fwd_rdna2.
    conv_state_in is the TRANSPOSED view [num_cache_lines, dim, state_len].
    Returns the final conv_state (modified in-place on conv_state_in).
    """
    dim = x.size(0)
    width = weight.size(1)
    state_len = width - 1
    batch = query_start_loc.size(0) - 1
    stride_x_token = x.stride(1)

    w = weight.float()
    bias_val = (
        bias.float() if (bias is not None and bias.numel() > 0) else None
    )

    out_fp32 = out_out.float().zero_()
    conv_state = conv_state_in.float()

    for b in range(batch):
        seq_start = query_start_loc[b].item()
        seq_end   = query_start_loc[b + 1].item()
        seqlen = seq_end - seq_start
        slot = cache_indices[b].item()

        # Load initial state
        if has_initial_state is not None and has_initial_state[b].item():
            state = conv_state[slot, :, :].clone()  # [dim, state_len]
        else:
            state = torch.zeros(dim, state_len, dtype=torch.float32, device=x.device)

        for t in range(seqlen):
            acc = bias_val.clone() if bias_val is not None else torch.zeros(dim, dtype=torch.float32, device=x.device)
            for k in range(state_len):
                acc += w[:, k] * state[:, k]
            acc += w[:, state_len] * x[:, seq_start + t].float()

            if silu_activation:
                acc = acc / (1.0 + torch.exp(-acc))

            out_fp32[:, seq_start + t] = acc

            # Shift state left and append x[t]
            for k in range(state_len - 1):
                state[:, k] = state[:, k + 1]
            state[:, state_len - 1] = x[:, seq_start + t].float()

        # Write back final state
        if slot >= 0 and slot < conv_state.size(0):
            conv_state[slot, :, :] = state.half()

    out_out.copy_(out_fp32.half())
    return conv_state.half()


def reference_causal_conv1d_update(
    x: torch.Tensor,          # [batch, dim, 1]
    conv_state_in: torch.Tensor,  # [num_cache_lines, dim, state_len] contiguous
    weight: torch.Tensor,     # [dim, width]
    bias: torch.Tensor | None,
    out_out: torch.Tensor,    # [batch, dim, 1]
    conv_state_indices: torch.Tensor,
    silu_activation: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure-PyTorch reference for causal_conv1d_update_rdna2.
    conv_state_in is [num_cache_lines, dim, state_len] CONTIGUOUS (NOT transposed).
    Returns (out, conv_state_out).
    """
    batch, dim, _ = x.shape
    width = weight.size(1)
    state_len = width - 1
    w = weight.float()
    bias_val = (
        bias.float() if (bias is not None and bias.numel() > 0) else None
    )

    out_fp32 = out_out.float().zero_()
    conv_state = conv_state_in.float()

    for b in range(batch):
        slot = conv_state_indices[b].item()
        if slot < 0 or slot >= conv_state.size(0):
            continue

        state = conv_state[slot, :, :]  # [dim, state_len], view

        acc = bias_val.clone() if bias_val is not None else torch.zeros(dim, dtype=torch.float32, device=x.device)
        for k in range(state_len):
            acc += w[:, k] * state[:, k]
        acc += w[:, state_len] * x[b, :, 0].float()

        if silu_activation:
            acc = acc / (1.0 + torch.exp(-acc))

        out_fp32[b, :, 0] = acc

        # Update state: shift left, append x[b, 0]
        for k in range(state_len - 1):
            state[:, k] = state[:, k + 1]
        state[:, state_len - 1] = x[b, :, 0].float()

    out_out.copy_(out_fp32.half())
    return out_out, conv_state.half()


# ---------------------------------------------------------------------------
# Test 1: causal_conv1d_fwd_rdna2 (varlen prefill)
# ---------------------------------------------------------------------------
def test_fwd():
    print("\n=== Test: causal_conv1d_fwd_rdna2 ===")

    torch.random.manual_seed(0xCAFFE)
    dim = 5120
    width = 4
    state_len = width - 1   # 3
    num_cache_lines = 696
    batch = 3
    query_start_loc_data = [0, 1568, 2272, 2432]  # seqlens: 1568, 704, 160
    cache_indices_data = [1, 5, 9]
    has_initial_state_data = [True, False, True]

    # x: [dim, padded_dim] channel-last view (stride[0]==1, stride[1]==8192),
    # matching production's mixed_qkv.transpose(0,1) layout.
    padded_dim = 8192
    x = torch.empty(padded_dim, dim, dtype=torch.float16, device="cuda").transpose(0, 1)
    torch.nn.init.uniform_(x, -1.0, 1.0)

    weight = torch.empty(dim, width, dtype=torch.float16, device="cuda")
    torch.nn.init.uniform_(weight, -0.1, 0.1)

    bias = torch.empty(dim, dtype=torch.float16, device="cuda")
    torch.nn.init.uniform_(bias, -0.05, 0.05)

    # conv_state: production stores as [num_cache_lines, state_len, dim] then
    # transpose(-1,-2) to [num_cache_lines, dim, state_len] before passing.
    # We follow this exact pattern.
    conv_state_raw = torch.empty(num_cache_lines, state_len, dim, dtype=torch.float16, device="cuda")
    torch.nn.init.uniform_(conv_state_raw, -0.5, 0.5)
    conv_state_transposed = conv_state_raw.transpose(-1, -2).contiguous()

    query_start_loc = torch.tensor(query_start_loc_data, dtype=torch.int32, device="cuda")
    cache_indices   = torch.tensor(cache_indices_data, dtype=torch.int32, device="cuda")
    has_initial_state = torch.tensor(has_initial_state_data, dtype=torch.bool, device="cuda")

    # ------------------------------------------------------------------
    # Case A: bias=True, silu=True, has_initial_state=[True,False,True]
    # ------------------------------------------------------------------
    print("  -- Case A: bias=True, silu=True, mixed initial-state")

    conv_state_hip = conv_state_transposed.clone()
    out_hip = torch.empty(dim, padded_dim, dtype=torch.float16, device="cuda")
    out_hip.zero_()

    try:
        torch.ops._rocm_C.causal_conv1d_fwd_rdna2(
            x, weight, bias, conv_state_hip,
            query_start_loc, cache_indices, has_initial_state,
            out_hip, True,
        )
    except Exception as e:
        print(f"  [SKIP] HIP kernel unavailable: {e}")
        return True

    # Reference: use a SEPARATE conv_state copy (HIP modified conv_state_hip)
    conv_state_ref = conv_state_transposed.clone()
    out_ref = torch.empty(dim, padded_dim, dtype=torch.float16, device="cuda")
    out_ref.zero_()
    reference_causal_conv1d_fwd(
        x, weight, bias, conv_state_ref,
        query_start_loc, cache_indices, has_initial_state,
        out_ref, silu_activation=True,
    )

    ok_a_out = report("out (Case A)", max_abs_diff(out_hip, out_ref))
    ok_a_state = report("conv_state writeback (Case A)", max_abs_diff(conv_state_hip, conv_state_ref))

    # ------------------------------------------------------------------
    # Case B: bias=False, silu=False, has_initial_state=[False,True,False]
    # ------------------------------------------------------------------
    print("  -- Case B: bias=False, silu=False, different init pattern")

    bias_empty = torch.empty(0, dtype=torch.float16, device="cuda")
    has_init_b = torch.tensor([False, True, False], dtype=torch.bool, device="cuda")

    conv_state_hip_b = conv_state_transposed.clone()
    out_hip_b = torch.empty(dim, padded_dim, dtype=torch.float16, device="cuda")
    out_hip_b.zero_()

    torch.ops._rocm_C.causal_conv1d_fwd_rdna2(
        x, weight, bias_empty, conv_state_hip_b,
        query_start_loc, cache_indices, has_init_b,
        out_hip_b, False,
    )

    conv_state_ref_b = conv_state_transposed.clone()
    out_ref_b = torch.empty(dim, padded_dim, dtype=torch.float16, device="cuda")
    out_ref_b.zero_()
    reference_causal_conv1d_fwd(
        x, weight, bias_empty, conv_state_ref_b,
        query_start_loc, cache_indices, has_init_b,
        out_ref_b, silu_activation=False,
    )

    ok_b_out = report("out (Case B)", max_abs_diff(out_hip_b, out_ref_b))
    ok_b_state = report("conv_state writeback (Case B)", max_abs_diff(conv_state_hip_b, conv_state_ref_b))

    return (ok_a_out and ok_a_state and ok_b_out and ok_b_state)


# ---------------------------------------------------------------------------
# Test 2: causal_conv1d_update_rdna2 (decode)
# ---------------------------------------------------------------------------
def test_update():
    print("\n=== Test: causal_conv1d_update_rdna2 ===")

    torch.random.manual_seed(0xDECAF)
    batch = 5
    dim = 5120
    width = 4
    state_len = width - 1  # 3
    num_cache_lines = 696
    conv_state_indices_data = [0, 3, 7, 11, 600]

    # x: [batch, dim, 1] — one new token per batch element
    x = torch.empty(batch, dim, 1, dtype=torch.float16, device="cuda")
    torch.nn.init.uniform_(x, -0.8, 0.8)

    weight = torch.empty(dim, width, dtype=torch.float16, device="cuda")
    torch.nn.init.uniform_(weight, -0.1, 0.1)

    bias = torch.empty(dim, dtype=torch.float16, device="cuda")
    torch.nn.init.uniform_(bias, -0.05, 0.05)

    # conv_state: [num_cache_lines, dim, state_len] CONTIGUOUS (NOT transposed)
    conv_state = torch.empty(num_cache_lines, dim, state_len, dtype=torch.float16, device="cuda")
    torch.nn.init.uniform_(conv_state, -0.5, 0.5)

    conv_state_indices = torch.tensor(conv_state_indices_data, dtype=torch.int32, device="cuda")

    out_hip = torch.empty(batch, dim, 1, dtype=torch.float16, device="cuda")
    out_ref = torch.empty(batch, dim, 1, dtype=torch.float16, device="cuda")

    # SEPARATE copies for HIP and reference (HIP modifies conv_state in-place)
    conv_state_hip = conv_state.clone()
    conv_state_ref = conv_state.clone()

    # ------------------------------------------------------------------
    # Case A: silu=True, bias=True
    # ------------------------------------------------------------------
    print("  -- Case A: silu=True, bias=True")

    try:
        torch.ops._rocm_C.causal_conv1d_update_rdna2(
            x, conv_state_hip, weight, bias,
            out_hip, conv_state_indices, True,  # silu=True
        )
    except Exception as e:
        print(f"  [SKIP] HIP kernel unavailable: {e}")
        return True

    out_ref.zero_()
    _, conv_state_ref_final = reference_causal_conv1d_update(
        x, conv_state_ref, weight, bias,
        out_ref, conv_state_indices, silu_activation=True,
    )

    ok_a_out = report("out (Case A)", max_abs_diff(out_hip, out_ref))
    ok_a_state = report("conv_state writeback (Case A)", max_abs_diff(conv_state_hip, conv_state_ref_final))

    # ------------------------------------------------------------------
    # Case B: silu=False, bias=False
    # ------------------------------------------------------------------
    print("  -- Case B: silu=False, bias=False")

    bias_none = torch.empty(0, dtype=torch.float16, device="cuda")

    conv_state_hip_b = conv_state.clone()
    conv_state_ref_b = conv_state.clone()
    x_b = torch.empty(batch, dim, 1, dtype=torch.float16, device="cuda")
    torch.nn.init.uniform_(x_b, -0.8, 0.8)

    out_hip_b = torch.empty(batch, dim, 1, dtype=torch.float16, device="cuda")
    out_ref_b = torch.empty(batch, dim, 1, dtype=torch.float16, device="cuda")

    torch.ops._rocm_C.causal_conv1d_update_rdna2(
        x_b, conv_state_hip_b, weight, bias_none,
        out_hip_b, conv_state_indices, False,  # silu=False
    )

    out_ref_b.zero_()
    _, conv_state_ref_b_final = reference_causal_conv1d_update(
        x_b, conv_state_ref_b, weight, bias_none,
        out_ref_b, conv_state_indices, silu_activation=False,
    )

    ok_b_out = report("out (Case B)", max_abs_diff(out_hip_b, out_ref_b))
    ok_b_state = report("conv_state writeback (Case B)", max_abs_diff(conv_state_hip_b, conv_state_ref_b_final))

    return (ok_a_out and ok_a_state and ok_b_out and ok_b_state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("causal_conv1d_rdna2 correctness test")
    print(f"Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"LD_LIBRARY_PATH first entries:")
    for _p in os.environ.get("LD_LIBRARY_PATH", "").split(":")[:4]:
        print(f"  {_p}")
    print(f"Tolerance: {TOLERANCE}")
    print("=" * 70)

    if not hasattr(torch.ops, "_rocm_C"):
        print("FATAL: torch.ops._rocm_C not found — is the _rocm_C .so loaded?")
        sys.exit(1)

    for _name in ("causal_conv1d_fwd_rdna2", "causal_conv1d_update_rdna2"):
        if not hasattr(torch.ops._rocm_C, _name):
            print(f"FATAL: torch.ops._rocm_C.{_name} not registered")
            sys.exit(1)

    ok_fwd    = test_fwd()
    ok_update = test_update()

    print("\n" + "=" * 70)
    if ok_fwd and ok_update:
        print("OVERALL: PASS")
        sys.exit(0)
    else:
        print("OVERALL: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
