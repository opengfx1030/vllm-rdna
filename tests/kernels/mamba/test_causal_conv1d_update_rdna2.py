# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stride tests for the gfx1030 HIP ``causal_conv1d_update_rdna2`` kernel.

The GDN layer passes ``kv_cache[0].transpose(-1, -2)`` as the conv state with
the default SD layout: storage is ``[lines, state_len, dim]`` and the view is
``[lines, dim, state_len]``, so time is not contiguous. The kernel used to
index the state as packed ``slot*dim*state_len + c*state_len + k`` and read
another channel (NaN tokens in eager mode and in oversize decode batches,
which bypass the contiguous cudagraph state arenas).

Run ``pytest tests/kernels/mamba/test_causal_conv1d_update_rdna2.py``.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 conv1d update kernel is ROCm-only", allow_module_level=True)

from vllm.platforms.rdna import on_rdna2  # noqa: E402

if not on_rdna2():
    pytest.skip("RDNA2 conv1d update kernel is gfx10x-only", allow_module_level=True)

import vllm._custom_ops  # noqa: E402, F401
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_update,
)

if not hasattr(torch.ops._rocm_C, "causal_conv1d_update_rdna2"):
    pytest.skip("causal_conv1d_update_rdna2 not built", allow_module_level=True)

DEVICE = "cuda"


def _reference(x, state, weight, bias, indices, silu):
    """fp32 reference. ``state`` is a [lines, dim, state_len] view (any
    strides); returns (out [batch, dim], new_state contiguous fp16)."""
    state = state.float().clone()
    w = weight.float()
    state_len = w.size(1) - 1
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for b, slot in enumerate(indices.tolist()):
        if slot < 0 or slot >= state.size(0):
            continue
        s = state[slot]
        acc = bias.float().clone() if bias is not None else torch.zeros_like(w[:, 0])
        acc = acc + (w[:, :state_len] * s).sum(-1) + w[:, state_len] * x[b].float()
        if silu:
            acc = acc * torch.sigmoid(acc)
        out[b] = acc
        state[slot] = torch.cat([s[:, 1:], x[b].float().unsqueeze(-1)], dim=-1)
    return out.half(), state.half()


def _make(batch, dim, width, lines, seed, transposed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    state_len = width - 1
    if transposed:
        # Paged SD layout: storage [lines, state_len, dim], view transposed.
        storage = torch.randn(
            lines, state_len, dim, device=DEVICE, dtype=torch.float16, generator=g
        )
        state = storage.transpose(-1, -2)
        assert not state.is_contiguous()
    else:
        state = torch.randn(
            lines, dim, state_len, device=DEVICE, dtype=torch.float16, generator=g
        )
    x = torch.randn(batch, dim, device=DEVICE, dtype=torch.float16, generator=g)
    weight = (
        torch.randn(dim, width, device=DEVICE, dtype=torch.float16, generator=g) * 0.5
    )
    bias = torch.randn(dim, device=DEVICE, dtype=torch.float16, generator=g)
    # Skip line 0: it is the null block the Triton path ignores.
    perm = torch.randperm(lines - 1, device="cpu")[:batch] + 1
    indices = perm.to(device=DEVICE, dtype=torch.int32)
    return x, state, weight, bias, indices


def _run_hip(x, state, weight, bias, indices, silu, monkeypatch):
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "1")
    return causal_conv1d_update(
        x.clone(),
        state,
        weight,
        bias,
        "silu" if silu else None,
        conv_state_indices=indices,
    )


@pytest.mark.parametrize("batch", [1, 7, 64])
@pytest.mark.parametrize("dim", [256, 2048])
@pytest.mark.parametrize("silu", [True, False])
@pytest.mark.parametrize("transposed", [False, True])
def test_update_matches_reference(batch, dim, silu, transposed, monkeypatch):
    width, lines = 4, 96
    x, state, weight, bias, indices = _make(batch, dim, width, lines, 0, transposed)
    ref_out, ref_state = _reference(x, state, weight, bias, indices, silu)
    out = _run_hip(x, state, weight, bias, indices, silu, monkeypatch)
    torch.accelerator.synchronize()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref_out, atol=2e-3, rtol=2e-3)
    # The state shift is a pure fp16 copy: exact.
    torch.testing.assert_close(state, ref_state, atol=0, rtol=0)


@pytest.mark.parametrize("silu", [True, False])
def test_transposed_matches_contiguous_bitwise(silu, monkeypatch):
    """Same values in both layouts must give bit-identical outputs/states:
    the arithmetic is layout-independent."""
    batch, dim, width, lines = 32, 2048, 4, 64
    x, state_c, weight, bias, indices = _make(batch, dim, width, lines, 1, False)
    state_t = state_c.contiguous().transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not state_t.is_contiguous()
    assert torch.equal(state_t, state_c)
    out_c = _run_hip(x, state_c, weight, bias, indices, silu, monkeypatch)
    out_t = _run_hip(x, state_t, weight, bias, indices, silu, monkeypatch)
    torch.accelerator.synchronize()
    assert torch.equal(out_c, out_t)
    assert torch.equal(state_c, state_t)


def test_matches_triton_path_transposed(monkeypatch):
    batch, dim, width, lines = 16, 1024, 4, 32
    x, state, weight, bias, indices = _make(batch, dim, width, lines, 2, True)
    state_tri = state.clone()  # clone keeps the transposed strides
    out_hip = _run_hip(x, state, weight, bias, indices, True, monkeypatch)
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "0")
    out_tri = causal_conv1d_update(
        x.clone(), state_tri, weight, bias, "silu", conv_state_indices=indices
    )
    torch.accelerator.synchronize()
    torch.testing.assert_close(out_hip, out_tri, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(state, state_tri, atol=0, rtol=0)


def test_strided_x_inplace_and_padded_slot(monkeypatch):
    """X is a column slice of a wider buffer (non-contiguous rows); the
    default out=x is written in place; slot -1 leaves its row and state
    untouched."""
    batch, dim, width, lines = 8, 512, 4, 16
    _, state, weight, bias, indices = _make(batch, dim, width, lines, 3, True)
    indices[3] = -1
    wide = torch.randn(batch, 3 * dim, device=DEVICE, dtype=torch.float16)
    x = wide[:, dim : 2 * dim]
    assert not x.is_contiguous()
    x_before = x.clone()
    ref_out, ref_state = _reference(x_before, state, weight, bias, indices, True)
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "1")
    out = causal_conv1d_update(
        x, state, weight, bias, "silu", conv_state_indices=indices
    )
    torch.accelerator.synchronize()
    assert out.data_ptr() == x.data_ptr()
    keep = torch.ones(batch, dtype=torch.bool)
    keep[3] = False
    torch.testing.assert_close(out[keep], ref_out[keep], atol=2e-3, rtol=2e-3)
    assert torch.equal(out[3], x_before[3])
    torch.testing.assert_close(state, ref_state, atol=0, rtol=0)
