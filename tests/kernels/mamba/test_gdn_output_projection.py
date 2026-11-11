# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape contract for ``QwenGatedDeltaNetAttention._output_projection``.

Every ``forward_*`` already allocates ``core_attn_out`` and ``z`` as
``(N, H, D)``. The helper now norms those tensors as-is and flattens once
for ``out_proj``; it used to be rank-agnostic via ``reshape(z_shape_og)``.
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)


@pytest.mark.parametrize("dtype", [torch.float32])
@torch.inference_mode()
def test_output_projection_norms_per_head_and_flattens(
    default_vllm_config,
    dtype: torch.dtype,
) -> None:
    num_tokens, num_heads, head_dim = 3, 4, 8
    hidden = num_heads * head_dim

    layer = types.SimpleNamespace()
    layer.norm = RMSNormGated(
        head_dim,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        device="cpu",
        dtype=dtype,
    )
    layer.out_proj = lambda x: (x, None)
    layer._output_projection = types.MethodType(
        QwenGatedDeltaNetAttention._output_projection, layer
    )

    core_attn_out = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    z = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    out = layer._output_projection(core_attn_out, z)

    assert core_attn_out.shape == (num_tokens, num_heads, head_dim)
    assert z.shape == (num_tokens, num_heads, head_dim)
    assert out.shape == (num_tokens, hidden)
    assert out.dtype == dtype


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gated_rms_norm")
    ),
    reason="RDNA HIP gated_rms_norm not built",
)
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_hip_gated_norm_takes_per_head_3d_input(
    default_vllm_config,
    activation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GDN hands the norm ``(N, H, D)``; the HIP kernel must cover it (a 2D-only
    check sent every RDNA decode step to the ~9-kernel native path)."""
    num_tokens, num_heads, head_dim = 5, 6, 128
    norm = RMSNormGated(
        head_dim,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        activation=activation,
        device="cuda",
        dtype=torch.float16,
    )
    torch.nn.init.normal_(norm.weight, mean=1.0, std=0.1)
    x = torch.randn(num_tokens, num_heads, head_dim, device="cuda", dtype=torch.float16)
    z = torch.randn_like(x)

    ref = norm.forward_native(x, z)

    def _no_native(*args, **kwargs):
        raise AssertionError("3D input fell back to forward_native")

    monkeypatch.setattr(norm, "forward_native", _no_native)
    out = norm.forward_hip(x, z)

    assert out.shape == x.shape
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def _packed_z(num_tokens: int, num_heads: int, head_dim: int, pad: int):
    """Z as the layer holds it: a [T, H, D] view into the packed qkvz rows (the
    token stride is wider than H * D; heads are dense)."""
    packed = torch.randn(
        num_tokens,
        3 * num_heads * head_dim + pad,
        device="cuda",
        dtype=torch.float16,
    )
    z = packed[:, num_heads * head_dim : 2 * num_heads * head_dim].view(
        num_tokens, num_heads, head_dim
    )
    assert z.stride(-2) == head_dim and (num_tokens == 1 or not z.is_contiguous())
    return z


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gated_rms_norm")
    ),
    reason="RDNA HIP gated_rms_norm not built",
)
@pytest.mark.parametrize("activation", [0, 1])
@pytest.mark.parametrize("num_tokens", [1, 18, 24])
@torch.inference_mode()
def test_gated_rms_norm_op_reads_strided_z_in_place(
    activation: int, num_tokens: int
) -> None:
    """The op reads a token-strided 3D z at its own stride: bitwise equal to the
    call on a packed copy, with no copy of z on the caller's side."""
    num_heads, head_dim = 12, 128
    x = torch.randn(
        num_tokens * num_heads, head_dim, device="cuda", dtype=torch.float16
    )
    weight = torch.randn(head_dim, device="cuda", dtype=torch.float16)
    z = _packed_z(num_tokens, num_heads, head_dim, pad=96)

    out_strided = torch.empty_like(x)
    torch.ops._rocm_C.gated_rms_norm(out_strided, x, z, weight, 1e-6, activation)
    out_packed = torch.empty_like(x)
    torch.ops._rocm_C.gated_rms_norm(
        out_packed, x, z.reshape(-1, head_dim).contiguous(), weight, 1e-6, activation
    )
    torch.accelerator.synchronize()
    assert torch.equal(out_strided, out_packed)

    bad = z[:, ::2]  # heads strided: not a view the kernel can address
    with pytest.raises(RuntimeError):
        torch.ops._rocm_C.gated_rms_norm(
            torch.empty_like(x[: bad.shape[0] * bad.shape[1]]),
            x[: bad.shape[0] * bad.shape[1]],
            bad,
            weight,
            1e-6,
            activation,
        )


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gated_rms_norm")
    ),
    reason="RDNA HIP gated_rms_norm not built",
)
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_hip_gated_norm_takes_token_strided_z(
    default_vllm_config,
    activation: str,
) -> None:
    """forward_hip must agree on a token-strided z (view into the packed
    qkvz) after flattening it for the 2D-only kernel."""
    num_tokens, num_heads, head_dim = 18, 12, 128
    norm = RMSNormGated(
        head_dim,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        activation=activation,
        device="cuda",
        dtype=torch.float16,
    )
    torch.nn.init.normal_(norm.weight, mean=1.0, std=0.1)
    x = torch.randn(num_tokens, num_heads, head_dim, device="cuda", dtype=torch.float16)
    z = _packed_z(num_tokens, num_heads, head_dim, pad=64)

    ref = norm.forward_native(x, z)
    out = norm.forward_hip(x, z)
    out_packed = norm.forward_hip(x, z.contiguous())
    torch.accelerator.synchronize()
    assert out.shape == x.shape
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    assert torch.equal(out, out_packed)
