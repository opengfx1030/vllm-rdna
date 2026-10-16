# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the RDNA2 (gfx1030) W8A8 sdot4 dense op and its wiring.

* ``pack_weight``: CPU checks of the packed dword layout against the kernel
  contract in ``csrc/rocm/w8a8_sdot4_rdna2.cuh`` (run anywhere).
* ``w8a8_gemm_rdna2`` vs an fp32 reference that replicates the kernel's
  per-(token, group) int8 activation quant, over per-channel and per-group
  weight scales, at decode (split-K) and prefill shapes.
* ``RDNA2Int8ScaledMMLinearKernel`` end to end on a synthetic layer, and the
  opt-in latch.

GPU cases skip cleanly when the op is absent (no GPU, non-gfx1030, partial
build).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization import rdna2_w8a8
from vllm.model_executor.layers.quantization.rdna2_w8a8 import (
    A_PERM,
    act_group,
    pack_weight,
    unpack_weight,
)


def _has_ops() -> bool:
    return rdna2_w8a8.on_gfx1030() and rdna2_w8a8.dense_ops_built()


gfx1030_w8a8 = pytest.mark.skipif(
    not _has_ops(), reason="requires gfx1030 with w8a8_gemm_rdna2 built in"
)
device = "cuda"


# ---------------------------------------------------------------------------
# Packing (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n, k", [(4, 32), (12, 64), (64, 256)])
def test_pack_weight_round_trip(n, k):
    torch.manual_seed(0)
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8)
    packed = pack_weight(w)
    assert packed.shape == (k // 4, n) and packed.dtype == torch.int32
    assert torch.equal(unpack_weight(packed), w)


def test_pack_weight_dword_layout():
    """Dword row 2c+h of column n holds w[8c + A_PERM[4h + i]] in byte i."""
    n, k = 8, 64
    torch.manual_seed(1)
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8)
    packed = pack_weight(w)
    for c in range(k // 8):
        for col in range(n):
            for h in range(2):
                dword = int(packed[2 * c + h, col]) & 0xFFFFFFFF
                for i in range(4):
                    byte = (dword >> (8 * i)) & 0xFF
                    expect = int(w[col, 8 * c + A_PERM[4 * h + i]]) & 0xFF
                    assert byte == expect, (c, col, h, i)


def test_act_group_picks_largest_divisor():
    assert act_group(5120) == 128
    assert act_group(192) == 64
    assert act_group(96) == 32
    with pytest.raises(ValueError):
        act_group(48)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def _quant_act(x: torch.Tensor, group: int):
    """Per-(token, group) absmax/127 int8 quant, matching the kernel."""
    m, k = x.shape
    g = k // group
    xf = x.float().reshape(m, g, group)
    gmax = xf.abs().amax(dim=2, keepdim=True)
    inv = torch.where(gmax > 0, 127.0 / gmax, torch.zeros_like(gmax))
    q = torch.round(xf * inv).clamp_(-128, 127)
    return q, gmax / 127.0


def _dequant_weight(w: torch.Tensor, w_scales: torch.Tensor) -> torch.Tensor:
    """int8 [N, K] x fp16 [K/WG, N] -> fp32 [N, K]."""
    n, k = w.shape
    wg = k // w_scales.shape[0]
    s = w_scales.float().t().repeat_interleave(wg, dim=1)
    return w.float() * s


def _reference(x, w, w_scales, a_group):
    q, a_scale = _quant_act(x, a_group)
    a_deq = (q * a_scale).reshape(x.shape)
    return (a_deq @ _dequant_weight(w, w_scales).t()).to(torch.float16)


def _make_scales(k, n, w_group):
    return (0.02 * torch.rand(k // w_group, n, device=device) + 0.005).half()


def _rel_l2(got, ref):
    got_f = got.float()
    ref_f = ref.float()
    denom = ref_f.norm()
    return ((got_f - ref_f).norm() / denom).item() if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# The op
# ---------------------------------------------------------------------------

# M, K, N, a_group, w_group (w_group == K is per-channel)
CASES = [
    (1, 5120, 5120, 128, 5120),  # decode: split-K fp32 path
    (4, 2560, 1536, 128, 2560),
    (17, 5120, 32, 128, 5120),  # one N tile, many K windows
    (33, 2048, 1024, 64, 2048),
    (64, 2560, 768, 128, 128),  # per-group weight scales
    (300, 1024, 512, 32, 32),
    (2048, 4096, 1024, 128, 4096),  # prefill: direct fp16 stores
]


@gfx1030_w8a8
@pytest.mark.parametrize("M, K, N, a_group, w_group", CASES)
def test_w8a8_gemm_matches_reference(M, K, N, a_group, w_group):
    """Only fp32 accumulation order separates the op from the reference."""
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device=device)
    scales = _make_scales(K, N, w_group)

    from vllm import _custom_ops as ops

    out = ops.w8a8_gemm_rdna2(x, pack_weight(w), scales, a_group)
    assert out.shape == (M, N) and out.dtype == torch.float16
    assert torch.isfinite(out).all()
    rel = _rel_l2(out, _reference(x, w, scales, a_group))
    assert rel < 2e-3, f"rel-L2 {rel:.5f} >= 2e-3"


@gfx1030_w8a8
def test_w8a8_gemm_tracks_fp16_matmul():
    """The int8 activation error stays inside the usual W8A8 band."""
    M, K, N = 128, 2048, 1024
    torch.manual_seed(7)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device=device)
    scales = _make_scales(K, N, K)

    from vllm import _custom_ops as ops

    out = ops.w8a8_gemm_rdna2(x, pack_weight(w), scales, 128)
    ref = x.float() @ _dequant_weight(w, scales).t()
    rel = _rel_l2(out, ref)
    assert rel < 0.05, f"rel-L2 {rel:.4f} >= 0.05"


@gfx1030_w8a8
def test_w8a8_gemm_accepts_strided_rows():
    """A sliced activation with aligned row stride matches the contiguous run."""
    M, K, N = 24, 1024, 512
    torch.manual_seed(3)
    big = torch.randn(M, K + 64, dtype=torch.float16, device=device)
    x = big[:, :K]
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device=device)
    scales = _make_scales(K, N, K)

    from vllm import _custom_ops as ops

    packed = pack_weight(w)
    strided = ops.w8a8_gemm_rdna2(x, packed, scales, 128)
    contiguous = ops.w8a8_gemm_rdna2(x.contiguous(), packed, scales, 128)
    assert torch.equal(strided, contiguous)


@gfx1030_w8a8
def test_w8a8_gemm_rejects_bad_shapes():
    from vllm import _custom_ops as ops

    x = torch.randn(8, 256, dtype=torch.float16, device=device)
    w = torch.randint(-128, 128, (64, 256), dtype=torch.int8, device=device)
    packed = pack_weight(w)
    with pytest.raises(RuntimeError):  # a_group does not divide the weight group
        ops.w8a8_gemm_rdna2(x, packed, _make_scales(256, 64, 64), 128)
    with pytest.raises(RuntimeError):  # unsupported activation group
        ops.w8a8_gemm_rdna2(x, packed, _make_scales(256, 64, 256), 16)
    with pytest.raises(RuntimeError):  # K mismatch between a and w_packed
        ops.w8a8_gemm_rdna2(x[:, :128], packed, _make_scales(256, 64, 256), 128)


# ---------------------------------------------------------------------------
# Linear kernel wiring
# ---------------------------------------------------------------------------


def _make_kernel(monkeypatch, is_channelwise: bool):
    from vllm.model_executor.kernels.linear.scaled_mm.rdna2_w8a8_int8 import (
        RDNA2Int8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        Int8ScaledMMLinearLayerConfig,
    )

    monkeypatch.setenv(rdna2_w8a8.ENV_VAR, "1")
    config = Int8ScaledMMLinearLayerConfig(
        is_channelwise=is_channelwise,
        is_static_input_scheme=False,
        input_symmetric=True,
    )
    return RDNA2Int8ScaledMMLinearKernel(
        config,
        layer_param_names=[
            "weight",
            "weight_scale",
            "input_scale",
            "input_zero_point",
            "azp_adj",
        ],
    )


@gfx1030_w8a8
def test_kernel_latch(monkeypatch):
    from vllm.model_executor.kernels.linear.scaled_mm.rdna2_w8a8_int8 import (
        RDNA2Int8ScaledMMLinearKernel,
    )

    monkeypatch.delenv(rdna2_w8a8.ENV_VAR, raising=False)
    assert RDNA2Int8ScaledMMLinearKernel.is_supported()[0] is False
    monkeypatch.setenv(rdna2_w8a8.ENV_VAR, "1")
    assert RDNA2Int8ScaledMMLinearKernel.is_supported()[0] is True


@gfx1030_w8a8
def test_kernel_declines_static_and_asymmetric():
    from vllm.model_executor.kernels.linear.scaled_mm.rdna2_w8a8_int8 import (
        RDNA2Int8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        Int8ScaledMMLinearLayerConfig,
    )

    static = Int8ScaledMMLinearLayerConfig(
        is_channelwise=True, is_static_input_scheme=True, input_symmetric=True
    )
    asym = Int8ScaledMMLinearLayerConfig(
        is_channelwise=True, is_static_input_scheme=False, input_symmetric=False
    )
    assert RDNA2Int8ScaledMMLinearKernel.can_implement(static)[0] is False
    assert RDNA2Int8ScaledMMLinearKernel.can_implement(asym)[0] is False


@gfx1030_w8a8
@pytest.mark.parametrize("is_channelwise", [True, False])
def test_kernel_layer_end_to_end(monkeypatch, is_channelwise):
    """process_weights_after_loading + apply_weights on a synthetic layer.

    The per-tensor case is a fused module (two logical matrices, one scale
    each), the path that expands scales to per-channel before packing.
    """
    M, K, widths = 40, 1024, [384, 128]
    N = sum(widths)
    torch.manual_seed(11)
    kernel = _make_kernel(monkeypatch, is_channelwise)
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device=device)
    if is_channelwise:
        w_s = 0.02 * torch.rand(N, 1, device=device) + 0.005
        expanded = w_s
    else:
        w_s = 0.02 * torch.rand(len(widths), device=device) + 0.005
        expanded = torch.cat(
            [w_s[i].expand(width) for i, width in enumerate(widths)]
        ).reshape(N, 1)
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(w, requires_grad=False),
        weight_scale=torch.nn.Parameter(w_s, requires_grad=False),
        input_scale=None,
        input_zero_point=None,
        azp_adj=None,
        logical_widths=widths,
    )
    kernel.process_weights_after_loading(layer)
    assert layer.weight.shape == (K // 4, N) and layer.weight.dtype == torch.int32
    assert layer.weight_scale.shape == (1, N) and layer.weight_scale.dtype == torch.half

    x = torch.randn(2, M // 2, K, dtype=torch.float16, device=device)
    bias = torch.randn(N, dtype=torch.float16, device=device)
    out = kernel.apply_weights(layer, x, bias)
    assert out.shape == (2, M // 2, N)
    ref = _reference(x.reshape(M, K), w, expanded.t().half(), act_group(K)) + bias
    rel = _rel_l2(out.reshape(M, N), ref)
    assert rel < 5e-3, f"rel-L2 {rel:.5f} >= 5e-3"
