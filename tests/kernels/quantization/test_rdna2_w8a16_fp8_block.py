# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA2W8A16Fp8BlockLinearKernel: selection and numerics on gfx1030."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 kernel", allow_module_level=True)

from vllm.model_executor.kernels.linear import init_fp8_linear_kernel  # noqa: E402
from vllm.model_executor.kernels.linear.scaled_mm.rdna2_w8a16_fp8_block import (  # noqa: E402
    W8A16_FP8_MAX_TOKENS,
    RDNA2W8A16Fp8BlockLinearKernel,
    w8a16_fp8_mm,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)

if not RDNA2W8A16Fp8BlockLinearKernel.is_supported()[0]:
    pytest.skip("needs gfx1030 + gemm_w8a16_fp8_dense", allow_module_level=True)


def _layer(n: int, k: int, g: torch.Generator, is_bmm: bool = False):
    layer = torch.nn.Module()
    # Every finite E4M3 code, including the exponent-0xF binade (256..448)
    # real checkpoints use with small block scales.
    codes = torch.randint(0, 256, (n, k), generator=g, dtype=torch.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x7E  # no NaN
    w = codes.view(torch.float8_e4m3fn)
    s = torch.exp2(-torch.randint(10, 15, (n // 128, k // 128), generator=g).float())
    layer.weight = torch.nn.Parameter(w.cuda(), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(s.cuda(), requires_grad=False)
    layer.input_scale = None
    if is_bmm:
        layer.is_bmm = True
    ref_w = w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return layer, ref_w


@pytest.mark.parametrize("m", [1, 4, 33, 256, 257, 2048])
@pytest.mark.parametrize("n,k", [(1536, 4096), (8192, 1024), (4096, 2048)])
def test_selected_and_matches_reference(m, n, k, default_vllm_config):
    kernel = init_fp8_linear_kernel(
        activation_quant_key=kFp8Dynamic128Sym,
        weight_quant_key=kFp8Static128BlockSym,
        input_dtype=torch.float16,
        out_dtype=torch.float16,
        weight_shape=(n, k),
    )
    assert isinstance(kernel, RDNA2W8A16Fp8BlockLinearKernel)
    g = torch.Generator().manual_seed(m + n)
    layer, ref_w = _layer(n, k, g)
    kernel.process_weights_after_loading(layer)
    # Logical [N, K] shape is kept; storage is K-major for the kernel.
    assert layer.weight.shape == (n, k) and layer.weight.t().is_contiguous()
    # A row-strided view, like the q-lora half of a fused projection.
    x_wide = torch.randn(m, k + 512, generator=g).to(torch.float16)
    x = x_wide[:, :k]
    out = kernel.apply_weights(layer, x_wide.cuda()[:, :k])
    ref = x.float() @ ref_w.t()
    assert out.shape == (m, n) and out.dtype == torch.float16
    torch.testing.assert_close(out.float().cpu(), ref, atol=2e-2, rtol=2e-2)


def test_bmm_weight_left_untouched(default_vllm_config):
    """Batched weights (wo_a) keep their [N, K] row-major layout."""
    kernel = RDNA2W8A16Fp8BlockLinearKernel(
        init_fp8_linear_kernel(
            activation_quant_key=kFp8Dynamic128Sym,
            weight_quant_key=kFp8Static128BlockSym,
            input_dtype=torch.float16,
            out_dtype=torch.float16,
            weight_shape=(2048, 1024),
        ).config
    )
    layer, _ = _layer(2048, 1024, torch.Generator().manual_seed(0), is_bmm=True)
    kernel.process_weights_after_loading(layer)
    # Row-major [N, K] (possibly a K-padded view), not the K-major copy.
    assert layer.weight.stride(1) == 1
    assert not hasattr(layer, "_rdna2_w8a16_scales")


@pytest.mark.parametrize("m", [1, W8A16_FP8_MAX_TOKENS, W8A16_FP8_MAX_TOKENS + 1, 512])
@pytest.mark.parametrize("zeroed", [False, True])
def test_w8a16_fp8_mm_out_both_paths(m, zeroed):
    """Both M paths of the helper (W8A16 kernel / dequant + rocBLAS) into a
    caller-owned output slice, as the DeepSeek-V4 wo_a groups use it."""
    k, n, gs = 4096, 1024, 128
    g = torch.Generator().manual_seed(m)
    codes = torch.randint(0, 256, (k, n), generator=g, dtype=torch.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x7E
    s = torch.exp2(-torch.randint(8, 12, (k // gs, n), generator=g).float())
    ref_w = codes.view(torch.float8_e4m3fn).float() * s.repeat_interleave(gs, 0)
    x = torch.randn(m, k, generator=g).to(torch.float16)
    out = torch.zeros(2, m, n, dtype=torch.float16, device="cuda")
    if not zeroed:
        out.fill_(7.0)  # stale contents the helper must overwrite
    w8a16_fp8_mm(
        x.cuda(), codes.cuda(), s.half().cuda(), gs, out=out[1], out_zeroed=zeroed
    )
    ref = x.float() @ ref_w
    torch.testing.assert_close(out[1].float().cpu(), ref, atol=5e-2, rtol=2e-2)
