# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from tests.kernels.quant_utils import FP8_DTYPE
from tests.kernels.utils import fp8_allclose, fp8_ulp_distance, opcheck
from vllm import ir
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx90a

    on_mi250 = on_gfx90a()
else:
    on_mi250 = False

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [8, 768, 769, 5120, 5125, 8192]  # Arbitrary values for testing
ADD_RESIDUAL = [False, True] if not on_mi250 else [True]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


def _rms_norm_tolerance(dtype: torch.dtype) -> dict[str, float]:
    return ir.ops.rms_norm.get_tolerance(dtype)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x[..., :hidden_size]
    assert x.is_contiguous() != strided_input
    x *= scale
    residual = x.new_empty_strided(x.size(), x.stride()) if add_residual else None
    if residual is not None:
        residual.normal_(std=scale)

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    if residual is not None:
        opcheck(
            torch.ops._C.fused_add_rms_norm,
            (x, residual, layer.weight.data, layer.variance_epsilon),
        )
    else:
        opcheck(
            torch.ops._C.rms_norm, (out, x, layer.weight.data, layer.variance_epsilon)
        )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm_weightless(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size, has_weight=False).to(dtype=dtype)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    residual = torch.randn_like(x) if add_residual else None

    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    tol = _rms_norm_tolerance(dtype)
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], **tol)
        torch.testing.assert_close(out[1], ref_out[1], **tol)
    else:
        torch.testing.assert_close(out, ref_out, **tol)

    if residual is not None:
        opcheck(
            torch.ops._C.fused_add_rms_norm,
            (x, residual, None, layer.variance_epsilon),
        )
    else:
        opcheck(
            torch.ops._C.rms_norm,
            (out, x, None, layer.variance_epsilon),
        )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_scale", [0.01, 1.0, 10.0])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
def test_fused_rms_norm_quant(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    quant_scale: float,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)

    weight = torch.empty(hidden_size, dtype=dtype).normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x_base = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x_base[..., :hidden_size]
    assert x.is_contiguous() != strided_input

    x *= scale
    if add_residual:
        residual = torch.randn_like(x) * scale
        residual_fused = residual.clone()
    else:
        residual = residual_fused = None

    out_norm = torch.empty_like(x)
    out_quant = torch.empty_like(x, dtype=FP8_DTYPE)
    out_quant_fused = torch.empty_like(out_quant)

    quant_scale_t = torch.tensor(quant_scale, dtype=torch.float32)

    if add_residual:
        torch.ops._C.fused_add_rms_norm_static_fp8_quant(
            out_quant_fused, x, residual_fused, weight, quant_scale_t, 1e-6
        )

        # Unfused kernel is in-place so it goes second
        # Also use a separate clone of x to avoid modifying the input
        x_unfused_base = x_base.clone()
        x_unfused = x_unfused_base[..., :hidden_size]
        assert x_unfused.is_contiguous() != strided_input
        torch.ops._C.fused_add_rms_norm(x_unfused, residual, weight, 1e-6)
        torch.ops._C.static_scaled_fp8_quant(
            out_quant, x_unfused.contiguous(), quant_scale_t
        )

        torch.accelerator.synchronize()
        torch.testing.assert_close(residual_fused, residual, atol=1e-2, rtol=1e-2)
        opcheck(
            torch.ops._C.fused_add_rms_norm_static_fp8_quant,
            (out_quant_fused, x, residual_fused, weight, quant_scale_t, 1e-6),
        )
    else:
        torch.ops._C.rms_norm_static_fp8_quant(
            out_quant_fused, x, weight, quant_scale_t, 1e-6
        )

        torch.ops._C.rms_norm(out_norm, x, weight, 1e-6)
        torch.ops._C.static_scaled_fp8_quant(out_quant, out_norm, quant_scale_t)

        opcheck(
            torch.ops._C.rms_norm_static_fp8_quant,
            (out_quant_fused, x, weight, quant_scale_t, 1e-6),
        )

    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx950

        if on_gfx950() and dtype == torch.float16 and not add_residual:
            # Fusion may round normalized FP16 across an E4M3 boundary on gfx950.
            assert fp8_allclose(out_quant_fused, out_quant, rtol=0.125, atol=2e-3)
            assert int(fp8_ulp_distance(out_quant_fused, out_quant).max()) <= 1
        else:
            # Fused and unfused FP8 paths can land on opposite sides of an E4M3
            # tie; tolerate a tiny number of isolated fp8 outliers on ROCm.
            ulp = fp8_ulp_distance(out_quant, out_quant_fused)
            max_outliers = ulp.numel() // 100_000 + 8
            num_outliers = int((ulp > 0).sum().item())
            assert num_outliers <= max_outliers, (
                f"FP8 quant mismatch: {num_outliers} fp8 outliers "
                f"(allowed {max_outliers})"
            )
    else:
        torch.testing.assert_close(
            out_quant.to(dtype=torch.float32),
            out_quant_fused.to(dtype=torch.float32),
            atol=1e-3,
            rtol=1e-3,
        )


@torch.inference_mode()
def test_gemma_rms_norm_mixed_input_weight_dtype(default_vllm_config) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = CUDA_DEVICES[0]
    torch.set_default_device(device)

    num_tokens, hidden_size = 32, 1024
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    layer = GemmaRMSNorm(hidden_size, eps=1e-6).to(device=device)
    layer.weight.data.normal_(mean=0.0, std=0.1)

    # Gemma uses fp32 weight parameter while activations can be bf16.
    assert layer.weight.dtype == torch.float32
    out = layer(x)

    x_fp32 = x.float()
    weight_fp32 = layer.weight.data.float() + 1.0
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    ref = (x_fp32 * torch.rsqrt(variance + layer.variance_epsilon) * weight_fp32).to(
        x.dtype
    )

    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


_RDNA_GEMMA_NORM = (
    current_platform.is_rocm()
    and hasattr(torch.ops, "_rocm_C")
    and hasattr(torch.ops._rocm_C, "gemma_rms_norm")
)


def _gemma_reference(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # (1 + w) rounded to the activation dtype first, fp32 mean of squares.
    scale = (w.float() + 1.0).to(x.dtype).float()
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * scale).to(x.dtype)


@pytest.mark.skipif(not _RDNA_GEMMA_NORM, reason="RDNA HIP gemma_rms_norm not built")
@pytest.mark.parametrize(
    "shape", [(1, 2560), (24, 2560), (5, 1024), (7, 3, 256), (24, 16, 256)]
)
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_gemma_rms_norm_hip_matches_reference(default_vllm_config, shape, strided):
    torch.manual_seed(0)
    if strided:
        # A token-strided view of a wider tensor (q after the qkv split): 2D
        # row-strided views take the HIP kernel, 3D ones stay on the old path.
        inner = math.prod(shape[1:])
        base = torch.randn(shape[0], 2 * inner, dtype=torch.half, device="cuda")
        x = base[:, :inner].view(shape)
    else:
        x = torch.randn(shape, dtype=torch.half, device="cuda")
    w = torch.randn(shape[-1], dtype=torch.half, device="cuda") * 0.1
    out = torch.ops.vllm.gemma_rms_norm(x, w, 1e-6)
    assert out.shape == x.shape and out.dtype == torch.half and out.is_contiguous()
    torch.testing.assert_close(out, _gemma_reference(x, w, 1e-6), atol=4e-3, rtol=2e-3)
    # The eager path (cast, add, cast, _C norm): the same fp32 math and a single
    # rounding. Only the order of the sum of squares differs, which flips an
    # fp16 ulp in a few elements per hundred thousand.
    scale = (w.float() + 1.0).to(x.dtype)
    old = torch.empty(shape, dtype=torch.half, device="cuda")
    torch.ops._C.rms_norm(
        old.view(-1, shape[-1]), x.reshape(-1, shape[-1]), scale, 1e-6
    )
    if not strided or len(shape) == 2:  # else the eager path ran itself
        assert (out != old).float().mean() < 1e-3
    torch.testing.assert_close(out, old, atol=4e-3, rtol=1.5e-3)
    assert torch.equal(out, torch.ops.vllm.gemma_rms_norm(x, w, 1e-6))


@pytest.mark.skipif(not _RDNA_GEMMA_NORM, reason="RDNA HIP gemma_rms_norm not built")
@pytest.mark.parametrize("offset", [0, 3])
@pytest.mark.parametrize("lead", [(24,), (24, 1)])
@torch.inference_mode()
def test_gemma_rms_norm_hip_row_strided_is_one_kernel(
    default_vllm_config, offset, lead
):
    """A row-strided view (pooled keys cut out of a wider buffer, 2D or with a
    unit middle dim) is normed in place: one launch, no copy, bitwise equal to
    the contiguous input."""
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(1)
    base = torch.randn(24, 640, dtype=torch.half, device="cuda")
    x = base[:, offset : offset + 128].view(*lead, 128)
    assert not x.is_contiguous() and x.stride(0) == 640 and x.stride(-1) == 1
    w = torch.randn(128, dtype=torch.half, device="cuda") * 0.1
    out = torch.ops.vllm.gemma_rms_norm(x, w, 1e-6)
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.ops.vllm.gemma_rms_norm(x, w, 1e-6)
        torch.accelerator.synchronize()
    names = [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    if names:
        assert len(names) == 1 and "rms_norm_kernel" in names[0], names
    xc = x.contiguous()
    torch.testing.assert_close(out, _gemma_reference(xc, w, 1e-6), atol=4e-3, rtol=2e-3)
    assert torch.equal(out, torch.ops.vllm.gemma_rms_norm(xc, w, 1e-6))


@pytest.mark.skipif(not _RDNA_GEMMA_NORM, reason="RDNA HIP gemma_rms_norm not built")
@torch.inference_mode()
def test_gemma_rms_norm_hip_empty_batch(default_vllm_config):
    x = torch.empty(0, 256, dtype=torch.half, device="cuda")
    w = torch.zeros(256, dtype=torch.half, device="cuda")
    out = torch.ops.vllm.gemma_rms_norm(x, w, 1e-6)
    assert out.shape == (0, 256) and out.dtype == torch.half


@pytest.mark.skipif(not _RDNA_GEMMA_NORM, reason="RDNA HIP gemma_rms_norm not built")
@torch.inference_mode()
def test_gemma_rms_norm_hip_keeps_old_path_for_other_dtypes(default_vllm_config):
    torch.manual_seed(0)
    x = torch.randn(8, 256, dtype=torch.half, device="cuda")
    w = torch.randn(256, dtype=torch.float32, device="cuda") * 0.1  # fp32 weight
    out = torch.ops.vllm.gemma_rms_norm(x, w, 1e-6)
    old = ir.ops.rms_norm(x, (w + 1.0).to(x.dtype), 1e-6)
    assert torch.equal(out, old)
    xb = x.bfloat16()
    outb = torch.ops.vllm.gemma_rms_norm(xb, w.bfloat16(), 1e-6)
    oldb = ir.ops.rms_norm(xb, (w.bfloat16().float() + 1.0).to(xb.dtype), 1e-6)
    assert torch.equal(outb, oldb)
