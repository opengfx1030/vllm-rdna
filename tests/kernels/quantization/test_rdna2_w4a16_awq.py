# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness tests for the ROCm RDNA2 W4A16 kernel (gfx1030) with AWQ format.

AWQ (uint4, no bias) stores zero points as literal values (no GPTQv1 +1 quirk).
The model's compressed-tensors loader stores zeros as [N//8, G] with
packed_dim=0, and process_weights_after_loading transposes to [G, N//8]
for the kernel.

The exllama fallback cannot handle AWQ (it's GPTQ-only and always adds +1
to stored zeros), so the dispatcher must never route AWQ to exllama.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 W4A16 kernel is ROCm-only", allow_module_level=True)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.rdna2_w4a16 import (
    RDNA2W4A16LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.platforms.rocm import on_gfx10x
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_random_seed

device = "cuda"
WEIGHT_TYPE = scalar_types.uint4
PACK_FACTOR = 8

gfx1030_only = pytest.mark.skipif(
    not (
        on_gfx10x()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna2")
    ),
    reason="requires gfx1030 with the _rocm_C.gptq_gemm_rdna2 op built in",
)


def _reference_awq(x, q, scales, zeros, G):
    K, N = q.shape
    s = scales.repeat_interleave(G, dim=0).float()
    z = zeros.repeat_interleave(G, dim=0).float()
    w = (q.float() - z) * s
    return (x.float() @ w).half()


def _build_layer_awq(q, scales, zeros_logical):
    """Build layer matching the model's compressed-tensors zero layout.

    The model stores zeros as [N//8, G] with packed_dim=0 (AWQ repack format).
    process_weights_after_loading transposes to [G, N//8] for the kernel.
    """
    no_loader = lambda *a, **kw: None
    K, N = q.shape
    qzeros_gN = pack_quantized_values_into_int32(
        zeros_logical, WEIGHT_TYPE, packed_dim=1
    )
    qzeros_packed = qzeros_gN.T.contiguous()
    qweight = pack_quantized_values_into_int32(q, WEIGHT_TYPE, packed_dim=0)

    class L(torch.nn.Module):
        pass

    layer = L()
    layer.register_parameter(
        "qweight",
        PackedvLLMParameter(
            data=qweight,
            weight_loader=no_loader,
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=PACK_FACTOR,
        ),
    )
    layer.register_parameter(
        "scales",
        GroupQuantScaleParameter(
            data=scales, weight_loader=no_loader, input_dim=0, output_dim=1
        ),
    )
    layer.register_parameter(
        "qzeros",
        PackedvLLMParameter(
            data=qzeros_packed,
            weight_loader=no_loader,
            input_dim=1,
            output_dim=0,
            packed_dim=0,
            packed_factor=PACK_FACTOR,
        ),
    )
    return layer


# Realistic Qwen3.8-27B-AWQ shapes (group_size=32, asymmetric)
AWQ_SHAPES = [
    (1, 4096, 4096, 32, "decode-qkv"),
    (1, 4096, 12288, 32, "decode-mlp-up"),
    (1, 12288, 4096, 32, "decode-mlp-down"),
    (16, 4096, 4096, 32, "prefill-qkv"),
    (16, 4096, 12288, 32, "prefill-mlp-up"),
    (16, 12288, 4096, 32, "prefill-mlp-down"),
    (128, 4096, 4096, 32, "chunked-prefill"),
    (2048, 4096, 4096, 32, "full-prefill"),
]


@gfx1030_only
@pytest.mark.parametrize("M,K,N,G,label", AWQ_SHAPES, ids=[s[4] for s in AWQ_SHAPES])
def test_rdna2_w4a16_awq_matches_reference(M, K, N, G, label, dist_init):
    set_random_seed(42)
    groups = K // G
    x = (0.25 * torch.randn(M, K, device=device, dtype=torch.float32)).half()
    q = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales = (
        0.05 * torch.rand(groups, N, device=device, dtype=torch.float32) + 0.01
    ).half()
    zeros = torch.randint(0, 16, (groups, N), device=device, dtype=torch.int32)
    ref = _reference_awq(x, q, scales, zeros, G)

    layer = _build_layer_awq(q, scales, zeros)
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=WEIGHT_TYPE,
        act_type=torch.float16,
        group_size=G,
        zero_points=True,
    )
    ok, reason = RDNA2W4A16LinearKernel.can_implement(config)
    assert ok, f"can_implement rejected: {reason}"
    kernel = RDNA2W4A16LinearKernel(
        config,
        w_q_param_name="qweight",
        w_s_param_name="scales",
        w_zp_param_name="qzeros",
    )
    kernel.process_weights_after_loading(layer)
    out = kernel.apply_weights(layer, x, bias=None)

    rel_l2 = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel_l2 < 5e-2, (
        f"AWQ rel_l2={rel_l2:.4f} exceeds 5e-2 for {label} (M={M},K={K},N={N},G={G})"
    )
    print(
        f"  {label:25s} M={M:5d} K={K:5d} N={N:5d} G={G:3d}  rel_l2={rel_l2:.4f}  PASS"
    )


def _fixed_dense_inputs(M, K, N, *, seed=918):
    from vllm import _custom_ops as ops

    gen = torch.Generator(device=device).manual_seed(seed)
    x = (0.25 * torch.randn(M, K, generator=gen, device=device)).half()
    q = torch.randint(0, 16, (K, N), generator=gen, device=device, dtype=torch.int32)
    scales = (
        0.01 + 0.02 * torch.rand(K // 128, N, generator=gen, device=device)
    ).half()
    zeros = torch.full_like(scales, 8, dtype=torch.int32)
    packed = pack_quantized_values_into_int32(q, WEIGHT_TYPE, packed_dim=0).contiguous()
    ops.gptq_shuffle(packed, 4)
    packed_zeros = pack_quantized_values_into_int32(
        zeros, WEIGHT_TYPE, packed_dim=1
    ).contiguous()
    empty = torch.empty(0, device=device, dtype=torch.int32)
    # Match prep_zero_scale_fp16's HALF-rounded baked constants, then its
    # half FMA. Low/high pairs alternate every two original K elements.
    scale = scales.repeat_interleave(128, 0).float()
    low = ((1024 + q.float()) * scale + (-1032 * scale).half().float()).half()
    high = (
        (1024 + 16 * q.float()) * (scale / 16).half().float()
        + (-72 * scale).half().float()
    ).half()
    low_pair = (torch.arange(K, device=device) % 4 < 2).view(-1, 1)
    weight = torch.where(low_pair, low, high)
    return x, packed, packed_zeros, scales, empty, weight


@gfx1030_only
@pytest.mark.parametrize("op_name", ["gptq_gemm_rdna2", "gptq_gemm_rdna2_prefill"])
@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("M", [1, 3, 24, 128])
@pytest.mark.parametrize("K,N", [(2560, 7168), (3072, 2560), (2560, 48)])
def test_deterministic_dense_repeat_and_reference(op_name, use_v2, M, K, N):
    op = getattr(torch.ops._rocm_C, op_name)
    x, packed, zeros, scales, empty, weight = _fixed_dense_inputs(M, K, N)
    if not use_v2:
        stored_zeros = torch.full((K // 128, N), 7, device=device, dtype=torch.int32)
        zeros = pack_quantized_values_into_int32(
            stored_zeros, WEIGHT_TYPE, packed_dim=1
        )
    args = (x, packed, zeros, scales, empty, use_v2)
    reference = (x.float() @ weight.float()).half()
    first = op(*args, True)
    torch.testing.assert_close(first, reference, rtol=3e-3, atol=1e-3)
    retained = first.clone()
    for repeat in range(8):
        actual = op(*args, True)
        assert torch.equal(actual.view(torch.uint8), retained.view(torch.uint8)), repeat
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(*args, True)
    for repeat in range(8):
        graph.replay()
        assert torch.equal(captured.view(torch.uint8), retained.view(torch.uint8)), (
            repeat
        )
    # Independent invocations must not alias the retained output bank.
    assert torch.equal(first.view(torch.uint8), retained.view(torch.uint8))


@gfx1030_only
@pytest.mark.parametrize("op_name", ["gptq_gemm_rdna2", "gptq_gemm_rdna2_prefill"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 65504.0])
def test_deterministic_dense_fail_closed_per_row(op_name, bad):
    from vllm import _custom_ops as ops

    K, N = 512, 16
    x = torch.full((2, K), 0.001, device=device, dtype=torch.float16)
    x[0] = bad
    q = torch.full((K, N), 9, device=device, dtype=torch.int32)
    packed = pack_quantized_values_into_int32(q, WEIGHT_TYPE, packed_dim=0).contiguous()
    ops.gptq_shuffle(packed, 4)
    scales = torch.full((K // 128, N), 60, device=device, dtype=torch.float16)
    zeros = torch.full((K // 128, N), 8, device=device, dtype=torch.int32)
    zeros = pack_quantized_values_into_int32(
        zeros, WEIGHT_TYPE, packed_dim=1
    ).contiguous()
    out = getattr(torch.ops._rocm_C, op_name)(
        x,
        packed,
        zeros,
        scales,
        torch.empty(0, device=device, dtype=torch.int32),
        True,
        True,
    )
    assert bool(torch.isnan(out[0]).all())
    assert bool(torch.isfinite(out[1]).all())
    torch.testing.assert_close(out[1], (x[1].float().sum() * 60).half().expand(N))


@gfx1030_only
def test_deterministic_decode_rejects_group_boundary_inside_k_step():
    # All packed shapes are valid, but a 40-element group crosses the decode
    # kernel's 32-element dequant step. Reject before launching with stale scales.
    K, N, groups = 5120, 128, 128
    x = torch.zeros((1, K), device=device, dtype=torch.float16)
    packed = torch.zeros((K // 8, N), device=device, dtype=torch.int32)
    zeros = torch.zeros((groups, N // 8), device=device, dtype=torch.int32)
    scales = torch.ones((groups, N), device=device, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="incompatible weight/group shapes"):
        torch.ops._rocm_C.gptq_gemm_rdna2(
            x,
            packed,
            zeros,
            scales,
            torch.empty(0, device=device, dtype=torch.int32),
            True,
            True,
        )
