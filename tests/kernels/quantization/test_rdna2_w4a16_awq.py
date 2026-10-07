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
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearLayerConfig
from vllm.model_executor.kernels.linear.mixed_precision.rdna2_w4a16 import RDNA2W4A16LinearKernel
from vllm.model_executor.layers.quantization.utils.quant_utils import pack_quantized_values_into_int32
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.platforms.rocm import on_gfx10x
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_random_seed

device = "cuda"
WEIGHT_TYPE = scalar_types.uint4
PACK_FACTOR = 8

gfx1030_only = pytest.mark.skipif(
    not (on_gfx10x() and hasattr(torch.ops, "_rocm_C") and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna2")),
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
    qzeros_gN = pack_quantized_values_into_int32(zeros_logical, WEIGHT_TYPE, packed_dim=1)
    qzeros_packed = qzeros_gN.T.contiguous()
    qweight = pack_quantized_values_into_int32(q, WEIGHT_TYPE, packed_dim=0)
    class L(torch.nn.Module): pass
    layer = L()
    layer.register_parameter("qweight", PackedvLLMParameter(
        data=qweight, weight_loader=no_loader, input_dim=0, output_dim=1, packed_dim=0, packed_factor=PACK_FACTOR))
    layer.register_parameter("scales", GroupQuantScaleParameter(
        data=scales, weight_loader=no_loader, input_dim=0, output_dim=1))
    layer.register_parameter("qzeros", PackedvLLMParameter(
        data=qzeros_packed, weight_loader=no_loader, input_dim=1, output_dim=0, packed_dim=0, packed_factor=PACK_FACTOR))
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
    scales = (0.05 * torch.rand(groups, N, device=device, dtype=torch.float32) + 0.01).half()
    zeros = torch.randint(0, 16, (groups, N), device=device, dtype=torch.int32)
    ref = _reference_awq(x, q, scales, zeros, G)

    layer = _build_layer_awq(q, scales, zeros)
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N), partition_weight_shape=(K, N),
        weight_type=WEIGHT_TYPE, act_type=torch.float16,
        group_size=G, zero_points=True, has_g_idx=False)
    ok, reason = RDNA2W4A16LinearKernel.can_implement(config)
    assert ok, f"can_implement rejected: {reason}"
    kernel = RDNA2W4A16LinearKernel(
        config, w_q_param_name="qweight", w_s_param_name="scales",
        w_zp_param_name="qzeros", w_gidx_param_name=None)
    kernel.process_weights_after_loading(layer)
    out = kernel.apply_weights(layer, x, bias=None)

    rel_l2 = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel_l2 < 5e-2, (
        f"AWQ rel_l2={rel_l2:.4f} exceeds 5e-2 for {label} "
        f"(M={M},K={K},N={N},G={G})")
    print(f"  {label:25s} M={M:5d} K={K:5d} N={N:5d} G={G:3d}  "
          f"rel_l2={rel_l2:.4f}  PASS")
