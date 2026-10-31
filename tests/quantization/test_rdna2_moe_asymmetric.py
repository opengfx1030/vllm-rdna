# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asymmetric W4A16 MoE checkpoints on the native RDNA2 MoE path."""

from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("compressed_tensors")

from compressed_tensors.quantization import QuantizationArgs  # noqa: E402

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E402, E501
    rocm_moe_rdna,
)
from vllm.platforms import current_platform  # noqa: E402


def test_asymmetric_8bit_moe_is_not_native_rdna2():
    args = QuantizationArgs(
        num_bits=8, symmetric=False, strategy="group", group_size=32
    )
    assert rocm_moe_rdna.is_supported(args) is False


def _on_gfx10x_with_op() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx10x

    return (
        on_gfx10x()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna2")
    )


needs_rdna2 = pytest.mark.skipif(
    not _on_gfx10x_with_op(), reason="needs gfx10x with the RDNA2 MoE op"
)


@needs_rdna2
def test_asymmetric_4bit_moe_is_native_rdna2():
    args = QuantizationArgs(
        num_bits=4, symmetric=False, strategy="group", group_size=32
    )
    assert rocm_moe_rdna.is_supported(args) is True


def _dequant(q, zp, scale, group):
    return (q - zp.repeat_interleave(group, dim=2)).float() * scale.repeat_interleave(
        group, dim=2
    ).float()


def _make_quantized_weights(num_experts, n, k, group, symmetric, device):
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32,
    )
    from vllm.scalar_type import scalar_types

    q = torch.randint(0, 16, (num_experts, n, k), dtype=torch.int32, device=device)
    if symmetric:
        zp = torch.full(
            (num_experts, n, k // group), 8, dtype=torch.int32, device=device
        )
    else:
        zp = torch.randint(
            0, 16, (num_experts, n, k // group), dtype=torch.int32, device=device
        )
    scale = (
        torch.rand(num_experts, n, k // group, device=device) * 0.01 + 0.005
    ).half()
    packed = pack_quantized_values_into_int32(q, scalar_types.uint4, packed_dim=2)
    zp_packed = pack_quantized_values_into_int32(zp, scalar_types.uint4, packed_dim=1)
    return (
        packed.contiguous(),
        scale,
        zp_packed.contiguous(),
        _dequant(q, zp, scale, group),
    )


@needs_rdna2
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 5, 64])
def test_rdna2_moe_matches_dequant_reference(symmetric, num_tokens):
    """Compressed-tensors layout in, fused MoE out, against a torch reference."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_rdna2 import (  # noqa: E501
        CompressedTensorsWNA16RDNA2MoEMethod,
        _rdna2_fused_moe,
    )

    torch.manual_seed(0)
    dev = torch.device("cuda")
    num_experts, hidden, inter, group, top_k = 4, 256, 128, 32, 2

    def make(n, k):
        return _make_quantized_weights(num_experts, n, k, group, symmetric, dev)

    w13_p, w13_s, w13_zp, w13 = make(2 * inter, hidden)
    w2_p, w2_s, w2_zp, w2 = make(hidden, inter)

    layer = torch.nn.Module()
    params = {
        "w13_weight_packed": w13_p,
        "w2_weight_packed": w2_p,
        "w13_weight_scale": w13_s,
        "w2_weight_scale": w2_s,
    }
    if not symmetric:
        params["w13_weight_zero_point"] = w13_zp
        params["w2_weight_zero_point"] = w2_zp
    for name, t in params.items():
        layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))

    method = object.__new__(CompressedTensorsWNA16RDNA2MoEMethod)
    method.group_size = group
    method.symmetric = symmetric
    method.process_weights_after_loading(layer)
    assert method._zero_v2 is (not symmetric)

    x = (torch.randn(num_tokens, hidden, device=dev) * 0.5).half()
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=dev)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_w = torch.softmax(torch.randn(num_tokens, top_k, device=dev), dim=-1)

    out = _rdna2_fused_moe(
        x,
        topk_w,
        topk_ids,
        layer=layer,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        global_num_experts=num_experts,
        expert_map=None,
        zero_v2=method._zero_v2,
    )

    ref = torch.zeros(num_tokens, hidden, device=dev)
    for t in range(num_tokens):
        for j in range(top_k):
            e = int(topk_ids[t, j])
            h = w13[e] @ x[t].float()
            a = torch.nn.functional.silu(h[:inter]) * h[inter:]
            ref[t] += topk_w[t, j] * (w2[e] @ a)

    rel = (out.float() - ref).norm() / ref.norm()
    assert rel < 1e-2, f"relative error {rel:.4f}"


@needs_rdna2
@pytest.mark.parametrize("copy_mode", ["tensor", "native"])
@pytest.mark.parametrize("overlap", [False, True])
def test_rdna2_ct_expert_cache_matches_resident_across_overflow_and_streams(
    copy_mode, overlap
):
    """Real CT packing and HIP execution retain every routed pair under churn."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.rdna_expert_cache import PAYLOAD_NAMES
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_rdna2 import (  # noqa: E501
        CompressedTensorsWNA16RDNA2MoEMethod,
    )

    torch.manual_seed(17)
    device = torch.device("cuda")
    experts, hidden, intermediate, group = 4, 256, 128, 128
    w13, s13, _, _ = _make_quantized_weights(
        experts, intermediate * 2, hidden, group, True, device
    )
    w2, s2, _, _ = _make_quantized_weights(
        experts, hidden, intermediate, group, True, device
    )
    resident, pooled = torch.nn.Module(), torch.nn.Module()
    for name, tensor in {
        "w13_weight_packed": w13,
        "w2_weight_packed": w2,
        "w13_weight_scale": s13,
        "w2_weight_scale": s2,
    }.items():
        resident.register_parameter(
            name, torch.nn.Parameter(tensor, requires_grad=False)
        )
        pooled.register_parameter(
            name, torch.nn.Parameter(tensor.cpu(), requires_grad=False)
        )
    for layer in (resident, pooled):
        layer.activation = MoEActivation.SILU
        layer.apply_router_weight_on_input = False
        layer.global_num_experts = experts
        layer.expert_map = None

    method = object.__new__(CompressedTensorsWNA16RDNA2MoEMethod)
    method.group_size = group
    method.symmetric = True
    method.process_weights_after_loading(resident)
    method.moe = Mock()
    method.moe.moe_parallel_config.ep_rank = 0
    pooled.layer_name = "model.layers.0.mlp.experts"
    pooled._rdna_expert_cache_overlap = overlap
    pooled._rdna_expert_cache_slots = 2
    pooled._rdna_expert_cache_device = device
    pooled._rdna_expert_cache_policy = "lfu_lru"
    pooled._rdna_expert_cache_decay = 128
    pooled._rdna_expert_cache_copy = copy_mode
    method.process_weights_after_loading(pooled)
    cache = pooled._rdna_expert_cache
    for name in PAYLOAD_NAMES:
        assert torch.equal(cache.host_payload[name], getattr(resident, name).cpu())
        assert cache.host_payload[name].is_pinned()
    assert cache.logical_num_experts == 4 and cache.slot_count == 2
    pointers = {name: tensor.data_ptr() for name, tensor in cache.payload.items()}
    x = (torch.randn(4, hidden, device=device) * 0.5).half()
    weights = torch.softmax(torch.randn(4, 2, device=device), dim=-1)
    routes = [
        torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]], device=device).int(),
        torch.tensor([[2, 3]] * 4, device=device).int(),
        torch.tensor([[2, 3]] * 4, device=device).int(),
        torch.tensor([[0, 1]] * 4, device=device).int(),
    ]
    references = [method.apply(resident, x, weights, ids, None, None) for ids in routes]
    torch.accelerator.synchronize()
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    outputs = []
    for index, ids in enumerate(routes):
        before = cache.copied_bytes
        with torch.cuda.stream(streams[index % 2]):
            outputs.append(method.apply(pooled, x, weights, ids, None, None))
        if index in (1, 2):
            assert cache.copied_bytes == before, "hot experts must not be copied again"
    for stream in streams:
        stream.synchronize()
    for actual, expected in zip(outputs, references, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)
    assert {
        name: tensor.data_ptr() for name, tensor in cache.payload.items()
    } == pointers
