# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asymmetric W4A16 MoE checkpoints on the native RDNA2 MoE path."""

import os

import pytest
import torch

pytest.importorskip("compressed_tensors")

from compressed_tensors.quantization import QuantizationArgs  # noqa: E402

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E402, E501
    rocm_moe_rdna,
)
from vllm.platforms import current_platform  # noqa: E402
from tests.utils import spawn_new_process_for_each_test  # noqa: E402


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
@spawn_new_process_for_each_test
def test_rdna2_ct_rank7_global_routes_match_local_dequant_reference():
    """Exercise the EP8 tile8 dispatch and real CT loading without collectives."""
    os.environ["VLLM_RDNA2_MOE_TILE8_MIN_TOKENS"] = "5"
    os.environ["VLLM_RDNA2_W4A8_SDOT4"] = "0"
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        ExpertMapManager,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_rdna2 import (  # noqa: E501
        CompressedTensorsWNA16RDNA2MoEMethod,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_rdna2 import (  # noqa: E501
        _TILE8_MIN_TOKENS,
    )

    assert _TILE8_MIN_TOKENS == 5
    torch.manual_seed(71)
    device = torch.device("cuda")
    hidden, inter, group, top_k = 2560, 640, 32, 10
    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=8,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=7,
        sp_size=1,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    moe = FusedMoEConfig(
        num_experts=512,
        experts_per_token=top_k,
        hidden_dim=hidden,
        intermediate_size=inter,
        num_local_experts=64,
        num_logical_experts=512,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel,
        in_dtype=torch.float16,
    )
    weights = QuantizationArgs(
        num_bits=4, type="int", strategy="group", group_size=group, symmetric=False
    )
    quant = CompressedTensorsConfig(
        target_scheme_map={
            "Linear": {
                "weights": weights,
                "input_activations": None,
                "format": "pack-quantized",
            }
        },
        ignore=[],
        quant_format="pack-quantized",
    )
    with device:
        manager = ExpertMapManager(
            max_num_batched_tokens=5,
            top_k=top_k,
            global_num_experts=512,
            num_redundant_experts=0,
            num_expert_group=None,
            moe_parallel_config=parallel,
            placement_strategy="linear",
            enable_eplb=False,
        )
        layer = RoutedExperts(
            layer_name="rank7_group32.experts",
            params_dtype=torch.float16,
            moe_config=moe,
            quant_config=quant,
            expert_map_manager=manager,
        )
    method = layer.quant_method
    assert isinstance(method, CompressedTensorsWNA16RDNA2MoEMethod)
    assert layer.expert_map[448].item() == 0
    assert layer.expert_map[511].item() == 63
    assert layer.expert_map[0].item() == layer.expert_map[447].item() == -1

    projections = (
        ("gate_proj", "w1", "w13", inter, hidden),
        ("up_proj", "w3", "w13", inter, hidden),
        ("down_proj", "w2", "w2", hidden, inter),
    )
    reference_weights = {}
    for source_expert, destinations in (
        (448, (0, 447, *range(448, 511))),
        (511, (511,)),
    ):
        reference_weights[source_expert] = []
        for projection, shard, stem, n, k in projections:
            packed, scale, zeros, dequantized = _make_quantized_weights(
                1, n, k, group, False, device
            )
            reference_weights[source_expert].append(dequantized[0])
            for suffix, tensor in (
                ("weight_packed", packed),
                ("weight_scale", scale),
                ("weight_zero_point", zeros),
            ):
                param = getattr(layer, f"{stem}_{suffix}")
                for global_id in destinations:
                    loaded = param.weight_loader(
                        param,
                        tensor[0],
                        f"experts.{global_id}.{projection}.{suffix}",
                        shard,
                        global_id,
                        return_success=True,
                    )
                    assert loaded is (global_id >= 448)

    method.process_weights_after_loading(layer)
    assert method._zero_v2
    assert layer.w13_weight_packed.shape == (64, 320, 1280)
    assert layer.w2_weight_packed.shape == (64, 80, 2560)
    assert layer.w13_weight_scale.shape == (64, 80, 1280)
    assert layer.w2_weight_scale.shape == (64, 20, 2560)
    assert layer.w13_qzeros.shape == (64, 80, 160)
    assert layer.w2_qzeros.shape == (64, 20, 320)

    x = (torch.randn(5, hidden, device=device) * 0.25).half()
    routes = [
        [448, 511, 0, 447, 1, 2, 3, 4, 5, 6],
        [511, 0, 447, 1, 2, 3, 4, 5, 6, 7],
        [448, 0, 447, 1, 2, 3, 4, 5, 6, 7],
        [0, 447, 1, 2, 3, 4, 5, 6, 7, 8],
        [447, 511, 0, 448, 1, 2, 3, 4, 5, 6],
    ]
    router_weights = torch.softmax(torch.randn(5, top_k, device=device), dim=-1)
    reference = torch.zeros(5, hidden, device=device, dtype=torch.float32)
    for token, ids in enumerate(routes):
        for slot, global_id in enumerate(ids):
            if global_id not in reference_weights:
                continue
            gate, up, down = reference_weights[global_id]
            activated = torch.nn.functional.silu(gate @ x[token].float()) * (
                up @ x[token].float()
            )
            reference[token] += router_weights[token, slot] * (down @ activated)
    assert reference[[0, 1, 2, 4]].norm(dim=1).min() > 0
    for ids_dtype in (torch.int32, torch.int64):
        # int32 exercises fused route/zero; int64 exercises block alignment.
        ids = torch.tensor(routes, device=device, dtype=ids_dtype)
        output = method.apply(layer, x, router_weights, ids, None, None)
        assert output.shape == reference.shape
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output[3]) == 0
        for token in (0, 1, 2, 4):
            relative_l2 = (
                (output[token].float() - reference[token]).norm()
                / reference[token].norm()
            )
            assert relative_l2 < 1e-2, (
                f"token {token}, ids {ids_dtype}: relative error {relative_l2:.4f}"
            )


@needs_rdna2
@pytest.mark.parametrize("copy_mode", ["tensor", "native"])
def test_rdna2_ct_expert_cache_matches_resident_across_overflow_and_streams(copy_mode):
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
