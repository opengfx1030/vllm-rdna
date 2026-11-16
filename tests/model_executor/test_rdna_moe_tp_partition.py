# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checkpoint ownership tests; these do not qualify the native HIP kernels."""

import math
from types import SimpleNamespace as NS

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import config as moe_config
from vllm.model_executor.layers.fused_moe import layer as factory
from vllm.model_executor.layers.fused_moe.oracle import int_wna16 as oracle
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.platforms import current_platform, rocm


@pytest.fixture
def rdna_tp_cpu(monkeypatch):
    one = NS(world_size=1, rank_in_group=0)
    rank = {"value": 0}
    monkeypatch.setattr(
        moe_config, "get_tensor_model_parallel_rank", lambda: rank["value"]
    )
    monkeypatch.setattr(moe_config, "get_dp_group", lambda: one)
    monkeypatch.setattr(moe_config, "get_pcp_group", lambda: one)
    monkeypatch.setattr(factory, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(factory, "get_dp_group", lambda: one)
    monkeypatch.setattr(factory, "get_pcp_group", lambda: one)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm, "on_gfx10x", lambda: True)
    # GateLinear asks about CUDA/MI350 capabilities even for CPU allocations.
    # This fixture represents gfx1030 without probing any physical device.
    monkeypatch.setattr(current_platform, "is_device_capability", lambda *args: False)
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda *args: False
    )
    monkeypatch.setattr(rocm, "on_gfx950", lambda: False)

    def no_hip_execution(*args, **kwargs):
        raise AssertionError("CPU ownership tests must not execute the native HIP op")

    # Keep the real compressed-tensors dispatcher and oracle. Only advertise
    # the extension's availability: construction/loading must never call it.
    monkeypatch.setattr(
        torch.ops._rocm_C,
        "moe_gptq_gemm_rdna2",
        no_hip_execution,
        raising=False,
    )
    monkeypatch.setattr(ops, "_RDNA2_MOE_FP32_ACCUM", True)
    monkeypatch.setenv("VLLM_RDNA2_W4A8_SDOT4", "0")
    monkeypatch.setattr(
        oracle,
        "_get_priority_backends",
        lambda: [oracle.WNA16MoEBackend.RDNA2_W4A16],
    )
    parallel = NS(
        enable_expert_parallel=False,
        enable_eplb=False,
        enable_dbo=False,
        all2all_backend="allgather_reducescatter",
        expert_placement_strategy="linear",
        elastic_ep_max_dp_size=None,
        tensor_parallel_size=3,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        use_sequence_parallel_moe=False,
    )
    context = NS(
        parallel_config=parallel,
        kernel_config=NS(moe_backend="auto"),
        scheduler_config=NS(max_num_batched_tokens=16),
        model_config=NS(dtype=torch.float16),
        lora_config=None,
        device_config=NS(device=torch.device("cpu")),
        compilation_config=CompilationConfig(mode=0, custom_ops=["none"]),
        additional_config={"rdna_moe_output_fp32": True, "rdna_moe_tp_group_size": 128},
    )
    return NS(rank=rank, context=context)


def _quant(group=128):
    return CompressedTensorsConfig.from_config(
        {
            "format": "pack-quantized",
            "quant_method": "compressed-tensors",
            "ignore": [],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "input_activations": None,
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": group,
                        "actorder": None,
                    },
                }
            },
        }
    )


def _runner(env, rank, partition, *, group=128, quantized=True):
    env.rank["value"] = rank
    env.context.compilation_config.static_forward_context.clear()
    env.context.compilation_config.static_all_moe_layers.clear()
    with set_current_vllm_config(env.context):
        return factory.FusedMoEFactory(
            num_experts=2,
            top_k=2,
            hidden_size=128,
            intermediate_size=640,
            params_dtype=torch.float16,
            quant_config=_quant(group) if quantized else None,
            tp_size=3,
            dp_size=1,
            pcp_size=1,
            prefix="model.layers.0.mlp.experts",
            output_dtype=torch.float32,
            intermediate_partition=partition,
        )


def _sources():
    sources = {}
    for projection, out_dim, in_dim in (
        ("gate_proj", 640, 128),
        ("up_proj", 640, 128),
        ("down_proj", 128, 640),
    ):
        seed = len(sources) + 1
        packed_shape = (2, out_dim, in_dim // 8)
        scale_shape = (2, out_dim, in_dim // 128)
        words = torch.arange(math.prod(packed_shape), dtype=torch.int64)
        packed = (
            ((words * (1234567 * seed) + 0x12345678 * seed) & 0xFFFFFFFF)
            .to(torch.int32)
            .reshape(packed_shape)
        )
        scales = torch.arange(math.prod(scale_shape)).reshape(scale_shape) % 113 + seed
        sources[projection] = {
            "weight_packed": packed,
            "weight_scale": (scales.float() / 4096).half(),
            "weight_shape": torch.tensor([out_dim, in_dim], dtype=torch.int64).repeat(
                2, 1
            ),
        }
    return sources


def _load(layer, sources, batched):
    weights = []
    if batched:
        for suffix in ("weight_packed", "weight_scale"):
            weights.append(
                (
                    f"gate_up_proj{suffix.removeprefix('weight')}",
                    torch.cat(
                        [sources["gate_proj"][suffix], sources["up_proj"][suffix]],
                        dim=1,
                    ),
                )
            )
            weights.append(
                (
                    f"down_proj{suffix.removeprefix('weight')}",
                    sources["down_proj"][suffix],
                )
            )
    for projection, tensors in sources.items():
        for suffix, value in tensors.items():
            if not batched or suffix == "weight_shape":
                weights.extend(
                    (f"{expert}.{projection}.{suffix}", value[expert])
                    for expert in range(value.shape[0])
                )
    list(layer.routed_experts.load_weights(weights))


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "partitions",
    [
        ((0, 256), (256, 256), (512, 128)),
        ((512, 128), (0, 256), (256, 256)),
        ((256, 256), (512, 128), (0, 256)),
    ],
)
def test_group_aligned_tp_reconstructs_every_packed_column_and_scale(
    rdna_tp_cpu, batched, partitions
):
    source = _sources()
    shards = []
    for rank, partition in enumerate(partitions):
        layer = _runner(rdna_tp_cpu, rank, partition)
        _load(layer, source, batched)
        shards.append((partition[0], layer.routed_experts))
    ordered = [layer for _, layer in sorted(shards)]
    for suffix in ("weight_packed", "weight_scale"):
        gate = torch.cat(
            [getattr(layer, "w13_" + suffix).chunk(2, dim=1)[0] for layer in ordered],
            dim=1,
        )
        up = torch.cat(
            [getattr(layer, "w13_" + suffix).chunk(2, dim=1)[1] for layer in ordered],
            dim=1,
        )
        down = torch.cat([getattr(layer, "w2_" + suffix) for layer in ordered], dim=2)
        assert torch.equal(gate, source["gate_proj"][suffix])
        assert torch.equal(up, source["up_proj"][suffix])
        assert torch.equal(down, source["down_proj"][suffix])


@pytest.mark.parametrize("shape", [(128, 32), (128, 79)])
def test_group_aligned_tp_rejects_incomplete_down_projection(rdna_tp_cpu, shape):
    layer = _runner(rdna_tp_cpu, 2, (512, 128)).routed_experts
    before = layer.w2_weight_packed.detach().clone()
    with pytest.raises(ValueError):
        layer.weight_loader(
            layer.w2_weight_packed,
            torch.zeros(shape, dtype=torch.int32),
            "down_proj.weight_packed",
            "w2",
            0,
        )
    assert torch.equal(layer.w2_weight_packed, before)


def test_group_aligned_tp_rejects_actual_scale_group_crossing(rdna_tp_cpu):
    with pytest.raises((ValueError, NotImplementedError)):
        _runner(rdna_tp_cpu, 2, (512, 128), group=256)


def test_group_aligned_tp_rejects_unquantized_experts(rdna_tp_cpu):
    with pytest.raises((ValueError, NotImplementedError)):
        _runner(rdna_tp_cpu, 0, (0, 256), quantized=False)


def test_group_aligned_tp_preserves_full_shared_expert(rdna_tp_cpu, monkeypatch):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import activation, linear
    from vllm.model_executor.models import qwen3_next
    from vllm.platforms.cpu import CpuPlatform

    # Execute the real native CPU activation, not an unavailable HIP extension.
    monkeypatch.setattr(activation, "current_platform", CpuPlatform())

    env = rdna_tp_cpu
    env.rank["value"] = 2
    group = NS(world_size=3, rank_in_group=2, device_group=None)
    monkeypatch.setattr(qwen3_next, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(qwen3_next, "get_ep_group", lambda: group)
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 2)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.float16)
    monkeypatch.setattr(
        linear,
        "dispatch_unquantized_gemm",
        lambda *args, **kwargs: (
            lambda layer, x, weight, bias: torch.nn.functional.linear(x, weight, bias)
        ),
    )
    env.context.kernel_config.linear_backend = "auto"
    env.context.parallel_config.eplb_config = NS(num_redundant_experts=0)
    env.context.model_config.hf_text_config = NS(
        hidden_size=128,
        num_experts=6,
        num_experts_per_tok=2,
        hidden_act="silu",
        shared_expert_intermediate_size=640,
        moe_intermediate_size=640,
    )
    env.context.quant_config = _quant()
    env.context.quant_config.ignore = ["re:.*shared_expert.*"]
    with set_current_vllm_config(env.context):
        block = qwen3_next.Qwen3NextSparseMoeBlock(
            env.context,
            prefix="model.layers.0.mlp",
            output_dtype=torch.float32,
            intermediate_partition=(512, 128),
        )
        shared = block.shared_expert
        generator = torch.Generator().manual_seed(620)
        gate, up = torch.randn(2, 640, 128, generator=generator).half() / 32
        down = torch.randn(128, 640, generator=generator).half() / 32
        shared.gate_up_proj.weight.weight_loader(shared.gate_up_proj.weight, gate, 0)
        shared.gate_up_proj.weight.weight_loader(shared.gate_up_proj.weight, up, 1)
        shared.down_proj.weight.weight_loader(shared.down_proj.weight, down)
        with torch.no_grad():
            block.shared_expert_gate.weight.fill_(1 / 128)
        hidden = torch.randn(3, 128, generator=generator).half()
        actual = shared(hidden)
    linear_op = torch.nn.functional.linear
    expected = linear_op(
        torch.nn.functional.silu(linear_op(hidden, gate)) * linear_op(hidden, up),
        down,
    ) * torch.sigmoid(linear_op(hidden, block.shared_expert_gate.weight))
    torch.testing.assert_close(actual, expected)
