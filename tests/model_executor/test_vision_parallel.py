# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unequal vision projection arithmetic and real three-process Gloo gathers."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.utils import AutoWeightsLoader


@pytest.mark.parametrize("text_tp_size", [1, 3])
@pytest.mark.parametrize("separate", [False, True])
def test_vision_qkv_loader_preserves_major_order_and_all_real_heads(
    monkeypatch, separate, text_tp_size
):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers.vision_parallel import VisionColumnParallelLinear
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

    generator = torch.Generator().manual_seed(37)
    weight = torch.randn(3 * 1152, 4, generator=generator)
    bias = torch.randn(3 * 1152, generator=generator)
    x = torch.randn(2, 4, generator=generator)
    local_outputs = []
    starts = []
    for rank in range(3):
        monkeypatch.setattr(
            parameter,
            "get_tensor_model_parallel_rank",
            lambda rank=rank: rank if text_tp_size == 3 else 0,
        )
        monkeypatch.setattr(
            parameter, "get_tensor_model_parallel_world_size", lambda: text_tp_size
        )
        layer = VisionColumnParallelLinear(
            4,
            1152,
            alignment=72,
            packed_qkv=True,
            vision_group=SimpleNamespace(world_size=3, rank_in_group=rank),
        )
        weights = {}
        if separate:
            for index, shard_id in enumerate(("q", "k", "v")):
                weights[f"attn.{shard_id}.weight"] = weight[
                    index * 1152 : (index + 1) * 1152
                ]
                weights[f"attn.{shard_id}.bias"] = bias[
                    index * 1152 : (index + 1) * 1152
                ]
            mapper = Qwen3_VisionTransformer.hf_to_vllm_mapper
        else:
            weights = {"attn.qkv.weight": weight, "attn.qkv.bias": bias}
            mapper = None
        tower = torch.nn.Module()
        tower.add_module("attn", torch.nn.Module())
        tower.attn.add_module("qkv", layer)
        loaded = AutoWeightsLoader(tower).load_weights(weights.items(), mapper=mapper)
        assert loaded == {"attn.qkv.weight", "attn.qkv.bias"}
        layer.quant_method.process_weights_after_loading(layer)
        local, _ = layer(x)
        local_outputs.append(local.reshape(2, 3, -1))
        starts.append((layer.partition_start, layer.partition_size))
    assert starts == [(0, 432), (432, 360), (792, 360)]
    actual = torch.cat(local_outputs, dim=-1).reshape(2, 3456)
    torch.testing.assert_close(actual, torch.nn.functional.linear(x, weight, bias))


def _vision_output_gloo_worker(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist

    from vllm.distributed import parallel_state
    from vllm.model_executor.layers import linear
    from vllm.model_executor.layers.vision_parallel import VisionOutputParallelLinear

    class Group:
        world_size = 3

        def __init__(self):
            self.rank_in_group = rank

        def all_gather(self, value, dim):
            assert value.is_contiguous()
            pieces = [torch.empty_like(value) for _ in range(3)]
            dist.all_gather(pieces, value)
            return torch.cat(pieces, dim=dim)

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=45),
    )
    try:
        with pytest.MonkeyPatch.context() as patch, torch.inference_mode():
            patch.setattr(
                linear,
                "dispatch_unquantized_gemm",
                lambda *args, **kwargs: lambda layer, x, weight, bias: (
                    torch.nn.functional.linear(x, weight, bias)
                ),
            )
            generator = torch.Generator().manual_seed(42)
            for text_tp_size in (1, 3):
                patch.setattr(
                    parallel_state,
                    "_TP",
                    SimpleNamespace(
                        world_size=text_tp_size,
                        rank_in_group=rank if text_tp_size == 3 else 0,
                    ),
                )
                for width, alignment, outputs, has_bias in (
                    (1152, 72, 7, True),
                    (4304, 16, 7, True),
                    (4608, 1, 9, False),
                ):
                    source = torch.randint(
                        -3,
                        4,
                        (outputs, width),
                        generator=generator,
                    ).bfloat16()
                    backing = torch.randint(
                        -2,
                        3,
                        (3, width * 2),
                        generator=generator,
                    ).half()
                    hidden = backing[:, ::2]
                    second_start = ((width // alignment + 2) // 3) * alignment
                    source[0].zero_()
                    source[0, 0], source[0, 1] = 2048, 1
                    source[0, second_start] = -2048
                    hidden[:, (0, 1, second_start)] = 1
                    bias = torch.arange(outputs).half() if has_bias else None
                    if has_bias:
                        source[1].zero_()
                        source[1, 0], source[1, 1] = 2048, 1
                        bias[1] = -2048
                    layer = VisionOutputParallelLinear(
                        width,
                        outputs,
                        alignment=alignment,
                        vision_group=Group(),
                        bias=has_bias,
                        params_dtype=torch.float16,
                        return_bias=False,
                    )
                    weights = [("weight", source)]
                    if bias is not None:
                        weights.append(("bias", bias))
                    AutoWeightsLoader(layer).load_weights(weights)
                    start, count = (
                        layer.input_partition_start,
                        layer.input_partition_size,
                    )
                    actual = layer(hidden[:, start : start + count])
                    expected = torch.nn.functional.linear(hidden, source.half(), bias)
                    torch.testing.assert_close(
                        actual[:, 0],
                        torch.ones(3, dtype=torch.float16),
                        atol=0,
                        rtol=0,
                    )
                    if has_bias:
                        torch.testing.assert_close(
                            actual[:, 1],
                            torch.ones(3, dtype=torch.float16),
                            atol=0,
                            rtol=0,
                        )
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_vision_output_preserves_fp16_cancellation_bias_and_strided_inputs(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _vision_output_gloo_worker,
        args=(str(tmp_path / "gloo"),),
        nprocs=3,
        join=True,
    )


@pytest.mark.parametrize("text_tp_size", [1, 3])
@pytest.mark.parametrize("width,alignment", [(4304, 16), (4608, 1)])
def test_vision_mlp_column_shards_cover_nonlinear_dense_projection(
    monkeypatch, width, alignment, text_tp_size
):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers.vision_parallel import VisionColumnParallelLinear

    generator = torch.Generator().manual_seed(56)
    x = torch.randn(3, 5, generator=generator)
    weight = torch.randn(width, 5, generator=generator)
    bias = torch.randn(width, generator=generator)
    outputs = []
    for rank in range(3):
        monkeypatch.setattr(
            parameter,
            "get_tensor_model_parallel_rank",
            lambda rank=rank: rank if text_tp_size == 3 else 0,
        )
        monkeypatch.setattr(
            parameter, "get_tensor_model_parallel_world_size", lambda: text_tp_size
        )
        layer = VisionColumnParallelLinear(
            5,
            width,
            alignment=alignment,
            vision_group=SimpleNamespace(world_size=3, rank_in_group=rank),
            return_bias=False,
        )
        AutoWeightsLoader(layer).load_weights([("weight", weight), ("bias", bias)])
        layer.quant_method.process_weights_after_loading(layer)
        outputs.append(torch.nn.functional.gelu(layer(x)))
    torch.testing.assert_close(
        torch.cat(outputs, -1),
        torch.nn.functional.gelu(torch.nn.functional.linear(x, weight, bias)),
    )
