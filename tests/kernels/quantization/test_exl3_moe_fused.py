# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-shape EXL3 MoE apply matches the per-expert reference loop."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA EXL3 kernels are ROCm-only", allow_module_level=True)

from vllm.model_executor.layers.quantization.exl3 import (  # noqa: E402
    _rdna_exl3_available,
)
from vllm.model_executor.layers.quantization.exl3_moe import (  # noqa: E402
    Exl3MoEMethod,
)

pytestmark = pytest.mark.skipif(
    not _rdna_exl3_available(), reason="EXL3 RDNA ops not built"
)


def _layer(num_experts, hidden, inter, bits, expert_map=None, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def trellis(*shape):
        return torch.randint(
            -32768, 32767, shape, dtype=torch.int16, device="cuda", generator=g
        )

    def scale(*shape):
        signs = torch.randint(0, 2, shape, device="cuda", generator=g) * 2 - 1
        mags = torch.rand(shape, device="cuda", generator=g) * 0.02 + 0.01
        return (signs * mags).half()

    w = 16 * bits
    return SimpleNamespace(
        w13_trellis=trellis(num_experts, 2, hidden // 16, inter // 16, w),
        w13_suh=scale(num_experts, 2, hidden),
        w13_svh=scale(num_experts, 2, inter),
        w2_trellis=trellis(num_experts, inter // 16, hidden // 16, w),
        w2_suh=scale(num_experts, inter),
        w2_svh=scale(num_experts, hidden),
        expert_map=expert_map,
        activation="silu",
    )


def _method(bits):
    m = object.__new__(Exl3MoEMethod)
    m.hadamard = "both"
    m.cb = 2
    from vllm.model_executor.layers.quantization.exl3 import Exl3LinearMethod

    m._linear = Exl3LinearMethod(bits=None, hadamard="both")
    m.w13_bits = bits
    m.w2_bits = bits
    return m


@pytest.mark.parametrize("tokens", [1, 3, 8, 37])
@pytest.mark.parametrize("use_map", [False, True])
def test_fused_apply_matches_reference(tokens, use_map):
    torch.manual_seed(tokens)
    global_experts, top_k, hidden, inter, bits = 16, 4, 256, 128, 3
    expert_map = None
    local_experts = global_experts
    if use_map:
        # Expert parallel: this rank owns the odd experts.
        local_experts = global_experts // 2
        expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
        expert_map[1::2] = torch.arange(local_experts, dtype=torch.int32, device="cuda")
    layer = _layer(local_experts, hidden, inter, bits, expert_map)
    method = _method(bits)

    x = (torch.randn(tokens, hidden, device="cuda") * 0.5).half()
    topk_ids = torch.stack(
        [torch.randperm(global_experts, device="cuda")[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    topk_weights = torch.rand(tokens, top_k, device="cuda").float()

    want = method.apply_reference(layer, x, topk_weights, topk_ids)
    method.process_weights_after_loading(layer_with_params := _as_module(layer))
    got = method.apply(layer_with_params, x, topk_weights, topk_ids)

    assert got.shape == want.shape == (tokens, hidden)
    torch.testing.assert_close(got.float(), want.float(), atol=2e-3, rtol=2e-2)


def _as_module(ns):
    mod = torch.nn.Module()
    for name, value in vars(ns).items():
        if isinstance(value, torch.Tensor) and name.startswith("w"):
            mod.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
        else:
            setattr(mod, name, value)
    return mod


@pytest.mark.parametrize("tokens", [1, 8])
def test_fused_apply_is_cudagraph_capturable(tokens):
    torch.manual_seed(100 + tokens)
    global_experts, top_k, hidden, inter, bits = 16, 4, 256, 128, 3
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
    expert_map[::2] = torch.arange(
        global_experts // 2, dtype=torch.int32, device="cuda"
    )
    layer = _as_module(_layer(global_experts // 2, hidden, inter, bits, expert_map))
    method = _method(bits)
    method.process_weights_after_loading(layer)

    x = (torch.randn(tokens, hidden, device="cuda") * 0.5).half()
    topk_ids = torch.stack(
        [torch.randperm(global_experts, device="cuda")[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    topk_weights = torch.rand(tokens, top_k, device="cuda").float()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        method.apply(layer, x, topk_weights, topk_ids)  # warm up
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = method.apply(layer, x, topk_weights, topk_ids)

    # New routing and inputs in the same buffers, then replay.
    x.copy_((torch.randn(tokens, hidden, device="cuda") * 0.5).half())
    topk_ids.copy_(
        torch.stack(
            [
                torch.randperm(global_experts, device="cuda")[:top_k]
                for _ in range(tokens)
            ]
        ).to(torch.int32)
    )
    graph.replay()
    want = method.apply(layer, x, topk_weights, topk_ids)
    torch.testing.assert_close(out.float(), want.float(), atol=2e-3, rtol=2e-2)
