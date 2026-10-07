# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

from ...utils import create_new_process_for_each_test


def _ple_grouped_norm_reference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int | None,
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    if group_size is None:
        grouped = hidden_states.unsqueeze(-2)
    else:
        grouped = hidden_states.unflatten(
            -1, (hidden_states.shape[-1] // group_size, group_size)
        )
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = grouped * torch.rsqrt(variance + eps)
    return (normalized.flatten(-2) * (1.0 + weight.float())).to(input_dtype)


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("group_size", [None, 8])
def test_amd_ple_grouped_norm_cpu_fallback_matches_reference(
    group_size: int | None,
) -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLEGroupedNorm

    hidden_size = 32
    eps = 1e-6
    norm = Qwen4ExpPLEGroupedNorm(hidden_size, eps, group_size, dtype=torch.float16)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.25, 0.25, hidden_size))
    hidden_states = torch.randn(3, hidden_size, dtype=torch.float16)

    actual = norm(hidden_states)
    expected = _ple_grouped_norm_reference(hidden_states, norm.weight, eps, group_size)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="AMD PLE fused norm requires a ROCm GPU",
)
@pytest.mark.parametrize(
    ("dtype", "group_size"),
    [
        (torch.float16, None),
        (torch.float16, 2560),
        (torch.bfloat16, None),
        (torch.bfloat16, 2560),
    ],
)
def test_amd_ple_grouped_norm_fused_matches_reference(
    dtype: torch.dtype, group_size: int | None
) -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLEGroupedNorm

    hidden_size = 10240
    eps = 1e-6
    norm = Qwen4ExpPLEGroupedNorm(hidden_size, eps, group_size, dtype=dtype).cuda()
    weight = torch.linspace(-0.25, 0.25, hidden_size, device="cuda")
    weight[:3] = torch.tensor([-1.0, -0.999, -1.001], device="cuda")
    with torch.no_grad():
        norm.weight.copy_(weight)
    hidden_states = torch.randn(1, 3, hidden_size, dtype=dtype, device="cuda")

    actual = norm(hidden_states)
    expected = _ple_grouped_norm_reference(hidden_states, norm.weight, eps, group_size)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="AMD PLE fused norm requires a ROCm GPU",
)
def test_amd_ple_grouped_norm_strided_input_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.models.qwen4_exp.amd.ple_layer as amd_ple_layer_module
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLEGroupedNorm

    hidden_size = 32
    eps = 1e-6
    norm = Qwen4ExpPLEGroupedNorm(hidden_size, eps, 8, dtype=torch.float16).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.25, 0.25, hidden_size, device="cuda"))
    hidden_states = torch.randn(3, hidden_size * 2, dtype=torch.float16, device="cuda")[
        :, ::2
    ]
    assert hidden_states.stride(-1) != 1

    def fail_if_called(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("fused grouped norm called for strided input")

    monkeypatch.setattr(amd_ple_layer_module, "grouped_gemma_rmsnorm", fail_if_called)
    actual = norm(hidden_states)
    expected = _ple_grouped_norm_reference(
        hidden_states, norm.weight, eps, norm.group_size
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="AMD PLE fused norm requires a ROCm GPU",
)
def test_amd_ple_grouped_norm_graph_replay() -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLEGroupedNorm

    hidden_size = 128
    eps = 1e-6
    group_size = 32
    norm = Qwen4ExpPLEGroupedNorm(
        hidden_size, eps, group_size, dtype=torch.float16
    ).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.25, 0.25, hidden_size, device="cuda"))
    hidden_states = torch.randn(3, hidden_size, dtype=torch.float16, device="cuda")
    norm(hidden_states)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = norm(hidden_states)

    hidden_states.copy_(torch.randn_like(hidden_states))
    graph.replay()
    torch.accelerator.synchronize()
    expected = _ple_grouped_norm_reference(hidden_states, norm.weight, eps, group_size)
    torch.testing.assert_close(graph_output, expected, atol=2e-3, rtol=2e-3)
