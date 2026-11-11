# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only execution of the real dispatcher without ROCm import side effects."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm/model_executor/kernels/linear/mixed_precision/rdna2_w4a16.py"
)


def _dispatcher(enabled):
    wanted = {"_w4a8_lds_fits", "_rdna2_w4a16_select_kernel", "_rdna2_w4a16_gemm"}
    nodes: list[ast.stmt] = [
        node
        for node in ast.parse(_SOURCE.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    calls = []

    def operation(name):
        def run(*args):
            calls.append((name, args))
            return args[0]

        return run

    env = {
        "torch": torch,
        "W4A8_MIN_ROWS": 33,
        "_DETERMINISTIC_DENSE": enabled,
        "ops": SimpleNamespace(
            **{
                name: operation(name)
                for name in (
                    "gptq_gemm_rdna2",
                    "gptq_gemm_rdna2_prefill",
                    "gptq_gemm",
                    "w4a8_gemm_rdna2",
                )
            }
        ),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_SOURCE), "exec"), env)
    return env, calls


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("is_awq", [False, True])
@pytest.mark.parametrize("M", [1, 128, 4096])
def test_dense_deterministic_dispatch_covers_symmetric_large_prefill(
    enabled, is_awq, M
):
    env, calls = _dispatcher(enabled)
    x = torch.empty(M, 2560, device="meta")
    empty = torch.empty(0, device="meta")
    env["_rdna2_w4a16_gemm"](x, empty, empty, empty, empty, 7168, is_awq, 4)
    selected = env["_rdna2_w4a16_select_kernel"](M, 2560, 7168, is_awq=is_awq)
    expected = (
        "gptq_gemm_rdna2_prefill" if enabled or selected == "prefill" else "gptq_gemm"
    )
    assert len(calls) == 1 and calls[0][0] == expected
    if expected == "gptq_gemm_rdna2_prefill":
        assert calls[0][1][-1] is enabled


@pytest.mark.parametrize("M", [1, 128])
def test_dense_deterministic_dispatch_refuses_w4a8(M):
    env, calls = _dispatcher(True)
    x = torch.empty(M, 2560, device="meta")
    empty = torch.empty(0, device="meta")
    with pytest.raises(ValueError, match="W4A8 disabled"):
        env["_rdna2_w4a16_gemm"](x, empty, empty, empty, empty, 7168, False, 4, True)
    assert not calls
