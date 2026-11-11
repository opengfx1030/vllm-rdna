# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for refusing unsegmented host-cache planning in PW graphs."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.parametrize(
    "piecewise,segmented", [(True, False), (True, True), (False, False)]
)
def test_cache_piecewise_requires_segmented_backend(piecewise, segmented):
    source = Path(__file__).resolve().parents[3] / (
        "vllm/models/qwen4_exp/amd/mono_decode_rdna2.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        node for node in tree.body if getattr(node, "name", "") == "check_config"
    )

    class RemoveImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None

    module = ast.Module(body=[RemoveImports().visit(function)], type_ignores=[])
    opts = SimpleNamespace(
        expert_cache_slots=32, k2=True, modules=False, mtp_drafter=False, k1=True
    )
    namespace: dict[str, Any] = {
        "enabled": lambda config: True,
        "_on_gfx10x": lambda: True,
        "is_qwen4exp": lambda config: True,
        "options": lambda config: opts,
        "config_refusal": lambda config: None,
        "is_breakable_cudagraph_enabled": lambda: segmented,
        "os": SimpleNamespace(environ={"VLLM_RAISE_ON_LOGIT_NANS": "1"}),
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    config = SimpleNamespace(
        model_config=object(),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_piecewise_cudagraphs=lambda: piecewise),
            splitting_ops=[],
        ),
    )
    if piecewise and not segmented:
        with pytest.raises(ValueError, match="VLLM_USE_BREAKABLE_CUDAGRAPH=1"):
            namespace["check_config"](config)
    else:
        namespace["check_config"](config)
