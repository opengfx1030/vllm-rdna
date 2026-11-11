# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PLE pipeline-parallel placement check (no GPU, no vLLM runtime needed)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm"
    / "models"
    / "qwen4_exp"
    / "common"
    / "ple_pp.py"
)
_spec = importlib.util.spec_from_file_location("_qwen4_exp_ple_pp", _PATH)
ple_pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ple_pp)


@pytest.mark.parametrize(
    ("ple_layer_ids", "first_stage_end", "expected"),
    [
        ([1], 1, []),  # decoder layer 0 on a 1-layer stage 0
        ([3], 17, []),  # Flash-Next style: layer 2 on stage 0
        ([2], 1, [2]),  # layer 1 lands on stage 1
        ([1, 3, 3, 20], 17, [20]),  # dedup, only the off-stage id
        ([], 4, []),
    ],
)
def test_ple_layers_outside_first_stage(ple_layer_ids, first_stage_end, expected):
    assert (
        ple_pp.ple_layers_outside_first_stage(ple_layer_ids, first_stage_end)
        == expected
    )


def test_no_ple_or_pp1_is_always_supported():
    with_ple = SimpleNamespace(ple_layer_ids=[1], num_hidden_layers=2)
    without_ple = SimpleNamespace(ple_layer_ids=[], num_hidden_layers=2)
    assert ple_pp.ple_pp_unsupported_reason(with_ple, 1, allow_pp=False) is None
    assert ple_pp.ple_pp_unsupported_reason(without_ple, 3, allow_pp=False) is None
    assert ple_pp.rdna_ple_pp_unsupported_reason(with_ple, 1) is None
    assert ple_pp.rdna_ple_pp_unsupported_reason(without_ple, 3) is None


def test_pp_rejected_when_not_allowed():
    cfg = SimpleNamespace(ple_layer_ids=[1], num_hidden_layers=2)
    reason = ple_pp.ple_pp_unsupported_reason(cfg, 2, allow_pp=False)
    assert reason == ple_pp.PLE_PP_UNSUPPORTED
    assert "pipeline_parallel_size=1" in reason
