# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-count-agnostic validation of the expert-cache control RPC."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
# qualify_expert_copy imports helpers via the absolute tools.* path; provide
# the namespace without pulling vllm's package machinery into a CPU test.
if "tools.rdna.mono.qualify_offload_long" not in sys.modules:
    for name in ("tools", "tools.rdna", "tools.rdna.mono"):
        module = sys.modules.setdefault(name, types.ModuleType(name))
        module.__path__ = [str(_REPO / name.replace(".", "/"))]
    stub = types.ModuleType("tools.rdna.mono.qualify_offload_long")
    stub.bench = lambda *a, **k: None
    stub.post = lambda *a, **k: None
    sys.modules["tools.rdna.mono.qualify_offload_long"] = stub

_PATH = _REPO / "tools/rdna/mono/qualify_expert_copy.py"
_SPEC = importlib.util.spec_from_file_location("qualify_expert_copy_tp4", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
qualify = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = qualify
_SPEC.loader.exec_module(qualify)


def _ranks(n, caches=48, modes=("tensor",)):
    return [
        dict(rank=r, caches=caches, copy_modes=list(modes)) for r in range(n)
    ]


def test_tp2_two_ranks_still_accepted():
    result = _ranks(2)
    assert qualify.check_control_result(result) == result


def test_tp4_four_ranks_accepted():
    result = _ranks(4)
    assert qualify.check_control_result(result) == result


def test_mixed_cache_counts_refused():
    result = _ranks(4)
    result[2]["caches"] = 47
    with pytest.raises(ValueError, match="expert caches"):
        qualify.check_control_result(result)


def test_empty_reply_refused():
    with pytest.raises(ValueError, match="no ranks"):
        qualify.check_control_result([])


def test_mode_switch_checked_per_rank():
    result = _ranks(4, modes=("native",))
    assert qualify.check_control_result(result, mode="native") == result
    result[3]["copy_modes"] = ["tensor"]
    with pytest.raises(ValueError, match="copy backend"):
        qualify.check_control_result(result, mode="native")
