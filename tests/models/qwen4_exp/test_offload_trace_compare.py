# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[3] / "tools/rdna/mono/compare_offload_trace.py"
_SPEC = importlib.util.spec_from_file_location("trace_compare", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def receipt():
    return {
        "version": 1,
        "rank": 0,
        "layer": 2,
        "call": 31,
        "rows": torch.tensor([0]),
        "metadata": {"attention": {"seq_lens": torch.tensor([4125, 0, 0])}},
        "tensors": {
            name: torch.tensor([[1.0, 2.0]])
            for name in ("residual", "attention", "injection", "output")
        },
    }


def test_matching_inputs_with_changed_output_is_detected():
    left = receipt()
    right = copy.deepcopy(left)
    right["tensors"]["output"][0, 1] += 0.125
    report = _MODULE.compare_records(left, right)
    assert report["inputs_identical"]
    assert report["tensors"]["output"]["max_abs"] == 0.125
    assert not report["tensors"]["output"]["within_atol"]


def test_equal_bucket_with_different_sequence_position_is_refused():
    left = receipt()
    right = copy.deepcopy(left)
    right["metadata"]["attention"]["seq_lens"][0] += 1
    with pytest.raises(ValueError, match="alignment differs"):
        _MODULE.compare_records(left, right)


def test_missing_alignment_is_refused():
    left = receipt()
    left["metadata"] = {}
    with pytest.raises(ValueError, match="no sequence/token identifiers"):
        _MODULE.compare_records(left, left)


def test_directory_requires_identical_generation_proof(tmp_path):
    proof = {
        "gpu_cached": {"tokens": [1], "text": "a"},
        "gpu_cached_repeat": {"tokens": [2], "text": "a"},
    }
    with pytest.raises(ValueError, match="different generated tokens"):
        _MODULE.compare_directory(tmp_path, 31, 62, proof)


def test_padding_does_not_define_alignment():
    left = receipt()
    right = copy.deepcopy(left)
    right["metadata"]["attention"]["seq_lens"][1:] = 17
    assert _MODULE.compare_records(left, right)["inputs_identical"]


def test_nested_decode_sequence_metadata_is_used():
    left = receipt()
    left["metadata"]["attention"] = {"decode": {"seq_lens": torch.tensor([4125])}}
    assert _MODULE.compare_records(left, left)["inputs_identical"]
