# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``loader.expert_formats``: which layers carry NVFP4 and which MXFP8 experts.

The format is read from the ``parameter_groups`` of ``hf_quant_config.json`` (same
structure as the real checkpoint's: float32 / bfloat16 groups for the dense tensors, an
``nvfp4`` group and an ``mxfp8`` group listing ``layers.N.mlp.experts.*`` tensors),
never from a layer-index constant.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "_i2l_loader_under_test", _REPO / "vllm/models/inkling2_loop/amd/loader.py"
)
assert _spec is not None and _spec.loader is not None
loader = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = loader
_spec.loader.exec_module(loader)

_TAILS = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")


def _experts(layers) -> list[str]:
    return [f"model.llm.layers.{i}.mlp.experts.{t}" for i in layers for t in _TAILS]


def _quant_config(nvfp4, mxfp8) -> dict:
    return {
        "model": {
            "parameter_groups": [
                {"compute_dtype": "float32", "tensors": ["model.llm.norm.weight"]},
                {
                    "compute_dtype": "bfloat16",
                    "tensors": ["model.llm.layers.0.attn.wq_du.weight"],
                },
                {"compute_dtype": "nvfp4", "tensors": _experts(nvfp4)},
                {"compute_dtype": "mxfp8", "tensors": _experts(mxfp8)},
                {"compute_dtype": "bfloat16", "tensors": ["model.llm.unembed.weight"]},
            ]
        },
        "quant_method": "mixed",
    }


def test_real_checkpoint_layout():
    fmt = loader.expert_formats(_quant_config(range(20), range(20, 24)), 24)
    assert fmt == ["nvfp4"] * 20 + ["mxfp8"] * 4


def test_tiny_checkpoint_layout():
    # tiny1: MXFP8 on the last layer only (not on a layer-index threshold of the
    # real one)
    fmt = loader.expert_formats(_quant_config(range(5), [5]), 6)
    assert fmt == ["nvfp4"] * 5 + ["mxfp8"]


def test_formats_follow_the_groups_not_the_layer_index():
    even, odd = range(0, 8, 2), range(1, 8, 2)
    fmt = loader.expert_formats(_quant_config(even, odd), 8)
    assert fmt == ["nvfp4", "mxfp8"] * 4


def test_dense_groups_do_not_define_a_format():
    cfg = _quant_config(range(2), [])
    cfg["model"]["parameter_groups"][1]["tensors"] += _experts([2])  # bfloat16 group
    with pytest.raises(ValueError, match=r"\[2\]"):
        loader.expert_formats(cfg, 3)


def test_layer_without_expert_group_is_an_error():
    with pytest.raises(ValueError, match=r"\[3\]"):
        loader.expert_formats(_quant_config(range(3), []), 4)


def test_read_next_to_a_local_model_dir(tmp_path):
    pytest.importorskip("vllm.transformers_utils.repo_utils")
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(_quant_config(range(5), [5]))
    )
    assert loader.read_expert_formats(str(tmp_path), None, 6) == ["nvfp4"] * 5 + [
        "mxfp8"
    ]
