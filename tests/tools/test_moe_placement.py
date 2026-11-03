# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for tools/rdna/prof/moe_placement.py.

The tool only needs stdlib + optional numpy, so the tests run on a laptop
and on the box without a GPU.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "moe_placement", ROOT / "tools/rdna/prof/moe_placement.py"
)
assert _SPEC is not None and _SPEC.loader is not None
MP = importlib.util.module_from_spec(_SPEC)
sys.modules["moe_placement"] = MP
_SPEC.loader.exec_module(MP)

META = {
    "meta": {
        "num_experts": 8,
        "ep_size": 2,
        "top_k": 2,
        "block_size_m": 4,
        "linear_placement": True,
    }
}


def _record(seq, step, layer_idx, tokens, counts):
    return {
        "seq": seq,
        "step": step,
        "layer": f"model.layers.{layer_idx}.mlp.experts",
        "layer_idx": layer_idx,
        "tokens": tokens,
        "counts": counts,
    }


def _write_hist(path, records, meta=True):
    with open(path, "w", encoding="utf-8") as f:
        if meta:
            f.write(json.dumps(META) + "\n")
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_load_hist_tolerates_truncated_last_line(tmp_path):
    path = tmp_path / "hist.jsonl"
    _write_hist(
        path,
        [_record(0, 0, 0, 512, [1, 2, 3, 4, 5, 6, 7, 8])],
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"seq": 1, "step": 0, "layer": "model.layers.1.mlp.experts"')
    meta, by_layer = MP.load_hist(str(path), min_tokens=0)
    assert meta["num_experts"] == 8
    assert list(by_layer) == [0]
    assert by_layer[0][0]["counts"] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_load_hist_filters_min_tokens(tmp_path):
    path = tmp_path / "hist.jsonl"
    _write_hist(
        path,
        [
            _record(0, 0, 0, 256, [1] * 8),
            _record(1, 0, 0, 512, [2] * 8),
        ],
    )
    _, by_layer = MP.load_hist(str(path), min_tokens=512)
    assert len(by_layer[0]) == 1
    assert by_layer[0][0]["tokens"] == 512


def test_load_hist_rejects_wrong_count_length(tmp_path):
    path = tmp_path / "hist.jsonl"
    _write_hist(path, [_record(0, 0, 0, 512, [1, 2, 3])])
    _, by_layer = MP.load_hist(str(path), min_tokens=0)
    assert by_layer == {}


def test_load_hist_requires_meta(tmp_path):
    path = tmp_path / "hist.jsonl"
    _write_hist(path, [_record(0, 0, 0, 512, [1] * 8)], meta=False)
    with pytest.raises(ValueError, match="no meta line"):
        MP.load_hist(str(path), min_tokens=0)


def test_linear_and_lpt_are_valid_placements():
    records = [_record(0, 0, 0, 512, [i for i in range(8)])]
    T = MP.build_T(records, 8, 4)
    linear = MP.linear_placement(8, 2)
    assert linear == [[0, 1, 2, 3], [4, 5, 6, 7]]
    lpt = MP.build_lpt(T, 8, 2)
    MP.validate_placement({0: lpt}, 8, 2)
    swapped = MP.swap_improve(T, [list(x) for x in lpt], "max")
    MP.validate_placement({0: swapped}, 8, 2)


def test_validate_placement_rejects_duplicate():
    bad = [[0, 1, 2, 3], [3, 4, 5, 6]]
    with pytest.raises(AssertionError, match="appears more than once"):
        MP.validate_placement({0: bad}, 8, 2)


def test_validate_placement_rejects_uneven_capacity():
    bad = [[0, 1, 2], [3, 4, 5, 6, 7]]
    with pytest.raises(AssertionError, match="experts, expected 4"):
        MP.validate_placement({0: bad}, 8, 2)


def test_split_folds_is_contiguous_and_complete():
    steps = list(range(10))
    blocks = MP._split_folds(steps, 4)
    assert blocks == [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]
    assert [s for block in blocks for s in block] == steps


def test_analyze_end_to_end_systematic(tmp_path):
    path = tmp_path / "sys.jsonl"
    MP.generate_synthetic(
        str(path),
        num_layers=2,
        num_steps=16,
        num_experts=16,
        tokens=512,
        top_k=4,
        skew="systematic",
        seed=0,
        alpha=0.1,
    )
    report, placement = MP.analyze(
        str(path),
        ranks=2,
        bsm=4,
        min_tokens=512,
        folds=4,
        objective="max",
        mean_moe_ms=100.0,
        seed=0,
    )
    assert set(report["methods"]) == {"linear", "lpt", "lpt+swap"}
    assert report["best_method"] in report["methods"]
    for method in report["methods"]:
        held = report["methods"][method]["held_out"]
        assert "mean_fraction" in held
        assert "projected_saving_ms" in held
        assert len(report["methods"][method]["folds"]) == 4
    MP.validate_placement(placement, 16, 2)
    assert any("held-out gain" in note for note in report["notes"])


def test_analyze_end_to_end_iid(tmp_path):
    path = tmp_path / "iid.jsonl"
    MP.generate_synthetic(
        str(path),
        num_layers=4,
        num_steps=40,
        num_experts=64,
        tokens=512,
        top_k=10,
        skew="iid",
        seed=1,
        alpha=0.5,
    )
    report, placement = MP.analyze(
        str(path),
        ranks=4,
        bsm=4,
        min_tokens=512,
        folds=4,
        objective="mean",
        mean_moe_ms=100.0,
        seed=0,
    )
    MP.validate_placement(placement, 64, 4)
    linear_out = report["methods"]["linear"]["held_out"]["mean_fraction"]
    best_out = report["methods"][report["best_method"]]["held_out"]["mean_fraction"]
    assert linear_out - best_out <= 0.02


def test_run_selftest():
    MP.run_selftest()
