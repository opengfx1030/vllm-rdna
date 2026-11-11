# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Invariants of tools/rdna/rdna3/perop_expectations.py (stdlib only)."""

import importlib.util
from pathlib import Path

import pytest

TOOL = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "rdna"
    / "rdna3"
    / "perop_expectations.py"
)


@pytest.fixture(scope="module")
def pe():
    spec = importlib.util.spec_from_file_location("perop_expectations", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_distinct_experts_match_the_design_doc(pe):
    # design doc 2.2 lists 28.8 / 174 / 282 distinct experts per layer at W = 3 / 24
    # / 48 (the closed form gives 29.4 at W = 3; the doc rounds its own fit)
    assert pe.distinct_experts(3) == pytest.approx(28.8, abs=1)
    assert pe.distinct_experts(24) == pytest.approx(174, abs=1)
    assert pe.distinct_experts(48) == pytest.approx(282, abs=1)


@pytest.mark.parametrize("scenario", ["pessimistic", "central", "optimistic"])
def test_every_row_is_finite_and_never_shrinks_with_width(pe, scenario):
    table = pe.table(scenario)
    for name, row in table["rows"].items():
        us = [row["us"][w] for w in pe.WIDTHS]
        assert all(v > 0 for v in us), name
        assert us == sorted(us), f"{name}: {us}"


def test_scenarios_are_ordered(pe):
    totals = {
        s: pe.table(s)["target_forward_ms_without_ar_and_gaps"][3]
        for s in ("pessimistic", "central", "optimistic")
    }
    assert totals["optimistic"] < totals["central"] < totals["pessimistic"]


def test_expert_streaming_rate_is_the_design_doc_constant(pe):
    # 2.53 MB over 0.8 x 864 GB/s = 3.67 us per expert (design doc 5.3)
    per = pe.EXPERT_BYTES / (0.8 * pe.PEAK_BW) * 1e6
    assert per == pytest.approx(3.67, abs=0.01)


def test_spec_gate_of_five_us_per_expert_is_slower_than_central(pe):
    row = pe.table("central")["rows"]["routed W4A16 experts (per layer, one EP2 rank)"]
    for w in pe.WIDTHS:
        assert row["us_at_5us_per_expert"][w] > row["us"][w]


def test_markdown_lists_every_row(pe):
    table = pe.table("central")
    text = pe.markdown(table)
    for name in table["rows"]:
        assert name in text
