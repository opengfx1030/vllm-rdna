# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode KL compares predictions conditioned on the same input prefix."""

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "decode_kl", ROOT / "tools/rdna/port_v031/decode_kl.py"
)
assert _spec is not None and _spec.loader is not None
decode_kl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(decode_kl)


def _run(ids: list[int], probabilities: list[tuple[float, float]]) -> dict:
    return {
        "meta": {"gen_tokens": 3},
        "runs": [
            {
                "ids": ids,
                "top": [{"1": math.log(p), "2": math.log(q)} for p, q in probabilities],
            }
        ],
    }


@pytest.mark.parametrize("divergence", [0, 1])
def test_first_divergent_prediction_still_has_the_same_input(divergence: int):
    ref = _run([1, 1, 1], [(0.75, 0.25)] * 3)
    cand = _run(
        [1] * divergence + [2] * (3 - divergence),
        [(0.75, 0.25)] * divergence
        + [(0.25, 0.75)]
        + [(0.01, 0.99)] * (2 - divergence),
    )

    result = decode_kl.pair(ref, cand)

    assert result["positions"] == divergence + 1
    # Earlier predictions agree exactly; later predictions use different inputs.
    divergent_kl = 0.5 * math.log(3)
    assert result["kl_mean"] == pytest.approx(divergent_kl / (divergence + 1))
    assert result["kl_max"] == pytest.approx(divergent_kl)
    assert result["confident_flips"] == 1


def test_identical_generated_tokens_compare_every_prediction():
    ref = _run([1, 1, 1], [(0.75, 0.25)] * 3)
    cand = _run([1, 1, 1], [(0.70, 0.30)] * 3)

    result = decode_kl.pair(ref, cand)

    expected_kl = 0.75 * math.log(0.75 / 0.70) + 0.25 * math.log(0.25 / 0.30)
    assert result["positions"] == 3
    assert result["kl_mean"] == pytest.approx(expected_kl)
    assert result["identical"] == 1
    assert result["confident_flips"] == 0
