# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the production greedy checker (not a reimplementation)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "probe_greedy_correctness",
    ROOT / "tools" / "probe_greedy_correctness.py",
)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_pass_full_completions():
    assert mod.check_paris(
        " Paris\nParis is the capital of France and the largest city"
    ) is None
    assert mod.check_one_plus_one("2=3\n2+1=3\n") is None
    assert mod.check_germany(
        " Berlin is a city of contrasts. It is a city of history"
    ) is None


def test_fail_first_token_then_garbage():
    assert mod.check_paris(" Parisgages苦") is not None
    assert mod.check_one_plus_one("2臣ucesFromArrayductductduct") is not None
    assert mod.check_germany(" Berlinductductduct") is not None


def test_fail_missing_target():
    assert mod.check_paris(" London is a city") is not None
    assert mod.check_one_plus_one("3+3=6") is not None
    assert mod.check_germany(" Munich is a city") is not None
