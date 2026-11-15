# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape formulas and census groups of the 27B HI2 TP=2 TunableOp campaign (CPU).

``tunable_hi2.py`` imports only the standard library until a GPU subcommand runs, and
``tunableop_hi2.sh shapes`` needs no GPU, so both are exercised here against the
Qwen3.8-27B text config.
"""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools/rdna/rdna_hint_flex/tunable_hi2.py"
CAMPAIGN = REPO / "tools/rdna/rdna_hint_flex/tunableop_hi2.sh"

CFG = {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "head_dim": 256,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "attn_output_gate": True,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
    "vocab_size": 248320,
}


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("tunable_hi2", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def model(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": CFG}))
    return tmp_path


def test_rank_shapes_of_the_27b_at_tp2(tool):
    assert tool.rank_shapes(CFG, 2) == {
        "gate_up": (17408, 5120),
        "down": (5120, 8704),
        "qkvz": (8192, 5120),
        "ba": (48, 5120),
        "out": (5120, 3072),
        "qkv": (7168, 5120),
        "o": (5120, 3072),
        "lm_head": (124160, 5120),
        "fc": (2560, 10240),
        "d_qkv": (7168, 5120),
        "d_o": (5120, 3072),
        "d_gate_up": (17408, 5120),
        "d_down": (5120, 8704),
    }


def test_key_and_m_lists(tool):
    assert tool.tn_key(17408, 2048, 5120) == "tn_17408_2048_5120_ld_5120_5120_17408"
    assert tool.parse_ints("1,2,5-8") == [1, 2, 5, 6, 7, 8]
    assert tool.parse_ints("1023,1025-1027,2046") == [1023, 1025, 1026, 1027, 2046]
    assert tool.parse_ints("") == []


def test_shapes_split_hi2_and_dense_m_and_dedupe(model, capsys, tool, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "tunable_hi2.py",
            "shapes",
            str(model),
            "--tp",
            "2",
            "--slots",
            "gate_up,d_gate_up,lm_head",
            "--m",
            "1024",
            "--dense-m",
            "12,1024",
        ],
    )
    assert tool.main() == 0
    keys = [line.split(",")[1] for line in capsys.readouterr().out.splitlines()]
    assert keys == [
        "tn_17408_1024_5120_ld_5120_5120_17408",  # gate_up at the hi2 M
        "tn_17408_12_5120_ld_5120_5120_17408",  # d_gate_up: 1024 repeats gate_up's key
        "tn_124160_12_5120_ld_5120_5120_124160",
        "tn_124160_1024_5120_ld_5120_5120_124160",
    ]


def test_campaign_groups(model, tmp_path):
    work = tmp_path / "work"
    env = {
        **os.environ,
        "W": str(work),
        "MODEL": str(model),
        "VENV": str(tmp_path / "venv"),
        "GPU": "4",
    }
    result = subprocess.run(
        ["bash", str(CAMPAIGN), "shapes"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    counts = {
        name: len((work / "shapes" / f"{name}.csv").read_text().splitlines())
        for name in (
            "prefill",
            "sched0",
            "sched1",
            "sched2",
            "small",
            "lmhead_small",
            "lmhead_pl",
        )
    }
    # prefill: 7 hi2 slots + fc at 2 chunk sizes, o and out share a key (8 - 1 = 7 x 2)
    # sched*: 5 slots x 5 M; small: 5 dense slots x 6 M (d_* repeat the hi2 keys only at
    # M > 48, so none are dropped; fc is one key per M); heads: 6 / 2
    assert counts == {
        "prefill": 14,
        "sched0": 25,
        "sched1": 25,
        "sched2": 25,
        "small": 30,
        "lmhead_small": 6,
        "lmhead_pl": 2,
    }
    sched0 = (work / "shapes/sched0.csv").read_text()
    for m in (1023, 1029, 1030, 1031, 2046):
        assert f"tn_17408_{m}_5120_ld_5120_5120_17408" in sched0
    assert "tn_8192_1023_5120" in sched0 and "tn_5120_1023_3072" in sched0
    assert "tn_5120_1023_8704" not in sched0  # down gained nothing at 1024 / 2048
