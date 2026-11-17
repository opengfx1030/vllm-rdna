# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Qwen4 config probe (``qwen4_prep.config_probe``).

No torch, no GPU and no vLLM import: the probe is loaded from its file so the
tests run on any box. The anchor cases are the published Flash-Next config
(supported) and a dense 27B-shaped config (blocked, because the decoder has no
dense MLP).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "vllm/models/qwen4_prep/config_probe.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load("qwen4_config_probe", PROBE_PATH)


def _flash_next() -> dict:
    """The published Qwen3.8-Flash-Next geometry, trimmed to what matters."""
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "dtype": "bfloat16",
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "full_attention_interval": 4,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "hc_count": 4,
            "hc_lowrank": 320,
            "ple_layer_ids": [2],
            "ngram_size": 3,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "mtp_num_hidden_layers": 1,
            "linear_num_value_heads": 48,
        },
    }


def _dense_27b() -> dict:
    """Qwen4-27B as described by the leaked Intel kernels: dense, 4 streams."""
    return {
        "model_type": "qwen4",
        "text_config": {
            "model_type": "qwen4_text",
            "dtype": "bfloat16",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "full_attention_interval": 4,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
            "hc_count": 4,
            "hc_lowrank": 320,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "mtp_num_hidden_layers": 1,
        },
    }


def test_flash_next_is_supported():
    report = probe.probe_config(_flash_next())
    assert report.supported
    assert report.blockers == ()
    assert report.model_type == "qwen4_exp"
    assert report.text_model_type == "qwen4_exp_text"
    assert report.geometry["num_experts"] == 512
    assert report.geometry["mtp_depth"] == 1
    assert report.kv_bytes_per_token == 12 * 2 * 256 * 2 * 2


def test_flash_next_warns_about_bf16_on_gfx1030():
    report = probe.probe_config(_flash_next())
    assert any("bfloat16" in warning for warning in report.warnings)


def test_dense_qwen4_is_blocked():
    report = probe.probe_config(_dense_27b())
    assert not report.supported
    assert any("dense MLP" in blocker for blocker in report.blockers)
    assert any("no model class" in blocker for blocker in report.blockers)
    assert report.kv_bytes_per_token == 16 * 4 * 256 * 2 * 2


def test_dense_config_with_known_model_type_only_reports_dense():
    config = _dense_27b()
    config["model_type"] = "qwen4_exp"
    config["text_config"]["model_type"] = "qwen4_exp_text"
    report = probe.probe_config(config)
    assert not report.supported
    assert len(report.blockers) == 1
    assert "dense MLP" in report.blockers[0]


def test_unknown_layer_type_is_blocked():
    config = _flash_next()
    config["text_config"]["layer_types"] = ["linear_attention", "mystery_attention"]
    report = probe.probe_config(config)
    assert any("unknown layer_types" in blocker for blocker in report.blockers)


def test_qwen_sparse_attention_alias_is_accepted():
    config = _flash_next()
    config["text_config"]["layer_types"] = ["qwen_sparse_attention"] * 4
    report = probe.probe_config(config)
    assert not any("unknown layer_types" in blocker for blocker in report.blockers)


def test_indexer_with_two_kv_heads_is_blocked():
    config = _flash_next()
    config["text_config"]["indexer_kv_heads"] = 2
    report = probe.probe_config(config)
    assert any("indexer_kv_heads" in blocker for blocker in report.blockers)


def test_missing_indexer_warns_about_dense_fallback():
    config = _flash_next()
    for key in ("indexer_n_heads", "indexer_kv_heads", "indexer_head_dim"):
        del config["text_config"][key]
    report = probe.probe_config(config)
    assert report.supported
    assert any("QSA is disabled" in warning for warning in report.warnings)


def test_fp8_quantization_is_blocked():
    config = _flash_next()
    config["quantization_config"] = {"quant_method": "fp8"}
    report = probe.probe_config(config)
    assert any("cannot run on gfx1030" in blocker for blocker in report.blockers)


def test_non_qwen4_config_raises():
    with pytest.raises(ValueError, match="not a Qwen4-family config"):
        probe.probe_config({"model_type": "qwen3_5"})


def test_load_config_accepts_directory(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_flash_next()))
    assert probe.load_config(tmp_path)["model_type"] == "qwen4_exp"


def test_cli_exit_codes_and_json(tmp_path, capsys):
    supported = tmp_path / "supported.json"
    supported.write_text(json.dumps(_flash_next()))
    assert probe.main([str(supported)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["supported"] is True

    blocked = tmp_path / "blocked.json"
    blocked.write_text(json.dumps(_dense_27b()))
    assert probe.main([str(blocked)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["supported"] is False
