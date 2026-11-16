# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The vllm-gguf-plugin seam reports a missing or mismatched plugin clearly."""

from __future__ import annotations

import types

import pytest
import torch

from vllm.model_executor.model_loader import gguf_plugin_compat as compat


def _fake_import(modules: dict[str, object]):
    def fake(name: str):
        entry = modules[name]
        if isinstance(entry, Exception):
            raise entry
        return entry

    return fake


def test_missing_plugin_says_to_install_it(monkeypatch):
    err = ModuleNotFoundError("No module named 'vllm_gguf_plugin'")
    err.name = "vllm_gguf_plugin"
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        _fake_import({compat._MOE_MODULE: err}),
    )
    with pytest.raises(ImportError, match="needs the out-of-tree vllm-gguf-plugin"):
        compat.get_gguf_moe_method_cls()


def test_plugin_that_does_not_import_against_this_vllm_says_so(monkeypatch):
    # plugin 0.0.5 on vLLM >= v0.32: the plugin imports a vLLM module that is gone
    err = ModuleNotFoundError(
        "No module named 'vllm.transformers_utils.configs.qwen3_5'"
    )
    err.name = "vllm.transformers_utils.configs.qwen3_5"
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        _fake_import({compat._MOE_MODULE: err}),
    )
    with pytest.raises(ImportError) as exc:
        compat.get_gguf_moe_method_cls()
    message = str(exc.value)
    assert "is installed but" in message
    assert "does not match this vLLM" in message
    assert "vllm.transformers_utils.configs.qwen3_5" in message
    assert "needs the out-of-tree" not in message


def test_missing_symbol_is_named(monkeypatch):
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        _fake_import({compat._MOE_MODULE: types.SimpleNamespace()}),
    )
    with pytest.raises(ImportError, match="no 'GGUFMoEMethod'"):
        compat.get_gguf_moe_method_cls()


def test_symbols_are_returned_and_called_through(monkeypatch):
    class FakeMoEMethod:
        pass

    calls: list[tuple[object, torch.Tensor, object]] = []

    def fake_store(param, loaded_weight, shard_id=None):
        calls.append((param, loaded_weight, shard_id))

    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        _fake_import(
            {
                compat._MOE_MODULE: types.SimpleNamespace(GGUFMoEMethod=FakeMoEMethod),
                compat._PARAMS_MODULE: types.SimpleNamespace(
                    _store_gguf_weight_type=fake_store
                ),
            }
        ),
    )
    assert compat.get_gguf_moe_method_cls() is FakeMoEMethod
    param = torch.nn.Parameter(torch.zeros(1))
    value = torch.tensor(12)
    compat.store_gguf_weight_type(param, value)
    compat.store_gguf_weight_type(param, value, 3)
    assert calls == [(param, value, None), (param, value, 3)]
