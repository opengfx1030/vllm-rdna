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


def test_unrouted_expert_slots_are_made_harmless(monkeypatch):
    # topk_ids == -1 is "no expert" in vLLM; the plugin kernels would read
    # weights[-1] unchecked (GPU memory fault on gfx1030 in graph warmup).
    seen: dict[str, torch.Tensor] = {}

    class FakeMoEMethod:
        def apply(self, layer, x, topk_weights, topk_ids, shared, shared_in):
            seen["w"], seen["ids"] = topk_weights, topk_ids
            return x

    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        _fake_import(
            {compat._MOE_MODULE: types.SimpleNamespace(GGUFMoEMethod=FakeMoEMethod)}
        ),
    )
    method = compat.get_loader_moe_method_cls()()
    assert isinstance(method, FakeMoEMethod)
    ids = torch.tensor([[3, -1], [-1, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.25], [0.125, 0.0625]])
    x = torch.zeros(2, 4)
    assert method.apply(None, x, weights, ids, None, None) is x
    assert seen["ids"].dtype == torch.int32
    assert seen["ids"].tolist() == [[3, 0], [0, 0]]
    assert seen["w"].tolist() == [[0.5, 0.0], [0.0, 0.0]]
    assert ids.tolist() == [[3, -1], [-1, -1]]  # caller's tensors are untouched


def _chunk_probe(monkeypatch):
    calls: list[int] = []

    class FakeMoEMethod:
        def apply(self, layer, x, topk_weights, topk_ids, shared, shared_in):
            calls.append(x.shape[0])
            return x + 1

    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        _fake_import(
            {compat._MOE_MODULE: types.SimpleNamespace(GGUFMoEMethod=FakeMoEMethod)}
        ),
    )
    return compat.get_loader_moe_method_cls()(), calls


def _run(method, rows, experts):
    x = torch.arange(rows, dtype=torch.float32).view(rows, 1)
    ids = torch.zeros(rows, 2, dtype=torch.int32)
    out = method.apply(
        types.SimpleNamespace(global_num_experts=experts),
        x,
        torch.ones(rows, 2),
        ids,
        None,
        None,
    )
    assert torch.equal(out, x + 1)  # slices come back in order


def test_big_batches_of_a_many_expert_model_stay_on_the_vector_kernel(monkeypatch):
    # the plugin's MMQ kernel drops experts 256+ and handles > 64 tokens only
    method, calls = _chunk_probe(monkeypatch)
    _run(method, 150, 512)
    assert calls == [64, 64, 22]


def test_small_batches_and_small_expert_counts_are_passed_through(monkeypatch):
    method, calls = _chunk_probe(monkeypatch)
    _run(method, 64, 512)
    _run(method, 150, 256)
    assert calls == [64, 150]


def test_a_fixed_plugin_can_lift_the_expert_limit(monkeypatch):
    monkeypatch.setenv(compat._MMQ_MAX_EXPERTS_ENV, "512")
    method, calls = _chunk_probe(monkeypatch)
    _run(method, 150, 512)
    assert calls == [150]
