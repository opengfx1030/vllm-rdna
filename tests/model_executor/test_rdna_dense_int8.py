# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU quantization contracts; no HIP operator registration or device discovery."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

_SRC = (
    Path(__file__).resolve().parents[2]
    / "vllm/model_executor/layers/rdna_dense_int8.py"
)


@pytest.fixture
def dense(monkeypatch):
    # Follow test_rdna_ar's isolated import: only the logger/operator-registration
    # dependencies are substituted. All quantization and consumer math is real.
    logger = ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    layers = ModuleType("vllm.model_executor.layers")
    layers.rdna_ops = ModuleType("rdna_ops")
    for name, module in (
        ("vllm", ModuleType("vllm")),
        ("vllm.logger", logger),
        ("vllm.model_executor", ModuleType("vllm.model_executor")),
        ("vllm.model_executor.layers", layers),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("rdna_dense_int8_cpu", _SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._ENABLED = True
    monkeypatch.setenv("VLLM_RDNA_DENSE_INT8_MIN_ROWS", "64")
    return module


def _layer(weight, global_shape):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.output_size, layer.input_size = global_shape
    return layer


def _reference(weight):
    scale = weight.abs().amax(dim=1).float().clamp_min(1e-8) / 127.0
    q = torch.round(weight.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.half()


@pytest.mark.parametrize("source_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_input_shards_preserve_full_width_quantization(dense, source_dtype):
    generator = torch.Generator().manual_seed(128)
    source = torch.randn(80, 96, generator=generator).to(source_dtype)
    source[:, 48:80] *= 8
    source[0].zero_()
    weight = source.half()
    expected_q, expected_scale = _reference(weight)
    start = 0
    for width in (48, 32, 16):
        layer = _layer(weight[:, start : start + width], weight.shape)
        dense._prepare_source_absmax(layer, source)
        q, scale = dense._quantize_weight(layer.weight, layer._rdna_i8_row_absmax)
        assert torch.equal(q, expected_q[:, start : start + width])
        assert torch.equal(scale, expected_scale)
        assert layer._rdna_i8_row_absmax.shape == (80,)
        start += width
    # This is the consumer-visible failure of quantizing each local K alone.
    local_q, _ = _reference(weight[:, :48])
    assert not torch.equal(local_q, expected_q[:, :48])


def test_small_uneven_output_shards_keep_global_eligibility(dense):
    weight = torch.randn(96, 32, generator=torch.Generator().manual_seed(96)).half()
    expected_q, expected_scale = _reference(weight)
    start = 0
    for width in (36, 30, 30):
        layer = _layer(weight[start : start + width], weight.shape)
        assert dense._eligible(layer)
        q, scale = dense._quantize_weight(layer.weight)
        assert torch.equal(q, expected_q[start : start + width])
        assert torch.equal(scale, expected_scale[start : start + width])
        start += width
    assert not dense._eligible(_layer(weight[:30], (30, 32)))


def test_vision_local_k360_keeps_existing_fallback(dense):
    layer = _layer(torch.zeros(80, 360, dtype=torch.float16), (80, 1152))
    assert not dense._eligible(layer)
    dense._prepare_source_absmax(layer, torch.ones(80, 1152))
    assert not hasattr(layer, "_rdna_i8_row_absmax")


@pytest.mark.parametrize("enabled", [False, True])
def test_full_width_replicated_weights_need_no_source_statistics(dense, enabled):
    dense._ENABLED = enabled
    weight = torch.randn(80, 32, generator=torch.Generator().manual_seed(1)).half()
    layer = _layer(weight, weight.shape)
    dense._prepare_source_absmax(layer, weight)
    assert not hasattr(layer, "_rdna_i8_row_absmax")
    q, scale = dense._quantize_weight(weight)
    expected_q, expected_scale = _reference(weight)
    assert torch.equal(q, expected_q)
    assert torch.equal(scale, expected_scale)


def test_partial_source_is_rejected_and_reload_replaces_statistics(dense):
    source = torch.ones(80, 48, dtype=torch.float16)
    layer = _layer(source[:, :16], source.shape)
    with pytest.raises(ValueError):
        dense._prepare_source_absmax(layer, source[:, :16])
    dense._prepare_source_absmax(layer, source)
    source[:, 16:] *= 7
    dense._prepare_source_absmax(layer, source)
    q, scale = dense._quantize_weight(layer.weight, layer._rdna_i8_row_absmax)
    expected_q, expected_scale = _reference(source)
    assert torch.equal(q, expected_q[:, :16])
    assert torch.equal(scale, expected_scale)


def test_zero_underflow_and_extreme_rows_preserve_original_rounding(dense):
    weight = torch.randn(80, 32, generator=torch.Generator().manual_seed(127)).half()
    weight[0].zero_()
    weight[1].fill_(1e-6)
    weight[2, :2] = torch.tensor([-65504, 65504], dtype=torch.float16)
    q, scale = dense._quantize_weight(weight)
    expected_q, expected_scale = _reference(weight)
    assert torch.equal(q, expected_q)
    assert torch.equal(scale, expected_scale)
    assert scale.dtype == torch.float16 and scale.is_contiguous()
    assert q.dtype == torch.int8 and q.is_contiguous()
    assert scale[0] == 0 and scale[1] == 0
    assert torch.equal(q[0], torch.zeros(32, dtype=torch.int8))
    assert q[2, 0] == -127 and q[2, 1] == 127


def test_released_weight_consumer_preserves_sharded_reconstruction_and_bias(dense):
    generator = torch.Generator().manual_seed(42)
    weight = torch.randn(80, 48, generator=generator).half()
    q, scale = dense._quantize_weight(weight)
    x = torch.randn(2, 3, 48, generator=generator).half()
    bias = torch.randn(80, generator=generator).half()
    expected_weight = (q.float() * scale.float()[:, None]).half()
    expected = torch.nn.functional.linear(x, expected_weight, bias)
    actual = dense.linear_released(x, q, scale, bias, block_rows=17)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_explicit_fp16_consumer_is_not_shadowed_or_released(dense, monkeypatch):
    source = torch.randn(80, 32, generator=torch.Generator().manual_seed(7)).half()
    layer = _layer(source, source.shape)
    layer._rdna_dense_fp16 = True
    dense._ONLY = True
    storage = layer.weight.data_ptr()

    def quantization_must_not_run(*args):
        raise AssertionError("FP16 consumer reached INT8 conversion")

    # The opt-out precedes global policy/device checks; no target policy is
    # mutated and even ONLY mode must retain the original full FP16 storage.
    monkeypatch.setattr(dense, "enabled", quantization_must_not_run)
    dense.make_shadow(layer)
    assert layer.weight.data_ptr() == storage
    assert layer.weight.dtype == torch.float16 and layer.weight.shape == source.shape
    assert not hasattr(layer, "weight_i8") and not hasattr(layer, "weight_i8_scale")
    x = torch.randn(2, 32).half()
    torch.testing.assert_close(
        torch.nn.functional.linear(x, layer.weight),
        torch.nn.functional.linear(x, source),
        atol=0,
        rtol=0,
    )
