# SPDX-License-Identifier: Apache-2.0
import importlib.util
import pathlib
import struct

import pytest
import torch
from gguf import quants

ROOT = pathlib.Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "gguf_packed", ROOT / "vllm/model_executor/model_loader/gguf_packed.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

FORMATS = [
    ("Q5_1", 32, 24),
    ("Q4_K", 256, 144),
    ("Q5_K", 256, 176),
    ("Q8_0", 32, 34),
    ("IQ4_NL", 32, 18),
]


@pytest.mark.parametrize("name,width,size", FORMATS)
def test_decode_matches_official_oracle(name, width, size):
    raw = torch.arange(size * 2, dtype=torch.uint8).reshape(2, size)
    got = mod.dequantize(raw, name)
    expected = getattr(quants, name).dequantize_blocks(raw.numpy()).reshape(-1)
    assert got.dtype == torch.float32
    assert got.shape == (2 * width,)
    torch.testing.assert_close(got, torch.from_numpy(expected), rtol=0, atol=0)


@pytest.mark.parametrize("name,width,size", FORMATS)
def test_varied_finite_scales_and_high_bits_match_oracle(name, width, size):
    raw = (
        ((torch.arange(32 * size, dtype=torch.int64) * 73 + 19) % 256)
        .to(torch.uint8)
        .reshape(32, size)
    )
    scales = [0.0, -0.0, 1.0, -2.0, 2**-24, -(2**-14), 0.125, -0.03125]
    for i in range(32):
        raw[i, :2] = torch.tensor(
            list(struct.pack("<e", scales[i % len(scales)])), dtype=torch.uint8
        )
        if name in ("Q4_K", "Q5_K", "Q5_1"):
            raw[i, 2:4] = torch.tensor(
                list(struct.pack("<e", scales[(i + 3) % len(scales)])),
                dtype=torch.uint8,
            )
    got = mod.dequantize(raw, name, budget_bytes=32 * width * 4)
    expected = getattr(quants, name).dequantize_blocks(raw.numpy()).reshape(-1)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, torch.from_numpy(expected.copy()), rtol=0, atol=0)


@pytest.mark.parametrize("name,width,size", FORMATS)
def test_selected_rows_obey_budget(name, width, size):
    raw = torch.arange(size * 3, dtype=torch.uint8).reshape(3, size)
    got = mod.dequantize(raw, name, rows=[2, 0], budget_bytes=2 * width * 4)
    expected = getattr(quants, name).dequantize_blocks(raw[[2, 0]].numpy()).reshape(-1)
    torch.testing.assert_close(got, torch.from_numpy(expected), rtol=0, atol=0)
    with pytest.raises(ValueError):
        mod.dequantize(raw, name, rows=[2, 0], budget_bytes=1)


@pytest.mark.parametrize("name,width,size", FORMATS)
def test_rejects_bad_payload(name, width, size):
    with pytest.raises(ValueError):
        mod.dequantize(torch.zeros(size - 1, dtype=torch.uint8), name)


@pytest.mark.parametrize("name,width,size", FORMATS)
def test_selected_model_rows_can_span_multiple_blocks(name, width, size):
    raw = torch.arange(size * 6, dtype=torch.uint8).reshape(3, 2 * size)
    got = mod.dequantize(
        raw, name, rows=[2, 0], row_width=2 * width, budget_bytes=4 * width * 4
    )
    expected = (
        getattr(quants, name)
        .dequantize_blocks(raw[[2, 0]].reshape(-1, size).numpy())
        .reshape(-1)
    )
    torch.testing.assert_close(got, torch.from_numpy(expected), rtol=0, atol=0)


@pytest.mark.parametrize("rows", [[0.5], [[0]], [True]])
def test_row_ids_are_not_silently_coerced(rows):
    with pytest.raises(ValueError, match="integer"):
        mod.dequantize(
            torch.zeros(34 * 3, dtype=torch.uint8), "Q8_0", rows=rows, budget_bytes=1024
        )


@pytest.mark.parametrize("budget", [float("nan"), True, 3.5])
def test_budget_requires_integer_bytes(budget):
    with pytest.raises(ValueError, match="budget"):
        mod.dequantize(torch.zeros(34, dtype=torch.uint8), "Q8_0", budget_bytes=budget)


@pytest.mark.parametrize("data", [5, "not-bytes", [0] * 34])
def test_only_real_byte_buffers_or_uint8_tensors_are_accepted(data):
    with pytest.raises(TypeError):
        mod.dequantize(data, "Q8_0")


def test_budget_is_checked_before_quant_decode(monkeypatch):
    def forbidden_decode(*args):
        pytest.fail("decoded before budget admission")

    monkeypatch.setattr(mod, "_f16", forbidden_decode)
    with pytest.raises(ValueError, match="budget"):
        mod.dequantize(
            torch.zeros(34 * 100, dtype=torch.uint8), "Q8_0", rows=[0], budget_bytes=1
        )


@pytest.mark.parametrize(
    "name,dtype", [("F32", torch.float32), ("BF16", torch.bfloat16)]
)
def test_plain_formats(name, dtype):
    x = torch.tensor([1.25, -2.5], dtype=dtype)
    got = mod.dequantize(x.view(torch.uint8), name)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, x.float())
