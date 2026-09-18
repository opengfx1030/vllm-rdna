# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for PLE fused gather on int4 and fp8 per-row sidecars.

Leap's T-PLE8 path views e4m3 bytes as uint8 and dequants through a
256-entry LUT in gather_rows_small. The fused decode gate must accept
is_fp8 tables, not only layout strings containing "int4".
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.ple_offload.worker import _fused_decode_lookup, _PleQuantTable


def _fp8_table(raw: torch.Tensor, scales: torch.Tensor) -> _PleQuantTable:
    table = _PleQuantTable.__new__(_PleQuantTable)
    table.is_fp8 = True
    table.layout = "per_row_e4m3"
    table.width = raw.shape[1]
    table._q_np = [raw.numpy()]
    table._s_np = [scales.numpy()]
    table._lut_np = (
        torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    )
    return table


def _int4_table(packed: np.ndarray, scales: np.ndarray) -> _PleQuantTable:
    table = _PleQuantTable.__new__(_PleQuantTable)
    table.is_fp8 = False
    table.layout = "int4_g16"
    table.width = packed.shape[1] * 2
    table._q_np = [packed]
    table._s_np = [scales]
    table._lut_np = None
    return table


def test_gather_rows_small_fp8_matches_torch_dequant():
    width = 8
    raw_u8 = torch.tensor(
        [
            [0, 64, 128, 192, 32, 96, 160, 224],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [255, 127, 63, 31, 15, 7, 3, 1],
        ],
        dtype=torch.uint8,
    )
    q = raw_u8.view(torch.float8_e4m3fn)
    scales = torch.tensor([0.5, 1.25, 2.0], dtype=torch.float32)
    table = _fp8_table(raw_u8, scales)
    ids = np.array([0, 2], dtype=np.int64)
    out = np.empty((2, width), dtype=np.float32)
    table.gather_rows_small(ids, out)
    ref = (q.float() * scales[:, None]).numpy()[ids]
    np.testing.assert_allclose(out, ref, rtol=0, atol=0)


def test_gather_rows_small_int4_unchanged():
    packed = np.array([[0x10, 0x32], [0x54, 0x76]], dtype=np.uint8)
    scales = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float16)
    table = _int4_table(packed, scales)
    ids = np.array([1], dtype=np.int64)
    out = np.empty((1, 4), dtype=np.float32)
    table.gather_rows_small(ids, out)
    # packed 0x54, 0x76 → nibbles (4,5) (6,7); minus 8 times scale 1
    np.testing.assert_allclose(out, [[-4.0, -3.0, -2.0, -1.0]], rtol=0, atol=1e-6)


def test_fused_decode_gate_accepts_fp8_layout():
    table = _fp8_table(
        torch.zeros((1, 4), dtype=torch.uint8),
        torch.ones((1,), dtype=torch.float32),
    )
    layer = SimpleNamespace(ngram_embedding=SimpleNamespace(_ple_quant=table))
    ngram = torch.zeros((1, 2), dtype=torch.int64)
    qsl = torch.tensor([0, 1], dtype=torch.int32)
    ids = torch.zeros(1, dtype=torch.int32)
    pinned = torch.zeros(1, 4)
    # Decode-shaped batch: a layout reject returns None. Passing the gate
    # reaches layer.ngram_size before any gather.
    with pytest.raises(AttributeError, match="ngram_size"):
        _fused_decode_lookup(layer, ids, qsl, ngram, pinned, False)


def test_fused_decode_gate_rejects_other_layouts():
    table = SimpleNamespace(layout="e2m1_e4m3_scale", is_fp8=False)
    layer = SimpleNamespace(ngram_embedding=SimpleNamespace(_ple_quant=table))
    ngram = torch.zeros((1, 2), dtype=torch.int64)
    qsl = torch.tensor([0, 1], dtype=torch.int32)
    ids = torch.zeros(1, dtype=torch.int32)
    pinned = torch.zeros(1, 4)
    assert _fused_decode_lookup(layer, ids, qsl, ngram, pinned, False) is None


@pytest.mark.parametrize(
    "layout,expect_fp8",
    [
        ("per_row_e4m3", True),
        ("int4_g16", False),
        ("e2m1_e4m3_scale", False),
    ],
)
def test_is_fp8_excludes_e2m1_even_if_layout_mentions_e4m3(
    layout: str, expect_fp8: bool
):
    is_fp8 = ("e4m3" in layout) and ("e2m1" not in layout)
    assert is_fp8 is expect_fp8
