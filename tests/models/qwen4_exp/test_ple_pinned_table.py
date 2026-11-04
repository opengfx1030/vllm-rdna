# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx10x pinned PLE table: allocated at its exact size."""

import pytest
import torch

import vllm.models.qwen4_exp.common.ngram_embedding as ne


def test_rdna_pinned_table_is_allocated_at_exact_size(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list = []

    def fake_alloc(shape, dtype):
        calls.append((shape, dtype))
        return torch.empty(shape, dtype=dtype)

    monkeypatch.setattr(ne, "alloc_pinned_exact", fake_alloc)
    table = ne._allocate_rdna_pinned_table((5, 7), torch.bfloat16)
    assert calls == [((5, 7), torch.bfloat16)]
    assert table.shape == (5, 7) and table.dtype == torch.bfloat16
