# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx10x pinned PLE table: exact size, mlock()ed, usable when mlock fails."""

import errno

import pytest
import torch

import vllm.models.qwen4_exp.common.ngram_embedding as ne


@pytest.fixture
def allocations(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []

    def fake_alloc(shape, dtype):
        calls.append((shape, dtype))
        return torch.empty(shape, dtype=dtype)

    monkeypatch.setattr(ne, "alloc_pinned_exact", fake_alloc)
    return calls


def test_rdna_pinned_table_is_exact_and_locked(
    monkeypatch: pytest.MonkeyPatch, allocations: list
):
    locked: list = []

    def fake_mlock(t):
        locked.append(t)
        return 0

    monkeypatch.setattr(ne, "mlock_tensor", fake_mlock)
    monkeypatch.setattr(ne.logger, "warning", lambda *a: pytest.fail("warned"))
    table = ne._allocate_rdna_pinned_table((5, 7), torch.bfloat16)
    assert allocations == [((5, 7), torch.bfloat16)]
    assert locked == [table]
    assert table.shape == (5, 7) and table.dtype == torch.bfloat16


def test_rdna_pinned_table_falls_back_when_mlock_fails(
    monkeypatch: pytest.MonkeyPatch, allocations: list
):
    warnings: list = []
    monkeypatch.setattr(ne, "mlock_tensor", lambda t: errno.ENOMEM)
    monkeypatch.setattr(ne.logger, "warning", lambda *a: warnings.append(a))
    table = ne._allocate_rdna_pinned_table((5, 7), torch.bfloat16)
    assert table.shape == (5, 7)
    assert len(warnings) == 1
    assert "RLIMIT_MEMLOCK" in warnings[0][0]
