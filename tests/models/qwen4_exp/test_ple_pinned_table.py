# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx10x pinned PLE table: exact-size allocation and the GTT headroom check."""

import pytest
import torch

import vllm.models.qwen4_exp.common.ngram_embedding as ne
import vllm.platforms.rocm_host_memory as rhm


class _TPGroup:
    """Stand-in TP group: broadcast_object hands every rank rank 0's verdict."""

    def __init__(self, rank: int, world_size: int, verdict: str | None):
        self.rank_in_group = rank
        self.world_size = world_size
        self.verdict = verdict

    def broadcast_object(self, obj, src: int = 0):
        assert src == 0
        return obj if self.rank_in_group == 0 else self.verdict


@pytest.fixture
def allocations(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []

    def fake_alloc(shape, dtype):
        calls.append((shape, dtype))
        return torch.empty(shape, dtype=dtype)

    monkeypatch.setattr(ne, "alloc_pinned_exact", fake_alloc)
    return calls


def test_userptr_mode_allocates_exact_without_collectives(
    monkeypatch: pytest.MonkeyPatch, allocations: list
):
    monkeypatch.delenv(rhm.PAGED_HOST_MEMORY_ENV, raising=False)
    monkeypatch.setattr(ne, "get_tp_group", lambda: pytest.fail("no TP collective"))
    table = ne._allocate_rdna_pinned_table((5, 7), torch.bfloat16)
    assert allocations == [((5, 7), torch.bfloat16)]
    assert table.shape == (5, 7)


def test_gtt_mode_rank0_checks_the_whole_tp_group(
    monkeypatch: pytest.MonkeyPatch, allocations: list
):
    monkeypatch.setenv(rhm.PAGED_HOST_MEMORY_ENV, "0")
    checked = []
    monkeypatch.setattr(
        rhm,
        "gtt_headroom_error",
        lambda nbytes, ranks, what: checked.append((nbytes, ranks)),
    )
    monkeypatch.setattr(ne, "get_tp_group", lambda: _TPGroup(0, 4, None))
    ne._allocate_rdna_pinned_table((5, 7), torch.bfloat16)
    assert checked == [(5 * 7 * 2, 4)]
    assert allocations == [((5, 7), torch.bfloat16)]


@pytest.mark.parametrize("rank", [0, 1])
def test_gtt_mode_every_rank_fails_fast_before_allocating(
    monkeypatch: pytest.MonkeyPatch, allocations: list, rank: int
):
    monkeypatch.setenv(rhm.PAGED_HOST_MEMORY_ENV, "0")

    def headroom(nbytes, ranks, what):
        assert rank == 0, "only TP rank 0 reads sysfs"
        return "no room"

    monkeypatch.setattr(rhm, "gtt_headroom_error", headroom)
    monkeypatch.setattr(ne, "get_tp_group", lambda: _TPGroup(rank, 4, "no room"))
    with pytest.raises(RuntimeError, match="no room"):
        ne._allocate_rdna_pinned_table((5, 7), torch.bfloat16)
    assert allocations == []
