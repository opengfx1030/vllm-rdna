# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 rank-map and per-rank accounting for the offline policy sweep.

torch is imported lazily by read_calls, so the rank contract is testable
wherever torch exists (the sweep itself is CPU-only analysis).
"""

import importlib.util
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[3] / "tools/rdna/mono/sweep_offload_policy.py"
)
_SPEC = importlib.util.spec_from_file_location("policy_sweep_tp4", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)


def _write(tmp_path, rank, layer, call, ids):
    import torch

    torch.save(
        dict(
            rank=rank, layer=layer, call=call, tensors={"route_ids": torch.tensor([ids])}
        ),
        tmp_path / f"rank{rank}-layer{layer}-call{call}.pt",
    )


def test_tp4_rank_map_is_per_rank_local_shift(tmp_path):
    # 128 local experts, 512 global: rank r owns [128r, 128r+128).
    _write(tmp_path, 0, 1, 1, [0, 127])
    _write(tmp_path, 1, 1, 1, [128, 255])
    _write(tmp_path, 3, 1, 1, [384, 511])
    records = sweep.read_calls(tmp_path, local_experts=128)
    assert records[(0, 1)] == [[0, 127]]
    assert records[(1, 1)] == [[0, 127]]
    assert records[(3, 1)] == [[0, 127]]


def test_tp2_default_behavior_preserved(tmp_path):
    _write(tmp_path, 1, 2, 1, [256, 511])
    records = sweep.read_calls(tmp_path)  # defaults: 256 local, 512 global
    assert records[(1, 2)] == [[0, 255]]


def test_rank_outside_topology_refused(tmp_path):
    _write(tmp_path, 4, 0, 1, [0])  # rank 4 does not exist at TP4
    with pytest.raises(ValueError, match="rank"):
        sweep.read_calls(tmp_path, local_experts=128)
    _write(tmp_path, 2, 0, 2, [0])  # rank 2 does not exist at TP2
    with pytest.raises(ValueError, match="rank"):
        sweep.read_calls(tmp_path)


def test_global_id_outside_owned_range_refused(tmp_path):
    _write(tmp_path, 0, 0, 1, [512])
    with pytest.raises(ValueError, match="invalid global"):
        sweep.read_calls(tmp_path, local_experts=128)


def test_local_experts_that_do_not_partition_global_refused(tmp_path):
    _write(tmp_path, 0, 0, 1, [0])
    with pytest.raises(ValueError, match="divide"):
        sweep.read_calls(tmp_path, local_experts=100, global_experts=512)


def test_sweep_reports_all_four_ranks_separately(tmp_path):
    for rank in range(4):
        _write(tmp_path, rank, 5, 1, [rank * 128, rank * 128 + 1])
    records = sweep.read_calls(tmp_path, local_experts=128)
    report = sweep.sweep(records, [2], ["lru"], local_experts=128)
    per_rank = report["results"][0]["per_rank"]
    assert sorted(per_rank) == ["0", "1", "2", "3"]
    assert all(per_rank[str(rank)]["calls"] == 1 for rank in range(4))
    assert report["results"][0]["totals"]["expert_demands"] == 8
