# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[3] / "tools/rdna/mono/sweep_offload_policy.py"
_SPEC = importlib.util.spec_from_file_location("policy_sweep", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)


def test_counts_unique_transfers_but_observes_duplicate_routes():
    counts = sweep.simulate([[0, 0, 1, -1, 256], [0, 1], [2, 3]], 2, "lfu_lru")
    assert counts == dict(
        calls=3,
        routed_pairs=7,
        expert_demands=6,
        hits=2,
        misses=4,
        passes=3,
        copied_expert_units=4,
    )


def test_overflow_and_empty_calls_preserve_accounting():
    counts = sweep.simulate([[0, 1, 2], [], [0, 1, 2]], 2, "lru")
    assert counts["passes"] == 4
    assert counts["hits"] + counts["misses"] == 6
    assert counts["calls"] == 3


def test_rank_mapping_numeric_call_order_and_safe_load(tmp_path):
    for call, ids in [(10, [257]), (2, [256, 257, 511, 0])]:
        torch.save(
            dict(
                rank=1, layer=7, call=call, tensors={"route_ids": torch.tensor([ids])}
            ),
            tmp_path / f"rank1-layer7-call{call}.pt",
        )
    records = sweep.read_calls(tmp_path)
    assert records[(1, 7)] == [[0, 1, 255, -256], [1]]
    report = sweep.sweep(records, [2], ["lru"], payload_bytes=100)
    result = report["results"][0]
    assert result["modeled_copy_bytes"] == result["totals"]["misses"] * 100
    assert result["per_rank"]["1"]["calls"] == 2


def test_invalid_global_ids_fail(tmp_path):
    torch.save(
        dict(rank=0, layer=1, call=1, tensors={"route_ids": torch.tensor([512])}),
        tmp_path / "rank0-layer1-call1.pt",
    )
    with pytest.raises(ValueError, match="invalid global"):
        sweep.read_calls(tmp_path)
