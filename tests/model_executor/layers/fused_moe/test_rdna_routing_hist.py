# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the opt-in MoE routing-histogram capture."""

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.rdna_routing_hist import (
    HIST_DIR_ENV,
    HIST_FILE_NAME,
    HIST_MIN_TOKENS_ENV,
    RoutingHistogram,
    layer_index,
)

NUM_EXPERTS = 16
TOP_K = 2


def _layer(idx: int = 3, ep_rank: int = 0, name: str | None = None):
    config = SimpleNamespace(ep_rank=ep_rank, ep_size=4, tp_rank=0, dp_rank=0)
    return SimpleNamespace(
        layer_name=f"model.layers.{idx}.mlp.experts" if name is None else name,
        moe_config=config,
        global_num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        expert_placement_strategy="linear",
    )


def _ids(tokens: int) -> torch.Tensor:
    # token t routes to experts (t % 16, (t + 5) % 16)
    t = torch.arange(tokens)
    return torch.stack([t % NUM_EXPERTS, (t + 5) % NUM_EXPERTS], dim=1).to(torch.int32)


def _read(tmp_path):
    path = tmp_path / HIST_FILE_NAME
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.12.mlp.experts", 12),
        ("language_model.model.layers.7.mlp.experts", 7),
        ("mtp.layers.0.mlp.experts", 0),
        ("model.embed_tokens", -1),
        ("model.layers12.mlp", -1),
    ],
)
def test_layer_index(name, expected):
    assert layer_index(name) == expected


def test_records_counts_and_steps(tmp_path):
    hist = RoutingHistogram(str(tmp_path), min_tokens=8)
    ids = _ids(32)
    for idx in (0, 1, 2, 0, 1):  # two forward passes
        hist.record(_layer(idx), ids)
    lines = _read(tmp_path)
    meta, records = lines[0]["meta"], lines[1:]
    assert meta["num_experts"] == NUM_EXPERTS
    assert meta["ep_size"] == 4
    assert meta["top_k"] == TOP_K
    assert meta["linear_placement"] is True
    assert [r["step"] for r in records] == [0, 0, 0, 1, 1]
    assert [r["layer_idx"] for r in records] == [0, 1, 2, 0, 1]
    assert [r["seq"] for r in records] == [1, 2, 3, 4, 5]
    expected = torch.bincount(ids.reshape(-1).long(), minlength=NUM_EXPERTS).tolist()
    for record in records:
        assert record["tokens"] == 32
        assert record["counts"] == expected
        assert sum(record["counts"]) == 32 * TOP_K


def test_skips_small_batches_other_ranks_and_unindexed_layers(tmp_path):
    hist = RoutingHistogram(str(tmp_path), min_tokens=8)
    hist.record(_layer(0), _ids(7))  # decode-sized
    hist.record(_layer(0, ep_rank=1), _ids(32))  # not the recording rank
    hist.record(_layer(name="model.norm"), _ids(32))  # no layer index
    assert not (tmp_path / HIST_FILE_NAME).exists()
    hist.record(_layer(5), _ids(8))
    records = _read(tmp_path)[1:]
    assert [r["layer_idx"] for r in records] == [5]


def test_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv(HIST_DIR_ENV, raising=False)
    assert RoutingHistogram.from_env() is None
    monkeypatch.setenv(HIST_DIR_ENV, str(tmp_path / "capture"))
    monkeypatch.setenv(HIST_MIN_TOKENS_ENV, "64")
    hist = RoutingHistogram.from_env()
    assert hist is not None
    assert hist.min_tokens == 64
    assert (tmp_path / "capture").is_dir()
