# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the lane P QSA cache binding (two-lane, mono QSA)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.lane.qsa_cache_bind import (
    bind_prefill_lane_qsa_caches,
)


class _Cache:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.kv_cache = tensor
        self.bound: list[torch.Tensor] = []

    def bind_kv_cache(self, tensor: torch.Tensor) -> None:
        if tensor.shape != self.kv_cache.shape:
            raise ValueError("shape mismatch")
        self.kv_cache = tensor
        self.bound.append(tensor)


def _runner(context: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=context)
        )
    )


def _tensor(value: float) -> torch.Tensor:
    return torch.full((2, 3), value)


def test_binds_p_qsa_caches_to_d() -> None:
    c = _Cache(_tensor(1.0))
    r = _Cache(_tensor(2.0))
    p = SimpleNamespace(kv_cache=_tensor(3.0))
    primary_c = _Cache(_tensor(4.0))
    primary_r = _Cache(_tensor(5.0))
    primary_p = SimpleNamespace(kv_cache=_tensor(6.0))
    prefill = _runner(
        {
            "model.layers.0.self_attn.indexer.raw_key_cache": r,
            "model.layers.0.self_attn.indexer.compressed_key_cache": c,
            "model.layers.0.self_attn.attn": p,
        }
    )
    primary = _runner(
        {
            "model.layers.0.self_attn.indexer.raw_key_cache": primary_r,
            "model.layers.0.self_attn.indexer.compressed_key_cache": primary_c,
            "model.layers.0.self_attn.attn": primary_p,
        }
    )

    assert bind_prefill_lane_qsa_caches(prefill, primary) == 3
    for mine, theirs in (
        (c, primary_c),
        (r, primary_r),
        (p, primary_p),
    ):
        assert mine.kv_cache.data_ptr() == theirs.kv_cache.data_ptr()
    # The QSA modules use bind_kv_cache; the attention module a plain attribute.
    assert len(c.bound) == 1 and len(r.bound) == 1


def test_already_bound_is_noop() -> None:
    shared = _Cache(_tensor(7.0))
    original = shared.kv_cache
    prefill = _runner({"model.layers.1.self_attn.indexer.raw_key_cache": shared})
    primary = _runner({"model.layers.1.self_attn.indexer.raw_key_cache": shared})
    assert bind_prefill_lane_qsa_caches(prefill, primary) == 0
    assert shared.kv_cache is original
    assert not shared.bound


def test_shape_mismatch_is_refused() -> None:
    mine = _Cache(torch.zeros(2, 3))
    theirs = _Cache(torch.zeros(4, 3))
    prefill = _runner({"model.layers.2.self_attn.indexer.raw_key_cache": mine})
    primary = _runner({"model.layers.2.self_attn.indexer.raw_key_cache": theirs})
    with pytest.raises(RuntimeError, match="does not match lane D"):
        bind_prefill_lane_qsa_caches(prefill, primary)


def test_unrelated_modules_are_ignored() -> None:
    prefill = _runner(
        {
            "model.layers.3.mono_rdna2": SimpleNamespace(kv_cache=_tensor(0.0)),
        }
    )
    primary = _runner(
        {
            "model.layers.3.mono_rdna2": SimpleNamespace(kv_cache=_tensor(1.0)),
        }
    )
    assert bind_prefill_lane_qsa_caches(prefill, primary) == 0
