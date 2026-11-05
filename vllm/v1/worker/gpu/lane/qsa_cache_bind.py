# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind lane P's mono QSA caches to lane D's after KV initialization.

The mono QSA kernel reads and writes three caches per layer: the compressed
indexer K cache (``c``), the raw indexer K cache plus its RoPE-position ring
(``r``), and the main attention K/V cache (``p``).  A workspace lane is not
enough: the fused stores key the compressed/raw caches by ``c_slots`` and
``r_slots``, and the epilogue keys ``p`` by the main attention slots.  If the
two lanes are handed distinct cache tensors, lane P writes a cache lane D never
reads (and shared-pool block ids can address outside P's tensor).

In lane mode both model runners share the model, so today the QSA cache modules
and their buffers are usually the same objects.  This guard makes the binding
explicit and stable: it runs after ``initialize_kv_cache`` (the buffers exist)
and before lane P captures its graphs.  If a future lane setup builds a
per-lane QSA module, its ``kv_cache`` is rebound to the primary's tensor; the
module stays per-lane, the cache storage does not.
"""

from __future__ import annotations

from typing import Any

_QSA_CACHE_SUFFIXES = (
    ".indexer.raw_key_cache",
    ".indexer.compressed_key_cache",
    ".self_attn.attn",
)


def _has_cache_tensor(module: Any) -> bool:
    cache = getattr(module, "kv_cache", None)
    return cache is not None and hasattr(cache, "data_ptr")


def bind_prefill_lane_qsa_caches(
    prefill_runner: Any,
    primary_runner: Any,
) -> int:
    """Make lane P's QSA caches alias lane D's; return the number bound.

    P must already have run ``initialize_kv_cache`` (it needs D's buffers) and
    must not have captured a graph yet.  The primary's matching module supplies
    the storage; a P-only module keeps its own Python object but points at that
    storage, so both lanes see every cache row.
    """
    prefill_context = (
        prefill_runner.vllm_config.compilation_config.static_forward_context
    )
    primary_context = (
        primary_runner.vllm_config.compilation_config.static_forward_context
    )
    bound = 0
    for prefix, module in prefill_context.items():
        if prefix not in primary_context:
            continue
        if not prefix.endswith(_QSA_CACHE_SUFFIXES):
            continue
        primary_module = primary_context[prefix]
        if not _has_cache_tensor(module) or not _has_cache_tensor(primary_module):
            continue
        mine = module.kv_cache
        theirs = primary_module.kv_cache
        if mine.data_ptr() == theirs.data_ptr():
            continue
        if mine.shape != theirs.shape or mine.dtype != theirs.dtype:
            raise RuntimeError(
                f"lane P QSA cache {prefix} does not match lane D's "
                f"(shape {tuple(mine.shape)} vs {tuple(theirs.shape)}, dtype "
                f"{mine.dtype} vs {theirs.dtype})"
            )
        bind = getattr(module, "bind_kv_cache", None)
        if bind is not None:
            bind(theirs)
        else:
            # The main attention module owns kv_cache as a plain attribute.
            module.kv_cache = theirs
        bound += 1
    return bound


__all__ = ["bind_prefill_lane_qsa_caches"]
