# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""v2 prefix-cache salt for filled blocks.

Inactive unless ``VLLM_RDNA_RIDGEFILL`` is on and ``VLLM_RDNA_RIDGEFILL_APC=v2``.
The salt is attached in ``generate_block_hash_extra_keys``. Exact blocks
and the flag-off path add nothing, so their hashes stay what they are.

A filled block's hash then matches only another fill-on request with the
same split, map sha, and mode. It does not match an exact block.
"""

from __future__ import annotations


def aligned_filled_end(num_tokens: int, tail: int, block_size: int) -> int:
    """Same cut as ``policy.filled_end(..., align=True)``."""
    raw = max(0, num_tokens - tail)
    if block_size <= 1:
        return raw
    return (raw // block_size) * block_size


def extra_key(
    *,
    num_prompt_tokens: int,
    start: int,
    end: int,
    block_size: int,
    split: int,
    tail: int,
    map_sha: str,
    mode: str,
    apc: str,
) -> tuple[str, int, str, str] | None:
    """Salt for one full block, or None when the block is exact or v2 is off.

    The block ``[start, end)`` is filled only when it ends at or before the
    block-aligned cut. A block that would mix filled and exact tokens is not
    produced, because the cut is a multiple of ``block_size``.
    """
    if apc != "v2":
        return None
    if end <= start:
        return None
    cut = aligned_filled_end(num_prompt_tokens, tail, block_size)
    if cut <= 0 or end > cut:
        return None
    sha = map_sha or "missing-sha"
    return ("ridgefill", int(split), sha, mode)


def extra_key_for_request(request, start: int, end: int):
    """Read the live env and the request length. Used by the hash hook."""
    from vllm.model_executor.layers.ridgefill.config import load_config

    cfg = load_config()
    if cfg.apc != "v2":
        return None
    block = end - start
    return extra_key(
        num_prompt_tokens=request.num_prompt_tokens,
        start=start,
        end=end,
        block_size=block,
        split=cfg.split,
        tail=cfg.tail,
        map_sha=cfg.map_sha,
        mode=cfg.mode,
        apc=cfg.apc,
    )
