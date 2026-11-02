# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row policy for the RidgeFill draft. No torch, no model import.

v1 (the default) does not cache filled blocks. v2 salts them so a later
fill-on request can reuse the same approximation, and an exact request
cannot. The tail is rounded up to a hash-block boundary so a block is
entirely filled or entirely exact.
"""

from __future__ import annotations

from dataclasses import dataclass


class RidgeFillRefused(RuntimeError):
    """The flag is on and this serve contradicts the v1 contract."""


def filled_end(num_tokens: int, tail: int, block_size: int, align: bool) -> int:
    """First exact token. With ``align``, round the cut down to a block.

    Rounding down lengthens the exact tail. A filled token is never stored
    in a block that an exact request could hash.
    """
    raw = max(0, num_tokens - tail)
    if not align or block_size <= 1:
        return raw
    return (raw // block_size) * block_size


@dataclass(frozen=True)
class RowPlan:
    """Which prompt rows are filled. ``[0, filled_end)`` filled, the rest exact."""

    split: int
    tail: int
    num_tokens: int
    filled_end: int
    active: bool

    def kind(self, index: int) -> str:
        if index < 0 or index >= self.num_tokens:
            raise IndexError(index)
        if self.active and index < self.filled_end:
            return "filled"
        return "exact"


def plan_prompt(
    num_tokens: int,
    *,
    split: int,
    tail: int,
    block_size: int = 0,
    align: bool = False,
) -> RowPlan:
    """Build the row plan. Prompts of length ``<= tail`` are entirely exact."""
    if num_tokens < 0 or tail < 0 or split < 0:
        raise ValueError("num_tokens, tail and split must be non-negative")
    cut = filled_end(num_tokens, tail, block_size, align)
    return RowPlan(
        split=split,
        tail=tail,
        num_tokens=num_tokens,
        filled_end=cut,
        active=cut > 0,
    )


def chunk_class(plan: RowPlan, start: int, end: int) -> str:
    """``filled``, ``exact``, or ``mixed`` for a token interval ``[start, end)``."""
    if start < 0 or end < start or end > plan.num_tokens:
        raise ValueError(f"chunk [{start}, {end}) outside 0..{plan.num_tokens}")
    if start == end or not plan.active or start >= plan.filled_end:
        return "exact"
    if end <= plan.filled_end:
        return "filled"
    return "mixed"


def v1_guards(
    *,
    speculative: bool,
    prefix_caching: bool,
    apc_mode: str,
) -> None:
    """Refuse MTP/speculative decode, and refuse APC unless v2 is selected.

    Args:
        speculative: True when any speculative config is set, including MTP.
        prefix_caching: True when prefix caching is enabled.
        apc_mode: ``off`` or ``v2``.

    Raises:
        RidgeFillRefused: The combination is outside this draft.
    """
    if speculative:
        raise RidgeFillRefused(
            "speculative decoding, including MTP, is forced off while "
            "VLLM_RIDGEFILL=1. Filled prompt rows have no final hidden state "
            "for a drafter. A later option is exact decode with MTP enabled "
            "only after prefill (ENGINE.md C2)."
        )
    if prefix_caching and apc_mode != "v2":
        raise RidgeFillRefused(
            "prefix caching is forced off while RidgeFill v1 is on. "
            "Filled blocks must not be reused as exact. Set "
            "ENABLE_PREFIX_CACHING=0, or VLLM_RIDGEFILL_APC=v2 to salt "
            "filled blocks (ENGINE.md)."
        )


def apply_v1_guards(
    *,
    flag_on: bool,
    speculative: bool,
    prefix_caching: bool,
    apc_mode: str,
) -> None:
    """No-op when the flag is off, so the default serve is unchanged."""
    if not flag_on:
        return
    v1_guards(
        speculative=speculative,
        prefix_caching=prefix_caching,
        apc_mode=apc_mode,
    )


def cpu_tier_may_store(*, flag_on: bool, block_filled: bool) -> bool:
    """False when a filled block would enter the CPU KV tier.

    v1 and v2 both refuse that store. An exact block, and every block while
    the flag is off, may be stored.
    """
    if flag_on and block_filled:
        return False
    return True
