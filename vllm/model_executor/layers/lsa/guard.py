# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve-side switch for the LSA explore stub.

The variable is ``VLLM_LSA`` in the server process. Clients and gateways
never send it. ``0`` (the default) and ``1`` both leave attention exact.
``1`` only records that a softmax layer was visited. Any other value
raises, so a request for real selection cannot fall through silently.

This module does not import torch and does not touch tensors, KV blocks,
or prefix-cache hashes.
"""

from __future__ import annotations

import os

_OFF = frozenset({"", "0", "false", "off", "no"})
_EXACT = frozenset({"1", "true", "on", "exact", "guard"})
_MODE: str | None = None


class LSAConfigError(ValueError):
    """``VLLM_LSA`` asked for a selector this tree does not have."""


def reset_lsa_state_for_tests() -> None:
    """Drop the cached parse. Tests only."""
    global _MODE
    _MODE = None


def lsa_mode() -> str:
    """Return ``off`` or ``exact``. Raise if the variable names a selector.

    The first call reads ``VLLM_LSA`` and caches it. Engine workers import
    this module after the server environment is in place.
    """
    global _MODE
    if _MODE is None:
        _MODE = _parse(os.environ.get("VLLM_LSA", "0"))
    return _MODE


def note_exact_layer(layer_type: str) -> str:
    """Acknowledge a softmax layer while attention stays exact.

    Returns ``layer_type`` unchanged. GDN layers are not LSA carriers;
    calling this with ``linear_attention`` while the guard is armed raises.
    """
    mode = lsa_mode()
    if mode == "exact" and layer_type == "linear_attention":
        raise LSAConfigError(
            "LSA does not apply to linear_attention (GDN). "
            "Only softmax layers can carry it, and this stub does not "
            "select keys."
        )
    return layer_type


def _parse(raw: str) -> str:
    value = raw.strip().lower()
    if value in _OFF:
        return "off"
    if value in _EXACT:
        return "exact"
    raise LSAConfigError(
        f"VLLM_LSA={raw!r} is not implemented. "
        "Leave it unset or set 0. Set 1 to arm the guard; attention "
        "stays exact either way. select and tree are not wired."
    )
