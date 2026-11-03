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
    this module after the server environment is in place. An exact boot
    also checks the port policy below. Those checks are skipped while the
    flag is off, so a normal serve does not read them.
    """
    global _MODE
    if _MODE is None:
        mode = _parse(os.environ.get("VLLM_LSA", "0"))
        if mode == "exact":
            enforce_port_env()
        _MODE = mode
    return _MODE


def upstream_would_pick_cute(
    kernels: str,
    device_type: str,
    capability: tuple[int, int],
) -> bool:
    """Mirror ``runs_cutedsl`` at commit ``906be195``.

    Their table treats ``(10, 3)`` as B300. This fork's
    ``_capability_from_gcn_arch("gfx1030")`` is also ``(10, 3)``.
    """
    cute = {(9, 0), (10, 3)}
    return kernels == "cutedsl" and device_type == "cuda" and capability in cute


def enforce_port_env() -> None:
    """Refuse a CuTe or tf32 Ward request. Attention stays exact either way.

    ``VLLM_LSA_KERNELS``, ``VLLM_LSA_WARD_PRECISION``, and
    ``VLLM_LSA_SCORE`` are server-process variables. Clients do not send
    them. Unset means ``triton``, ``ieee``, and ``fp32``.
    """
    kernels = os.environ.get("VLLM_LSA_KERNELS", "triton").strip().lower()
    if kernels != "triton":
        raise LSAConfigError(
            f"VLLM_LSA_KERNELS={kernels!r} is refused. "
            "Force triton. Their dispatch.py selects CuTeDSL for "
            "capability (10, 3), and gfx1030 parses to (10, 3)."
        )
    precision = os.environ.get("VLLM_LSA_WARD_PRECISION", "ieee").strip().lower()
    if precision != "ieee":
        raise LSAConfigError(
            f"VLLM_LSA_WARD_PRECISION={precision!r} is refused. "
            "Pin ieee. Their ward.py defaults to tf32x3, and gfx1030 "
            "has no tf32 matrix path."
        )
    score = os.environ.get("VLLM_LSA_SCORE", "fp32").strip().lower()
    if score not in {"fp32", "fp16"}:
        raise LSAConfigError(
            f"VLLM_LSA_SCORE={score!r} is refused. "
            "Use fp32 (one ieee dot) or fp16 with fp32 accumulation. "
            "The bf16 three-dot split in their block_scores.py is not "
            "a gfx1030 path."
        )


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
