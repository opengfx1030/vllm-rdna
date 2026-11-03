# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment for the RidgeFill explore draft.

Every read is live. Unset and ``0`` both mean off, so importing this module
does not change a serve. Workers that never see the caller's environment stay
on the exact path, which is the fail-safe direction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_OFF = frozenset({"", "0", "false", "False", "off", "OFF", "no", "NO"})

# gfx1030 default. bf16 is the overflow fallback, not the serve dtype.
_FP16 = frozenset({"fp16", "float16", "half"})
_BF16 = frozenset({"bf16", "bfloat16"})


class RidgeFillConfigError(ValueError):
    """A RidgeFill flag is set to something this draft does not implement."""


def _raw(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def enabled() -> bool:
    """True only when ``VLLM_RIDGEFILL`` is explicitly on."""
    return _raw("VLLM_RIDGEFILL", "0") not in _OFF


def map_dtype() -> str:
    """``fp16`` (default) or ``bf16`` from ``VLLM_RIDGEFILL_MAP_DTYPE``."""
    raw = _raw("VLLM_RIDGEFILL_MAP_DTYPE", "fp16").lower()
    if raw in _FP16:
        return "fp16"
    if raw in _BF16:
        return "bf16"
    raise RidgeFillConfigError(
        f"VLLM_RIDGEFILL_MAP_DTYPE must be fp16 (default) or bf16, got {raw!r}"
    )


def cache_only() -> bool:
    """True when filled rows do only their cache side effects.

    ``VLLM_RIDGEFILL_CACHE_ONLY=1`` selects the rework path: filled rows write
    QSA K/V and indexer keys and update GDN state, and the layer's output work
    runs on the compacted exact rows. Default off, so a boot without the flag
    keeps the fill path the eager A/B measured until the compiled A/B runs.
    """
    return _raw("VLLM_RIDGEFILL_CACHE_ONLY", "0") not in _OFF


def apc_mode() -> str:
    """``off`` (v1) or ``v2`` (salt filled blocks). Anything else is an error."""
    raw = _raw("VLLM_RIDGEFILL_APC", "off").lower()
    if raw in _OFF or raw == "off":
        return "off"
    if raw == "v2":
        return "v2"
    raise RidgeFillConfigError(
        "VLLM_RIDGEFILL_APC must be off (default) or v2, got " + raw
    )


def _int(name: str, default: int) -> int:
    raw = _raw(name, str(default))
    try:
        return int(raw, 10)
    except ValueError as exc:
        raise RidgeFillConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = _raw(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise RidgeFillConfigError(f"{name} must be a float, got {raw!r}") from exc


@dataclass(frozen=True)
class RidgeConfig:
    """Knobs for one fill. Defaults match the Flash-Next paper configuration."""

    split: int = 24
    tail: int = 2048
    din: int = 10240
    dout: int = 2560
    lam: float = 0.03
    group: int = 128
    map_dtype: str = "fp16"
    apc: str = "off"
    mode: str = "plain"
    map_path: str = ""
    map_sha: str = ""
    pins_path: str = ""
    block_size: int = 0


def load_config() -> RidgeConfig:
    """Read the current environment. Does not require the flag to be on."""
    mode = _raw("VLLM_RIDGEFILL_MODE", "plain").lower()
    if mode not in ("plain", "quality"):
        raise RidgeFillConfigError(
            "VLLM_RIDGEFILL_MODE must be plain or quality, got " + mode
        )
    return RidgeConfig(
        split=_int("VLLM_RIDGEFILL_SPLIT", 24),
        tail=_int("VLLM_RIDGEFILL_TAIL", 2048),
        din=_int("VLLM_RIDGEFILL_DIN", 10240),
        dout=_int("VLLM_RIDGEFILL_DOUT", 2560),
        lam=_float("VLLM_RIDGEFILL_LAMBDA", 0.03),
        group=_int("VLLM_RIDGEFILL_GROUP", 128),
        map_dtype=map_dtype(),
        apc=apc_mode(),
        mode=mode,
        map_path=_raw("VLLM_RIDGEFILL_MAP", ""),
        map_sha=_raw("VLLM_RIDGEFILL_MAP_SHA", ""),
        pins_path=_raw("VLLM_RIDGEFILL_PINS", ""),
        block_size=_int("VLLM_RIDGEFILL_BLOCK", 0),
    )
