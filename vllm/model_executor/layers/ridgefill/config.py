# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment for the RidgeFill explore draft.

Every knob is registered in ``vllm/envs.py`` as ``VLLM_RDNA_RIDGEFILL*`` and
read through ``vllm.envs``; no model code touches ``os.environ``. Unset and
``0`` both mean off, so importing this module does not change a serve.
Workers that never see the caller's environment stay on the exact path,
which is the fail-safe direction.
"""

from __future__ import annotations

from dataclasses import dataclass

# gfx1030 default. bf16 is the overflow fallback, not the serve dtype.
_FP16 = frozenset({"fp16", "float16", "half"})
_BF16 = frozenset({"bf16", "bfloat16"})
_MAP_MODES = frozenset({"dot2", "dot4"})


class RidgeFillConfigError(ValueError):
    """A RidgeFill flag is set to something this draft does not implement."""


def _env(name: str):
    """Read a knob registered in ``vllm.envs``.

    Imported lazily so this module stays importable on a host without vllm,
    which is where the pure-Python CPU proofs run.
    """
    from vllm import envs

    return getattr(envs, name)


def enabled() -> bool:
    """True only when ``VLLM_RDNA_RIDGEFILL`` is explicitly on."""
    return _env_bool("VLLM_RDNA_RIDGEFILL")


def map_dtype() -> str:
    """``fp16`` (default) or ``bf16`` from ``..._MAP_DTYPE``."""
    raw = str(_env("VLLM_RDNA_RIDGEFILL_MAP_DTYPE")).lower()
    if raw in _FP16:
        return "fp16"
    if raw in _BF16:
        return "bf16"
    raise RidgeFillConfigError(
        f"VLLM_RDNA_RIDGEFILL_MAP_DTYPE must be fp16 (default) or bf16, got {raw!r}"
    )


def _env_bool(name: str) -> bool:
    """A ``VLLM_RDNA_RIDGEFILL*`` boolean knob, false unless set to 1."""
    raw = str(_env(name)).strip().lower()
    return raw not in ("", "0", "false", "off", "no")


def map_mode() -> str:
    """``dot2`` (default) or ``dot4`` from ``..._MAP_MODE``.

    dot2 is the fp16 map GEMM (rocBLAS per layer, or the fused HIP op with
    ``..._MAP_FUSED=1``). dot4 is the fused HIP op on int8 x int8 with
    Hadamard-rotated, per-row quantized maps and activations, so it needs
    ``..._MAP_FUSED=1`` (checked in ``load_config``). dot8 (int4) is not
    implemented and refused rather than silently run as dot2.
    """
    raw = str(_env("VLLM_RDNA_RIDGEFILL_MAP_MODE")).lower()
    if raw not in _MAP_MODES:
        raise RidgeFillConfigError(
            f"VLLM_RDNA_RIDGEFILL_MAP_MODE must be dot2 (default) or dot4, got {raw!r}"
        )
    return raw


def cache_only() -> bool:
    """True when filled rows do only their cache side effects.

    ``VLLM_RDNA_RIDGEFILL_CACHE_ONLY=1`` selects the rework path: filled rows
    write QSA K/V and indexer keys and update GDN state, and the layer's
    output work runs on the compacted exact rows. Default off, so a boot
    without the flag keeps the fill path the eager A/B measured until the
    compiled A/B runs.
    """
    return _env_bool("VLLM_RDNA_RIDGEFILL_CACHE_ONLY")


def debug() -> bool:
    """True when ``VLLM_RDNA_RIDGEFILL_DEBUG`` asks for the row-plan log."""
    return _env_bool("VLLM_RDNA_RIDGEFILL_DEBUG")


def apc_mode() -> str:
    """``off`` (v1) or ``v2`` (salt filled blocks). Anything else is an error."""
    raw = str(_env("VLLM_RDNA_RIDGEFILL_APC")).lower()
    if raw in ("", "off"):
        return "off"
    if raw == "v2":
        return "v2"
    raise RidgeFillConfigError(
        "VLLM_RDNA_RIDGEFILL_APC must be off (default) or v2, got " + raw
    )


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
    map_mode: str = "dot2"
    map_fused: bool = False
    apc: str = "off"
    mode: str = "plain"
    map_path: str = ""
    map_sha: str = ""
    pins_path: str = ""
    block_size: int = 0


def load_config() -> RidgeConfig:
    """Read the current environment. Does not require the flag to be on."""
    mode = str(_env("VLLM_RDNA_RIDGEFILL_MODE")).lower()
    if mode not in ("plain", "quality"):
        raise RidgeFillConfigError(
            "VLLM_RDNA_RIDGEFILL_MODE must be plain or quality, got " + mode
        )
    cfg = RidgeConfig(
        split=int(_env("VLLM_RDNA_RIDGEFILL_SPLIT")),
        tail=int(_env("VLLM_RDNA_RIDGEFILL_TAIL")),
        din=int(_env("VLLM_RDNA_RIDGEFILL_DIN")),
        dout=int(_env("VLLM_RDNA_RIDGEFILL_DOUT")),
        lam=float(_env("VLLM_RDNA_RIDGEFILL_LAMBDA")),
        group=int(_env("VLLM_RDNA_RIDGEFILL_GROUP")),
        map_dtype=map_dtype(),
        map_mode=map_mode(),
        map_fused=_env_bool("VLLM_RDNA_RIDGEFILL_MAP_FUSED"),
        apc=apc_mode(),
        mode=mode,
        map_path=str(_env("VLLM_RDNA_RIDGEFILL_MAP")),
        map_sha=str(_env("VLLM_RDNA_RIDGEFILL_MAP_SHA")),
        pins_path=str(_env("VLLM_RDNA_RIDGEFILL_PINS")),
        block_size=int(_env("VLLM_RDNA_RIDGEFILL_BLOCK")),
    )
    if cfg.map_mode == "dot4" and (not cfg.map_fused or cfg.map_dtype != "fp16"):
        raise RidgeFillConfigError(
            "VLLM_RDNA_RIDGEFILL_MAP_MODE=dot4 runs on the fused HIP map op: "
            "set VLLM_RDNA_RIDGEFILL_MAP_FUSED=1 and keep MAP_DTYPE=fp16"
        )
    return cfg
