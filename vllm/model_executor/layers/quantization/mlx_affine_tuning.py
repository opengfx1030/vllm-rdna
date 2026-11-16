# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-independent, finite native affine dispatch configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_OMITTED = object()


@dataclass(frozen=True)
class MlxAffineTuning:
    decode_tile_n: int = 8
    prefill_tile_n: int = 8
    prefill_block_m: int = 4
    dense_prefill_block_m: int = 8
    chunk_tokens: int = 256

    def __post_init__(self) -> None:
        domains = {
            "decode_tile_n": (4, 8, 16),
            "prefill_tile_n": (4, 8),
            "prefill_block_m": (2, 4, 8),
            "dense_prefill_block_m": (2, 4, 8),
            "chunk_tokens": (128, 256, 512, 1024),
        }
        for name, allowed in domains.items():
            value = getattr(self, name)
            if type(value) is not int or value not in allowed:
                raise ValueError(f"rdna_oq_affine.{name} must be one of {allowed}")

    @classmethod
    def from_config(cls, config: Any = _OMITTED):
        """Read the rdna_oq_affine object, not the outer additional_config."""
        if config is _OMITTED:
            return cls()
        if not isinstance(config, Mapping):
            raise ValueError("rdna_oq_affine must be an object")
        if set(config) != set(cls.__dataclass_fields__):
            raise ValueError("rdna_oq_affine requires exactly the five tuning fields")
        return cls(**config)
