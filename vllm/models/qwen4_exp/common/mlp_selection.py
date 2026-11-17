# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Provisional dense-vs-MoE selection for the Qwen4 decoder layer.

Qwen3.8-Flash-Next (``qwen4_exp``) is a sparse MoE model and must keep
building ``Qwen4ExpSparseMoeBlock``. The leaked Qwen4-27B kernels are dense
and target no expert GEMM, so a released dense checkpoint needs a plain MLP
in the same decoder layer.

The rule is deliberately small and provisional:

- an explicit ``dense_mlp`` config field wins when present. No released
  checkpoint defines it; it exists so a future config can be unambiguous.
- otherwise a config with no routed experts (``num_experts`` missing or 0)
  is dense. Qwen3.8-27B, the backbone the 27B leak keeps, has no
  ``num_experts``; Flash-Next has 512.
- ``mlp_only_layers`` is not consulted: no Qwen config defines it, and
  guessing a DeepSeek-style field would silently change Flash-Next.

This module is stdlib-only so the rule can be unit-tested without torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def uses_dense_mlp(config: Any) -> bool:
    """Whether a Qwen4-family config needs a dense MLP instead of sparse MoE."""
    explicit = _get(config, "dense_mlp")
    if explicit is not None:
        return bool(explicit)
    return not _get(config, "num_experts", 0)
