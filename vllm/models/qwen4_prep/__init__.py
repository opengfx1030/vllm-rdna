# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in prep package for the Qwen4 decode monokernel on gfx1030.

Nothing here is imported by the serving path. ``mono_plan`` derives the
per-layer stage graph from a model config; the model registry has no entry
for it yet because Qwen4 is unreleased. The architecture gate below is the
only "registration": it matches a Qwen4 name and never the Qwen4Exp
(Flash-Next) family that the live monokernel serves today.
"""

# Qwen4Exp* is the Flash-Next family (``vllm/models/qwen4_exp/``); its
# monokernel behavior must not change. New architecture strings that start
# with ``Qwen4`` and are not ``Qwen4Exp`` belong to the Qwen4 line.
_QWEN4_EXCLUDED_PREFIX = "Qwen4Exp"


def is_qwen4_architecture(name: str) -> bool:
    """Whether an HF architecture string belongs to the Qwen4 line.

    Exact-name knowledge waits for the release; the rule is a prefix match
    that explicitly excludes the Flash-Next family, so an unknown
    ``Qwen4*`` class can be planned but ``Qwen4Exp*`` never is.
    """
    return name.startswith("Qwen4") and not name.startswith(_QWEN4_EXCLUDED_PREFIX)
