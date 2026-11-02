# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in RidgeFill draft. Off unless ``VLLM_RIDGEFILL=1``.

The math is an original implementation of the closed-form fit in Johnston
and Smith, RidgeFill (OpenReview ymLpDkJIGs). No file from ridgefill-data,
llama.cpp-ridgefill, or the tcclaviger image is copied here.
"""

from vllm.model_executor.layers.ridgefill.config import enabled

__all__ = ["enabled"]
