# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling-2-Mini (thinkingmachines, ``inkling2_loop``), RDNA2/ROCm only."""

__all__ = ["Inkling2LoopForConditionalGeneration"]


def __getattr__(name: str):
    if name == "Inkling2LoopForConditionalGeneration":
        from .amd.model import Inkling2LoopForConditionalGeneration

        return Inkling2LoopForConditionalGeneration
    raise AttributeError(name)
