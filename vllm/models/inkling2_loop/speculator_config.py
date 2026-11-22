# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculator config of the Inkling-2 DSpark draft.

Kept apart from ``amd/dspark.py``: config modules are imported in every process and
must not import the speculator, which is resolved by name on the GPU worker only.
"""

from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig


class Inkling2DSparkSpeculatorConfig(DSparkSpeculatorConfig):
    speculator_class = "vllm.models.inkling2_loop.amd.dspark.Inkling2DSparkSpeculator"
