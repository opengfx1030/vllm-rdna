# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The checkpoint byte table behind the Flash-Next geometry of the EPLB study.

``Geometry`` and ``CostModel`` (byte costs and the MoE time model, every constant with
its source) live in ``vllm/model_executor/layers/fused_moe/rdna_eplb_policy.py``, the
pure-numpy policy shared with the online planner; ``from_bytes_table`` rebuilds the
geometry from the checkpoint headers and the tests check it equals ``core.FLASHNEXT``.
"""

import json
import os

from tools.rdna.eplb._core import core

HERE = os.path.dirname(os.path.abspath(__file__))
BYTES_TABLE = os.path.join(
    HERE, os.pardir, "hetero_moe", "data", "flashnext_w4a16_bytes.json"
)
GIB = 1 << 30


def from_bytes_table(path: str = BYTES_TABLE) -> core.Geometry:
    """Geometry read from a ``hetero-moe-model-bytes/1`` table (checkpoint headers)."""
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    return core.Geometry(
        layers=len(range(table["layer_ids"][0], table["layer_ids"][1] + 1)),
        experts=table["num_experts"],
        expert_bytes=table["expert_bytes"],
        resident_extra_bytes=table["expert_resident_extra_bytes_rdna2"],
        mtp_expert_bytes=table["mtp"]["expert_bytes"],
    )
