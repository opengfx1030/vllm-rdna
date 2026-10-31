# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup-only expert capacity selection; no device allocations or imports."""

import regex as re


def validate_slot_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("expert_cache_slot_plan must be a rank-to-layer dictionary")
    for rank, layers in plan.items():
        if not isinstance(rank, str) or not rank.isdecimal() or str(int(rank)) != rank:
            raise ValueError(
                "expert_cache_slot_plan rank keys must be canonical integers"
            )
        if not isinstance(layers, dict) or not layers:
            raise ValueError("expert_cache_slot_plan ranks require layer capacities")
        for layer, slots in layers.items():
            if (
                not isinstance(layer, str)
                or not layer.isdecimal()
                or str(int(layer)) != layer
            ):
                raise ValueError(
                    "expert_cache_slot_plan layer keys must be canonical integers"
                )
            if type(slots) is not int or not 1 <= slots <= 256:
                raise ValueError(
                    "expert_cache_slot_plan capacities must be integers in [1, 256]"
                )


def select_slots(default, plan, rank, layer_name):
    """A supplied plan is complete: never silently use a default for missing layers."""
    validate_slot_plan(plan)
    if not plan:
        return default
    matches = re.findall(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if len(matches) != 1:
        raise ValueError(f"cannot identify expert-cache layer: {layer_name}")
    key = str(int(matches[0]))
    try:
        return plan[str(rank)][key]
    except KeyError as exc:
        raise ValueError(
            f"expert_cache_slot_plan missing rank {rank} layer {key}"
        ) from exc
