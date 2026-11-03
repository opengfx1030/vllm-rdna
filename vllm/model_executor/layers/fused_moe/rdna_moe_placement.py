# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in static expert-to-rank placement for expert-parallel MoE (RDNA2).

``VLLM_RDNA_MOE_PLACEMENT=<placement.json>`` replaces the uniform linear
expert placement with a per-layer expert-to-rank assignment produced by
``tools/rdna/prof/moe_placement.py --out`` from a routing-histogram capture
(``VLLM_RDNA_MOE_HIST_DIR``). The file's ``layers["<decoder layer index>"][r]``
list names the GLOBAL experts that run on EP rank ``r``. Only the
``expert_map`` changes: the weight loader places each global expert into the
local slot the map gives it, so neither the checkpoint nor the kernels change.

The file format (see ``moe_placement.py``)::

    {"version": 1, "num_experts": 512, "ep_size": 4, "method": "lpt+swap",
     "layers": {"3": [[...128 experts...], ..., [...]]}}

A layer that is missing from the file, and any rank whose list length is not
the uniform ``num_experts // ep_size``, keep the linear placement, so a
partial capture still boots. Default off (``expert_map`` from the strategy).
"""

import json
import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.rdna_routing_hist import layer_index

logger = init_logger(__name__)

PLACEMENT_ENV = "VLLM_RDNA_MOE_PLACEMENT"

# path -> parsed document; a placement file is read once per process.
_CACHE: dict[str, dict[str, Any]] = {}
# path -> a coverage warning was already logged.
_WARNED: dict[str, bool] = {}


def load_placement(path: str) -> dict[str, Any]:
    """Parse (once) and validate a placement file."""
    cached = _CACHE.get(path)
    if cached is not None:
        return cached
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or "layers" not in doc:
        raise ValueError(f"{path}: expected a JSON object with a 'layers' key")
    if doc.get("version") != 1:
        raise ValueError(
            f"{path}: unsupported placement version {doc.get('version')!r}"
        )
    for key, placement in doc["layers"].items():
        if not isinstance(placement, list) or not all(
            isinstance(rank_experts, list) for rank_experts in placement
        ):
            raise ValueError(f"{path}: layers[{key!r}] is not a list of rank lists")
        if len(placement) != doc["ep_size"]:
            raise ValueError(
                f"{path}: layers[{key!r}] has {len(placement)} ranks, "
                f"expected {doc['ep_size']}"
            )
    _CACHE[path] = doc
    return doc


def local_experts_for_layer(
    doc: dict[str, Any],
    layer_name: str,
    ep_rank: int,
    global_num_experts: int,
) -> list[int] | None:
    """Global experts that run on ``ep_rank`` for ``layer_name``, or None.

    None means "keep the linear placement": the layer is not in the file, the
    file was made for another expert count, or this rank's list is not the
    uniform size (a non-uniform placement would change the local tensor shape
    and is not supported).
    """
    if doc.get("num_experts") != global_num_experts:
        return None
    capacity = global_num_experts // doc["ep_size"]
    entry = doc["layers"].get(str(layer_index(layer_name)))
    if entry is None or not 0 <= ep_rank < len(entry):
        return None
    local = entry[ep_rank]
    if len(local) != capacity:
        return None
    return local


def experts_to_expert_map(
    local_experts: list[int],
    global_num_experts: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the global-to-local map for one rank (``-1`` = not on this rank).

    Local slot ``i`` holds ``local_experts[i]``; the slots are numbered in the
    order of the list, which is the order the routing capture was scored in.
    """
    if len(set(local_experts)) != len(local_experts):
        raise ValueError("placement repeats an expert id")
    if any(not 0 <= expert < global_num_experts for expert in local_experts):
        raise ValueError(
            f"placement has an expert id outside 0..{global_num_experts - 1}"
        )
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32, device=device)
    expert_map[torch.tensor(local_experts, dtype=torch.long, device=device)] = (
        torch.arange(len(local_experts), dtype=torch.int32, device=device)
    )
    return expert_map


def placement_from_env(
    layer_name: str,
    ep_rank: int,
    global_num_experts: int,
) -> list[int] | None:
    """The env-selected placement for one layer, or None when off/unusable."""
    path = os.environ.get(PLACEMENT_ENV, "")
    if not path:
        return None
    local = local_experts_for_layer(
        load_placement(path), layer_name, ep_rank, global_num_experts
    )
    if local is None and not _WARNED.get(path):
        _WARNED[path] = True
        logger.warning(
            "RDNA MoE placement %s: layer %r (rank %d) is not covered; "
            "keeping the linear placement for it",
            path,
            layer_name,
            ep_rank,
        )
    return local
