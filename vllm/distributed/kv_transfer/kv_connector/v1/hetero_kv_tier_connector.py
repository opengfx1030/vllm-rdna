# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV connector that spills full-attention blocks to the cold pool.

The policy object is :class:`HeteroKVTier`. ``start_load_kv`` copies
blocks back before attention and does not leave a remote view.
``get_num_new_matched_tokens`` returns no prefix hits: with automatic
prefix caching off, the tier is a preemption swap and a long-context
spill, not a prefix cache.

The in-process tier is what the CPU tests exercise. This connector does
not issue HIP copies onto a V620. Select it with
``--kv-transfer-config`` only when ``VLLM_HETERO_MOE_KV_TIER=1``.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.hetero_moe.kv_tier import REFUSED_GROUPS, HeteroKVTier
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)


@dataclass
class HeteroKVMetadata(KVConnectorMetadata):
    """Block hashes to copy back before this step's attention."""

    restore: list = field(default_factory=list)
    spill: list = field(default_factory=list)


def group_kind_from_layer_name(layer_name: str) -> str:
    """Classify a layer name for the spill policy.

    Args:
        layer_name: Attention layer name.

    Returns:
        ``full_attention``, a refused group, or a split QSA part.
    """
    lowered = layer_name.lower()
    for part in ("gdn", "ple", "mamba", "kda", "recurrent", "ssm"):
        if part in lowered:
            return part
    if "qsa" in lowered and "compressed" in lowered:
        return "qsa_compressed_key"
    if "qsa" in lowered and "index" in lowered:
        return "qsa_indexer"
    if "qsa" in lowered:
        return "qsa_main_kv"
    return "full_attention"


class HeteroKVTierConnector(KVConnectorBase_V1):
    """Scheduler and worker facade over :class:`HeteroKVTier`."""

    def __init__(self, vllm_config, role, kv_cache_config) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        capacity = int(os.environ.get("VLLM_HETERO_MOE_KV_CAPACITY_BLOCKS", "0"))
        self.tier = HeteroKVTier(capacity)

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """Copy-back runs in Python between graph pieces."""
        return True

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """No prefix hits. APC stays off on this tree."""
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return None

    def build_connector_meta(self, scheduler_output) -> HeteroKVMetadata:
        """Ask the worker to copy back whatever is currently cold."""
        return HeteroKVMetadata(restore=self.tier.cold_hashes(), spill=[])

    def start_load_kv(self, forward_context, **kwargs) -> None:
        """Copy spilled blocks back before attention. No remote view."""
        meta = self._connector_metadata
        if not isinstance(meta, HeteroKVMetadata):
            return
        self.tier.prepare_for_attention(list(meta.restore))

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Ignore refused groups. Do not remote-read the others."""
        kind = group_kind_from_layer_name(layer_name)
        if kind in REFUSED_GROUPS or self.tier.is_refused(kind):
            return None
        return None

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        """Do not store refused groups. Spill is an explicit metadata list."""
        kind = group_kind_from_layer_name(layer_name)
        if self.tier.is_refused(kind):
            return None
        return None

    def wait_for_save(self) -> None:
        return None

    def handle_preemptions(self, kv_connector_metadata) -> None:
        """Spill listed blocks before the paged buffer reuses them."""
        if not isinstance(kv_connector_metadata, HeteroKVMetadata):
            return
        for item in kv_connector_metadata.spill:
            block_hash, payload, group = item
            if group == "qsa_heap":
                main_hash, comp_hash = block_hash
                main_payload, comp_payload = payload
                self.tier.evict_qsa_heap(
                    main_hash,
                    main_payload,
                    comp_hash,
                    comp_payload,
                )
            else:
                self.tier.evict(block_hash, payload, group)
