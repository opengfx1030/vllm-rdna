# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-Next geometry, byte costs and the MoE time model of the EPLB study.

Every constant carries its source. The time model has two measured anchors:

* decode (bandwidth bound): one touched expert costs ``expert_us`` whatever its
  row count, the weight read of 2.53 MB at about 380 GB/s (``hetero_moe``
  constants for the V620, ``--expert-us 6.7 --expert-fixed-us 3.1``). Check: at
  c=1 MTP2 (3 rows) the model gives 58 us per layer against 52 us measured in the
  decode-step trace (``docs/rdna2/async-stall-profile-2026-11-02.md``);
* prefill (compute bound): ``pair_us`` per routed (row, expert) pair, derived from
  the 2026-11-03 rocprof table of a 2048-token step (MoE int4 GEMM tile 4,
  208.6 ms / 48 layers = 4.35 ms per layer, ~5,120 local pairs on 128 touched
  experts: (4.35 ms - 128 x 6.7 us) / 5,120 = 0.68 us). The shipped W4A8 v2 tile-8
  kernel is faster (1.75x at 2048 tokens), so absolute prefill times are an upper
  bound; only ratios between placements are used.

A rank's time in a layer is ``fixed_us + sum over its touched experts of
max(expert_us, pair_us * rows_of_the_expert)``: the same roofline for both phases.
The layer finishes with its slowest rank (the next collective waits for it).
"""

import json
import os
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BYTES_TABLE = os.path.join(
    HERE, os.pardir, "hetero_moe", "data", "flashnext_w4a16_bytes.json"
)
GIB = 1 << 30


@dataclass(frozen=True)
class Geometry:
    """Qwen3.8-Flash-Next W4A16 on a TP4/EP4 V620 island."""

    layers: int = 48
    experts: int = 512
    top_k: int = 10
    # packed int4 2,457,600 + fp16 scales 76,800 (symmetric, group 128)
    expert_bytes: int = 2_534_400
    # constant synthetic qzeros (0x77777777) the RDNA2 layout keeps as parameters
    resident_extra_bytes: int = 19_200
    # the MTP layer's 512 BF16 experts (kind "mtp", layer index 48)
    mtp_expert_bytes: int = 9_830_400
    # measured, TP4 flashnext-mtp2: 266,945 KV tokens in a 3.75 GiB KV budget
    kv_tokens: int = 266_945
    kv_bytes: int = 4_026_531_840

    @property
    def param_expert_bytes(self) -> int:
        """Bytes one expert occupies in the parameters (what a slot costs)."""
        return self.expert_bytes + self.resident_extra_bytes

    @property
    def slot_bytes_all_layers(self) -> int:
        """One extra expert slot on every layer, per rank."""
        return self.layers * self.param_expert_bytes

    def redundancy_bytes(self, extra_slots_per_rank: int) -> int:
        """Per-rank bytes of ``extra_slots_per_rank`` slots on every layer."""
        return int(extra_slots_per_rank) * self.slot_bytes_all_layers

    def kv_tokens_lost(self, nbytes: float) -> float:
        """KV capacity (tokens) that ``nbytes`` of weights take from the budget."""
        return nbytes * self.kv_tokens / self.kv_bytes

    def kv_fraction_lost(self, nbytes: float) -> float:
        return nbytes / self.kv_bytes

    def local_experts(self, ranks: int) -> int:
        if self.experts % ranks:
            raise ValueError(f"{self.experts} experts do not divide over {ranks} ranks")
        return self.experts // ranks

    def moved_bytes(self, experts_moved: int, layers: int | None = None) -> int:
        """Bytes a rebalance transfers: payload per moved (expert, layer)."""
        return int(experts_moved) * self.expert_bytes * (layers or 1)


@dataclass(frozen=True)
class CostModel:
    """Time model of one MoE layer on one rank (see the module docstring)."""

    expert_us: float = 6.7
    fixed_us: float = 3.1
    pair_us: float = 0.68

    def rank_us(self, rows_per_expert: np.ndarray) -> np.ndarray:
        """Per-expert time ``max(expert_us, pair_us * rows)``, zero for idle experts."""
        rows = np.asarray(rows_per_expert, dtype=np.float64)
        return np.where(rows > 0, np.maximum(self.expert_us, self.pair_us * rows), 0.0)


FLASHNEXT = Geometry()
DEFAULT_COST = CostModel()


def from_bytes_table(path: str = BYTES_TABLE) -> Geometry:
    """Geometry read from a ``hetero-moe-model-bytes/1`` table (checkpoint headers)."""
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    return Geometry(
        layers=len(range(table["layer_ids"][0], table["layer_ids"][1] + 1)),
        experts=table["num_experts"],
        expert_bytes=table["expert_bytes"],
        resident_extra_bytes=table["expert_resident_extra_bytes_rdna2"],
        mtp_expert_bytes=table["mtp"]["expert_bytes"],
    )
