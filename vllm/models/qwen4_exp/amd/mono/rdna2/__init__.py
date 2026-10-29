# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp MonoKernel on gfx1030 (RDNA2): the parts behind ``--monokernel``.

``k1`` attention-side HC preamble, ``k2`` persistent FFN block, ``gdn`` fused
GDN decode core, ``qsa`` QSA decode; ``layout`` holds the shared weight views,
workspace and all-reduce attach, ``debug`` the in-serve timing. The switch,
the deployment checks and the hook installation are in
``vllm/models/qwen4_exp/amd/mono_decode_rdna2.py``; see README.md.
"""

from dataclasses import dataclass, field, fields


@dataclass
class Options:
    """``--additional-config '{"rocm_mono_decode": {...}}'`` (only read with
    ``--monokernel``). The defaults are the validated all-on configuration."""

    k1: bool = True  # attention-side HC preamble on the persistent kernel
    k1_max_rows: int = 24
    k2: bool = True  # decode FFN block in one persistent launch
    k2_max_rows: int = 8  # width gate: K2 up to this many rows, modules above
    mtp_drafter: bool = False  # opt-in: use K1/K2 for supported MTP MoE layers
    mtp_k1_max_rows: int = 24
    mtp_k2_max_rows: int = 24
    k2_modes: str = "full,piecewise,eager"
    k2_num_wg: int = 0  # 0: residency cap
    k2_sleep: int = 0  # tuning: barrier poll backoff
    k2_flags: int = 0  # tuning flags (workspace tail)
    k2_ar: bool = False  # FFN all-reduce inside K2 on the rdna_ar buffers
    gdn: bool = True  # fused GDN decode core
    gdn_max_rows: int = 64
    gdn_graph: bool = False  # record the fused GDN core in FULL graphs
    qsa: bool = True  # QSA decode: bounded indexer, HIP sparse decode, fusions
    timing: bool = False  # in-serve timing marks / per-width stats (debug)
    modules: bool = False  # timing reference: today's modules under the marks
    hw_queues: int = 4  # GPU_MAX_HW_QUEUES floor for the workers (0: leave)
    expert_cache_slots: int = 0  # per-layer GPU expert capacity; 0: resident
    expert_cache_slot_plan: dict = field(default_factory=dict)
    expert_cache_policy: str = "lfu_lru"
    expert_cache_decay: int = 128
    kv_cache_live_offload: bool = False
    kv_cache_live_offload_watermark: float = 0.9
    kv_cache_live_offload_pause_target_s: float = 10.0
    expert_cache_overlap: bool = False  # known misses only; extra passes may cost more
    expert_cache_copy: str = "tensor"  # opt-in native batched HIP submissions

    @classmethod
    def from_dict(cls, extra: dict) -> "Options":
        known = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(extra) - set(known))
        if unknown:
            raise ValueError(
                f"--monokernel: unknown rocm_mono_decode keys {unknown} "
                f"(known: {sorted(known)})"
            )
        opts = cls()
        for k, v in extra.items():
            if k in ("expert_cache_slots", "expert_cache_decay") and (
                isinstance(v, bool) or not isinstance(v, int)
            ):
                raise ValueError(f"--monokernel: {k} must be an integer")
            if k == "expert_cache_slot_plan" and not isinstance(v, dict):
                raise ValueError("expert_cache_slot_plan must be a dictionary")
            if k in (
                "expert_cache_overlap",
                "kv_cache_live_offload",
            ) and not isinstance(v, bool):
                raise ValueError(f"{k} must be a boolean")
            if k in (
                "kv_cache_live_offload_watermark",
                "kv_cache_live_offload_pause_target_s",
            ) and (isinstance(v, bool) or not isinstance(v, (int, float))):
                raise ValueError(f"{k} must be a number")
            cur = getattr(opts, k)
            setattr(opts, k, type(cur)(v) if not isinstance(cur, bool) else bool(v))
        if not 0 <= opts.kv_cache_live_offload_watermark <= 1:
            raise ValueError("kv_cache_live_offload_watermark must be in [0, 1]")
        if not 0 < opts.kv_cache_live_offload_pause_target_s <= 3600:
            raise ValueError(
                "kv_cache_live_offload_pause_target_s must be in (0, 3600]"
            )
        if opts.expert_cache_slot_plan:
            from vllm.model_executor.layers.fused_moe.rdna_expert_cache_config import (
                validate_slot_plan,
            )

            validate_slot_plan(opts.expert_cache_slot_plan)
        if opts.expert_cache_slot_plan and not opts.expert_cache_slots:
            raise ValueError("expert_cache_slot_plan requires expert_cache_slots > 0")
        if not 0 <= opts.expert_cache_slots <= 256:
            raise ValueError("--monokernel: expert_cache_slots must be in [0, 256]")
        if opts.expert_cache_policy not in ("lfu_lru", "lru"):
            raise ValueError("--monokernel: expert_cache_policy must be lfu_lru or lru")
        if opts.expert_cache_decay <= 0:
            raise ValueError("--monokernel: expert_cache_decay must be positive")
        if opts.expert_cache_copy not in ("tensor", "native"):
            raise ValueError("--monokernel: expert_cache_copy must be tensor or native")
        return opts

    def assign(self, other: "Options") -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(other, f.name))


# The active options of this process (set by mono_decode_rdna2.install).
OPTS = Options()
