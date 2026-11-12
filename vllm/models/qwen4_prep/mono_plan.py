# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config-to-stage-graph planner for a Qwen4 decode monokernel on gfx1030.

Prep scaffolding for the Qwen4 release. Nothing imports this from the serving
path, and the Qwen4Exp (Flash-Next) monokernel under
``vllm/models/qwen4_exp/amd/mono/rdna2/`` is untouched: the planner only
describes the plan that kernel implements today and the plan a released Qwen4
config would need.

The unit of planning is one decode step of one decoder layer. A layer is a
sequence of parts: the PLE add (a module), the attention-side hyper-connection
preamble (K1), the attention core (the fused GDN launch or the QSA indexer and
sparse attention), the attention all-reduce, the FFN block (K2 or the vLLM
modules) and the FFN all-reduce. ``plan_for`` derives that sequence, the row
caps, the persistent grid and a roofline traffic estimate from a model config
mapping; ``flash_next_plan`` reproduces the shipped Flash-Next plan exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Graph modes the persistent launches are allowed in today (k2_modes).
GRAPH_MODES = ("full", "piecewise", "eager")


@dataclass(frozen=True)
class Limits:
    """gfx1030 / V620 budgets and the live monokernel's build constants.

    Every field is taken from code or measured on the V620; the tests
    cross-check them against the sources so the planner cannot drift.
    """

    threads: int = 256  # kThreads in all three kernels
    waves: int = 8  # threads / 32, wave32
    max_rows_k1: int = 24  # layout.MAX_ROWS
    max_rows_k2: int = 24  # mono_k2_rdna2.cu kMaxM
    k2_gate_rows: int = 8  # Options.k2_max_rows
    max_rows_gdn: int = 64  # mono_gdn_rdna2.cu kCapRows
    min_rows_qsa: int = 3  # mono/rdna2/qsa.py MIN_ROWS
    max_rows_qsa: int = 64  # mono/rdna2/qsa.py _hip_ok upper bound
    max_group_qsa: int = 8  # mono_qsa_decode_rdna2.cu kMaxG
    qsa_head_dim: int = 256  # mono_qsa_decode_rdna2.cu kHD
    max_splits_qsa: int = 64  # mono_qsa_decode_rdna2.cu kMaxS
    max_topk: int = 10  # kTopK
    max_local_experts: int = 128  # kLocalE (resident kernel)
    max_local_experts_offload: int = 256  # Stage-1/2/3 specializations
    hc_count: int = 4  # kHC, layout.py
    max_lowrank: int = 320  # "lora rank <= 320"
    lds_bytes: int = 64 * 1024  # per workgroup (WGP mode 2 x 64 KiB)
    vgprs_per_lane: int = 128  # K2 build for two WGs per CU
    cus: int = 72  # 8 x V620, 2 x 36 WGPs
    wg_per_cu: int = 2  # measured grid cap 144
    hw_queues: int = 4  # GPU_MAX_HW_QUEUES floor with the persistent kernel
    bandwidth_gbs: float = 512.0  # V620 GDDR6, roofline denominator
    quant_bits: int = 4  # symmetric W4A16 routed experts
    group_size: int = 128  # GPTQ group for the expert scales

    def residency_cap(self, vgprs: int, lds_bytes: int) -> int:
        """Port of the launch-time grid cap in ``mono_launch``.

        ``hipOccupancy*`` under-reports by 2x on gfx10, so the kernel derives
        the cap from the code object: waves per SIMD = 1024 / VGPRs (wave32),
        2 SIMDs per CU, 64 KiB LDS per CU, CUs = 2 x the reported WGP count.
        """
        vgpr = max(8, (vgprs + 7) // 8 * 8)
        by_waves = 2 * min(16, 1024 // vgpr) // self.waves
        by_lds = self.lds_bytes // lds_bytes if lds_bytes else 8
        return max(1, min(by_waves, by_lds) * 2 * (self.cus // 2))


def _shard(total: int, tp: int) -> int:
    """Per-rank width of a tensor-parallel shard, replicating short heads."""
    if total < tp:
        return total  # KV heads are replicated when there are fewer than ranks
    if total % tp:
        raise ValueError(f"cannot shard {total} over TP={tp}")
    return total // tp


@dataclass(frozen=True)
class Geometry:
    """Model geometry the plan needs, independent of weight dtypes."""

    hidden: int
    num_layers: int
    layer_types: tuple[str, ...]
    hc_count: int
    hc_lowrank: int
    q_heads: int
    kv_heads: int
    head_dim: int
    lin_k_heads: int
    lin_v_heads: int
    lin_k_dim: int
    lin_v_dim: int
    conv_width: int
    experts: int
    top_k: int
    moe_inter: int
    shared_inter: int
    dense_inter: int
    indexer_budget: int
    indexer_compress: int
    indexer_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    ple_layers: tuple[int, ...]
    mtp_layers: int
    mtp_layer_types: tuple[str, ...]
    tp: int

    @property
    def dense(self) -> bool:
        """Whether the FFN is dense (no routed experts)."""
        return self.experts == 0

    @property
    def local_experts(self) -> int:
        """Routed experts this rank owns under expert parallelism."""
        if self.dense:
            return 0
        return _shard(self.experts, self.tp)

    @property
    def hc_width(self) -> int:
        """Width of the multi-stream residual state (``hc_count * H``)."""
        return self.hidden * self.hc_count

    @property
    def hc_down_rows(self) -> int:
        """Rows of the HC down/inject weight: low-rank plus the inject block."""
        return self.hc_lowrank + self.hc_count * self.hc_count

    @property
    def q_heads_local(self) -> int:
        """Query heads this rank owns."""
        return _shard(self.q_heads, self.tp)

    @property
    def kv_heads_local(self) -> int:
        """KV heads this rank owns (replicated when fewer than TP)."""
        return _shard(self.kv_heads, self.tp)

    @property
    def qsa_group(self) -> int:
        """Query heads per KV head; the sparse kernel is built for <= 8."""
        return self.q_heads_local // self.kv_heads_local

    @property
    def lin_k_heads_local(self) -> int:
        """GDN key heads this rank owns."""
        return _shard(self.lin_k_heads, self.tp)

    @property
    def lin_v_heads_local(self) -> int:
        """GDN value heads this rank owns."""
        return _shard(self.lin_v_heads, self.tp)

    @property
    def ple_layer_indices(self) -> tuple[int, ...]:
        """Zero-based indices of the layers that add the PLE n-gram vector."""
        return tuple(i - 1 for i in self.ple_layers)

    @property
    def qsa_layers(self) -> tuple[int, ...]:
        """Indices of the full-attention (QSA) layers."""
        return tuple(i for i, t in enumerate(self.layer_types) if t == "full_attention")

    @classmethod
    def from_hf(cls, cfg: Mapping[str, Any], tp: int) -> Geometry:
        """Build the geometry from an HF-style config mapping.

        Accepts the top-level config or one nesting the text config under
        ``text_config``/``llm_config``. Missing optional fields fall back to
        the values Qwen4Exp uses; a field the plan cannot guess raises.
        """
        if not isinstance(cfg, Mapping):
            raise TypeError("config must be a mapping")
        for key in ("text_config", "llm_config"):
            nested = cfg.get(key)
            if isinstance(nested, Mapping):
                merged = dict(cfg)
                merged.pop(key)
                merged.update(nested)
                cfg = merged
                break
        hidden = int(cfg["hidden_size"])
        num_layers = int(cfg["num_hidden_layers"])
        layer_types = _layer_types(cfg, num_layers)
        q_heads = int(cfg["num_attention_heads"])
        kv_heads = int(cfg.get("num_key_value_heads", q_heads))
        head_dim = int(cfg.get("head_dim") or hidden // q_heads)
        lin_k = int(cfg.get("linear_num_key_heads", 0))
        lin_v = int(cfg.get("linear_num_value_heads", 0))
        lin_k_dim = int(cfg.get("linear_key_head_dim", 128))
        lin_v_dim = int(cfg.get("linear_value_head_dim", 128))
        conv = int(cfg.get("linear_conv_kernel_dim", 4))
        experts = int(cfg.get("num_experts", 0) or 0)
        top_k = int(cfg.get("num_experts_per_tok", 0) or 0)
        moe_inter = int(cfg.get("moe_intermediate_size", 0) or 0)
        shared = int(cfg.get("shared_expert_intermediate_size", 0) or 0)
        dense_inter = int(cfg.get("intermediate_size", 0) or 0)
        budget = cfg.get("indexer_budget")
        mtp_raw = cfg.get("mtp")
        mtp: Mapping[str, Any] = mtp_raw if isinstance(mtp_raw, Mapping) else {}
        mtp_layers = int(mtp.get("num_hidden_layers", 0) or 0)
        mtp_types = tuple(str(t) for t in mtp.get("layer_types", ()))
        if mtp_layers and not mtp_types:
            mtp_types = ("full_attention",) * mtp_layers
        return cls(
            hidden=hidden,
            num_layers=num_layers,
            layer_types=layer_types,
            hc_count=int(cfg.get("hc_count", 4)),
            hc_lowrank=int(cfg.get("hc_lowrank", 320)),
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            lin_k_heads=lin_k,
            lin_v_heads=lin_v,
            lin_k_dim=lin_k_dim,
            lin_v_dim=lin_v_dim,
            conv_width=conv,
            experts=experts,
            top_k=top_k,
            moe_inter=moe_inter,
            shared_inter=shared,
            dense_inter=dense_inter,
            indexer_budget=int(budget or 0),
            indexer_compress=int(cfg.get("indexer_compress_ratio", 0) or 0),
            indexer_heads=int(cfg.get("indexer_n_heads", 0) or 0),
            indexer_kv_heads=int(cfg.get("indexer_kv_heads", 1) or 1),
            indexer_head_dim=int(cfg.get("indexer_head_dim", 128) or 128),
            ple_layers=tuple(int(i) for i in cfg.get("ple_layer_ids", ()) or ()),
            mtp_layers=mtp_layers,
            mtp_layer_types=mtp_types,
            tp=int(tp),
        )


def _layer_types(cfg: Mapping[str, Any], num_layers: int) -> tuple[str, ...]:
    """Normalise ``layer_types`` or expand ``full_attention_interval``."""
    raw = cfg.get("layer_types")
    if raw:
        types = tuple(str(t) for t in raw)
        if len(types) != num_layers:
            raise ValueError(
                f"layer_types has {len(types)} entries for {num_layers} layers"
            )
        for t in types:
            if t not in ("linear_attention", "full_attention"):
                raise ValueError(f"unknown layer type {t!r}")
        return types
    interval = int(
        cfg.get("full_attention_interval") or cfg.get("full_attn_interval") or 0
    )
    if interval <= 0:
        raise ValueError(
            "config needs layer_types or full_attention_interval "
            "(a Qwen4 hybrid pattern is not guessable)"
        )
    pattern = ("linear_attention",) * (interval - 1) + ("full_attention",)
    return (pattern * ((num_layers + interval - 1) // interval))[:num_layers]


@dataclass(frozen=True)
class Stage:
    """One part of a decode step for one layer.

    ``rows`` is the row cap the part is built for, ``engine`` names where the
    code lives (``mono`` persistent launch, ``hip`` one-shot kernel, ``triton``
    fused Triton launch, ``module`` vLLM module, ``ar`` all-reduce), and
    ``replaces`` is how many launches the part removes.
    """

    part: str
    rows: int
    engine: str
    launches: int = 1
    replaces: int = 1

    def as_dict(self) -> dict[str, Any]:
        """Plain dict for logs, tests and the design doc tables."""
        return {
            "part": self.part,
            "rows": self.rows,
            "engine": self.engine,
            "launches": self.launches,
            "replaces": self.replaces,
        }


@dataclass(frozen=True)
class LayerPlan:
    """Stage graph of one decoder layer."""

    index: int
    layer_type: str
    stages: tuple[Stage, ...]

    @property
    def parts(self) -> tuple[str, ...]:
        """Part names in execution order."""
        return tuple(s.part for s in self.stages)

    def as_dict(self) -> dict[str, Any]:
        """Plain dict for logs, tests and the design doc tables."""
        return {
            "index": self.index,
            "layer_type": self.layer_type,
            "stages": [s.as_dict() for s in self.stages],
        }


@dataclass(frozen=True)
class Traffic:
    """Roofline estimate of the bytes one decode step reads per rank.

    ``dense`` covers the sharded dense weights and the replicated HC mixes and
    router; ``expert`` the routed experts the batch touches; ``kv`` the
    selected QSA keys/values and the indexer's compressed-key scoring.
    """

    rows: int
    context: int
    dense_bytes: int
    expert_bytes: int
    kv_bytes: int

    @property
    def total_bytes(self) -> int:
        """Bytes read by one decode step on this rank."""
        return self.dense_bytes + self.expert_bytes + self.kv_bytes

    def floor_ms(self, bandwidth_gbs: float) -> float:
        """Lower bound on step time if DRAM bandwidth were the only limit."""
        return self.total_bytes / (bandwidth_gbs * 1e6)

    def as_dict(self) -> dict[str, Any]:
        """Plain dict for logs, tests and the design doc tables."""
        return {
            "rows": self.rows,
            "context": self.context,
            "dense_bytes": self.dense_bytes,
            "expert_bytes": self.expert_bytes,
            "kv_bytes": self.kv_bytes,
            "total_bytes": self.total_bytes,
        }


def _expert_bytes(geo: Geometry, limits: Limits) -> int:
    """Bytes of one routed expert: packed weights plus scales and zeros."""
    if geo.dense:
        return 0
    i = geo.moe_inter
    rows13 = 2 * i
    w13 = geo.hidden * rows13 * limits.quant_bits // 8
    w2 = i * geo.hidden * limits.quant_bits // 8
    groups_k = geo.hidden // limits.group_size
    groups_i = max(1, i // limits.group_size)
    s13 = groups_k * rows13 * 2
    s2 = groups_i * geo.hidden * 2
    z13 = groups_k * (rows13 // 8) * 4
    z2 = groups_i * (geo.hidden // 8) * 4
    return w13 + w2 + s13 + s2 + z13 + z2


def _touched_experts(local: int, pairs: int) -> float:
    """Expected distinct experts a batch of ``pairs`` token-expert draws hits."""
    if local <= 0 or pairs <= 0:
        return 0.0
    if local == 1:
        return 1.0
    return local * (1.0 - math.exp(pairs * math.log1p(-1.0 / local)))


def estimate_traffic(
    geo: Geometry,
    *,
    rows: int,
    context: int,
    limits: Limits | None = None,
) -> Traffic:
    """Estimate the bytes one decode step reads per rank.

    Roofline, not a prediction: it counts every weight a stage streams and the
    QSA key traffic for ``context`` tokens, assuming the expert union of the
    batch (no reuse across layers). Compute, latency and the all-reduces are
    outside the model.
    """
    lim = limits or Limits()
    dense = 0
    expert = 0
    kv = 0
    n_gdn = sum(1 for t in geo.layer_types if t == "linear_attention")
    n_qsa = geo.num_layers - n_gdn
    # HC mixes are replicated per rank; the inject block is hc_count squared.
    dense += (
        geo.num_layers
        * 2
        * (geo.hc_down_rows * geo.hc_width + geo.hc_width * geo.hc_lowrank)
        * 2
    )
    if geo.dense:
        # Dense FFN, TP-sharded, gate and up then down.
        ffn = (2 * geo.dense_inter * geo.hidden + geo.dense_inter * geo.hidden) * 2
        dense += geo.num_layers * ffn // geo.tp
    else:
        # Router logits: one replicated [E, H] fp16 matrix per layer.
        dense += geo.num_layers * geo.experts * geo.hidden * 2
        # Shared expert: gate and up then down, TP-sharded.
        ise = geo.shared_inter // geo.tp
        dense += geo.num_layers * (2 * ise * geo.hidden + geo.hidden * ise) * 2
        per_expert = _expert_bytes(geo, lim)
        pairs = rows * geo.top_k
        touched = _touched_experts(geo.local_experts, pairs)
        expert += int(geo.num_layers * touched * per_expert)
    # GDN layers: in_proj q|k|v|z and ba, then out_proj, TP-sharded.
    kd, vd = geo.lin_k_dim, geo.lin_v_dim
    qkvz = 2 * geo.lin_k_heads_local * kd + 2 * geo.lin_v_heads_local * vd
    ba = 2 * geo.lin_v_heads_local
    out = geo.lin_v_heads_local * vd
    gdn = (geo.hidden * (qkvz + ba + out)) * 2
    dense += n_gdn * gdn
    # QSA layers: q, k, v and o projections (TP-sharded), plus the indexer.
    q_size = geo.q_heads_local * geo.head_dim
    kv_size = geo.kv_heads_local * geo.head_dim
    proj = geo.hidden * (2 * q_size + 2 * kv_size) * 2
    indexer = geo.hidden * 2 * geo.indexer_heads * geo.indexer_head_dim * 2
    dense += n_qsa * (proj + indexer)
    if n_qsa and geo.indexer_budget and geo.indexer_compress:
        # Selected K and V for every row, then the compressed keys scored.
        selected = min(geo.indexer_budget, max(context, 0))
        kv += n_qsa * rows * selected * geo.kv_heads_local * geo.head_dim * 2 * 2
        scored = max(context, 0) // geo.indexer_compress
        kv += n_qsa * rows * scored * geo.indexer_kv_heads * geo.indexer_head_dim * 2
    return Traffic(
        rows=rows,
        context=context,
        dense_bytes=int(dense),
        expert_bytes=int(expert),
        kv_bytes=int(kv),
    )


@dataclass(frozen=True)
class MonoPlan:
    """The planned decode-step structure of a Qwen4 model on gfx1030."""

    geometry: Geometry
    limits: Limits
    layers: tuple[LayerPlan, ...]
    drafter: tuple[LayerPlan, ...]
    widths: tuple[int, ...]
    row_cap: int
    grid_wg: int
    warnings: tuple[str, ...] = ()
    refusals: tuple[str, ...] = ()
    traffic: dict[int, Traffic] = field(default_factory=dict)

    @property
    def runs(self) -> bool:
        """Whether the monokernel can run at all for this deployment."""
        return not self.refusals

    def layer_parts(self, layer_type: str) -> tuple[str, ...]:
        """The part sequence of the first layer of ``layer_type``.

        Layers of one type share the graph except for the PLE add, which the
        live kernel keys on ``ple_layer_ids``; use ``layers`` for the exact
        per-index graph.
        """
        for layer in self.layers:
            if layer.layer_type == layer_type:
                return layer.parts
        raise KeyError(layer_type)

    def as_dict(self) -> dict[str, Any]:
        """Plain dict for logs, tests and the design doc tables."""
        return {
            "hidden": self.geometry.hidden,
            "tp": self.geometry.tp,
            "layers": self.geometry.num_layers,
            "row_cap": self.row_cap,
            "widths": list(self.widths),
            "grid_wg": self.grid_wg,
            "warnings": list(self.warnings),
            "refusals": list(self.refusals),
            "traffic": {str(rows): t.as_dict() for rows, t in self.traffic.items()},
            "parts": dict.fromkeys(self.layer_parts("linear_attention"), 1)
            if not self.refusals
            else {},
        }


def widths_for(
    spec_tokens: int, ladder: Sequence[int] = (1, 2, 4, 8)
) -> tuple[int, ...]:
    """Capture widths for a speculative depth: ``(k + 1) * ladder``."""
    return tuple((spec_tokens + 1) * int(b) for b in ladder)


def _refusals(geo: Geometry, lim: Limits, rows: int, offload: bool) -> tuple[str, ...]:
    """Deployment-wide reasons the monokernel cannot run (today's checks)."""
    why: list[str] = []
    if geo.hc_count != lim.hc_count:
        why.append(
            f"HC count {geo.hc_count}: K1/K2 are built for {lim.hc_count} streams"
        )
    if geo.hidden % 128:
        why.append(f"hidden {geo.hidden} is not a multiple of 128")
    if geo.hc_lowrank > lim.max_lowrank:
        why.append(f"HC low-rank {geo.hc_lowrank} exceeds {lim.max_lowrank}")
    if geo.layer_types.count("linear_attention") and not geo.lin_k_heads:
        why.append("linear_attention layers without GDN head counts")
    if not geo.dense:
        if geo.top_k != lim.max_topk:
            why.append(f"top-{geo.top_k} routing: K2 is built for top-10")
        if geo.experts % geo.tp:
            why.append(f"{geo.experts} experts do not split over TP={geo.tp}")
        cap = lim.max_local_experts_offload if offload else lim.max_local_experts
        if geo.local_experts > cap:
            why.append(f"{geo.local_experts} local experts exceed the kernel's {cap}")
        if geo.moe_inter % 64:
            why.append(f"moe_intermediate_size {geo.moe_inter} is not a multiple of 64")
    return tuple(why)


def _warnings(
    geo: Geometry,
    lim: Limits,
    *,
    k1_rows: int,
    k2_rows: int,
    gdn_rows: int,
    qsa_rows: int,
) -> tuple[str, ...]:
    """Per-part fallbacks: the monokernel runs, these parts use modules."""
    warn: list[str] = []
    if k1_rows > lim.max_rows_k1:
        warn.append(f"k1 rows {k1_rows} > {lim.max_rows_k1}: modules above the cap")
    if k2_rows > lim.max_rows_k2:
        warn.append(f"k2 rows {k2_rows} > {lim.max_rows_k2}: modules above the cap")
    if gdn_rows > lim.max_rows_gdn:
        warn.append(f"gdn rows {gdn_rows} > {lim.max_rows_gdn}: modules above the cap")
    if qsa_rows > lim.max_rows_qsa or (qsa_rows and qsa_rows < lim.min_rows_qsa):
        warn.append(
            f"qsa rows {qsa_rows} outside [{lim.min_rows_qsa},"
            f" {lim.max_rows_qsa}]: Triton sparse attention"
        )
    if geo.head_dim != lim.qsa_head_dim:
        warn.append(
            f"head_dim {geo.head_dim} != {lim.qsa_head_dim}: "
            "HIP sparse decode does not cover it"
        )
    if geo.qsa_group > lim.max_group_qsa:
        warn.append(
            f"{geo.qsa_group} query heads per KV head > {lim.max_group_qsa}: "
            f"HIP sparse decode does not cover TP={geo.tp}"
        )
    if not geo.indexer_budget or not geo.indexer_compress:
        warn.append("no indexer fields: full attention has no QSA selection")
    return tuple(warn)


def plan_for(
    geo: Geometry,
    *,
    spec_tokens: int = 0,
    ladder: Sequence[int] = (1, 2, 4, 8),
    k1_rows: int | None = None,
    k2_rows: int | None = None,
    gdn_rows: int | None = None,
    offload: bool = False,
    k2_ar: bool = False,
    limits: Limits | None = None,
    context: int = 16384,
) -> MonoPlan:
    """Derive the decode-stage graph, caps and budget for one deployment.

    ``spec_tokens`` is the speculative depth (MTP=k); ``exec_tokens`` is
    ``spec_tokens + 1`` rows per request. The row caps default to the live
    kernel's gates and are raised to the widest captured width only when the
    caller asks for it, which is what the Qwen4 work has to decide.
    """
    lim = limits or Limits()
    widths = widths_for(spec_tokens, ladder)
    row_cap = max(widths)
    k1 = row_cap if k1_rows is None else k1_rows
    k2 = min(row_cap, lim.k2_gate_rows) if k2_rows is None else k2_rows
    gdn = row_cap if gdn_rows is None else gdn_rows
    refusals = _refusals(geo, lim, row_cap, offload)
    warnings = _warnings(
        geo, lim, k1_rows=k1, k2_rows=k2, gdn_rows=gdn, qsa_rows=row_cap
    )
    if refusals:
        return MonoPlan(
            geometry=geo,
            limits=lim,
            layers=(),
            drafter=(),
            widths=widths,
            row_cap=row_cap,
            grid_wg=lim.cus * lim.wg_per_cu,
            warnings=warnings,
            refusals=refusals,
        )
    layers = tuple(
        _layer_plan(geo, lim, i, k1_rows=k1, k2_rows=k2, gdn_rows=gdn, k2_ar=k2_ar)
        for i in range(geo.num_layers)
    )
    drafter = tuple(
        _drafter_plan(geo, lim, i, k1_rows=k1, k2_rows=k2)
        for i in range(geo.mtp_layers)
    )
    traffic = {
        rows: estimate_traffic(geo, rows=rows, context=context, limits=lim)
        for rows in widths
    }
    return MonoPlan(
        geometry=geo,
        limits=lim,
        layers=layers,
        drafter=drafter,
        widths=widths,
        row_cap=row_cap,
        grid_wg=lim.cus * lim.wg_per_cu,
        warnings=warnings,
        refusals=refusals,
        traffic=traffic,
    )


def _layer_plan(
    geo: Geometry,
    lim: Limits,
    index: int,
    *,
    k1_rows: int,
    k2_rows: int,
    gdn_rows: int,
    k2_ar: bool,
) -> LayerPlan:
    """Stage graph of layer ``index`` with the live kernel's part choices."""
    layer_type = geo.layer_types[index]
    rows = min(k1_rows, lim.max_rows_k1)
    stages: list[Stage] = []
    if index in geo.ple_layer_indices:
        # The PLE add runs before the pending HC state is materialized.
        stages.append(Stage("ple", rows, "module", launches=3))
    stages.append(Stage("k1", rows, "mono", replaces=4))
    if layer_type == "linear_attention":
        stages.append(Stage("gdn", min(gdn_rows, lim.max_rows_gdn), "mono", replaces=9))
        stages.append(Stage("attn_ar", rows, "ar"))
    elif layer_type == "full_attention":
        # Fused prepare writes the main K/V and the gate with the indexer q/k.
        stages.append(Stage("qsa_prepare", rows, "hip", replaces=3))
        stages.append(
            Stage("qsa_indexer", rows, "hip", replaces=3)
        )  # norm, pool, 3 stores
        stages.append(Stage("qsa_mqa", rows, "hip"))
        stages.append(Stage("qsa_select", rows, "triton"))
        if (
            geo.indexer_budget
            and geo.head_dim == lim.qsa_head_dim
            and geo.qsa_group <= lim.max_group_qsa
            and lim.min_rows_qsa <= rows <= lim.max_rows_qsa
        ):
            stages.append(Stage("qsa_sparse", rows, "hip", replaces=2))  # split + merge
        else:
            stages.append(Stage("qsa_sparse", rows, "triton"))
        stages.append(Stage("attn_ar", rows, "ar"))
    else:
        raise ValueError(f"unknown layer type {layer_type!r}")
    stages.append(Stage("k2", min(k2_rows, lim.max_rows_k2), "mono", replaces=21))
    if not k2_ar:
        stages.append(Stage("ffn_ar", rows, "ar"))
    return LayerPlan(index=index, layer_type=layer_type, stages=tuple(stages))


def _drafter_plan(
    geo: Geometry, lim: Limits, index: int, *, k1_rows: int, k2_rows: int
) -> LayerPlan:
    """Stage graph of one MTP drafter layer.

    The drafter is a separate one-layer model with its own vLLM config and
    workspace. Today K1/K2 serve it only behind ``mtp_drafter`` and only for
    the dense FP16 route; otherwise every part is a module.
    """
    rows = min(k1_rows, lim.max_rows_k1)
    stages = (
        Stage("k1", rows, "mono", replaces=4),
        Stage("drafter_attn", rows, "module"),
        Stage("attn_ar", rows, "ar"),
        Stage("k2", min(k2_rows, lim.max_rows_k2), "mono", replaces=21),
        Stage("ffn_ar", rows, "ar"),
    )
    return LayerPlan(index=index, layer_type="mtp", stages=stages)


# The published Qwen3.8-Flash-Next config (``model_type: qwen4_exp``), the
# geometry the live monokernel implements. Values from the HF config.json;
# see docs/rdna2/qwen4/monokernel-design.md for the link.
FLASH_NEXT_CONFIG: dict[str, Any] = {
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "model_type": "qwen4_exp",
    "hidden_size": 2560,
    "num_hidden_layers": 48,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    * 12,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "hc_count": 4,
    "hc_lowrank": 320,
    "ple_layer_ids": [2],
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 128,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "mtp": {"num_hidden_layers": 1, "layer_types": ["full_attention"]},
}

# Qwen4-27B as reported from an Intel XPU kernel PR: dense, H=5120, HC 4/320,
# GDN 16 key / 48 value heads d128, 24 Q / 4 KV heads d256, FFN 14336.
# [UNCONFIRMED] -- no weights, config or model card exist; the layer count is
# a fit to ~27 B parameters, not a published field.
QWEN4_27B_LEAK: dict[str, Any] = {
    "architectures": ["Qwen4ForCausalLM"],
    "model_type": "qwen4",
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    * 16,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 0,
    "intermediate_size": 14336,
    "hc_count": 4,
    "hc_lowrank": 320,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 128,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "mtp": {"num_hidden_layers": 1, "layer_types": ["full_attention"]},
}


def flash_next_geometry(tp: int = 4) -> Geometry:
    """Geometry of the shipped Flash-Next checkpoint at ``tp``."""
    return Geometry.from_hf(FLASH_NEXT_CONFIG, tp)


def flash_next_plan(
    tp: int = 4,
    *,
    spec_tokens: int = 2,
    limits: Limits | None = None,
    context: int = 16384,
) -> MonoPlan:
    """The plan the live monokernel runs for Flash-Next.

    Reproduces ``Options`` defaults: K1 <= 24 rows, K2 <= 8 rows, GDN <= 64
    rows, QSA from 3 rows, no in-kernel FFN all-reduce.
    """
    return plan_for(
        flash_next_geometry(tp),
        spec_tokens=spec_tokens,
        offload=False,
        k2_ar=False,
        limits=limits,
        context=context,
    )
