# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Algorithm: LeapQuant (Pan et al., arXiv 2609.38166). This script is original
# code written for this fork; it only calls the KV-cache planner.
"""KV-cache capacity of LeapQuant GDN state on the real vLLM planner (CPU only).

It drives the same functions as the engine (platform block alignment,
``get_global_kv_cache_groups``, ``get_kv_cache_configs`` and the capacity log
formula) through the harness of ``tests/v1/core/test_leapquant_page_packing.py``
and swaps the model geometry. No GPU, no weights, no network. Run it from the
repository root in a venv that imports vLLM:

    HIP_VISIBLE_DEVICES= PYTHONPATH=. python tools/rdna/leapquant_capacity.py all
    ... selfcheck | sweep | anatomy | breakdown | ceilings | options
    ... --model 27b            # Qwen3.8-27B AWQ at TP2
    ... --kv-bytes 6000000000  # another KV budget per rank
    ... --json out.json        # every number of the run

Every row is one planner run. "As served" rows go through the real
``Platform._align_hybrid_block_size``. "What-if" rows replace only that
function (``bypass_platform``), so a block the platform would raise stays put;
``selfcheck`` shows the replacement equals the real function wherever the
platform keeps the block. Other what-ifs patch one planner formula while a
capacity is read and say so in their header. Nothing here changes the engine.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import pathlib
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from unittest import mock

for _root in (os.getcwd(), str(pathlib.Path(__file__).resolve().parents[2])):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import torch  # noqa: E402
from transformers import OPTConfig  # noqa: E402

import tests.v1.core.test_leapquant_page_packing as H  # noqa: E402
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops import leapquant_gdn  # noqa: E402
from vllm.model_executor.models import ModelRegistry  # noqa: E402
from vllm.platforms.interface import Platform  # noqa: E402
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum  # noqa: E402
from vllm.v1.core import kv_cache_planning, kv_cache_utils  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    iter_layer_specs,
)

logging.getLogger("vllm").setLevel(logging.ERROR)

MAX_MODEL_LEN = H.MAX_MODEL_LEN
NUM_SPEC = H.NUM_SPEC
# The recipes' SEQS and MAXBAT; H.make_vllm_config builds the same scheduler.
MAX_NUM_SEQS = 8
MAX_BATCHED = 2048


@dataclass(frozen=True)
class Geometry:
    """One model per TP rank, as the harness needs it."""

    key: str
    title: str
    tp: int
    kv_bytes: int  # kv_cache_memory_bytes of the recipe, per rank
    gdn_layers: int
    attn_layers: int  # target layers; one MTP draft layer is added
    key_heads: int  # global
    value_heads: int  # global
    kv_heads: int  # per rank
    qsa: bool  # compressed-key page and key ring next to each main page
    ple: bool  # one replicated PLE short-conv layer
    recipe_off_block: int  # block of the flag-off recipe
    recipe_lq_block: int  # block of the LeapQuant recipe


FLASHNEXT = Geometry(
    key="flashnext",
    title="Qwen3.8-Flash-Next AWQ-W4A16, TP4, MTP2",
    tp=4,
    kv_bytes=4_026_531_840,
    gdn_layers=36,
    attn_layers=12,
    key_heads=16,
    value_heads=48,
    kv_heads=1,
    qsa=True,
    ple=True,
    recipe_off_block=1024,
    recipe_lq_block=512,
)
QWEN27B = Geometry(
    key="27b",
    title="Qwen3.8-27B AWQ-INT4, TP2, MTP2",
    tp=2,
    kv_bytes=12_000_000_000,
    gdn_layers=48,
    attn_layers=16,
    key_heads=16,
    value_heads=48,
    kv_heads=2,
    qsa=False,
    ple=False,
    recipe_off_block=1024,
    recipe_lq_block=1024,
)
GEOMETRIES = {g.key: g for g in (FLASHNEXT, QWEN27B)}

# (model, block, mode) -> (tokens, where the number comes from)
LOGGED = {
    ("flashnext", 1024, "off"): (266_945, "boot log"),
    ("flashnext", 512, "lq"): (273_456, "boot log"),
    ("flashnext", 640, "pack"): (273_220, "planner, not booted"),
    ("27b", 1024, "off"): (328_658, "boot log"),
    ("27b", 1024, "lq"): (325_019, "boot log"),
    ("27b", 1024, "pack"): (331_129, "planner, not booted"),
    ("27b", 512, "lq"): (334_768, "planner, not booted"),
    ("27b", 512, "pack"): (334_768, "planner, not booted"),
}
# Boot of 2026-11-13 at 512 with 4 state ids per group (before the 5-id bound).
SMOKE_BOOT_512_FOUR_IDS = 275_523

MODES = {
    "off": "flag off (FP32 state)",
    "lq": "LeapQuant #168 (padded pages)",
    "pack": "LeapQuant + packing #170",
}


class Model:
    """``get_mamba_specs_from_config`` of the model, as the platform reads it."""

    def __init__(self, geom: Geometry):
        self.geom = geom

    def gdn_layer(self, vllm_config) -> H._StateLayer:
        g = self.geom
        cache = vllm_config.cache_config
        state_dtype = cache.replayssm_state_dtype
        return H._StateLayer(
            MambaStateShapeCalculator.gated_delta_net_state_shape(
                tp_world_size=g.tp,
                num_k_heads=g.key_heads,
                num_v_heads=g.value_heads,
                head_k_dim=128,
                head_v_dim=128,
                conv_kernel_size=4,
                num_spec=NUM_SPEC,
                replayssm_state_dtype=state_dtype,
            ),
            MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                torch.float16, cache.mamba_cache_dtype, "float32", state_dtype
            ),
            MambaAttentionBackendEnum.GDN_ATTN,
            state_format=(
                leapquant_gdn.leapquant_gdn_layout(128, 128, NUM_SPEC).state_format
                if state_dtype == "leapquant"
                else None
            ),
        )

    def ple_layer(self, vllm_config) -> H._StateLayer:
        return H._StateLayer(
            MambaStateShapeCalculator.short_conv_state_shape(
                tp_world_size=1,
                intermediate_size=2560 * 4,
                conv_kernel=(4 - 1) * 3 + 1,
                num_spec=NUM_SPEC,
            ),
            MambaStateDtypeCalculator.short_conv_state_dtype(
                torch.float16, vllm_config.cache_config.mamba_cache_dtype
            ),
            MambaAttentionBackendEnum.SHORT_CONV,
            tp_replicated=True,
        )

    def get_mamba_specs_from_config(self, vllm_config):
        layers = [self.gdn_layer(vllm_config)]
        if self.geom.ple:
            layers.append(self.ple_layer(vllm_config))
        return tuple(
            replace(layer.spec(vllm_config), block_size=-1, page_size_padded=None)
            for layer in layers
        )


def attention_main_spec(geom: Geometry, block: int) -> FullAttentionSpec:
    return H._RdnaStyleBackend.customize_spec(
        FullAttentionSpec(
            block_size=block,
            num_kv_heads=geom.kv_heads,
            head_size=256,
            head_size_v=256,
            dtype=torch.float16,
        )
    )


def attention_bytes_per_token(geom: Geometry) -> int:
    """Main-KV bytes of one token in one layer: the page the platform pads to."""
    return attention_main_spec(geom, 1).page_size_bytes


def attention_page(geom: Geometry, block: int) -> int:
    return block * attention_bytes_per_token(geom)


def build_specs(
    geom: Geometry,
    vllm_config,
    *,
    ple_joins_gdn: bool = False,
    drop: frozenset[str] = frozenset(),
) -> dict:
    """The per-rank spec dict in the engine's registration order (the key ring,
    then the compressed key, then the main KV of each QSA layer: the boot log
    prints the group block sizes in that order). ``drop`` removes owners for
    the ceilings: ``gdn``, ``ple``, ``ring``, ``compressed``."""
    block = vllm_config.cache_config.block_size
    model = Model(geom)
    gdn = model.gdn_layer(vllm_config).spec(vllm_config)
    ple = model.ple_layer(vllm_config).spec(vllm_config) if geom.ple else None
    main = attention_main_spec(geom, block)
    compressed = MLAAttentionSpec(
        block_size=block,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        tokens_per_state=4,
    )
    ring = CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=128, head_size_v=0, dtype=torch.float16
    )
    specs: dict = {}

    def add_attention(prefix: str, is_draft: bool) -> None:
        def mark(spec):
            return replace(spec, is_draft=True) if is_draft else spec

        if geom.qsa and "ring" not in drop:
            specs[f"{prefix}.indexer.raw_key_cache"] = mark(ring)
        if geom.qsa and "compressed" not in drop:
            specs[f"{prefix}.indexer.compressed_key_cache"] = mark(compressed)
        specs[prefix] = mark(main)

    attn_ids = {3 + 4 * i for i in range(geom.attn_layers)}
    for layer in range(geom.gdn_layers + geom.attn_layers):
        if layer in attn_ids:
            add_attention(f"model.layers.{layer}.self_attn", False)
        elif "gdn" not in drop:
            specs[f"model.layers.{layer}.linear_attn"] = gdn
        if geom.ple and layer == 2 and "ple" not in drop:
            specs["model.layers.2.ple"] = gdn if ple_joins_gdn else ple
    add_attention("mtp.layers.0.self_attn", True)
    return specs


def write_stub_model(directory: str, kv_heads: int) -> None:
    """A stub whose KV page is the main page of one rank (``kv_heads`` x 256)."""
    OPTConfig(
        architectures=["OPTForCausalLM"],
        hidden_size=256 * kv_heads,
        num_attention_heads=kv_heads,
        ffn_dim=256 * kv_heads,
        word_embed_proj_dim=256 * kv_heads,
        num_hidden_layers=1,
        max_position_embeddings=MAX_MODEL_LEN,
    ).save_pretrained(directory)


class Infeasible(Exception):
    """The block cannot hold the state page when the page is padded."""


def bypass_platform(vllm_config, geom: Geometry, *, pack: bool) -> None:
    """``Platform._align_hybrid_block_size`` without the block raise: it pads
    the Mamba page to the attention page like the real function and keeps the
    block as requested. ``selfcheck`` compares it with the real one."""
    cache = vllm_config.cache_config
    cache.mamba_block_size = cache.block_size
    if pack:
        return
    mamba_page = max(
        spec.page_size_bytes
        for spec in Model(geom).get_mamba_specs_from_config(vllm_config)
    )
    attn_page = attention_page(geom, cache.block_size)
    if attn_page < mamba_page:
        raise Infeasible(f"attention page {attn_page} < state page {mamba_page}")
    if attn_page != mamba_page:
        cache.mamba_page_size_padded = attn_page


def make_plan(
    geom: Geometry,
    model_dir: str,
    block: int,
    *,
    leapquant: bool,
    pack: bool,
    platform: bool = True,
    ple_joins_gdn: bool = False,
    drop: frozenset[str] = frozenset(),
) -> H.Plan:
    """Plan the KV cache of one serve. ``platform=False`` keeps ``block``."""
    with H.serving_env(pack=pack):
        vllm_config = H.make_vllm_config(model_dir, block, "BLNHC", leapquant=leapquant)
        if platform:
            with mock.patch.object(
                ModelRegistry,
                "resolve_model_cls",
                lambda *args, **kwargs: (Model(geom), "Model"),
            ):
                Platform._align_hybrid_block_size(vllm_config, H._RdnaStyleBackend)
        else:
            bypass_platform(vllm_config, geom, pack=pack)
        specs = build_specs(geom, vllm_config, ple_joins_gdn=ple_joins_gdn, drop=drop)
        groups = kv_cache_planning.get_global_kv_cache_groups(vllm_config, [specs])
        (config,) = kv_cache_planning.get_kv_cache_configs(
            vllm_config, [specs], [geom.kv_bytes], groups
        )
        return H.Plan(
            block_size=vllm_config.cache_config.block_size,
            leapquant=leapquant,
            pack=pack,
            vllm_config=vllm_config,
            specs=specs,
            groups=groups,
            config=config,
            scheduler_config=kv_cache_utils.generate_scheduler_kv_cache_config(
                [config]
            ),
            mamba_page_size_padded=vllm_config.cache_config.mamba_page_size_padded,
        )


@contextlib.contextmanager
def mamba_reserve(*, retained: int | None = None, draft: int | None = None):
    """What-if: change the resident-state reserve of every Mamba spec while a
    capacity is read. A request keeps ``1 + retained + draft`` block ids per
    group (align mode). The engine's values: ``retained`` is
    ``max_concurrent_batches`` with LeapQuant and 1 without, ``draft`` is
    ``num_speculative_blocks`` (2 at MTP2)."""
    original = MambaSpec.max_memory_usage_bytes

    def patched(self, vllm_config):
        if vllm_config.cache_config.mamba_cache_mode != "align":
            return original(self, vllm_config)
        kept = retained
        if kept is None:
            leap = vllm_config.cache_config.replayssm_state_dtype == "leapquant"
            kept = vllm_config.max_concurrent_batches if leap else 1
        drafts = self.num_speculative_blocks if draft is None else draft
        return self.page_size_bytes * (
            1 + kept + drafts + self.num_prefill_checkpoint_blocks
        )

    with mock.patch.object(MambaSpec, "max_memory_usage_bytes", patched):
        yield


@contextlib.contextmanager
def leapquant_shape(*, window: int | None = None, rank: int | None = None):
    """What-if: a LeapQuant blob with another window ``p`` or rank ``r``."""
    with contextlib.ExitStack() as stack:
        if window is not None:
            stack.enter_context(
                mock.patch.object(leapquant_gdn, "LEAPQUANT_WINDOW", window)
            )
        if rank is not None:
            stack.enter_context(
                mock.patch.object(leapquant_gdn, "LEAPQUANT_NUM_COMP", rank)
            )
        yield


@contextlib.contextmanager
def pack_any_state():
    """What-if: the planner's byte-fit split for buckets of any state format
    (the build applies it to LeapQuant buckets only)."""
    with mock.patch.object(leapquant_gdn, "is_leapquant_state_format", lambda _: True):
        yield


# ---------------------------------------------------------------------------
# One plan, analyzed
# ---------------------------------------------------------------------------


def analyze(p: H.Plan, geom: Geometry, requested: int | None = None) -> dict:
    """Everything the tables read from one plan."""
    cfg = p.config
    (A,) = {t.block_stride for t in cfg.kv_cache_tensors}
    N = cfg.num_blocks
    b = p.block_size
    need = p.blocks_per_request()
    groups = []
    for group, ids in zip(cfg.kv_cache_groups, need):
        specs = list(iter_layer_specs(group.kv_cache_spec))
        if len(specs) == 1:
            specs = specs * len(group.layer_names)
        padded = sum(s.page_size_bytes for s in specs)
        natural = sum(H._natural_page_bytes(s) for s in specs)
        groups.append(
            {
                "kind": H._group_kind(group),
                "layers": len(group.layer_names),
                "block": group.kv_cache_spec.block_size,
                "ids": ids,
                "page": specs[0].page_size_bytes,
                "natural_page": H._natural_page_bytes(specs[0]),
                "held": ids * A,
                "used": ids * natural,
                "page_padding": ids * (padded - natural),
                "slack": ids * (A - padded),
            }
        )
    kinds: dict[str, dict] = {}
    for g in groups:
        k = kinds.setdefault(
            g["kind"],
            {
                "groups": 0,
                "layers": [],
                "page": g["page"],
                "natural_page": g["natural_page"],
                **{f: 0 for f in ("ids", "held", "used", "page_padding", "slack")},
            },
        )
        k["groups"] += 1
        k["layers"].append(g["layers"])
        for f in ("ids", "held", "used", "page_padding", "slack"):
            k[f] += g[f]
    tokens, conc = p.capacity()
    attn_ids = kinds["attention"]["ids"]
    state_ids = sum(k["ids"] for n, k in kinds.items() if n != "attention")
    with H.serving_env(pack=p.pack):
        sched_block, hash_block = kv_cache_utils.resolve_kv_cache_block_sizes(
            p.scheduler_config, p.vllm_config
        )
        hit_alignment = kv_cache_utils.resolve_cache_hit_alignment_tokens(
            p.scheduler_config, p.vllm_config, sched_block, hash_block
        )
    gdn = kinds.get("GDN")
    return {
        "requested": requested if requested is not None else b,
        "block": b,
        "A": A,
        "N": N,
        "per_token": A / b,
        "remainder": geom.kv_bytes - N * A,
        "groups": groups,
        "kinds": kinds,
        "group_sizes": [g["block"] for g in groups],
        "attn_ids": attn_ids,
        "state_ids": state_ids,
        "tokens": tokens,
        "conc": conc,
        "longest_8": p.longest_context(MAX_NUM_SEQS),
        "padded_page": p.mamba_page_size_padded,
        "sched_block": sched_block,
        "hash_block": hash_block,
        "hit_alignment": hit_alignment,
        "prefill_step": (MAX_BATCHED // b) * b,
        "layers_per_block": (A // gdn["page"]) if gdn else None,
        "gdn_page": gdn["page"] if gdn else None,
        "gdn_natural_page": gdn["natural_page"] if gdn else None,
    }


class Study:
    """Plans and rows of one geometry, cached by their arguments."""

    def __init__(self, geom: Geometry, kv_bytes: int | None = None):
        self.geom = replace(geom, kv_bytes=kv_bytes or geom.kv_bytes)
        self._dir = tempfile.TemporaryDirectory()
        write_stub_model(self._dir.name, geom.kv_heads)
        self._plans: dict = {}
        self._rows: dict = {}

    def plan(self, block, *, mode, platform=True, **kw) -> H.Plan:
        """``mode``: off (FP32 state), lq (LeapQuant, padded pages), pack, or
        fp32pack (what-if: FP32 pages through the byte-fit split)."""
        key = (block, mode, platform, tuple(sorted((k, str(v)) for k, v in kw.items())))
        if key not in self._plans:
            with contextlib.ExitStack() as stack:
                if mode == "fp32pack":
                    stack.enter_context(pack_any_state())
                self._plans[key] = make_plan(
                    self.geom,
                    self._dir.name,
                    block,
                    leapquant=mode in ("lq", "pack"),
                    pack=mode in ("pack", "fp32pack"),
                    platform=platform and mode != "fp32pack",
                    **kw,
                )
        return self._plans[key]

    def row(self, block, *, mode, platform=True, **kw) -> dict:
        key = (block, mode, platform, tuple(sorted((k, str(v)) for k, v in kw.items())))
        if key not in self._rows:
            try:
                p = self.plan(block, mode=mode, platform=platform, **kw)
                self._rows[key] = analyze(p, self.geom, requested=block)
            except Infeasible as exc:
                self._rows[key] = {"requested": block, "infeasible": str(exc)}
            except ValueError as exc:
                if "To serve at least one request" not in str(exc):
                    raise
                self._rows[key] = {
                    "requested": block,
                    "infeasible": "one max-length request does not fit the pool",
                }
        return self._rows[key]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def mb(n: float) -> str:
    return f"{n / 1e6:,.1f}"


def describe_groups(layers: list[int]) -> str:
    """Layers per group: [3] * 12 -> "12 x 3", [5, 5, 4, 4] -> "2 x 5 + 2 x 4"."""
    counts = Counter(layers)
    return " + ".join(
        f"{n} x {size}" for size, n in sorted(counts.items(), reverse=True)
    )


def pct(a: float, b: float) -> str:
    return f"{100 * (a - b) / b:+.2f}%"


def tokens_vs(tokens: int, base: int) -> str:
    return f"{tokens:,} ({pct(tokens, base)})"


def md_table(headers: list[str], rows: list[list]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def served(row: dict) -> bool:
    """The row exists and the platform kept the requested block."""
    return "infeasible" not in row and row["block"] == row["requested"]


def smallest_block(geom: Geometry, state_page: int) -> int:
    """First multiple of 16 whose attention page holds the state page."""
    per_token = attention_bytes_per_token(geom)
    need = -(-state_page // per_token)
    return -(-need // 16) * 16


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

SWEEP_REQUESTS = (
    16, 64, 128, 192, 256, 320, 384, 448, 464, 480, 496, 512, 576, 608, 640, 704,
    768, 800, 832, 896, 1024,
)  # fmt: skip
BLOCK_TABLE = (
    128, 192, 256, 320, 384, 448, 464, 512, 576, 608, 640, 704, 768, 800, 896, 1024,
)  # fmt: skip
COMBINED_BLOCKS = (112, 128, 144, 208, 256, 304, 400, 464, 512, 608, 640, 1024)
ISOLATION_BLOCKS = (64, 112, 128, 176, 256, 400, 464, 512, 608, 640, 800, 1024)


def _cell(r: dict, base: int) -> str:
    if "infeasible" in r:
        return "infeasible"
    shown = f"{r['tokens']:,} ({pct(r['tokens'], base)})"
    if r["block"] != r["requested"]:
        return f"-> {r['block']}: {shown}"
    return shown


def table_sweep(study: Study, requests=SWEEP_REQUESTS) -> str:
    """Tokens by requested block, per mode, as served and what-if."""
    base = study.row(1024, mode="off")["tokens"]
    rows = []
    for req in requests:
        wi = study.row(req, mode="pack", platform=False)
        rows.append(
            [
                req,
                _cell(study.row(req, mode="off"), base),
                _cell(study.row(req, mode="lq"), base),
                _cell(study.row(req, mode="pack"), base),
                "(as served)"
                if served(study.row(req, mode="pack"))
                else _cell(wi, base),
            ]
        )
    return md_table(
        [
            "requested block",
            "flag off: tokens (vs flag off at 1,024)",
            "LeapQuant #168",
            "LeapQuant + packing #170",
            "packing, platform rule lifted (what-if)",
        ],
        rows,
    )


def table_sweep_all(study: Study, lo: int, hi: int) -> tuple[str, dict]:
    """Every multiple of 16 (the QSA kernel's rule): tokens per mode."""
    base = study.row(1024, mode="off")["tokens"]
    cols = {
        "off": [],
        "lq": [],
        "pack": [],
        "wi": [],
        "fp32pack": [],
    }
    data = []
    for b in range(lo, hi + 1, 16):
        r_off = study.row(b, mode="off")
        r_lq = study.row(b, mode="lq")
        r_pack = study.row(b, mode="pack")
        r_wi = study.row(b, mode="pack", platform=False)
        r_f = study.row(b, mode="fp32pack", platform=False)
        data.append((b, r_off, r_lq, r_pack, r_wi, r_f))
        for name, r in zip(cols, (r_off, r_lq, r_pack, r_wi, r_f)):
            if "infeasible" not in r and (name in ("wi", "fp32pack") or served(r)):
                cols[name].append((r["tokens"], b))
    best = {k: max(v) if v else (None, None) for k, v in cols.items()}

    def fmt(r, name, b):
        if "infeasible" in r or (name not in ("wi", "fp32pack") and not served(r)):
            return "-"
        text = f"{r['tokens']:,}"
        return f"**{text}**" if r["tokens"] == best[name][0] else text

    rows = []
    for b, r_off, r_lq, r_pack, r_wi, r_f in data:
        if "infeasible" in r_wi:
            continue
        gdn = r_wi["kinds"].get("GDN")
        rows.append(
            [
                b,
                f"{r_wi['A'] / 1e6:.3f}",
                r_wi["N"],
                fmt(r_off, "off", b),
                fmt(r_lq, "lq", b),
                fmt(r_pack, "pack", b),
                fmt(r_wi, "wi", b),
                describe_groups(gdn["layers"]) if gdn else "-",
                r_wi["state_ids"],
                fmt(r_f, "fp32pack", b),
                r_wi["prefill_step"],
            ]
        )
    text = md_table(
        [
            "block",
            "A (MB)",
            "blocks N",
            "flag off",
            "LeapQuant #168",
            "LeapQuant + packing #170",
            "packing, rule lifted (what-if)",
            "GDN groups (what-if)",
            "state ids/request (what-if)",
            "FP32 + packing (what-if)",
            "prefill tokens/step at 2,048",
        ],
        rows,
    )
    return text, {"best": best, "base": base}


def table_blocks(study: Study, blocks) -> str:
    """One row per block and mode: pages, padding, groups, state reserve,
    tokens and prefix-cache grain. A what-if row lifts the platform floor."""
    geom = study.geom
    base = study.row(geom.recipe_off_block, mode="off")["tokens"]
    variants = (("off", True), ("lq", True), ("pack", True), ("pack", False))
    rows = []
    for b in blocks:
        for mode, platform in variants:
            r = study.row(b, mode=mode, platform=platform)
            if "infeasible" in r or (platform and not served(r)):
                continue
            if not platform and served(study.row(b, mode="pack")):
                continue
            k = r["kinds"]
            others = [v for n, v in k.items() if n != "attention"]
            held = sum(v["held"] for v in others)
            used = sum(v["used"] for v in others)
            ids = " + ".join(str(k[n]["ids"]) for n in ("GDN", "PLE", "ring") if n in k)
            padding = r["gdn_page"] - r["gdn_natural_page"]
            label = MODES[mode] if platform else "packing, floor lifted (what-if)"
            rows.append(
                [
                    b,
                    label,
                    f"{attention_page(geom, b):,}",
                    f"{r['gdn_natural_page']:,} -> {r['gdn_page']:,}",
                    f"{padding:,} ({100 * padding / r['gdn_natural_page']:.1f}%)",
                    "attention page" if r["padded_page"] else "none",
                    describe_groups(k["GDN"]["layers"]),
                    ids,
                    f"{mb(held)} / {mb(used)}",
                    r["N"],
                    f"{r['tokens']:,}",
                    pct(r["tokens"], base),
                    r["hit_alignment"],
                    f"{r['prefill_step']:,}",
                ]
            )
    return md_table(
        [
            "block",
            "mode",
            "attention page (B)",
            "state page: natural -> in plan (B)",
            "padding per page (B)",
            "padded to",
            "GDN groups (layers)",
            "state ids GDN + PLE + ring",
            "state held / needed (MB)",
            "blocks N",
            "tokens",
            f"vs {base:,}-token flag off",
            "prefix hit grain (tokens)",
            "prefill tokens per step at 2,048",
        ],
        rows,
    )


def table_why(study: Study, configs) -> str:
    """Tokens = L * N / (attention ids + state ids): the terms, per config."""
    rows = []
    for label, block, mode, kw in configs:
        r = study.row(block, mode=mode, **kw)
        b, c = r["block"], r["per_token"]
        rows.append(
            [
                label,
                b,
                f"{r['A']:,}",
                r["N"],
                f"{r['remainder']:,} ({r['remainder'] / c:,.0f})",
                r["attn_ids"],
                f"{r['attn_ids'] * b - MAX_MODEL_LEN:,}",
                r["state_ids"],
                f"{r['state_ids'] * b:,}",
                f"{r['N'] * b:,}",
                f"{r['tokens']:,}",
            ]
        )
    return md_table(
        [
            "config",
            "block b",
            "A (bytes)",
            "N = floor(KV / A)",
            "pool remainder bytes (tokens-eq)",
            "attention ids",
            "attention rounding (tokens over 262,144)",
            "state ids S",
            "S x b (tokens-eq)",
            "N x b",
            "tokens = 262,144 N / (attention ids + S)",
        ],
        rows,
    )


def breakdown_rows(study: Study, label: str, row: dict) -> list[list]:
    """Where the pool goes, per rank: one max-length request and
    ``MAX_NUM_SEQS`` resident requests (their state reserve)."""
    pool = study.geom.kv_bytes
    exact_attn = MAX_MODEL_LEN * row["per_token"]
    k = row["kinds"]
    rows = []

    def add(name, ids, held, used, note=""):
        rows.append(
            [
                label,
                name,
                ids,
                mb(held),
                mb(used),
                f"{100 * held / pool:.2f}%",
                mb(MAX_NUM_SEQS * held) if name != "attention" else "",
                f"{100 * MAX_NUM_SEQS * held / pool:.2f}%"
                if name != "attention"
                else "",
                note,
            ]
        )

    a = k["attention"]
    add(
        "attention",
        a["ids"],
        a["held"],
        exact_attn,
        f"{row['per_token']:,.0f} B per token per rank",
    )
    g = k["GDN"]
    add(
        f"GDN state ({g['groups']} groups of {'+'.join(map(str, g['layers']))})",
        g["ids"],
        g["held"],
        g["used"],
        f"page padding {mb(g['page_padding'])} MB, unused block bytes "
        f"{mb(g['slack'])} MB",
    )
    for kind in ("PLE", "ring"):
        if kind in k:
            v = k[kind]
            add(kind, v["ids"], v["held"], v["used"], "")
    rows.append(
        [
            label,
            "pool remainder",
            "",
            mb(row["remainder"]),
            "",
            f"{100 * row['remainder'] / pool:.2f}%",
            "",
            "",
            f"N = {row['N']} blocks of {row['A']:,} B",
        ]
    )
    return rows


def table_breakdown(study: Study, configs) -> str:
    rows = []
    for label, block, mode, kw in configs:
        rows += breakdown_rows(study, label, study.row(block, mode=mode, **kw))
    return md_table(
        [
            "config",
            "owner",
            "block ids per max-length request",
            "bytes held MB",
            "bytes needed MB",
            "% of KV pool",
            f"state reserve of {MAX_NUM_SEQS} requests MB",
            f"{MAX_NUM_SEQS} requests, % of pool",
            "note",
        ],
        rows,
    )


def table_pool(study: Study, configs) -> str:
    rows = []
    for label, block, mode, kw in configs:
        r = study.row(block, mode=mode, **kw)
        k = r["kinds"]
        held = sum(v["held"] for n, v in k.items() if n != "attention")
        used = sum(v["used"] for n, v in k.items() if n != "attention")
        left = (r["N"] - MAX_NUM_SEQS * r["state_ids"]) * r["block"]
        rows.append(
            [
                label,
                r["block"],
                r["N"],
                f"{r['tokens']:,}",
                f"{100 * r['state_ids'] / (r['attn_ids'] + r['state_ids']):.2f}%",
                mb(held),
                mb(used),
                f"{100 * used / held:.1f}%",
                f"{left:,}",
                f"{r['longest_8']:,}",
            ]
        )
    return md_table(
        [
            "config",
            "block",
            "N",
            "tokens (log)",
            "state share of a max-length request",
            "state held MB per request",
            "state needed MB per request",
            "needed / held",
            f"attention tokens left with {MAX_NUM_SEQS} requests resident",
            f"longest max-model-len at {MAX_NUM_SEQS}.0x concurrency",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# Ceilings and options
# ---------------------------------------------------------------------------


def exact_state_bytes(row: dict) -> int:
    """Bytes of state one request reserves at natural page size."""
    return sum(v["used"] for n, v in row["kinds"].items() if n != "attention")


def ceilings(study: Study, block: int, mode: str, **kw) -> dict:
    """Capacity ceilings on the block grid of one configuration."""
    geom = study.geom
    served_row = study.row(block, mode=mode, **kw)
    A, N, alpha = served_row["A"], served_row["N"], served_row["attn_ids"]
    b = served_row["block"]
    L = MAX_MODEL_LEN
    out = {"block": b, "tokens": served_row["tokens"]}
    plat = {"platform": False, **kw}
    out["gdn_state_zero"] = study.row(b, mode=mode, drop=frozenset({"gdn"}), **plat)[
        "tokens"
    ]
    out["pure_attention"] = study.row(
        b, mode=mode, drop=frozenset({"gdn", "ple", "ring"}), **plat
    )["tokens"]
    exact = exact_state_bytes(served_row)
    out["exact_bytes_one_request"] = exact
    out["no_padding_shared_pool"] = int(L * N / (alpha + exact / A))
    attn_block = served_row["per_token"] * b
    out["separate_pool_one_request"] = int(
        L * int((geom.kv_bytes - exact) // attn_block) / alpha
    )
    with mamba_reserve(retained=1):
        out["reserve_flag_off_pattern"] = study.plan(b, mode=mode, **kw).capacity()[0]
    out["attention_tokens_8_shared"] = (N - MAX_NUM_SEQS * served_row["state_ids"]) * b
    out["attention_tokens_8_separate"] = (
        int((geom.kv_bytes - MAX_NUM_SEQS * exact) // attn_block) * b
    )
    out["pool_ceiling"] = int(geom.kv_bytes / served_row["per_token"])
    return out


def table_ceilings(study: Study, configs) -> tuple[str, dict]:
    rows = []
    data = {}
    for label, block, mode, kw in configs:
        c = ceilings(study, block, mode, **kw)
        data[label] = c
        base = c["tokens"]
        rows.append(
            [
                label,
                c["block"],
                f"{c['tokens']:,}",
                *(
                    tokens_vs(c[key], base)
                    for key in (
                        "reserve_flag_off_pattern",
                        "no_padding_shared_pool",
                        "separate_pool_one_request",
                        "gdn_state_zero",
                        "pure_attention",
                    )
                ),
                f"{c['attention_tokens_8_shared']:,}",
                f"{c['attention_tokens_8_separate']:,}",
            ]
        )
    text = md_table(
        [
            "config",
            "block",
            "tokens (log)",
            "reserve = flag-off pattern (4 ids per group)",
            "state at exact bytes, shared pool",
            "separate state pool, exact bytes",
            "GDN state = 0 bytes",
            "pure attention pool (all state 0)",
            f"attention tokens with {MAX_NUM_SEQS} requests resident, shared pool",
            f"... separate state pool for {MAX_NUM_SEQS} requests",
        ],
        rows,
    )
    return text, data


def table_reserve(study: Study, configs) -> tuple[str, dict]:
    """What-if (c): fewer resident-state ids per group (all Mamba groups)."""
    variants = [
        ("5 = 1 + 2 retained + 2 draft (LeapQuant today)", None, None),
        ("4 = 1 + 1 retained + 2 draft (flag-off pattern)", 1, None),
        ("3 = 1 + 2 retained + 0 draft", None, 0),
        ("2 = 1 + 1 retained + 0 draft", 1, 0),
    ]
    rows = []
    data = {}
    for label, block, mode, kw in configs:
        p = study.plan(block, mode=mode, **kw)
        base = p.capacity()[0]
        cells = [label, p.block_size]
        for name, retained, draft in variants:
            with mamba_reserve(retained=retained, draft=draft):
                tokens = p.capacity()[0]
                l8 = p.longest_context(MAX_NUM_SEQS)
            data[(label, name)] = (tokens, l8)
            cells.append(f"{tokens:,} ({pct(tokens, base)}); 8-req {l8:,}")
        rows.append(cells)
    return md_table(["config", "block"] + [v[0] for v in variants], rows), data


def staircase(study: Study, lo=48, hi=1536) -> tuple[str, list]:
    """What-if (b): per GDN group count, the first block that holds it and the
    best tokens at any block with that count (packing, rule lifted)."""
    by_groups: dict[int, list] = {}
    for b in range(lo, hi + 1, 16):
        r = study.row(b, mode="pack", platform=False)
        if "infeasible" in r:
            continue
        g = r["kinds"]["GDN"]["groups"]
        by_groups.setdefault(g, []).append((b, r))
    rows = []
    data = []
    for g in sorted(by_groups, reverse=True):
        entries = by_groups[g]
        first_b, first = entries[0]
        best_b, best = max(entries, key=lambda e: (e[1]["tokens"], -e[0]))
        rows.append(
            [
                g,
                describe_groups(first["kinds"]["GDN"]["layers"]),
                first_b,
                f"{first['tokens']:,}",
                first["state_ids"],
                best_b,
                f"{best['tokens']:,}",
                best["state_ids"],
                "yes" if first_b >= 464 else "no (below 464)",
            ]
        )
        data.append((g, first_b, first["tokens"], best_b, best["tokens"]))
    return (
        md_table(
            [
                "GDN groups",
                "layers per group",
                "first block with this count",
                "tokens there",
                "state ids",
                "best block with this count",
                "tokens there",
                "state ids",
                "platform accepts the first block",
            ],
            rows,
        ),
        data,
    )


def table_window(study: Study) -> tuple[str, dict]:
    """What-if (e): a smaller blob. The window is a module constant today."""
    geom = study.geom
    rows = []
    data = {}
    for window, rank in ((32, 4), (24, 4), (20, 4), (16, 4), (32, 2), (16, 2)):
        with leapquant_shape(window=window, rank=rank):
            local = Study(geom)
            r512 = local.row(512, mode="lq")
            page = r512["gdn_natural_page"]
            min_b = smallest_block(geom, page)
            r_min = local.row(min_b, mode="lq")
            cands = [
                (local.row(b, mode="pack")["tokens"], b)
                for b in range(min_b, 1537, 16)
                if served(local.row(b, mode="pack"))
            ]
            best_pack = max(cands)
            lq_best = max(
                (local.row(b, mode="lq")["tokens"], b)
                for b in range(min_b, 1537, 16)
                if served(local.row(b, mode="lq"))
            )
            layout = leapquant_gdn.leapquant_gdn_layout(128, 128, NUM_SPEC)
            data[(window, rank)] = (layout.nbytes, page, min_b)
            rows.append(
                [
                    f"p={window}, r={rank}",
                    f"{layout.nbytes:,}",
                    f"{page:,}",
                    min_b,
                    f"{r_min['tokens']:,}",
                    f"{r512['tokens']:,}",
                    f"{lq_best[0]:,} at {lq_best[1]}",
                    f"{best_pack[0]:,} at {best_pack[1]}",
                ]
            )
    return (
        md_table(
            [
                "blob",
                "bytes per head",
                "state page (12 heads + conv)",
                "smallest block",
                "tokens there (#168)",
                "tokens at 512 (#168)",
                "best #168 block",
                "best #170 block",
            ],
            rows,
        ),
        data,
    )


def blob_composition() -> str:
    layout = leapquant_gdn.leapquant_gdn_layout(128, 128, NUM_SPEC)
    offs = layout.offsets
    names = [s[0] for s in layout._sections()]
    sizes = {}
    for i, name in enumerate(names):
        end = offs[names[i + 1]] if i + 1 < len(names) else offs["_total"]
        sizes[name] = end - offs[name]
    total = layout.nbytes
    groups = [
        ("INT8 residual z", ["z"]),
        ("scales (bscale, csmooth)", ["bscale", "csmooth"]),
        ("compensators (comp_k, comp_u)", ["comp_k", "comp_u"]),
        ("record ring (rec_k, rec_u, rec_g)", ["rec_k", "rec_u", "rec_g"]),
        ("header", ["header"]),
    ]
    rows = []
    for title, members in groups:
        n = sum(sizes[m] for m in members)
        rows.append([title, f"{n:,}", f"{100 * n / total:.1f}%"])
    rows.append(["total", f"{total:,}", "100%"])
    return md_table(["section", "bytes per head", "share"], rows)


def table_ple(study: Study, blocks) -> str:
    """What-if (d-lite): the PLE layer takes a GDN-sized page in a GDN group."""
    base = study.row(1024, mode="off")["tokens"]
    rows = []
    for b in blocks:
        for mode in ("lq", "pack"):
            plain = study.row(b, mode=mode)
            merged = study.row(b, mode=mode, ple_joins_gdn=True)
            if "infeasible" in plain or not served(plain):
                continue
            rows.append(
                [
                    b,
                    MODES[mode],
                    f"{plain['tokens']:,}",
                    f"{merged['tokens']:,} ({pct(merged['tokens'], plain['tokens'])})",
                    f"{plain['state_ids']} -> {merged['state_ids']}",
                    pct(merged["tokens"], base),
                ]
            )
    return md_table(
        [
            "block",
            "mode",
            "tokens",
            "PLE merged into a GDN group",
            "state ids per request",
            "merged vs flag off at 1,024",
        ],
        rows,
    )


def table_combined(study: Study, blocks) -> tuple[str, dict]:
    """What-if: the levers together, each column adds to the one before.
    Packing with the platform rule lifted, the PLE layer merged into a GDN
    group, then fewer resident-state ids per group."""
    base = study.row(1024, mode="off")["tokens"]
    merge = {"ple_joins_gdn": True} if study.geom.ple else {}
    variants = [("packing, rule lifted", {}, {})]
    if merge:
        variants.append(("+ PLE merged", merge, {}))
    variants += [
        ("+ 4 ids per group", merge, {"retained": 1}),
        ("+ 3 ids per group, no draft block", merge, {"draft": 0}),
    ]
    rows = []
    data = {}
    for b in blocks:
        cells = [b]
        for name, plan_kw, reserve_kw in variants:
            plan = study.plan(b, mode="pack", platform=False, **plan_kw)
            with mamba_reserve(**reserve_kw):
                tokens = plan.capacity()[0]
                limit = plan.longest_context(MAX_NUM_SEQS)
            data[(b, name)] = (tokens, limit)
            cells.append(f"{tokens:,} ({pct(tokens, base)}) / {limit:,}")
        rows.append(cells)
    return md_table(["block"] + [v[0] for v in variants], rows), data


def table_isolation(study: Study, blocks) -> str:
    """What-if: the same planner on FP32 pages and on LeapQuant pages, so the
    difference is the state format alone."""
    rows = []
    for b in blocks:
        fp32 = study.row(b, mode="fp32pack", platform=False)
        leap = study.row(b, mode="pack", platform=False)
        if "infeasible" in fp32 or "infeasible" in leap:
            continue
        f, q = fp32["kinds"]["GDN"], leap["kinds"]["GDN"]
        rows.append(
            [
                b,
                describe_groups(f["layers"]),
                fp32["state_ids"],
                f"{fp32['tokens']:,}",
                describe_groups(q["layers"]),
                leap["state_ids"],
                f"{leap['tokens']:,}",
                pct(leap["tokens"], fp32["tokens"]),
            ]
        )
    return md_table(
        [
            "block",
            "FP32: GDN groups",
            "FP32: state ids",
            "FP32 + packing tokens",
            "LeapQuant: GDN groups",
            "LeapQuant: state ids",
            "LeapQuant + packing tokens",
            "LeapQuant over FP32",
        ],
        rows,
    )


def best_row(study: Study, mode: str, *, platform: bool) -> dict:
    """The multiple of 16 with the most tokens. With ``platform`` the real
    platform must keep the block."""
    best = None
    for b in range(16, 1537, 16):
        r = study.row(b, mode=mode, platform=platform)
        if "infeasible" in r or (platform and not served(r)):
            continue
        if best is None or r["tokens"] > best["tokens"]:
            best = r
    return best


def table_summary(study: Study) -> tuple[str, list]:
    geom = study.geom
    base = study.row(geom.recipe_off_block, mode="off")
    entries = [
        ("flag off, recipe block", base, "recipe"),
        ("flag off, best block", best_row(study, "off", platform=True), "recipe"),
        (
            "LeapQuant #168, recipe block",
            study.row(geom.recipe_lq_block, mode="lq"),
            "recipe",
        ),
        ("LeapQuant #168, best block", best_row(study, "lq", platform=True), "recipe"),
        (
            "LeapQuant + packing #170, best block",
            best_row(study, "pack", platform=True),
            "recipe",
        ),
        (
            "packing, platform rule lifted, best block",
            best_row(study, "pack", platform=False),
            "what-if: platform",
        ),
        (
            "FP32 + packing, platform rule lifted, best block",
            best_row(study, "fp32pack", platform=False),
            "what-if: platform and planner",
        ),
    ]
    rows = [
        [
            label,
            r["block"],
            f"{r['tokens']:,}",
            pct(r["tokens"], base["tokens"]),
            f"{r['longest_8']:,}",
            f"{r['prefill_step']:,}",
            needs,
        ]
        for label, r, needs in entries
    ]
    text = md_table(
        [
            "configuration",
            "block",
            "tokens",
            f"vs {base['block']:,}-token flag off",
            f"longest max-model-len at {MAX_NUM_SEQS}.0x concurrency",
            "prefill tokens per step at 2,048",
            "needs",
        ],
        rows,
    )
    return text, [(label, r["block"], r["tokens"]) for label, r, _ in entries]


def table_attribution(study: Study) -> tuple[str, list]:
    """Steps from the flag-off recipe to the LeapQuant recipe, one cause each.
    The steps are ordered, so the split depends on the order."""
    geom = study.geom
    steps = [
        (
            "flag off, recipe block",
            study.row(geom.recipe_off_block, mode="off"),
            "recipe",
        ),
        (
            "flag off, best block the platform keeps",
            best_row(study, "off", platform=True),
            "recipe change",
        ),
        (
            "FP32 state, packing, at the LeapQuant block",
            study.row(geom.recipe_lq_block, mode="fp32pack", platform=False),
            "what-if: platform and planner",
        ),
        (
            "LeapQuant INT8 state, as shipped",
            study.row(geom.recipe_lq_block, mode="lq"),
            "PR #168",
        ),
    ]
    first = steps[0][1]["tokens"]
    total = steps[-1][1]["tokens"] - first
    rows = []
    data = []
    previous = first
    for index, (label, r, needs) in enumerate(steps):
        gain = r["tokens"] - previous
        share = f"{100 * gain / total:.0f}%" if index and total > 0 else ""
        rows.append(
            [
                label,
                r["block"],
                f"{r['tokens']:,}",
                f"{gain:+,}" if index else "",
                share,
                needs,
            ]
        )
        data.append((label, r["block"], r["tokens"]))
        previous = r["tokens"]
    headers = ["step", "block", "tokens", "gain", "share of the total gain", "needs"]
    return md_table(headers, rows), data


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_selfcheck(study: Study, out: dict) -> None:
    geom = study.geom
    print("### Logged capacities against the planner\n")
    checks = []
    if geom.kv_bytes == GEOMETRIES[geom.key].kv_bytes:
        for (model, block, mode), (logged, source) in LOGGED.items():
            if model == geom.key:
                got = study.row(block, mode=mode)["tokens"]
                checks.append((block, MODES[mode], logged, source, got))
        if geom.key == "flashnext":
            with mamba_reserve(retained=1):
                got = study.plan(512, mode="lq").capacity()[0]
            label = "LeapQuant, 4 state ids per group"
            checks.append(
                (512, label, SMOKE_BOOT_512_FOUR_IDS, "boot log 2026-11-13", got)
            )
    if checks:
        rows = [
            [b, m, f"{want:,}", s, f"{got:,}", "equal" if got == want else "DIFFERENT"]
            for b, m, want, s, got in checks
        ]
        headers = [
            "block",
            "mode",
            "logged tokens",
            "source",
            "planner tokens",
            "check",
        ]
        print(md_table(headers, rows))
    else:
        print("The KV budget differs from the recipe, so no boot log applies.")
    ok = all(want == got for _, _, want, _, got in checks)
    print("\n### Platform bypass against the real platform\n")
    bad = []
    count = 0
    for mode in ("off", "lq", "pack"):
        for b in range(16, 1537, 16):
            real = study.row(b, mode=mode)
            if not served(real):
                continue
            fake = study.row(b, mode=mode, platform=False)
            count += 1
            if (
                fake.get("padded_page") != real["padded_page"]
                or fake.get("tokens") != real["tokens"]
                or fake.get("block") != real["block"]
            ):
                bad.append((mode, b))
    print(
        f"{count} (mode, block) pairs where the real platform keeps the block: "
        f"{'all equal' if not bad else 'DIFFERENT: ' + str(bad)}"
    )
    out["selfcheck"] = {"logged_equal": ok, "bypass_equal": not bad, "pairs": count}


def cmd_summary(study: Study, out: dict) -> None:
    text, data = table_summary(study)
    print("### Summary\n")
    print(text)
    text, steps = table_attribution(study)
    out["attribution"] = steps
    if steps[-1][2] > steps[0][2]:
        print("\n### Attribution of the recipe gain\n")
        print(text)
    out["summary"] = data


def cmd_sweep(study: Study, out: dict) -> None:
    geom = study.geom
    print(f"### Sweep by requested block, {geom.title}, KV {geom.kv_bytes:,} B\n")
    print(table_sweep(study))
    print("\n### Every multiple of 16\n")
    text, info = table_sweep_all(study, 16, 1536)
    print(text)
    out["sweep_best"] = info["best"]


def cmd_anatomy(study: Study, out: dict) -> None:
    print("### Pages, padding, state reserve and prefix-cache grain by block\n")
    print(table_blocks(study, BLOCK_TABLE))


def default_configs(study: Study) -> list[tuple]:
    if study.geom.qsa:
        return [
            ("flag off, 1,024", 1024, "off", {}),
            ("LeapQuant #168, 512", 512, "lq", {}),
            ("LeapQuant #168, 464 (smallest block)", 464, "lq", {}),
            ("LeapQuant + packing, 608", 608, "pack", {}),
            ("LeapQuant + packing, 640", 640, "pack", {}),
        ]
    return [
        ("flag off, 1,024", 1024, "off", {}),
        ("LeapQuant #168, 1,024", 1024, "lq", {}),
        ("LeapQuant #168, 512", 512, "lq", {}),
        ("LeapQuant #168, 464 (smallest block)", 464, "lq", {}),
        ("LeapQuant + packing, 1,024", 1024, "pack", {}),
    ]


def cmd_breakdown(study: Study, out: dict) -> None:
    configs = default_configs(study)
    print("### Where the pool goes\n")
    print(table_breakdown(study, configs))
    print("\n### One line per configuration\n")
    print(table_pool(study, configs))
    print("\n### Why the tokens move\n")
    print(table_why(study, configs))


def cmd_ceilings(study: Study, out: dict) -> None:
    text, data = table_ceilings(study, default_configs(study))
    print("### Ceilings\n")
    print(text)
    out["ceilings"] = data


def cmd_options(study: Study, out: dict) -> None:
    geom = study.geom
    print("### (a) Smallest block\n")
    for mode in ("off", "lq"):
        page = study.row(1024, mode=mode)["gdn_natural_page"]
        small = smallest_block(geom, page)
        print(
            f"- {MODES[mode]}: state page {page:,} B, smallest block {small} "
            f"(attention page {attention_page(geom, small):,} B)"
        )
    print("\n### (b) Layers per block: GDN groups against block size\n")
    text, stair = staircase(study)
    print(text)
    out["staircase"] = stair
    print("\n### (c) Reserve per group\n")
    configs = [c for c in default_configs(study) if c[2] != "off"]
    text, reserve = table_reserve(study, configs)
    print(text)
    out["reserve"] = {f"{k[0]} | {k[1]}": v for k, v in reserve.items()}
    if geom.ple:
        print("\n### (d) PLE merged into a GDN group\n")
        print(table_ple(study, (464, 512, 608, 640)))
    print("\n### (e) Smaller blob\n")
    print(blob_composition())
    print()
    text, shrink = table_window(study)
    print(text)
    out["window"] = {str(k): v for k, v in shrink.items()}
    print("\n### (f) The levers together (what-if)\n")
    print("Each cell: tokens (vs flag off) / longest max-model-len at 8.0x.\n")
    text, combined = table_combined(study, COMBINED_BLOCKS)
    print(text)
    out["combined"] = {f"{k[0]} | {k[1]}": v for k, v in combined.items()}
    print("\n### (g) State format alone, FP32 against LeapQuant (what-if)\n")
    print(table_isolation(study, ISOLATION_BLOCKS))


COMMANDS = {
    "selfcheck": cmd_selfcheck,
    "summary": cmd_summary,
    "sweep": cmd_sweep,
    "anatomy": cmd_anatomy,
    "breakdown": cmd_breakdown,
    "ceilings": cmd_ceilings,
    "options": cmd_options,
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["all", *COMMANDS])
    ap.add_argument("--model", choices=sorted(GEOMETRIES), default="flashnext")
    ap.add_argument("--kv-bytes", type=int, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    study = Study(GEOMETRIES[args.model], args.kv_bytes)
    print(f"## {study.geom.title}, KV {study.geom.kv_bytes:,} B per rank\n")
    out: dict = {}
    for name in COMMANDS if args.command == "all" else [args.command]:
        COMMANDS[name](study, out)
        print()
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
