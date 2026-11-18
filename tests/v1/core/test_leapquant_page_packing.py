# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Algorithm: LeapQuant (Pan et al., arXiv 2609.38166). The page packing and these
# tests are original code written for this fork.
"""LeapQuant GDN page packing, on the real KV-cache planner (CPU only).

The specs are Qwen3.8-Flash-Next per TP rank (TP4, MTP2): 36 GDN layers, 12 QSA
layers plus the MTP draft layer (main KV, compressed key and key ring each) and
one replicated PLE short-conv layer. The geometry comes from the HF config. The
flag-off numbers were first checked against two real boot logs (273,456 tokens
for 512-token LeapQuant blocks, 266,945 for 1024-token FP32 blocks).

``python tests/v1/core/test_leapquant_page_packing.py`` prints the capacity
table of ``docs/rdna2/leapquant-page-packing-2026-11-14.md``.
"""

import contextlib
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from unittest import mock

import pytest
import torch
from transformers import OPTConfig

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops import leapquant_gdn
from vllm.model_executor.models import ModelRegistry
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.interface import Platform
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core import kv_cache_planning, kv_cache_utils
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    iter_layer_specs,
)
from vllm.v1.worker.utils import allocate_kv_cache

pytestmark = pytest.mark.cpu_test

# Per TP rank. KV_BYTES is ``kv_cache_memory_bytes`` of the flashnext recipes.
GDN_LAYERS = 36
ATTN_LAYERS = 12
KV_BYTES = 4_026_531_840
MAX_MODEL_LEN = 262_144
NUM_SPEC = 2
# The block-size column of the table; 512 is the shipped LeapQuant recipe.
BLOCK_SIZES = (512, 608, 640, 1024)
# Bytes of one QSA main page per token: K and V, 256 wide, FP16, one KV head.
ATTN_BYTES_PER_TOKEN = 1024
# One LeapQuant GDN layer per rank (window 32, rank 4, MTP2):
# conv 2560 x 5 FP16 plus 12 value heads of 37,024 B each.
LQ_PAGE = 2560 * 5 * 2 + 12 * 37_024
FP32_PAGE = 2560 * 5 * 2 + 12 * 128 * 128 * 4
# The replicated PLE short-conv layer: 10,240 channels x 11 taps, FP16.
PLE_PAGE = 10_240 * 11 * 2
# The key ring of one QSA layer (13 with the MTP draft): 8 rows of 128, FP16.
RING_BYTES = (ATTN_LAYERS + 1) * 8 * 128 * 2


class _RdnaStyleBackend:
    """The two hooks ``_align_hybrid_block_size`` reads from the attention
    backend, as ``RdnaAttentionBackend`` defines them."""

    @staticmethod
    def customize_spec(spec):
        return replace(
            spec,
            num_head_slots=2,
            state_content_bytes=spec.num_kv_heads
            * spec.head_size
            * spec.dtype.itemsize,
        )

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]


class _StateLayer:
    """A Mamba layer reduced to what ``MambaBase.get_kv_cache_spec`` and
    ``bind_kv_cache`` read, so both run unchanged."""

    def __init__(
        self, shapes, dtypes, mamba_type, tp_replicated=False, state_format=None
    ):
        self._shapes, self._dtypes = shapes, dtypes
        self.mamba_type = mamba_type
        self.is_kv_cache_tp_replicated = tp_replicated
        self._state_format = state_format

    def get_state_shape(self):
        return self._shapes

    def get_state_dtype(self):
        return self._dtypes

    def spec(self, vllm_config) -> MambaSpec:
        spec = MambaBase.get_kv_cache_spec(self, vllm_config)
        if self._state_format is not None:
            spec = replace(spec, state_format=self._state_format)
        return spec

    def bind(self, kv_cache: torch.Tensor) -> None:
        MambaBase.bind_kv_cache(self, kv_cache)


class _FlashNextModel:
    """``get_mamba_specs_from_config`` of Qwen4Exp: the GDN and PLE states."""

    @staticmethod
    def gdn_layer(vllm_config) -> _StateLayer:
        cache = vllm_config.cache_config
        state_dtype = cache.replayssm_state_dtype
        return _StateLayer(
            MambaStateShapeCalculator.gated_delta_net_state_shape(
                tp_world_size=4,
                num_k_heads=16,
                num_v_heads=48,
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
            # The layer stamps the format identity on its spec
            # (QwenGatedDeltaNetAttention.get_kv_cache_spec).
            state_format=(
                leapquant_gdn.leapquant_gdn_layout(128, 128, NUM_SPEC).state_format
                if state_dtype == "leapquant"
                else None
            ),
        )

    @staticmethod
    def ple_layer(vllm_config) -> _StateLayer:
        return _StateLayer(
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

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config):
        return tuple(
            replace(layer.spec(vllm_config), block_size=-1, page_size_padded=None)
            for layer in (cls.gdn_layer(vllm_config), cls.ple_layer(vllm_config))
        )


@contextlib.contextmanager
def flags(*, pack: bool):
    """Set the packing switch as on gfx10x; only the HIP probe is faked. The
    LeapQuant format is not an environment switch: it is the
    ``--replayssm-state-dtype leapquant`` field of the CacheConfig."""
    with (
        mock.patch.dict(
            os.environ, {"VLLM_RDNA_GDN_LEAPQUANT_PACK": "1" if pack else "0"}
        ),
        mock.patch("vllm.platforms.rdna.on_rdna2_hip", lambda: True),
    ):
        leapquant_gdn.leapquant_gdn_pack_requested.cache_clear()
        try:
            yield
        finally:
            leapquant_gdn.leapquant_gdn_pack_requested.cache_clear()


@contextlib.contextmanager
def serving_env(*, pack: bool):
    """CPU stand-in for one serve: async scheduling (two concurrent batches),
    MTP2, the packing switch."""
    with (
        mock.patch("vllm.platforms.current_platform", CpuPlatform()),
        mock.patch.object(
            VllmConfig, "max_concurrent_batches", property(lambda self: 2)
        ),
        mock.patch.object(
            VllmConfig, "num_speculative_tokens", property(lambda self: NUM_SPEC)
        ),
        flags(pack=pack),
    ):
        yield


@contextlib.contextmanager
def max_model_len(vllm_config: VllmConfig, value: int):
    original = vllm_config.model_config.max_model_len
    vllm_config.model_config.max_model_len = value
    try:
        yield
    finally:
        vllm_config.model_config.max_model_len = original


@dataclass
class Plan:
    """One planned KV-cache config and what the logs would say about it."""

    block_size: int
    leapquant: bool
    pack: bool
    vllm_config: VllmConfig
    specs: dict[str, KVCacheSpec]
    groups: list[KVCacheGroupSpec]
    config: KVCacheConfig
    scheduler_config: KVCacheConfig
    mamba_page_size_padded: int | None

    def gdn_groups(self) -> list[KVCacheGroupSpec]:
        return [g for g in self.config.kv_cache_groups if _gdn_group(g)]

    def blocks_per_request(self, length: int = MAX_MODEL_LEN) -> list[int]:
        """Blocks each group takes for one request of ``length`` tokens."""
        with (
            serving_env(pack=self.pack),
            max_model_len(self.vllm_config, length),
        ):
            return [
                -(
                    -group.kv_cache_spec.max_memory_usage_bytes(self.vllm_config)
                    // group.kv_cache_spec.page_size_bytes
                )
                for group in self.scheduler_config.kv_cache_groups
            ]

    def gdn_blocks_per_request(self) -> int:
        return sum(
            blocks
            for group, blocks in zip(
                self.scheduler_config.kv_cache_groups, self.blocks_per_request()
            )
            if _gdn_group(group)
        )

    def capacity(self, length: int = MAX_MODEL_LEN) -> tuple[int, float]:
        """The tokens and concurrency of ``GPU KV cache size`` in the log."""
        with (
            serving_env(pack=self.pack),
            max_model_len(self.vllm_config, length),
        ):
            return kv_cache_utils.get_kv_cache_capacity(
                self.vllm_config, self.scheduler_config
            )

    def longest_context(self, concurrency: int) -> int:
        """Longest max_model_len at which ``concurrency`` requests still fit."""
        lo, hi = 1, MAX_MODEL_LEN
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.capacity(mid)[1] >= concurrency:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def fixed_cost(self) -> dict[str, tuple[int, int, int]]:
        """What one request of 262,144 tokens holds outside the attention group.

        Per owner (GDN, PLE, ring): the block ids, the bytes those blocks span
        and the bytes of state in them."""
        assert [g.layer_names for g in self.config.kv_cache_groups] == [
            g.layer_names for g in self.scheduler_config.kv_cache_groups
        ]
        (block,) = {t.block_stride for t in self.config.kv_cache_tensors}
        cost: dict[str, tuple[int, int, int]] = {}
        for group, blocks in zip(
            self.config.kv_cache_groups, self.blocks_per_request()
        ):
            owner = _group_kind(group)
            if owner == "attention":
                continue
            used = blocks * sum(
                _natural_page_bytes(s) for s in iter_layer_specs(group.kv_cache_spec)
            )
            was = cost.get(owner, (0, 0, 0))
            cost[owner] = (was[0] + blocks, was[1] + blocks * block, was[2] + used)
        return cost


def _gdn_group(group: KVCacheGroupSpec) -> bool:
    specs = list(iter_layer_specs(group.kv_cache_spec))
    return all(
        isinstance(s, MambaSpec) and s.mamba_type == MambaAttentionBackendEnum.GDN_ATTN
        for s in specs
    )


def _group_kind(group: KVCacheGroupSpec) -> str:
    specs = list(iter_layer_specs(group.kv_cache_spec))
    if _gdn_group(group):
        return "GDN"
    if all(isinstance(s, MambaSpec) for s in specs):
        return "PLE"
    if all(isinstance(s, CircularBufferSpec) for s in specs):
        return "ring"
    return "attention"


def _natural_page_bytes(spec: KVCacheSpec) -> int:
    """The bytes of state one layer keeps per block, without page padding."""
    if getattr(spec, "page_size_padded", None) is not None:
        spec = replace(spec, page_size_padded=None)
    return spec.page_size_bytes


def make_vllm_config(
    model_dir, block_size: int, layout: str = "BLNHC", *, leapquant: bool = True
) -> VllmConfig:
    model_config = ModelConfig(
        model=str(model_dir),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
        max_model_len=MAX_MODEL_LEN,
    )
    config = VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=2048,
            max_model_len=MAX_MODEL_LEN,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
        ),
        parallel_config=ParallelConfig(),
    )
    cache = config.cache_config
    # The layout the engine core resolves for QSA: block-outermost (BLNHC).
    cache.kv_cache_layout = layout
    cache.mamba_cache_mode = "align"
    cache.mamba_block_size = block_size
    # The CacheConfig field behind ``--replayssm-state-dtype leapquant``. Set
    # after the VllmConfig is built: the start-up validator needs a GDN model
    # config and the V2 runner, which the planner tests do not model
    # (tests/config/test_leapquant_config.py covers it).
    if leapquant:
        cache.replayssm_state_dtype = "leapquant"
    register_all_kvcache_specs(config)
    return config


def flash_next_specs(
    vllm_config: VllmConfig, *, ple_joins_gdn: bool = False
) -> dict[str, KVCacheSpec]:
    """The per-rank spec dict, built like the runner does after the platform
    aligned the block size and padded (or not) the Mamba pages.

    ``ple_joins_gdn`` is a what-if for the design note, not a shipped layout:
    the PLE layer takes a GDN-sized page in the GDN bucket."""
    block = vllm_config.cache_config.block_size
    gdn = _FlashNextModel.gdn_layer(vllm_config).spec(vllm_config)
    ple = _FlashNextModel.ple_layer(vllm_config).spec(vllm_config)
    main = _RdnaStyleBackend.customize_spec(
        FullAttentionSpec(
            block_size=block,
            num_kv_heads=1,
            head_size=256,
            head_size_v=256,
            dtype=torch.float16,
        )
    )
    compressed = MLAAttentionSpec(
        block_size=block,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        tokens_per_state=4,
    )
    # QSARawKeyCache: capacity = round_up(compress_ratio + num_spec, ratio) = 8.
    ring = CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=128, head_size_v=0, dtype=torch.float16
    )
    specs: dict[str, KVCacheSpec] = {}
    attn_ids = {3 + 4 * i for i in range(ATTN_LAYERS)}
    for layer in range(GDN_LAYERS + ATTN_LAYERS):
        if layer in attn_ids:
            prefix = f"model.layers.{layer}.self_attn"
            specs[prefix] = main
            specs[f"{prefix}.indexer.raw_key_cache"] = ring
            specs[f"{prefix}.indexer.compressed_key_cache"] = compressed
        else:
            specs[f"model.layers.{layer}.linear_attn"] = gdn
        if layer == 2:
            specs["model.layers.2.ple"] = gdn if ple_joins_gdn else ple
    prefix = "mtp.layers.0.self_attn"
    specs[prefix] = replace(main, is_draft=True)
    specs[f"{prefix}.indexer.raw_key_cache"] = replace(ring, is_draft=True)
    specs[f"{prefix}.indexer.compressed_key_cache"] = replace(compressed, is_draft=True)
    return specs


def plan(
    model_dir,
    block_size: int,
    *,
    leapquant: bool = True,
    pack: bool = False,
    memory: int = KV_BYTES,
    layout: str = "BLNHC",
    ple_joins_gdn: bool = False,
) -> Plan:
    """Plan the KV cache as one serve would: platform alignment, specs, groups,
    config. Flags are read exactly where production reads them."""
    with serving_env(pack=pack):
        vllm_config = make_vllm_config(
            model_dir, block_size, layout, leapquant=leapquant
        )
        with mock.patch.object(
            ModelRegistry,
            "resolve_model_cls",
            lambda *args, **kwargs: (_FlashNextModel, "Qwen4ExpForCausalLM"),
        ):
            Platform._align_hybrid_block_size(vllm_config, _RdnaStyleBackend)
        specs = flash_next_specs(vllm_config, ple_joins_gdn=ple_joins_gdn)
        groups = kv_cache_planning.get_global_kv_cache_groups(vllm_config, [specs])
        (config,) = kv_cache_planning.get_kv_cache_configs(
            vllm_config, [specs], [memory], groups
        )
        return Plan(
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


def layout_fingerprint(config: KVCacheConfig) -> dict:
    """What the workers and the scheduler read from a KV-cache config."""
    return {
        "num_blocks": config.num_blocks,
        "groups": [
            (len(g.layer_names), g.kv_cache_spec.block_size)
            for g in config.kv_cache_groups
        ],
        "tensors": [
            (len(t.layers), t.layer_stride, t.block_stride, t.offset, t.size)
            for t in config.kv_cache_tensors
        ],
    }


def capacity_table(model_dir) -> str:
    """The markdown table of the design note, from the real planner."""
    rows = [
        (
            "| Block | Mode | GDN groups | Blocks per request (GDN+PLE+ring / attn) "
            "| Blocks | Tokens at 262,144 | Concurrency | 1k | 4k | 32k | 8 requests |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def row(label, p: Plan):
        need = p.blocks_per_request()
        gdn = [len(g.layer_names) for g in p.gdn_groups()]
        attn = max(need)
        tokens, conc = p.capacity()
        rows.append(
            f"| {p.block_size} | {label} | {len(gdn)} x {gdn[0]}"
            f"{'' if len(set(gdn)) == 1 else '+'} "
            f"| {sum(need) - attn} / {attn} | {p.config.num_blocks} "
            f"| {tokens:,} | {conc:.4f} "
            f"| {p.capacity(1024)[0]:,} | {p.capacity(4096)[0]:,} "
            f"| {p.capacity(32768)[0]:,} | {p.longest_context(8):,} |"
        )

    row("FP32 (flag off)", plan(model_dir, 1024, leapquant=False))
    for block in BLOCK_SIZES:
        row("LeapQuant padded (PR 1)", plan(model_dir, block))
        row("LeapQuant packed", plan(model_dir, block, pack=True))
    for block in (512, 640):
        row(
            "what-if: PLE in a GDN group (not built)",
            plan(model_dir, block, pack=True, ple_joins_gdn=True),
        )
    return "\n".join(rows)


def fixed_cost_table(model_dir) -> str:
    """Where the non-attention blocks of one 262,144-token request go."""
    rows = [
        "| Block | Mode | Owner | Block ids | Bytes held | Bytes used | Used |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, p in (
        ("LeapQuant padded (PR 1)", plan(model_dir, 512)),
        ("LeapQuant packed", plan(model_dir, 640, pack=True)),
    ):
        cost = p.fixed_cost()
        total = tuple(map(sum, zip(*cost.values())))
        for owner, (blocks, held, used) in [*cost.items(), ("total", total)]:
            rows.append(
                f"| {p.block_size} | {label} | {owner} | {blocks} | {held:,} "
                f"| {used:,} | {used / held:.1%} |"
            )
    return "\n".join(rows)


def write_stub_model(directory) -> None:
    """A one-head, 256-wide model config: its KV page is one QSA main page."""
    OPTConfig(
        architectures=["OPTForCausalLM"],
        hidden_size=256,
        num_attention_heads=1,
        ffn_dim=256,
        word_embed_proj_dim=256,
        num_hidden_layers=1,
        max_position_embeddings=MAX_MODEL_LEN,
    ).save_pretrained(directory)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("flash_next_stub")
    write_stub_model(directory)
    return directory


def attention_block_bytes(block_size: int) -> int:
    """Bytes of one block of the attention group: the layers' main and
    compressed-key pages (the compressed key is one 128-wide row per 4 tokens)."""
    layers = ATTN_LAYERS + 1  # plus the MTP draft layer
    compressed = ATTN_BYTES_PER_TOKEN // 16
    return layers * block_size * (ATTN_BYTES_PER_TOKEN + compressed)


def _golden(num_blocks, size, block_size, comp_page):
    """Layout of a flag-off plan, recorded from the base commit 7b4aff7fe3
    (the first LeapQuant PR) before this change."""
    page = block_size * ATTN_BYTES_PER_TOKEN
    stride = attention_block_bytes(block_size)
    return {
        "num_blocks": num_blocks,
        "groups": [(12, block_size)] * 3 + [(1, block_size), (26, block_size), (13, 8)],
        "tensors": [(12, page, stride, 0, size)] * 3
        + [
            (1, page, stride, 0, size),
            (13, page, stride, 0, size),
            (13, comp_page, stride, 13 * page, size),
            (13, 2048, stride, 0, size),
        ],
    }


BASE_LAYOUTS = {
    # (block size, LeapQuant): padded GDN pages, one block per attention page.
    (512, True): _golden(556, 4_026_400_768, 512, 32_768),
    (640, True): _golden(444, 4_019_159_040, 640, 40_960),
    (1024, False): _golden(278, 4_026_400_768, 1024, 65_536),
}


def test_pack_flag_needs_leapquant():
    def cache(state_dtype: str) -> CacheConfig:
        return CacheConfig(replayssm_state_dtype=state_dtype)

    with flags(pack=False):
        assert leapquant_gdn.leapquant_gdn_pack_enabled(cache("auto")) is False
        assert leapquant_gdn.leapquant_gdn_pack_enabled(cache("leapquant")) is False
    with flags(pack=True):
        assert leapquant_gdn.leapquant_gdn_pack_enabled(cache("leapquant")) is True
    with (
        flags(pack=True),
        pytest.raises(
            ValueError,
            match="VLLM_RDNA_GDN_LEAPQUANT_PACK=1 needs --replayssm-state-dtype "
            "leapquant",
        ),
    ):
        leapquant_gdn.leapquant_gdn_pack_enabled(cache("auto"))


def test_pack_switch_without_the_state_dtype_stops_the_platform(model_dir):
    """The refusal is on the serving path: the platform's block alignment reads
    the switch against the config."""
    with pytest.raises(ValueError, match="needs --replayssm-state-dtype leapquant"):
        plan(model_dir, 640, leapquant=False, pack=True)


def test_state_page_sizes(model_dir):
    for leapquant, page in ((True, LQ_PAGE), (False, FP32_PAGE)):
        with serving_env(pack=False):
            vllm_config = make_vllm_config(model_dir, 1024, leapquant=leapquant)
            gdn = _FlashNextModel.gdn_layer(vllm_config)
            assert gdn.spec(vllm_config).page_size_bytes == page
    assert LQ_PAGE == 469_888
    assert FP32_PAGE == 812_032
    with serving_env(pack=False):
        vllm_config = make_vllm_config(model_dir, 512)
        ple = _FlashNextModel.ple_layer(vllm_config).spec(vllm_config)
        assert ple.page_size_bytes == 225_280


@pytest.mark.parametrize(
    ("block_size", "tokens", "num_blocks", "need"),
    [
        # Real boot, 512-token LeapQuant blocks: "GPU KV cache size: 273,456".
        (512, 273_456, 556, [5, 5, 5, 5, 512, 1]),
        # Real boot, 1024-token FP32 blocks: "GPU KV cache size: 266,945".
        (1024, 266_945, 278, [4, 4, 4, 4, 256, 1]),
    ],
)
def test_flag_off_reproduces_the_real_boot_logs(
    model_dir, block_size, tokens, num_blocks, need
):
    p = plan(model_dir, block_size, leapquant=block_size == 512)
    assert [len(g.layer_names) for g in p.config.kv_cache_groups] == [
        12,
        12,
        12,
        1,
        26,
        13,
    ]
    assert p.config.num_blocks == num_blocks
    assert p.blocks_per_request() == need
    assert p.capacity()[0] == tokens


@pytest.mark.parametrize(("block_size", "leapquant"), sorted(BASE_LAYOUTS))
def test_flag_off_layout_is_identical_to_the_base_branch(
    model_dir, block_size, leapquant
):
    p = plan(model_dir, block_size, leapquant=leapquant)
    assert p.mamba_page_size_padded == block_size * ATTN_BYTES_PER_TOKEN
    assert layout_fingerprint(p.config) == BASE_LAYOUTS[(block_size, leapquant)]


def _packing_logs(log) -> list[tuple]:
    """The arguments of the page-packing log calls of one plan."""
    return [
        call.args[1:] for call in log.call_args_list if "page packing" in call.args[0]
    ]


@pytest.mark.parametrize(
    ("block_size", "groups"), [(512, 3), (576, 3), (608, 2), (640, 2), (1024, 2)]
)
def test_group_count_follows_the_layers_one_block_holds(model_dir, block_size, groups):
    """Two groups of 18 layers need a block of 8.46 MB: 598 tokens or more.
    A 512-token block holds 15 layers, so the recipe stays at 3 groups."""
    block = attention_block_bytes(block_size)
    assert -(-GDN_LAYERS // (block // LQ_PAGE)) == groups
    with mock.patch.object(kv_cache_planning.logger, "info_once") as log:
        packed = plan(model_dir, block_size, pack=True)
    # The boot log tells how the layers were split, also when nothing is gained.
    assert (GDN_LAYERS, LQ_PAGE, groups, block // LQ_PAGE) in _packing_logs(log)
    assert len(packed.gdn_groups()) == groups
    for group in packed.gdn_groups():
        assert len(group.layer_names) * LQ_PAGE <= block
    # One request keeps 5 state blocks per group: 1 live, 2 unresolved, 2 MTP.
    assert packed.gdn_blocks_per_request() == 5 * groups


def test_packing_changes_nothing_at_the_512_recipe(model_dir):
    with mock.patch.object(kv_cache_planning.logger, "info_once") as log:
        padded = plan(model_dir, 512)
    assert _packing_logs(log) == [], "the flag is off: no packing log"
    packed = plan(model_dir, 512, pack=True)
    assert packed.config.num_blocks == padded.config.num_blocks == 556
    assert packed.blocks_per_request() == padded.blocks_per_request()
    assert packed.capacity() == padded.capacity()
    assert packed.capacity()[0] == 273_456


def test_unpadded_pages_alone_free_no_capacity(model_dir):
    """The planner change makes the gain, not the missing padding: the old
    repeat-pattern split still makes 3 groups of unpadded pages."""
    padded = plan(model_dir, 1024)
    unpadded = plan(model_dir, 1024, pack=True)
    with serving_env(pack=False):
        groups = kv_cache_planning.get_global_kv_cache_groups(
            unpadded.vllm_config, [unpadded.specs]
        )
        (config,) = kv_cache_planning.get_kv_cache_configs(
            unpadded.vllm_config, [unpadded.specs], [KV_BYTES], groups
        )
    assert [len(g.layer_names) for g in config.kv_cache_groups] == [
        12,
        12,
        12,
        1,
        26,
        13,
    ]
    assert config.num_blocks == padded.config.num_blocks


def test_packing_leaves_buckets_without_the_leapquant_format_alone(model_dir):
    """The switch packs LeapQuant buckets only: the same unpadded specs without
    ``state_format`` keep the repeat-pattern split (3 groups of 12), the stamped
    ones pack to 2 groups of 18."""
    packed = plan(model_dir, 640, pack=True)
    plain_specs = {
        name: replace(spec, state_format=None) if isinstance(spec, MambaSpec) else spec
        for name, spec in packed.specs.items()
    }
    assert any(
        isinstance(s, MambaSpec) and s.state_format for s in packed.specs.values()
    )
    with serving_env(pack=True):
        stamped = kv_cache_planning.get_global_kv_cache_groups(
            packed.vllm_config, [packed.specs]
        )
        plain = kv_cache_planning.get_global_kv_cache_groups(
            packed.vllm_config, [plain_specs]
        )

    def gdn_sizes(groups):
        return sorted(len(g.layer_names) for g in groups if _gdn_group(g))

    assert gdn_sizes(stamped) == [18, 18]
    assert gdn_sizes(plain) == [12, 12, 12]


# block size: (tokens padded, tokens packed), at 262,144 tokens per request.
PACKED_CAPACITY = {
    608: (270_824, 273_846),
    640: (270_050, 273_220),
    1024: (263_090, 267_926),
}


@pytest.mark.parametrize("block_size", sorted(PACKED_CAPACITY))
def test_packing_cuts_gdn_blocks_and_adds_capacity(model_dir, block_size):
    padded = plan(model_dir, block_size)
    packed = plan(model_dir, block_size, pack=True)
    block = attention_block_bytes(block_size)

    # The attention group sets the block; both plans share one pool geometry.
    assert block_size * ATTN_BYTES_PER_TOKEN % 16 == 0
    assert packed.config.num_blocks == padded.config.num_blocks == KV_BYTES // block
    assert {t.block_stride for t in packed.config.kv_cache_tensors} == {block}

    # 36 layers go from 3 groups of 12 padded pages to the fewest groups that fit.
    assert [len(g.layer_names) for g in padded.gdn_groups()] == [12, 12, 12]
    sizes = sorted(len(g.layer_names) for g in packed.gdn_groups())
    assert sizes == [18, 18]
    assert sizes[-1] * LQ_PAGE <= block < GDN_LAYERS * LQ_PAGE
    placed = [n for g in packed.gdn_groups() for n in g.layer_names]
    assert len(placed) == len(set(placed)) == GDN_LAYERS

    # Each group holds one state block per resident state of every request.
    assert (padded.gdn_blocks_per_request(), packed.gdn_blocks_per_request()) == (
        15,
        10,
    )

    tokens_padded, tokens_packed = PACKED_CAPACITY[block_size]
    assert padded.capacity()[0] == tokens_padded
    assert packed.capacity()[0] == tokens_packed
    assert tokens_packed > tokens_padded


def test_packing_keeps_speculative_slots_and_checkpoints(model_dir):
    packed = plan(model_dir, 640, pack=True)
    for group in packed.gdn_groups():
        for spec in iter_layer_specs(group.kv_cache_spec):
            assert spec.page_size_bytes == LQ_PAGE
            assert spec.num_speculative_blocks == NUM_SPEC
            assert spec.num_prefill_checkpoint_blocks == 0
            assert spec.mamba_cache_mode == "align"
    assert packed.mamba_page_size_padded is None


def test_layers_without_block_outermost_packing_still_pad(model_dir):
    """A layer-outermost layout cannot pack: the unified-page path pads the
    unpadded pages back, so the flag changes nothing there."""
    padded = plan(model_dir, 512, layout="LBNHC")
    packed = plan(model_dir, 512, pack=True, layout="LBNHC")
    assert packed.mamba_page_size_padded is None
    assert len(packed.config.kv_cache_groups) == len(padded.config.kv_cache_groups)
    assert packed.config.num_blocks == padded.config.num_blocks
    assert packed.capacity() == padded.capacity()


@pytest.mark.parametrize("pack", [False, True])
@pytest.mark.parametrize("block_size", [512, 640, 1024])
def test_capacity_log_math(model_dir, block_size, pack):
    p = plan(model_dir, block_size, pack=pack)
    need = p.blocks_per_request()
    groups = p.scheduler_config.kv_cache_groups
    slots = 1 + 2 + NUM_SPEC  # live state, two unresolved batches, MTP2 slots
    for group, blocks in zip(groups, need):
        if isinstance(group.kv_cache_spec, MambaSpec):
            assert blocks == slots
    attention = [
        n
        for g, n in zip(groups, need)
        if isinstance(g.kv_cache_spec, FullAttentionSpec)
    ]
    assert attention == [-(-MAX_MODEL_LEN // block_size)]

    num_blocks = KV_BYTES // attention_block_bytes(block_size)
    assert p.config.num_blocks == num_blocks
    tokens, concurrency = p.capacity()
    assert concurrency == num_blocks / sum(need)
    assert tokens == int(concurrency * MAX_MODEL_LEN)

    with (
        serving_env(pack=pack),
        mock.patch.object(kv_cache_utils.logger, "info_once") as log,
    ):
        kv_cache_utils.update_kv_cache_capacity(p.vllm_config, p.scheduler_config)
    _, _, shown_tokens, shown_len, shown_concurrency = log.call_args.args
    assert (shown_tokens, shown_len) == (f"{tokens:,}", "262,144")
    assert shown_concurrency == concurrency
    assert p.vllm_config.cache_config.kv_cache_size_tokens == tokens


def _bind_group(p: Plan, config: KVCacheConfig, kv_caches, group):
    """Bind every GDN layer of ``group`` like ``bind_kv_cache_to_layers``."""
    layers = []
    with serving_env(pack=p.pack):
        for name in group.layer_names:
            layer = _FlashNextModel.gdn_layer(p.vllm_config)
            layer.bind(kv_caches[name])
            layers.append(layer)
    return layers


@pytest.mark.parametrize("block_size", [640, 1024])
def test_packed_views_are_aligned_disjoint_and_inside_one_block(model_dir, block_size):
    p = plan(model_dir, block_size, pack=True)
    block = attention_block_bytes(block_size)
    num_blocks = 4
    with serving_env(pack=True):
        config = kv_cache_planning.get_kv_cache_config_from_groups(
            p.vllm_config, p.groups, num_blocks * block
        )
        assert config.num_blocks == num_blocks
        with mock.patch("vllm.v1.worker.utils.current_platform", CpuPlatform()):
            kv_caches = allocate_kv_cache(
                config,
                torch.device("cpu"),
                p.vllm_config.cache_config.get_resolved_kv_cache_layout(),
            )
    gdn_groups = [g for g in config.kv_cache_groups if _gdn_group(g)]
    assert sorted(len(g.layer_names) for g in gdn_groups) == [18, 18]
    base = kv_caches[gdn_groups[0].layer_names[0]].untyped_storage().data_ptr()

    conv_bytes = 2560 * 5 * 2
    blob_bytes = 12 * 37_024
    assert conv_bytes + blob_bytes == LQ_PAGE
    for group in gdn_groups:
        layers = _bind_group(p, config, kv_caches, group)
        ranges = []
        for index, (name, layer) in enumerate(zip(group.layer_names, layers)):
            conv, blob = layer.kv_cache
            page = kv_caches[name]
            # One layer page per block, one layer after the other in the block.
            assert page.shape == (num_blocks, 1, 1, LQ_PAGE)
            assert page.data_ptr() - base == index * LQ_PAGE
            assert page.stride(0) == block
            # The gdn_lq_* kernels check exactly this (check_blob).
            assert blob.dtype == torch.uint8 and blob.shape == (num_blocks, 12, 37_024)
            assert blob.stride(2) == 1
            assert blob.data_ptr() % 16 == 0
            assert blob.stride(0) % 16 == 0 and blob.stride(1) % 16 == 0
            assert conv.data_ptr() % 2 == 0 and conv.stride(0) * 2 == block
            for b in range(num_blocks):
                window = (base + b * block, base + (b + 1) * block)
                conv_start = conv[b].data_ptr()
                ranges.append((conv_start, conv_start + conv_bytes, window))
                for head in range(12):
                    start = blob[b, head].data_ptr()
                    ranges.append((start, start + 37_024, window))
        # Inside the block of its own index, and no two ranges share a byte.
        for start, end, (low, high) in ranges:
            assert low <= start < end <= high
        ordered = sorted((start, end) for start, end, _ in ranges)
        assert all(
            a_end <= b_start for (_, a_end), (b_start, _) in zip(ordered, ordered[1:])
        )
        used = len(group.layer_names) * LQ_PAGE
        assert used <= block
    # Groups overlay the same bytes; a block id belongs to one group at a time.
    first, second = gdn_groups
    assert (
        kv_caches[first.layer_names[0]].data_ptr()
        == kv_caches[second.layer_names[0]].data_ptr()
    )


def test_normalize_group_offsets_follow_the_packed_stride(model_dir):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        _LqNormalizeGroup,
    )

    p = plan(model_dir, 640, pack=True)
    block = attention_block_bytes(640)
    with serving_env(pack=True):
        config = kv_cache_planning.get_kv_cache_config_from_groups(
            p.vllm_config, p.groups, 2 * block
        )
        with mock.patch("vllm.v1.worker.utils.current_platform", CpuPlatform()):
            kv_caches = allocate_kv_cache(
                config,
                torch.device("cpu"),
                p.vllm_config.cache_config.get_resolved_kv_cache_layout(),
            )
    (group, _) = [g for g in config.kv_cache_groups if _gdn_group(g)]
    layers = _bind_group(p, config, kv_caches, group)
    normalize = _LqNormalizeGroup(layers)
    # One launch per group: every layer's blob is a whole number of unpadded
    # pages after the leader's, and each offset keeps the 16-byte alignment.
    assert normalize.offsets.tolist() == [i * LQ_PAGE for i in range(18)]
    assert all(offset % 16 == 0 for offset in normalize.offsets.tolist())


def test_fixed_cost_of_one_request_by_owner(model_dir):
    """The PLE and ring blocks of a request are almost empty: block ids, not
    bytes, are what they cost. GDN blocks are full to the slack of the block."""
    gdn_state = GDN_LAYERS * LQ_PAGE * 5
    ple_state = 5 * PLE_PAGE
    block_512, block_640 = attention_block_bytes(512), attention_block_bytes(640)
    assert plan(model_dir, 512).fixed_cost() == {
        "GDN": (15, 15 * block_512, gdn_state),
        "PLE": (5, 5 * block_512, ple_state),
        "ring": (1, block_512, RING_BYTES),
    }
    assert plan(model_dir, 640, pack=True).fixed_cost() == {
        "GDN": (10, 10 * block_640, gdn_state),
        "PLE": (5, 5 * block_640, ple_state),
        "ring": (1, block_640, RING_BYTES),
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        write_stub_model(directory)
        print(capacity_table(directory))
        print()
        print(fixed_cost_table(directory))
    sys.exit(0)
