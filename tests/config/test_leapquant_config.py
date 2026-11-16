# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--replayssm-state-dtype leapquant``: the flag, its one validator, and the
state format identity. CPU only; no GPU and no model weights.

The rejection list follows the safety section of the upstream LeapQuant RFC
(vllm-project/vllm#64725, lihaoyang-amd). The flag name follows draft
vllm-project/vllm#52962 (askliar). The tests are original to this fork.
"""

import dataclasses

import pytest
import torch
from transformers import OPTConfig, Qwen3NextConfig

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    MambaConfig,
    VllmConfig,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    get_conv_copy_spec,
    get_leapquant_copy_spec,
    get_temporal_copy_spec,
)
from vllm.model_executor.layers.mamba.ops import leapquant_gdn as lq
from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.mamba_utils import select_state_copy_funcs

pytestmark = pytest.mark.cpu_test

GDN = MambaAttentionBackendEnum.GDN_ATTN
SHORT_CONV = MambaAttentionBackendEnum.SHORT_CONV
# MTP2 serving layout: the format string is the contract of the state bytes.
SERVING_FORMAT = "leapquant-int8/v1/k128v128/r4/p32/cap34/rec-float16"
CONNECTOR = KVTransferConfig(kv_connector="ExampleConnector", kv_role="kv_both")


@pytest.fixture
def gfx10x(monkeypatch):
    """Pretend to run on RDNA2 with the HIP kernel set, whatever the host is."""
    monkeypatch.setattr("vllm.platforms.rdna.on_rdna2_hip", lambda: True)


@pytest.fixture
def v2_runner(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")


@pytest.fixture
def v1_runner(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")


@pytest.fixture
def make_model(tmp_path):
    """Save a tiny Qwen3-Next config. A fresh directory per call: ModelConfig
    applies the model-specific config update once per ModelConfig object."""

    def make(head_dim=128):
        path = tmp_path / f"qwen3-next-{len(list(tmp_path.iterdir()))}"
        Qwen3NextConfig(
            architectures=["Qwen3NextForCausalLM"],
            vocab_size=512,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=head_dim,
            linear_value_head_dim=head_dim,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            max_position_embeddings=4096,
        ).save_pretrained(path)
        return str(path)

    return make


def engine_config(model, **kwargs):
    """The EngineArgs path: the CLI flag, the config, then the validator."""
    kwargs.setdefault("enable_prefix_caching", True)
    return EngineArgs(
        model=model,
        dtype="float16",
        skip_tokenizer_init=True,
        max_model_len=2048,
        **kwargs,
    ).create_engine_config()


def make_vllm_config(*, cache=None, **kwargs):
    """A config without a model, for the rules that do not read the model."""
    return VllmConfig(
        cache_config=CacheConfig(**(cache or {})),
        device_config=DeviceConfig("cpu"),
        **kwargs,
    )


def make_spec(**kwargs):
    return MambaSpec(
        block_size=512,
        shapes=((5, 2560), (12, 37024)),
        dtypes=(torch.float16, torch.uint8),
        mamba_type=GDN,
        mamba_cache_mode="align",
        num_speculative_blocks=2,
        **kwargs,
    )


# --- the flag ------------------------------------------------------------


def test_flag_defaults_to_auto_and_rejects_other_values():
    assert CacheConfig().replayssm_state_dtype == "auto"
    assert CacheConfig(replayssm_state_dtype="leapquant").replayssm_state_dtype == (
        "leapquant"
    )
    with pytest.raises(ValueError, match="'auto' or 'leapquant'"):
        CacheConfig(replayssm_state_dtype="fp8")


def test_cli_flag_parses_and_restricts_its_values():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--model", "dummy"])
    assert EngineArgs.from_cli_args(args).replayssm_state_dtype == "auto"
    args = parser.parse_args(
        ["--model", "dummy", "--replayssm-state-dtype", "leapquant"]
    )
    assert EngineArgs.from_cli_args(args).replayssm_state_dtype == "leapquant"
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "dummy", "--replayssm-state-dtype", "fp8"])


def test_flag_is_not_a_compile_cache_factor():
    # The flag picks HIP ops inside the opaque GDN layer op and the page
    # layout. It does not change the compiled graph.
    assert (
        CacheConfig().compute_hash()
        == CacheConfig(replayssm_state_dtype="leapquant").compute_hash()
    )


# --- the supported setup ---------------------------------------------------


def test_supported_setup_stores_the_gdn_state_as_blobs(gfx10x, v2_runner, make_model):
    config = engine_config(make_model(), replayssm_state_dtype="leapquant")
    assert config.cache_config.replayssm_state_dtype == "leapquant"
    # Prefix caching on a hybrid model: align mode, which needs the V2 runner.
    assert config.cache_config.mamba_cache_mode == "align"

    # The model classmethods feed the page size unification.
    _, temporal_shape = Qwen3NextForCausalLM.get_mamba_state_shape_from_config(config)
    _, temporal_dtype = Qwen3NextForCausalLM.get_mamba_state_dtype_from_config(config)
    assert temporal_shape == (4, lq.leapquant_gdn_layout(128, 128, 0).nbytes)
    assert temporal_dtype == torch.uint8


def test_mtp_stays_supported(gfx10x, v2_runner, make_model):
    # The RFC rejects speculative decoding. This fork keeps MTP.
    config = engine_config(
        make_model(),
        replayssm_state_dtype="leapquant",
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
    )
    assert config.speculative_config.num_speculative_tokens == 2
    _, temporal_shape = Qwen3NextForCausalLM.get_mamba_state_shape_from_config(config)
    assert temporal_shape == (4, 37024)

    # A drafter gets its own VllmConfig that holds the draft model config
    # (the Qwen4Exp MTP model builds it this way). The validator runs again.
    drafter = dataclasses.replace(
        config, model_config=config.speculative_config.draft_model_config
    )
    assert drafter.cache_config.replayssm_state_dtype == "leapquant"


def test_validator_allows_the_v1_runner_without_align_mode(
    gfx10x, v1_runner, make_model
):
    config = engine_config(
        make_model(), replayssm_state_dtype="leapquant", enable_prefix_caching=False
    )
    assert config.cache_config.mamba_cache_mode == "none"


# --- the refused setups ------------------------------------------------------


def test_refuses_a_host_that_is_not_gfx10x(monkeypatch):
    monkeypatch.setattr("vllm.platforms.rdna.on_rdna2_hip", lambda: False)
    with pytest.raises(ValueError, match=r"RDNA2 \(gfx10x\)") as error:
        make_vllm_config(cache={"replayssm_state_dtype": "leapquant"})
    assert "VLLM_RDNA_ALL_HIP=1" in str(error.value)


def test_refuses_use_replayssm(gfx10x):
    with pytest.raises(ValueError, match="--use-replayssm"):
        make_vllm_config(
            cache={"replayssm_state_dtype": "leapquant", "use_replayssm": True}
        )


def test_refuses_stochastic_rounding(gfx10x):
    # MambaConfig refuses stochastic rounding off CUDA when it is built, so the
    # LeapQuant rule only shows for a config changed after that.
    mamba_config = MambaConfig()
    mamba_config.enable_stochastic_rounding = True
    with pytest.raises(ValueError, match="stochastic-rounding"):
        make_vllm_config(
            cache={"replayssm_state_dtype": "leapquant"}, mamba_config=mamba_config
        )


def test_refuses_a_kv_connector(gfx10x):
    with pytest.raises(ValueError, match="KV connector"):
        make_vllm_config(
            cache={"replayssm_state_dtype": "leapquant"}, kv_transfer_config=CONNECTOR
        )


def test_refuses_offloading_before_its_connector_exists(gfx10x):
    # kv_offloading_size becomes a connector only late in VllmConfig.__post_init__.
    with pytest.raises(ValueError, match="offloading"):
        make_vllm_config(
            cache={"replayssm_state_dtype": "leapquant", "kv_offloading_size": 1.0}
        )


@pytest.mark.parametrize(
    "predicate",
    ["are_gdn_triton_kernels_available", "is_rdna_gdn_triton_kernels_available"],
)
def test_refuses_the_aiter_gdn_path(gfx10x, monkeypatch, predicate):
    monkeypatch.setattr(rocm_aiter_ops, predicate, lambda: True)
    with pytest.raises(ValueError, match="AITER"):
        make_vllm_config(cache={"replayssm_state_dtype": "leapquant"})


def test_refuses_align_mode_on_the_v1_runner(gfx10x, v1_runner, make_model):
    # `mamba_cache_mode` becomes "align" inside the model-specific config
    # update, so the validator must run after that update.
    with pytest.raises(ValueError, match="V2 model runner"):
        engine_config(make_model(), replayssm_state_dtype="leapquant")


def test_refuses_head_dims_other_than_128(gfx10x, v2_runner, make_model):
    with pytest.raises(
        ValueError, match="head_k_dim == head_v_dim == 128, got 64 / 64"
    ):
        engine_config(make_model(head_dim=64), replayssm_state_dtype="leapquant")


def test_refuses_a_model_without_gdn_layers(gfx10x, v2_runner, tmp_path):
    OPTConfig(architectures=["OPTForCausalLM"]).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="Gated DeltaNet"):
        engine_config(str(tmp_path), replayssm_state_dtype="leapquant")


@pytest.mark.parametrize(
    ("facts", "message"),
    [
        ({"fused_cuda_spec_decode": True}, "fused CUDA"),
        ({"flashinfer_checkpoints": True}, "FlashInfer"),
        ({"head_dims": (128, 64)}, "got 128 / 64"),
    ],
    ids=["fused-cuda", "flashinfer", "head-dim"],
)
def test_refuses_what_only_the_layer_knows(gfx10x, facts, message):
    # Each LeapQuant layer runs the validator again with the facts that exist
    # once the layer is built.
    config = make_vllm_config()
    with pytest.raises(ValueError, match=message) as error:
        lq.verify_leapquant_gdn(config, **facts)
    assert str(error.value).startswith(lq.LEAPQUANT_FLAG)


@pytest.mark.parametrize(
    ("cache", "kv_transfer_config"),
    [
        pytest.param({"use_replayssm": True}, None, id="replayssm"),
        pytest.param({"kv_offloading_size": 1.0}, None, id="offloading"),
        pytest.param({}, CONNECTOR, id="connector"),
    ],
)
def test_config_refusal_names_the_flag(gfx10x, cache, kv_transfer_config):
    # No silent fallback: a refused setup stops the start-up and says why.
    with pytest.raises(
        ValueError, match="Value error, --replayssm-state-dtype leapquant"
    ):
        make_vllm_config(
            cache={"replayssm_state_dtype": "leapquant", **cache},
            kv_transfer_config=kv_transfer_config,
        )


def test_auto_accepts_what_leapquant_refuses(v1_runner):
    # The validator runs only for --replayssm-state-dtype leapquant. (Triton
    # ReplaySSM needs the V1 runner: that is a check of its own.)
    make_vllm_config(cache={"use_replayssm": True})
    make_vllm_config(cache={"kv_offloading_size": 1.0})
    make_vllm_config(kv_transfer_config=CONNECTOR)


def test_cli_flag_reaches_the_validator(gfx10x, tmp_path):
    # EngineArgs -> CacheConfig -> VllmConfig: the refusal names the flag.
    OPTConfig(architectures=["OPTForCausalLM"]).save_pretrained(tmp_path)
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--model",
            str(tmp_path),
            "--skip-tokenizer-init",
            "--replayssm-state-dtype",
            "leapquant",
        ]
    )
    with pytest.raises(ValueError, match="--replayssm-state-dtype leapquant"):
        EngineArgs.from_cli_args(args).create_engine_config()


# --- auto keeps the model's own state -------------------------------------


def test_auto_keeps_the_gdn_state_dtype_and_shape():
    auto = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        torch.float16, "auto", "auto"
    )
    assert auto == MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        torch.float16, "auto", "auto", replayssm_state_dtype="auto"
    )
    assert MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        torch.float16, "auto", "auto", replayssm_state_dtype="leapquant"
    ) == (auto[0], torch.uint8)
    assert auto[1] != torch.uint8

    # Flash-Next TP4: 12 value heads per rank.
    conv, temporal = MambaStateShapeCalculator.gated_delta_net_state_shape(
        4, 16, 48, 128, 128, 4, 2
    )
    assert temporal == (12, 128, 128)
    conv_lq, temporal_lq = MambaStateShapeCalculator.gated_delta_net_state_shape(
        4, 16, 48, 128, 128, 4, 2, replayssm_state_dtype="leapquant"
    )
    assert conv_lq == conv
    assert temporal_lq == (12, 37024)


def test_state_shape_refuses_head_dims_other_than_128():
    with pytest.raises(ValueError, match="head_k_dim == head_v_dim == 128"):
        MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, 4, 8, 64, 64, 4, 0, replayssm_state_dtype="leapquant"
        )


def test_max_memory_usage_follows_the_flag_not_the_environment(
    gfx10x, v2_runner, monkeypatch
):
    monkeypatch.setenv("VLLM_RDNA_GDN_LEAPQUANT", "1")  # the removed switch
    spec = make_spec()
    page = spec.page_size_bytes

    def usage(state_dtype):
        config = make_vllm_config(
            cache={"mamba_cache_mode": "align", "replayssm_state_dtype": state_dtype}
        )
        return spec.max_memory_usage_bytes(config), config.max_concurrent_batches

    auto_bytes, auto_batches = usage("auto")
    leapquant_bytes, leapquant_batches = usage("leapquant")
    assert auto_batches == leapquant_batches > 1  # the flag must make a difference
    # One live state per concurrent batch under LeapQuant, one otherwise.
    assert auto_bytes == page * (1 + 1 + 2)
    assert leapquant_bytes == page * (1 + leapquant_batches + 2)


# --- state format identity ------------------------------------------------


def test_serving_layout_bytes_are_pinned_to_the_layout_version():
    # If this fails, the byte layout changed. Raise LEAPQUANT_LAYOUT_VERSION
    # and update the table, so old and new state are never mixed.
    layout = lq.leapquant_gdn_layout(128, 128, 2)
    assert layout.offsets == {
        "z": 0,
        "bscale": 16384,
        "csmooth": 16896,
        "comp_k": 17408,
        "comp_u": 18432,
        "rec_k": 19456,
        "rec_u": 28160,
        "rec_g": 36864,
        "header": 37008,
        "_total": 37024,
    }
    assert lq.LEAPQUANT_LAYOUT_VERSION == 1
    assert layout.state_format == SERVING_FORMAT


def test_state_format_names_every_field_of_the_layout():
    base = lq.leapquant_gdn_layout(128, 128, 2)
    variants = [
        dataclasses.replace(base, window=16),
        dataclasses.replace(base, num_comp=2),
        dataclasses.replace(base, max_query_len=1),
        dataclasses.replace(base, head_k_dim=64),
        dataclasses.replace(base, head_v_dim=64),
    ]
    assert len({base.state_format, *(v.state_format for v in variants)}) == 6
    # The fold fit changes values, not what a blob holds.
    assert dataclasses.replace(base, power_iters=0).state_format == base.state_format


def test_state_format_changes_with_the_layout_version(monkeypatch):
    before = lq.leapquant_gdn_layout(128, 128, 2).state_format
    monkeypatch.setattr(lq, "LEAPQUANT_LAYOUT_VERSION", 2)
    after = lq.leapquant_gdn_layout(128, 128, 2).state_format
    assert after != before
    assert lq.is_leapquant_state_format(before) and lq.is_leapquant_state_format(after)


def test_is_leapquant_state_format():
    assert lq.is_leapquant_state_format(SERVING_FORMAT)
    assert not lq.is_leapquant_state_format(None)
    assert not lq.is_leapquant_state_format("fp8/v1")
    assert not lq.is_leapquant_state_format("leapquant-int8x/v1")


def test_state_format_is_part_of_the_spec_key():
    fp32, quantized = make_spec(), make_spec(state_format=SERVING_FORMAT)
    assert fp32.state_format is None
    # Same shapes, dtypes and page size: only the identity tells them apart.
    assert fp32.page_size_bytes == quantized.page_size_bytes
    assert fp32 != quantized and len({fp32, quantized}) == 2
    assert fp32 == make_spec(state_format=None)
    assert not fp32.is_uniform_with_collection({"a": quantized})
    assert not quantized.is_uniform_with_collection({"a": fp32})
    assert quantized.is_uniform_with_collection(
        {"a": make_spec(state_format=SERVING_FORMAT)}
    )


def test_layouts_of_equal_size_but_different_version_are_not_uniform():
    # The shapes and dtypes of two layout versions can be the same.
    v1 = make_spec(state_format="leapquant-int8/v1/k128v128/r4/p32/cap34/rec-float16")
    v2 = make_spec(state_format="leapquant-int8/v2/k128v128/r4/p32/cap34/rec-float16")
    assert v1.shapes == v2.shapes and v1.dtypes == v2.dtypes
    assert not v1.is_uniform_with_collection({"a": v2})


def test_layer_stamps_its_state_format_on_the_kv_cache_spec(monkeypatch):
    # The layer is built without its constructor (the GPU tests do the same):
    # only the spec stamping is under test. MambaBase.get_kv_cache_spec returns
    # the unstamped spec of the model's own state.
    base = make_spec()
    monkeypatch.setattr(MambaBase, "get_kv_cache_spec", lambda self, config: base)
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    layer._checkpoint_alignment = None

    layer._lq_layout = None
    assert layer.get_kv_cache_spec(None) is base

    layer._lq_layout = lq.leapquant_gdn_layout(128, 128, 2)
    spec = layer.get_kv_cache_spec(None)
    assert spec.state_format == SERVING_FORMAT
    assert dataclasses.replace(spec, state_format=None) == base


def test_kv_cache_config_lists_the_quantized_state_formats():
    attention = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
    )

    def kv_cache_config(*specs):
        return KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec([f"l{i}"], spec) for i, spec in enumerate(specs)
            ],
        )

    assert kv_cache_config(attention, make_spec()).mamba_state_formats == frozenset()
    quantized = make_spec(state_format=SERVING_FORMAT)
    assert kv_cache_config(attention, quantized).mamba_state_formats == {SERVING_FORMAT}
    # Layer specs inside a uniform wrapper count too.
    wrapped = UniformTypeKVCacheSpecs(
        block_size=512, kv_cache_specs={"gdn0": quantized, "gdn1": quantized}
    )
    assert kv_cache_config(wrapped).mamba_state_formats == {SERVING_FORMAT}


# --- the align copy functions follow the spec, not the config --------------


def test_gdn_copy_funcs_do_not_depend_on_the_config():
    assert MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func() == (
        get_conv_copy_spec,
        get_temporal_copy_spec,
    )


def test_select_state_copy_funcs_marks_leapquant_state_only():
    funcs = {
        GDN: MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func(),
        SHORT_CONV: MambaStateCopyFuncCalculator.short_conv_state_copy_func(),
    }
    ple = MambaSpec(
        block_size=512,
        shapes=((8, 3),),
        dtypes=(torch.float16,),
        mamba_type=SHORT_CONV,
        tp_replicated=True,
    )
    assert select_state_copy_funcs({make_spec(): [0], ple: [1]}, funcs) == funcs

    selected = select_state_copy_funcs(
        {make_spec(state_format=SERVING_FORMAT): [0], ple: [1]}, funcs
    )
    assert selected[GDN] == (get_conv_copy_spec, get_leapquant_copy_spec)
    assert selected[SHORT_CONV] == funcs[SHORT_CONV]
    assert funcs[GDN] == (get_conv_copy_spec, get_temporal_copy_spec)


def test_select_state_copy_funcs_still_validates():
    with pytest.raises(AssertionError, match="missing state copy funcs"):
        select_state_copy_funcs({make_spec(): [0]}, {})
