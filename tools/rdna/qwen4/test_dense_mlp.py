# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the opt-in dense Qwen4 MLP.

The selection rule is stdlib-only and always runs. The module-tree and
weight-name assertions instantiate the decoder on the meta device, so they
need torch and vLLM (a CPU box with the fork installed is enough; no GPU).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SELECTION_PATH = ROOT / "vllm/models/qwen4_exp/common/mlp_selection.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


selection = _load("qwen4_mlp_selection", SELECTION_PATH)


def test_flash_next_keeps_sparse_moe():
    # Published Qwen3.8-Flash-Next text config: 512 routed experts.
    assert not selection.uses_dense_mlp({"num_experts": 512})


def test_missing_experts_means_dense():
    # Qwen3.8-27B (the backbone the leaked 27B keeps) has no num_experts.
    assert selection.uses_dense_mlp({"hidden_size": 5120})


def test_zero_experts_means_dense():
    assert selection.uses_dense_mlp({"num_experts": 0})


def test_explicit_dense_flag_wins():
    assert selection.uses_dense_mlp({"num_experts": 512, "dense_mlp": True})
    assert not selection.uses_dense_mlp({"num_experts": 0, "dense_mlp": False})


def test_object_config_is_supported():
    class Config:
        num_experts = 8

    assert not selection.uses_dense_mlp(Config())
    Config.num_experts = 0
    assert selection.uses_dense_mlp(Config())


@pytest.fixture(scope="module")
def model_parallel_group():
    """Initialize a 1x1 TP/PP group for layer construction.

    vLLM layers call ``get_tensor_model_parallel_world_size`` while they are
    built, which asserts unless the group exists. The meta-device tests need
    one visible GPU only because importing vLLM on ROCm queries the device.
    """
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.config import (
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.utils.network_utils import get_open_port

    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=1))
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1)
        try:
            yield
        finally:
            cleanup_dist_env_and_memory()


def _tiny_text_config(dense: bool):
    from transformers import Qwen4ExpTextConfig

    kwargs = dict(
        vocab_size=128,
        eos_token_id=0,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        output_gate_type="sigmoid",
        num_experts_per_tok=2,
        hc_count=2,
        hc_lowrank=16,
        ple_layer_ids=[],
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
    )
    config = Qwen4ExpTextConfig(**kwargs)
    if dense:
        config.num_experts = 0
        config.dense_mlp = True
    else:
        config.num_experts = 4
    return config


def _build_layer(text_config):
    from types import SimpleNamespace

    import torch

    from vllm.config import (
        EngramConfig,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.models.qwen4_exp.amd import model as qwen4_model

    vllm_config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=1))
    vllm_config.model_config = SimpleNamespace(
        architecture="Qwen4ExpForCausalLM",
        is_moe=True,
        sleep_mode_offload_cudagraph=False,
        hf_text_config=text_config,
        dtype=torch.float16,
    )
    vllm_config.engram_config = EngramConfig(cpu_offload=False)
    with set_current_vllm_config(vllm_config), torch.device("meta"):
        layer = qwen4_model.Qwen4ExpDecoderLayer(
            vllm_config,
            layer_type="linear_attention",
            prefix="model.layers.0",
        )
    return layer, qwen4_model, vllm_config


def test_flash_next_module_tree_is_unchanged(model_parallel_group):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    layer, qwen4_model, vllm_config = _build_layer(_tiny_text_config(dense=False))
    assert not layer.is_dense_mlp
    assert isinstance(layer.mlp, qwen4_model.Qwen4ExpSparseMoeBlock)

    # The selection branch must not change what the MoE path builds.
    import torch

    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config), torch.device("meta"):
        reference = qwen4_model.Qwen4ExpSparseMoeBlock(
            vllm_config,
            prefix="model.layers.0.reference_mlp",
            reduce_results=True,
        )
    assert [name for name, _ in layer.mlp.named_modules()] == [
        name for name, _ in reference.named_modules()
    ]
    assert [name for name, _ in layer.mlp.named_parameters()] == [
        name for name, _ in reference.named_parameters()
    ]


def test_dense_config_builds_dense_mlp(model_parallel_group):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    layer, qwen4_model, _ = _build_layer(_tiny_text_config(dense=True))
    assert layer.is_dense_mlp
    assert isinstance(layer.mlp, qwen4_model.Qwen4DenseMLP)
    assert [name for name, _ in layer.mlp.named_modules()] == [
        "",
        "gate_up_proj",
        "down_proj",
        "act_fn",
    ]
    assert [name for name, _ in layer.mlp.named_parameters()] == [
        "gate_up_proj.weight",
        "down_proj.weight",
    ]


def test_dense_weight_names_map_from_hf():
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.models.qwen4_exp.amd.model import Qwen4ExpForCausalLM

    mapping = Qwen4ExpForCausalLM.packed_modules_mapping
    # HF Qwen3.8-27B ships model.language_model.layers.N.mlp.{gate,up,down}_proj;
    # the prefix mapper strips model.language_model. and this mapping fuses
    # gate/up into the two gate_up_proj shards.
    assert mapping["gate_up_proj"] == ["gate_proj", "up_proj"]
    assert "qkv_proj" in mapping
