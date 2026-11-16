# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real compact factories: mixed segments, native identities and owned copies."""

import hashlib
import json
from math import prod

import pytest
import torch

from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.quantization.mlx_affine import MlxAffineConfig
from vllm.model_executor.layers.quantization.mlx_affine_tuning import MlxAffineTuning


def _native_config(tmp_path, modules, *, view="target"):
    specs = {
        name: {"bits": bits, "group_size": group, "mode": "affine"}
        for name, (_, _, bits, group) in modules.items()
    }
    quant = {"quant_method": "mlx_affine", "format_version": 1, "module_specs": specs}
    config = MlxAffineConfig.from_config(quant)
    tensors = {}
    offset = 0
    for original, (rows, logical_k, bits, group) in modules.items():
        runtime = original
        if view == "mtp":
            runtime = "model." + original.removeprefix("mtp.")
        runtime = runtime.replace(".switch_mlp.", ".experts.")
        for component, dtype, columns, itemsize in (
            ("weight", "U32", (logical_k * bits + 31) // 32, 4),
            ("scales", "BF16", (logical_k + group - 1) // group, 2),
            ("biases", "BF16", (logical_k + group - 1) // group, 2),
        ):
            shape = [rows, columns]
            size = prod(shape) * itemsize
            tensors[f"{original}.{component}"] = {
                "file": "model.safetensors",
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + size],
                "runtime_name": f"{runtime}.{component}",
                "layout": "identity",
            }
            offset += size
    manifest = {
        "format_version": 1,
        "view": view,
        "source_revision": "e171af86f499f1855b0fb71d781105e8dd609610",
        "module_specs_sha256": config.specs.sha256,
        "tensors": tensors,
    }
    root = tmp_path / "native"
    root.mkdir()
    # Construction binds metadata and protected links, never tensor payloads.
    (tmp_path / "original.safetensors").write_bytes(b"not read during construction")
    (root / "model.safetensors").symlink_to("../original.safetensors")
    raw = json.dumps(manifest).encode()
    (root / "mlx-affine-manifest.json").write_bytes(raw)
    hf = {
        "quantization_config": quant,
        "num_hidden_layers": 48,
        "mlx_affine": {
            "manifest": "mlx-affine-manifest.json",
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "view": view,
        },
    }
    config.maybe_update_config(str(root), hf)
    return config


def test_mixed_fused_factory_keeps_precision_and_incremental_copies(
    tmp_path,
    default_vllm_config,
):
    prefix = "language_model.model.layers.0.mlp.shared_expert"
    config = _native_config(
        tmp_path,
        {
            prefix + ".gate_proj": (3, 128, 6, 64),
            prefix + ".up_proj": (5, 128, 8, 128),
        },
    )
    layer = MergedColumnParallelLinear(
        128,
        [3, 5],
        bias=False,
        params_dtype=torch.float16,
        quant_config=config,
        prefix="model." + prefix + ".gate_up_proj",
        disable_tp=True,
    )
    assert [
        (s.spec.bits, s.spec.group_size, s.output_offset, s.output_size)
        for s in layer._mlx_affine_segments
    ] == [(6, 64, 0, 3), (8, 128, 3, 5)]
    assert not hasattr(layer, "weight")
    for binding in layer._mlx_affine_load_bindings:
        entry = config.tensor(binding.original_name)
        dtype = torch.int32 if entry["dtype"] == "U32" else torch.bfloat16
        values = torch.arange(prod(entry["shape"]), dtype=dtype).reshape(entry["shape"])
        source = values.view(torch.uint32) if entry["dtype"] == "U32" else values
        binding.load(source)
        parameter = getattr(layer, binding.loaded_parameters[0])
        assert parameter.dtype == dtype
        assert torch.equal(parameter, values)
    assert len(layer._mlx_affine_load_bindings) == 6


def test_mtp_expert_alias_preserves_original_and_registered_layer_zero(
    tmp_path,
    default_vllm_config,
):
    original = "mtp.layers.0.mlp.switch_mlp.gate_proj"
    config = _native_config(tmp_path, {original: (2, 128, 6, 64)}, view="mtp")
    assert config._resolve("mtp.layers.48.mlp.experts.gate_proj") == original
    assert config._resolve("model.layers.0.mlp.experts.gate_proj") == original
    assert config.resolve_checkpoint_quantization(original).original_prefix == original
    assert (
        config.resolve_checkpoint_quantization("mtp.layers.48.mlp.experts.gate_proj")
        is None
    )


def test_gdn_qkv_source_has_one_binding_and_three_explicit_shard_ids(
    tmp_path,
    default_vllm_config,
):
    prefix = "language_model.model.layers.0.linear_attn"
    config = _native_config(
        tmp_path,
        {
            prefix + ".in_proj_qkv": (10, 128, 6, 64),
            prefix + ".in_proj_z": (6, 128, 8, 128),
        },
    )
    layer = MergedColumnParallelLinear(
        128,
        [2, 2, 6, 6],
        bias=False,
        params_dtype=torch.float16,
        quant_config=config,
        prefix=prefix + ".in_proj_qkvz",
        disable_tp=True,
    )
    assert [part.shard_id for part in layer._mlx_affine_parts] == [0, 1, 2, 3]
    assert [part.source_rows for part in layer._mlx_affine_parts] == [
        (0, 2),
        (2, 4),
        (4, 10),
        (0, 6),
    ]
    binding = next(
        binding
        for binding in layer._mlx_affine_load_bindings
        if binding.original_name == prefix + ".in_proj_qkv.weight"
    )
    assert len(binding.loaded_parameters) == 3
    source = torch.arange(240, dtype=torch.int32).reshape(10, 24)
    binding.load(source.view(torch.uint32))
    for part in layer._mlx_affine_parts[:3]:
        start, stop = part.source_rows
        assert torch.equal(getattr(layer, part.parameters[0]), source[start:stop])


def test_hc_factory_declares_padding_without_synthetic_checkpoint_parameters(
    tmp_path,
    default_vllm_config,
):
    prefix = "language_model.model.layers.0.attn_hyper_connection"
    config = _native_config(
        tmp_path,
        {
            prefix + ".input_mix_weight_down": (320, 10240, 6, 64),
            prefix + ".block_inject_weight": (4, 10240, 8, 128),
        },
    )
    layer = MergedColumnParallelLinear(
        10240,
        [320, 4, 12],
        bias=False,
        params_dtype=torch.float16,
        quant_config=config,
        prefix=prefix + ".input_mix_weight_down_block_inject",
        disable_tp=True,
    )
    assert layer._mlx_affine_zero_padding == (324, 336)
    assert [(s.output_offset, s.output_size) for s in layer._mlx_affine_segments] == [
        (0, 320),
        (320, 4),
    ]
    assert (
        sum(p.shape[0] for n, p in layer.named_parameters() if n.endswith("_weight"))
        == 324
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        True,
        {
            "decode_tile_n": True,
            "prefill_tile_n": 8,
            "prefill_block_m": 4,
            "dense_prefill_block_m": 8,
            "chunk_tokens": 256,
        },
        {
            "decode_tile_n": 8,
            "prefill_tile_n": 16,
            "prefill_block_m": 4,
            "dense_prefill_block_m": 8,
            "chunk_tokens": 256,
        },
        {
            "decode_tile_n": 8,
            "prefill_tile_n": 8,
            "prefill_block_m": 4,
            "dense_prefill_block_m": 8,
            "chunk_tokens": 2048,
        },
        {
            "decode_tile_n": 8,
            "prefill_tile_n": 8,
            "prefill_block_m": 4,
            "dense_prefill_block_m": True,
            "chunk_tokens": 256,
        },
        {
            "decode_tile_n": 8,
            "prefill_tile_n": 8,
            "prefill_block_m": 4,
            "dense_prefill_block_m": 1,
            "chunk_tokens": 256,
        },
        {
            "decode_tile_n": 8,
            "prefill_tile_n": 8,
            "prefill_block_m": 4,
            "chunk_tokens": 256,
        },
        {
            "decode_tile_n": 8,
            "prefill_tile_n": 8,
            "prefill_block_m": 4,
            "dense_prefill_block_m": 8,
            "chunk_tokens": 256,
            "extra": 1,
        },
    ],
)
def test_tuning_rejects_malformed_or_unbounded_variants(value):
    with pytest.raises(ValueError):
        MlxAffineTuning.from_config(value)
