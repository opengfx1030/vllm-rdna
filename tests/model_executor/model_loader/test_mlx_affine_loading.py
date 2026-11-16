# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tensor tests for source admission, owned copies and deferred sharing."""

from functools import partial
from math import prod
from pathlib import Path

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization.mlx_affine import MlxAffineConfig
from vllm.model_executor.layers.quantization.mlx_affine_format import (
    MlxAffineDestination as Destination,
)
from vllm.model_executor.layers.quantization.mlx_affine_format import (
    MlxAffineInterval as Interval,
)
from vllm.model_executor.layers.quantization.mlx_affine_format import (
    MlxAffineShard,
    MlxAffineSpec,
)
from vllm.model_executor.layers.quantization.mlx_affine_format import (
    MlxAffineSourcePlan as SourcePlan,
)
from vllm.model_executor.model_loader.mlx_affine_checkpoint import MlxAffineCheckpoint
from vllm.model_executor.model_loader.mlx_affine_loading import (
    MlxAffineLoadBinding,
    copy_mlx_affine_tensor,
    finish_mlx_affine_sharing,
    is_mlx_affine_loading,
    load_mlx_affine_weights,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.parameter import MlxAffineParameter


def _config(records, specs=None, view="target", layouts=None):
    config = MlxAffineConfig.from_config(
        {
            "quant_method": "mlx_affine",
            "format_version": 1,
            "module_specs": specs or {},
        }
    )
    tensors = {}
    offset = 0
    sizes = {"U32": 4, "BF16": 2, "F16": 2}
    for name, (dtype, shape, runtime_name) in records.items():
        size = prod(shape) * sizes[dtype]
        tensors[name] = {
            "file": "model.safetensors",
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
            "runtime_name": runtime_name,
            "layout": (layouts or {}).get(name, "identity"),
        }
        offset += size
    manifest = {
        "format_version": 1,
        "view": view,
        "source_revision": "e171af86f499f1855b0fb71d781105e8dd609610",
        "module_specs_sha256": config.specs.sha256,
        "tensors": tensors,
    }
    # Manifest pin/path admission is exercised by test_mlx_affine_checkpoint;
    # this suite begins at the real construction-bound descriptor boundary.
    config.checkpoint = MlxAffineCheckpoint(
        Path("/native-view"),
        manifest,
        frozenset(tensors),
        (),
        "a" * 64,
    )
    return config


class _OwnedExperts(nn.Module):
    def __init__(self, local=True):
        super().__init__()
        records = {
            "model.experts.weight": ("U32", (4, 2, 12), "codes"),
            "model.experts.scales": ("BF16", (4, 2, 2), "scales"),
            "model.experts.biases": ("BF16", (4, 2, 2), "biases"),
            "model.norm": ("BF16", (2,), "norm"),
        }
        self.quant_config = _config(
            records,
            {
                "model.experts": {"bits": 6, "group_size": 32, "mode": "affine"},
            },
        )
        self.norm = nn.Parameter(
            torch.empty(2, dtype=torch.float16), requires_grad=False
        )
        self.source_tensors = {
            "model.experts.weight": torch.arange(96, dtype=torch.int32)
            .reshape(4, 2, 12)
            .view(torch.uint32),
            "model.experts.scales": torch.arange(16, dtype=torch.bfloat16).reshape(
                4, 2, 2
            ),
            "model.experts.biases": -torch.arange(16, dtype=torch.bfloat16).reshape(
                4, 2, 2
            ),
            "model.norm": torch.tensor([2, 3], dtype=torch.bfloat16),
        }
        spec = self.quant_config.specs.module_specs["model.experts"]
        shard = MlxAffineShard.from_interval(spec, 0, 64, logical_k=64)
        bindings = []
        for suffix, parameter_name in (
            ("weight", "codes"),
            ("scales", "scales"),
            ("biases", "biases"),
        ):
            name = "model.experts." + suffix
            shape = records[name][1]
            if local:
                parameter = MlxAffineParameter(
                    torch.empty(
                        (2, *shape[1:]),
                        dtype=(torch.int32 if suffix == "weight" else torch.bfloat16),
                    ),
                    component=suffix,
                )
                self.register_parameter(parameter_name, parameter)
                plan = SourcePlan(
                    (Interval(1, 3),),
                    (Interval(0, 1), Interval(3, 4)),
                    (Destination(parameter_name, Interval(0, 2)),),
                    "expert ownership",
                )
                callback = partial(
                    parameter.copy_source,
                    source_shape=shape,
                    shard=shard,
                    source_slices=(slice(1, 3), slice(None)),
                )
                sizes = {parameter_name: 2}
                actual = (parameter_name,)
            else:
                plan = SourcePlan((), (Interval(0, 4),), (), "expert ownership")
                callback = self._omit_nonlocal
                sizes, actual = {}, ()
            bindings.append(MlxAffineLoadBinding(name, plan, sizes, callback, actual))
        self._mlx_affine_load_bindings = tuple(bindings)

    @staticmethod
    def _omit_nonlocal(source):
        # A source still reaches its explicit ownership callback; no local write.
        assert source.device.type == "cpu"

    def load(self, weights=None):
        return load_mlx_affine_weights(
            self,
            iter(self.source_tensors.items()) if weights is None else weights,
            AutoWeightsLoader(self).load_weights,
        )


def test_owned_expert_components_copy_only_local_slices_and_report_all_sources():
    model = _OwnedExperts()
    # Lazy workspace is not a checkpoint destination and need not be materialized.
    model.register_buffer("workspace", torch.empty(2, device="meta"), persistent=False)
    assert model.load() == {"codes", "scales", "biases", "norm"}
    assert torch.equal(
        model.codes, model.source_tensors["model.experts.weight"][1:3].view(torch.int32)
    )
    assert torch.equal(model.scales, model.source_tensors["model.experts.scales"][1:3])
    assert torch.equal(model.biases, model.source_tensors["model.experts.biases"][1:3])
    assert torch.equal(model.norm, torch.tensor([2, 3], dtype=torch.float16))
    receipt = model._mlx_affine_load_receipt
    assert receipt.coverage.source_components == 4
    assert receipt.coverage.nonlocal_components == 3
    assert receipt.compact_components == 3
    assert receipt.ordinary_components == 1
    assert receipt.gpu_copy_scratch_bytes == 0
    assert receipt.max_copy_tile_bytes <= 1024 * 1024
    assert not is_mlx_affine_loading()


def test_meta_destinations_cannot_receive_a_completed_load_receipt():
    model = _OwnedExperts()
    # Meta construction is permitted, but copying to it is a silent Torch no-op.
    # Use the same storage-free destination through the public loading path.
    model.norm = nn.Parameter(model.norm.to("meta"), requires_grad=False)
    with pytest.raises(ValueError):
        model.load()
    assert not hasattr(model, "_mlx_affine_completed_load")
    assert not hasattr(model, "_mlx_affine_load_receipt")
    assert not is_mlx_affine_loading()


def test_bounded_copy_rejects_storage_free_destination():
    source = torch.tensor([2, 3], dtype=torch.bfloat16)
    destination = torch.empty(2, dtype=torch.float16, device="meta")
    with pytest.raises(ValueError):
        copy_mlx_affine_tensor(destination, source)


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "dtype", "shape", "unknown"]
)
def test_invalid_original_components_fail_before_admission_completes(failure):
    model = _OwnedExperts()
    weights = list(model.source_tensors.items())
    if failure == "missing":
        weights.pop(1)
    elif failure == "duplicate":
        weights.append(weights[0])
    elif failure == "dtype":
        weights[0] = (weights[0][0], weights[0][1].to(torch.int32))
    elif failure == "shape":
        weights[0] = (weights[0][0], weights[0][1][:3])
    else:
        weights.append(("undeclared.weight", torch.empty(1)))
    with pytest.raises(ValueError):
        model.load(iter(weights))
    assert not is_mlx_affine_loading()
    assert not hasattr(model, "_mlx_affine_load_receipt")


def test_wholly_nonlocal_experts_are_accounted_without_fake_loaded_parameters():
    model = _OwnedExperts(local=False)
    assert model.load() == {"norm"}
    report = model._mlx_affine_load_receipt.coverage
    assert report.source_components == 4
    assert report.local_components == 1
    assert report.nonlocal_components == 3


def test_metadata_ownership_must_match_its_codes():
    model = _OwnedExperts()
    weight, scales, biases = model._mlx_affine_load_bindings
    wrong_plan = SourcePlan(
        (Interval(0, 2),),
        (Interval(2, 4),),
        scales.plan.destinations,
        "wrong ownership",
    )
    model._mlx_affine_load_bindings = (
        weight,
        MlxAffineLoadBinding(
            scales.original_name,
            wrong_plan,
            scales.destination_sizes,
            scales.load,
            scales.loaded_parameters,
        ),
        biases,
    )
    with pytest.raises(ValueError, match="metadata ownership differs"):
        model.load()
    assert model.codes.copy_bytes == 0


def test_unsigned_words_and_bfloat16_metadata_survive_misaligned_tp_copy():
    spec = MlxAffineSpec(6, 128, "affine", "layer")
    shard = MlxAffineShard.from_interval(spec, 1, 81, logical_k=640)
    source = torch.tensor(
        [[0xFFFFFFFF, 0x80000000, 0x12345678] * 40] * 3,
        dtype=torch.uint32,
    )
    parameter = MlxAffineParameter(
        torch.empty(2, 16, dtype=torch.int32), component="weight"
    )
    parameter.copy_source(
        source,
        source_shape=(3, 120),
        shard=shard,
        source_slices=(slice(1, 3),),
    )
    assert torch.equal(parameter, source[1:3, :16].view(torch.int32))
    assert shard.code_bit_offset == 6
    metadata_bits = torch.tensor([[1, 0x7F7F, -32767, 0, 3]] * 3, dtype=torch.int16)
    metadata = metadata_bits.view(torch.bfloat16)
    scale = MlxAffineParameter(
        torch.empty(2, 1, dtype=torch.bfloat16), component="scales"
    )
    scale.copy_source(
        metadata,
        source_shape=(3, 5),
        shard=shard,
        source_slices=(slice(1, 3),),
    )
    assert torch.equal(scale.view(torch.int16), metadata_bits[1:3, :1])


def test_expert_gate_up_leading_axis_slices_do_not_overwrite_other_constituents():
    spec = MlxAffineSpec(6, 32, "affine", "layer")
    shard = MlxAffineShard.from_interval(spec, 0, 64, logical_k=64)
    source = (
        torch.arange(4 * 3 * 12, dtype=torch.int32).reshape(4, 3, 12).view(torch.uint32)
    )
    parameter = MlxAffineParameter(
        torch.full((2, 6, 12), -1, dtype=torch.int32), component="weight"
    )
    parameter.copy_source(
        source,
        source_shape=(4, 3, 12),
        shard=shard,
        source_slices=(slice(1, 3), slice(None)),
        destination_slices=(slice(None), slice(3, 6)),
    )
    assert torch.equal(parameter[:, 3:], source[1:3].view(torch.int32))
    assert torch.equal(parameter[:, :3], torch.full((2, 3, 12), -1, dtype=torch.int32))


def test_bounded_layout_copy_casts_cpu_tiles_without_full_transformed_storage():
    source = torch.arange(1025 * 512, dtype=torch.float32).reshape(512, 1025).t()
    destination = torch.empty(source.shape, dtype=torch.float16)
    stats = copy_mlx_affine_tensor(destination, source)
    assert torch.equal(destination, source.to(torch.float16))
    assert stats.copy_tiles > 1
    assert stats.max_copy_tile_bytes <= 1024 * 1024
    assert stats.copy_bytes == destination.numel() * destination.element_size()


class _AliasModel(nn.Module):
    def __init__(self, draft=False):
        super().__init__()
        if draft:
            self.norm = nn.Parameter(
                torch.empty(2, dtype=torch.float16), requires_grad=False
            )
            self.quant_config = _config(
                {"mtp.norm": ("BF16", (2,), "norm")}, view="mtp"
            )
            self._mlx_affine_shared_aliases = {"embed": "embed", "head": "head"}
            self.weights = {"mtp.norm": torch.ones(2, dtype=torch.bfloat16)}
        else:
            self.embed = nn.Embedding(2, 2, dtype=torch.float16)
            self.head = nn.Linear(2, 2, bias=False, dtype=torch.float16)
            self.quant_config = _config(
                {
                    "embed.weight": ("BF16", (2, 2), "embed.weight"),
                    "head.weight": ("BF16", (2, 2), "head.weight"),
                }
            )
            self.weights = {
                "embed.weight": torch.ones(2, 2, dtype=torch.bfloat16),
                "head.weight": torch.full((2, 2), 2, dtype=torch.bfloat16),
            }

    def load(self):
        return load_mlx_affine_weights(
            self, iter(self.weights.items()), self._logical_load
        )

    def _logical_load(self, weights):
        # Exercise a nested public wrapper with the real ordinary loader.
        return load_mlx_affine_weights(
            self, weights, AutoWeightsLoader(self).load_weights
        )


def test_sharing_is_deferred_until_actual_whole_modules_are_identical():
    target, draft = _AliasModel(), _AliasModel(draft=True)
    target.load()
    draft.load()
    assert draft._mlx_affine_load_receipt.coverage.pending_aliases == (
        "draft.embed",
        "draft.head",
    )
    draft.embed = nn.Embedding(2, 2, dtype=torch.float16)
    draft.head = nn.Linear(2, 2, bias=False, dtype=torch.float16)
    with pytest.raises(ValueError, match="identity mismatch"):
        finish_mlx_affine_sharing(target, draft)
    draft.embed = target.embed
    draft.head = target.head
    receipt = finish_mlx_affine_sharing(target, draft)
    assert receipt.coverage.pending_aliases == ()
    assert target.embed is not target.head


def test_distinct_target_embedding_head_owners_cannot_be_collapsed():
    target, draft = _AliasModel(), _AliasModel(draft=True)
    target.load()
    draft.load()
    target.head = target.embed
    draft.embed = target.embed
    draft.head = target.head
    with pytest.raises(ValueError, match="distinct shared owners alias"):
        finish_mlx_affine_sharing(target, draft)


def test_compact_parameter_rejects_numeric_int32_sources_and_wrong_original_shape():
    spec = MlxAffineSpec(6, 32, "affine", "layer")
    shard = MlxAffineShard.from_interval(spec, 0, 64, logical_k=64)
    parameter = MlxAffineParameter(
        torch.empty(2, 12, dtype=torch.int32), component="weight"
    )
    source = torch.zeros(2, 12, dtype=torch.uint32)
    with pytest.raises(ValueError, match="original CPU shape/dtype"):
        parameter.copy_source(
            source.view(torch.int32), source_shape=(2, 12), shard=shard
        )
    with pytest.raises(ValueError, match="original CPU shape/dtype"):
        parameter.copy_source(source, source_shape=(3, 12), shard=shard)
    assert parameter.copy_bytes == 0


class _VisionLayout(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Parameter(
            torch.empty(1152, 3, 2, 16, 16, dtype=torch.float16),
            requires_grad=False,
        )
        self.quant_config = _config(
            {"vision_tower.patch.weight": ("BF16", (1152, 2, 16, 16, 3), "patch")},
            layouts={"vision_tower.patch.weight": "vision_patch_mlx"},
        )


def test_original_layout_is_validated_before_cpu_permutation_and_logical_copy():
    model = _VisionLayout()
    source = (
        torch.arange(1152 * 2 * 16 * 16 * 3, dtype=torch.int32)
        .remainder(17)
        .to(torch.bfloat16)
        .reshape(1152, 2, 16, 16, 3)
    )
    loaded = load_mlx_affine_weights(
        model,
        iter((("vision_tower.patch.weight", source),)),
        AutoWeightsLoader(model).load_weights,
    )
    assert loaded == {"patch"}
    assert torch.equal(model.patch, source.permute(0, 4, 1, 2, 3).to(torch.float16))
    invalid_model = _VisionLayout()
    with pytest.raises(ValueError, match="shape/dtype mismatch"):
        load_mlx_affine_weights(
            invalid_model,
            iter((("vision_tower.patch.weight", source.permute(0, 4, 1, 2, 3)),)),
            AutoWeightsLoader(invalid_model).load_weights,
        )


def test_postshare_owner_paths_can_rebase_but_draft_alias_keys_cannot_change():
    target, draft = _AliasModel(), _AliasModel(draft=True)
    target.load()
    draft.load()
    target.language_model = nn.Module()
    target.language_model.embed = target.embed
    target.language_model.head = target.head
    draft.embed, draft.head = target.embed, target.head
    draft._mlx_affine_shared_aliases = {
        "embed": "language_model.embed",
        "head": "language_model.head",
    }
    receipt = finish_mlx_affine_sharing(target, draft)
    assert receipt.shared_aliases == (
        ("embed", "language_model.embed"),
        ("head", "language_model.head"),
    )
    draft._mlx_affine_shared_aliases = {
        "other": "language_model.embed",
        "head": "language_model.head",
    }
    with pytest.raises(ValueError, match="declared two distinct"):
        finish_mlx_affine_sharing(target, draft)


def test_transposed_destination_copies_contiguous_spans_not_individual_elements():
    destination = torch.empty(1025, 512, dtype=torch.float16).t()
    source = (
        torch.arange(512 * 1025, dtype=torch.int32)
        .remainder(64)
        .reshape(
            512,
            1025,
        )
        .to(torch.bfloat16)
    )
    stats = copy_mlx_affine_tensor(destination, source)
    assert torch.equal(destination, source.to(torch.float16))
    assert stats.copy_tiles <= destination.shape[1]
    assert stats.max_copy_tile_bytes <= 1024 * 1024
    assert stats.copy_bytes == destination.numel() * destination.element_size()


def test_failed_replacement_load_invalidates_prior_admission_before_any_writes():
    target, draft = _AliasModel(), _AliasModel(draft=True)
    target.load()
    draft.load()
    draft.embed, draft.head = target.embed, target.head
    finish_mlx_affine_sharing(target, draft)
    replacement = torch.full((2, 2), 9, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="missing source components"):
        load_mlx_affine_weights(
            target,
            iter((("embed.weight", replacement),)),
            target._logical_load,
        )
    # The partial replacement really changed final storage, not just metadata.
    assert torch.equal(target.embed.weight, replacement.to(torch.float16))
    assert not hasattr(target, "_mlx_affine_completed_load")
    assert not hasattr(target, "_mlx_affine_load_receipt")
    with pytest.raises(ValueError, match="both native target and draft"):
        finish_mlx_affine_sharing(target, draft)
