# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MLX affine storage, source ownership and RDNA2 execution methods.

Checkpoint identities are resolved once, before allocating compact parameters.
Architectural biases are not the published affine ``biases`` arrays.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.mlx_affine_format import (
    MlxAffineDestination,
    MlxAffineIdentity,
    MlxAffineInterval,
    MlxAffineSegment,
    MlxAffineShard,
    MlxAffineSourcePlan,
    MlxAffineSpec,
    MlxAffineSpecMap,
)
from vllm.model_executor.layers.quantization.mlx_affine_tuning import MlxAffineTuning
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.mlx_affine_checkpoint import (
    load_mlx_affine_checkpoint,
)
from vllm.model_executor.parameter import MlxAffineParameter

_COMPONENTS = ("weight", "scales", "biases")
_DTYPES = {"weight": torch.int32, "scales": torch.bfloat16, "biases": torch.bfloat16}


def _complement(intervals, size):
    result = []
    cursor = 0
    for start, stop in sorted(set(intervals)):
        if start < cursor or not start < stop <= size:
            raise ValueError("overlapping or out-of-range affine source rows")
        if cursor < start:
            result.append(MlxAffineInterval(cursor, start))
        cursor = stop
    if cursor < size:
        result.append(MlxAffineInterval(cursor, size))
    return tuple(result)


@dataclass(frozen=True)
class _DensePart:
    segment: MlxAffineSegment
    source_rows: tuple[int, int]
    shard_id: str | int | tuple[int, ...] | None
    parameters: tuple[str, str, str]


class MlxAffineConfig(QuantizationConfig):
    def __init__(self, specs: MlxAffineSpecMap):
        super().__init__()
        self.specs = specs
        self.module_specs = specs.module_specs
        self.checkpoint = None
        self._original_by_prefix = MappingProxyType({})
        self._tuning: MlxAffineTuning | None = None
        self.backbone_layers = 48

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        return cls(MlxAffineSpecMap.from_config(config))

    def get_name(self) -> str:
        return "mlx_affine"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @staticmethod
    def get_checkpoint_weight_mapper():
        from vllm.model_executor.models.utils import WeightsMapper

        return WeightsMapper()

    def resolve_checkpoint_quantization(
        self, original_prefix: str
    ) -> MlxAffineSpec | None:
        return self.specs.resolve_checkpoint_quantization(original_prefix)

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        if hf_config is None:
            raise ValueError("mlx_affine requires the normalized view configuration")
        config = hf_config.to_dict() if hasattr(hf_config, "to_dict") else hf_config
        checkpoint = load_mlx_affine_checkpoint(model_name, config)
        if checkpoint.manifest["module_specs_sha256"] != self.specs.sha256:
            raise ValueError("native view and quantization map disagree")
        if self.checkpoint is not None:
            if self.checkpoint.manifest_sha256 != checkpoint.manifest_sha256:
                raise ValueError("an affine config cannot be rebound to another view")
            return
        text = config.get("text_config", config)
        self.backbone_layers = text.get("num_hidden_layers", 48)
        if self.backbone_layers != 48:
            raise ValueError(
                "the native oQ6e backbone must retain its 48-layer identity"
            )
        original_by_prefix = {}
        for original, entry in checkpoint.manifest["tensors"].items():
            if not original.endswith(".weight"):
                continue
            original_prefix = original.removesuffix(".weight")
            if self.resolve_checkpoint_quantization(original_prefix) is None:
                continue
            runtime_prefix = entry["runtime_name"].removesuffix(".weight")
            aliases = {runtime_prefix, original_prefix}
            if original.startswith("mtp.layers.0."):
                if not runtime_prefix.startswith("model.layers.0."):
                    raise ValueError("MTP decoder must retain registered layer 0")
                # Manifest remapping also changes switch_mlp to experts.
                aliases.add(
                    f"mtp.layers.{self.backbone_layers}."
                    + runtime_prefix.removeprefix("model.layers.0.")
                )
            for alias in aliases:
                previous = original_by_prefix.setdefault(alias, original_prefix)
                if previous != original_prefix:
                    raise ValueError(f"ambiguous affine construction prefix: {alias}")
                # The default public constructor has a "model" wrapper that
                # is absent from registered multimodal parameter names.
                if alias.startswith(("language_model.", "mtp.", "model.layers.")):
                    wrapped = "model." + alias
                    previous = original_by_prefix.setdefault(wrapped, original_prefix)
                    if previous != original_prefix:
                        raise ValueError(f"ambiguous affine wrapper prefix: {wrapped}")
        self.checkpoint = checkpoint
        self._original_by_prefix = MappingProxyType(original_by_prefix)

    @property
    def tuning(self) -> MlxAffineTuning:
        # HF config binding precedes set_current_vllm_config. Resolve only at
        # the first native method's construction, never in the token path.
        if self._tuning is None:
            current = get_current_vllm_config_or_none()
            if current is None:
                raise ValueError(
                    "native affine construction requires active VllmConfig"
                )
            additional = current.additional_config
            if additional is not None and not isinstance(additional, Mapping):
                raise ValueError("additional_config must be an object")
            self._tuning = (
                MlxAffineTuning.from_config(additional["rdna_oq_affine"])
                if additional is not None and "rdna_oq_affine" in additional
                else MlxAffineTuning()
            )
        return self._tuning

    def _resolve(self, prefix: str) -> str | None:
        if self.checkpoint is None:
            raise ValueError(
                "bind mlx_affine to its native manifest before construction"
            )
        return self._original_by_prefix.get(prefix)

    def _sources(self, prefix: str) -> tuple[str, ...]:
        direct = self._resolve(prefix)
        if direct is not None:
            return (direct,)
        parent, _, leaf = prefix.rpartition(".")
        explicit = {
            "qkv_proj": ("q_proj", "k_proj", "v_proj"),
            "gate_up_proj": ("gate_proj", "up_proj"),
            "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
            "in_proj_ba": ("in_proj_b", "in_proj_a"),
            "kv_proj": ("key_proj", "value_proj"),
            "input_mix_weight_down_block_inject": (
                "input_mix_weight_down",
                "block_inject_weight",
            ),
        }
        constituents = explicit.get(leaf, self.packed_modules_mapping.get(leaf, ()))
        resolved = tuple(
            self._resolve(f"{parent}.{name}" if parent else name)
            for name in constituents
            if name != "_input_mix_padding"
        )
        if not any(resolved):
            return ()
        if not all(resolved):
            raise ValueError(f"partly quantized fused module is unsupported: {prefix}")
        return resolved

    def get_quant_method(self, layer: nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        if isinstance(layer, RoutedExperts):
            sources = tuple(
                self._resolve(f"{prefix}.{projection}")
                for projection in ("gate_proj", "up_proj", "down_proj")
            )
            if not all(sources):
                raise ValueError(f"missing native expert specification: {prefix}")
            return MlxAffineMoEMethod(self, sources, layer.moe_config)
        if isinstance(layer, (LinearBase, VocabParallelEmbedding)):
            sources = self._sources(prefix)
            if not sources:
                if isinstance(layer, LinearBase):
                    return UnquantizedLinearMethod()
                return None
            method = (
                MlxAffineVocabMethod
                if isinstance(layer, VocabParallelEmbedding)
                else MlxAffineLinearMethod
            )
            return method(self, sources, prefix)
        return None

    def tensor(self, original_name):
        return self.checkpoint.manifest["tensors"][original_name]

    def identity(self, original_prefix, construction_prefix):
        original = original_prefix + ".weight"
        runtime = self.tensor(original)["runtime_name"]
        return MlxAffineIdentity(original, runtime, construction_prefix)


def _install_dense_bindings(layer, config, parts):
    from vllm.model_executor.model_loader.mlx_affine_loading import MlxAffineLoadBinding

    by_source = defaultdict(list)
    for part in parts:
        by_source[part.segment.source.original_name.removesuffix(".weight")].append(
            part
        )
    bindings = []
    for original_prefix, source_parts in by_source.items():
        for component_index, component in enumerate(_COMPONENTS):
            original_name = f"{original_prefix}.{component}"
            source_shape = tuple(config.tensor(original_name)["shape"])
            intervals = [part.source_rows for part in source_parts]
            local = tuple(MlxAffineInterval(*interval) for interval in intervals)
            nonlocal_ = _complement(intervals, source_shape[0])
            destinations = tuple(
                MlxAffineDestination(
                    part.parameters[component_index],
                    MlxAffineInterval(0, part.segment.output_size),
                )
                for part in source_parts
            )
            sizes = {
                part.parameters[component_index]: part.segment.output_size
                for part in source_parts
            }

            def copy(
                source,
                *,
                entries=tuple(source_parts),
                ci=component_index,
                shape=source_shape,
            ):
                for part in entries:
                    getattr(layer, part.parameters[ci]).copy_source(
                        source,
                        source_shape=shape,
                        shard=part.segment.shard,
                        source_slices=(slice(*part.source_rows),),
                    )

            bindings.append(
                MlxAffineLoadBinding(
                    original_name=original_name,
                    plan=MlxAffineSourcePlan(
                        local,
                        nonlocal_,
                        destinations,
                        "tensor-parallel output ownership" if nonlocal_ else None,
                    ),
                    destination_sizes=MappingProxyType(sizes),
                    load=copy,
                    loaded_parameters=tuple(sizes),
                )
            )
    layer._mlx_affine_load_bindings = tuple(bindings)


class MlxAffineLinearMethod(LinearMethodBase):
    def __init__(self, config, sources, prefix):
        self.config = config
        self.sources = sources
        self.prefix = prefix
        self.tuning = config.tuning

    def _output_rows(self, layer, partitions):
        """Return original rows and logical destination slices, including GDN IDs."""
        if self.prefix.endswith(".in_proj_qkvz") and len(self.sources) == 2:
            if len(partitions) != 4:
                raise ValueError("GDN QKVZ requires Q,K,V,Z destination partitions")
            assignments = [(self.sources[0], i, i) for i in range(3)]
            assignments.append((self.sources[1], 3, 3))
        elif len(self.sources) == len(partitions):
            assignments = [(source, i, i) for i, source in enumerate(self.sources)]
        elif len(self.sources) == 1 and len(partitions) == 1:
            assignments = [(self.sources[0], 0, None)]
        elif self.prefix.endswith(".input_mix_weight_down_block_inject"):
            if len(partitions) != 3 or tuple(partitions) != (320, 4, 12):
                raise ValueError("HC down/inject requires 320+4 rows and 12 zero pads")
            assignments = [(self.sources[0], 0, 0), (self.sources[1], 1, 1)]
        else:
            raise ValueError(f"undeclared fused affine layout: {self.prefix}")
        outputs = []
        source_offsets = defaultdict(int)
        output_offset = 0
        for source, partition, shard_id in assignments:
            width = partitions[partition]
            source_rows = self.config.tensor(source + ".weight")["shape"][0]
            rank = layer.tp_rank
            size = layer.tp_size
            if isinstance(layer, QKVParallelLinear) and partition in (1, 2):
                rank //= layer.num_kv_head_replicas
                size //= layer.num_kv_head_replicas
                shard_id = ("q", "k", "v")[partition]
            elif isinstance(layer, QKVParallelLinear):
                shard_id = "q"
            if source == self.sources[0] and self.prefix.endswith(".in_proj_qkvz"):
                global_width = width * size
                source_offset = source_offsets[source]
                source_offsets[source] += global_width
            else:
                global_width = source_rows
                source_offset = 0
            # Row-parallel and replicated linears retain all source output rows.
            if width == global_width:
                start = source_offset
            elif global_width == width * size:
                start = source_offset + rank * width
            else:
                raise ValueError(f"affine output shape does not match {self.prefix}")
            outputs.append((source, start, start + width, output_offset, shard_id))
            output_offset += width
        return outputs

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.float16:
            raise ValueError("native affine execution requires FP16 activations")
        if input_size % input_size_per_partition:
            raise ValueError("invalid native affine TP input partition")
        input_divisor = input_size // input_size_per_partition
        if input_divisor not in (1, layer.tp_size):
            raise ValueError("native affine input ownership must match TP")
        k0 = layer.tp_rank * input_size_per_partition if input_divisor > 1 else 0
        parts = []
        for index, (source, r0, r1, offset, shard_id) in enumerate(
            self._output_rows(layer, output_partition_sizes)
        ):
            spec = self.config.resolve_checkpoint_quantization(source)
            shard = MlxAffineShard.from_interval(
                spec, k0, k0 + input_size_per_partition, logical_k=input_size
            )
            source_shape = tuple(self.config.tensor(source + ".weight")["shape"])
            if (
                len(source_shape) != 2
                or source_shape[-1] != (input_size * spec.bits + 31) // 32
            ):
                raise ValueError(f"undeclared matrix layout or logical K: {source}")
            declared_hc = {
                "input_mix_weight_down": (320, 10240),
                "block_inject_weight": (4, 10240),
                "input_mix_weight_up": (10240, 320),
            }
            hc_shape = declared_hc.get(source.rsplit(".", 1)[-1])
            if hc_shape is not None and (source_shape[0], input_size) != hc_shape:
                raise ValueError(f"undeclared HC stream/low-rank geometry: {source}")
            for component in ("scales", "biases"):
                entry = self.config.tensor(f"{source}.{component}")
                expected = (
                    source_shape[0],
                    (input_size + spec.group_size - 1) // spec.group_size,
                )
                if entry["dtype"] != "BF16" or tuple(entry["shape"]) != expected:
                    raise ValueError(f"invalid affine metadata shape/dtype: {source}")
            if self.config.tensor(source + ".weight")["dtype"] != "U32":
                raise ValueError(f"native codes must be original U32: {source}")
            if not 0 <= r0 < r1 <= source_shape[0]:
                raise ValueError("native affine output ownership exceeds source rows")
            names = tuple(f"oq{index}_{component}" for component in _COMPONENTS)
            for name, component in zip(names, _COMPONENTS):
                columns = (
                    shard.copied_word_count
                    if component == "weight"
                    else shard.metadata_group_count
                )
                parameter = MlxAffineParameter(
                    torch.empty(r1 - r0, columns, dtype=_DTYPES[component]),
                    component=component,
                )
                layer.register_parameter(name, parameter)
            parts.append(
                _DensePart(
                    MlxAffineSegment(
                        self.config.identity(source, self.prefix),
                        spec,
                        shard,
                        offset,
                        r1 - r0,
                    ),
                    (r0, r1),
                    shard_id,
                    names,
                )
            )
        layer._mlx_affine_parts = tuple(parts)
        layer._mlx_affine_segments = tuple(part.segment for part in parts)
        layer._mlx_affine_output_size = sum(output_partition_sizes)
        layer._mlx_affine_input_size = input_size_per_partition
        layer._mlx_affine_zero_padding = (
            (324, 336)
            if self.prefix.endswith(".input_mix_weight_down_block_inject")
            else None
        )
        if isinstance(layer, VocabParallelEmbedding):
            actual = parts[0].segment.output_size
            total = layer._mlx_affine_output_size
            layer._mlx_affine_zero_padding = (actual, total) if actual < total else None
        padding = layer._mlx_affine_zero_padding
        declared_padding = (MlxAffineInterval(*padding),) if padding else ()
        holes = _complement(
            [
                (
                    part.segment.output_offset,
                    part.segment.output_offset + part.segment.output_size,
                )
                for part in parts
            ],
            layer._mlx_affine_output_size,
        )
        if holes != declared_padding:
            raise ValueError(
                "affine segments and declared zeros do not cover the output"
            )
        layer._mlx_affine_output_padding = declared_padding
        self._execution_view = all(
            source.endswith(".mlp.shared_expert_gate")
            or (
                source.startswith("language_model.model.layers.")
                and source.endswith(
                    (".linear_attn.in_proj_b", ".linear_attn.in_proj_a")
                )
            )
            for source in self.sources
        )
        if self._execution_view:
            layer.register_buffer(
                "weight",
                torch.empty(
                    layer._mlx_affine_output_size,
                    input_size_per_partition,
                    dtype=torch.float16,
                ),
            )
        _install_dense_bindings(layer, self.config, parts)

    def process_weights_after_loading(self, layer):
        if not self._execution_view:
            return
        # Only the explicitly admitted gates and target GDN B/A are expanded.
        # Reconstruct bounded rows using the same native rounding as lookup.
        for part in layer._mlx_affine_parts:
            segment = part.segment
            if segment.shard.logical_k_start or segment.shard.code_bit_offset:
                raise ValueError("execution views require complete input rows")
            codes, scales, biases = (getattr(layer, name) for name in part.parameters)
            rows_per_tile = max(1, (1024 * 1024) // (2 * layer._mlx_affine_input_size))
            for start in range(0, segment.output_size, rows_per_tile):
                stop = min(start + rows_per_tile, segment.output_size)
                ids = torch.arange(start, stop, dtype=torch.int64, device=codes.device)
                out = layer.weight[
                    segment.output_offset + start : segment.output_offset + stop
                ]
                ops.oq_affine_embedding_rdna2(
                    ids,
                    codes,
                    scales,
                    biases,
                    out,
                    segment.spec.bits,
                    segment.spec.group_size,
                    layer._mlx_affine_input_size,
                )

    def apply(self, layer, x, bias=None):
        if x.dtype != torch.float16 or x.shape[-1] != layer._mlx_affine_input_size:
            raise ValueError("native affine linear input must be FP16 with declared K")
        if self._execution_view:
            return torch.nn.functional.linear(x, layer.weight, bias)
        flat = x.reshape(-1, x.shape[-1])
        out = torch.empty(
            flat.shape[0],
            layer._mlx_affine_output_size,
            dtype=torch.float16,
            device=x.device,
        )
        padding = layer._mlx_affine_zero_padding
        if padding is not None:
            out[:, padding[0] : padding[1]].zero_()
        block_m = self.tuning.dense_prefill_block_m
        tile_n = (
            self.tuning.decode_tile_n
            if flat.shape[0] <= 32
            else self.tuning.prefill_tile_n
        )
        for part in layer._mlx_affine_parts:
            segment = part.segment
            codes, scales, biases = (getattr(layer, name) for name in part.parameters)
            shard = segment.shard
            ops.oq_affine_linear_rdna2(
                flat,
                codes,
                scales,
                biases,
                out,
                segment.spec.bits,
                segment.spec.group_size,
                shard.logical_k_start,
                shard.code_bit_offset,
                shard.metadata_group_start,
                segment.output_offset,
                block_m,
                tile_n,
            )
        if bias is not None:
            out.add_(bias)
        return out.view(*x.shape[:-1], layer._mlx_affine_output_size)


class MlxAffineVocabMethod(MlxAffineLinearMethod):
    def _output_rows(self, layer, partitions):
        if len(self.sources) != 1 or layer.num_added_embeddings:
            raise ValueError(
                "native affine vocabulary requires unchanged published IDs"
            )
        indices = layer.shard_indices
        return [
            (
                self.sources[0],
                indices.org_vocab_start_index,
                indices.org_vocab_end_index,
                0,
                None,
            )
        ]

    def embedding(self, layer, input_):
        part = layer._mlx_affine_parts[0]
        codes, scales, biases = (getattr(layer, name) for name in part.parameters)
        out = torch.empty(
            *input_.shape,
            layer.embedding_dim,
            dtype=torch.float16,
            device=input_.device,
        )
        ops.oq_affine_embedding_rdna2(
            input_,
            codes,
            scales,
            biases,
            out,
            part.segment.spec.bits,
            part.segment.spec.group_size,
            layer.embedding_dim,
        )
        return out

    def tie_weights(self, layer, embed_tokens):
        raise ValueError(
            "native oQ6e has untied target embedding/head; share whole modules"
        )


class MlxAffineMoEMethod(FusedMoEMethodBase):
    def __init__(self, config, sources, moe):
        super().__init__(moe)
        self.config = config
        self.sources = sources
        self.tuning = config.tuning

    def get_fused_moe_quant_config(self, layer):
        return None

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        from vllm.model_executor.model_loader.mlx_affine_loading import (
            MlxAffineLoadBinding,
        )

        if (
            params_dtype != torch.float16
            or hidden_size != 2560
            or intermediate_size_per_partition != 640
            or layer.global_num_experts != 512
            or layer.top_k != 10
        ):
            raise ValueError("native oQ6e MoE requires H2560/I640/E512/topk10")
        if layer.activation.value != "silu" or layer.apply_router_weight_on_input:
            raise ValueError(
                "native affine MoE requires SiLU and output router weights"
            )
        if any(
            getattr(layer, name, None) is not None
            for name in (
                "swiglu_limit",
                "swiglu_alpha",
                "swiglu_beta",
            )
        ):
            raise ValueError(
                "native affine MoE does not alter published SiLU semantics"
            )
        specs = tuple(
            self.config.resolve_checkpoint_quantization(s) for s in self.sources
        )
        if any((s.bits, s.group_size, s.mode) != (6, 64, "affine") for s in specs):
            raise ValueError("published native experts require affine6/group64")
        self.spec = specs[0]
        manager = layer.expert_map_manager
        source_map = manager.expert_map
        if source_map is not None and source_map.dtype != torch.int32:
            raise ValueError("native expert maps must be int32")
        if source_map is None:
            mapping = list(range(512))
        else:
            # One bounded construction-time ownership read, never a
            # request-path route-count/scalar synchronization.
            mapping = source_map.detach().cpu().tolist()
        if (
            len(mapping) != 512
            or any(type(i) is not int or i < -1 or i >= num_experts for i in mapping)
            or sorted(i for i in mapping if i >= 0) != list(range(num_experts))
        ):
            raise ValueError("invalid native global-to-local expert ownership")
        local = tuple(
            (global_id, local_id)
            for global_id, local_id in enumerate(mapping)
            if local_id >= 0
        )
        self._ownership = local
        bindings = []
        for projection, output_rows, logical_k in (
            ("w13", 1280, 2560),
            ("w2", 2560, 640),
        ):
            for component in _COMPONENTS:
                columns = (
                    (logical_k * 6 + 31) // 32
                    if component == "weight"
                    else (logical_k + 63) // 64
                )
                layer.register_parameter(
                    f"{projection}_{component}",
                    MlxAffineParameter(
                        torch.empty(
                            num_experts, output_rows, columns, dtype=_DTYPES[component]
                        ),
                        component=component,
                    ),
                )
        for index, source in enumerate(self.sources):
            projection = "w2" if index == 2 else "w13"
            logical_k = 640 if index == 2 else 2560
            shard = MlxAffineShard.from_interval(
                specs[index],
                0,
                logical_k,
                logical_k=logical_k,
            )
            rows = 2560 if index == 2 else 640
            output_offset = 640 if index == 1 else 0
            for component in _COMPONENTS:
                name = f"{source}.{component}"
                shape = tuple(self.config.tensor(name)["shape"])
                dtype = self.config.tensor(name)["dtype"]
                if dtype != ("U32" if component == "weight" else "BF16"):
                    raise ValueError(f"invalid native expert component dtype: {name}")
                expected_columns = (
                    shard.copied_word_count
                    if component == "weight"
                    else shard.metadata_group_count
                )
                if shape != (512, rows, expected_columns):
                    raise ValueError(f"invalid native expert component shape: {name}")
                parameter_name = f"{projection}_{component}"
                # Destination units are explicit expert/output rows, not an
                # inference from tensor element counts or packing geometry.
                destinations = tuple(
                    MlxAffineDestination(
                        parameter_name,
                        MlxAffineInterval(
                            local_id * (2560 if index == 2 else 1280) + output_offset,
                            local_id * (2560 if index == 2 else 1280)
                            + output_offset
                            + rows,
                        ),
                    )
                    for _, local_id in local
                )
                intervals = tuple((global_id, global_id + 1) for global_id, _ in local)
                nonlocal_ = _complement(intervals, 512)

                def copy(
                    tensor,
                    *,
                    pname=parameter_name,
                    source_shape=shape,
                    source_shard=shard,
                    offset=output_offset,
                    width=rows,
                ):
                    parameter = getattr(layer, pname)
                    for global_id, local_id in self._ownership:
                        parameter.copy_source(
                            tensor,
                            source_shape=source_shape,
                            shard=source_shard,
                            source_slices=(
                                slice(global_id, global_id + 1),
                                slice(0, width),
                            ),
                            destination_slices=(
                                slice(local_id, local_id + 1),
                                slice(offset, offset + width),
                            ),
                        )

                bindings.append(
                    MlxAffineLoadBinding(
                        original_name=name,
                        plan=MlxAffineSourcePlan(
                            tuple(
                                MlxAffineInterval(*interval) for interval in intervals
                            ),
                            nonlocal_,
                            destinations,
                            "expert-parallel ownership" if nonlocal_ else None,
                        ),
                        destination_sizes=MappingProxyType(
                            {
                                parameter_name: num_experts
                                * (2560 if index == 2 else 1280)
                            }
                        ),
                        load=copy,
                        loaded_parameters=(parameter_name,),
                    )
                )
        layer._mlx_affine_load_bindings = tuple(bindings)
        layer._mlx_affine_local_experts = tuple(global_id for global_id, _ in local)

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    ):
        if x.dtype != torch.float16 or x.ndim != 2 or x.shape[1] != 2560:
            raise ValueError("native MoE activations must be FP16 [M,2560]")
        if (
            topk_ids.shape != (x.shape[0], 10)
            or topk_weights.shape != topk_ids.shape
            or topk_ids.dtype not in (torch.int32, torch.int64)
            or topk_weights.dtype != torch.float32
        ):
            raise ValueError(
                "native MoE requires integer topk10 IDs and FP32 routing weights"
            )
        if not all(t.is_contiguous() for t in (x, topk_ids, topk_weights)):
            raise ValueError("native MoE inputs must be contiguous")
        output = torch.empty_like(x)
        chunk = min(self.tuning.chunk_tokens, x.shape[0])
        if not chunk:
            return output
        # Allocator-owned invocation storage becomes graph-pool-owned under
        # capture; there is no persistent shape-keyed scratch or device readback.
        gate_up = torch.empty(chunk * 10, 1280, dtype=torch.float32, device=x.device)
        act = torch.empty(chunk * 10, 640, dtype=torch.float16, device=x.device)
        slots = torch.empty(chunk * 10, 2560, dtype=torch.float32, device=x.device)
        empty = torch.empty(0, dtype=torch.int32, device=x.device)
        expert_map = layer.expert_map if layer.expert_map is not None else empty
        decode = x.shape[0] <= 32
        block_m = 1 if decode else self.tuning.prefill_block_m
        tile_n = self.tuning.decode_tile_n if decode else self.tuning.prefill_tile_n
        for start in range(0, x.shape[0], chunk):
            stop = min(start + chunk, x.shape[0])
            count = (stop - start) * 10
            ids = topk_ids[start:stop]
            if decode:
                sorted_ids, expert_ids, post_pad = empty, empty, empty
            else:
                sorted_ids, expert_ids, post_pad = moe_align_block_size(
                    ids,
                    block_m,
                    512,
                    layer.expert_map,
                    ignore_invalid_experts=True,
                )
            gate = gate_up[:count]
            activation = act[:count]
            down = slots[:count]
            ops.moe_oq_affine_gemm_rdna2(
                x[start:stop],
                layer.w13_weight,
                layer.w13_scales,
                layer.w13_biases,
                ids,
                expert_map,
                sorted_ids,
                expert_ids,
                post_pad,
                gate,
                6,
                64,
                10,
                block_m,
                tile_n,
            )
            ops.oq_silu_mul_rdna2(gate, activation)
            ops.moe_oq_affine_gemm_rdna2(
                activation,
                layer.w2_weight,
                layer.w2_scales,
                layer.w2_biases,
                ids,
                expert_map,
                sorted_ids,
                expert_ids,
                post_pad,
                down,
                6,
                64,
                1,
                block_m,
                tile_n,
            )
            ops.oq_moe_reduce_rdna2(
                down.view(stop - start, 10, 2560),
                topk_weights[start:stop],
                output[start:stop],
                True,
            )
        return output
