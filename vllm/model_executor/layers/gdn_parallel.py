# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense GDN projections with contiguous, whole-key-group TP ownership."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import Parameter

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class GDNHeadPartition:
    key_start: int
    key_count: int
    value_start: int
    value_count: int
    head_k_dim: int
    head_v_dim: int

    def __post_init__(self) -> None:
        values = (
            self.key_start,
            self.key_count,
            self.value_start,
            self.value_count,
            self.head_k_dim,
            self.head_v_dim,
        )
        if not all(_is_int(value) for value in values):
            raise ValueError("GDN partition fields must be integers")
        if (
            self.key_start < 0
            or self.value_start < 0
            or any(
                value <= 0
                for value in (
                    self.key_count,
                    self.value_count,
                    self.head_k_dim,
                    self.head_v_dim,
                )
            )
        ):
            raise ValueError(
                "GDN partitions require nonempty heads and positive dimensions"
            )
        ratio, remainder = divmod(self.value_count, self.key_count)
        if remainder or ratio < 1 or self.value_start != self.key_start * ratio:
            raise ValueError("GDN partitions must own complete contiguous V/K groups")

    @property
    def key_dim(self) -> int:
        return self.key_count * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.value_count * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim


def gdn_head_partition(
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    rank: int,
    world_size: int,
) -> GDNHeadPartition:
    """Balance complete key groups, giving earlier ranks the remainder."""
    if not all(
        _is_int(value)
        for value in (
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            rank,
            world_size,
        )
    ):
        raise ValueError("GDN head dimensions, rank and world size must be integers")
    if min(num_k_heads, num_v_heads, head_k_dim, head_v_dim, world_size) <= 0:
        raise ValueError("GDN head dimensions and world size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("GDN rank must be in the TP range")
    if num_k_heads < world_size or num_v_heads % num_k_heads:
        raise ValueError("GDN TP requires one key group per rank and integral V/K")
    groups, remainder = divmod(num_k_heads, world_size)
    start = rank * groups + min(rank, remainder)
    count = groups + (rank < remainder)
    ratio = num_v_heads // num_k_heads
    return GDNHeadPartition(
        start, count, start * ratio, count * ratio, head_k_dim, head_v_dim
    )


def _slice_layout(
    global_sizes: tuple[int, ...], local_slices: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int, int], ...]:
    if not global_sizes or len(global_sizes) != len(local_slices):
        raise ValueError("GDN requires one local slice per global component")
    layout = []
    target_offset = 0
    for size, row_slice in zip(global_sizes, local_slices):
        if not _is_int(size) or size <= 0 or len(row_slice) != 2:
            raise ValueError("GDN component sizes and slices must be positive integers")
        start, count = row_slice
        if not _is_int(start) or not _is_int(count):
            raise ValueError("GDN slice offsets and sizes must be integers")
        if start < 0 or count <= 0 or start + count > size:
            raise ValueError(
                "GDN local slice must be nonempty and within its component"
            )
        layout.append((target_offset, start, count))
        target_offset += count
    return tuple(layout)


def _require_dense(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if (
            tensor.layout != torch.strided
            or tensor.dtype
            not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or getattr(tensor, "packed_dim", None) is not None
            or getattr(tensor, "is_gguf_weight", False)
        ):
            raise ValueError("Unequal GDN TP supports only unquantized dense weights")


class GDNPackedColumnParallelLinear(ReplicatedLinear):
    """Load each original packed component's owned rows into a local projection."""

    def __init__(
        self,
        input_size: int,
        output_sizes: tuple[int, ...],
        row_slices: tuple[tuple[int, int], ...],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        if not _is_int(input_size) or input_size <= 0:
            raise ValueError("GDN input size must be a positive integer")
        self._row_layout = _slice_layout(output_sizes, row_slices)
        self.full_output_sizes = tuple(output_sizes)
        self.full_output_size = sum(output_sizes)
        self.full_input_size = input_size
        self.row_slices = tuple(row_slices)
        self.output_sizes = [count for _, _, count in self._row_layout]
        self._rdna_i8_global_shape = (self.full_output_size, input_size)
        super().__init__(
            input_size,
            sum(self.output_sizes),
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=True,
        )
        if not isinstance(self.quant_method, UnquantizedLinearMethod):
            raise ValueError("Unequal GDN TP requires unquantized dense projections")
        self.update_param_tp_status()

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        shard_id: int | tuple[int, ...] | None = None,
    ) -> None:
        if shard_id is None:
            shard_id = getattr(loaded_weight, "shard_id", None)
        if shard_id is None:
            components = tuple(range(len(self.full_output_sizes)))
        elif _is_int(shard_id):
            components = (shard_id,)
        elif isinstance(shard_id, tuple) and shard_id:
            components = shard_id
        else:
            raise ValueError(f"Invalid GDN packed shard ID: {shard_id}")
        if any(
            not _is_int(index) or not 0 <= index < len(self.full_output_sizes)
            for index in components
        ) or len(set(components)) != len(components):
            raise ValueError(f"Invalid GDN packed component IDs: {components}")
        expected = (
            sum(self.full_output_sizes[index] for index in components),
            self.full_input_size,
        )
        if loaded_weight.shape != expected:
            raise ValueError(
                f"GDN packed source requires full component shape {expected}"
            )
        if param.shape != (self.output_size, self.full_input_size):
            raise ValueError("GDN packed target parameter has an invalid shape")
        _require_dense(param, loaded_weight)
        source_offset = 0
        for index in components:
            target_offset, start, count = self._row_layout[index]
            param.data.narrow(0, target_offset, count).copy_(
                loaded_weight.narrow(0, source_offset + start, count)
            )
            source_offset += self.full_output_sizes[index]


class GDNOutputParallelLinear(ReplicatedLinear):
    """Load owned V columns and optionally reduce once on the existing TP group."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        partition: GDNHeadPartition,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        if not all(_is_int(size) and size > 0 for size in (input_size, output_size)):
            raise ValueError("GDN output projection sizes must be positive integers")
        if not isinstance(partition, GDNHeadPartition):
            raise ValueError("GDN output projection requires a GDN head partition")
        if (
            input_size % partition.head_v_dim
            or (partition.value_start + partition.value_count) * partition.head_v_dim
            > input_size
        ):
            raise ValueError("GDN output partition exceeds the global V head columns")
        self.partition = partition
        self.full_input_size = input_size
        self.full_output_size = output_size
        self.partition_start = partition.value_start * partition.head_v_dim
        self.partition_size = partition.value_dim
        self.reduce_results = reduce_results
        self._rdna_i8_global_shape = (output_size, input_size)
        super().__init__(
            self.partition_size,
            output_size,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=True,
        )
        if not isinstance(self.quant_method, UnquantizedLinearMethod):
            raise ValueError("Unequal GDN TP requires unquantized dense projections")
        self.update_param_tp_status()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        expected = (self.full_output_size, self.full_input_size)
        if loaded_weight.shape != expected:
            raise ValueError(f"GDN output requires full source shape {expected}")
        if param.shape != (self.full_output_size, self.partition_size):
            raise ValueError("GDN output target parameter has an invalid shape")
        _require_dense(param, loaded_weight)
        if current_platform.is_rocm():
            from vllm.model_executor.layers import rdna_dense_int8

            rdna_dense_int8._prepare_source_absmax(self, loaded_weight)
        super().weight_loader(
            param,
            loaded_weight.narrow(1, self.partition_start, self.partition_size),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if x.shape[-1] != self.partition_size:
            raise ValueError(
                "GDN output input must contain exactly the owned V columns"
            )
        output = self.quant_method.apply(self, x, None)
        if self.reduce_results:
            output = tensor_model_parallel_all_reduce(output)
        return output, None


def gdn_sharded_weight_loader(
    global_sizes: tuple[int, ...],
    local_slices: tuple[tuple[int, int], ...],
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """Load conv rows or per-V values without changing trailing dimensions/dtype."""
    layout = _slice_layout(global_sizes, local_slices)
    global_rows = sum(global_sizes)
    local_rows = sum(count for _, _, count in layout)
    source_offset = 0
    copies = []
    for size, (target_offset, start, count) in zip(global_sizes, layout):
        copies.append((target_offset, source_offset + start, count))
        source_offset += size
    copies = tuple(copies)

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if param.ndim == 1:
            trailing = ()
        elif param.ndim == 3 and param.shape[1] == 1 and param.shape[2] > 0:
            trailing = tuple(param.shape[1:])
        else:
            raise ValueError(
                "GDN sharded targets must be per-head or [rows, 1, kernel]"
            )
        if param.shape != (local_rows, *trailing):
            raise ValueError("GDN sharded target parameter has an invalid shape")
        expected = (global_rows, *trailing)
        if loaded_weight.shape != expected:
            raise ValueError(f"GDN sharded source requires full shape {expected}")
        _require_dense(param, loaded_weight)
        for target_offset, source_start, count in copies:
            param.data.narrow(0, target_offset, count).copy_(
                loaded_weight.narrow(0, source_start, count)
            )

    return loader
