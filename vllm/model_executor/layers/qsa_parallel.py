# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-GQA-group ownership for rotating, unequal QSA tensor parallelism."""

from dataclasses import dataclass

import torch
from torch.nn import Parameter

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform


@dataclass(frozen=True)
class QSAHeadPartition:
    q_start: int
    q_count: int
    kv_start: int
    kv_count: int


def qsa_head_partition(
    num_heads: int,
    num_kv_heads: int,
    ordinal: int,
    rank: int,
    world_size: int,
) -> QSAHeadPartition:
    """Rotate the two complete GQA groups and one truly empty owner."""
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (num_heads, num_kv_heads, ordinal, rank, world_size)
    ):
        raise ValueError(
            "QSA head counts, ordinal, rank and world size must be integers"
        )
    if world_size != 3 or num_kv_heads != 2:
        raise ValueError("Rotating QSA ownership requires TP3 and two KV heads")
    if num_heads <= 0 or num_heads % num_kv_heads:
        raise ValueError("QSA query heads must be positive and divisible by KV heads")
    if ordinal < 0 or not 0 <= rank < world_size:
        raise ValueError("QSA ordinal must be nonnegative and rank must be in TP range")
    slot = (rank - ordinal) % world_size
    group_size = num_heads // num_kv_heads
    if slot == 2:
        return QSAHeadPartition(num_heads, 0, num_kv_heads, 0)
    return QSAHeadPartition(slot * group_size, group_size, slot, 1)


class QSAQKVParallelLinear(ReplicatedLinear):
    """Load local [Q/gates, K, V] rows, keeping each Q/gate head intact."""

    def __init__(
        self,
        hidden_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        head_dim: int,
        *,
        partition: QSAHeadPartition,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        if hidden_size <= 0 or head_dim <= 0:
            raise ValueError("QSA hidden size and head dimension must be positive")
        valid_partitions = (
            qsa_head_partition(total_num_heads, total_num_kv_heads, 0, rank, 3)
            for rank in range(3)
        )
        if partition not in valid_partitions:
            raise ValueError(
                "QSA partition must own one complete GQA group or be empty"
            )
        self.partition = partition
        self.hidden_size = hidden_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.head_dim = head_dim
        # Checkpoint Q rows interleave a full Q vector and gate vector per head.
        self.full_q_size = 2 * total_num_heads * head_dim
        self.full_kv_size = total_num_kv_heads * head_dim
        self.q_size = 2 * partition.q_count * head_dim
        self.kv_size = partition.kv_count * head_dim
        self.output_sizes = [self.q_size, self.kv_size, self.kv_size]
        self.full_output_size = self.full_q_size + 2 * self.full_kv_size
        self._rdna_i8_global_shape = (self.full_output_size, hidden_size)
        super().__init__(
            hidden_size,
            sum(self.output_sizes),
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=True,
        )
        if not isinstance(self.quant_method, UnquantizedLinearMethod):
            raise ValueError("Rotating QSA TP requires unquantized projections")
        self.update_param_tp_status()

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str | None = None,
    ):
        if shard_id is None:
            shard_id = getattr(loaded_weight, "shard_id", None)
        if shard_id is None:
            expected = (self.full_output_size, self.hidden_size)
            if loaded_weight.shape != expected:
                raise ValueError(f"Fused QSA QKV requires full source shape {expected}")
            # Views only: never materialize the full checkpoint tensor again.
            for name, source in zip(
                ("q", "k", "v"),
                loaded_weight.split(
                    (self.full_q_size, self.full_kv_size, self.full_kv_size), dim=0
                ),
            ):
                self.weight_loader(param, source, name)
            return
        if shard_id not in ("q", "k", "v"):
            raise ValueError(f"Unknown QSA QKV shard ID: {shard_id}")
        is_query = shard_id == "q"
        full_rows = self.full_q_size if is_query else self.full_kv_size
        expected = (full_rows, self.hidden_size)
        if loaded_weight.shape != expected:
            raise ValueError(f"QSA {shard_id} requires full source shape {expected}")
        if param.shape != (self.output_size, self.hidden_size):
            raise ValueError("QSA QKV target parameter has an invalid shape")
        start = (
            2 * self.partition.q_start * self.head_dim
            if is_query
            else self.partition.kv_start * self.head_dim
        )
        count = self.q_size if is_query else self.kv_size
        offset = 0 if is_query else self.q_size
        if shard_id == "v":
            offset += self.kv_size
        param.data.narrow(0, offset, count).copy_(loaded_weight.narrow(0, start, count))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if x.shape[-1] != self.hidden_size:
            raise ValueError("QSA QKV input must contain the full hidden width")
        if self.partition.q_count == 0:
            return x.new_zeros((*x.shape[:-1], 0)), None
        return self.quant_method.apply(self, x, None), None


class QSAOutputParallelLinear(ReplicatedLinear):
    """Load original Q input columns and reduce once over the existing TP group."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        head_dim: int,
        *,
        partition: QSAHeadPartition,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        if input_size <= 0 or output_size <= 0 or head_dim <= 0:
            raise ValueError("QSA projection dimensions must be positive")
        if input_size % head_dim:
            raise ValueError("QSA input size must contain complete heads")
        valid_partitions = (
            qsa_head_partition(input_size // head_dim, 2, 0, rank, 3)
            for rank in range(3)
        )
        if partition not in valid_partitions:
            raise ValueError(
                "QSA partition must own one complete GQA group or be empty"
            )
        self.partition = partition
        self.head_dim = head_dim
        self.full_input_size = input_size
        self.full_output_size = output_size
        self.reduce_results = reduce_results
        self.partition_start = partition.q_start * head_dim
        self.partition_size = partition.q_count * head_dim
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
            raise ValueError("Rotating QSA TP requires unquantized projections")
        self.update_param_tp_status()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        expected = (self.full_output_size, self.full_input_size)
        if loaded_weight.shape != expected:
            raise ValueError(f"QSA output requires full source shape {expected}")
        if param.shape != (self.output_size, self.partition_size):
            raise ValueError("QSA output target parameter has an invalid shape")
        if current_platform.is_rocm():
            from vllm.model_executor.layers import rdna_dense_int8

            # Full original rows define the scale, before discarding any columns.
            rdna_dense_int8._prepare_source_absmax(self, loaded_weight)
        super().weight_loader(
            param,
            loaded_weight.narrow(1, self.partition_start, self.partition_size),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if x.shape[-1] != self.partition_size:
            raise ValueError(
                "QSA output input must contain exactly the owned Q columns"
            )
        if self.partition.q_count == 0:
            output = x.new_zeros((*x.shape[:-1], self.output_size))
        else:
            output = self.quant_method.apply(self, x, None)
        if self.reduce_results:
            output = tensor_model_parallel_all_reduce(output)
        return output, None
