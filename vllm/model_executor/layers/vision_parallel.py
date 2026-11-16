# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unquantized vision projections on an explicit, independent process group."""

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


def vision_partition(
    size: int, group: GroupCoordinator, alignment: int = 1
) -> tuple[int, int]:
    """Return the contiguous balanced partition, without dummy channels."""
    if size % alignment:
        raise ValueError(f"Vision width {size} is not aligned to {alignment}")
    units, remainder = divmod(size // alignment, group.world_size)
    rank = group.rank_in_group
    start = (rank * units + min(rank, remainder)) * alignment
    width = (units + (rank < remainder)) * alignment
    return start, width


class VisionColumnParallelLinear(ReplicatedLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        vision_group: GroupCoordinator,
        alignment: int = 1,
        packed_qkv: bool = False,
        quant_config: QuantizationConfig | None = None,
        disable_tp: bool = False,
        **kwargs,
    ):
        self.partition_start, self.partition_size = vision_partition(
            output_size, vision_group, alignment
        )
        self.full_output_size = output_size
        self.packed_qkv = packed_qkv
        self._rdna_i8_global_shape = (
            output_size * (3 if packed_qkv else 1),
            input_size,
        )
        super().__init__(
            input_size,
            self.partition_size * (3 if packed_qkv else 1),
            quant_config=quant_config,
            disable_tp=True,
            **kwargs,
        )
        if not isinstance(self.quant_method, UnquantizedLinearMethod):
            raise ValueError("Independent vision TP requires unquantized projections")

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str | None = None,
    ):
        if shard_id is None:
            shard_id = getattr(loaded_weight, "shard_id", None)
        if not self.packed_qkv:
            return super().weight_loader(
                param,
                loaded_weight.narrow(0, self.partition_start, self.partition_size),
            )
        if shard_id is None:
            if loaded_weight.shape[0] != 3 * self.full_output_size:
                raise ValueError("Fused vision QKV must contain complete Q, K and V")
            for index, shard in enumerate(
                loaded_weight.split(self.full_output_size, 0)
            ):
                self.weight_loader(param, shard, ("q", "k", "v")[index])
            return
        index = ("q", "k", "v").index(shard_id)
        target = param.data.narrow(0, index * self.partition_size, self.partition_size)
        source = loaded_weight.narrow(0, self.partition_start, self.partition_size)
        if target.shape != source.shape:
            raise ValueError(
                f"Vision QKV shape mismatch: {source.shape}, {target.shape}"
            )
        target.copy_(source)


class VisionOutputParallelLinear(ReplicatedLinear):
    """Gather inputs, project owned output rows with bias, then gather outputs."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        vision_group: GroupCoordinator,
        alignment: int = 1,
        quant_config: QuantizationConfig | None = None,
        disable_tp: bool = False,
        **kwargs,
    ):
        self.vision_group = vision_group
        self.full_input_size = input_size
        self.full_output_size = output_size
        self._rdna_i8_global_shape = (output_size, input_size)
        self.input_partition_start, self.input_partition_size = vision_partition(
            input_size, vision_group, alignment
        )
        self.partition_start, self.partition_size = vision_partition(
            output_size, vision_group
        )
        units, remainder = divmod(input_size // alignment, vision_group.world_size)
        input_sizes = tuple(
            (units + (rank < remainder)) * alignment
            for rank in range(vision_group.world_size)
        )
        units, remainder = divmod(output_size, vision_group.world_size)
        output_sizes = tuple(
            units + (rank < remainder) for rank in range(vision_group.world_size)
        )
        self.input_gather_width = max(input_sizes)
        self.input_gather_slices = (
            tuple(
                (rank * self.input_gather_width, size)
                for rank, size in enumerate(input_sizes)
            )
            if any(size != self.input_gather_width for size in input_sizes)
            else ()
        )
        self.output_gather_width = max(output_sizes)
        self.output_gather_slices = (
            tuple(
                (rank * self.output_gather_width, size)
                for rank, size in enumerate(output_sizes)
            )
            if any(size != self.output_gather_width for size in output_sizes)
            else ()
        )
        super().__init__(
            input_size,
            self.partition_size,
            quant_config=quant_config,
            disable_tp=True,
            **kwargs,
        )
        if not isinstance(self.quant_method, UnquantizedLinearMethod):
            raise ValueError("Independent vision TP requires unquantized projections")

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        trailing = () if param.ndim == 1 else (self.full_input_size,)
        if (
            param.ndim not in (1, 2)
            or loaded_weight.shape != (self.full_output_size, *trailing)
            or not loaded_weight.is_floating_point()
            or getattr(loaded_weight, "packed_dim", None) is not None
        ):
            raise ValueError("Vision output loader requires full dense output rows")
        super().weight_loader(
            param,
            loaded_weight.narrow(0, self.partition_start, self.partition_size),
        )

    def _gather_columns(
        self,
        x: torch.Tensor,
        width: int,
        slices: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        if self.vision_group.world_size == 1:
            return x
        if x.shape[-1] != width:
            x = F.pad(x, (0, width - x.shape[-1]))
        else:
            x = x.contiguous()
        gathered = self.vision_group.all_gather(x, dim=-1)
        if not slices:
            return gathered
        return torch.cat(
            [gathered.narrow(-1, start, count) for start, count in slices],
            dim=-1,
        )

    def forward(self, x: torch.Tensor):
        x = self._gather_columns(
            x,
            self.input_gather_width,
            self.input_gather_slices,
        )
        output = self.quant_method.apply(self, x, self.bias)
        output = self._gather_columns(
            output,
            self.output_gather_width,
            self.output_gather_slices,
        )
        return (output, None) if self.return_bias else output
