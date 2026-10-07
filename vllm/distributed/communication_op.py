# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from vllm.utils.torch_utils import direct_register_custom_op

from .parallel_state import get_tp_group


def _tp_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_tp_group().all_reduce(input_)


def _tp_all_reduce_fake(input_: torch.Tensor) -> torch.Tensor:
    # Must be a new tensor. Returning `input_` made inductor treat
    # all_reduce as identity (TP=2 Qwen3.5 hybrid greedy garbage).
    return torch.empty_like(input_)


direct_register_custom_op(
    op_name="tensor_model_parallel_all_reduce",
    op_func=_tp_all_reduce,
    fake_impl=_tp_all_reduce_fake,
)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return torch.ops.vllm.tensor_model_parallel_all_reduce(input_)


def _tp_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return get_tp_group().all_gather(input_, dim)


def _tp_all_gather_fake(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    world = get_tp_group().world_size
    out_shape = list(input_.shape)
    out_shape[dim] = out_shape[dim] * world
    return input_.new_empty(out_shape)


direct_register_custom_op(
    op_name="tensor_model_parallel_all_gather",
    op_func=_tp_all_gather,
    fake_impl=_tp_all_gather_fake,
)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return torch.ops.vllm.tensor_model_parallel_all_gather(input_, dim)


def _tp_reduce_scatter(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return get_tp_group().reduce_scatter(input_, dim)


def _tp_reduce_scatter_fake(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    world = get_tp_group().world_size
    out_shape = list(input_.shape)
    out_shape[dim] = out_shape[dim] // world
    return input_.new_empty(out_shape)


direct_register_custom_op(
    op_name="tensor_model_parallel_reduce_scatter",
    op_func=_tp_reduce_scatter,
    fake_impl=_tp_reduce_scatter_fake,
)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return torch.ops.vllm.tensor_model_parallel_reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
