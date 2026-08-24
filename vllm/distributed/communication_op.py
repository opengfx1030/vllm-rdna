# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from .parallel_state import get_tp_group

_FAKE_TENSOR_TYPES = (
    torch._subclasses.fake_tensor.FakeTensor,
    torch._subclasses.functional_tensor.FunctionalTensor,
)


def _is_fake_tensor(t: torch.Tensor) -> bool:
    return isinstance(t, _FAKE_TENSOR_TYPES)


@torch.compiler.allow_in_graph
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    if _is_fake_tensor(input_):
        return input_
    return get_tp_group().all_reduce(input_)


@torch.compiler.allow_in_graph
def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    if _is_fake_tensor(input_):
        return input_
    return get_tp_group().all_gather(input_, dim)


@torch.compiler.allow_in_graph
def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    if _is_fake_tensor(input_):
        return input_
    return get_tp_group().reduce_scatter(input_, dim)


@torch.compiler.allow_in_graph
def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    if _is_fake_tensor(input_):
        return input_
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
