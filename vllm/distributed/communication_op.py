# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from vllm.platforms.rdna import on_rdna_family
from vllm.utils.torch_utils import direct_register_custom_op

from .parallel_state import get_tp_group

# Resolved once: these wrappers run inside Dynamo-traced model code.
_ON_RDNA = on_rdna_family()


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
    if _ON_RDNA:
        # RDNA: opaque op so Dynamo does not trace into the communicator.
        return torch.ops.vllm.tensor_model_parallel_all_reduce(input_)
    return get_tp_group().all_reduce(input_)


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
    if _ON_RDNA:
        return torch.ops.vllm.tensor_model_parallel_all_gather(input_, dim)
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_all_gather_via_all_reduce(
    input_: torch.Tensor,
) -> torch.Tensor:
    """Last-dim all-gather built from an all-reduce of a zero-padded buffer.

    Rank r owns columns [r*shard, (r+1)*shard); every other column is zero, so
    the sum is exactly the layout ``tensor_model_parallel_all_gather`` returns.
    Only for hosts where ``all_gather_into_tensor`` is corrupt (P2P-less
    gfx1030: wrong-rank shard at offset 0, zero holes) while ``all_reduce`` is
    exact. It moves ``tp`` times the data of a real gather, so it is opt-in
    (``VLLM_RDNA_HEAD_ALLREDUCE`` / ``VLLM_RDNA_GATHER_ALLREDUCE``).

    Goes through ``tensor_model_parallel_all_reduce`` so small tensors take the
    HIP one-shot ``rdna_ar`` path. GPU-resident throughout, so CUDA-graph
    capture stays legal.
    """
    group = get_tp_group()
    shard = input_.shape[-1]
    gathered = input_.new_zeros((*input_.shape[:-1], shard * group.world_size))
    lo = group.rank_in_group * shard
    gathered[..., lo : lo + shard].copy_(input_)
    return tensor_model_parallel_all_reduce(gathered)


def tensor_model_parallel_argmax_pair(
    vals: torch.Tensor, idx: torch.Tensor
) -> torch.Tensor | None:
    """Global per-row argmax over TP ranks of (value, global index) pairs.

    One rdna_ar HIP launch (push, barrier, pick) replaces the pair all-gather
    and the argmax/gather/cast kernels behind it. Returns None when this group
    has no eligible rdna_ar instance; callers then use their generic path.
    """
    communicator = get_tp_group().device_communicator
    rdna_ar_comm = getattr(communicator, "rdna_ar_comm", None)
    if rdna_ar_comm is not None and rdna_ar_comm.should_argmax(vals, idx):
        return rdna_ar_comm.argmax(vals, idx)
    return None


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
    if _ON_RDNA:
        return torch.ops.vllm.tensor_model_parallel_reduce_scatter(input_, dim)
    return get_tp_group().reduce_scatter(input_, dim)


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
