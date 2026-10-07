# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP collectives must be opaque to inductor.

Returning the same FakeTensor from tensor_model_parallel_all_reduce made
inductor treat all-reduce as identity. TP=2 Qwen3.5 hybrid greedy then
emitted looping garbage while TP=1 (no reduction) stayed correct.
"""

import torch

from vllm.distributed.communication_op import (
    _tp_all_reduce_fake,
    tensor_model_parallel_all_reduce,
)


def test_tp_all_reduce_is_opaque_custom_op():
    assert hasattr(torch.ops.vllm, "tensor_model_parallel_all_reduce")
    assert hasattr(torch.ops.vllm, "tensor_model_parallel_all_gather")
    assert hasattr(torch.ops.vllm, "tensor_model_parallel_reduce_scatter")


def test_tp_all_reduce_fake_is_not_identity():
    t = torch.zeros(2, 4)
    out = _tp_all_reduce_fake(t)
    assert out is not t
    assert out.shape == t.shape
    assert out.dtype == t.dtype


def test_tp_all_reduce_fake_dispatch_is_not_identity():
    t = torch.zeros(2, 4)
    with torch._subclasses.fake_tensor.FakeTensorMode():
        fake_in = torch.empty(2, 4)
        fake_out = torch.ops.vllm.tensor_model_parallel_all_reduce(fake_in)
        assert fake_out is not fake_in
        assert tuple(fake_out.shape) == tuple(fake_in.shape)


def test_tp_all_reduce_python_wrapper_calls_custom_op():
    import inspect

    body = inspect.getsource(tensor_model_parallel_all_reduce)
    assert "torch.ops.vllm.tensor_model_parallel_all_reduce" in body
    assert "return input_" not in body
