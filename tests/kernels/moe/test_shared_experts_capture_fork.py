# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx10x: no shared-experts aux-stream fork inside a CUDA-graph capture."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner import shared_experts as se_mod
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)


def _shared_experts(no_fork_in_capture: bool) -> SharedExperts:
    se = SharedExperts.__new__(SharedExperts)
    torch.nn.Module.__init__(se)
    se._layer = SimpleNamespace()
    se._moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            enable_eplb=False,
            all2all_backend="allgather_reducescatter",
            use_fi_nvl_two_sided_kernels=False,
        )
    )
    se._mk_can_overlap_shared_experts = lambda: False
    se._stream = object()
    se._no_fork_in_capture = no_fork_in_capture
    return se


@pytest.mark.parametrize("no_fork_in_capture", [False, True])
@pytest.mark.parametrize("capturing", [False, True])
def test_capture_fork_gate(monkeypatch, no_fork_in_capture, capturing):
    monkeypatch.setattr(se_mod.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    se = _shared_experts(no_fork_in_capture)
    order = se._determine_shared_experts_order(torch.empty(3, 8))
    if no_fork_in_capture and capturing:
        assert order == SharedExpertsOrder.NO_OVERLAP
    else:
        assert order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
