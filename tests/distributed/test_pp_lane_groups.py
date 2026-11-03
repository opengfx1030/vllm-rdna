# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Each lane's PP tensor-dict send/recv uses its own device group.

The lane P communicator set exists for collectives, but the PP
intermediate-tensor p2p used the production ``device_group``: two lane
threads on one PP rank then interleave their sends/receives and the first
decode hangs (TP1/PP2, 27B, 2026-11-03). These tests pin the routing.
"""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed import parallel_state as ps


class _FakeCoordinator(ps.GroupCoordinator):
    @property
    def device_communicator(self):
        # Mirrors the real property: only lane P sees the lane set.
        if ps.current_lane() == ps.Lane.PREFILL:
            return self._fake_lane_comm
        return None


def _fake_send(monkeypatch, lane, current):
    production_group = object()
    lane_group = object()
    cpu_group = object()
    coord = object.__new__(_FakeCoordinator)
    coord.world_size = 2
    coord.rank_in_group = 0
    coord.ranks = [0, 1]
    coord.device_group = production_group
    coord.cpu_group = cpu_group
    coord.use_cpu_custom_send_recv = False
    coord._fake_lane_comm = SimpleNamespace(
        device_group=lane_group, cpu_group=cpu_group
    )
    coord._pending_isends = deque()
    coord.isend_object = lambda obj, dst: object()
    coord._reap_completed_isends = lambda: None
    coord._should_use_all_gather = lambda *args: False
    monkeypatch.setattr(ps, "lane_mode_enabled", lambda: True)
    monkeypatch.setattr(ps, "current_lane", lambda: current)
    # A "meta" tensor is a real Tensor whose is_cpu is False, so it takes the
    # device-group branch without initializing a GPU.
    tensor = torch.empty(1, device="meta")
    monkeypatch.setitem(
        ps.__dict__,
        "_split_tensor_dict",
        lambda tensor_dict: ([], [tensor]),
    )
    seen: list[object] = []

    def fake_isend(tensor, dst, group):
        seen.append(group)
        return object()

    monkeypatch.setattr(torch.distributed, "isend", fake_isend)
    ps.GroupCoordinator.isend_tensor_dict(coord, {"x": tensor}, dst=1)
    return seen[0], production_group, lane_group


def test_pp_lane_p_sends_on_its_own_device_group(monkeypatch):
    group, production_group, lane_group = _fake_send(
        monkeypatch, lane=object(), current=ps.Lane.PREFILL
    )
    assert group is lane_group
    assert group is not production_group


def test_pp_lane_d_sends_on_the_production_group(monkeypatch):
    group, production_group, _ = _fake_send(
        monkeypatch, lane=None, current=ps.Lane.DECODE
    )
    assert group is production_group


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_pp_lane_send_routing_needs_gpu():
    pytest.skip("routing unit test above; this keeps GPU CI honest")
