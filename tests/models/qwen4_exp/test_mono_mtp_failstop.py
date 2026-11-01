# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contracts for the Mono MTP and expert-cache post-step fail-stop."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.amd import mono_decode_rdna2
from vllm.models.qwen4_exp.amd.mono.rdna2 import failstop


@pytest.fixture
def isolated_scope(monkeypatch):
    # The real workspace registry may be populated by other test modules.
    spaces: dict[tuple[int, str], torch.Tensor] = {}
    zeros: dict[int, torch.Tensor] = {}
    monkeypatch.setattr(failstop, "_WORKSPACES", spaces)
    monkeypatch.setattr(failstop, "_ZERO", zeros)
    return spaces, zeros


def test_opted_in_mono_drafter_or_expert_cache_enables_fault_check(monkeypatch):
    cfg = object()
    for options, expected in (
        (None, False),
        (SimpleNamespace(mtp_drafter=False), False),
        (SimpleNamespace(mtp_drafter=True), True),
        (SimpleNamespace(mtp_drafter=False, expert_cache_slots=0), False),
        (SimpleNamespace(mtp_drafter=False, expert_cache_slots=96), True),
        (SimpleNamespace(mtp_drafter=True, expert_cache_slots=96), True),
    ):
        monkeypatch.setattr(mono_decode_rdna2, "options", lambda _, opts=options: opts)
        assert failstop.enabled(cfg) is expected


@pytest.mark.parametrize(("local_fault", "remote_fault"), [(1, 0), (0, 1)])
def test_tp_fault_is_not_lost_when_this_rank_falls_back(
    monkeypatch, isolated_scope, local_fault, remote_fault
):
    spaces, zeros = isolated_scope
    device = torch.device("cpu:0")
    # A target Mono arena on the same device must never be selected.
    spaces[(0, "model")] = torch.full((64,), 9, dtype=torch.uint8)
    if local_fault:
        spaces[(0, "mtp")] = torch.zeros(64, dtype=torch.uint8)
        spaces[(0, "mtp")].view(torch.int32)[2] = local_fault

    class OutOfPlaceTP:
        def __init__(self) -> None:
            self.calls: list[torch.Tensor] = []

        def all_reduce(self, local: torch.Tensor) -> torch.Tensor:
            self.calls.append(local)
            result = local.clone()
            result += remote_fault
            assert result.data_ptr() != local.data_ptr()
            return result

    group = OutOfPlaceTP()
    monkeypatch.setattr(failstop, "get_tp_group", lambda: group)
    result = failstop.collect_tp_error(device)
    assert len(group.calls) == 1
    assert result.item() == 1
    with pytest.raises(RuntimeError, match="restart the engine"):
        failstop.raise_if_error(result)
    if local_fault:
        assert group.calls[0].data_ptr() == spaces[(0, "mtp")].data_ptr() + 8
        assert not zeros
    else:
        assert group.calls[0].data_ptr() == zeros[0].data_ptr()
        assert zeros[0].item() == 0  # no sticky poison in the fallback sentinel
        assert failstop.local_error_view(device).data_ptr() == zeros[0].data_ptr()


def test_clean_fallback_participates_once_without_mutating_zero(
    monkeypatch, isolated_scope
):
    _, zeros = isolated_scope
    device = torch.device("cpu:0")
    calls = 0

    class OutOfPlaceTP:
        def all_reduce(self, local: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return local.clone()

    monkeypatch.setattr(failstop, "get_tp_group", lambda: OutOfPlaceTP())
    result = failstop.collect_tp_error(device)
    assert calls == 1
    assert result.item() == 0
    assert result.data_ptr() != zeros[0].data_ptr()
    assert zeros[0].item() == 0
    failstop.raise_if_error(result)


def test_diagnostic_wait_only_retains_local_fault_without_collective(
    monkeypatch, isolated_scope
):
    spaces, _ = isolated_scope
    arena = torch.zeros(64, dtype=torch.uint8)
    arena.view(torch.int32)[2] = 1
    spaces[(0, "model-offload")] = arena
    monkeypatch.setattr(failstop, "_USE_TP_COLLECTIVE", False)
    monkeypatch.setattr(
        failstop, "get_tp_group", lambda: pytest.fail("unexpected TP collective")
    )
    result = failstop.collect_tp_error(torch.device("cpu:0"))
    assert result.item() == 1
    assert result.data_ptr() == arena.data_ptr() + 8


def test_offline_recovery_clears_entire_arena_and_preserves_pointer(
    monkeypatch, isolated_scope
):
    spaces, _ = isolated_scope
    device = torch.device("cpu:0")
    arena = torch.full((256,), 7, dtype=torch.uint8)
    spaces[(0, "mtp")] = arena
    spaces[(0, "model")] = torch.full((256,), 5, dtype=torch.uint8)
    before_ptr = arena.data_ptr()
    syncs: list[str] = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: syncs.append("sync"))
    failstop.clear_mtp_fault_for_recovery(device)
    assert syncs == ["sync", "sync"]
    assert arena.data_ptr() == before_ptr
    assert torch.count_nonzero(arena).item() == 0
    assert torch.all(spaces[(0, "model")] == 5).item()


def test_async_fault_event_precedes_token_parsing(monkeypatch):
    from vllm.v1.worker.gpu import async_utils

    order: list[str] = []

    class Event:
        def __init__(self, *, blocking: bool):
            assert blocking

        def record(self, stream):
            order.append("fault-record")

        def synchronize(self):
            order.append("fault-wait")

    class CopyStream:
        def wait_stream(self, main):
            order.append("wait-main")

    class Fault:
        device = torch.device("cpu:0")

        def to(self, device, *, non_blocking):
            assert device == "cpu" and non_blocking
            order.append("fault-copy")
            return torch.tensor([1], dtype=torch.int32)

        def record_stream(self, stream):
            order.append("retain-until-copy")

    class NeverParse:
        def tolist(self):
            raise AssertionError("faulty tokens were parsed")

    monkeypatch.setattr(async_utils.torch.cuda, "Event", Event)
    monkeypatch.setattr(async_utils, "stream", lambda *_: nullcontext())
    output = object.__new__(async_utils.AsyncOutput)
    output.copy_event = SimpleNamespace(synchronize=lambda: order.append("copy-wait"))
    output.sampled_token_ids = NeverParse()
    output._mtp_fault_event = None
    fault = Fault()
    copy_stream = CopyStream()
    output.enqueue_mtp_fault(fault, object(), copy_stream)
    assert output._mtp_fault_gpu is fault  # retain GPU value until copy/event
    with pytest.raises(RuntimeError, match="restart the engine"):
        output.get_output()
    assert order == [
        "wait-main",
        "fault-copy",
        "retain-until-copy",
        "fault-record",
        "copy-wait",
        "fault-wait",
    ]


def test_async_without_fault_keeps_token_count_path(monkeypatch):
    from vllm.v1.worker.gpu import async_utils

    output = object.__new__(async_utils.AsyncOutput)
    output.copy_event = SimpleNamespace(synchronize=lambda: None)
    output._mtp_fault_event = None
    output.sampled_token_ids = np.array([[12, 13]])
    output.num_sampled_tokens_np = np.array([1])
    output.model_runner_output = SimpleNamespace(req_ids=["r"])
    output.sampling_mask_tensors = None
    output.num_nans = None
    output.logprobs_tensors = None
    output.prompt_logprobs_dict = {}
    output.prompt_token_id_logprobs_dict = {}
    output.pending_aux_output = None
    output._has_fault = None
    assert output.get_output().sampled_token_ids == [[12]]


def test_offload_only_fault_keeps_direct_header_view(isolated_scope):
    spaces, zeros = isolated_scope
    arena = torch.zeros(64, dtype=torch.uint8)
    arena.view(torch.int32)[2:5] = torch.tensor([1, 6, 13])
    spaces[(0, "model-offload")] = arena
    error = failstop.local_error_view(torch.device("cpu:0"))
    assert error.data_ptr() == arena.data_ptr() + 8
    assert error.item() == 1
    assert not zeros


def test_combined_fault_scopes_sum_without_mutating_headers(isolated_scope):
    spaces, zeros = isolated_scope
    for scope, code in (("mtp", 1), ("model-offload", 2)):
        arena = torch.zeros(64, dtype=torch.uint8)
        arena.view(torch.int32)[2] = code
        spaces[(0, scope)] = arena
    before = {scope: arena.clone() for (_, scope), arena in spaces.items()}
    error = failstop.local_error_view(torch.device("cpu:0"))
    assert error.item() == 3
    assert error.data_ptr() not in [arena.data_ptr() + 8 for arena in spaces.values()]
    for (_, scope), arena in spaces.items():
        assert torch.equal(arena, before[scope])
    assert not zeros


def test_resident_model_fault_is_not_an_offload_fault(isolated_scope):
    spaces, zeros = isolated_scope
    spaces[(0, "model")] = torch.full((64,), 9, dtype=torch.uint8)
    error = failstop.local_error_view(torch.device("cpu:0"))
    assert error.item() == 0
    assert error.data_ptr() == zeros[0].data_ptr()
    failstop.raise_if_error(error, torch.device("cpu:0"))


def test_fault_details_include_only_local_monitored_headers(isolated_scope):
    spaces, _ = isolated_scope
    for scope, header in (("mtp", [1, 3, 7]), ("model-offload", [1, 6, 11])):
        arena = torch.zeros(64, dtype=torch.uint8)
        arena.view(torch.int32)[2:5] = torch.tensor(header)
        spaces[(0, scope)] = arena
    spaces[(0, "model")] = torch.full((64,), 9, dtype=torch.uint8)
    with pytest.raises(RuntimeError) as exc:
        failstop.raise_if_error(torch.tensor([2]), torch.device("cpu:0"))
    message = str(exc.value)
    assert "'mtp': [1, 3, 7]" in message
    assert "'model-offload': [1, 6, 11]" in message
    assert "'model':" not in message


def test_remote_fault_reports_clean_local_headers(isolated_scope):
    spaces, _ = isolated_scope
    spaces[(0, "model-offload")] = torch.zeros(64, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match=r"'model-offload': \[0, 0, 0\]"):
        failstop.raise_if_error(torch.tensor([1]), torch.device("cpu:0"))


def test_success_does_not_read_diagnostic_headers(monkeypatch, isolated_scope):
    spaces, _ = isolated_scope

    class NeverRead:
        def __getitem__(self, key):
            raise AssertionError("success path inspected a diagnostic header")

    spaces[(0, "mtp")] = NeverRead()
    spaces[(0, "model-offload")] = NeverRead()
    failstop.raise_if_error(torch.tensor([0]), torch.device("cpu:0"))
