# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.phase_timer import PhaseTimer
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


class _DefaultSpeculator(BaseSpeculator):
    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(self, *args, **kwargs):
        raise NotImplementedError


class _StopDummyRun(Exception):
    pass


def test_graph_phase_events_require_opt_in_capture(monkeypatch):
    """Warmups and ordinary runs must not create diagnostic event nodes."""
    capture = False
    recorded = []

    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True, "external": True}

        def record(self):
            recorded.append(self)

    monkeypatch.setattr("torch.cuda.Event", Event)
    monkeypatch.setattr("torch.cuda.is_current_stream_capturing", lambda: capture)
    timer = PhaseTimer(50)
    timer.graph_mark("disabled")
    timer.graph_marks = []
    timer.graph_mark("warmup")
    assert not recorded
    capture = True
    timer.graph_mark("body")
    assert timer.graph_marks == [("body", recorded[0])]


def test_graph_op_capture_and_sampled_replay(monkeypatch):
    phase_mod = importlib.import_module("vllm.v1.worker.gpu.phase_timer")
    graph_mod = importlib.import_module(
        "vllm.v1.worker.gpu.spec_decode.dflash.cudagraph"
    )
    capturing = False
    events: list[object] = []
    logs = []

    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True, "external": True}
            self.position = -1

        def record(self):
            self.position = len(events)
            events.append(self)

        def elapsed_time(self, other):
            return (other.position - self.position) * 0.1

        def synchronize(self):
            pass

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(phase_mod, "GRAPH_OP_TIMING", True)
    monkeypatch.setattr(graph_mod, "GRAPH_OP_TIMING", True)
    monkeypatch.setattr(graph_mod, "GRAPH_TIMING", True)
    monkeypatch.setattr(
        graph_mod, "_prepare_dflash_inputs_to_capture", lambda *a, **kw: (None, None)
    )
    monkeypatch.setattr(graph_mod.logger, "info", lambda *args: logs.append(args))

    timer = PhaseTimer(2)
    timer.cur = (0, [], [])
    previous_marks = [("previous", Event())]
    previous_ops = [("previous", Event(), Event())]
    timer.graph_marks = previous_marks
    timer.graph_ops = previous_ops
    monkeypatch.setattr(graph_mod, "phase_timer", timer)
    monkeypatch.setattr(phase_mod, "phase_timer", timer)
    desc = BatchExecutionDescriptor(
        CUDAGraphMode.FULL, num_tokens=24, num_reqs=8, uniform_token_count=3
    )
    x = torch.empty(3, 8, dtype=torch.bfloat16)
    w = torch.empty(4, 8, dtype=torch.bfloat16)
    calls = []

    def op(a, b):
        calls.append((a, b))
        return a

    wrapped_op = phase_mod.wrap_op_timing("rocm_unquantized_gemm", op)

    def base_capture(self, create_forward_fn, _progress_bar_desc):
        nonlocal capturing
        create_forward_fn(desc, True)(CUDAGraphMode.FULL)
        assert not self._phase_events
        assert not self._op_events
        capturing = True
        create_forward_fn(desc, False)(CUDAGraphMode.FULL)
        capturing = False

    monkeypatch.setattr(CudaGraphManager, "capture", base_capture)
    monkeypatch.setattr(CudaGraphManager, "run_fullgraph", lambda self, desc: None)
    manager = DFlashCudaGraphManager.__new__(DFlashCudaGraphManager)
    manager.dp_size = 1
    manager.capture(
        forward_fn=lambda *args: (wrapped_op(x, w), wrapped_op(x, w)),
        input_buffers=None,
        block_tables=None,
        attn_groups=[],
        kv_cache_config=None,
        max_model_len=16,
        causal=False,
        precompute_context_kv=lambda _: wrapped_op(x, w),
    )
    assert len(calls) == 6  # Warmup and capture both execute the Python body.
    assert timer.graph_marks is previous_marks
    assert timer.graph_ops is previous_ops
    assert timer.cur is not None
    assert not timer.cur[2]
    assert len(manager._op_events[desc]) == 3
    assert all(
        "3x8:bfloat16,4x8:bfloat16" in name for name, *_ in manager._op_events[desc]
    )
    retained_ops = manager._op_events[desc]
    manager.run_fullgraph(desc)
    assert not logs
    manager.run_fullgraph(desc)
    assert manager._op_events[desc] is retained_ops
    assert len(logs) == 2
    assert "[DFLASH GRAPH OPS" in logs[1][0]
    assert "#3" in logs[1][-1]


def test_graph_op_wrapper_disabled_by_default(monkeypatch):
    phase_mod = importlib.import_module("vllm.v1.worker.gpu.phase_timer")
    monkeypatch.setattr(phase_mod, "OP_TIMING", False)
    monkeypatch.setattr(phase_mod, "GRAPH_OP_TIMING", False)
    fn = lambda x: x
    assert phase_mod.wrap_op_timing("an_op", fn) is fn


@pytest.mark.parametrize(("mode", "wrapped"), [("0", False), ("3", False), ("4", True)])
def test_custom_op_registration_installs_graph_wrapper(monkeypatch, mode, wrapped):
    torch_utils = importlib.import_module("vllm.utils.torch_utils")
    phase_mod = importlib.import_module("vllm.v1.worker.gpu.phase_timer")
    monkeypatch.setenv("VLLM_RDNA_OP_TIMING", mode)
    registered = {}
    sentinel = lambda x: x

    class Library:
        def define(self, *args, **kwargs):
            pass

        def impl(self, _name, implementation, dispatch_key):
            assert dispatch_key == "CPU"
            registered["implementation"] = implementation

    def op(x: torch.Tensor) -> torch.Tensor:
        return x

    monkeypatch.setattr(torch_utils, "vllm_lib", Library())
    monkeypatch.setattr(phase_mod, "wrap_op_timing", lambda *args: sentinel)
    torch_utils.direct_register_custom_op(
        "test_dflash_graph_timing", op, dispatch_key="CPU"
    )
    assert (registered["implementation"] is sentinel) == wrapped


def _run_dummy_batch(
    speculator,
    *,
    max_num_reqs=512,
    is_profile=False,
    skip_attn=False,
):
    captured = {}

    def execute_model(scheduler_output, **kwargs):
        captured["scheduler_output"] = scheduler_output
        raise _StopDummyRun

    runner = SimpleNamespace(
        max_num_reqs=max_num_reqs,
        max_num_tokens=2048,
        decode_query_len=1,
        speculator=speculator,
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
        is_first_pp_rank=True,
        lora_config=None,
        maybe_dummy_run_with_lora=lambda *args, **kwargs: nullcontext(),
        execute_model=execute_model,
    )

    with pytest.raises(_StopDummyRun):
        GPUModelRunner._dummy_run(
            runner,
            num_tokens=2048,
            skip_attn=skip_attn,
            is_profile=is_profile,
        )

    return captured["scheduler_output"]


def test_default_speculator_query_width_is_one():
    assert _DefaultSpeculator().num_query_per_req == 1


@pytest.mark.parametrize("speculator_cls", [DFlashSpeculator, DSparkSpeculator])
@pytest.mark.parametrize(
    ("is_profile", "skip_attn"),
    [(False, False), (True, False), (True, True)],
)
def test_dummy_run_caps_parallel_draft_requests(speculator_cls, is_profile, skip_attn):
    speculator = speculator_cls.__new__(speculator_cls)
    speculator.num_query_per_req = 5

    scheduler_output = _run_dummy_batch(
        speculator,
        is_profile=is_profile,
        skip_attn=skip_attn,
    )
    assert len(scheduler_output.num_scheduled_tokens) == 409
    assert scheduler_output.total_num_scheduled_tokens == 2048


def test_dummy_run_keeps_already_fitting_parallel_draft_batch():
    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.num_query_per_req = 5

    scheduler_output = _run_dummy_batch(speculator, max_num_reqs=128)
    assert len(scheduler_output.num_scheduled_tokens) == 128


def test_dummy_run_keeps_default_speculator_batch():
    scheduler_output = _run_dummy_batch(_DefaultSpeculator())
    assert len(scheduler_output.num_scheduled_tokens) == 512
