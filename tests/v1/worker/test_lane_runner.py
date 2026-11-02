# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-lane runner tests (S5): second runner, sharing, stream binding.

The prefill-lane runner shares the primary's weights and KV/mamba tensors and
owns everything else. These tests exercise the runner-side contract without a
model; the forward-equivalence smoke test needs one GPU and is skipped without
one.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.lane as lane_module
import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.v1.lane import Lane, lane_scope
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

HAS_GPU = torch.cuda.is_available()


@pytest.fixture
def lane_mode(monkeypatch: pytest.MonkeyPatch):
    """Enable lane mode for one test without leaking it to its neighbours."""
    monkeypatch.setattr(lane_module, "_lane_mode", True)


def _lane_ctx(stream):
    return SimpleNamespace(current_stream=lambda: stream, on_batch_done=Mock())


# --------------------------------------------------------------------------
# construction and sharing (R-S5-1, R-S5-2)
# --------------------------------------------------------------------------


def test_prefill_runner_requires_a_primary() -> None:
    runner = SimpleNamespace(_is_prefill_lane=True)
    with pytest.raises(ValueError, match="must share a primary runner"):
        GPUModelRunner.load_model(runner)


def test_decode_runner_cannot_share_a_primary() -> None:
    runner = SimpleNamespace(_is_prefill_lane=False)
    with pytest.raises(ValueError, match="Only the prefill-lane runner"):
        GPUModelRunner.load_model(runner, shared_from=SimpleNamespace())


def test_prefill_load_model_binds_the_primary_weights() -> None:
    primary = SimpleNamespace(model=object(), speculator=None)
    runner = SimpleNamespace(
        _is_prefill_lane=True,
        speculator=None,
        _init_components_after_model_load=Mock(),
    )
    GPUModelRunner.load_model(runner, shared_from=primary)
    assert runner.model is primary.model
    # Sharing is not a second weight load: no memory is charged to P.
    assert runner.model_memory_usage == 0
    runner._init_components_after_model_load.assert_called_once_with(False, False)


def test_prefill_initialize_kv_cache_requires_a_primary() -> None:
    runner = SimpleNamespace(_is_prefill_lane=True)
    with pytest.raises(ValueError, match="must share a primary runner"):
        GPUModelRunner.initialize_kv_cache(runner, object())


def test_shared_kv_cache_verification_accepts_the_primary_tensors() -> None:
    cache = torch.zeros(8)
    primary = SimpleNamespace(kv_caches=[cache, cache[:4]])
    runner = SimpleNamespace(kv_caches=[cache, cache[:4]])
    GPUModelRunner._verify_shared_kv_cache(runner, primary)


def test_shared_kv_cache_verification_rejects_a_second_allocation() -> None:
    primary = SimpleNamespace(kv_caches=[torch.zeros(8)])
    runner = SimpleNamespace(kv_caches=[torch.zeros(8)])
    with pytest.raises(RuntimeError, match="not the primary's tensor"):
        GPUModelRunner._verify_shared_kv_cache(runner, primary)


def test_shared_kv_cache_verification_rejects_a_count_mismatch() -> None:
    primary = SimpleNamespace(kv_caches=[torch.zeros(8)])
    runner = SimpleNamespace(kv_caches=[torch.zeros(8), torch.zeros(8)])
    with pytest.raises(RuntimeError, match="different number of KV caches"):
        GPUModelRunner._verify_shared_kv_cache(runner, primary)


def test_prefill_runner_never_captures_graphs() -> None:
    runner = SimpleNamespace(_is_prefill_lane=True)
    with pytest.raises(RuntimeError, match="never captures"):
        GPUModelRunner.capture_model(runner)


def test_prefill_shutdown_releases_only_its_own_state() -> None:
    runner = SimpleNamespace(
        _is_prefill_lane=True,
        _ple_offload_connector=None,
        cudagraph_manager=object(),
        fast_prefill=object(),
        speculator=object(),
        kv_caches=[object()],
        attn_groups=[[object()]],
        model_state=object(),
        model=object(),
        primary=object(),
    )
    GPUModelRunner.shutdown(runner)
    assert not hasattr(runner, "model")
    assert runner.primary is None
    assert runner.model_state is None
    assert runner.kv_caches == []
    assert runner.attn_groups == []


# --------------------------------------------------------------------------
# stream binding (R-S5-3, I5)
# --------------------------------------------------------------------------


def test_main_stream_follows_the_lane_context() -> None:
    lane_stream = object()
    runner = SimpleNamespace(
        lane_ctx=_lane_ctx(lane_stream), _main_stream_cache=object()
    )
    assert GPUModelRunner.__dict__["main_stream"].fget(runner) is lane_stream


def test_main_stream_is_cached_when_lane_mode_is_off() -> None:
    cached = object()
    runner = SimpleNamespace(lane_ctx=None, _main_stream_cache=cached)
    assert GPUModelRunner.__dict__["main_stream"].fget(runner) is cached


def test_output_copy_stream_is_the_lane_stream() -> None:
    lane_stream = object()
    runner = SimpleNamespace(
        _output_copy_stream_impl=None, lane_ctx=_lane_ctx(lane_stream)
    )
    assert GPUModelRunner.__dict__["output_copy_stream"].fget(runner) is lane_stream


def test_output_copy_stream_keeps_the_helper_stream_when_flag_off() -> None:
    helper = object()
    runner = SimpleNamespace(_output_copy_stream_impl=helper, lane_ctx=None)
    assert GPUModelRunner.__dict__["output_copy_stream"].fget(runner) is helper


@pytest.mark.skipif(not HAS_GPU, reason="Requires a GPU")
def test_draft_tokens_handler_creates_no_stream_in_lane_mode(
    lane_mode: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[object] = []

    def no_stream(*args, **kwargs):
        created.append((args, kwargs))
        return object()

    monkeypatch.setattr(torch.cuda, "Stream", no_stream)
    handler = DraftTokensHandler(torch.device("cuda"))
    # No helper stream exists; the property resolves the current (lane) stream.
    assert created == []
    assert handler._copy_stream is None


@pytest.mark.skipif(not HAS_GPU, reason="Requires a GPU")
def test_draft_tokens_handler_keeps_its_stream_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = object()
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: helper)
    handler = DraftTokensHandler(torch.device("cuda"))
    assert handler.copy_stream is helper


def test_lane_done_hook_is_called_once_with_the_batch(lane_mode: None) -> None:
    ctx = _lane_ctx(object())
    batch = SimpleNamespace(lane_seq=7)
    runner = SimpleNamespace(lane_ctx=ctx, _lane_batch=batch)
    GPUModelRunner._notify_lane_batch_done(runner)
    ctx.on_batch_done.assert_called_once_with(batch)
    assert runner._lane_batch is None
    # A batch is never reported twice, e.g. when the next execute_model fails.
    GPUModelRunner._notify_lane_batch_done(runner)
    ctx.on_batch_done.assert_called_once()


def test_lane_done_hook_is_a_no_op_when_flag_off() -> None:
    runner = SimpleNamespace(lane_ctx=None, _lane_batch=None)
    GPUModelRunner._notify_lane_batch_done(runner)


# --------------------------------------------------------------------------
# warmup and dummy runs (R-S5-7)
# --------------------------------------------------------------------------


def test_lane_warmup_is_prefill_only() -> None:
    runner = SimpleNamespace(_is_prefill_lane=False)
    with pytest.raises(ValueError, match="only valid"):
        GPUModelRunner.lane_warmup(runner, [8])


def test_lane_warmup_runs_under_workspace_lane_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lanes: list[int] = []
    real_use_workspace_lane = model_runner_module.use_workspace_lane

    def spy(lane):
        lanes.append(lane)
        return real_use_workspace_lane(lane)

    monkeypatch.setattr(model_runner_module, "use_workspace_lane", spy)
    calls: list[tuple[int, dict]] = []
    runner = SimpleNamespace(
        _is_prefill_lane=True,
        _lane_warmup_done=False,
        lane_ctx=None,
        _dummy_run=lambda num_tokens, **kwargs: calls.append((num_tokens, kwargs)),
    )
    GPUModelRunner.lane_warmup(runner, [8, 32])
    assert lanes == [int(Lane.PREFILL)]
    assert [num_tokens for num_tokens, _ in calls] == [8, 32]
    assert all(kwargs["valid_dummy_state_slots"] for _, kwargs in calls)
    assert all(kwargs["randomize_inputs"] for _, kwargs in calls)
    assert runner._lane_warmup_done

    with pytest.raises(RuntimeError, match="exactly once"):
        GPUModelRunner.lane_warmup(runner, [8])


def test_dummy_run_is_refused_after_serving_starts(lane_mode: None) -> None:
    runner = SimpleNamespace(_lane_served_batch=True)
    with pytest.raises(RuntimeError, match="forbids dummy runs"):
        GPUModelRunner._dummy_run(runner, 8)


def test_dummy_run_is_allowed_before_serving(lane_mode: None) -> None:
    """The guard must not fire during startup warmup/profile runs."""
    runner = SimpleNamespace(_lane_served_batch=False)
    # The guard is the first statement that can reject the call; the rest of
    # _dummy_run needs a real runner, so reaching the next error is enough.
    with pytest.raises(AttributeError):
        GPUModelRunner._dummy_run(runner, 8)


# --------------------------------------------------------------------------
# speculative decoding (R-S5-2)
# --------------------------------------------------------------------------


def test_shared_speculator_binds_the_primary_draft_model() -> None:
    draft = object()
    primary = SimpleNamespace(
        model=draft,
        draft_attn_layer_names={"mtp.layers.0"},
        supports_mm_inputs=False,
    )
    speculator = SimpleNamespace(
        load_draft_model=Mock(side_effect=AssertionError("second draft load")),
        _validate_local_argmax_reduction=Mock(),
        use_acceptance_estimator=False,
        use_local_argmax_reduction=False,
    )
    DraftModelSpeculator.load_model(
        speculator, target_model=object(), shared_from=primary
    )
    assert speculator.model is draft
    assert speculator.draft_attn_layer_names == {"mtp.layers.0"}
    assert speculator.supports_mm_inputs is False
    speculator.load_draft_model.assert_not_called()
    speculator._validate_local_argmax_reduction.assert_called_once()


def test_lane_scope_is_visible_to_the_model() -> None:
    """The model keys its per-lane scratch on the runner thread's lane."""
    assert lane_module.current_lane() == Lane.DECODE
    with lane_scope(Lane.PREFILL):
        assert lane_module.current_lane() == Lane.PREFILL
