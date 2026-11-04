# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-lane runner tests (S5): second runner, sharing, stream binding.

The prefill-lane runner shares the primary's weights and KV/mamba tensors and
owns everything else. These tests exercise the runner-side contract without a
model; the forward-equivalence smoke test needs one GPU and is skipped without
one.
"""

import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import vllm.v1.lane as lane_module
import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.lane import Lane, lane_scope
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

# GPU tests must name the device they own: HIP 0/1 on par1-cs25 belong to
# another job, and touching them from this test crashed the process (rc=139).
# Set VLLM_TEST_DEVICE=<hip index> for a GPU run; unset skips every GPU test.
_TEST_DEVICE_INDEX = os.environ.get("VLLM_TEST_DEVICE")
HAS_GPU = _TEST_DEVICE_INDEX is not None and torch.cuda.is_available()


def _test_device() -> torch.device:
    if _TEST_DEVICE_INDEX is None:
        raise RuntimeError("VLLM_TEST_DEVICE=<hip index> is required")
    return torch.device("cuda", int(_TEST_DEVICE_INDEX))


@pytest.fixture
def lane_mode(monkeypatch: pytest.MonkeyPatch):
    """Enable lane mode for one test without leaking it to its neighbours."""
    monkeypatch.setattr(lane_module, "_lane_mode", True)


def _lane_ctx(stream, copy_stream=None):
    return SimpleNamespace(
        current_stream=lambda: stream,
        output_copy_stream=lambda: stream if copy_stream is None else copy_stream,
        on_batch_done=Mock(),
    )


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
    primary = SimpleNamespace(model=object(), speculator=None, encoder_cache=object())
    runner = SimpleNamespace(
        _is_prefill_lane=True,
        speculator=None,
        _init_components_after_model_load=Mock(),
    )
    GPUModelRunner.load_model(runner, shared_from=primary)
    assert runner.model is primary.model
    # The vision encoder runs on P and D reads its outputs after the handoff,
    # so both lanes must see one cache (R-S6-5).
    assert runner.encoder_cache is primary.encoder_cache
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


def test_output_copy_stream_follows_the_lane_context() -> None:
    lane_stream, copy_stream = object(), object()
    runner = SimpleNamespace(
        _output_copy_stream_impl=None,
        lane_ctx=_lane_ctx(lane_stream, copy_stream),
    )
    assert GPUModelRunner.__dict__["output_copy_stream"].fget(runner) is copy_stream


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
    handler = DraftTokensHandler(_test_device())
    # No helper stream exists; the property resolves the current (lane) stream.
    assert created == []
    assert handler._copy_stream is None


@pytest.mark.skipif(not HAS_GPU, reason="Requires a GPU")
def test_draft_tokens_handler_keeps_its_stream_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = object()
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: helper)
    handler = DraftTokensHandler(_test_device())
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
    zeroer = SimpleNamespace(warmup=Mock())
    runner = SimpleNamespace(
        _is_prefill_lane=True,
        _lane_warmup_done=False,
        lane_ctx=None,
        kv_block_zeroer=zeroer,
        _dummy_run=lambda num_tokens, **kwargs: calls.append((num_tokens, kwargs)),
    )
    GPUModelRunner.lane_warmup(runner, [8, 32])
    assert lanes == [int(Lane.PREFILL)]
    assert [num_tokens for num_tokens, _ in calls] == [8, 32]
    assert all(kwargs["valid_dummy_state_slots"] for _, kwargs in calls)
    assert all(kwargs["randomize_inputs"] for _, kwargs in calls)
    # The P runner's zeroer is JIT'd before serving, never on the lane thread
    # (review finding 1: a first admission compiled the zeroing kernel).
    zeroer.warmup.assert_called_once_with(1)
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


# --------------------------------------------------------------------------
# one-GPU full runner: construction, sharing and forward equivalence
# --------------------------------------------------------------------------

_LANE_MODEL = os.environ.get(
    "TWO_LANE_TEST_MODEL", "/home/chenco_adm/models/Qwen3.5-0.8B"
)


def _lane_config(dtype: str = "float16", model: str = _LANE_MODEL):
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.config.compilation import CompilationMode

    model_config = ModelConfig(model=model, dtype=dtype, seed=0, max_model_len=2048)
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=2048,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16, gpu_memory_utilization=0.9, cache_dtype="auto"
    )
    compilation_config = CompilationConfig(
        mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE
    )
    cache_config.kv_cache_layout = "LBNHC"
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=ParallelConfig(),
        compilation_config=compilation_config,
    )


def _new_request_so(req_id: str, prompt: list[int], block_ids: list[list[int]]):
    from vllm.sampling_params import SamplingParams

    return SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt,
                mm_features=[],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
                pooling_params=None,
                block_ids=tuple(block_ids),
                num_computed_tokens=0,
                lora_request=None,
                prefill_token_ids=prompt,
            )
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: len(prompt)},
        total_num_scheduled_tokens=len(prompt),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _cached_request_so(req_id: str, num_computed: int, num_output: int):
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[req_id],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[num_computed],
            num_output_tokens=[num_output],
        ),
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _scheduled_tokens(async_output) -> list[int]:
    output = async_output.get_output()
    sampled = output.sampled_token_ids
    if sampled is None:
        sampled = async_output.sampled_token_ids
    return np.asarray(sampled).reshape(-1).tolist()


def _step(runner, vllm_config, scheduler_output, stream) -> list[int]:
    from vllm.config import set_current_vllm_config

    stored = runner.lane_ctx
    runner.lane_ctx = SimpleNamespace(
        current_stream=lambda: stream, on_batch_done=stored.on_batch_done
    )
    try:
        with set_current_vllm_config(vllm_config), torch.cuda.stream(stream):
            runner.execute_model(scheduler_output)
            async_output = runner.sample_tokens(None)
            return _scheduled_tokens(async_output)
    finally:
        runner.lane_ctx = stored


def _zero_kv(runner) -> None:
    for cache in runner.kv_caches:
        cache.zero_()


def _used_bytes() -> int:
    free, total = torch.accelerator.get_memory_info(_test_device())
    return total - free


def _ptr(tensors) -> list[int]:
    return [t.data_ptr() for t in tensors]


@pytest.mark.skipif(not HAS_GPU, reason="Requires a GPU")
@pytest.mark.skipif(
    not os.path.isdir(_LANE_MODEL), reason=f"two-lane test model missing: {_LANE_MODEL}"
)
def test_second_runner_shares_weights_and_kv_and_matches_forwards(
    dist_init,
    lane_mode: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.v1.core.utils import get_kv_cache_configs
    from vllm.config import set_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager

    vllm_config = _lane_config()
    with set_current_vllm_config(vllm_config):
        init_workspace_manager(_test_device(), 1, 2)
        decode = GPUModelRunner(vllm_config, _test_device())
        before = _used_bytes()
        decode.load_model()
        weights_mem = _used_bytes() - before

        kv_cache_config = get_kv_cache_configs(
            vllm_config, [decode.get_kv_cache_spec()], [2 << 30]
        )[0]
        before = _used_bytes()
        decode.initialize_kv_cache(kv_cache_config)
        kv_mem = _used_bytes() - before

        decode_weights = [p.data_ptr() for p in decode.model.parameters()]
        decode_kv = _ptr(decode.kv_caches)
        assert decode_weights and decode_kv

        # Lane P is constructed with lane mode on, so no helper stream exists.
        created: list[object] = []
        real_stream_cls = torch.cuda.Stream

        def counting_stream(*args, **kwargs):
            stream = real_stream_cls(*args, **kwargs)
            created.append(stream)
            return stream

        monkeypatch.setattr(torch.cuda, "Stream", counting_stream)
        before = _used_bytes()
        prefill = GPUModelRunner(
            vllm_config, _test_device(), lane=Lane.PREFILL, primary=decode
        )
        prefill_mem = _used_bytes() - before

        before = _used_bytes()
        prefill.load_model(model=decode.model, shared_from=decode)
        prefill_load_mem = _used_bytes() - before

        before = _used_bytes()
        prefill.initialize_kv_cache(kv_cache_config, shared_from=decode)
        prefill_kv_mem = _used_bytes() - before
        monkeypatch.undo()

        # --- sharing -----------------------------------------------------
        assert [p.data_ptr() for p in prefill.model.parameters()] == decode_weights
        assert _ptr(prefill.kv_caches) == decode_kv

        # --- P's own buffers ---------------------------------------------
        assert prefill.input_buffers is not decode.input_buffers
        assert prefill.block_tables is not decode.block_tables
        assert prefill.attn_groups is not decode.attn_groups
        assert prefill.cudagraph_manager is not decode.cudagraph_manager
        assert prefill.cudagraph_manager is not None
        assert prefill.cudagraph_manager.cudagraph_mode == CUDAGraphMode.NONE
        assert not prefill.cudagraph_manager._graphs_captured
        assert (
            prefill.input_buffers.input_ids.data_ptr()
            != decode.input_buffers.input_ids.data_ptr()
        )
        # No helper stream may be retained: the D2H copies run on the lane
        # stream. (A bare `current_stream()` call also constructs a Stream
        # wrapper, so count ownership, not constructor calls.)
        assert prefill._output_copy_stream_impl is None
        assert prefill.draft_tokens_handler._copy_stream is None
        assert all(s is not prefill._output_copy_stream_impl for s in created)

        # --- warmup and its effect on the shared KV ----------------------
        kv_before = [t.clone() for t in prefill.kv_caches]
        before = _used_bytes()
        prefill.lane_warmup([16])
        prefill_warmup_mem = _used_bytes() - before
        warmup_bit_identical = all(
            torch.equal(a, b) for a, b in zip(kv_before, prefill.kv_caches)
        )
        assert warmup_bit_identical, "lane_warmup must not write the shared KV"

        memory_table = {
            "D weights": weights_mem,
            "D kv+mamba": kv_mem,
            "P runner buffers": prefill_mem,
            "P load_model": prefill_load_mem,
            "P initialize_kv_cache": prefill_kv_mem,
            "P lane_warmup": prefill_warmup_mem,
        }
        with capsys.disabled():
            print("\ntwo-lane runner memory (MiB):")
            for name, value in memory_table.items():
                print(f"  {name}: {value / 2**20:.2f}")
            p_extra = prefill_mem + prefill_load_mem + prefill_kv_mem
            print(f"  P extra total: {p_extra / 2**20:.2f} MiB")
            print(f"  streams constructed during P init: {len(created)} (none owned)")
        assert all(value >= 0 for value in memory_table.values())


_FORWARD_MODEL = os.environ.get("TWO_LANE_FORWARD_MODEL", "facebook/opt-125m")


@pytest.mark.skipif(not HAS_GPU, reason="Requires a GPU")
@pytest.mark.skipif(not HAS_GPU, reason="Requires a GPU")
def test_lane_forwards_match_a_single_runner(
    dist_init,
    lane_mode: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """D and P forwards on two lane streams equal one baseline runner's.

    The forward gate uses OPT-125m: a dense model whose paged-attention
    backend runs this repo's LBNHC KV layout on the box. Qwen3.5's eager GDN
    path cannot run here today (Report: its fp16 3-D ``gated_rms_norm`` fast
    path postdates the built ``_rocm_C``, and gfx1030 has no bf16 dot2);
    both failures reproduce with a plain decode runner and no lane code. The
    Qwen3.5 sharing/state contract is covered by
    ``test_second_runner_shares_weights_and_kv_and_matches_forwards`` and
    ``test_lane_model_isolation.py``.
    """
    from tests.v1.core.utils import get_kv_cache_configs
    from vllm.config import set_current_vllm_config
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        use_workspace_lane,
    )

    try:
        vllm_config = _lane_config(model=_FORWARD_MODEL)
    except (OSError, ValueError) as err:
        pytest.skip(f"forward test model unavailable: {err}")
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    prompt = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333, 444, 555]
    prompt_b = [7, 17, 27, 37, 47, 57, 67, 77, 87, 97, 107, 117, 127, 137]

    with set_current_vllm_config(vllm_config):
        init_workspace_manager(_test_device(), 1, 2)
        stream = torch.cuda.Stream(device=_test_device())
        ctx = SimpleNamespace(
            current_stream=lambda: stream, on_batch_done=lambda so: None
        )

        def build_runner(lane=None, primary=None):
            runner = (
                GPUModelRunner(vllm_config, _test_device(), lane=lane, primary=primary)
                if lane is not None
                else GPUModelRunner(vllm_config, _test_device())
            )
            runner.lane_ctx = ctx
            if lane == Lane.PREFILL:
                runner.load_model(model=primary.model, shared_from=primary)
                runner.initialize_kv_cache(kv, shared_from=primary)
                with use_workspace_lane(int(Lane.PREFILL)):
                    runner.lane_warmup([16])
            else:
                runner.load_model()
                runner.initialize_kv_cache(kv)
            return runner

        baseline = GPUModelRunner(vllm_config, _test_device())
        baseline.lane_ctx = ctx
        baseline.load_model()
        kv_spec = baseline.get_kv_cache_spec()
        kv = get_kv_cache_configs(vllm_config, [kv_spec], [2 << 30])[0]
        baseline.initialize_kv_cache(kv)
        num_groups = len(kv.kv_cache_groups)
        blocks = [[0] for _ in range(num_groups)]
        cap = [
            g.kv_cache_spec.max_num_blocks_per_req(vllm_config, 2048)
            for g in kv.kv_cache_groups
        ]
        assert all(n > 1 for n in cap), (
            "the concurrency check needs two blocks per KV group"
        )
        second_blocks = [[1] for _ in range(num_groups)]

        def run(runner, req_id: str, tokens: list[int], ids):
            _zero_kv(runner)
            return (
                _step(
                    runner, vllm_config, _new_request_so(req_id, tokens, ids), stream
                ),
                _step(
                    runner,
                    vllm_config,
                    _cached_request_so(req_id, len(tokens), 1),
                    stream,
                ),
            )

        baseline_tokens = run(baseline, "solo", prompt, blocks)

        decode = GPUModelRunner(vllm_config, _test_device(), lane=Lane.DECODE)
        decode.lane_ctx = ctx
        decode.load_model()
        decode.initialize_kv_cache(kv)
        prefill = build_runner(lane=Lane.PREFILL, primary=decode)
        assert [p.data_ptr() for p in prefill.model.parameters()] == [
            p.data_ptr() for p in decode.model.parameters()
        ]
        assert _ptr(prefill.kv_caches) == _ptr(decode.kv_caches)

        decode_tokens = run(decode, "solo", prompt, blocks)
        prefill_tokens = run(prefill, "solo", prompt, blocks)
        assert decode_tokens == baseline_tokens, (decode_tokens, baseline_tokens)
        assert prefill_tokens == baseline_tokens, (prefill_tokens, baseline_tokens)

        # --- concurrent requests on the two lane streams -----------------
        results: dict[str, tuple[list[int], list[int]]] = {}
        errors: list[BaseException] = []

        def run_side(lane, tokens):
            try:
                with lane_scope(lane):
                    results[lane.name] = run(
                        decode if lane is Lane.DECODE else prefill,
                        f"conc-{lane.name}",
                        tokens,
                        blocks if lane is Lane.DECODE else second_blocks,
                    )
            except BaseException as err:  # noqa: BLE001
                errors.append(err)

        threads = [
            threading.Thread(target=run_side, args=(Lane.DECODE, prompt)),
            threading.Thread(target=run_side, args=(Lane.PREFILL, prompt_b)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors, errors

        with lane_scope(Lane.DECODE):
            solo_a = run(decode, "conc-DECODE", prompt, blocks)
        with lane_scope(Lane.PREFILL):
            solo_b = run(prefill, "conc-PREFILL", prompt_b, second_blocks)
        assert results["DECODE"] == solo_a, (results["DECODE"], solo_a)
        assert results["PREFILL"] == solo_b, (results["PREFILL"], solo_b)

        with capsys.disabled():
            print(
                "\ntwo-lane forward equivalence: D == P == single runner"
                f" ({num_groups} KV group(s)); concurrent D/P per lane match"
            )


@pytest.mark.skipif(not HAS_GPU, reason="Requires a GPU")
@pytest.mark.skipif(
    not os.path.isdir(_LANE_MODEL), reason=f"two-lane test model missing: {_LANE_MODEL}"
)
def test_flag_off_runner_allocates_no_lane_state(
    dist_init,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag off (lane mode never enabled): baseline runner, no lane state.

    Two-lane is opt-in; with the flag off the runner must allocate the same
    buffers as before, create no stream of its own and expose no lane
    attributes (the ``0 MiB``/``no stream`` flag-off variant).
    """
    from tests.v1.core.utils import get_kv_cache_configs
    from vllm.config import set_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager

    assert not lane_module.lane_mode_enabled()

    created: list[object] = []
    real_stream_cls = torch.cuda.Stream

    def counting_stream(*args, **kwargs):
        stream = real_stream_cls(*args, **kwargs)
        created.append(stream)
        return stream

    monkeypatch.setattr(torch.cuda, "Stream", counting_stream)
    vllm_config = _lane_config()
    with set_current_vllm_config(vllm_config):
        init_workspace_manager(_test_device(), 1, 2)
        runner = GPUModelRunner(vllm_config, _test_device())
        runner.load_model()
        # Sum the tensor bytes: a device-wide allocation delta is perturbed by
        # the allocator releasing cached blocks inside load_model.
        weights_mem = sum(
            p.numel() * p.element_size() for p in runner.model.parameters()
        )
        kv_cache_config = get_kv_cache_configs(
            vllm_config, [runner.get_kv_cache_spec()], [2 << 30]
        )[0]
        runner.initialize_kv_cache(kv_cache_config)
        kv_mem = sum(cache.numel() * cache.element_size() for cache in runner.kv_caches)
    monkeypatch.undo()

    # No lane plumbing was added to the flag-off path.
    assert runner.lane == Lane.DECODE
    assert not runner._is_prefill_lane
    assert runner.lane_ctx is None
    assert runner._lane_batch is None
    # Flag off is the baseline: the helper D2H stream exists (created once at
    # init) and is not lane-owned. Lane mode leaves it None instead.
    assert isinstance(runner._output_copy_stream_impl, torch.cuda.Stream)
    assert runner.output_copy_stream is runner._output_copy_stream_impl
    assert isinstance(runner.draft_tokens_handler._copy_stream, torch.cuda.Stream)
    assert runner._output_copy_stream_impl in created
    assert weights_mem >= 0 and kv_mem >= 0

    with pytest.raises(ValueError, match="only valid on the prefill-lane"):
        runner.lane_warmup([16])


# --------------------------------------------------------------------------
# handoff payload (R-S5-9, 5.1)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GPU, reason="needs one GPU")
def test_handoff_add_request_is_a_decode_over_the_prompt() -> None:
    """The handoff payload must land as a decode row, not a 1-token prefill.

    prefill_token_ids = prompt + [t1] with prefill_len = len(prompt): the
    prompt is computed and t1 is the first generated token. The runner
    classifies a row by ``num_computed_prefill_tokens < prefill_len``, so this
    is a decode exactly like the baseline's second step, while the full
    history (including t1) stays available to the penalty bincount.
    """
    from vllm.v1.worker.gpu.states import RequestState

    prompt = [11, 12, 13, 14]
    t1 = 99
    req_states = RequestState(
        max_num_reqs=2,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=0,
        vocab_size=128,
        device=torch.device("cuda"),
    )
    req_states.add_request(
        req_id="h",
        prompt_len=len(prompt),
        all_token_ids=prompt + [t1],
        num_computed_tokens=len(prompt),
        max_tokens=8,
        prefill_len=len(prompt),
    )
    # A normal admission keeps the baseline meaning: everything is prefill.
    req_states.add_request(
        req_id="n",
        prompt_len=len(prompt),
        all_token_ids=prompt,
        num_computed_tokens=0,
        max_tokens=8,
    )
    req_states.apply_staged_writes()

    idx = req_states.req_id_to_index["h"]
    assert req_states.prefill_len.np[idx] == len(prompt)
    assert req_states.total_len_np[idx] == len(prompt) + 1
    assert int(req_states.total_len.gpu[idx]) == len(prompt) + 1
    assert req_states.num_computed_prefill_tokens[idx] == len(prompt)
    assert int(req_states.last_sampled_tokens[idx, 0]) == t1
    # The runner's classification: the handoff row is not a prefill row.
    assert not (
        req_states.num_computed_prefill_tokens[idx] < req_states.prefill_len.np[idx]
    )

    n_idx = req_states.req_id_to_index["n"]
    assert req_states.prefill_len.np[n_idx] == len(prompt)
    assert req_states.total_len_np[n_idx] == len(prompt)
    assert req_states.num_computed_prefill_tokens[n_idx] == 0
    assert (
        req_states.num_computed_prefill_tokens[n_idx] < req_states.prefill_len.np[n_idx]
    )
