# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S0 tests for the two-lane interface freeze.

Covers the frozen config fields and their defaults, the ``VllmConfig``
refusal matrix (contract section S0 item 4 / D6), the RDNA2 gate, and the
thread-local helpers in ``vllm/v1/lane.py``. No GPU is required.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import vllm.config.scheduler as scheduler_mod
import vllm.envs as envs
import vllm.platforms.rdna as rdna_mod
from vllm.config import SchedulerConfig, VllmConfig
from vllm.v1.lane import (
    NUM_LANES,
    Lane,
    current_lane,
    enable_lane_mode,
    lane_mode_enabled,
    lane_scope,
)

LANE_SCHEDULER = "vllm.v1.core.sched.two_lane_scheduler.TwoLaneScheduler"


def _scheduler_config(**overrides) -> SimpleNamespace:
    base = dict(
        two_lane=True,
        async_scheduling=True,
        prefill_schedule_interval=1,
        decode_stall_budget_ms=None,
        decode_time_share=0.0,
        long_prefill_token_threshold=0,
        two_lane_decode_depth=2,
        two_lane_prefill_depth=1,
        two_lane_decode_wgps=20,
        two_lane_prefill_budget=None,
        two_lane_solo_full=True,
        two_lane_mem_reserve_mb=256,
        two_lane_trace=False,
        two_lane_strict=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _parallel_config(**overrides) -> SimpleNamespace:
    base = dict(
        distributed_executor_backend="mp",
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_dbo=False,
        use_ubatching=False,
        enable_eplb=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _spec_config(**overrides) -> SimpleNamespace:
    base = dict(
        method="mtp",
        draft_sample_method="greedy",
        enable_adaptive_verification=False,
        uses_dynamic_speculative_decoding=lambda: False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _fake_vllm_config(**overrides) -> SimpleNamespace:
    base = dict(
        scheduler_config=_scheduler_config(),
        use_v2_model_runner=True,
        parallel_config=_parallel_config(),
        speculative_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        kernel_config=SimpleNamespace(monokernel=False),
        additional_config=None,
        lora_config=None,
        is_mm_encoder_only=False,
        model_config=SimpleNamespace(runner_type="generate", enable_sleep_mode=False),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        weight_transfer_config=None,
        profiler_config=SimpleNamespace(profiler="none"),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeExecutor:
    @staticmethod
    def supports_two_lane() -> bool:
        return True


@pytest.fixture(autouse=True)
def _two_lane_env(monkeypatch):
    # Patch the platform probe, not two_lane_supported(), so the gating test
    # exercises the real two_lane_supported() -> on_rdna2() call.
    monkeypatch.setattr(rdna_mod, "on_rdna2", lambda: True)
    for name in (
        "VLLM_ROCM_MONO_DECODE",
        "VLLM_PLE_CPU_OFFLOAD",
        "VLLM_RDNA_DYNAMIC_PREFILL",
        "VLLM_RDNA_DECODE_STALL_MS",
        "VLLM_RDNA_DECODE_SHARE",
    ):
        monkeypatch.delenv(name, raising=False)
    import vllm.v1.lane as lane_mod

    monkeypatch.setattr(lane_mod, "_lane_mode", False, raising=False)


@pytest.fixture(autouse=True)
def _executor_support(monkeypatch):
    from vllm.v1.executor.abstract import Executor

    monkeypatch.setattr(
        Executor, "get_class", staticmethod(lambda _config: _FakeExecutor)
    )


def _run_verify(config: SimpleNamespace) -> None:
    VllmConfig._verify_two_lane(config)


def test_lane_enum_and_count():
    assert Lane.DECODE == 0
    assert Lane.PREFILL == 1
    assert NUM_LANES == 2


def test_lane_scope_restores_previous_lane():
    assert current_lane() == Lane.DECODE
    with lane_scope(Lane.PREFILL):
        assert current_lane() == Lane.PREFILL
    assert current_lane() == Lane.DECODE


def test_lane_scope_nested():
    with lane_scope(Lane.PREFILL):
        with lane_scope(Lane.DECODE):
            assert current_lane() == Lane.DECODE
        assert current_lane() == Lane.PREFILL
    assert current_lane() == Lane.DECODE


def test_lane_scope_is_thread_local():
    seen: list[Lane] = []

    def worker() -> None:
        seen.append(current_lane())
        with lane_scope(Lane.PREFILL):
            seen.append(current_lane())

    with lane_scope(Lane.PREFILL):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert seen == [Lane.DECODE, Lane.PREFILL]
    assert current_lane() == Lane.DECODE


def test_enable_lane_mode():
    assert lane_mode_enabled() is False
    enable_lane_mode()
    assert lane_mode_enabled() is True


def test_scheduler_config_defaults():
    config = SchedulerConfig.default_factory()
    assert config.two_lane is False
    assert config.two_lane_decode_wgps == 20
    assert config.two_lane_prefill_budget is None
    assert config.two_lane_decode_depth == 2
    assert config.two_lane_prefill_depth == 1
    assert config.two_lane_solo_full is True
    assert config.two_lane_mem_reserve_mb == 256
    assert config.two_lane_trace is False
    assert config.two_lane_strict is False
    # Flag off leaves the pre-existing knobs at their HEAD defaults.
    assert config.prefill_schedule_interval == 1
    assert config.decode_stall_budget_ms is None
    assert config.decode_time_share == 0.0
    assert config.long_prefill_token_threshold == 0


def test_max_concurrent_batches_lane_mode():
    fake = SimpleNamespace(
        scheduler_config=_scheduler_config(
            two_lane_decode_depth=3, two_lane_prefill_depth=2
        )
    )
    assert VllmConfig.max_concurrent_batches.__get__(fake) == 5


def test_max_concurrent_batches_flag_off_ignores_lane_depths():
    fake = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            two_lane=False,
            async_scheduling=False,
            two_lane_decode_depth=99,
            two_lane_prefill_depth=99,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    # Baseline path must not add the lane depths.
    assert VllmConfig.max_concurrent_batches.__get__(fake) == 1


def test_get_scheduler_cls_uses_two_lane_class(monkeypatch):
    monkeypatch.setattr(scheduler_mod, "resolve_obj_by_qualname", lambda name: name)
    config = SchedulerConfig.default_factory(two_lane=True)
    assert config.get_scheduler_cls() == LANE_SCHEDULER


def test_get_scheduler_cls_flag_off_uses_baseline(monkeypatch):
    monkeypatch.setattr(scheduler_mod, "resolve_obj_by_qualname", lambda name: name)
    config = SchedulerConfig.default_factory(two_lane=False)
    assert config.get_scheduler_cls() != LANE_SCHEDULER


def test_verify_two_lane_flag_off_returns_early():
    config = _fake_vllm_config(scheduler_config=_scheduler_config(two_lane=False))
    _run_verify(config)  # no exception


def test_verify_two_lane_baseline_ok():
    _run_verify(_fake_vllm_config())


def test_verify_two_lane_requires_async_scheduling():
    config = _fake_vllm_config(
        scheduler_config=_scheduler_config(async_scheduling=False)
    )
    with pytest.raises(ValueError, match="async scheduling"):
        _run_verify(config)


def test_verify_two_lane_requires_v2_runner():
    config = _fake_vllm_config(use_v2_model_runner=False)
    with pytest.raises(ValueError, match="Model Runner V2"):
        _run_verify(config)


def test_verify_two_lane_requires_executor_support(monkeypatch):
    from vllm.v1.executor.abstract import Executor

    monkeypatch.setattr(Executor, "get_class", staticmethod(lambda _c: object))
    config = _fake_vllm_config()
    with pytest.raises(ValueError, match="does not support"):
        _run_verify(config)


def test_verify_two_lane_requires_rdna2(monkeypatch):
    monkeypatch.setattr(rdna_mod, "two_lane_supported", lambda: False)
    config = _fake_vllm_config()
    with pytest.raises(ValueError, match="RDNA2"):
        _run_verify(config)


@pytest.mark.parametrize(
    "spec, match",
    [
        (_spec_config(method="eagle3"), "method 'mtp'"),
        (_spec_config(draft_sample_method="probabilistic"), "draft_sample_method"),
        (
            _spec_config(enable_adaptive_verification=True),
            "adaptive speculative verification",
        ),
        (
            _spec_config(uses_dynamic_speculative_decoding=lambda: True),
            "dynamic speculative decoding",
        ),
    ],
)
def test_verify_two_lane_spec_refusals(spec, match):
    config = _fake_vllm_config(speculative_config=spec)
    with pytest.raises(ValueError, match=match):
        _run_verify(config)


_REFUSALS = [
    (
        lambda c, _mp: setattr(c.parallel_config, "pipeline_parallel_size", 2),
        "pipeline parallelism",
    ),
    (
        lambda c, _mp: setattr(c.parallel_config, "data_parallel_size", 2),
        "data parallelism",
    ),
    (
        lambda c, _mp: setattr(c.parallel_config, "decode_context_parallel_size", 2),
        "context parallelism",
    ),
    (
        lambda c, _mp: setattr(c.parallel_config, "prefill_context_parallel_size", 2),
        "context parallelism",
    ),
    (
        lambda c, _mp: setattr(c.parallel_config, "enable_dbo", True),
        "dual batch overlap",
    ),
    (
        lambda c, _mp: setattr(c.parallel_config, "use_ubatching", True),
        "dual batch overlap",
    ),
    (
        lambda c, _mp: setattr(c.parallel_config, "enable_eplb", True),
        "expert parallelism load balancing",
    ),
    (
        lambda c, _mp: setattr(
            c, "kv_transfer_config", SimpleNamespace(kv_connector="x")
        ),
        "KV connectors",
    ),
    (
        lambda c, _mp: setattr(c.kernel_config, "monokernel", True),
        "monokernel",
    ),
    (
        lambda c, _mp: _mp.setenv("VLLM_ROCM_MONO_DECODE", "1"),
        "monokernel",
    ),
    (
        lambda c, _mp: setattr(
            c,
            "additional_config",
            {"rocm_mono_decode": {"expert_cache_slots": 1}},
        ),
        "expert-cache",
    ),
    (lambda c, _mp: setattr(c, "lora_config", object()), "LoRA"),
    (
        lambda c, _mp: setattr(c, "is_mm_encoder_only", True),
        "mm-encoder-only",
    ),
    (
        lambda c, _mp: setattr(c.model_config, "runner_type", "pooling"),
        "generate runners",
    ),
    (
        lambda c, _mp: setattr(c, "ec_transfer_config", object()),
        "encoder-cache transfer",
    ),
    (
        lambda c, _mp: setattr(c.model_config, "enable_sleep_mode", True),
        "sleep mode",
    ),
    (
        lambda c, _mp: setattr(c, "weight_transfer_config", object()),
        "weight transfer",
    ),
    (
        lambda c, _mp: setattr(c.scheduler_config, "prefill_schedule_interval", 2),
        "prefill-schedule-interval",
    ),
    (
        lambda c, _mp: setattr(c.scheduler_config, "decode_stall_budget_ms", 100),
        "decode-stall-budget-ms",
    ),
    (
        lambda c, _mp: setattr(c.scheduler_config, "decode_time_share", 0.5),
        "decode-time-share",
    ),
    (
        lambda c, _mp: setattr(c.scheduler_config, "long_prefill_token_threshold", 100),
        "long-prefill-token-threshold",
    ),
    (
        lambda c, _mp: _mp.setenv("VLLM_RDNA_DYNAMIC_PREFILL", "1"),
        "VLLM_RDNA_DYNAMIC_PREFILL",
    ),
    (
        lambda c, _mp: _mp.setenv("VLLM_RDNA_DECODE_STALL_MS", "100"),
        "VLLM_RDNA_DECODE_STALL_MS",
    ),
    (
        lambda c, _mp: _mp.setenv("VLLM_RDNA_DECODE_SHARE", "0.5"),
        "VLLM_RDNA_DECODE_SHARE",
    ),
    (
        lambda c, _mp: _mp.setenv("VLLM_RDNA_DECODE_STALL_MS", "abc"),
        "VLLM_RDNA_DECODE_STALL_MS",
    ),
    (
        lambda c, _mp: _mp.setenv("VLLM_RDNA_DECODE_SHARE", "abc"),
        "VLLM_RDNA_DECODE_SHARE",
    ),
    (
        lambda c, _mp: setattr(
            c.aux_output_config, "enable_return_routed_experts", True
        ),
        "enable-return-routed-experts",
    ),
]


@pytest.mark.parametrize("modifier, match", _REFUSALS)
def test_verify_two_lane_refusals(modifier, match, monkeypatch):
    config = _fake_vllm_config()
    modifier(config, monkeypatch)
    with pytest.raises(ValueError, match=match):
        _run_verify(config)


def test_verify_two_lane_allows_the_torch_profiler():
    """The profiler is process-wide; lane ranges separate the two lanes."""
    config = _fake_vllm_config(profiler_config=SimpleNamespace(profiler="torch"))
    _run_verify(config)


def test_verify_two_lane_allows_ple_cpu_offload(monkeypatch):
    """PLE offload is lane-safe: one connector per lane, sidecar keyed by lane."""
    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", "1")
    _run_verify(_fake_vllm_config())


def test_verify_two_lane_allows_zero_stall_budget_and_share():
    config = _fake_vllm_config(
        scheduler_config=_scheduler_config(
            decode_stall_budget_ms=0, decode_time_share=0.0
        )
    )
    _run_verify(config)


def test_two_lane_supported_follows_rdna2(monkeypatch):
    monkeypatch.setattr(rdna_mod, "on_rdna2", lambda: True)
    assert rdna_mod.two_lane_supported() is True
    monkeypatch.setattr(rdna_mod, "on_rdna2", lambda: False)
    assert rdna_mod.two_lane_supported() is False


def test_rdna_env_defaults_off():
    assert envs.VLLM_RDNA_TWO_LANE is False
    assert envs.VLLM_RDNA_LANE_DECODE_WGPS == 20
    assert envs.VLLM_RDNA_LANE_PREFILL_BUDGET is None
    assert envs.VLLM_RDNA_LANE_DECODE_DEPTH == 2
    assert envs.VLLM_RDNA_LANE_PREFILL_DEPTH == 1
    assert envs.VLLM_RDNA_LANE_SOLO_FULL is True
    assert envs.VLLM_RDNA_LANE_MEM_RESERVE_MB == 256
    assert envs.VLLM_RDNA_LANE_TRACE is False
    assert envs.VLLM_RDNA_LANE_STRICT is False
