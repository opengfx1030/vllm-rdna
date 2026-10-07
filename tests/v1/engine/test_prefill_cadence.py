# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.engine.core import DPEngineCoreProc, EngineCore


def test_prefill_defer_step_interval_one_never_defers():
    assert not any(EngineCore._prefill_defer_step(step, 1) for step in range(1, 9))


@pytest.mark.parametrize("interval", [2, 3, 4])
def test_prefill_defer_step_releases_on_interval_multiples(interval: int):
    steps = range(1, 3 * interval + 1)
    deferred = [EngineCore._prefill_defer_step(step, interval) for step in steps]
    assert deferred == [step % interval != 0 for step in steps]
    assert deferred.count(False) == 3


def test_base_engine_core_defers_from_config_and_step_counter():
    engine_core = EngineCore.__new__(EngineCore)
    engine_core.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(prefill_schedule_interval=4)
    )
    engine_core._prefill_step_counter = 0
    sequence = []
    for _ in range(8):
        engine_core._prefill_step_counter += 1
        sequence.append(engine_core._should_throttle_prefills())
    assert sequence == [True, True, True, False, True, True, True, False]


def test_dp_engine_core_defers_from_its_own_step_counter():
    dp_engine_core = DPEngineCoreProc.__new__(DPEngineCoreProc)
    dp_engine_core.prefill_schedule_interval = 4
    dp_engine_core.step_counter = 0
    sequence = []
    for _ in range(8):
        dp_engine_core.step_counter += 1
        sequence.append(dp_engine_core._should_throttle_prefills())
    assert sequence == [True, True, True, False, True, True, True, False]
