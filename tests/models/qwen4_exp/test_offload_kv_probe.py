# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate cold-prefill rounding from errors introduced by CPU KV reload."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_PATH = Path(__file__).resolve().parents[3] / "tools/rdna/mono/probe_offload_kv.py"
_SPEC = importlib.util.spec_from_file_location("offload_kv_probe", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)


def _completion(logprob, token="token_id:17"):
    return {
        "text": "The fence prevents slot reuse.",
        "tokens": [token],
        "token_logprobs": [logprob],
        "finish_reason": "stop",
    }


def test_prefill_shape_delta_is_diagnostic_but_reload_uses_cached_reference():
    cold = _completion(-0.1)
    gpu_cached = _completion(-0.185)
    reloaded = _completion(-0.186)
    baseline = _PROBE.compare(cold, gpu_cached, 0.05, enforce_logprobs=False)
    assert baseline["within_logprob_tolerance"] is False
    assert baseline["max_abs_token_logprob_delta"] == pytest.approx(0.085)
    parity = _PROBE.compare(gpu_cached, reloaded, 0.05)
    assert parity["within_logprob_tolerance"] is True
    assert parity["logprob_tolerance_enforced"] is True


def test_reload_drift_still_fails_original_tolerance():
    with pytest.raises(RuntimeError, match="exceeds 0.05"):
        _PROBE.compare(_completion(-0.185), _completion(-0.25), 0.05)


def test_baseline_token_divergence_is_never_just_diagnostic():
    with pytest.raises(RuntimeError, match="text/token IDs differ"):
        _PROBE.compare(
            _completion(-0.1),
            _completion(-0.185, token="token_id:18"),
            0.05,
            enforce_logprobs=False,
        )


def _mock_run(monkeypatch, values, gpu_cached_tokens=None, loaded_cached_tokens=None):
    state: dict[str, Any] = {"count": 0, "reset": False}
    values = iter(values)

    def complete(*args, **kwargs):
        state["count"] += 1
        result: dict[str, Any] = _completion(next(values))
        result["usage"] = {"prompt_tokens": 4096, "completion_tokens": 1}
        if state["count"] in (2, 3) and gpu_cached_tokens is not None:
            result["usage"]["prompt_tokens_details"] = {
                "cached_tokens": gpu_cached_tokens
            }
        if state["count"] == 5:
            state["reload_limit"] = kwargs.get("max_load_tokens")
            if loaded_cached_tokens is not None:
                result["usage"]["prompt_tokens_details"] = {
                    "cached_tokens": loaded_cached_tokens
                }
        return result

    def metrics(*args):
        return {
            "store_bytes": 8192 if state["count"] else 0,
            "load_bytes": 8192 if state["reset"] and state["count"] == 5 else 0,
            "prefix_cache_hits": min(max(state["count"] - 1, 0), 2) * 2048,
        }

    def reset(*args):
        assert "reset_external=false" in args[1]
        state["reset"] = True
        return {"success": True}

    monkeypatch.setattr(_PROBE, "complete", complete)
    monkeypatch.setattr(_PROBE, "metrics", metrics)
    monkeypatch.setattr(_PROBE, "request", reset)
    args = SimpleNamespace(
        base_url="http://unused",
        model="test",
        prompt_tokens=4096,
        max_tokens=32,
        request_timeout=1,
        wait_timeout=1,
        logprob_atol=0.05,
        match_reload_boundary=False,
    )
    return args, {"passed": False}, state


def test_gpu_repeat_cannot_replace_original_reference_to_hide_reload_drift(monkeypatch):
    args, report, _ = _mock_run(monkeypatch, [-0.1, -0.185, -0.25, -0.1, -0.25])
    with pytest.raises(RuntimeError, match="exceeds 0.05"):
        _PROBE.run(args, report)
    assert report["passed"] is False
    assert report["gpu_cached_repeat_parity"]["within_logprob_tolerance"] is False
    assert report["gpu_cached_repeat_hit_evidence"]["prefix_cache_hits_delta"] == 2048
    assert report["gpu_cached"]["token_logprobs"] == [-0.185]
    assert report["reloaded"]["token_logprobs"] == [-0.25]
    assert report["gpu_repeat_vs_reloaded_parity"]["within_logprob_tolerance"] is True


@pytest.mark.parametrize("loaded_cached_tokens", [2048, 1024, None])
def test_controlled_reload_must_match_exact_original_gpu_boundary(
    monkeypatch, loaded_cached_tokens
):
    args, report, state = _mock_run(
        monkeypatch,
        [-0.1, -0.185, -0.185, -0.1, -0.186],
        gpu_cached_tokens=2048,
        loaded_cached_tokens=loaded_cached_tokens,
    )
    args.match_reload_boundary = True
    if loaded_cached_tokens == 2048:
        _PROBE.run(args, report)
        assert report["passed"] is True
        assert report["parity_reference"] == "gpu_cached"
        assert report["parity"]["max_abs_token_logprob_delta"] == pytest.approx(0.001)
    else:
        with pytest.raises(RuntimeError, match="expected exactly 2048"):
            _PROBE.run(args, report)
        assert report["passed"] is False
    assert state["reload_limit"] == 2048


@pytest.mark.parametrize("gpu_cached_tokens", [None, 0])
def test_controlled_reload_refuses_missing_gpu_boundary_before_reset(
    monkeypatch, gpu_cached_tokens
):
    args, report, state = _mock_run(
        monkeypatch,
        [-0.1, -0.185, -0.185],
        gpu_cached_tokens=gpu_cached_tokens,
    )
    args.match_reload_boundary = True
    with pytest.raises(RuntimeError, match="requires positive GPU usage cached_tokens"):
        _PROBE.run(args, report)
    assert state["reset"] is False


@pytest.mark.parametrize("loaded_cached_tokens", [2048, 1024, 3072])
def test_mamba_cap_slack_changes_request_cap_but_not_accepted_restore_boundary(
    monkeypatch, loaded_cached_tokens
):
    args, report, state = _mock_run(
        monkeypatch,
        [-0.1, -0.185, -0.185, -0.1, -0.186],
        gpu_cached_tokens=2048,
        loaded_cached_tokens=loaded_cached_tokens,
    )
    args.match_reload_boundary = True
    args.reload_cap_slack = 1
    if loaded_cached_tokens == 2048:
        _PROBE.run(args, report)
        assert report["passed"] is True
        assert report["logprob_atol"] == 0.05
    else:
        with pytest.raises(RuntimeError, match="expected exactly 2048"):
            _PROBE.run(args, report)
        assert report["passed"] is False
    assert state["reload_limit"] == 2049
    assert report["reload_request_max_load_tokens"] == 2049
    assert report["matched_reload_boundary_tokens"] == 2048


def test_completion_only_sends_boundary_cap_when_explicitly_requested(monkeypatch):
    bodies = []

    def request(base, path, timeout, body):
        bodies.append(body)
        result = _completion(-0.1)
        result["logprobs"] = {
            "tokens": result["tokens"],
            "token_logprobs": result["token_logprobs"],
        }
        return {"choices": [result], "usage": {"completion_tokens": 1}}

    monkeypatch.setattr(_PROBE, "request", request)
    args = SimpleNamespace(base_url="http://unused", model="test", request_timeout=1)
    _PROBE.complete(args, "prompt", "salt", 32)
    _PROBE.complete(args, "prompt", "salt", 32, max_load_tokens=2048)
    assert "kv_transfer_params" not in bodies[0]
    assert bodies[1]["kv_transfer_params"] == {"max_load_tokens": 2048}


def test_unstable_gpu_reference_cannot_qualify_even_when_cpu_matches(monkeypatch):
    args, report, _ = _mock_run(monkeypatch, [-0.1, -0.185, -0.26, -0.1, -0.186])
    with pytest.raises(RuntimeError, match="unstable GPU-resident reference"):
        _PROBE.run(args, report)
    assert report["passed"] is False
    assert report["parity"]["within_logprob_tolerance"] is True
    assert report["parity"]["logprob_tolerance_enforced"] is True
    assert report["load_bytes_delta"] == 8192
    assert len(report["validation_failures"]) == 1


def test_stable_gpu_reference_reports_reload_failure_separately(monkeypatch):
    args, report, _ = _mock_run(monkeypatch, [-0.1, -0.185, -0.186, -0.1, -0.26])
    with pytest.raises(RuntimeError, match="CPU reload versus original GPU reference"):
        _PROBE.run(args, report)
    assert report["gpu_cached_repeat_parity"]["within_logprob_tolerance"] is True
    assert report["parity"]["within_logprob_tolerance"] is False
    assert len(report["validation_failures"]) == 1
