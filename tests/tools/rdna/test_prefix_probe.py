# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validity gates of tools/rdna/port_v031/prefix_probe.py.

A fake HTTP layer supplies /tokenize and /metrics and TTFT is injected, so the
cases are deterministic and need no server, GPU or sleeping.
"""

import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Literal

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "prefix_probe", ROOT / "tools/rdna/port_v031/prefix_probe.py"
)
assert _spec is not None and _spec.loader is not None
pp = importlib.util.module_from_spec(_spec)
sys.modules["prefix_probe"] = pp
_spec.loader.exec_module(pp)

COMPLETIONS = "http://127.0.0.1:18120/v1/completions"
METRICS = "http://127.0.0.1:18120/metrics"
TOKENIZE = "http://127.0.0.1:18120/tokenize"


class _Reply:
    def __init__(self, body: str | bytes):
        self._body = body.encode() if isinstance(body, str) else body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_Reply":
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        return False


def _serve(monkeypatch: pytest.MonkeyPatch, routes: dict[str, list[object]]):
    """Answer each URL from a queue of bodies/exceptions; record every call."""
    pending = {url: list(items) for url, items in routes.items()}
    calls: list[str] = []

    def urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        calls.append(url)
        assert pending.get(url), f"unexpected or exhausted request: {url}"
        item = pending[url].pop(0)
        if isinstance(item, Exception):
            raise item
        return _Reply(item)

    monkeypatch.setattr(pp.urllib.request, "urlopen", urlopen)
    return calls


def _metrics(queries: float, hits: float) -> str:
    return (
        'vllm:prefix_cache_queries_total{engine="0",model_name="m"} '
        f"{queries}\n"
        'vllm:prefix_cache_hits_total{engine="0",model_name="m"} '
        f"{hits}\n"
    )


def _fake_ttft(values: list[float]):
    pending = iter(values)

    def fake(url, model, prompt, max_tokens):
        return next(pending), "text"

    return fake


def _no_ttft():
    def fake(url, model, prompt, max_tokens):
        pytest.fail("no completion request should be sent")

    return fake


def _argv(*extra: str) -> list[str]:
    return ["--url", COMPLETIONS, "--model", "m", *extra]


# ------------------------------------------------------------------ metrics


def test_counters_are_read_as_the_server_reports_them(monkeypatch):
    calls = _serve(monkeypatch, {METRICS: [_metrics(3854, 1024)]})
    assert pp.prefix_counters(COMPLETIONS) == (3854.0, 1024.0)
    assert calls == [METRICS]


def test_counters_sum_every_engine_label_set(monkeypatch):
    body = (
        'vllm:prefix_cache_queries_total{engine="0"} 100\n'
        'vllm:prefix_cache_queries_total{engine="1"} 50\n'
        'vllm:prefix_cache_hits_total{engine="0"} 40\n'
        'vllm:prefix_cache_hits_total{engine="1"} 10\n'
    )
    _serve(monkeypatch, {METRICS: [body]})
    assert pp.prefix_counters(COMPLETIONS) == (150.0, 50.0)


def test_unavailable_metrics_fail_instead_of_returning_zero(monkeypatch):
    _serve(monkeypatch, {METRICS: [OSError("connection refused")]})
    with pytest.raises(pp.ProbeError):
        pp.prefix_counters(COMPLETIONS)


def test_missing_hit_counter_fails_instead_of_returning_zero(monkeypatch):
    _serve(monkeypatch, {METRICS: ["vllm:prefix_cache_queries_total 10\n"]})
    with pytest.raises(pp.ProbeError):
        pp.prefix_counters(COMPLETIONS)


def test_a_malformed_counter_value_fails_closed(monkeypatch):
    _serve(monkeypatch, {METRICS: ["vllm:prefix_cache_queries_total 1e\n"]})
    with pytest.raises(pp.ProbeError):
        pp.prefix_counters(COMPLETIONS)


def test_an_unreadable_metrics_body_fails_closed(monkeypatch):
    _serve(monkeypatch, {METRICS: [b"\xff\xfe"]})
    with pytest.raises(pp.ProbeError):
        pp.prefix_counters(COMPLETIONS)


# ----------------------------------------------------------------- tokenizer


def test_tokenize_count_uses_the_server_value(monkeypatch):
    payload = json.dumps({"count": 1927, "max_model_len": 4096, "tokens": [1, 2]})
    calls = _serve(monkeypatch, {TOKENIZE: [payload]})
    assert pp.tokenize_count(COMPLETIONS, "m", "prompt") == 1927
    assert calls == [TOKENIZE]


@pytest.mark.parametrize(
    "payload",
    [
        '{"tokens": []}',  # no count
        '{"count": "lots"}',  # wrong type
        '{"count": 0}',  # no tokens
        '{"count": true}',  # bool is not a count
        "not json",
    ],
)
def test_a_malformed_tokenize_response_fails_closed(monkeypatch, payload):
    _serve(monkeypatch, {TOKENIZE: [payload]})
    with pytest.raises(pp.ProbeError):
        pp.tokenize_count(COMPLETIONS, "m", "prompt")


def test_an_unreachable_tokenize_endpoint_fails_closed(monkeypatch):
    _serve(monkeypatch, {TOKENIZE: [OSError("connection refused")]})
    with pytest.raises(pp.ProbeError):
        pp.tokenize_count(COMPLETIONS, "m", "prompt")


# -------------------------------------------------------------------- prompt


def test_max_model_len_is_enforced_with_the_tokenized_length():
    # Ten words would pass any word-to-token estimate; the real 4100 tokens
    # plus 32 generated tokens do not fit 4096.
    with pytest.raises(pp.ProbeError):
        pp.check_prompt(4100, 32, 4096)
    pp.check_prompt(4064, 32, 4096)


# ------------------------------------------------------------------- verdict


def test_recorded_arm_passes_on_its_verified_hit_rate():
    verdict = pp.evaluate(0.74, 0.40, 3854, 1024, None)
    assert verdict.passed
    assert verdict.latency_met is None
    assert verdict.hit_tokens == 1024
    assert verdict.warm_queries == pytest.approx(1927)
    assert verdict.hit_rate == pytest.approx(1024 / 1927)


def test_recorded_arm_still_fails_the_explicit_half_cold_target():
    verdict = pp.evaluate(0.74, 0.40, 3854, 1024, 0.5)
    assert verdict.latency_met is False
    assert not verdict.passed


def test_the_default_reports_ttft_without_gating_on_it():
    # A slower warm response still passes when the cache hits are verified.
    verdict = pp.evaluate(0.40, 0.74, 3854, 1024, None)
    assert verdict.latency_met is None
    assert verdict.passed


def test_an_explicit_met_target_is_still_required():
    verdict = pp.evaluate(0.74, 0.30, 3854, 1024, 0.5)
    assert verdict.latency_met is True
    assert verdict.passed


def test_a_fast_warm_request_without_hits_fails():
    verdict = pp.evaluate(60.0, 0.5, 3854, 0, None)
    assert verdict.hit_rate == 0.0
    assert not verdict.passed


def test_exactly_half_the_warm_prompt_is_not_a_majority():
    verdict = pp.evaluate(60.0, 0.5, 4000, 1000, None)
    assert verdict.hit_rate == pytest.approx(0.5)
    assert not verdict.passed


def test_an_explicit_latency_target_cannot_rescue_missing_hits():
    verdict = pp.evaluate(60.0, 0.5, 4000, 100, 0.5)
    assert verdict.latency_met is True
    assert not verdict.passed


@pytest.mark.parametrize(
    ("queries", "hits"),
    [(0, 0), (100, -1), (100, 101), (2000, 1001)],
)
def test_unusable_counter_deltas_are_an_error(queries, hits):
    with pytest.raises(pp.ProbeError):
        pp.evaluate(1.0, 1.0, queries, hits, None)


# ----------------------------------------------------------------------- CLI


def _main_routes(monkeypatch, token_count: int, counters: list[tuple[float, float]]):
    routes: dict[str, list[object]] = {
        TOKENIZE: [json.dumps({"count": token_count, "max_model_len": 4096})],
        METRICS: [_metrics(queries, hits) for queries, hits in counters],
    }
    return _serve(monkeypatch, routes)


def test_cli_default_passes_the_recorded_arm_while_ratio_half_fails(monkeypatch):
    for extra, expected in (([], 0), (["--ratio", "0.5"], 1)):
        monkeypatch.setattr(pp, "ttft", _fake_ttft([0.74, 0.40]))
        calls = _main_routes(monkeypatch, 1927, [(0.0, 0.0), (3854.0, 1024.0)])
        assert pp.main(_argv(*extra)) == expected
        assert calls.count(METRICS) == 2


def test_cli_fails_when_metrics_are_unavailable(monkeypatch):
    monkeypatch.setattr(pp, "ttft", _no_ttft())
    _serve(
        monkeypatch,
        {
            TOKENIZE: [json.dumps({"count": 1927})],
            METRICS: [OSError("connection refused")],
        },
    )
    assert pp.main(_argv()) == 1


def test_cli_fails_when_tokenize_is_unavailable(monkeypatch):
    monkeypatch.setattr(pp, "ttft", _no_ttft())
    _serve(monkeypatch, {TOKENIZE: [OSError("connection refused")]})
    assert pp.main(_argv()) == 1


def test_cli_uses_the_tokenized_length_not_the_word_count(monkeypatch):
    monkeypatch.setattr(pp, "ttft", _no_ttft())
    _serve(
        monkeypatch,
        {
            TOKENIZE: [json.dumps({"count": 4100})],
            METRICS: [_metrics(0, 0)],
        },
    )
    assert pp.main(_argv("--words", "10")) == 1


def test_small_context_uses_measured_hits_not_assumed_block_geometry(monkeypatch):
    _main_routes(monkeypatch, 64, [(0, 0), (128, 48)])
    monkeypatch.setattr(pp, "ttft", _fake_ttft([0.10, 0.08]))
    assert pp.main(_argv("--words", "1", "--max-model-len", "256")) == 0


def test_empty_completion_stream_has_no_measurable_ttft(monkeypatch):
    payload = b'data: {"choices":[{"text":""}]}\n\ndata: [DONE]\n\n'
    monkeypatch.setattr(
        pp.urllib.request,
        "urlopen",
        lambda *args, **kwargs: BytesIO(payload),
    )
    with pytest.raises(pp.ProbeError):
        pp.ttft(COMPLETIONS, "m", "prompt", 32)
