# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Failure gates and metric semantics of the standalone long-workload client."""

import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

_PATH = Path(__file__).resolve().parents[3] / "tools/rdna/mono/qualify_offload_long.py"
_SPEC = importlib.util.spec_from_file_location("offload_long_probe", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)


def test_copy_benchmark_resets_external_prefix_and_retries_unreleased_blocks(
    monkeypatch,
):
    copy_probe = _PROBE.load_tool(
        "copy_reset_probe", "tools/rdna/mono/qualify_expert_copy.py"
    )
    requests = []
    responses = iter((b'{"success":false}', b'{"success":true}'))

    class Response(io.BytesIO):
        status = 200

    def urlopen(request, timeout):
        requests.append(request)
        return Response(next(responses))

    monkeypatch.setattr(copy_probe.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(copy_probe.time, "sleep", lambda _: None)
    copy_probe.reset_prefix(SimpleNamespace(base_url="http://unused"))
    assert len(requests) == 2
    assert all(r.method == "POST" for r in requests)
    assert all(
        parse_qs(urlsplit(r.full_url).query) == {"reset_external": ["true"]}
        for r in requests
    )


@pytest.mark.parametrize("hits", [0, 128, None])
@pytest.mark.parametrize(
    "name", ["vllm:prefix_cache_hits", "vllm:external_prefix_cache_hits"]
)
def test_copy_benchmark_cannot_accept_warm_or_unverified_prefixes(name, hits):
    copy_probe = _PROBE.load_tool(
        "copy_counter_probe", "tools/rdna/mono/qualify_expert_copy.py"
    )
    result = {
        "counter_deltas": {
            "vllm:prefix_cache_hits": 0,
            "vllm:external_prefix_cache_hits": 0,
        }
    }
    result["counter_deltas"][name] = hits
    if hits == 0:
        copy_probe.check_cold_prefix(result)
    else:
        with pytest.raises(RuntimeError, match="cold-prefix benchmark"):
            copy_probe.check_cold_prefix(result)


def test_metric_parser_sums_labeled_transfers_but_preserves_peak_occupancy():
    result = _PROBE.parse_metrics(
        "\n".join(
            [
                'vllm:kv_offload_load_bytes_total{rank="0"} 100',
                'vllm:kv_offload_load_bytes_total{rank="1"} 200',
                'vllm:gpu_cache_usage_perc{engine="0"} 0.7',
                'vllm:gpu_cache_usage_perc{engine="1"} 0.9',
                'vllm:num_preemptions_total{engine="0"} 2',
            ]
        )
    )
    assert result["vllm:kv_offload_load_bytes_total"] == 300
    assert result["vllm:gpu_cache_usage_perc"] == 0.9
    assert result["vllm:num_preemptions_total"] == 2


@pytest.mark.parametrize(
    "bad", ["failed", "incomplete", "missing", "nonstring", "short"]
)
def test_forced_length_performance_cannot_pass_missing_or_partial_requests(bad):
    result = {
        "completed": 1,
        "failed": 0,
        "generated_texts": ["The request completed."],
        "output_lens": [1024],
    }
    if bad == "failed":
        result["failed"] = 1
    elif bad == "incomplete":
        result["completed"] = 0
    elif bad == "missing":
        del result["generated_texts"]
    elif bad == "nonstring":
        result["generated_texts"] = [None]
    else:
        result["output_lens"] = [1023]
    with pytest.raises(RuntimeError):
        _PROBE.check_bench(result, 1, 1024, 16384)


def test_preemptions_are_reported_without_turning_completed_work_into_failure(tmp_path):
    samples = _PROBE.MetricSamples("http://unused", tmp_path / "metrics.jsonl")
    samples.rows = [
        {"metrics": _boundary_metrics(1)},
        {"metrics": _boundary_metrics(4)},
    ]
    assert samples.summary()["counter_deltas"]["vllm:num_preemptions"] == 3


def _actual_random_dataset():
    """Execute the pinned sampler source without importing GPU/MM dependencies."""
    import ast
    import logging
    import math
    import random
    import sys
    import types
    from abc import ABC, abstractmethod
    from dataclasses import dataclass

    import numpy as np

    root = _PATH.parents[3]
    module = types.ModuleType("_actual_offload_random_dataset")
    sys.modules[module.__name__] = module
    module.__dict__.update(
        ABC=ABC,
        abstractmethod=abstractmethod,
        dataclass=dataclass,
        np=np,
        math=math,
        random=random,
        logger=logging.getLogger(__name__),
    )
    selected: list[ast.stmt] = []
    for filename, names in [
        ("utils.py", {"_resolve_range_ratios", "get_sampling_params"}),
        (
            "datasets.py",
            {
                "SampleRequest",
                "BenchmarkDataset",
                "RandomDataset",
                "gen_prompt_decode_to_target_len",
            },
        ),
    ]:
        source = root / "vllm/benchmarks/datasets" / filename
        tree = ast.parse(source.read_text(), filename=str(source))
        selected.extend(
            node for node in tree.body if getattr(node, "name", None) in names
        )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    tree = ast.fix_missing_locations(
        ast.Module(body=[future, *selected], type_ignores=[])
    )
    exec(compile(tree, "<pinned random dataset>", "exec"), module.__dict__)
    return module.RandomDataset


@pytest.mark.parametrize("input_len,output_len,concurrency,requests", _PROBE.CELLS)
def test_fixed_long_cells_execute_pinned_sampler_with_exact_lengths(
    input_len, output_len, concurrency, requests
):
    class Tokenizer:
        vocab_size = 128
        all_special_ids = []

        def num_special_tokens_to_add(self):
            return 0

        def decode(self, ids):
            return " ".join(map(str, ids))

        def encode(self, text, add_special_tokens=False):
            return [int(token) for token in text.split()]

    dataset = _actual_random_dataset()(random_seed=17)
    samples = dataset.sample(
        tokenizer=Tokenizer(),
        num_requests=requests,
        input_len=input_len,
        output_len=output_len,
        range_ratio=_PROBE.FIXED_RANGE_RATIO,
    )
    assert len(samples) == requests
    assert all(sample.prompt_len == input_len for sample in samples)
    assert all(sample.expected_output_len == output_len for sample in samples)
    # Reproduce the failed launch through the real sampler guard.
    with pytest.raises(ValueError, match="minimum possible total input tokens"):
        dataset.sample(
            tokenizer=Tokenizer(),
            num_requests=1,
            input_len=input_len,
            output_len=output_len,
            range_ratio=1.0,
        )


def _boundary_metrics(preemptions=0):
    return {
        "vllm:num_preemptions_total": preemptions,
        "vllm:kv_offload_load_bytes_total": 0,
        "vllm:kv_offload_store_bytes_total": 0,
    }


@pytest.mark.parametrize(
    "missing", ["before", "after", "load_counter", "store_counter"]
)
def test_missing_real_boundary_or_transfer_metrics_fail(tmp_path, missing):
    samples = _PROBE.MetricSamples("http://unused", tmp_path / "metrics.jsonl")
    samples.rows = [{"metrics": _boundary_metrics()} for _ in range(4)]
    if missing in ("before", "after"):
        samples.rows[0 if missing == "before" else -1] = {"error": "timeout"}
    else:
        key = "vllm:kv_offload_" + missing.split("_")[0] + "_bytes_total"
        del samples.rows[-1]["metrics"][key]
    with pytest.raises(RuntimeError, match="boundary|before/after"):
        samples.summary()
    if missing in ("load_counter", "store_counter"):
        # Expert-copy qualification does not require KV restoration, and must
        # report an unexported transfer counter as unavailable rather than zero.
        result = samples.summary(require_boundary_counters=False)
        assert result["counter_deltas"][key.removesuffix("_total")] is None


@pytest.mark.parametrize("length", [16000, 17000, 32700])
def test_short_or_over_context_actual_prompt_fails(length):
    result = dict(
        completed=1,
        failed=0,
        generated_texts=["valid output"],
        output_lens=[1024],
        input_lens=[length],
    )
    with pytest.raises(RuntimeError, match="prompt lengths"):
        _PROBE.check_bench(result, 1, 1024, 16384)


def test_successful_client_without_new_result_cannot_reuse_stale_json(
    tmp_path, monkeypatch
):
    import json
    from contextlib import nullcontext
    from types import SimpleNamespace

    path = tmp_path / "bench-16384x1024-c1.json"
    path.write_text(
        json.dumps(
            dict(
                completed=1,
                failed=0,
                generated_texts=["old success"],
                input_lens=[16384],
                output_lens=[1024],
            )
        )
    )
    monkeypatch.setattr(_PROBE, "MetricSamples", lambda *args: nullcontext())

    def client(*args, **kwargs):
        assert not path.exists()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(_PROBE.subprocess, "run", client)
    args = SimpleNamespace(
        base_url="http://unused",
        model="m",
        tokenizer="t",
        out=tmp_path,
        cell_timeout=10,
    )
    with pytest.raises(FileNotFoundError):
        _PROBE.bench(args, (16384, 1024, 1, 1), 17)


@pytest.mark.parametrize("text", ["", "\n" * 344])
def test_forced_random_empty_outputs_are_quality_warnings(text):
    result = dict(
        completed=1,
        failed=0,
        generated_texts=[text],
        output_lens=[1024],
        input_lens=[16384],
    )
    warnings = _PROBE.check_bench(result, 1, 1024, 16384)
    assert warnings[0]["request"] == 0
    assert warnings[0]["garbage"] == "empty"


def test_selected_cells_report_partial_scope(tmp_path, monkeypatch):
    import json
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--tokenizer", "fake", "--out", str(tmp_path), "--cells", "32k-c1"],
    )
    calls = []

    def bench(args, cell, seed):
        calls.append(cell)
        return {}

    monkeypatch.setattr(_PROBE, "bench", bench)
    monkeypatch.setattr(_PROBE, "retrieval", lambda args: {"marker_found": True})
    assert _PROBE.main() == 0
    report = json.loads((tmp_path / "qualification.json").read_text())
    assert report["selected_cells"] == ["32k-c1"]
    assert report["partial_scope"] is True
    assert calls == [(32256, 256, 1, 1)]
