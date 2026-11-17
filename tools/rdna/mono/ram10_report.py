#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report the TP4 RAM-backed expert-cache study from real run artifacts.

The study runner (``ram10_study.sh``) boots one server per arm and leaves, per
run, ``metadata.json``, ``serve.log`` and ``cells/<cell>/{bench,warmup}.json``
plus a ``cell.json`` health record.  This tool reads those files, reuses
``port_v031/bench_metrics.py`` for the headline rates, and writes:

  * ``ram10-report.json``  -- machine-readable cells and comparisons;
  * ``ram10-per-request.csv`` -- one row per request from ``--save-detailed``;
  * ``ram10-report.md`` -- baseline-relative Markdown tables.

Missing measurements stay ``None`` / ``n/a`` / blank; they are never zero.
``--selftest`` builds a deterministic fixture and checks the rate arithmetic.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import regex as re

ARM_ORDER = {"R": 0, "F": 1, "D": 2, "Z": 3}
ARM_NAMES = {"R": "resident", "F": "fullcache", "D": "demand", "Z": "zeros"}
ARM_RE = re.compile(r"^([RFDZ])(?:$|[-_.])")
CELL_RE = re.compile(r"bench-(\d+)x(\d+)-c(\d+)")
KV_GIB_RE = re.compile(r"Available KV cache memory:\s*([0-9]+(?:\.[0-9]+)?)\s*GiB")
KV_TOK_RE = re.compile(
    r"\b(?:GPU|CPU|NPU|HPU|XPU) KV cache size:\s*([0-9][0-9,]*)\s*tokens"
)
KV_CONC_RE = re.compile(
    r"Maximum concurrency for\s*(?:[0-9][0-9,]*)\s*tokens per request:\s*"
    r"([0-9]+(?:\.[0-9]+)?)x"
)
PREEMPT_RE = re.compile(r"Preemptions:\s*([0-9]+)")
FAULT_RE = re.compile(
    r"Memory Fault|HSA_STATUS_ERROR|page fault|Segmentation fault|core dumped"
)
EXPERT_RE = re.compile(
    r"RDNA2 expert cache .*?:\s*(\d+) logical experts,\s*(\d+) GPU slots;\s*"
    r"device\s*([0-9.]+)\s*GiB,\s*host\s*([0-9.]+)\s*GiB"
)
DMESG_RE = re.compile(r"new amdgpu dmesg faults:\s*(\d+|n/a)", re.IGNORECASE)
READY_RE = re.compile(r"READY in\s*(\d+)s")
WORKER_MEM_RE = re.compile(
    r"Actual usage is ([0-9]+(?:\.[0-9]+)?) GiB for consumed memory"
    r".*?([0-9]+(?:\.[0-9]+)?) GiB for peak activation"
    r".*?([0-9]+(?:\.[0-9]+)?) GiB for CUDAGraph memory"
    r".*?Current kv cache memory in use is ([0-9]+(?:\.[0-9]+)?) GiB"
)
GRAPH_POOL_RE = re.compile(
    r"(?:CUDA|ROCm) graph pool memory:\s*([0-9]+(?:\.[0-9]+)?) GiB "
    r"\(actual\),\s*([0-9]+(?:\.[0-9]+)?) GiB \(estimated\)"
)

COMPARE_KEYS = (
    "prefill_req_tps",
    "prefill_agg_bench_tps",
    "prefill_span_tps",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "decode_req_tps",
    "decode_only_tps",
    "itl_p50_ms",
    "itl_p99_ms",
    "output_throughput_e2e",
    "kv_tp_min_gib",
    "kv_size_tokens",
)
METRIC_LABELS = {
    "prefill_req_tps": "prefill req",
    "prefill_agg_bench_tps": "prefill agg",
    "prefill_span_tps": "prefill span",
    "ttft_p50_ms": "TTFT p50",
    "ttft_p99_ms": "TTFT p99",
    "decode_req_tps": "decode req",
    "decode_only_tps": "decode-only",
    "itl_p50_ms": "ITL p50",
    "itl_p99_ms": "ITL p99",
    "output_throughput_e2e": "out e2e",
    "kv_tp_min_gib": "KV TP-min GiB",
    "kv_size_tokens": "KV tokens (global)",
}

NOTES = [
    "F is the full 128-slot offload backend control: its device payload holds "
    "all 128 experts, so F-vs-R isolates the offload backend, graph and copy "
    "path at equal expert residency.",
    "D and Z use 115 of 128 local expert slots. The cache also allocates one "
    "all-zero dummy slot, so the device payload is 116 rows and the physical "
    "saving is 12 rows, not 13; 89.84% of local experts are resident.",
    "Offload arms are not precision- or graph-matched to R. "
    "`mono_decode_rdna2.check_config` refuses `expert_cache_slots` when "
    "`VLLM_RDNA2_W4A8_SDOT4=1`, so F/D/Z run with "
    "`VLLM_RDNA2_W4A8_SDOT4=0` and `_prepare_expert_cache` also sets "
    "`_w4a8 = False`; on gfx10x the expert cache auto-enables breakable "
    "graphs, which sets compilation mode NONE, while R keeps the recipe's "
    "compile mode 3 / FULL_AND_PIECEWISE. F isolates the offload "
    "backend/graph/copy path at full residency; D-vs-F isolates demand "
    "misses. Do not attribute the whole R-vs-offload delta to PCIe. A "
    "resident-W4A16 control can be run explicitly with `RESIDENT_SDOT4=0`; "
    "it is not the default.",
    "KV bytes come from `ExpertCacheWorker.rdna_cache_memory` on every arm "
    "before the matrix; `cache-memory.json` reports exact per-rank storage "
    "bytes after view/storage deduplication. The table shows TP-min and the "
    "sum across ranks; TP-min is the global capacity limit. `Available KV "
    "cache memory` is logged by rank 0 only and is shown separately, never as "
    "TP-min. If the RPC snapshot is missing, TP-min falls back to each rank's "
    "`Current kv cache memory in use` line, which is rounded to 0.01 GiB and "
    "is marked `~` in comparisons. Global KV token capacity is `GPU KV cache "
    "size` from startup, already the TP-min/global limit, and is the exact "
    "capacity-gain metric. Expert host/device bytes are reported "
    "separately and are not added to KV capacity or called KV gain. KV "
    "capacity is measured, not assumed: an offload arm can have less KV than "
    "R when weights, activations or graph pools consume more device memory; a "
    "negative ΔKV is a real capacity tradeoff, not a gain.",
    "Preemptions in the measured window are recorded as counts, not treated "
    "as a valid isolated-kernel window or zero. A window with preemptions is "
    "capacity-limited: its end-to-end, TTFT and ITL values include recompute "
    "and should not be read as isolated kernel cost.",
    "The worker-memory table parses the actual `gpu_worker` lines "
    "(`Actual usage is ... consumed memory ... peak activation ... CUDAGraph "
    "memory`) and the separate graph-pool actual/estimated line. It does not "
    "infer a reserve from other values; missing lines stay n/a.",
    "Expert-cache counters are deltas between `rdna_expert_cache_control` "
    "snapshots taken before and after each measured cell. The RPC runs "
    "between workloads, not inside the measured wall time; the counters are "
    "not phase timings. `copied_bytes` is host-to-device expert payload copy "
    "traffic, and `device_bytes` includes the dummy slot; neither is KV.",
    "`output_throughput_e2e` is total output tokens / benchmark duration, "
    "including prefill and queueing. `decode_only_tps` is total decoded tokens "
    "/ sum(latency - TTFT) over successful requests with output_tokens > 1; it "
    "is a decode-only estimate, not a wall-clock aggregate.",
    "`prefill_req_tps` is mean prompt tokens / mean TTFT. "
    "`prefill_agg_bench_tps` reuses bench_metrics: "
    "min(max_concurrency, completed) * mean prompt tokens / p99 TTFT, a "
    "one-wave approximation. `prefill_span_tps` is total prompt tokens / "
    "(last first-token time - first request start), which includes queueing "
    "across waves.",
    "Baseline-relative rows match cell name, replicate and measured request "
    "count. A same-named cell run with a different request count is marked "
    "`request_mismatch` and gets no delta; do not compare a 4-wave pilot cell "
    "with a 1-wave screen cell.",
    "Per-request decode uses actual output token counts and the request's "
    "decode interval (latency - TTFT), not 1/median ITL. ITL p50/p99 come from "
    "the benchmark result; `itl_partial` means streamed chunk intervals do not "
    "cover every generated token.",
    "Missing measurements are null in JSON, `n/a` in Markdown and blank in "
    "CSV; they are never reported as zero.",
]

_BENCH_METRICS: Any = None


def _bench_metrics() -> Any:
    global _BENCH_METRICS
    if _BENCH_METRICS is None:
        path = Path(__file__).resolve().parent.parent / "port_v031" / "bench_metrics.py"
        spec = importlib.util.spec_from_file_location("ram10_bench_metrics", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load bench_metrics.py from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BENCH_METRICS = module
    return _BENCH_METRICS


def _read_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ("1", "true", "yes", "ok"):
            return True
        if value.lower() in ("0", "false", "no"):
            return False
    if value in (0, 1):
        return bool(value)
    return None


def normalize_arm(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.upper() in ARM_ORDER:
        return text.upper()
    low = text.lower()
    for arm, name in ARM_NAMES.items():
        if low == name or low.startswith(name):
            return arm
    match = ARM_RE.match(text.upper())
    return match.group(1) if match else None


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (NumPy's default), stdlib only."""
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * p / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return xs[low]
    return xs[low] + (xs[high] - xs[low]) * (rank - low)


def parse_cell_name(name: str) -> tuple[int, int, int] | None:
    match = CELL_RE.fullmatch(name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _sort_key(name: str) -> tuple[int, int, int, str]:
    parsed = parse_cell_name(name)
    if parsed is None:
        return (10**9, 10**9, 10**9, name)
    return (*parsed, name)


def parse_serve_log(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "present": False,
        "kv_available_gib_per_rank": [],
        "kv_available_gib_min": None,
        "kv_available_gib_max": None,
        "kv_available_bytes_min_derived": None,
        "kv_rank0_available_gib_min": None,
        "kv_rank0_available_gib_max": None,
        "kv_rank0_available_bytes_derived": None,
        "kv_tp_min_in_use_gib": None,
        "kv_tp_min_in_use_gib_values": [],
        "kv_tp_min_in_use_bytes_derived": None,
        "kv_size_tokens": None,
        "kv_size_tokens_values": [],
        "kv_max_concurrency": None,
        "expert_cache_layers": 0,
        "expert_cache_slots": [],
        "expert_cache_device_gib_sum": None,
        "expert_cache_host_gib_sum": None,
        "worker_memory": [],
        "worker_consumed_gib_min": None,
        "worker_consumed_gib_max": None,
        "worker_activation_gib_min": None,
        "worker_activation_gib_max": None,
        "worker_graph_gib_min": None,
        "worker_graph_gib_max": None,
        "worker_kv_in_use_gib_per_rank": [],
        "worker_kv_in_use_gib_min": None,
        "worker_kv_in_use_gib_max": None,
        "graph_pool_actual_gib_min": None,
        "graph_pool_actual_gib_max": None,
        "graph_pool_estimated_gib_min": None,
        "graph_pool_estimated_gib_max": None,
        "preemptions_total": 0,
        "faults_total": 0,
        "dmesg_faults": None,
        "ready_s": None,
        "boot_failed": False,
        "line_count": 0,
    }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    out["present"] = True
    lines = text.splitlines()
    out["line_count"] = len(lines)
    kv_gib = [float(v) for v in KV_GIB_RE.findall(text)]
    kv_tokens = [int(v.replace(",", "")) for v in KV_TOK_RE.findall(text)]
    conc = [float(v) for v in KV_CONC_RE.findall(text)]
    expert = [
        (int(logical), int(slots), float(dev), float(host))
        for logical, slots, dev, host in EXPERT_RE.findall(text)
    ]
    preemptions = [int(v) for v in PREEMPT_RE.findall(text)]
    out["kv_available_gib_per_rank"] = kv_gib
    if kv_gib:
        out["kv_available_gib_min"] = min(kv_gib)
        out["kv_available_gib_max"] = max(kv_gib)
        out["kv_available_bytes_min_derived"] = int(round(min(kv_gib) * (2**30)))
        out["kv_rank0_available_gib_min"] = min(kv_gib)
        out["kv_rank0_available_gib_max"] = max(kv_gib)
        out["kv_rank0_available_bytes_derived"] = int(round(min(kv_gib) * (2**30)))
    out["kv_size_tokens_values"] = kv_tokens
    if kv_tokens:
        out["kv_size_tokens"] = min(kv_tokens)
    if conc:
        out["kv_max_concurrency"] = conc[0]
    out["expert_cache_layers"] = len(expert)
    out["expert_cache_slots"] = sorted({slots for _, slots, _, _ in expert})
    if expert:
        out["expert_cache_device_gib_sum"] = sum(dev for _, _, dev, _ in expert)
        out["expert_cache_host_gib_sum"] = sum(host for _, _, _, host in expert)
    worker = [
        {
            "consumed_gib": float(consumed),
            "activation_gib": float(activation),
            "graph_gib": float(graph),
            "kv_in_use_gib": float(kv_in_use),
        }
        for consumed, activation, graph, kv_in_use in WORKER_MEM_RE.findall(text)
    ]
    out["worker_memory"] = worker
    for key in ("consumed", "activation", "graph", "kv_in_use"):
        values = [row[f"{key}_gib"] for row in worker]
        if key == "kv_in_use":
            out["worker_kv_in_use_gib_per_rank"] = values
        if values:
            out[f"worker_{key}_gib_min"] = min(values)
            out[f"worker_{key}_gib_max"] = max(values)
    if worker:
        kv_values = [row["kv_in_use_gib"] for row in worker]
        out["kv_tp_min_in_use_gib_values"] = kv_values
        tp_min = min(kv_values)
        out["kv_tp_min_in_use_gib"] = tp_min
        out["kv_tp_min_in_use_bytes_derived"] = int(round(tp_min * (2**30)))
    graph_pool = [
        (float(actual), float(estimated))
        for actual, estimated in GRAPH_POOL_RE.findall(text)
    ]
    if graph_pool:
        out["graph_pool_actual_gib_min"] = min(v[0] for v in graph_pool)
        out["graph_pool_actual_gib_max"] = max(v[0] for v in graph_pool)
        out["graph_pool_estimated_gib_min"] = min(v[1] for v in graph_pool)
        out["graph_pool_estimated_gib_max"] = max(v[1] for v in graph_pool)
    out["preemptions_total"] = max(preemptions) if preemptions else 0
    out["faults_total"] = sum(1 for line in lines if FAULT_RE.search(line))
    out["boot_failed"] = any("BOOT FAILED" in line for line in lines)
    ready = READY_RE.search(text)
    if ready:
        out["ready_s"] = int(ready.group(1))
    try:
        summary = (path.parent / "summary.txt").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        summary = ""
    match = DMESG_RE.search(summary)
    if match and match.group(1).lower() != "n/a":
        out["dmesg_faults"] = int(match.group(1))
    return out


def parse_cache_memory(data: dict[str, Any] | None) -> dict[str, Any]:
    """Exact per-worker KV storage bytes from ExpertCacheWorker.rdna_cache_memory."""
    out: dict[str, Any] = {
        "present": False,
        "ranks": 0,
        "bytes_per_rank": [],
        "storage_bytes_min": None,
        "storage_bytes_sum": None,
        "storage_gib_min": None,
        "storage_gib_sum": None,
        "views": None,
        "storages": None,
        "devices": [],
    }
    if not isinstance(data, dict):
        return out
    results = data.get("results")
    if not isinstance(results, list) or not results:
        return out
    values: list[int] = []
    views = 0
    storages = 0
    devices: set[str] = set()
    for rank in results:
        if not isinstance(rank, dict):
            return out
        size = _to_int(rank.get("kv_cache_storage_bytes"))
        if size is None or size <= 0:
            return out
        values.append(size)
        views += _to_int(rank.get("kv_cache_views")) or 0
        storages += _to_int(rank.get("kv_cache_storages")) or 0
        for device in rank.get("kv_cache_devices") or []:
            devices.add(str(device))
    total = sum(values)
    out.update(
        {
            "present": True,
            "ranks": len(values),
            "bytes_per_rank": values,
            "storage_bytes_min": min(values),
            "storage_bytes_sum": total,
            "storage_gib_min": min(values) / (2**30),
            "storage_gib_sum": total / (2**30),
            "views": views,
            "storages": storages,
            "devices": sorted(devices),
        }
    )
    return out


def parse_cache_counters(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> dict[str, Any]:
    """Per-cell deltas from two rdna_expert_cache_control RPC snapshots."""
    out: dict[str, Any] = {
        "present": False,
        "ranks": 0,
        "demands": None,
        "misses": None,
        "miss_rate_pct": None,
        "copied_bytes": None,
        "committed_passes": None,
        "native_copy_calls": None,
        "batch_copy_calls": None,
        "map_uploads": None,
        "device_bytes": None,
        "host_bytes": None,
        "device_bytes_min_rank": None,
        "host_bytes_min_rank": None,
        "copy_modes": [],
        "debug_timing": None,
    }
    if not isinstance(before, dict) or not isinstance(after, dict):
        return out
    old = before.get("results")
    new = after.get("results")
    if not isinstance(old, list) or not isinstance(new, list):
        return out
    if not old or len(old) != len(new):
        return out
    delta_keys = (
        "expert_demands",
        "expert_misses",
        "copied_bytes",
        "committed_passes",
        "native_copy_calls",
        "batch_copy_calls",
        "map_uploads",
    )
    deltas = {key: 0 for key in delta_keys}
    device = []
    host = []
    modes: set[str] = set()
    debug_timing = False
    for before_rank, after_rank in zip(old, new):
        if not isinstance(before_rank, dict) or not isinstance(after_rank, dict):
            return out
        for key in delta_keys:
            start = _to_int(before_rank.get(key))
            end = _to_int(after_rank.get(key))
            if start is None or end is None:
                return out
            deltas[key] += end - start
        if _to_int(after_rank.get("device_bytes")) is not None:
            device.append(_to_int(after_rank.get("device_bytes")))
        if _to_int(after_rank.get("host_bytes")) is not None:
            host.append(_to_int(after_rank.get("host_bytes")))
        if after_rank.get("debug_timing"):
            debug_timing = True
        for mode in after_rank.get("copy_modes") or []:
            modes.add(str(mode))
    out.update(
        {
            "present": True,
            "ranks": len(new),
            "demands": deltas["expert_demands"],
            "misses": deltas["expert_misses"],
            "copied_bytes": deltas["copied_bytes"],
            "committed_passes": deltas["committed_passes"],
            "native_copy_calls": deltas["native_copy_calls"],
            "batch_copy_calls": deltas["batch_copy_calls"],
            "map_uploads": deltas["map_uploads"],
            "device_bytes": sum(device) if device else None,
            "device_bytes_min_rank": min(device) if device else None,
            "host_bytes": sum(host) if host else None,
            "host_bytes_min_rank": min(host) if host else None,
            "copy_modes": sorted(modes),
            "debug_timing": debug_timing,
        }
    )
    if deltas["expert_demands"] and deltas["expert_demands"] > 0:
        out["miss_rate_pct"] = (
            deltas["expert_misses"] / deltas["expert_demands"] * 100.0
        )
    return out


def _request_rows(result: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    keys = ("input_lens", "output_lens", "ttfts", "latencies", "itls", "errors")
    lists = {key: result.get(key) for key in keys}
    if not all(isinstance(lists[key], list) for key in keys):
        return [], False
    lengths = {len(lists[key]) for key in keys}
    if len(lengths) != 1:
        return [], False
    rows: list[dict[str, Any]] = []
    for index in range(next(iter(lengths))):
        err = lists["errors"][index]
        prompt = _to_int(lists["input_lens"][index])
        output = _to_int(lists["output_lens"][index])
        ttft = _to_float(lists["ttfts"][index])
        latency = _to_float(lists["latencies"][index])
        itl = lists["itls"][index]
        itl_values = [_to_float(v) for v in itl] if isinstance(itl, list) else []
        itl_ms = [v * 1000.0 for v in itl_values if v is not None and v >= 0]
        success = (
            not err
            and output is not None
            and output > 0
            and latency is not None
            and latency > 0
        )
        timing_ok = success and ttft is not None and ttft > 0 and latency >= ttft
        decode_tokens = output - 1 if output is not None and output > 1 else None
        decode_seconds = (
            latency - ttft
            if decode_tokens is not None and success and timing_ok and latency > ttft
            else None
        )
        decode_tps = (
            decode_tokens / decode_seconds
            if decode_tokens is not None and decode_seconds
            else None
        )
        prefill_tps = (
            prompt / ttft if prompt is not None and prompt > 0 and ttft else None
        )
        expected_itl = max((output or 0) - 1, 0) if output is not None else None
        itl_missing = (
            max(expected_itl - len(itl_values), 0) if expected_itl is not None else None
        )
        rows.append(
            {
                "request_index": index,
                "success": success,
                "timing_ok": timing_ok,
                "prompt_tokens": prompt,
                "output_tokens": output,
                "ttft_s": ttft,
                "latency_s": latency,
                "decode_s": decode_seconds,
                "decode_tokens": decode_tokens,
                "prefill_tok_s": prefill_tps,
                "decode_tok_s": decode_tps,
                "itl_n": len(itl_ms) if isinstance(itl, list) else None,
                "itl_expected": expected_itl,
                "itl_missing_tokens": itl_missing,
                "itl_mean_ms": (sum(itl_ms) / len(itl_ms) if itl_ms else None),
                "itl_p50_ms": percentile(itl_ms, 50),
                "itl_p99_ms": percentile(itl_ms, 99),
                "error": None if not err else str(err),
            }
        )
    return rows, True


def compute_cell(
    result: dict[str, Any] | None,
    *,
    status: dict[str, Any] | None,
    warmup: dict[str, Any] | None,
    cell: str,
    arm: str | None,
    replicate: str | None,
    cache_before: dict[str, Any] | None = None,
    cache_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = status or {}
    if not arm:
        arm = _infer_arm(status.get("arm"))
    if replicate in (None, "", "unknown") and status.get("replicate") is not None:
        replicate = str(status.get("replicate"))
    row: dict[str, Any] = {
        "cell": cell,
        "arm": arm,
        "replicate": replicate,
        "requested": None,
        "completed": None,
        "failed": None,
        "success_count": None,
        "prompt_tokens_mean": None,
        "input_tokens": None,
        "output_tokens": None,
        "duration_s": None,
        "ttft_mean_ms": None,
        "ttft_p50_ms": None,
        "ttft_p99_ms": None,
        "itl_mean_ms": None,
        "itl_p50_ms": None,
        "itl_p99_ms": None,
        "itl_complete": None,
        "prefill_req_tps": None,
        "prefill_req_tps_p50": None,
        "prefill_req_tps_p99": None,
        "prefill_agg_bench_tps": None,
        "prefill_span_tps": None,
        "decode_req_tps": None,
        "decode_req_tps_p50": None,
        "decode_req_tps_p99": None,
        "decode_only_tps": None,
        "output_throughput_e2e": None,
        "decode_measured": False,
        "mtp_accepted_tokens": None,
        "mtp_num_drafts": None,
        "mtp_draft_tokens": None,
        "mtp_acceptance_rate": None,
        "mtp_acceptance_length": None,
        "preemptions": _to_int(status.get("preemptions_in_window")),
        "capacity_limited": False,
        "evictions": status.get("evictions"),
        "faults": _to_int(status.get("faults_in_window")),
        "cache": None,
        "cache_demands": None,
        "cache_misses": None,
        "cache_miss_rate_pct": None,
        "cache_copied_bytes": None,
        "cache_device_bytes": None,
        "cache_host_bytes": None,
        "cache_ranks": None,
        "cache_copy_modes": [],
        "warmup_completed": None,
        "warmup_failed": None,
        "valid": False,
        "valid_for_ttft": False,
        "valid_for_decode": False,
        "invalid_reasons": [],
        "warnings": [],
        "per_request": [],
    }
    invalid: list[str] = []
    warnings: list[str] = []
    if result is None:
        invalid.append("no_result")
        row["invalid_reasons"] = invalid
        row["warnings"] = warnings
        return row

    completed = _to_int(result.get("completed"))
    failed = _to_int(result.get("failed"))
    requested = _to_int(result.get("num_prompts"))
    total_input = _to_float(result.get("total_input_tokens"))
    total_output = _to_float(result.get("total_output_tokens"))
    duration = _to_float(result.get("duration"))
    max_conc = _to_int(result.get("max_concurrency")) or 1
    mean_ttft = _to_float(result.get("mean_ttft_ms"))
    p50_ttft = _to_float(result.get("median_ttft_ms"))
    p99_ttft = _to_float(result.get("p99_ttft_ms"))
    mean_itl = _to_float(result.get("mean_itl_ms"))
    p50_itl = _to_float(result.get("median_itl_ms"))
    p99_itl = _to_float(result.get("p99_itl_ms"))
    mean_tpot = _to_float(result.get("mean_tpot_ms"))
    out_e2e = _to_float(result.get("output_throughput"))
    if out_e2e is None and duration and total_output is not None:
        out_e2e = total_output / duration
    ttft_pool = [
        float(value) * 1000.0
        for value in result.get("ttfts") or []
        if isinstance(value, (int, float)) and value >= 0
    ]
    itl_pool = [
        float(value) * 1000.0
        for values in result.get("itls") or []
        if isinstance(values, list)
        for value in values
        if isinstance(value, (int, float)) and value >= 0
    ]
    if ttft_pool:
        if mean_ttft is None:
            mean_ttft = sum(ttft_pool) / len(ttft_pool)
        if p50_ttft is None:
            p50_ttft = percentile(ttft_pool, 50)
        if p99_ttft is None:
            p99_ttft = percentile(ttft_pool, 99)
    if itl_pool:
        if mean_itl is None:
            mean_itl = sum(itl_pool) / len(itl_pool)
        if p50_itl is None:
            p50_itl = percentile(itl_pool, 50)
        if p99_itl is None:
            p99_itl = percentile(itl_pool, 99)
    rows, details = _request_rows(result)
    success_rows = [r for r in rows if r["success"]]
    success_count = len(success_rows) if details else None

    if completed is None:
        completed = success_count
    if requested is None and details:
        requested = len(rows)
    if failed is None and completed is not None and requested is not None:
        failed = max(requested - completed, 0)
    if completed is not None and completed <= 0:
        invalid.append("no_completed")
    if failed is not None and failed > 0:
        invalid.append("failed_requests")
    if not details:
        invalid.append("missing_per_request_detail")
    if duration is not None and duration <= 0:
        invalid.append("invalid_duration")

    prompt_mean = None
    if total_input is not None and completed and completed > 0:
        prompt_mean = total_input / completed
    bm: dict[str, Any] = {}
    if completed and completed > 0:
        try:
            bm = _bench_metrics().metrics(result)
        except (KeyError, TypeError, ValueError, ArithmeticError):
            warnings.append("bench_metrics_failed")
    prefill_req = (
        bm.get("prefill_req_tps")
        if mean_ttft is not None
        and mean_ttft > 0
        and prompt_mean is not None
        and prompt_mean > 0
        else None
    )
    if (
        prefill_req is None
        and mean_ttft is not None
        and mean_ttft > 0
        and prompt_mean is not None
        and prompt_mean > 0
    ):
        prefill_req = prompt_mean * 1000.0 / mean_ttft
    prefill_agg = (
        bm.get("prefill_agg_tps")
        if p99_ttft is not None
        and p99_ttft > 0
        and prompt_mean is not None
        and prompt_mean > 0
        else None
    )
    if (
        prefill_agg is None
        and p99_ttft is not None
        and p99_ttft > 0
        and prompt_mean is not None
        and prompt_mean > 0
    ):
        wave = min(max_conc, completed or 1)
        prefill_agg = wave * prompt_mean * 1000.0 / p99_ttft
    decode_req = (
        bm.get("decode_req_tps") if mean_tpot is not None and mean_tpot > 0 else None
    )
    if decode_req is None and mean_tpot is not None and mean_tpot > 0:
        decode_req = 1000.0 / mean_tpot

    prefill_p50 = percentile(
        [r["prefill_tok_s"] for r in success_rows if r["prefill_tok_s"] is not None],
        50,
    )
    prefill_p99 = percentile(
        [r["prefill_tok_s"] for r in success_rows if r["prefill_tok_s"] is not None],
        99,
    )
    decode_rates = [
        r["decode_tok_s"] for r in success_rows if r["decode_tok_s"] is not None
    ]
    decode_p50 = percentile(decode_rates, 50)
    decode_p99 = percentile(decode_rates, 99)
    decode_tokens = sum(r["decode_tokens"] for r in success_rows if r["decode_tokens"])
    decode_seconds = sum(
        r["decode_s"] for r in success_rows if r["decode_s"] is not None
    )
    decode_only = (
        decode_tokens / decode_seconds
        if decode_tokens and decode_seconds and decode_seconds > 0
        else None
    )
    if details:
        timing_bad = any(r["success"] and not r["timing_ok"] for r in rows)
        if timing_bad:
            invalid.append("invalid_request_timing")
        if not any(r["decode_tok_s"] is not None for r in success_rows):
            invalid.append("no_decode_interval")
    itl_complete = None
    if details:
        decode_rows = [r for r in success_rows if (r["output_tokens"] or 0) > 1]
        if decode_rows:
            itl_complete = all(r["itl_missing_tokens"] == 0 for r in decode_rows)
            if not itl_complete:
                warnings.append("itl_partial")

    start_times = result.get("start_times")
    prefill_span = None
    if (
        isinstance(start_times, list)
        and details
        and len(start_times) == len(rows)
        and total_input is not None
    ):
        points = []
        for index, start in enumerate(start_times):
            start_f = _to_float(start)
            ttft_f = _to_float(rows[index]["ttft_s"])
            if (
                rows[index]["success"]
                and start_f is not None
                and ttft_f is not None
                and ttft_f > 0
            ):
                points.append((start_f, start_f + ttft_f))
        if points:
            first_start = min(p[0] for p in points)
            last_first = max(p[1] for p in points)
            span = last_first - first_start
            if span > 0:
                prefill_span = total_input / span

    mtp_keys = {
        "mtp_accepted_tokens": "spec_decode_accepted_tokens",
        "mtp_num_drafts": "spec_decode_num_drafts",
        "mtp_draft_tokens": "spec_decode_draft_tokens",
        "mtp_acceptance_rate": "spec_decode_acceptance_rate",
        "mtp_acceptance_length": "spec_decode_acceptance_length",
    }
    mtp_present = False
    for out_key, src_key in mtp_keys.items():
        if src_key in result:
            mtp_present = True
            row[out_key] = _to_float(result.get(src_key))
    if not mtp_present:
        warnings.append("mtp_metrics_missing")

    cache = parse_cache_counters(cache_before, cache_after)
    if arm in ("F", "D", "Z"):
        if not cache["present"]:
            warnings.append("cache_counters_missing")
        else:
            if any(mode == "native" for mode in cache["copy_modes"]):
                invalid.append("native_copy")
            if cache.get("native_copy_calls"):
                invalid.append("native_copy")
        if cache.get("debug_timing"):
            invalid.append("debug_timing_enabled")

    if status:
        if _to_int(status.get("bench_rc")) not in (None, 0):
            invalid.append("bench_failed")
        if _to_int(status.get("warmup_rc")) not in (None, 0):
            invalid.append("warmup_failed")
        if _as_bool(status.get("reset_before_warmup_ok")) is False:
            invalid.append("prefix_reset_failed")
        if _as_bool(status.get("reset_after_warmup_ok")) is False:
            invalid.append("prefix_reset_failed")
        if _as_bool(status.get("server_alive_after")) is False:
            invalid.append("server_died")
        if _as_bool(status.get("result_present")) is False:
            invalid.append("no_result")
        if (
            _as_bool(status.get("warmup_present")) is False
            and _as_bool(status.get("warmup_enabled")) is not False
        ):
            invalid.append("warmup_missing")
        evictions = status.get("evictions")
        if evictions == "INVALID":
            invalid.append("kfd_evictions")
        elif evictions in ("unavailable", None, ""):
            warnings.append("kfd_evictions_unavailable")
        if (row["preemptions"] or 0) > 0:
            row["capacity_limited"] = True
            if "preemptions_observed" not in warnings:
                warnings.append("preemptions_observed")
        if (row["faults"] or 0) > 0:
            invalid.append("faults")
        for reason in status.get("invalid_reasons") or []:
            if reason not in invalid:
                invalid.append(str(reason))
        for warning in status.get("warnings") or []:
            if warning not in warnings:
                warnings.append(str(warning))
    else:
        warnings.append("cell_status_missing")

    if warmup is not None:
        row["warmup_completed"] = _to_int(warmup.get("completed"))
        row["warmup_failed"] = _to_int(warmup.get("failed"))
    if (
        details
        and completed is not None
        and success_count is not None
        and completed != success_count
    ):
        warnings.append("completed_count_mismatch")
    if (
        details
        and requested is not None
        and completed is not None
        and failed is not None
        and completed + failed != requested
    ):
        warnings.append("request_count_mismatch")

    row.update(
        {
            "requested": requested,
            "completed": completed,
            "failed": failed,
            "success_count": success_count,
            "prompt_tokens_mean": prompt_mean,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "duration_s": duration,
            "ttft_mean_ms": mean_ttft,
            "ttft_p50_ms": p50_ttft,
            "ttft_p99_ms": p99_ttft,
            "itl_mean_ms": mean_itl,
            "itl_p50_ms": p50_itl,
            "itl_p99_ms": p99_itl,
            "itl_complete": itl_complete,
            "prefill_req_tps": prefill_req,
            "prefill_req_tps_p50": prefill_p50,
            "prefill_req_tps_p99": prefill_p99,
            "prefill_agg_bench_tps": prefill_agg,
            "prefill_span_tps": prefill_span,
            "decode_req_tps": decode_req,
            "decode_req_tps_p50": decode_p50,
            "decode_req_tps_p99": decode_p99,
            "decode_only_tps": decode_only,
            "output_throughput_e2e": out_e2e,
            "decode_measured": decode_only is not None,
            "cache": cache,
            "cache_demands": cache["demands"],
            "cache_misses": cache["misses"],
            "cache_miss_rate_pct": cache["miss_rate_pct"],
            "cache_copied_bytes": cache["copied_bytes"],
            "cache_device_bytes": cache["device_bytes"],
            "cache_host_bytes": cache["host_bytes"],
            "cache_ranks": cache["ranks"],
            "cache_copy_modes": cache["copy_modes"],
            "invalid_reasons": invalid,
            "warnings": warnings,
            "per_request": rows,
        }
    )
    row["valid"] = not invalid
    row["valid_for_ttft"] = row["valid"] and p50_ttft is not None
    row["valid_for_decode"] = row["valid"] and row["decode_measured"]
    return row


def finalize_cell_dir(cell_dir: Path) -> dict[str, Any]:
    """Merge report-derived validity/metrics into a runner cell.json."""
    cell_dir = Path(cell_dir)
    result = _read_json(cell_dir / "bench.json")
    status = _read_json(cell_dir / "cell.json") or {}
    warmup = _read_json(cell_dir / "warmup.json")
    cache_before = _read_json(cell_dir / "cache-before.json")
    cache_after = _read_json(cell_dir / "cache-after.json")
    name = status.get("cell") or cell_dir.name
    row = compute_cell(
        result,
        status=status,
        warmup=warmup,
        cell=str(name),
        arm=status.get("arm"),
        replicate=(
            str(status.get("replicate"))
            if status.get("replicate") is not None
            else None
        ),
        cache_before=cache_before,
        cache_after=cache_after,
    )
    summary_keys = (
        "completed",
        "failed",
        "requested",
        "prefill_req_tps",
        "prefill_agg_bench_tps",
        "prefill_span_tps",
        "decode_req_tps",
        "decode_only_tps",
        "output_throughput_e2e",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "itl_p50_ms",
        "itl_p99_ms",
        "cache_demands",
        "cache_misses",
        "cache_miss_rate_pct",
        "cache_copied_bytes",
        "cache_device_bytes",
        "cache_host_bytes",
        "cache_ranks",
        "capacity_limited",
    )
    status.update(
        {
            "valid": row["valid"],
            "valid_for_ttft": row["valid_for_ttft"],
            "valid_for_decode": row["valid_for_decode"],
            "invalid_reasons": row["invalid_reasons"],
            "warnings": row["warnings"],
            "decode_measured": row["decode_measured"],
            "itl_complete": row["itl_complete"],
            "metrics": {key: row[key] for key in summary_keys},
        }
    )
    _write_json(cell_dir / "cell.json", status)
    return row


def _infer_arm(*values: Any) -> str | None:
    for value in values:
        arm = normalize_arm(value)
        if arm:
            return arm
    return None


def _is_trace_path(path: Path) -> bool:
    """Post-matrix trace artifacts live under OUT/trace and never count as cells."""
    return "trace" in path.parts


def _discover_runs(run_root: Path) -> list[Path]:
    if (run_root / "metadata.json").is_file():
        return [] if _is_trace_path(run_root) else [run_root]
    found: set[Path] = set()
    for meta in run_root.glob("*/metadata.json"):
        if not _is_trace_path(meta):
            found.add(meta.parent)
    for bench in run_root.glob("*/cells/*/bench.json"):
        if len(bench.parents) >= 3 and not _is_trace_path(bench):
            found.add(bench.parents[2])
    for bench in run_root.glob("*/bench-*.json"):
        if not _is_trace_path(bench):
            found.add(bench.parent)
    return sorted(found)


def _cell_paths(run_dir: Path) -> list[tuple[str, Path]]:
    cells: dict[str, Path] = {}
    for bench in sorted(run_dir.glob("cells/*/bench.json")):
        if not _is_trace_path(bench):
            cells[bench.parent.name] = bench
    for bench in sorted(run_dir.glob("bench-*.json")):
        if not _is_trace_path(bench):
            cells.setdefault(bench.stem, bench)
    return sorted(cells.items(), key=lambda item: _sort_key(item[0]))


def build_report(
    run_root: Path,
    *,
    baseline_arm: str = "R",
    replicate: str | None = None,
    arms: list[str] | None = None,
) -> dict[str, Any]:
    run_root = Path(run_root)
    baseline_arm = _infer_arm(baseline_arm) or baseline_arm
    run_dirs = _discover_runs(run_root)
    runs: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    per_request_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run_dir in run_dirs:
        meta = _read_json(run_dir / "metadata.json") or {}
        serve = parse_serve_log(run_dir / "serve.log")
        cache_memory = parse_cache_memory(_read_json(run_dir / "cache-memory.json"))
        cell_paths = _cell_paths(run_dir)
        first_result = _read_json(cell_paths[0][1]) if cell_paths else None
        first_status = (
            _read_json(cell_paths[0][1].parent / "cell.json") if cell_paths else None
        )
        arm = _infer_arm(
            meta.get("arm"),
            meta.get("arm_name"),
            (first_result or {}).get("ram10_arm"),
            (first_status or {}).get("arm"),
        )
        replicate_value = (
            meta.get("replicate")
            or (first_result or {}).get("ram10_replicate")
            or (first_status or {}).get("replicate")
            or "unknown"
        )
        tag = (
            meta.get("tag")
            or (first_result or {}).get("ram10_tag")
            or (first_status or {}).get("tag")
            or run_dir.name
        )
        if model_arm := _infer_arm(tag):
            arm = arm or model_arm
        baseline_name = _infer_arm(baseline_arm) or baseline_arm.upper()
        if arms and arm not in arms and arm != baseline_name:
            continue
        if replicate is not None and str(replicate_value) != str(replicate):
            continue
        run_cells: list[dict[str, Any]] = []
        for name, bench_path in cell_paths:
            if _is_trace_path(bench_path):
                continue
            cell_dir = bench_path.parent if bench_path.parent.name == name else run_dir
            result = _read_json(bench_path)
            status = _read_json(cell_dir / "cell.json")
            warmup = _read_json(cell_dir / "warmup.json")
            cache_before = _read_json(cell_dir / "cache-before.json")
            cache_after = _read_json(cell_dir / "cache-after.json")
            row = compute_cell(
                result,
                status=status,
                warmup=warmup,
                cell=name,
                arm=arm,
                replicate=str(replicate_value),
                cache_before=cache_before,
                cache_after=cache_after,
            )
            row["tag"] = tag
            row["run_path"] = str(run_dir)
            row["kv_available_gib_min"] = serve["kv_available_gib_min"]
            row["kv_available_bytes_min_derived"] = serve[
                "kv_available_bytes_min_derived"
            ]
            row["kv_rank0_available_gib_min"] = serve["kv_rank0_available_gib_min"]
            row["kv_rank0_available_bytes_derived"] = serve[
                "kv_rank0_available_bytes_derived"
            ]
            row["kv_tp_min_in_use_gib"] = serve["kv_tp_min_in_use_gib"]
            row["kv_tp_min_in_use_bytes_derived"] = serve[
                "kv_tp_min_in_use_bytes_derived"
            ]
            row["kv_size_tokens"] = serve["kv_size_tokens"]
            row["kv_tp_min_storage_bytes"] = cache_memory["storage_bytes_min"]
            row["kv_tp_sum_storage_bytes"] = cache_memory["storage_bytes_sum"]
            row["kv_tp_min_storage_gib"] = cache_memory["storage_gib_min"]
            row["kv_tp_sum_storage_gib"] = cache_memory["storage_gib_sum"]
            row["kv_tp_min_gib"] = None
            row["kv_tp_min_bytes"] = None
            row["kv_tp_min_source"] = None
            if cache_memory["present"]:
                row["kv_tp_min_gib"] = cache_memory["storage_gib_min"]
                row["kv_tp_min_bytes"] = cache_memory["storage_bytes_min"]
                row["kv_tp_min_source"] = "worker_rpc_exact"
            elif row["kv_tp_min_in_use_gib"] is not None:
                row["kv_tp_min_gib"] = row["kv_tp_min_in_use_gib"]
                row["kv_tp_min_bytes"] = row["kv_tp_min_in_use_bytes_derived"]
                row["kv_tp_min_source"] = "serve_log_rounded"
            row["preemptions_boot"] = serve["preemptions_total"]
            row["faults_boot"] = serve["faults_total"]
            for request in row.pop("per_request"):
                per_request_rows.append(
                    {
                        "arm": arm,
                        "replicate": replicate_value,
                        "tag": tag,
                        "cell": name,
                        **request,
                    }
                )
            run_cells.append(row)
            cells.append(row)
        if not cell_paths:
            warnings.append(f"{run_dir.name}: no benchmark result files")
        if not serve["present"]:
            warnings.append(f"{run_dir.name}: serve.log missing")
        if serve["boot_failed"]:
            warnings.append(f"{run_dir.name}: BOOT FAILED in serve.log")
        if serve["present"] and serve["ready_s"] is None:
            warnings.append(f"{run_dir.name}: READY line missing from serve.log")
        if serve["kv_available_gib_min"] is None:
            warnings.append(
                f"{run_dir.name}: rank0 Available KV cache memory line missing"
            )
        if serve["kv_tp_min_in_use_gib"] is None:
            warnings.append(
                f"{run_dir.name}: per-rank Current kv cache memory lines missing"
            )
        if serve["kv_size_tokens"] is None:
            warnings.append(f"{run_dir.name}: KV token capacity missing from serve.log")
        if serve["present"] and not serve["worker_memory"]:
            warnings.append(
                f"{run_dir.name}: actual worker memory lines missing from serve.log"
            )
        if not cache_memory["present"]:
            warnings.append(
                f"{run_dir.name}: exact rdna_cache_memory snapshot missing; "
                "KV TP-min falls back to rounded serve.log values"
            )
        runs.append(
            {
                "arm": arm,
                "name": ARM_NAMES.get(arm or "", "unknown"),
                "replicate": str(replicate_value),
                "tag": tag,
                "path": str(run_dir),
                "status": meta.get("status"),
                "exit_code": _to_int(meta.get("exit_code")),
                "slots": _to_int(meta.get("slots")),
                "local_experts": _to_int(meta.get("local_experts")),
                "dummy_slot": meta.get("dummy_slot"),
                "device_cache_rows": _to_int(meta.get("device_cache_rows")),
                "rows_saved_vs_128": _to_int(meta.get("rows_saved_vs_128")),
                "resident_fraction": _to_float(meta.get("resident_fraction")),
                "w4a8_prefill": meta.get("w4a8_prefill"),
                "w4a8_sdot4": meta.get("w4a8_sdot4"),
                "compile_mode": meta.get("compile_mode"),
                "cudagraph_mode": meta.get("cudagraph_mode"),
                "breakable_cudagraph": meta.get("breakable_cudagraph"),
                "deterministic_dense": meta.get("deterministic_dense"),
                "mono_knobs": meta.get("mono_knobs"),
                "after_matrix_hook": meta.get("after_matrix_hook"),
                "tree_sha": meta.get("tree_sha"),
                "git_head": meta.get("git_head"),
                "git_dirty": meta.get("git_dirty"),
                "serve": serve,
                "cache_memory": cache_memory,
                "cell_count": len(run_cells),
            }
        )

    baseline_runs = [run for run in runs if run["arm"] == baseline_arm]
    comparisons: list[dict[str, Any]] = []
    for cell in cells:
        candidates = [
            run
            for run in baseline_runs
            if any(
                other["cell"] == cell["cell"] and other["replicate"] == run["replicate"]
                for other in cells
                if other["run_path"] == run["path"]
            )
        ]
        same_rep = [run for run in candidates if run["replicate"] == cell["replicate"]]
        chosen = same_rep
        comparison: dict[str, Any] = {
            "cell": cell["cell"],
            "arm": cell["arm"],
            "replicate": cell["replicate"],
            "baseline_arm": baseline_arm,
            "baseline_replicate": None,
            "baseline_tag": None,
            "status": "ok",
            "metrics": {},
        }
        if not chosen:
            status = "replicate_mismatch" if candidates else "missing_baseline"
            comparison["status"] = status
            for key in COMPARE_KEYS:
                comparison["metrics"][key] = {
                    "candidate": cell.get(key),
                    "baseline": None,
                    "delta": None,
                    "pct": None,
                    "status": status,
                }
            comparisons.append(comparison)
            continue
        if len(chosen) > 1:
            comparison["status"] = "ambiguous_baseline"
            for key in COMPARE_KEYS:
                comparison["metrics"][key] = {
                    "candidate": cell.get(key),
                    "baseline": None,
                    "delta": None,
                    "pct": None,
                    "status": "ambiguous_baseline",
                }
            comparisons.append(comparison)
            continue
        baseline_run = chosen[0]
        comparison["baseline_replicate"] = baseline_run["replicate"]
        comparison["baseline_tag"] = baseline_run["tag"]
        baseline_cell = next(
            (
                other
                for other in cells
                if other["run_path"] == baseline_run["path"]
                and other["cell"] == cell["cell"]
            ),
            None,
        )
        if baseline_cell is None:
            comparison["status"] = "missing_baseline"
            comparisons.append(comparison)
            continue
        candidate_requests = cell.get("requested")
        baseline_requests = baseline_cell.get("requested")
        if (
            candidate_requests is not None
            and baseline_requests is not None
            and candidate_requests != baseline_requests
        ):
            comparison["status"] = "request_mismatch"
            comparison["candidate_requests"] = candidate_requests
            comparison["baseline_requests"] = baseline_requests
            for key in COMPARE_KEYS:
                comparison["metrics"][key] = {
                    "candidate": cell.get(key),
                    "baseline": baseline_cell.get(key),
                    "delta": None,
                    "pct": None,
                    "status": "request_mismatch",
                }
            comparisons.append(comparison)
            continue
        for key in COMPARE_KEYS:
            candidate_value = cell.get(key)
            baseline_value = baseline_cell.get(key)
            metric: dict[str, Any] = {
                "candidate": candidate_value,
                "baseline": baseline_value,
                "delta": None,
                "pct": None,
                "status": "ok",
            }
            if candidate_value is None or baseline_value is None:
                metric["status"] = "missing_metric"
            elif not cell["valid"] or not baseline_cell["valid"]:
                metric["status"] = "invalid_window"
            elif key.startswith("itl_") and (
                cell.get("itl_complete") is False
                or baseline_cell.get("itl_complete") is False
            ):
                metric["status"] = "partial_itl"
            else:
                if key == "kv_tp_min_gib" and (
                    cell.get("kv_tp_min_source") != "worker_rpc_exact"
                    or baseline_cell.get("kv_tp_min_source") != "worker_rpc_exact"
                ):
                    metric["status"] = "rounded_fallback"
                delta = candidate_value - baseline_value
                metric["delta"] = delta
                if baseline_value == 0:
                    metric["status"] = "baseline_zero"
                else:
                    metric["pct"] = (candidate_value / baseline_value - 1.0) * 100.0
            comparison["metrics"][key] = metric
        comparisons.append(comparison)

    arms_summary = []
    seen_arms: set[str | None] = set()
    for run in runs:
        if run["arm"] in seen_arms:
            continue
        seen_arms.add(run["arm"])
        arms_summary.append(run)

    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "run_root": str(run_root),
        "baseline_arm": baseline_arm,
        "filters": {"replicate": replicate, "arms": arms},
        "arms": arms_summary,
        "runs": runs,
        "cells": cells,
        "comparisons": comparisons,
        "per_request_rows": per_request_rows,
        "warnings": warnings,
        "notes": NOTES,
    }


def _f(value: Any, digits: int = 1) -> str:
    number = _to_float(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _i(value: Any) -> str:
    number = _to_int(value)
    return "n/a" if number is None else str(number)


def _bool(value: Any) -> str:
    flag = _as_bool(value)
    if flag is None:
        return "n/a"
    return "yes" if flag else "no"


def _pct(value: Any) -> str:
    number = _to_float(value)
    return "n/a" if number is None else f"{number:+.1f}%"


def _kv_gib_cell(cell: dict[str, Any]) -> str:
    value = _to_float(cell.get("kv_tp_min_gib"))
    if value is None:
        return "n/a"
    prefix = "~" if cell.get("kv_tp_min_source") == "serve_log_rounded" else ""
    return prefix + f"{value:.3f}"


def _minmax(lo: Any, hi: Any, digits: int = 2) -> str:
    low = _to_float(lo)
    high = _to_float(hi)
    if low is None:
        return "n/a"
    if high is None or abs(high - low) < 0.005:
        return f"{low:.{digits}f}"
    return f"{low:.{digits}f}-{high:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _cmp_cell(metric: dict[str, Any] | None) -> str:
    if not metric:
        return "n/a"
    if metric.get("pct") is not None:
        prefix = "~" if metric.get("status") == "rounded_fallback" else ""
        return prefix + _pct(metric["pct"])
    if metric.get("status") == "baseline_zero":
        return "base0"
    return "n/a"


def markdown_report(report: dict[str, Any]) -> str:
    out: list[str] = ["# RAM10 study report", ""]
    out.append(f"Generated: `{report['generated_utc']}`")
    out.append(f"Run root: `{report['run_root']}`")
    out.append(f"Baseline arm: `{report['baseline_arm']}`")
    out.append("")

    out.append("## Settings")
    rows = []
    for arm in report["arms"]:
        rows.append(
            [
                f"{arm['arm'] or '?'} ({arm['name']})",
                _i(arm.get("slots")),
                _i(arm.get("local_experts")),
                _bool(arm.get("dummy_slot")),
                _i(arm.get("device_cache_rows")) if arm.get("slots") else "resident",
                _i(arm.get("rows_saved_vs_128")),
                (
                    f"{arm['resident_fraction'] * 100:.2f}%"
                    if arm.get("resident_fraction") is not None
                    else "n/a"
                ),
                _bool(arm.get("w4a8_prefill")),
                _i(arm.get("w4a8_sdot4")),
                _bool(arm.get("breakable_cudagraph")),
                (
                    f"compile {_i(arm.get('compile_mode'))} / "
                    f"{arm.get('cudagraph_mode') or 'n/a'}"
                ),
                _bool(arm.get("deterministic_dense")),
                f"`{json.dumps(arm.get('mono_knobs'), sort_keys=True)}`"
                if arm.get("mono_knobs")
                else "n/a",
            ]
        )
    out.append(
        _table(
            [
                "Arm",
                "Slots",
                "Local experts",
                "Dummy slot",
                "Device cache rows",
                "Rows saved vs 128",
                "Resident",
                "W4A8 prefill",
                "W4A8 SDOT4",
                "Breakable graphs",
                "Graph config",
                "Deterministic dense",
                "mono knobs",
            ],
            rows,
        )
    )
    out.append("")

    exact_rows = []
    for run in report["runs"]:
        memory = run.get("cache_memory") or {}
        if not memory.get("present"):
            continue
        exact_rows.append(
            [
                f"{run['arm'] or '?'} ({run['name']})",
                str(run["replicate"]),
                str(memory.get("ranks")),
                _i(memory.get("storage_bytes_min")),
                _f(memory.get("storage_gib_min"), 3),
                _i(memory.get("storage_bytes_sum")),
                _f(memory.get("storage_gib_sum"), 3),
                _i(memory.get("views")),
                _i(memory.get("storages")),
                ", ".join(memory.get("devices") or []) or "n/a",
            ]
        )
    out.append("## KV allocation (exact worker storage)")
    if exact_rows:
        out.append(
            _table(
                [
                    "Arm",
                    "Rep",
                    "Ranks",
                    "TP-min bytes",
                    "TP-min GiB",
                    "Sum bytes",
                    "Sum GiB",
                    "Views",
                    "Storages",
                    "Devices",
                ],
                exact_rows,
            )
        )
        out.append("")
        out.append(
            "TP-min is the smallest per-rank `kv_cache_storage_bytes`; global "
            "capacity is limited by it. Byte counts include group/page padding "
            "and deduplicate KV views sharing the same untyped storage."
        )
    else:
        out.append(
            "No run provided `cache-memory.json`; KV TP-min uses rounded "
            "serve.log values and is marked `~` in comparisons."
        )
    out.append("")

    out.append("## Runs")
    rows = []
    for run in report["runs"]:
        serve = run["serve"]
        rows.append(
            [
                f"{run['arm'] or '?'} ({run['name']})",
                str(run["replicate"]),
                run["tag"],
                str(run.get("status") or "unknown"),
                _i(run.get("exit_code")),
                _i(serve.get("ready_s")),
                _f(serve.get("kv_rank0_available_gib_min"), 2),
                _f(serve.get("kv_tp_min_in_use_gib"), 2),
                _i(serve.get("kv_size_tokens")),
                _i(serve.get("preemptions_total")),
                _i(serve.get("faults_total")),
                _i(serve.get("dmesg_faults")),
                (run.get("tree_sha") or run.get("git_head") or "n/a")[:12],
                Path(run["path"]).name,
            ]
        )
    out.append(
        _table(
            [
                "Arm",
                "Rep",
                "Tag",
                "Status",
                "Exit",
                "Ready s",
                "KV rank0 avail GiB (log est.)",
                "KV TP-min in-use GiB (rounded)",
                "KV tokens (global)",
                "Preemptions",
                "Faults",
                "dmesg faults",
                "Tree",
                "Run dir",
            ],
            rows,
        )
    )
    out.append("")

    out.append("## Worker memory (actual, serve.log)")
    rows = []
    for run in report["runs"]:
        serve = run["serve"]
        rows.append(
            [
                f"{run['arm'] or '?'} ({run['name']})",
                str(run["replicate"]),
                _minmax(
                    serve.get("worker_consumed_gib_min"),
                    serve.get("worker_consumed_gib_max"),
                ),
                _minmax(
                    serve.get("worker_activation_gib_min"),
                    serve.get("worker_activation_gib_max"),
                ),
                _minmax(
                    serve.get("worker_graph_gib_min"),
                    serve.get("worker_graph_gib_max"),
                ),
                _minmax(
                    serve.get("worker_kv_in_use_gib_min"),
                    serve.get("worker_kv_in_use_gib_max"),
                ),
                (
                    _minmax(
                        serve.get("graph_pool_actual_gib_min"),
                        serve.get("graph_pool_actual_gib_max"),
                    )
                    + " / "
                    + _minmax(
                        serve.get("graph_pool_estimated_gib_min"),
                        serve.get("graph_pool_estimated_gib_max"),
                    )
                ),
            ]
        )
    if rows:
        out.append(
            _table(
                [
                    "Arm",
                    "Rep",
                    "Consumed GiB",
                    "Peak activation GiB",
                    "CUDAGraph GiB",
                    "KV in use GiB",
                    "Graph pool actual / estimated GiB",
                ],
                rows,
            )
        )
    else:
        out.append("_No actual worker memory lines found in serve.log._")
    out.append("")

    out.append("## Measured cells: prefill and end-to-end")
    rows = []
    for cell in report["cells"]:
        requested = _i(cell.get("requested"))
        completed = _i(cell.get("completed"))
        rows.append(
            [
                cell["cell"],
                cell["arm"] or "?",
                str(cell["replicate"]),
                f"{completed}/{requested}",
                _f(cell.get("prompt_tokens_mean"), 0),
                _f(cell.get("ttft_p50_ms")),
                _f(cell.get("ttft_p99_ms")),
                _f(cell.get("prefill_req_tps")),
                _f(cell.get("prefill_agg_bench_tps")),
                _f(cell.get("prefill_span_tps")),
                _f(cell.get("output_throughput_e2e"), 2),
                "yes" if cell["valid"] else "no",
            ]
        )
    out.append(
        _table(
            [
                "Cell",
                "Arm",
                "Rep",
                "ok/req",
                "Prompt tok",
                "TTFT p50 ms",
                "TTFT p99 ms",
                "Prefill req tok/s",
                "Prefill agg tok/s",
                "Prefill span tok/s",
                "Out e2e tok/s",
                "Valid",
            ],
            rows,
        )
    )
    out.append("")

    out.append("## Measured cells: decode and health")
    rows = []
    for cell in report["cells"]:
        mtp = "n/a"
        if cell.get("mtp_acceptance_rate") is not None:
            mtp = f"{_f(cell['mtp_acceptance_rate'], 1)}%"
            if cell.get("mtp_accepted_tokens") is not None:
                mtp += (
                    f" ({_i(cell['mtp_accepted_tokens'])}/"
                    f"{_i(cell.get('mtp_num_drafts'))})"
                )
        rows.append(
            [
                cell["cell"],
                cell["arm"] or "?",
                str(cell["replicate"]),
                _f(cell.get("decode_req_tps"), 2),
                _f(cell.get("decode_only_tps"), 2),
                _f(cell.get("decode_req_tps_p50"), 2),
                _f(cell.get("decode_req_tps_p99"), 2),
                _f(cell.get("itl_p50_ms")),
                _f(cell.get("itl_p99_ms")),
                mtp,
                _i(cell.get("preemptions")),
                str(cell.get("evictions") or "n/a"),
                _i(cell.get("faults")),
                _kv_gib_cell(cell),
                _i(cell.get("kv_size_tokens")),
                "yes" if cell["valid"] else "no",
            ]
        )
    out.append(
        _table(
            [
                "Cell",
                "Arm",
                "Rep",
                "Decode req tok/s",
                "Decode-only tok/s",
                "Decode p50",
                "Decode p99",
                "ITL p50 ms",
                "ITL p99 ms",
                "MTP acc (acc/drafts)",
                "Preempt",
                "Evict",
                "Faults",
                "KV TP-min GiB (rounded)",
                "KV tokens (global)",
                "Valid",
            ],
            rows,
        )
    )
    out.append("")

    out.append("## Expert-cache counters (between-workload RPC deltas)")
    rows = []
    for cell in report["cells"]:
        rows.append(
            [
                cell["cell"],
                cell["arm"] or "?",
                str(cell["replicate"]),
                _i(cell.get("cache_ranks")),
                _i(cell.get("cache_demands")),
                _i(cell.get("cache_misses")),
                _f(cell.get("cache_miss_rate_pct"), 2),
                _f(
                    (
                        cell["cache_copied_bytes"] / (1024**2)
                        if cell.get("cache_copied_bytes") is not None
                        else None
                    ),
                    2,
                ),
                _f(
                    (
                        cell["cache_device_bytes"] / (1024**3)
                        if cell.get("cache_device_bytes") is not None
                        else None
                    ),
                    3,
                ),
                _f(
                    (
                        cell["cache_host_bytes"] / (1024**3)
                        if cell.get("cache_host_bytes") is not None
                        else None
                    ),
                    3,
                ),
                ",".join(cell.get("cache_copy_modes") or []) or "n/a",
            ]
        )
    out.append(
        _table(
            [
                "Cell",
                "Arm",
                "Rep",
                "Ranks",
                "Demands",
                "Misses",
                "Miss %",
                "Copied MiB",
                "Device GiB",
                "Host GiB",
                "Copy mode",
            ],
            rows,
        )
    )
    out.append("")

    out.append("## Baseline-relative metrics (matched cells)")
    headers = ["Cell", "Arm", "Rep", "Baseline"]
    headers += [f"Δ {METRIC_LABELS[key]}" for key in COMPARE_KEYS]
    headers.append("Status")
    rows = []
    for comparison in report["comparisons"]:
        metrics = comparison["metrics"]
        rows.append(
            [
                comparison["cell"],
                comparison["arm"] or "?",
                str(comparison["replicate"]),
                comparison["baseline_tag"] or "n/a",
                *[_cmp_cell(metrics.get(key)) for key in COMPARE_KEYS],
                comparison["status"],
            ]
        )
    out.append(_table(headers, rows))
    out.append("")

    out.append("## Missing and invalid windows")
    missing = []
    for cell in report["cells"]:
        if cell["invalid_reasons"]:
            missing.append(
                f"- {cell['arm']}/{cell['replicate']}/{cell['cell']}: invalid "
                f"({', '.join(cell['invalid_reasons'])})"
            )
        if cell["warnings"]:
            missing.append(
                f"- {cell['arm']}/{cell['replicate']}/{cell['cell']}: "
                f"{', '.join(cell['warnings'])}"
            )
    missing.extend(f"- run: {warning}" for warning in report["warnings"])
    out.extend(missing or ["- none"])
    out.append("")

    out.append("## Notes")
    out.extend(f"- {note}" for note in report["notes"])
    out.append("")
    return "\n".join(out)


def _csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    columns = (
        "arm",
        "replicate",
        "tag",
        "cell",
        "request_index",
        "success",
        "prompt_tokens",
        "output_tokens",
        "ttft_s",
        "latency_s",
        "decode_s",
        "decode_tokens",
        "prefill_tok_s",
        "decode_tok_s",
        "itl_n",
        "itl_expected",
        "itl_missing_tokens",
        "itl_mean_ms",
        "itl_p50_ms",
        "itl_p99_ms",
        "error",
    )
    rows = []
    for row in report["per_request_rows"]:
        out = {}
        for key in columns:
            value = row.get(key)
            if key == "success":
                out[key] = "" if value is None else int(bool(value))
            elif value is None:
                out[key] = ""
            elif isinstance(value, float):
                out[key] = f"{value:.9g}"
            else:
                out[key] = value
        rows.append(out)
    return rows


def write_outputs(
    report: dict[str, Any],
    out_dir: Path,
    *,
    json_path: Path | None = None,
    csv_path: Path | None = None,
    markdown_path: Path | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_path or out_dir / "ram10-report.json"
    csv_path = csv_path or out_dir / "ram10-per-request.csv"
    markdown_path = markdown_path or out_dir / "ram10-report.md"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in report.items() if key != "per_request_rows"}
    _write_json(Path(json_path), payload)
    rows = _csv_rows(report)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write(
                "arm,replicate,tag,cell,request_index,success,prompt_tokens,"
                "output_tokens,ttft_s,latency_s,decode_s,decode_tokens,"
                "prefill_tok_s,decode_tok_s,itl_n,itl_expected,"
                "itl_missing_tokens,itl_mean_ms,itl_p50_ms,itl_p99_ms,error\n"
            )
    Path(markdown_path).write_text(markdown_report(report), encoding="utf-8")
    return {
        "json": Path(json_path),
        "csv": Path(csv_path),
        "markdown": Path(markdown_path),
    }


def _fixture_result(
    *,
    completed: int,
    total_input: int,
    total_output: int,
    duration: float,
    output_throughput: float,
    mean_ttft: float,
    p50_ttft: float,
    p99_ttft: float,
    mean_tpot: float,
    mean_itl: float,
    p50_itl: float,
    p99_itl: float,
    input_lens: list[int],
    output_lens: list[int],
    ttfts: list[float],
    latencies: list[float],
    itls: list[list[float]],
) -> dict[str, Any]:
    return {
        "completed": completed,
        "failed": 0,
        "num_prompts": len(input_lens),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "duration": duration,
        "output_throughput": output_throughput,
        "max_concurrency": 1,
        "mean_ttft_ms": mean_ttft,
        "median_ttft_ms": p50_ttft,
        "p99_ttft_ms": p99_ttft,
        "mean_tpot_ms": mean_tpot,
        "mean_itl_ms": mean_itl,
        "median_itl_ms": p50_itl,
        "p99_itl_ms": p99_itl,
        "input_lens": input_lens,
        "output_lens": output_lens,
        "ttfts": ttfts,
        "latencies": latencies,
        "itls": itls,
        "errors": [None] * len(input_lens),
        "start_times": [0.0] * len(input_lens),
    }


def _selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="ram10-report-") as tmp:
        root = Path(tmp) / "runs"
        out_dir = Path(tmp) / "out"
        trace_cell = root / "trace" / "cells" / "bench-1024x512-c1"
        trace_cell.mkdir(parents=True)
        _write_json(trace_cell / "bench.json", {"completed": 1})
        _write_json(trace_cell / "cell.json", {"arm": "D", "cell": "trace"})
        _write_json(root / "trace" / "metadata.json", {"arm": "D"})

        def write_cache_memory(run: Path, sizes: list[int]) -> None:
            _write_json(
                run / "cache-memory.json",
                {
                    "results": [
                        {
                            "kv_cache_storage_bytes": size,
                            "kv_cache_views": 2,
                            "kv_cache_storages": 1,
                            "kv_cache_devices": ["cuda:0"],
                        }
                        for size in sizes
                    ]
                },
            )

        def write_run(arm: str, slots: int, kv_gib: list[float], tokens: int):
            run = root / f"ram10-{arm.lower()}-r1"
            cell = run / "cells" / "bench-1024x512-c1"
            cell.mkdir(parents=True)
            _write_json(
                run / "metadata.json",
                {
                    "arm": arm,
                    "arm_name": ARM_NAMES[arm],
                    "replicate": "1",
                    "tag": run.name,
                    "slots": slots,
                    "local_experts": 128,
                    "dummy_slot": slots not in (0, 128),
                    "device_cache_rows": (
                        None if slots == 0 else slots + (1 if slots < 128 else 0)
                    ),
                    "rows_saved_vs_128": (
                        0 if slots in (0, 128) else 128 - (slots + 1)
                    ),
                    "resident_fraction": (None if slots == 0 else slots / 128.0),
                    "w4a8_prefill": slots == 0,
                    "deterministic_dense": True,
                    "mono_knobs": (
                        None
                        if slots == 0
                        else {
                            "expert_cache_slots": slots,
                            "expert_cache_policy": "lfu_lru",
                            "expert_cache_decay": 128,
                        }
                    ),
                    "tree_sha": "fixture",
                    "status": "done",
                    "exit_code": 0,
                },
            )
            (run / "serve.log").write_text(
                "\n".join(
                    [
                        "[00:00:01] READY in 10s served=flash-next",
                        *[
                            f"Available KV cache memory: {gib:.2f} GiB"
                            for gib in kv_gib
                        ],
                        f"GPU KV cache size: {tokens:,} tokens, "
                        "Maximum concurrency for 131,072 tokens per request: 0.01x",
                        "RDNA2 expert cache layer.0: 128 logical experts, "
                        f"{slots} GPU slots; device 0.250 GiB, host 0.500 GiB",
                        "Actual usage is 20.32 GiB for consumed memory "
                        "(weights + non-torch), 3.66 GiB for peak activation, "
                        "and 1.57 GiB for CUDAGraph memory. Replace "
                        "gpu_memory_utilization config with `--kv-cache-memory=1` "
                        "(3.60 GiB) to fit into requested memory, or "
                        "`--kv-cache-memory=2` (4.18 GiB) to fully utilize gpu "
                        "memory. Current kv cache memory in use is 4.18 GiB.",
                        "CUDA graph pool memory: 1.57 GiB (actual), "
                        "2.44 GiB (estimated), difference: 0.87 GiB (55.4%).",
                        "Preemptions: 0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            return run, cell

        baseline_run, baseline_cell = write_run("R", 0, [10.0, 9.5], 1234)
        resident_exact_min = 4_831_838_208
        write_cache_memory(baseline_run, [resident_exact_min, 5_368_709_120])
        baseline = _fixture_result(
            completed=2,
            total_input=400,
            total_output=4,
            duration=4.0,
            output_throughput=1.0,
            mean_ttft=1000.0,
            p50_ttft=1000.0,
            p99_ttft=1500.0,
            mean_tpot=750.0,
            mean_itl=750.0,
            p50_itl=750.0,
            p99_itl=1000.0,
            input_lens=[100, 300],
            output_lens=[3, 1],
            ttfts=[0.5, 1.5],
            latencies=[2.0, 2.2],
            itls=[[0.75, 0.75], []],
        )
        _write_json(baseline_cell / "bench.json", baseline)
        _write_json(baseline_cell / "warmup.json", {"completed": 1, "failed": 0})
        _write_json(
            baseline_cell / "cell.json",
            {
                "arm": "R",
                "replicate": "1",
                "cell": "bench-1024x512-c1",
                "bench_rc": 0,
                "warmup_rc": 0,
                "reset_before_warmup_ok": True,
                "reset_after_warmup_ok": True,
                "evictions": "ok",
                "preemptions_in_window": 0,
                "faults_in_window": 0,
                "server_alive_after": True,
                "result_present": True,
                "warmup_present": True,
            },
        )
        baseline_c8 = baseline_run / "cells" / "bench-1024x512-c8"
        baseline_c8.mkdir(parents=True)
        _write_json(baseline_c8 / "bench.json", {**baseline, "num_prompts": 2})
        _write_json(baseline_c8 / "warmup.json", {"completed": 1, "failed": 0})
        _write_json(
            baseline_c8 / "cell.json",
            {
                "arm": "R",
                "replicate": "1",
                "cell": "bench-1024x512-c8",
                "bench_rc": 0,
                "warmup_rc": 0,
                "reset_before_warmup_ok": True,
                "reset_after_warmup_ok": True,
                "evictions": "ok",
                "preemptions_in_window": 0,
                "faults_in_window": 0,
                "server_alive_after": True,
                "result_present": True,
                "warmup_present": True,
            },
        )

        demand_run, demand_cell = write_run("D", 115, [11.25, 11.25], 1500)
        demand_exact_min = 5_046_586_163
        write_cache_memory(demand_run, [demand_exact_min, demand_exact_min])
        demand = _fixture_result(
            completed=2,
            total_input=400,
            total_output=4,
            duration=1.0,
            output_throughput=4.0,
            mean_ttft=800.0,
            p50_ttft=800.0,
            p99_ttft=1200.0,
            mean_tpot=250.0,
            mean_itl=250.0,
            p50_itl=250.0,
            p99_itl=300.0,
            input_lens=[100, 300],
            output_lens=[3, 1],
            ttfts=[0.4, 1.2],
            latencies=[0.9, 1.4],
            itls=[[0.25, 0.25], []],
        )
        _write_json(demand_cell / "bench.json", demand)
        _write_json(demand_cell / "warmup.json", {"completed": 1, "failed": 0})
        cache_rank_before = {
            "expert_demands": 100,
            "expert_misses": 10,
            "copied_bytes": 1000,
            "committed_passes": 10,
            "native_copy_calls": 0,
            "batch_copy_calls": 0,
            "map_uploads": 10,
            "device_bytes": 4096,
            "host_bytes": 8192,
            "copy_modes": ["tensor"],
            "debug_timing": False,
        }
        cache_rank_after = dict(
            cache_rank_before,
            expert_demands=150,
            expert_misses=20,
            copied_bytes=2000,
            committed_passes=15,
            map_uploads=15,
        )
        _write_json(
            demand_cell / "cache-before.json",
            {"results": [cache_rank_before, cache_rank_before]},
        )
        _write_json(
            demand_cell / "cache-after.json",
            {"results": [cache_rank_after, cache_rank_after]},
        )
        _write_json(
            demand_cell / "cell.json",
            {
                "arm": "D",
                "replicate": "1",
                "cell": "bench-1024x512-c1",
                "bench_rc": 0,
                "warmup_rc": 0,
                "reset_before_warmup_ok": True,
                "reset_after_warmup_ok": True,
                "evictions": "ok",
                "preemptions_in_window": 0,
                "faults_in_window": 0,
                "server_alive_after": True,
                "result_present": True,
                "warmup_present": True,
            },
        )

        invalid_cell = demand_run / "cells" / "bench-1024x512-c8"
        invalid_cell.mkdir()
        _write_json(invalid_cell / "bench.json", {**demand, "num_prompts": 4})
        _write_json(invalid_cell / "warmup.json", {"completed": 1, "failed": 0})
        _write_json(
            invalid_cell / "cell.json",
            {
                "arm": "D",
                "replicate": "1",
                "cell": "bench-1024x512-c8",
                "bench_rc": 0,
                "warmup_rc": 0,
                "reset_before_warmup_ok": True,
                "reset_after_warmup_ok": True,
                "evictions": "INVALID",
                "preemptions_in_window": 0,
                "faults_in_window": 0,
                "server_alive_after": True,
                "result_present": True,
                "warmup_present": True,
            },
        )

        report = build_report(root, baseline_arm="R")
        cells = {(c["arm"], c["cell"]): c for c in report["cells"]}
        resident = cells[("R", "bench-1024x512-c1")]
        candidate = cells[("D", "bench-1024x512-c1")]
        broken = cells[("D", "bench-1024x512-c8")]
        demand_run_record = next(r for r in report["runs"] if r["arm"] == "D")
        checks = [
            ("resident prefill req", resident["prefill_req_tps"], 200.0),
            ("resident decode req", resident["decode_req_tps"], 4.0 / 3.0),
            ("resident decode-only", resident["decode_only_tps"], 4.0 / 3.0),
            (
                "resident prefill agg",
                resident["prefill_agg_bench_tps"],
                200.0 / 1.5,
            ),
            ("resident prefill span", resident["prefill_span_tps"], 400.0 / 1.5),
            ("candidate prefill req", candidate["prefill_req_tps"], 250.0),
            ("candidate decode req", candidate["decode_req_tps"], 4.0),
            ("candidate decode-only", candidate["decode_only_tps"], 4.0),
            ("candidate out e2e", candidate["output_throughput_e2e"], 4.0),
            (
                "candidate prefill agg",
                candidate["prefill_agg_bench_tps"],
                200.0 / 1.2,
            ),
            ("candidate prefill span", candidate["prefill_span_tps"], 400.0 / 1.2),
            (
                "kv rank0 min",
                demand_run_record["serve"]["kv_available_gib_min"],
                11.25,
            ),
            (
                "exact kv min bytes",
                demand_run_record["cache_memory"]["storage_bytes_min"],
                demand_exact_min,
            ),
            (
                "exact kv sum bytes",
                demand_run_record["cache_memory"]["storage_bytes_sum"],
                2 * demand_exact_min,
            ),
            (
                "exact kv ranks",
                demand_run_record["cache_memory"]["ranks"],
                2,
            ),
            (
                "cell exact kv min bytes",
                candidate["kv_tp_min_storage_bytes"],
                demand_exact_min,
            ),
            (
                "cell exact kv sum bytes",
                candidate["kv_tp_sum_storage_bytes"],
                2 * demand_exact_min,
            ),
            ("kv tokens", demand_run_record["serve"]["kv_size_tokens"], 1500),
            (
                "kv host bytes are separate",
                demand_run_record["serve"]["expert_cache_host_gib_sum"],
                0.5,
            ),
            (
                "worker consumed actual",
                demand_run_record["serve"]["worker_consumed_gib_min"],
                20.32,
            ),
            (
                "worker activation actual",
                demand_run_record["serve"]["worker_activation_gib_min"],
                3.66,
            ),
            (
                "worker graph actual",
                demand_run_record["serve"]["worker_graph_gib_min"],
                1.57,
            ),
            (
                "graph estimate separate",
                demand_run_record["serve"]["graph_pool_estimated_gib_max"],
                2.44,
            ),
            ("cache demands", candidate["cache_demands"], 100),
            ("cache misses", candidate["cache_misses"], 20),
            ("cache miss rate", candidate["cache_miss_rate_pct"], 20.0),
            ("cache copied bytes", candidate["cache_copied_bytes"], 2000),
            ("cache device bytes", candidate["cache_device_bytes"], 8192),
            ("cache host bytes", candidate["cache_host_bytes"], 16384),
            ("cache ranks", candidate["cache_ranks"], 2),
        ]
        for label, got, want in checks:
            if got is None or abs(float(got) - float(want)) > 1e-9:
                print(f"selftest FAIL: {label}: got {got!r}, want {want!r}")
                return 1
        if (
            resident["kv_tp_min_source"] != "worker_rpc_exact"
            or candidate["kv_tp_min_source"] != "worker_rpc_exact"
        ):
            print("selftest FAIL: exact KV source was not preferred")
            return 1
        preempt_status = _read_json(demand_cell / "cell.json") or {}
        preempt_status["preemptions_in_window"] = 2
        preempt_row = compute_cell(
            demand,
            status=preempt_status,
            warmup={"completed": 1, "failed": 0},
            cell="bench-1024x512-c1",
            arm="D",
            replicate="1",
            cache_before={"results": [cache_rank_before, cache_rank_before]},
            cache_after={"results": [cache_rank_after, cache_rank_after]},
        )
        if not preempt_row["valid"] or not preempt_row["capacity_limited"]:
            print("selftest FAIL: preemption window was hidden or invalidated")
            return 1
        if "preemptions_observed" not in preempt_row["warnings"]:
            print("selftest FAIL: preemption warning missing")
            return 1
        if broken["valid"]:
            print("selftest FAIL: KFD-evicted cell was not marked invalid")
            return 1
        if "kfd_evictions" not in broken["invalid_reasons"]:
            print("selftest FAIL: KFD reason missing")
            return 1
        broken_comparison = next(
            c
            for c in report["comparisons"]
            if c["arm"] == "D" and c["cell"] == "bench-1024x512-c8"
        )
        if broken_comparison["status"] != "request_mismatch":
            print("selftest FAIL: different request counts were compared")
            return 1
        comparison = next(
            c
            for c in report["comparisons"]
            if c["arm"] == "D" and c["cell"] == "bench-1024x512-c1"
        )
        expected = {
            "prefill_req_tps": 25.0,
            "prefill_agg_bench_tps": 25.0,
            "prefill_span_tps": 25.0,
            "ttft_p50_ms": -20.0,
            "ttft_p99_ms": -20.0,
            "decode_req_tps": 200.0,
            "decode_only_tps": 200.0,
            "itl_p50_ms": (250.0 / 750.0 - 1.0) * 100.0,
            "itl_p99_ms": (300.0 / 1000.0 - 1.0) * 100.0,
            "output_throughput_e2e": 300.0,
            "kv_tp_min_gib": (demand_exact_min / resident_exact_min - 1.0) * 100.0,
            "kv_size_tokens": (1500.0 / 1234.0 - 1.0) * 100.0,
        }
        for key, want in expected.items():
            got = comparison["metrics"][key]["pct"]
            if got is None or abs(got - want) > 1e-9:
                print(f"selftest FAIL: delta {key}: got {got!r}, want {want!r}")
                return 1
        paths = write_outputs(report, out_dir)
        with open(paths["csv"], encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 8:
            print(f"selftest FAIL: CSV rows {len(rows)} != 8")
            return 1
        candidate_rows = [
            r for r in rows if r["arm"] == "D" and r["cell"] == "bench-1024x512-c1"
        ]
        if abs(float(candidate_rows[0]["decode_tok_s"]) - 4.0) > 1e-9:
            print("selftest FAIL: CSV decode rate")
            return 1
        if candidate_rows[1]["decode_tok_s"] != "":
            print("selftest FAIL: one-token request decode should be blank")
            return 1
        finalized = finalize_cell_dir(demand_cell)
        if not finalized["valid"]:
            print("selftest FAIL: finalize rejected valid cell")
            return 1
        status = _read_json(demand_cell / "cell.json")
        if not status.get("valid"):
            print("selftest FAIL: finalize did not persist validity")
            return 1
        finalized_broken = finalize_cell_dir(invalid_cell)
        if finalized_broken["valid"]:
            print("selftest FAIL: finalize accepted KFD-evicted cell")
            return 1
        markdown = paths["markdown"].read_text(encoding="utf-8")
        for phrase in (
            "decode-only",
            "end-to-end",
            "Baseline-relative",
            "Worker memory",
            "KV allocation (exact worker storage)",
            "TP-min bytes",
            "capacity-limited",
        ):
            if phrase not in markdown:
                print(f"selftest FAIL: Markdown missing {phrase!r}")
                return 1
    print("selftest PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", nargs="?", help="study run root")
    parser.add_argument(
        "--run-root", "--root", dest="run_root_opt", help="study run root"
    )
    parser.add_argument(
        "--out", "--out-dir", help="output directory (default: run root)"
    )
    parser.add_argument("--baseline", default="R", help="baseline arm (default R)")
    parser.add_argument("--replicate", help="only this replicate id")
    parser.add_argument("--arms", help="comma-separated arms to include")
    parser.add_argument("--json", dest="json_path", help="report JSON path")
    parser.add_argument("--csv", dest="csv_path", help="per-request CSV path")
    parser.add_argument("--markdown", dest="markdown_path", help="Markdown report path")
    parser.add_argument(
        "--finalize-cell",
        help="recompute one cell dir's cell.json and exit 0 valid / 1 invalid",
    )
    parser.add_argument(
        "--selftest", action="store_true", help="run the deterministic fixture"
    )
    parser.add_argument("--quiet", action="store_true", help="print only output paths")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.finalize_cell:
        row = finalize_cell_dir(Path(args.finalize_cell))
        print(
            json.dumps(
                {
                    "cell": row["cell"],
                    "valid": row["valid"],
                    "invalid_reasons": row["invalid_reasons"],
                    "warnings": row["warnings"],
                    "decode_req_tps": row["decode_req_tps"],
                    "decode_only_tps": row["decode_only_tps"],
                },
                sort_keys=True,
            )
        )
        return 0 if row["valid"] else 1

    run_root = args.run_root_opt or args.run_root
    if not run_root:
        run_root = os.environ.get("RAM10_RUN_ROOT") or str(
            Path.home() / "w4a8_runs" / "ram10-study"
        )
    root = Path(run_root)
    if not root.exists():
        print(f"ram10_report: run root does not exist: {root}", file=sys.stderr)
        return 2
    arms = None
    if args.arms:
        arms = []
        for item in args.arms.split(","):
            arm = normalize_arm(item)
            if arm is None:
                print(f"ram10_report: unknown arm {item!r}", file=sys.stderr)
                return 2
            arms.append(arm)
    baseline = normalize_arm(args.baseline)
    if baseline is None:
        print(f"ram10_report: unknown baseline arm {args.baseline!r}", file=sys.stderr)
        return 2
    report = build_report(
        root,
        baseline_arm=baseline,
        replicate=args.replicate,
        arms=arms,
    )
    out_dir = Path(args.out) if args.out else root
    paths = write_outputs(
        report,
        out_dir,
        json_path=Path(args.json_path) if args.json_path else None,
        csv_path=Path(args.csv_path) if args.csv_path else None,
        markdown_path=Path(args.markdown_path) if args.markdown_path else None,
    )
    if args.quiet:
        print(paths["json"])
        print(paths["csv"])
        print(paths["markdown"])
    else:
        print(f"report: {paths['json']}")
        print(f"per-request CSV: {paths['csv']}")
        print(f"markdown: {paths['markdown']}")
        print(
            f"runs={len(report['runs'])} cells={len(report['cells'])} "
            f"invalid={sum(1 for c in report['cells'] if not c['valid'])}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
