#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report the TP4 RAM-backed expert-cache study from run artifacts.

Reads ROOT/<arm>-r<rep>/{metadata,serve,summary,cache-memory} and
ROOT/<arm>-r<rep>/cells/<cell>/{bench,warmup}.json, cell.json, cache-*.json
and metrics-*.txt.  Writes ram10-report.json, ram10-report.md and
ram10-per-request.csv.  Missing measurements stay null / n/a / blank.
`--finalize-cell` recomputes one cell.json and exits 0 valid / 1 invalid.
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
CELL_RE = re.compile(r"bench-(\d+)x(\d+)-c(\d+)")
KV_RE = re.compile(r"GPU KV cache size:\s*([0-9][0-9,]*)\s*tokens")
MEM_RE = re.compile(
    r"Actual usage is ([0-9]+(?:\.[0-9]+)?) GiB for consumed memory"
    r".*?([0-9]+(?:\.[0-9]+)?) GiB for peak activation"
    r".*?([0-9]+(?:\.[0-9]+)?) GiB for CUDAGraph memory",
    re.S,
)
POOL_RE = re.compile(
    r"(?:CUDA|ROCm) graph pool memory:\s*([0-9]+(?:\.[0-9]+)?) GiB \(actual\),"
    r"\s*([0-9]+(?:\.[0-9]+)?) GiB \(estimated\)"
)
PPL_RE = re.compile(r"\bPPL\s+([0-9]+(?:\.[0-9]+)?)")
FAULT_RE = re.compile(
    r"Memory Fault|HSA_STATUS_ERROR|page fault|Segmentation fault|core dumped"
    r"|GPU Hang|HW Exception"
)
PREEMPT_RE = re.compile(
    r"^vllm:num_preemptions(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+]+)\s*$", re.M
)
MTP = {
    "mtp_accepted_tokens": "spec_decode_accepted_tokens",
    "mtp_num_drafts": "spec_decode_num_drafts",
    "mtp_draft_tokens": "spec_decode_draft_tokens",
    "mtp_acceptance_rate": "spec_decode_acceptance_rate",
    "mtp_acceptance_length": "spec_decode_acceptance_length",
}
COMPARE = (
    "prefill_req_tps",
    "prefill_agg_bench_tps",
    "prefill_span_tps",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "decode_req_tps",
    "decode_only_tps",
    "decode_span_agg_tps",
    "itl_p50_ms",
    "itl_p99_ms",
    "output_throughput_e2e",
    "kv_tp_min_gib",
    "kv_size_tokens",
)
META = (
    "arm_name",
    "cells",
    "slots",
    "device_cache_rows",
    "rows_saved_vs_128",
    "model",
    "recipe",
    "gpus",
    "tp",
    "gmem",
    "kv",
    "maxlen",
    "maxbat",
    "seqs",
    "block_size",
    "source_sha",
    "tree_sha",
    "kernelSHA",
    "kernel_sha256",
    "w4a8_prefill",
    "w4a8_sdot4",
    "compile_mode",
    "cudagraph_mode",
    "breakable_cudagraph",
    "deterministic_dense",
    "mono_knobs",
)
FIELDS = (
    "cell",
    "arm",
    "replicate",
    "requested",
    "completed",
    "failed",
    "success_count",
    "prompt_tokens_mean",
    "input_tokens",
    "output_tokens",
    "duration_s",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "itl_mean_ms",
    "itl_p50_ms",
    "itl_p99_ms",
    "itl_complete",
    "prefill_req_tps",
    "prefill_req_tps_p50",
    "prefill_req_tps_p99",
    "prefill_agg_bench_tps",
    "prefill_span_tps",
    "decode_req_tps",
    "decode_req_tps_p50",
    "decode_req_tps_p99",
    "decode_only_tps",
    "decode_span_agg_tps",
    "output_throughput_e2e",
    "mtp_accepted_tokens",
    "mtp_num_drafts",
    "mtp_draft_tokens",
    "mtp_acceptance_rate",
    "mtp_acceptance_length",
    "preemptions",
    "capacity_limited",
    "faults",
    "evictions",
    "cache_demands",
    "cache_misses",
    "cache_miss_rate_pct",
    "cache_copied_bytes",
    "cache_device_bytes",
    "cache_host_bytes",
    "cache_ranks",
    "cache_copy_modes",
    "valid",
    "valid_for_ttft",
    "valid_for_decode",
    "invalid_reasons",
    "warnings",
)
FINAL = (
    "completed",
    "failed",
    "requested",
    "prefill_req_tps",
    "prefill_agg_bench_tps",
    "prefill_span_tps",
    "decode_req_tps",
    "decode_only_tps",
    "decode_span_agg_tps",
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
CSV_COLS = (
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
NOTES = [
    (
        "D/Z have 115 real slots plus one dummy: 116 physical rows, saving 12, "
        "not 13. F is the full-capacity cache control. The CPU bank mirrors all "
        "128 local experts, not only the 13 nonresident ones."
    ),
    (
        "R uses W4A8 prefill and compiled F&P; F/D/Z use W4A16 prefill and "
        "breakable F&P, compile NONE. R-to-offload measures deployment cost, "
        "not PCIe alone. No resident W4A16 control was measured."
    ),
    (
        "KV bytes are exact deduplicated rdna_cache_memory storage: TP-min "
        "limits capacity; sum covers all ranks. Token capacity is the engine's "
        "GPU KV cache size. Missing RPC data stays null, never inferred from "
        "weight savings. Expert host/device bytes are separate."
    ),
    (
        "Logged peak activation includes estimated graph reserve: it is not "
        "isolated transient memory. Graph-pool actual/estimated values are "
        "reported separately; startup memory fields must not be summed blindly."
    ),
    (
        "Output e2e is total output/duration. Decode wtd is "
        "sum(out-1)/sum(latency-TTFT); decode span is "
        "sum(out-1)/(last completion-first first token), not pure GPU decode. "
        "Prefill req is mean prompt/mean TTFT; agg is bench_metrics' one-wave "
        "estimate. Span rates include queueing and overlapping other phases."
    ),
    (
        "CSV rates use each request's prompt/TTFT and "
        "(out-1)/(latency-TTFT), never 1/median ITL. ITL measures delivery chunks "
        "under MTP; missing token intervals are not fabricated as zeros."
    ),
    (
        "Preemptions are cumulative metric deltas, excluding _created. Positive "
        "counts flag capacity limits; faults/KFD evictions invalidate. Baseline "
        "comparisons match cell, replicate, request and completed counts. "
        "Missing observations stay null/n/a/blank."
    ),
]

_SPEC = importlib.util.spec_from_file_location(
    "ram10_bench_metrics",
    Path(__file__).resolve().parent.parent / "port_v031" / "bench_metrics.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load port_v031/bench_metrics.py")
_BENCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCH)


def _load(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _dump(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _f(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _i(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _b(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return (
        {
            "1": True,
            "true": True,
            "yes": True,
            "ok": True,
            "0": False,
            "false": False,
            "no": False,
        }.get(value.strip().lower())
        if isinstance(value, str)
        else None
    )


def _pos(value: Any) -> float | None:
    number = _f(value)
    return number if number is not None and number > 0 else None


def _arm(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return (
        text.upper()
        if text.upper() in ARM_ORDER
        else next(
            (arm for arm, name in ARM_NAMES.items() if text.lower() == name), None
        )
    )


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    rank = (len(xs) - 1) * p / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    return (
        xs[0]
        if len(xs) == 1
        else xs[low]
        if low == high
        else xs[low] + (xs[high] - xs[low]) * (rank - low)
    )


def parse_serve(path: Path, summary_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "present": False,
        "boot_failed": False,
        "boot_faults": None,
        "kv_size_tokens": None,
        "ppl": None,
        "consumed_gib_min": None,
        "consumed_gib_max": None,
        "activation_gib_min": None,
        "activation_gib_max": None,
        "graph_gib_min": None,
        "graph_gib_max": None,
        "graph_pool_actual_gib_min": None,
        "graph_pool_actual_gib_max": None,
        "graph_pool_estimated_gib_min": None,
        "graph_pool_estimated_gib_max": None,
    }
    text = _text(path)
    if text:
        out["present"], out["boot_failed"] = True, "BOOT FAILED" in text
        out["boot_faults"] = sum(
            bool(FAULT_RE.search(line)) for line in text.splitlines()
        )
        tokens = KV_RE.findall(text)
        if tokens:
            out["kv_size_tokens"] = min(int(token.replace(",", "")) for token in tokens)
        for key, index in (("consumed", 0), ("activation", 1), ("graph", 2)):
            values = [float(row[index]) for row in MEM_RE.findall(text)]
            out[f"{key}_gib_min"], out[f"{key}_gib_max"] = (
                (min(values), max(values)) if values else (None, None)
            )
        for key, index in (("actual", 0), ("estimated", 1)):
            values = [float(row[index]) for row in POOL_RE.findall(text)]
            out[f"graph_pool_{key}_gib_min"], out[f"graph_pool_{key}_gib_max"] = (
                (min(values), max(values)) if values else (None, None)
            )
    match = PPL_RE.search(_text(summary_path))
    if match:
        out["ppl"] = float(match.group(1))
    return out


def parse_memory(data: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "present": False,
        "ranks": None,
        "bytes_per_rank": [],
        "storage_bytes_min": None,
        "storage_bytes_sum": None,
        "storage_gib_min": None,
        "storage_gib_sum": None,
        "views": None,
        "storages": None,
        "devices": [],
    }
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or not results:
        return out
    sizes: list[int] = []
    views: list[int | None] = []
    storages: list[int | None] = []
    devices: set[str] = set()
    for rank in results:
        if not isinstance(rank, dict):
            return out
        size = _i(rank.get("kv_cache_storage_bytes"))
        if size is None or size <= 0:
            return out
        sizes.append(size)
        views.append(_i(rank.get("kv_cache_views")))
        storages.append(_i(rank.get("kv_cache_storages")))
        devices.update(str(device) for device in rank.get("kv_cache_devices") or [])
    out.update(
        present=True,
        ranks=len(sizes),
        bytes_per_rank=sizes,
        storage_bytes_min=min(sizes),
        storage_bytes_sum=sum(sizes),
        storage_gib_min=min(sizes) / 2**30,
        storage_gib_sum=sum(sizes) / 2**30,
        views=sum(views) if None not in views else None,
        storages=sum(storages) if None not in storages else None,
        devices=sorted(devices),
    )
    return out


def parse_counters(before: Any, after: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "present": False,
        "ranks": None,
        "demands": None,
        "misses": None,
        "miss_rate_pct": None,
        "copied_bytes": None,
        "device_bytes": None,
        "host_bytes": None,
        "native_copy_calls": None,
        "copy_modes": [],
        "debug_timing": None,
    }
    old = before.get("results") if isinstance(before, dict) else None
    new = after.get("results") if isinstance(after, dict) else None
    if (
        not isinstance(old, list)
        or not isinstance(new, list)
        or not old
        or len(old) != len(new)
    ):
        return out
    total = dict.fromkeys(("expert_demands", "expert_misses", "copied_bytes"), 0)
    native: list[int | None] = []
    device: list[int | None] = []
    host: list[int | None] = []
    debug: list[Any] = []
    modes: set[str] = set()
    for old_rank, new_rank in zip(old, new):
        if not isinstance(old_rank, dict) or not isinstance(new_rank, dict):
            return out
        for key in total:
            start, end = _i(old_rank.get(key)), _i(new_rank.get(key))
            if start is None or end is None:
                return out
            total[key] += end - start
        native.append(_i(new_rank.get("native_copy_calls")))
        device.append(_i(new_rank.get("device_bytes")))
        host.append(_i(new_rank.get("host_bytes")))
        debug.append(new_rank.get("debug_timing"))
        modes.update(str(mode) for mode in new_rank.get("copy_modes") or [])
    if any(value < 0 for value in total.values()):
        return out
    demands = total["expert_demands"]
    out.update(
        present=True,
        ranks=len(new),
        demands=demands,
        misses=total["expert_misses"],
        copied_bytes=total["copied_bytes"],
        miss_rate_pct=total["expert_misses"] / demands * 100.0 if demands > 0 else None,
        device_bytes=sum(device) if None not in device else None,
        host_bytes=sum(host) if None not in host else None,
        native_copy_calls=sum(native) if None not in native else None,
        copy_modes=sorted(modes),
        debug_timing=True
        if True in debug
        else (False if debug and all(value is False for value in debug) else None),
    )
    return out


def _pre_sum(path: Path) -> int | None:
    values = [float(value) for value in PREEMPT_RE.findall(_text(path))]
    return round(sum(values)) if values else None


def preemption_delta(cell_dir: Path) -> int | None:
    before, after = (
        _pre_sum(cell_dir / "metrics-before.txt"),
        _pre_sum(cell_dir / "metrics-after.txt"),
    )
    return (
        after - before
        if before is not None and after is not None and after >= before
        else None
    )


def request_rows(result: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    lists = [
        result.get(key)
        for key in ("input_lens", "output_lens", "ttfts", "latencies", "itls", "errors")
    ]
    if (
        not all(isinstance(value, list) for value in lists)
        or len({len(value) for value in lists}) != 1
    ):
        return [], False
    starts = result.get("start_times")
    if not isinstance(starts, list) or len(starts) != len(lists[0]):
        starts = None
    rows: list[dict[str, Any]] = []
    for index, (prompt, output, ttft, latency, itls, error) in enumerate(zip(*lists)):
        prompt, output, ttft, latency = _i(prompt), _i(output), _f(ttft), _f(latency)
        chunks = (
            [
                value * 1000.0
                for value in (_f(raw) for raw in itls)
                if value is not None and value >= 0
            ]
            if isinstance(itls, list)
            else []
        )
        start = _f(starts[index]) if starts is not None else None
        success = (
            not error
            and output is not None
            and output > 0
            and latency is not None
            and latency > 0
        )
        timing = success and ttft is not None and ttft > 0 and latency >= ttft
        tokens = output - 1 if output is not None and output > 1 else None
        seconds = (
            latency - ttft if tokens is not None and timing and latency > ttft else None
        )
        rows.append(
            {
                "request_index": index,
                "success": success,
                "timing_ok": timing,
                "prompt_tokens": prompt,
                "output_tokens": output,
                "ttft_s": ttft,
                "latency_s": latency,
                "start_s": start,
                "completion_s": start + latency
                if start is not None and latency is not None
                else None,
                "decode_s": seconds,
                "decode_tokens": tokens,
                "prefill_tok_s": prompt / ttft
                if prompt is not None and prompt > 0 and ttft is not None and ttft > 0
                else None,
                "decode_tok_s": tokens / seconds
                if tokens is not None and seconds
                else None,
                "itl_n": len(chunks) if isinstance(itls, list) else None,
                "itl_expected": max(output - 1, 0) if output is not None else None,
                "itl_missing_tokens": max(output - 1 - len(chunks), 0)
                if output is not None
                else None,
                "itl_mean_ms": sum(chunks) / len(chunks) if chunks else None,
                "itl_p50_ms": _pct(chunks, 50),
                "itl_p99_ms": _pct(chunks, 99),
                "error": None if not error else str(error),
            }
        )
    return rows, True


def analyse_cell(
    cell_dir: Path, cell: str, arm: str | None, replicate: str | None
) -> dict[str, Any]:
    cell_dir = Path(cell_dir)
    status, result, warmup = (
        _load(cell_dir / "cell.json") or {},
        _load(cell_dir / "bench.json"),
        _load(cell_dir / "warmup.json"),
    )
    arm = arm or _arm(status.get("arm"))
    replicate = (
        str(status.get("replicate") or "unknown")
        if replicate in (None, "", "unknown")
        else str(replicate)
    )
    row: dict[str, Any] = {key: None for key in FIELDS}
    row.update(
        cell=str(cell),
        arm=arm,
        replicate=replicate,
        invalid_reasons=[],
        warnings=[],
        capacity_limited=False,
        evictions=status.get("evictions"),
    )
    inv, warn = row["invalid_reasons"], row["warnings"]
    if not status:
        inv.append("cell_status_missing")
    for key, reason in (
        ("reset_before_warmup_ok", "prefix_reset_failed"),
        ("reset_after_warmup_ok", "prefix_reset_failed"),
        ("server_alive_after", "server_died"),
        ("result_present", "no_result"),
    ):
        if _b(status.get(key)) is False:
            inv.append(reason)
    if (
        _b(status.get("warmup_present")) is False
        and _b(status.get("warmup_enabled")) is not False
    ):
        inv.append("warmup_missing")
    if isinstance(warmup, dict):
        if (
            _i(warmup.get("failed")) not in (None, 0)
            or _i(warmup.get("completed")) == 0
        ):
            inv.append("warmup_failed")
    elif _b(status.get("warmup_enabled")) is not False:
        inv.append("warmup_missing")
    if _i(status.get("bench_rc")) not in (None, 0):
        inv.append("bench_failed")
    if _i(status.get("warmup_rc")) not in (None, 0):
        inv.append("warmup_failed")
    evictions = status.get("evictions")
    if evictions == "INVALID":
        inv.append("kfd_evictions")
    elif evictions in (None, "", "unavailable"):
        inv.append("kfd_evictions_unavailable")
    faults = _i(status.get("faults_in_window"))
    row["faults"] = faults
    if faults is None:
        inv.append("faults_missing")
    elif faults > 0:
        inv.append("faults")
    reported, raw = _i(status.get("preemptions_in_window")), preemption_delta(cell_dir)
    if (
        raw is None
        and (cell_dir / "metrics-before.txt").is_file()
        and (cell_dir / "metrics-after.txt").is_file()
    ):
        inv.append("preemptions_unparsable")
    preemptions = raw if raw is not None else reported
    if raw is not None and reported is not None and reported != raw:
        warn.append("preemptions_mismatch")
    row["preemptions"] = preemptions
    if preemptions is None:
        inv.append("preemptions_missing")
    elif preemptions > 0:
        row["capacity_limited"] = True
        warn.append("preemptions_observed")
    inv.extend(str(reason) for reason in status.get("invalid_reasons") or [])
    warn.extend(str(reason) for reason in status.get("warnings") or [])
    cache = parse_counters(
        _load(cell_dir / "cache-before.json"), _load(cell_dir / "cache-after.json")
    )
    row.update(
        {
            f"cache_{key}": cache[key]
            for key in (
                "demands",
                "misses",
                "miss_rate_pct",
                "copied_bytes",
                "device_bytes",
                "host_bytes",
                "ranks",
            )
        }
    )
    row["cache_copy_modes"] = cache["copy_modes"]
    if arm in ("F", "D", "Z"):
        if not cache["present"]:
            warn.append("cache_counters_missing")
        elif cache["native_copy_calls"] or "native" in cache["copy_modes"]:
            inv.append("native_copy")
        elif cache["debug_timing"] is True:
            inv.append("debug_timing_enabled")
    if not isinstance(result, dict):
        inv.append("no_result")
        row.update(
            valid=False,
            valid_for_ttft=False,
            valid_for_decode=False,
            invalid_reasons=list(dict.fromkeys(inv)),
            warnings=list(dict.fromkeys(warn)),
        )
        return row
    completed, failed, requested = (
        _i(result.get("completed")),
        _i(result.get("failed")),
        _i(result.get("num_prompts")),
    )
    total_in, total_out, duration = (
        _f(result.get("total_input_tokens")),
        _f(result.get("total_output_tokens")),
        _f(result.get("duration")),
    )
    rows, details = request_rows(result)
    success = [request for request in rows if request["success"]]
    count = len(success) if details else None
    if requested is None and details:
        requested = len(rows)
    if completed is None:
        completed = count
    if details and completed is not None and count is not None and completed != count:
        warn.append("completed_count_mismatch")
    if (
        details
        and None not in (requested, completed, failed)
        and completed + failed != requested
    ):
        warn.append("request_count_mismatch")
    if completed is None:
        inv.append("completed_missing")
    elif completed <= 0:
        inv.append("no_completed")
    if failed is None:
        inv.append("failed_missing")
    elif failed > 0:
        inv.append("failed_requests")
    if requested is None:
        inv.append("requested_missing")
    if not details:
        inv.append("missing_per_request_detail")
    if total_in is None:
        inv.append("input_tokens_missing")
    if total_out is None:
        inv.append("output_tokens_missing")
    if duration is None:
        inv.append("duration_missing")
    elif duration <= 0:
        inv.append("invalid_duration")
    if details and any(
        request["success"] and not request["timing_ok"] for request in rows
    ):
        inv.append("invalid_request_timing")
    if details and not any(request["decode_tok_s"] is not None for request in success):
        inv.append("no_decode_interval")
    bm: dict[str, Any] = {}
    if completed is not None and completed > 0:
        try:
            bm = _BENCH.metrics(result)
        except (KeyError, TypeError, ValueError, ArithmeticError):
            warn.append("bench_metrics_failed")
    prompt_mean = (
        total_in / completed
        if total_in is not None and completed and completed > 0
        else None
    )
    ttft_mean, ttft_p50, ttft_p99 = (
        _pos(bm.get("ttft_mean_ms")),
        _pos(bm.get("ttft_p50_ms")),
        _pos(bm.get("ttft_p99_ms")),
    )
    itl_mean, itl_p50, itl_p99 = (
        _pos(bm.get("itl_mean_ms")),
        _pos(bm.get("itl_p50_ms")),
        _pos(bm.get("itl_p99_ms")),
    )
    prefill_req, prefill_agg, decode_req = (
        _pos(bm.get("prefill_req_tps")),
        _pos(bm.get("prefill_agg_tps")),
        _pos(bm.get("decode_req_tps")),
    )
    e2e = _f(result.get("output_throughput"))
    if e2e is None and duration and total_out is not None:
        e2e = total_out / duration
    pre_rates = [
        request["prefill_tok_s"]
        for request in success
        if request["prefill_tok_s"] is not None
    ]
    dec_rates = [
        request["decode_tok_s"]
        for request in success
        if request["decode_tok_s"] is not None
    ]
    dec_pairs = [
        (request["decode_tokens"], request["decode_s"])
        for request in success
        if request["decode_tokens"] is not None and request["decode_s"] is not None
    ]
    dec_tokens, dec_seconds = (
        sum(pair[0] for pair in dec_pairs),
        sum(pair[1] for pair in dec_pairs),
    )
    decode_only = (
        dec_tokens / dec_seconds
        if dec_tokens and dec_seconds and dec_seconds > 0
        else None
    )
    pre_points = [
        (request["start_s"], request["start_s"] + request["ttft_s"])
        for request in success
        if request["start_s"] is not None
        and request["ttft_s"] is not None
        and request["ttft_s"] > 0
    ]
    prefill_span = None
    if pre_points and total_in is not None:
        span = max(end for _, end in pre_points) - min(start for start, _ in pre_points)
        prefill_span = total_in / span if span > 0 else None
    dec_points = [
        (
            request["start_s"] + request["ttft_s"],
            request["completion_s"],
            request["decode_tokens"] or 0,
        )
        for request in success
        if request["start_s"] is not None
        and request["ttft_s"] is not None
        and request["completion_s"] is not None
    ]
    decode_span = None
    if dec_points:
        first, last = (
            min(point[0] for point in dec_points),
            max(point[1] for point in dec_points),
        )
        tokens = sum(point[2] for point in dec_points)
        decode_span = tokens / (last - first) if last > first and tokens > 0 else None
    itl_complete = None
    if details:
        decode_rows = [
            request for request in success if (request["output_tokens"] or 0) > 1
        ]
        if decode_rows:
            itl_complete = all(
                request["itl_missing_tokens"] == 0 for request in decode_rows
            )
            if not itl_complete:
                warn.append("itl_partial")
    mtp = {
        key: _f(result.get(source)) for key, source in MTP.items() if source in result
    }
    if not mtp:
        warn.append("mtp_metrics_missing")
    row.update(
        requested=requested,
        completed=completed,
        failed=failed,
        success_count=count,
        prompt_tokens_mean=prompt_mean,
        input_tokens=total_in,
        output_tokens=total_out,
        duration_s=duration,
        ttft_mean_ms=ttft_mean,
        ttft_p50_ms=ttft_p50,
        ttft_p99_ms=ttft_p99,
        itl_mean_ms=itl_mean,
        itl_p50_ms=itl_p50,
        itl_p99_ms=itl_p99,
        itl_complete=itl_complete,
        prefill_req_tps=prefill_req,
        prefill_req_tps_p50=_pct(pre_rates, 50),
        prefill_req_tps_p99=_pct(pre_rates, 99),
        prefill_agg_bench_tps=prefill_agg,
        prefill_span_tps=prefill_span,
        decode_req_tps=decode_req,
        decode_req_tps_p50=_pct(dec_rates, 50),
        decode_req_tps_p99=_pct(dec_rates, 99),
        decode_only_tps=decode_only,
        decode_span_agg_tps=decode_span,
        output_throughput_e2e=e2e,
        **mtp,
    )
    row["valid"] = not inv
    row["valid_for_ttft"] = row["valid"] and ttft_p50 is not None
    row["valid_for_decode"] = row["valid"] and decode_only is not None
    row["invalid_reasons"] = list(dict.fromkeys(inv))
    row["warnings"] = list(dict.fromkeys(warn))
    row["_per_request"] = rows
    return row


def finalize_cell(cell_dir: Path) -> dict[str, Any]:
    cell_dir = Path(cell_dir)
    status = _load(cell_dir / "cell.json") or {}
    row = analyse_cell(
        cell_dir,
        str(status.get("cell") or cell_dir.name),
        _arm(status.get("arm")),
        status.get("replicate"),
    )
    status.update(
        valid=row["valid"],
        valid_for_ttft=row["valid_for_ttft"],
        valid_for_decode=row["valid_for_decode"],
        invalid_reasons=row["invalid_reasons"],
        warnings=row["warnings"],
        capacity_limited=row["capacity_limited"],
        decode_measured=row["decode_only_tps"] is not None,
        itl_complete=row["itl_complete"],
        metrics={key: row.get(key) for key in FINAL},
    )
    if row["preemptions"] is not None:
        status["preemptions_in_window"] = row["preemptions"]
    _dump(cell_dir / "cell.json", status)
    return row


def _cell_key(name: str) -> tuple[int, int, int, str]:
    match = CELL_RE.fullmatch(name)
    return (
        (int(match.group(1)), int(match.group(2)), int(match.group(3)), name)
        if match
        else (10**9, 10**9, 10**9, name)
    )


def _discover_runs(root: Path) -> list[Path]:
    if (root / "metadata.json").is_file() or (root / "cells").is_dir():
        return [root]
    found: set[Path] = set()
    for path in root.glob("*/metadata.json"):
        if "trace" not in path.relative_to(root).parts:
            found.add(path.parent)
    for path in root.glob("*/cells/*/bench.json"):
        if "trace" not in path.relative_to(root).parts:
            found.add(path.parents[2])
    return sorted(found)


def compare_cells(
    cells: list[dict[str, Any]], baseline_arm: str
) -> list[dict[str, Any]]:
    index: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for cell in cells:
        index.setdefault((cell["arm"], cell["replicate"], cell["cell"]), []).append(
            cell
        )
    baseline_names = {cell["cell"] for cell in cells if cell["arm"] == baseline_arm}
    comparisons = []
    for cell in cells:
        if cell["arm"] == baseline_arm:
            continue
        bases = index.get((baseline_arm, cell["replicate"], cell["cell"]), [])
        baseline = bases[0] if len(bases) == 1 else None
        status = "ok"
        if not bases:
            status = (
                "replicate_mismatch"
                if cell["cell"] in baseline_names
                else "missing_baseline"
            )
        elif baseline is None:
            status = "ambiguous_baseline"
        elif cell["requested"] is None or baseline["requested"] is None:
            status = "missing_request_count"
        elif cell["completed"] is None or baseline["completed"] is None:
            status = "missing_completed"
        elif cell["requested"] != baseline["requested"]:
            status = "request_mismatch"
        elif cell["completed"] != baseline["completed"]:
            status = "completion_mismatch"
        comparison: dict[str, Any] = {
            "cell": cell["cell"],
            "arm": cell["arm"],
            "replicate": cell["replicate"],
            "baseline_arm": baseline_arm,
            "baseline_replicate": baseline["replicate"] if baseline else None,
            "baseline_tag": baseline.get("tag") if baseline else None,
            "status": status,
            "metrics": {},
        }
        for key in COMPARE:
            candidate = cell.get(key)
            reference = baseline.get(key) if baseline else None
            metric = {
                "candidate": candidate,
                "baseline": reference,
                "delta": None,
                "pct": None,
                "status": status,
            }
            if status == "ok":
                assert baseline is not None
                if candidate is None or reference is None:
                    metric["status"] = "missing_metric"
                elif not cell["valid"] or not baseline["valid"]:
                    metric["status"] = "invalid_window"
                else:
                    # ITL compares observed delivery chunks under MTP, not
                    # fabricated per-token intervals. Token coverage may differ.
                    metric["delta"] = candidate - reference
                    if reference == 0:
                        metric["status"] = "baseline_zero"
                    else:
                        metric["pct"] = (candidate / reference - 1.0) * 100.0
            comparison["metrics"][key] = metric
        comparisons.append(comparison)
    return comparisons


def build_report(run_root: Path, baseline_arm: str = "R") -> dict[str, Any]:
    run_root = Path(run_root)
    baseline_arm = _arm(baseline_arm) or baseline_arm.upper()
    runs: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    per_request: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run_dir in _discover_runs(run_root):
        meta = _load(run_dir / "metadata.json") or {}
        serve = parse_serve(run_dir / "serve.log", run_dir / "summary.txt")
        memory = parse_memory(_load(run_dir / "cache-memory.json"))
        arm = _arm(meta.get("arm")) or _arm(meta.get("arm_name"))
        replicate, tag = (
            str(meta.get("replicate") or "unknown"),
            meta.get("tag") or run_dir.name,
        )
        run_cells: list[dict[str, Any]] = []
        for bench in sorted(
            run_dir.glob("cells/*/bench.json"),
            key=lambda path: _cell_key(path.parent.name),
        ):
            if "trace" in bench.relative_to(run_dir).parts:
                continue
            row = analyse_cell(bench.parent, bench.parent.name, arm, replicate)
            row.update(
                tag=tag,
                run_path=str(run_dir),
                kv_tp_min_bytes=memory["storage_bytes_min"],
                kv_tp_sum_bytes=memory["storage_bytes_sum"],
                kv_tp_min_gib=memory["storage_gib_min"],
                kv_tp_sum_gib=memory["storage_gib_sum"],
                kv_views=memory["views"],
                kv_storages=memory["storages"],
                kv_devices=memory["devices"],
                kv_size_tokens=serve["kv_size_tokens"],
            )
            for request in row.pop("_per_request", []):
                per_request.append(
                    {
                        "arm": row["arm"],
                        "replicate": row["replicate"],
                        "tag": tag,
                        "cell": bench.parent.name,
                        **request,
                    }
                )
            run_cells.append(row)
            cells.append(row)
        arm = arm or (run_cells[0]["arm"] if run_cells else None)
        replicate = (
            str(run_cells[0]["replicate"])
            if replicate == "unknown" and run_cells
            else replicate
        )
        expert = next(
            (
                cell
                for cell in reversed(run_cells)
                if cell["cache_device_bytes"] is not None
            ),
            None,
        )
        run: dict[str, Any] = {
            "arm": arm,
            "name": ARM_NAMES.get(arm or "", "unknown"),
            "replicate": replicate,
            "tag": tag,
            "path": str(run_dir),
            "status": meta.get("status"),
            "exit_code": _i(meta.get("exit_code")),
            "cell_count": len(run_cells),
            "expert_device_bytes": expert["cache_device_bytes"] if expert else None,
            "expert_host_bytes": expert["cache_host_bytes"] if expert else None,
            "serve": serve,
            "cache_memory": memory,
        }
        run.update({key: meta.get(key) for key in META})
        runs.append(run)
        if run["status"] != "done":
            warnings.append(f"{run_dir.name}: run status {run['status']!r}, not done")
        declared = set()
        for spec in meta.get("cells", []):
            input_len, output_len, concurrency, *_ = spec.split(":")
            declared.add(f"bench-{input_len}x{output_len}-c{concurrency}")
        run["missing_declared_cells"] = sorted(
            declared - {cell["cell"] for cell in run_cells}
        )
        if run["missing_declared_cells"]:
            warnings.append(
                f"{run_dir.name}: missing declared cells: "
                + ", ".join(run["missing_declared_cells"])
            )
        if not serve["present"]:
            warnings.append(f"{run_dir.name}: serve.log missing")
        elif serve["boot_failed"]:
            warnings.append(f"{run_dir.name}: BOOT FAILED in serve.log")
        if serve["boot_faults"]:
            warnings.append(
                f"{run_dir.name}: {serve['boot_faults']} boot fault marker(s)"
            )
        if serve["kv_size_tokens"] is None:
            warnings.append(f"{run_dir.name}: GPU KV cache size missing")
        if not memory["present"]:
            warnings.append(
                f"{run_dir.name}: exact rdna_cache_memory snapshot missing; "
                "KV capacity is null"
            )
        if not run_cells:
            warnings.append(f"{run_dir.name}: no cells/*/bench.json")
    runs.sort(
        key=lambda run: (
            ARM_ORDER.get(run["arm"], 9),
            str(run["replicate"]),
            run["path"],
        )
    )
    cells.sort(
        key=lambda cell: (
            ARM_ORDER.get(cell["arm"], 9),
            str(cell["replicate"]),
            _cell_key(cell["cell"]),
        )
    )
    per_request.sort(
        key=lambda row: (
            ARM_ORDER.get(row["arm"], 9),
            str(row["replicate"]),
            _cell_key(row["cell"]),
            row["request_index"],
        )
    )
    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "run_root": str(run_root),
        "baseline_arm": baseline_arm,
        "runs": runs,
        "cells": cells,
        "comparisons": compare_cells(cells, baseline_arm),
        "per_request_rows": per_request,
        "warnings": warnings,
        "notes": NOTES,
    }


def _fstr(value: Any, digits: int = 1) -> str:
    number = _f(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _istr(value: Any) -> str:
    number = _i(value)
    return "n/a" if number is None else str(number)


def _range(low: Any, high: Any, digits: int = 2) -> str:
    low, high = _f(low), _f(high)
    if low is None:
        return "n/a"
    return (
        f"{low:.{digits}f}"
        if high is None or abs(high - low) < 0.005
        else f"{low:.{digits}f}-{high:.{digits}f}"
    )


def _flag(value: Any) -> str:
    flag = _b(value)
    return "n/a" if flag is None else "1" if flag else "0"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def _cmp(metric: dict[str, Any] | None) -> str:
    if not metric:
        return "n/a"
    status = metric.get("status")
    if status in ("partial_itl", "invalid_window", "baseline_zero"):
        return {
            "partial_itl": "partial",
            "invalid_window": "invalid",
            "baseline_zero": "base0",
        }[status]
    number = _f(metric.get("pct"))
    return "n/a" if number is None else f"{number:+.1f}%"


def _cmp_group(comparison: dict[str, Any], keys: tuple[str, ...]) -> str:
    return " / ".join(_cmp(comparison["metrics"].get(key)) for key in keys)


def _capacity_row(run: dict[str, Any]) -> list[str]:
    memory, serve, dev, host = (
        run["cache_memory"],
        run["serve"],
        run.get("expert_device_bytes"),
        run.get("expert_host_bytes"),
    )
    return [
        f"{run['arm'] or '?'}/{run['replicate']}",
        str(run.get("tag") or Path(run["path"]).name),
        _istr(run.get("slots")),
        f"{_flag(run.get('w4a8_prefill'))}/{_flag(run.get('w4a8_sdot4'))}",
        f"{_istr(run.get('compile_mode'))} / {run.get('cudagraph_mode') or 'n/a'}",
        _istr(memory.get("ranks")),
        f"{_fstr(memory.get('storage_gib_min'), 4)} "
        f"({_istr(memory.get('storage_bytes_min'))})"
        if memory.get("present")
        else "n/a",
        _fstr(memory.get("storage_gib_sum"), 3),
        _istr(serve.get("kv_size_tokens")),
        _fstr(serve.get("ppl"), 2),
        " / ".join(
            _range(serve.get(f"{name}_min"), serve.get(f"{name}_max"))
            for name in ("consumed_gib", "activation_gib", "graph_gib")
        ),
        " / ".join(
            _range(serve.get(f"{name}_min"), serve.get(f"{name}_max"))
            for name in ("graph_pool_actual_gib", "graph_pool_estimated_gib")
        ),
        "n/a"
        if dev is None and host is None
        else f"{_fstr(dev / 2**30 if dev is not None else None, 2)} / "
        f"{_fstr(host / 2**30 if host is not None else None, 2)}",
    ]


def _cell_row(cell: dict[str, Any]) -> list[str]:
    cache = "n/a"
    if cell.get("cache_demands") is not None:
        cache = (
            f"{_fstr(cell.get('cache_miss_rate_pct'), 1)}% "
            f"({_istr(cell.get('cache_misses'))}/"
            f"{_istr(cell.get('cache_demands'))})"
        )
        if cell.get("cache_copied_bytes") is not None:
            cache += f" / {_fstr(cell['cache_copied_bytes'] / 2**20, 2)} MiB"
    mtp = "n/a"
    if cell.get("mtp_acceptance_rate") is not None:
        mtp = f"{_fstr(cell['mtp_acceptance_rate'], 1)}%"
        if cell.get("mtp_accepted_tokens") is not None:
            mtp += (
                f" ({_istr(cell['mtp_accepted_tokens'])}/"
                f"{_istr(cell.get('mtp_draft_tokens'))})"
            )
    valid = "yes" if cell["valid"] else "no: " + ",".join(cell["invalid_reasons"])
    if cell["valid"] and cell["warnings"]:
        valid = "yes (warn)"
    return [
        cell["cell"],
        f"{cell['arm'] or '?'}/{cell['replicate']}",
        f"{_istr(cell['completed'])}/{_istr(cell['requested'])}",
        _fstr(cell.get("prompt_tokens_mean"), 0),
        f"{_fstr(cell.get('ttft_p50_ms'))} / {_fstr(cell.get('ttft_p99_ms'))}",
        f"{_fstr(cell.get('prefill_req_tps'))} / "
        f"{_fstr(cell.get('prefill_agg_bench_tps'))} / "
        f"{_fstr(cell.get('prefill_span_tps'))}",
        f"{_fstr(cell.get('decode_req_tps'), 2)} / "
        f"{_fstr(cell.get('decode_only_tps'), 2)} / "
        f"{_fstr(cell.get('decode_span_agg_tps'), 2)}",
        _fstr(cell.get("output_throughput_e2e"), 2),
        f"{_fstr(cell.get('itl_p50_ms'))} / {_fstr(cell.get('itl_p99_ms'))}",
        mtp,
        cache,
        _istr(cell.get("preemptions")),
        _istr(cell.get("faults")),
        str(cell.get("evictions") or "n/a"),
        valid,
    ]


def _comparison_row(comparison: dict[str, Any]) -> list[str]:
    baseline = (
        f"{comparison['baseline_tag'] or comparison['baseline_arm']} "
        f"r{comparison['baseline_replicate']}"
        if comparison.get("baseline_replicate") is not None
        else "n/a"
    )
    return [
        comparison["cell"],
        f"{comparison['arm'] or '?'}/{comparison['replicate']}",
        baseline,
        _cmp_group(
            comparison, ("prefill_req_tps", "prefill_agg_bench_tps", "prefill_span_tps")
        ),
        _cmp_group(comparison, ("ttft_p50_ms", "ttft_p99_ms")),
        _cmp_group(
            comparison, ("decode_req_tps", "decode_only_tps", "decode_span_agg_tps")
        ),
        _cmp_group(comparison, ("itl_p50_ms", "itl_p99_ms")),
        _cmp(comparison["metrics"].get("output_throughput_e2e")),
        _cmp_group(comparison, ("kv_tp_min_gib", "kv_size_tokens")),
        comparison["status"],
    ]


def markdown_report(report: dict[str, Any]) -> str:
    problems = []
    for cell in report["cells"]:
        if not cell["invalid_reasons"] and not cell["warnings"]:
            continue
        parts = [
            f"{label}: {', '.join(values)}"
            for label, values in (
                ("invalid", cell["invalid_reasons"]),
                ("warnings", cell["warnings"]),
            )
            if values
        ]
        problems.append(
            f"- {cell['arm']}/{cell['replicate']}/{cell['cell']}: " + "; ".join(parts)
        )
    problems.extend(f"- run: {warning}" for warning in report["warnings"])
    return "\n".join(
        [
            "# RAM10 TP4 expert-offload study",
            "",
            f"Generated `{report['generated_utc']}` from `{report['run_root']}`; "
            f"baseline arm `{report['baseline_arm']}`.",
            "",
            "## Capacity and startup",
            "",
            _table(
                [
                    "Arm/rep",
                    "Run",
                    "Slots",
                    "W4A8 pre/sd",
                    "Compile / graph",
                    "KV ranks",
                    "KV TP-min GiB (B)",
                    "KV sum GiB",
                    "KV tokens",
                    "PPL",
                    "Startup c/a/g GiB",
                    "Graph pool a/e GiB",
                    "Expert dev/host GiB",
                ],
                [_capacity_row(run) for run in report["runs"]],
            ),
            "",
            "## Cells",
            "",
            _table(
                [
                    "Cell",
                    "Arm/rep",
                    "ok/req",
                    "Prompt",
                    "TTFT p50/p99 ms",
                    "Prefill req/agg/span tok/s",
                    "Decode req/wtd/span tok/s",
                    "Out e2e tok/s",
                    "ITL p50/p99 ms",
                    "MTP acc",
                    "Cache miss% (m/d) / copy",
                    "Preempt",
                    "Fault",
                    "Evict",
                    "Valid",
                ],
                [_cell_row(cell) for cell in report["cells"]],
            ),
            "",
            "## Baseline-relative deltas",
            "",
            _table(
                [
                    "Cell",
                    "Arm/rep",
                    "Baseline",
                    "Δ prefill req/agg/span %",
                    "Δ TTFT p50/p99 %",
                    "Δ decode req/wtd/span %",
                    "Δ ITL p50/p99 %",
                    "Δ out e2e %",
                    "Δ KV TP-min/tokens %",
                    "Status",
                ],
                [_comparison_row(item) for item in report["comparisons"]],
            ),
            "",
            "## Health and missing data",
            "",
            *(problems or ["- none"]),
            "",
            "## Notes",
            "",
            *(f"- {note}" for note in report["notes"]),
            "",
        ]
    )


def _csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in report["per_request_rows"]:
        out: dict[str, Any] = {}
        for key in CSV_COLS:
            value = row.get(key)
            out[key] = (
                ("" if value is None else int(bool(value)))
                if key == "success"
                else (
                    ""
                    if value is None
                    else f"{value:.9g}"
                    if isinstance(value, float)
                    else value
                )
            )
        rows.append(out)
    return rows


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "ram10-report.json",
        "csv": out_dir / "ram10-per-request.csv",
        "markdown": out_dir / "ram10-report.md",
    }
    _dump(
        paths["json"],
        {key: value for key, value in report.items() if key != "per_request_rows"},
    )
    with open(paths["csv"], "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLS))
        writer.writeheader()
        writer.writerows(_csv_rows(report))
    paths["markdown"].write_text(markdown_report(report), encoding="utf-8")
    return paths


def _selftest() -> int:
    failures: list[str] = []

    def true(label: str, condition: bool) -> None:
        if not condition:
            failures.append(label)

    def check(label: str, got: Any, want: Any, tol: float = 1e-9) -> None:
        try:
            ok = got is not None and abs(float(got) - float(want)) <= tol
        except (TypeError, ValueError):
            ok = got == want
        if not ok:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    health = {
        "bench_rc": 0,
        "warmup_rc": 0,
        "warmup_enabled": 1,
        "reset_before_warmup_ok": 1,
        "reset_after_warmup_ok": 1,
        "evictions": "ok",
        "preemptions_in_window": 0,
        "faults_in_window": 0,
        "server_alive_after": 1,
        "result_present": 1,
        "warmup_present": 1,
    }
    base = {
        "completed": 1,
        "failed": 0,
        "num_prompts": 1,
        "total_input_tokens": 100,
        "total_output_tokens": 3,
        "duration": 2.0,
        "output_throughput": 1.5,
        "max_concurrency": 1,
        "mean_ttft_ms": 500.0,
        "median_ttft_ms": 500.0,
        "p99_ttft_ms": 500.0,
        "mean_tpot_ms": 250.0,
        "mean_itl_ms": 250.0,
        "median_itl_ms": 250.0,
        "p99_itl_ms": 300.0,
        "input_lens": [100],
        "output_lens": [3],
        "ttfts": [0.5],
        "latencies": [1.0],
        "itls": [[0.25, 0.25]],
        "errors": [None],
        "start_times": [0.0],
        "spec_decode_accepted_tokens": 2,
        "spec_decode_num_drafts": 4,
    }
    broken = {
        "completed": 1,
        "failed": 0,
        "num_prompts": 1,
        "total_input_tokens": 100,
        "total_output_tokens": 1,
        "duration": 1.0,
        "output_throughput": 1.0,
        "max_concurrency": 1,
        "mean_ttft_ms": 1000.0,
        "median_ttft_ms": 1000.0,
        "p99_ttft_ms": 1000.0,
        "mean_tpot_ms": 0.0,
        "mean_itl_ms": 0.0,
        "median_itl_ms": 0.0,
        "p99_itl_ms": 0.0,
    }
    with tempfile.TemporaryDirectory(prefix="ram10-report-") as tmp:
        fault_probe = Path(tmp) / "fault-probe.log"
        for signal in ("GPU Hang", "HW Exception"):
            fault_probe.write_text(signal + "\n", encoding="utf-8")
            assert (
                parse_serve(fault_probe, Path(tmp) / "summary.txt")["boot_faults"] == 1
            )

        def write_cell(
            path: Path,
            result: dict[str, Any],
            overrides: dict[str, Any] | None = None,
            before: int = 0,
            after: int = 0,
        ) -> Path:
            path.mkdir(parents=True)
            _dump(path / "bench.json", result)
            _dump(path / "warmup.json", {"completed": 1, "failed": 0})
            _dump(path / "cell.json", {**health, **(overrides or {})})
            (path / "metrics-before.txt").write_text(
                f'vllm:num_preemptions_total{{model_name="x"}} {before}\n',
                encoding="utf-8",
            )
            (path / "metrics-after.txt").write_text(
                f'vllm:num_preemptions_total{{model_name="x"}} {after}\n'
                'vllm:num_preemptions_total_created{model_name="x"} 999\n',
                encoding="utf-8",
            )
            return path

        valid = write_cell(Path(tmp) / "valid", base)
        row = analyse_cell(valid, "bench-1024x512-c1", "R", "1")
        for label, got, want in (
            ("prefill req", row["prefill_req_tps"], 200.0),
            ("prefill agg", row["prefill_agg_bench_tps"], 200.0),
            ("prefill span", row["prefill_span_tps"], 200.0),
            ("decode req", row["decode_req_tps"], 4.0),
            ("decode-only", row["decode_only_tps"], 4.0),
            ("decode-span", row["decode_span_agg_tps"], 4.0),
            ("e2e", row["output_throughput_e2e"], 1.5),
            ("itl p99", row["itl_p99_ms"], 300.0),
        ):
            check(label, got, want)
        true("valid cell", row["valid"])
        chunk_baseline = {**row, "itl_complete": False}
        chunk_candidate = {
            **chunk_baseline,
            "arm": "D",
            "itl_p50_ms": row["itl_p50_ms"] * 2,
        }
        chunk_delta = compare_cells([chunk_baseline, chunk_candidate], "R")[0]
        check(
            "MTP delivery-chunk ITL comparison",
            chunk_delta["metrics"]["itl_p50_ms"]["pct"],
            100.0,
        )
        memory = parse_memory(
            {
                "results": [
                    {
                        "kv_cache_storage_bytes": size,
                        "kv_cache_views": 2,
                        "kv_cache_storages": 1,
                        "kv_cache_devices": ["cuda:0"],
                    }
                    for size in (5_046_586_163, 4_831_838_208, 4_831_838_208)
                ]
            }
        )
        check("KV min", memory["storage_bytes_min"], 4_831_838_208.0)
        check("KV sum", memory["storage_bytes_sum"], 14_710_262_579.0)
        check("KV ranks", memory["ranks"], 3.0)
        true("missing RPC null", parse_memory(None)["storage_bytes_min"] is None)
        invalid = write_cell(
            Path(tmp) / "invalid",
            broken,
            {"evictions": "INVALID", "preemptions_in_window": 2},
            after=2,
        )
        bad = analyse_cell(invalid, "bench-1024x512-c8", "D", "1")
        true("invalid KFD", "kfd_evictions" in bad["invalid_reasons"])
        unknown_kfd = write_cell(
            Path(tmp) / "unknown-kfd", base, {"evictions": "unavailable"}
        )
        unobserved = analyse_cell(unknown_kfd, "bench-1024x512-c1", "R", "1")
        true(
            "missing KFD invalid",
            "kfd_evictions_unavailable" in unobserved["invalid_reasons"],
        )
        true("invalid detail", "missing_per_request_detail" in bad["invalid_reasons"])
        true("capacity limited", bad["capacity_limited"] and bad["preemptions"] == 2)
        true(
            "missing decode",
            bad["decode_only_tps"] is None and bad["decode_span_agg_tps"] is None,
        )
        check("finalize invalid", finalize_cell(invalid)["valid"], False)
        true("finalize persisted", _load(invalid / "cell.json")["valid"] is False)
        true("finalize valid", finalize_cell(valid)["valid"])
        partial_root = Path(tmp) / "partial"
        partial_run = partial_root / "resident"
        write_cell(partial_run / "cells" / "bench-1024x512-c1", base)
        _dump(
            partial_run / "metadata.json",
            {
                "arm": "R",
                "replicate": "1",
                "status": "running",
                "cells": ["1024:512:1:1", "16384:1024:8:8"],
            },
        )
        partial = build_report(partial_root, "R")
        true(
            "incomplete run warning",
            any("not done" in warning for warning in partial["warnings"]),
        )
        true(
            "missing declared cell",
            partial["runs"][0]["missing_declared_cells"] == ["bench-16384x1024-c8"],
        )
    if failures:
        for failure in failures:
            print(f"selftest FAIL: {failure}")
        return 1
    print("selftest PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", help="study run root")
    parser.add_argument("--baseline", default="R", help="baseline arm (default R)")
    parser.add_argument("--out", help="output directory (default: run root)")
    parser.add_argument(
        "--finalize-cell", help="recompute one cell.json; exit 0 valid / 1 invalid"
    )
    parser.add_argument(
        "--selftest", action="store_true", help="run the deterministic fixture"
    )
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.finalize_cell:
        row = finalize_cell(Path(args.finalize_cell))
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "cell",
                        "valid",
                        "invalid_reasons",
                        "warnings",
                        "decode_req_tps",
                        "decode_only_tps",
                        "decode_span_agg_tps",
                    )
                },
                sort_keys=True,
            )
        )
        return 0 if row["valid"] else 1
    if not args.run_root:
        parser.error("--run-root is required")
    baseline = _arm(args.baseline)
    if baseline is None:
        parser.error(f"unknown baseline arm {args.baseline!r}")
    root = Path(args.run_root)
    if not root.is_dir():
        print(f"ram10_report: run root does not exist: {root}", file=sys.stderr)
        return 2
    report = build_report(root, baseline_arm=baseline)
    paths = write_outputs(report, Path(args.out) if args.out else root)
    print(f"report: {paths['json']}")
    print(f"per-request CSV: {paths['csv']}")
    print(f"markdown: {paths['markdown']}")
    print(
        f"runs={len(report['runs'])} cells={len(report['cells'])} "
        f"invalid={sum(1 for cell in report['cells'] if not cell['valid'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
