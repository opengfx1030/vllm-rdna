# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in CPU-only local expert route receipts for offline policy replay.

Call only at a safe host planning boundary. The low-level recorder deliberately
has no GPU/runtime dependency; phase and rank must be supplied by its caller.
Each stream has an immutable header and an atomically replaced count manifest.
"""

import json
import os
import threading
import uuid
from pathlib import Path

import regex as re

ENABLED = bool(os.environ.get("VLLM_RDNA_EXPERT_ROUTE_DIR"))
_SESSION = uuid.uuid4().hex
_STREAMS: dict[tuple[int, str, str], dict] = {}
_LOCK = threading.Lock()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def record_routes(
    layer_name, local_ids, logical_experts, *, phase, rank, expert_bytes=None
):
    """Record complete invocation routes, normalizing nonlocal sentinels to -1.

    VLLM_RDNA_EXPERT_ROUTE_MAX_CALLS bounds each rank/layer/phase stream (128
    default). No truncation of a recorded invocation or valid local demand.
    Returns False when disabled or the stream's call limit has been reached.
    """
    if not ENABLED:
        return False
    arm_file = os.environ.get("VLLM_RDNA_EXPERT_ROUTE_ARM_FILE")
    if arm_file and not Path(arm_file).exists():
        return False
    if phase not in ("prefill", "decode", "profile"):
        raise ValueError("phase must be prefill, decode or profile")
    selected_phase = os.environ.get("VLLM_RDNA_EXPERT_ROUTE_PHASE", "all")
    if selected_phase not in ("all", "prefill", "decode"):
        raise ValueError("VLLM_RDNA_EXPERT_ROUTE_PHASE must be all, prefill or decode")
    if selected_phase != "all" and phase != selected_phase:
        return False
    rank = _integer(rank, "rank")
    logical_experts = _integer(logical_experts, "logical_experts", 1)
    if expert_bytes is not None:
        _integer(expert_bytes, "expert_bytes", 1)
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if match is None:
        raise ValueError("layer_name must contain a layers.<index> component")
    layer = int(match[1])
    limit = int(os.environ.get("VLLM_RDNA_EXPERT_ROUTE_MAX_CALLS", "128"))
    _integer(limit, "VLLM_RDNA_EXPERT_ROUTE_MAX_CALLS")
    key = rank, layer_name, phase
    with _LOCK:
        stream = _STREAMS.get(key)
        if limit == 0 or (stream is not None and stream["recorded_calls"] >= limit):
            return False
        # Only already-materialized CPU integers are accepted: never implicitly
        # iterate a GPU tensor or call .item()/.cpu() here.
        if not isinstance(local_ids, (list, tuple)):
            raise ValueError("local_ids must be a CPU Python list or tuple")
        ids = list(local_ids)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
            raise ValueError("local_ids must contain CPU Python integers")
        ids = [value if 0 <= value < logical_experts else -1 for value in ids]
        if stream is None:
            directory = Path(os.environ["VLLM_RDNA_EXPERT_ROUTE_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            filename = f"routes-rank{rank}-layer{layer}-{phase}-{_SESSION}.jsonl"
            stream = dict(
                schema="rdna-expert-routes",
                version=1,
                stream_id=_SESSION,
                rank=rank,
                layer=layer,
                layer_name=layer_name,
                phase=phase,
                logical_experts=logical_experts,
                expert_bytes=expert_bytes,
                file=filename,
                recorded_calls=0,
                routed_pairs=0,
                local_pairs=0,
            )
            path = directory / filename
            with path.open("x") as handle:
                handle.write(json.dumps(dict(type="header", **stream)) + "\n")
            _STREAMS[key] = stream
        elif (stream["logical_experts"], stream["expert_bytes"]) != (
            logical_experts,
            expert_bytes,
        ):
            raise ValueError("trace stream expert geometry changed")
        directory = Path(os.environ["VLLM_RDNA_EXPERT_ROUTE_DIR"])
        record = dict(
            type="routes",
            stream_id=stream["stream_id"],
            rank=rank,
            layer=layer,
            phase=phase,
            call=stream["recorded_calls"],
            ids=ids,
        )
        with (directory / stream["file"]).open("a") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        stream["recorded_calls"] += 1
        stream["routed_pairs"] += len(ids)
        stream["local_pairs"] += sum(value >= 0 for value in ids)
        manifest = directory / (stream["file"] + ".manifest.json")
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(stream, indent=2) + "\n")
        temporary.replace(manifest)
        return True


def record_cache_routes(cache, layer_name, ids, rank):
    """Classify metadata without device reads; arm after warmup for real traces."""
    if not ENABLED:
        return False
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    capture = BreakableCUDAGraphCapture.current()
    if capture is not None and not capture.replaying:
        return False
    if not is_forward_context_available():
        return False
    context = get_forward_context()
    metadata = context.attn_metadata
    if getattr(context, "is_profile", False) or not isinstance(metadata, dict):
        return False
    values = list(metadata.values())
    prefill = any(
        getattr(md, "num_prefills", 0) or getattr(md, "num_prefill_tokens", 0)
        for md in values
    )
    decode = any(
        getattr(md, "num_decodes", 0)
        or getattr(md, "num_decode_tokens", 0)
        or getattr(md, "num_spec_decodes", 0)
        or getattr(md, "num_spec_decode_tokens", 0)
        for md in values
    )
    if not prefill and not decode:
        return False  # Missing/profile metadata is not an observed workload.
    return record_routes(
        layer_name,
        ids,
        cache.logical_num_experts,
        phase="prefill" if prefill else "decode",
        rank=rank,
        expert_bytes=cache._expert_bytes,
    )
