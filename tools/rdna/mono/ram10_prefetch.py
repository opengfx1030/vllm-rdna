#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay CPU expert-cache receipts; count transfers, never estimate runtime.

Input is the JSON response from rdna_expert_cache_trace(action="stop").
Ordinary replay must reproduce its recorded miss counters exactly. Predictors
use only already observed demands and the SAME number of real cache slots.
Instantaneous speculative fills are an optimistic accounting model, not proof
of safe overlap, avoided wall time, or a deployable prefetch implementation.
"""

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_policy():
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm/model_executor/layers/fused_moe/rdna_expert_cache_policy.py"
    )
    spec = importlib.util.spec_from_file_location("_ram10_policy", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ExpertCachePolicy


def restore(record, policy_class):
    policy = policy_class(
        record["local_experts"],
        record["capacity"],
        record["policy"],
        record["decay_interval"],
    )
    policy.epoch = record["initial_epoch"]
    policy._frequency = record["initial_frequencies"][:]
    policy._last_used = record["initial_last_used"][:]
    policy._expert_slots = {
        int(expert): slot for expert, slot in record["initial_mapping"].items()
    }
    for expert, slot in policy.mapping.items():
        policy._slot_experts[slot] = expert
    return policy


def replay(response, mode="ordinary", budget=2, warmup_calls=128):
    if mode not in ("ordinary", "previous_step", "transition", "transition_missing"):
        raise ValueError(f"unknown predictor {mode}")
    policy_class = load_policy()
    nodes = {}
    for rank, result in enumerate(response["results"]):
        for record in result["layers"]:
            key = (rank, record["layer"])
            if key in nodes:
                raise ValueError(f"duplicate rank/layer {key}")
            nodes[key] = {
                "record": record,
                "policy": restore(record, policy_class),
                "pending": {},
                "previous": set(),
                "model": defaultdict(Counter),
                "native_misses": record["initial_misses"],
            }
    lengths = {len(node["record"]["ids"]) for node in nodes.values()}
    if not nodes:
        raise ValueError("trace must contain expert-cache layers")
    # A per-worker byte cap can end midway through a forward. Score only the
    # shared complete prefix, retaining its exact observed boundary counters.
    calls = min(lengths)
    if not calls or not 0 <= warmup_calls < calls:
        raise ValueError("warmup_calls must leave at least one scored call")
    totals = Counter()
    whole = Counter()
    for counters in (totals, whole):
        for field in (
            "demand_experts",
            "demand_misses",
            "demand_bytes",
            "demand_passes",
            "prediction_candidates",
            "predictions_already_resident",
            "prefetch_misses",
            "prefetch_bytes",
            "useful_prefetch_loads",
            "unused_prefetch_evictions",
            "prefetch_not_used_by_trace_end",
        ):
            counters[field] = 0

    def count(step, field, amount=1, *, eligible=True):
        whole[field] += amount
        if step >= warmup_calls and eligible:
            totals[field] += amount

    def retire(node, step):
        mapping = node["policy"].mapping
        for expert in list(node["pending"]):
            if expert not in mapping:
                issued = node["pending"].pop(expert)
                count(
                    step, "unused_prefetch_evictions", eligible=issued >= warmup_calls
                )

    for step in range(calls):
        observed = {}
        for key, node in sorted(nodes.items()):
            record, policy = node["record"], node["policy"]
            source = observed.get((key[0], key[1] - 1))
            scores = Counter()
            if mode.startswith("transition") and source is not None:
                for expert in source:
                    scores.update(node["model"][expert])
            elif mode == "previous_step":
                scores.update({expert: 1 for expert in node["previous"]})
            elif mode != "ordinary" and not mode.startswith("transition"):
                raise ValueError(f"unknown predictor {mode}")
            mapping = policy.mapping
            candidates = sorted(scores, key=lambda expert: (-scores[expert], expert))
            if mode == "transition_missing":
                candidates = [expert for expert in candidates if expert not in mapping]
            for expert in candidates[:budget]:
                count(step, "prediction_candidates")
                if expert in policy.mapping:
                    count(step, "predictions_already_resident")
                    continue
                lease = policy.acquire([expert])
                retire(node, step)
                lease.commit()
                lease.release()
                node["pending"][expert] = step
                count(step, "prefetch_misses")
                count(step, "prefetch_bytes", record["bytes_per_miss"])

            ids = record["ids"][step]
            unique = {expert for expert in ids if 0 <= expert < policy.num_experts}
            if mode == "ordinary" and (
                node["native_misses"] != record["misses_before"][step]
            ):
                raise ValueError(f"native replay mismatch before {key} call {step}")
            policy.observe(ids)
            for experts in policy.plan_passes(ids):
                lease = policy.acquire(experts)
                retire(node, step)
                for expert, _, miss in lease.bindings:
                    count(step, "demand_experts")
                    if miss:
                        node["native_misses"] += 1
                        count(step, "demand_misses")
                        count(step, "demand_bytes", record["bytes_per_miss"])
                    elif expert in node["pending"]:
                        issued = node["pending"].pop(expert)
                        count(
                            step,
                            "useful_prefetch_loads",
                            eligible=issued >= warmup_calls,
                        )
                lease.commit()
                lease.release()
                count(step, "demand_passes")
            # Fit only AFTER this target has actually executed. No future routes
            # enter ranking; only physically adjacent, already observed layers do.
            if source is not None:
                for expert in source:
                    node["model"][expert].update(unique)
            node["previous"] = unique
            observed[key] = unique

    if mode == "ordinary":
        for key, node in nodes.items():
            record = node["record"]
            end = (
                record["misses_before"][calls]
                if len(record["ids"]) > calls
                else record["captured_end_misses"]
            )
            if node["native_misses"] != end:
                raise ValueError(f"native replay end mismatch at {key}")
    for node in nodes.values():
        unused = sum(step >= warmup_calls for step in node["pending"].values())
        totals["prefetch_not_used_by_trace_end"] += unused
        whole["prefetch_not_used_by_trace_end"] += len(node["pending"])
    for counters in (totals, whole):
        counters["total_copy_bytes"] = (
            counters["demand_bytes"] + counters["prefetch_bytes"]
        )
    return {
        "mode": mode,
        "budget_per_target_call": budget,
        "warmup_calls_per_layer": warmup_calls,
        "scored_calls_per_layer": calls - warmup_calls,
        "layer_records": len(nodes),
        "distinct_captured_call_counts": sorted(lengths),
        "unscored_tail_calls": sum(
            len(node["record"]["ids"]) - calls for node in nodes.values()
        ),
        "trace_byte_limit_reached": any(
            result.get("byte_limit_reached", False) for result in response["results"]
        ),
        "capacities": sorted({node["policy"].capacity for node in nodes.values()}),
        "native_replay_exact": True if mode == "ordinary" else None,
        "scored": dict(totals),
        "whole_trace": dict(whole),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--warmup-calls", type=int, default=128)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.budget <= 8:
        parser.error("budget must be in [1, 8]")
    response = json.loads(args.trace.read_text())
    results = [
        replay(response, mode, args.budget, args.warmup_calls)
        for mode in ("ordinary", "previous_step", "transition", "transition_missing")
    ]
    baseline = results[0]["scored"]
    for result in results:
        scored = result["scored"]
        result["demand_misses_avoided_vs_ordinary"] = (
            baseline["demand_misses"] - scored["demand_misses"]
        )
        result["extra_copy_bytes_vs_ordinary"] = (
            scored["total_copy_bytes"] - baseline["total_copy_bytes"]
        )
    output = {
        "input": str(args.trace),
        "results": results,
        "limits": [
            "Diagnostic trace overhead excluded from all performance claims.",
            "Same real slots and native demand policy; no extra VRAM assumed.",
            "Instantaneous fills: no copy/compute contention or timing proof.",
            "Online training prefix, not an independent held-out workload.",
            "Transition targets are recorded physically adjacent layers only.",
            "Never substitute these predictors for the model's learned router.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
