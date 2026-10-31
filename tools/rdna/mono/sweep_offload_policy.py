# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline, empty-cache decode-trace policy comparison; no GPU execution."""

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import regex as re


def load_policy():
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm/model_executor/layers/fused_moe/rdna_expert_cache_policy.py"
    )
    spec = importlib.util.spec_from_file_location("_sweep_expert_policy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ExpertCachePolicy


def simulate(calls, capacity, policy, local_experts=256, decay_interval=128):
    cache = load_policy()(local_experts, capacity, policy, decay_interval)
    totals = dict(calls=0, routed_pairs=0, expert_demands=0, hits=0, misses=0, passes=0)
    for ids in calls:
        valid = [i for i in ids if 0 <= i < local_experts]
        totals["calls"] += 1
        totals["routed_pairs"] += len(valid)
        totals["expert_demands"] += len(set(valid))
        cache.observe(ids)
        for experts in cache.plan_passes(ids):
            lease = cache.acquire(experts)
            totals["passes"] += 1
            totals["misses"] += sum(miss for _, _, miss in lease.bindings)
            totals["hits"] += sum(not miss for _, _, miss in lease.bindings)
            lease.commit()
            lease.release()
    totals["copied_expert_units"] = totals["misses"]
    return totals


class TraceRecords(dict):
    def __init__(self, values, phase, excluded_phases=(), runtime_costs=None):
        super().__init__(values)
        self.trace_phase = phase
        self.excluded_phases = tuple(excluded_phases)
        self.runtime_costs = runtime_costs


def _read_route_calls(paths, local_experts, phase):
    records, phases, identities, runtime_costs = {}, set(), set(), {}
    for path in paths:
        lines = path.read_text().splitlines()
        if not lines:
            raise ValueError(f"empty route trace: {path}")
        header, *events = [json.loads(line) for line in lines]
        fields = {
            "schema",
            "version",
            "stream_id",
            "rank",
            "layer",
            "layer_name",
            "phase",
            "logical_experts",
            "expert_bytes",
            "file",
            "recorded_calls",
            "routed_pairs",
            "local_pairs",
        }
        if (
            set(header) != fields | {"type"}
            or header["type"] != "header"
            or header["schema"] != "rdna-expert-routes"
            or header["version"] != 1
        ):
            raise ValueError(f"invalid route trace header: {path}")
        rank = _integer(header["rank"], "rank")
        layer = _integer(header["layer"], "layer")
        _integer(header["version"], "version", 1)
        _integer(header["logical_experts"], "logical_experts", 1)
        for name in ("recorded_calls", "routed_pairs", "local_pairs"):
            _integer(header[name], name)
        layer_match = (
            re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", header["layer_name"])
            if isinstance(header["layer_name"], str)
            else None
        )
        if (
            header["phase"] not in ("prefill", "decode", "profile")
            or header["logical_experts"] != local_experts
            or isinstance(header["logical_experts"], bool)
            or not isinstance(header["stream_id"], str)
            or not header["stream_id"]
            or not isinstance(header["layer_name"], str)
            or layer_match is None
            or int(layer_match[1]) != layer
            or header["file"] != path.name
            or any(
                header[name] != 0
                for name in ("recorded_calls", "routed_pairs", "local_pairs")
            )
        ):
            raise ValueError(f"invalid route trace geometry/scope: {path}")
        if header["expert_bytes"] is not None:
            _integer(header["expert_bytes"], "expert_bytes", 1)
        identity = rank, layer, header["phase"]
        if identity in identities:
            raise ValueError(f"duplicate route stream for rank/layer/phase: {identity}")
        identities.add(identity)
        phases.add(header["phase"])
        calls, pairs, valid_pairs = [], 0, 0
        for ordinal, event in enumerate(events):
            if (
                set(event)
                != {"type", "stream_id", "rank", "layer", "phase", "call", "ids"}
                or event["type"] != "routes"
                or any(
                    event[key] != header[key]
                    for key in ("stream_id", "rank", "layer", "phase")
                )
                or _integer(event["call"], "call") != ordinal
                or _integer(event["rank"], "rank") != rank
                or _integer(event["layer"], "layer") != layer
                or not isinstance(event["ids"], list)
            ):
                raise ValueError(f"invalid route invocation: {path}")
            ids = event["ids"]
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < -1
                or value >= local_experts
                for value in ids
            ):
                raise ValueError(f"invalid local route ID: {path}")
            calls.append(ids)
            pairs += len(ids)
            valid_pairs += sum(value >= 0 for value in ids)
        manifest_path = path.with_name(path.name + ".manifest.json")
        if not manifest_path.exists():
            raise ValueError(f"missing route count manifest: {path}")
        manifest = json.loads(manifest_path.read_text())
        expected = {key: value for key, value in header.items() if key != "type"}
        expected.update(
            recorded_calls=len(calls), routed_pairs=pairs, local_pairs=valid_pairs
        )
        if (
            json.dumps(manifest, sort_keys=True) != json.dumps(expected, sort_keys=True)
            or not calls
        ):
            raise ValueError(f"route manifest count/schema mismatch: {path}")
        records[identity] = calls
        runtime_costs[identity] = dict(
            expert_bytes=header["expert_bytes"],
            map_bytes=local_experts * 4,
            logical_experts=local_experts,
        )
    if phase is None:
        if len(phases) != 1:
            raise ValueError("mixed route phases require an explicit phase selection")
        phase = next(iter(phases))
    if phase not in phases:
        raise ValueError(f"no route traces for phase {phase!r}")
    selected = {
        (rank, layer): calls
        for (rank, layer, scope), calls in records.items()
        if scope == phase
    }
    selected_costs = {
        (rank, layer): costs
        for (rank, layer, scope), costs in runtime_costs.items()
        if scope == phase
    }
    return TraceRecords(selected, phase, sorted(phases - {phase}), selected_costs)


def read_calls(directory, local_experts=256, phase=None):
    paths = sorted(Path(directory).glob("*.jsonl"))
    legacy = sorted(Path(directory).glob("rank*-layer*-call*.pt"))
    if paths and legacy:
        raise ValueError("mixed .pt and JSONL expert trace formats are not supported")
    if paths:
        return _read_route_calls(paths, local_experts, phase)
    if phase not in (None, "decode"):
        raise ValueError("legacy .pt traces are decode-only")
    import torch

    records = defaultdict(list)
    seen = set()
    for path in legacy:
        receipt = torch.load(path, map_location="cpu", weights_only=True)
        rank, layer, call = (int(receipt[k]) for k in ("rank", "layer", "call"))
        key = rank, layer, call
        if key in seen:
            raise ValueError(f"duplicate trace identity: {key}")
        seen.add(key)
        routes = receipt["tensors"]["route_ids"]
        if routes.dtype.is_floating_point or routes.dtype == torch.bool:
            raise ValueError(f"route IDs must be integers: {path}")
        global_ids = routes.reshape(-1).tolist()
        if any(i < -1 or i >= 2 * local_experts for i in global_ids):
            raise ValueError(f"invalid global route ID: {path}")
        if rank not in (0, 1):
            raise ValueError("this sweep expects TP2 rank IDs 0 and 1")
        ids = [i - rank * local_experts for i in global_ids]
        records[rank, layer].append((call, ids))
    if not records:
        raise ValueError("no expert trace receipts found")
    return TraceRecords(
        {key: [ids for _, ids in sorted(values)] for key, values in records.items()},
        "decode",
    )


def sweep(records, capacities, policies, payload_bytes=None, decay_interval=128):
    results = []
    for policy in policies:
        for capacity in capacities:
            layers = []
            total = defaultdict(int)
            ranks = defaultdict(lambda: defaultdict(int))
            for (rank, layer), calls in sorted(records.items()):
                counts = simulate(
                    calls, capacity, policy, decay_interval=decay_interval
                )
                layers.append(dict(rank=rank, layer=layer, **counts))
                for name, value in counts.items():
                    total[name] += value
                    ranks[rank][name] += value
            demands = total["expert_demands"]
            result = dict(
                policy=policy,
                capacity=capacity,
                totals=dict(total),
                hit_fraction=total["hits"] / demands if demands else None,
                per_rank={str(k): dict(v) for k, v in ranks.items()},
                per_layer=layers,
            )
            if payload_bytes is not None:
                result["modeled_copy_bytes"] = total["misses"] * payload_bytes
            results.append(result)
    trace_phase = getattr(records, "trace_phase", "decode")
    return dict(
        trace_phase=trace_phase,
        excluded_trace_phases=getattr(records, "excluded_phases", ()),
        scope=(
            f"{trace_phase}-only trace, empty initial caches; excludes unrecorded "
            "residency, timing, and transfer overlap; informational, "
            "not workload optimality or physical residency replay. Callers "
            "must exclude padding/capture routes; every supplied demand is retained"
        ),
        decay_interval=decay_interval,
        max_calls_per_rank_layer=max(map(len, records.values())),
        decay_exercised=any(len(calls) >= decay_interval for calls in records.values()),
        payload_bytes_per_expert=payload_bytes,
        results=results,
    )


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def read_plan_profile(path):
    """Read explicit per-rank/layer costs; duplicate identities fail closed."""
    document = json.loads(Path(path).read_text())
    profiles = {}
    for entry in document["layers"]:
        key = (_integer(entry["rank"], "rank"), _integer(entry["layer"], "layer"))
        if key in profiles:
            raise ValueError(f"duplicate layer profile: {key}")
        profiles[key] = {k: v for k, v in entry.items() if k not in ("rank", "layer")}
    budgets = {}
    for rank, value in document["rank_budgets"].items():
        parsed = int(rank)
        if str(parsed) != rank or parsed < 0:
            raise ValueError("rank budget keys must be canonical nonnegative integers")
        budgets[parsed] = value
    return profiles, budgets


def plan_slots(
    records,
    profiles,
    rank_budgets,
    capacities,
    policy="lfu_lru",
    baseline_capacity=96,
    decay_interval=128,
    local_experts=256,
):
    """Greedily buy measured miss-byte savings under independent rank budgets.

    Every candidate uses the existing policy replay, including overflow passes.
    Consider all higher-capacity jumps (miss curves need not be convex or
    monotonic). This deterministic heuristic is not a global optimum or a
    latency prediction. Optional objective_weight scales measured miss bytes,
    e.g. a caller-supplied relative transfer-time weight.
    """
    _integer(local_experts, "local_experts", 1)
    _integer(decay_interval, "decay_interval", 1)
    _integer(baseline_capacity, "baseline_capacity", 1)
    if baseline_capacity > local_experts:
        raise ValueError("baseline_capacity exceeds local_experts")
    if policy not in ("lfu_lru", "lru"):
        raise ValueError("policy must be lfu_lru or lru")
    if not records or set(profiles) != set(records):
        raise ValueError("profiles must cover exactly all traced rank/layer identities")
    ranks = sorted({rank for rank, _ in records})
    if set(rank_budgets) != set(ranks):
        raise ValueError("rank_budgets must cover exactly the traced ranks")
    runtime_costs = getattr(records, "runtime_costs", None)
    if runtime_costs is not None:
        if set(runtime_costs) != set(records):
            raise ValueError("runtime trace costs must cover exactly all traced layers")
        # Validate all measured costs before replaying even the first capacity.
        # Legacy/manual inputs deliberately retain caller-supplied modeled costs.
        for key in sorted(records):
            actual = runtime_costs[key]
            if actual["expert_bytes"] is None:
                raise ValueError(
                    f"missing measured expert_bytes in JSONL trace for {key}"
                )
            if actual["logical_experts"] != local_experts:
                raise ValueError(f"runtime logical_experts mismatch for {key}")
            payload = _integer(profiles[key].get("expert_bytes"), "expert_bytes", 1)
            fixed = _integer(
                profiles[key].get("map_bytes", local_experts * 4), "map_bytes"
            )
            if payload != actual["expert_bytes"] or fixed != actual["map_bytes"]:
                raise ValueError(
                    f"profile costs contradict measured runtime trace for {key}"
                )
    available = sorted(set(_integer(c, "capacity", 1) for c in capacities))
    if not available or available[-1] > local_experts:
        raise ValueError("capacities must be nonempty and <= local_experts")
    curves, costs, weights, choices, bounds, baseline = {}, {}, {}, {}, {}, {}
    for key, calls in sorted(records.items()):
        rank, layer = key
        _integer(rank, "rank")
        _integer(layer, "layer")
        if not calls:
            raise ValueError(f"empty trace for {key}")
        profile = profiles[key]
        unknown = set(profile) - {
            "expert_bytes",
            "map_bytes",
            "min_capacity",
            "max_capacity",
            "capacities",
            "objective_weight",
        }
        if unknown:
            raise ValueError(f"unknown profile fields for {key}: {sorted(unknown)}")
        payload = _integer(profile.get("expert_bytes"), "expert_bytes", 1)
        fixed = _integer(profile.get("map_bytes", local_experts * 4), "map_bytes")
        low = _integer(profile.get("min_capacity", 1), "min_capacity", 1)
        high = _integer(profile.get("max_capacity", local_experts), "max_capacity", 1)
        if low > high or high > local_experts:
            raise ValueError(f"invalid capacity bounds for {key}")
        candidates = sorted(
            set(
                _integer(c, "capacity", 1) for c in profile.get("capacities", available)
            )
        )
        if any(c > local_experts for c in candidates):
            raise ValueError(f"capacity exceeds local_experts for {key}")
        candidates = [c for c in candidates if low <= c <= high]
        if not candidates:
            raise ValueError(f"no available capacity within bounds for {key}")
        weight = profile.get("objective_weight", 1)
        try:
            finite_weight = math.isfinite(weight)
        except (TypeError, OverflowError):
            finite_weight = False
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not finite_weight
            or weight <= 0
        ):
            raise ValueError(f"objective_weight must be finite and positive for {key}")
        weights[key] = Fraction(str(weight))
        costs[key] = payload, fixed
        bounds[key] = low, high
        curves[key] = {
            capacity: simulate(calls, capacity, policy, local_experts, decay_interval)
            for capacity in candidates
        }
        baseline[key] = curves[key].get(baseline_capacity) or simulate(
            calls, baseline_capacity, policy, local_experts, decay_interval
        )
        choices[key] = candidates[0]

    per_rank = {}
    for rank in ranks:
        budget = _integer(rank_budgets[rank], "rank budget", 1)
        keys = [key for key in sorted(records) if key[0] == rank]
        fixed_bytes = sum(costs[key][1] for key in keys)
        used = fixed_bytes + sum(costs[key][0] * choices[key] for key in keys)
        if used > budget:
            raise ValueError(
                f"rank {rank} minimum plan needs {used} bytes, budget is {budget}"
            )
        while True:
            best = None
            for key in keys:
                current = choices[key]
                for candidate, counts in sorted(curves[key].items()):
                    extra = (candidate - current) * costs[key][0]
                    if extra <= 0 or used + extra > budget:
                        continue
                    gain = (
                        (curves[key][current]["misses"] - counts["misses"])
                        * costs[key][0]
                        * weights[key]
                    )
                    if gain <= 0:
                        continue
                    # Density ties prefer larger gains, then lower layer ID
                    # and smaller capacity jumps.
                    score = (gain / extra, gain, -key[1], -candidate)
                    if best is None or score > best[0]:
                        best = score, key, candidate, extra
            if best is None:
                break
            _, key, candidate, extra = best
            choices[key] = candidate
            used += extra
        copied = sum(
            curves[key][choices[key]]["misses"] * costs[key][0] for key in keys
        )
        baseline_copied = sum(baseline[key]["misses"] * costs[key][0] for key in keys)
        baseline_bytes = fixed_bytes + sum(
            costs[key][0] * baseline_capacity for key in keys
        )
        baseline_available = all(baseline_capacity in curves[key] for key in keys)
        per_rank[str(rank)] = dict(
            budget_bytes=budget,
            used_bytes=used,
            payload_bytes=used - fixed_bytes,
            map_bytes=fixed_bytes,
            unused_bytes=budget - used,
            modeled_copy_bytes=copied,
            weighted_miss_bytes=float(
                sum(
                    curves[key][choices[key]]["misses"] * costs[key][0] * weights[key]
                    for key in keys
                )
            ),
            layer_slots={str(key[1]): choices[key] for key in keys},
            baseline=dict(
                capacity=baseline_capacity,
                used_bytes=baseline_bytes,
                feasible=baseline_bytes <= budget and baseline_available,
                capacities_available=baseline_available,
                modeled_copy_bytes=baseline_copied,
            ),
            gpu_bytes_saved_vs_baseline=baseline_bytes - used,
            modeled_copy_bytes_saved_vs_baseline=baseline_copied - copied,
        )
    return dict(
        version=1,
        runtime_costs_verified=runtime_costs is not None,
        cost_scope=(
            "JSONL expert payload and fixed int32 slot-map costs verified"
            if runtime_costs is not None
            else "Caller-supplied modeled costs; runtime allocations unverified"
        ),
        trace_phase=getattr(records, "trace_phase", "unspecified"),
        excluded_trace_phases=getattr(records, "excluded_phases", ()),
        policy=policy,
        decay_interval=decay_interval,
        algorithm="greedy measured weighted-miss-byte reduction per added GPU byte",
        scope=(
            "Only supplied traced layers/ranks; empty initial caches, "
            "supplied recorded routes, "
            "all demanded experts and overflow passes retained. "
            "Excludes untraced model "
            "weights, KV, graphs, unrecorded phases, allocator overhead and overlap. "
            "Heuristic, not a global optimum or latency prediction."
        ),
        local_experts=local_experts,
        layer_slots={rank: result["layer_slots"] for rank, result in per_rank.items()},
        per_rank=per_rank,
        per_layer=[
            dict(
                rank=key[0],
                layer=key[1],
                expert_bytes=costs[key][0],
                map_bytes=costs[key][1],
                min_capacity=bounds[key][0],
                max_capacity=bounds[key][1],
                objective_weight=float(weights[key]),
                selected_capacity=choices[key],
                curve=[
                    dict(
                        capacity=c,
                        payload_bytes=c * costs[key][0],
                        modeled_copy_bytes=counts["misses"] * costs[key][0],
                        **counts,
                    )
                    for c, counts in sorted(curves[key].items())
                ],
            )
            for key in sorted(records)
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacities", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument(
        "--policies", choices=["lfu_lru", "lru"], nargs="+", default=["lfu_lru", "lru"]
    )
    parser.add_argument("--payload-bytes", type=int)
    parser.add_argument("--decay-interval", type=int, default=128)
    parser.add_argument(
        "--plan-profile",
        type=Path,
        help="JSON layers [{rank,layer,expert_bytes,...}], rank_budgets {rank:bytes}",
    )
    parser.add_argument(
        "--plan-output", type=Path, help="separate nonuniform plan JSON"
    )
    parser.add_argument("--plan-policy", choices=["lfu_lru", "lru"], default="lfu_lru")
    parser.add_argument("--baseline-capacity", type=int, default=96)
    parser.add_argument(
        "--trace-phase",
        choices=["prefill", "decode", "profile"],
        help="required for mixed-phase JSONL directories",
    )
    args = parser.parse_args()
    if bool(args.plan_profile) != bool(args.plan_output):
        parser.error("--plan-profile and --plan-output must be supplied together")
    if args.payload_bytes is not None and args.payload_bytes <= 0:
        parser.error("--payload-bytes must be positive")
    records = read_calls(args.directory, phase=args.trace_phase)
    if args.plan_profile:
        profiles, budgets = read_plan_profile(args.plan_profile)
        plan = plan_slots(
            records,
            profiles,
            budgets,
            args.capacities,
            args.plan_policy,
            args.baseline_capacity,
            args.decay_interval,
        )
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(json.dumps(plan, indent=2) + "\n")
    report = sweep(
        records,
        args.capacities,
        args.policies,
        args.payload_bytes,
        args.decay_interval,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["scope"])
    for result in report["results"]:
        totals = result["totals"]
        print(
            f"{result['policy']:7} slots={result['capacity']:3} "
            f"copies={totals['misses']:7} hits={totals['hits']:7} "
            f"passes={totals['passes']:6} hit_fraction={result['hit_fraction']}"
        )


if __name__ == "__main__":
    main()
