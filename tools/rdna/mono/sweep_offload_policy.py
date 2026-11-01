# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline, empty-cache decode-trace policy comparison; no GPU execution."""

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_policy():
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm/model_executor/layers/fused_moe/rdna_expert_cache_policy.py"
    )
    spec = importlib.util.spec_from_file_location("_sweep_expert_policy", path)
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


def read_calls(directory, local_experts=256):
    import torch

    records = defaultdict(list)
    seen = set()
    for path in sorted(Path(directory).glob("rank*-layer*-call*.pt")):
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
    return {key: [ids for _, ids in sorted(values)] for key, values in records.items()}


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
    return dict(
        scope=(
            "decode-only short trace, empty initial caches; excludes prefill "
            "residency, timing, and transfer overlap; informational, "
            "not workload optimality; real rows only, excludes historical "
            "padding demand and is not physical residency replay"
        ),
        decay_interval=decay_interval,
        max_calls_per_rank_layer=max(map(len, records.values())),
        decay_exercised=any(len(calls) >= decay_interval for calls in records.values()),
        payload_bytes_per_expert=payload_bytes,
        results=results,
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
    args = parser.parse_args()
    if args.payload_bytes is not None and args.payload_bytes <= 0:
        parser.error("--payload-bytes must be positive")
    report = sweep(
        read_calls(args.directory),
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
