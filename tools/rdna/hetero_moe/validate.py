# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Invariant checker for ``hetero-moe-placement/1`` plans.

    python -m tools.rdna.hetero_moe.validate PLAN.json [--stats STATS]

Checks (exit status 1 and one line per violation on failure):

* format/version, device records, island partition (same shape and switch per
  island, devices of an island on one switch);
* every layer lists ``num_experts`` owner lists of valid, duplicate-free island-0
  device indices, hub devices first, then expert devices (primary first,
  replicas ascending);
* with ``cold_complete``: every expert of every non-hub-only layer has exactly
  one primary and distinct replicas on expert devices, so each expert is placed
  exactly once per island plus its replicas; primaries are balanced (a device
  holds ``E / D`` primaries per layer, remainder spread by at most one);
* ``island_mode`` ``partition`` (default ``replicate``): owners are global device
  indices, every expert is owned inside exactly one island, island expert counts
  per layer differ by at most one and primaries are balanced per island;
* hub lists: a hub device only lists hot experts it holds, the hub and the
  V620 byte budgets are respected;
* with ``--stats``: every statistics layer (except an omitted MTP layer) has a
  plan entry.
"""

import argparse
import json
import sys
from collections import Counter

PLAN_FORMAT = "hetero-moe-placement/1"


def validate_plan(doc: dict, record=None) -> list[str]:  # noqa: C901
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(msg)

    if doc.get("format") != PLAN_FORMAT or doc.get("version") != 1:
        err(f"format/version must be {PLAN_FORMAT} / 1")
        return errors
    devices = doc.get("devices", [])
    islands = doc.get("islands", [])
    if not devices or not islands:
        err("devices and islands are required")
        return errors
    ids = [d.get("id") for d in devices]
    if len(set(ids)) != len(ids):
        err("device ids are not unique")
    hips = [d.get("hip") for d in devices]
    if len(set(hips)) != len(hips):
        err("device HIP indices are not unique")
    for d in devices:
        if d.get("role") not in ("hub", "expert"):
            err(f"device {d.get('id')}: role must be hub or expert")
        if (
            not isinstance(d.get("expert_budget_bytes"), int)
            or d["expert_budget_bytes"] < 0
        ):
            err(f"device {d.get('id')}: bad expert_budget_bytes")
    seen = [i for group in islands for i in group]
    if sorted(seen) != list(range(len(devices))):
        err("islands must partition the device indices exactly once")
        return errors
    shapes = {tuple(devices[i]["role"] for i in group) for group in islands}
    if len(shapes) != 1:
        err("islands differ in device roles/order")
    for k, group in enumerate(islands):
        if len({devices[i]["switch"] for i in group}) != 1:
            err(f"island {k} spans PCIe switches")
    if errors:
        return errors

    mode = doc.get("island_mode", "replicate")
    if mode not in ("replicate", "partition"):
        err(f"island_mode must be replicate or partition, not {mode!r}")
        return errors
    partition = mode == "partition"
    island_of_dev = {d: k for k, group in enumerate(islands) for d in group}
    hub_idx_of = [[i for i in g if devices[i]["role"] == "hub"] for g in islands]
    exp_idx_of = [[i for i in g if devices[i]["role"] == "expert"] for g in islands]
    if not all(exp_idx_of):
        err("island has no expert device")
        return errors
    model = doc.get("model", {})
    num_experts = model.get("num_experts")
    expert_bytes = doc.get("expert_bytes")
    mtp_bytes = model.get("mtp_expert_bytes", expert_bytes)
    if not isinstance(num_experts, int) or not isinstance(expert_bytes, int):
        err("model.num_experts and expert_bytes are required")
        return errors
    cold_complete = bool(doc.get("cold_complete"))
    hub_mode = doc.get("hub_mode", "replicate")
    layer_ids = [x["layer"] for x in doc["layers"]]
    if len(set(layer_ids)) != len(layer_ids):
        err("duplicate layer entries")
    hub_bytes = Counter()
    exp_bytes = Counter()
    for layer in doc["layers"]:
        lid = layer["layer"]
        size = mtp_bytes if layer.get("kind") == "mtp" else expert_bytes
        owners = layer["owners"]
        if len(owners) != num_experts:
            err(f"layer {lid}: {len(owners)} owner lists, expected {num_experts}")
            continue
        primaries = Counter()
        per_island = Counter()
        for e, row in enumerate(owners):
            if len(set(row)) != len(row):
                err(f"layer {lid} expert {e}: duplicate owner")
                continue
            if partition:
                spans = {island_of_dev.get(d) for d in row}
                if None in spans or len(spans) > 1:
                    err(
                        f"layer {lid} expert {e}: owners must lie in exactly one island"
                    )
                    continue
                k = spans.pop() if spans else 0
            else:
                k = 0
                bad = [d for d in row if d not in islands[0]]
                if bad:
                    err(f"layer {lid} expert {e}: owner {bad} outside island 0")
                    continue
            hub_idx, exp_idx = hub_idx_of[k], exp_idx_of[k]
            hubs = [d for d in row if d in hub_idx]
            cold = [d for d in row if d in exp_idx]
            if row != hubs + cold:
                err(f"layer {lid} expert {e}: hubs must precede expert devices")
            if cold and cold[1:] != sorted(cold[1:]):
                err(f"layer {lid} expert {e}: replicas must be ascending")
            per_island[k] += 1
            share = size // len(hub_idx) if hub_mode == "shard" else size
            if layer.get("hub_only"):
                if cold or not hubs:
                    err(f"layer {lid} expert {e}: hub_only layer needs hub owners only")
                for d in hubs:
                    hub_bytes[d] += share
                continue
            if cold_complete and not cold:
                err(f"layer {lid} expert {e}: no expert-device copy (cold_complete)")
            if not hubs and not cold:
                err(f"layer {lid} expert {e}: placed nowhere")
            if cold:
                primaries[cold[0]] += 1
            for d in cold:
                exp_bytes[d] += size
            for d in hubs:
                hub_bytes[d] += share
        if layer.get("hub_only"):
            continue
        if (
            partition
            and per_island
            and max(per_island.values())
            - min(per_island.get(k, 0) for k in range(len(islands)))
            > 1
        ):
            err(f"layer {lid}: experts unbalanced over islands {dict(per_island)}")
        if cold_complete and primaries:
            for k in range(len(islands) if partition else 1):
                counts = [primaries.get(d, 0) for d in exp_idx_of[k]]
                want = per_island.get(k, 0) if partition else num_experts
                if max(counts) - min(counts) > 1 or sum(counts) != want:
                    err(f"layer {lid}: unbalanced primaries {counts}")
    for d, dev in enumerate(devices):
        used = hub_bytes if dev["role"] == "hub" else exp_bytes
        if used[d] > dev["expert_budget_bytes"]:
            what = "hub" if dev["role"] == "hub" else "expert"
            err(f"{what} device {d}: {used[d]} B exceeds its budget")
    if record is not None:
        have = set(layer_ids)
        for li, ls in record.layers.items():
            if li not in have and ls.kind != "mtp":
                err(f"stats layer {li} missing from the plan")
    return errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("plan")
    p.add_argument("--stats", nargs="*")
    args = p.parse_args(argv)
    with open(args.plan, encoding="utf-8") as f:
        doc = json.load(f)
    record = None
    if args.stats:
        from .stats_io import load_stats

        record = load_stats(args.stats)
    errors = validate_plan(doc, record)
    for line in errors:
        print(line, file=sys.stderr)
    print("plan OK" if not errors else f"{len(errors)} violation(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
