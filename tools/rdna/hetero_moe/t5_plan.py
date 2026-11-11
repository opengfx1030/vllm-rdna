# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-1 placement plans of the T5 harness (one island: 1 hub + 4 expert servers).

    python -m tools.rdna.hetero_moe.t5_plan --kind stripe --out plan.json \\
        [--hub-hip 0 --expert-hips 1,2,3,4 --switch c0 --spill-k 3:0,24:0]

Both kinds describe the Flash-Next model (48 target layers, 512 experts, top-10) on
devices ``hub0``, ``exp0``..``exp3`` and add the MTP layer (48) as a hub-only layer
that holds all 512 experts (like ``solve.py --mtp-policy hub``). Every expert of the
target layers has a server copy (``cold_complete``), so the servers can serve any
pair.

``stripe``     the hub keeps the stripe-4 set (``e % 4 == 0``, 128 experts per layer);
               ``runtime.spill_k`` is K = 0 at every capture width, so at the decode
               widths every pair still goes to a server. Rows above the largest
               capture width keep the hub-resident pairs on the hub (prefill rule).
``coldonly``   the hub holds no target-layer expert: every pair goes to a server.

The document is checked with ``validate.validate_plan`` before it is written.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.rdna.hetero_moe import model_bytes
from tools.rdna.hetero_moe.placement import (
    LayerPlacement,
    SolverConfig,
    Topology,
    fallback_plan,
    plan_document,
)
from tools.rdna.hetero_moe.solve import DEFAULT_TABLE, GIB
from tools.rdna.hetero_moe.validate import validate_plan

KINDS = ("stripe", "coldonly")
CAPTURE_WIDTHS = (3, 6, 12, 24, 48)
# Budgets as `solve.py` derives them (W7800 48 GB, 12.1 GiB reserve; V620 26 GiB).
HUB_VRAM_GIB = 44.98
HUB_RESERVE_GIB = 12.1
EXPERT_BUDGET_GIB = 26.0
TOP_K = 10


def parse_hips(text: str) -> list[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def build_doc(
    kind: str,
    hub_hip: int,
    expert_hips: list[int],
    switch: str,
    spill_k: dict[int, int] | None = None,
    table_path: str = DEFAULT_TABLE,
) -> dict:
    """The plan document of ``kind``; ``spill_k`` (capture width -> K) defaults to
    K = 0 at every width."""
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r}: choose from {KINDS}")
    table = model_bytes.load(table_path)
    n_layers, n_experts = table["num_layers"], table["num_experts"]
    topo = Topology(
        islands=1,
        n_hubs=1,
        n_experts=len(expert_hips),
        hub_hips=[[hub_hip]],
        expert_hips=[list(expert_hips)],
        switches=[switch],
    )
    hub_budget = (
        round(HUB_VRAM_GIB * GIB)
        - table["hub_active_bytes"]
        - round(HUB_RESERVE_GIB * GIB)
    )
    cfg = SolverConfig(
        expert_bytes=table["expert_bytes"],
        hub_budget_bytes=hub_budget,
        expert_budget_bytes=round(EXPERT_BUDGET_GIB * GIB),
        mtp_expert_bytes=table["expert_bytes"],
        max_hub_per_layer=0 if kind == "coldonly" else 130,
        cold_complete=True,
        mtp_policy="hub",
    )
    placements = fallback_plan(n_layers, n_experts, topo, cfg)
    placements[n_layers] = LayerPlacement(
        n_layers,
        "mtp",
        hub=[list(range(n_experts))],
        owner=-np.ones(n_experts, dtype=np.int64),
        replicas={},
        hub_only=True,
    )
    model = {
        "name": table.get("model"),
        "num_layers": n_layers,
        "num_experts": n_experts,
        "top_k": TOP_K,
        "mtp_layers": 1,
    }
    estimate = {"plan": {"analytic": {"note": f"T5 {kind} plan, no statistics"}}}
    meta = {"t5_kind": kind, "mtp_policy": "hub"}
    doc = plan_document(placements, topo, cfg, model, estimate, f"t5-{kind}", [], meta)
    k_table = spill_k or dict.fromkeys(CAPTURE_WIDTHS, 0)
    doc["runtime"] = {"spill_k": {str(w): int(k) for w, k in sorted(k_table.items())}}
    return doc


def parse_spill_k(text: str) -> dict[int, int]:
    out = {}
    for item in text.replace(" ", "").split(","):
        width, sep, k = item.partition(":")
        if not (sep and width.isdigit() and k.isdigit()):
            raise argparse.ArgumentTypeError(f"bad spill K entry {item!r}: WIDTH:K")
        out[int(width)] = int(k)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hub-hip", type=int, default=0)
    p.add_argument("--expert-hips", type=parse_hips, default=[1, 2, 3, 4])
    p.add_argument("--switch", default="c0")
    p.add_argument(
        "--spill-k",
        type=parse_spill_k,
        default=None,
        help='runtime.spill_k, "3:10,24:20" (default K = 0 at every capture width)',
    )
    p.add_argument("--bytes", dest="table", default=DEFAULT_TABLE)
    args = p.parse_args(argv)
    doc = build_doc(
        args.kind, args.hub_hip, args.expert_hips, args.switch, args.spill_k, args.table
    )
    errors = validate_plan(doc)
    if errors:
        for line in errors[:20]:
            print(line, file=sys.stderr)
        print(f"T5 plan FAIL {len(errors)} violation(s)")
        return 1
    Path(args.out).write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    n_hub = sum(
        1
        for row in doc["layers"][0]["owners"]
        if row and row[0] == doc["islands"][0][0]
    )
    print(
        f"T5 plan PASS kind={args.kind} plan_id={doc['plan_id']} "
        f"layers={len(doc['layers'])} hub_experts_layer0={n_hub} -> {args.out}"
    )
    print(doc["plan_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
