# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan command line: what the launcher, the recipes and an operator ask of a plan.

Run it by path, not with ``-m``: ``python vllm/distributed/hetero_moe/cli.py
<command> ...`` loads only the numpy-level modules of this package (plan,
tables), so it never imports the vLLM stack, ``vllm.platforms`` or amdsmi and
cannot wake a GPU (``tools/rdna/rdna_hetero_launch.sh`` relies on that).

Commands (all print plain text; an unusable plan exits with status 1 and the
reason on stderr):

``validate PLAN``
    Load and validate the plan (ExpertStats' checker, runtime limits, content
    hash) and print its one-line description.
``hubs PLAN`` / ``servers PLAN``
    One line per hub / expert device in plan order, tab separated:
    ``island id hip switch arch``.
``buckets PLAN [--spill-k TABLE]``
    The capture widths of island 0's spill table as ``3,6,12,24,48``: what the
    servers' ``--buckets`` must be (``TABLE`` is ``VLLM_HETERO_MOE_SPILL_K``).
``retarget PLAN --out OUT --arch ID=ARCH ...``
    Copy of the plan with some devices' arch replaced (the V620 stand-in hub).
``stripe --out PLAN [--hub-hips 0] [--expert-hips 1,2,3,4] [--hub-arch ARCH]
[-- SOLVER...]``
    The uniform-stripe plan of one island (H1 by default) from ExpertStats'
    solver without statistics; arguments after ``--`` go to the solver.
"""

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PKG = "_hetero_moe_cli"

# HeteroDesign's H1 solver command (spec 6.11): one island, one hub, four
# servers, the checkpoint's bf16 drafter, the section 7.2 cost constants.
H1_SOLVER_ARGS = (
    "--islands",
    "1",
    "--hubs-per-island",
    "1",
    "--island-mode",
    "replicate",
    "--hub-active-gib",
    "9.4",
    "--hub-vram-gib",
    "44.98",
    "--hub-reserve-gib",
    "12.1",
    "--expert-budget-gib",
    "26",
    "--max-hub-per-layer",
    "130",
    "--mtp-policy",
    "hub",
    "--mtp-expert-bytes",
    "9830400",
    "--hub-us",
    "3.7",
    "--expert-us",
    "6.7",
    "--hub-fixed-us",
    "30.8",
    "--expert-fixed-us",
    "3.1",
    "--latency-us",
    "13.3",
    "--gbps",
    "25",
)


def _package():
    """This package under a private name: no ``vllm`` import."""
    if PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            PKG, HERE / "__init__.py", submodule_search_locations=[str(HERE)]
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[PKG] = module
        spec.loader.exec_module(module)
    return sys.modules[PKG]


def _module(name: str):
    _package()
    return importlib.import_module(f"{PKG}.{name}")


def _devices(plan, devices_of) -> list[str]:
    lines = []
    for island in range(plan.num_islands):
        for dev in devices_of(island):
            d = plan.doc["devices"][dev]
            lines.append(
                f"{island}\t{d['id']}\t{d['hip']}\t{d.get('switch', '-')}\t{d['arch']}"
            )
    return lines


def _stripe(args: argparse.Namespace, rest: list[str]) -> int:
    plan_mod = _module("plan")
    solver = list(H1_SOLVER_ARGS)
    expert_hips = args.expert_hips
    solver += [
        "--hub-hips",
        args.hub_hips,
        "--expert-hips",
        expert_hips,
        "--experts-per-island",
        str(len(expert_hips.split(","))),
        "--switches",
        args.switches,
        "--stripe",
        str(args.stripe),
    ]
    solver += rest
    sys.path.insert(0, str(ROOT))
    plan = plan_mod.solve_plan(args.out, solver)
    if args.hub_arch:
        hubs = {plan.device_id(dev): args.hub_arch for dev in plan.hubs(0)}
        doc = plan_mod.with_device_arch(plan.doc, hubs)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
        plan = plan_mod.load_plan(args.out)
    print(plan.describe())
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, rest = argv[:cut], argv[cut + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "hubs", "servers"):
        sub.add_parser(name).add_argument("plan")
    p = sub.add_parser("buckets")
    p.add_argument("plan")
    p.add_argument("--spill-k", default="")
    p = sub.add_parser("retarget")
    p.add_argument("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--arch", action="append", default=[], metavar="ID=ARCH")
    p = sub.add_parser("stripe")
    p.add_argument("--out", required=True)
    p.add_argument("--hub-hips", default="0")
    p.add_argument("--expert-hips", default="1,2,3,4")
    p.add_argument("--switches", default="c0")
    p.add_argument("--stripe", type=int, default=4)
    p.add_argument("--hub-arch", default="")
    args = parser.parse_args(argv)
    plan_mod = _module("plan")
    try:
        if args.command == "stripe":
            return _stripe(args, rest)
        if rest:
            parser.error("arguments after -- belong to `stripe`")
        if args.command == "retarget":
            plan = plan_mod.load_plan(args.plan)
            arch = dict(a.split("=", 1) for a in args.arch)
            doc = plan_mod.with_device_arch(plan.doc, arch)
            plan_mod.plan_from_doc(doc)  # validate before writing
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=1)
            return 0
        plan = plan_mod.load_plan(args.plan)
        if args.command == "validate":
            print(plan.describe())
        elif args.command == "hubs":
            print("\n".join(_devices(plan, plan.hubs)))
        elif args.command == "servers":
            print("\n".join(_devices(plan, plan.experts)))
        else:  # buckets
            tables = _module("tables")
            spill = tables.parse_spill_k(args.spill_k) if args.spill_k else None
            island = tables.build_island_tables(plan, plan.hubs(0)[0], spill_k=spill)
            print(",".join(str(w) for w in island.widths))
    except plan_mod.PlanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
