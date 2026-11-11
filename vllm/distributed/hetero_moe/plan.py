# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Placement plan of the heterogeneous MoE runtime (spec section 9.2).

A ``hetero-moe-placement/1`` plan says, per MoE layer and expert, which device
holds the expert: the hub (W7800, hot experts, slot pool) and/or the expert
servers (V620, cold copies). ``tools/rdna/hetero_moe`` writes and validates it;
this module loads it for the runtime, re-validates it with the same checker and
answers the questions the hub and the servers ask:

* ``hub_resident(hub, layer)``: the experts in the hub slot pool, in slot order;
* ``server_experts(server, layer)``: the experts a server holds, ascending, which
  is the server's local slot order;
* ``cold_copies(layer, island)``: per expert the island servers that hold a copy,
  in the plan's owner order (primary first, replicas ascending).

``island_mode`` ``replicate`` (stage 1, H1 is one island): ``owners`` are written
for island 0 and island ``k`` maps every index by its position inside the island.
``partition`` (H2): ``owners`` are global device indices and an expert that is
not owned by an island is dropped by that island (the other island computes it).

Standalone (numpy only; no torch, no vLLM import) so plans are usable from the
launcher and from tests without a GPU stack.
"""

import copy
import hashlib
import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import regex as re

PLAN_FORMAT = "hetero-moe-placement/1"
MAX_COLD = 8  # HX_MAX_COLD: expert-device copies per expert encoded in the tables
MAX_SERVERS = 8  # expert servers per hub
MAX_LAYERS = 64  # plan layers (targets + MTP)

# Cold-copy rule of the decode steps (user decision 16, spec C.5). "row" is the plan
# format's (row + layer) % n_cold on every step (as built); "primary" sends a cold
# pair to the first listed copy of its expert on graph steps at the capture widths
# (eager and prefill steps keep the row rule). The plans of ExpertStats also carry a
# third table, "expert", which the hub does not offer.
DECODE_DISPATCH_RULES = ("row", "primary")

_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_MTP_SEGMENT = re.compile(r"(?:^|\.)mtp(?:\.|$)")


class PlanError(ValueError):
    """The plan file is unusable: format, invariants or this process' role."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_plan_fn():
    """``tools.rdna.hetero_moe.validate.validate_plan`` (ExpertStats' checker)."""
    try:
        from tools.rdna.hetero_moe import validate
    except ImportError:
        path = repo_root() / "tools" / "rdna" / "hetero_moe" / "validate.py"
        if not path.is_file():
            raise PlanError(
                "plan validation needs tools/rdna/hetero_moe/validate.py from the "
                "repository checkout (not found next to this vllm package)"
            ) from None
        spec = importlib.util.spec_from_file_location("_hetero_moe_validate", path)
        assert spec is not None and spec.loader is not None
        validate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validate)
    return validate.validate_plan


def compute_plan_id(doc: dict) -> str:
    """Content hash of ``layers`` + ``devices`` (the solver's ``plan_id`` rule)."""
    payload = {"layers": doc["layers"], "devices": doc["devices"]}
    blob = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def plan_layer_id(layer_name: str, num_target_layers: int) -> int | None:
    """Plan layer of a module name: ``layers.N`` for the target model, and
    ``num_target_layers + N`` for the MTP draft (the stats collector's rule).
    ``None`` when the name carries no layer index."""
    match = _LAYER_INDEX.search(layer_name)
    if match is None:
        return None
    index = int(match.group(1))
    if _MTP_SEGMENT.search(layer_name):
        return num_target_layers + index
    return index


class HeteroPlan:
    """Read-only view of one validated plan; every array is int32 numpy."""

    def __init__(self, doc: dict, path: str | None = None):
        self.doc = doc
        self.path = path
        self.stored_plan_id: str | None = doc.get("plan_id")
        self.plan_id = compute_plan_id(doc)
        self.devices: list[dict] = doc["devices"]
        self.islands: list[list[int]] = [list(g) for g in doc["islands"]]
        self.island_mode: str = doc.get("island_mode", "replicate")
        self.hub_mode: str = doc.get("hub_mode", "replicate")
        self.cold_complete = bool(doc.get("cold_complete"))
        model = doc.get("model", {})
        self.num_experts = int(model["num_experts"])
        self.top_k = int(model["top_k"])
        self.expert_bytes = int(doc["expert_bytes"])
        self._layers = {int(x["layer"]): x for x in doc["layers"]}
        self.layer_ids = sorted(self._layers)
        self.target_layers = [
            li for li in self.layer_ids if self._layers[li].get("kind") != "mtp"
        ]
        self.num_layers = int(model.get("num_layers", len(self.target_layers)))
        self._island_of = {d: k for k, g in enumerate(self.islands) for d in g}
        self._ids = {d["id"]: i for i, d in enumerate(self.devices)}
        self._views: dict[tuple[int, int], list[list[int]]] = {}

    # ---- devices -------------------------------------------------------------

    def device_index(self, dev_id: str) -> int:
        try:
            return self._ids[dev_id]
        except KeyError:
            raise PlanError(
                f"device {dev_id!r} is not in plan {self.plan_id} "
                f"(devices: {', '.join(self._ids)})"
            ) from None

    def device_id(self, dev: int) -> str:
        return self.devices[dev]["id"]

    def role(self, dev: int) -> str:
        return self.devices[dev]["role"]

    def arch(self, dev: int) -> str:
        return self.devices[dev]["arch"]

    def island_of(self, dev: int) -> int:
        return self._island_of[dev]

    @property
    def num_islands(self) -> int:
        return len(self.islands)

    def hubs(self, island: int) -> list[int]:
        return [d for d in self.islands[island] if self.devices[d]["role"] == "hub"]

    def experts(self, island: int) -> list[int]:
        """Expert devices of an island in plan order (= server index order)."""
        return [d for d in self.islands[island] if self.devices[d]["role"] == "expert"]

    def server_index(self, dev: int) -> int:
        """Position of an expert device among its island's expert devices."""
        return self.experts(self.island_of(dev)).index(dev)

    # ---- layers --------------------------------------------------------------

    def has_layer(self, layer: int) -> bool:
        return layer in self._layers

    def layer_kind(self, layer: int) -> str:
        return self._layers[layer].get("kind", "target")

    def is_hub_only(self, layer: int) -> bool:
        return bool(self._layers[layer].get("hub_only"))

    def island_owners(self, layer: int, island: int) -> list[list[int]]:
        """Per expert the GLOBAL device indices of the island's owners (hubs
        first, then expert devices as listed); empty when the island does not
        hold the expert (``partition`` mode)."""
        key = (layer, island)
        view = self._views.get(key)
        if view is not None:
            return view
        rows = self._layers[layer]["owners"]
        if self.island_mode == "partition":
            view = [[d for d in row if self._island_of[d] == island] for row in rows]
        else:
            base, mine = self.islands[0], self.islands[island]
            mapped = {d: mine[i] for i, d in enumerate(base)}
            view = [[mapped[d] for d in row] for row in rows]
        self._views[key] = view
        return view

    def hub_resident(self, hub_dev: int, layer: int) -> np.ndarray:
        """Expert ids in the hub's slot pool, in slot order: ``layers[].hub_order``
        (hot first) when the plan has it, ascending id otherwise."""
        island = self.island_of(hub_dev)
        owners = self.island_owners(layer, island)
        ids = [e for e, row in enumerate(owners) if hub_dev in row]
        order = self._layers[layer].get("hub_order")
        if order is not None:
            rank = {int(e): i for i, e in enumerate(order)}
            ids.sort(key=lambda e: (rank.get(e, len(rank)), e))
        return np.asarray(ids, dtype=np.int32)

    def server_experts(self, dev: int, layer: int) -> np.ndarray:
        """Expert ids a server holds, ascending (its local slot order)."""
        owners = self.island_owners(layer, self.island_of(dev))
        return np.asarray(
            [e for e, row in enumerate(owners) if dev in row], dtype=np.int32
        )

    def cold_copies(self, layer: int, island: int) -> list[list[int]]:
        """Per expert the server indices (inside the island) that hold a copy, in
        owner order: primary first, replicas ascending."""
        servers = {d: i for i, d in enumerate(self.experts(island))}
        owners = self.island_owners(layer, island)
        return [[servers[d] for d in row if d in servers] for row in owners]

    # ---- optional fields -------------------------------------------------------

    def spill_k(self, rule: str | None = None) -> dict[int, int] | None:
        """``runtime.spill_k`` of the plan (bucket -> K) when it carries one. With a
        decode dispatch ``rule``: that rule's table of ``runtime.spill_k_by_dispatch``
        when the plan has one, else ``runtime.spill_k`` (``spill_k_source`` says
        which)."""
        runtime = self.doc.get("runtime") or {}
        raw = None
        if rule is not None:
            raw = (runtime.get("spill_k_by_dispatch") or {}).get(rule)
        raw = raw or runtime.get("spill_k")
        if not raw:
            return None
        return {int(k): int(v) for k, v in raw.items()}

    def spill_k_source(self, rule: str) -> str:
        """Where ``spill_k(rule)`` comes from, for the plan check and the logs."""
        runtime = self.doc.get("runtime") or {}
        if (runtime.get("spill_k_by_dispatch") or {}).get(rule):
            return f"runtime.spill_k_by_dispatch.{rule}"
        if runtime.get("spill_k"):
            declared = runtime.get("spill_k_dispatch") or "row"
            if declared == rule:
                return "runtime.spill_k"
            return (
                f"runtime.spill_k, which the plan declares for {declared} "
                f"(it has no table for {rule})"
            )
        return "the cost-model defaults (the plan has no runtime.spill_k)"

    def describe(self) -> str:
        hubs = sum(len(self.hubs(k)) for k in range(self.num_islands))
        return (
            f"plan {self.plan_id} ({self.island_mode}): {self.num_islands} island(s), "
            f"{hubs} hub(s), {len(self.devices) - hubs} expert device(s), "
            f"{len(self.target_layers)} target + "
            f"{len(self.layer_ids) - len(self.target_layers)} MTP layer(s), "
            f"{self.num_experts} experts top-{self.top_k}"
        )


def _check_runtime_limits(plan: HeteroPlan) -> None:
    errors = []
    # Hub-only layers (the MTP draft, numbered num_target_layers + N by the stats
    # collector) never reach the servers: their ids are not hx table indexes.
    hx_layers = [li for li in plan.layer_ids if not plan.is_hub_only(li)]
    if len(hx_layers) > MAX_LAYERS or (hx_layers and hx_layers[-1] >= MAX_LAYERS):
        errors.append(f"layer ids must be below {MAX_LAYERS}")
    for k in range(plan.num_islands):
        if not plan.hubs(k):
            errors.append(f"island {k} has no hub device")
        if len(plan.experts(k)) > MAX_SERVERS:
            errors.append(f"island {k} has more than {MAX_SERVERS} expert devices")
        for li in plan.layer_ids:
            if plan.is_hub_only(li):
                continue
            cold = plan.cold_copies(li, k)
            if max((len(c) for c in cold), default=0) > MAX_COLD:
                errors.append(f"layer {li}: an expert has more than {MAX_COLD} copies")
                break
    if errors:
        raise PlanError("plan exceeds the runtime limits: " + "; ".join(errors))


def _check_spill_tables(plan: HeteroPlan) -> None:
    """The tables of ``runtime.spill_k_by_dispatch`` (the rules the hub offers) must
    cover the same capture widths as ``runtime.spill_k`` and as each other: the
    widths are the servers' graph buckets and do not depend on the dispatch rule."""
    runtime = plan.doc.get("runtime") or {}
    by_rule = runtime.get("spill_k_by_dispatch") or {}
    widths: dict[str, list[int]] = {}
    if runtime.get("spill_k"):
        widths["runtime.spill_k"] = sorted(int(k) for k in runtime["spill_k"])
    for rule in DECODE_DISPATCH_RULES:
        table = by_rule.get(rule)
        if not table:
            continue
        try:
            widths[f"runtime.spill_k_by_dispatch.{rule}"] = sorted(
                int(k) for k in table
            )
            if min(int(v) for v in table.values()) < 0:
                raise ValueError("negative K")
        except (AttributeError, TypeError, ValueError) as exc:
            raise PlanError(
                f"runtime.spill_k_by_dispatch.{rule} is not a table of capture "
                f"width -> K ({exc})"
            ) from exc
    if len({tuple(w) for w in widths.values()}) > 1:
        shown = "; ".join(f"{name} has widths {w}" for name, w in widths.items())
        raise PlanError(
            f"the spill K tables of the plan disagree on the widths: {shown}"
        )


def plan_from_doc(
    doc: dict,
    *,
    validate: bool = True,
    check_id: bool = True,
    path: str | None = None,
) -> HeteroPlan:
    """Build a plan from a parsed document; raises ``PlanError`` on any problem."""
    if validate:
        errors = _validate_plan_fn()(doc)
        if errors:
            shown = "\n  ".join(errors[:12])
            more = f"\n  ... and {len(errors) - 12} more" if len(errors) > 12 else ""
            raise PlanError(f"invalid plan {path or '<doc>'}:\n  {shown}{more}")
    try:
        plan = HeteroPlan(doc, path)
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanError(f"malformed plan {path or '<doc>'}: {exc!r}") from exc
    if check_id and plan.stored_plan_id not in (None, plan.plan_id):
        raise PlanError(
            f"plan {path or '<doc>'}: stored plan_id {plan.stored_plan_id} does not "
            f"match its content hash {plan.plan_id} (edited after it was solved?)"
        )
    _check_runtime_limits(plan)
    _check_spill_tables(plan)
    return plan


def load_plan(
    path: str | Path, *, validate: bool = True, check_id: bool = True
) -> HeteroPlan:
    """Read, validate and index a plan file."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as exc:
        raise PlanError(f"cannot read plan {path}: {exc}") from exc
    return plan_from_doc(doc, validate=validate, check_id=check_id, path=str(path))


def with_device_arch(doc: dict, arch_by_id: dict[str, str]) -> dict:
    """Copy of a plan document with some devices' ``arch`` replaced and the
    ``plan_id`` recomputed. Used to run the hub role on another GPU generation
    (the V620 stand-in of stage 1) without re-solving."""
    out = copy.deepcopy(doc)
    known = {d["id"] for d in out["devices"]}
    missing = sorted(set(arch_by_id) - known)
    if missing:
        raise PlanError(f"unknown device id(s) {missing}")
    for dev in out["devices"]:
        if dev["id"] in arch_by_id:
            dev["arch"] = arch_by_id[dev["id"]]
    out["plan_id"] = compute_plan_id(out)
    return out


def solve_plan(
    out_path: str | Path, solver_args: Sequence[str] = (), *, validate: bool = True
) -> HeteroPlan:
    """Run ExpertStats' solver CLI in process and load its output.

    Without ``--stats`` among ``solver_args`` it writes the deterministic stripe
    plan (no statistics needed). ``tools/`` must be importable (repository root)."""
    try:
        from tools.rdna.hetero_moe import solve
    except ImportError as exc:
        raise PlanError(f"tools/rdna/hetero_moe is not importable: {exc}") from exc
    status = solve.main([*solver_args, "--out", str(out_path)])
    if status:
        raise PlanError(f"solver exited with status {status}")
    return load_plan(out_path, validate=validate)
