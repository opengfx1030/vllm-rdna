# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Placement solver core: hub (W7800) hot sets, V620 owners and replicas.

Model. One *island* is one PCIe switch: ``n_hubs`` hub devices (gfx1100) and
``n_experts`` expert devices (gfx1030) that serve one full copy of the model.
Per MoE layer ``l`` and expert ``e`` the stats give ``q[l, e]``, the probability
that a decode row selects ``e`` (``counts / rows``; ``sum_e q = top_k``), and the
same for prefill. Every cold (not hub-resident) expert is computed by a V620.

Objective (from the design spec): minimise, over decode steps, the sum over
layers of the maximum over expert devices of the number of *distinct cold
experts touched* on that device, weighting decode 1.0 and prefill 0.05 (prefill
only pushes the hub set and the balance toward its own hot experts). The solver
minimises it in four stages, each optimal or greedy for its sub-problem:

1. **Hub hot set**: every expert has the same size, so the hub set that removes
   the most expected cold work under a global byte budget is the global top-N
   (layer, expert) pairs by the weighted marginal ``mass = w_d q_dec + w_p q_pre``
   (greedy by ``mass / bytes`` when sizes differ), subject to per-layer bounds
   ``[min_hub_per_layer, max_hub_per_layer]``. Flat layers get fewer slots,
   skewed layers more (water-filling).
2. **V620 primaries**: per layer the cold experts are assigned by LPT (longest
   processing time first) on the expected touched count over the decode step
   shape, under an exact capacity of ``E / D`` experts per device (the hub's
   experts then fill the remaining slots, so with ``cold_complete`` every expert
   has a V620 copy and the hub set can change at runtime without moving V620
   data). With a co-activation matrix, a bounded swap search then lowers
   ``max_d (mu_d + sigma_d)`` where ``sigma_d`` is the row-to-row standard
   deviation of the touched count (positively co-selected experts on one device
   make the maximum worse).
3. **Replicas**: cold experts, hottest first, round by round, get another copy
   on the least loaded expert device that lacks one, until every device's byte
   budget is used. Runtime splits a cold expert's rows uniformly over its cold
   copies, and the load model does the same.
4. **Estimate**: expected hub hit rate, expected touched experts per device and
   the critical path, for the plan and for the stripe baseline.

Without statistics ``fallback_plan`` returns deterministic striping: hub holds
``e % R == 0`` (``R = 4``, 128 experts per layer when the budget allows, else an
evenly spaced stripe), cold owner is ``(e + e // D) % D``, no replicas.
"""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .stats_io import StatsRecord

PLAN_FORMAT = "hetero-moe-placement/1"

HUB_ARCH = "gfx1100"
EXPERT_ARCH = "gfx1030"


@dataclass
class Topology:
    """``islands`` PCIe switches of ``n_hubs`` hubs + ``n_experts`` expert devices."""

    islands: int = 2
    n_hubs: int = 1
    n_experts: int = 4
    hub_hips: list[list[int]] = field(default_factory=list)
    expert_hips: list[list[int]] = field(default_factory=list)
    switches: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        per = self.n_hubs + self.n_experts
        if not self.hub_hips:
            self.hub_hips = [
                [i * per + h for h in range(self.n_hubs)] for i in range(self.islands)
            ]
        if not self.expert_hips:
            self.expert_hips = [
                [i * per + self.n_hubs + d for d in range(self.n_experts)]
                for i in range(self.islands)
            ]
        if not self.switches:
            self.switches = [f"s{i}" for i in range(self.islands)]
        if not (
            len(self.hub_hips)
            == len(self.expert_hips)
            == len(self.switches)
            == self.islands
        ):
            raise ValueError("topology lists must have one entry per island")
        for i in range(self.islands):
            if len(self.hub_hips[i]) != self.n_hubs:
                raise ValueError(f"island {i}: need {self.n_hubs} hub HIP indices")
            if len(self.expert_hips[i]) != self.n_experts:
                raise ValueError(
                    f"island {i}: need {self.n_experts} expert HIP indices"
                )
        hips = [h for group in (*self.hub_hips, *self.expert_hips) for h in group]
        if len(set(hips)) != len(hips):
            raise ValueError("HIP indices must be unique")

    @classmethod
    def brief_default(cls) -> "Topology":
        """par1-cs25: switch c0 = W7800 HIP 0 + V620 HIP 1-4; 00 = HIP 5 + 6-9."""
        return cls(
            islands=2,
            n_hubs=1,
            n_experts=4,
            hub_hips=[[0], [5]],
            expert_hips=[[1, 2, 3, 4], [6, 7, 8, 9]],
            switches=["c0", "00"],
        )

    def devices(self, hub_budget: int, expert_budget: int) -> list[dict]:
        out: list[dict] = []
        hub_n = exp_n = 0
        for i in range(self.islands):
            for h in self.hub_hips[i]:
                out.append(
                    {
                        "id": f"hub{hub_n}",
                        "role": "hub",
                        "arch": HUB_ARCH,
                        "hip": h,
                        "switch": self.switches[i],
                        "expert_budget_bytes": int(hub_budget),
                    }
                )
                hub_n += 1
            for d in self.expert_hips[i]:
                out.append(
                    {
                        "id": f"exp{exp_n}",
                        "role": "expert",
                        "arch": EXPERT_ARCH,
                        "hip": d,
                        "switch": self.switches[i],
                        "expert_budget_bytes": int(expert_budget),
                    }
                )
                exp_n += 1
        return out

    def island_groups(self) -> list[list[int]]:
        per = self.n_hubs + self.n_experts
        return [list(range(i * per, (i + 1) * per)) for i in range(self.islands)]


@dataclass
class SolverConfig:
    expert_bytes: int
    hub_budget_bytes: int  # per hub, for routed experts
    expert_budget_bytes: int  # per expert device, for routed experts
    mtp_expert_bytes: int | None = None
    hub_mode: str = "replicate"  # replicate | shard | partition
    island_mode: str = "replicate"  # replicate | partition
    max_hub_per_layer: int = 160
    min_hub_per_layer: int = 0
    w_decode: float = 1.0
    w_prefill: float = 0.05
    rows_per_step: dict[int, float] | None = None  # None: stats hist, else {3: 1}
    cold_complete: bool = True
    replicate: bool = True
    coact_refine: bool = True
    refine_iters: int = 40
    refine_z: float = 1.0
    mtp_policy: str = "omit"  # omit | hub | solve
    fallback_stripe: int = 4
    num_layers: int | None = None  # fallback only
    num_experts: int | None = None
    top_k: int | None = None


@dataclass
class LayerModel:
    layer: int
    kind: str
    q_dec: np.ndarray  # [E] selection probability per decode row
    q_pre: np.ndarray
    mass: np.ndarray  # hub-selection value
    touch: np.ndarray  # expected distinct touches per decode step (LPT weight)
    var: np.ndarray  # q (1 - q)
    cov: np.ndarray | None  # [E, E] float32 row-level covariance, diag 0
    expert_bytes: int


@dataclass
class LayerPlacement:
    layer: int
    kind: str
    hub: list[list[int]]  # per hub: sorted expert ids
    owner: np.ndarray  # int [E] local expert-device index of the primary, -1 none
    replicas: dict[int, list[int]]  # expert -> extra local expert-device indices
    hub_only: bool = False
    island_of: np.ndarray | None = None  # int [E]; island mode partition only


# ---- step shape ---------------------------------------------------------------


def step_shape(cfg: SolverConfig, record: StatsRecord | None) -> dict[int, float]:
    """Normalised distribution of decode rows per step."""
    dist = cfg.rows_per_step
    if dist is None and record is not None and record.rows_hist:
        dist = record.rows_hist
    if not dist:
        dist = {3: 1.0}
    total = float(sum(dist.values()))
    return {int(k): float(v) / total for k, v in sorted(dist.items()) if v > 0}


def expected_touch(q: np.ndarray, shape: dict[int, float]) -> np.ndarray:
    """E[distinct touches] of an expert with row-probability ``q`` per step."""
    out = np.zeros_like(q, dtype=np.float64)
    for rows, p in shape.items():
        out += p * (1.0 - (1.0 - np.clip(q, 0.0, 1.0)) ** rows)
    return out


# ---- stats -> layer models ----------------------------------------------------


def layer_models(
    record: StatsRecord, cfg: SolverConfig, shape: dict[int, float]
) -> dict[int, LayerModel]:
    models: dict[int, LayerModel] = {}
    for li, ls in record.layers.items():
        dec_rows, pre_rows = ls.decode_tokens, ls.prefill_tokens
        q_dec = ls.decode / dec_rows if dec_rows > 0 else np.zeros_like(ls.decode)
        q_pre = ls.prefill / pre_rows if pre_rows > 0 else np.zeros_like(ls.prefill)
        q_dec = np.clip(q_dec, 0.0, 1.0)
        q_pre = np.clip(q_pre, 0.0, 1.0)
        if dec_rows > 0:
            mass = cfg.w_decode * q_dec + cfg.w_prefill * q_pre
            touch_q = q_dec
        else:  # no decode traffic in this layer: prefill alone decides
            mass = q_pre.copy()
            touch_q = q_pre
        cov = None
        if cfg.coact_refine and record.coact is not None and li in record.coact_layers:
            cov = _covariance(record.coact[record.coact_layers.index(li)], record.top_k)
        nbytes = (
            cfg.mtp_expert_bytes
            if ls.kind == "mtp" and cfg.mtp_expert_bytes
            else cfg.expert_bytes
        )
        models[li] = LayerModel(
            li,
            ls.kind,
            q_dec,
            q_pre,
            mass,
            expected_touch(touch_q, shape),
            touch_q * (1.0 - touch_q),
            cov,
            int(nbytes),
        )
    return models


def _covariance(pairs: np.ndarray, top_k: int) -> np.ndarray | None:
    """Row-level covariance from an upper-triangular pair-count matrix."""
    upper = pairs.astype(np.float64)
    total = upper.sum()
    if total <= 0:
        return None
    rows = total / (top_k * (top_k - 1) / 2)
    joint = (upper + upper.T) / rows
    q = joint.sum(axis=1) / (top_k - 1)
    cov = joint - np.outer(q, q)
    np.fill_diagonal(cov, 0.0)
    return cov.astype(np.float32)


# ---- stage 1: hub hot sets ----------------------------------------------------


def hub_capacity_bytes(cfg: SolverConfig, n_hubs: int) -> int:
    """Bytes of routed experts the hubs of one island can hold together."""
    if cfg.hub_mode == "shard":  # every hub keeps a 1/n slice of each hot expert
        return int(cfg.hub_budget_bytes) * n_hubs
    if cfg.hub_mode == "partition":  # whole experts, disjoint between hubs
        slots = int(cfg.hub_budget_bytes) // cfg.expert_bytes
        return slots * cfg.expert_bytes * n_hubs
    return int(cfg.hub_budget_bytes)


def select_hub(
    models: dict[int, LayerModel], cfg: SolverConfig, n_hubs: int
) -> dict[int, np.ndarray]:
    """Boolean hub mask per layer: global greedy by mass / bytes with per-layer caps."""
    capacity = hub_capacity_bytes(cfg, n_hubs)
    masks = {li: np.zeros(m.mass.shape[0], dtype=bool) for li, m in models.items()}
    used = 0
    counts = {li: 0 for li in models}
    # MTP policy: the draft layer is either all on the hub or solved like the rest.
    for li, m in models.items():
        if m.kind == "mtp" and cfg.mtp_policy == "hub":
            masks[li][:] = True
            counts[li] = m.mass.shape[0]
            used += m.expert_bytes * m.mass.shape[0]
    if used > capacity:
        raise ValueError(
            f"MTP experts need {used} bytes on the hub, budget is {capacity}"
        )
    solved = [
        li
        for li, m in models.items()
        if not (m.kind == "mtp" and cfg.mtp_policy in ("hub", "omit"))
    ]
    # Per-layer minimum first (their best experts), then the global greedy.
    if cfg.min_hub_per_layer:
        for li in solved:
            m = models[li]
            order = np.lexsort((np.arange(m.mass.shape[0]), -m.mass))
            for e in order[: cfg.min_hub_per_layer]:
                if used + m.expert_bytes > capacity:
                    raise ValueError(
                        "hub budget cannot hold min_hub_per_layer experts everywhere"
                    )
                masks[li][e] = True
                counts[li] += 1
                used += m.expert_bytes
    entries = []
    for li in solved:
        m = models[li]
        n = m.mass.shape[0]
        ratio = m.mass / m.expert_bytes
        for e in range(n):
            if not masks[li][e]:
                entries.append((-ratio[e], li, e))
    entries.sort()
    for _, li, e in entries:
        m = models[li]
        if counts[li] >= cfg.max_hub_per_layer:
            continue
        if used + m.expert_bytes > capacity:
            continue
        masks[li][e] = True
        counts[li] += 1
        used += m.expert_bytes
    return masks


def split_hubs(
    mask: np.ndarray,
    mass: np.ndarray,
    n_hubs: int,
    mode: str,
    totals: np.ndarray | None = None,
) -> list[list[int]]:
    """Per-hub sorted expert lists of one layer.

    ``partition`` splits the hot set into disjoint, mass-balanced lists whose sizes
    differ by at most one; ``totals`` (experts held per hub so far, updated here)
    hands the odd expert of each layer to the hub holding the fewest.
    """
    hot = [int(e) for e in np.flatnonzero(mask)]
    if mode != "partition" or n_hubs == 1:
        return [list(hot) for _ in range(n_hubs)]
    if totals is None:
        totals = np.zeros(n_hubs, dtype=np.int64)
    base, rem = divmod(len(hot), n_hubs)
    cap = np.full(n_hubs, base, dtype=np.int64)
    for h in sorted(range(n_hubs), key=lambda x: (totals[x], x))[:rem]:
        cap[h] += 1
    lists: list[list[int]] = [[] for _ in range(n_hubs)]
    loads = np.zeros(n_hubs)
    for e in sorted(hot, key=lambda x: (-mass[x], x)):
        free = [h for h in range(n_hubs) if len(lists[h]) < cap[h]]
        h = min(free, key=lambda x: (loads[x], x))
        lists[h].append(e)
        loads[h] += mass[e]
    totals += cap
    return [sorted(x) for x in lists]


# ---- stage 2: V620 primaries --------------------------------------------------


def _capacities(total: int, devices: int, layer: int) -> np.ndarray:
    """Per-device slot counts summing to ``total``; the remainder rotates by layer."""
    base, rem = divmod(total, devices)
    cap = np.full(devices, base, dtype=np.int64)
    for k in range(rem):
        cap[(layer + k) % devices] += 1
    return cap


def assign_primaries(
    model: LayerModel, hot: np.ndarray, devices: int, cfg: SolverConfig, t_bar: float
) -> np.ndarray:
    n = model.touch.shape[0]
    cold = [e for e in np.argsort(-model.touch, kind="stable") if not hot[e]]
    if cfg.cold_complete:
        cap = _capacities(n, devices, model.layer)
    else:
        cap = _capacities(len(cold), devices, model.layer)
    owner = -np.ones(n, dtype=np.int64)
    counts = np.zeros(devices, dtype=np.int64)
    loads = np.zeros(devices)
    for e in cold:
        free = np.flatnonzero(counts < cap)
        d = int(free[np.lexsort((free, loads[free]))[0]])
        owner[e] = d
        counts[d] += 1
        loads[d] += model.touch[e]
    if model.cov is not None and cfg.refine_iters > 0 and devices > 1:
        _refine(model, owner, cold, devices, cfg, t_bar)
    if cfg.cold_complete:
        for e in np.flatnonzero(hot):
            free = np.flatnonzero(counts < cap)
            spare = cap[free] - counts[free]
            d = int(free[np.lexsort((free, -spare))[0]])
            owner[e] = d
            counts[d] += 1
    return owner


def _refine(
    model: LayerModel,
    owner: np.ndarray,
    cold: list[int],
    devices: int,
    cfg: SolverConfig,
    t_bar: float,
) -> None:
    """Swap cold experts between devices to lower max_d(mu_d + z sigma_d)."""
    cov = model.cov
    assert cov is not None
    idx = np.asarray(cold)
    nc = idx.shape[0]
    if nc < 2:
        return
    sub = cov[np.ix_(idx, idx)].astype(np.float64)
    v = model.var[idx]
    mu_e = model.touch[idx]
    dev = owner[idx].copy()
    S = np.zeros((nc, devices))
    for d in range(devices):
        S[:, d] = sub[:, dev == d].sum(axis=1)
    mu = np.array([mu_e[dev == d].sum() for d in range(devices)])
    var = np.array(
        [
            v[dev == d].sum() + sub[np.ix_(dev == d, dev == d)].sum()
            for d in range(devices)
        ]
    )  # sum over ordered pairs == 2 * sum over unordered pairs
    z = cfg.refine_z

    def score(mu_d, var_d):
        return mu_d + z * np.sqrt(np.maximum(t_bar * var_d, 0.0))

    for _ in range(cfg.refine_iters):
        scores = score(mu, var)
        i = int(np.argmax(scores))
        best = None
        best_max = scores[i] - 1e-9
        a_idx = np.flatnonzero(dev == i)
        for j in range(devices):
            if j == i:
                continue
            b_idx = np.flatnonzero(dev == j)
            if a_idx.size == 0 or b_idx.size == 0:
                continue
            c_ab = sub[np.ix_(a_idx, b_idx)]
            d_mu = mu_e[b_idx][None, :] - mu_e[a_idx][:, None]
            d_var_i = (v[b_idx][None, :] - v[a_idx][:, None]) + 2.0 * (
                -S[a_idx, i][:, None] + S[b_idx, i][None, :] - c_ab
            )
            d_var_j = (v[a_idx][:, None] - v[b_idx][None, :]) + 2.0 * (
                -S[b_idx, j][None, :] + S[a_idx, j][:, None] - c_ab
            )
            s_i = score(mu[i] + d_mu, var[i] + d_var_i)
            s_j = score(mu[j] - d_mu, var[j] + d_var_j)
            others = [scores[d] for d in range(devices) if d not in (i, j)]
            rest = max(others) if others else -np.inf
            new_max = np.maximum(np.maximum(s_i, s_j), rest)
            flat = int(np.argmin(new_max))
            ai, bj = divmod(flat, new_max.shape[1])
            if new_max[ai, bj] < best_max:
                best_max = new_max[ai, bj]
                best = (
                    j,
                    int(a_idx[ai]),
                    int(b_idx[bj]),
                    d_mu[ai, bj],
                    d_var_i[ai, bj],
                    d_var_j[ai, bj],
                )
        if best is None:
            break
        j, a, b, d_mu_v, dvi, dvj = best
        mu[i] += d_mu_v
        mu[j] -= d_mu_v
        var[i] += dvi
        var[j] += dvj
        S[:, i] += sub[:, b] - sub[:, a]
        S[:, j] += sub[:, a] - sub[:, b]
        dev[a], dev[b] = j, i
    owner[idx] = dev


# ---- stage 3: replicas --------------------------------------------------------


def _split_loads(
    touch: np.ndarray,
    owners: dict[int, list[int]],
    devices: int,
    cold_experts: list[int],
) -> np.ndarray:
    loads = np.zeros(devices)
    for e in cold_experts:
        copies = owners[e]
        share = touch[e] / len(copies)
        for d in copies:
            loads[d] += share
    return loads


def add_replicas(
    models: dict[int, LayerModel],
    hot: dict[int, np.ndarray],
    owner: dict[int, np.ndarray],
    devices: int,
    cfg: SolverConfig,
) -> dict[int, dict[int, list[int]]]:
    """Replicas as ``layer -> expert -> extra device list``; fills the byte budget."""
    used = np.zeros(devices, dtype=np.int64)
    for li, own in owner.items():
        for d in range(devices):
            used[d] += int((own == d).sum()) * models[li].expert_bytes
    if (used > cfg.expert_budget_bytes).any():
        raise ValueError(
            "expert devices cannot hold their primary experts: need "
            f"{int(used.max())} bytes on the fullest device, budget "
            f"{cfg.expert_budget_bytes} (try --no-cold-complete)"
        )
    spare = cfg.expert_budget_bytes - used
    copies: dict[int, dict[int, list[int]]] = {}
    loads: dict[int, np.ndarray] = {}
    for li, m in models.items():
        n = m.touch.shape[0]
        copies[li] = {
            e: [int(owner[li][e])]
            for e in range(n)
            if owner[li][e] >= 0 and not hot[li][e]
        }
        loads[li] = _split_loads(m.touch, copies[li], devices, list(copies[li]))
    replicas: dict[int, dict[int, list[int]]] = {li: {} for li in models}
    if not cfg.replicate or devices < 2:
        return replicas
    order = sorted(
        ((li, e) for li in copies for e in copies[li]),
        key=lambda x: (-models[x[0]].touch[x[1]], x[0], x[1]),
    )
    for _ in range(devices - 1):
        progressed = False
        for li, e in order:
            cur = copies[li][e]
            if len(cur) >= devices:
                continue
            size = models[li].expert_bytes
            cand = [d for d in range(devices) if d not in cur and spare[d] >= size]
            if not cand:
                continue
            t = models[li].touch[e]
            ld = loads[li]
            d = min(cand, key=lambda x: (ld[x], x))
            old_share, new_share = t / len(cur), t / (len(cur) + 1)
            for s in cur:
                ld[s] -= old_share - new_share
            ld[d] += new_share
            cur.append(d)
            replicas[li].setdefault(e, []).append(d)
            spare[d] -= size
            progressed = True
        if not progressed:
            break
    return replicas


def split_islands(touch: np.ndarray, islands: int, layer: int) -> np.ndarray:
    """Island of each expert: LPT on expected touches, exact count balance."""
    n = touch.shape[0]
    cap = _capacities(n, islands, layer)
    island = np.empty(n, dtype=np.int64)
    counts = np.zeros(islands, dtype=np.int64)
    loads = np.zeros(islands)
    for e in np.argsort(-touch, kind="stable"):
        free = np.flatnonzero(counts < cap)
        k = int(free[np.lexsort((free, loads[free]))[0]])
        island[e] = k
        counts[k] += 1
        loads[k] += touch[e]
    return island


def _sub_model(m: LayerModel, ids: np.ndarray) -> LayerModel:
    cov = None if m.cov is None else m.cov[np.ix_(ids, ids)]
    return LayerModel(
        m.layer,
        m.kind,
        m.q_dec[ids],
        m.q_pre[ids],
        m.mass[ids],
        m.touch[ids],
        m.var[ids],
        cov,
        m.expert_bytes,
    )


def _solve_partition(
    record: StatsRecord, topo: Topology, cfg: SolverConfig
) -> tuple[dict[int, LayerPlacement], dict[int, LayerModel], dict[int, np.ndarray]]:
    """Island mode ``partition``: each expert lives in exactly one island.

    Per layer the experts are split over the islands (LPT on expected touches,
    equal counts), then every island is solved on its own experts with its own hub
    and V620 budgets: hub hot set, V620 primaries (``cold_complete`` per island),
    replicas.
    """
    shape = step_shape(cfg, record)
    t_bar = sum(k * p for k, p in shape.items())
    models = layer_models(record, cfg, shape)
    if cfg.mtp_policy == "omit":
        models = {li: m for li, m in models.items() if m.kind != "mtp"}
    n_isl = topo.islands
    island_of = {li: split_islands(m.touch, n_isl, li) for li, m in models.items()}
    ids = {
        li: [np.flatnonzero(island_of[li] == k) for k in range(n_isl)] for li in models
    }
    subs: list[dict[int, LayerModel]] = []
    solved: list[tuple[dict, dict, dict]] = []
    for k in range(n_isl):
        sub = {li: _sub_model(m, ids[li][k]) for li, m in models.items()}
        hub = select_hub(sub, cfg, topo.n_hubs)
        owner: dict[int, np.ndarray] = {}
        for li, m in sub.items():
            if m.kind == "mtp" and cfg.mtp_policy == "hub":
                continue
            owner[li] = assign_primaries(m, hub[li], topo.n_experts, cfg, t_bar)
        replicas = add_replicas(
            {li: sub[li] for li in owner},
            {li: hub[li] for li in owner},
            owner,
            topo.n_experts,
            cfg,
        )
        subs.append(sub)
        solved.append((hub, owner, replicas))
    totals = [np.zeros(topo.n_hubs, dtype=np.int64) for _ in range(n_isl)]
    placements: dict[int, LayerPlacement] = {}
    hub_masks: dict[int, np.ndarray] = {}
    for li, m in models.items():
        n = m.mass.shape[0]
        hub_lists: list[list[int]] = [[] for _ in range(topo.n_hubs)]
        owner_all = -np.ones(n, dtype=np.int64)
        replicas_all: dict[int, list[int]] = {}
        mask = np.zeros(n, dtype=bool)
        hub_only = False
        for k in range(n_isl):
            hub, owner, replicas = solved[k]
            g = ids[li][k]
            sub = subs[k][li]
            lists = split_hubs(hub[li], sub.mass, topo.n_hubs, cfg.hub_mode, totals[k])
            for h in range(topo.n_hubs):
                hub_lists[h].extend(int(g[j]) for j in lists[h])
            mask[g[hub[li]]] = True
            if li in owner:
                owner_all[g] = owner[li]
                for j, devs in replicas[li].items():
                    replicas_all[int(g[j])] = devs
            else:
                hub_only = True
        placements[li] = LayerPlacement(
            li,
            m.kind,
            [sorted(x) for x in hub_lists],
            owner_all,
            replicas_all,
            hub_only=hub_only,
            island_of=island_of[li],
        )
        hub_masks[li] = mask
    return placements, models, hub_masks


# ---- solve --------------------------------------------------------------------


def solve(
    record: StatsRecord, topo: Topology, cfg: SolverConfig
) -> tuple[dict[int, LayerPlacement], dict[int, LayerModel], dict[int, np.ndarray]]:
    if cfg.island_mode == "partition":
        return _solve_partition(record, topo, cfg)
    if cfg.island_mode != "replicate":
        raise ValueError(f"unknown island_mode {cfg.island_mode!r}")
    shape = step_shape(cfg, record)
    t_bar = sum(k * p for k, p in shape.items())
    models = layer_models(record, cfg, shape)
    if cfg.mtp_policy == "omit":
        models = {li: m for li, m in models.items() if m.kind != "mtp"}
    hub = select_hub(models, cfg, topo.n_hubs)
    owner: dict[int, np.ndarray] = {}
    placements: dict[int, LayerPlacement] = {}
    hub_totals = np.zeros(topo.n_hubs, dtype=np.int64)
    for li, m in models.items():
        if m.kind == "mtp" and cfg.mtp_policy == "hub":
            continue
        owner[li] = assign_primaries(m, hub[li], topo.n_experts, cfg, t_bar)
    replicas = add_replicas(
        {li: models[li] for li in owner},
        {li: hub[li] for li in owner},
        owner,
        topo.n_experts,
        cfg,
    )
    for li, m in models.items():
        n = m.mass.shape[0]
        if li not in owner:  # MTP on the hub only
            placements[li] = LayerPlacement(
                li,
                m.kind,
                split_hubs(hub[li], m.mass, topo.n_hubs, cfg.hub_mode, hub_totals),
                -np.ones(n, dtype=np.int64),
                {},
                hub_only=True,
            )
            continue
        placements[li] = LayerPlacement(
            li,
            m.kind,
            split_hubs(hub[li], m.mass, topo.n_hubs, cfg.hub_mode, hub_totals),
            owner[li],
            replicas[li],
        )
    return placements, models, hub


# ---- fallback -----------------------------------------------------------------


def fallback_plan(
    num_layers: int, num_experts: int, topo: Topology, cfg: SolverConfig
) -> dict[int, LayerPlacement]:
    """Deterministic striping when no statistics exist.

    ``partition`` island mode: expert ``e`` lives in island ``e % islands`` and the
    stripe is applied to its local index ``e // islands`` inside that island.
    """
    d_n = topo.n_experts
    stripe = max(1, cfg.fallback_stripe)
    partition = cfg.island_mode == "partition"
    if partition and num_experts % topo.islands:
        raise ValueError("experts must divide evenly over the islands")
    n_loc = num_experts // topo.islands if partition else num_experts
    full = n_loc // stripe
    capacity = hub_capacity_bytes(cfg, topo.n_hubs) // cfg.expert_bytes
    per_layer = min(full, cfg.max_hub_per_layer, capacity // max(1, num_layers))
    if cfg.min_hub_per_layer > per_layer:
        raise ValueError("hub budget cannot hold min_hub_per_layer experts everywhere")
    if per_layer == full:
        hot_loc = [e for e in range(n_loc) if e % stripe == 0]
    elif per_layer > 0:
        hot_loc = sorted({int((i + 0.5) * n_loc / per_layer) for i in range(per_layer)})
    else:
        hot_loc = []
    local = (
        np.arange(num_experts) // topo.islands if partition else np.arange(num_experts)
    )
    owner_loc = np.array([(j + j // d_n) % d_n for j in range(n_loc)], dtype=np.int64)
    hot_mask = np.zeros(n_loc, dtype=bool)
    hot_mask[hot_loc] = True
    mask = hot_mask[local]
    owner = owner_loc[local]
    island_of = np.arange(num_experts) % topo.islands if partition else None
    out: dict[int, LayerPlacement] = {}
    hub_totals = np.zeros(topo.n_hubs, dtype=np.int64)
    for layer in range(num_layers):
        out[layer] = LayerPlacement(
            layer,
            "target",
            split_hubs(
                mask, np.ones(num_experts), topo.n_hubs, cfg.hub_mode, hub_totals
            ),
            owner.copy() if cfg.cold_complete else np.where(mask, -1, owner),
            {},
            island_of=island_of,
        )
    return out


# ---- plan document ------------------------------------------------------------


def owners_lists(
    lp: LayerPlacement, topo: Topology, num_experts: int
) -> list[list[int]]:
    """``owners[e]``: device indices; hubs first, then V620s (primary first,
    replicas ascending). Island-0 numbering in ``replicate`` mode; global indices
    of the expert's own island when the layer has ``island_of`` (``partition``)."""
    groups = topo.island_groups()
    hub_sets = [set(x) for x in lp.hub]
    out: list[list[int]] = []
    for e in range(num_experts):
        group = groups[int(lp.island_of[e])] if lp.island_of is not None else groups[0]
        row = [group[h] for h in range(topo.n_hubs) if e in hub_sets[h]]
        if lp.owner[e] >= 0:
            row.append(group[topo.n_hubs + int(lp.owner[e])])
            row.extend(group[topo.n_hubs + d] for d in sorted(lp.replicas.get(e, [])))
        out.append(row)
    return out


def plan_document(
    placements: dict[int, LayerPlacement],
    topo: Topology,
    cfg: SolverConfig,
    model: dict,
    estimate: dict,
    source: str,
    stats_sha256: list[str],
    extra_meta: dict | None = None,
) -> dict:
    num_experts = model["num_experts"]
    layers = []
    for li in sorted(placements):
        lp = placements[li]
        entry: dict = {
            "layer": li,
            "kind": lp.kind,
            "owners": owners_lists(lp, topo, num_experts),
        }
        if lp.hub_only:
            entry["hub_only"] = True
        layers.append(entry)
    doc = {
        "format": PLAN_FORMAT,
        "version": 1,
        "model": model,
        "expert_bytes": cfg.expert_bytes,
        "devices": topo.devices(cfg.hub_budget_bytes, cfg.expert_budget_bytes),
        "islands": topo.island_groups(),
        "cold_complete": bool(cfg.cold_complete),
        "hub_mode": cfg.hub_mode,
        "island_mode": cfg.island_mode,
        "source": source,
        "stats_sha256": stats_sha256,
        "layers": layers,
        "estimate": estimate,
        "notes": {
            "owners": (
                "owners are GLOBAL device indices; every expert is owned inside "
                "exactly one island (hub devices holding it first, then that "
                "island's expert devices, primary first, replicas ascending)."
                if cfg.island_mode == "partition"
                else "owners are written for island 0 (device indices); island k "
                "uses the same lists with each index mapped by its position inside "
                "the island (islands[k][position]). Per expert: hub devices holding "
                "it first, then expert devices, primary first, replicas ascending."
            ),
            "dispatch": "hub copy wins; otherwise cold copy index = (row + layer) % "
            "n_cold_copies over the expert-role owners in listed order.",
        },
    }
    if extra_meta:
        doc["meta"] = extra_meta
    doc["plan_id"] = hashlib.sha256(
        json.dumps(
            {"layers": layers, "devices": doc["devices"]}, separators=(",", ":")
        ).encode()
    ).hexdigest()[:16]
    return doc
