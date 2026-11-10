# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hub-side runtime of the heterogeneous MoE (spec 6.2).

One ``HubRuntime`` per hub process. It owns the plan tables, the connection to
the expert servers (``hx``) and the per-step protocol:

* ``pre_forward`` / ``post_forward`` are called by the model runner around every
  target-model forward. Before a graph replay they wake the servers
  (``post_step``: host code, outside any graph); ``post_forward`` polls the
  abort record.
* ``begin_step`` runs in the first MoE layer of a forward. Kernel launches that
  Python enqueues itself (eager steps, capture warm-ups) wake the servers from
  here unless the runner already did; during a graph capture nothing is posted
  (a capture executes nothing) and the capture width is baked in.
* ``apply_layer`` is one MoE layer: dispatch, shared expert, hub fused MoE on the
  resident experts, combine. The returned tensor is the island's partial sum;
  any all-reduce is the runner's.

The ``hx`` backend is a module with ``begin_step`` / ``dispatch`` / ``combine``
(the HIP ops of ``hetero_moe.hx``, or ``ref_hx`` in tests).
"""

import atexit
import functools
import importlib
import json
import logging
import os
import time
from collections.abc import Callable
from types import ModuleType
from typing import Protocol

import numpy as np
import torch

from .plan import HeteroPlan, PlanError, load_plan, plan_layer_id
from .tables import IslandTables, build_island_tables, parse_spill_k

logger = logging.getLogger(__name__)

# The hub MoE of one plan layer on its own slot pool: fn(x, topk_weights, topk_ids)
# -> y. Layers register it when their weights are final; the boot self-test calls it.
LayerFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

_PLAN: HeteroPlan | None = None
_RUNTIME: "HubRuntime | None" = None
_LAYER_FNS: dict[int, LayerFn] = {}

# hx_status word layout (hx_ops.cu / hx_proto.HX_ST_*): the abort record, the
# epoch, the live record, then steps, dispatches, combines and the pairs kept on
# the hub and sent to the servers.
ST_RECORD, ST_EPOCH, ST_LIVE, ST_STEPS = 0, 1, 2, 3
ST_DISPATCHES, ST_COMBINES, ST_PAIRS_HUB, ST_PAIRS_REMOTE = 4, 5, 6, 7
STATS_EVERY = 1000  # forwards between two counter lines


class HubLike(Protocol):
    """What the runtime needs of ``hx.HxHub`` (``ref_hx.RefHub`` has it too)."""

    def post_step(self, rows: int, bucket: int) -> None: ...

    def check(self) -> None: ...

    def close(self) -> None: ...


def _env():
    from vllm import envs

    return envs


def hetero_enabled() -> bool:
    return bool(_env().VLLM_HETERO_MOE)


def get_plan() -> HeteroPlan | None:
    """The plan of this process (``VLLM_HETERO_MOE=1``), loaded and validated once."""
    global _PLAN
    if not hetero_enabled():
        return None
    if _PLAN is None:
        path = _env().VLLM_HETERO_MOE_PLAN
        if not path:
            raise PlanError("VLLM_HETERO_MOE=1 needs VLLM_HETERO_MOE_PLAN=<plan.json>")
        _PLAN = load_plan(path)
    return _PLAN


def reset_for_tests() -> None:
    global _PLAN, _RUNTIME
    _PLAN = None
    if _RUNTIME is not None:
        _RUNTIME.close()
    _RUNTIME = None
    _LAYER_FNS.clear()


def moe_rank(parallel) -> int:
    """Index of this process in the hub device list: the EP rank under expert
    parallelism, else the TP rank (``FusedMoEParallelConfig``)."""
    return parallel.ep_rank if parallel.use_ep else parallel.tp_rank


def register_layer(plan_layer: int, fn: LayerFn) -> None:
    """A quant method registers the hub MoE of a layer that uses the servers."""
    _LAYER_FNS[plan_layer] = fn


def check_device_arch(plan: HeteroPlan, dev: int, device) -> str:
    """The plan's arch of ``dev`` must be the arch this process runs on (the
    per-role ``gcnArchName`` assertion of spec 6.2)."""
    want = plan.arch(dev)
    have = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if have != want:
        raise PlanError(
            f"device {plan.device_id(dev)} is {want} in plan {plan.plan_id}, but "
            f"this process runs on {have}"
        )
    return have


def hub_device_index(plan: HeteroPlan, rank: int) -> int:
    """Plan device of the hub with this TP/EP rank: ``VLLM_HETERO_MOE_DEVICE_ID``
    (comma list in rank order) or the plan's hubs in plan order."""
    ids = [s for s in _env().VLLM_HETERO_MOE_DEVICE_ID.split(",") if s]
    if ids:
        if rank >= len(ids):
            raise PlanError(
                f"VLLM_HETERO_MOE_DEVICE_ID lists {len(ids)} device(s), rank {rank}"
            )
        dev = plan.device_index(ids[rank])
    else:
        hubs = [d for k in range(plan.num_islands) for d in plan.hubs(k)]
        if rank >= len(hubs):
            raise PlanError(f"plan has {len(hubs)} hub(s), rank {rank}")
        dev = hubs[rank]
    if plan.role(dev) != "hub":
        raise PlanError(f"device {plan.device_id(dev)} is not a hub")
    return dev


def hub_slot_experts(
    layer_name: str, global_num_experts: int, rank: int
) -> list[int] | None:
    """Expert ids of this layer's slot pool on this hub, in slot order, or
    ``None`` when the layer follows the standard path (hetero off, a layer the
    plan does not list, or a hub-only layer that holds every expert)."""
    plan = get_plan()
    if plan is None:
        return None
    layer = plan_layer_id(layer_name, plan.num_layers)
    if layer is None or not plan.has_layer(layer):
        return None
    if plan.num_experts != global_num_experts:
        raise PlanError(
            f"plan {plan.plan_id} has {plan.num_experts} experts, layer "
            f"{layer_name} has {global_num_experts}"
        )
    resident = plan.hub_resident(hub_device_index(plan, rank), layer)
    identity = np.array_equal(resident, np.arange(global_num_experts))
    if plan.is_hub_only(layer) and identity:
        return None
    return [int(e) for e in resident]


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class HubRuntime:
    def __init__(
        self,
        plan: HeteroPlan,
        tables: IslandTables,
        hub: HubLike,
        hx: ModuleType,
        *,
        max_rows: int,
        device: torch.device | str,
        hidden: int = 0,
        stats: bool = False,
        step_log: str = "",
    ):
        self.plan = plan
        self.tables = tables
        self.hub = hub
        self.hx = hx
        self.max_rows = max_rows
        self.hx_layers = sorted(li for li, lt in tables.layers.items() if lt.uses_hx)
        if not self.hx_layers:
            raise PlanError(f"plan {plan.plan_id} has no layer that uses the servers")
        self.first_layer = self.hx_layers[0]
        self._widths = frozenset(tables.widths)
        self.hidden = hidden
        self.self_tested = False
        self.stats = stats
        self._forwards = 0
        # One buffer for all layers: graph captures bake its address.
        self.hub_ids = torch.empty(
            (max_rows, tables.top_k), dtype=torch.int32, device=device
        )
        self._armed = False
        self._closed = False
        # VLLM_HETERO_MOE_STEP_LOG=<path>: one JSON line per announced step in
        # <path>.<pid> (diagnosis of a wrong row; off by default). The handle lives
        # as long as the runtime and is closed in ``close``.
        self._step_log = (
            open(  # noqa: SIM115
                f"{step_log}.{os.getpid()}", "a", buffering=1
            )
            if step_log
            else None
        )

    # ---- per-step protocol -----------------------------------------------------

    def pre_forward(self, rows: int, replay: bool, batch=None, req_states=None) -> None:
        """Runner hook before a target forward of ``rows`` rows (the padded width
        when ``replay``). Wakes the servers; the forward must follow. ``batch`` and
        ``req_states`` (the runner's ``InputBatch`` and ``RequestState``) only feed
        the step log."""
        if replay and rows not in self._widths:
            raise RuntimeError(
                f"hetero MoE: graph replay of width {rows}, but the servers have "
                f"graphs for {sorted(self._widths)} only"
            )
        bucket = rows if replay else 0
        self.hub.post_step(rows, bucket)
        self._armed = True
        if self._step_log is not None:
            self._log_step("runner", rows, bucket, batch, req_states)

    def _log_step(self, who: str, rows: int, bucket: int, batch, req_states) -> None:
        """One JSON line: wall clock ns, the epoch just posted, who announced it
        (``runner`` hook or ``begin`` of an unannounced forward), rows, bucket and
        the running requests as ``[req_id, prompt_len, num_computed (the runner's
        CPU upper bound), num_scheduled]``. A failure here turns the log off, it
        never fails a forward."""
        try:
            reqs = []
            if batch is not None and req_states is not None:
                for i in range(batch.num_reqs):
                    s = int(batch.idx_mapping_np[i])
                    reqs.append(
                        [
                            batch.req_ids[i],
                            int(req_states.prompt_len.np[s]),
                            int(req_states.num_computed_tokens_np[s]),
                            int(batch.num_scheduled_tokens[i]),
                        ]
                    )
            rec = {
                "ns": time.time_ns(),
                "epoch": getattr(self.hub, "_epoch", None),
                "who": who,
                "rows": rows,
                "bucket": bucket,
                "reqs": reqs,
            }
            assert self._step_log is not None
            self._step_log.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except Exception:  # noqa: BLE001 - diagnosis only, never fail the forward
            logger.exception("hetero MoE: step log turned off after an error")
            self._close_step_log()

    def _close_step_log(self) -> None:
        f, self._step_log = self._step_log, None
        if f is not None:
            f.close()

    def post_forward(self) -> None:
        self._armed = False
        self.hub.check()
        if self.stats:
            self._forwards += 1
            if self._forwards % STATS_EVERY == 0:
                self.log_stats()

    def check(self) -> None:
        self.hub.check()

    def begin_step(self, rows: int) -> None:
        if rows > self.max_rows:
            raise RuntimeError(
                f"hetero MoE: {rows} rows exceed the transport's {self.max_rows}"
            )
        if _capturing():
            if rows not in self._widths:
                raise RuntimeError(
                    f"hetero MoE: capture of width {rows}, but the servers have "
                    f"graphs for {sorted(self._widths)} only (align "
                    "cudagraph_capture_sizes, the spill K table and the server "
                    "--buckets)"
                )
            bucket = rows
        else:
            bucket = 0
            if self._armed:
                self._armed = False  # the runner woke the servers for this forward
            else:
                self.hub.post_step(rows, 0)
                if self._step_log is not None:
                    self._log_step("begin", rows, 0, None, None)
        self.hx.begin_step(self.hub, rows, bucket)

    # ---- one MoE layer ----------------------------------------------------------

    def apply_layer(
        self,
        layer: int,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        hub_moe: Callable[[torch.Tensor], torch.Tensor],
        shared: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Dispatch -> shared expert -> hub MoE(masked ids) -> combine."""
        rows = x.shape[0]
        if layer == self.first_layer:
            self.begin_step(rows)
        hub_ids = self.hub_ids[:rows]
        self.hx.dispatch(self.hub, x, topk_ids, topk_weights, hub_ids, layer)
        if shared is not None:
            shared()
        y_hub = hub_moe(hub_ids)
        return self.hx.combine(self.hub, y_hub, y_hub, layer)

    def self_test(
        self,
        layer_fns: dict[int, LayerFn] | None = None,
        *,
        rows: int = 3,
        seed: int = 1234,
        rtol: float = 0.03,
        atol: float = 1e-3,
    ) -> dict[str, float | int]:
        """Boot known-answer check across the transport (spec C.2 step 3).

        One eager step of ``rows`` rows through every layer that uses the servers.
        The pairs are experts that are resident on the hub AND have a server copy,
        so the same experts can be computed on the hub alone (``fn``, no
        transport) and through dispatch / servers / combine (the spill rule sends
        some of them to the servers). The two must agree up to the numerics of
        two kernels (``rtol`` of the largest output, plus ``atol``); a NaN (the
        abort output) or an expert id mapped to the wrong server slot does not.
        Raises ``RuntimeError`` on a mismatch."""
        fns = _LAYER_FNS if layer_fns is None else layer_fns
        gen = torch.Generator().manual_seed(seed)
        device = self.hub_ids.device
        top_k = self.tables.top_k
        tested, worst = 0, 0.0
        for layer in self.hx_layers:
            lt = self.tables.layers[layer]
            fn = fns.get(layer)
            if fn is None:
                raise RuntimeError(
                    f"hetero MoE self-test: layer {layer} has no hub MoE"
                )
            pool = torch.from_numpy(
                np.flatnonzero((lt.owned != 0) & (lt.slot_map >= 0) & (lt.n_cold > 0))
            )
            if len(pool) < top_k:
                continue  # no resident expert with a server copy to cross-check
            ids = torch.stack(
                [
                    pool[torch.randperm(len(pool), generator=gen)[:top_k]]
                    for _ in range(rows)
                ]
            ).to(device=device, dtype=torch.int32)
            x = (torch.randn(rows, self.hidden, generator=gen) * 0.5).to(
                device=device, dtype=torch.float16
            )
            w = torch.softmax(torch.randn(rows, top_k, generator=gen), dim=-1).to(
                device
            )
            want = fn(x, w, ids).float()
            got = self.apply_layer(
                layer, x, w, ids, hub_moe=functools.partial(fn, x, w)
            ).float()
            scale = float(want.abs().max())
            err = float((got - want).abs().max())
            if not err <= rtol * scale + atol:  # also true for NaN
                raise RuntimeError(
                    f"hetero MoE boot self-test failed in layer {layer}: remote + hub "
                    f"output differs from the hub-only output by {err:g} "
                    f"(largest hub-only value {scale:g}, tolerance "
                    f"{rtol * scale + atol:g})"
                )
            tested += 1
            worst = max(worst, err / max(scale, 1e-12))
        if not tested:
            logger.warning(
                "hetero MoE boot self-test: no layer has a hub-resident expert with "
                "a server copy, nothing cross-checked"
            )
        else:
            logger.info(
                "hetero MoE boot self-test passed: %d layers, largest relative "
                "difference %.2e",
                tested,
                worst,
            )
        self.self_tested = True
        return {"layers": tested, "max_rel_err": worst}

    def log_stats(self, *, final: bool = False) -> None:
        """Counters of the transport (``VLLM_HETERO_MOE_STATS=1``): the hub's
        ``status_words`` (steps, pairs kept on the hub, pairs sent to servers; the
        raw words in a second line, so a failed arm can say whether the device
        saw an abort or a missing exchange) and, when the hub offers them, the
        per-layer ``layer_stats`` (hub-local, remote because the expert is not
        resident, remote by the spill rule)."""
        words = getattr(self.hub, "status_words", None)
        if words is not None:
            w = [int(v) for v in words()]
            steps, hub_pairs, remote = w[ST_STEPS], w[ST_PAIRS_HUB], w[ST_PAIRS_REMOTE]
            total = max(hub_pairs + remote, 1)
            logger.info(
                "hetero MoE counters%s: %d steps, pairs per step: hub %.1f, servers "
                "%.1f (%.1f %% remote)",
                " (final)" if final else "",
                steps,
                hub_pairs / max(steps, 1),
                remote / max(steps, 1),
                100.0 * remote / total,
            )
            logger.info(
                "hetero MoE hx status: record %#x, epoch %d, live %#x, steps %d, "
                "dispatches %d, combines %d, pairs hub %d, pairs servers %d",
                w[ST_RECORD] & 0xFFFFFFFFFFFFFFFF,
                w[ST_EPOCH],
                w[ST_LIVE] & 0xFFFFFFFFFFFFFFFF,
                steps,
                w[ST_DISPATCHES],
                w[ST_COMBINES],
                hub_pairs,
                remote,
            )
        layer_stats = getattr(self.hub, "layer_stats", None)
        table = layer_stats() if layer_stats is not None else None
        if table is not None:  # None: the hub was created without layer_stats=True
            rows = [[int(v) for v in row] for row in table]
            tot = [sum(r[i] for r in rows) for i in range(3)]
            logger.info(
                "hetero MoE pairs over all layers: hub %d, servers (not resident) %d, "
                "servers (spill rule) %d",
                *tot,
            )
            if final:
                body = "\n".join(
                    f"  layer {li}: hub {r[0]}, cold {r[1]}, spilled {r[2]}"
                    for li, r in enumerate(rows)
                    if any(r)
                )
                logger.info("hetero MoE per-layer pairs (final):\n%s", body)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self.stats:
                try:
                    self.log_stats(final=True)
                except Exception:  # noqa: BLE001 - the transport may already be down
                    logger.exception("hetero MoE: final counters unavailable")
            self._close_step_log()
            self.hub.close()


# ---- process singleton and the runner hooks ----------------------------------------


def get_runtime(*, rank: int, hidden: int, max_rows: int, device) -> "HubRuntime":
    """The process' ``HubRuntime``; the first call connects to the servers."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    if _capturing():
        raise RuntimeError("hetero MoE: the hub must connect before a graph capture")
    plan = get_plan()
    assert plan is not None
    hub_dev = hub_device_index(plan, rank)
    check_device_arch(plan, hub_dev, device)
    override = _env().VLLM_HETERO_MOE_SPILL_K
    tables = build_island_tables(
        plan, hub_dev, spill_k=parse_spill_k(override) if override else None
    )
    hx = importlib.import_module("vllm.distributed.hetero_moe.hx")
    stats = bool(_env().VLLM_HETERO_MOE_STATS)
    # layer_stats=True: the dispatch kernel counts per layer (hub / cold / spilled
    # pairs); off by default, no cost when off
    extra = {"layer_stats": True} if stats else {}
    hub = hx.HxHub.create(
        plan,
        plan.device_id(hub_dev),
        tables=tables,
        max_rows=max_rows,
        top_k=plan.top_k,
        hidden=hidden,
        wait_ms=_env().VLLM_HETERO_MOE_WAIT_MS,
        **extra,
    )
    _RUNTIME = HubRuntime(
        plan,
        tables,
        hub,
        hx,
        max_rows=max_rows,
        device=device,
        hidden=hidden,
        stats=stats,
        step_log=_env().VLLM_HETERO_MOE_STEP_LOG,
    )
    atexit.register(_RUNTIME.close)
    _RUNTIME.self_test()
    return _RUNTIME


def install_runtime(runtime: "HubRuntime | None") -> None:
    """Test hook: use ``runtime`` as the process' runtime."""
    global _RUNTIME
    _RUNTIME = runtime


def current_runtime() -> "HubRuntime | None":
    return _RUNTIME


def runner_pre_forward(
    num_tokens: int, replay: bool, batch=None, req_states=None
) -> None:
    """Model-runner hook before a target forward (no-op until the first MoE layer
    has connected the runtime, and whenever hetero MoE is off)."""
    runtime = _RUNTIME
    if runtime is not None:
        runtime.pre_forward(num_tokens, replay, batch, req_states)


def runner_post_forward() -> None:
    runtime = _RUNTIME
    if runtime is not None:
        runtime.post_forward()


def runner_check() -> None:
    runtime = _RUNTIME
    if runtime is not None:
        runtime.check()


def hetero_env_report() -> str:
    envs = _env()
    return (
        f"VLLM_HETERO_MOE={int(envs.VLLM_HETERO_MOE)} "
        f"plan={envs.VLLM_HETERO_MOE_PLAN or '-'} "
        f"device_id={envs.VLLM_HETERO_MOE_DEVICE_ID or '-'} "
        f"spill_k={envs.VLLM_HETERO_MOE_SPILL_K or 'default'} "
        f"wait_ms={envs.VLLM_HETERO_MOE_WAIT_MS} pid={os.getpid()}"
    )
