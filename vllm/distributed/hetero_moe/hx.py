# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hub side of the ``hx`` transport (spec 6.4, 9.4; docs/rdna2/hetero-hx-*.md).

``runtime.HubRuntime`` talks to this module only through ``HxHub`` and the three
module-level functions (the same API as ``ref_hx``):

* ``HxHub.create(...)`` connects to the island's expert servers (IPC handles
  under ``/dev/shm/hx_<plan_id>/``), allocates the reply region, uploads the plan
  tables and builds the device ctx.
* ``hub.post_step(T, bucket)`` (host, outside any graph, once per executed
  forward) wakes the servers through the step ring.
* ``begin_step`` / ``dispatch`` / ``combine`` enqueue the kernels of
  ``csrc/rocm/hetero_moe/hx_core.cuh`` on the current stream (graph capturable).
* ``hub.check()`` raises ``HxAbort`` when the device saw a timeout/mismatch or a
  server process died. Never syncs the compute stream.
"""

import functools
import os
import threading

import numpy as np
import torch

from vllm.distributed.hetero_moe import hx_proto as P
from vllm.distributed.hetero_moe import rdv
from vllm.distributed.hetero_moe.ctx import (
    Exported,
    Geometry,
    alloc_exported,
    calibrate_clock,
    close_handle,
    free_exported,
    hub_image,
    open_handle,
    ops,
    register_ctx,
    upload_tables,
)
from vllm.distributed.hetero_moe.ring import StepRing
from vllm.logger import init_logger

logger = init_logger(__name__)

SRV_FORMAT = "hx-srv/1"
HUB_FORMAT = "hx-hub/1"


class HxAbort(RuntimeError):
    """The transport failed: a wait timed out, a header did not match, the plan
    has an uncovered expert or a peer process died. The step's output is NaN and
    the process must not continue with this transport."""

    def __init__(self, message: str, record: int = 0):
        super().__init__(message)
        self.record = record


def srv_path(d, device_id: str):
    return d / f"srv_{device_id}.json"


def hub_path(d, device_id: str):
    return d / f"hub_{device_id}.json"


def ack_path(d, srv_id: str, hub_id: str):
    return d / f"ack_{srv_id}_{hub_id}.json"


def ring_path(d, hub_id: str):
    return d / f"ring_{hub_id}.bin"


# Digest mode: ``check()`` (once per forward) reads the counters every this
# many calls and logs them when a mismatch was counted, plus every 16th time.
DIGEST_LOG_EVERY = 256


class HxHub:
    """One hub process' connection to its island's expert servers."""

    _closed: bool
    _dead: tuple[int, str] | None
    _epoch: int
    _lock: threading.Lock
    _stats_lock: threading.Lock
    _side: torch.cuda.Stream | None
    _opened: list[int]
    _reply: Exported | None
    _lstats: torch.Tensor | None
    ctx: torch.Tensor | None
    digest: bool
    _checks: int
    _dig_bad: int
    ring: StepRing | None
    watcher: rdv.PeerWatcher | None

    def __init__(self):
        raise TypeError("use HxHub.create")

    @property
    def n_cold_table(self) -> torch.Tensor:
        """The device ``n_cold`` table (int8 ``[L, E]``) that the dispatch kernel reads
        on every call. ``HubRuntime`` rewrites it in place (``copy_``) to switch the
        decode dispatch rule; never rebind or reallocate it: the hx ctx holds its
        address."""
        return self._tables.n_cold

    @classmethod
    def create(
        cls,
        plan,
        hub_device_id: str,
        *,
        tables,
        max_rows: int = P.HX_DEFAULT_ROWS_CAP,
        top_k: int = P.HX_DEFAULT_TOPK,
        hidden: int = P.HX_DEFAULT_HIDDEN,
        wait_ms: int = P.HX_DEFAULT_WAIT_MS,
        connect_timeout_s: float = 300.0,
        nslots: int = P.HX_DEFAULT_NSLOTS,
        flush: int | None = None,
        alloc: str | None = None,
        device: str | torch.device = "cuda",
        layer_stats: bool = False,
    ) -> "HxHub":
        self = object.__new__(cls)
        self._closed = False
        self._dead = None
        self._epoch = 0
        self._checks = 0
        self._dig_bad = 0
        self.digest = False
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._side = None
        self._opened = []
        self._reply = None
        self._lstats = None
        self.ctx = None
        self.ring = None
        self.watcher = None
        try:
            self._connect(
                plan,
                hub_device_id,
                tables,
                max_rows,
                top_k,
                hidden,
                wait_ms,
                connect_timeout_s,
                nslots,
                os.environ.get("VLLM_HX_FLUSH", "0") if flush is None else flush,
                alloc or os.environ.get("VLLM_HX_ALLOC", "uncached"),
                device,
                layer_stats,
            )
        except BaseException:
            self.close()
            raise
        return self

    # ---- connect ----------------------------------------------------------------

    def _connect(
        self,
        plan,
        hub_id,
        tables,
        max_rows,
        top_k,
        hidden,
        wait_ms,
        timeout_s,
        nslots,
        flush,
        alloc,
        device,
        layer_stats,
    ):
        hub_dev = plan.device_index(hub_id)
        island = plan.island_of(hub_dev)
        servers = plan.experts(island)
        if not servers:
            raise RuntimeError(
                f"hx: island {island} of plan {plan.plan_id} has no servers"
            )
        if len(servers) > P.HX_MAX_SERVERS:
            raise RuntimeError(f"hx: {len(servers)} servers exceed {P.HX_MAX_SERVERS}")
        self.plan_id = plan.plan_id
        self.hub_id = hub_id
        self.hub_index = plan.hubs(island).index(hub_dev)
        self.server_ids = [plan.device_id(d) for d in servers]
        self.nsrv = len(servers)
        self.max_rows = max_rows
        geom = Geometry(
            rows_cap=max_rows,
            topk=top_k,
            hidden=hidden,
            nslots=nslots,
            num_experts=plan.num_experts,
            nlayers=48,
        )
        self.geometry = geom
        d = rdv.hx_dir(plan.plan_id)
        self.dir = d
        deadline_note = f"{timeout_s:g} s"

        # stale files of an earlier run of this hub
        rdv.remove_quiet(hub_path(d, hub_id))
        for sid in self.server_ids:
            rdv.remove_quiet(ack_path(d, sid, hub_id))
        rdv.remove_quiet(ring_path(d, hub_id))

        tpm, attr_khz = calibrate_clock()
        logger.info(
            "hx hub %s: device clock %d ticks/ms measured (attribute %.0f kHz)",
            hub_id,
            tpm,
            attr_khz,
        )
        if abs(tpm - attr_khz) > 0.05 * max(tpm, 1):
            logger.warning(
                "hx hub %s: measured clock rate %d ticks/ms differs from the "
                "device attribute %.0f by more than 5%%; the measured rate is used",
                hub_id,
                tpm,
                attr_khz,
            )
        self.ticks_per_ms = tpm

        tdev = upload_tables(tables, max_rows, device)
        self._tables = tdev
        self.nlayers = tdev.nlayers
        if layer_stats:
            self._lstats = torch.zeros(
                P.HX_MAX_LAYERS, P.HX_LSTAT_COLS, dtype=torch.int64, device=device
            )
        reply_layout = geom.reply(self.nsrv)
        self._reply = alloc_exported(reply_layout.total_bytes, alloc)
        logger.info(
            "hx hub %s: reply region %d MiB, allocation %s",
            hub_id,
            self._reply.nbytes >> 20,
            self._reply.kind_name,
        )

        self.digest = bool(int(flush) & P.HX_DIGEST)

        # servers publish after load, self-test and graph capture
        infos = []
        for sid in self.server_ids:
            info = rdv.wait_json(
                srv_path(d, sid),
                timeout_s,
                what=f"server {sid} (plan {plan.plan_id})",
            )
            self._validate_server(plan, sid, info, geom)
            if not rdv.pid_alive(int(info["pid"])):
                raise RuntimeError(f"hx: server {sid} pid {info['pid']} is not alive")
            infos.append(info)
        inbox_ptrs = []
        for sid, info in zip(self.server_ids, infos):
            ptr = open_handle(bytes.fromhex(info["inbox"]["handle"]))
            self._opened.append(ptr)
            inbox_ptrs.append(ptr)
        self._server_pids = [int(i["pid"]) for i in infos]

        image = hub_image(
            geom,
            plan_id=plan.plan_id,
            hub_index=self.hub_index,
            nsrv=self.nsrv,
            reply=self._reply,
            inbox_ptrs=inbox_ptrs,
            tables=tdev,
            ticks_per_ms=tpm,
            wait_ms=wait_ms,
            flush_mode=int(flush),
            layer_stats=self._lstats,
        )
        self.ctx = register_ctx(image, device)
        self.ring = StepRing.create(ring_path(d, hub_id))
        rdv.write_json(
            hub_path(d, hub_id),
            {
                "format": HUB_FORMAT,
                "plan_id": plan.plan_id,
                "device_id": hub_id,
                "hub_index": self.hub_index,
                "pid": os.getpid(),
                "geometry": geom.as_dict(),
                "nsrv": self.nsrv,
                "digest": self.digest,
                "servers": self.server_ids,
                "ticks_per_ms": tpm,
                "reply": {
                    "handle": self._reply.handle_hex(),
                    "nbytes": self._reply.nbytes,
                    "kind": self._reply.kind_name,
                },
            },
        )
        for sid in self.server_ids:
            rdv.wait_json(
                ack_path(d, sid, hub_id),
                timeout_s,
                what=f"ack of server {sid} (waited {deadline_note})",
                alive=functools.partial(self._peer_alive_error, sid),
            )
        self.watcher = rdv.PeerWatcher(self._on_peer_death)
        self.watcher.start()
        for sid, pid in zip(self.server_ids, self._server_pids):
            self.watcher.add(pid, sid)
        logger.info(
            "hx hub %s connected to %d server(s): %s (plan %s)",
            hub_id,
            self.nsrv,
            ", ".join(self.server_ids),
            plan.plan_id,
        )

    def _validate_server(self, plan, sid: str, info: dict, geom: Geometry) -> None:
        if info.get("format") != SRV_FORMAT or info.get("plan_id") != plan.plan_id:
            raise RuntimeError(
                f"hx: {sid} published format {info.get('format')!r} plan "
                f"{info.get('plan_id')!r}; hub expects {SRV_FORMAT!r} plan "
                f"{plan.plan_id!r}"
            )
        theirs = Geometry.from_dict(info["geometry"])
        for name in ("rows_cap", "topk", "hidden", "nslots"):
            if getattr(theirs, name) != getattr(geom, name):
                raise RuntimeError(
                    f"hx: geometry mismatch with {sid}: {name} hub "
                    f"{getattr(geom, name)} server {getattr(theirs, name)}"
                )
        if bool(info.get("digest", False)) != self.digest:
            raise RuntimeError(
                f"hx: digest mode differs: hub {self.digest}, {sid} "
                f"{bool(info.get('digest', False))}; set VLLM_HX_FLUSH with the "
                f"digest bit (256, decimal) on the hub and on every server, or "
                f"on none"
            )
        dev = plan.device_index(sid)
        if info.get("arch") and plan.arch(dev) and plan.arch(dev) not in info["arch"]:
            logger.warning(
                "hx: server %s runs on %s, the plan says %s",
                sid,
                info["arch"],
                plan.arch(dev),
            )

    def _peer_alive_error(self, sid: str) -> str | None:
        pid = self._server_pids[self.server_ids.index(sid)]
        return None if rdv.pid_alive(pid) else f"server {sid} (pid {pid}) died"

    def _on_peer_death(self, pid: int, name: str) -> None:
        # Runs on the watcher thread: poke the control page so waits end.
        self._dead = (pid, name)
        logger.error("hx hub: server %s (pid %d) died, aborting waits", name, pid)
        try:
            ops().hx_host_abort(self.ctx)
        except Exception:  # noqa: BLE001 - nothing else to do on this thread
            logger.exception("hx hub: host abort poke failed")

    # ---- per step ---------------------------------------------------------------

    def post_step(self, T: int, bucket: int) -> None:
        """Host, outside any graph: announce the next forward to the servers.
        Exactly once per executed forward (before its kernels are launched),
        never for a capture."""
        if self._dead is not None:
            raise HxAbort(f"hx server {self._dead[1]} (pid {self._dead[0]}) died")
        ring = self.ring
        if ring is None:
            raise HxAbort("hx hub is not connected")
        with self._lock:
            self._epoch += 1
            ring.post(self._epoch, T, bucket)

    def status_words(self) -> list[int]:
        return [int(v) for v in ops().hx_status(self.ctx)]

    def digest_stats(self) -> list[int]:
        """The ``HX_DS_*`` words of digest mode (ctx ``flush`` bit ``HX_DIGEST``,
        also on every server; all zero otherwise), through the same side-stream
        copy as ``status_words``."""
        return [int(v) for v in ops().hx_dig_stats(self.ctx)]

    def describe_digest(self) -> str:
        """One line of the digest counters (``hx_proto.describe_dig_stats``)."""
        return P.describe_dig_stats(self.digest_stats(), "hub")

    def _note_digest(self, final: bool = False) -> None:
        words = self.digest_stats()
        bad = words[P.HX_DS_BAD]
        if final or bad != self._dig_bad or self._checks % (16 * DIGEST_LOG_EVERY) == 0:
            self._dig_bad = bad
            (logger.warning if bad else logger.info)(
                "hx hub %s: %s", self.hub_id, P.describe_dig_stats(words, "hub")
            )

    def check(self) -> None:
        """Raise ``HxAbort`` if the device or the host saw the transport fail."""
        st = self.status_words()
        rec = st[P.HX_ST_RECORD]
        if rec != 0:
            raise HxAbort(P.describe_record(rec, "hub"), rec)
        if self._dead is not None:
            raise HxAbort(f"hx server {self._dead[1]} (pid {self._dead[0]}) died")
        if self.digest:
            self._checks += 1
            if self._checks % DIGEST_LOG_EVERY == 0:
                self._note_digest()

    def layer_stats(self) -> np.ndarray | None:
        """Cumulative pair counts per layer, int64 ``[nlayers, 3]``: kept on the
        hub, sent to a server because the expert is not resident, sent to a
        server by the spill rule. ``None`` unless created with
        ``layer_stats=True``. Read through a side stream copy: the compute stream
        is not waited for, so the numbers may trail the running step."""
        if self._lstats is None:
            return None
        with self._stats_lock:
            if self._side is None:
                self._side = torch.cuda.Stream(device=self._lstats.device)
            with torch.cuda.stream(self._side):
                host = self._lstats.to("cpu")
        cols = [P.HX_LSTAT_HUB, P.HX_LSTAT_COLD, P.HX_LSTAT_SPILL]
        return host.numpy()[: self.nlayers, cols].copy()

    def reset_layer_stats(self) -> None:
        if self._lstats is not None:
            with self._stats_lock:
                if self._side is None:
                    self._side = torch.cuda.Stream(device=self._lstats.device)
                with torch.cuda.stream(self._side):
                    self._lstats.zero_()
                self._side.synchronize()

    # ---- shutdown ---------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.digest and self.ctx is not None:
            try:
                self._note_digest(final=True)
            except Exception:  # noqa: BLE001 - the device may be gone
                logger.exception("hx hub: reading the digest counters failed")
        if self.ring is not None:
            self.ring.close_writer()
        if self.watcher is not None:
            self.watcher.stop()
        if self.ctx is not None:
            try:
                ops().hx_ctx_unregister(self.ctx)
            except Exception:  # noqa: BLE001
                logger.exception("hx hub: ctx unregister failed")
        for ptr in self._opened:
            try:
                close_handle(ptr)
            except Exception:  # noqa: BLE001
                logger.exception("hx hub: closing a server handle failed")
        self._opened.clear()
        d = getattr(self, "dir", None)
        if d is not None:
            rdv.remove_quiet(hub_path(d, self.hub_id))
            for sid in self.server_ids:
                rdv.remove_quiet(ack_path(d, sid, self.hub_id))
        if self.ring is not None:
            self.ring.close()
            if d is not None:
                rdv.remove_quiet(ring_path(d, self.hub_id))
        if self._reply is not None:
            free_exported(self._reply)
            self._reply = None


# ---- module-level ops (same names and signatures as ref_hx) ----------------------


def begin_step(hub: HxHub, rows: int, bucket: int) -> None:
    ops().hx_begin_step(hub.ctx, rows, bucket)


def dispatch(
    hub: HxHub,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    hub_ids_out: torch.Tensor,
    layer: int,
) -> None:
    ops().hx_dispatch(x, topk_ids, topk_w, hub_ids_out, hub.ctx, layer)


def combine(
    hub: HxHub, y_hub: torch.Tensor, out: torch.Tensor, layer: int
) -> torch.Tensor:
    if hub.digest and out.data_ptr() == y_hub.data_ptr():
        # Digest mode heals a late chunk by summing the replies again, so the
        # hub's own part must survive the first pass: never combine in place.
        res = torch.empty_like(y_hub)
        ops().hx_combine(y_hub, res, hub.ctx, layer)
        out.copy_(res)
        return out
    ops().hx_combine(y_hub, out, hub.ctx, layer)
    return out


def status(hub: HxHub) -> torch.Tensor:
    return ops().hx_status(hub.ctx)
