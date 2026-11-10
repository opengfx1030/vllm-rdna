# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert server of the heterogeneous MoE (spec 6.3, Appendix C.2/C.6).

``python -m vllm.distributed.hetero_moe.server --plan P --device-id exp0 --model M``

One process per V620. It loads only the experts the plan assigns to its device
(all target layers, through the TP4-EP V620 W4A16 repack, ``server_model.py``),
runs a known-answer self-test, captures one CUDA graph per decode bucket, then
publishes its inbox (``/dev/shm/hx_<plan_id>/srv_<device>.json``) and waits for
the hub of its island. Per step the host thread reads the step ring and either
replays the graph of the announced bucket or enqueues the eager chain (prefill
widths, capture warm-ups): ``hx_srv_begin`` (header check), then per layer
``hx_srv_wait`` -> MoE on views of the inbox slot -> ``hx_srv_push``.

Exit codes: 0 clean (SIGTERM, hub closed the ring), 3 transport abort or peer
death, 4 setup failure (rendezvous timeout, self-test, plan problem).
"""

import argparse
import collections
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.distributed.hetero_moe import hx_proto as P
from vllm.distributed.hetero_moe import rdv
from vllm.distributed.hetero_moe.ctx import (
    ALLOC_KINDS,
    Exported,
    Geometry,
    alloc_exported,
    calibrate_clock,
    close_handle,
    free_exported,
    open_handle,
    ops,
    register_ctx,
    reregister_ctx,
    server_image,
)
from vllm.distributed.hetero_moe.hx import (
    HUB_FORMAT,
    SRV_FORMAT,
    ack_path,
    hub_path,
    ring_path,
    srv_path,
)
from vllm.distributed.hetero_moe.ring import RingOverrun, StepDesc, StepRing
from vllm.logger import init_logger

logger = init_logger(__name__)

EXIT_OK = 0
EXIT_ABORT = 3
EXIT_SETUP = 4

# Environment of the V620 production recipe (tools/rdna/recipes/flashnext-mtp2.env):
# the server chain must be the kernel path the TP4 engine runs. Set before the
# first HIP call, never override what the caller exported.
ENV_DEFAULTS = {
    "VLLM_RDNA2_W4A8_SDOT4": "1",
    "VLLM_RDNA2_MOE_DETERMINISTIC": "1",
    "VLLM_RDNA2_W4A8_BLOCK_M": "8",
    "VLLM_RDNA_W4A8_KBLOCK": "-1",
    "GPU_MAX_HW_QUEUES": "4",
    "HSA_FORCE_FINE_GRAIN_PCIE": "1",
    "HSA_ENABLE_SDMA": "0",
    "PYTORCH_TUNABLEOP_ENABLED": "0",
}

# ---- opcode registry (stage 2 adds lm_head = 2, dense = 3) -----------------------
OpHandler = Callable[["ExpertServer", StepDesc], None]
_OP_HANDLERS: dict[int, OpHandler] = {}


def register_op(opcode: int, handler: OpHandler) -> None:
    """Handler of a step opcode: enqueues the step's device work on the current
    stream. An opcode without a handler is a protocol error (abort)."""
    if opcode in _OP_HANDLERS:
        raise ValueError(f"hx server: opcode {opcode} already registered")
    _OP_HANDLERS[opcode] = handler


def _moe_op(server: "ExpertServer", desc: StepDesc) -> None:
    server.run_moe_step(desc)


register_op(P.HX_OPCODE_MOE, _moe_op)


class SyntheticModel:
    """Weight-free stand-in with the ``ExpertServerModel`` call interface, for
    dry runs of the transport (``--synthetic``): ``y[r, c] = fp16(x[r, c] *
    sum_k w[r, k] * ((id & 7) + 1))`` over the pairs with ``id >= 0``."""

    def __init__(self, num_layers: int, hidden: int, top_k: int):
        self.num_layers = num_layers
        self.hidden = hidden
        self.top_k = top_k
        self.expert_sets = [[0]] * num_layers

    def moe(self, layer, x, topk_ids, topk_w):
        scale = ((topk_ids & 7) + 1).to(torch.float32) * topk_w
        scale = torch.where(topk_ids >= 0, scale, torch.zeros_like(scale))
        return (x.to(torch.float32) * scale.sum(dim=1, keepdim=True)).to(torch.float16)

    def known_answer_selftest(self, **_):
        return {"synthetic": True}

    def memory_bytes(self) -> int:
        return 0


def load_expert_model(
    plan,
    dev: int,
    model_dir: str,
    device,
    *,
    max_capture_rows: int,
    progress: Callable[[int, int], None] | None = None,
):
    """``ExpertServerModel`` of plan device ``dev``: its experts of every target
    layer (layers of the plan the device does not serve stay empty), loaded from
    the checkpoint through the V620 W4A16 repack."""
    from vllm.distributed.hetero_moe.server_model import ExpertServerModel

    layers = set(plan.target_layers)
    sets = [
        [int(e) for e in plan.server_experts(dev, layer)] if layer in layers else []
        for layer in range(max(layers) + 1)
    ]
    model = ExpertServerModel(
        model_dir,
        sets,
        device,
        num_experts=plan.num_experts,
        top_k=plan.top_k,
        max_capture_rows=max_capture_rows,
    )
    model.load(progress)
    return model


class ServerError(RuntimeError):
    """Setup problem (exit code 4)."""


class ExpertServer:
    def __init__(
        self,
        plan,
        device_id: str,
        model_dir: str | None,
        *,
        buckets: list[int],
        max_rows: int = P.HX_DEFAULT_ROWS_CAP,
        wait_ms: int = P.HX_DEFAULT_WAIT_MS,
        begin_wait_ms: int = P.HX_DEFAULT_BEGIN_WAIT_MS,
        nslots: int = P.HX_DEFAULT_NSLOTS,
        alloc: str = "uncached",
        flush: int = 0,
        selftest: bool = True,
        graphs: bool = True,
        synthetic: bool = False,
        hub_timeout_s: float = 7200.0,
        ready_file: str | None = None,
        expect_bus: str | None = None,
    ):
        self.plan = plan
        self.device_id = device_id
        self.model_dir = model_dir
        self.buckets = sorted(set(buckets))
        self.max_rows = max_rows
        self.wait_ms = wait_ms
        self.begin_wait_ms = begin_wait_ms
        self.nslots = nslots
        self.alloc = alloc
        self.flush = flush
        self.selftest = selftest
        self.use_graphs = graphs
        self.synthetic = synthetic
        self.hub_timeout_s = hub_timeout_s
        self.ready_file = ready_file
        self.expect_bus = expect_bus
        self._stop = False
        self._hub_dead = False
        self._abort_exit: int | None = None
        self._dig_bad = 0  # digest mismatches logged so far (HX_DIGEST mode)
        self._inflight: collections.deque = collections.deque()
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._opened: list[int] = []
        self._inbox: Exported | None = None
        self.ctx: torch.Tensor | None = None
        self.ring: StepRing | None = None
        self.watcher: rdv.PeerWatcher | None = None
        self._next_no = 1

        dev = plan.device_index(device_id)
        if plan.role(dev) != "expert":
            raise ServerError(
                f"device {device_id} of plan {plan.plan_id} is not an expert"
            )
        self.dev = dev
        self.island = plan.island_of(dev)
        self.server_index = plan.server_index(dev)
        self.hubs = [plan.device_id(h) for h in plan.hubs(self.island)]
        if len(self.hubs) != 1:
            raise ServerError(
                f"island {self.island} has {len(self.hubs)} hubs; a server serves "
                "exactly one hub in stage 1 (multi-hub inboxes are a stage-2 extension)"
            )
        self.hub_id = self.hubs[0]
        self.layers = sorted(plan.target_layers)
        if not self.layers or self.layers[-1] >= P.HX_MAX_LAYERS:
            raise ServerError(f"plan layers {self.layers[:3]}... outside the protocol")
        self.nlayers = self.layers[-1] + 1
        if not self.buckets or self.buckets[0] < 1 or self.buckets[-1] > max_rows:
            raise ServerError(f"buckets {self.buckets} must be within 1..{max_rows}")
        self.geometry = Geometry(
            rows_cap=max_rows,
            topk=plan.top_k,
            hidden=P.HX_DEFAULT_HIDDEN,
            nslots=nslots,
            num_experts=plan.num_experts,
            nlayers=self.nlayers,
        )

    # ---- setup ------------------------------------------------------------------

    def prepare(self) -> None:
        """Everything that does not need the hub: allocation, ctx, weights, self
        test, graphs."""
        plan = self.plan
        d = rdv.hx_dir(plan.plan_id)
        self.dir = d
        rdv.remove_quiet(srv_path(d, self.device_id))
        rdv.remove_quiet(ack_path(d, self.device_id, self.hub_id))

        anchor = torch.empty(1, device="cuda")  # resolves the visible device
        self.device = anchor.device
        props = torch.cuda.get_device_properties(self.device)
        self.arch = props.gcnArchName
        want = plan.arch(self.dev)
        if want and want not in self.arch:
            raise ServerError(
                f"device {self.device_id} is {self.arch}, the plan says {want}"
            )
        bus = f"{props.pci_bus_id:02x}"
        if self.expect_bus and bus != self.expect_bus.lower().removeprefix("0x"):
            raise ServerError(
                f"device {self.device_id} sits on PCI bus {bus}, expected "
                f"{self.expect_bus} (HIP indices move; resolve by bus)"
            )
        self.bus = bus
        tpm, attr_khz = calibrate_clock()
        logger.info(
            "hx server %s on %s (%s, bus %s): device clock %d ticks/ms measured "
            "(attribute %.0f kHz)",
            self.device_id,
            props.name,
            self.arch,
            bus,
            tpm,
            attr_khz,
        )
        if abs(tpm - attr_khz) > 0.05 * max(tpm, 1):
            logger.warning(
                "hx server %s: measured clock rate %d ticks/ms differs from the "
                "attribute %.0f by more than 5%%; the measured rate is used",
                self.device_id,
                tpm,
                attr_khz,
            )
        self.ticks_per_ms = tpm

        geom = self.geometry
        self._inbox_layout = geom.inbox()
        self._inbox = alloc_exported(self._inbox_layout.total_bytes, self.alloc)
        logger.info(
            "hx server %s: inbox %d MiB, allocation %s",
            self.device_id,
            self._inbox.nbytes >> 20,
            self._inbox.kind_name,
        )
        # cached workspaces hx_srv_wait fills from the uncached inbox slot
        self.ids32 = torch.zeros(
            geom.rows_cap * geom.topk, dtype=torch.int32, device=self.device
        )
        self.x_ws = torch.zeros(
            geom.rows_cap, geom.hidden, dtype=torch.float16, device=self.device
        )
        self.w_ws = torch.zeros(
            geom.rows_cap, geom.topk, dtype=torch.float32, device=self.device
        )
        hub_nsrv = len(self.plan.experts(self.island))
        self._hub_nsrv = hub_nsrv
        image = server_image(
            geom,
            plan_id=plan.plan_id,
            server_index=self.server_index,
            nsrv=hub_nsrv,
            inbox=self._inbox,
            hub_reply_ptr=self._inbox.ptr,  # placeholder until the hub connects
            ids32=self.ids32,
            x_ws=self.x_ws,
            w_ws=self.w_ws,
            ticks_per_ms=tpm,
            wait_ms=self.wait_ms,
            begin_wait_ms=self.begin_wait_ms,
            hub_nsrv=hub_nsrv,
            flush_mode=self.flush,
        )
        self._image = image
        self.ctx = register_ctx(image, self.device)

        self._load_model()
        if self.use_graphs:
            self._capture_graphs()

    def _load_model(self) -> None:
        geom = self.geometry
        if self.synthetic:
            self.model = SyntheticModel(self.nlayers, geom.hidden, geom.topk)
            return
        if not self.model_dir:
            raise ServerError("--model is required (or --synthetic)")
        t0 = time.monotonic()

        def progress(layer: int, total: int) -> None:
            if (layer + 1) % 8 == 0 or layer + 1 == total:
                logger.info(
                    "hx server %s: loaded layer %d/%d (%.0f s)",
                    self.device_id,
                    layer + 1,
                    total,
                    time.monotonic() - t0,
                )

        self.model = load_expert_model(
            self.plan,
            self.dev,
            self.model_dir,
            self.device,
            max_capture_rows=self.buckets[-1],
            progress=progress,
        )
        logger.info(
            "hx server %s: %d experts over %d layers, %.2f GiB on the device",
            self.device_id,
            sum(len(s) for s in self.model.expert_sets),
            len(self.layers),
            self.model.memory_bytes() / 2**30,
        )
        if self.selftest:
            res = self.model.known_answer_selftest()
            logger.info("hx server %s: known-answer self-test %s", self.device_id, res)

    def _warm(self, T: int) -> None:
        """Run the chain once on dummy inputs so no kernel loads lazily inside a
        capture (the protocol kernels are not involved)."""
        geom = self.geometry
        gen = torch.Generator(device=self.device).manual_seed(7)
        x = torch.randn(T, geom.hidden, device=self.device, generator=gen).to(
            torch.float16
        )
        w = torch.rand(T, geom.topk, device=self.device, generator=gen)
        for layer in self.layers:
            held = getattr(self.model, "expert_sets", None)
            ids = torch.full((T, geom.topk), -1, dtype=torch.int32, device=self.device)
            if held is not None and held[layer]:
                mine = torch.tensor(held[layer], dtype=torch.int32, device=self.device)
                ids[:, :] = mine[
                    torch.arange(T * geom.topk, device=self.device) % len(mine)
                ].view(T, geom.topk)
            self.model.moe(layer, x, ids, w)

    def _chain(self, T: int) -> None:
        """The enqueued work of one step at T rows (also what a graph records)."""
        c = self.ctx
        geom = self.geometry
        o = ops()
        o.hx_srv_begin(c)
        ids = self.ids32[: T * geom.topk].view(T, geom.topk)
        for layer in self.layers:
            o.hx_srv_wait(c, layer, T)
            y = self.model.moe(layer, self.x_ws[:T], ids, self.w_ws[:T])
            o.hx_srv_push(y, c, layer, T)

    def _capture_graphs(self) -> None:
        for T in self.buckets:
            self._warm(T)
        torch.cuda.current_stream().synchronize()
        for T in self.buckets:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._chain(T)
            self._graphs[T] = g
            logger.info("hx server %s: captured graph for %d rows", self.device_id, T)

    # ---- rendezvous ----------------------------------------------------------

    def publish(self) -> None:
        geom = self.geometry
        inbox = self._inbox
        if inbox is None:
            raise ServerError("publish before prepare")
        doc = {
            "format": SRV_FORMAT,
            "plan_id": self.plan.plan_id,
            "device_id": self.device_id,
            "server_index": self.server_index,
            "island": self.island,
            "pid": os.getpid(),
            "arch": self.arch,
            "bus": self.bus,
            "geometry": geom.as_dict(),
            "buckets": self.buckets,
            "ticks_per_ms": self.ticks_per_ms,
            "digest": bool(self.flush & P.HX_DIGEST),
            "inbox": {
                "handle": inbox.handle_hex(),
                "nbytes": inbox.nbytes,
                "kind": inbox.kind_name,
            },
            "state": "ready",
        }
        rdv.write_json(srv_path(self.dir, self.device_id), doc)
        if self.ready_file:
            rdv.write_json(Path(self.ready_file), doc)

    def connect_hub(self) -> bool:
        """Wait for the hub, handshake and open its reply region. False: a stop
        request (SIGTERM) arrived while waiting, nothing was connected."""
        d = self.dir
        try:
            info = rdv.wait_json(
                hub_path(d, self.hub_id),
                self.hub_timeout_s,
                what=f"hub {self.hub_id} (plan {self.plan.plan_id})",
                alive=lambda: "stop requested" if self._stop else None,
            )
        except RuntimeError:
            if self._stop:
                return False
            raise
        if info.get("format") != HUB_FORMAT or info.get("plan_id") != self.plan.plan_id:
            raise ServerError(
                f"hub published format {info.get('format')!r} plan "
                f"{info.get('plan_id')!r}; server expects {HUB_FORMAT!r} plan "
                f"{self.plan.plan_id!r}"
            )
        theirs = Geometry.from_dict(info["geometry"])
        for name in ("rows_cap", "topk", "hidden", "nslots"):
            if getattr(theirs, name) != getattr(self.geometry, name):
                raise ServerError(
                    f"geometry mismatch with hub {self.hub_id}: {name} hub "
                    f"{getattr(theirs, name)} server {getattr(self.geometry, name)}"
                )
        if bool(info.get("digest", False)) != bool(self.flush & P.HX_DIGEST):
            raise ServerError(
                f"digest mode differs: hub {bool(info.get('digest', False))}, "
                f"server {bool(self.flush & P.HX_DIGEST)}; set VLLM_HX_FLUSH with "
                f"the digest bit (256, decimal) on the hub and on every server, "
                f"or on none"
            )
        if self.device_id not in info["servers"]:
            raise ServerError(f"hub {self.hub_id} does not list {self.device_id}")
        listed = info["servers"].index(self.device_id)
        if listed != self.server_index:
            raise ServerError(
                f"hub lists {self.device_id} at index {listed}, "
                f"plan says {self.server_index}"
            )
        inbox = self._inbox
        if inbox is None:
            raise ServerError("connect_hub before prepare")
        if not rdv.pid_alive(int(info["pid"])):
            raise ServerError(f"hub pid {info['pid']} is not alive")
        ptr = open_handle(bytes.fromhex(info["reply"]["handle"]))
        self._opened.append(ptr)
        self._hub_pid = int(info["pid"])
        # re-upload the ctx with the hub's reply region (state words restart at 0)
        image = server_image(
            self.geometry,
            plan_id=self.plan.plan_id,
            server_index=self.server_index,
            nsrv=self._hub_nsrv,
            inbox=inbox,
            hub_reply_ptr=ptr,
            ids32=self.ids32,
            x_ws=self.x_ws,
            w_ws=self.w_ws,
            ticks_per_ms=self.ticks_per_ms,
            wait_ms=self.wait_ms,
            begin_wait_ms=self.begin_wait_ms,
            hub_nsrv=int(info["nsrv"]),
            flush_mode=self.flush,
        )
        reregister_ctx(self.ctx, image)
        self.ring = StepRing.open(ring_path(d, self.hub_id), timeout_s=60.0)
        self.watcher = rdv.PeerWatcher(self._on_hub_death)
        self.watcher.start()
        self.watcher.add(self._hub_pid, self.hub_id)
        rdv.write_json(
            ack_path(d, self.device_id, self.hub_id),
            {"pid": os.getpid(), "hub": self.hub_id, "server": self.device_id},
        )
        logger.info("hx server %s connected to hub %s", self.device_id, self.hub_id)
        return True

    def _on_hub_death(self, pid: int, name: str) -> None:
        self._hub_dead = True
        logger.error("hx server %s: hub %s (pid %d) died", self.device_id, name, pid)
        self._poke()

    def _poke(self) -> None:
        try:
            ops().hx_host_abort(self.ctx)
        except Exception:  # noqa: BLE001 - nothing better to do on a signal path
            logger.exception("hx server: host abort poke failed")

    # ---- steps ---------------------------------------------------------------

    def run_moe_step(self, desc: StepDesc) -> None:
        if not 1 <= desc.T <= self.geometry.rows_cap:
            raise ServerError(f"step T={desc.T} outside 1..{self.geometry.rows_cap}")
        ops().hx_srv_post_desc(self.ctx, desc.epoch_abs, desc.T, desc.bucket)
        if desc.bucket:
            g = self._graphs.get(desc.bucket)
            if g is None or desc.bucket != desc.T:
                raise ServerError(
                    f"hub announced bucket {desc.bucket} (T={desc.T}); this server "
                    f"has graphs for {sorted(self._graphs)}"
                )
            g.replay()
        else:
            self._chain(desc.T)

    def _wait_desc(self, n: int) -> StepDesc | None:
        """Next step descriptor; ``None`` when the server should stop (signal,
        hub closed the ring, hub died, or the device recorded an abort while
        idle)."""
        ring = self.ring
        if ring is None:
            raise ServerError("no step ring: the hub is not connected")
        t0 = time.monotonic()
        spin_until = t0 + 200e-6
        next_check = t0 + 0.5
        while True:
            desc = ring.get(n)
            if desc is not None:
                return desc
            if self._stop or ring.closed or self._hub_dead:
                return None
            now = time.monotonic()
            if now >= next_check:
                next_check = now + 0.5
                if self._check_abort():
                    self._abort_exit = EXIT_ABORT
                    return None
            if now < spin_until:
                time.sleep(0)
            elif now - t0 > 2.0:
                time.sleep(1e-3)
            else:
                time.sleep(1e-4)

    def _check_abort(self) -> bool:
        """True (and a log line) when the device recorded an abort."""
        rec = int(ops().hx_status(self.ctx)[P.HX_ST_RECORD])
        if rec == 0:
            return False
        logger.error(
            "hx server %s: %s", self.device_id, P.describe_record(rec, "server")
        )
        return True

    def _digest_note(self, final: bool = False) -> None:
        """Digest mode only (``flush`` has ``HX_DIGEST``): log the counters of
        the request checks when a mismatch was counted since the last call, and
        once at the end."""
        if not self.flush & P.HX_DIGEST:
            return
        words = [int(v) for v in ops().hx_dig_stats(self.ctx)]
        bad = words[P.HX_DS_BAD]
        if final or bad != self._dig_bad:
            self._dig_bad = bad
            (logger.warning if bad else logger.info)(
                "hx server %s: %s",
                self.device_id,
                P.describe_dig_stats(words, "server"),
            )

    def serve(self) -> int:
        try:
            while True:
                desc = self._wait_desc(self._next_no)
                if desc is None:
                    break
                handler = _OP_HANDLERS.get(desc.opcode)
                if handler is None:
                    logger.error(
                        "hx server %s: unknown opcode %d in step %d",
                        self.device_id,
                        desc.opcode,
                        desc.step_no,
                    )
                    self._poke()
                    return EXIT_ABORT
                handler(self, desc)
                self._next_no += 1
                ev = torch.cuda.Event()
                ev.record()
                self._inflight.append(ev)
                while len(self._inflight) > 1:
                    self._inflight.popleft().synchronize()
                if self._check_abort() and not self._stop:
                    return EXIT_ABORT
                if self._next_no % 16 == 0:
                    self._digest_note()
        except RingOverrun as e:
            logger.error("hx server %s: %s", self.device_id, e)
            self._poke()
            return EXIT_ABORT
        except ServerError as e:
            logger.error("hx server %s: %s", self.device_id, e)
            self._poke()
            return EXIT_ABORT
        # leaving: drain what is queued
        for ev in self._inflight:
            ev.synchronize()
        if self._abort_exit is not None and not self._stop:
            return self._abort_exit
        if self._hub_dead and not self._stop:
            return EXIT_ABORT
        return EXIT_OK

    def request_stop(self) -> None:
        self._stop = True
        self._poke()

    # ---- shutdown ------------------------------------------------------------

    def close(self) -> None:
        if self.ctx is not None:
            try:
                self._digest_note(final=True)
            except Exception:  # noqa: BLE001 - the device may be gone
                logger.exception("hx server: reading the digest counters failed")
        if self.watcher is not None:
            self.watcher.stop()
        if self.ctx is not None:
            try:
                ops().hx_ctx_unregister(self.ctx)
            except Exception:  # noqa: BLE001
                logger.exception("hx server: ctx unregister failed")
        for ptr in self._opened:
            try:
                close_handle(ptr)
            except Exception:  # noqa: BLE001
                logger.exception("hx server: closing the hub handle failed")
        self._opened.clear()
        d = getattr(self, "dir", None)
        if d is not None:
            rdv.remove_quiet(srv_path(d, self.device_id))
            rdv.remove_quiet(ack_path(d, self.device_id, self.hub_id))
        if self.ring is not None:
            self.ring.close()
        if self._inbox is not None:
            free_exported(self._inbox)
            self._inbox = None


def _start_signal_thread(server: ExpertServer) -> None:
    """SIGTERM/SIGINT stop the server in any state, also while the main thread sits
    in a blocking device sync (a Python handler would not run until the device
    wait times out). The C-level handler writes the signal number to a wakeup pipe
    in signal context, whichever thread receives the signal; a thread reads the
    pipe and calls ``request_stop`` (which pokes the host abort word). The Python
    handler is a no-op: it only removes the default action (terminate), which
    would otherwise hit the threads that torch and the HIP runtime created before
    any mask could be set. Main thread only."""
    rfd, wfd = os.pipe()
    os.set_blocking(wfd, False)
    signal.set_wakeup_fd(wfd, warn_on_full_buffer=False)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: None)

    def wait() -> None:
        while True:
            data = os.read(rfd, 16)
            if not data:
                return
            logger.info("hx server %s: signal %d, stopping", server.device_id, data[0])
            server.request_stop()

    threading.Thread(target=wait, name="hx-signal", daemon=True).start()


def apply_env_defaults(env: dict[str, str] | None = None) -> None:
    target = os.environ if env is None else env
    for key, value in ENV_DEFAULTS.items():
        target.setdefault(key, value)


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m vllm.distributed.hetero_moe.server",
        description="hx expert server: one V620, the experts of one plan device.",
    )
    ap.add_argument("--plan", required=True, help="placement plan JSON")
    ap.add_argument("--device-id", required=True, help="plan device id, e.g. exp0")
    ap.add_argument("--model", default=None, help="checkpoint directory")
    ap.add_argument("--ready-file", default=None)
    ap.add_argument("--buckets", default="3,6,12,24,48")
    ap.add_argument("--max-rows", type=int, default=P.HX_DEFAULT_ROWS_CAP)
    ap.add_argument("--nslots", type=int, default=P.HX_DEFAULT_NSLOTS)
    ap.add_argument("--wait-ms", type=int, default=P.HX_DEFAULT_WAIT_MS)
    ap.add_argument("--begin-wait-ms", type=int, default=P.HX_DEFAULT_BEGIN_WAIT_MS)
    ap.add_argument("--hub-timeout", type=float, default=7200.0)
    ap.add_argument(
        "--alloc",
        default=os.environ.get("VLLM_HX_ALLOC", "uncached"),
        choices=sorted(ALLOC_KINDS),
    )
    ap.add_argument(
        "--flush", type=int, default=int(os.environ.get("VLLM_HX_FLUSH", 0))
    )
    ap.add_argument(
        "--no-graphs", action="store_true", help="eager chain for every step"
    )
    ap.add_argument("--no-selftest", action="store_true")
    ap.add_argument(
        "--expect-bus",
        default=None,
        help="PCI bus (hex) the device must sit on; refuse to start otherwise",
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="weight-free dry run of the transport (no checkpoint)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    apply_env_defaults()
    # The known-answer reference runs on the CPU: a loaded 96-core box makes the
    # default thread count 150x slower on these small tensors.
    torch.set_num_threads(int(os.environ.get("VLLM_HX_SERVER_THREADS", "4")))
    from vllm.distributed.hetero_moe.plan import load_plan

    server = None
    try:
        plan = load_plan(args.plan)
        server = ExpertServer(
            plan,
            args.device_id,
            args.model,
            buckets=[int(b) for b in args.buckets.split(",") if b],
            max_rows=args.max_rows,
            wait_ms=args.wait_ms,
            begin_wait_ms=args.begin_wait_ms,
            nslots=args.nslots,
            alloc=args.alloc,
            flush=args.flush,
            selftest=not args.no_selftest,
            graphs=not args.no_graphs,
            synthetic=args.synthetic,
            hub_timeout_s=args.hub_timeout,
            ready_file=args.ready_file,
            expect_bus=args.expect_bus,
        )

        _start_signal_thread(server)
        server.prepare()
        server.publish()
        code = server.serve() if server.connect_hub() else EXIT_OK
        logger.info("hx server %s: exit %d", args.device_id, code)
        return code
    except (ServerError, TimeoutError, RuntimeError, ValueError, OSError) as e:
        logger.exception("hx server %s: setup failed: %s", args.device_id, e)
        return EXIT_SETUP
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    sys.exit(main())
