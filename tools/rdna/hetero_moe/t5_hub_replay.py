# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T5 hub side: replays layer calls through the real ``hx`` transport. Needs a GPU.

One process on the hub GPU (the W7800, or a V620 stand-in) connects to the island's
expert-server processes (``t5_run.py run`` starts them) and replays the
deterministic input family of ``t5_common`` for every hx layer of the plan:

``verify``  eager steps. Right after each ``combine`` the four reply slots are
            copied into tap buffers (same stream), so the servers' replies of every
            layer are compared with the golden file (digests, and raw bytes up to
            width 48), the combine output with the fp32 reference, and
            ``hub_ids_out`` with the split rule.
``graph``   widths <= 48: the whole step (begin_step + 48 x dispatch/combine/taps)
            is captured once and replayed; every replay is verified like ``verify``.
``bench``   latency without taps: per-layer CUDA events in eager mode, step events
            around the replay in graph mode (per layer = step / layers).
``soak``    graph replays for ``--iters`` or ``--seconds`` per width; the digest of
            the last layer's output is checked on every replay, ``hub.check()`` too.
``serve``   graph replays until SIGTERM, no verification (fault injection); prints
            ``STEP <n>`` every 100 replays and ``HX_ABORT <why>`` (exit 3) when the
            transport aborts.

``--mode`` takes a comma separated list: the servers serve exactly one hub process
(they exit when the hub closes its ring), so one process runs all requested modes.

    HIP_VISIBLE_DEVICES=0 PYTHONPATH=<tree> \\
      python -m tools.rdna.hetero_moe.t5_hub_replay --plan P --hub hub0 \\
        --golden-dir D --seed 1 --mode verify,graph,bench --out r.json

Exit codes: 0 pass, 1 mismatch, 3 transport abort, 4 setup failure.
"""

import argparse
import collections
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from tools.rdna.hetero_moe import t5_common as C
from vllm.distributed.hetero_moe import hx
from vllm.distributed.hetero_moe import hx_proto as P
from vllm.distributed.hetero_moe.ctx import ops
from vllm.distributed.hetero_moe.hx import HxAbort, HxHub

HIDDEN = 2560
X_POOL = 4
GRAPH_MAX_WIDTH = C.RAW_MAX_WIDTH
MODES = ("verify", "graph", "bench", "soak", "serve")
DEFAULT_ITERS = {"graph": 8, "bench": 200, "soak": 1000}


def parse_modes(text: str) -> list[str]:
    modes = [m for m in text.replace(" ", "").split(",") if m]
    bad = [m for m in modes if m not in MODES]
    if bad or not modes:
        raise argparse.ArgumentTypeError(f"modes {bad or text!r}: choose from {MODES}")
    if "serve" in modes and len(modes) > 1:
        raise argparse.ArgumentTypeError("serve runs alone")
    return modes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--plan", required=True)
    p.add_argument("--hub", default="hub0")
    p.add_argument("--golden-dir", default=None, help="needed by verify/graph/soak")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mode", type=parse_modes, default=["verify"])
    p.add_argument(
        "--widths",
        "--width-list",
        dest="widths",
        type=C.parse_int_list,
        default=None,
        help="default 3,24,48,2048; graph/soak/serve use the widths <= 48",
    )
    p.add_argument("--patterns", type=int, default=C.DEFAULT_PATTERNS)
    p.add_argument(
        "--layers", type=int, default=None, help="compare the first N layers"
    )
    p.add_argument("--iters", type=int, default=None, help="steps/replays per width")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seconds", type=float, default=0.0, help="soak seconds per width")
    p.add_argument("--spill-k", default=None, help='override the plan K, "3:10,24:20"')
    p.add_argument("--connect-timeout", type=float, default=300.0)
    p.add_argument("--wait-ms", type=int, default=P.HX_DEFAULT_WAIT_MS)
    p.add_argument("--nslots", type=int, default=P.HX_DEFAULT_NSLOTS)
    p.add_argument("--max-rows", type=int, default=P.HX_DEFAULT_ROWS_CAP)
    p.add_argument("--out", default=None, help="JSON results")
    p.add_argument(
        "--serve-eager",
        action="store_true",
        help="serve: eager steps instead of graph replays (freeze-hub case)",
    )
    p.add_argument("--freeze-trigger", default=None, help="serve-eager: SIGSTOP self")
    p.add_argument("--freeze-layer", type=int, default=5, help="layer index to stop at")
    return p.parse_args(argv)


# ---- host and device step data -----------------------------------------------


class HostStep:
    """The inputs of one (width, pattern) on the host, plus the hub partials."""

    def __init__(self, inp: C.StepInputs, y_hub: np.ndarray):
        self.inp = inp
        self.y_hub = y_hub  # fp16 [L, T, H]


class Buffers:
    """Static device buffers of one width (a graph needs fixed addresses)."""

    def __init__(self, r: "Replay", T: int, taps: bool):
        dev, n, k = r.device, r.L, r.top_k
        self.T = T
        self.x = torch.empty((X_POOL, T, HIDDEN), dtype=torch.float16, device=dev)
        self.ids = torch.empty((n, T, k), dtype=torch.int32, device=dev)
        self.w = torch.empty((n, T, k), dtype=torch.float32, device=dev)
        self.y_hub = torch.empty((n, T, HIDDEN), dtype=torch.float16, device=dev)
        self.out = torch.empty((n, T, HIDDEN), dtype=torch.float16, device=dev)
        self.hub_ids = torch.empty((n, T, k), dtype=torch.int32, device=dev)
        self.tap = None
        self.views = None
        if taps:
            self.tap = torch.empty((n, r.S, T, HIDDEN), dtype=torch.float16, device=dev)
            self.views = r.reply_views(T)

    def upload(self, host: HostStep) -> None:
        self.x.copy_(torch.from_numpy(host.inp.x))
        self.ids.copy_(torch.from_numpy(host.inp.ids))
        self.w.copy_(torch.from_numpy(host.inp.w))
        self.y_hub.copy_(torch.from_numpy(host.y_hub))

    def load(self, stage: "Stage") -> None:
        self.x.copy_(stage.x)
        self.ids.copy_(stage.ids)
        self.w.copy_(stage.w)
        if stage.y_hub is not None:
            self.y_hub.copy_(stage.y_hub)


class Stage:
    """Device copy of the inputs of one pattern (device-to-device reloads)."""

    def __init__(self, r: "Replay", host: HostStep, with_y_hub: bool):
        dev = r.device
        self.x = torch.from_numpy(host.inp.x).to(dev)
        self.ids = torch.from_numpy(host.inp.ids).to(dev)
        self.w = torch.from_numpy(host.inp.w).to(dev)
        self.y_hub = torch.from_numpy(host.y_hub).to(dev) if with_y_hub else None


class LayerEvents:
    def __init__(self, n: int):
        self.start = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        self.end = [torch.cuda.Event(enable_timing=True) for _ in range(n)]


class Stopped(Exception):
    """SIGTERM/SIGINT arrived."""


# ---- the replay --------------------------------------------------------------


class Replay:
    def __init__(self, args, plan, tables, hub: HxHub, goldens, stop: threading.Event):
        self.args = args
        self.plan = plan
        self.tables = tables
        self.hub = hub
        self.goldens = goldens
        self.stop = stop
        self.layers = C.hx_layers(tables)
        self.L = len(self.layers)
        self.S = hub.nsrv
        self.top_k = plan.top_k
        self.nslots = hub.geometry.nslots
        self.reply = hub.geometry.reply(hub.nsrv)
        self.dev_index = torch.empty(1, device="cuda").device.index
        self.device = torch.device("cuda", self.dev_index)
        self.compare_n = self.L if args.layers is None else min(args.layers, self.L)

    # ---- helpers ----

    def reply_views(self, T: int):
        base = self.hub._reply.ptr
        return [
            [
                ops().hx_view(
                    base + self.reply.y_off(s, j),
                    [T, HIDDEN],
                    torch.float16,
                    self.dev_index,
                )
                for j in range(self.nslots)
            ]
            for s in range(self.S)
        ]

    def host_step(self, W: int, pattern: int) -> HostStep:
        a = self.args
        inp = C.make_inputs(
            a.seed,
            W,
            pattern,
            nlayers=self.L,
            num_experts=self.plan.num_experts,
            top_k=self.top_k,
            hidden=HIDDEN,
            x_pool=X_POOL,
        )
        y_hub = np.stack(
            [
                C.synthetic_y_hub(a.seed, W, pattern, layer, W, HIDDEN)
                for layer in self.layers
            ]
        )
        return HostStep(inp, y_hub)

    def check_stop(self) -> None:
        if self.stop.is_set():
            raise Stopped

    def check_goldens(self, widths: list[int]) -> None:
        """The golden files must describe this plan, layers, K and pattern count."""
        for g in self.goldens:
            if g.layers != self.layers:
                raise C.GoldenError(
                    f"golden of {g.meta['server_id']}: layers {g.layers[:3]}... "
                    f"differ from the hub's {self.layers[:3]}..."
                )
            for W in widths:
                if g.k_keep(W) != self.tables.k_for(W):
                    raise C.GoldenError(
                        f"golden of {g.meta['server_id']}: K({W}) = {g.k_keep(W)}, "
                        f"the hub's tables say {self.tables.k_for(W)} (--spill-k?)"
                    )
                if g.patterns(W) < self.args.patterns:
                    raise C.GoldenError(
                        f"golden of {g.meta['server_id']}: {g.patterns(W)} patterns "
                        f"at W={W}, {self.args.patterns} requested"
                    )

    def input_problems(self, W: int, pattern: int, host: HostStep) -> list[str]:
        mine = C.inputs_digest(host.inp)
        return [
            f"W={W} pattern {pattern}: input generation differs from the golden of "
            f"{g.meta['server_id']} ({mine} != {g.inputs_digest(W, pattern)})"
            for g in self.goldens
            if g.inputs_digest(W, pattern) != mine
        ]

    # ---- enqueue ----

    def enqueue(
        self, buf: Buffers, bucket: int, *, taps: bool, ev=None, hook=None
    ) -> None:
        """One step on the current stream: also what a graph capture records. No host
        sync in here. Taps are enqueued right after each combine: the reply slot of
        layer l is not rewritten before dispatch(l + nslots). ``hook(i)`` runs on the
        host after the i-th layer was enqueued (eager serve only)."""
        hub = self.hub
        hx.begin_step(hub, buf.T, bucket)
        for i, layer in enumerate(self.layers):
            if ev is not None:
                ev.start[i].record()
            hx.dispatch(
                hub,
                buf.x[layer % X_POOL],
                buf.ids[i],
                buf.w[i],
                buf.hub_ids[i],
                layer,
            )
            hx.combine(hub, buf.y_hub[i], buf.out[i], layer)
            if ev is not None:
                ev.end[i].record()
            if taps:
                for s in range(self.S):
                    buf.tap[i, s].copy_(buf.views[s][layer % self.nslots])
            if hook is not None:
                hook(i)

    def step_eager(self, buf: Buffers, *, taps: bool, ev=None) -> None:
        self.hub.post_step(buf.T, 0)
        self.enqueue(buf, 0, taps=taps, ev=ev)
        torch.cuda.current_stream().synchronize()
        self.hub.check()

    def capture(self, buf: Buffers, *, taps: bool) -> torch.cuda.CUDAGraph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.enqueue(buf, buf.T, taps=taps)
        return graph

    def replay(self, graph: torch.cuda.CUDAGraph, W: int) -> None:
        self.hub.post_step(W, W)
        graph.replay()

    # ---- verification ----

    def compare(self, buf: Buffers, host: HostStep, W: int, pattern: int) -> list[str]:
        """Compare the first ``--layers`` layers of the step in ``buf``."""
        problems: list[str] = []
        k_keep = self.tables.k_for(W)
        for i in range(self.compare_n):
            layer = self.layers[i]
            want_hub, _ = C.expected_split(self.tables, layer, host.inp.ids[i], k_keep)
            try:
                problems += C.check_layer(
                    layer,
                    buf.tap[i].cpu().numpy(),
                    buf.out[i].cpu().numpy(),
                    buf.hub_ids[i].cpu().numpy(),
                    host.y_hub[i],
                    want_hub,
                    [g.layer_digest(W, pattern, layer) for g in self.goldens],
                    [g.layer_y(W, pattern, i) for g in self.goldens],
                )
            except KeyError as exc:
                problems.append(f"layer {layer}: golden entry {exc} missing")
        return [f"W={W} pattern {pattern}: {m}" for m in problems]

    # ---- modes ----

    def verify(self, widths: list[int]) -> tuple[bool, list[dict]]:
        records = []
        ok = True
        for W in widths:
            t0 = time.monotonic()
            problems: list[str] = []
            buf = Buffers(self, W, taps=True)
            for pattern in range(self.args.patterns):
                self.check_stop()
                host = self.host_step(W, pattern)
                problems += self.input_problems(W, pattern, host)
                buf.upload(host)
                self.step_eager(buf, taps=True)
                problems += self.compare(buf, host, W, pattern)
            del buf
            torch.accelerator.empty_cache()
            ok &= self._report("verify", W, problems, t0, records)
        return ok, records

    def graph(self, widths: list[int]) -> tuple[bool, list[dict]]:
        records = []
        ok = True
        replays = max(self.args.iters or DEFAULT_ITERS["graph"], 1)
        for W in widths:
            t0 = time.monotonic()
            problems: list[str] = []
            buf = Buffers(self, W, taps=True)
            hosts = [self.host_step(W, p) for p in range(self.args.patterns)]
            for p, host in enumerate(hosts):
                problems += self.input_problems(W, p, host)
            # eager warm-up: kernels load before the capture, servers run their chain
            buf.upload(hosts[0])
            self.step_eager(buf, taps=True)
            problems += ["warm-up " + m for m in self.compare(buf, hosts[0], W, 0)]
            graph = self.capture(buf, taps=True)
            for r in range(replays):
                self.check_stop()
                pattern = r % self.args.patterns
                buf.upload(hosts[pattern])
                self.replay(graph, W)
                torch.cuda.current_stream().synchronize()
                self.hub.check()
                problems += [
                    f"replay {r}: {m}"
                    for m in self.compare(buf, hosts[pattern], W, pattern)
                ]
            del graph, buf
            torch.accelerator.empty_cache()
            ok &= self._report("graph", W, problems, t0, records, replays=replays)
        return ok, records

    def bench(self, widths: list[int]) -> tuple[bool, list[dict]]:
        iters = self.args.iters or DEFAULT_ITERS["bench"]
        rows, records = [], []
        ok = True
        for W in widths:
            samples = self._bench_eager(W, iters)
            row, rec = C.latency_row(W, "eager", samples)
            rows.append(row)
            records.append(rec)
            if W <= GRAPH_MAX_WIDTH:
                steps = self._bench_graph(W, iters)
                row, rec = C.latency_row(W, "graph", [s / self.L for s in steps])
                rec["step_us"] = C.percentiles(steps)
                rows.append(row)
                records.append(rec)
                if rec.get("ratio_mean") is not None:
                    limit = C.MODEL_TOLERANCE
                    good = rec["ratio_mean"] <= limit
                    ok &= C.emit(
                        "bench",
                        good,
                        f"W={W} graph mean {rec['mean']:.1f} us/layer = "
                        f"{rec['ratio_mean']:.2f} x model {rec['model_us']:.0f} us "
                        f"(limit {limit} x)",
                    )
        print(C.format_table(C.LATENCY_HEADERS, rows), flush=True)
        return ok, records

    def _bench_stages(self, W: int, buf: Buffers) -> list[Stage]:
        host0 = self.host_step(W, 0)
        buf.y_hub.copy_(torch.from_numpy(host0.y_hub))
        stages = [Stage(self, host0, with_y_hub=False)]
        for p in range(1, self.args.patterns):
            stages.append(Stage(self, self.host_step(W, p), with_y_hub=False))
        return stages

    def _bench_eager(self, W: int, iters: int) -> list[float]:
        buf = Buffers(self, W, taps=False)
        stages = self._bench_stages(W, buf)
        ev = LayerEvents(self.L)
        samples: list[float] = []
        for it in range(self.args.warmup + iters):
            self.check_stop()
            buf.load(stages[it % len(stages)])
            self.step_eager(buf, taps=False, ev=ev)
            if it >= self.args.warmup:
                samples += [
                    ev.start[i].elapsed_time(ev.end[i]) * 1e3 for i in range(self.L)
                ]
        del buf, stages
        torch.accelerator.empty_cache()
        return samples

    def _bench_graph(self, W: int, iters: int) -> list[float]:
        buf = Buffers(self, W, taps=False)
        stages = self._bench_stages(W, buf)
        buf.load(stages[0])
        self.step_eager(buf, taps=False)
        graph = self.capture(buf, taps=False)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        steps: list[float] = []
        for it in range(self.args.warmup + iters):
            self.check_stop()
            buf.load(stages[it % len(stages)])
            self.hub.post_step(W, W)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            self.hub.check()
            if it >= self.args.warmup:
                steps.append(start.elapsed_time(end) * 1e3)
        del graph, buf, stages
        torch.accelerator.empty_cache()
        return steps

    def soak(self, widths: list[int]) -> tuple[bool, list[dict]]:
        records = []
        ok = True
        for W in widths:
            ok &= self._soak_width(W, records)
        return ok, records

    def _soak_width(self, W: int, records: list[dict]) -> bool:
        a = self.args
        hosts = [self.host_step(W, p) for p in range(a.patterns)]
        problems: list[str] = []
        want_last: list[str] = []
        for p, host in enumerate(hosts):
            problems += self.input_problems(W, p, host)
            ys = [g.layer_y(W, p, self.L - 1) for g in self.goldens]
            if any(y is None for y in ys):
                raise C.GoldenError(
                    f"soak needs raw golden bytes: W={W} <= {GRAPH_MAX_WIDTH}"
                )
            want_last.append(C.digest(C.combine_reference(host.y_hub[-1], ys)))
        buf = Buffers(self, W, taps=False)
        stages = [Stage(self, h, with_y_hub=True) for h in hosts]
        buf.load(stages[0])
        self.step_eager(buf, taps=False)
        graph = self.capture(buf, taps=False)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        iters = max(a.iters or DEFAULT_ITERS["soak"], 1)
        deadline = time.monotonic() + a.seconds if a.seconds > 0 else None
        t0 = time.monotonic()
        n = mismatches = 0
        max_ms = 0.0
        while True:
            if deadline is not None:
                if time.monotonic() >= deadline:
                    break
            elif n >= iters:
                break
            self.check_stop()
            pattern = n % len(stages)
            buf.load(stages[pattern])
            self.hub.post_step(W, W)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            self.hub.check()
            max_ms = max(max_ms, start.elapsed_time(end))
            if C.digest(buf.out[-1].cpu().numpy()) != want_last[pattern]:
                mismatches += 1
                if mismatches <= 10:
                    problems.append(
                        f"replay {n} pattern {pattern}: last-layer output digest "
                        "differs from the golden-derived one"
                    )
            n += 1
        del graph, buf, stages
        torch.accelerator.empty_cache()
        rec = {
            "iterations": n,
            "mismatches": mismatches,
            "max_step_us": max_ms * 1e3,
        }
        return self._report("soak", W, problems, t0, records, **rec)

    def serve(self, W: int) -> None:
        """Replays until SIGTERM. ``--serve-eager`` enqueues the eager step instead of
        a graph and, once ``--freeze-trigger`` exists, stops the process itself
        (SIGSTOP) after layer ``--freeze-layer`` of a step: the servers are then
        left waiting for a request mid-step."""
        a = self.args
        buf = Buffers(self, W, taps=False)
        stage = Stage(self, self.host_step(W, 0), with_y_hub=True)
        buf.load(stage)
        self.step_eager(buf, taps=False)
        if a.serve_eager:

            def launch() -> None:
                self.hub.post_step(W, 0)
                self.enqueue(buf, 0, taps=False, hook=self._maybe_freeze)
        else:
            graph = self.capture(buf, taps=False)

            def launch() -> None:
                self.replay(graph, W)

        print(f"SERVING W={W} layers={self.L} eager={a.serve_eager}", flush=True)
        inflight: collections.deque = collections.deque()
        n = 0
        while not self.stop.is_set():
            launch()
            ev = torch.cuda.Event()
            ev.record()
            inflight.append(ev)
            while len(inflight) > 2:
                inflight.popleft().synchronize()
            self.hub.check()
            n += 1
            if n % 100 == 0:
                print(f"STEP {n}", flush=True)
        torch.cuda.current_stream().synchronize()

    def _maybe_freeze(self, index: int) -> None:
        a = self.args
        if (
            index == a.freeze_layer
            and a.freeze_trigger
            and os.path.exists(a.freeze_trigger)
        ):
            os.kill(os.getpid(), signal.SIGSTOP)

    # ---- report ----

    def _report(
        self,
        tag: str,
        W: int,
        problems: list[str],
        t0: float,
        records: list[dict],
        **extra,
    ) -> bool:
        ok = not problems
        detail = f"W={W} patterns={self.args.patterns} layers={self.compare_n}"
        detail += "".join(f" {k}={v}" for k, v in extra.items())
        detail += f" servers={self.S} {time.monotonic() - t0:.1f} s"
        if problems:
            detail += f" {len(problems)} problem(s); first: {problems[0]}"
        C.emit(tag, ok, detail)
        for line in problems[1:10]:
            print(f"T5 {tag} detail: {line}", flush=True)
        records.append(
            {
                "mode": tag,
                "width": W,
                "ok": ok,
                "problems": problems[:20],
                "n_problems": len(problems),
                **extra,
            }
        )
        return ok


# ---- process -------------------------------------------------------------------------


def abort_line(exc: HxAbort) -> str:
    """``HX_ABORT <describe> phase=<n> peer=<k>`` (phase 0 / peer -1: host-side)."""
    rec = P.decode_record(exc.record) if exc.record else None
    phase, peer = (rec.phase, rec.peer) if rec and rec.valid else (0, -1)
    return f"HX_ABORT {exc} phase={phase} peer={peer}"


def write_results(path: str | None, results: dict) -> None:
    if path:
        Path(path).write_text(json.dumps(results, indent=1, default=str))


def hard_exit(hub: HxHub, code: int) -> None:
    """Leave now: close the hub on a side thread (at most 3 s: after an abort a
    device call may hang) and skip the interpreter teardown."""
    sys.stdout.flush()
    sys.stderr.flush()
    closer = threading.Thread(target=hub.close, daemon=True)
    closer.start()
    closer.join(3.0)
    os._exit(code)


def setup(args, stop):
    """Plan, tables, hub, goldens -> ``Replay``; raises on any setup problem."""
    from vllm.distributed.hetero_moe.plan import load_plan

    plan = load_plan(args.plan)
    tables = C.build_tables(plan, args.hub, args.spill_k)
    need_golden = any(m in ("verify", "graph", "soak") for m in args.mode)
    hub = HxHub.create(
        plan,
        args.hub,
        tables=tables,
        max_rows=args.max_rows,
        wait_ms=args.wait_ms,
        connect_timeout_s=args.connect_timeout,
        nslots=args.nslots,
    )
    try:
        goldens = []
        if need_golden:
            if not args.golden_dir:
                raise C.GoldenError("--golden-dir is required by verify/graph/soak")
            goldens = [
                C.load_golden(
                    C.golden_path(args.golden_dir, sid),
                    plan_id=plan.plan_id,
                    server_id=sid,
                )
                for sid in hub.server_ids
            ]
        return plan, hub, Replay(args, plan, tables, hub, goldens, stop)
    except BaseException:
        hub.close()
        raise


def widths_for(mode: str, requested: list[int] | None) -> list[int]:
    widths = list(requested or C.DEFAULT_WIDTHS)
    if mode in ("graph", "soak", "serve"):
        widths = [w for w in widths if w <= GRAPH_MAX_WIDTH]
    if mode == "serve":
        widths = widths[:1] or [3]
    return widths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # The recipe environment must be in place before the first HIP call.
    from vllm.distributed.hetero_moe import server

    server.apply_env_defaults()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    t_setup = time.monotonic()
    try:
        plan, hub, replay = setup(args, stop)
    except Exception as exc:  # noqa: BLE001 - one clear line, exit 4
        C.emit(
            "connect",
            False,
            f"{type(exc).__name__}: {exc} (waited {time.monotonic() - t_setup:.1f} s)",
        )
        return C.EXIT_SETUP
    results: dict = {
        "plan_id": plan.plan_id,
        "hub": args.hub,
        "seed": args.seed,
        "modes": {},
    }
    code = C.EXIT_OK
    try:
        for mode in args.mode:
            widths = widths_for(mode, args.widths)
            if mode in ("verify", "graph", "soak"):
                replay.check_goldens(widths)
            if mode == "serve":
                replay.serve(widths[0])
                continue
            ok, records = getattr(replay, mode)(widths)
            results["modes"][mode] = {"ok": ok, "records": records}
            if not ok:
                code = C.EXIT_MISMATCH
    except Stopped:
        C.emit("stop", True, "signal received")
    except HxAbort as exc:
        line = abort_line(exc)
        print(line, flush=True)
        results["abort"] = line
        write_results(args.out, results)
        hard_exit(hub, C.EXIT_ABORT)
    except C.GoldenError as exc:
        C.emit("golden", False, str(exc))
        code = C.EXIT_SETUP
    finally:
        hub.close()
    write_results(args.out, results)
    return code


if __name__ == "__main__":
    sys.exit(main())
