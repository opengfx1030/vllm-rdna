# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-benchmark of the Inkling-2-Mini glue kernels against the torch code.

Needs a gfx1030 GPU and a build with ``csrc/rocm/inkling2_glue_rdna2.cu``::

    python tools/rdna/inkling2/bench_glue.py --tokens 1,16,128,1024

Every scenario is the fused op sequence of ``docs/rdna2/inkling2-glue-kernels.md``
against the torch composition it replaces (``vllm/models/inkling2_loop/amd``:
``blocks.py``, ``moe.py``), at the real checkpoint dimensions.  Columns:

    fused us     the fused sequence replayed from a HIP graph (GPU time only)
    torch gpu    the torch composition replayed from a HIP graph when it is
                 capturable, else the sum of its kernel durations (profiler)
    torch eager  wall time of the eager torch composition, i.e. what the model
                 pays today when it is launch bound
    launches     GPU kernels / memory ops of the torch composition vs the fused
                 sequence (profiler event counts)

The torch composition is an oracle for the numbers only; nothing here runs in
the serving path.
"""

import argparse
import json
import time
import types
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.models.inkling2_loop.amd import blocks

H, R, E, K, INTER = 2304, 64, 128, 8, 1152
EPS = 1e-6
DEV = "cuda"
INNER = 20  # calls captured per graph


@dataclass
class Row:
    name: str
    tokens: int
    fused_us: float
    torch_gpu_us: float
    torch_eager_us: float
    torch_launches: int
    fused_launches: int


def _cfg() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        hidden_size=H,
        rms_norm_eps=EPS,
        loop_gate_rank=R,
        loop_layer_index_start=6,
        loop_layer_index_end=18,
    )


def _rand(*shape, mag=1.0, dtype=torch.float32):
    return (torch.randn(*shape, device=DEV) * mag).to(dtype)


def graph_us(fn: Callable[[], object], reps: int) -> float | None:
    """GPU us of one ``fn()`` call, replayed from a HIP graph (None: not capturable)."""
    try:
        fn()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(INNER):
                fn()
    except Exception:
        torch.accelerator.synchronize()
        return None
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) * 1e3 / (reps * INNER)


def eager_us(fn: Callable[[], object], reps: int) -> float:
    fn()
    torch.accelerator.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - t0) * 1e6 / reps


def profile(fn: Callable[[], object], reps: int = 5) -> tuple[int, float]:
    """(GPU events per call, summed GPU kernel us per call) from the profiler."""
    fn()
    torch.accelerator.synchronize()
    acts = [torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(activities=acts) as prof:
        for _ in range(reps):
            fn()
        torch.accelerator.synchronize()
    n, us = 0, 0.0
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            n += 1
            us += e.device_time if hasattr(e, "device_time") else e.cuda_time
    return n // reps, us / reps


def measure(name, tokens, fused, torch_fn, reps) -> Row:
    f_us = graph_us(fused, reps)
    n_f, k_f = profile(fused)
    t_us = graph_us(torch_fn, max(reps // 4, 20))
    n_t, k_t = profile(torch_fn)
    return Row(
        name,
        tokens,
        f_us if f_us is not None else k_f,
        t_us if t_us is not None else k_t,
        eager_us(torch_fn, max(reps // 4, 20)),
        n_t,
        n_f,
    )


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
def _pack(w_mod: torch.Tensor, fp16: bool) -> torch.Tensor:
    G, Hh, Rr = w_mod.shape
    if not fp16:
        return w_mod.transpose(1, 2).contiguous()
    w = w_mod.to(torch.float16).reshape(G, Hh, Rr // 2, 2)
    return w.permute(0, 2, 1, 3).contiguous()


class _Bench:
    def __init__(self, T: int, reps: int) -> None:
        self.T, self.reps = T, reps
        cfg = _cfg()
        self.hid = _rand(T, H)
        self.x1 = _rand(T, H)
        self.nb = 5
        self.blk = _rand(T, 8, H)
        self.prev_blk = _rand(T, 1, H)
        self.msgs = _rand(T, 4, H)
        self.bar3 = blocks.BlockAttentionResidual(cfg, 3).to(DEV)
        self.bar1 = blocks.BlockAttentionResidual(cfg, 1).to(DEV)
        self.mar = blocks.MessageAttentionResidual(cfg).to(DEV)
        for m in (self.bar3, self.bar1, self.mar):
            for p in m.parameters():
                p.data.copy_(_rand(*p.shape))
        self.gates = [
            blocks.MessageGate(cfg, dtype=torch.float16).to(DEV) for _ in range(4)
        ]
        for m in self.gates:
            m.down_proj.weight.data.copy_(_rand(R, 2 * H, mag=0.02).half())
            m.forget_gate_proj.weight.data.copy_(_rand(H, R, mag=0.05))
            m.message_gate_proj.weight.data.copy_(_rand(H, R, mag=0.05))
        self.norms = [
            blocks.Inkling2RMSNorm(H, EPS, dtype=torch.float16).to(DEV)
            for _ in range(4)
        ]
        for n in self.norms:
            n.weight.data.copy_((1 + 0.3 * _rand(H)).half())
        self.nw3 = torch.stack([n.weight.data for n in self.norms[:3]])
        self.nw1 = self.norms[3].weight.data[None]
        self.q3 = self.bar3.query.detach().view(3, H)
        self.q1 = self.bar1.query.detach().view(1, H)
        self.qmsg = self.mar.query[5].detach()

    # ---- attention residual -------------------------------------------------
    def prologue(self, full: bool) -> Row:
        T, nb = self.T, self.nb
        Q = 3 if full else 1
        bar = self.bar3 if full else self.bar1
        out = torch.empty(T, Q, H, dtype=torch.float16, device=DEV)
        hsum = torch.empty(T, H, device=DEV)
        nw = self.nw3 if full else self.nw1
        q = self.q3 if full else self.q1

        def fused():
            ops.inkling2_glue_attn_res(
                self.blk, nb, self.hid, q, out, x1=self.x1, partial_out=hsum,
                norm_w=nw, eps=EPS, norm_eps=EPS,
            )  # fmt: skip

        def torch_fn():
            with torch.no_grad():
                partial = self.hid + self.x1
                hs = bar(self.blk[:, :nb], partial)
                if full:
                    return [self.norms[i](hs[i]).to(torch.float16) for i in range(3)]
                return self.norms[3](hs[0]).to(torch.float16)

        return measure(
            "attn_res prologue " + ("Q=3 full" if full else "Q=1 sliding"),
            T, fused, torch_fn, self.reps,
        )  # fmt: skip

    def mid(self) -> Row:
        T, nb = self.T, self.nb
        out = torch.empty(T, 1, H, dtype=torch.float16, device=DEV)
        blk = self.blk.clone()

        def fused():  # block boundary layer: append the layer input
            ops.inkling2_glue_attn_res(
                blk, nb, self.x1, self.q1, out, append=self.hid, norm_w=self.nw1,
                eps=EPS, norm_eps=EPS,
            )  # fmt: skip

        def torch_fn():
            with torch.no_grad():
                states = torch.cat([self.blk[:, :nb], self.hid.unsqueeze(1)], 1)
                h = self.bar1(states, self.x1)[0]
                return self.norms[3](h).to(torch.float16)

        return measure("attn_res mid (boundary, norm)", T, fused, torch_fn, self.reps)

    def messages(self) -> Row:
        T = self.T
        out = torch.empty(T, 4, H, device=DEV)

        def fused():
            ops.inkling2_glue_attn_res(
                self.prev_blk, 1, self.hid, self.qmsg, out, x1=self.x1, eps=EPS
            )

        def torch_fn():
            with torch.no_grad():
                return self.mar(self.prev_blk, self.hid + self.x1, 5)

        return measure("attn_res messages Q=4", T, fused, torch_fn, self.reps)

    # ---- message gates ------------------------------------------------------
    def gates_op(self, G: int, w16: bool) -> Row:
        T = self.T
        gs = self.gates[:G]
        wd = torch.stack([m.down_proj.weight.data for m in gs])
        wf = _pack(torch.stack([m.forget_gate_proj.weight.data for m in gs]), w16)
        wg = _pack(torch.stack([m.message_gate_proj.weight.data for m in gs]), w16)
        h = _rand(T, G, H)
        m_in = self.msgs[:, :G]
        nw = self.nw3 if G == 3 else self.nw1
        d = torch.empty(G, T, R, dtype=torch.float16, device=DEV)
        out = torch.empty(G, T, H, dtype=torch.float16, device=DEV)
        scratch = torch.empty(G, T, H, device=DEV)
        parts = torch.empty(G, T, (H + 127) // 128, device=DEV)
        counters = torch.zeros(G * T, dtype=torch.int32, device=DEV)

        def fused():
            ops.inkling2_glue_msg_gate_down(h, m_in, wd, d, EPS)
            ops.inkling2_glue_msg_gate_mix(
                h, m_in, d, wf, wg, out, norm_w=nw, scratch=scratch,
                partials=parts, counters=counters, norm_eps=EPS,
            )  # fmt: skip

        def torch_fn():
            with torch.no_grad():
                return [
                    self.norms[g](gs[g](h[:, g], m_in[:, g])).to(torch.float16)
                    for g in range(G)
                ]

        tag = "fp16" if w16 else "fp32"
        return measure(
            f"msg gates G={G} ({tag} gain/forget)", T, fused, torch_fn, self.reps
        )

    # ---- router / MoE -------------------------------------------------------
    def route(self, w16: bool) -> Row:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        T = self.T
        BM = 1 if T <= 1 else 2 if T <= 2 else 4 if T <= 4 else 8
        gate_w = _rand(E, H, mag=1 / H**0.5)
        bias = 0.1 * _rand(E)
        x = _rand(T, H).half()
        max_padded, max_blocks, n_counters = ops.inkling2_glue_moe_route_sizes(
            T, E, K, BM
        )
        logits = torch.empty(T, E, device=DEV)
        tw = torch.empty(T, K, device=DEV)
        ti = torch.empty(T, K, dtype=torch.int32, device=DEV)
        sid = torch.empty(max_padded, dtype=torch.int32, device=DEV)
        eid = torch.empty(max_blocks, dtype=torch.int32, device=DEV)
        pp = torch.empty(1, dtype=torch.int32, device=DEV)
        counters = torch.zeros(n_counters, dtype=torch.int32, device=DEV)
        z1 = torch.empty(T * K * 2 * INTER, dtype=torch.float16, device=DEV)
        z2 = torch.empty(T * H, dtype=torch.float16, device=DEV)
        w_k = gate_w.half() if w16 else gate_w

        def fused():
            ops.inkling2_glue_moe_route(
                x, w_k, bias, logits, tw, ti, sid, eid, pp, counters, BM, 2.5,
                zero1=z1, zero2=z2,
            )  # fmt: skip

        def torch_fn():
            with torch.no_grad():
                lg = F.linear(x.float(), gate_w)
                w, ids = blocks.route_topk(lg, bias, K, 2.5)
                s, e, n = moe_align_block_size(ids, BM, E)
                a = torch.zeros(T * K, 2 * INTER, dtype=torch.float16, device=DEV)
                b = torch.zeros(T, H, dtype=torch.float16, device=DEV)
                return w, s, e, n, a, b

        tag = "fp16" if w16 else "fp32"
        return measure(
            f"moe route+align+zero ({tag} router)", T, fused, torch_fn, self.reps
        )

    def swiglu(self) -> Row:
        T = self.T
        gu = _rand(T * K, 2 * INTER, mag=4.0).half()
        out = torch.empty(T * K, INTER, dtype=torch.float16, device=DEV)

        def fused():
            ops.inkling2_glue_swiglu(gu, out, 4.0, 25.0)

        def torch_fn():
            g, u = gu.chunk(2, -1)
            return (
                blocks.softcapped_swiglu(g, u, 4.0, 25.0).to(torch.float16).contiguous()
            )

        return measure("softcapped swiglu", T, fused, torch_fn, self.reps)

    # ---- norms --------------------------------------------------------------
    def final_norm(self) -> Row:
        T = self.T
        out = torch.empty(T, H, dtype=torch.float16, device=DEV)
        hsum = torch.empty(T, H, device=DEV)

        def fused():
            ops.inkling2_glue_rms_norm(
                self.hid, out, weight=self.nw1[0], x1=self.x1, sum_out=hsum, eps=EPS
            )

        def torch_fn():
            with torch.no_grad():
                return self.norms[3](self.hid + self.x1).to(torch.float16)

        return measure("rms_norm (+lazy add)", T, fused, torch_fn, self.reps)

    def embed(self) -> Row:
        T = self.T
        V = 8192
        table = _rand(V, H, mag=0.05).half()
        ids = torch.randint(0, V, (T,), device=DEV)
        out = torch.empty(T, H, device=DEV)

        def fused():
            ops.inkling2_glue_embed_norm(ids, table, self.nw1[0], out, EPS)

        def torch_fn():
            with torch.no_grad():
                return self.norms[3](F.embedding(ids, table)).float()

        return measure("embed gather + norm", T, fused, torch_fn, self.reps)


def _fmt(rows: list[Row]) -> str:
    head = (
        f"{'scenario':<42}{'T':>6}{'fused us':>10}{'torch gpu':>11}"
        f"{'torch eager':>12}{'speedup':>9}{'launches':>12}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r.name:<42}{r.tokens:>6}{r.fused_us:>10.1f}{r.torch_gpu_us:>11.1f}"
            f"{r.torch_eager_us:>12.1f}{r.torch_gpu_us / r.fused_us:>8.1f}x"
            f"{r.torch_launches:>7} ->{r.fused_launches:>2}"
        )
    return "\n".join(lines)


def visit_summary(rows: list[Row], T: int) -> list[str]:
    """Glue totals of one forward (48 layer visits) from the measured rows."""

    def pick(prefix: str) -> Row:
        return next(r for r in rows if r.tokens == T and r.name.startswith(prefix))

    pro_f, pro_s = pick("attn_res prologue Q=3"), pick("attn_res prologue Q=1")
    mid, msg = pick("attn_res mid"), pick("attn_res messages")
    g3, g1 = pick("msg gates G=3 (fp16"), pick("msg gates G=1 (fp16")
    route, act = pick("moe route+align+zero (fp16"), pick("softcapped swiglu")
    # 48 visits: 8 full-layer + 40 sliding; 24 of them (loop passes >= 1) have messages
    plan = [  # (rows, visits)
        ([pro_f, mid, route, act], 4),
        ([pro_s, mid, route, act], 20),
        ([pro_f, mid, route, act, msg, g3, g1], 4),
        ([pro_s, mid, route, act, msg, g3, g1], 20),
    ]
    tot = dict(f=0.0, tg=0.0, te=0.0, ln_t=0, ln_f=0)
    for group, n in plan:
        for r in group:
            tot["f"] += n * r.fused_us
            tot["tg"] += n * r.torch_gpu_us
            tot["te"] += n * r.torch_eager_us
            tot["ln_t"] += n * r.torch_launches
            tot["ln_f"] += n * r.fused_launches
    return [
        f"T={T}: glue of one forward (48 layer visits, no embed / final norm):",
        f"  fused  : {tot['f'] / 1e3:8.2f} ms GPU, {tot['ln_f']:6d} launches",
        (
            f"  torch  : {tot['tg'] / 1e3:8.2f} ms GPU, {tot['ln_t']:6d} launches, "
            f"{tot['te'] / 1e3:8.2f} ms eager wall"
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,16,128,1024")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    rows: list[Row] = []
    for T in (int(t) for t in args.tokens.split(",")):
        b = _Bench(T, args.reps)
        rows += [
            b.prologue(True),
            b.prologue(False),
            b.mid(),
            b.messages(),
            b.gates_op(3, True),
            b.gates_op(3, False),
            b.gates_op(1, True),
            b.gates_op(1, False),
            b.route(True),
            b.route(False),
            b.swiglu(),
            b.final_norm(),
            b.embed(),
        ]
    print(_fmt(rows))
    print()
    for T in sorted({r.tokens for r in rows}):
        print("\n".join(visit_summary(rows, T)))
    if args.json:
        with open(args.json, "w") as f:
            json.dump([r.__dict__ for r in rows], f, indent=1)


if __name__ == "__main__":
    main()
