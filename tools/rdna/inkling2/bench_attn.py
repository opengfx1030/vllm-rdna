# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-benchmark: Inkling-2 gfx1030 attention kernels vs the eager torch path.

Kernel side: ``inkling2_attention``, ``inkling2_kv_prep`` and
``inkling2_sconv_static`` (``csrc/rocm/inkling2_attn_rdna2.cu``), timed as
CUDA-graph replays of ``--reps`` back-to-back launches (device time plus the
inter-kernel gap, which is what a captured decode step pays).

Torch side: the repo's own eager path (``Inkling2Attention._attend``,
``conv_taps``, ``kv_write`` ... in ``vllm/models/inkling2_loop/amd``), timed
eagerly because it synchronises on the host (``.tolist()``); that is the cost the
kernels replace.  The torch path keeps fp32 K/V copies, so the 128k contexts
need ~3 GB of scratch.

    python tools/rdna/inkling2/bench_attn.py --md          # full table
    python tools/rdna/inkling2/bench_attn.py --sweep-splits --ctx 131072
"""

from __future__ import annotations

import argparse
import json
import math
import types

import torch

from vllm import _custom_ops as ops
from vllm.models.inkling2_loop.amd import attention as attn_mod
from vllm.models.inkling2_loop.amd import blocks
from vllm.models.inkling2_loop.amd import state as state_mod

D = 128
NH, NKV = 24, 8  # TP=1 geometry
D_REL = 16
WINDOW = 1024
REL_EXTENT = 1024
KV_BLOCK = 64
CONV_BLOCK = 4
Q_CLIP = 22.627416997969522
REL_CLIP = 8.0
CFG = types.SimpleNamespace(
    log_scaling_alpha=0.1,
    log_scaling_n_floor=8192,
    qk_l2_clip_threshold=Q_CLIP,
    rel_l2_clip_threshold=REL_CLIP,
    rms_norm_eps=1e-6,
)
STREAMS = ((0, D), (D, D), (2 * D, 288), (2 * D + 288, 288))  # TP=1 widths
DPAD = 1024


def cdiv(a, b):
    return -(-a // b)


LAYOUT = "bhc"  # physical page layout: "bhc" = [B, bs, H, C], "hbc" = [B, H, bs, C]


def engine_cache(pages, heads, bs, width, dtype, fill):
    """[B, H, bs, C] cache; physically [B, bs, H, C] (the engine's block-outermost
    page array seen through a permuted view) unless ``LAYOUT == "hbc"``."""
    if LAYOUT == "hbc":
        page = torch.empty(pages, heads, bs, width, dtype=dtype, device="cuda")
    else:
        page = torch.empty(pages, bs, heads, width, dtype=dtype, device="cuda")
    if fill == "rand":
        page.normal_()
    else:
        page.fill_(float("nan"))
    return page if LAYOUT == "hbc" else page.permute(0, 2, 1, 3)


class World:
    """Decode / prefill world with random K|V pages and engine-layout caches.

    ``conv=False`` skips the conv caches and conv-side tensors (attention-only
    timing worlds)."""

    def __init__(self, q_lens, seq_lens, sliding: bool, seed=0, conv=True):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.q_lens, self.seq_lens, self.sliding = q_lens, seq_lens, sliding
        self.R, self.T = len(q_lens), sum(q_lens)
        self.window = WINDOW if sliding else 0
        nb = [cdiv(L, KV_BLOCK) for L in seq_lens]
        kv_pages = sum(nb) + 1
        self.kv = engine_cache(kv_pages, NKV, KV_BLOCK, 2 * D, torch.float16, "rand")
        perm = torch.randperm(kv_pages - 1, device="cuda", generator=g) + 1
        self.table = torch.zeros(self.R, max(nb), dtype=torch.int32, device="cuda")
        i = 0
        for r in range(self.R):
            self.table[r, : nb[r]] = perm[i : i + nb[r]].int()
            i += nb[r]
        self.qsl = torch.tensor(
            [0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device="cuda"
        )
        self.seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
        pos = torch.cat(
            [torch.arange(L - q, L) for q, L in zip(q_lens, seq_lens)]
        ).cuda()
        self.positions = pos
        T = self.T

        def rnd(*shape, scale=1.0, dtype=torch.float16):
            return (torch.randn(*shape, device="cuda", generator=g) * scale).to(dtype)

        self.q = rnd(T, NH * D)
        self.rel = rnd(T, NH * D_REL, scale=2.0)
        self.sink = rnd(T, NH) + 4.0
        self.gate = rnd(T, NH * D, scale=1.5)
        self.q_scale = 0.7 + 0.6 * torch.rand(NH, D, device="cuda", generator=g)
        self.rel_scale = 0.7 + 0.6 * torch.rand(NH, D_REL, device="cuda", generator=g)
        self.proj = rnd(D_REL, REL_EXTENT, scale=0.1, dtype=torch.float32)
        self.proj_t = self.proj.t().contiguous().half()
        self.out = torch.empty(T, NH * D, dtype=torch.float16, device="cuda")
        self.seq_idx = torch.cat(
            [torch.full((q,), r, dtype=torch.int32) for r, q in enumerate(q_lens)]
        ).cuda()

        def slots(table, bs):
            out = []
            for r, (q, L) in enumerate(zip(q_lens, seq_lens)):
                p = torch.arange(L - q, L, device="cuda")
                out.append(table[r, p // bs].long() * bs + p % bs)
            return torch.cat(out)

        self.kv_slot = slots(self.table, KV_BLOCK)
        self.md_kv = state_mod.Inkling2Metadata(
            block_table=self.table,
            slot_mapping=self.kv_slot,
            seq_lens=self.seq_lens_t,
            query_start_loc=self.qsl,
            seq_idx=self.seq_idx,
            num_actual_tokens=T,
            num_reqs=self.R,
            max_query_len=max(q_lens),
            max_seq_len=max(seq_lens),
        )
        self.kv_owner = types.SimpleNamespace(kv_cache=self.kv, head_dim=D)
        if not conv:
            return
        # conv world (block size 4)
        cb = [cdiv(L, CONV_BLOCK) for L in seq_lens]
        conv_pages = sum(cb) + 1
        self.conv = engine_cache(
            conv_pages, NKV, CONV_BLOCK, DPAD, torch.float32, "rand"
        )
        cperm = torch.randperm(conv_pages - 1, device="cuda", generator=g) + 1
        self.conv_table = torch.zeros(self.R, max(cb), dtype=torch.int32, device="cuda")
        i = 0
        for r in range(self.R):
            self.conv_table[r, : cb[r]] = cperm[i : i + cb[r]].int()
            i += cb[r]
        self.conv_slot = slots(self.conv_table, CONV_BLOCK)
        self.md_conv = state_mod.Inkling2Metadata(
            block_table=self.conv_table,
            slot_mapping=self.conv_slot,
            seq_lens=self.seq_lens_t,
            query_start_loc=self.qsl,
            seq_idx=self.seq_idx,
            num_actual_tokens=T,
            num_reqs=self.R,
            max_query_len=max(q_lens),
            max_seq_len=max(seq_lens),
        )
        self.conv_owner = types.SimpleNamespace(
            kv_cache=self.conv,
            kernel_size=4,
            num_kv_heads=NKV,
            stream_ranges=STREAMS,
        )
        self.xk = rnd(T, NKV * D, scale=2.0)
        self.xv = rnd(T, NKV * D, scale=2.0)
        self.x_post = rnd(T, 2304, scale=2.0, dtype=torch.float32)
        self.conv_w = rnd(2304, 1, 4, scale=0.5, dtype=torch.float32)
        self.kv_w = rnd(NKV * D, 1, 4, scale=0.5, dtype=torch.float32)
        self.dyn_static_w = rnd(4, NKV * D, scale=0.5, dtype=torch.float32)
        self.dyn_w = rnd(T, NKV * 4, scale=0.4)
        self.hidden = rnd(T, 2304)
        self.post_out = torch.empty(T, 2304, dtype=torch.float32, device="cuda")

    def me(self):
        return types.SimpleNamespace(
            num_heads=NH,
            num_kv_heads=NKV,
            groups=NH // NKV,
            window=self.window or None,
            rel_extent=REL_EXTENT,
            scaling=D**-0.5,
            is_sliding=self.sliding,
            config=CFG,
        )


# ---------------------------------------------------------------------------
# kernel calls
# ---------------------------------------------------------------------------
def kernel_attention(w: World, splits=None):
    """Graph-capturable call of ``inkling2_attention``; ``call.splits`` holds
    the split factors (the shipped policy unless given)."""
    max_q = max(w.q_lens)
    if splits is None:
        splits = ops.inkling2_attention_splits(
            w.R, NKV, w.window, max(w.seq_lens), max_q
        )
    ws = torch.empty(
        ops.inkling2_attention_workspace_numel(w.R, w.T, NH, *splits),
        dtype=torch.float32,
        device="cuda",
    )

    def call():
        ops.inkling2_attention(
            w.q, w.rel, w.sink, w.gate, w.q_scale, w.rel_scale, w.proj_t,
            w.positions, w.kv, w.table, w.seq_lens_t, w.qsl, w.out, ws,
            w.window, REL_EXTENT, not w.sliding, 0.1, 8192.0, Q_CLIP, REL_CLIP,
            max_q, splits[0], splits[1],
        )  # fmt: skip

    call.splits = splits
    return call


def kernel_kv_prep(w: World, dynamic: bool):
    wk = w.dyn_static_w if dynamic else w.kv_w
    dyn = w.dyn_w if dynamic else None

    def call():
        ops.inkling2_kv_prep(
            w.xk, w.xv, wk, wk, dyn, dyn, w.conv, w.conv_table, w.conv_slot,
            w.seq_idx, w.qsl, w.seq_lens_t, w.kv, w.kv_slot, 0, D, Q_CLIP, 1e-6,
        )  # fmt: skip

    return call


def kernel_sconv(w: World):
    def call():
        ops.inkling2_sconv_static(
            w.x_post, w.conv_w, w.post_out, w.conv, w.conv_table, w.conv_slot,
            w.seq_idx, w.qsl, w.seq_lens_t, STREAMS[2][0], NKV,
        )  # fmt: skip

    return call


# ---------------------------------------------------------------------------
# torch baseline (the repo's eager path)
# ---------------------------------------------------------------------------
def torch_attention(w: World):
    me = w.me()

    def call():
        q = attn_mod.Inkling2Attention._scaled_clip(
            me, w.q.view(w.T, NH, D), w.q_scale, Q_CLIP, w.positions
        )
        rel = attn_mod.Inkling2Attention._scaled_clip(
            me, w.rel.view(w.T, NH, D_REL), w.rel_scale, REL_CLIP, w.positions
        )
        rel_logits = torch.einsum("thd,dr->thr", rel.float(), w.proj)
        attn = attn_mod.Inkling2Attention._attend(
            me, q, rel_logits, w.sink.float(), w.positions, w.kv_owner, w.md_kv
        )
        gate = torch.sigmoid(w.gate.float()).view(w.T, NH, D)
        return (attn.float() * gate).to(torch.float16).reshape(w.T, NH * D)

    return call


def torch_kv_prep(w: World, dynamic: bool):
    k_mod = (
        attn_mod.DynamicSconv(NKV * D, 2304, NKV, 4).cuda()
        if dynamic
        else attn_mod.StaticSconv(NKV * D, 4).cuda()
    )
    if dynamic:
        k_mod.static_weight.data.copy_(w.dyn_static_w)
        k_mod.weight_proj.weight = torch.nn.Parameter(
            torch.randn(NKV * 4, 2304, device="cuda", dtype=torch.float16) * 0.01,
            requires_grad=False,
        )
    else:
        k_mod.conv1d.weight.data.copy_(w.kv_w)

    def stream(x, which):
        taps = state_mod.conv_taps(
            w.conv_owner, which, x.float(), w.positions, w.md_conv
        )
        y = k_mod.forward_taps(taps, w.hidden) if dynamic else k_mod.forward_taps(taps)
        return y.view(-1, NKV, D)

    def call():
        k = blocks.l2_clip(stream(w.xk, state_mod.STREAM_K), Q_CLIP)
        v = stream(w.xv, state_mod.STREAM_V)
        v = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)
        state_mod.kv_write(w.kv_owner, k, v, w.md_kv)

    return call


def torch_sconv(w: World):
    mod = attn_mod.StaticSconv(2304, 4).cuda()
    mod.conv1d.weight.data.copy_(w.conv_w)

    def call():
        taps = state_mod.conv_taps(
            w.conv_owner, state_mod.STREAM_ATTN, w.x_post, w.positions, w.md_conv
        )
        return mod.forward_taps(taps)

    return call


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def time_graph(call, reps: int, rounds: int = 5) -> float:
    """Microseconds per call, CUDA-graph replay of ``reps`` launches.

    ``call`` may be a list of calls: one launch of each is one repetition and
    the result is per call (distinct caches, so the working set can exceed
    the L3)."""
    calls = call if isinstance(call, list) else [call]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            for c in calls:
                c()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(reps):
            for c in calls:
                c()
    graph.replay()
    torch.accelerator.synchronize()
    best = float("inf")
    for _ in range(rounds):
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        graph.replay()
        b.record()
        torch.accelerator.synchronize()
        best = min(best, a.elapsed_time(b) * 1e3 / (reps * len(calls)))
    return best


def time_eager(call, iters: int) -> float:
    for _ in range(2):
        call()
    torch.accelerator.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        call()
    b.record()
    torch.accelerator.synchronize()
    return a.elapsed_time(b) * 1e3 / iters


def visible_keys(L, q, sliding):
    """Total (query, key) pairs visited by a request."""
    pairs = 0
    p0 = L - q
    for i in range(q):
        pq = p0 + i
        pairs += min(pq + 1, WINDOW) if sliding else pq + 1
    return pairs


def attention_bytes(w: World):
    """DRAM bytes of the K|V rows a decode launch must read."""
    total = 0
    for _q, L in zip(w.q_lens, w.seq_lens):
        keys = min(L, WINDOW + 15) if w.sliding else L  # window + super-tile slack
        total += keys * NKV * 2 * D * 2
    return total


L3_BYTES = 128 << 20  # V620 Infinity Cache


def cold_us(w: World, warm_us: float, reps: int) -> float:
    """Per-call time with the L3 flushed by the traffic of other calls: the
    launches cycle over enough distinct caches (as the 48 layer visits of the
    model do) that a call's K|V rows are no longer cache-resident.  A sliding
    layer only reads its window, so its worlds use a context of 2 * window.
    Equals the warm time when one call alone streams more than the L3."""
    n = min(48, max(1, math.ceil(2 * L3_BYTES / attention_bytes(w))))
    if n == 1:
        return warm_us
    seq_lens = [min(L, 2 * WINDOW) for L in w.seq_lens] if w.sliding else w.seq_lens
    worlds = [World(w.q_lens, seq_lens, w.sliding, i, conv=False) for i in range(n)]
    return time_graph([kernel_attention(x) for x in worlds], max(reps // n, 2))


def run_case(name, w, torch_iters, reps, cold=False):
    row = {"case": name}
    try:
        call = kernel_attention(w)
        row["splits"] = "{}/{}".format(*call.splits)
        k_us = time_graph(call, reps)
    except Exception as e:  # noqa: BLE001
        k_us = float("nan")
        row["kernel_error"] = str(e)[:200]
    row["kernel_us"] = k_us
    if max(w.q_lens) <= 16:  # one query tile: every K|V row is read once
        row["kernel_GBps"] = attention_bytes(w) / (k_us * 1e3)
        if cold:
            row["cold_us"] = cold_us(w, k_us, reps)
            row["cold_GBps"] = attention_bytes(w) / (row["cold_us"] * 1e3)
    if w.T > 8:
        pairs = sum(visible_keys(L, q, w.sliding) for q, L in zip(w.q_lens, w.seq_lens))
        row["kernel_TFLOPs"] = 4.0 * pairs * D * NH / (k_us * 1e6)
    if torch_iters:
        try:
            row["torch_us"] = time_eager(torch_attention(w), torch_iters)
            row["speedup"] = row["torch_us"] / k_us
        except Exception as e:  # noqa: BLE001
            row["torch_error"] = str(e)[:200]
    return row


def _cell(row, key, fmt):
    value = row.get(key)
    if value is None or math.isnan(value):
        return "-"
    return fmt.format(value)


def print_markdown(rows):
    print(
        "\n| case | splits | kernel us | GB/s | cold us | cold GB/s | TFLOP/s "
        "| torch us | speedup |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        cells = [
            row.get("splits", "-"),
            _cell(row, "kernel_us", "{:.1f}"),
            _cell(row, "kernel_GBps", "{:.0f}"),
            _cell(row, "cold_us", "{:.1f}"),
            _cell(row, "cold_GBps", "{:.0f}"),
            _cell(row, "kernel_TFLOPs", "{:.2f}"),
            _cell(row, "torch_us", "{:.0f}"),
            _cell(row, "speedup", "{:.0f}x"),
        ]
        print(f"| {row['case']} | " + " | ".join(cells) + " |")


SWEEP_DECODE = (1, 2, 4, 8, 16, 32, 64, 128, 256)  # splits_decode
SWEEP_TILE = (1, 2, 4, 8, 16)  # splits_tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, choices=[1, 2, 4], default=1)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024, 16384, 131072])
    ap.add_argument("--prefill", type=int, default=1024)
    ap.add_argument(
        "--verify-q",
        type=int,
        nargs="*",
        default=[],
        help="also time speculative-verify shapes: q query tokens per request",
    )
    ap.add_argument(
        "--cold",
        action="store_true",
        help="also time decode / verify with the L3 flushed (distinct caches)",
    )
    ap.add_argument(
        "--chunk-ctx",
        type=int,
        nargs="*",
        default=[],
        help="also time a --prefill-token chunk of a long prefill: the chunk is "
        "the last tokens of a request of this total length (the torch path is "
        "not run)",
    )
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--torch-iters", type=int, default=3, help="0 skips the torch path")
    ap.add_argument("--sweep-splits", action="store_true")
    ap.add_argument("--only", choices=["attn", "kv", "sconv"], nargs="*")
    ap.add_argument("--md", action="store_true", help="print markdown tables")
    ap.add_argument("--json", type=str, default="")
    ap.add_argument(
        "--layout",
        choices=["bhc", "hbc"],
        default="bhc",
        help="physical page layout: engine block-outermost [B,bs,H,C] or "
        "head-major [B,H,bs,C]",
    )
    args = ap.parse_args()
    global NH, NKV, LAYOUT
    NH, NKV = 24 // args.tp, 8 // args.tp
    LAYOUT = args.layout
    only = set(args.only or ["attn", "kv", "sconv"])
    rows = []

    if "attn" in only:
        for sliding in (True, False):
            kind = "sliding" if sliding else "full"
            shapes = [(q, B, c) for q in [1, *args.verify_q] for B in args.batch
                      for c in args.ctx]  # fmt: skip
            for q, B, ctx in shapes:
                tag = "decode" if q == 1 else f"verify q={q}"
                w = World([q] * B, [ctx] * B, sliding, conv=False)
                name = f"attn {tag} {kind} B={B} ctx={ctx}"
                if args.sweep_splits:
                    for s in SWEEP_DECODE if q == 1 else SWEEP_TILE:
                        splits = (s, 1) if q == 1 else (1, s)
                        us = time_graph(kernel_attention(w, splits), args.reps)
                        rows.append(
                            dict(case=f"{name} {'S' if q == 1 else 'Sb'}={s}",
                                 kernel_us=us,
                                 kernel_GBps=attention_bytes(w) / (us * 1e3))
                        )  # fmt: skip
                        print(rows[-1], flush=True)
                else:
                    rows.append(
                        run_case(name, w, args.torch_iters, args.reps, args.cold)
                    )
                    print(rows[-1], flush=True)
                del w
                torch.accelerator.empty_cache()
            if args.prefill:
                q = args.prefill
                w = World([q], [q], sliding, conv=False)
                if args.sweep_splits:
                    for s in SWEEP_TILE:
                        us = time_graph(kernel_attention(w, (1, s)), args.reps)
                        print(
                            dict(
                                case=f"attn prefill {kind} q={q} Sb={s}", kernel_us=us
                            ),
                            flush=True,
                        )
                else:
                    rows.append(
                        run_case(
                            f"attn prefill {kind} q={q}", w, args.torch_iters, args.reps
                        )
                    )
                    print(rows[-1], flush=True)
                del w
                torch.accelerator.empty_cache()
            q = args.prefill
            for ctx in sorted(
                {min(c, 2 * WINDOW) if sliding else c for c in args.chunk_ctx}
            ):
                w = World([q], [ctx], sliding, conv=False)
                rows.append(run_case(f"attn chunk {kind} q={q} ctx={ctx}", w, 0, 5))
                print(rows[-1], flush=True)
                del w
                torch.accelerator.empty_cache()

    if "kv" in only:
        for dynamic in (False, True):
            kind = "dynamic(full)" if dynamic else "static(sliding)"
            for B, q, ctx in [
                (1, 1, 1024),
                (4, 1, 1024),
                (1, args.prefill, args.prefill),
            ]:
                w = World([q] * B, [ctx] * B, not dynamic)
                row = {"case": f"kv_prep {kind} B={B} q={q}"}
                row["kernel_us"] = time_graph(kernel_kv_prep(w, dynamic), args.reps)
                if args.torch_iters:
                    row["torch_us"] = time_eager(
                        torch_kv_prep(w, dynamic), args.torch_iters
                    )
                    row["speedup"] = row["torch_us"] / row["kernel_us"]
                rows.append(row)
                print(row, flush=True)

    if "sconv" in only:
        for B, q, ctx in [(1, 1, 1024), (4, 1, 1024), (1, args.prefill, args.prefill)]:
            w = World([q] * B, [ctx] * B, True)
            row = {"case": f"sconv_static post-conv C=2304 B={B} q={q}"}
            row["kernel_us"] = time_graph(kernel_sconv(w), args.reps)
            if args.torch_iters:
                row["torch_us"] = time_eager(torch_sconv(w), args.torch_iters)
                row["speedup"] = row["torch_us"] / row["kernel_us"]
            rows.append(row)
            print(row, flush=True)

    if args.md:
        print_markdown(rows)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
