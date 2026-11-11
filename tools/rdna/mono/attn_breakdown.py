#!/usr/bin/env python3
"""E0: attention-block breakdown per decode width from one rank's vLLM torch-
profiler trace (eager serve, e0_profile.sh), with the attention segments split
into QSA and GDN layers. Derived from monokernel-p0-probe/analyze_trace.py.

  python3 attn_breakdown.py TRACE.json.gz

A step is a CPU `execute_context_C(..)_generation_G(..)` annotation; decode-only
steps have C == 0. Each GPU kernel is tied to the CPU runtime call that issued
it through its correlation id (a replayed graph's kernels all carry the
hipGraphLaunch id), so async GPU lag does not blur step boundaries.

  python3 analyze_trace.py TRACE.json.gz [--dump-names] [--per-graph]
"""

import argparse
import bisect
import collections
import gzip
import json
import re
import statistics

# First match wins.
BUCKETS = [
    ("allreduce", r"rdna_ar|allreduce|all_reduce|cross_device_reduce|nccl|rccl|AllGather|allgather"),
    ("moe_routed", r"moe|expert|topk_softmax|topkGating|count_and_sort|align_block"),
    ("shared_expert", r"fused_se|shared_exp|_se_"),
    ("gdn_conv", r"gdn|gated_delta|fused_recurrent|recurrent|delta|causal_conv|conv1d|l2norm|ssm"),
    ("attn_qsa", r"fa_rdna2|flash|attn|attention|paged|qsa|indexer|rope|rotary|reshape_and_cache|kv_cache|compress|mqa"),
    ("gemm_gemv", r"gemv|gemm|Cijk|gptq|awq|wvSplit|rocblas|matmul|dot2|w4a16|marlin|exllama|hc_|hyper"),
    ("ple", r"ple|ngram"),
    ("norm_act_ew", r"rms|norm|silu|act_and_mul|gelu|sigmoid|softplus|elementwise|vectorized|triton_poi|triton_per|triton_red|fused_add|reduce_kernel"),
    ("sample_spec", r"argmax|sample|softmax|topk|top_k|gather|scatter|index|cumsum|sort|copy|memcpy|fill|zero|cat|rejection|spec|eagle|mtp"),
]
BUCKET_RE = [(n, re.compile(p, re.I)) for n, p in BUCKETS]
NAMES = [b for b, _ in BUCKETS] + ["other"]
STEP_RE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def bucket(name):
    for n, r in BUCKET_RE:
        if r.search(name):
            return n
    return "other"


AR_RE = re.compile(r"rdna_ar|ncclDevKernel|allreduce|all_reduce|cross_device_reduce", re.I)
ATTN_RE = re.compile(r"gated_delta|delta_rule|causal_conv|qsa|fa_rdna2|flash|paged_attention|mqa", re.I)
FFN_RE = re.compile(r"moe_gemm|fused_moe|topkGating|se_gate_up|se_down", re.I)
QSA_RE = re.compile(r"qsa|fa_rdna2|mqa|paged_attention|top_k_per_row", re.I)


def segment_report(steps, show_seq, classes=("attn_gdn", "attn_qsa")):
    """Split each step at its all-reduce kernels. A segment holding MoE /
    shared-expert kernels is the FFN block between the attention all-reduce and
    the FFN all-reduce (what a 2a FFN kernel replaces: HC combine+norm, router,
    top-k, align, routed experts, shared expert, HC mix); a segment holding
    GDN/QSA kernels is the attention block; the rest is embedding, lm_head,
    drafter glue and sampling."""
    per = []
    samples = {}
    for r in steps:
        segs, cur = [], []
        for k in r["ks"]:
            if AR_RE.search(k["name"]):
                segs.append(("seg", cur))
                segs.append(("ar", [k]))
                cur = []
            else:
                cur.append(k)
        segs.append(("seg", cur))
        acc = collections.defaultdict(lambda: [0, 0, 0.0])  # segments, kernels, us
        seq = []
        for kind, ks in segs:
            if not ks:
                continue
            names = " ".join(k["name"] for k in ks)
            if kind == "ar":
                cls = "allreduce"
            elif FFN_RE.search(names):
                cls = "ffn"
            elif QSA_RE.search(names):
                cls = "attn_qsa"
            elif ATTN_RE.search(names):
                cls = "attn_gdn"
            else:
                cls = "other"
            acc[cls][0] += 1
            acc[cls][1] += len(ks)
            acc[cls][2] += sum(k.get("dur", 0) for k in ks)
            seq.append((cls, len(ks), sum(k.get("dur", 0) for k in ks)))
            if kind != "ar":
                sig = (cls, len(ks))
                samples.setdefault(sig, []).append(ks)
        per.append((acc, seq))
    print("\n== per-step segments (split at all-reduce kernels), medians over steps")
    print(f"  {'class':10s} {'segments':>8s} {'kernels':>8s} {'busy_ms':>8s} {'%busy':>6s} "
          f"{'kern/seg':>8s} {'us/seg':>8s}")
    tot = statistics.median(sum(v[2] for v in acc.values()) for acc, _ in per)
    for cls in ("attn_gdn", "attn_qsa", "ffn", "allreduce", "other"):
        segs = statistics.median(acc[cls][0] for acc, _ in per)
        kern = statistics.median(acc[cls][1] for acc, _ in per)
        us = statistics.median(acc[cls][2] for acc, _ in per)
        print(f"  {cls:10s} {segs:8.0f} {kern:8.0f} {us/1e3:8.3f} {100*us/tot:6.1f} "
              f"{(kern/segs if segs else 0):8.1f} {(us/segs if segs else 0):8.1f}")
    # Kernel-level view of the most common FFN / attention segment shapes:
    # per-position median duration over every occurrence.
    common = sorted(samples.items(), key=lambda kv: -len(kv[1]))
    shown = set()
    for (cls, n), occ in common:
        if cls not in classes or any(c == cls for c, _ in shown):
            continue
        shown.add((cls, n))
        tot_us = statistics.median(sum(k.get("dur", 0) for k in ks) for ks in occ)
        print(f"\n== {cls} segment with {n} kernels: {len(occ)} occurrences, median {tot_us:.1f} us")
        for i in range(n):
            d = statistics.median(ks[i].get("dur", 0) for ks in occ)
            print(f"  {i:3d} {d:7.1f} us  [{bucket(occ[0][i]['name']):12s}] {occ[0][i]['name'][:110]}")
    if show_seq:
        _, seq = per[len(per) // 2]
        print("\n== segment sequence of the median step (class, kernels, us)")
        print(" ".join(f"{c[0]}{n}:{us:.0f}" for c, n, us in seq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--dump-names", action="store_true")
    ap.add_argument("--show-seq", action="store_true",
                    help="print the classified segment sequence of one step")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--classes", default="attn_gdn,attn_qsa",
                    help="segment classes to list kernel by kernel (also: ffn)")
    a = ap.parse_args()
    op = gzip.open if a.trace.endswith(".gz") else open
    with op(a.trace, "rt") as f:
        ev = json.load(f)["traceEvents"]

    kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    rt = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")]
    anns = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "user_annotation" and STEP_RE.match(e["name"])]
    by_corr = collections.defaultdict(list)
    for k in kern:
        by_corr[(k.get("args") or {}).get("correlation")].append(k)
    rt.sort(key=lambda e: e["ts"])
    rt_ts = [e["ts"] for e in rt]

    if a.dump_names:
        cnt, dur = collections.Counter(), collections.Counter()
        for k in kern:
            cnt[k["name"]] += 1
            dur[k["name"]] += k.get("dur", 0)
        tot = sum(dur.values())
        for n, d in dur.most_common(a.top):
            print(f"{d/1e3:9.2f} ms {100*d/tot:5.1f}% {cnt[n]:6d}x [{bucket(n):12s}] {n[:140]}")
        return

    rows = []
    for an in sorted(anns, key=lambda e: e["ts"]):
        m = STEP_RE.match(an["name"])
        ctx_reqs, ctx_toks, gen_reqs, gen_toks = map(int, m.groups())
        i0 = bisect.bisect_left(rt_ts, an["ts"])
        i1 = bisect.bisect_right(rt_ts, an["ts"] + an["dur"])
        ks, calls, graphs = [], collections.Counter(), []
        for r in rt[i0:i1]:
            c = (r.get("args") or {}).get("correlation")
            kk = by_corr.get(c, [])
            if kk:
                calls[r["name"]] += 1
                ks.extend(kk)
                if "GraphLaunch" in r["name"]:
                    graphs.append(len(kk))
        if not ks:
            continue
        t0 = min(k["ts"] for k in ks)
        t1 = max(k["ts"] + k.get("dur", 0) for k in ks)
        busy, n = collections.Counter(), collections.Counter()
        for k in ks:
            b = bucket(k["name"])
            busy[b] += k.get("dur", 0)
            n[b] += 1
        ks.sort(key=lambda k: k["ts"])
        rows.append(dict(ks=ks, name=an["name"], decode=ctx_reqs == 0, gen=gen_reqs, toks=gen_toks + ctx_toks,
                         cpu_ms=an["dur"] / 1e3, span_ms=(t1 - t0) / 1e3, nk=len(ks),
                         busy_ms=sum(busy.values()) / 1e3, busy=busy, n=n, calls=calls, graphs=graphs))

    print(f"{len(kern)} GPU ops, {len(anns)} step annotations, {len(rows)} steps with GPU work")
    print("step                                       cpu_ms span_ms kernels busy_ms graphs(kernels each)  launch calls")
    for r in rows:
        print(f"{r['name'][:42]:42s} {r['cpu_ms']:6.2f} {r['span_ms']:7.2f} {r['nk']:7d} {r['busy_ms']:7.2f} "
              f"{str(r['graphs'])[:22]:22s} {dict(r['calls'].most_common(4))}")
    dec = [r for r in rows if r["decode"]]
    by_w = collections.defaultdict(list)
    for r in dec:
        by_w[r["toks"]].append(r)
    for toks, g in sorted(by_w.items()):
        if len(g) < 3:
            continue
        print(f"\n######## decode width {toks} ({len(g)} steps)")
        segment_report(g, False, tuple(a.classes.split(",")))
    for label, sel in (("decode-only steps", dec),):
        if not sel:
            continue
        groups = collections.defaultdict(list)
        for r in sel:
            groups[r["toks"]].append(r)
        for toks, g in sorted(groups.items()):
            print(f"\n== {label}, {toks} tokens/step (n={len(g)}), medians")
            span = statistics.median(r["span_ms"] for r in g)
            busy = statistics.median(r["busy_ms"] for r in g)
            nk = statistics.median(r["nk"] for r in g)
            print(f"  GPU span {span:.3f} ms, GPU busy {busy:.3f} ms, kernels {nk:.0f}, "
                  f"graph launches {statistics.median(len(r['graphs']) for r in g):.0f}, "
                  f"kernels per graph {g[len(g)//2]['graphs']}")
            print(f"  {'bucket':12s} {'kernels':>8s} {'ms':>8s} {'%busy':>6s}")
            for b in NAMES:
                kb = statistics.median(r["n"][b] for r in g)
                mb = statistics.median(r["busy"][b] for r in g) / 1e3
                if kb or mb:
                    print(f"  {b:12s} {kb:8.0f} {mb:8.3f} {100*mb/busy:6.1f}")
    if dec:
        segment_report(dec, a.show_seq)


if __name__ == "__main__":
    main()
