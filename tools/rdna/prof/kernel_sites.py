# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute GPU glue kernels in a torch profiler trace to their launch sites.

usage: kernel_sites.py TRACE.json[.gz] [--marker REGEX] [--top N] [--depth N]
                       [--steps all|decode|mixed] [--json OUT]

A trace comes from vLLM's torch profiler with ``torch_profiler_with_stack``
(``serve_profile.sh PROFILER=torch``). Boot with CG_MODE=NONE so the model runs
eagerly and every kernel has a Python launch stack.

Steps are the intervals between consecutive ``--marker`` kernels (one per
engine step, ``get_num_sampled_and_rejected`` by default). Steps are split into
decode-like and mixed (prefill-carrying) by GPU busy time, and per-step counts
are reported for the chosen split. Each GPU event (kernel, memcpy, memset) is
tied to its launch thread and time (through the HIP launch event of the same
correlation id, else the CPU op's external id), then to the innermost vllm/
Python frames and the CPU ops enclosing that moment.
"""

import argparse
import bisect
import gzip
import json
import statistics
from collections import Counter, defaultdict

import regex as re

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")

# First match wins; the label is what the per-step table is keyed on.
KIND_RULES = (
    ("direct_copy", re.compile(r"direct_copy_kernel_cuda")),
    ("fill", re.compile(r"FillFunctor")),
    ("binary", re.compile(r"BinaryFunctor")),
    ("cast_copy", re.compile(r"_copy_kernel_cuda")),
    ("rocclr_copy", re.compile(r"__amd_rocclr_copyBuffer")),
    ("rocclr_fill", re.compile(r"__amd_rocclr_fillBuffer")),
    (
        "index",
        re.compile(
            r"index_elementwise_kernel|indexSelect|index_put|index_copy|"
            r"gather_kernel|_scatter_gather_elementwise_kernel"
        ),
    ),
    ("cat", re.compile(r"CatArrayBatchedCopy")),
    ("sigmoid", re.compile(r"sigmoid_kernel_cuda")),
    ("silu", re.compile(r"silu_kernel")),
    ("add", re.compile(r"CUDAFunctor(?:OnSelf)?_add|AddFunctor")),
)
GENERIC = re.compile(r"at::native::(?:\(anonymous namespace\)::)?(\w+)")
SKIP_GENERIC = {
    "elementwise_kernel",
    "elementwise_kernel_manual_unroll",
    "vectorized_elementwise_kernel",
    "unrolled_elementwise_kernel",
    "gpu_kernel_impl",
    "gpu_kernel_impl_nocast",
}
DTYPE = re.compile(r"\{lambda\(([\w:: ]+?)(?:, [\w:: ]+)*\)#1\}")
FILL_DTYPE = re.compile(r"FillFunctor<([\w:]+)>")
BIN_OP = re.compile(r"BinaryFunctor<.*?binary_internal::(\w+?)Functor")
FRAME = re.compile(r"^(.*)\((\d+)\): (.*)$")
INDUCTOR = re.compile(r"/(?:torchinductor[^/]*|inductor)/\w\w/(c\w+\.py)")


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)["traceEvents"]


def kind_of(name):
    """(kind, detail) for a GPU event name; detail adds dtype / functor."""
    for kind, rx in KIND_RULES:
        if rx.search(name):
            if kind == "fill":
                m = FILL_DTYPE.search(name)
                return kind, kind + (
                    f"({m.group(1).replace('c10::', '')})" if m else ""
                )
            if kind == "binary":
                m = BIN_OP.search(name)
                return kind, kind + (f"({m.group(1)})" if m else "")
            if kind == "direct_copy":
                m = DTYPE.search(name)
                return kind, kind + (
                    f"({m.group(1).replace('c10::', '')})" if m else ""
                )
            return kind, kind
    if name.startswith("Memcpy") or name.startswith("Memset"):
        return "memcpy", name.split(" (")[0]
    for m in GENERIC.finditer(name):
        if m.group(1) not in SKIP_GENERIC:
            return "other_native", m.group(1)
    return "other", re.sub(r"\(.*$", "", name)[:60]


def frame_label(raw):
    """(class, label) of a python_function name: repo / inductor / other."""
    m = FRAME.match(raw)
    if not m:
        return "other", raw[:80]
    path, line, fn = m.groups()
    if "site-packages" not in path and re.search(r"(?:^|/)vllm/", path):
        rel = re.sub(r"^.*?(?:^|/)(vllm/)", r"\1", path, count=1)
        return "repo", f"{rel}:{line} {fn}"
    mi = INDUCTOR.search(path)
    if mi:
        return "inductor", f"inductor/{mi.group(1)}:{line} {fn}"
    tail = path.split("site-packages/")[-1] if "site-packages/" in path else path
    return "other", f"{tail}:{line} {fn}"


class Intervals:
    """Per-thread nested intervals; stack sweep answers "what encloses t"."""

    def __init__(self):
        self.by_tid = defaultdict(list)

    def add(self, tid, ts, end, payload):
        self.by_tid[tid].append((ts, end, payload))

    def sweep(self, queries):
        """queries: {tid: [(t, key)]} -> {key: [payload outermost..innermost]}."""
        out = {}
        for tid, qs in queries.items():
            frames = sorted(self.by_tid.get(tid, ()), key=lambda f: (f[0], -f[1]))
            qs = sorted(qs, key=lambda q: q[0])
            stack = []
            fi = 0
            for t, key in qs:
                while fi < len(frames) and frames[fi][0] <= t:
                    while stack and stack[-1][1] <= frames[fi][0]:
                        stack.pop()
                    stack.append(frames[fi])
                    fi += 1
                while stack and stack[-1][1] <= t:
                    stack.pop()
                out[key] = [s[2] for s in stack]
        return out


def split_steps(durs, ratio):
    """Boolean mixed flag per step from GPU busy time (decode = near the low end)."""
    if not durs:
        return []
    base = statistics.median(sorted(durs)[: max(1, len(durs) // 4)])
    return [d > base * ratio for d in durs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--marker", default="get_num_sampled_and_rejected")
    ap.add_argument(
        "--match",
        default=r"at::native|CatArrayBatchedCopy|__amd_rocclr|Memcpy|Memset|Functor",
        help="GPU event names counted as glue",
    )
    ap.add_argument("--top", type=int, default=45)
    ap.add_argument("--depth", type=int, default=3, help="repo frames per site")
    ap.add_argument("--steps", default="decode", choices=("all", "decode", "mixed"))
    ap.add_argument(
        "--mixed-ratio",
        type=float,
        default=1.5,
        help="a step is mixed when its GPU busy time exceeds this x the "
        "decode baseline (median of the quickest quarter)",
    )
    ap.add_argument("--json", default=None, help="write the aggregates here")
    ap.add_argument("--chain", type=int, default=0, help="print N full chains")
    a = ap.parse_args()

    ev = load(a.trace)
    cats = Counter(e.get("cat") for e in ev)
    print("event categories:", dict(cats.most_common(12)))

    ext_op = {}
    runtime = {}
    cpu_ops = Intervals()
    frames = Intervals()
    gpu = []
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        args = e.get("args") or {}
        if cat == "cpu_op":
            cpu_ops.add(e["tid"], e["ts"], e["ts"] + e["dur"], e["name"])
            ext = args.get("External id")
            if ext is not None:
                ext_op[ext] = (e["tid"], e["ts"], e["name"])
        elif cat == "python_function":
            frames.add(e["tid"], e["ts"], e["ts"] + e["dur"], e["name"])
        elif cat in GPU_CATS:
            gpu.append(e)
        elif cat.endswith("_runtime") or cat == "cuda_driver":
            corr = args.get("correlation")
            if corr is not None:
                runtime[corr] = (e["tid"], e["ts"])
    del ev

    gpu.sort(key=lambda e: e["ts"])
    markers = [e for e in gpu if re.search(a.marker, e["name"])]
    print(
        f"gpu events {len(gpu)}  markers {len(markers)}  runtime events "
        f"{len(runtime)}  cpu_ops {sum(len(v) for v in cpu_ops.by_tid.values())}  "
        f"python frames {sum(len(v) for v in frames.by_tid.values())}"
    )
    if len(markers) < 2:
        raise SystemExit("need at least two marker kernels to delimit a step")
    bounds = [m["ts"] + m["dur"] for m in markers]
    nsteps = len(markers) - 1

    # Launch point (tid, t) of every glue event.
    rx = re.compile(a.match)
    rows = []
    queries = defaultdict(list)
    via = Counter()
    for i, e in enumerate(gpu):
        end = e["ts"] + e["dur"]
        if end <= bounds[0] or end > bounds[-1]:
            continue
        step = bisect.bisect_left(bounds, end) - 1
        rows.append((i, step, e))
    step_dur = defaultdict(float)
    step_n = Counter()
    for _, step, e in rows:
        step_dur[step] += e["dur"]
        step_n[step] += 1
    durs = [step_dur[s] for s in range(nsteps)]
    mixed = split_steps(durs, a.mixed_ratio)
    want = {
        "all": lambda s: True,
        "decode": lambda s: not mixed[s],
        "mixed": lambda s: mixed[s],
    }[a.steps]
    sel = [s for s in range(nsteps) if want(s)]
    print(
        f"steps {nsteps}: decode-like {sum(1 for m in mixed if not m)}, "
        f"mixed {sum(mixed)}; selected ({a.steps}) {len(sel)}"
    )
    if durs:
        ds = sorted(durs)
        print(
            f"step GPU busy ms: min {ds[0] / 1e3:.1f} median "
            f"{statistics.median(ds) / 1e3:.1f} p90 {ds[int(len(ds) * 0.9)] / 1e3:.1f} "
            f"max {ds[-1] / 1e3:.1f}; kernels/step median "
            f"{statistics.median(step_n[s] for s in range(nsteps)):.0f}"
        )
    selset = set(sel)
    if not sel:
        raise SystemExit("no steps selected")

    glue = []
    for i, step, e in rows:
        if step not in selset or not rx.search(e["name"]):
            continue
        args = e.get("args") or {}
        corr = args.get("correlation")
        ext = args.get("External id")
        if corr in runtime:
            tid, t = runtime[corr]
            via["correlation"] += 1
        elif ext in ext_op:
            tid, t, _ = ext_op[ext]
            via["external id"] += 1
        else:
            via["unresolved"] += 1
            glue.append((e, None))
            continue
        key = len(glue)
        queries[tid].append((t, key))
        glue.append((e, key))
    print("launch resolution:", dict(via))
    cpu_at = cpu_ops.sweep(queries)
    py_at = frames.sweep(queries)

    agg = Counter()
    dur = Counter()
    kinds = Counter()
    kdur = Counter()
    scopes = Counter()
    first_chain = {}
    first_ops = {}
    for e, key in glue:
        kind, det = kind_of(e["name"])
        kinds[(kind, det)] += 1
        kdur[(kind, det)] += e["dur"]
        if key is None:
            k0 = (kind, det, "(unresolved launch)", "", "")
            agg[k0] += 1
            dur[k0] += e["dur"]
            continue
        ops = cpu_at.get(key, [])
        py = py_at.get(key, [])
        top = ops[0] if ops else "(no cpu op)"
        leaf = ops[-1] if ops else ""
        repo = []
        ind = None
        for raw in reversed(py):
            cls, lab = frame_label(raw)
            if cls == "repo":
                repo.append(lab)
            elif cls == "inductor" and ind is None and not repo:
                ind = lab
        site = " <- ".join(repo[: a.depth]) if repo else "(no vllm frame)"
        if ind:
            site = f"[{ind}] " + site
        k1 = (kind, det, top, leaf if leaf != top else "", site)
        agg[k1] += 1
        dur[k1] += e["dur"]
        first_chain.setdefault(k1, [frame_label(r)[1] for r in py[-12:]])
        first_ops.setdefault(k1, list(ops))
        # in-model vs out-of-model (what a FULL graph replays vs host work)
        scope = (
            "model"
            if any(
                re.search(
                    r"vllm/(models|model_executor|compilation)/", frame_label(r)[1]
                )
                for r in py
            )
            else "runner/other"
        )
        scopes[(kind, scope)] += 1

    n = len(sel)
    total = sum(kinds.values())
    print(
        f"\nper-step glue kernels, {a.steps} steps (n={n}): {total / n:.1f} kernels, "
        f"{sum(kdur.values()) / n:.0f} us GPU time per step"
    )
    by_kind = Counter()
    by_kind_dur = Counter()
    for (kind, _), v in kinds.items():
        by_kind[kind] += v
    for (kind, _), v in kdur.items():
        by_kind_dur[kind] += v
    for kind, v in by_kind.most_common():
        sc = {s: scopes[(kind, s)] / n for s in ("model", "runner/other")}
        print(
            f"  {v / n:7.1f}  {by_kind_dur[kind] / n:8.0f} us  {kind:14s} "
            f"model {sc['model']:6.1f}  runner/other {sc['runner/other']:6.1f}"
        )
    print("  by dtype/functor (count/step, us/step):")
    for (kind, det), v in kinds.most_common(24):
        print(f"    {v / n:7.1f}  {kdur[(kind, det)] / n:8.0f} us  {det}")
    print(
        f"\nper-step by launch site (top {a.top}): count, GPU us, kernel, cpu op, site"
    )
    for k1, v in agg.most_common(a.top):
        kind, det, top, leaf, site = k1
        op = top + (f" > {leaf}" if leaf else "")
        print(f"  {v / n:6.1f} {dur[k1] / n:7.0f}  {det:22s} {op:34s} {site}")
    roll = Counter()
    roll_dur = Counter()
    for k1, v in agg.items():
        roll[k1[4]] += v
        roll_dur[k1[4]] += dur[k1]
    print(
        f"\nper-step rollup by innermost vllm frame chain (top {a.top}): count, GPU us"
    )
    for site, v in roll.most_common(a.top):
        print(f"  {v / n:6.1f} {roll_dur[site] / n:7.0f}  {site}")
    if a.chain:
        print("\nfull chains (innermost last) for the top sites:")
        for k, v in agg.most_common(a.chain):
            print(f"  {v / n:.1f}  {k[1]} {k[2]}")
            for fr in first_chain.get(k, []):
                print(f"      {fr}")
            print("      cpu ops (outer > inner): " + " > ".join(first_ops.get(k, [])))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(
                {
                    "steps": n,
                    "kinds": {
                        f"{k}|{d}": {"per_step": v / n, "us_per_step": kdur[(k, d)] / n}
                        for (k, d), v in kinds.items()
                    },
                    "sites": [
                        {
                            "per_step": v / n,
                            "us_per_step": dur[k] / n,
                            "kind": k[0],
                            "detail": k[1],
                            "op": k[2],
                            "leaf": k[3],
                            "site": k[4],
                        }
                        for k, v in agg.most_common()
                    ],
                },
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
