# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute GPU kernels in a torch profiler trace to Python frames.

usage: kernel_sites.py TRACE.json[.gz] [--match REGEX] [--marker REGEX] [--top N]
Counts kernels whose name matches --match per engine step (step = one --marker
kernel) and groups them by the innermost vllm/ Python frame (+ parent) that
launched them.
"""

import argparse
import bisect
import gzip
import json
from collections import Counter, defaultdict

import regex as re


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)["traceEvents"]


def short(name):
    m = re.search(
        r"(direct_copy_kernel_cuda|FillFunctor|BinaryFunctor|_copy_kernel_cuda|"
        r"sigmoid_kernel_cuda|CatArrayBatchedCopy|index_elementwise_kernel|"
        r"CUDAFunctorOnSelf_add|CUDAFunctor_add|gpu_kernel_impl_nocast|"
        r"[A-Za-z]+Functor)",
        name,
    )
    return m.group(1) if m else re.sub(r"\(.*$", "", name)[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--match", default=r"at::native|CatArrayBatchedCopy")
    ap.add_argument("--marker", default="get_num_sampled_and_rejected")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--depth", type=int, default=3)
    a = ap.parse_args()
    ev = load(a.trace)
    ops = {}
    py = defaultdict(list)
    kernels = []
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat == "cpu_op":
            ext = e.get("args", {}).get("External id")
            if ext is not None:
                ops[ext] = e
        elif cat == "python_function":
            py[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"]))
        elif cat == "kernel":
            kernels.append(e)
    for v in py.values():
        v.sort()
    starts = {t: [x[0] for x in v] for t, v in py.items()}
    steps = sum(1 for k in kernels if re.search(a.marker, k["name"]))
    print(f"kernels {len(kernels)}  steps(marker) {steps}  cpu ops {len(ops)}")
    steps = max(steps, 1)
    rx = re.compile(a.match)
    agg = Counter()
    kinds = Counter()
    unresolved = Counter()
    for k in kernels:
        if not rx.search(k["name"]):
            continue
        kinds[short(k["name"])] += 1
        op = ops.get(k.get("args", {}).get("External id"))
        if op is None:
            unresolved[short(k["name"])] += 1
            continue
        frames = py.get(op["tid"], [])
        st = starts.get(op["tid"], [])
        i = bisect.bisect_right(st, op["ts"])
        enclosing = [
            f for f in frames[max(0, i - 4000) : i] if f[1] >= op["ts"] + op["dur"]
        ]
        enclosing.sort(key=lambda f: f[0])
        chain = [
            re.sub(r"^.*?/vllm/", "vllm/", f[2])
            for f in enclosing
            if "/vllm/" in f[2] and "torch/" not in f[2].split("/vllm/")[-1]
        ]
        site = " <- ".join(reversed(chain[-a.depth :])) if chain else "(no vllm frame)"
        agg[(short(k["name"]), op["name"], site)] += 1
    print("per-step kernel kinds:")
    for k, v in kinds.most_common(15):
        print(f"  {v / steps:7.1f}  {k}")
    if unresolved:
        print("unresolved (no cpu op):", dict(unresolved.most_common(5)))
    print("per-step by launch site:")
    for (kn, opn, site), v in agg.most_common(a.top):
        print(f"  {v / steps:6.1f}  {kn:24s} {opn:22s} {site}")


if __name__ == "__main__":
    main()
