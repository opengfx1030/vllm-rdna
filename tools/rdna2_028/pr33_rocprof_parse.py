#!/usr/bin/env python3
"""Window a rocprofv3 kernel trace and report FA-RDNA2 decode cost in-model.

    python pr33_rocprof_parse.py <label>=<trace.csv|dir> [more...]

A trace covers one serve process. Under rocprofv3 --run the launcher execs the
target, so each GPU worker writes its own `prof*_kernel_trace.csv`; pass the
campaign dir and the per-process files are merged. Decode FA dispatches are
clustered by GPU-time gaps; each cluster is one bench cell (or a warmup/capture
burst). Per cluster we report the FA decode total, the call count, the elapsed
GPU time of the whole window, the FA share, and FA microseconds per decoded
token.

gfx1030 in-model shape (Flash-Next TP4): H_q/H_kv = 6/1, so the per-head
`fa_decode_paged_splitk_kernel_256` runs for c<=3 and the GQA variant only for
`num_tokens * H_kv >= 4`. FA decode call arguments: Grid_X = num_tokens for
both kernels.
"""
import csv
import glob
import os
import sys

FA_DECODE = "fa_decode_paged_splitk"
FA_GQA = "fa_decode_paged_splitk_gqa"
GAP_NS = 1_000_000_000  # 1 s GPU-time gap splits bursts
STEPS = 512  # random-output-len of the measurement cells


def load(path):
    rows = []  # (start, end, kind, gx, gy, gz)
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            name = rec.get("Kernel_Name", "")
            try:
                st = int(rec["Start_Timestamp"])
                en = int(rec["End_Timestamp"])
            except (KeyError, ValueError):
                continue
            if FA_DECODE in name and FA_GQA not in name:
                kind, gx = "d256", int(rec["Grid_Size_X"])
            elif FA_GQA in name:
                kind, gx = "gqa", int(rec["Grid_Size_X"])
            else:
                kind, gx = "", 0
            rows.append((st, en, kind, gx,
                         int(rec.get("Grid_Size_Y", 0) or 0),
                         int(rec.get("Grid_Size_Z", 0) or 0)))
    return rows


def clusters(rows):
    fa = sorted((r for r in rows if r[2]), key=lambda r: r[0])
    out, cur = [], []
    for r in fa:
        if cur and r[0] - cur[-1][0] > GAP_NS:
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def file_bursts(path):
    rows = load(path)
    res = []
    for cl in clusters(rows):
        if len(cl) < 100:
            continue
        wstart = min(r[0] for r in cl)
        wend = max(r[1] for r in cl)
        win = [r for r in rows if wstart <= r[0] and r[1] <= wend]
        elapsed = max(r[1] for r in win) - min(r[0] for r in win)
        res.append({"kind": cl[0][2], "gx": cl[0][3],
                    "calls": len(cl), "fa_ns": sum(r[1] - r[0] for r in cl),
                    "elapsed_ns": elapsed})
    return res


def summarize(label, target):
    if os.path.isfile(target):
        files = [target]
    else:
        files = sorted(glob.glob(os.path.join(target, "prof*_kernel_trace.csv")))
    files = [f for f in files if "agent_info" not in f]
    if not files:
        print(f"\n=== {label} ===\n  no traces in {target}")
        return
    per_file = [b for b in (file_bursts(f) for f in files) if b]
    if not per_file:
        print(f"\n=== {label} ===\n  no FA decode bursts")
        return
    n = min(len(b) for b in per_file)
    print(f"\n=== {label} ===  files={len(files)} "
          f"bursts/file={[len(b) for b in per_file]}")
    for i in range(n):
        burst = [b[i] for b in per_file]
        gx = max(b["gx"] for b in burst)
        kind = burst[0]["kind"]
        calls = sum(b["calls"] for b in burst)
        fa_ns = sum(b["fa_ns"] for b in burst)
        elapsed = max(b["elapsed_ns"] for b in burst)
        tokens = STEPS * max(gx, 1)
        print(f"  burst {i}: kind={kind} grid_x={gx} "
              f"fa_calls(total)={calls} fa_ms(total)={fa_ns/1e6:.3f} "
              f"elapsed_s={elapsed/1e9:.3f} fa_share={100*fa_ns/elapsed:.2f}% "
              f"fa_us_per_tok={fa_ns/tokens/1e3:.3f} "
              f"step_ms~={elapsed/STEPS/1e6:.3f} (files={len(burst)})")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    for a in args:
        label, _, target = a.partition("=")
        summarize(label, target or label)


if __name__ == "__main__":
    main()
