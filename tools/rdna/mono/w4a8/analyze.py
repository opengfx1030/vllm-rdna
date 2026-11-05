#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Arm checks and the result tables of the mono + W4A8 comparison.

analyze.py check-arm --arm-dir D --mono 1 --w4a8 1 --hwq 4 [--tp 4]
    The logs of one arm must show the arm that was asked for: the mono boot
    line (or none), the W4A8 prefill part and the sdot4 MoE warning (or
    none), the K2 dispatch lines of the decode graphs, and the HW queue
    budget the workers were spawned with. Exit 1 on any mismatch.

analyze.py report --run RUN --tree TREE --arms "c a b d c2 a2"
    Reads RUN/arms/<arm>/ (serve_validate files + arm/ from arm_driver.py)
    and writes RUN/results/{tables.md,results.json}: arm table, perf cells
    with deltas against production and the boot-to-boot noise of the
    repeated arms, probes, teacher-forced KL / dNLL / top-1 / PPL pairs
    (``topk_kl.py`` statistics) and the decode logprob parity pairs.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import regex as re

# nats/token a candidate may sit above the boot-to-boot floor of its own
# configuration: the measured W4A8 vs W4A16 KL increment is 0.0019 - 0.0028.
KL_MARGIN = 0.003
# a delta below this (percent) is never called a difference
MIN_NOISE_PCT = 1.0

GROUPS = (
    ("prod", "production (a, a2)", ("a", "a2")),
    ("prod_q4", "production, 4 queues (e)", ("e",)),
    ("prod_ple", "production, recipe PLE (a0)", ("a0",)),
    ("mono", "mono, W4A16 (b)", ("b",)),
    ("monow", "mono + W4A8 (c, c2)", ("c", "c2")),
    ("monow2", "mono + W4A8, 2 queues (d)", ("d",)),
)

# row label, cell, key, format, True when a larger value is better
METRICS = (
    ("TTFT 4k solo, median (s)", "ttft4k", "ttft_median_s", "{:.2f}", False),
    ("prefill 4k (tok/s/req)", "ttft4k", "prefill_tok_s_req", "{:.0f}", True),
    ("TTFT 16k solo, median (s)", "ttft16k", "ttft_median_s", "{:.2f}", False),
    ("prefill 16k (tok/s/req)", "ttft16k", "prefill_tok_s_req", "{:.0f}", True),
    ("16k c8 aggregate prefill (tok/s)", "agg16k", "prefill_tok_s_agg", "{:.0f}", True),
    ("16k c8 mean TTFT (s)", "agg16k", "ttft_mean_s", "{:.2f}", False),
    ("16k/1k c8 decode (tok/s/req)", "agg16k", "decode_tok_s_req", "{:.1f}", True),
    ("c1 ITL median (ms)", "c1", "itl_median_ms", "{:.1f}", False),
    ("c1 decode (tok/s/req)", "c1", "decode_tok_s_req", "{:.1f}", True),
    ("c6 ITL median (ms)", "c6", "itl_median_ms", "{:.1f}", False),
    ("c6 decode (tok/s/req)", "c6", "decode_tok_s_req", "{:.1f}", True),
    ("c8 ITL median (ms)", "c8", "itl_median_ms", "{:.1f}", False),
    ("c8 decode (tok/s/req)", "c8", "decode_tok_s_req", "{:.1f}", True),
    ("c8 output (tok/s total)", "c8", "out_tok_s", "{:.0f}", True),
    ("c1 accepted length", "c1", "accept_len", "{:.2f}", True),
    ("c8 accepted length", "c8", "accept_len", "{:.2f}", True),
)


def count(path: Path, pattern: str) -> int:
    if not path.is_file():
        return 0
    rx = re.compile(pattern)
    with open(path, errors="replace") as f:
        return sum(1 for line in f if rx.search(line))


def first_match(path: Path, pattern: str) -> re.Match | None:
    if not path.is_file():
        return None
    rx = re.compile(pattern)
    with open(path, errors="replace") as f:
        for line in f:
            if m := rx.search(line):
                return m
    return None


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# ---- check-arm ---------------------------------------------------------------


def check_arm(arm_dir: Path, mono: int, w4a8: int, hwq: int, tp: int) -> list[tuple]:
    """(name, ok, detail) per check."""
    log = arm_dir / "serve.log"
    out: list[tuple] = []

    def add(name: str, ok: bool, detail: str) -> None:
        out.append((name, ok, detail))

    n_mono = count(log, r"mono rdna2: --monokernel on TP rank")
    n_part = count(log, r"W4A8 MoE prefill from \d+ tokens")
    n_warn = count(log, r"RDNA2 W4A8 sdot4 MoE path active")
    add(
        "mono boot lines",
        n_mono >= tp if mono else n_mono == 0,
        f"{n_mono} (want {'>= ' + str(tp) if mono else 0})",
    )
    want_part = bool(mono and w4a8)
    add(
        "W4A8 part in the mono boot line",
        n_part >= tp if want_part else n_part == 0,
        f"{n_part} (want {'>= ' + str(tp) if want_part else 0})",
    )
    add(
        "sdot4 MoE path active (W4A8 prefill ran)",
        n_warn >= tp if w4a8 else n_warn == 0,
        f"{n_warn} (want {'>= ' + str(tp) if w4a8 else 0})",
    )
    if mono:
        for rows in (3, 6):
            n = count(log, rf"mono k2_rdna2: {rows} rows \(full\) -> mono kernel")
            add(f"K2 serves the {rows}-row FULL graph", n >= 1, f"{n} line(s)")
        off = count(log, r"mono k[12]_rdna2 disabled for")
        add("no K1/K2 'disabled for' warning", off == 0, f"{off} line(s)")
    raised = count(log, r"mono rdna2: GPU_MAX_HW_QUEUES \S+ -> \d+")
    want_raise = bool(mono and hwq == 4)
    add(
        "mono raised the queue budget",
        raised >= 1 if want_raise else raised == 0,
        f"{raised} line(s) (want {'>= 1' if want_raise else 0})",
    )
    summary = load_json(arm_dir / "arm" / "summary.json") or {}
    workers = summary.get("worker_env", {})
    good = [
        pid
        for pid, e in workers.items()
        if e.get("GPU_MAX_HW_QUEUES") == str(hwq)
        and e.get("VLLM_RDNA2_W4A8_SDOT4") == str(w4a8)
    ]
    add(
        "workers spawned with the arm's knobs",
        len(good) >= tp,
        f"{len(good)}/{len(workers)} KFD processes have GPU_MAX_HW_QUEUES={hwq} "
        f"VLLM_RDNA2_W4A8_SDOT4={w4a8}",
    )
    return out


def cmd_check_arm(args) -> int:
    checks = check_arm(args.arm_dir, args.mono, args.w4a8, args.hwq, args.tp)
    for name, ok, detail in checks:
        print(f"{'OK      ' if ok else 'MISMATCH'} {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


# ---- report ------------------------------------------------------------------


def load_arm(run: Path, arm: str) -> dict:
    d = run / "arms" / arm
    cells: dict[str, dict] = {}
    invalid: list[str] = []
    for line in (
        (d / "arm" / "cells.jsonl").read_text().splitlines()
        if (d / "arm" / "cells.jsonl").is_file()
        else []
    ):
        rec = json.loads(line)
        if rec["status"] == "valid":
            cells[rec["cell"]] = rec
        else:
            invalid.append(f"{rec['cell']} try {rec['attempt']}: {rec['status']}")
    summ = d / "summary.txt"
    log = d / "serve.log"

    def verdict(label: str) -> str:
        if first_match(summ, rf"{label} PASS"):
            return "PASS"
        return "FAIL" if first_match(summ, rf"{label} FAIL") else "n/a"

    ppl = first_match(d / "ppl.txt", r"PPL ([\d.]+) \(mean NLL")
    prefix = first_match(d / "prefix.txt", r"cold TTFT ([\d.]+) s, warm TTFT ([\d.]+)")
    boot = first_match(summ, r"READY in (\d+)s")
    load = first_match(log, r"Model loading took ([\d.]+) GiB")
    kv = first_match(log, r"GPU KV cache size: ([\d,]+) tokens")
    arm_summary = load_json(d / "arm" / "summary.json") or {}
    return {
        "arm": arm,
        "dir": str(d),
        "done": (d / "DONE").exists(),
        "cells": cells,
        "invalid": invalid,
        "summary": arm_summary,
        "probe": verdict("PROBE"),
        "ppl_probe": verdict("PPL PROBE"),
        "ppl": float(ppl.group(1)) if ppl else None,
        "prefix": verdict("PREFIX"),
        "prefix_ttft": (
            [float(prefix.group(1)), float(prefix.group(2))] if prefix else None
        ),
        "conc": verdict("CONCURRENCY"),
        "after": verdict("PROBE-AFTER-LOAD"),
        "boot_s": int(boot.group(1)) if boot else None,
        "load_gib": float(load.group(1)) if load else None,
        "kv_tokens": kv.group(1) if kv else None,
        "n_mono": count(log, r"mono rdna2: --monokernel on TP rank"),
        "n_w4a8_part": count(log, r"W4A8 MoE prefill from \d+ tokens"),
        "n_w4a8_active": count(log, r"RDNA2 W4A8 sdot4 MoE path active"),
        "n_k2_mono": count(log, r"mono k2_rdna2: .*\(full\) -> mono kernel"),
        "hwq_raise": count(log, r"mono rdna2: GPU_MAX_HW_QUEUES"),
        "tracebacks": count(log, r"Traceback"),
        "check": (d / "check.txt").read_text() if (d / "check.txt").is_file() else "",
        "arm_txt": (d / "arm.txt").read_text() if (d / "arm.txt").is_file() else "",
    }


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def spread_pct(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2 or not mean(xs):
        return None
    return (max(xs) - min(xs)) / mean(xs) * 100.0


def value(arm: dict, cell: str, key: str):
    rec = arm["cells"].get(cell)
    return None if rec is None else rec["metrics"].get(key)


def fmt(x, f: str) -> str:
    return "n/a" if x is None else f.format(x)


def pct(x: float | None, ref: float | None) -> float | None:
    if x is None or not ref:
        return None
    return (x / ref - 1.0) * 100.0


def perf_tables(arms: dict[str, dict]) -> tuple[str, dict]:
    order = list(arms)
    lines = [
        "| metric | " + " | ".join(order) + " |",
        "|---|" + "---:|" * len(order),
    ]
    for label, cell, key, f, _ in METRICS:
        row = [fmt(value(arms[a], cell, key), f) for a in order]
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    per_arm = "\n".join(lines)

    groups = [
        (gid, name, [a for a in members if a in arms]) for gid, name, members in GROUPS
    ]
    groups = [g for g in groups if g[2]]
    by_id = {gid: members for gid, _, members in groups}
    data: dict = {}
    head = ["metric"] + [name for _, name, _ in groups]
    cmp_cols = [
        ("monow", "prod", "mono+W4A8 vs prod"),
        ("mono", "prod", "mono vs prod"),
        ("monow2", "prod", "mono+W4A8 2q vs prod"),
        ("monow", "mono", "W4A8 inside mono"),
        ("monow", "monow2", "4q vs 2q (W4A8)"),
    ]
    cmp_cols = [c for c in cmp_cols if c[0] in by_id and c[1] in by_id]
    head += [name for _, _, name in cmp_cols] + ["noise prod %", "noise mono+W4A8 %"]
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for label, cell, key, f, higher in METRICS:
        means = {
            gid: mean([value(arms[a], cell, key) for a in members])
            for gid, _, members in groups
        }
        cells = [fmt(means[gid], f) for gid, _, _ in groups]
        noise_p = spread_pct([value(arms[a], cell, key) for a in by_id.get("prod", [])])
        noise_m = spread_pct(
            [value(arms[a], cell, key) for a in by_id.get("monow", [])]
        )
        floor = max(MIN_NOISE_PCT, noise_p or 0.0, noise_m or 0.0)
        row_data = {
            "means": means,
            "noise_prod_pct": noise_p,
            "noise_monow_pct": noise_m,
        }
        for new, ref, name in cmp_cols:
            d = pct(means[new], means[ref])
            row_data[name] = d
            if d is None:
                cells.append("n/a")
                continue
            beyond = abs(d) > floor
            better = (d > 0) == higher
            mark = ("+" if better else "-") if beyond else "~"
            cells.append(f"{d:+.1f}% {mark}")
        cells += [
            "n/a" if noise_p is None else f"{noise_p:.1f}",
            "n/a" if noise_m is None else f"{noise_m:.1f}",
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
        data[label] = row_data
    grouped = "\n".join(lines)
    note = (
        "`+` better than the reference beyond the noise, `-` worse beyond it, "
        f"`~` within the noise (the larger of {MIN_NOISE_PCT:.0f} % and the "
        "boot-to-boot spread of the repeated arms, a/a2 and c/c2). Each cell is "
        "one boot per arm; a spread needs the repeats."
    )
    return f"{per_arm}\n\n{grouped}\n\n{note}\n", data


def arm_table(arms: dict[str, dict]) -> str:
    head = (
        "| arm | boot s | model GiB | KV tokens | mono lines | W4A8 part | "
        "sdot4 active | K2 full lines | queue raise | workers' queues | "
        "eviction (whole hook) | invalid cells | status |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|\n"
    )
    rows = []
    for a, d in arms.items():
        s = d["summary"]
        queues = sorted(
            {
                f"{e.get('GPU_MAX_HW_QUEUES')}"
                for e in s.get("worker_env", {}).values()
                if e.get("GPU_MAX_HW_QUEUES")
            }
        )
        ev = s.get("evict_whole_hook", {})
        evict = (
            "n/a" if not ev else ("ok" if ev.get("ok") else "; ".join(ev["reasons"]))
        )
        rows.append(
            f"| {a} | {fmt(d['boot_s'], '{}')} | {fmt(d['load_gib'], '{:.2f}')} | "
            f"{d['kv_tokens'] or 'n/a'} | {d['n_mono']} | {d['n_w4a8_part']} | "
            f"{d['n_w4a8_active']} | {d['n_k2_mono']} | {d['hwq_raise']} | "
            f"{'/'.join(queues) or 'n/a'} | {evict} | "
            f"{'; '.join(d['invalid']) or 'none'} | "
            f"{'DONE' if d['done'] else 'NOT DONE'} |"
        )
    return head + "\n".join(rows) + "\n"


def probe_table(arms: dict[str, dict]) -> str:
    head = (
        "| arm | greedy probe | PPL probe (261 tok) | prefix | concurrency | "
        "probe after load | prefix cold/warm TTFT (s) | tracebacks |\n"
        "|---|---|---|---|---|---|---|---:|\n"
    )
    rows = []
    for a, d in arms.items():
        ttft = d["prefix_ttft"]
        rows.append(
            f"| {a} | {d['probe']} | {d['ppl_probe']} "
            f"({fmt(d['ppl'], '{:.2f}')}) | {d['prefix']} | {d['conc']} | "
            f"{d['after']} | "
            f"{'n/a' if not ttft else f'{ttft[0]:.2f} / {ttft[1]:.2f}'} | "
            f"{d['tracebacks']} |"
        )
    return head + "\n".join(rows) + "\n"


def load_topk_module(tree: Path):
    path = tree / "tools" / "rdna" / "port_v031" / "topk_kl.py"
    spec = importlib.util.spec_from_file_location("topk_kl", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["topk_kl"] = mod
    spec.loader.exec_module(mod)
    return mod


def topk_pair(tkl, ref_path: Path, cand_path: Path, reps: int = 2000) -> dict:
    """Per-domain pair statistics, on the windows both runs scored."""
    ma, da = tkl.load(str(ref_path))
    mb, db = tkl.load(str(cand_path))
    if ma["windows_sha256"] != mb["windows_sha256"] or ma["top"] != mb["top"]:
        raise ValueError("different windows or top-k")
    res = {}
    for name in ma["domains"]:
        ref = {f: da[f"{name}.{f}"] for f in (*tkl.FIELDS, "act_id")}
        cand = {f: db[f"{name}.{f}"] for f in (*tkl.FIELDS, "act_id")}
        n = min(ref["act_lp"].shape[0], cand["act_lp"].shape[0])
        ref = {k: v[:n] for k, v in ref.items()}
        cand = {k: v[:n] for k, v in cand.items()}
        res[name] = tkl.domain_pair(ref, cand, reps, 3)
    return res


def quality_pairs(run: Path, arms: dict[str, dict]) -> list[tuple[str, Path, Path]]:
    def f(arm: str, name: str = "topk.npz") -> Path:
        return run / "arms" / arm / name

    pairs = [
        ("a2 vs a (boot floor, W4A8)", f("a"), f("a2")),
        ("c2 vs c (boot floor, mono+W4A8)", f("c"), f("c2")),
        ("c vs a (headline)", f("a"), f("c")),
        ("c2 vs a2", f("a2"), f("c2")),
        ("b vs a", f("a"), f("b")),
        ("c vs b (W4A8 inside mono)", f("b"), f("c")),
        ("d vs a", f("a"), f("d")),
        ("e vs a", f("a"), f("e")),
        ("a0 vs a", f("a0"), f("a")),
    ]
    for arm in arms:
        pairs.append((f"{arm} repeat (same boot)", f(arm), f(arm, "topk_rep.npz")))
    return [p for p in pairs if p[1].is_file() and p[2].is_file()]


def quality_tables(tkl, run: Path, arms: dict[str, dict]) -> tuple[str, dict]:
    head = (
        "| pair (ref, cand) | domain | windows | PPL ref | PPL cand | dNLL [95% CI] | "
        "KL nats/token [95% CI] | top-1 | identical tokens |\n"
        "|---|---|---:|---:|---:|---|---|---:|---:|\n"
    )
    rows = []
    data: dict = {}
    for label, ref, cand in quality_pairs(run, arms):
        try:
            res = topk_pair(tkl, ref, cand)
        except (ValueError, KeyError, OSError) as exc:
            rows.append(f"| {label} | error: {exc} | | | | | | | |")
            continue
        data[label] = res
        for dom, d in res.items():
            rows.append(
                f"| {label} | {dom} | {d['windows']} | {d['ppl_ref']:.4f} | "
                f"{d['ppl_cand']:.4f} | {d['dnll']:+.5f} "
                f"[{d['dnll_ci'][0]:+.5f}, {d['dnll_ci'][1]:+.5f}] | "
                f"{d['kl']:.5f} [{d['kl_ci'][0]:.5f}, {d['kl_ci'][1]:.5f}] | "
                f"{d['top1']:.4f} | {d['identical_positions']:.1%} |"
            )
    return head + "\n".join(rows) + "\n", data


def quality_verdict(data: dict) -> list[str]:
    """Mechanical reading of the headline pair against the boot-to-boot floors."""
    head = data.get("c vs a (headline)")
    if not head:
        return ["quality: no c vs a pair (an arm or its topk file is missing)"]
    floors = [data[k] for k in data if "boot floor" in k]
    out = []
    for dom, d in head.items():
        floor = max((p[dom]["kl"] for p in floors if dom in p), default=None)
        ci_zero = d["dnll_ci"][0] <= 0.0 <= d["dnll_ci"][1]
        if floor is None:
            out.append(f"{dom}: KL {d['kl']:.5f}, no boot floor (needs a2 and c2)")
            continue
        ok = d["kl"] <= floor + KL_MARGIN and ci_zero
        out.append(
            f"{dom}: KL(c vs a) {d['kl']:.5f} vs floor {floor:.5f} + {KL_MARGIN} -> "
            f"{'within noise' if d['kl'] <= floor + KL_MARGIN else 'ABOVE the floor'}; "
            f"dNLL {d['dnll']:+.5f} CI "
            f"{'contains' if ci_zero else 'EXCLUDES'} 0 -> "
            f"{'PASS' if ok else 'CHECK'}"
        )
    return out


def parity_tables(tree: Path, run: Path, arms: dict[str, dict]) -> str:
    script = tree / "tools" / "rdna" / "mono" / "logprob_parity.py"

    def f(arm: str) -> Path:
        return run / "arms" / arm / "parity.json"

    pairs = [
        ("a vs a2", "a", "a2"),
        ("c vs c2", "c", "c2"),
        ("a vs c", "a", "c"),
        ("a vs b", "a", "b"),
        ("b vs c", "b", "c"),
        ("a vs d", "a", "d"),
    ]
    rows = [
        "| pair | identical-prefix tokens | worst abs dlogprob on the shared prefix |",
        "|---|---:|---:|",
    ]
    for label, x, y in pairs:
        if not (f(x).is_file() and f(y).is_file()):
            continue
        res = subprocess.run(
            [sys.executable, str(script), "compare", str(f(x)), str(f(y))],
            capture_output=True,
            text=True,
            check=False,
        )
        (run / "results" / f"parity-{x}-{y}.txt").write_text(res.stdout + res.stderr)
        m = re.search(
            r"identical-prefix tokens (\d+)/(\d+), worst \|dlogprob\| on shared "
            r"prefix ([\d.]+)",
            res.stdout,
        )
        rows.append(
            f"| {label} | {m.group(1)}/{m.group(2)} | {m.group(3)} |"
            if m
            else f"| {label} | error | see parity-{x}-{y}.txt |"
        )
    return "\n".join(rows) + "\n" if len(rows) > 2 else "no parity pairs\n"


def cmd_report(args) -> int:
    run = args.run
    (run / "results").mkdir(exist_ok=True)
    names = [a for a in args.arms.split() if (run / "arms" / a).is_dir()]
    arms = {a: load_arm(run, a) for a in names}
    sha = (
        (args.tree / ".sha").read_text().strip()
        if (args.tree / ".sha").is_file()
        else "?"
    )
    prov = args.tree / ".so-provenance"
    perf_md, perf_data = perf_tables(arms)
    tkl = load_topk_module(args.tree)
    qual_md, qual_data = quality_tables(tkl, run, arms)
    md = [
        "# Mono + W4A8 comparison",
        "",
        f"- tree `{args.tree}` at `{sha}`",
        f"- `.so` provenance: {prov.read_text().strip() if prov.is_file() else 'n/a'}",
        "- arms: "
        + "; ".join(f"{a}: {describe(d['arm_txt'])}" for a, d in arms.items()),
        "",
        "## Arms",
        "",
        arm_table(arms),
        "## Performance (valid cells only)",
        "",
        perf_md,
        "## Probes",
        "",
        probe_table(arms),
        "## Teacher-forced windows (prefill path, W4A8 MoE when on)",
        "",
        qual_md,
        "Mechanical reading of `c vs a` against the boot-to-boot floors "
        f"(KL margin {KL_MARGIN}):",
        "",
        *[f"- {line}" for line in quality_verdict(qual_data)],
        "",
        "## Decode logprob parity (greedy, top-5, solo and c8)",
        "",
        parity_tables(args.tree, run, arms),
    ]
    (run / "results" / "tables.md").write_text("\n".join(md))
    (run / "results" / "results.json").write_text(
        json.dumps(
            {
                "sha": sha,
                "arms": {
                    a: {k: v for k, v in d.items() if k not in ("check", "arm_txt")}
                    for a, d in arms.items()
                },
                "perf": perf_data,
                "quality": qual_data,
            },
            indent=1,
            default=str,
        )
    )
    print(f"wrote {run / 'results' / 'tables.md'}")
    return 0


def describe(arm_txt: str) -> str:
    m = re.search(r"try=(\d+) desc=(.*)", arm_txt)
    return f"{m.group(2)} (try {m.group(1)})" if m else "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-arm")
    c.add_argument("--arm-dir", type=Path, required=True)
    c.add_argument("--mono", type=int, required=True)
    c.add_argument("--w4a8", type=int, required=True)
    c.add_argument("--hwq", type=int, required=True)
    c.add_argument("--tp", type=int, default=4)
    c.set_defaults(func=cmd_check_arm)
    r = sub.add_parser("report")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--tree", type=Path, required=True)
    r.add_argument("--arms", required=True)
    r.set_defaults(func=cmd_report)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
