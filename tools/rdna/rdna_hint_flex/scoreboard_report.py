#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Markdown table rows below are long f-strings.
# ruff: noqa: E501

"""Scoreboard report for HI2 serving arms (one JSON + one markdown per arm).

``scoreboard.sh`` boots one arm (a set of env switches / recipe overrides) and
leaves its raw results in one directory; this script turns that directory into
``scoreboard.json`` and ``scoreboard.md``:

1. PER REQUEST, per bench cell: TTFT, prefill tok/s (prompt tokens / TTFT),
   decode tok/s (output tokens after the first / time between first and last
   token) and ITL p50/p99 (pooled over the cell's streamed gaps). Needs the
   detailed bench JSON (``--save-detailed``, which ``scoreboard.sh`` sets); older
   JSONs fall back to the cell summary fields and are marked ``per_request: false``.
2. AGGREGATE: output tok/s, total tok/s, request/s and goodput (output tokens of
   the requests that met the SLO, per second of cell wall time; see SLO below).
3. QUALITY: short PPL, long-context PPL (``quality_ab.py``, 6 passages of about
   15k tokens), and, against a baseline arm of the SAME checkpoint,
   prompt-level top-k KL / dNLL / top-1 (``topk_kl.py``) and decode-path KL
   (``decode_kl.py``), next to the boot-to-boot noise floor (the baseline booted
   twice): a delta is only a finding when it exceeds the floor.
4. VALIDITY: greedy / PPL / prefix-cache / concurrency probes, KFD queue
   evictions per cell, amdgpu fault markers, junction temperature (rolling 5 s
   mean, 90 C limit), speculative acceptance when MTP is on, and the measurement
   window label (quiet protocol, or taken during an image build).

    python scoreboard_report.py arm DIR [--baseline DIR] [--floor DIR]
    python scoreboard_report.py table DIR_A DIR_B ...      # first is the reference

The quality tools are reused unchanged (``tools/rdna/port_v031``). SLO for
goodput: TTFT <= ``--slo-ttft-base-s`` + ``--slo-ttft-s-per-ktok`` x prompt
ktokens and per-request ITL p99 <= ``--slo-itl-p99-ms``; the number only means
something between arms judged with the same SLO. Stdlib only (the topk_kl
comparison needs numpy, so run with the serving venv's python).
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import regex as re

KIT = Path(__file__).resolve().parents[1] / "port_v031"
CELL_RE = re.compile(r"^bench-(\d+)x(\d+)-c(\d+)\.json$")
LIMIT_JUNCTION_C = 90.0


def pctl(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def spread(values: list[float]) -> dict:
    return {"mean": mean(values), "p50": pctl(values, 0.5), "p99": pctl(values, 0.99)}


def cell_stats(path: Path, args: argparse.Namespace) -> dict:
    d = json.loads(path.read_text())
    m = CELL_RE.match(path.name)
    cell = {
        "input_len": int(m.group(1)),
        "output_len": int(m.group(2)),
        "conc": int(m.group(3)),
        "completed": d.get("completed"),
        "failed": d.get("failed"),
        "aggregate": {
            "output_tok_s": d.get("output_throughput"),
            "total_tok_s": d.get("total_token_throughput"),
            "request_per_s": d.get("request_throughput"),
            "duration_s": d.get("duration"),
        },
    }
    ttfts, itls = d.get("ttfts"), d.get("itls")
    in_lens, out_lens = d.get("input_lens"), d.get("output_lens")
    if not (ttfts and itls is not None and in_lens and out_lens):
        cell["per_request"] = False
        cell["summary_fields"] = {
            k: d.get(k)
            for k in (
                "mean_ttft_ms",
                "median_ttft_ms",
                "p99_ttft_ms",
                "mean_tpot_ms",
                "median_itl_ms",
                "p99_itl_ms",
            )
        }
        return cell
    errors = d.get("errors") or [""] * len(ttfts)
    rows, pooled, good_tokens = [], [], 0
    for i, ttft in enumerate(ttfts):
        if errors[i] or not ttft or ttft <= 0:
            continue
        gaps = [g * 1000.0 for g in (itls[i] or [])]
        pooled.extend(gaps)
        n_in, n_out = in_lens[i], out_lens[i]
        decode_s = sum(itls[i] or [])
        row = {
            "input_len": n_in,
            "output_len": n_out,
            "ttft_ms": ttft * 1000.0,
            "prefill_tok_s": n_in / ttft,
            "decode_tok_s": (n_out - 1) / decode_s if decode_s > 0 else None,
            "itl_p50_ms": pctl(gaps, 0.5),
            "itl_p99_ms": pctl(gaps, 0.99),
        }
        rows.append(row)
        ttft_slo = args.slo_ttft_base_s + args.slo_ttft_s_per_ktok * n_in / 1000.0
        if ttft <= ttft_slo and (row["itl_p99_ms"] or 0.0) <= args.slo_itl_p99_ms:
            good_tokens += n_out
    cell["per_request"] = True
    cell["requests"] = rows
    cell["ttft_ms"] = spread([r["ttft_ms"] for r in rows])
    cell["prefill_tok_s"] = spread([r["prefill_tok_s"] for r in rows])
    cell["decode_tok_s"] = spread(
        [r["decode_tok_s"] for r in rows if r["decode_tok_s"] is not None]
    )
    cell["itl_ms"] = {"p50": pctl(pooled, 0.5), "p99": pctl(pooled, 0.99)}
    dur = d.get("duration") or 0.0
    cell["aggregate"]["goodput_tok_s"] = good_tokens / dur if dur else None
    cell["aggregate"]["good_requests_note"] = (
        f"SLO ttft <= {args.slo_ttft_base_s} s + {args.slo_ttft_s_per_ktok} s/ktok, "
        f"itl p99 <= {args.slo_itl_p99_ms} ms"
    )
    return cell


def thermal(path: Path) -> dict:
    series: dict[str, list[tuple[float, float, float, float]]] = {}
    for line in read_text(path).splitlines()[1:]:
        f = line.split()
        if len(f) < 6:
            continue
        try:
            series.setdefault(f[1], []).append(
                (
                    float(f[2]) / 1000.0,
                    float(f[3]) / 1000.0,
                    float(f[4]) / 1000.0,
                    float(f[5]),
                )
            )
        except ValueError:
            continue
    out: dict = {}
    for bus, rows in series.items():
        junction = [r[1] for r in rows]
        roll = [mean(junction[max(0, i - 4) : i + 1]) for i in range(len(junction))]
        out[bus] = {
            "samples": len(rows),
            "edge_max_c": max(r[0] for r in rows),
            "junction_max_c": max(junction),
            "junction_5s_mean_max_c": max(roll),
            "mem_max_c": max(r[2] for r in rows),
            "vram_max_mib": max(r[3] for r in rows),
        }
    out["limit_ok"] = (
        all(v["junction_5s_mean_max_c"] < LIMIT_JUNCTION_C for v in out.values())
        if out
        else None
    )
    return out


def last_match(text: str, pattern: str) -> str | None:
    hits = re.findall(pattern, text, flags=re.M)
    return hits[-1] if hits else None


def validity(d: Path) -> dict:
    summary = read_text(d / "summary.txt")
    ppl_txt = read_text(d / "ppl.txt")
    prefix = read_text(d / "prefix.txt")
    conc = read_text(d / "concurrency.txt")
    v: dict = {
        "ready_s": last_match(summary, r"READY in (\d+)s"),
        "greedy_probe": last_match(read_text(d / "probe.txt"), r"^(RESULT .*)$"),
        "greedy_probe_after_load": last_match(
            read_text(d / "probe-after.txt"), r"^(RESULT .*)$"
        ),
        "ppl_short": last_match(ppl_txt, r"PPL ([0-9.]+)"),
        "ppl_probe": last_match(ppl_txt, r"^(RESULT .*)$"),
        "prefix_probe": last_match(prefix, r"^(cold TTFT .*)$"),
        "prefix_hit": last_match(prefix, r"^(prefix cache: .*)$"),
        "concurrency": last_match(conc, r"(RESULT .*?)\(")
        or last_match(conc, r"^(.*identical.*)$"),
        "kfd_evictions": re.findall(r"evicted_ms.*", summary),
        "invalid_cells": [ln for ln in summary.splitlines() if "INVALID" in ln],
        "fault_markers": last_match(summary, r"fault markers in serve.log: (\S+)"),
        "dmesg_faults": last_match(summary, r"new amdgpu dmesg faults: (\S+)"),
        "load_hook": last_match(summary, r"LOAD HOOK rc=(.*)"),
    }
    v["kfd_ok"] = (
        all("-> ok" in e for e in v["kfd_evictions"]) and not v["invalid_cells"]
    )
    spec = read_text(d / "quality" / "spec.txt")
    if spec:
        v["spec_accept"] = [ln for ln in spec.splitlines() if ln.strip()][-8:]
    v["thermal"] = thermal(d / "thermal.csv")
    return v


def window_label(meta: dict) -> str:
    start = str(meta.get("build_running_start")) == "1"
    end = str(meta.get("build_running_end")) == "1"
    quiet = str(meta.get("quiet_hold")) == "1"
    if quiet and not end:
        return "quiet protocol (QUIET_HOLD=1, no image build at the end)"
    if start or end:
        return "taken during image build, not for comparison"
    return "no image build seen, no quiet protocol (QUIET_HOLD=0): not for comparison"


def qab_pair(a: dict, b: dict) -> dict:
    deltas, exact, firsts, big = [], 0, [], 0
    dn, dn_n = 0.0, 0
    for pa, pb in zip(a["passages"], b["passages"]):
        if pa["tokens"] != pb["tokens"]:
            continue
        for x, y in zip(pa["token_logprobs"], pb["token_logprobs"]):
            if x is not None and y is not None:
                deltas.append(abs(x - y))
                big += abs(x - y) > 0.1
                dn += x - y
                dn_n += 1
        ga, gb = pa["greedy_tokens"], pb["greedy_tokens"]
        exact += ga == gb
        firsts.append(
            next(
                (i for i, (x, y) in enumerate(zip(ga, gb)) if x != y),
                min(len(ga), len(gb)),
            )
        )
    return {
        "ppl_ref": a["ppl"],
        "ppl_cand": b["ppl"],
        "ppl_rel": b["ppl"] / a["ppl"] - 1.0,
        "dnll": dn / dn_n if dn_n else None,
        "mean_abs_dlogp": mean(deltas),
        "frac_abs_dlogp_gt_0p1": big / len(deltas) if deltas else None,
        "greedy_exact": f"{exact}/{len(a['passages'])}",
        "mean_first_divergence": mean(firsts),
    }


def run_tool(py: str, tool: str, argv: list[str]) -> tuple[int, str]:
    p = subprocess.run([py, str(KIT / tool), *argv], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def quality(d: Path, base: Path | None, floor: Path | None, py: str) -> dict:
    q: dict = {"ppl_short": None, "ppl_long": None}
    m = last_match(read_text(d / "ppl.txt"), r"PPL ([0-9.]+)")
    q["ppl_short"] = float(m) if m else None
    qd = d / "quality"
    arm_qab = load_json(qd / "qab.json")
    if arm_qab:
        q["ppl_long"] = arm_qab["ppl"]
        q["long_prompt_tokens"] = [p["prompt_tokens"] for p in arm_qab["passages"]]
    if base is None:
        return q
    cmp: dict = {"baseline": str(base), "floor": str(floor) if floor else None}
    bq = base / "quality"
    fq = floor / "quality" if floor else None
    base_qab = load_json(bq / "qab.json")
    if base_qab and arm_qab:
        cmp["long_ppl"] = {"arm": qab_pair(base_qab, arm_qab)}
        floor_qab = load_json(fq / "qab.json") if fq else None
        if floor_qab:
            cmp["long_ppl"]["floor"] = qab_pair(base_qab, floor_qab)
    out = d / "quality"
    if (bq / "topk.npz").exists() and (qd / "topk.npz").exists():
        argv = ["compare", "--pair", f"arm={bq / 'topk.npz'},{qd / 'topk.npz'}"]
        if fq and (fq / "topk.npz").exists():
            argv += ["--pair", f"floor={bq / 'topk.npz'},{fq / 'topk.npz'}"]
        argv += [
            "--json",
            str(out / "topk_compare.json"),
            "--md",
            str(out / "topk_compare.md"),
        ]
        rc, text = run_tool(py, "topk_kl.py", argv)
        j = load_json(out / "topk_compare.json")
        cmp["prompt_kl"] = {
            label: res["domains"] for label, res in (j or {}).get("pairs", {}).items()
        } or {"error": text[-400:], "rc": rc}
    if (bq / "decode.json").exists() and (qd / "decode.json").exists():
        argv = ["compare", str(bq / "decode.json"), str(qd / "decode.json")]
        if fq and (fq / "decode.json").exists():
            argv += ["--floor", str(fq / "decode.json")]
        argv += ["--json", str(out / "decode_compare.json")]
        rc, text = run_tool(py, "decode_kl.py", argv)
        cmp["decode_kl"] = load_json(out / "decode_compare.json") or {
            "error": text[-400:],
            "rc": rc,
        }
    q["vs_baseline"] = cmp
    return q


def fmt(x, nd=2, dash="-"):
    if x is None:
        return dash
    if isinstance(x, float) and abs(x) < 10 ** (-nd) and x != 0:
        return f"{x:.2e}"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def judge(arm, floor, tol=1.25):
    if arm is None or floor is None:
        return ""
    return "within floor" if abs(arm) <= abs(floor) * tol else "ABOVE floor"


def markdown(r: dict) -> str:
    L = [f"# Scoreboard: arm `{r['arm']}`", ""]
    meta = r["meta"]
    L += [
        f"**Window: {r['window']}.**",
        "",
        f"- recipe `{meta.get('recipe')}`, overrides `{meta.get('overrides')}`, "
        f"GPUs {meta.get('gpus')} (PCI {meta.get('buses')}), tree sha `{meta.get('tree_sha')}`",
        f"- started {meta.get('start')}, build running at start/end: "
        f"{meta.get('build_running_start')}/{meta.get('build_running_end')}, "
        f"QUIET_HOLD={meta.get('quiet_hold')}",
        "",
        "## 1-2. Performance (per request, then aggregate)",
        "",
        "| cell | n | TTFT mean / p99 (s) | prefill tok/s mean | decode tok/s/req mean "
        "| ITL p50 / p99 (ms) | output tok/s | total tok/s | goodput tok/s |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for name, c in r["cells"].items():
        a = c["aggregate"]
        if c.get("per_request"):
            t, p, dd, itl = (
                c["ttft_ms"],
                c["prefill_tok_s"],
                c["decode_tok_s"],
                c["itl_ms"],
            )
            L.append(
                f"| {name} | {c['completed']} | {fmt(t['mean'] / 1000)} / "
                f"{fmt(t['p99'] / 1000)} | {fmt(p['mean'], 0)} | {fmt(dd['mean'], 1)} | "
                f"{fmt(itl['p50'], 1)} / {fmt(itl['p99'], 1)} | {fmt(a['output_tok_s'], 1)} "
                f"| {fmt(a['total_tok_s'], 0)} | {fmt(a.get('goodput_tok_s'), 1)} |"
            )
        else:
            s = c["summary_fields"]
            L.append(
                f"| {name} | {c['completed']} | {fmt((s['mean_ttft_ms'] or 0) / 1000)} / "
                f"{fmt((s['p99_ttft_ms'] or 0) / 1000)} | "
                f"{fmt(c['input_len'] / (s['mean_ttft_ms'] / 1000), 0) if s['mean_ttft_ms'] else '-'} | "
                f"{fmt(1000 / s['mean_tpot_ms'], 1) if s['mean_tpot_ms'] else '-'} | "
                f"{fmt(s['median_itl_ms'], 1)} / {fmt(s['p99_itl_ms'], 1)} | "
                f"{fmt(a['output_tok_s'], 1)} | {fmt(a['total_tok_s'], 0)} | - |"
            )
    q = r["quality"]
    L += [
        "",
        "## 3. Quality",
        "",
        f"- short PPL {fmt(q.get('ppl_short'), 3)}; long-context PPL "
        f"{fmt(q.get('ppl_long'), 4)} over passages of {q.get('long_prompt_tokens')} tokens",
    ]
    vb = q.get("vs_baseline")
    if vb:
        L += [
            f"- baseline `{vb['baseline']}`, floor (baseline booted again) `{vb['floor']}`",
            "",
        ]
        lp = vb.get("long_ppl")
        if lp:
            L += [
                "| long-context (quality_ab) | PPL ref -> cand | dPPL | dNLL | mean abs dlogp "
                "| greedy identical | mean first divergence |",
                "|---|---|---:|---:|---:|---|---:|",
            ]
            for lab in ("arm", "floor"):
                s = lp.get(lab)
                if s:
                    L.append(
                        f"| {lab} | {fmt(s['ppl_ref'], 4)} -> {fmt(s['ppl_cand'], 4)} | "
                        f"{s['ppl_rel']:+.3%} | {fmt(s['dnll'], 5)} | {fmt(s['mean_abs_dlogp'], 5)} "
                        f"| {s['greedy_exact']} | {fmt(s['mean_first_divergence'], 1)} |"
                    )
            if lp.get("floor"):
                L.append(
                    f"| judged | | | | {judge(lp['arm']['mean_abs_dlogp'], lp['floor']['mean_abs_dlogp'])} | | |"
                )
            L.append("")
        pk = vb.get("prompt_kl")
        if pk and "error" not in pk:
            L += [
                "| prompt KL (topk_kl) | domain | KL nats/token [95% CI] | dNLL [95% CI] "
                "| top-1 [95% CI] | abs dlogp p99 |",
                "|---|---|---|---|---|---:|",
            ]
            for lab, doms in pk.items():
                for dom, s in doms.items():
                    L.append(
                        f"| {lab} | {dom} | {s['kl']:.5f} [{s['kl_ci'][0]:.5f}, {s['kl_ci'][1]:.5f}]"
                        f" | {s['dnll']:+.5f} [{s['dnll_ci'][0]:+.5f}, {s['dnll_ci'][1]:+.5f}]"
                        f" | {s['top1']:.4f} [{s['top1_ci'][0]:.4f}, {s['top1_ci'][1]:.4f}]"
                        f" | {s['absdlp_p99']:.4f} |"
                    )
            if "floor" in pk and "arm" in pk:
                for dom in pk["arm"]:
                    L.append(
                        f"| judged | {dom} | KL {judge(pk['arm'][dom]['kl'], pk['floor'][dom]['kl'])}"
                        f" | dNLL {judge(pk['arm'][dom]['dnll'], pk['floor'][dom]['dnll'])} | | |"
                    )
            L.append("")
        dk = vb.get("decode_kl")
        if dk and "error" not in dk:
            L += [
                "| decode KL (decode_kl) | KL mean | KL p99 | divergence median (tokens) "
                "| identical | confident flips | positions |",
                "|---|---:|---:|---:|---|---:|---:|",
            ]
            for lab, key in (("arm", "ref-cand"), ("floor", "ref-floor")):
                s = dk.get(key)
                if s:
                    L.append(
                        f"| {lab} | {s['kl_mean']:.3e} | {s['kl_p99']:.3e} | "
                        f"{s['divergence_median']} | {s['identical']}/{s['passages']} | "
                        f"{s['confident_flips']} | {s['positions']} |"
                    )
            if "ref-floor" in dk:
                L.append(
                    f"| judged | {judge(dk['ref-cand']['kl_mean'], dk['ref-floor']['kl_mean'])} "
                    "| | | | | |"
                )
            L.append("")
    v = r["validity"]
    th = v.get("thermal", {})
    L += [
        "## 4. Validity",
        "",
        f"- READY after {v.get('ready_s')} s; greedy probe: {v.get('greedy_probe')}; after load: "
        f"{v.get('greedy_probe_after_load')}",
        f"- PPL probe: {v.get('ppl_short')} {v.get('ppl_probe')}",
        f"- prefix cache: {v.get('prefix_probe')}; {v.get('prefix_hit')}",
        f"- concurrency: {v.get('concurrency')}",
        f"- KFD evictions: {'all zero' if v.get('kfd_ok') else 'NON-ZERO or INVALID cell'}"
        f" ({len(v.get('kfd_evictions', []))} windows); fault markers {v.get('fault_markers')}, "
        f"dmesg faults {v.get('dmesg_faults')}",
    ]
    for bus, s in th.items():
        if isinstance(s, dict):
            L.append(
                f"- bus {bus}: junction max {s['junction_max_c']:.0f} C, worst 5 s mean "
                f"{s['junction_5s_mean_max_c']:.1f} C (limit {LIMIT_JUNCTION_C:.0f}), "
                f"VRAM max {s['vram_max_mib']:.0f} MiB"
            )
    if v.get("spec_accept"):
        L += ["- speculative acceptance:", "", "```"] + v["spec_accept"] + ["```"]
    return "\n".join(L) + "\n"


def build_arm(args: argparse.Namespace) -> int:
    d = Path(args.dir)
    meta = load_json(d / "meta.json") or {}
    cells = {}
    files = [p for p in d.glob("bench-*.json") if CELL_RE.match(p.name)]
    for p in sorted(
        files, key=lambda p: [int(x) for x in CELL_RE.match(p.name).groups()]
    ):
        i, o, c = CELL_RE.match(p.name).groups()
        cells[f"{i}x{o}-c{c}"] = cell_stats(p, args)
    report = {
        "arm": meta.get("arm", d.name),
        "meta": meta,
        "window": window_label(meta),
        "cells": cells,
        "quality": quality(
            d,
            Path(args.baseline) if args.baseline else None,
            Path(args.floor) if args.floor else None,
            args.python,
        ),
        "validity": validity(d),
    }
    (d / "scoreboard.json").write_text(json.dumps(report, indent=1))
    md = markdown(report)
    (d / "scoreboard.md").write_text(md)
    print(md)
    return 0


def build_table(args: argparse.Namespace) -> int:
    reps = [json.loads((Path(p) / "scoreboard.json").read_text()) for p in args.dirs]
    ref = reps[0]
    L = [f"# Scoreboard comparison (reference: `{ref['arm']}`)", ""]
    for r in reps:
        L.append(f"- `{r['arm']}`: {r['window']}")
    L += [
        "",
        "| cell | arm | output tok/s | vs ref | TTFT mean (s) | prefill tok/s | decode tok/s/req "
        "| ITL p99 (ms) | goodput tok/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ref["cells"]:
        base_out = ref["cells"][name]["aggregate"]["output_tok_s"]
        for r in reps:
            c = r["cells"].get(name)
            if not c:
                continue
            a = c["aggregate"]
            ratio = (
                (a["output_tok_s"] / base_out)
                if base_out and a["output_tok_s"]
                else None
            )
            if c.get("per_request"):
                row = (
                    fmt(c["ttft_ms"]["mean"] / 1000),
                    fmt(c["prefill_tok_s"]["mean"], 0),
                    fmt(c["decode_tok_s"]["mean"], 1),
                    fmt(c["itl_ms"]["p99"], 1),
                    fmt(a.get("goodput_tok_s"), 1),
                )
            else:
                row = ("-",) * 5
            L.append(
                f"| {name} | {r['arm']} | {fmt(a['output_tok_s'], 1)} | "
                f"{'' if ratio is None else f'{ratio:.3f}x'} | "
                + " | ".join(row)
                + " |"
            )
    L += [
        "",
        "| arm | short PPL | long PPL | prompt KL (arm vs ref) | decode KL mean | KFD ok | thermal ok |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in reps:
        q = r["quality"]
        vb = q.get("vs_baseline") or {}
        pk = (vb.get("prompt_kl") or {}).get("arm") or {}
        kl = next(iter(pk.values()))["kl"] if pk else None
        dk = (vb.get("decode_kl") or {}).get("ref-cand")
        dkl = f"{dk['kl_mean']:.3e}" if dk else "-"
        L.append(
            f"| {r['arm']} | {fmt(q.get('ppl_short'), 3)} | {fmt(q.get('ppl_long'), 4)} | "
            f"{fmt(kl, 5)} | {dkl} | "
            f"{r['validity'].get('kfd_ok')} | {r['validity'].get('thermal', {}).get('limit_ok')} |"
        )
    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm", help="report for one arm directory")
    a.add_argument("dir")
    a.add_argument("--baseline", help="baseline arm directory (same checkpoint)")
    a.add_argument("--floor", help="second boot of the baseline (noise floor)")
    a.add_argument(
        "--python", default=sys.executable, help="python with numpy for topk_kl"
    )
    a.add_argument("--slo-ttft-base-s", type=float, default=2.0)
    a.add_argument("--slo-ttft-s-per-ktok", type=float, default=1.0)
    a.add_argument("--slo-itl-p99-ms", type=float, default=100.0)
    a.set_defaults(func=build_arm)
    t = sub.add_parser("table", help="compare arm reports (first = reference)")
    t.add_argument("dirs", nargs="+")
    t.add_argument("--out")
    t.set_defaults(func=build_table)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
