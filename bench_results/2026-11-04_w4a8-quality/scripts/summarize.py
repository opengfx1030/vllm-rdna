#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tables for the W4A8 quality run (A = W4A16, B = W4A8, A2/B2 repeats; Ad/Bd =
A/B with the fp32-accumulator MoE epilogue, windows 0-31 scored twice).

    python summarize.py RUN_DIR OUT_DIR [TOKENIZER_DIR]

RUN_DIR holds arms/<arm>/{topk.npz,topk_rep.npz,bench.json,lp.json,metrics-*.txt,
serve.log} (the arm's final boot; discarded wedged boots in arms/<arm>/attempt<k>/),
tools/ (topk_kl.py), windows.json and hi2/ (the hi2 checkout). Writes into OUT_DIR:
topk.md/json (topk_kl pairs), tasks.md/json (hi2 bench/compare.py),
teacher_forced.json, added.json (W4A8 effect over the floors), topk_repeat.json
(within-boot repeats), topk_fp32accum.json (kernel view), nondeterminism_positions
.json (largest within-boot logprob moves), arms.md/json (MTP acceptance, W4A8
shapes, rdna_ar health and wedges) and summary.md (every table, effect pairs next
to their noise floors). TOKENIZER_DIR (the model snapshot) decodes window text.
"""

import json
import math
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import regex as re

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
ARMS = ("A", "B", "A2", "B2")
SHOW = ("A", "A2", "B", "B2")  # column order of the per-arm tables
EFFECT = (("A", "B"), ("A", "B2"), ("A2", "B"), ("A2", "B2"))
FLOOR = (("A", "A2"), ("B", "B2"))
MTP = 2
PY = sys.executable
sys.path.insert(0, str(RUN / "hi2"))
from bench import compare as hi2_compare  # noqa: E402

WEDGE = re.compile(
    r"(\d\d:\d\d:\d\d) \[rdna_all_reduce\.py:\d+\] rdna_ar: WEDGED -- rank (\d+) "
    r"timed out after ~(\d+) ms of spinning at collective #(\d+): "
    r"peer rank (\d+)'s flag never arrived"
)
LATE = re.compile(
    r"(\d\d:\d\d:\d\d) \[rdna_all_reduce\.py:\d+\] rdna_ar: late peer -- rank (\d+) "
    r"waited ~(\d+) ms for peer rank (\d+) at collective #(\d+)"
)


def arm(name: str) -> Path:
    return RUN / "arms" / name


def metrics(path: Path) -> dict[str, float]:
    """Prometheus text -> {name: value summed over label sets} (+ per-pos keys)."""
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, value = line.rpartition(" ")
        name = key.split("{", 1)[0]
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            pos = key.split('position="', 1)[1].split('"', 1)[0]
            name = f"{name}[{pos}]"
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return out


def arm_facts(name: str) -> dict:
    """The arm's final boot (the only one whose data is used): MTP acceptance over
    the suite phase from /metrics, W4A8 fast-path shapes, all-reduce health; plus
    the rdna_ar wedges and late peers of the discarded boots (attempt<k>/)."""
    d = arm(name)
    m0 = metrics(d / "metrics-after-topk.txt")
    m1 = metrics(d / "metrics-after-suite.txt")

    def delta(key: str) -> float:
        return m1.get(key, 0.0) - m0.get(key, 0.0)

    drafts = delta("vllm:spec_decode_num_drafts_total")
    drafted = delta("vllm:spec_decode_num_draft_tokens_total")
    acc = delta("vllm:spec_decode_num_accepted_tokens_total")
    pos_key = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
    log = (d / "serve.log").read_text(errors="replace")
    shapes = {
        line.split("fast path fired: ", 1)[1].strip()
        for line in log.splitlines()
        if "W4A8-MOE-DEBUG] fast path fired" in line
    }
    ms = sorted({int(s.split("M=", 1)[1].split()[0]) for s in shapes})
    discarded = []
    for a in sorted(d.glob("attempt*"), key=lambda p: int(p.name[7:])):
        alog = (a / "serve.log").read_text(errors="replace")
        discarded.append(
            {
                "dir": a.name,
                "wedged": [
                    dict(zip(("time", "rank", "spin_ms", "collective", "peer"), w))
                    for w in WEDGE.findall(alog)
                ],
                "late_peers": [
                    dict(zip(("time", "rank", "wait_ms", "peer", "collective"), w))
                    for w in LATE.findall(alog)
                ],
            }
        )
    return {
        "acceptance_rate": acc / drafted if drafted else math.nan,
        "mean_acceptance_length": 1 + acc / drafts if drafts else math.nan,
        "acceptance_per_pos": [
            delta(f"{pos_key}[{p}]") / drafts if drafts else math.nan
            for p in range(MTP)
        ],
        "drafted_tokens": int(drafted),
        "w4a8_shapes": sorted(shapes),
        "w4a8_m_min": ms[0] if ms else None,
        "w4a8_m_max": ms[-1] if ms else None,
        "rdna_ar_active": log.count("rdna_ar: one-shot all-reduce active"),
        "rdna_ar_disabled": log.count("rdna_ar: disabled"),
        "wedged": "rdna_ar: WEDGED" in log,
        "late_peers": [
            dict(zip(("time", "rank", "wait_ms", "peer", "collective"), w))
            for w in LATE.findall(log)
        ],
        "discarded_boots": discarded,
    }


def teacher_forced(ref: dict, cand: dict, reps: int = 10000) -> dict:
    """Per task: token-weighted mean of lp_ref - lp_cand with a bootstrap CI."""
    out = {}
    rng = np.random.default_rng(0)
    for task in ref["results"]:
        sums, lens = [], []
        for i, a in ref["results"][task].items():
            b = cand["results"].get(task, {}).get(i)
            if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and a:
                sums.append(sum(x - y for x, y in zip(a, b)))
                lens.append(len(a))
        s, n = np.asarray(sums), np.asarray(lens)
        idx = rng.integers(0, len(s), size=(reps, len(s)))
        boot = s[idx].sum(1) / n[idx].sum(1)
        out[task] = {
            "items": len(s),
            "tokens": int(n.sum()),
            "gap": float(s.sum() / n.sum()),
            "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        }
    return out


def boot(stat, n: int, reps: int = 10000, seed: int = 7) -> list[float]:
    """95% bootstrap interval of stat(idx) over n paired units."""
    rng = np.random.default_rng(seed)
    vals = [stat(rng.integers(0, n, size=n)) for _ in range(reps)]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def cross_minus_floor(per_unit: dict, idx: np.ndarray) -> float:
    """Mean over the four cross pairs minus the mean over the two floor pairs."""
    cross = np.mean([per_unit[p][idx].mean() for p in EFFECT])
    return float(cross - np.mean([per_unit[p][idx].mean() for p in FLOOR]))


def b_minus_a(per_arm: dict, idx: np.ndarray) -> float:
    """Mean of B and B2 minus the mean of A and A2."""
    b = (per_arm["B"][idx] + per_arm["B2"][idx]) / 2
    return float((b - (per_arm["A"][idx] + per_arm["A2"][idx]) / 2).mean())


def added_windows(paths: dict[str, Path] | None = None) -> dict:
    """Per domain: W4A8 effect over the boot-to-boot floor, windows resampled
    jointly for all arms. KL and top-1 disagreement: mean of the four cross pairs
    (A-B, A-B2, A2-B, A2-B2) minus the mean of the two floor pairs (A-A2, B-B2).
    NLL: mean of B and B2 minus the mean of A and A2. `paths` maps the four roles
    to topk npz files (default: each arm's topk.npz)."""
    sys.path.insert(0, str(RUN / "tools"))
    import topk_kl

    paths = paths or {a: arm(a) / "topk.npz" for a in ARMS}
    data = {a: topk_kl.load(str(paths[a]))[1] for a in ARMS}
    doms = sorted({k.split(".", 1)[0] for k in data["A"] if k.endswith(".act_lp")})
    out = {}
    for dom in doms:
        z = {
            a: {f: data[a][f"{dom}.{f}"] for f in (*topk_kl.FIELDS, "act_id")}
            for a in ARMS
        }
        n = z["A"]["act_lp"].shape[0]
        kl, dis = {}, {}
        for a, b in EFFECT + FLOOR:
            st = [topk_kl.window_stats(z[a], z[b], w) for w in range(n)]
            kl[a, b] = np.array([s["kl"].mean() for s in st])
            dis[a, b] = np.array([1.0 - s["top1"].mean() for s in st])
        nll = {a: -z[a]["act_lp"].mean(1) for a in ARMS}
        every = np.arange(n)
        out[dom] = {
            "windows": n,
            "kl_floor": {f"{a}-{b}": float(kl[a, b].mean()) for a, b in FLOOR},
            "kl_cross": {f"{a}-{b}": float(kl[a, b].mean()) for a, b in EFFECT},
            "kl_added": cross_minus_floor(kl, every),
            "kl_added_ci": boot(partial(cross_minus_floor, kl), n),
            "top1_disagree_floor": {
                f"{a}-{b}": float(dis[a, b].mean()) for a, b in FLOOR
            },
            "top1_disagree_cross": {
                f"{a}-{b}": float(dis[a, b].mean()) for a, b in EFFECT
            },
            "top1_disagree_added": cross_minus_floor(dis, every),
            "top1_disagree_added_ci": boot(partial(cross_minus_floor, dis), n),
            "nll_shift": b_minus_a(nll, every),
            "nll_shift_ci": boot(partial(b_minus_a, nll), n),
            "nll_floor_A2_minus_A": float((nll["A2"] - nll["A"]).mean()),
            "nll_floor_B2_minus_B": float((nll["B2"] - nll["B"]).mean()),
        }
    return out


def added_tasks(runs: dict) -> dict:
    """Per family and pooled, items resampled jointly for all arms: accuracy shift
    (mean of B, B2 minus mean of A, A2) and flip rate added over the floor (mean
    flip rate of the four cross pairs minus that of A-A2 and B-B2)."""
    fams = list(runs["A"]["results"])
    ok = {}
    for fam in fams:
        rows = [runs[a]["results"].get(fam, {}) for a in ARMS]
        ids = sorted(set.intersection(*(set(r) for r in rows)), key=str)
        ids = [i for i in ids if all(hi2_compare.completed(r[i]) for r in rows)]
        ok[fam] = {
            a: np.array([bool(r[i]["ok"]) for i in ids], float)
            for a, r in zip(ARMS, rows)
        }
    ok["pooled"] = {a: np.concatenate([ok[f][a] for f in fams]) for a in ARMS}
    out = {}
    for fam, m in ok.items():
        n = len(m["A"])
        flip = {(a, b): (m[a] != m[b]).astype(float) for a, b in EFFECT + FLOOR}
        every = np.arange(n)
        out[fam] = {
            "n": n,
            "acc": {a: float(m[a].mean()) for a in ARMS},
            "shift": b_minus_a(m, every),
            "shift_ci": boot(partial(b_minus_a, m), n),
            "flip_floor": {f"{a}-{b}": float(flip[a, b].mean()) for a, b in FLOOR},
            "flip_cross": {f"{a}-{b}": float(flip[a, b].mean()) for a, b in EFFECT},
            "flips_added": cross_minus_floor(flip, every),
            "flips_added_ci": boot(partial(cross_minus_floor, flip), n),
        }
    return out


def gap_added(sums: dict, toks: np.ndarray, idx: np.ndarray) -> float:
    """Token-weighted mean(gap A-B, gap A-B2) - gap A-A2 over items idx."""
    t = toks[idx].sum()
    cross = (sums["B"][idx].sum() + sums["B2"][idx].sum()) / (2 * t)
    return float(cross - sums["A2"][idx].sum() / t)


def added_teacher_forced(lp: dict) -> dict:
    """Per family, items resampled jointly: the teacher-forced gap on A's
    completions added by W4A8 = mean(gap A-B, gap A-B2) - gap A-A2."""
    out = {}
    for fam in lp["A"]["results"]:
        rows = {a: lp[a]["results"].get(fam, {}) for a in ARMS}
        ids = [
            i
            for i, x in rows["A"].items()
            if isinstance(x, list)
            and x
            and all(
                isinstance(rows[a].get(i), list) and len(rows[a][i]) == len(x)
                for a in ARMS
            )
        ]
        ref = {i: np.asarray(rows["A"][i]) for i in ids}
        sums = {
            a: np.array([(ref[i] - np.asarray(rows[a][i])).sum() for i in ids])
            for a in ("A2", "B", "B2")
        }
        toks = np.array([len(ref[i]) for i in ids], float)
        out[fam] = {
            "items": len(ids),
            "tokens": int(toks.sum()),
            "gap": {f"A-{a}": float(sums[a].sum() / toks.sum()) for a in sums},
            "added": gap_added(sums, toks, np.arange(len(ids))),
            "added_ci": boot(partial(gap_added, sums, toks), len(ids)),
        }
    return out


def repeat_pairs() -> dict:
    """First REPEAT windows per domain: within-boot repeats (topk.npz vs
    topk_rep.npz of one boot) next to cross-boot pairs on the same windows."""
    sys.path.insert(0, str(RUN / "tools"))
    import topk_kl

    full = {a: np.load(arm(a) / "topk.npz") for a in ARMS}
    rep = {
        a: np.load(arm(a) / "topk_rep.npz")
        for a in ARMS
        if (arm(a) / "topk_rep.npz").exists()
    }
    if not rep:
        return {}
    any_rep = next(iter(rep.values()))
    doms = sorted({k.split(".", 1)[0] for k in any_rep.files if k.endswith(".act_lp")})
    n = {d: any_rep[f"{d}.act_lp"].shape[0] for d in doms}

    def sub(z, d: str) -> dict:
        return {f: z[f"{d}.{f}"][: n[d]] for f in (*topk_kl.FIELDS, "act_id")}

    pairs = {f"{a} run 1 - {a} run 2": ("within-boot", full[a], rep[a]) for a in rep}
    for a, b in FLOOR:
        pairs[f"{a}-{b}"] = ("cross-boot floor", full[a], full[b])
    for a, b in EFFECT:
        pairs[f"{a}-{b}"] = ("W4A8", full[a], full[b])
    return {
        label: {
            "kind": kind,
            "domains": {
                d: topk_kl.domain_pair(sub(x, d), sub(y, d), 2000, 1) for d in doms
            },
        }
        for label, (kind, x, y) in pairs.items()
    }


def deterministic_pairs() -> dict:
    """Kernel view: arms Ad / Bd run the MoE epilogue with the fp32 scratch
    accumulator (VLLM_RDNA2_MOE_FP32_ACCUM=1) and score the first 32 windows per
    domain twice (topk.npz, topk_rep.npz). Pairs on those windows: each arm's
    within-boot repeat, W4A16 vs W4A8 on each pass, and production vs fp32."""
    ad, bd = arm("Ad"), arm("Bd")
    if not ((ad / "topk_rep.npz").exists() and (bd / "topk_rep.npz").exists()):
        return {}
    sys.path.insert(0, str(RUN / "tools"))
    import topk_kl

    z = {
        "Ad1": np.load(ad / "topk.npz"),
        "Ad2": np.load(ad / "topk_rep.npz"),
        "Bd1": np.load(bd / "topk.npz"),
        "Bd2": np.load(bd / "topk_rep.npz"),
        "A": np.load(arm("A") / "topk.npz"),
        "B": np.load(arm("B") / "topk.npz"),
    }
    doms = sorted({k.split(".", 1)[0] for k in z["Ad1"].files if k.endswith(".act_lp")})
    n = {d: z["Ad1"][f"{d}.act_lp"].shape[0] for d in doms}

    def sub(key: str, d: str) -> dict:
        return {f: z[key][f"{d}.{f}"][: n[d]] for f in (*topk_kl.FIELDS, "act_id")}

    pairs = {
        "Ad run 1 - Ad run 2": ("fp32 floor, W4A16", "Ad1", "Ad2"),
        "Bd run 1 - Bd run 2": ("fp32 floor, W4A8", "Bd1", "Bd2"),
        "Ad-Bd run 1": ("fp32, W4A8 effect", "Ad1", "Bd1"),
        "Ad-Bd run 2": ("fp32, W4A8 effect", "Ad2", "Bd2"),
        "A-Ad": ("CAS vs fp32 epilogue, W4A16", "A", "Ad1"),
        "B-Bd": ("CAS vs fp32 epilogue, W4A8", "B", "Bd1"),
    }
    return {
        label: {
            "kind": kind,
            "domains": {
                d: topk_kl.domain_pair(sub(x, d), sub(y, d), 2000, 1) for d in doms
            },
        }
        for label, (kind, x, y) in pairs.items()
    }


def first_moves(absd: np.ndarray, t: float) -> list[int]:
    """Per window: the first position with abs dlogp >= t (-1: none)."""
    return [int(r.argmax()) if r.any() else -1 for r in absd >= t]


def worst_positions(windows: dict, tok, top: int = 50, big: float = 1.0) -> dict:
    """Within-boot repeats (topk.npz vs topk_rep.npz of one boot): the `top`
    positions per domain with the largest change in the actual token's logprob,
    with the window's earlier moves, plus counts of large moves per domain."""
    out = {}
    for a in (*ARMS, "Ad", "Bd"):
        rep_path = arm(a) / "topk_rep.npz"
        if not rep_path.exists():
            continue
        z1, z2 = np.load(arm(a) / "topk.npz"), np.load(rep_path)
        doms = sorted({k.split(".", 1)[0] for k in z2.files if k.endswith(".act_lp")})
        res = {}
        for d in doms:
            n = z2[f"{d}.act_lp"].shape[0]
            if not np.array_equal(z1[f"{d}.act_id"][:n], z2[f"{d}.act_id"]):
                raise SystemExit(f"{a} {d}: the two passes scored different tokens")
            lp1, lp2 = z1[f"{d}.act_lp"][:n], z2[f"{d}.act_lp"]
            absd = np.abs(lp2 - lp1)
            bigm = absd >= big
            order = np.argsort(absd, axis=None)[::-1][:top]
            rows = []
            for flat in order:
                w, p = (int(i) for i in np.unravel_index(flat, absd.shape))
                ids = windows[d]["windows"][w]
                t1 = int(z1[f"{d}.top_ids"][w, p, 0])
                t2 = int(z2[f"{d}.top_ids"][w, p, 0])
                earlier = np.flatnonzero(bigm[w, :p])
                first = np.flatnonzero(bigm[w])
                row = {
                    "window": w,
                    "position": p,
                    "token_id": int(ids[p + 1]),
                    "logp_run1": float(lp1[w, p]),
                    "logp_run2": float(lp2[w, p]),
                    "rank_run1": int(z1[f"{d}.act_rank"][w, p]),
                    "rank_run2": int(z2[f"{d}.act_rank"][w, p]),
                    "top1_run1": t1,
                    "top1_logp_run1": float(z1[f"{d}.top_lp"][w, p, 0]),
                    "top1_run2": t2,
                    "top1_logp_run2": float(z2[f"{d}.top_lp"][w, p, 0]),
                    "earlier_moves_ge_1nat": int(earlier.size),
                    "earlier_max_abs_dlogp": float(absd[w, :p].max()) if p else 0.0,
                    "first_move_ge_1nat_position": int(first[0]) if first.size else -1,
                    "window_moves_ge_1nat": int(bigm[w].sum()),
                }
                if tok is not None:
                    row["token"] = tok.decode([ids[p + 1]])
                    row["top1_run1_text"] = tok.decode([t1])
                    row["top1_run2_text"] = tok.decode([t2])
                    row["context"] = tok.decode(ids[max(0, p - 15) : p + 1])
                rows.append(row)
            hist = np.histogram(np.nonzero(bigm)[1], bins=8, range=(0, 2048))[0]
            res[d] = {
                "windows": n,
                "positions": int(absd.size),
                "identical_positions": int((absd == 0).sum()),
                "moves_ge": {
                    str(t): int((absd >= t).sum()) for t in (0.1, 1.0, 5.0, 10.0)
                },
                "windows_with_move_ge_5nat": int((absd.max(1) >= 5).sum()),
                "moves_ge_1nat_by_position_bin_256": [int(x) for x in hist],
                "first_move_ge_0.1nat_by_window": first_moves(absd, 0.1),
                "first_move_ge_1nat_by_window": first_moves(absd, 1.0),
                "worst": rows,
            }
        out[a] = res
    return out


def prefix_cache_rows() -> list[str]:
    """Prefix-cache and prompt-token counters at every /metrics snapshot."""
    keys = (
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:prompt_tokens_total",
    )
    snaps = ("0-start", "after-topk", "after-suite", "after-tfkld", "after-repeat")
    rows = [
        "| arm | snapshot | prefix-cache queries | prefix-cache hits | prompt tokens |",
        "|---|---|---:|---:|---:|",
    ]
    for a in (*ARMS, "Ad", "Bd"):
        for snap in snaps:
            path = arm(a) / f"metrics-{snap}.txt"
            if path.exists():
                m = metrics(path)
                vals = " | ".join(f"{m.get(k, 0.0):,.0f}" for k in keys)
                rows.append(f"| {a} | {snap} | {vals} |")
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # 1. teacher-forced top-20 windows
    pairs = [f"{a}-{b}={arm(a)}/topk.npz,{arm(b)}/topk.npz" for a, b in EFFECT + FLOOR]
    subprocess.run(
        [
            PY,
            str(RUN / "tools/topk_kl.py"),
            "compare",
            *[x for p in pairs for x in ("--pair", p)],
            "--json",
            str(OUT / "topk.json"),
            "--md",
            str(OUT / "topk.md"),
        ],
        check=True,
    )
    topk = json.loads((OUT / "topk.json").read_text())["pairs"]
    md_topk = OUT / "topk.md"
    md_topk.write_text(
        "# Teacher-forced top-20 windows, per pair\n\n" + md_topk.read_text()
    )
    # 2. task suite: every arm against A, and B2 against B
    runs = {a: json.loads((arm(a) / "bench.json").read_text()) for a in ARMS}
    tasks = {
        f"{a}-{b}": hi2_compare.compare(runs[a], runs[b]) for a, b in EFFECT + FLOOR
    }
    (OUT / "tasks.json").write_text(
        json.dumps(hi2_compare.json_values(tasks), indent=1, allow_nan=False)
    )
    md = [
        hi2_compare.markdown(f"{a}.json", f"{b}.json", tasks[f"{a}-{b}"], True)
        for a, b in (("A", "B"), ("A", "A2"), ("B", "B2"))
    ]
    (OUT / "tasks.md").write_text(
        "# Task suite, hi2 bench/compare.py\n\n" + "\n".join(md)
    )
    # 3. teacher-forced gaps on A's completions
    lp = {a: json.loads((arm(a) / "lp.json").read_text()) for a in ARMS}
    tf = {
        f"{a}-{b}": teacher_forced(lp[a], lp[b])
        for a, b in (("A", "B"), ("A", "A2"), ("A", "B2"), ("B", "B2"), ("A2", "B2"))
    }
    (OUT / "teacher_forced.json").write_text(json.dumps(tf, indent=1))
    # 4. arm facts
    facts = {a: arm_facts(a) for a in ARMS}
    (OUT / "arms.json").write_text(json.dumps(facts, indent=1))
    lines = [
        "| arm | W4A8 | discarded boots | rdna_ar on, final boot | "
        "MTP acceptance (pos 1 / pos 2) | mean accepted length | drafted tokens | "
        "W4A8 fast-path shapes (M range) |",
        "|---|---|---:|---|---|---:|---:|---|",
    ]
    wedges = [
        "| arm | boot | time | collective | ranks timed out (spin ms) | missing peer |",
        "|---|---|---|---:|---|---:|",
    ]
    for a in ARMS:
        f = facts[a]
        rng = (
            f"{len(f['w4a8_shapes'])} ({f['w4a8_m_min']}-{f['w4a8_m_max']})"
            if f["w4a8_shapes"]
            else "0"
        )
        healthy = f["rdna_ar_active"] and not f["rdna_ar_disabled"] and not f["wedged"]
        lines.append(
            f"| {a} | {'on' if a.startswith('B') else 'off'} | "
            f"{len(f['discarded_boots'])} | {'yes' if healthy else 'NO'} | "
            f"{f['acceptance_rate']:.4f} ({f['acceptance_per_pos'][0]:.4f} / "
            f"{f['acceptance_per_pos'][1]:.4f}) | {f['mean_acceptance_length']:.3f} | "
            f"{f['drafted_tokens']} | {rng} |"
        )
        for boot in f["discarded_boots"]:
            for coll in sorted({w["collective"] for w in boot["wedged"]}):
                ws = [w for w in boot["wedged"] if w["collective"] == coll]
                wedges.append(
                    f"| {a} | {boot['dir']} | {ws[0]['time']} | #{coll} | "
                    + ", ".join(f"{w['rank']} ({w['spin_ms']})" for w in ws)
                    + f" | {ws[0]['peer']} |"
                )
    (OUT / "arms.md").write_text("\n".join(["# Arms", "", *lines, "", *wedges]) + "\n")
    # 5. headline: what W4A8 adds over the boot-to-boot floor
    eff = {
        "windows": added_windows(),
        "tasks": added_tasks(runs),
        "teacher_forced": added_teacher_forced(lp),
    }
    (OUT / "added.json").write_text(json.dumps(eff, indent=1))

    def ci(v: list[float], scale: float = 1.0, fmt: str = "+.5f") -> str:
        return f"[{scale * v[0]:{fmt}}, {scale * v[1]:{fmt}}]"

    s = [
        "# W4A8 quality run 2026-11-04 (tree 4a3db777ae): all tables",
        "",
        "## W4A8 effect over the boot-to-boot floor",
        "",
        "Cross = mean of A-B, A-B2, A2-B, A2-B2; floor = mean of A-A2 and B-B2; "
        "added = cross - floor. 95% intervals resample windows (or items) jointly "
        "for all four arms.",
        "",
        "| domain | KL floor A-A2 / B-B2 | KL cross | KL added [95% CI] | "
        "top-1 disagreement floor / cross | added [95% CI] | "
        "NLL shift (B+B2)/2-(A+A2)/2 [95% CI] |",
        "|---|---|---:|---|---|---|---|",
    ]
    for dom, e in eff["windows"].items():
        fl, cr = e["kl_floor"], e["kl_cross"]
        dfl, dcr = e["top1_disagree_floor"], e["top1_disagree_cross"]
        s.append(
            f"| {dom} | {fl['A-A2']:.5f} / {fl['B-B2']:.5f} | "
            f"{np.mean(list(cr.values())):.5f} | {e['kl_added']:+.5f} "
            f"{ci(e['kl_added_ci'])} | {100 * np.mean(list(dfl.values())):.2f}% / "
            f"{100 * np.mean(list(dcr.values())):.2f}% | "
            f"{100 * e['top1_disagree_added']:+.2f} pt "
            f"{ci(e['top1_disagree_added_ci'], 100, '+.2f')} | "
            f"{e['nll_shift']:+.5f} {ci(e['nll_shift_ci'])} |"
        )
    s += [
        "",
        "| family | n | acc A / A2 / B / B2 % | accuracy shift [95% CI] | "
        "flip rate floor / cross | flips added [95% CI] |",
        "|---|---:|---|---|---|---|",
    ]
    for fam, e in eff["tasks"].items():
        acc = " / ".join(f"{100 * e['acc'][a]:.1f}" for a in SHOW)
        s.append(
            f"| {fam} | {e['n']} | {acc} | {100 * e['shift']:+.1f} pt "
            f"{ci(e['shift_ci'], 100, '+.1f')} | "
            f"{100 * np.mean(list(e['flip_floor'].values())):.1f}% / "
            f"{100 * np.mean(list(e['flip_cross'].values())):.1f}% | "
            f"{100 * e['flips_added']:+.1f} pt {ci(e['flips_added_ci'], 100, '+.1f')} |"
        )
    s += [
        "",
        "| family | items / tokens | gap A-A2 (floor) | gap A-B / A-B2 | "
        "added [95% CI] (nats/token) |",
        "|---|---|---:|---|---|",
    ]
    for fam, e in eff["teacher_forced"].items():
        g = e["gap"]
        s.append(
            f"| {fam} | {e['items']} / {e['tokens']} | {g['A-A2']:+.5f} | "
            f"{g['A-B']:+.5f} / {g['A-B2']:+.5f} | {e['added']:+.5f} "
            f"{ci(e['added_ci'])} |"
        )
    # 6. detail tables: effect pairs against the floor pairs
    s += ["", "## Teacher-forced windows (top-20, per domain)", ""]
    s.append("| domain | A PPL | A2 PPL | B PPL | B2 PPL |")
    s.append("|---|---:|---:|---:|---:|")
    for dom in topk["A-B"]["domains"]:
        ppl = {
            "A": topk["A-B"]["domains"][dom]["ppl_ref"],
            "A2": topk["A-A2"]["domains"][dom]["ppl_cand"],
            "B": topk["A-B"]["domains"][dom]["ppl_cand"],
            "B2": topk["B-B2"]["domains"][dom]["ppl_cand"],
        }
        s.append(f"| {dom} | " + " | ".join(f"{ppl[a]:.4f}" for a in SHOW) + " |")
    s += [
        "",
        "| domain | pair | kind | KL nats/token [95% CI] | coverage | "
        "dNLL [95% CI] | top-1 agree | abs dlogp p50 / p99 / max |",
        "|---|---|---|---|---:|---|---:|---|",
    ]
    for dom in topk["A-B"]["domains"]:
        for label, res in topk.items():
            d = res["domains"][dom]
            kind = "floor" if label in ("A-A2", "B-B2") else "W4A8"
            s.append(
                f"| {dom} | {label} | {kind} | {d['kl']:.5f} "
                f"[{d['kl_ci'][0]:.5f}, {d['kl_ci'][1]:.5f}] | {d['coverage']:.4f} | "
                f"{d['dnll']:+.5f} [{d['dnll_ci'][0]:+.5f}, {d['dnll_ci'][1]:+.5f}] | "
                f"{d['top1']:.4f} | {d['absdlp_p50']:.4f} / {d['absdlp_p99']:.4f}"
                f" / {d['absdlp_max']:.3f} |"
            )
    reps = repeat_pairs()
    (OUT / "topk_repeat.json").write_text(json.dumps(reps, indent=1))
    if reps:
        any_pair = next(iter(reps.values()))["domains"]
        nwin = {d: v["windows"] for d, v in any_pair.items()}
        s += [
            "",
            "## Within-boot repeat vs cross-boot, same windows "
            + ", ".join(f"{d} 0-{n - 1}" for d, n in nwin.items()),
            "",
            "| domain | pair | kind | KL nats/token [95% CI] | top-1 agree | "
            "abs dlogp p99 / max | identical positions |",
            "|---|---|---|---|---:|---|---:|",
        ]
        for dom in nwin:
            for label, res in reps.items():
                d = res["domains"][dom]
                s.append(
                    f"| {dom} | {label} | {res['kind']} | {d['kl']:.5f} "
                    f"[{d['kl_ci'][0]:.5f}, {d['kl_ci'][1]:.5f}] | {d['top1']:.4f} | "
                    f"{d['absdlp_p99']:.4f} / {d['absdlp_max']:.3f} | "
                    f"{100 * d['identical_positions']:.1f}% |"
                )
    det = deterministic_pairs()
    kern = {}
    if det:
        # same 2 x 2 design with the two passes of one boot as the "repeat" arm
        kern = added_windows(
            {
                "A": arm("Ad") / "topk.npz",
                "A2": arm("Ad") / "topk_rep.npz",
                "B": arm("Bd") / "topk.npz",
                "B2": arm("Bd") / "topk_rep.npz",
            }
        )
    (OUT / "topk_fp32accum.json").write_text(
        json.dumps({"pairs": det, "added": kern}, indent=1)
    )
    if det:
        any_pair = next(iter(det.values()))["domains"]
        nwin = {d: v["windows"] for d, v in any_pair.items()}
        s += [
            "",
            "## Kernel view: fp32-accumulator MoE epilogue (arms Ad, Bd), windows "
            + ", ".join(f"{d} 0-{n - 1}" for d, n in nwin.items()),
            "",
            "| domain | pair | kind | PPL ref / cand | dNLL [95% CI] | "
            "KL nats/token [95% CI] | top-1 agree | abs dlogp p50 / p99 / max | "
            "identical positions |",
            "|---|---|---|---|---|---|---:|---|---:|",
        ]
        for dom in nwin:
            for label, res in det.items():
                d = res["domains"][dom]
                s.append(
                    f"| {dom} | {label} | {res['kind']} | {d['ppl_ref']:.4f} / "
                    f"{d['ppl_cand']:.4f} | {d['dnll']:+.5f} [{d['dnll_ci'][0]:+.5f}, "
                    f"{d['dnll_ci'][1]:+.5f}] | {d['kl']:.5f} [{d['kl_ci'][0]:.5f}, "
                    f"{d['kl_ci'][1]:.5f}] | {d['top1']:.4f} | "
                    f"{d['absdlp_p50']:.4f} / {d['absdlp_p99']:.4f} / "
                    f"{d['absdlp_max']:.3f} | {100 * d['identical_positions']:.1f}% |"
                )
        s += [
            "",
            "Kernel view, W4A8 effect over the fp32 within-boot floor: cross = mean "
            "of Ad-Bd over the four pass pairings; floor = mean of Ad run 1 - run 2 "
            "and Bd run 1 - run 2; windows resampled jointly.",
            "",
            "| domain | KL floor Ad / Bd | KL cross | KL added [95% CI] | "
            "top-1 disagreement floor / cross | added [95% CI] | "
            "NLL shift Bd - Ad [95% CI] |",
            "|---|---|---:|---|---|---|---|",
        ]
        for dom, e in kern.items():
            fl, cr = e["kl_floor"], e["kl_cross"]
            dfl = np.mean(list(e["top1_disagree_floor"].values()))
            dcr = np.mean(list(e["top1_disagree_cross"].values()))
            s.append(
                f"| {dom} | {fl['A-A2']:.5f} / {fl['B-B2']:.5f} | "
                f"{np.mean(list(cr.values())):.5f} | {e['kl_added']:+.5f} "
                f"{ci(e['kl_added_ci'])} | {100 * dfl:.2f}% / {100 * dcr:.2f}% | "
                f"{100 * e['top1_disagree_added']:+.2f} pt "
                f"{ci(e['top1_disagree_added_ci'], 100, '+.2f')} | "
                f"{e['nll_shift']:+.5f} {ci(e['nll_shift_ci'])} |"
            )
    s += ["", "## Task suite (accuracy %, paired against the first arm)", ""]
    s.append(
        "| family | n | A | A2 | B | B2 | B-A [95% CI] (p) | A2-A [95% CI] (p) | "
        "B2-B [95% CI] (p) | flips A/B, A/A2, B/B2 |"
    )
    s.append("|---|---:|---:|---:|---:|---:|---|---|---|---|")

    def cell(t: dict) -> str:
        lo, hi = t["ci95"]
        return (
            f"{100 * t['diff']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] "
            f"({t['mcnemar_p']:.2g})"
        )

    for fam in tasks["A-B"]["tasks"]:
        ab, aa, bb = (tasks[k]["tasks"][fam] for k in ("A-B", "A-A2", "B-B2"))
        s.append(
            f"| {fam} | {ab['n']} | {100 * ab['ref_acc']:.1f} | "
            f"{100 * aa['cand_acc']:.1f} | {100 * ab['cand_acc']:.1f} | "
            f"{100 * bb['cand_acc']:.1f} | {cell(ab)} | {cell(aa)} | {cell(bb)} | "
            f"{ab['b'] + ab['c']}, {aa['b'] + aa['c']}, {bb['b'] + bb['c']} |"
        )

    def pooled(k: str) -> str:
        o = tasks[k]["overall"]
        lo, hi = o["pooled_ci95"]
        return (
            f"{100 * o['pooled_diff']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] "
            f"({o['pooled_mcnemar_p']:.2g})"
        )

    s.append(
        f"| pooled | {tasks['A-B']['overall']['pooled_n']} | | | | | "
        + " | ".join(pooled(k) for k in ("A-B", "A-A2", "B-B2"))
        + " | |"
    )
    s += [
        "",
        "## Prompt and completion tokens per family (arm A, server usage)",
        "",
        "| family | items | prompt min / median / max | completion median / max |",
        "|---|---:|---|---|",
    ]
    for fam, rows in runs["A"]["results"].items():
        use = [r.get("usage") or {} for r in rows.values()]
        p = sorted(u["prompt_tokens"] for u in use if "prompt_tokens" in u)
        c = sorted(u["completion_tokens"] for u in use if "completion_tokens" in u)
        if p:
            s.append(
                f"| {fam} | {len(p)} | {p[0]} / {p[len(p) // 2]} / {p[-1]} | "
                f"{c[len(c) // 2]} / {c[-1]} |"
            )
    s += ["", "## Teacher-forced gap on A's completions (nats/token)", ""]
    s.append("| family | " + " | ".join(tf) + " |")
    s.append("|---|" + "---|" * len(tf))
    for fam in tf["A-B"]:
        row = [
            f"{tf[k][fam]['gap']:+.5f} [{tf[k][fam]['ci'][0]:+.5f}, "
            f"{tf[k][fam]['ci'][1]:+.5f}]"
            for k in tf
        ]
        s.append(
            f"| {fam} ({tf['A-B'][fam]['tokens']} tok) | " + " | ".join(row) + " |"
        )
    s += ["", "## Worst windows by KL (pair A-B)", ""]
    s.append("| domain | window | KL | dNLL | top-1 | max abs dlogp | text start |")
    s.append("|---|---:|---:|---:|---:|---:|---|")
    windows = json.loads((RUN / "windows.json").read_text())["domains"]
    tok = None
    if len(sys.argv) > 3:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(sys.argv[3])
    for dom, d in topk["A-B"]["domains"].items():
        for w in d["worst"]:
            text = "-"
            if tok is not None:
                ids = windows[dom]["windows"][w["window"]][:16]
                text = json.dumps(tok.decode(ids), ensure_ascii=False)[1:-1]
                text = text.replace("|", "/")
                text = f"`{text.replace('`', '')}`"
            s.append(
                f"| {dom} | {w['window']} | {w['kl']:.5f} | {w['dnll']:+.5f} | "
                f"{w['top1']:.4f} | {w['absdlp_max']:.3f} | {text} |"
            )
    nd = worst_positions(windows, tok)
    (OUT / "nondeterminism_positions.json").write_text(
        json.dumps(nd, indent=1, ensure_ascii=False)
    )
    s += [
        "",
        "## Within-boot repeats: size of the logprob moves (actual token)",
        "",
        "| arm | domain | windows | identical positions | moves >= 0.1 / 1 / 5 / 10 "
        "nats | windows with a >= 5 nat move | moves >= 1 nat by position "
        "(256-token bins) |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for a, res in nd.items():
        for dom, r in res.items():
            m = r["moves_ge"]
            s.append(
                f"| {a} | {dom} | {r['windows']} | {r['identical_positions']} / "
                f"{r['positions']} | {m['0.1']} / {m['1.0']} / {m['5.0']} / "
                f"{m['10.0']} | {r['windows_with_move_ge_5nat']} | "
                + " ".join(str(x) for x in r["moves_ge_1nat_by_position_bin_256"])
                + " |"
            )
    s += [
        "",
        "## Prefix-cache counters at every /metrics snapshot",
        "",
        "Requests with prompt_logprobs skip the prefix-cache lookup and are not "
        "counted as queries (topk, tfkld, repeat phases).",
        "",
        *prefix_cache_rows(),
    ]
    s += ["", "## Arms", "", *lines, "", "## rdna_ar wedges (discarded boots)", ""]
    s += wedges
    (OUT / "summary.md").write_text("\n".join(s) + "\n")
    print("\n".join(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
