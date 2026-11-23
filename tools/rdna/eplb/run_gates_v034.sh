#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Open-gate chain of online EPLB (PR 191 and the stacked Mono PR 193) on the v0.34 merge.
#
# READY TO RUN, NOT RUN BY ITS AUTHOR: the b4 cards (HIP 2,3,4,5) were held by
# another lane.  Every unit cost below is a duration measured in an earlier
# receipt on this box.  A threshold marked [INFERENCE] is a proposal that no
# receipt supports yet; each one is an environment variable.  Nothing here
# changes a qualify_final.sh, serve_validate.sh or pr_receipt.sh rule.
#
#   BASE_TREE=/built/base HEAD_TREE=/built/head VENV=/venv MODEL=/model \
#     OUT=/fresh/run bash tools/rdna/eplb/run_gates_v034.sh
#   GATES="inject ppl" ...  only these gates, in the order given
#   LIST=1                  print the gates, boots, measured unit costs and ETA; stop
#   DRY_RUN=1               check the inputs and print every launch plan; no GPU, no lease
#   CONTINUE=1              go on after a failed gate (default: stop at the first)
#   BUILD=1                 build a base/head tree that has no vllm/_rocm_C*.so
#                           (CPU, build.lock, MAX_JOBS<=8, nice 19; ~13 min per tree)
#   BUILD_SEED=<tree>       donor of .deps/triton_kernels-src, vllm/vllm-rs and vllm/_version.py for
#                           BUILD=1 (default ~/w4a8_runs/leapquant/tree-eplb-034)
#
# Gate order (cheap and most informative first); each gate is one open item of
# the PR bodies, unit costs are measured on this box (source in brackets):
#
#   gate        open item                                    boots  unit
#   build       (a) independent base/head extension builds     0    757 s head, 760 s base [2026-11-23 builds]
#   inject      (d) mapped K2 under >8-row prefill injection   4    9 min 6 s per boot, stall probe 3 min 4 s
#               (e) reverse-arrival gap, paired off/chaos           [gated_mono: 18 min 13 s for 2 boots;
#               (h) stall probe with one slot reserved               pr_receipt 20261117-094408 A1]
#   ppl         (g) full quality gate (KL floor, PPL, top-1)   3    6 min 48 s per floor boot, of which 1 min 52 s
#               on a chaos boot against two off boots               is the quality hook [pr194 receipt]
#   r2          (a) flag-on ABBA; (f) prefix probe in every    4    13-14.5 min per boot, 54 min 58 s
#               boot (existing qualify_final.sh MODE=r2)             [prefix-contract-06e01fee103a]
#   r1          (a) flag-off ABBA, independent builds, token-  6    2 h 48 min: 2 x 8 min exact + 4 x
#               exact, stall; zero REGRESSION labels (hard)          37.5 min speed [final-2ea7a364848c]
#   stationary  (b) stationary traffic quiescence, (f) post-   1    STATIONARY_MINUTES + boot/probes
#               load prefix probe                                   [no prior receipt: ~8 min is INFERENCE]
#   soak        (c) PCIe/GPU-hang recurrence budget under      1    SOAK_MINUTES + boot/probes
#               chaos; never a root cause                           [no prior receipt: ~8 min is INFERENCE]
#
# Every boot is one fresh server through qualify_final.sh's guard wrapper (HIP
# 2,3,4,5 resolved to the b4 buses d9 dc df e2, temperature watcher, AER
# counters, serve.log fault scan, abort latch).  A latched ABORT stops the whole
# chain, even with CONTINUE=1.  A stage's own checks never relax a guard.
#
# What each gate asserts (all hard; CPU judge below, replayable on saved boots):
#   inject      J2off J2on J7off J7on, MTP2 so a decoder is 3 rows: 2 decoders = 6 rows, under
#               the mapped K2 cap of 8; 7 decoders = 21 rows, the modules path.  Stall probe
#               solo,inject,reverse x3, lengths 4096,16384, --max-starved 0, one decode token
#               per decoder per window, --gate-stat worst, verdict exactly PASS (NO-GATES and
#               INVALID fail).  On boots: chaos 8 swaps/512 steps; plans must overlap the
#               probe windows; the K2 3- and 6-row witnesses (mono kernel) and the >=24-row
#               modules witness on all four ranks; the iteration log must show both K2-eligible
#               decode-only steps and >8-row prefill steps in the probe.  Paired: the mean
#               maximum gap of the inject and reverse windows on <= off * (1 + tolerance).
#   ppl         P1 off, P2 chaos, P3 off (floor) with the existing pr_receipt_hook.sh collects
#               (topk_kl 64 windows x 2048 tokens, quality_ab passages).  The chaos arm first
#               carries decode traffic until PPL_MIN_PLANS plans committed.  KL and top-1 within
#               the P1-vs-P3 floor (pr_receipt_report.within_floor), PPL rise <= MAX_PPL_REL.
#   r2          unchanged qualify_final.sh MODE=r2 (itl_gate 0.5 %/off-spread envelope).
#   r1          unchanged qualify_final.sh MODE=r1, then the native cell verdicts must hold zero
#               REGRESSION labels (r1 itself only reports them).
#   stationary  planner on (no chaos), one fixed 1k/512 c=8 cell repeated for STATIONARY_MINUTES
#               with a new seed each time, every engine step logged.  The design-doc rule (online
#               phase): at most one rebalance per 10,000 steps.  After a one-third warm-up the
#               steady window needs >= 20,000 steps and no more plan starts than steps/10,000;
#               the median ITL of the last third of the cells stays within STATIONARY_ITL_DRIFT of
#               the first; the repaired prefix probe (verified reuse), PPL probe and greedy probe
#               pass after the load.  Old soak1 (mixed 1k/16k, 14 plans over 90,880 steps) exceeded
#               the rate but was not a fixed workload, so it does not decide this gate.
#   soak        chaos, the soak1 mix (1k/512 c=8 then 16k/1k c=8) for SOAK_MINUTES; every guard
#               clean.  PASS means NO RECURRENCE in the recorded budget; it does not explain the
#               PCIe completion-timeout event.  The final ledger sums every EPLB boot of the run.
#
# Prior art, measured but not on this merge: the earlier v0.34 attempt on the old rebase
# (eplb/online-mono-034, 38f3024159; receipts r2-034-20261121-111250 and -111950 under
# ~/w4a8_runs/eplb-online) stopped twice in its flag-OFF arm A1 (qualify_final.sh r2, rc 1): a KFD
# queue eviction of ~18.7-19.5 s on the first run, HSA_STATUS_ERROR_EXCEPTION code 0x1016 in
# at::native::indexFuncSmallIndex on GPU index 1, 2 and 3 on the second; AER before == after on both.
# Whether the merged tree repeats it is unknown.  Do not retry around an ABORT.
#
# Deploying a final head: extract it to a fresh tree with .sha = its commit, build it
# (BUILD=1) or copy vllm/*.so, vllm/_version.py, vllm/vllm-rs and .deps from a built tree
# whose csrc/, cmake/, CMakeLists.txt and setup.py are unchanged (git diff on the Mac).  qualify_final.sh
# refuses HEAD_TREE whose .sha differs from this script's tree, and refuses base and head
# sharing one _rocm_C binary.
#
# Exit: 0 every selected gate passed (the summary says COMPLETE or PARTIAL), 1 a gate failed
# or is incomplete, 2 bad input, 3 abort latched (thermal, AER, serve.log fault, busy GPUs).
# shellcheck disable=SC2030,SC2031
set -uo pipefail

SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
QUALIFY_FINAL=$TOOLS/tools/rdna/eplb/qualify_final.sh
PORT_DIR=$TOOLS/tools/rdna/port_v031
REPORT=$TOOLS/tools/rdna/pr_receipt_report.py
GATES_ALL="build inject ppl r2 r1 stationary soak"
GPUS_FIXED=2,3,4,5
RECIPE_FIXED=flashnext-mtp2
GATE_PORT=18248
K2_ROW_CAP=8
ROWS_PER_DECODER=3
QUALIFY_PYTHON=${QUALIFY_PYTHON:-/usr/bin/python3}

# Measured units, in whole minutes (see the table above).
ETA_BUILD_PER_TREE=13
ETA_INJECT=37
ETA_PPL=25
ETA_R2=55
ETA_R1=169
BOOT_OVERHEAD=8

die() {
    echo "gates: $*" >&2
    exit 2
}

note() {
    echo "[$(date +%H:%M:%S)] gates: $*"
}

is_uint() { [[ ${1:-} =~ ^[0-9]+$ ]]; }
is_frac() { [[ ${1:-} =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

gate_known() {
    local g
    for g in $GATES_ALL; do
        [[ $g == "$1" ]] && return 0
    done
    return 1
}

gate_label() {
    case $1 in
        build) echo "(a) independent base/head extension builds" ;;
        inject) echo "(d) mapped K2 under >8-row injection, (e) reverse-arrival gap, (h) reserved slot" ;;
        ppl) echo "(g) full quality gate on a chaos boot" ;;
        r2) echo "(a) flag-on ABBA, (f) prefix probe in every boot" ;;
        r1) echo "(a) flag-off ABBA with independent builds, zero REGRESSION labels" ;;
        stationary) echo "(b) stationary-traffic quiescence, (f) post-load prefix probe" ;;
        soak) echo "(c) PCIe/GPU-hang recurrence budget under chaos" ;;
        ledger) echo "(c) ledger of every EPLB boot of this run (no recurrence, never a cause)" ;;
    esac
}

gate_boots() {
    case $1 in
        build) echo 0 ;;
        inject) echo 4 ;;
        ppl) echo 3 ;;
        r2) echo 4 ;;
        r1) echo 6 ;;
        stationary | soak) echo 1 ;;
        ledger) echo 0 ;;
    esac
}

gate_eta() { # minutes, from the measured units
    local trees=0 t
    case $1 in
        build)
            if [[ ${BUILD:-0} == 1 ]]; then
                for t in ${BASE_TREE:-} ${HEAD_TREE:-}; do
                    compgen -G "$t/vllm/_rocm_C*.so" > /dev/null || trees=$((trees + 1))
                done
            fi
            echo $((trees * ETA_BUILD_PER_TREE))
            ;;
        inject) echo "$ETA_INJECT" ;;
        ppl) echo "$ETA_PPL" ;;
        r2) echo "$ETA_R2" ;;
        r1) echo "$ETA_R1" ;;
        stationary) echo $((STATIONARY_MINUTES + BOOT_OVERHEAD)) ;;
        soak) echo $((SOAK_MINUTES + BOOT_OVERHEAD)) ;;
    esac
}

defaults() {
    GATES=${GATES:-$GATES_ALL}
    DRY_RUN=${DRY_RUN:-0}
    LIST=${LIST:-0}
    CONTINUE=${CONTINUE:-0}
    BUILD=${BUILD:-0}
    BUILD_SEED=${BUILD_SEED:-$HOME/w4a8_runs/leapquant/tree-eplb-034}
    OWNER=${OWNER:-EplbGatesV034}
    LEASE_HELD=${LEASE_HELD:-0}
    LEASE_SH=${LEASE_SH:-$HOME/w4a8_runs/leapquant/lease.sh}
    # Sourced defaults: MAX_PPL_REL (0.02) and QUALITY_LIMIT/TOP/PASSAGES (64/20/6) are those of
    # pr_receipt.sh, STATIONARY_STEPS_PER_PLAN (10,000) is the design-doc rule, PREFIX_WORDS (1800)
    # the serve_validate.sh default, INJECT_REPEATS/INJECT_LENS (3, 4096,16384) the stall probe's.
    # [INFERENCE] proposals that no receipt supports yet: GAP_TOL, REVERSE_TOL, INJECT_MIN_OVERLAP,
    # INJECT_MIN_STEPS, PPL_MIN_PLANS, PPL_PRETRAFFIC_MAX_S, STATIONARY_MINUTES,
    # STATIONARY_ITL_DRIFT, SOAK_MINUTES and the one-third warm-up of the stationary window.
    INJECT_REPEATS=${INJECT_REPEATS:-3}
    INJECT_LENS=${INJECT_LENS:-4096,16384}
    INJECT_MIN_OVERLAP=${INJECT_MIN_OVERLAP:-1}
    INJECT_MIN_STEPS=${INJECT_MIN_STEPS:-10}
    GAP_TOL=${GAP_TOL:-0.10}
    REVERSE_TOL=${REVERSE_TOL:-0.10}
    QUALITY_LIMIT=${QUALITY_LIMIT:-64}
    QUALITY_TOP=${QUALITY_TOP:-20}
    QUALITY_PASSAGES=${QUALITY_PASSAGES:-6}
    MAX_PPL_REL=${MAX_PPL_REL:-0.02}
    PPL_MIN_PLANS=${PPL_MIN_PLANS:-4}
    PPL_PRETRAFFIC_MAX_S=${PPL_PRETRAFFIC_MAX_S:-1500}
    STATIONARY_MINUTES=${STATIONARY_MINUTES:-90}
    STATIONARY_ITL_DRIFT=${STATIONARY_ITL_DRIFT:-0.05}
    STATIONARY_STEPS_PER_PLAN=${STATIONARY_STEPS_PER_PLAN:-10000}
    SOAK_MINUTES=${SOAK_MINUTES:-240}
    PREFIX_WORDS=${PREFIX_WORDS:-1800}
    QUALIFY_POLL=${QUALIFY_POLL:-5}
    local v
    for v in DRY_RUN LIST CONTINUE BUILD LEASE_HELD; do
        [[ ${!v} == 0 || ${!v} == 1 ]] || die "$v is 0 or 1, not ${!v}"
    done
    for v in INJECT_REPEATS INJECT_MIN_OVERLAP INJECT_MIN_STEPS QUALITY_LIMIT QUALITY_TOP \
        QUALITY_PASSAGES PPL_MIN_PLANS PPL_PRETRAFFIC_MAX_S STATIONARY_MINUTES SOAK_MINUTES \
        STATIONARY_STEPS_PER_PLAN \
        PREFIX_WORDS; do
        if ! is_uint "${!v}" || ((10#${!v} < 1)); then
            die "$v is a positive integer, not ${!v}"
        fi
    done
    for v in GAP_TOL REVERSE_TOL MAX_PPL_REL STATIONARY_ITL_DRIFT; do
        is_frac "${!v}" || die "$v is a non-negative number, not ${!v}"
    done
    [[ $INJECT_LENS =~ ^[0-9]+(,[0-9]+)*$ ]] || die "INJECT_LENS is a comma list of token counts"
    local g
    for g in $GATES; do
        gate_known "$g" || die "unknown gate '$g' (known: $GATES_ALL)"
    done
}

list_gates() {
    local g total=0 eta
    printf '%-11s %-5s %-6s %s\n' gate boots 'ETA' 'open item'
    for g in $GATES; do
        eta=$(gate_eta "$g")
        total=$((total + eta))
        printf '%-11s %-5s %-6s %s\n' "$g" "$(gate_boots "$g")" "${eta}m" "$(gate_label "$g")"
    done
    printf '%-11s %-5s %-6s %s\n' total '' "${total}m" \
        "for GATES='$GATES' (build is CPU; the b4 lease is taken after it)"
    cat << EOF
ETA basis: R1 6 boots 2 h 48 min and R2 4 boots 54 min 58 s measured; inject 9 min 6 s per boot;
ppl 3 boots of 6 min 48 s (quality hook included) plus [INFERENCE] a few minutes of pre-traffic on the chaos boot;
stationary $STATIONARY_MINUTES min and soak $SOAK_MINUTES min are parameters plus ~$BOOT_OVERHEAD min of boot and probes [INFERENCE].
EOF
}

# ------------------------------------------------------------------ judge
# The CPU judge: imports qualify_boot.py (guards, summary gates, plan lines,
# fault patterns), stall_probe.py (iteration log) and pr_receipt_report.py
# (quality rule).  It never imports torch and sees no GPU.  --fixture skips the
# guard files a saved pre-wrapper boot does not have.
judge() {
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u LD_LIBRARY_PATH HIP_VISIBLE_DEVICES= \
        GATES_TOOLS="$TOOLS" PYTHONPATH="$TOOLS" "$QUALIFY_PYTHON" - "$@" << 'PY'
import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import regex as re

TOOLS = Path(os.environ["GATES_TOOLS"])
sys.path.insert(0, str(TOOLS / "tools/rdna/eplb"))
sys.path.insert(0, str(TOOLS / "tools/rdna/port_v031"))
import qualify_boot as qb  # noqa: E402
import stall_probe as sp  # noqa: E402

STAMP = re.compile(r"\b(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\b")
OLD_RANK = re.compile(r"\(VllmWorker-(\d+) pid=\d+\)")
K2_CAP = 8
ROWS_PER_DECODER = 3  # MTP2: the token and two drafts
RANKS = {0, 1, 2, 3}


def epoch(line: str, year: int) -> float | None:
    m = STAMP.search(line)
    if m is None:
        return None
    mo, d, h, mi, s = (int(x) for x in m.groups())
    return time.mktime((year, mo, d, h, mi, s, 0, 0, -1))


def rank_of(line: str) -> int | None:
    r = qb._worker_rank(line)
    if r is not None:
        return r
    m = OLD_RANK.search(line)
    return int(m.group(1)) if m else None


def year_of(path: Path) -> int:
    return time.localtime(path.stat().st_mtime).tm_year


def plan_walls(text: str, year: int) -> dict[int, tuple[float, float | None, int]]:
    """plan number -> (start, done or None, swaps) on the log's wall clock, 1 s resolution."""
    starts: dict[int, float] = {}
    dones: dict[int, float] = {}
    swaps: dict[int, int] = {}
    for line in text.splitlines():
        if "Online EPLB: plan " not in line:
            continue
        t = epoch(line, year)
        if t is None:
            continue
        s = qb.PLAN_START.search(line)
        d = qb.PLAN_DONE.search(line)
        if s is not None:
            n = int(s.group(1))
            starts[n] = min(t, starts.get(n, t))
            swaps[n] = int(s.group(3))
        elif d is not None:
            n = int(d.group(1))
            dones[n] = max(t, dones.get(n, t))
    return {n: (starts[n], dones.get(n), swaps[n]) for n in sorted(starts)}


def witnesses(text: str) -> dict:
    mono: set[int] = set()
    rows: dict[tuple[int, str, str], set[int]] = {}
    eplb_on = counters = 0
    offs: list[str] = []
    configs: list[str] = []
    for line in text.splitlines():
        m = qb.MONO_RANK.search(line)
        if m:
            mono.add(int(m.group(1)))
        if "Online EPLB: off" in line:
            offs.append(line.strip())
        r = qb.K2_ROW.search(line)
        if r:
            rk = rank_of(line)
            if rk is not None:
                rows.setdefault((int(r.group(1)), r.group(2), r.group(3)), set()).add(rk)
        if qb.EPLB_ON in line:
            eplb_on += 1
            configs.append(line.strip())
        if qb.EPLB_COUNTERS in line:
            counters += 1
    return {"mono": mono, "rows": rows, "eplb_on": eplb_on, "counters": counters,
            "offs": offs, "configs": configs}


CONFIG = {
    "off": None,
    "chaos": {"chaos": 8, "decision_steps": 512},
    "planner": {"chaos": 0, "decision_steps": 4096},
}


def compact_rows(rows: dict) -> dict:
    """'path/mode' -> the row counts logged on all four ranks; a count missing on a rank is flagged."""
    out: dict[str, str] = {}
    for path, mode in sorted({(p, m) for (_n, m, p) in rows}):
        mine = {n: rk for (n, m, p), rk in rows.items() if (p, m) == (path, mode)}
        full = sorted(n for n, rk in mine.items() if rk == RANKS)
        part = sorted((n, sorted(rk)) for n, rk in mine.items() if rk != RANKS)
        out[f"{path}/{mode}"] = ",".join(map(str, full)) + (f"  [on fewer ranks: {part}]" if part else "")
    return out


def common_checks(d: Path, kind: str, fixture: bool) -> tuple[list[str], dict, str]:
    """Probes, guards, faults, Mono ranks and the EPLB on/off markers of one boot."""
    problems: list[str] = []
    try:
        summary = qb._required(d / "summary.txt")
        serve = qb._required(d / "serve.log")
    except qb.QualificationError as exc:
        return [str(exc)], {}, ""
    problems += qb.summary_problems(summary, "none", require_dmesg=False)
    if not fixture:
        problems += qb.guard_problems(d)
    try:
        faults = qb.fault_lines(d / "serve.log")
    except qb.QualificationError as exc:
        faults = []
        problems.append(str(exc))
    if faults:
        problems.append(f"{len(faults)} serve.log fault line(s): {faults[0].strip()}")
    w = witnesses(serve)
    if w["mono"] != RANKS:
        problems.append(f"Mono ranks are {sorted(w['mono'])}, expected TP ranks 0-3")
    want = CONFIG[kind]
    if want is None:
        if w["eplb_on"] or w["counters"] or "Online EPLB" in serve:
            problems.append("an off boot logged online EPLB")
    else:
        if w["eplb_on"] != 4:
            problems.append(f"'{qb.EPLB_ON}' appears {w['eplb_on']} time(s), expected 4")
        if w["counters"] != 4:
            problems.append(f"'{qb.EPLB_COUNTERS}' appears {w['counters']} time(s), expected 4")
        if w["offs"]:
            problems.append(f"EPLB turned itself off: {w['offs'][0]}")
        for cfg in w["configs"]:
            for key, value in want.items():
                if not re.search(rf"\b{key}={value}(?:,|\))", cfg):
                    problems.append(f"EPLB config does not show {key}={value}: {cfg}")
                    break
            else:
                continue
            break
    ev = {
        "mono_ranks": sorted(w["mono"]),
        "k2_rows_on_all_four_ranks": compact_rows(w["rows"]),
        "eplb_on_lines": w["eplb_on"],
        "counter_lines": w["counters"],
        "faults": len(faults),
    }
    return problems, ev, serve


def k2_witness_problems(w_rows: dict, decoders: int | None) -> list[str]:
    problems = []
    full = {n for (n, mode, path), rk in w_rows.items()
            if mode == "full" and path == "mono kernel" and rk == RANKS}
    wanted = {qb.K2_FULL_ROWS}
    if decoders is not None and decoders * ROWS_PER_DECODER <= K2_CAP:
        wanted.add(decoders * ROWS_PER_DECODER)
    for n in sorted(wanted - full):
        problems.append(f"no {n}-row FULL K2 (-> mono kernel) witness on all four ranks")
    mods = {n for (n, mode, path), rk in w_rows.items()
            if path == "modules" and n >= qb.MODULE_MIN_ROWS and rk == RANKS}
    if not mods:
        problems.append(f"no >= {qb.MODULE_MIN_ROWS}-row modules witness on all four ranks")
    return problems


def stall_windows(js: dict) -> list[dict]:
    out = []
    for run in js.get("runs", []):
        for w in run.get("windows") or []:
            out.append({**w, "scenario": run.get("scenario"), "repeat": run.get("repeat")})
    return out


def cmd_inject_boot(a: argparse.Namespace) -> int:
    d = Path(a.dir)
    kind = "off" if a.arm == "off" else "chaos"
    problems, ev, serve = common_checks(d, kind, a.fixture)
    decoders = a.decoders
    if serve:
        w = witnesses(serve)
        problems += k2_witness_problems(w["rows"], decoders)
    # the stall probe verdict, never NO-GATES or INVALID
    try:
        txt = qb._required(d / "stall.txt")
        js = json.loads(qb._required(d / "stall" / "stall_probe.json"))
    except (qb.QualificationError, ValueError) as exc:
        problems.append(f"stall probe artefacts: {exc}")
        js = None
        txt = ""
    verdicts = re.findall(r"^verdict: (\S+)", txt, re.M)
    verdict = verdicts[-1] if verdicts else "missing"
    if verdict != "PASS":
        problems.append(f"stall probe verdict is {verdict}, not PASS (NO-GATES/INVALID/ERROR all fail)")
    iters_summary: dict = {}
    if js is not None:
        cfg = js.get("config", {})
        if cfg.get("decoders") != decoders:
            problems.append(f"stall probe ran {cfg.get('decoders')} decoders, expected {decoders}")
        if not {"solo", "inject", "reverse"} <= set(cfg.get("scenarios", [])):
            problems.append(f"stall scenarios are {cfg.get('scenarios')}")
        if cfg.get("max_starved") != 0 or not cfg.get("min_decode_tokens_during_prefill"):
            problems.append("the hard stall gates (max starved 0, decode tokens per window >= 1) are not set")
        gates = js.get("gates") or []
        if not gates:
            problems.append("stall probe evaluated no gate")
        bad = [g for g in gates if not g.get("ok")]
        if bad:
            problems.append(f"{len(bad)} stall gate(s) failed: {bad[0]}")
        errs = [r for r in js.get("runs", []) if r.get("error") or r.get("stream_errors")]
        if errs:
            problems.append(f"stall run error: {errs[0].get('error') or errs[0].get('stream_errors')}")
        wins = stall_windows(js)
        inj = [w for w in wins if w["scenario"] == "inject"]
        if not inj:
            problems.append("no inject window was measured")
        year = time.localtime(js["clock_wall0"]).tm_year
        t0 = js["clock_wall0"]
        t1 = t0 + max((w["t1"] for w in wins), default=0.0) + 10.0
        iters = sp.parse_iteration_log(str(d / "serve.log"), year)
        sel = [it for it in iters if math.floor(t0) <= it.wall <= math.ceil(t1)]
        # one scheduler step = one batch of ctx + gen tokens = the rows the MoE sees (an MTP2
        # decoder schedules 3): K = decode-only step at or under the mapped K2 cap, B = a step
        # with prefill rows over it.  K<->B flips are the K2 <-> modules switches.
        kinds = []
        for it in sel:
            if it.ctx_tokens == 0 and 0 < it.gen_tokens <= K2_CAP:
                kinds.append("K")
            elif it.ctx_tokens > 0 and it.ctx_tokens + it.gen_tokens > K2_CAP:
                kinds.append("B")
            else:
                kinds.append("-")
        k2_steps = kinds.count("K")
        big_steps = kinds.count("B")
        flips = sum(1 for x, y in zip(kinds, kinds[1:], strict=False) if {x, y} == {"K", "B"})
        iters_summary = {
            "probe_steps": len(sel),
            "decode_only_steps_within_k2_cap": k2_steps,
            "steps_with_prefill_rows_over_cap": big_steps,
            "k2_to_modules_flips": flips,
            "max_prefill_tokens_per_step": max((it.ctx_tokens for it in sel), default=0),
        }
        for w in inj:
            if (w.get("server") or {}).get("max_prefill_tokens_per_step", 0) <= K2_CAP:
                problems.append(f"inject window {w['len']} t={w['t0']:.0f}s saw no prefill step over {K2_CAP} rows")
        if big_steps < a.min_steps:
            problems.append(f"only {big_steps} step(s) with prefill rows over the K2 cap in the probe (< {a.min_steps})")
        if decoders * ROWS_PER_DECODER <= K2_CAP:
            if k2_steps < a.min_steps:
                problems.append(f"only {k2_steps} decode-only step(s) within the K2 cap in the probe (< {a.min_steps})")
            if flips < a.min_steps:
                problems.append(f"only {flips} K2 <-> modules step flip(s) in the probe (< {a.min_steps})")
        if kind == "chaos":
            plans = plan_walls(serve, year)
            done = {n: v for n, v in plans.items() if v[1] is not None}
            if not done:
                problems.append("no EPLB plan started and committed during the boot")
            over = 0
            for w in inj + [w for w in wins if w["scenario"] == "reverse"]:
                lo, hi = t0 + w["t0"] - 1.0, t0 + w["t1"] + 1.0
                if any(s < hi and e > lo for s, e, _k in done.values()):
                    over += 1
            iters_summary["windows_overlapping_a_plan"] = over
            iters_summary["windows"] = len(inj) + len([w for w in wins if w["scenario"] == "reverse"])
            iters_summary["plans_committed"] = len(done)
            if over < a.min_overlap:
                problems.append(f"only {over} injection/reverse window(s) overlap a committed plan (< {a.min_overlap})")
    ev.update({"verdict": verdict, "iterations": iters_summary, "decoders": decoders, "arm": a.arm})
    ev["ok"] = not problems
    ev["problems"] = problems
    if a.out:
        Path(a.out).write_text(json.dumps(ev, indent=1) + "\n")
    print(f"inject boot {d.name} ({a.arm}, {decoders} decoders): " + ("OK" if not problems else "NOT OK"))
    for p in problems:
        print(f"  - {p}")
    return 1 if problems else 0


def cmd_inject_pair(a: argparse.Namespace) -> int:
    problems = []
    rows = []
    try:
        off = json.loads((Path(a.off) / "stall" / "stall_probe.json").read_text())["aggregate"]["windows"]
        on = json.loads((Path(a.on) / "stall" / "stall_probe.json").read_text())["aggregate"]["windows"]
    except (OSError, ValueError, KeyError) as exc:
        print(f"inject pair {a.label}: cannot read the aggregates: {exc}")
        return 1
    keys = sorted(k for k in set(off) | set(on) if k.split(":")[0] in ("inject", "reverse"))
    if not keys or any(k not in off or k not in on for k in keys):
        problems.append(f"windows differ between the arms: off {sorted(off)} on {sorted(on)}")
    for k in keys:
        if k not in off or k not in on:
            continue
        tol = a.reverse_tol if k.startswith("reverse") else a.gap_tol
        va, vb = off[k]["max_gap_ms"]["mean"], on[k]["max_gap_ms"]["mean"]
        limit = va * (1.0 + tol)
        ok = vb <= limit
        rows.append({"window": k, "off_ms": va, "on_ms": vb, "limit_ms": limit, "tol": tol, "ok": ok})
        if not ok:
            problems.append(f"{k}: on {vb:.1f} ms > {limit:.1f} ms (= off {va:.1f} ms x {1 + tol:.2f})")
    print(f"inject pair {a.label}: " + ("OK" if not problems else "NOT OK"))
    for r in rows:
        print(f"  {r['window']:<16} off {r['off_ms']:9.1f} ms  on {r['on_ms']:9.1f} ms  "
              f"{(r['on_ms'] / r['off_ms'] - 1) * 100:+6.2f} %  limit +{r['tol'] * 100:.0f} %  "
              f"{'ok' if r['ok'] else 'FAIL'}")
    for p in problems:
        print(f"  - {p}")
    if a.out:
        Path(a.out).write_text(json.dumps({"rows": rows, "problems": problems}, indent=1) + "\n")
    return 1 if problems else 0


def cmd_ppl(a: argparse.Namespace) -> int:
    root = Path(a.root)
    report = qb.report_module()
    problems: list[str] = []
    detail: dict = {}
    for arm, kind in (("P1", "off"), ("P2", "chaos"), ("P3", "off")):
        d = root / "boots" / arm
        p, ev, serve = common_checks(d, kind, a.fixture)
        problems += [f"{arm}: {x}" for x in p]
        hook = d / "load_hook.txt"
        try:
            htxt = qb._required(hook)
        except qb.QualificationError as exc:
            problems.append(f"{arm}: {exc}")
            htxt = ""
        if not re.search(r"^PASS pr_receipt quality collects", htxt, re.M):
            problems.append(f"{arm}: the quality collects did not end in PASS")
        if arm == "P2" and serve:
            year = year_of(d / "serve.log")
            plans = plan_walls(serve, year)
            try:
                cstart = float((d / "collect-start.epoch").read_text().strip())
            except (OSError, ValueError):
                problems.append("P2: collect-start.epoch is missing")
                cstart = 0.0
            before = [n for n, (s, e, k) in plans.items() if e is not None and e <= cstart + 1.0]
            detail["P2_plans_committed_before_collect"] = len(before)
            detail["P2_swaps_before_collect"] = sum(plans[n][2] for n in before)
            if len(before) < a.min_plans:
                problems.append(f"P2: only {len(before)} plan(s) committed before the quality collect (< {a.min_plans})")
        detail[arm] = ev
    try:
        kl = json.loads((root / "kl.json").read_text())["pairs"]
        floor, head = kl["floor"], kl["head"]
        rowsq = {}
        for name, f in floor["domains"].items():
            h = head["domains"][name]
            kl_ok, top1_ok = report.within_floor(h, f)
            rowsq[name] = {"kl": h["kl"], "floor_kl_hi": f["kl_ci"][1], "top1": h["top1"],
                           "floor_top1_lo": f["top1_ci"][0], "kl_ok": kl_ok, "top1_ok": top1_ok}
            if not kl_ok:
                problems.append(f"{name}: KL {h['kl']:.5f} > floor upper bound {f['kl_ci'][1]:.5f}")
            if not top1_ok:
                problems.append(f"{name}: top-1 {h['top1']:.4f} < floor lower bound {f['top1_ci'][0]:.4f}")
        if not rowsq:
            problems.append("kl.json has no domain")
        detail["kl"] = rowsq
    except (OSError, ValueError, KeyError) as exc:
        problems.append(f"kl.json: {exc}")
    for label in ("head", "floor"):
        try:
            total = report.parse_ppl_total((root / f"ppl-{label}.txt").read_text())
        except OSError:
            total = None
        detail[f"ppl_{label}"] = total
        if total is None:
            problems.append(f"ppl-{label}.txt has no TOTAL line")
        elif label == "head" and total["verdict"] != "PASS":
            problems.append(f"PPL rise {total['rel_pct']:+.3f} % is not within {a.max_ppl_rel * 100:.1f} %")
    detail["ok"] = not problems
    detail["problems"] = problems
    Path(root / "verdict.json").write_text(json.dumps(detail, indent=1, default=str) + "\n")
    print("ppl gate: " + ("PASS" if not problems else "FAIL"))
    print(json.dumps({k: v for k, v in detail.items() if k.startswith(("P2_", "ppl_", "kl"))}, indent=1, default=str))
    for p in problems:
        print(f"  - {p}")
    return 1 if problems else 0


def cmd_traffic(a: argparse.Namespace) -> int:
    d = Path(a.dir)
    kind = "planner" if a.kind == "stationary" else "chaos"
    problems, ev, serve = common_checks(d, kind, a.fixture)
    try:
        htxt = qb._required(d / "load_hook.txt")
        t0 = float((d / "traffic-start.epoch").read_text())
        t1 = float((d / "traffic-end.epoch").read_text())
    except (qb.QualificationError, OSError, ValueError) as exc:
        problems.append(f"traffic artefacts: {exc}")
        htxt, t0, t1 = "", 0.0, 0.0
    if not re.search(r"^PASS gates traffic hook", htxt, re.M):
        problems.append("the traffic hook did not end in PASS (see load_hook.txt)")
    span = t1 - t0
    if span < a.minutes * 60 * 0.95:
        problems.append(f"the traffic window lasted {span / 60:.1f} min, expected {a.minutes}")
    result: dict = {"window_minutes": round(span / 60, 2)}
    if serve and span > 0:
        year = year_of(d / "serve.log")
        plans = plan_walls(serve, year)
        done = {n: v for n, v in plans.items() if v[1] is not None}
        unfinished = sorted(set(plans) - set(done))
        thirds = [t0 + span * i / 3.0 for i in range(4)]
        count = [sum(1 for s, _e, _k in plans.values() if thirds[i] <= s < thirds[i + 1] + (1.0 if i == 2 else 0.0))
                 for i in range(3)]
        result.update({"plans_started": len(plans), "plans_committed": len(done),
                       "swaps_committed": sum(k for _s, _e, k in done.values()),
                       "unfinished": unfinished, "plans_started_per_third": count})
        if a.kind == "stationary":
            # design doc, online phase: no oscillation, at most one rebalance per 10,000 steps on
            # stationary traffic.  The first third is warm-up (the first plans move the initial
            # placement: soak1's first five moved 32 swaps each); the rest is the steady window.
            lo = t0 + span * a.warmup_frac
            starts = sorted(s for s, _e, _k in plans.values() if lo <= s <= t1 + 1.0)
            iters = [it for it in sp.parse_iteration_log(str(d / "serve.log"), year) if lo <= it.wall <= t1]
            if len(iters) < 2:
                problems.append("no engine iteration lines in the steady window (the boot needs --enable-logging-iteration-details)")
            else:
                steps = iters[-1].index - iters[0].index
                allowed = steps // a.steps_per_plan
                result.update({"steady_window_steps": steps, "steady_plan_starts": len(starts),
                               "steady_plans_allowed": allowed})
                if steps < 2 * a.steps_per_plan:
                    problems.append(f"the steady window holds {steps} engine steps, fewer than two plan "
                                    f"intervals of {a.steps_per_plan}: raise STATIONARY_MINUTES")
                elif len(starts) > allowed:
                    problems.append(f"{len(starts)} plan(s) started in {steps} steady-state steps: more than "
                                    f"one per {a.steps_per_plan} (allowed {allowed})")
            cells = sorted((d / "traffic").glob("cell-*.json"),
                           key=lambda p: int(re.search(r"cell-(\d+)-", p.name).group(1)))
            itl = []
            for p in cells:
                try:
                    itl.append(float(json.loads(p.read_text())["median_itl_ms"]))
                except (OSError, ValueError, KeyError):
                    problems.append(f"{p.name}: no median_itl_ms")
            if len(itl) >= 6:
                k = len(itl) // 3
                first, last = statistics.median(itl[:k]), statistics.median(itl[-k:])
                result.update({"cells": len(itl), "itl_first_third_ms": first, "itl_last_third_ms": last})
                if last > first * (1.0 + a.drift):
                    problems.append(f"median ITL drifted {first:.2f} -> {last:.2f} ms (> +{a.drift * 100:.0f} %)")
            else:
                problems.append(f"only {len(itl)} traffic cell(s): too few for a drift check (>= 6)")
        else:
            if not done:
                problems.append("no EPLB plan committed during the soak")
        for probe in ("prefix-after.txt", "ppl-after.txt"):
            try:
                ptxt = qb._required(d / probe)
            except qb.QualificationError as exc:
                problems.append(str(exc))
                continue
            if probe.startswith("prefix"):
                hit = re.search(r"warm hit rate ([\d.]+)%", ptxt)
                result["prefix_after_warm_hit_rate_pct"] = float(hit.group(1)) if hit else None
                if not re.search(r"probe PASS\b", ptxt):
                    problems.append("the post-load prefix probe did not pass (verified reuse, repaired probe)")
            elif "PASS" not in ptxt:
                problems.append("the post-load PPL probe did not pass")
    ev.update(result)
    ev["ok"] = not problems
    ev["problems"] = problems
    if a.out:
        Path(a.out).write_text(json.dumps(ev, indent=1, default=str) + "\n")
    print(f"{a.kind} gate: " + ("PASS" if not problems else "FAIL"))
    print(json.dumps(result, indent=1, default=str))
    for p in problems:
        print(f"  - {p}")
    return 1 if problems else 0


def cmd_ledger(a: argparse.Namespace) -> int:
    root = Path(a.root)
    boots = on_boots = plans = swaps = 0
    hours = 0.0
    bad: list[str] = []
    for path in sorted(root.rglob("ABORT")):
        bad.append(f"abort latch {path.relative_to(root)}: {path.read_text().strip()[:160]}")
    for serve in sorted(root.rglob("serve.log")):
        d = serve.parent
        if "cache" in d.parts:
            continue
        text = serve.read_text(errors="replace")
        boots += 1
        guard = d / "guard.json"
        if guard.is_file():
            g = json.loads(guard.read_text())
            if g.get("ok") is not True or g.get("faults") != 0:
                bad.append(f"{d.relative_to(root)}: guard ok={g.get('ok')} faults={g.get('faults')} {str(g.get('note'))[:120]}")
        try:
            faults = qb.fault_lines(serve)
        except qb.QualificationError:
            faults = []
        if faults:
            bad.append(f"{d.relative_to(root)}: {len(faults)} fault line(s): {faults[0].strip()[:160]}")
        if qb.EPLB_ON not in text:
            continue
        on_boots += 1
        year = year_of(serve)
        lines = text.splitlines()
        first = next((t for t in (epoch(x, year) for x in lines) if t), None)
        last = next((t for t in (epoch(x, year) for x in reversed(lines)) if t), None)
        if first is not None and last is not None:
            hours += (last - first) / 3600.0
        for _n, (_s, e, k) in plan_walls(text, year).items():
            if e is not None:
                plans += 1
                swaps += k
    verdict = "RECURRED" if bad else "NO RECURRENCE"
    print(f"ledger: {verdict} in {on_boots} EPLB boot(s) of {boots}, ~{hours:.1f} h, {plans} committed plan(s), {swaps} swap(s)")
    print("ledger: the PCIe completion-timeout/DPC event of the first chaos run is NOT explained by this count")
    for b in bad:
        print(f"  - {b}")
    if a.out:
        Path(a.out).write_text(json.dumps({"verdict": verdict, "boots": boots, "eplb_boots": on_boots,
                                           "hours": round(hours, 2), "plans": plans, "swaps": swaps,
                                           "problems": bad}, indent=1) + "\n")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inject-boot")
    p.add_argument("--dir", required=True)
    p.add_argument("--arm", choices=["off", "chaos"], required=True)
    p.add_argument("--decoders", type=int, required=True)
    p.add_argument("--min-steps", type=int, default=10)
    p.add_argument("--min-overlap", type=int, default=1)
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_inject_boot)
    p = sub.add_parser("inject-pair")
    p.add_argument("--off", required=True)
    p.add_argument("--on", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--gap-tol", dest="gap_tol", type=float, default=0.10)
    p.add_argument("--reverse-tol", dest="reverse_tol", type=float, default=0.10)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_inject_pair)
    p = sub.add_parser("ppl")
    p.add_argument("--root", required=True)
    p.add_argument("--min-plans", type=int, default=4)
    p.add_argument("--max-ppl-rel", type=float, default=0.02)
    p.add_argument("--fixture", action="store_true")
    p.set_defaults(fn=cmd_ppl)
    p = sub.add_parser("traffic")
    p.add_argument("--dir", required=True)
    p.add_argument("--kind", choices=["stationary", "soak"], required=True)
    p.add_argument("--minutes", type=int, required=True)
    p.add_argument("--drift", type=float, default=0.05)
    p.add_argument("--steps-per-plan", dest="steps_per_plan", type=int, default=10000)
    p.add_argument("--warmup-frac", dest="warmup_frac", type=float, default=1.0 / 3.0)
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_traffic)
    p = sub.add_parser("ledger")
    p.add_argument("--root", required=True)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_ledger)
    a = ap.parse_args()
    return a.fn(a)


sys.exit(main())
PY
}

# ------------------------------------------------------------------- hooks
# serve_validate.sh runs a LOAD_HOOK with URL PORT SERVED MODEL OUT exported and
# needs a final PASS/FAIL line.  The hooks are written into the run directory so
# the chain is one file.
write_hooks() {
    mkdir -p "$RUN/hooks"
    cat > "$RUN/hooks/traffic.sh" << 'HOOK'
#!/usr/bin/env bash
# bench cells in a loop for HOOK_MINUTES, then the post-load probes
set -uo pipefail
: "${URL:?}" "${PORT:?}" "${SERVED:?}" "${MODEL:?}" "${OUT:?}" "${HOOK_CELLS:?}" "${HOOK_MINUTES:?}"
: "${RECEIPT_PY:?}" "${RECEIPT_TOOLS:?}"
say() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$OUT/traffic"
end=$(($(date +%s) + HOOK_MINUTES * 60))
date +%s > "$OUT/traffic-start.epoch"
i=0
bad=0
say "traffic: cells '$HOOK_CELLS' for $HOOK_MINUTES min"
while (($(date +%s) < end)); do
    for cell in $HOOK_CELLS; do
        (($(date +%s) < end)) || break
        IFS=: read -r in out conc n <<< "$cell"
        i=$((i + 1))
        t0=$(date +%s)
        "$RECEIPT_PY" -m vllm.entrypoints.cli.main bench serve --backend openai \
            --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" --model "$SERVED" \
            --tokenizer "$MODEL" --dataset-name random --random-input-len "$in" \
            --random-output-len "$out" --num-prompts "$n" --max-concurrency "$conc" \
            --ignore-eos --request-rate inf --seed $((${HOOK_SEED:-7000} + i)) --temperature 0 \
            --save-result --result-dir "$OUT/traffic" \
            --result-filename "cell-$i-${in}x${out}-c${conc}.json" \
            > "$OUT/traffic/cell-$i.log" 2>&1
        rc=$?
        say "cell $i ${in}x${out} c=$conc n=$n rc=$rc in $(($(date +%s) - t0)) s"
        ((rc == 0)) || bad=1
        kill -0 "$(cat "$OUT/serve.pid" 2> /dev/null)" 2> /dev/null || { say "server gone"; bad=1; break 2; }
    done
done
date +%s > "$OUT/traffic-end.epoch"
if "$RECEIPT_PY" "$RECEIPT_TOOLS/tools/rdna/port_v031/prefix_probe.py" --url "$URL" \
    --model "$SERVED" --words "${PREFIX_WORDS:-1800}" > "$OUT/prefix-after.txt" 2>&1; then
    say "post-load prefix probe PASS"
else
    say "post-load prefix probe FAIL"
    bad=1
fi
if "$RECEIPT_PY" "$RECEIPT_TOOLS/tools/rdna/port_v031/ppl_probe.py" --url "$URL" \
    --model "$SERVED" --max-ppl "${MAX_PPL:-60}" > "$OUT/ppl-after.txt" 2>&1; then
    say "post-load PPL probe PASS"
else
    say "post-load PPL probe FAIL"
    bad=1
fi
if ((bad == 0)); then
    echo "PASS gates traffic hook ($i cells)"
else
    echo "FAIL gates traffic hook"
fi
exit "$bad"
HOOK
    cat > "$RUN/hooks/quality.sh" << 'HOOK'
#!/usr/bin/env bash
# decode traffic until the chaos boot has committed PPL_NEED_PLANS plans (0 on an off boot: none),
# then the quality collects of pr_receipt_hook.sh
set -uo pipefail
: "${URL:?}" "${PORT:?}" "${SERVED:?}" "${MODEL:?}" "${OUT:?}" "${RECEIPT_PY:?}" "${RECEIPT_TOOLS:?}"
mkdir -p "$OUT/traffic"
need=${PPL_NEED_PLANS:-0}
plans_done() { grep -a -c -E "Worker_TP0_EP0.*Online EPLB: plan [0-9]+ done" "$OUT/serve.log"; }
waves=0
limit=$((SECONDS + ${PPL_PRETRAFFIC_MAX_S:-1500}))
while (($(plans_done) < need && SECONDS < limit)); do
    waves=$((waves + 1))
    "$RECEIPT_PY" -m vllm.entrypoints.cli.main bench serve --backend openai \
        --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" --model "$SERVED" \
        --tokenizer "$MODEL" --dataset-name random --random-input-len 1024 --random-output-len 512 \
        --num-prompts 16 --max-concurrency 8 --ignore-eos --request-rate inf \
        --seed $((4242 + waves)) --temperature 0 > "$OUT/traffic/pretraffic-$waves.log" 2>&1
done
echo "[$(date +%H:%M:%S)] pretraffic: $waves wave(s), $(plans_done) of $need plan(s) committed"
date +%s > "$OUT/collect-start.epoch"
exec bash "$RECEIPT_TOOLS/tools/rdna/pr_receipt_hook.sh"
HOOK
    chmod +x "$RUN/hooks/traffic.sh" "$RUN/hooks/quality.sh"
}

# -------------------------------------------------------------------- arms
# The EPLB KEY=value words of one boot.  EXTRA_ARGS is an environment variable of run_arm, never a
# trailing KEY=value: serve_validate.sh appends --enable-logging-iteration-details to the
# environment's EXTRA_ARGS when STALL_PROBE=1, and a trailing EXTRA_ARGS=... would override it.
arm_args() {
    case $1 in
        off) echo "VLLM_RDNA_EPLB=0" ;;
        chaos) echo "VLLM_RDNA_EPLB=1 VLLM_RDNA_EPLB_CHAOS=8 VLLM_RDNA_EPLB_DECISION_STEPS=512" ;;
        planner) echo "VLLM_RDNA_EPLB=1" ;;
        *) die "arm kind $1" ;;
    esac
}

# One fresh boot through qualify_final.sh's guard wrapper.  The per-arm exports stay in
# the subshell.  run_arm DIR TAG KIND CELLS STALL_ARGS LOAD_HOOK [VAR=value ...]
run_arm() {
    local dir=$1 tag=$2 kind=$3 cells=$4 stall=$5 hook=$6 rc=0 kv
    shift 6
    local -a flags=()
    read -r -a flags <<< "$(arm_args "$kind")"
    [[ ! -e $RUN/ABORT ]] || { note "abort latched: $(tail -n 1 "$RUN/ABORT"); not booting $tag"; return 3; }
    mkdir -p "$dir" "$RUN/cache/triton" "$RUN/cache/inductor" "$RUN/logs"
    note "boot $tag: $kind cells=${cells:-none} stall=${stall:+yes}${stall:-no} hook=${hook:+yes}"
    (
        for kv in $(compgen -e | grep '^VLLM_RDNA_EPLB' || true); do unset "$kv"; done
        unset VLLM_RDNA2_MOE_DETERMINISTIC ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES PYTHONPATH EAGER
        unset QUALITY_WINDOWS QUALITY_LIMIT QUALITY_TOP QUALITY_PASSAGES LOAD_HOOK STALL_ARGS STALL_ITER_LOG
        export OUT=$dir TAG="gates-$STAMP-$tag" SERVE_TREE=$HEAD_TREE VENV MODEL RECIPE=$RECIPE_FIXED
        export GPUS=$GPUS_FIXED HIP_VISIBLE_DEVICES=$GPUS_FIXED REFUSED_GPUS=0,1 PORT=$GATE_PORT
        export CELLS="${cells:-none}" BENCH_EXTRA='--temperature 0' CONC_PROBE=1 RESET_PREFIX_CACHE=0
        export READY_TIMEOUT=2400 STOP_GRACE=120 QUIET_OWN=EplbGates QUIET_WIN_MAX=240
        export BOOT_LOCK QUIET_DIR
        export TRITON_CACHE_DIR=$RUN/cache/triton TORCHINDUCTOR_CACHE_DIR=$RUN/cache/inductor
        export QUALIFY_WRAPPER=1 QUALIFY_CAMPAIGN_OUT=$RUN QUALIFY_ABORT=$RUN/ABORT
        export QUALIFY_PYTHON QUALIFY_POLL EXTRA_ARGS=--monokernel
        export RECEIPT_PY=$VENV/bin/python RECEIPT_TOOLS=$TOOLS PREFIX_WORDS
        if [[ -n $stall ]]; then
            export STALL_PROBE=1 "STALL_ARGS=$stall" QUIET_HOLD=1
        else
            export STALL_PROBE=0 QUIET_HOLD=0
        fi
        [[ -z $hook ]] || export LOAD_HOOK=$hook
        for kv in "$@"; do export "${kv?}"; done
        bash "$QUALIFY_FINAL" "${flags[@]}"
    ) > "$RUN/logs/$tag.log" 2>&1
    rc=$?
    note "boot $tag exited $rc (log $RUN/logs/$tag.log)"
    return "$rc"
}

# ------------------------------------------------------------------ stages
stage_build() {
    local role tree var
    for role in BASE HEAD; do
        var=${role}_TREE
        tree=${!var}
        if compgen -G "$tree/vllm/_rocm_C*.so" > /dev/null; then
            note "build: $role $tree already has $(basename "$(compgen -G "$tree/vllm/_rocm_C*.so" | head -n 1)")"
            continue
        fi
        if ((BUILD == 0)); then
            echo "build: $role tree $tree has no vllm/_rocm_C*.so; rerun with BUILD=1" >&2
            return 2
        fi
        ((DRY_RUN == 0)) || { note "build (dry run): would build $tree"; continue; }
        [[ -d $BUILD_SEED/.deps/triton_kernels-src && -e $BUILD_SEED/vllm/vllm-rs ]] \
            || { echo "build: BUILD_SEED=$BUILD_SEED has no .deps/triton_kernels-src and vllm/vllm-rs" >&2; return 2; }
        build_tree "$tree" || return 1
    done
    note "build: BASE $(tree_sha "$BASE_TREE") _rocm_C $(rocm_md5 "$BASE_TREE")"
    note "build: HEAD $(tree_sha "$HEAD_TREE") _rocm_C $(rocm_md5 "$HEAD_TREE")"
    return 0
}

tree_sha() {
    local sha=""
    [[ -f $1/.sha ]] && sha=$(tr -d '[:space:]' < "$1/.sha")
    echo "${sha:-unknown}"
}

rocm_md5() {
    local f
    f=$(compgen -G "$1/vllm/_rocm_C*.so" | head -n 1)
    [[ -n $f ]] && md5sum "$f" | cut -d' ' -f1 || echo none
}

# CPU-only in-place build of one source tree, the recipe of the Inkling-2 receipt trees on this
# box (setup.py build_ext --inplace, MAX_JOBS<=8, nice 19, build.lock).  .deps/triton_kernels-src,
# the precompiled vllm/vllm-rs and vllm/_version.py come from the donor tree BUILD_SEED (never the
# -build/-subbuild directories).
build_tree() {
    local tree=$1 s rc=0 bt
    bt=$RUN/build/$(basename "$tree")
    s=$VENV/lib/python3.12/site-packages
    mkdir -p "$bt" "$RUN/tmp"
    (
        unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES PYTHONPATH
        unset VLLM_USE_PRECOMPILED VLLM_USE_PRECOMPILED_RUST VLLM_PRECOMPILED_WHEEL_LOCATION
        export HIP_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1030
        export SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION:-0.34.0+rdna}
        export VLLM_PYTHON_EXECUTABLE=$VENV/bin/python MAX_JOBS=${MAX_JOBS:-8} CMAKE_BUILD_TYPE=RelWithDebInfo
        ((MAX_JOBS <= 8)) || { echo "refusing MAX_JOBS=$MAX_JOBS (> 8): box rule"; exit 2; }
        export ROCM_HOME=/opt/rocm/core-7.14 ROCM_PATH=/opt/rocm/core-7.14 HIP_PATH=/opt/rocm/core-7.14
        export HIP_ROOT_DIR=/opt/rocm/core-7.14 CMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-7.14
        export CMAKE_HIP_COMPILER=/opt/rocm/core-7.14/bin/hipcc
        export HIP_DEVICE_LIB_PATH=/opt/rocm/core-7.14/lib/llvm/amdgcn/bitcode
        export TRITON_KERNELS_SRC_DIR=$tree/.deps/triton_kernels-src/python/triton_kernels/triton_kernels
        export PATH=$VENV/bin:/opt/rocm/core-7.14/bin:$PATH TMPDIR=$RUN/tmp
        export LD_LIBRARY_PATH="$s/_rocm_sdk_libraries/lib:$s/_rocm_sdk_core/lib/host-math/lib:$s/_rocm_sdk_core/lib/rocm_sysdeps/lib:$s/_rocm_sdk_core/lib:$s/torch/lib"
        cd "$tree" || exit 1
        mkdir -p .deps
        [[ -d .deps/triton_kernels-src ]] || cp -a "$BUILD_SEED/.deps/triton_kernels-src" .deps/
        [[ -e vllm/vllm-rs ]] || ln "$BUILD_SEED/vllm/vllm-rs" vllm/vllm-rs 2> /dev/null || cp -p "$BUILD_SEED/vllm/vllm-rs" vllm/
        [[ -f vllm/_version.py ]] || cp -p "$BUILD_SEED/vllm/_version.py" vllm/
        exec 9> "$HOME/w4a8_runs/audit/build.lock"
        echo "[$(date +%T)] waiting for build.lock"
        flock 9
        echo "[$(date +%T)] build starts: $tree sha $(tree_sha "$tree")"
        nice -n 19 "$VENV/bin/python" setup.py build_ext --inplace --build-temp "$bt"
    ) >> "$RUN/logs/build-$(basename "$tree").log" 2>&1
    rc=$?
    note "build: $tree exit $rc (log $RUN/logs/build-$(basename "$tree").log)"
    return "$rc"
}

# The mapped K2 cap is 8 rows and an MTP2 decoder is 3 rows: 2 decoders (6 rows, capture size 6)
# stay on K2 between prefill chunks, 7 decoders (21 rows, capture size 24) never do.  Order: the
# new K2 pair first, then the historical modules pair.
stage_inject() {
    local dec2 dec7 spec boot decoders kind stall failed=0 rc=0 off_boot on_boot label
    dec2=$((K2_ROW_CAP / ROWS_PER_DECODER))
    dec7=$((MAX_SEQS - 1))
    ((dec2 * ROWS_PER_DECODER <= K2_ROW_CAP && dec7 * ROWS_PER_DECODER > K2_ROW_CAP)) \
        || { echo "inject: $dec2 decoders must fit the K2 cap and $dec7 must not" >&2; return 2; }
    for spec in "J2off:$dec2:off" "J2on:$dec2:chaos" "J7off:$dec7:off" "J7on:$dec7:chaos"; do
        IFS=: read -r boot decoders kind <<< "$spec"
        stall="--scenarios solo,inject,reverse --decoders $decoders --prefill-lens $INJECT_LENS --repeats $INJECT_REPEATS --max-starved 0 --min-decode-tokens-during-prefill 1 --gate-stat worst"
        if ((DRY_RUN)); then
            note "inject $boot: $kind, $decoders decoders ($((decoders * ROWS_PER_DECODER)) rows), STALL_ARGS=$stall"
            continue
        fi
        run_arm "$RUN/inject/boots/$boot" "inject-$boot" "$kind" none "$stall" "" || { failed=1; break; }
        judge inject-boot --dir "$RUN/inject/boots/$boot" --arm "$kind" --decoders "$decoders" \
            --min-steps "$INJECT_MIN_STEPS" --min-overlap "$INJECT_MIN_OVERLAP" \
            --out "$RUN/inject/boots/$boot/evidence.json" || { failed=1; break; }
    done
    ((DRY_RUN == 0)) || return 0
    ((failed == 0)) || return 1
    for spec in "J2:J2off:J2on" "J7:J7off:J7on"; do
        IFS=: read -r label off_boot on_boot <<< "$spec"
        judge inject-pair --off "$RUN/inject/boots/$off_boot" --on "$RUN/inject/boots/$on_boot" \
            --label "$label" --gap-tol "$GAP_TOL" --reverse-tol "$REVERSE_TOL" \
            --out "$RUN/inject/pair-$label.json" || rc=1
    done
    return "$rc"
}

stage_ppl() {
    local d=$RUN/ppl spec boot kind need rc=0 py=$VENV/bin/python s
    if ((DRY_RUN)); then
        note "ppl: P1 off, P2 chaos (decode traffic until $PPL_MIN_PLANS plans committed, cap ${PPL_PRETRAFFIC_MAX_S}s), P3 off; $QUALITY_LIMIT windows, top $QUALITY_TOP, $QUALITY_PASSAGES passages; PPL rise <= $MAX_PPL_REL"
        return 0
    fi
    mkdir -p "$d"
    s=$VENV/lib/python3.12/site-packages
    find "$TOOLS/docs" -name '*.md' | sort | xargs cat > "$d/docs.txt"
    LD_LIBRARY_PATH="$s/_rocm_sdk_libraries/lib:$s/_rocm_sdk_core/lib/host-math/lib:$s/_rocm_sdk_core/lib/rocm_sysdeps/lib:$s/_rocm_sdk_core/lib:$s/torch/lib" \
        HIP_VISIBLE_DEVICES='' "$py" "$PORT_DIR/topk_kl.py" windows --tokenizer "$MODEL" \
        --domain "docs=$d/docs.txt@0" --count "$QUALITY_LIMIT" --seq-len 2048 --out "$d/windows.json" \
        > "$d/windows.log" 2>&1 || { echo "ppl: cannot cut the quality windows (see $d/windows.log)" >&2; return 1; }
    for spec in "P1:off" "P2:chaos" "P3:off"; do
        IFS=: read -r boot kind <<< "$spec"
        need=0
        [[ $kind != chaos ]] || need=$PPL_MIN_PLANS
        run_arm "$d/boots/$boot" "ppl-$boot" "$kind" none "" "$RUN/hooks/quality.sh" \
            "QUALITY_WINDOWS=$d/windows.json" "QUALITY_LIMIT=$QUALITY_LIMIT" "QUALITY_TOP=$QUALITY_TOP" \
            "QUALITY_PASSAGES=$QUALITY_PASSAGES" "PPL_NEED_PLANS=$need" \
            "PPL_PRETRAFFIC_MAX_S=$PPL_PRETRAFFIC_MAX_S" || return 1
    done
    "$py" "$PORT_DIR/topk_kl.py" compare --pair "floor=$d/boots/P1/topk.npz,$d/boots/P3/topk.npz" \
        --pair "head=$d/boots/P1/topk.npz,$d/boots/P2/topk.npz" --json "$d/kl.json" --md "$d/kl.md" \
        > "$d/kl.log" 2>&1 || { echo "ppl: topk_kl compare failed (see $d/kl.log)" >&2; return 1; }
    "$py" "$PORT_DIR/quality_ab.py" compare "$d/boots/P1/quality.json" "$d/boots/P2/quality.json" \
        --max-ppl-rel "$MAX_PPL_REL" > "$d/ppl-head.txt" 2>&1 || true
    "$py" "$PORT_DIR/quality_ab.py" compare "$d/boots/P1/quality.json" "$d/boots/P3/quality.json" \
        --max-ppl-rel "$MAX_PPL_REL" > "$d/ppl-floor.txt" 2>&1 || true
    judge ppl --root "$d" --min-plans "$PPL_MIN_PLANS" --max-ppl-rel "$MAX_PPL_REL" || rc=1
    return "$rc"
}

stage_r2() {
    local rc=0
    if ((DRY_RUN)); then
        MODE=r2 DRY_RUN=1 BASE_TREE=$BASE_TREE HEAD_TREE=$HEAD_TREE VENV=$VENV MODEL=$MODEL \
            OUT=$RUN/r2-dry bash "$QUALIFY_FINAL" || return 1
        return 0
    fi
    MODE=r2 BASE_TREE=$BASE_TREE HEAD_TREE=$HEAD_TREE VENV=$VENV MODEL=$MODEL OUT=$RUN/r2 \
        QUALIFY_POLL=$QUALIFY_POLL bash "$QUALIFY_FINAL"
    rc=$?
    return "$rc"
}

stage_r1() {
    local rc=0 labels
    if ((DRY_RUN)); then
        MODE=r1 DRY_RUN=1 BASE_TREE=$BASE_TREE HEAD_TREE=$HEAD_TREE VENV=$VENV MODEL=$MODEL \
            OUT=$RUN/r1-dry bash "$QUALIFY_FINAL" || return 1
        return 0
    fi
    MODE=r1 BASE_TREE=$BASE_TREE HEAD_TREE=$HEAD_TREE VENV=$VENV MODEL=$MODEL OUT=$RUN/r1 \
        QUALIFY_POLL=$QUALIFY_POLL bash "$QUALIFY_FINAL"
    rc=$?
    ((rc == 0)) || return "$rc"
    labels=$(cat "$RUN/r1/speed-labels.txt" 2> /dev/null || echo missing)
    if [[ $labels != "0 REGRESSION labels" ]]; then
        echo "r1: the native speed verdicts are not clean (hard gate of this chain):" >&2
        echo "    ${labels//$'\n'/$'\n'    }" >&2
        return 1
    fi
    note "r1: 0 REGRESSION labels"
    return 0
}

# stationary: the planner alone (no chaos) on one fixed cell; the boot logs every engine iteration so the
# judge can count steps.  soak: chaos on the soak1 mix; no iteration log (4 h of lines nothing reads).
stage_traffic() { # stationary|soak
    local kind=$1 minutes cells arm_kind d=$RUN/$1/boots/S seed
    local -a extra=()
    if [[ $kind == stationary ]]; then
        minutes=$STATIONARY_MINUTES
        cells="1024:512:8:32"
        arm_kind=planner
        seed=7000
        extra=("EXTRA_ARGS=--monokernel --enable-logging-iteration-details")
    else
        minutes=$SOAK_MINUTES
        cells="1024:512:8:32 16384:1024:8:32"
        arm_kind=chaos
        seed=9000
    fi
    if ((DRY_RUN)); then
        note "$kind: $arm_kind boot, cells '$cells' for $minutes min, then prefix/PPL/greedy probes"
        return 0
    fi
    run_arm "$d" "$kind" "$arm_kind" none "" "$RUN/hooks/traffic.sh" \
        "HOOK_CELLS=$cells" "HOOK_MINUTES=$minutes" "HOOK_SEED=$seed" "${extra[@]}" || return 1
    if [[ $kind == stationary ]]; then
        judge traffic --dir "$d" --kind "$kind" --minutes "$minutes" --drift "$STATIONARY_ITL_DRIFT" \
            --steps-per-plan "$STATIONARY_STEPS_PER_PLAN" --out "$d/evidence.json"
    else
        judge traffic --dir "$d" --kind "$kind" --minutes "$minutes" --out "$d/evidence.json"
    fi
}

stage_stationary() { stage_traffic stationary; }
stage_soak() { stage_traffic soak; }

# ------------------------------------------------------------------- driver
SUMMARY_ROWS=()

run_stage() { # gate
    local gate=$1 t0 rc log=$RUN/$1.log follow verdict
    mkdir -p "$RUN/$gate"
    : > "$log"
    note "gate $gate: $(gate_label "$gate")"
    t0=$(date +%s)
    tail -n +1 -f "$log" &
    follow=$!
    "stage_$gate" >> "$log" 2>&1
    rc=$?
    sleep 0.5
    kill "$follow" 2> /dev/null
    wait "$follow" 2> /dev/null
    case $rc in
        0) verdict=PASS ;;
        1) verdict=FAIL ;;
        2) verdict=ERROR ;;
        *) verdict=ABORT ;;
    esac
    [[ ! -e $RUN/ABORT ]] || verdict=ABORT
    ((DRY_RUN == 0 || rc != 0)) || verdict=PLAN-OK
    SUMMARY_ROWS+=("$gate|$verdict|$(($(date +%s) - t0))|$(gate_boots "$gate")")
    note "gate $gate: $verdict in $((($(date +%s) - t0) / 60)) min"
    [[ $verdict == PASS || $verdict == PLAN-OK ]]
}

summarize() {
    local row g v s b pass=0 ran="" missing="" complete=1 verdict
    {
        echo "# EPLB v0.34 gate chain"
        echo
        echo "head: $HEAD_TREE ($(tree_sha "$HEAD_TREE")) _rocm_C $(rocm_md5 "$HEAD_TREE")"
        echo "base: $BASE_TREE ($(tree_sha "$BASE_TREE")) _rocm_C $(rocm_md5 "$BASE_TREE")"
        echo
        echo "| gate | verdict | boots | minutes | open item |"
        echo "|---|---|---|---|---|"
        for row in ${SUMMARY_ROWS[@]+"${SUMMARY_ROWS[@]}"}; do
            IFS='|' read -r g v s b <<< "$row"
            echo "| $g | $v | $b | $((s / 60)) | $(gate_label "$g") |"
            ran="$ran $g"
            [[ $v == PASS ]] && pass=$((pass + 1))
        done
    } > "$RUN/gates-summary.md"
    if ((DRY_RUN)); then
        {
            echo
            echo "DRY RUN: inputs and launch plans were checked; no boot ran and no gate was evaluated."
            echo "This is NOT a qualification."
        } >> "$RUN/gates-summary.md"
        cat "$RUN/gates-summary.md"
        return 0
    fi
    for g in $GATES_ALL; do
        [[ " $ran " == *" $g "* ]] || { missing="$missing $g"; complete=0; }
    done
    if ((pass == ${#SUMMARY_ROWS[@]} && ${#SUMMARY_ROWS[@]} > 0)); then verdict=PASS; else verdict=NOT-QUALIFIED; fi
    {
        echo
        echo "chain: $verdict ($pass of ${#SUMMARY_ROWS[@]} gates ran passed)"
        if ((complete)); then
            echo "coverage: COMPLETE, every open gate of the chain ran"
        else
            echo "coverage: PARTIAL, not run:$missing"
        fi
    } >> "$RUN/gates-summary.md"
    cat "$RUN/gates-summary.md"
}

preflight() {
    : "${BASE_TREE:?BASE_TREE required}" "${HEAD_TREE:?HEAD_TREE required}"
    : "${VENV:?VENV required}" "${MODEL:?MODEL required}" "${OUT:?OUT required}"
    [[ -d $BASE_TREE && -d $HEAD_TREE ]] || die "BASE_TREE and HEAD_TREE must be directories"
    BASE_TREE=$(cd "$BASE_TREE" && pwd)
    HEAD_TREE=$(cd "$HEAD_TREE" && pwd)
    [[ -z ${GPUS:-} || $GPUS == "$GPUS_FIXED" ]] || die "GPUS is fixed to $GPUS_FIXED, not $GPUS"
    [[ -z ${RECIPE:-} || $RECIPE == "$RECIPE_FIXED" ]] || die "RECIPE is fixed to $RECIPE_FIXED"
    export GPUS=$GPUS_FIXED REFUSED_GPUS=0,1
    BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
    QUIET_DIR=${QUIET_DIR:-$HOME/w4a8_runs/lanes/quiet-b4}
    if [[ -e $OUT ]]; then
        [[ -d $OUT && -z $(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit) ]] \
            || die "OUT=$OUT must be a fresh run directory"
    fi
    mkdir -p "$OUT" || die "cannot create OUT=$OUT"
    RUN=$(cd "$OUT" && pwd)
    STAMP=$(date -u +%Y%m%d-%H%M%S)
    export BASE_TREE HEAD_TREE VENV MODEL BOOT_LOCK QUIET_DIR
    local v
    for v in $(compgen -e | grep '^VLLM_RDNA_EPLB' || true); do unset "$v"; done
    unset VLLM_RDNA2_MOE_DETERMINISTIC
    write_hooks
}

# qualify_final.sh's own input checks (trees, shas, distinct _rocm_C, recipe, GPUS/REFUSED_GPUS)
# and the head launch plan, with no GPU.  MAX_SEQS comes from that plan.
preflight_inputs() {
    note "preflight: qualify_final.sh MODE=r2 DRY_RUN=1 (no GPU)"
    MODE=r2 DRY_RUN=1 BASE_TREE=$BASE_TREE HEAD_TREE=$HEAD_TREE VENV=$VENV MODEL=$MODEL \
        OUT=$RUN/preflight bash "$QUALIFY_FINAL" > "$RUN/preflight.log" 2>&1 \
        || { tail -n 20 "$RUN/preflight.log" >&2; die "preflight failed (see $RUN/preflight.log)"; }
    MAX_SEQS=$("$QUALIFY_PYTHON" "$REPORT" plan-field "$RUN/preflight/plans/head.txt" max-num-seqs 2> /dev/null)
    if ! is_uint "$MAX_SEQS" || ((MAX_SEQS < 4)); then
        die "no usable --max-num-seqs in $RUN/preflight/plans/head.txt (got '$MAX_SEQS')"
    fi
    note "preflight: ok; head --max-num-seqs $MAX_SEQS"
    {
        echo "gates=$GATES"
        echo "base_sha=$(tree_sha "$BASE_TREE") base_rocm_c=$(rocm_md5 "$BASE_TREE")"
        echo "head_sha=$(tree_sha "$HEAD_TREE") head_rocm_c=$(rocm_md5 "$HEAD_TREE")"
        echo "script_sha256=$(sha256sum "$SELF" | cut -d' ' -f1)"
        echo "model=$MODEL venv=$VENV"
        echo "inject_lens=$INJECT_LENS repeats=$INJECT_REPEATS gap_tol=$GAP_TOL reverse_tol=$REVERSE_TOL"
        echo "ppl limit=$QUALITY_LIMIT passages=$QUALITY_PASSAGES max_rel=$MAX_PPL_REL min_plans=$PPL_MIN_PLANS"
        echo "stationary_minutes=$STATIONARY_MINUTES drift=$STATIONARY_ITL_DRIFT steps_per_plan=$STATIONARY_STEPS_PER_PLAN soak_minutes=$SOAK_MINUTES"
    } > "$RUN/manifest.txt"
}

dry_plans() { # the launch plan of each arm kind, no GPU
    local spec name kind extra
    for spec in "off:--monokernel" "chaos:--monokernel" "planner:--monokernel" \
        "chaos-iterlog:--monokernel --enable-logging-iteration-details"; do
        IFS=: read -r name extra <<< "$spec"
        kind=${name%-iterlog}
        local -a flags=()
        read -r -a flags <<< "$(arm_args "$kind")"
        note "launch plan ($name): EXTRA_ARGS='$extra' serve_rdna.sh PRINT=1 ${flags[*]}"
        EXTRA_ARGS=$extra HIP_VISIBLE_DEVICES='' PRINT=1 bash "$HEAD_TREE/tools/rdna/serve_rdna.sh" \
            RECIPE="$RECIPE_FIXED" MODEL="$MODEL" VENV="$VENV" VLLM_TREE="$HEAD_TREE" PORT="$GATE_PORT" \
            HOST=127.0.0.1 "${flags[@]}" > "$RUN/preflight/plan-$name.txt" 2>&1 \
            || { tail -n 8 "$RUN/preflight/plan-$name.txt" >&2; die "launch plan $name failed"; }
    done
    note "launch plans written to $RUN/preflight/plan-*.txt"
}

take_lease() {
    local total=30 g
    for g in $GATES; do
        case $g in build) ;; *) total=$((total + $(gate_eta "$g"))) ;; esac
    done
    [[ -r $LEASE_SH ]] || die "no lease helper at $LEASE_SH (set LEASE_HELD=1 if you hold b4 yourself)"
    local saved_model=$MODEL
    export LEASE=b4 OWNER
    # shellcheck source=/dev/null
    source "$LEASE_SH"
    MODEL=$saved_model
    trap 'note "interrupted"; exit 130' INT TERM
    lease_take "EPLB v0.34 open gates: $GATES" "$total"
    lease_check_buses
}

# Every EPLB boot of this run, pooled: the bounded evidence for (c).  Never a root cause.
stage_ledger() {
    local rc
    judge ledger --root "$RUN" --out "$RUN/ledger.json" | tee "$RUN/ledger.txt"
    rc=${PIPESTATUS[0]}
    return "$rc"
}

main() {
    local g failed=0
    defaults
    if ((LIST)); then
        list_gates
        exit 0
    fi
    preflight
    # the CPU build comes first, before the lease
    if [[ " $GATES " == *" build "* ]]; then
        run_stage build || failed=1
    fi
    ((failed == 0)) || { summarize; exit 1; }
    preflight_inputs
    ((DRY_RUN == 0)) || dry_plans
    if ((DRY_RUN == 0 && LEASE_HELD == 0)); then
        take_lease
    fi
    for g in $GATES; do
        [[ $g != build ]] || continue
        ((failed == 0)) || { [[ $CONTINUE == 1 && ! -e $RUN/ABORT ]] || break; }
        run_stage "$g" || failed=1
        [[ ! -e $RUN/ABORT ]] || break
    done
    if ((DRY_RUN == 0)) && ((${#SUMMARY_ROWS[@]})); then
        run_stage ledger || failed=1
    fi
    summarize
    [[ ! -e $RUN/ABORT ]] || exit 3
    ((failed == 0)) || exit 1
    exit 0
}

# `source` the file for the judge and helpers only (RUN_GATES_SOURCE_ONLY=1); run it for the chain.
if [[ ${RUN_GATES_SOURCE_ONLY:-0} != 1 ]]; then
    main "$@"
fi
