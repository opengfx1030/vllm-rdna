#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Windows -> plans -> held-out simulation of the T10 collection (arm B1 of
# tools/rdna/hetero_moe/t10_run.sh), constants of the hetero-MoE spec section 7.2
# (origin/feat/hetero-moe 30e22d0b1a). Run from the repo root with
# RAW=<expert_stats-dp0-pid*.jsonl of B1 rank 0> (no .pairs.npz next to it: windows
# cannot be cut out of the cumulative pair counts), PY=<python with numpy>.
#
# Windows are whole agentic passes of the load hook (elapsed_s of the collector):
#   W1 = passes 1-3 (hook start .. pass 4 start), W2 = passes 4-7 (.. end of load).
# Every plan is solved on one window and simulated on both (the other is held out).
set -euo pipefail
: "${RAW:?}" "${PY:?}"
D=${D:-bench_results/2026-11-07_expert-stats-t10}
SHA=eeec2b0d38a40a6443ad0ff30104fdef104cef8e # tree of the collection (arm B1)
export PYTHONPATH=.

for w in W1 W2; do
    case $w in
        W1) t0=843.19 t1=2694.316 ;;
        W2) t0=2694.316 t1=4998.605 ;;
    esac
    "$PY" -m tools.rdna.hetero_moe.stats_io convert "$RAW" --out "$D/$w.stats.json" \
        --t0 "$t0" --t1 "$t1" --engine-commit "$SHA" \
        --workload "agentic_bench hi2 suite behind cline_act, seeds 1-7, 8 workers, flashnext-mtp2 TP4 EP4"
done

COMMON=(--hub-vram-gib 44.98 --hub-reserve-gib 12.1 --expert-budget-gib 26
    --mtp-policy hub --mtp-expert-bytes 2534400 --hub-us 3.7 --expert-us 6.7
    --expert-fixed-us 3.1 --latency-us 13.3 --gbps 25)
H1=(--islands 1 --hub-hips 0 --expert-hips '1,2,3,4' --switches c0
    --island-mode replicate --hub-active-gib 9.4 --max-hub-per-layer 130
    --hub-fixed-us 30.8)
H2=(--islands 2 --hub-hips '0;5' --expert-hips '1,2,3,4;6,7,8,9' --switches 'c0,00'
    --island-mode partition --hub-active-gib 5.8 --max-hub-per-layer 115
    --hub-fixed-us 20.7)

for w in W1 W2; do
    other=$([[ $w == W1 ]] && echo W2 || echo W1)
    for h in H1 H2; do
        case $h in
            H1) opts=("${H1[@]}") fixed=30.8 ;;
            H2) opts=("${H2[@]}") fixed=20.7 ;;
        esac
        plan=$D/$h.$w.plan.json
        "$PY" -m tools.rdna.hetero_moe.solve --stats "$D/$w.stats.json" "${opts[@]}" \
            "${COMMON[@]}" --out "$plan"
        "$PY" -m tools.rdna.hetero_moe.validate "$plan" --stats "$D/$w.stats.json"
        for on in "$w" "$other"; do
            # hx broadcast exchange and the correlated row model (the simulate
            # defaults) with the plan format's row dispatch, and with the primary
            # copy only
            "$PY" -m tools.rdna.hetero_moe.simulate "$plan" --stats "$D/$on.stats.json" \
                --hub-fixed-us "$fixed" --expert-fixed-us 3.1 \
                > "$D/$h.$w.sim-on-$on.json"
            "$PY" -m tools.rdna.hetero_moe.simulate "$plan" --stats "$D/$on.stats.json" \
                --hub-fixed-us "$fixed" --expert-fixed-us 3.1 \
                --decode-dispatch primary > "$D/$h.$w.sim-on-$on.primary.json"
            # independent rows instead of the fitted correlated row model
            "$PY" -m tools.rdna.hetero_moe.simulate "$plan" --stats "$D/$on.stats.json" \
                --hub-fixed-us "$fixed" --expert-fixed-us 3.1 \
                --row-model independent > "$D/$h.$w.sim-on-$on.independent.json"
        done
    done
done
