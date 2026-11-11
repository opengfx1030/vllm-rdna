#!/usr/bin/env bash
# serve_validate.sh LOAD_HOOK for one W4A8-quality arm. serve_validate.sh exports
# PORT, SERVED and OUT; run_arms.sh exports W4Q (run dir), TASKS, LIMIT, PHASES,
# TOPK_LIMIT, REPEAT and REF_BENCH (arm A's bench.json; empty for arm A itself).
# Ends with PASS or FAIL.
#   topk  : tools/rdna/port_v031/topk_kl.py collect (top-20, 2 domains x 2048-token
#           windows; TOPK_LIMIT windows per domain, 0 = all 128)
#   suite : hi2 bench/run.py with the Cline system prompt (agentic_bench.py run)
#   tfkld : hi2 bench/tfkld.py score of arm A's completions (agentic_bench.py tfkld)
# then a within-boot repeat of the first REPEAT (16) windows per domain (topk_rep).
# Every boot measures the whole arm: outputs left by an earlier (wedged) boot move
# into that boot's attempt<k>/ directory first, so nothing is resumed or mixed.
# /metrics is saved before and after every phase.
set -uo pipefail
: "${PORT:?}" "${SERVED:?}" "${OUT:?}" "${W4Q:?}" "${TASKS:?}" "${LIMIT:?}"
PY=${PY:-$HOME/Apps/vllm/venv-7.14.0_0.32.0/bin/python}
BASE=http://127.0.0.1:$PORT
T=$W4Q/tools
export HI2_BENCH_DATA=$HOME/w4a8_runs/hadamard-int2/bench_data/export
export HF_DATASETS_OFFLINE=1
rc=0
say() { echo "[$(date +%T)] $*"; }
metrics() {
    curl -sf "$BASE/metrics" > "$OUT/metrics-$1.txt" || { say "metrics $1 FAILED"; rc=1; }
}
bench() {
    "$PY" "$T/agentic_bench.py" --hi2 "$W4Q/hi2" --system "$W4Q/cline_act.md" "$@"
}
mapfile -t stale < <(find "$OUT" -maxdepth 1 \( -name 'topk.*' -o -name 'topk_rep.*' \
    -o -name 'bench.*' -o -name 'lp.*' -o -name 'metrics-*.txt' \) -printf '%f\n')
if ((${#stale[@]})); then
    last=$(find "$OUT" -maxdepth 1 -type d -name 'attempt*' -printf '%f\n' |
        sort -V | tail -n 1)
    dest=$OUT/${last:-discarded-$(date +%H%M%S)}
    mkdir -p "$dest"
    for f in "${stale[@]}"; do mv "$OUT/$f" "$dest/"; done
    say "discarded earlier outputs into $dest: ${stale[*]}"
fi
metrics 0-start
for phase in ${PHASES:-topk suite tfkld}; do
    say "$phase start"
    case $phase in
        topk)
            "$PY" "$T/topk_kl.py" collect --url "$BASE" --model "$SERVED" \
                --windows "$W4Q/windows.json" --top 20 --limit "${TOPK_LIMIT:-0}" \
                --out "$OUT/topk.npz" > "$OUT/topk.log" 2>&1
            ;;
        suite)
            bench run --url "$BASE" --model "$SERVED" --tasks "$TASKS" \
                --limit "$LIMIT" --workers 8 --seed 0 --out "$OUT/bench.json" \
                > "$OUT/bench.log" 2>&1
            ;;
        tfkld)
            bench tfkld score --url "$BASE" --model "$SERVED" \
                --ref-results "${REF_BENCH:-$OUT/bench.json}" --limit "$LIMIT" \
                --workers 8 --out "$OUT/lp.json" > "$OUT/lp.log" 2>&1
            ;;
        *) say "unknown phase $phase"; false ;;
    esac
    prc=$?
    say "$phase rc=$prc"
    ((prc == 0)) || rc=1
    metrics "after-$phase"
done
# Within-boot repeat of the first REPEAT windows per domain: separates per-request
# nondeterminism from boot-to-boot differences in the noise floor.
if ((${REPEAT:-16} > 0)); then
    say "topk repeat start"
    "$PY" "$T/topk_kl.py" collect --url "$BASE" --model "$SERVED" \
        --windows "$W4Q/windows.json" --top 20 --limit "${REPEAT:-16}" \
        --out "$OUT/topk_rep.npz" > "$OUT/topk_rep.log" 2>&1
    prc=$?
    say "topk repeat rc=$prc"
    ((prc == 0)) || rc=1
    metrics after-repeat
fi
if ((rc == 0)); then echo PASS; else echo FAIL; fi
exit "$rc"
