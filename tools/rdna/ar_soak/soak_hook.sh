#!/usr/bin/env bash
# LOAD_HOOK of the rdna_ar soak (serve_validate.sh exports URL PORT SERVED MODEL
# OUT; driver.sh exports R TREE_T VENV SOAK_MIN HIP_CLIENT BENCH_VENV).
# Runs after the boot probes with the server up, for SOAK_MIN minutes:
#   - continuous mixed load: 1k/512 and 16k/1k bench cells alternate, concurrency
#     1..8 and the seed random, back to back (no gap, no quiet window: the
#     neighbour boots on the other slot are the point of the soak)
#   - every CHECK_S (900) seconds: greedy probe, PPL probe, and three fixed
#     reference cells (1k/512 c1, 1k/512 c8, 16k/1k c1) for the ITL drift line;
#     a concurrency probe every CONC_S (3600) seconds
#   - every cell: wall time, failed/completed from the result JSON, client rc
# Results under $OUT/soak/: cells.tsv, checks.tsv, cells/, probes/.
# Ends with PASS or FAIL on the last line.
set -uo pipefail
: "${OUT:?}" "${TREE_T:?}" "${VENV:?}" "${PORT:?}" "${SERVED:?}" "${MODEL:?}" "${R:?}"
SOAK_MIN=${SOAK_MIN:-135}
CHECK_S=${CHECK_S:-900}
CONC_S=${CONC_S:-3600}
CELL_TIMEOUT=${CELL_TIMEOUT:-1500}
HIP_CLIENT=${HIP_CLIENT:?}
PY=$VENV/bin/python
BENCH_VENV=${BENCH_VENV:-$HOME/Apps/vllm/venv-7.14.0_0.32.0}
BENCH_SITE=$BENCH_VENV/lib/python3.12/site-packages
BENCH_LD=$BENCH_SITE/_rocm_sdk_libraries/lib:$BENCH_SITE/_rocm_sdk_core/lib/host-math/lib:$BENCH_SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$BENCH_SITE/_rocm_sdk_core/lib:$BENCH_SITE/torch/lib
S=$OUT/soak
URLC=http://127.0.0.1:$PORT/v1/completions
mkdir -p "$S/cells" "$S/probes"
say() { echo "[$(date +%T)] $*"; }

t_start=$(date +%s)
t_end=$((t_start + SOAK_MIN * 60))
next_check=$(($(date +%s) + CHECK_S))
next_conc=$(($(date +%s) + CONC_S))
printf 'cell\tshape\tconc\tn\tseed\tstart\tend\twall_s\trc\tcompleted\tfailed\titl_med\titl_mean\titl_p99\tttft_mean\tout_tps\tref\n' > "$S/cells.tsv"
printf 'check\ttime\tprobe\tppl\tppl_rc\tprobe_rc\n' > "$S/checks.tsv"

cell() { # cell NAME IN OUT CONC N SEED REF
    local name=$1 in=$2 out=$3 conc=$4 n=$5 seed=$6 ref=$7 t0 t1 rc=0
    t0=$(date +%s)
    (cd "$S/cells" && LD_LIBRARY_PATH=$BENCH_LD HIP_VISIBLE_DEVICES=$HIP_CLIENT \
        timeout --kill-after=30 "$CELL_TIMEOUT" "$BENCH_VENV/bin/python" -m vllm.entrypoints.cli.main bench serve \
        --backend openai --endpoint /v1/completions \
        --base-url "http://127.0.0.1:$PORT" --model "$SERVED" --tokenizer "$MODEL" \
        --dataset-name random --random-input-len "$in" --random-output-len "$out" \
        --num-prompts "$n" --max-concurrency "$conc" --ignore-eos --temperature 0 \
        --request-rate inf --seed "$seed" --save-result --result-dir "$S/cells" \
        --result-filename "$name.json" > "$S/cells/$name.log" 2>&1) || rc=$?
    t1=$(date +%s)
    "$PY" - "$S/cells/$name.json" "$name" "$in" "$conc" "$n" "$seed" "$t0" "$t1" "$rc" "$ref" >> "$S/cells.tsv" << 'PYEOF'
import json, sys, time
path, name, shape, conc, n, seed, t0, t1, rc, ref = sys.argv[1:]
t0, t1 = int(t0), int(t1)
try:
    d = json.load(open(path))
    done = int(d.get("completed", 0))
    failed = int(d.get("failed", int(n) - done))
    vals = [d.get(k, 0.0) for k in ("median_itl_ms", "mean_itl_ms", "p99_itl_ms", "mean_ttft_ms", "output_throughput")]
except (OSError, ValueError):
    done, failed, vals = 0, int(n), [0.0] * 5
f = time.strftime
print("\t".join([name, shape, conc, n, seed, f("%H:%M:%S", time.localtime(t0)), f("%H:%M:%S", time.localtime(t1)),
                 str(t1 - t0), rc, str(done), str(failed)] + [f"{v:.2f}" for v in vals] + [ref]))
PYEOF
    tail -1 "$S/cells.tsv" | awk -F'\t' '{printf "  %s %s c%s n=%s wall %ss rc=%s ok=%s fail=%s ITL med %s p99 %s TTFT %s\n", $1,$2,$3,$4,$8,$9,$10,$11,$12,$14,$15}'
}

check() { # check LABEL
    local label=$1 prc=0 lrc=0 ppl
    "$PY" "$TREE_T/tools/probe_greedy_correctness.py" --url "$URLC" --model "$SERVED" --max-tokens 48 \
        > "$S/probes/probe-$label.txt" 2>&1 || prc=$?
    "$PY" "$TREE_T/tools/rdna/port_v031/ppl_probe.py" --url "$URLC" --model "$SERVED" --max-ppl "${MAX_PPL:-60}" \
        > "$S/probes/ppl-$label.txt" 2>&1 || lrc=$?
    ppl=$(sed -n 's/^PPL \([0-9.]*\).*/\1/p' "$S/probes/ppl-$label.txt" | tail -1)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$(date +%T)" "$([[ $prc == 0 ]] && echo PASS || echo FAIL)" "${ppl:-NA}" "$lrc" "$prc" >> "$S/checks.tsv"
    say "check $label: probe rc=$prc ppl=${ppl:-NA} rc=$lrc"
}

conc_check() {
    local rc=0
    "$PY" "$TREE_T/tools/rdna/port_v031/concurrency_probe.py" --url "$URLC" --model "$SERVED" \
        > "$S/probes/concurrency-$1.txt" 2>&1 || rc=$?
    printf 'conc-%s\t%s\t%s\t-\t-\t%s\n' "$1" "$(date +%T)" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)" "$rc" >> "$S/checks.tsv"
    say "concurrency probe $1 rc=$rc"
}

stop_reason() {
    local f
    for f in "$OUT"/soak/FAIL.* "$R"/ABORT_TEMP; do
        [[ -e $f && $f != */FAIL.traceback ]] && { echo "$f"; return 0; }
    done
    kill -0 "$(cat "$OUT/serve.pid")" 2> /dev/null || { echo "server gone"; return 0; }
    return 1
}

fails=0
k=0
check boot
while (($(date +%s) < t_end)); do
    if reason=$(stop_reason); then
        say "stopping the load: $reason"
        fails=$((fails + 1))
        break
    fi
    if (($(date +%s) >= next_check)); then
        next_check=$((next_check + CHECK_S))
        check "t$((($(date +%s) - t_start) / 60))m"
        cell "ref-1k-c1-$k" 1024 512 1 4 $((RANDOM * 32768 + RANDOM)) ref
        cell "ref-1k-c8-$k" 1024 512 8 16 $((RANDOM * 32768 + RANDOM)) ref
        cell "ref-16k-c1-$k" 16384 1024 1 2 $((RANDOM * 32768 + RANDOM)) ref
        if (($(date +%s) >= next_conc)); then
            next_conc=$((next_conc + CONC_S))
            conc_check "$k"
        fi
    fi
    k=$((k + 1))
    conc=$((RANDOM % 8 + 1))
    seed=$((RANDOM * 32768 + RANDOM))
    if ((k % 2)); then
        n=$((conc * 3 < 4 ? 4 : conc * 3))
        cell "mix-$k-1k-c$conc" 1024 512 "$conc" "$n" "$seed" mix
    else
        n=$((conc * 2 < 3 ? 3 : conc * 2))
        cell "mix-$k-16k-c$conc" 16384 1024 "$conc" "$n" "$seed" mix
    fi
done
check final
touch "$S/STOP" # the monitor ends before the server is shut down
sleep 6

# summary of failures in the client results
bad=$(awk -F'\t' 'NR>1 && ($9 != 0 || $11 != 0 || $10 == 0)' "$S/cells.tsv" | wc -l)
chk_bad=$(awk -F'\t' 'NR>1 && ($3 != "PASS" || ($5 != 0 && $5 != "-"))' "$S/checks.tsv" | wc -l)
crit=$(find "$S" -maxdepth 1 -name 'FAIL.*' | wc -l)
say "monitor FAIL markers: $crit"
say "cells: $(($(wc -l < "$S/cells.tsv") - 1)), bad cells: $bad, bad checks: $chk_bad, load-stop reasons: $fails"
if ((bad == 0 && chk_bad == 0 && fails == 0 && crit == 0)); then echo "PASS soak load"; else echo "FAIL soak load"; fi
