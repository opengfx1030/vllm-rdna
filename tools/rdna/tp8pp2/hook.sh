#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# LOAD_HOOK of the harness arms (driver.sh HOOK=1). serve_validate.sh exports URL PORT
# SERVED MODEL OUT; the driver exports ARM GPUS TREE_T VENV OWNER. Runs after the
# probes, PPL and the concurrency probe, with the server up:
#  - CONC_REPEAT=N (default 0): N more runs of concurrency_probe.py against the same
#    server, one file per run in $OUT/conc/ and a tally in $OUT/conc/runs.tsv (the
#    probe's own pass criterion per run: exit code 0 = PASS)
#  - QUALITY=1 (default): long-context quality collect (quality_ab.py collect)
#  - CELLS_SPEC (default the four 1k/16k c1/c8 cells; "none" = no measured window):
#    measured windows follow the lease-README quiet protocol: quiet/$OWNER touched
#    (re-touched every 2 min), then wait for `flock -n boot.lock true`, re-checked
#    before every cell and every rerun; no window while docker buildx build runs; a
#    window never exceeds ~12 min (closed, 2 min gap, reopened)
#  - every cell sits in a KFD evicted_ms window; INVALID -> rerun in the same boot with
#    a fresh seed (the original prompts are prefix-cached by then)
#  - cells and seeds are identical in every arm
set -uo pipefail
: "${OUT:?}" "${ARM:?}" "${GPUS:?}" "${TREE_T:?}" "${VENV:?}" "${PORT:?}" "${SERVED:?}" "${MODEL:?}" "${OWNER:?}"
# The server runs with PYTHONPATH=memhook (a sitecustomize that imports torch); the
# bench client, quality_ab.py and the KFD helpers must not inherit it (the client then
# aborts with the LLVM error).
unset PYTHONPATH RDNA_MEMHOOK_DIR RDNA_MEMHOOK_PERIOD
# shellcheck source=slots.sh
source "$(dirname "${BASH_SOURCE[0]}")/slots.sh"
PY=$VENV/bin/python
# vllm bench serve does not import under venv-7.14.0 (its editable vllm-rdna-mtp-mono
# aborts with "LLVM ERROR: Option 'spirv-expand-step' registered more than once"); the
# client only sends HTTP, so it runs from the venv every earlier harness run used.
BENCH_VENV=${BENCH_VENV:-$HOME/Apps/vllm/venv-7.14.0_0.32.0}
BENCH_SITE=$(echo "$BENCH_VENV"/lib/python3*/site-packages)
BENCH_LD=$BENCH_SITE/_rocm_sdk_libraries/lib:$BENCH_SITE/_rocm_sdk_core/lib/host-math/lib:$BENCH_SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$BENCH_SITE/_rocm_sdk_core/lib/core/lib:$BENCH_SITE/torch/lib
Q=$QUIET/$OWNER
BL=$BOOT_LOCK
ROOT_PID=$(cat "$OUT/serve.pid")
KFD=$TREE_T/tools/rdna/port_v031/kfd_evictions.py
BM=$TREE_T/tools/rdna/port_v031/bench_metrics.py
BUILDX_PAT='^(/usr/sbin/|/usr/bin/)?docker buildx build'
CELLS=${CELLS_SPEC:-"1024:512:1:4:12301 1024:512:8:32:12303 16384:1024:1:4:12304 16384:1024:8:32:12305"}
MAX_TRIES=3
WIN_MAX_S=${WIN_MAX_S:-480} # do not START a cell after the window is this old (16k c8 runs ~5 min; 15 min cap)
mkdir -p "$OUT/cells"
say() { echo "[$(date +%T)] $*"; }

"$PY" "$H/qprobe.py" "$TREE_T" "$ROOT_PID" "pre-cells" | tee -a "$OUT/queues.txt"

# conc_runs LABEL COUNT [probe args]: COUNT runs of concurrency_probe.py against the
# running server -> conc/<LABEL><n>.{txt,json}; one tally line per run in conc/runs.tsv
# (label, run, exit code, the probe's RESULT line). Exit code 0 = the probe's own PASS.
conc_runs() {
    local label=$1 count=$2 i crc npass=0 nrun=0
    shift 2
    for ((i = 1; i <= count; i++)); do
        kill -0 "$ROOT_PID" 2> /dev/null || {
            say "SERVER DIED before concurrency run $label$i"
            rc=1
            break
        }
        "$PY" "$TREE_T/tools/rdna/port_v031/concurrency_probe.py" \
            --url "http://127.0.0.1:$PORT/v1/completions" \
            --model "$SERVED" --verbose --save-results "$OUT/conc/$label$i.json" \
            "$@" > "$OUT/conc/$label$i.txt" 2>&1
        crc=$?
        nrun=$i
        printf '%s\t%s\t%s\t%s\n' "$label" "$i" "$crc" "$(grep -aE '^RESULT' "$OUT/conc/$label$i.txt" | tail -1)" >> "$OUT/conc/runs.tsv"
        ((crc == 0)) && npass=$((npass + 1))
    done
    say "CONC_REPEAT $label pass=$npass fail=$((nrun - npass)) of $nrun runs (tally: $OUT/conc/runs.tsv)"
    grep -ahE '^FAIL' "$OUT"/conc/"$label"[0-9]*.txt | sed 's/^/    /' | head -20
}

rc=0
if ((${CONC_REPEAT:-0} > 0 || ${CONC_REPEAT_NOEOS:-0} > 0)); then
    mkdir -p "$OUT/conc"
    read -r -a conc_args <<< "${CONC_PROBE_ARGS:-}"
    : > "$OUT/conc/runs.tsv"
    # CONC_REPEAT_NOEOS: the probe's default forces generation past the stop token
    # (--ignore-eos) so every request runs 64 tokens; this variant lets requests stop
    # where they would in serving
    ((${CONC_REPEAT:-0} > 0)) && conc_runs run "$CONC_REPEAT" "${conc_args[@]}"
    ((${CONC_REPEAT_NOEOS:-0} > 0)) && conc_runs noeos "$CONC_REPEAT_NOEOS" "${conc_args[@]}" --no-ignore-eos
fi

if [[ ${QUALITY:-1} == 1 ]]; then
    say "quality collect (long-context teacher-forced logprobs + greedy continuation)"
    "$PY" "$TREE_T/tools/rdna/port_v031/quality_ab.py" collect --url "http://127.0.0.1:$PORT" \
        --model "$SERVED" --out "$OUT/quality.json" --passages "${Q_PASSAGES:-6}" \
        --passage-chars "${Q_CHARS:-12000}" --gen-tokens 96 > "$OUT/quality.log" 2>&1 ||
        say "quality collect rc!=0 (see quality.log)"
fi

if [[ $CELLS == none ]]; then
    if ((rc == 0)); then echo "PASS hook $ARM (no measured cells)"; else echo "FAIL hook $ARM"; fi
    exit "$rc"
fi

TOUCHER=""
WIN_T0=0
win_open() {
    while pgrep -f "$BUILDX_PAT" > /dev/null; do
        say "docker buildx build running: measured windows wait"
        sleep 30
    done
    touch "$Q"
    (while :; do
        sleep 120
        touch "$Q"
    done) &
    TOUCHER=$!
    WIN_T0=$(date +%s)
    say "quiet window open"
}
win_close() {
    if [[ -n $TOUCHER ]]; then
        kill "$TOUCHER" 2> /dev/null
        wait "$TOUCHER" 2> /dev/null
        TOUCHER=""
    fi
    rm -f "$Q"
    say "quiet window closed"
}
lock_wait() {
    local t waited=0
    t=$(date +%s)
    while ! flock -n "$BL" true 2> /dev/null; do
        say "boot.lock held (a boot is in flight): waiting"
        touch "$Q"
        sleep 5
        waited=1
        if (($(date +%s) - t > 900)); then
            say "boot.lock held >15 min"
            return 1
        fi
    done
    ((waited)) && sleep 30
    return 0
}
trap 'win_close' EXIT

: > "$OUT/cells.tsv"
for spec in $CELLS; do
    IFS=: read -r in out conc n seed <<< "$spec"
    name=bench-${in}x${out}-c${conc}
    ok=0
    for ((try = 1; try <= MAX_TRIES; try++)); do
        if [[ -e $R/ABORT_TEMP.$ARM ]]; then
            say "thermal abort marker: stopping"
            rc=1
            break 2
        fi
        kill -0 "$ROOT_PID" 2> /dev/null || {
            say "SERVER DIED before $name"
            rc=1
            break 2
        }
        if [[ -n $TOUCHER ]] && (($(date +%s) - WIN_T0 > WIN_MAX_S)); then
            win_close
            say "window gap 130 s"
            sleep 130
        fi
        [[ -n $TOUCHER ]] || win_open
        if [[ -n ${BARRIER_DIR:-} ]] && ((try == 1)); then
            # replicas measure the same cell together: wait for the peer to arrive
            mkdir -p "$BARRIER_DIR"
            touch "$BARRIER_DIR/$ARM.$name"
            bw=0
            until [[ -e $BARRIER_DIR/$BARRIER_PEER.$name ]]; do
                if [[ -e $R/arms/$BARRIER_PEER/DONE ]] || ((bw > 2400)); then
                    say "barrier: peer $BARRIER_PEER is gone or never arrived at $name"
                    rc=1
                    break 2
                fi
                sleep 2
                touch "$Q"
                bw=$((bw + 2))
            done
        fi
        lock_wait || {
            rc=1
            break 2
        }
        # buildx may have appeared since the window opened
        if pgrep -f "$BUILDX_PAT" > /dev/null; then
            win_close
            win_open
            lock_wait || {
                rc=1
                break 2
            }
        fi
        cseed=$((seed + (try - 1) * 1000))
        cname=$name
        ((try > 1)) && cname=$name-r$try
        t_start=$(date +%H:%M:%S)
        "$PY" "$KFD" "$ROOT_PID" --save "$OUT/$cname.kfd0.json" > /dev/null 2>&1
        say "bench $cname (n=$n seed=$cseed try=$try)"
        (cd "$OUT/cells" && env -u PYTHONPATH -u RDNA_MEMHOOK_DIR LD_LIBRARY_PATH="$BENCH_LD" HIP_VISIBLE_DEVICES="$GPUS" \
            "$BENCH_VENV/bin/python" -m vllm.entrypoints.cli.main bench serve \
            --backend openai --endpoint /v1/completions \
            --base-url "http://127.0.0.1:$PORT" --model "$SERVED" --tokenizer "$MODEL" \
            --dataset-name random --random-input-len "$in" --random-output-len "$out" \
            --num-prompts "$n" --max-concurrency "$conc" --ignore-eos --temperature 0 \
            --request-rate inf --seed "$cseed" --save-result --result-dir "$OUT/cells" \
            --result-filename "$cname.json" > "$OUT/cells/$cname.log" 2>&1) || say "bench client rc!=0 for $cname"
        t_end=$(date +%H:%M:%S)
        kfd=$("$PY" "$KFD" "$ROOT_PID" --since "$OUT/$cname.kfd0.json" 2>&1 | tail -1)
        say "$cname kfd: $kfd"
        "$PY" "$BM" "$OUT/cells/$cname.json" | sed 's/^/    /'
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cname" "$name" "$try" "$cseed" "$t_start" "$t_end" "$kfd" >> "$OUT/cells.tsv"
        if [[ $kfd == *INVALID* ]]; then
            say "$cname INVALID (KFD eviction): rerun"
            continue
        fi
        if [[ ! -s $OUT/cells/$cname.json ]]; then
            say "$cname produced no JSON"
            continue
        fi
        ok=1
        break
    done
    ((ok)) || {
        say "cell $name: no valid run"
        rc=1
    }
done
win_close
"$PY" "$H/qprobe.py" "$TREE_T" "$ROOT_PID" "post-cells" | tee -a "$OUT/queues.txt"
if ((rc == 0)); then echo "PASS hook $ARM"; else echo "FAIL hook $ARM"; fi
exit "$rc"
