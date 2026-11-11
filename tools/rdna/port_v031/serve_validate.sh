#!/usr/bin/env bash
# Boot one recipe, probe greedy correctness, run bench cells, collect markers.
#
#   RECIPE=flashnext-mtp0 MODEL=/path GPUS=2,3,4,5 PORT=18120 TAG=fn-mtp0 \
#     CELLS="1024:512:1 1024:512:8 16384:1024:8:32" \
#     bash tools/rdna/port_v031/serve_validate.sh [KEY=value recipe overrides...]
#
# Results: ~/w4a8_runs/port-v031/serve-$TAG/{serve.log,probe.txt,bench-*.json,summary.txt}
# CONC_PROBE=0 skips the greedy concurrency-correctness probe (on by default).
# CONC_PROBE_ARGS="--no-ignore-eos" checks normal EOS instead of forced continuation.
# RESET_PREFIX_CACHE=1 clears cached prompts before each benchmark cell;
# requires VLLM_SERVER_DEV_MODE=1 on the benchmark server.
# BENCH_EXTRA="--temperature 0" passes extra args to `vllm bench serve`. Without
# it the server's generation_config sampling applies (e.g. temperature 1.0,
# top_k 20, top_p 0.95 for Qwen3.8); greedy reference numbers need the flag.
# CELLS=none skips the bench cells. WARMUP=0 skips the capture-ladder warm-up
# (tools/rdna/port_v031/ladder_warmup.py) that runs before the stall probe.
#
# LOAD_HOOK=/path/script.sh runs after the bench cells (before the after-load
# probe) with URL, PORT, SERVED, MODEL and OUT exported, e.g. a real-traffic
# workload; its output goes to $OUT/load_hook.txt and must end in PASS/FAIL.
#
# STALL_PROBE=1 (default 0) runs tools/rdna/port_v031/stall_probe.py (mixed
# prefill/decode stall probe, docs/rdna2/mixed-batch-stall-probe.md) after the
# probes, with STALL_ARGS passed through (gates, lengths, repeats). It also
# boots the server with --enable-logging-iteration-details (log only, no
# behaviour change; STALL_ITER_LOG=0 turns that off) so the probe can tie
# stalls to scheduler steps. Results: $OUT/stall/{summary.txt,stall_probe.json,
# timeline.csv}.
#
# Every bench cell (and the stall probe, per window) records each worker's
# KFD queue-eviction time (tools/rdna/port_v031/kfd_evictions.py): a cell
# whose workers were evicted is marked INVALID in summary.txt and must be rerun.
#
# QUIET_DIR=/path (default ~/w4a8_runs/lanes/quiet; QUIET_SKIP=1 opts out): the
# quiet windows of ~/w4a8_runs/lanes/lease/README.md.
# Booter side: before the server is launched, wait until no file in that
# directory except QUIET_OWN* (default QUIET_NAME, else EvictFix) was touched in the last
# QUIET_MIN minutes (default 6; polling every QUIET_POLL seconds, 30), take
# BOOT_LOCK, look again under it and, if a window opened meanwhile, let go of the
# lock and wait again. Measurer side, with QUIET_HOLD=1: the bench cells are one
# window: wait for any docker buildx build, touch $QUIET_DIR/$QUIET_OWN (touched
# again every QUIET_TOUCH seconds, 60, for at most QUIET_CAP, 900, and not once
# this script is gone), wait until BOOT_LOCK is free (a boot that began before
# the touch still runs; QUIET_SETTLE seconds, 30, more after a wait) and again
# before every cell, run the cells, remove the file, and sleep QUIET_GAP seconds
# (120) before the LOAD_HOOK,
# which opens a window of its own for each load.
#
# STOP_GRACE=SECONDS (default 10): how long a server that is still shutting down
# gets after the SIGTERM before the SIGKILL. A TP4 worker holding 24 GiB of
# pinned memory can take longer than 10 s to leave KFD; a SIGKILL in the middle
# of that teardown is what a driver that cannot afford a wedged box avoids.
set -uo pipefail

TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0}
# Baseline A/B: SERVE_TREE=~/vllm-rdna-0.28.0 VENV=.../venv-7.14.0_0.28.0
# SERVE_SCRIPT=~/vllm-rdna-0.28.0/scripts/serve_rdna.sh (probes stay in TREE).
SERVE_TREE=${SERVE_TREE:-$TREE}
SERVE_SCRIPT=${SERVE_SCRIPT:-$SERVE_TREE/tools/rdna/serve_rdna.sh}
: "${RECIPE:?RECIPE required}" "${MODEL:?MODEL required}" "${GPUS:?GPUS required}"
PORT=${PORT:-18120}
TAG=${TAG:-$RECIPE}
CELLS=${CELLS:-"1024:512:1 1024:512:8 16384:1024:1 16384:1024:8"}
[[ $CELLS == none ]] && CELLS=""
READY_TIMEOUT=${READY_TIMEOUT:-3600}
STOP_GRACE=${STOP_GRACE:-10}
OUT=${OUT:-$HOME/w4a8_runs/port-v031/serve-$TAG}
mkdir -p "$OUT"
SUM=$OUT/summary.txt
: > "$SUM"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUM"; }

# HIP indices on par1-cs25: 0-1 = W7800 (other job), 2-5 = V620 PLX-B,
# 6-9 = V620 PLX-A. rocm-smi numbers them differently. This refusal is host
# allocation policy, not a vLLM constraint, so it is explicit and portable:
# REFUSED_GPUS keeps the par1-cs25 default; a host whose HIP 0/1 ARE the
# intended V620s (e.g. a four-card 0,1,2,3 allocation) clears or repoints
# the list via env instead of editing the driver.
REFUSED_GPUS=${REFUSED_GPUS-0,1}
for _refuse in ${REFUSED_GPUS//,/ }; do
    case ",$GPUS," in
        *,${_refuse},*)
            log "refusing HIP GPU ${_refuse} (reserved by this allocation)"
            exit 2 ;;
    esac
done

SITE=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib:$SITE/torch/lib
CACHE=${CACHE:-$HOME/w4a8_runs/port-v031/cache-$TAG}
# VLLM_CACHE_ROOT stays per serve (it also holds the rdna_ar wedge marker);
# the Triton / Inductor caches may be shared by serves of one tree (keyed by
# kernel / graph hash) to skip recompiling at boot.
export VLLM_CACHE_ROOT=$CACHE/vllm
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$CACHE/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$CACHE/inductor}
export ROCPROFILER_REGISTER_ENABLED=${ROCPROFILER_REGISTER_ENABLED:-0}  # bench clients too

if [[ ${STALL_PROBE:-0} == 1 && ${STALL_ITER_LOG:-1} == 1 ]]; then
    export EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--enable-logging-iteration-details"
fi

log "boot TREE=$SERVE_TREE RECIPE=$RECIPE MODEL=$MODEL GPUS=$GPUS PORT=$PORT overrides=$* bench_extra=${BENCH_EXTRA:-}"
dmesg_before=$(sudo -n dmesg 2>/dev/null | wc -l || echo 0)
# BOOT_LOCK=/path serializes boots across concurrent drivers (two boots at
# once have stalled on this box): held from launch until the server answers
# or the boot fails, never inherited by the server (9>&-).
# With QUIET_DIR the README's booter protocol applies: wait for no fresh quiet
# file, take the lock, look again under it (and start over if a window opened).
if [[ ${QUIET_SKIP:-0} == 1 ]]; then
    QUIET_DIR=""
else
    QUIET_DIR=${QUIET_DIR:-$HOME/w4a8_runs/lanes/quiet}
fi
QUIET_OWN=${QUIET_OWN:-${QUIET_NAME:-EvictFix}}
quiet_recent() {
    [[ -n ${QUIET_DIR:-} ]] || return 0
    find "$QUIET_DIR" -maxdepth 1 -type f -mmin "-${QUIET_MIN:-6}" \
        ! -name "${QUIET_OWN}*" -print -quit 2> /dev/null
}
[[ -n ${BOOT_LOCK:-} ]] && exec 9> "$BOOT_LOCK"
while :; do
    while q=$(quiet_recent) && [[ -n $q ]]; do
        log "quiet window (${q##*/}): waiting"
        sleep "${QUIET_POLL:-30}"
    done
    if [[ -n ${BOOT_LOCK:-} ]]; then
        log "waiting for boot lock $BOOT_LOCK"
        flock 9
        q=$(quiet_recent)
        if [[ -n $q ]]; then
            flock -u 9
            log "quiet window under the boot lock (${q##*/}): lock released"
            continue
        fi
    fi
    break
done
if [[ -n ${SO_BUILD_DIR:-} ]]; then
    bash "$TREE/tools/rdna/port_v031/attach_so.sh" "$SO_BUILD_DIR" "$SERVE_TREE" \
        ${SO_FORCE:+--force} || exit 1
fi
setsid bash "$SERVE_SCRIPT" RECIPE="$RECIPE" MODEL="$MODEL" \
    VENV="$VENV" VLLM_TREE="$SERVE_TREE" HIP_VISIBLE_DEVICES="$GPUS" ROCR_VISIBLE_DEVICES= PORT="$PORT" \
    HOST=127.0.0.1 VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
    TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" "$@" \
    > "$OUT/serve.log" 2>&1 < /dev/null 9>&- &
SPID=$!
echo "$SPID" > "$OUT/serve.pid"

QUIET_FILE=""
QUIET_PID=""
quiet_end() {
    if [[ -n $QUIET_PID ]]; then
        kill "$QUIET_PID" 2>/dev/null
        wait "$QUIET_PID" 2>/dev/null
        QUIET_PID=""
    fi
    if [[ -n $QUIET_FILE ]]; then
        rm -f "$QUIET_FILE"
        QUIET_FILE=""
        log "quiet window closed"
    fi
}
# A measurer waits until `flock -n BOOT_LOCK true` succeeds after touching its file
# and again before every measured cell (a booter can take the lock just before the
# file appears), and lets the boot settle for QUIET_SETTLE seconds after a wait.
quiet_lock_wait() {
    [[ -n ${BOOT_LOCK:-} ]] || return 0
    local t_lock waited=0
    t_lock=$(date +%s)
    while ! flock -n "$BOOT_LOCK" true 2>/dev/null; do
        if (( $(date +%s) - t_lock > ${QUIET_LOCK_WAIT:-600} )); then
            log "$BOOT_LOCK is still held: measuring anyway"
            return 0
        fi
        log "$BOOT_LOCK is held (a boot is in flight): waiting"
        sleep "${QUIET_POLL:-30}"
        waited=1
    done
    (( waited )) && sleep "${QUIET_SETTLE:-30}"
    return 0
}
# QUIET_HOLD=1: open the window of the bench cells (see the header).
quiet_begin() {
    [[ -n ${QUIET_DIR:-} && ${QUIET_HOLD:-0} == 1 ]] || return 0
    local shell=$$
    # anchored: another agent's waiter carries the words in its own command line
    while pgrep -f "${BUILDX_PAT:-^(/usr/sbin/|/usr/bin/)?docker buildx build}" > /dev/null 2>&1; do
        log "a docker buildx build is running: measured windows wait"
        sleep "${QUIET_POLL:-30}"
    done
    QUIET_FILE=$QUIET_DIR/${QUIET_OWN:-EvictFix}
    touch "$QUIET_FILE"
    (
        stop=$(( $(date +%s) + ${QUIET_CAP:-900} ))
        while sleep "${QUIET_TOUCH:-60}" && (( $(date +%s) < stop )) \
            && kill -0 "$shell" 2>/dev/null; do
            touch "$QUIET_FILE"
        done
    ) &
    QUIET_PID=$!
    log "quiet window open: $QUIET_FILE"
    quiet_lock_wait
}
stop_server() {
    kill -- -"$SPID" 2>/dev/null
    sleep 10
    local waited=10
    while (( waited < STOP_GRACE )) && kill -0 -- -"$SPID" 2>/dev/null; do
        sleep 2
        waited=$(( waited + 2 ))
    done
    kill -9 -- -"$SPID" 2>/dev/null
}
cleanup() {
    quiet_end
    stop_server
}
trap cleanup EXIT

t0=$(date +%s)
ready=0
while (( $(date +%s) - t0 < READY_TIMEOUT )); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$SPID" 2>/dev/null; then
        break
    fi
    sleep 10
done
boot_s=$(( $(date +%s) - t0 ))
[[ -n ${BOOT_LOCK:-} ]] && exec 9>&-
if (( ! ready )); then
    log "BOOT FAILED after ${boot_s}s"
    grep -aE "Error|error:|Traceback|raise |Exception|Memory Fault|HSA_STATUS|out of memory" \
        "$OUT/serve.log" | tail -25 | tee -a "$SUM"
    exit 1
fi
SERVED=$(python3 -c "import json;print(json.load(open('$OUT/models.json'))['data'][0]['id'])")
log "READY in ${boot_s}s served=$SERVED"
evict() { python3 "$TREE/tools/rdna/port_v031/kfd_evictions.py" "$SPID" "$@"; }
evict --save "$OUT/evicted-ready.json"

URL=http://127.0.0.1:$PORT/v1/completions
if "$VENV/bin/python" "$TREE/tools/probe_greedy_correctness.py" --url "$URL" \
    --model "$SERVED" --max-tokens 48 > "$OUT/probe.txt" 2>&1; then
    log "PROBE PASS"
else
    log "PROBE FAIL"
fi
tail -15 "$OUT/probe.txt" >> "$SUM"

# Perplexity + garbage check: the pass/fail signal for degraded checkpoints
# (e.g. pruned DeepSeek-V4-Flash) that cannot be held to exact answers.
if [[ ${PPL_PROBE:-1} == 1 ]]; then
    if "$VENV/bin/python" "$TREE/tools/rdna/port_v031/ppl_probe.py" --url "$URL" \
        --model "$SERVED" --max-ppl "${MAX_PPL:-60}" > "$OUT/ppl.txt" 2>&1; then
        log "PPL PROBE PASS"
    else
        log "PPL PROBE FAIL"
    fi
    sed 's/^/    /' "$OUT/ppl.txt" | tail -6 | tee -a "$SUM"
fi

if [[ ${PREFIX_PROBE:-1} == 1 ]]; then
    if "$VENV/bin/python" "$TREE/tools/rdna/port_v031/prefix_probe.py" --url "$URL" \
        --model "$SERVED" --words "${PREFIX_WORDS:-1800}" \
        > "$OUT/prefix.txt" 2>&1; then
        log "PREFIX PASS"
    else
        log "PREFIX FAIL"
    fi
    sed 's/^/    /' "$OUT/prefix.txt" | tail -4 | tee -a "$SUM"
fi

# Warm every capture size (one exact-length prefill each) and decode batches
# 1..SEQS before measuring: first use of a size JIT-compiles/tunes for up to
# tens of seconds and otherwise lands in a bench cell. WARMUP=0 skips it.
if [[ ${WARMUP:-1} == 1 ]]; then
    wsizes=${CG_SIZES:-}
    for a in "$@"; do [[ $a == CG_SIZES=* ]] && wsizes=${a#CG_SIZES=}; done
    wseqs=8
    for a in "$@"; do [[ $a == SEQS=* ]] && wseqs=${a#SEQS=}; done
    t_w=$(date +%s)
    "$VENV/bin/python" "$TREE/tools/rdna/port_v031/ladder_warmup.py" --url "http://127.0.0.1:$PORT" \
        --model "$SERVED" --max-seqs "$wseqs" \
        ${wsizes:+--sizes "$wsizes"} > "$OUT/warmup.txt" 2>&1
    log "warmup rc=$? in $(( $(date +%s) - t_w ))s (sizes ${wsizes:-default}); slowest:"
    sort -t: -k2 -g -r "$OUT/warmup.txt" 2>/dev/null | head -3 | sed 's/^/    /' | tee -a "$SUM"
fi

if [[ ${STALL_PROBE:-0} == 1 ]]; then
    log "stall probe: ${STALL_ARGS:-defaults}"
    # shellcheck disable=SC2086
    "$VENV/bin/python" "$TREE/tools/rdna/port_v031/stall_probe.py" \
        --base-url "http://127.0.0.1:$PORT" --model "$SERVED" --out "$OUT/stall" \
        --server-log "$OUT/serve.log" --server-pid "$SPID" --timeline ${STALL_ARGS:-} \
        > "$OUT/stall.txt" 2>&1
    rc=$?
    verdict=$(sed -n 's/^verdict: //p' "$OUT/stall.txt" | tail -1)
    log "STALL PROBE ${verdict:-ERROR} (rc=$rc, see $OUT/stall.txt)"
    sed -n '/^stall_probe:/,$p' "$OUT/stall.txt" | sed 's/^/    /' | tee -a "$SUM"
fi

# Concurrency correctness: greedy outputs at c=N and under slot churn must
# match the solo outputs. The c=1 probes above always run in slot 0.
if [[ ${CONC_PROBE:-1} == 1 ]]; then
    read -r -a conc_probe_args <<< "${CONC_PROBE_ARGS:-}"
    if "$VENV/bin/python" "$TREE/tools/rdna/port_v031/concurrency_probe.py" --url "$URL" \
        --model "$SERVED" --verbose --save-results "$OUT/concurrency.json" \
        "${conc_probe_args[@]}" > "$OUT/concurrency.txt" 2>&1; then
        log "CONCURRENCY PASS"
    else
        log "CONCURRENCY FAIL"
    fi
    grep -E "^FAIL|^WARN|RESULT" "$OUT/concurrency.txt" | tail -8 | sed 's/^/    /' | tee -a "$SUM"
fi

cd "$OUT"
quiet_begin
for cell in $CELLS; do
    # a booter may take the lock just before the window's file appeared: every cell
    # re-checks it (a no-op while no window is open)
    [[ -n $QUIET_FILE ]] && quiet_lock_wait
    IFS=: read -r in out conc nreq <<< "$cell"
    n=$(( conc * 4 > 8 ? conc * 4 : 8 ))
    (( conc == 1 )) && n=4
    [[ -n ${nreq:-} ]] && n=$nreq  # optional 4th field: request count
    name=bench-${in}x${out}-c${conc}
    if [[ ${RESET_PREFIX_CACHE:-0} == 1 ]]; then
        reset_ok=0
        for _ in 1 2 3 4 5; do
            if curl -sf -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" \
                > "$OUT/reset-prefix-$name.json" && \
                "$VENV/bin/python" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("success") is not True)' \
                "$OUT/reset-prefix-$name.json"; then
                reset_ok=1
                break
            fi
            sleep 1
        done
        if (( ! reset_ok )); then
            log "PREFIX RESET FAILED before $name"
            exit 1
        fi
        log "PREFIX RESET PASS before $name"
    fi
    log "bench $name (n=$n)"
    evict --save "$OUT/evicted-$name.json"
    # BENCH_EXTRA is a deliberately word-split list of extra CLI flags.
    # shellcheck disable=SC2086
    "$VENV/bin/python" -m vllm.entrypoints.cli.main bench serve \
        --backend openai --endpoint /v1/completions \
        --base-url "http://127.0.0.1:$PORT" --model "$SERVED" --tokenizer "$MODEL" \
        --dataset-name random --random-input-len "$in" --random-output-len "$out" \
        --num-prompts "$n" --max-concurrency "$conc" --ignore-eos \
        --request-rate inf --seed 12345 --save-result --result-dir "$OUT" \
        --result-filename "$name.json" ${BENCH_EXTRA:-} > "$OUT/$name.log" 2>&1
    python3 "$TREE/tools/rdna/port_v031/bench_metrics.py" "$OUT/$name.json" \
        | sed 's/^/    /' | tee -a "$SUM"
    evict --since "$OUT/evicted-$name.json" | sed 's/^/    /' | tee -a "$SUM"
    if ! kill -0 "$SPID" 2>/dev/null; then
        log "SERVER DIED during $name"
        break
    fi
done
quiet_end
if [[ -n ${LOAD_HOOK:-} && -n ${QUIET_DIR:-} && ${QUIET_HOLD:-0} == 1 ]] \
    && kill -0 "$SPID" 2>/dev/null; then
    log "quiet gap: ${QUIET_GAP:-120} s without a window before the load hook"
    sleep "${QUIET_GAP:-120}"
fi
if [[ -n ${LOAD_HOOK:-} ]] && kill -0 "$SPID" 2>/dev/null; then
    log "load hook: $LOAD_HOOK"
    URL=$URL PORT=$PORT SERVED=$SERVED MODEL=$MODEL OUT=$OUT \
        bash "$LOAD_HOOK" > "$OUT/load_hook.txt" 2>&1
    log "LOAD HOOK rc=$? $(grep -aoE "(PASS|FAIL)[^\n]*" "$OUT/load_hook.txt" | tail -1)"
    grep -aE "^\[|PASS|FAIL|garbage" "$OUT/load_hook.txt" | tail -12 | sed 's/^/    /' | tee -a "$SUM"
fi
# One more probe after load: corruption under load shows up here.
if "$VENV/bin/python" "$TREE/tools/probe_greedy_correctness.py" --url "$URL" \
    --model "$SERVED" --max-tokens 48 > "$OUT/probe-after.txt" 2>&1; then
    log "PROBE-AFTER-LOAD PASS"
else
    log "PROBE-AFTER-LOAD FAIL"
fi

log "$(evict --since "$OUT/evicted-ready.json" | sed 's/^evicted_ms/evicted_ms since READY:/')"
log "markers:"
grep -aoE "Using [A-Za-z0-9_]+(MoEMethod|LinearKernel|Kernel)[^\"]*|Selected [A-Za-z0-9_]+ |rdna_ar: [a-z]+[^\"]{0,60}|Using [A-Z_]+ backend[^\"]{0,40}|VLLM_USE_RDNA2_FA=1[^\"]{0,40}|qwen_gdn_full_forward[^\"]{0,30}|Initialized AMD PLE embedding[^\"]{0,120}|Model loading took[^\"]{0,60}|Available KV cache memory[^\"]{0,40}|GPU KV cache size[^\"]{0,40}|Maximum concurrency[^\"]{0,60}|Captur[a-z]+ [^\"]{0,60}took[^\"]{0,30}|WNA16 MoE backend[^\"]{0,60}|Mxfp4 MoE backend[^\"]{0,60}" \
    "$OUT/serve.log" | sort | uniq -c | sort -rn | head -40 | sed 's/^/    /' | tee -a "$SUM"
faults=$(grep -acE "Memory Fault|HSA_STATUS_ERROR|page fault|Segmentation fault|core dumped" "$OUT/serve.log")
log "fault markers in serve.log: $faults"
dmesg_new=$(sudo -n dmesg 2>/dev/null | tail -n +$((dmesg_before + 1)) | grep -iE "amdgpu.*(fault|timeout|reset)|UTCL2|AER|Hardware Error" | grep -vc "uses VM inv eng" || echo "n/a")
log "new amdgpu dmesg faults: $dmesg_new"
log "done"
