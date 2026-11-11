#!/usr/bin/env bash
# Profile a live server under mixed decode + long-prefill load with rocprofv3.
#
#   bash tools/rdna/prof/serve_profile.sh LABEL GPUS PORT rdna|offload [KEY=VAL ...]
#
# The server runs under rocprofv3 (via serve_rdna.sh LAUNCH_WRAPPER) with
# ROCPROFILER_REGISTER_ENABLED=1 (the launcher defaults it to 0, which leaves
# every process with an idle tool and an empty trace). After warmup, decode
# load (1k/512 at CDEC concurrency) plus a 16k-token prefill injector run for
# LOAD_S seconds; the workers are then sent SIGTERM so the tool flushes (the
# API server's handler deadlocks under rocprofv3, so it is killed afterwards).
# PERIOD=DELAY:DUR:1 (seconds after each process starts the tool) records only
# that window; without it the whole run is traced. Analyse the tail with:
#   python3 tools/rdna/prof/rocprof_gaps.py OUT/trace --window-ms 15000
#
# Env: OUT_ROOT (default ~/w4a8_runs/prof), VENV, MODEL, PLE_QUANT_DIR,
#      LOAD_S (60), CDEC (6), DEC_NUM (requests per decode bench, default CDEC; set it above CDEC so that CDEC requests
#      stay in flight while the bench client restarts), INJ_GAP (10), INJECT (1; 0 = decode load only), SEED (777),
#      TRACE_OPTS, PERIOD,
#      ROCPROFV3 (default: the venv's own rocprofv3; the /opt build registers
#      against a different rocprofiler-sdk and aborts at startup).
#
# PROFILER=torch swaps rocprofv3 for vLLM's torch profiler (Python stacks; traces land in OUT/torch, one file per rank
# and window). The rdna launch prepends tools/monokernel-p0-probe/kineto_shim to PYTHONPATH so that the workers record
# GPU kernels and HIP API events as well; the offload launcher hard-sets PYTHONPATH, so offload torch traces are
# CPU-only (use the rocprofv3 branch there). The stop-time device-time table is disabled (it blocks every worker for
# minutes on a with_stack trace).
# Windows are triggered by server state: TORCH_STATES="mixed decode" opens one TORCH_S-second window (default 3) per
# state, each waiting up to TORCH_WAIT s (120) on /metrics for CDEC running requests and prompt tokens moving (mixed)
# or not moving (decode); the load clients are stopped after the last window. Without TORCH_STATES: TORCH_AT seconds
# into the load it POSTs /start_profile, TORCH_S seconds later /stop_profile; optional second window: TORCH_GAP
# (seconds after the first stop returned), TORCH_S2 (its length; default TORCH_S). Kernels inside CUDA graphs are not
# attributed, so boot with CG_MODE=NONE (compile stays on) to map graph contents to their launch sites:
#   python3 tools/rdna/prof/kernel_sites.py OUT/torch/<rank0 trace>.json.gz
set -uo pipefail
LB=$1 G=$2 PT=$3 LAUNCH=$4
shift 4
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
V=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.32.0}
O=${OUT_ROOT:-$HOME/w4a8_runs/prof}/$LB
rm -rf "$O"
mkdir -p "$O"
M=${MODEL:-$HOME/.cache/huggingface/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
ROCPROFV3=${ROCPROFV3:-$V/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3}
log() { echo "[$LB $(date +%T)] $*"; }
# The server runs in its own session (setsid); kill only that session so a
# concurrent server on other GPUs survives.
server_sid() {
    local p
    p=$(pgrep -f "[-]-port $PT" | head -1)
    [[ -n $p ]] && ps -o sid= -p "$p" | tr -d ' '
}
stop_server() {
    local sid
    sid=$(server_sid)
    [[ -n $sid ]] && pkill -9 -s "$sid"
    pkill -9 -f "[-]-port $PT"
}

if [[ ${PROFILER:-rocprof} == torch ]]; then
    mkdir -p "$O/torch"
    # The API server imports torch first; its HIP runtime leaves ROCPROFILER_REGISTER_LIBRARY in the C
    # environment, every spawned worker inherits it, and kineto then records CPU ops and Python frames only
    # (no kernels, no HIP API events). The shim drops it before torch loads (tools/monokernel-p0-probe/README.md).
    KSHIM=$TREE/tools/monokernel-p0-probe/kineto_shim
    export ROCPROFILER_REGISTER_ENABLED=1
    # no device-time table at stop: key_averages() over a with_stack trace takes minutes and blocks every worker
    export EXTRA_ARGS="${EXTRA_ARGS:-} --profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"$O/torch\",\"torch_profiler_with_stack\":true,\"torch_profiler_dump_cuda_time_total\":false,\"ignore_frontend\":true}"
else
    TRACE_OPTS=${TRACE_OPTS:---kernel-trace --hip-runtime-trace --memory-copy-trace}
    [[ -n ${PERIOD:-} ]] && TRACE_OPTS+=" --collection-period $PERIOD --collection-period-unit sec"
    export ROCPROFILER_REGISTER_ENABLED=1
    export LAUNCH_WRAPPER="$ROCPROFV3 $TRACE_OPTS --output-format csv -d $O/trace -o %pid% --"
fi

cd "$TREE" || exit 1
# BOOT_LOCK=/path serializes boots across concurrent drivers; held until the
# server answers or fails, never inherited by the server (9>&-).
if [[ -n ${BOOT_LOCK:-} ]]; then
    exec 9> "$BOOT_LOCK"
    log "waiting for boot lock $BOOT_LOCK"
    flock 9
fi
T0=$(date +%s)
if [[ $LAUNCH == offload ]]; then
    (
        export TP=${TP:-2} GPUS=$G ALLOWED_GPU_SETS=$G VENV=$V PORT=$PT TAG=prof-$LB
        setsid nohup bash tools/rdna/mono/serve_offload.sh "$@" > "$O/serve.log" 2>&1 < /dev/null 9>&- &
    )
else
    (
        export HIP_VISIBLE_DEVICES=$G PYTHONPATH="${KSHIM:+$KSHIM:}$TREE"
        setsid nohup bash tools/rdna/serve_rdna.sh VENV="$V" VLLM_TREE="$TREE" PORT="$PT" \
            MODEL="$M" "$@" > "$O/serve.log" 2>&1 < /dev/null 9>&- &
    )
fi
for _ in $(seq 1 360); do
    curl -sf "localhost:$PT/v1/models" > "$O/models.json" 2>/dev/null && break
    if grep -qaE "exited unexpectedly|initialization failed|Traceback|caught signal" "$O/serve.log"; then
        log BOOTFAIL
        grep -aE "Error|caught signal" "$O/serve.log" | tail -5
        stop_server
        exit 1
    fi
    sleep 5
done
[[ -n ${BOOT_LOCK:-} ]] && exec 9>&-
[[ -s $O/models.json ]] || { log "BOOTFAIL timeout"; stop_server; exit 1; }
SN=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"][0]["id"])' "$O/models.json")
log "up after $(($(date +%s) - T0))s served=$SN"

bench() { # name in out n conc
    "$V/bin/python" -m vllm.entrypoints.cli.main bench serve --backend openai \
        --endpoint /v1/completions --base-url "http://127.0.0.1:$PT" --model "$SN" \
        --tokenizer "$M" --dataset-name random --random-input-len "$2" \
        --random-output-len "$3" --num-prompts "$4" --max-concurrency "$5" \
        --ignore-eos --request-rate inf --seed "${SEED:-777}" --save-result \
        --result-dir "$O" --result-filename "$1.json" > "$O/$1.log" 2>&1
}
bench warm 1024 64 8 4
log "warm done at +$(($(date +%s) - T0))s"
END=$(($(date +%s) + ${LOAD_S:-60}))
(
    k=0
    while (($(date +%s) < END)); do
        # DEC_NUM > CDEC keeps CDEC requests in flight for the whole bench (the client itself takes ~15 s to start between benches)
        bench "decode$k" 1024 512 "${DEC_NUM:-${CDEC:-6}}" "${CDEC:-6}"
        k=$((k + 1))
    done
) &
LD=$!
(
    k=0
    # INJECT=0 skips the 16k injector (offload: its ~100 s prefill fills the window)
    while [[ ${INJECT:-1} == 1 ]] && (($(date +%s) < END)); do
        bench "inject$k" 16384 16 1 1
        k=$((k + 1))
        sleep "${INJ_GAP:-10}"
    done
) &
LI=$!
metric() { # summed value of a /metrics series family (0 when the server does not answer)
    curl -sf "localhost:$PT/metrics" | awk -v n="$1" '$1 ~ "^"n"(\\{|$)" {s+=$NF} END {printf "%d\n", s}'
}
wait_state() { # wait_state decode|mixed MAX_S: all CDEC decoders running and prompt tokens flowing (mixed) / not flowing (decode)
    local t0 run p0 p1
    t0=$(date +%s)
    while (($(date +%s) - t0 < $2)); do
        run=$(metric 'vllm:num_requests_running')
        p0=$(metric 'vllm:prompt_tokens_total')
        sleep 2
        p1=$(metric 'vllm:prompt_tokens_total')
        if ((run >= ${CDEC:-6})); then
            [[ $1 == mixed ]] && ((p1 > p0)) && return 0
            [[ $1 == decode ]] && ((p1 == p0)) && return 0
        fi
    done
    return 1
}
torch_window() { # torch_window NAME STATE SECONDS [MAXWAIT]: trigger on server state, not on a fixed delay
    if [[ $2 == fixed ]]; then :; elif wait_state "$2" "${4:-120}"; then log "state $2 reached"; else log "state $2 NOT reached; profiling anyway"; fi
    curl -sf -X POST "localhost:$PT/start_profile" && log "torch profile $1 started"
    sleep "$3"
    curl -sf -X POST "localhost:$PT/stop_profile" && log "torch profile $1 stopped"
}
if [[ ${PROFILER:-rocprof} == torch ]]; then
    if [[ -n ${TORCH_STATES:-} ]]; then
        # TORCH_STATES="mixed decode": one window per state (TORCH_S seconds each, waiting up to TORCH_WAIT s for the state).
        # The decode/inject clients pause ~15 s between benches, so fixed delays can land on an idle engine.
        w=0
        for st in $TORCH_STATES; do
            w=$((w + 1))
            torch_window "w$w" "$st" "${TORCH_S:-3}" "${TORCH_WAIT:-120}"
        done
    else
        sleep "${TORCH_AT:-20}"
        torch_window w1 fixed "${TORCH_S:-3}"
        if [[ -n ${TORCH_GAP:-} ]]; then # second window, TORCH_GAP s after the first stop returned
            sleep "$TORCH_GAP"
            torch_window w2 fixed "${TORCH_S2:-${TORCH_S:-3}}"
        fi
    fi
fi
if [[ ${PROFILER:-rocprof} == torch && -n ${TORCH_STATES:-} ]]; then
    # windows are done: stop the load clients now instead of waiting for LOAD_S
    kill $LD $LI 2> /dev/null
    pkill -f "[-]-base-url http://127.0.0.1:$PT"
fi
wait $LD $LI
log "load done at +$(($(date +%s) - T0))s; stopping"
if [[ ${PROFILER:-rocprof} != torch ]]; then
    # rocprofv3 owns SIGTERM in every traced process, and the API server's
    # handler then waits for children that are never told to stop. Flush the
    # workers first: each worker's tool handler writes its trace and chains to
    # vLLM's.
    SID=$(server_sid)
    WPIDS=$([[ -n $SID ]] && pgrep -s "$SID" -f "[W]orker""_TP" | tr '\n' ' ')
    # shellcheck disable=SC2086 # one PID per word
    [[ -n $WPIDS ]] && kill -TERM $WPIDS
    for _ in $(seq 1 300); do
        alive=0
        for p in $WPIDS; do kill -0 "$p" 2> /dev/null && alive=1; done
        ((alive)) || break
        sleep 2
    done
    log "workers flushed ($WPIDS)"
fi
stop_server
log "trace csv=$(find "$O/trace" -name '*.csv' 2>/dev/null | wc -l) size=$(du -sh "$O/trace" 2>/dev/null | cut -f1)"
for f in "$O"/decode*.json "$O"/inject0.json; do
    [[ -f $f ]] && python3 "$TREE/tools/rdna/port_v031/bench_metrics.py" "$f" | sed 's/^/  /'
done
log FINISHED
