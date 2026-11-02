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
#      LOAD_S (60), CDEC (6), INJ_GAP (10), SEED (777), TRACE_OPTS, PERIOD,
#      ROCPROFV3 (default: the venv's own rocprofv3; the /opt build registers
#      against a different rocprofiler-sdk and aborts at startup).
#
# PROFILER=torch swaps rocprofv3 for vLLM's torch profiler (Python stacks):
# TORCH_AT seconds into the load it POSTs /start_profile, TORCH_S seconds later
# /stop_profile; traces land in OUT/torch. Kernels inside CUDA graphs are not
# attributed, so boot with CG_MODE=NONE (compile stays on) to map graph
# contents to their launch sites:
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
    export EXTRA_ARGS="${EXTRA_ARGS:-} --profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"$O/torch\",\"torch_profiler_with_stack\":true,\"ignore_frontend\":true}"
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
        export HIP_VISIBLE_DEVICES=$G PYTHONPATH=$TREE
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
        bench "decode$k" 1024 512 "${CDEC:-6}" "${CDEC:-6}"
        k=$((k + 1))
    done
) &
LD=$!
(
    k=0
    while (($(date +%s) < END)); do
        bench "inject$k" 16384 16 1 1
        k=$((k + 1))
        sleep "${INJ_GAP:-10}"
    done
) &
LI=$!
if [[ ${PROFILER:-rocprof} == torch ]]; then
    sleep "${TORCH_AT:-20}"
    curl -sf -X POST "localhost:$PT/start_profile" && log "torch profile started"
    sleep "${TORCH_S:-3}"
    curl -sf -X POST "localhost:$PT/stop_profile" && log "torch profile stopped"
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
