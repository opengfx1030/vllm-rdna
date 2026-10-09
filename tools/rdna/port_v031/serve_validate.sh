#!/usr/bin/env bash
# Boot one recipe, probe greedy correctness, run bench cells, collect markers.
#
#   RECIPE=flashnext-mtp0 MODEL=/path GPUS=2,3,4,5 PORT=18120 TAG=fn-mtp0 \
#     CELLS="1024:512:1 1024:512:8 16384:1024:8:32" \
#     bash tools/rdna/port_v031/serve_validate.sh [KEY=value recipe overrides...]
#
# Results: ~/w4a8_runs/port-v031/serve-$TAG/{serve.log,probe.txt,bench-*.json,summary.txt}
# CONC_PROBE=0 skips the greedy concurrency-correctness probe (on by default).
# BENCH_EXTRA="--temperature 0" passes extra args to `vllm bench serve`. Without
# it the server's generation_config sampling applies (e.g. temperature 1.0,
# top_k 20, top_p 0.95 for Qwen3.8); greedy reference numbers need the flag.
# CELLS=none skips the bench cells.
#
# STALL_PROBE=1 (default 0) runs tools/rdna/port_v031/stall_probe.py (mixed
# prefill/decode stall probe, docs/rdna2/mixed-batch-stall-probe.md) after the
# probes, with STALL_ARGS passed through (gates, lengths, repeats). It also
# boots the server with --enable-logging-iteration-details (log only, no
# behaviour change; STALL_ITER_LOG=0 turns that off) so the probe can tie
# stalls to scheduler steps. Results: $OUT/stall/{summary.txt,stall_probe.json,
# timeline.csv}.
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
OUT=${OUT:-$HOME/w4a8_runs/port-v031/serve-$TAG}
mkdir -p "$OUT"
SUM=$OUT/summary.txt
: > "$SUM"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUM"; }

# HIP indices on par1-cs25: 0-1 = W7800 (other job), 2-5 = V620 PLX-B,
# 6-9 = V620 PLX-A. rocm-smi numbers them differently.
case ",$GPUS," in
    *,0,* | *,1,*) log "refusing HIP GPUs 0/1 (W7800s, other job)"; exit 2 ;;
esac

SITE=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib/core/lib:$SITE/torch/lib
CACHE=${CACHE:-$HOME/w4a8_runs/port-v031/cache-$TAG}
export VLLM_CACHE_ROOT=$CACHE/vllm TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor

if [[ ${STALL_PROBE:-0} == 1 && ${STALL_ITER_LOG:-1} == 1 ]]; then
    export EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--enable-logging-iteration-details"
fi

log "boot TREE=$SERVE_TREE RECIPE=$RECIPE MODEL=$MODEL GPUS=$GPUS PORT=$PORT overrides=$* bench_extra=${BENCH_EXTRA:-}"
dmesg_before=$(sudo -n dmesg 2>/dev/null | wc -l || echo 0)
setsid bash "$SERVE_SCRIPT" RECIPE="$RECIPE" MODEL="$MODEL" \
    VENV="$VENV" VLLM_TREE="$SERVE_TREE" HIP_VISIBLE_DEVICES="$GPUS" ROCR_VISIBLE_DEVICES= PORT="$PORT" \
    HOST=127.0.0.1 VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
    TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" "$@" \
    > "$OUT/serve.log" 2>&1 < /dev/null &
SPID=$!
echo "$SPID" > "$OUT/serve.pid"

stop_server() {
    kill -- -"$SPID" 2>/dev/null
    sleep 10
    kill -9 -- -"$SPID" 2>/dev/null
}
trap stop_server EXIT

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
if (( ! ready )); then
    log "BOOT FAILED after ${boot_s}s"
    grep -aE "Error|error:|Traceback|raise |Exception|Memory Fault|HSA_STATUS|out of memory" \
        "$OUT/serve.log" | tail -25 | tee -a "$SUM"
    exit 1
fi
SERVED=$(python3 -c "import json;print(json.load(open('$OUT/models.json'))['data'][0]['id'])")
log "READY in ${boot_s}s served=$SERVED"

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
        --model "$SERVED" > "$OUT/prefix.txt" 2>&1; then
        log "PREFIX PASS"
    else
        log "PREFIX FAIL"
    fi
    sed 's/^/    /' "$OUT/prefix.txt" | tail -4 | tee -a "$SUM"
fi

if [[ ${STALL_PROBE:-0} == 1 ]]; then
    log "stall probe: ${STALL_ARGS:-defaults}"
    # shellcheck disable=SC2086
    "$VENV/bin/python" "$TREE/tools/rdna/port_v031/stall_probe.py" \
        --base-url "http://127.0.0.1:$PORT" --model "$SERVED" --out "$OUT/stall" \
        --server-log "$OUT/serve.log" --timeline ${STALL_ARGS:-} \
        > "$OUT/stall.txt" 2>&1
    rc=$?
    verdict=$(sed -n 's/^verdict: //p' "$OUT/stall.txt" | tail -1)
    log "STALL PROBE ${verdict:-ERROR} (rc=$rc, see $OUT/stall.txt)"
    sed -n '/^stall_probe:/,$p' "$OUT/stall.txt" | sed 's/^/    /' | tee -a "$SUM"
fi

# Concurrency correctness: greedy outputs at c=N and under slot churn must
# match the solo outputs. The c=1 probes above always run in slot 0.
if [[ ${CONC_PROBE:-1} == 1 ]]; then
    if "$VENV/bin/python" "$TREE/tools/rdna/port_v031/concurrency_probe.py" --url "$URL" \
        --model "$SERVED" --verbose > "$OUT/concurrency.txt" 2>&1; then
        log "CONCURRENCY PASS"
    else
        log "CONCURRENCY FAIL"
    fi
    grep -E "^FAIL|^WARN|RESULT" "$OUT/concurrency.txt" | tail -8 | sed 's/^/    /' | tee -a "$SUM"
fi

cd "$OUT"
for cell in $CELLS; do
    IFS=: read -r in out conc nreq <<< "$cell"
    n=$(( conc * 4 > 8 ? conc * 4 : 8 ))
    (( conc == 1 )) && n=4
    [[ -n ${nreq:-} ]] && n=$nreq  # optional 4th field: request count
    name=bench-${in}x${out}-c${conc}
    log "bench $name (n=$n)"
    "$VENV/bin/python" -m vllm.entrypoints.cli.main bench serve \
        --backend openai --endpoint /v1/completions \
        --base-url "http://127.0.0.1:$PORT" --model "$SERVED" --tokenizer "$MODEL" \
        --dataset-name random --random-input-len "$in" --random-output-len "$out" \
        --num-prompts "$n" --max-concurrency "$conc" --ignore-eos \
        --request-rate inf --seed 12345 --save-result --result-dir "$OUT" \
        --result-filename "$name.json" ${BENCH_EXTRA:-} > "$OUT/$name.log" 2>&1
    python3 "$TREE/tools/rdna/port_v031/bench_metrics.py" "$OUT/$name.json" \
        | sed 's/^/    /' | tee -a "$SUM"
    if ! kill -0 "$SPID" 2>/dev/null; then
        log "SERVER DIED during $name"
        break
    fi
done
# One more probe after load: corruption under load shows up here.
if "$VENV/bin/python" "$TREE/tools/probe_greedy_correctness.py" --url "$URL" \
    --model "$SERVED" --max-tokens 48 > "$OUT/probe-after.txt" 2>&1; then
    log "PROBE-AFTER-LOAD PASS"
else
    log "PROBE-AFTER-LOAD FAIL"
fi

log "markers:"
grep -aoE "Using [A-Za-z0-9_]+(MoEMethod|LinearKernel|Kernel)[^\"]*|Selected [A-Za-z0-9_]+ |rdna_ar: [a-z]+[^\"]{0,60}|Using [A-Z_]+ backend[^\"]{0,40}|VLLM_USE_RDNA2_FA=1[^\"]{0,40}|qwen_gdn_full_forward[^\"]{0,30}|Initialized AMD PLE embedding[^\"]{0,120}|Model loading took[^\"]{0,60}|Available KV cache memory[^\"]{0,40}|GPU KV cache size[^\"]{0,40}|Maximum concurrency[^\"]{0,60}|Captur[a-z]+ [^\"]{0,60}took[^\"]{0,30}|WNA16 MoE backend[^\"]{0,60}|Mxfp4 MoE backend[^\"]{0,60}" \
    "$OUT/serve.log" | sort | uniq -c | sort -rn | head -40 | sed 's/^/    /' | tee -a "$SUM"
faults=$(grep -acE "Memory Fault|HSA_STATUS_ERROR|page fault|Segmentation fault|core dumped" "$OUT/serve.log")
log "fault markers in serve.log: $faults"
dmesg_new=$(sudo -n dmesg 2>/dev/null | tail -n +$((dmesg_before + 1)) | grep -iE "amdgpu.*(fault|timeout|reset)|UTCL2|AER|Hardware Error" | grep -vc "uses VM inv eng" || echo "n/a")
log "new amdgpu dmesg faults: $dmesg_new"
log "done"
