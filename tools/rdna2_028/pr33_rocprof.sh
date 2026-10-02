#!/usr/bin/env bash
# PR#33 in-model rocprof A/B. Wraps a Flash-Next / TP4 / F&P / prefix-cache
# serve from birth with venv-7.14.0 rocprofv3 (--run mode; torch here is not
# register-built, so --attach cannot be used). One arm per invocation.
#
#   ARM=baseline|patched CTX=16384 PORT=18130 bash pr33_rocprof.sh
#
# The full kernel trace is written (rocprof's --kernel-include-regex only
# filters counter/thread-trace, not dispatch rows); the parser windows it.
# No /tmp. Logs/results: /home/chenco_adm/w4a8_runs/pr33_retest/<tag>/.
set -uo pipefail

V=${V:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0}
T=${T:-/home/chenco_adm/vllm-rdna-0.28.0}
RUN=${RUN:-/home/chenco_adm/w4a8_runs/pr33_retest}
MODEL=${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
PLE=${PLE:-/home/chenco_adm/hfcache/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}
ARM=${ARM:?ARM required}
CTX=${CTX:?CTX required}
PORT=${PORT:-18130}
RECIPE=${RECIPE:-flashnext-mtp0}
CELLS=${CELLS:-"1 ${CTX} 512 711|3 ${CTX} 512 713"}
TAG=${TAG:-rocprof_${ARM}_${CTX}}
READY_MAX=${READY_MAX:-180}
D=$RUN/$TAG
mkdir -p "$D"
DRIVER_LOG=$D/driver.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }

our_pids() {
  local p e
  for p in $(pgrep -f "entrypoints.cli.main serve|entrypoints.openai.api_server|VLLM::Worker|VLLM::EngineCore|PleOffloadWorker|rocprofv3" 2>/dev/null); do
    e=$(tr '\0' '\n' <"/proc/$p/environ" 2>/dev/null || true)
    case "$e" in *"VLLM_CACHE_ROOT=$T/cache/pr33-"*) echo "$p" ;; esac
  done
}
hygiene() {
  local n=0 left p i
  for p in $(our_pids); do kill -TERM "$p" 2>/dev/null && n=$((n+1)); done
  [ "$n" -gt 0 ] && sleep 8
  for p in $(our_pids); do kill -KILL "$p" 2>/dev/null; done
  for i in $(seq 1 30); do left=$(our_pids | tr '\n' ' '); [ -z "${left// /}" ] && break; sleep 2; done
  left=$(our_pids | tr '\n' ' ')
  log "hygiene: terminated=$n remaining=[${left% }]"
  [ -z "${left// /}" ]
}
sel_count(){ timeout 10 sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }

SO=$RUN/_rocm_C.$ARM.so
[ -r "$SO" ] || { log "missing $SO"; exit 1; }
cp "$SO" "$T/vllm/_rocm_C.abi3.so"
so=$(sha256sum "$T/vllm/_rocm_C.abi3.so" | cut -c1-16)
log "=== ARM=$ARM ctx=$CTX so=$so recipe=$RECIPE port=$PORT ==="
hygiene || { log "hygiene failed"; exit 1; }
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then log "port $PORT busy"; exit 1; fi

CACHE=${CACHE:-$T/cache/pr33-$ARM-$CTX}
if [ "${KEEP_CACHE:-0}" = 1 ] && [ -d "$CACHE/inductor" ]; then
  log "KEEP_CACHE=1: reusing $CACHE"
else
  rm -rf "$CACHE"
fi
mkdir -p "$CACHE/inductor" "$CACHE/extensions" "$CACHE/triton" "$CACHE/vllm"
rm -f "$CACHE/rdna_ar_wedged"

# rocprof buffers per-process in $TMPDIR/.rocprofv3 and merges on clean exit.
# Keep it inside the campaign dir (persistent) rather than /tmp.
mkdir -p "$D/tmp"

RSL=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
RS=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$RSL:$RS/host-math/lib:$RS/rocm_sysdeps/lib:$RS/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

S0=$(sel_count); log "PCI-SERR before=$S0 uptime=$(uptime -p)"
T0=$(date +%s)
setsid nohup env \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  TMPDIR="$D/tmp" \
  VENV="$V" VLLM_TREE="$T" MODEL="$MODEL" VLLM_PLE_QUANT_DIR="$PLE" \
  VLLM_CACHE_ROOT="$CACHE" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
  TRITON_CACHE_DIR="$CACHE/triton" TORCH_EXTENSIONS_DIR="$CACHE/extensions" \
  HIP_VISIBLE_DEVICES=4,5,6,7 \
  "$V/bin/rocprofv3" --kernel-trace --stats -f csv -o "$D/prof_%pid%" -- \
  bash "$T/scripts/serve_rdna.sh" RECIPE="$RECIPE" PORT="$PORT" \
  >>"$D/serve.log" 2>&1 </dev/null &
RPID=$!
log "rocprof pid=$RPID"
ready=0; dead=0
for i in $(seq 1 "$READY_MAX"); do
  sleep 10
  curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q flash-next && { ready=1; break; }
  if kill -0 "$RPID" 2>/dev/null; then dead=0; else dead=$((dead+1)); fi
  [ "$dead" -ge 3 ] && { log "rocprof launcher gone during boot"; break; }
done
COLD=$(( $(date +%s) - T0 ))
if [ "$ready" != 1 ]; then log "NOT READY after ${COLD}s"; tail -40 "$D/serve.log" >>"$DRIVER_LOG"; hygiene; exit 1; fi
log "READY t=${COLD}s"; echo "$COLD" >"$D/cold_seconds.txt"
S1=$(sel_count); log "PCI-SERR after boot=$S1 uptime=$(uptime -p)"
if [ "$S1" -gt "$S0" ]; then echo "STOP NEW PCI SERR" >"$RUN/IM_STOP"; hygiene; exit 2; fi
if [ "${BOOT_ONLY:-0}" = 1 ]; then log "BOOT_ONLY ready"; exit 0; fi

# Prefill-only warmup (1 output token so it does not pollute decode counts much)
for wl in "1024 1 991" "16384 1 992"; do set -- $wl
  ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --served-model-name flash-next --dataset-name random --random-input-len "$1" --random-output-len "$2" --num-prompts 1 --max-concurrency 1 --ignore-eos --request-rate inf --seed "$3" --temperature 0 --save-result --result-dir "$D/warmup_$1" ) >"$D/warmup_$1.log" 2>&1
done
log "warmup done"

IFS='|' read -ra _cells <<<"$CELLS"
: >"$D/cell_windows.txt"
for cell in "${_cells[@]}"; do set -- $cell
  N=$1; IN=$2; OUTL=$3; SEED=$4
  CD="$D/cells/c${N}_${IN}"
  mkdir -p "$CD"
  WS=$(date +%s.%N)
  ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --served-model-name flash-next --dataset-name random --random-input-len "$IN" --random-output-len "$OUTL" --num-prompts "$N" --max-concurrency "$N" --ignore-eos --request-rate inf --seed "$SEED" --temperature 0 --save-result --result-dir "$CD" ) >"$CD.log" 2>&1
  WE=$(date +%s.%N)
  echo "$N $IN $OUTL $SEED $WS $WE" >>"$D/cell_windows.txt"
  log "cell c=$N in=$IN | $(grep -m1 'Output token throughput' "$CD.log" | tr -s ' ') | $(grep -m1 'Mean TPOT' "$CD.log" | tr -s ' ')"
  S2=$(sel_count); if [ "$S2" -gt "$S0" ]; then echo "STOP NEW PCI SERR" >"$RUN/IM_STOP"; break; fi
done

# The injected rocprofiler tool runs its finalizer from the SIGTERM handler and
# can take minutes to flush a multi-hundred-MB per-process trace; wait for the
# CSVs before reaping, or the flush is aborted and the run is lost.
log "graceful stop: SIGTERM to campaign pids"
for p in $(our_pids); do kill -TERM "$p" 2>/dev/null; done
for i in $(seq 1 300); do
  n=$(ls "$D"/prof_*_kernel_trace.csv 2>/dev/null | wc -l)
  [ "$n" -ge 4 ] && break
  sleep 3
done
sleep 20
n=$(ls "$D"/prof_*_kernel_trace.csv 2>/dev/null | wc -l)
log "CSV count after flush: $n"
hygiene
S3=$(sel_count); log "PCI-SERR after teardown=$S3 uptime=$(uptime -p)"
log "arm $ARM ctx $CTX done"
