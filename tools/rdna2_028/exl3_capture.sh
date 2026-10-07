#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# EXL3 27B TunableOp shape-capture driver.
#
# Boots the EXL3 27B launcher (scripts/serve_gfx1030_exl3_27b.sh) once per arm
# with the record-untuned census enabled:
#     PYTORCH_TUNABLEOP_RECORD_UNTUNED=1
#     PYTORCH_TUNABLEOP_UNTUNED_FILENAME=<cap_dir>/untuned.csv
# record_untuned DISABLES the results lookup, so the whole fp16 GEMM stream is
# recorded (not looked up) and the cells run on rocBLAS heuristics. This is
# shape discovery only -- a capture boot can never be a validation boot.
#
#   MODE=capture (only mode). ARMS default "exl3-m0 exl3-m2".
#   Cells: 1k/512 + 16k/1k at c=1 and c=8, short (64-token) outputs so the
#   decode/verify shape families fire without paying long generation.
#
# No /tmp. GPUs default 4-7 (the Flash-Next co-tenant owns 0-3).
set -uo pipefail

V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
T=/home/chenco_adm/vllm-rdna-0.28.0
MODEL=${MODEL:-/home/chenco_adm/models/Qwen3.8-27B-exl3-3.00bpw}
OUT=${OUT:-/home/chenco_adm/w4a8_runs}
CAP_ROOT=${CAP_ROOT:-$OUT/_captures-exl3}
CELLS=${CELLS:-"1 1024 64 221|8 1024 64 222|1 16384 64 111|8 16384 64 112"}
WARM=${WARM:-"1024 32 999|16384 32 998"}
ARMS=${ARMS:-"exl3-m0 exl3-m2"}
GPUS=${GPUS:-4,5,6,7}

ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DRIVER_LOG=$OUT/exl3_capture.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }

# --- our EXL3 api_server / orphaned engine group only ---------------------
# A crashed api_server leaves orphaned VLLM::EngineCore + VLLM::Worker_TP* whose
# cmdline is just the setproctitle. Identify them by the stdout log path (this
# driver's run dir), which is unique to our captures; the co-tenant's log lives
# elsewhere and is never matched.
our_pids() {
  local p fd
  for p in $(pgrep -f "entrypoints.openai.api_server.*exl3-27b-mul1" 2>/dev/null); do echo "$p"; done
  for p in $(pgrep -f "VLLM::EngineCore|VLLM::Worker_TP" 2>/dev/null); do
    fd=$(readlink "/proc/$p/fd/1" 2>/dev/null || true)
    case "$fd" in "$OUT"/2026-10-01_exl3-capture-*/serve.log) echo "$p" ;; esac
  done
}
our_pgids() {
  local p
  for p in $(our_pids); do ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '; done | sort -u
}
hygiene() {
  local pg n=0
  for pg in $(our_pgids); do
    kill -TERM "-$pg" 2>/dev/null && n=$((n + 1))
  done
  [ "$n" -gt 0 ] && sleep 8
  for pg in $(our_pgids); do kill -KILL "-$pg" 2>/dev/null; done
  sleep 2
  local left; left=$(our_pgids | tr '\n' ' ')
  log "hygiene: terminated_groups=$n remaining=[${left% }]"
  [ -n "${left// /}" ] && return 1
  return 0
}
cleanup() {
  local p
  for p in $(our_pids); do kill -TERM "$p" 2>/dev/null; done
  sleep 5
  for p in $(our_pids); do kill -KILL "$p" 2>/dev/null; done
  return 0
}
trap cleanup EXIT
sel_count() { timeout 8 sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }
sel_snap() {
  local tag=$1 c; c=$(sel_count)
  log "SEL[$tag] PCI-SERR=$c base=${SERR0:-?} uptime=$(uptime -p)"
  if [ -n "${SERR0:-}" ] && [ "$c" -gt "$SERR0" ]; then
    echo "STOP NEW PCI SERR after $tag ($c > $SERR0)" >"$OUT/STOP"
    log "STOP: new PCI SERR after $tag"; return 1
  fi
}

run_arm() {
  local arm=$1
  case "$arm" in
    exl3-m0) MTP=0 PORT=18400 TAG=2026-10-01_exl3-capture-m0 ;;
    exl3-m2) MTP=2 PORT=18401 TAG=2026-10-01_exl3-capture-m2 ;;
    *) log "unknown arm: $arm"; return 1 ;;
  esac
  local CACHE=$T/cache/arm-cap-$arm
  local D=$OUT/$TAG
  local CAP_DIR=$CAP_ROOT/$arm
  mkdir -p "$D/cells" "$CAP_DIR"

  log "=== ARM=$arm MTP=$MTP TAG=$TAG PORT=$PORT ==="
  hygiene || { log "aborting: hygiene failed"; return 1; }
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then log "aborting: port $PORT bound"; return 1; fi
  [ "${FRESH_CACHE:-0}" = "1" ] && rm -rf "$CACHE"
  mkdir -p "$CACHE/inductor" "$CACHE/extensions" "$CACHE/triton"
  rm -f "$CACHE/rdna_ar_wedged"
  rm -f "$CAP_DIR"/untuned*.csv
  SERR0=$(sel_count)
  log "cache=$CACHE capture_dir=$CAP_DIR baseline PCI-SERR=$SERR0"
  sel_snap preflight || return 2

  T0=$(date +%s)
  setsid nohup env \
    VENV="$V" VLLM_TREE="$T" MTP="$MTP" PORT="$PORT" MODEL="$MODEL" EAGER=0 \
    HIP_VISIBLE_DEVICES="$GPUS" \
    VLLM_CACHE_ROOT="$CACHE/vllm" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
    TRITON_CACHE_DIR="$CACHE/triton" TORCH_EXTENSIONS_DIR="$CACHE/extensions" \
    TUNABLEOP=1 \
    PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 \
    PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 \
    PYTORCH_TUNABLEOP_UNTUNED_FILENAME="$CAP_DIR/untuned.csv" \
    PYTHONFAULTHANDLER=1 \
    bash "$T/scripts/serve_gfx1030_exl3_27b.sh" >"$D/serve.log" 2>&1 </dev/null &
  log "launched EXL3 arm=$arm log=$D/serve.log"

  local ready=0 dead=0 i
  for i in $(seq 1 120); do
    sleep 10
    curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q exl3-27b-mul1 && { ready=1; break; }
    [ -f "$OUT/STOP" ] && break
    if [ -n "$(our_pids)" ]; then dead=0; else dead=$((dead + 1)); fi
    if [ "$dead" -ge 3 ]; then log "server gone during boot (after $((i*10))s)"; break; fi
  done
  local COLD=$(( $(date +%s) - T0 ))
  if [ "$ready" != "1" ]; then
    log "NOT READY after ${COLD}s"
    tail -40 "$D/serve.log" | tee -a "$DRIVER_LOG"
    hygiene
    echo "NOT_READY" >"$D/status.txt"
    return 1
  fi
  log "READY t=${COLD}s"; echo "$COLD" >"$D/cold_compile_seconds.txt"
  sel_snap after_boot || { hygiene; return 1; }

  # Warmup (throwaway) then coherence (chat endpoint).
  local wl
  IFS='|' read -ra _warm <<<"$WARM"
  for wl in "${_warm[@]}"; do
    set -- $wl
    log "warmup in=$1 out=$2"
    ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
      --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
      --model "$MODEL" --served-model-name exl3-27b-mul1 --dataset-name random \
      --random-input-len "$1" --random-output-len "$2" --num-prompts 1 --max-concurrency 1 \
      --ignore-eos --request-rate inf --seed "$3" --temperature 0 \
      --save-result --result-dir "$D/warmup_$1" ) >"$D/warmup_$1.log" 2>&1
  done
  log "warmup done"
  "$V/bin/python" - "$PORT" >"$D/coherence.txt" 2>&1 <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
def chat(prompt, mt=48):
    body = json.dumps({"model": "exl3-27b-mul1",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": mt, "temperature": 0.0}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as resp:
        return json.load(resp)["choices"][0]["message"]["content"] or ""
ok = True
for name, prompt, expect in [("france", "The capital of France is", "paris"),
                             ("math", "2 + 2 =", "4")]:
    try:
        t = chat(prompt)
    except Exception as e:
        t = f"<ERROR {e!r}>"
    good = expect in t.lower()
    ok = ok and good
    print(f"[{name}] coherent={good} | {t[:120]!r}")
print("COHERENCE_RESULT:", "PASS" if ok else "FAIL")
PY
  log "coherence rc=$? | $(grep -c 'coherent=True' "$D/coherence.txt")/2"

  # Measured (shape-firing) cells.
  local cell
  IFS='|' read -ra _cells <<<"$CELLS"
  for cell in "${_cells[@]}"; do
    set -- $cell
    local N=$1 IN=$2 OUTL=$3 SEED=$4
    local CD="$D/cells/c${IN}_${N}"
    log "bench c=$N in=$IN start"
    ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
      --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
      --model "$MODEL" --served-model-name exl3-27b-mul1 --dataset-name random \
      --random-input-len "$IN" --random-output-len "$OUTL" --num-prompts "$N" --max-concurrency "$N" \
      --ignore-eos --request-rate inf --seed "$SEED" --temperature 0 \
      --save-result --result-dir "$CD" ) >"$CD.log" 2>&1
    log "bench c=$N in=$IN done: $(grep -m1 'Output token throughput' "$CD.log" | tr -s ' ')"
    sel_snap "after_c${N}_${IN}" || { hygiene; return 1; }
  done

  # Capture artefacts.
  local raw=0
  for f in "$CAP_DIR"/untuned[0-9].csv; do [ -f "$f" ] && raw=$((raw + $(wc -l <"$f"))); done
  cat "$CAP_DIR"/untuned[0-9].csv 2>/dev/null | awk -F, '$1 ~ /^Gemm/ {print $1","$2}' | sort -u >"$CAP_DIR/shapes_$arm.txt"
  log "capture: raw lines=$raw unique keys=$(wc -l <"$CAP_DIR/shapes_$arm.txt") -> shapes_$arm.txt"
  {
    echo "=== TunableOp messages ==="
    grep -h "TunableOp\|record_untuned\|tuning results" "$D/serve.log" | sort -u | head -10 || echo "(none)"
    echo "=== attention backend ==="
    grep -m2 "RDNA_ATTN\|attention backend" "$D/serve.log" | head -4 || echo "(none)"
    echo "=== rdna_ar ==="
    grep -h "rdna_ar:" "$D/serve.log" | sort -u | head -4 || echo "(none)"
  } >"$D/markers.txt" 2>&1
  tail -6 "$D/markers.txt" | tee -a "$DRIVER_LOG"

  hygiene
  sel_snap after_teardown || true
  log "done $arm"
  echo "OK" >"$D/status.txt"
}

# --- driver ---------------------------------------------------------------
SERR0=$(sel_count)
log "=== exl3 capture start ARMS=[$ARMS] GPUs=$GPUS PCI-SERR=$SERR0 uptime=$(uptime -p) ==="
hygiene || { log "hygiene failed at start"; exit 1; }

rc=0
for arm in $ARMS; do
  log "---------------------------------------------------------------"
  if ! run_arm "$arm"; then
    log "ARM FAIL: $arm (continuing)"
    rc=1
    grep -q "PCI SERR" "$OUT/STOP" 2>/dev/null && { log "stopping: PCI SERR"; break; }
    hygiene || { log "hygiene failed after $arm"; break; }
  fi
done

SERR1=$(sel_count)
log "=== exl3 capture done rc=$rc PCI-SERR=$SERR1 (base $SERR0) ==="
[ "$SERR1" -gt "$SERR0" ] && { echo "STOP NEW PCI SERR" >"$OUT/STOP"; log "STOP: new PCI SERR"; exit 2; }

echo "=== capture summary ==="
for arm in $ARMS; do
  f=$CAP_ROOT/$arm/shapes_$arm.txt
  [ -f "$f" ] && echo "$arm: $(wc -l <"$f") unique Gemm keys"
done
exit $rc
