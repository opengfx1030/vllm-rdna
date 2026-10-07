#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Flash-Next TunableOp campaign driver: capture the observed GEMM shapes and
# validate the frozen rows across the serving matrix.
#
#   MODE=capture   boot with record_untuned_enable=1 pointed at a scratch file
#                  (PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 +
#                   PYTORCH_TUNABLEOP_UNTUNED_FILENAME=<dir>/untuned.csv).
#                  NOTE: record mode DISABLES the results lookup, so the cells
#                  run on rocBLAS heuristics -- they are shape discovery, not
#                  performance validation. Writes shapes_<arm>.txt.
#   MODE=validate  boot with the fork rows via tunableop_env.sh (lookup-only,
#                  no record), run the measured cells; those ARE validation.
#                  Writes cells.csv + coherence + markers.
#
# Env: MODE, ARMS (space list), OUT, MODEL, CELLS, PORT base.
# No /tmp. Captures + logs live under /home/chenco_adm/w4a8_runs/.
set -uo pipefail

MODE=${MODE:-capture}
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
T=/home/chenco_adm/vllm-rdna-0.28.0
MODEL=${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
OUT=${OUT:-/home/chenco_adm/w4a8_runs}
CELLS=${CELLS:-"1 1024 512 221|8 1024 512 222|1 16384 1024 111|8 16384 1024 112"}
ARMS=${ARMS:-"w4a16-fa-m2"}
CAP_ROOT=$OUT/_captures
TRITON_TEMPLATE=${TRITON_TEMPLATE:-$T/cache/triton}

ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DRIVER_LOG=$OUT/campaign_${MODE}.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }

our_pids() {
  local p exe
  for p in $(pgrep -f "entrypoints.cli.main serve|entrypoints.openai.api_server|VLLM::Worker|VLLM::EngineCore|PleOffloadWorker" 2>/dev/null); do
    exe=$(readlink -f "/proc/$p/exe" 2>/dev/null || true)
    case "$exe" in */venv-7.14.0_0.28.0/bin/python*) echo "$p" ;; esac
  done
}

hygiene() {
  local n=0 left p
  for p in $(our_pids); do kill -TERM "$p" 2>/dev/null && n=$((n + 1)); done
  [ "$n" -gt 0 ] && sleep 8
  for p in $(our_pids); do kill -KILL "$p" 2>/dev/null; done
  sleep 2
  left=$(our_pids | tr '\n' ' ')
  log "hygiene: terminated=$n remaining_our=[${left% }]"
  [ -n "${left// /}" ] && return 1
  return 0
}

sel_count() { timeout 10 sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }
sel_snap() {
  local tag=$1 c
  c=$(sel_count)
  log "SEL[$tag] PCI-SERR=$c base=${SERR0:-?} uptime=$(uptime -p)"
  if [ -n "${SERR0:-}" ] && [ "${c:-0}" -gt "$SERR0" ]; then
    echo "STOP NEW PCI SERR after $tag ($c > $SERR0)" >"$OUT/STOP"
    log "STOP: new PCI SERR after $tag"; return 1
  fi
  return 0
}

teardown() {
  local api p
  api=$(grep -oE "APIServer pid=[0-9]+" "$D/serve.log" 2>/dev/null | head -1 | cut -d= -f2)
  [ -n "$api" ] && { kill -TERM "$api" 2>/dev/null; sleep 10; }
  for p in $(our_pids); do kill -TERM "$p" 2>/dev/null; done
  sleep 6
  for p in $(our_pids); do kill -KILL "$p" 2>/dev/null; done
  sleep 2
}

run_arm() {
  local arm=$1
  case "$arm" in
    w4a16-fa-m0)     W4A8=0 MTP=0 ATTN=fa     PORT=18300 ;;
    w4a16-fa-m2)     W4A8=0 MTP=2 ATTN=fa     PORT=18301 ;;
    w4a16-triton-m0) W4A8=0 MTP=0 ATTN=triton PORT=18302 ;;
    w4a16-triton-m2) W4A8=0 MTP=2 ATTN=triton PORT=18303 ;;
    w4a8-fa-m0)      W4A8=1 MTP=0 ATTN=fa     PORT=18304 ;;
    w4a8-fa-m2)      W4A8=1 MTP=2 ATTN=fa     PORT=18305 ;;
    w4a8-triton-m0)  W4A8=1 MTP=0 ATTN=triton PORT=18306 ;;
    w4a8-triton-m2)  W4A8=1 MTP=2 ATTN=triton PORT=18307 ;;
    *) log "unknown arm: $arm"; return 1 ;;
  esac
  TAG=${TAG_OVERRIDE:-2026-09-30_${MODE}_$arm}
  local CACHE=$T/cache/arm-${MODE}-$arm
  D=$OUT/$TAG
  local CAP_DIR=$CAP_ROOT/$arm
  mkdir -p "$D/cells" "$CAP_DIR"

  log "=== ARM=$arm MODE=$MODE W4A8=$W4A8 MTP=$MTP ATTN=$ATTN TAG=$TAG PORT=$PORT ==="
  hygiene || { log "aborting: hygiene failed"; return 1; }
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then log "aborting: port $PORT bound"; return 1; fi
  rm -rf "$CACHE"
  mkdir -p "$CACHE/inductor" "$CACHE/extensions"
  if [ -d "$TRITON_TEMPLATE" ]; then
    cp -a "$TRITON_TEMPLATE" "$CACHE/triton"
    log "triton cache seeded ($(ls "$CACHE/triton" | wc -l) entries)"
  else
    mkdir -p "$CACHE/triton"
  fi
  rm -f "$CACHE/rdna_ar_wedged"
  SERR0=$(sel_count)
  log "arm cache=$CACHE baseline PCI-SERR=$SERR0"
  sel_snap preflight || return 2

  local EXTRA_ENV=()
  if [ "$MODE" = "capture" ]; then
    rm -f "$CAP_DIR"/untuned*.csv "$CAP_DIR"/untuned.csv
    EXTRA_ENV=(
      PYTORCH_TUNABLEOP_RECORD_UNTUNED=1
      PYTORCH_TUNABLEOP_UNTUNED_FILENAME="$CAP_DIR/untuned.csv"
      PYTORCH_TUNABLEOP_TUNING=0
    )
  fi

  T0=$(date +%s)
  setsid nohup env \
    VENV="$V" VLLM_TREE="$T" MTP="$MTP" ATTN="$ATTN" TP=4 PORT="$PORT" MODEL="$MODEL" \
    VLLM_CACHE_ROOT="$CACHE" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
    TRITON_CACHE_DIR="$CACHE/triton" TORCH_EXTENSIONS_DIR="$CACHE/extensions" \
    VLLM_RDNA2_W4A8_SDOT4="$W4A8" \
    VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=64 VLLM_RDNA_AR_ONESHOT_KB=64 \
    VLLM_RDNA_AR_BLOCKS=0 VLLM_RDNA_AR_PACE=0 VLLM_FORCE_CUSTOM_ALL_REDUCE=0 \
    NCCL_P2P_LEVEL=pxb RCCL_P2P_NET_DISABLE=1 RCCL_P2P_BATCH_ENABLE=1 \
    NCCL_PROTO=Simple RCCL_MSCCL_ENABLE=0 PYTHONFAULTHANDLER=1 \
    HIP_VISIBLE_DEVICES=0,1,2,3 "${EXTRA_ENV[@]}" \
    bash "$T/tools/rdna/serve_gfx1030_flashnext_mtp.sh" >"$D/serve.log" 2>&1 </dev/null &
  log "launched arm=$arm cache=$CACHE log=$D/serve.log"

  local ready=0 dead=0 i
  for i in $(seq 1 90); do
    sleep 10
    curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q flash-next && { ready=1; break; }
    [ -f "$OUT/STOP" ] && break
    if pgrep -f "[a]pi_server" >/dev/null; then dead=0; else dead=$((dead + 1)); fi
    if [ "$dead" -ge 3 ]; then log "server gone during boot (after $((i*10))s)"; break; fi
  done
  local COLD=$(( $(date +%s) - T0 ))
  if [ "$ready" != "1" ]; then
    log "NOT READY after ${COLD}s"; tail -30 "$D/serve.log" | tee -a "$DRIVER_LOG" | tail -20
    teardown; hygiene; echo "NOT_READY" >"$D/status.txt"; return 1
  fi
  log "READY t=${COLD}s"; echo "$COLD" >"$D/cold_compile_seconds.txt"
  sel_snap after_boot || { teardown; hygiene; return 1; }

  # Warmup (throwaway) then coherence.
  for wl in "1024 64 999" "16384 64 998"; do
    set -- $wl
    ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
      --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
      --model "$MODEL" --served-model-name flash-next --dataset-name random \
      --random-input-len "$1" --random-output-len "$2" --num-prompts 1 --max-concurrency 1 \
      --ignore-eos --request-rate inf --seed "$3" --temperature 0 \
      --save-result --result-dir "$D/warmup_$1" ) >"$D/warmup_$1.log" 2>&1
  done
  log "warmup done"
  "$V/bin/python" "$T/tools/rdna2_028/probe_w4a8.py" "http://127.0.0.1:$PORT/v1" flash-next 1 >"$D/coherence.txt" 2>&1
  log "coherence rc=$? $(grep -c 'OK ' "$D/coherence.txt")/$(grep -cE '^\[' "$D/coherence.txt") OK"

  # Measured cells.
  local cell
  IFS='|' read -ra _cells <<<"$CELLS"
  for cell in "${_cells[@]}"; do
    set -- $cell
    local N=$1 IN=$2 OUTL=$3 SEED=$4
    local CD="$D/cells/c${IN}_${N}"
    log "bench c=$N in=$IN start"
    ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
      --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
      --model "$MODEL" --served-model-name flash-next --dataset-name random \
      --random-input-len "$IN" --random-output-len "$OUTL" --num-prompts "$N" --max-concurrency "$N" \
      --ignore-eos --request-rate inf --seed "$SEED" --temperature 0 \
      --save-result --result-dir "$CD" ) >"$CD.log" 2>&1
    log "bench c=$N in=$IN done: $(grep -m1 'Output token throughput' "$CD.log" | tr -s ' ')"
    "$V/bin/python" - "$PORT" "$CD" <<'PY' >"$CD/coherence.txt" 2>&1
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
def ask(p, mt=16):
    body = json.dumps({"model": "flash-next", "prompt": p, "max_tokens": mt,
                       "temperature": 0.0, "ignore_eos": True}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as resp:
        return json.load(resp)["choices"][0]["text"] or ""
for name, p, exp in [("france", "The capital of France is", "paris"), ("math", "2 + 2 =", "4")]:
    try: t = ask(p)
    except Exception as e: t = f"<ERROR {e!r}>"
    print(f"[{name}] coherent={exp in t.lower()} | {t[:120]!r}")
PY
    sel_snap "after_c${N}_${IN}" || { teardown; hygiene; return 1; }
  done

  # Markers.
  {
    echo "=== W4A8 MoE marker (expected iff W4A8=1) ==="
    grep -m4 "W4A8\|w4a8\|sdot4" "$D/serve.log" | grep -v "W4A8-CALLTRACE" | head -8 || echo "(none)"
    echo "=== TunableOp files ==="
    grep -h "reading tuning results from\|TunableOp lookup\|record" "$D/serve.log" | sort -u | head -6 || echo "(none)"
    echo "=== rdna_ar ==="
    grep -h "rdna_ar:" "$D/serve.log" | sort -u | head -4 || echo "(none)"
    echo "=== attention backend ==="
    grep -h "RDNA_ATTN\|TRITON_ATTN\|attention backend" "$D/serve.log" | sort -u | head -4 || echo "(none)"
  } >"$D/markers.txt" 2>&1
  tail -12 "$D/markers.txt" | tee -a "$DRIVER_LOG"

  # Mode-specific artifacts.
  if [ "$MODE" = "capture" ]; then
    local raw=0
    for f in "$CAP_DIR"/untuned[0-9].csv; do [ -f "$f" ] && raw=$((raw + $(wc -l <"$f"))); done
    cat "$CAP_DIR"/untuned[0-9].csv 2>/dev/null | awk -F, '$1 ~ /^Gemm/ {print $1","$2}' | sort -u >"$CAP_DIR/shapes_$arm.txt"
    cp "$CAP_DIR"/untuned[0-9].csv "$CAP_DIR/" 2>/dev/null || true
    log "capture: raw lines=$raw unique keys=$(wc -l <"$CAP_DIR/shapes_$arm.txt") -> shapes_$arm.txt"
  else
    "$V/bin/python" "$T/tools/rdna2_028/flashnext_w4a8_extract.py" "$D" >"$D/cells.csv" 2>"$D/extract.err" || true
    cat "$D/cells.csv" | tee -a "$DRIVER_LOG" | tail -6
    grep "SpecDecoding metrics" "$D/serve.log" >"$D/acceptance.txt" 2>/dev/null || true
  fi

  teardown
  hygiene
  sel_snap after_teardown || true
  log "done $arm (W4A8=$W4A8 MTP=$MTP ATTN=$ATTN)"
  echo "OK" >"$D/status.txt"
}

# --- Driver ----------------------------------------------------------------
SERR0=$(sel_count)
log "=== campaign_${MODE} start ARMS=[$ARMS] PCI-SERR=$SERR0 uptime=$(uptime -p) ==="
hygiene || { log "hygiene failed at start"; exit 1; }

rc=0
for arm in $ARMS; do
  log "---------------------------------------------------------------"
  if ! run_arm "$arm"; then
    log "ARM FAIL: $arm"
    rc=1
    grep -q "PCI SERR" "$OUT/STOP" 2>/dev/null && { log "stopping: PCI SERR"; break; }
    hygiene || { log "hygiene failed after $arm"; break; }
  fi
done

SERR1=$(sel_count)
log "=== campaign_${MODE} done rc=$rc PCI-SERR=$SERR1 (base $SERR0) ==="
[ "$SERR1" -gt "$SERR0" ] && { echo "STOP NEW PCI SERR" >"$OUT/STOP"; log "STOP: new PCI SERR"; exit 2; }
echo "DONE_$MODE rc=$rc" >"$OUT/campaign_${MODE}.status"
exit $rc
