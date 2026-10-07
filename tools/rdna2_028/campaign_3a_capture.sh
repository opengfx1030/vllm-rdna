#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# STEP 3 capture driver: 4-config x MTP=2 capture with record_untuned_enable.
#
# For each of {W4A16,W4A8} x {FA-RDNA2,Triton AMD FA}:
#   * boot Flash-Next with the FROZEN rows active + record_untuned_enable=True
#     pointed at a scratch filename. Any shape NOT in the 719-row set will be
#     recorded (not tuned inline) for the offline tune pass.
#   * run the four standard cells (c=1/c=8 x 1k/512 + 16k/1k). The cells data
#     IS the STEP 4 validation, so this single pass serves both goals when no
#     new shapes are captured.
#
# Outputs (per arm, under /home/chenco_adm/w4a8_runs/<TAG>/):
#   serve.log, driver.log, coherence.txt, cells/c{IN}_{N}/{log, coherence.txt}
#   untuned_captures.csv -- every GemmTunableOp key not in the frozen rows
#   cells.csv, kfd.log, markers.txt
#
# No /tmp. Scratch rows live under /home/chenco_adm/w4a8_runs/_captures/<arm>/.
set -uo pipefail

V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
T=/home/chenco_adm/vllm-rdna-0.28.0
MODEL=${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
OUT=${OUT:-/home/chenco_adm/w4a8_runs}
ROWS=$T/tunableop/rocblas-f30bb442e9b5
CAP_ROOT=$OUT/_captures
CELLS=${CELLS:-"1 1024 512 221|8 1024 512 222|1 16384 1024 111|8 16384 1024 112"}
# Arms to run in order. Default: all 4 at MTP=2 (the superset over MTP=0).
ARMS=${ARMS:-"w4a16-fa-m2 w4a16-triton-m2 w4a8-fa-m2 w4a8-triton-m2"}

ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/roCM_SYSDEPS/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/campaign_3a.log"; }

# --- OUR processes only (0.28.0 venv) --------------------------------------
our_pids() {
  local p exe
  for p in $(pgrep -f "entrypoints.cli.main serve|entrypoints.openai.api_server|VLLM::Worker|VLLM::EngineCore|PleOffloadWorker" 2>/dev/null); do
    exe=$(readlink -f "/proc/$p/exe" 2>/dev/null || true)
    case "$exe" in
      */venv-7.14.0_0.28.0/bin/python*) echo "$p" ;;
    esac
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

sel_count() { timeout 8 sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }

teardown() {
  local api p
  api=$(grep -oE "APIServer pid=[0-9]+" "$D/serve.log" 2>/dev/null | head -1 | cut -d= -f2)
  [ -n "$api" ] && kill -TERM "$api" 2>/dev/null && sleep 10
  for p in $(our_pids); do kill -TERM "$p" 2>/dev/null; done
  sleep 6
  for p in $(our_pids); do kill -KILL "$p" 2>/dev/null; done
  sleep 2
}

run_arm() {
  local arm=$1
  case "$arm" in
    w4a16-fa-m2)        W4A8=0 MTP=2 ATTN=fa PORT=18290 TAG=2026-09-30_w4a16-fa-m2-cap ;;
    w4a16-triton-m2)    W4A8=0 MTP=2 ATTN=triton PORT=18291 TAG=2026-09-30_w4a16-triton-m2-cap ;;
    w4a8-fa-m2)         W4A8=1 MTP=2 ATTN=fa PORT=18292 TAG=2026-09-30_w4a8-fa-m2-cap ;;
    w4a8-triton-m2)     W4A8=1 MTP=2 ATTN=triton PORT=18293 TAG=2026-09-30_w4a8-triton-m2-cap ;;
    *) log "unknown arm: $arm"; return 1 ;;
  esac

  local ARM=w4a8$([ "$W4A8" = 1 ] && echo on || echo off)
  local CACHE=$T/cache/arm-cap-$arm
  local TRITON_TEMPLATE=$T/cache/triton
  local D=$OUT/$TAG
  local CAP_DIR=$CAP_ROOT/$arm
  mkdir -p "$D/cells" "$CAP_DIR"

  log "=== ARM=$arm W4A8=$W4A8 ATTN=$ATTN MTP=$MTP TAG=$TAG PORT=$PORT ==="
  hygiene || { log "aborting: hygiene failed"; return 1; }
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    log "aborting: port $PORT bound"; return 1
  fi
  rm -rf "$CACHE"
  mkdir -p "$CACHE/inductor" "$CACHE/extensions"
  if [ -d "$TRITON_TEMPLATE" ]; then
    cp -a "$TRITON_TEMPLATE" "$CACHE/triton"
    log "triton cache seeded from $TRITON_TEMPLATE ($(ls "$CACHE/triton" | wc -l) entries)"
  else
    mkdir -p "$CACHE/triton"
  fi
  rm -f "$CACHE/rdna_ar_wedged"
  local SERR0; SERR0=$(sel_count)
  log "arm cache=$CACHE baseline PCI-SERR=$SERR0 capture_dir=$CAP_DIR"
  if [ "$SERR0" -gt "${CAMP_SERR_BASE:-100}" ]; then
    log "STOP: PCI-SERR baseline $SERR0 > CAMP_SERR_BASE ${CAMP_SERR_BASE:-100}"; return 2
  fi

  local SCRATCH_DIR=$CAP_DIR/scratch
  mkdir -p "$SCRATCH_DIR"
  for r in 0 1 2 3; do
    cp "$ROWS/tunableop_results${r}.csv" "$SCRATCH_DIR/tunableop_results${r}.csv"
  done

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
    HIP_VISIBLE_DEVICES=0,1,2,3 \
    PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 \
    PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 TORCH_BLAS_PREFER_HIPBLASLT=0 \
    PYTORCH_TUNABLEOP_FILENAME="$SCRATCH_DIR/tunableop_results%d.csv" \
    PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 \
    bash "$T/scripts/serve_gfx1030_flashnext_mtp.sh" >"$D/serve.log" 2>&1 </dev/null &
  log "flashnext launched cache=$CACHE log=$D/serve.log scratch=$SCRATCH_DIR"

  # --- Wait for ready ---
  local ready=0 dead=0
  for i in $(seq 1 90); do
    sleep 10
    curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q flash-next && { ready=1; break; }
    [ -f "$D/STOP" ] && break
    if pgrep -f "[a]pi_server" >/dev/null; then dead=0; else dead=$((dead + 1)); fi
    if [ "$dead" -ge 3 ]; then log "server process gone during boot (after $((i*10))s)"; break; fi
  done
  local COLD=$(( $(date +%s) - T0 ))
  if [ "$ready" != "1" ]; then
    log "NOT READY after ${COLD}s"
    tail -40 "$D/serve.log" | tee -a "$D/driver.log"
    teardown; hygiene
    echo "NOT_READY" >"$D/status.txt"
    return 1
  fi
  log "READY t=${COLD}s"
  echo "$COLD" >"$D/cold_compile_seconds.txt"

  # --- Warmup: throwaway 1k c=1 to clear cold-start pollution ---
  for wl in "1024 64 999" "16384 64 998"; do
    set -- $wl
    log "warmup in=$1 out=$2"
    ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
      --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
      --model "$MODEL" --served-model-name flash-next --dataset-name random \
      --random-input-len "$1" --random-output-len "$2" --num-prompts 1 --max-concurrency 1 \
      --ignore-eos --request-rate inf --seed "$3" --temperature 0 \
      --save-result --result-dir "$D/warmup_$1" ) >"$D/warmup_$1.log" 2>&1
  done
  log "warmup done"

  # --- Coherence probe ---
  log "coherence probe start"
  "$V/bin/python" "$T/tools/rdna2_028/probe_w4a8.py" "http://127.0.0.1:$PORT/v1" flash-next 1 >"$D/coherence.txt" 2>&1
  log "coherence probe done rc=$?"
  grep -E "^\[|OK /" "$D/coherence.txt" | tee -a "$D/driver.log"

  # --- Measured cells ---
  IFS='|' read -ra _cells <<<"$CELLS"
  echo "arm,dir,n,in_len,out_len,ok,dur_s,in_tok,out_tok,ttft_mean_ms,ttft_med_ms,ttft_p99_ms,tpot_mean_ms,itl_med_ms,itl_p99_ms,out_tok_s_agg,total_tok_s,decode_tok_s_per_req,prefill_tok_s" >"$D/cells.csv"
  for cell in "${_cells[@]}"; do
    set -- $cell
    N=$1; IN=$2; OUTL=$3; SEED=$4
    local CD="$D/cells/c${IN}_${N}"
    log "bench c=$N in=$IN out=$OUTL start"
    ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
      --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
      --model "$MODEL" --served-model-name flash-next --dataset-name random \
      --random-input-len "$IN" --random-output-len "$OUTL" --num-prompts "$N" --max-concurrency "$N" \
      --ignore-eos --request-rate inf --seed "$SEED" --temperature 0 \
      --save-result --result-dir "$CD" ) >"$CD.log" 2>&1
    log "bench c=$N in=$IN done: $(grep -m1 'Output token throughput' "$CD.log" | tr -s ' ')"
    # Per-cell coherence (France -> Paris, 2+2 -> 4)
    "$V/bin/python" - "$PORT" "$CD" <<'PY' >"$CD/coherence.txt" 2>&1
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
def ask(prompt, mt=16):
    body = json.dumps({"model": "flash-next", "prompt": prompt, "max_tokens": mt,
                       "temperature": 0.0, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["text"] or ""
for name, prompt, expect in [("france", "The capital of France is", "paris"),
                             ("math", "2 + 2 =", "4")]:
    try:
        t = ask(prompt)
    except Exception as e:
        t = f"<ERROR {e!r}>"
    ok = expect in t.lower()
    print(f"[{name}] coherent={ok} | {t[:120]!r}")
PY
  done

  # --- Markers ---
  {
    echo "=== 'W4A8 sdot4 path active' (expected iff W4A8=1 AND a dense quantized linear ran) ==="
    grep -m4 "W4A8 sdot4 path active" "$D/serve.log" || echo "(absent)"
    echo "=== 'W4A8-DEBUG' fast-path shapes ==="
    grep -h "W4A8-DEBUG" "$D/serve.log" | sort -u | head -30 || echo "(none)"
    echo "=== rdna_ar lines (expected: one-shot active on every rank) ==="
    grep -h "rdna_ar:" "$D/serve.log" | sort -u || echo "(none)"
    echo "=== all-reduce backend selection ==="
    grep -m2 "all-reduce backends" "$D/serve.log" || echo "(none)"
    echo "=== attention backend ==="
    grep -m2 "attention backend" "$D/serve.log" | head -4 || echo "(none)"
    echo "=== TunableOp record_untuned_enable messages ==="
    grep -h "record_untuned\|TunableOp" "$D/serve.log" | sort -u | head -10 || echo "(none)"
  } >"$D/markers.txt" 2>&1
  tee -a "$D/driver.log" <"$D/markers.txt"

  if [ -d "$SCRATCH_DIR" ]; then
    for r in 0 1 2 3; do
      f=$SCRATCH_DIR/tunableop_results${r}.csv
      if [ -f "$f" ]; then
        local cap_lines; cap_lines=$(wc -l <"$f")
        local gemm_keys; gemm_keys=$(grep -c "^GemmTunableOp" "$f" 2>/dev/null || echo 0)
        log "capture rank $r: $f ($cap_lines lines, $gemm_keys gemm keys)"
      fi
    done
  else
    log "capture scratch dir: $SCRATCH_DIR ABSENT"
  fi

  # --- Teardown ---
  teardown
  sleep 3
  local left; left=$(our_pids | tr '\n' ' ')
  log "teardown done remaining_our=[${left% }]"
  echo "OK" >"$D/status.txt"
}

# --- Driver loop -----------------------------------------------------------
SERR0=$(sel_count)
log "=== campaign 3a start: ARMS=$ARMS PCI-SERR=$SERR0 ==="
log "preflight uptime=$(uptime -p) host=$(hostname)"
hygiene || { log "hygiene failed at start"; exit 1; }

local_rc=0
for arm in $ARMS; do
  log "---------------------------------------------------------------"
  if ! run_arm "$arm"; then
    log "ARM FAIL: $arm (continuing)"
    local_rc=1
    hygiene || { log "hygiene failed after $arm"; break; }
  fi
done

SERR1=$(sel_count)
log "=== campaign 3a done: rc=$local_rc PCI-SERR=$SERR1 (base $SERR0) ==="
[ "$SERR1" -gt "$SERR0" ] && { echo "STOP NEW PCI SERR" >"$OUT/STOP"; log "STOP: new PCI SERR"; exit 2; }

# Aggregate capture summary
echo "=== capture summary ==="
for arm in $ARMS; do
  f=$CAP_ROOT/$arm/untuned_captures.csv
  if [ -f "$f" ]; then
    n=$(grep -c "^GemmTunableOp" "$f" 2>/dev/null || echo 0)
    echo "$arm: $n captured GemmTunableOp keys"
  fi
done

exit $local_rc
