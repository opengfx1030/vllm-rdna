#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 27B dense W4A16 bring-up at TP=4 on the FULL HIP path with the fork's
# PCIe-peer all-reduce DISABLED (RDNA_AR=0) and FULL_AND_PIECEWISE cudagraphs.
# Fresh arm-tagged caches force a clean W4A16 compile (the shared cache holds a
# contaminated W4A8 graph that replays the `duct` collapse).
#
# IPMI is sampled before, after each cell, and after teardown; a new PCI SERR
# aborts the remaining cells and drops a STOP marker.
#
# Usage: TAG=2026-09-29_tp4-fp-aroff bash fp_aroff_27b.sh
set -uo pipefail
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
T=/home/chenco_adm/vllm-rdna-0.28.0
CACHE=$T/cache/hip-fp-aroff
MODEL=${MODEL:-/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea}
TAG=${TAG:-2026-09-29_tp4-fp-aroff}
PORT=${PORT:-18240}
OUT=${OUT:-/home/chenco_adm/w4a8_runs}
D=$OUT/$TAG
rm -rf "$D"; mkdir -p "$D"
ROCK_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCK_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCK_SDK_LIB:$ROCK_SDK/host-math/lib:$ROCK_SDK/rocm_sysdeps/lib:$ROCK_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/driver.log"; }

sel_snapshot() {
  local tag=$1
  sudo -n ipmitool sel list 2>/dev/null > "$D/$tag.sel.txt"
  local serr crit ecc
  serr=$(grep -c 'PCI SERR' "$D/$tag.sel.txt")
  crit=$(grep -c 'Critical Interrupt' "$D/$tag.sel.txt")
  ecc=$(grep -c 'Correctable ECC' "$D/$tag.sel.txt")
  {
    echo "=== [$tag] $(date -Is) uptime: $(uptime -p)"
    echo "PCI SERR: $serr (baseline $SERR0)  Critical Interrupt: $crit  Correctable ECC: $ecc"
    tail -3 "$D/$tag.sel.txt"
  } | tee -a "$D/ipmi.log"
  if [ "$serr" -gt "$SERR0" ]; then
    echo "STOP: NEW PCI SERR after $tag" | tee -a "$D/ipmi.log"
    echo "STOP NEW PCI SERR after $tag" > "$D/STOP"
    return 1
  fi
  return 0
}

teardown() {
  local api
  api=$(grep -oE "APIServer pid=[0-9]+" "$D/serve.log" 2>/dev/null | head -1 | cut -d= -f2)
  [ -n "$api" ] && kill -TERM "$api" 2>/dev/null && sleep 10
  for p in $(pgrep -f "VLLM::Worker|VLLM::EngineCore|entrypoints.cli.main serve" 2>/dev/null); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q "VLLM_CACHE_ROOT=$CACHE" && kill -9 "$p" 2>/dev/null
  done
}

# --- Preflight: baseline fabric events, clean caches, free port ---
log "preflight: TAG=$TAG PORT=$PORT CACHE=$CACHE"
mkdir -p "$(dirname "$CACHE")"
rm -rf "$CACHE"
mkdir -p "$CACHE/inductor" "$CACHE/triton" "$CACHE/extensions"
SERR0=$(sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR')
sel_snapshot before || { log "baseline snapshot failed; aborting"; exit 1; }
log "baseline PCI SERR count=$SERR0"

# --- Launch: FULL HIP path, no fork AR, F&P graphs, fresh caches ---
T0=$(date +%s)
setsid nohup env MTP=0 W4A8=0 RDNA_AR=0 EAGER=0 ATTN=fa TP=4 PORT=$PORT \
  KV=8000000000 SEQS=8 MAXBAT=2048 CG_MODE=FULL_AND_PIECEWISE MODEL="$MODEL" \
  VLLM_CACHE_ROOT="$CACHE" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
  TRITON_CACHE_DIR="$CACHE/triton" TORCH_EXTENSIONS_DIR="$CACHE/extensions" \
  bash "$T/scripts/serve_gfx1030_27b_dense.sh" > "$D/serve.log" 2>&1 < /dev/null &
log "serve launched pid $! (MTP=0 W4A8=0 RDNA_AR=0 CG_MODE=FULL_AND_PIECEWISE)"

ready=0
for i in $(seq 1 360); do
  sleep 10
  curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q q27d && { ready=1; break; }
done
COLD=$(( $(date +%s) - T0 ))
if [ "$ready" = "1" ]; then
  log "READY t=${COLD}s (cold compile+capture window)"
  echo "$COLD" > "$D/cold_compile_seconds.txt"
else
  log "NOT READY after ${COLD}s"
  tail -40 "$D/serve.log"
  sel_snapshot notready
  exit 1
fi
sel_snapshot after_boot || { teardown; exit 1; }

# --- Warm-up cell (throwaway; cold-start pollution skews the first measured cell) ---
log "warmup cell start"
cd "$D"
"$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
  --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" --served-model-name q27d --dataset-name random \
  --random-input-len 16384 --random-output-len 64 --num-prompts 1 --max-concurrency 1 \
  --ignore-eos --request-rate inf --seed 999 --temperature 0 \
  --save-result --result-dir "$D/warmup" > "$D/warmup.log" 2>&1
log "warmup cell done"
sel_snapshot after_warmup || { teardown; exit 1; }

# --- Coherence (greedy) ---
log "coherence probe start"
"$V/bin/python" "$T/tools/rdna2_028/probe_w4a8.py" "http://127.0.0.1:$PORT/v1" q27d 1 | tee "$D/coherence.txt"
log "coherence probe done"

# --- Marker audit: what should be OFF must not appear ---
{
  echo "=== 'W4A8 sdot4 path active' (must be ABSENT) ==="
  grep -n "W4A8 sdot4 path active" "$D/serve.log" || echo "(absent: good)"
  echo "=== 'W4A8-DEBUG' (must be ABSENT; debug not set) ==="
  grep -n "W4A8-DEBUG" "$D/serve.log" || echo "(absent: good)"
  echo "=== 'rdna_ar:' lines ==="
  grep -n "rdna_ar:" "$D/serve.log" || echo "(no rdna_ar lines: VLLM_RDNA_AR=0 never constructs the backend)"
  echo "=== all-reduce backend selection ==="
  grep -n "all-reduce backends" "$D/serve.log" || echo "(none)"
  echo "=== 'RDNA_ONESHOT' (must be ABSENT) ==="
  grep -n "RDNA_ONESHOT" "$D/serve.log" || echo "(absent: good)"
  echo "=== 'Custom allreduce force-enabled' (must be ABSENT) ==="
  grep -n "Custom allreduce force-enabled" "$D/serve.log" || echo "(absent: good)"
  echo "=== attention backend markers ==="
  grep -nE "RDNA_ATTN|RDNA_FA|Using .*attention backend" "$D/serve.log" | head -5 || true
} | tee "$D/markers.txt"

# --- Measured cells: 16k/1k at c=1, then c=8 ---
for cell in "1 16384 1024 111" "8 16384 1024 112"; do
  set -- $cell
  log "bench c=$1 in=$2 out=$3 start"
  "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
    --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" --served-model-name q27d --dataset-name random \
    --random-input-len "$2" --random-output-len "$3" --num-prompts "$1" --max-concurrency "$1" \
    --ignore-eos --request-rate inf --seed "$4" --temperature 0 \
    --save-result --result-dir "$D/c$2_$1" > "$D/c$2_$1.log" 2>&1
  log "bench c=$1 in=$2 out=$3 done: $(grep -m1 'Output token throughput' "$D/c$2_$1.log" | tr -s ' ')"
  sel_snapshot "after_c${1}_${2}" || { teardown; exit 1; }
done

# --- Teardown + post IPMI ---
teardown
sleep 3
sel_snapshot after_teardown || true
log "done $TAG"
echo "OK" > "$D/status.txt"
