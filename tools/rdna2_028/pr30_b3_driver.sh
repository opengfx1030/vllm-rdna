#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# PR #30 B3 + W4A8 freeze check: 27B AWQ dense serve matrix.
# One arm = (RDNA2_W4A16_RUNTIME_DISPATCH, VLLM_RDNA2_W4A8_SDOT4, CG_MODE).
# Arms are run sequentially by this script (one engine at a time). Each arm
# uses a fresh VLLM_CACHE_ROOT (the compile cache key does not include either
# dispatch env var). Coherence + the 4 production cells + duct/P-SERR guards.
#
#   ARMS="0 0 1 1" ... see the ARMS list below
#   bash tools/rdna2_028/pr30_b3_driver.sh
set -uo pipefail
V=${V:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0}
T=${T:-/home/chenco_adm/vllm-rdna-0.28.0}
MODEL=${MODEL:-/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea}
CG_MODE=${CG_MODE:-PIECEWISE}
MTP=${MTP:-0}
OUT=${OUT:-/home/chenco_adm/w4a8_runs}
# "RD W4A8" pairs.
ARMS=${ARMS:-"0 0|1 0|0 1|1 1"}

ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

serr() { dmesg -T 2>/dev/null | grep -ciE "SERR|AER" || echo 0; }

# aborts our leaked workers before the next arm boots into "free memory < util";
# exe-scoped so a co-tenant on another venv is never touched.
our_pids() {
  local p exe
  for p in $(pgrep -f "entrypoints.cli.main serve|VLLM::Worker|VLLM::EngineCore" 2>/dev/null); do
    exe=$(readlink -f "/proc/$p/exe" 2>/dev/null || true)
    case "$exe" in */venv-7.14.0_0.28.0/bin/python*) echo "$p" ;; esac
  done
}
hygiene() {
  local p n=0
  for p in $(our_pids); do kill -TERM "$p" 2>/dev/null && n=$((n + 1)); done
  [ "$n" -gt 0 ] && sleep 8
  for p in $(our_pids); do kill -KILL "$p" 2>/dev/null; done
  sleep 2
}

run_arm() { # $1=rd $2=w4a8
  local rd=$1 w4a8=$2
  local tag="2026-09-30_pr30-b3-rd${rd}-w${w4a8}-${CG_MODE}"
  local D=$OUT/$tag port=18320
  rm -rf "$D"; mkdir -p "$D"
  local s0; s0=$(serr)
  echo "[$(date +%H:%M:%S)] === arm rd=$rd w4a8=$w4a8 cg=$CG_MODE port=$port serr0=$s0 uptime=$(uptime -p) ===" | tee -a "$OUT/pr30_b3_driver.log"
  hygiene

  setsid nohup env MTP=$MTP W4A8=$w4a8 ATTN=fa TP=4 PORT=$port \
    KV=${KV:-8000000000} SEQS=${SEQS:-8} MAXBAT=${MAXBAT:-2048} \
    CG_MODE=$CG_MODE MODEL=$MODEL \
    VLLM_CACHE_ROOT="$D/cache" VLLM_DISABLE_COMPILE_CACHE=1 \
    VLLM_RDNA2_W4A16_RUNTIME_DISPATCH=$rd \
    bash "$T/tools/rdna/serve_gfx1030_27b_dense.sh" > "$D/serve.log" 2>&1 </dev/null &

  local ready=0
  for i in $(seq 1 90); do
    sleep 10
    curl -s --max-time 4 "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q q27d && { ready=1; echo "[$(date +%H:%M:%S)] READY t=$((i*10))s" | tee -a "$D/arm.log"; break; }
  done
  [ "$ready" = 1 ] || { echo "NOT READY" | tee -a "$D/arm.log"; tail -30 "$D/serve.log"; return 1; }

  # Warmup (throwaway; cold-start pollution is real).
  for k in 1 2; do
    curl -s --max-time 300 -X POST "http://127.0.0.1:$port/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"q27d\",\"prompt\":\"$(printf 'word %.0s' $(seq 1 2000))\",\"max_tokens\":8,\"temperature\":0,\"ignore_eos\":true}" >/dev/null
  done

  "$V/bin/python" "$T/tools/rdna2_028/probe_w4a8.py" "http://127.0.0.1:$port/v1" q27d 1 2>&1 | tee -a "$D/arm.log"

  cd "$D"
  for cell in "1 16384 1024 111" "8 16384 1024 112" "1 1024 512 221" "8 1024 512 222"; do
    set -- $cell
    echo "[$(date +%H:%M:%S)] bench c=$1 in=$2 out=$3" | tee -a "$D/arm.log"
    "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
      --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$port" \
      --model "$MODEL" --served-model-name q27d --dataset-name random \
      --random-input-len "$2" --random-output-len "$3" --num-prompts "$1" --max-concurrency "$1" \
      --ignore-eos --request-rate inf --seed "$4" --temperature 0 \
      --save-result --result-dir "$D/c$2_$1" > "$D/c$2_$1.log" 2>&1
    grep -E "Output token throughput|Mean TTFT|Mean TPOT|Input token throughput" "$D/c$2_$1.log" | tail -4 | tee -a "$D/arm.log"
  done

  echo "--- W4A8 gate:" | tee -a "$D/arm.log"
  grep -m1 "W4A8 sdot4 path active" "$D/serve.log" || echo "(not taken)" | tee -a "$D/arm.log"
  echo "--- duct markers:" | tee -a "$D/arm.log"
  grep -c "ductduct" "$D/arm.log" || true
  local s1; s1=$(serr)
  echo "[$(date +%H:%M:%S)] serr1=$s1 uptime=$(uptime -p)" | tee -a "$D/arm.log"

  local api; api=$(grep -oE "APIServer pid=[0-9]+" "$D/serve.log" | head -1 | cut -d= -f2)
  [ -n "$api" ] && kill -TERM "$api" 2>/dev/null && sleep 12
  for p in $(pgrep -f "VLLM::Worker|VLLM::EngineCore|entrypoints.cli.main serve" 2>/dev/null); do
    [ -r "/proc/$p/environ" ] || continue
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q "^VLLM_CACHE_ROOT=$D/cache$" && kill -9 "$p" 2>/dev/null
  done
  sleep 3
  echo "[$(date +%H:%M:%S)] === done rd=$rd w4a8=$w4a8 ===" | tee -a "$OUT/pr30_b3_driver.log"
}

IFS='|' read -ra _arms <<< "$ARMS"
for a in "${_arms[@]}"; do
  set -- $a
  run_arm "$1" "$2"
done
