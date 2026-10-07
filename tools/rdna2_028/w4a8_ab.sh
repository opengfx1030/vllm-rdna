#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Controlled W4A8 A/B driver for the 27B dense MTP stack. One arm per run:
#   MTP=2 W4A8=1 TAG=ab_m2_w1 bash w4a8_ab.sh
#   MTP=2 W4A8=0 TAG=ab_m2_w0 bash w4a8_ab.sh
#
# Starts the server via scripts/serve_gfx1030_27b_dense.sh with the env-gated
# shape diagnostic (VLLM_RDNA2_W4A8_DEBUG=1, csrc/rocm/w4a8_sdot4_rdna2.cu),
# WARMS UP with throwaway requests, runs greedy coherence probes, then the
# production bench cells with --temperature=0. Extracts:
#   - every per-10s SpecDecoding window (Mean acceptance length + acc/draft),
#   - the distinct (m,k,n,group) shapes that fired the W4A8 fast path,
#   - the W4A8 gate marker.
# Everything lands in /home/chenco_adm/w4a8_runs/<TAG>/ (persistent, no /tmp).
set -uo pipefail
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
T=/home/chenco_adm/vllm-rdna-0.28.0
MODEL=${MODEL:-/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea}
MTP=${MTP:-2}
W4A8=${W4A8:-0}
DBG=${DBG:-1}
TAG=${TAG:-ab_mtp${MTP}_w${W4A8}}
PORT=${PORT:-18230}
OUT=${OUT:-/home/chenco_adm/w4a8_runs}
D=$OUT/$TAG
rm -rf "$D"; mkdir -p "$D"
ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/driver.log"; }

# Cells: "n_prompts input_len output_len seed". 16k c=8 last (slowest, and the
# load under which the MTP acceptance collapse was observed).
CELLS=${CELLS:-"1 16384 1024 111|1 1024 512 221|8 1024 512 222|8 16384 1024 112"}

setsid nohup env MTP=$MTP W4A8=$W4A8 VLLM_RDNA2_W4A8_DEBUG=$DBG EAGER=${EAGER:-0} ATTN=fa TP=${TP:-4} PORT=$PORT \
  KV=${KV:-8000000000} SEQS=${SEQS:-8} MAXBAT=${MAXBAT:-2048} \
  CG_MODE=${CG_MODE:-FULL_AND_PIECEWISE} MODEL="$MODEL" \
  bash "$T/scripts/serve_gfx1030_27b_dense.sh" > "$D/serve.log" 2>&1 < /dev/null &
log "serve pid $! (MTP=$MTP W4A8=$W4A8 DBG=$DBG)"
ready=0
for i in $(seq 1 96); do
  sleep 10
  curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q q27d && { ready=1; log "READY t=$((i*10))s"; break; }
done
[ "$ready" = "1" ] || { log "NOT READY"; tail -40 "$D/serve.log"; exit 1; }

# --- Warmup (throwaway; cold-start pollution is real on this stack) ---
log "warmup start"
for k in 1 2; do
  curl -s --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"q27d\",\"prompt\":\"$(printf 'word %.0s' $(seq 1 2000))\",\"max_tokens\":8,\"temperature\":0,\"ignore_eos\":true}" >/dev/null
done
log "warmup done"

# --- Coherence probe (greedy) ---
log "coherence probe start"
"$V/bin/python" "$T/tools/rdna2_028/probe_w4a8.py" "http://127.0.0.1:$PORT/v1" q27d 1 | tee "$D/coherence.txt"
log "coherence probe done"

# Run the bench client from the output dir (no vllm/ package shadowing, and no
# /tmp on this box).
cd "$D"
IFS='|' read -ra _cells <<< "$CELLS"
for cell in "${_cells[@]}"; do
  set -- $cell
  log "bench c=$1 in=$2 out=$3 start"
  "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
    --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" --served-model-name q27d --dataset-name random \
    --random-input-len "$2" --random-output-len "$3" --num-prompts "$1" --max-concurrency "$1" \
    --ignore-eos --request-rate inf --seed "$4" --temperature 0 \
    --save-result --result-dir "$D/c$2_$1" > "$D/c$2_$1.log" 2>&1
  log "bench c=$1 in=$2 out=$3 done: $(grep -m1 'Output token throughput' "$D/c$2_$1.log" | tr -s ' ')"
done

# --- Extract ---
log "W4A8 gate: $(grep -m1 'W4A8 sdot4 path active' "$D/serve.log" || echo '(not taken)')"
{
  echo "=== W4A8 fast-path shapes (unique, all ranks) ==="
  grep -h "W4A8-DEBUG" "$D/serve.log" | sort -u
} | tee "$D/shapes.txt"
{
  echo "=== SpecDecoding windows (serve log) ==="
  grep "SpecDecoding metrics" "$D/serve.log" \
    | sed -E "s/^\(APIServer[^)]*\) INFO ([0-9-]+ [0-9:]+).*Mean acceptance length: ([0-9.]+).*Accepted: ([0-9]+) tokens, Drafted: ([0-9]+).*/\1 mean=\2 acc=\3 dr=\4/"
} | tee "$D/acceptance.txt"

# Teardown scoped to this tree's cache root.
api=$(grep -oE "APIServer pid=[0-9]+" "$D/serve.log" | head -1 | cut -d= -f2)
[ -n "$api" ] && kill -TERM "$api" 2>/dev/null && sleep 10
for p in $(pgrep -f "VLLM::Worker|VLLM::EngineCore|entrypoints.cli.main serve" 2>/dev/null); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "VLLM_CACHE_ROOT=$T" && kill -9 $p 2>/dev/null; done
log "done $TAG"
