#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 27B dense matrix: FA-RDNA2 + W4A8 sdot4 + RDNA2 HIP, prefix caching, F&P
# graphs, no --max-model-len. Runs the four production cells and greps the W4A8
# gate line. Uses the dedicated launcher (tools/rdna/serve_gfx1030_27b_dense.sh).
# Usage: MTP=0 TAG=m27b_m0 bash m27b_matrix.sh
set -uo pipefail
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
T=/home/chenco_adm/vllm-rdna-0.28.0
MODEL=${MODEL:-/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea}
MTP=${MTP:-0}
TAG=${TAG:-m27b_mtp$MTP}
PORT=${PORT:-18220}
OUT=${OUT:-/home/chenco_adm/w4a8_runs}
D=$OUT/$TAG
mkdir -p "$D"
ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib"

setsid nohup env MTP=$MTP W4A8=${W4A8:-0} ATTN=fa TP=${TP:-4} PORT=$PORT \
  KV=${KV:-8000000000} SEQS=${SEQS:-8} MAXBAT=${MAXBAT:-2048} \
  CG_MODE=${CG_MODE:-FULL_AND_PIECEWISE} MODEL="$MODEL" \
  bash "$T/tools/rdna/serve_gfx1030_27b_dense.sh" > "$D/serve.log" 2>&1 < /dev/null &
echo "serve pid $! (MTP=$MTP)"
ready=0
for i in $(seq 1 72); do
  sleep 15
  curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q q27d && { ready=1; echo "READY t=$((i*15))s"; break; }
done
[ "$ready" = "1" ] || { echo "NOT READY"; tail -20 "$D/serve.log"; exit 1; }
for p in "The capital of France is" "2 + 2 ="; do
  echo "--- $p"
  curl -s --max-time 180 -X POST "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"q27d\",\"prompt\":\"$p\",\"max_tokens\":12,\"temperature\":0}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(repr(d['choices'][0]['text']))" 2>/dev/null || echo "(no response)"
done | tee "$D/coherence.txt"
cd "$D"
for cell in "1 16384 1024 111" "8 16384 1024 112" "1 1024 512 221" "8 1024 512 222"; do
  set -- $cell
  echo "--- bench c=$1 in=$2 out=$3"
  "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
    --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" --served-model-name q27d --dataset-name random \
    --random-input-len "$2" --random-output-len "$3" --num-prompts "$1" --max-concurrency "$1" \
    --ignore-eos --request-rate inf --seed "$4" --save-result --result-dir "$D/c$2_$1" > "$D/c$2_$1.log" 2>&1
  grep -E "Mean TTFT|Mean TPOT|Mean ITL|Input token throughput|Output token throughput" "$D/c$2_$1.log" | tail -5
done
echo "--- W4A8 gate:"; grep -m2 "W4A8 sdot4 path active" "$D/serve.log" || echo "(not taken)"
api=$(grep -oE "APIServer pid=[0-9]+" "$D/serve.log" | head -1 | cut -d= -f2)
[ -n "$api" ] && kill -TERM "$api" 2>/dev/null && sleep 10
for p in $(pgrep -f "VLLM::Worker|VLLM::EngineCore|entrypoints.cli.main serve" 2>/dev/null); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "VLLM_CACHE_ROOT=$T" && kill -9 $p 2>/dev/null; done
echo "done $TAG"