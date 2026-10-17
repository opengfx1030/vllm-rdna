#!/usr/bin/env bash
# A/B one serving env gate against the server already listening on $PORT.
#
#   bash tools/rdna2/bench_serve_gates_ab.sh <label>
#
# Runs three shapes against the running server and prints the TTFT/TPOT lines
# `vllm bench serve` reports, so two arms launched with different gate values
# can be compared directly:
#
#   TUNABLEOP_PROFILE= VLLM_RDNA_QSA_HIP=0 VLLM_GDN_HIP_PREFILL=0 \
#       bash tools/rdna2/serve_v620_stable.sh          # arm A
#   bash tools/rdna2/bench_serve_gates_ab.sh off
#   ... restart with the gate on ...
#   bash tools/rdna2/bench_serve_gates_ab.sh on
#
# The gate env vars are read at import time, so each arm needs its own server
# process -- this script never restarts anything.
set -euo pipefail
repo=/home/wsantos/work/vllm-rdna
label=${1:?usage: bench_serve_gates_ab.sh <label>}
port=${PORT:-8000}
model=${SERVED_NAME:-active}
out=$repo/.pi/serve_ab_$label
mkdir -p "$out"
cd "$repo"

run() { # name input_len output_len prompts concurrency
  echo "=== $1 (input=$2 output=$3 prompts=$4 conc=$5) ==="
  .venv/bin/vllm bench serve \
    --backend openai --base-url "http://localhost:$port" --model "$model" \
    --tokenizer "$repo/models/qwen38-flash-next-uncensored-w4a16" \
    --trust-remote-code \
    --dataset-name random --seed 0 \
    --random-input-len "$2" --random-output-len "$3" \
    --num-prompts "$4" --max-concurrency "$5" --ignore-eos \
    --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    >"$out/$1.log" 2>&1 || { tail -20 "$out/$1.log"; return 1; }
  grep -E "TTFT \(ms\)|TPOT \(ms\)|token throughput" "$out/$1.log"
}

# The served prefill chunks are 1024/2048 (--max-num-batched-tokens 2048); the
# 8k/16k shapes are the long-context chunked-prefill sizes the PR also quotes.
run prefill_1k_c1 1024 1 64 1
run prefill_1k_c8 1024 1 64 8
# 2048 is the served prefill chunk (--max-num-batched-tokens 2048).
run prefill_2k_c1 2048 1 64 1
run prefill_2k_c8 2048 1 64 8
run prefill_8k_c1 8192 1 16 1
run prefill_8k_c8 8192 1 16 8
run prefill_16k_c1 16384 1 8 1
run prefill_16k_c8 16384 1 8 8
run decode_c1 128 256 8 1
