#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Diagnostic-only AFTER_MATRIX_HOOK for ram10_study.sh; never a timing arm.
set -euo pipefail
case ${ARM:?ARM required} in
    D|Z) ;;
    *) echo "TRACE SKIP arm=$ARM (no partial offload)"; exit 0 ;;
esac
: "${OUT:?}" "${PORT:?}" "${SERVED:?}" "${MODEL:?}" "${VENV:?}" "${GPUS:?}"
TREE=${TREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
base="http://127.0.0.1:$PORT"
mkdir -p "$OUT/trace"
started=0
cleanup() {
    if (( started )); then
        curl -fsS --max-time 120 -H 'Content-Type: application/json' \
            --data '{"method":"rdna_expert_cache_trace","kwargs":{"action":"stop"}}' \
            "$base/collective_rpc" > "$OUT/trace/interrupted.json" || true
    fi
}
trap cleanup EXIT
for cell in 16384:1 1024:8; do
    IFS=: read -r input concurrency <<< "$cell"
    dir="$OUT/trace/${input}-c${concurrency}"
    mkdir -p "$dir"
    curl -fsS --max-time 60 -X POST "$base/reset_prefix_cache" > "$dir/reset.json"
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        python3 -c 'import json,sys; assert json.load(open(sys.argv[1])).get("success") is True' "$dir/reset.json"
    curl -fsS --max-time 120 -H 'Content-Type: application/json' \
        --data '{"method":"rdna_expert_cache_trace","kwargs":{"action":"start","max_calls":1024,"layers":[0,1,12,13,24,25,36,37,46,47]}}' \
        "$base/collective_rpc" > "$dir/start.json"
    started=1
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        python3 -c 'import json,sys; r=json.load(open(sys.argv[1]))["results"]; assert len(r)==4 and all(x["caches"]==10 for x in r)' "$dir/start.json"
    HIP_VISIBLE_DEVICES="$GPUS" "$VENV/bin/python" -m vllm.entrypoints.cli.main bench serve \
        --backend openai --endpoint /v1/completions --base-url "$base" \
        --model "$SERVED" --tokenizer "$MODEL" --dataset-name random \
        --random-input-len "$input" --random-output-len 512 \
        --num-prompts "$concurrency" --max-concurrency "$concurrency" \
        --ignore-eos --request-rate inf --seed 271828 --temperature 0 \
        --save-result --save-detailed --result-dir "$dir" \
        --result-filename diagnostic-bench.json > "$dir/client.log" 2>&1
    curl -fsS --max-time 120 -H 'Content-Type: application/json' \
        --data '{"method":"rdna_expert_cache_trace","kwargs":{"action":"stop"}}' \
        "$base/collective_rpc" > "$dir/stop.json"
    started=0
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        python3 "$TREE/tools/rdna/mono/ram10_prefetch.py" "$dir/stop.json" \
        --warmup-calls 64 --budget 2 --out "$dir/prefetch.json" > "$dir/replay.log"
    echo "TRACE PASS arm=$ARM input=$input concurrency=$concurrency; diagnostic timings excluded"
done
