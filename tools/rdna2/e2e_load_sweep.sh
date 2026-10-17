#!/usr/bin/env bash
# TTFT / ITL / TPOT / throughput vs CONCURRENCY against an already-running server.
#
#   bash tools/rdna2/e2e_load_sweep.sh <label>
#
# Writes one JSON + one log per (scenario, concurrency) into .pi/sweep_<label>/
# and then prints the report. Use tools/rdna2/e2e_prefill_ab.sh to bring the two
# arms up around this.
#
# Scenarios are split so prefill and decode can be read separately:
#   prefill_*  output_len=1  -> TTFT is the whole request; throughput is prefill tok/s
#   decode     input_len=128 -> ITL/TPOT is the decode cost, input is negligible
#   mixed_2k   realistic     -> both, with a real prefill:decode ratio
set -euo pipefail
repo=/home/wsantos/work/vllm-rdna
label=${1:?usage: e2e_load_sweep.sh <label>}
port=${PORT:-8000}
model=${SERVED_NAME:-active}
tok=$repo/models/qwen38-flash-next-uncensored-w4a16
out=$repo/.pi/sweep_$label
mkdir -p "$out"
cd "$repo"

run() { # name in out prompts conc
  local name=$1 in=$2 o=$3 n=$4 c=$5
  printf '  %-14s in=%-6s out=%-4s n=%-4s c=%-3s ' "$name" "$in" "$o" "$n" "$c"
  if .venv/bin/vllm bench serve \
      --backend openai --base-url "http://localhost:$port" --model "$model" \
      --tokenizer "$tok" --trust-remote-code \
      --dataset-name random --seed 0 \
      --random-input-len "$in" --random-output-len "$o" \
      --num-prompts "$n" --max-concurrency "$c" --ignore-eos \
      --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99 \
      --save-result --result-dir "$out" --result-filename "${name}_c${c}.json" \
      >"$out/${name}_c${c}.log" 2>&1; then
    echo ok
  else
    echo FAILED
    tail -5 "$out/${name}_c${c}.log"
  fi
}

sweep() { # name in out concs...
  local name=$1 in=$2 o=$3
  shift 3
  local c n
  for c in "$@"; do
    n=$((c * 4))
    [ "$n" -lt 16 ] && n=16
    # The KV cache holds ~267k tokens (15,084 bytes/token into 3.75 GiB), so the
    # long-input sweeps must keep concurrency * input under that.
    [ "$in" -ge 8192 ] && n=$((c * 2))
    run "$name" "$in" "$o" "$n" "$c"
  done
}

echo "=== sweep $label ==="
# SWEEP_SCENARIOS limits the run to a subset (space-separated), for re-measuring a
# single scenario without restarting the whole 30-case sweep.
scen=${SWEEP_SCENARIOS:-all}
want() { [ "$scen" = all ] || [[ " $scen " == *" $1 "* ]]; }

want prefill_1k && sweep prefill_1k 1024 1 1 2 4 8 16 32
want prefill_2k && sweep prefill_2k 2048 1 1 2 4 8 16
want prefill_8k && sweep prefill_8k 8192 1 1 2 4 8
want prefill_16k && sweep prefill_16k 16384 1 1 2 4 8
want decode && sweep decode 128 256 1 2 4 8 16 32
want mixed_2k && sweep mixed_2k 2048 128 1 2 4 8 16

.venv/bin/python tools/rdna2/e2e_load_report.py "$label"
