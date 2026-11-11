#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# serve_validate.sh LOAD_HOOK of t10_run.sh: agentic traffic for LOAD_SECONDS.
# serve_validate.sh exports PORT, SERVED and OUT; t10_run.sh exports LOAD_SECONDS.
#
# One pass = the hi2 task suite behind the Cline system prompt (agentic_bench.py
# run: gsm8k, humaneval, mbpp, ifeval, toolcall, needle, decision; every prompt
# starts with the long agent system prompt, needle adds long haystacks), 8 workers
# (the recipe's 8 sequences), greedy. Pass k uses --seed k, which draws other
# items, so no pass repeats the previous one. New passes start until LOAD_SECONDS
# have elapsed. /metrics is saved before and after. Ends with PASS or FAIL.
set -uo pipefail
: "${PORT:?}" "${SERVED:?}" "${OUT:?}" "${LOAD_SECONDS:?}"
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.32.0}
PY=$VENV/bin/python
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
W4Q=${W4Q:-$HOME/w4a8_runs/w4a8-quality-2026-11-04}
TASKS=${TASKS:-gsm8k,humaneval,mbpp,ifeval,toolcall,needle,decision}
LIMIT=${LIMIT:-60}
BASE=http://127.0.0.1:$PORT
export HI2_BENCH_DATA=${HI2_BENCH_DATA:-$HOME/w4a8_runs/hadamard-int2/bench_data/export}
export HF_DATASETS_OFFLINE=1
export ROCPROFILER_REGISTER_ENABLED=0

say() { echo "[$(date +%T)] $*"; }
mkdir -p "$OUT/agentic"
curl -sf "$BASE/metrics" > "$OUT/metrics-load-start.txt" || say "metrics start FAILED"
start=$(date +%s)
seed=1
rc=0
while (($(date +%s) - start < LOAD_SECONDS)); do
    say "pass $seed start ($(($(date +%s) - start)) s of $LOAD_SECONDS)"
    "$PY" "$TREE/tools/rdna/port_v031/agentic_bench.py" --hi2 "$W4Q/hi2" \
        --system "$W4Q/cline_act.md" run --url "$BASE" --model "$SERVED" \
        --tasks "$TASKS" --limit "$LIMIT" --workers 8 --seed "$seed" \
        --out "$OUT/agentic/pass$seed.json" > "$OUT/agentic/pass$seed.log" 2>&1
    prc=$?
    say "pass $seed rc=$prc"
    ((prc == 0)) || rc=1
    seed=$((seed + 1))
done
curl -sf "$BASE/metrics" > "$OUT/metrics-load-end.txt" || { say "metrics end FAILED"; rc=1; }
say "passes: $((seed - 1)) in $(($(date +%s) - start)) s"
if ((rc == 0)); then echo PASS; else echo FAIL; fi
exit "$rc"
