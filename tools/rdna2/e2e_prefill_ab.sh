#!/usr/bin/env bash
# End-to-end prefill A/B: the Triton path vs the all-HIP umbrella, TP4 on all
# four GPUs. This is the one comparison the per-kernel benches cannot make.
#
#   bash tools/rdna2/e2e_prefill_ab.sh            # both arms, in order
#   bash tools/rdna2/e2e_prefill_ab.sh triton     # one arm
#
# The gates are read at import time, so each arm needs its own server process.
# Prefill chunks are capped at --max-num-batched-tokens 2048, so T=2048 is the
# served shape; 8k/16k are the long-context chunked-prefill sizes.
#
# NOTE: the arm difference is NOT prefill-only. `VLLM_RDNA_ALL_HIP` is the umbrella
# (`on_rdna2_hip()` = `on_rdna2() and VLLM_RDNA_ALL_HIP`) and gates 29 call sites, including
# the decode hot path: `v1/sample/ops/topk_topp_sampler.py`, all of `v1/worker/gpu/sample/*`,
# `spec_decode/{speculator,rejection_sampler}`, `input_batch`/`block_table`, and the QSA
# metadata builder (which runs twice per decode step). So this A/B is "whole HIP port vs
# Triton/FLA", i.e. the `VLLM_RDNA_ALL_HIP` default-flip decision, not a prefill isolation.
# A prefill-isolated arm is `VLLM_RDNA_ALL_HIP=1` plus the five individual gates set to 0.
#
# Env knobs:
#   SWEEP_SCENARIOS="prefill_1k prefill_2k"  limit the sweep to a subset
#   SWEEP_TAG=_rep                           suffix the results dir only
#   EXTRA_ARGS_APPEND=--no-enable-prefix-caching
#       appended to the server's EXTRA_ARGS by serve_v620_stable.sh, AFTER the launcher's
#       --enable-prefix-caching, so last-wins disables it. REQUIRED for a clean prefill
#       measurement: `--dataset-name random --seed 0` generates request i as
#       `allowed[(offset_i + i + arange(len))]`, so prompts collide across bench cases and
#       prefix caching serves whole prompts from cache (a 16k hit costs ~1.3 s regardless of
#       length). Without this, per-request latency inside one case spreads 15-57x at 8k/16k.
set -euo pipefail
repo=/home/wsantos/work/vllm-rdna
cd "$repo"
PORT=${PORT:-8000}
arms=${1:-both}

# One pattern per line, each bracketed so the pattern text cannot match this
# script's own command line and kill the caller's shell.
stop_server() {
  pkill -9 -f '[s]erve_v620_stable.sh|[s]erve_v620_v031.sh|[o]penai.api_server|[c]li.main serve|[V]LLM::' || true
  sleep 5
  fuser -k -9 /dev/kfd >/dev/null 2>&1 || true
  sleep 5
  pgrep -af '[o]penai.api_server|[c]li.main serve' && { echo "server still up"; return 1; }
  return 0
}

wait_ready() {
  for i in $(seq 1 90); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "  ready after $((i * 10))s"
      return 0
    fi
    sleep 10
  done
  echo "  server did NOT become ready in 900s"
  return 1
}

run_arm() { # label  env...
  local label=$1
  shift
  echo "=== arm $label ==="
  # SWEEP_TAG suffixes only the results directory, so re-measuring one scenario
  # does not overwrite the full-sweep JSONs.
  local tag="$label${SWEEP_TAG:-}"
  stop_server
  env "$@" bash tools/rdna2/serve_v620_stable.sh \
    >".pi/e2e_$tag.server.log" 2>&1 &
  if ! wait_ready; then
    tail -30 ".pi/e2e_$tag.server.log"
    stop_server || true
    return 1
  fi
  bash tools/rdna2/e2e_load_sweep.sh "$tag" || true
  stop_server
}

# Every prefill gate off: the Triton/FLA reference path.
TRITON_ENV=(
  VLLM_RDNA_ALL_HIP=0
  VLLM_RDNA_QSA_HIP=0
  VLLM_GDN_HIP_PREFILL=0
  VLLM_RDNA_HC_PREFILL_HIP=0
  VLLM_CAUSAL_CONV1D_RDNA2_FWD=0
  VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=0
)
# The umbrella plus every individual gate: the all-HIP path.
HIP_ENV=(
  VLLM_RDNA_ALL_HIP=1
  VLLM_RDNA_QSA_HIP=1
  VLLM_GDN_HIP_PREFILL=1
  VLLM_RDNA_HC_PREFILL_HIP=1
  VLLM_CAUSAL_CONV1D_RDNA2_FWD=1
  VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=1
)

# The five per-op gates ON with the umbrella OFF. This isolates the prefill ports
# from the umbrella's decode-side kernel set, and it is the arm pair the tracker's
# order-controlled A/B/A used (docs/rdna2/qwen4_exp_hip_tracker.md, Round 7).
GATES_ENV=(
  VLLM_RDNA_ALL_HIP=0
  VLLM_RDNA_QSA_HIP=1
  VLLM_GDN_HIP_PREFILL=1
  VLLM_RDNA_HC_PREFILL_HIP=1
  VLLM_CAUSAL_CONV1D_RDNA2_FWD=1
  VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=1
)

case "$arms" in
  triton) run_arm triton "${TRITON_ENV[@]}" || echo "arm triton FAILED" ;;
  hip) run_arm hip "${HIP_ENV[@]}" || echo "arm hip FAILED" ;;
  gates) run_arm gates "${GATES_ENV[@]}" || echo "arm gates FAILED" ;;
  both)
    run_arm triton "${TRITON_ENV[@]}" || echo "arm triton FAILED"
    run_arm hip "${HIP_ENV[@]}" || echo "arm hip FAILED"
    ;;
  *) echo "usage: $0 [both|triton|hip|gates]"; exit 2 ;;
esac
echo "=== logs in $repo/.pi/sweep_{triton,hip}/ ==="
