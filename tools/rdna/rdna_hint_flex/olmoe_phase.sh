#!/usr/bin/env bash
# OLMoE-1B-7B rdna_hint_flex campaign on one free V620 (holds a lock file
# while it runs): TunableOp tune/curate/merge of the census shapes, the
# serving matrix (FULL_AND_PIECEWISE, max context 4096: 1k/512 and 2k/512 at
# c=1/c=8) for both checkpoints and the dense model (RDNA2 HIP and Triton MoE
# backends) plus a per-pair-GEMV A/B arm, then held-out perplexity of the
# dense exports and of the original model in fp16.
#   GPU=3 BENCH_TREE=~/vllm-rdna-hi2-bench3 bash tools/rdna/rdna_hint_flex/olmoe_phase.sh
# SKIP_TUNE=1 / SKIP_PPL=1 skip those phases; ROOT moves the serving results.
set -uo pipefail
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
GPU=${GPU:?GPU}
BENCH_TREE=${BENCH_TREE:?BENCH_TREE}
R=$HOME/w4a8_runs/hi2-support
A=$HOME/w4a8_runs/hadamard-int2
M=$A/runsmoe
O=$HOME/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5
LOCK=$A/hi2support.gpu$GPU.lock
echo "rdna_hint_flex OLMoE serving campaign (fork side), started $(date)" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
log() { echo "[$(date +%H:%M:%S)] $*"; }

if [[ ${SKIP_TUNE:-0} != 1 ]]; then
  export WORK=$R/tunable-olmoe CENSUS_GLOB="$R/census/olmoe-*/untuned*.csv"
  C=$TREE/tools/rdna/rdna_hint_flex/tunableop_campaign.sh
  for phase in census tune measure curate merge; do
    log "tunableop $phase"
    GPU=$GPU bash "$C" "$phase" || { log "tunableop $phase failed"; exit 1; }
  done
  cp "$TREE"/tunableop/rocm7.14-rocblas5.5/tunableop_results*.csv \
    "$BENCH_TREE"/tunableop/rocm7.14-rocblas5.5/
fi

export ROOT=${ROOT:-$R}  # serving results: ROOT/serve-<arm>
mkdir -p "$ROOT"
export SERVE_OVERRIDES="MAXLEN=4096 MIN_MAXLEN=0"
export CELLS="1024:512:1 1024:512:8:32 2048:512:1:4 2048:512:8:16"
B=$BENCH_TREE/tools/rdna/rdna_hint_flex/bench_serve.sh
GPU=$GPU PORT=18741 bash "$B" "olmoe-r4=$M/olmoe-r4" "olmoe-b30=$M/olmoe-b3.0" "olmoe-dense=$O"
EXTRA_ARGS="--moe-backend triton" GPU=$GPU PORT=18741 bash "$B" "olmoe-densetriton=$O"
RDNA_HINT_FLEX_MOE_GROUPED_MIN=1000000 GPU=$GPU PORT=18741 bash "$B" "olmoe-r4-pairgemv=$M/olmoe-r4"

[[ ${SKIP_PPL:-0} == 1 ]] && { log "campaign finished (no ppl)"; exit 0; }
MIX=$A/eval_mix.txt
for arm in r4:$R/dense/olmoe-r4 b30:$R/dense/olmoe-b3.0 orig:$O; do
  tag=${arm%%:*}
  model=${arm#*:}
  GPU=$GPU TP=1 TAG=olmoe_${tag}export_study MODEL=$model bash "$TREE/tools/rdna/rdna_hint_flex/acc_tp.sh"
  GPU=$GPU TP=1 TAG=olmoe_${tag}export_mix MODEL=$model bash "$TREE/tools/rdna/rdna_hint_flex/acc_tp.sh" --text "$MIX" --skip-tokens 0
done
log "campaign finished"
