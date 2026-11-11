#!/usr/bin/env bash
# PR134 counterbalanced repeat phase: INT4, INT4, FP16.
#
# Combined with the pair phase (FP16, INT4) the full sequence is
# FP16 -> INT4 -> INT4 -> FP16 (ABBA), which controls for run-order drift.
# All four arms share one boot, one config, one source/binary manifest.
#
# The GPU set is a parameter (the paired runs use HIP 6,7,8,9: the clean PCIe
# root port). No source sync, no core rebuild, no dependency change.
set -uo pipefail

gpus=${1:?usage: $0 <gpus> <suffix>}
suffix=${2:?usage: $0 <gpus> <suffix>}

readonly results=/home/chenco_adm/pr134-results
readonly tree=/home/chenco_adm/vllm-pr-int4fp16
# Tag every output directory with the core binary that actually served it, so a
# rebuilt .so can never be filed under the previous build's results.
readonly core_tag=$(sha256sum "$tree/vllm/_rocm_C.abi3.so" | cut -c1-8)
readonly reference=$results/serve-27b-fp16-core$core_tag-$suffix
readonly int4_b=$results/serve-27b-int4-core$core_tag-$suffix-b
readonly fp16_b=$results/serve-27b-fp16-core$core_tag-$suffix-b
readonly stop_marker=$results/stop-qualification
readonly model=/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea

cd "$tree" || exit 10
export HIP_VISIBLE_DEVICES=$gpus
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

common=(--model "$model" --recipe 27b-awq --layout LBHNC --kv-bytes 8000000000
        --hw-queues 2 --port 18134 --gpus "$gpus" --run)

if [[ ! -s $reference/status.json ]]; then log "missing FP16 reference $reference"; exit 3; fi
mode=$(/home/chenco_adm/Apps/vllm/venv-pr134-int4fp16/bin/python \
  "$tree"/benchmarks/kernels/pr134/reference_mode.py "$reference" 2>/dev/null)
log "reference mode=$mode"
case "$mode" in
  strict-pass)   diagnostic=() ;;
  diagnostic-only) diagnostic=(--diagnostic-reference) ;;
  *) log "reference not usable as pairing basis"; exit 4 ;;
esac

# -------------------------------------------------- Arm 3: INT4 (second pass)
if [[ -e $stop_marker ]]; then log "stop marker present; aborting"; exit 2; fi
log "arm3=INT4 repeat output=$int4_b"
bash benchmarks/kernels/pr134/run_remote_micro.sh \
  benchmarks/kernels/pr134/qualify_serving.py "${common[@]}" \
  --cache-dtype int4_fp16_block --reference "$reference" "${diagnostic[@]}" \
  --output "$int4_b"
log "arm3 exit=$?"
[[ -s $int4_b/status.json ]] && cat "$int4_b/status.json"

# -------------------------------------------------- Arm 4: FP16 (second pass)
if [[ -e $stop_marker ]]; then log "stop marker present; aborting"; exit 2; fi
log "arm4=FP16 repeat output=$fp16_b"
bash benchmarks/kernels/pr134/run_remote_micro.sh \
  benchmarks/kernels/pr134/qualify_serving.py "${common[@]}" \
  --cache-dtype auto --output "$fp16_b"
log "arm4 exit=$?"
[[ -s $fp16_b/status.json ]] && cat "$fp16_b/status.json"
log "counterbalanced phase complete"
exit 0
