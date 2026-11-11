#!/usr/bin/env bash
# PR134 INT4 candidate arm on par1-cs25, HIP 6,7,8,9.
#
# Pairs against the strict-pass FP16 reference on the same boot, same config
# and same source/binary manifest. The raw gate is untouched; a corrected-link
# PCIe event would still fail the raw status and switch this arm to
# diagnostic-only attribution.
set -uo pipefail

readonly results=/home/chenco_adm/pr134-results
readonly tree=/home/chenco_adm/vllm-pr-int4fp16
readonly runtime=/home/chenco_adm/Apps/vllm/venv-pr134-int4fp16
# Tag every output directory with the core binary that actually served it, so a
# rebuilt .so can never be filed under the previous build's results.
readonly core_tag=$(sha256sum "$tree/vllm/_rocm_C.abi3.so" | cut -c1-8)
readonly reference=$results/serve-27b-fp16-core$core_tag-hip6789
readonly candidate=$results/serve-27b-int4-core$core_tag-hip6789
readonly stop_marker=$results/stop-qualification
readonly model=/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea

cd "$tree" || exit 10
export HIP_VISIBLE_DEVICES=6,7,8,9
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

if [[ -e $stop_marker ]]; then log "stop marker present; aborting"; exit 2; fi
mode=$("$runtime"/bin/python "$tree"/benchmarks/kernels/pr134/reference_mode.py "$reference" 2>/dev/null)
log "reference mode=$mode"
case "$mode" in
  strict-pass)   extra=() ;;
  diagnostic-only) extra=(--diagnostic-reference) ;;
  *) log "reference not usable as pairing basis"; exit 4 ;;
esac

log "boot=$(cat /proc/sys/kernel/random/boot_id) gpus=6,7,8,9 arm=INT4 candidate output=$candidate"
bash benchmarks/kernels/pr134/run_remote_micro.sh \
  benchmarks/kernels/pr134/qualify_serving.py \
  --model "$model" --recipe 27b-awq --layout LBHNC --kv-bytes 8000000000 \
  --hw-queues 2 --port 18134 --gpus 6,7,8,9 --run \
  --cache-dtype int4_fp16_block --reference "$reference" "${extra[@]}" \
  --output "$candidate"
log "arm exit=$?"
[[ -s $candidate/status.json ]] && cat "$candidate/status.json"
[[ -s $candidate/qualification.json ]] && cat "$candidate/qualification.json"
log "candidate arm complete"
exit 0
