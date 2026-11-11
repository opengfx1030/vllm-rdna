#!/usr/bin/env bash
# PR134 same-boot paired qualification on par1-cs25 (192.168.1.84).
#
# Usage: pr134-27b-pair.sh <gpus> <suffix> [qualify_serving.py args...]
#   <gpus>   comma-separated HIP indices, e.g. 6,7,8,9
#   <suffix> output-dir suffix, e.g. hip6789
#   args     extra qualify_serving.py flags, applied to BOTH arms so the pair
#            differs only in the KV cache dtype. Use them to pin the serving
#            levers under test:
#              --w4a8 1                          int8-activation prefill GEMMs
#              --decode-stall-ms 0               disable the decode-stall cap
#              --mamba-ssm-cache-dtype float16   fp16 mamba state
#
# Arm 1: FP16 reference (cache-dtype auto) on the current boot.
# Arm 2: INT4 candidate (cache-dtype int4_fp16_block) against that reference.
#
# The candidate requires a same-boot, same-config, same-manifest reference. The
# raw gate is untouched: a corrected-link PCIe/AER event still yields raw FAIL.
# When the reference's only kernel events are corrected-link noise, the
# candidate is run with --diagnostic-reference and both arms are recorded as
# diagnostic-only with contaminated hardware attribution.
set -uo pipefail

gpus=${1:?usage: $0 <gpus> <suffix> [args...]}
suffix=${2:?usage: $0 <gpus> <suffix> [args...]}
shift 2

readonly results=/home/chenco_adm/pr134-results
readonly tree=/home/chenco_adm/vllm-pr-int4fp16
# Tag every output directory with the core binary that actually served it, so a
# rebuilt .so can never be filed under the previous build's results.
readonly core_tag=$(sha256sum "$tree/vllm/_rocm_C.abi3.so" | cut -c1-8)
readonly reference=$results/serve-27b-fp16-core$core_tag-$suffix
readonly candidate=$results/serve-27b-int4-core$core_tag-$suffix
readonly stop_marker=$results/stop-qualification
# PR134_MODEL overrides the served model directory. The HF cache snapshot of
# this model lost its directory entry after the host reboot (a stale XFS
# entry), so the campaign runs against a repaired copy of the same blobs.
readonly model=${PR134_MODEL:-/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeaa}

cd "$tree" || exit 10
export HIP_VISIBLE_DEVICES=$gpus
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

common=(--model "$model" --recipe 27b-awq --layout LBHNC --kv-bytes 8000000000
        --hw-queues 2 --port 18134 --gpus "$gpus" --run "$@")

arm1=(benchmarks/kernels/pr134/qualify_serving.py "${common[@]}" --cache-dtype auto
      --output "$reference")
arm2_strict=(benchmarks/kernels/pr134/qualify_serving.py "${common[@]}" --cache-dtype int4_fp16_block
              --reference "$reference" --output "$candidate")
arm2_diag=(benchmarks/kernels/pr134/qualify_serving.py "${common[@]}" --cache-dtype int4_fp16_block
           --reference "$reference" --diagnostic-reference --output "$candidate")

# ---------------------------------------------------------------- Arm 1: FP16
log "boot=$(cat /proc/sys/kernel/random/boot_id) gpus=$gpus arm1=FP16 reference output=$reference"
if [[ -e $stop_marker ]]; then log "stop marker present; aborting before arm 1"; exit 2; fi
bash benchmarks/kernels/pr134/run_remote_micro.sh "${arm1[@]}"
arm1_rc=$?
log "arm1 exit=$arm1_rc"

if [[ ! -s $reference/status.json ]]; then log "arm1 produced no status.json; aborting"; exit 3; fi
cat "$reference/status.json"
if [[ -s $reference/qualification.json ]]; then cat "$reference/qualification.json"; fi

# ---------------------------------------------------- Arm 2 gate: decide mode
mode=$(/home/chenco_adm/Apps/vllm/venv-pr134-int4fp16/bin/python \
  "$tree"/benchmarks/kernels/pr134/reference_mode.py "$reference" 2>/dev/null)
log "reference mode=$mode"
case "$mode" in
  strict-pass)
    arm2=("${arm2_strict[@]}")
    ;;
  diagnostic-only)
    arm2=("${arm2_diag[@]}")
    log "reference raw FAIL is corrected-link-only; candidate runs diagnostic-only"
    ;;
  *)
    log "reference is not usable as a pairing basis; aborting before arm 2"
    exit 4
    ;;
esac

if [[ -e $stop_marker ]]; then log "stop marker present; aborting before arm 2"; exit 2; fi

# --------------------------------------------------------------- Arm 2: INT4
log "arm2=INT4 candidate output=$candidate mode=$mode"
bash benchmarks/kernels/pr134/run_remote_micro.sh "${arm2[@]}"
arm2_rc=$?
log "arm2 exit=$arm2_rc"
if [[ -s $candidate/status.json ]]; then cat "$candidate/status.json"; fi
if [[ -s $candidate/qualification.json ]]; then cat "$candidate/qualification.json"; fi
log "pair complete"
exit 0
