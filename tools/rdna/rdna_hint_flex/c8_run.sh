#!/usr/bin/env bash
# One c8_profile.py run under the box rules, on slot a2 (HIP 0,1 = PCI buses d1,d4):
# the measurement mutex (waits for it, holds it for the whole boot + workload +
# teardown, <= HOLD_S), boot.lock and the quiet files handled by the driver. Run it
# inside a gpu01.lock hold (c8_session.sh does), never next to a scoreboard arm.
#   bash tools/rdna/rdna_hint_flex/c8_run.sh LABEL c8|fast [KEY=VAL ...]
# KEY=VAL go to serve_rdna.sh after the recipe and the "final" switch set; a later
# KEY wins, so RDNA_HINT_FLEX_PREFILL_GEMM=hip, MAXBAT=2056, ... are plain arguments.
#   FINAL= (empty) boots the recipe defaults (all HI2 switches off).
#   C8_ARGS="--windows decode" and similar go to c8_profile.py.
set -uo pipefail
LABEL=${1:?usage: c8_run.sh LABEL c8|fast [KEY=VAL ...]}
WL=${2:?workload c8 or fast}
shift 2
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.32.0}
SNAP=models--BlivionIaG--Qwen3.8-27B-HI2-3.0bpw/snapshots/db2a3aa3cd5efb0f92403cd4864e1f0d99378149
MODEL=${MODEL:-$HOME/.cache/huggingface/hub/$SNAP}
GPUS=${GPUS:-0,1}
BUSES=${BUSES:-d1,d4}
PORT=${PORT:-18215}
LOCK=${MEASURE_LOCK:-$HOME/w4a8_runs/measure.lock}
FINAL=${FINAL-"RDNA_HINT_FLEX_GEMV=auto VLLM_FA_RDNA2_UNFUSED_PREFILL=1 \
VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1 VLLM_RDNA_GDN_DECODE_INPLACE=1 \
RDNA_HINT_FLEX_FUSE_SILU=1 TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2"}
export HIP_VISIBLE_DEVICES=$GPUS
cd "$TREE" || exit 2
echo "[$(date +%T)] $LABEL: waiting for $LOCK"
# shellcheck disable=SC2086
exec flock "$LOCK" timeout --kill-after=60 "${HOLD_S:-2400}" \
    "$VENV/bin/python" tools/rdna/rdna_hint_flex/c8_profile.py \
    --label "$LABEL" --gpus "$GPUS" --pci-buses "$BUSES" --port "$PORT" \
    --model "$MODEL" --workload "$WL" ${C8_ARGS:-} \
    -- RECIPE=27b-hi2-tp2 $FINAL "$@"
