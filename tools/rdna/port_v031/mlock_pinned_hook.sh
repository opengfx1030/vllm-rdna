#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# serve_validate.sh LOAD_HOOK of the pinned-host mlock proof: measures the booted
# TP server while a neighbour loads weights (pinned_proof.py hook). serve_validate.sh
# exports PORT, SERVED, MODEL and OUT; mlock_pinned_proof.sh exports the knobs below.
#
#   NEIGHBOUR=replay  a CPU-only replay of a TP2 weight load (pinned_proof.py
#                     neighbour): SHM_GIB of locked shared memory per rank, anonymous
#                     memory up to MemFree + OVERSHOOT_GIB, a read of $MODEL
#   NEIGHBOUR=cmd     NEIGHBOUR_CMD (word-split) is one load, e.g. a TP2 boot of the
#                     same model on other leased GPUs that exits when it is READY
#
# Each load is one quiet window (pinned_proof.py QuietWindow: touch quiet/EvictFix,
# wait for boot.lock, IDLE_S idle, the load, SETTLE_S settle, remove the file,
# GAP_S >= 120 s before the next). Ends with PASS or FAIL (the loads ran or not);
# mlock_pinned_proof.sh judges.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${PORT:?}" "${SERVED:?}" "${MODEL:?}" "${OUT:?}" "${BOOT_LOCK:?}" "${QUIET_DIR:?}"
PY=${PY:-$HOME/Apps/vllm/venv-7.14.0_0.32.0/bin/python}
NEIGHBOUR=${NEIGHBOUR:-replay}

case $NEIGHBOUR in
    replay)
        # CPU only: no GPU is visible to the replay.
        load=(env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=
            "$PY" "$HERE/pinned_proof.py" neighbour
            --workers "${REPLAY_WORKERS:-2}" --shm-gib "${SHM_GIB:-47.68}" --lock
            --overshoot-gib "${OVERSHOOT_GIB:-16}" --margin-gib "${MARGIN_GIB:-60}"
            --hold "${HOLD_S:-30}" --read "$MODEL")
        ;;
    cmd)
        [[ -n ${NEIGHBOUR_CMD:-} ]] || { echo "NEIGHBOUR=cmd needs NEIGHBOUR_CMD"; echo FAIL; exit 1; }
        read -r -a load <<< "$NEIGHBOUR_CMD"
        ;;
    *)
        echo "unknown NEIGHBOUR=$NEIGHBOUR"
        echo FAIL
        exit 1
        ;;
esac

exec "$PY" "$HERE/pinned_proof.py" hook \
    --loads "${LOADS:-3}" --idle "${IDLE_S:-30}" --gap "${GAP_S:-120}" \
    --settle "${SETTLE_S:-30}" --streams "${STREAMS:-4}" \
    --max-minutes "${MAX_MINUTES:-120}" --python "$PY" --kfd-raw "${KFD_RAW:-}" \
    --boot-lock "$BOOT_LOCK" --quiet-dir "$QUIET_DIR" --quiet-own "${QUIET_OWN:-EvictFix}" \
    -- "${load[@]}"
