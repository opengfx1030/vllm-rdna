#!/usr/bin/env bash
# SERVE_SCRIPT wrapper (runs from serve_validate.sh AFTER it took boot.lock).
#  1. quiet-window re-check under the lock (exit 75: the driver retries)
#  2. HIP -> PCI bus re-resolution for slot A (exit 76: the driver aborts)
#  3. PRINT=1 plan of the same arguments -> $OUT/plan.txt; refuse unless it
#     exports GPU_MAX_HW_QUEUES=4 and VLLM_RDNA_AR=1 (exit 77)
#  4. exec serve_rdna.sh with the same arguments
: "${R:?}" "${TREE_T:?}" "${OUT:?}" "${VENV:?}" "${SLOT:?}"
# shellcheck source=/dev/null
source "$R/slots.sh"
fresh=$(find "$HOME/w4a8_runs/lanes/quiet" -type f -mmin -6 ! -name "${OWNER:-Soak}" 2> /dev/null | tr '\n' ' ')
if [[ -n $fresh ]]; then
    echo "$(date -Is) quiet window active: $fresh" > "$R/abort_quiet"
    echo "serve_wrap: quiet window active ($fresh), releasing boot.lock"
    exit 75
fi
if ! msg=$(env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES bash -c "source $R/slots.sh; verify_slot $SLOT" 2>&1); then
    echo "$(date -Is) $msg" > "$R/abort_bus"
    echo "serve_wrap: $msg"
    exit 76
fi
echo "serve_wrap: $msg"
PRINT=1 bash "$TREE_T/tools/rdna/serve_rdna.sh" "$@" > "$OUT/plan.txt" 2>&1
for want in "export GPU_MAX_HW_QUEUES='4'" "export VLLM_RDNA_AR='1'"; do
    if ! grep -qx "$want" "$OUT/plan.txt"; then
        echo "$(date -Is) plan lacks $want" > "$R/abort_plan"
        echo "serve_wrap: plan lacks $want"
        exit 77
    fi
done
echo "serve_wrap: plan ok: $(grep -E 'VLLM_RDNA_AR|GPU_MAX_HW' "$OUT/plan.txt" | tr '\n' ' ')"
exec bash "$TREE_T/tools/rdna/serve_rdna.sh" "$@"
