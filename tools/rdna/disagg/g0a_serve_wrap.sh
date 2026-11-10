#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# SERVE_SCRIPT wrapper of the G0a driver (runs from serve_validate.sh AFTER it took
# boot.lock):
#  1. quiet-window re-check under the lock (exit 75 -> the driver retries)
#  2. HIP -> PCI bus re-resolution for the leased slot $SLOT (exit 76 -> the driver aborts)
#  3. PRINT=1 plan of the same arguments -> $ARM_OUT/plan.txt, and the G0/NIXL/UCX
#     environment the server will inherit -> $ARM_OUT/env.txt
#  4. exec serve_rdna.sh with the same arguments
: "${ARM:?}" "${TREE_T:?}" "${ARM_OUT:?}" "${VENV:?}" "${R:?}" "${SPIKE:?}" "${SLOT:?}"
# shellcheck source=tools/rdna/disagg/slots.sh
source "$SPIKE/slots.sh"
fresh=$(find "$HOME/w4a8_runs/lanes/quiet" -type f -mmin -6 ! -name Disagg 2> /dev/null | tr '\n' ' ')
if [[ -n $fresh ]]; then
    echo "$(date -Is) quiet window active: $fresh" > "$R/abort_quiet.$ARM"
    echo "serve_wrap: quiet window active ($fresh), releasing boot.lock"
    exit 75
fi
if ! msg=$(env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES bash -c "source $SPIKE/slots.sh; verify_slot $SLOT" 2>&1); then
    echo "$(date -Is) $msg" > "$R/abort_bus.$ARM"
    echo "serve_wrap: $msg"
    exit 76
fi
echo "serve_wrap: $msg"
# aiter (imported through flash_attn) takes the first gfx name rocminfo prints and raises
# KeyError for one it does not list: gfx1030 on the V620-only box (cpu_stubs/rocminfo)
if [[ ${G0A_ROCMINFO_STUB:-1} == 1 ]]; then
    export G0A_ROCMINFO_STUB=1
    export PATH="$SPIKE/cpu_stubs:$PATH"
fi
PRINT=1 bash "$TREE_T/tools/rdna/serve_rdna.sh" "$@" > "$ARM_OUT/plan.txt" 2>&1
env | grep -E '^(VLLM_(RDNA2_MOE_DETERMINISTIC|SSM_CONV_STATE_LAYOUT|SERVER_DEV_MODE|NIXL)|G0A_|NIXL_|UCX_|HSA_ENABLE_IPC|PYTHONPATH|EXTRA_ARGS)' \
    | sort > "$ARM_OUT/env.txt"
exec bash "$TREE_T/tools/rdna/serve_rdna.sh" "$@"
