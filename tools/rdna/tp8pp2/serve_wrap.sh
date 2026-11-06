#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# SERVE_SCRIPT wrapper (runs from serve_validate.sh AFTER it took boot.lock).
#  1. quiet-window re-check under the lock (exit 75 -> driver retries)
#  2. HIP -> PCI bus re-resolution for the arm's slot list (exit 76 -> driver aborts)
#  3. PRINT=1 plan of the exact same arguments -> $ARM_OUT/plan.txt
#  4. exec serve_rdna.sh with the same arguments
# shellcheck source=slots.sh
source "$(dirname "${BASH_SOURCE[0]}")/slots.sh"
: "${ARM:?}" "${SLOT:?}" "${TREE_T:?}" "${ARM_OUT:?}" "${VENV:?}" "${OWNER:?}"
fresh=$(find "$QUIET" -type f -mmin -6 ! -name "$OWNER" 2> /dev/null | tr '\n' ' ')
if [[ -n $fresh ]]; then
    echo "$(date -Is) quiet window active: $fresh" > "$R/abort_quiet.$ARM"
    echo "serve_wrap: quiet window active ($fresh), releasing boot.lock"
    exit 75
fi
# shellcheck disable=SC2016
if ! msg=$(env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES bash -c \
    'source "$1"; verify_slot "$2"' _ "$H/slots.sh" "$SLOT" 2>&1); then
    echo "$(date -Is) $msg" > "$R/abort_bus.$ARM"
    echo "serve_wrap: $msg"
    exit 76
fi
echo "serve_wrap: $msg"
args=("$@")
# RECIPE_RESOLVES_GPUS=1: the recipe's PCI_BUSES (not the driver's slot list) must
# pick the GPUs. serve_validate.sh passes the slot list both in the environment and as
# a trailing HIP_VISIBLE_DEVICES=<list> argument, and either one counts as an explicit
# device list that wins over PCI_BUSES: drop both, let serve_rdna.sh resolve the buses
# itself, then compare the plan with the verified slot list.
if [[ ${RECIPE_RESOLVES_GPUS:-0} == 1 ]]; then
    want=$HIP_VISIBLE_DEVICES
    args=()
    for a in "$@"; do
        [[ $a == HIP_VISIBLE_DEVICES=* ]] || args+=("$a")
    done
    unset HIP_VISIBLE_DEVICES GPUIDS ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    PRINT=1 bash "$TREE_T/tools/rdna/serve_rdna.sh" "${args[@]}" > "$ARM_OUT/plan.txt" 2>&1
    got=$(sed -n "s/^export HIP_VISIBLE_DEVICES='\(.*\)'$/\1/p" "$ARM_OUT/plan.txt")
    if [[ $got != "$want" ]]; then
        echo "$(date -Is) recipe resolved HIP '$got', verified slot list is '$want'" \
            > "$R/abort_bus.$ARM"
        echo "serve_wrap: recipe GPU resolution mismatch: '$got' vs '$want'"
        exit 76
    fi
    echo "serve_wrap: recipe PCI_BUSES resolved to HIP $got (matches the verified slot list)"
fi
PRINT=1 bash "$TREE_T/tools/rdna/serve_rdna.sh" "${args[@]}" > "$ARM_OUT/plan.txt" 2>&1
echo "serve_wrap: plan written to $ARM_OUT/plan.txt"
exec bash "$TREE_T/tools/rdna/serve_rdna.sh" "${args[@]}"
