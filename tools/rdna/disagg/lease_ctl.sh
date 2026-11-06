#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Manual lease control for the Disagg spikes (~/w4a8_runs/lanes/lease/README.md):
#   lease_ctl.sh take SLOT...      request, wait for the holders and >= 28 GiB free per GPU,
#                                  write ~/w4a8_runs/lanes/lease/Disagg.slotX, then exit
#                                  (the holder files stay; the drivers run with LEASE_HELD=1)
#   lease_ctl.sh release SLOT...   remove my holder file and request for each slot, and my
#                                  quiet file
#   lease_ctl.sh status            list the lease and quiet directories
set -uo pipefail
SPIKE=$(cd "$(dirname "$0")" && pwd)
R=${R:-$HOME/w4a8_runs/disagg}
export VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0}
mkdir -p "$R"
# shellcheck source=tools/rdna/disagg/slots.sh
source "$SPIKE/slots.sh"
# shellcheck source=tools/rdna/disagg/lease.sh
source "$SPIKE/lease.sh"
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
cmd=${1:?usage: lease_ctl.sh take|release|status [SLOT...]}
shift
case $cmd in
    take) take_leases "$@" ;;
    release)
        for s in "$@"; do release_slot "$s"; done
        rm -f "$QUIET/$OWNER"
        ;;
    status)
        say "lease: $(lease_ls)"
        say "quiet: $(for f in "$QUIET"/*; do [[ -e $f ]] && printf '%s ' "${f##*/}"; done)"
        ;;
    *) die "unknown command $cmd" ;;
esac
