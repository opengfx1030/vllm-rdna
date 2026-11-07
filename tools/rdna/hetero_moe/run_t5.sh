#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# T5 on par1-cs25 (hetero-MoE design section 11, row T5): runs
# tools/rdna/hetero_moe/t5_run.py (golden | run | faults) with the box rules around
# it: leases checked, HIP -> bus verified, venv-7.14.0 with PYTHONPATH at the exact
# tree, weight loads under boot.lock, measured modes inside a quiet window.
#
#   run_t5.sh --tag TAG [--tree DIR] [--dev-lib FILE] [--hub-hip N] [--quiet]
#             [--boot] -- golden|run|faults ARGS...
#
#   --tree      tree to import (default: this script's tree). It must carry the
#               vllm/*.abi3.so of a fat gfx1030;gfx1100 build whose _rocm_C holds
#               the hx_* ops, or pass --dev-lib.
#   --dev-lib   hx_dev.so of tools/rdna/hetero_moe/build_hx_dev.sh (dev shim; never
#               together with a _rocm_C that already has the hx ops).
#   --hub-hip   HIP index the hub process uses (0 = the W7800: needs the lease
#               <owner>.w7800cb; 1 = V620 stand-in hub). Default 0.
#   --quiet     measured window (bench/soak): quiet/<owner> + boot.lock wait.
#   --boot      the run loads weights (golden, run without --synthetic): quiet
#               check + boot.lock around the whole run.
#   Everything after -- goes to t5_run.py (--plan/--out/--model/...). t5_run starts
#   its own GPU children, each with HIP_VISIBLE_DEVICES on its own command line.
#
# Leases are NOT taken here: <owner>.slotA (and <owner>.w7800cb for --hub-hip 0)
# must exist (HX_OWNER, default HxServer); the script refuses otherwise. HIP 5 is
# refused. Logs: ~/w4a8_runs/<tag>/run_t5.log (plus whatever t5_run writes).
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=hx_box.sh
source "$HERE/hx_box.sh"

tag=""
tree=$(cd "$HERE/../../.." && pwd)
dev_lib=""
hub_hip=""
quiet=0
boot=0

die() {
    echo "run_t5: $*" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --tag) tag=${2:?}; shift 2 ;;
        --tree) tree=${2:?}; shift 2 ;;
        --dev-lib) dev_lib=${2:?}; shift 2 ;;
        --hub-hip) hub_hip=${2:?}; shift 2 ;;
        --quiet) quiet=1; shift ;;
        --boot) boot=1; shift ;;
        --) shift; break ;;
        -h | --help) sed -n '4,/^set -uo/p' "${BASH_SOURCE[0]}" | grep '^#' | cut -c3-; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done
[[ -n $tag ]] || die "--tag is required"
[[ $# -ge 1 ]] || die "missing t5_run subcommand (golden|run|faults)"
case $1 in golden | run | faults) ;; *) die "subcommand must be golden|run|faults, got $1" ;; esac
if [[ $1 == golden ]]; then
    hub_hip=""
elif [[ -z $hub_hip ]]; then
    die "--hub-hip is required for run/faults (0 = W7800 hub, 1 = V620 stand-in hub)"
fi
if [[ -n $hub_hip ]]; then
    [[ $hub_hip == 5 ]] && die "HIP 5 is forbidden"
    [[ $hub_hip =~ ^[0-9]+$ ]] || die "--hub-hip must be a number"
fi
[[ -d $tree/vllm ]] || die "tree $tree has no vllm/"

out=$HOME/w4a8_runs/$tag
mkdir -p "$out"
log=$out/run_t5.log
exec > >(tee -a "$log") 2>&1

[[ -e $LEASE/$HX_OWNER.slotA ]] || die "lease $LEASE/$HX_OWNER.slotA missing (take it first)"
hx_verify_slot_a || exit 4
if [[ $hub_hip == 0 ]]; then
    [[ -e $LEASE/$HX_OWNER.w7800cb ]] || die "lease $LEASE/$HX_OWNER.w7800cb missing (--hub-hip 0)"
    hx_verify_hub || exit 4
fi

export PYTHONPATH=$tree
[[ -n $dev_lib ]] && export VLLM_HX_DEV_LIB=$dev_lib
# the t5_run parent touches no GPU (VLLM_TARGET_DEVICE=cpu: no platform probe, amdsmi
# would enumerate every GPU); its children get their own HIP_VISIBLE_DEVICES and no
# VLLM_TARGET_DEVICE
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=
export VLLM_TARGET_DEVICE=cpu
args=("$@")
# the hub process of a stand-in run sits on a V620: addressed through --hub-hip
[[ -n $hub_hip ]] && args+=(--hub-hip "$hub_hip")

hx_log "tag $tag tree $tree hub HIP ${hub_hip:-none} dev-lib ${dev_lib:-none} quiet=$quiet boot=$boot"
run() { (cd "$tree" && "$VENV/bin/python" -m tools.rdna.hetero_moe.t5_run "${args[@]}"); }

status=0
if ((quiet)); then hx_quiet_begin || exit 3; fi
if ((boot)); then
    hx_boot_run run || status=$?
else
    run || status=$?
fi
if ((quiet)); then hx_quiet_end; fi
hx_log "t5_run exit $status"
exit "$status"
