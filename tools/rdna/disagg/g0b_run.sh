#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# One G0b run: a src probe on HIP $SRC_HIP (prefill rank, root complex c0) and a dst probe on
# HIP $DST_HIP (decode rank, root complex 00), each its own process.
#
#   g0b_run.sh NAME ENGINE VISIBLE TLS CELLS [LABEL]
#     ENGINE   nixl | hipipc | hostbounce
#     VISIBLE  disjoint (src sees only HIP 4, dst only HIP 6; production-like: separate TP
#              groups) | shared (both processes see HIP 4,6)
#     TLS      default | ipc_only (UCX_TLS=rocm_copy,rocm_ipc,tcp,self: no host RMA transport
#              (posix/sysv/cma), so a device-to-device put/get cannot be staged through host
#              memory; rocm_copy is only UCX's local mem-type copy endpoint, which UCX refuses to
#              start without; tcp carries wireup and notifications) |
#              no_ipc (UCX_TLS without rocm_ipc: host bounce through rocm_copy + posix/cma)
#     CELLS    g0b_xfer_probe.py --cells string
#   env: TREE_T (tree holding tools/rdna/disagg), G0B_ROOT, MEM=gpu|cpu, WINDOW=1|0 (measured
#        window protocol), SRC_HIP/DST_HIP/SRC_BUS/DST_BUS, UCX_LOG (e.g. info), HSA_IPC_LEGACY
#        (default 0: legacy KFD IPC breaks UCX rocm_ipc here), WARMUP
# The caller holds the leases (HIP 4 = slot A, HIP 6 = slot B) and, for measured cells, runs
# under the quiet-window protocol (WINDOW=1 does it per cell inside the probe).
set -uo pipefail
NAME=${1:?NAME} ENGINE=${2:?ENGINE} VISIBLE=${3:?VISIBLE} TLS=${4:?TLS} CELLS=${5:?CELLS} LABEL=${6:-}
TREE_T=${TREE_T:?TREE_T}
SPIKE=$TREE_T/tools/rdna/disagg
G0B_ROOT=${G0B_ROOT:-$HOME/w4a8_runs/disagg/g0b}
NIXL_ROOT=${NIXL_ROOT:-$HOME/w4a8_runs/disagg/nixl}
export VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0}
# shellcheck source=tools/rdna/disagg/slots.sh
source "$SPIKE/slots.sh"
MEM=${MEM:-gpu}
WINDOW=${WINDOW:-0}
SRC_HIP=${SRC_HIP:-4} DST_HIP=${DST_HIP:-6} SRC_BUS=${SRC_BUS:-db} DST_BUS=${DST_BUS:-0a}
WARMUP=${WARMUP:-3}
RUN=$G0B_ROOT/$NAME
rm -rf "$RUN"
mkdir -p "$RUN/rv"

common=(PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 GPU_MAX_HW_QUEUES=4 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1)
case $TLS in
    default) ;;
    ipc_only) common+=("UCX_TLS=rocm_copy,rocm_ipc,tcp,self") ;;
    no_ipc) common+=("UCX_TLS=rocm_copy,posix,sysv,tcp,cma,self") ;;
    *) echo "TLS must be default, ipc_only or no_ipc"; exit 64 ;;
esac
[[ -n ${UCX_LOG:-} ]] && common+=(UCX_LOG_LEVEL="$UCX_LOG" UCX_PROTO_INFO=y)
# Legacy KFD IPC (HSA_ENABLE_IPC_MODE_LEGACY=1) makes UCX rocm_ipc registration fail on this kernel
# ("Failed to create ipc ... Address not valid", G0b 2026-11-06); the dmabuf IPC default works.
legacy=${HSA_IPC_LEGACY:-0}
((legacy)) && common+=(HSA_ENABLE_IPC_MODE_LEGACY=1)
if [[ $ENGINE == nixl ]]; then
    common+=(PYTHONPATH="$NIXL_ROOT/site" NIXL_PLUGIN_DIR="$NIXL_ROOT/site/nixl_rocm.libs/nixl"
        UCX_ROCM_COPY_SIGPOOL_MAX_ELEMS=inf UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=inf)
fi

vis_src=$SRC_HIP vis_dst=$DST_HIP dev_dst=0
if [[ $VISIBLE == shared ]]; then
    vis_src=$SRC_HIP,$DST_HIP vis_dst=$SRC_HIP,$DST_HIP dev_dst=1
elif [[ $VISIBLE != disjoint ]]; then
    echo "VISIBLE must be disjoint or shared"; exit 64
fi
if [[ $MEM == cpu ]]; then
    vis_src='' vis_dst='' dev_dst=0
    expect_src=() expect_dst=()
else
    expect_src=(--expect-bus "$SRC_BUS") expect_dst=(--expect-bus "$DST_BUS")
fi
common_args=(--engine "$ENGINE" --mem "$MEM" --cells "$CELLS" --warmup "$WARMUP" --label "$LABEL" --run-dir "$RUN/rv" --timeout "${G0B_TIMEOUT:-180}")
((WINDOW)) && common_args+=(--window)
[[ -n ${SYNC_DIR:-} ]] && common_args+=(--sync-dir "$SYNC_DIR" --sync-n "${SYNC_N:?}" --pair "${PAIR:?}")

{
    echo "name=$NAME engine=$ENGINE visible=$VISIBLE tls=$TLS mem=$MEM window=$WINDOW"
    echo "cells=$CELLS"
    echo "src: HIP_VISIBLE_DEVICES=$vis_src bus $SRC_BUS; dst: HIP_VISIBLE_DEVICES=$vis_dst dev_index $dev_dst bus $DST_BUS"
    printf '%s\n' "${common[@]}"
    echo "sha $(cat "$TREE_T/.sha" 2> /dev/null)"
    date -Is
} > "$RUN/config.txt"
DMESG_BEFORE=$(sudo -n dmesg 2> /dev/null | tee "$RUN/dmesg.before.txt" | wc -l)

echo $$ > "$RUN/pgid" # the group leader when the caller runs this under setsid -w: tempwatch ends the group
python3 "$SPIKE/tempwatch.py" "$SRC_BUS,$DST_BUS" "$RUN/temps1s.log" "$RUN/temps15s.log" "$RUN/pgid" "$RUN/ABORT_TEMP" &
WATCH=$!
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES "${common[@]}" HIP_VISIBLE_DEVICES="$vis_src" \
    "$VENV/bin/python" "$SPIKE/g0b_xfer_probe.py" --role src --dev-index 0 "${expect_src[@]}" "${common_args[@]}" \
    > "$RUN/src.log" 2>&1 &
P1=$!
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES "${common[@]}" HIP_VISIBLE_DEVICES="$vis_dst" \
    "$VENV/bin/python" "$SPIKE/g0b_xfer_probe.py" --role dst --dev-index "$dev_dst" "${expect_dst[@]}" "${common_args[@]}" \
    > "$RUN/dst.log" 2>&1 &
P2=$!
wait "$P1"
R1=$?
wait "$P2"
R2=$?
kill "$WATCH" 2> /dev/null
rm -f "$HOME/w4a8_runs/lanes/quiet/${OWNER:-Disagg}"
# shellcheck disable=SC2024 # the redirect target is mine; only dmesg needs sudo
sudo -n dmesg > "$RUN/dmesg.after.txt" 2> /dev/null
{
    echo "dmesg lines before $DMESG_BEFORE, after $(wc -l < "$RUN/dmesg.after.txt")"
    echo "new amdgpu fault/timeout/reset/hang lines: $(tail -n +$((DMESG_BEFORE + 1)) "$RUN/dmesg.after.txt" | grep -iE 'amdgpu' | grep -iE 'fault|timeout|reset|hang|ras |ecc|UTCL2' | grep -vc 'uses VM inv eng')"
    echo "new AER / PCIe Bus Error lines: $(tail -n +$((DMESG_BEFORE + 1)) "$RUN/dmesg.after.txt" | grep -ciE 'AER:|PCIe Bus Error')"
} > "$RUN/dmesg_summary.txt"
cp "$RUN"/rv/*.jsonl "$RUN/" 2> /dev/null
echo "src rc=$R1 dst rc=$R2" | tee "$RUN/rc.txt"
[[ -e $RUN/ABORT_TEMP ]] && { echo "THERMAL ABORT: $(cat "$RUN/ABORT_TEMP")"; exit 3; }
exit $((R1 || R2))
