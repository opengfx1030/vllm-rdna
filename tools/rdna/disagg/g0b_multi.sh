#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# N concurrent cross-root pairs (the TP4 -> TP4 handoff: every prefill rank pushes to its
# decode rank at once). One g0b_run.sh per pair, each with its own src/dst process pair and
# run directory NAME-p<i>; the initiators wait for each other before every transfer
# (g0b_xfer_probe.py --sync-dir), so the N transfers of one iteration overlap.
#
#   g0b_multi.sh NAME ENGINE TLS CELLS SRC_HIP:SRC_BUS:DST_HIP:DST_BUS ...
#     e.g. g0b_multi.sh multi4 nixl ipc_only WRITE:contig:249.1M:50 1:ce:6:0a 2:d3:7:0f 3:d8:8:14 4:db:9:17
#   env as g0b_run.sh (TREE_T, G0B_ROOT, MEM, WINDOW, UCX_LOG, OWNER).
# Aggregate numbers: g0b_summary.py ROOT NAME-p0 NAME-p1 ... (runs named -p<i> are grouped).
set -uo pipefail
NAME=${1:?NAME} ENGINE=${2:?ENGINE} TLS=${3:?TLS} CELLS=${4:?CELLS}
shift 4
((${#} >= 2)) || { echo "need at least two pairs"; exit 64; }
TREE_T=${TREE_T:?TREE_T}
SPIKE=$TREE_T/tools/rdna/disagg
G0B_ROOT=${G0B_ROOT:-$HOME/w4a8_runs/disagg/g0b}
SYNC=$G0B_ROOT/$NAME.sync
rm -rf "$SYNC"
mkdir -p "$SYNC"
pids=()
i=0
for pair in "$@"; do
    IFS=: read -r src_hip src_bus dst_hip dst_bus <<< "$pair"
    env SYNC_DIR="$SYNC" SYNC_N=$# PAIR=$i SRC_HIP="$src_hip" SRC_BUS="$src_bus" DST_HIP="$dst_hip" DST_BUS="$dst_bus" \
        setsid -w bash "$SPIKE/g0b_run.sh" "$NAME-p$i" "$ENGINE" disjoint "$TLS" "$CELLS" "multi $NAME pair $i of $#" &
    pids+=($!)
    i=$((i + 1))
done
rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done
rm -rf "$SYNC"
echo "multi $NAME rc=$rc ($# pairs)" | tee "$G0B_ROOT/$NAME-p0/multi_rc.txt"
exit $rc
