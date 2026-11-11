#!/usr/bin/env bash
# Before/after P2P subset on the 8 V620 (HIP 1-4,6-9): probe subset (3 repeats)
# + RCCL TP8 all_reduce_perf (2 repeats, production env). usage: subset.sh <outdir> <tag>
set -uo pipefail
out=$1 tag=$2
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
cd "$here" || exit 1
touch ~/w4a8_runs/lanes/quiet/PcieAudit
python3 run_matrix.py "$out/$tag.probe.log" 1,2,3,4,6,7,8,9 sub
for r in 1 2; do
    touch ~/w4a8_runs/lanes/quiet/PcieAudit
    RCCL_MAX=8M RCCL_ITERS=50 ./rccl_matrix.sh "$out" "$tag.rccl$r" 1,2,3,4,6,7,8,9 \
        NCCL_PROTO=Simple RCCL_P2P_NET_DISABLE=1 RCCL_P2P_BATCH_ENABLE=1 \
        RCCL_MSCCL_ENABLE=0 NCCL_P2P_LEVEL=pxb >/dev/null
done
rm -f ~/w4a8_runs/lanes/quiet/PcieAudit
echo "subset finished"
