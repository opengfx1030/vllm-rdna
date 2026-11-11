#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Evidence extraction from one arm directory: bash extract.sh ARM_DIR > ARM_DIR/markers.txt
set -uo pipefail
D=${1:?ARM_DIR}
L=$D/serve.log
g() { grep -aE "$1" "$L" | sed -E 's/\x1b\[[0-9;]*m//g' | cut -c1-400; }
echo "== boot/ready"
grep -aE "READY in|BOOT FAILED|PROBE|PPL|CONCURRENCY|CONC_REPEAT|done$" "$D/summary.txt" 2> /dev/null
echo "== rdna_ar status lines"
g "rdna_ar:" | sort | uniq -c | sort -rn | head -40
echo "== RCCL / NCCL warnings in serve.log"
g "NCCL WARN|RCCL|nccl.*(error|warn)" | head -20
echo "== weights / KV / graph / profiling"
g "Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|Graph capturing finished|Memory profiling takes|Total non KV cache|torch peak memory|KV cache memory" | sort | uniq -c | head -60
echo "== PLE"
g "Pinned PLE table|Initialized AMD PLE|PLE sidecar|ple_offload" | head -20
echo "== distributed init"
g "world_size|rank|Starting to load model|torch_dtype|Using .* all.?reduce|custom all.?reduce|CustomAllreduce" | grep -aiE "custom all|all-reduce|allreduce|rdna_ar|quick" | head -20
echo "== attention / kernel markers"
g "Using [A-Za-z0-9_]+ backend|Selected [A-Za-z0-9_]+|WNA16 MoE backend|RDNA_ATTN|qsa|QSA.*(legacy|pipelined)|GQA" | sort | uniq -c | sort -rn | head -30
echo "== tuned configs / tunableop"
g "tuned|TunableOp|tunableop|Using default MoE config|Config file not found" | sort | uniq -c | sort -rn | head -20
echo "== faults / errors"
g "Memory Fault|HSA_STATUS|page fault|Segmentation fault|core dumped|Traceback|RuntimeError|ValueError|AssertionError|OutOfMemory|out of memory|wedged|late peer|evicted" | sort | uniq -c | sort -rn | head -30
echo "== spec decode"
g "SpecDecoding metrics" | tail -3
echo "== pipeline parallel / MTP drafter / head sharing"
g "pipeline|Pipeline|PP_PARTITION|PP[0-9]_TP|pp_group|partition|start_layer|end_layer|lm_head|Detected MTP|drafter|Sharing target|refus|not supported|NotImplemented" | sed -E 's/pid=[0-9]+/pid=N/' | sort | uniq -c | sort -rn | head -50
echo "== concurrency repeats"
cat "$D/conc/runs.tsv" 2> /dev/null
echo "== runlist / kfd"
sed -n '1,200p' "$D/dmesg_summary.txt" 2> /dev/null
echo "== temps (peak junction, 1 Hz)"
python3 - "$D/temps1s.log" << 'PY'
import sys

peak = {}
try:
    with open(sys.argv[1]) as f:
        for line in f:
            for tok in line.split()[1:]:
                k, _, v = tok.partition("=")
                try:
                    peak[k] = max(peak.get(k, 0), float(v))
                except ValueError:
                    pass
except OSError:
    pass
print(" ".join(f"{k}={v:.0f}" for k, v in sorted(peak.items())))
PY
