#!/usr/bin/env python3
"""Print per-kernel register/spill metadata from clang -S AMDGPU output."""

import re
import sys

txt = open(sys.argv[1]).read().splitlines()
cur = {}
rows = []

KEYS = (
    "sgpr_count",
    "sgpr_spill_count",
    "vgpr_count",
    "vgpr_spill_count",
    "private_segment_fixed_size",
    "group_segment_fixed_size",
)

for line in txt:
    s = line.strip()
    m = re.match(r"\.name:\s+(_Z\S+)", s)
    if m:
        if cur.get("name"):
            rows.append(cur)
        cur = {"name": m.group(1)}
        continue
    for key in KEYS:
        m = re.match(r"\." + key + r":\s+(\d+)", s)
        if m and cur.get("name") and key not in cur:
            cur[key] = int(m.group(1))

if cur.get("name"):
    rows.append(cur)

for r in rows:
    n = r["name"]
    n = n.replace("_ZN4vllm14moe_w4a8_rdna220moe_w4a8_gemm_kernelI", "GEMM<")
    n = n.replace("_ZN4vllm12explore_w4a816w4a8_gemm_kernelI", "DGMM<")
    n = n.replace("_ZN4vllm12explore_w4a821w4a8_act_quant_kernelI", "ACTQ<")
    n = n.replace("_Z24moe_gemm_q4_kernel_rdna2I", "W4A16M<")
    print(
        "%-34s sgpr=%4s sp=%3s vgpr=%4s vp=%3s priv=%s"
        % (
            n[:34],
            r.get("sgpr_count", "?"),
            r.get("sgpr_spill_count", "?"),
            r.get("vgpr_count", "?"),
            r.get("vgpr_spill_count", "?"),
            r.get("private_segment_fixed_size", "?"),
        )
    )
