# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-kernel ISA statistics of a gfx10 ``.s`` (hipcc -S) or disassembly.

For every kernel: VGPR/SGPR count, spills, LDS bytes, and for the hottest
loop (the backward branch with the most dot instructions) the instruction
mix and the VGPR bank pattern of each dot (bank = register % 4; gfx10 reads
one register per bank per cycle, so two sources in one bank cost a cycle
unless the hardware's operand cache already holds one of them).

    python3 isa_stats.py mapk.s
"""

import collections
import sys

import regex as re

DOT = re.compile(
    r"\b(v_dot2c_f32_f16|v_dot2_f32_f16|v_dot4c_i32_i8|v_dot4_i32_i8|"
    r"v_fma_mix_f32|v_fmac_f32)(?:_e32|_e64)?\s+v(\d+),\s*v(\d+),\s*v(\d+)"
)
LABEL = re.compile(r"^(\.?[\w$.]+):")
BRANCH = re.compile(r"\bs_cbranch_\w+\s+(\.?[\w$.]+)")


def bank_stats(lines):
    """Share of dots whose three VGPR reads collide in a bank."""
    total = conflict = after_reuse = 0
    prev = set()
    for line in lines:
        m = DOT.search(line)
        if not m:
            continue
        regs = [int(m.group(2)), int(m.group(3)), int(m.group(4))]
        worst = max(collections.Counter(r % 4 for r in regs).values())
        fresh = [r for r in regs if r not in prev]
        worst_fresh = (
            max(collections.Counter(r % 4 for r in fresh).values()) if fresh else 0
        )
        total += 1
        conflict += worst > 1
        after_reuse += worst_fresh > 1
        prev = set(regs)
    return total, conflict, after_reuse


def split_kernels(text):
    kernels = {}
    name = None
    for line in text.splitlines():
        m = re.match(r"^([\w$.]+):\s*(?:;|//|$)", line)
        if m and not line.startswith(".") and not m.group(1).startswith(".L"):
            name = m.group(1)
            kernels[name] = []
            continue
        if name:
            kernels[name].append(line)
    return kernels


def hottest_loop(lines):
    labels = {}
    best = None
    for idx, line in enumerate(lines):
        lm = LABEL.match(line.strip())
        if lm:
            labels[lm.group(1)] = idx
        bm = BRANCH.search(line)
        if bm and bm.group(1) in labels and labels[bm.group(1)] < idx:
            body = lines[labels[bm.group(1)] : idx + 1]
            dots = sum(1 for b in body if DOT.search(b))
            if best is None or dots > best[0]:
                best = (dots, body)
    return best[1] if best else []


def meta(text, name, key):
    if key == "group_segment_fixed_size":
        m = re.search(
            rf"\.amdhsa_kernel {re.escape(name)}\s*\n(?:.*\n)*?"
            r"\s+\.amdhsa_group_segment_fixed_size\s+(\d+)",
            text,
        )
        return int(m.group(1)) if m else -1
    m = re.search(
        rf"\.name:\s+{re.escape(name)}\s*\n(?:.*\n)*?\s+\.{key}:\s+(\d+)", text
    )
    return int(m.group(1)) if m else -1


def main() -> None:
    with open(sys.argv[1]) as fh:
        text = fh.read()
    for name, lines in split_kernels(text).items():
        loop = hottest_loop(lines)
        if not loop:
            continue
        ops = collections.Counter()
        for line in loop:
            tok = line.strip().split()
            if tok and not tok[0].endswith(":") and not tok[0].startswith(("//", ";")):
                ops[tok[0]] += 1
        total, conflict, after = bank_stats(loop)
        if not total:
            continue
        valu_other = sum(
            v
            for k, v in ops.items()
            if k.startswith("v_") and not DOT.search(k + " v0, v1, v2")
        )
        short = name if len(name) < 90 else name[:87] + "..."
        print(short)
        print(
            f"  vgpr {meta(text, name, 'vgpr_count')} spill "
            f"{meta(text, name, 'vgpr_spill_count')} sgpr "
            f"{meta(text, name, 'sgpr_count')} lds "
            f"{meta(text, name, 'group_segment_fixed_size')}"
        )
        reads = {k: v for k, v in ops.items() if k.startswith("ds_")}
        mem = {k: v for k, v in ops.items() if k.startswith(("global_", "buffer_"))}
        print(
            f"  loop: {total} dots, {valu_other} other VALU, "
            f"{sum(v for k, v in ops.items() if k.startswith('s_'))} SALU/branch, "
            f"LDS {reads}, VMEM {mem}"
        )
        print(
            f"  dot bank collisions: {100.0 * conflict / total:.1f}% raw, "
            f"{100.0 * after / total:.1f}% after previous-instruction reuse"
        )


if __name__ == "__main__":
    main()
