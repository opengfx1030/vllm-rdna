# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, SIM115
"""Instruction mix of the hot loop of one AMDGPU kernel in a .s file.

  hipcc -O3 --offload-arch=gfx1030 --cuda-device-only -S -o k.s <flags> gemm_variant.hip
  python isa_mix.py k.s [--kernel gemm_kernel] [--dump loop.txt]

The hot loop is the backward branch whose body holds the most v_dot2 / v_fma /
v_pk_fma instructions. Prints the mnemonic histogram of that body, the
ds_read / dot2 and VALU / dot2 ratios, and a count of dot2 instructions whose
two VGPR sources share a bank (index mod 4), the gfx10 register-file bank
conflict candidates. ``--objdump`` reads an ``llvm-objdump -d --mcpu=gfx1030``
disassembly (one kernel) instead, for the kernels of a built .so or rocBLAS's
Tensile code objects.
"""

from __future__ import annotations

import argparse
import collections
import sys

import regex as re

DOT = re.compile(
    r"^v_(dot2c?_f32_f16|dot4_i32_i8|fma_f32|pk_fma_f16|fmac_f32|mad_f32|fma_mix)"
)
DOT2 = re.compile(r"^v_dot2c?_f32_f16")
VREG = re.compile(r"\bv(\d+)\b|\bv\[(\d+):(\d+)\]")


OBJ_INSTR = re.compile(
    r"^\s+(\S.*?)\s+// ([0-9A-Fa-f]+): ([0-9A-Fa-f ]+?)\s*(<[^>+]*(?:\+0x([0-9A-Fa-f]+))?>)?\s*$"
)  # fmt: skip
SOPP_BRANCH_OPS = range(2, 10)  # s_branch, s_cbranch_scc0/1, vccz/vccnz, execz/execnz


def parse_objdump(path: str) -> list[str]:
    """llvm-objdump lines -> assembler-style lines: branch operands become
    .L<addr> labels and a label line precedes every branch target. The target
    comes from <sym+off> when objdump prints it, else from the SOPP encoding
    (a code object that has local symbols, like rocBLAS's Tensile kernels,
    prints a symbol name or a bare word count instead)."""
    start = 0
    insts: list[tuple[int, str, int | None]] = []
    for ln in open(path, errors="replace").read().splitlines():
        m = re.match(r"^([0-9a-f]{16}) <", ln)
        if m:
            start = int(m.group(1), 16)
            continue
        m = OBJ_INSTR.match(ln)
        if m:
            addr, text = int(m.group(2), 16), m.group(1).strip()
            w0 = int(m.group(3).split()[0], 16)
            tgt = start + int(m.group(5) or "0", 16) if m.group(4) else None
            if (
                tgt is None
                and (w0 >> 23) == 0x17F
                and ((w0 >> 16) & 0x7F) in SOPP_BRANCH_OPS
            ):
                simm = w0 & 0xFFFF
                tgt = addr + 4 + 4 * (simm - 0x10000 if simm & 0x8000 else simm)
            insts.append((addr, text, tgt))
    targets = {t for _, _, t in insts if t is not None}
    body: list[str] = []
    for addr, text, tgt in insts:
        if addr in targets:
            body.append(f".L{addr:x}:")
        if tgt is not None:
            text = (
                re.sub(r"\s+\S+\s*$", f" .L{tgt:x}", text)
                if text.startswith("s_")
                else text
            )
        body.append(text)
    return body


def parse(path: str, kernel: str | None, objdump: bool):
    if objdump:
        return parse_objdump(path)
    body: list[str] = []
    active = False
    for ln in open(path, errors="replace").read().splitlines():
        s = ln.strip()
        if kernel and re.match(rf"^_Z\w*{kernel}\w*:", s):
            active = True
            continue
        if active and s.startswith(".Lfunc_end"):
            break
        if active or not kernel:
            body.append(s)
    return body


def mnemonic(line: str) -> str | None:
    s = line.split(";")[0].strip()
    if not s or s.startswith((".", " ")) or s.endswith(":"):
        return None
    return s.split()[0]


def hot_loop(body: list[str]):
    labels = {}
    for i, ln in enumerate(body):
        m = re.match(r"^(\.L\w+):", ln.strip())
        if m:
            labels[m.group(1)] = i
    best = (0, 0, 0)
    for i, ln in enumerate(body):
        m = re.match(r"^s_(?:cbranch\w*|branch)\s+(\.L\w+)", ln.strip())
        if m and m.group(1) in labels and labels[m.group(1)] < i:
            lo = labels[m.group(1)]
            n = sum(1 for x in body[lo : i + 1] if DOT.match(x.strip()))
            if n > best[0]:
                best = (n, lo, i)
    return best


def banks(line: str):
    ops = (
        [o.strip() for o in line.split(";")[0].split(None, 1)[1].split(",")]
        if " " in line
        else []
    )
    regs = []
    for o in ops[1:]:  # sources only
        m = re.match(r"^(?:-|\|)?v(\d+)", o)
        if m:
            regs.append(int(m.group(1)))
    return regs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("asm")
    ap.add_argument("--kernel", default="gemm_kernel")
    ap.add_argument("--objdump", action="store_true")
    ap.add_argument("--dump")
    args = ap.parse_args()
    body = parse(args.asm, None if args.objdump else args.kernel, args.objdump)
    n, lo, hi = hot_loop(body)
    if n == 0:
        sys.exit("no loop with dot instructions found")
    loop = [x.strip() for x in body[lo : hi + 1]]
    hist = collections.Counter(m for m in (mnemonic(x) for x in loop) if m)
    total = sum(hist.values())
    fma_class = sum(c for m, c in hist.items() if DOT.match(m))
    # the useful work is v_dot2; decode / scale FMAs of the fused kernels are overhead and count as "other VALU"
    dots = sum(c for m, c in hist.items() if DOT2.match(m)) or fma_class
    print(
        f"hot loop: {total} instructions, {dots} v_dot2 ({100 * dots / total:.1f}%), {fma_class - dots} other FMA-class"
    )
    for m, c in hist.most_common(24):
        print(f"  {m:28s} {c:5d}")
    ds_r = sum(
        c for m, c in hist.items() if m.startswith("ds_read") or m.startswith("ds_load")
    )
    ds_w = sum(
        c
        for m, c in hist.items()
        if m.startswith("ds_write") or m.startswith("ds_store")
    )
    gl = sum(
        c
        for m, c in hist.items()
        if m.startswith(("global_load", "buffer_load", "flat_load"))
    )
    valu = sum(c for m, c in hist.items() if m.startswith("v_"))
    valu_other = valu - dots
    print(
        f"per v_dot2: ds_read {ds_r / dots:.3f}  ds_write {ds_w / dots:.3f}  global_load {gl / dots:.3f}  "
        f"other VALU {valu_other / dots:.3f}  total instr {total / dots:.3f}"
    )
    print(f"VALU instructions per v_dot2: {valu / dots:.3f} -> issue-bound ceiling {18.432 * dots / valu:.2f} TF/s per GHz "
          f"(dot2 rate 18.43)")  # fmt: skip
    conflict = 0
    ndot = 0
    isdot = DOT2 if any(DOT2.match(m) for m in hist) else DOT
    for x in loop:
        if isdot.match(x):
            r = banks(x)
            ndot += 1
            if len(r) >= 2 and len({v % 4 for v in r[:2]}) < 2:
                conflict += 1
    print(f"dot src0/src1 in the same VGPR bank: {conflict}/{ndot}")
    if args.dump:
        open(args.dump, "w").write("\n".join(loop) + "\n")


if __name__ == "__main__":
    main()
