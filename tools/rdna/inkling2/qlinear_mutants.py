#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand mutants of ``csrc/rocm/inkling2_qlinear_rdna2.cu``: ``qlinear_check.py`` must fail on
every one of them and pass on the unmutated control.

Each mutant is one deliberate bug in the arithmetic or the epilogue (the nibble order of
an int4 word, the int4 zero point, the byte selector of the int8 pairs, the scale of the
odd group of a pair, the tie rule of the argmax at the block / lane / wave level, the
vocabulary mask, the K-quarter reduction, the softcap, the scale of the high nibbles, the
tail mask).  A mutant the check survives would be a hole of the oracle test.

    python qlinear_mutants.py [--jobs 6]     # compiles on the CPU, checks on the GPU
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2] / "csrc" / "rocm" / "inkling2_qlinear_rdna2.cu"
FILTER = "(((M) == 1 || (M) == 4) && (C) == 8 && (W) == 8 && ((R) == 8 || (R) == 32))"
FORMATS = "8:64,4:32,4:64"

MUTANTS = {
    "m1_i4_nibble_map": (
        "plo[m] = dot2u(h2[j], x.z, plo[m]);",
        "plo[m] = dot2u(h3[j], x.z, plo[m]);",
    ),
    "m2_i4_zero_point": ("0x64086408u);  // - 1032", "0x64076407u);  // - 1032"),
    "m3_i8_perm": (
        "bytes_to_h2(wd[j], 0x04010400u)",
        "bytes_to_h2(wd[j], 0x04010401u)",
    ),
    "m4_scale_pair": (
        "const float sc = h2f(sw, (u / GC) & 1);",
        "const float sc = h2f(sw, ((u / GC) & 1) ^ 1);",
    ),
    "m5_block_tie": (
        "if (v > best_v || (v == best_v && ix < best_i)) {",
        "if (v > best_v || (v == best_v && ix > best_i)) {",
    ),
    "m6_lane_tie": (
        "if (val == val && val > bv[m]) {",
        "if (val == val && val >= bv[m]) {",
    ),
    "m7_vocab_mask": ("if (row >= sg.n_valid) val = -__builtin_inff();", ";"),
    "m8_kq_reduce": (
        "for (int off = R; off < 32; off <<= 1)",
        "for (int off = 2 * R; off < 32; off <<= 1)",
    ),
    "m9_softcap": ("return tanhf(v / p) * p;", "return tanhf(v) * p;"),
    "m10_wave_tie": ("if (ov > v || (ov == v && oi < ix)) {", "if (ov > v) {"),
    "m11_i4_hi_scale": (
        "Y[m] = fmaf(sc, fmaf(phi[m], 0.0625f, plo[m]), Y[m]);",
        "Y[m] = fmaf(sc, fmaf(phi[m], 0.125f, plo[m]), Y[m]);",
    ),
    "m12_tail_scale": (
        "s.n_valid = static_cast<int>(nv);",
        "s.n_valid = static_cast<int>(n);",
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument(
        "--work", help="directory of the mutated sources (default $I2_JIT_DIR/mutants)"
    )
    ap.add_argument("--jobs", type=int, default=4, help="parallel CPU compiles")
    args = ap.parse_args()
    text = Path(args.source).read_text()
    work = Path(
        args.work
        or Path(os.environ.get("I2_JIT_DIR", "~/.cache/vllm_inkling2_jit")).expanduser()
        / "mutants"
    )
    work.mkdir(parents=True, exist_ok=True)
    variants = {"m0_control": text}
    for name, (old, new) in MUTANTS.items():
        if old not in text:
            raise SystemExit(f"{name}: pattern not found in the kernel: {old!r}")
        variants[name] = text.replace(old, new, 1)
    for name, body in variants.items():
        (work / f"mut_{name}.cu").write_text(body)

    def compile_one(name: str) -> int:
        cmd = [
            sys.executable,
            str(HERE / "qlinear_jit.py"),
            "--name",
            f"mut_{name}",
            "--source",
            str(work / f"mut_{name}.cu"),
            "--formats",
            FORMATS,
            "--filter",
            FILTER,
        ]
        p = subprocess.run(
            cmd, capture_output=True, text=True, env={**os.environ, "MAX_JOBS": "3"}
        )
        print(f"compiled {name}: exit={p.returncode}", flush=True)
        return p.returncode

    with ThreadPoolExecutor(args.jobs) as pool:
        if any(pool.map(compile_one, variants)):
            print("a mutant did not compile")
            return 2
    names = sorted(variants, key=lambda n: int(re.match(r"m(\d+)", n).group(1)))
    results = {}
    for name in names:
        cmd = [
            sys.executable,
            str(HERE / "qlinear_check.py"),
            "--jit",
            "--source",
            str(work / f"mut_{name}.cu"),
            "--name",
            f"mut_{name}",
            "--filter",
            FILTER,
            "--formats",
            FORMATS,
            "--cpis",
            "8",
            "--wvs",
            "8",
            "--ms",
            "1,4",
            "--rs",
            "8,32",
            "--quick",
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        lines = (p.stdout + p.stderr).splitlines()
        fails = [ln for ln in lines if ln.startswith("FAIL")]
        verdict = next(
            (ln for ln in lines if ln.startswith("CHECK")),
            lines[-1][:120] if lines else "",
        )
        results[name] = p.returncode
        print(
            f"{name}: exit={p.returncode} {verdict} | first: {fails[0][:140] if fails else ''}",
            flush=True,
        )
    ok = results["m0_control"] == 0 and all(
        rc != 0 for n, rc in results.items() if n != "m0_control"
    )
    print(
        "MUTANTS",
        "ALL KILLED, CONTROL PASSES" if ok else "SURVIVOR OR CONTROL FAILURE",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
