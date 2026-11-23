#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand mutants of the prefill tile kernel of ``csrc/rocm/inkling2_attn_rdna2.cu``.

    python tools/rdna/inkling2/attn_tile_mutants.py --phase build  # CPU only
    python tools/rdna/inkling2/attn_tile_mutants.py --phase test   # needs a gfx1030 GPU

Each mutant is one edit inside the tile kernel (``--kernel``, default
``i2_tile_kernel``; the edit must match exactly once inside that kernel).  Phase
``build`` compiles the unmutated file and every mutant with ``attn_jit.py`` (``--jobs
N`` in parallel, the build trees are shared with the test through ``I2_JIT_DIR``), phase
``test`` runs ``tests/kernels/core/test_inkling2_attn_prefill_depth_rdna2.py`` once per
build with ``I2_ATTN_VARIANT_JIT`` set (the *variant* is the mutant; the shipped
reference stays the unmutated op, so a mutant is caught by the bit-identity check or by
the fp64 oracle) and ``-x`` (the first failing test ends a killed mutant).  The
unmutated file must pass; a mutant that passes is a hole of the battery (or an
equivalent mutant: listed under ``EQUIVALENT``, they are not run).

``--source`` is the file to mutate (default: the repo's), ``--macros`` its ``-D`` flags
(for a work-in-progress file whose kernels are chosen by macros), ``--shipped`` the file
of the shipped reference (``I2_ATTN_SHIPPED_JIT``: the 9285bf8523 ``.cu`` when the tree
already holds a variant that is selected by an argument; the op with ``tile_impl`` is
then the variant under test, so use ``--tile-impl``).  Exit status 0: the unmutated file
passes and every mutant is killed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
import pathlib
import subprocess
import sys

import regex as re

REPO = pathlib.Path(__file__).resolve().parents[3]
SRC = REPO / "csrc" / "rocm" / "inkling2_attn_rdna2.cu"
TEST = REPO / "tests" / "kernels" / "core" / "test_inkling2_attn_prefill_depth_rdna2.py"
sys.path.insert(0, str(REPO / "tools"))

# (name, what it breaks, old text, new text[, scope[, nth]]): the old text is in the
# kernel exactly once, or ``nth`` (0-based) picks one of several occurrences; ``scope``
# is the function to edit when it is not the tile kernel.
MUTANTS = [
    (
        "window_edge",
        "sliding window: the key one before the window is visible",
        "(a.window <= 0 || j > pq - a.window)",
        "(a.window <= 0 || j >= pq - a.window)",
    ),
    (
        "window_edge_in",
        "sliding window: the oldest key of the window is masked",
        "(a.window <= 0 || j > pq - a.window)",
        "(a.window <= 0 || j > pq - a.window + 1)",
    ),
    (
        "causal_edge",
        "causal mask: a row sees the key after it",
        "(j <= pq)",
        "(j <= pq + 1)",
    ),
    (
        "bias_reach_in",
        "relative bias: the key at distance rel_extent gets a bias",
        "const bool inb = dist < a.rel_extent;",
        "const bool inb = dist <= a.rel_extent;",
    ),
    (
        "bias_reach_out",
        "relative bias: the key at distance rel_extent - 1 gets none",
        "const bool inb = dist < a.rel_extent;",
        "const bool inb = dist < a.rel_extent - 1;",
    ),
    (
        "bias_clamp_self",
        "relative bias: the key at distance 0 reads proj row 1 instead of row 0",
        "const int idx = min(max(dist, 0), a.rel_extent - 1);",
        "const int idx = min(max(dist, 1), a.rel_extent - 1);",
    ),
    (
        "v_pair_swapped",
        "V staging: the pair (m, m + 32) is stored swapped",
        "lo4.x = (va.x & 0xFFFFu) | (vb.x << 16);",
        "lo4.x = (vb.x & 0xFFFFu) | (va.x << 16);",
    ),
    (
        "split_one_tile",
        "split-K: a split walks one tile too few",
        "const int t1 = min(ts.t_hi, t0 + ts.tps);",
        "const int t1 = min(ts.t_hi, t0 + ts.tps - (ts.tps > 1 ? 1 : 0));",
    ),
    (
        "window_start_tile",
        "key range: the first tile is one too late",
        "s.t_lo = s.lo / kI2BC;",
        "s.t_lo = (s.lo + 1) / kI2BC;",
        "i2_tile_split",
    ),
]

# Edits that only a restructured kernel has (the staging, the double-buffered page
# table, the skip of the bias of whole tiles); they are matched against ``--kernel``
# and are not applicable to the shipped kernel (``--variant-mutants`` selects them).
VARIANT_MUTANTS = [
    (
        "skip_bias_one_early",
        "bias skip: the tile whose last key is at distance rel_extent - 1 is skipped",
        "if (p_first - (n0 + kI2BC - 1) >= a.rel_extent) {",
        "if (p_first - (n0 + kI2BC - 1) >= a.rel_extent - 1) {",
    ),
    (
        "skip_bias_last_row",
        "bias skip: tested at the last row of the CTA instead of the first",
        "if (p_first - (n0 + kI2BC - 1) >= a.rel_extent) {",
        "if (p_first + ntok - 1 - (n0 + kI2BC - 1) >= a.rel_extent) {",
    ),
    (
        "mask_skip_one_late",
        "mask skip: an interior tile may end one key past the first row of the block",
        "if (ntok == kI2BR && n0 + kI2BC - 1 <= p_first &&",
        "if (ntok == kI2BR && n0 + kI2BC - 1 <= p_first + 1 &&",
    ),
    (
        "page_parity",
        "page table double buffer: the next tile's pages land in the current slot",
        "s_page[(tile + 1) & 1][tid] =",
        "s_page[tile & 1][tid] =",
    ),
    (
        "prefetch_same_tile",
        "K prefetch: the next tile's K rows are loaded from the current tile",
        "if (tile + 1 < t1) load_k(tile + 1);",
        "if (tile + 1 < t1) load_k(tile);",
    ),
    (
        "prefetch_guard",
        (
            "K prefetch: the guard of the last tile is dropped (the extra load is "
            "never used: an equivalent mutant unless it faults or reads a stale slot)"
        ),
        "if (tile + 1 < t1) load_k(tile + 1);",
        "load_k(tile + 1);",
    ),
    (
        "fast_path_hi",
        "V fast path: taken when the tile runs one key past the last row",
        (
            "if (uni && n0 >= lo_all && n0 + kI2BC <= hi_all) {\n"
            "      const char* base = tile_base(tile) + kI2D * 2;"
        ),
        (
            "if (uni && n0 >= lo_all && n0 + kI2BC <= hi_all + 1) {\n"
            "      const char* base = tile_base(tile) + kI2D * 2;"
        ),
    ),
    (
        "fast_path_lo",
        "V fast path: taken when the tile starts one key before the window",
        (
            "if (uni && n0 >= lo_all && n0 + kI2BC <= hi_all) {\n"
            "      const char* base = tile_base(tile) + kI2D * 2;"
        ),
        (
            "if (uni && n0 + 1 >= lo_all && n0 + kI2BC <= hi_all) {\n"
            "      const char* base = tile_base(tile) + kI2D * 2;"
        ),
    ),
    (
        "row_share_lane",
        "probabilities in registers: the DPP row_share reads the next lane",
        "i2_row_share<(m & 15)>(pk[ri][m >> 4])",
        "i2_row_share<((m + 1) & 15)>(pk[ri][m >> 4])",
    ),
    (
        "row_share_word",
        "probabilities in registers: the DPP row_share reads the other pair word",
        "i2_row_share<(m & 15)>(pk[ri][m >> 4])",
        "i2_row_share<(m & 15)>(pk[ri][(m >> 4) ^ 1])",
    ),
    (
        "pack_swapped",
        "probabilities in registers: the pair (p0, p2) is packed swapped",
        "pk[ri][0] = i2_pack(p[0], p[2]);",
        "pk[ri][0] = i2_pack(p[2], p[0]);",
        None,
        0,
    ),
    (
        "pack_swapped_edge",
        "probabilities in registers: the same on the edge-tile path",
        "pk[ri][0] = i2_pack(p[0], p[2]);",
        "pk[ri][0] = i2_pack(p[2], p[0]);",
        None,
        1,
    ),
    (
        "prefetch_v_same_tile",
        "V prefetch: the next tile's V rows are loaded from the current tile",
        "if (tile + 1 < t1) load_v(tile + 1);",
        "if (tile + 1 < t1) load_v(tile);",
    ),
    (
        "v_buffer_early",
        "single K/V buffer: V rows stored before every QK reader is done",
        "__syncthreads();  // (2)",
        "// (2) removed",
    ),
]

# Probes are reported but never fail the run: an edit that is equivalent unless the
# extra load faults (the dropped prefetch guard), and a data race (a missing barrier)
# that the hardware schedule may or may not expose.
PROBES = {"prefetch_guard", "v_buffer_early"}

# Edits that change nothing the output can show (documented, not run).
EQUIVALENT = {
    "k_fast_path_hi": "K fast path of a tile that runs one key past the last row "
    "(`n0 + kI2BC <= hi_all + 1`): the extra key is masked by the causal select, a "
    "NaN score never reaches the sum (the V twin of this edit is `fast_path_hi`)",
    "k_fast_path_lo": "K fast path of a tile that starts one key before the window "
    "(`n0 + 1 >= lo_all`): that key's score is masked (the V twin is `fast_path_lo`)",
    "bias_idx_clamp_high": "min(max(dist, 0), a.rel_extent - 1) -> rel_extent: only "
    "reads the proj row of a key that `inb` zeroes",
    "bias_clamp_low": "max(dist, 0) -> max(dist, 1): only changes the proj row of keys "
    "after the row, which the causal mask removes",
}


def kernel_span(text: str, kernel: str) -> tuple[int, int]:
    """[start, end) of the first definition of ``kernel`` (up to the closing brace at
    column 0)."""
    m = re.search(rf"\b(?:void|I2TileSplit)\s+{re.escape(kernel)}\s*\(", text)
    if m is None:
        raise SystemExit(f"kernel {kernel!r} not found")
    end = text.index("\n}\n", m.end())
    return m.start(), end + 3


def make_mutant(
    text: str, kernel: str, old: str, new: str, nth: int | None = None
) -> str | None:
    """``text`` with ``old`` replaced by ``new`` inside ``kernel`` (None: ``old`` is not
    there exactly once, or less than ``nth + 1`` times when ``nth`` is given)."""
    lo, hi = kernel_span(text, kernel)
    body = text[lo:hi]
    n = body.count(old)
    if (nth is None and n != 1) or (nth is not None and n <= nth):
        return None
    at = -1
    for _ in range((nth or 0) + 1):
        at = body.index(old, at + 1)
    return text[:lo] + body[:at] + new + body[at + len(old) :] + text[hi:]


def mutants(args) -> list[tuple]:
    return MUTANTS + (VARIANT_MUTANTS if args.variant_mutants else [])


def source_of(
    args, name: str, src: str, text: str, *, root: pathlib.Path
) -> pathlib.Path:
    path = root / f"{name}.cu"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def spec(path: pathlib.Path, macros: str) -> str:
    return f"{path}@{macros}" if macros else str(path)


def build_one(name: str, path: pathlib.Path, macros: str) -> str:
    from rdna.inkling2 import attn_jit

    attn_jit.build(spec(path, macros))
    return name


def prepare(args) -> list[tuple[str, str, pathlib.Path | None]]:
    """[(name, description, file)] with the unmutated file first; a file of None is a
    mutant whose anchor is missing."""
    src = pathlib.Path(args.source or SRC)
    text = src.read_text()
    root = pathlib.Path(args.work).expanduser()
    tag = hashlib.md5(text.encode()).hexdigest()[:8]
    out: list[tuple[str, str, pathlib.Path | None]] = [
        ("unmutated", "the file as it is (must pass)", src)
    ]
    for name, what, old, new, *extra in mutants(args):
        scope = extra[0] if extra and extra[0] else args.kernel
        mutated = make_mutant(
            text, scope, old, new, extra[1] if len(extra) > 1 else None
        )
        path = None
        if mutated is not None:
            path = source_of(args, name, what, mutated, root=root / tag)
        out.append((name, what, path))
    return out


def run_test(
    args, name: str, path: pathlib.Path, expect_pass: bool
) -> tuple[bool, str]:
    env = dict(os.environ)
    env["I2_ATTN_VARIANT_JIT"] = spec(path, args.macros)
    env["I2_ATTN_REQUIRE_VARIANT"] = "1"
    env.setdefault("I2_ATTN_DEPTH_MAX_L", str(args.max_l))
    if args.shipped:
        env["I2_ATTN_SHIPPED_JIT"] = args.shipped
    if args.tile_impl is not None:
        env["I2_ATTN_VARIANT_TILE_IMPL"] = str(args.tile_impl)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(TEST),
        "-q",
        "-x",
        "-p",
        "no:cacheprovider",
    ]
    if args.k:
        cmd += ["-k", args.k]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO)
    log = pathlib.Path(args.work).expanduser() / "logs"
    log.mkdir(parents=True, exist_ok=True)
    (log / f"{name}.log").write_text(proc.stdout + proc.stderr)
    failed = [ln for ln in proc.stdout.splitlines() if ln.startswith("FAILED")]
    out = proc.stdout + proc.stderr
    if failed:
        why = re.sub(r"\s+", " ", failed[0])[:200]
    elif "Fatal Python error" in out:
        why = "crash (Fatal Python error): " + proc.stdout.strip()[:40]
    else:
        why = proc.stdout.strip()[-200:]
    return proc.returncode == 0, why


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["build", "test"], required=True)
    ap.add_argument("--source", default="")
    ap.add_argument("--macros", default="")
    ap.add_argument("--kernel", default="i2_tile_kernel")
    ap.add_argument("--variant-mutants", action="store_true")
    ap.add_argument("--shipped", default="")
    ap.add_argument("--tile-impl", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="", help="comma list of mutant names")
    ap.add_argument("--k", default="", help="pytest -k expression")
    ap.add_argument("--max-l", type=int, default=20000)
    ap.add_argument(
        "--work",
        default=os.environ.get("I2_MUTANT_DIR", "~/.cache/vllm_inkling2_mutants"),
    )
    args = ap.parse_args()
    plan = prepare(args)
    if args.only:
        keep = set(args.only.split(",")) | {"unmutated"}
        plan = [p for p in plan if p[0] in keep]
    missing = [n for n, _, p in plan if p is None]
    if missing:
        print("ANCHOR MISSING (edit the mutant, not the kernel):", ", ".join(missing))
        return 2
    if args.phase == "build":
        with cf.ThreadPoolExecutor(args.jobs) as ex:
            futs = [ex.submit(build_one, n, p, args.macros) for n, _, p in plan]
            for f in futs:
                print("built", f.result(), flush=True)
        return 0
    bad = 0
    for name, what, path in plan:
        assert path is not None
        ok, why = run_test(args, name, path, expect_pass=name == "unmutated")
        if name == "unmutated":
            verdict = "PASS" if ok else "FAIL (the control must pass)"
            bad += not ok
        else:
            probe = name in PROBES
            verdict = (
                f"SURVIVED{' (probe)' if probe else '  <-- hole'}: {what}"
                if ok
                else "killed"
            )
            bad += ok and not probe
        print(
            f"{name:24s} {verdict}"
            + ("" if name == "unmutated" or ok else f"  [{why}]"),
            flush=True,
        )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
