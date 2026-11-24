#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand mutants of ``csrc/rocm/moe_prefill_bm_rdna2.cu``: each fails the kernel test.

    python tools/rdna/inkling2/moe_prefill_bm_mutants.py [--jobs 4] [--only NAME,...]

Phase 1 builds the unmutated file and every mutant with ``moe_prefill_bm_jit.py`` (CPU
only, ``--jobs`` compilations in parallel, the build trees are shared with the test
through ``I2_JIT_DIR``), phase 2 runs the kernel test
``tests/kernels/core/test_inkling2_moe_prefill_bm_rdna2.py`` on each build with
``I2_MOE_PREFILL_JIT`` set (``-x``: the first failing test ends a killed mutant).  The
unmutated file must pass; a mutant that passes is a hole of the test (or an equivalent
mutant) and makes the exit code 1.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
SRC = REPO / "csrc" / "rocm" / "moe_prefill_bm_rdna2.cu"
TEST = REPO / "tests" / "kernels" / "core" / "test_inkling2_moe_prefill_bm_rdna2.py"
JIT = REPO / "tools" / "rdna" / "inkling2" / "moe_prefill_bm_jit.py"

# (name, what it breaks, old text (exactly once), new text)
MUTANTS = [
    (
        "perm_k8_selector",
        "activation K permutation of a 8-K word (NVFP4)",
        "p.x = __builtin_amdgcn_perm(v.z, v.x, 0x05040100u);",
        "p.x = __builtin_amdgcn_perm(v.z, v.x, 0x07060302u);",
    ),
    (
        "nibble_sign_shift",
        "sign bit of a dequantised nibble goes to bit 14",
        "constexpr int SS = 12 - 4 * J;",
        "constexpr int SS = 11 - 4 * J;",
    ),
    (
        "global_scale",
        "per-expert global scale of NVFP4 halved",
        "b_g[expert_id] * 4194304.0f;",
        "b_g[expert_id] * 2097152.0f;",
    ),
    (
        "e4m3_scale_mask",
        "e4m3 scale mantissa bit dropped",
        "((b << 7) & 0x3F80u)",
        "((b << 7) & 0x3F00u)",
    ),
    (
        "weight_prefetch_group",
        "second register set prefetches the wrong group",
        "const int gn = min(G + 2, total_groups - 1);",
        "const int gn = min(G + 1, total_groups - 1);",
    ),
    (
        "k_chunk_count",
        "the last K chunk is never computed",
        "const int nchunks = (size_k + KC - 1) / KC;",
        "const int nchunks = (size_k + KC - 1) / KC - 1;",
    ),
    (
        "nvfp4_k_tail_groups",
        "NVFP4 K-tail drops the last 32-K pair",
        "const int ng = min(KC, size_k - ck * KC) >> 4;",
        "const int ng = max(0, (min(KC, size_k - ck * KC) >> 4) - 2);",
    ),
    ("mxfp8_block_factor", "UE8M0 block scale off by 2", "* 256.0f));", "* 128.0f));"),
    (
        "mxfp8_lds_row_offset",
        "activation window of the 32-K step",
        "const half* arow = lds_a + 32 * g;",
        "const half* arow = lds_a + 16 * g;",
    ),
    (
        "mxfp8_sign",
        "e4m3 sign bit of the second byte",
        "((lanes << 8) & 0x80008000u)",
        "((lanes << 7) & 0x80008000u)",
    ),
    (
        "padding_row_valid",
        "the padding id (token == size_m) is accepted as a row",
        "if (id / top_k >= size_m) id = -1;",
        "if (id / top_k > size_m) id = -1;",
    ),
    (
        "router_weight",
        "w2: router weight not applied",
        "for (int j = 0; j < 4; ++j) r[j] *= tw;",
        "for (int j = 0; j < 4; ++j) r[j] *= 1.0f;",
    ),
    (
        "topk_reduction_row",
        "w2: output row of a routed id",
        "(int64_t)(id / out_topk)",
        "(int64_t)(id % out_topk)",
    ),
    (
        "atomic_add_is_store",
        "w2: the top-k rows overwrite instead of adding",
        "atomic_add_pk4_f16(out, r01, r23);",
        (
            "{ uint2 u; u.x = __builtin_bit_cast(uint32_t, r01); "
            "u.y = __builtin_bit_cast(uint32_t, r23); *(uint2*)out = u; }"
        ),
    ),
    (
        "row_groups_counted",
        "only the first 8-row group of a block is computed",
        "n += (sh_id[8 * g] >= 0) ? 1 : 0;",
        "n = 1;",
    ),
    (
        "work_list_row_offset",
        "first row of the second full work block of an expert",
        "(s_lb[lo] + (c - s_nf[lo]) * R) * ba",
        "(s_lb[lo] + (c - s_nf[lo]) * (R - 1)) * ba",
    ),
    (
        "work_list_partial_order",
        "partial blocks shortest first",
        "(p2 > p || (p2 == p && e2 < e))",
        "(p2 < p || (p2 == p && e2 < e))",
    ),
    (
        "work_list_expert_search",
        "expert of a full block off by one",
        "info = make_int4(lo, (s_lb[lo]",
        "info = make_int4(lo > 0 ? lo - 1 : lo, (s_lb[lo]",
    ),
    # W4A8 kernel and the activation quantiser
    (
        "quant_pair_order",
        "act quantiser: K order inside an 8-K word",
        "pk(8 * kw + 0, 8 * kw + 2, 8 * kw + 4, 8 * kw + 6)",
        "pk(8 * kw + 2, 8 * kw + 0, 8 * kw + 4, 8 * kw + 6)",
    ),
    (
        "quant_inv_scale",
        "act quantiser: 126 / amax instead of 127 / amax",
        "127.0f / amax",
        "126.0f / amax",
    ),
    (
        "lut_neg_table",
        "e2m1 LUT: one entry of the negative table",
        "0xF4F8FAFCu",
        "0xF4F8FAFDu",
    ),
    (
        "lut_sign_select",
        "e2m1 LUT: sign bit position of the select",
        "((z >> 1) & 0x04040404u)",
        "((z >> 2) & 0x04040404u)",
    ),
    (
        "lut_odd_nibbles",
        "odd nibbles dequantised like the even ones",
        "dq[g][wd][j][1] = lut_e2m1_x2<4>(wc[j]);",
        "dq[g][wd][j][1] = lut_e2m1_x2<0>(wc[j]);",
    ),
    (
        "scale_align_mm1",
        "second group aligned by the first group's exponent gap",
        "mm1[j] = (int)((M1 << 6) >> (Em - E1));",
        "mm1[j] = (int)((M1 << 6) >> (Em - E0));",
    ),
    (
        "scale_align_mm0",
        "first group multiplier off by a factor 2",
        "mm0[j] = (int)((M0 << 6) >> (Em - E0));",
        "mm0[j] = (int)((M0 << 5) >> (Em - E0));",
    ),
    (
        "scale_exponent",
        "window scale 2^(Em - 16) instead of 2^(Em - 17)",
        "(Em + 110u) << 23",
        "(Em + 111u) << 23",
    ),
    (
        "window_scale_index",
        "activation scale of window wi of a chunk",
        "+ (k0 >> 5) + wi]",
        "+ (k0 >> 5)]",
    ),
    (
        "dot_chain_init",
        "second dot chain starts from the first one's sum",
        "int p1 = sdot4_init(dq[1][0][j][0], a1.x);",
        "int p1 = sdot4_acc(dq[1][0][j][0], a1.x, p0);",
    ),
    (
        "act_tile_offset",
        "int8 activation tile: store offset of a 16-K piece",
        "*(uint4*)(lds_a + r * KC + 16 * ch) = v;",
        "*(uint4*)(lds_a + r * KC + 8 * ch) = v;",
    ),
]


def _tag(path: pathlib.Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:10]


def _build(path: pathlib.Path, env: dict) -> tuple[str, int, str]:
    name = "i2_moe_prefill_test_" + _tag(path)
    r = subprocess.run(
        [sys.executable, str(JIT), "--source", str(path), "--name", name],
        env=env,
        capture_output=True,
        text=True,
    )
    return path.name, r.returncode, (r.stdout + r.stderr)[-1500:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument(
        "--phase",
        choices=("build", "test", "all"),
        default="all",
        help="build: compile only (no GPU needed); test: run the tests on the builds",
    )
    ap.add_argument(
        "--work", default=os.environ.get("I2_JIT_DIR", "/tmp") + "/moe_prefill_mutants"
    )
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    text = SRC.read_text()
    chosen = [m for m in MUTANTS if not args.only or m[0] in args.only.split(",")]
    files = {}
    for name, _what, old, new in chosen:
        assert text.count(old) == 1, f"{name}: pattern found {text.count(old)} times"
        p = work / f"mut_{name}.cu"
        p.write_text(text.replace(old, new))
        files[name] = p
    base = work / "base.cu"
    base.write_text(text)
    env = dict(os.environ)
    env["MAX_JOBS"] = "4"
    todo = [base, *files.values()]
    if args.phase in ("build", "all"):
        print(f"phase 1: building {len(todo)} files, {args.jobs} at a time", flush=True)
        with cf.ThreadPoolExecutor(args.jobs) as ex:
            for name, rc, tail in ex.map(lambda p: _build(p, env), todo):
                print(f"  build {name}: {'ok' if rc == 0 else 'FAILED'}", flush=True)
                if rc != 0:
                    print(tail)
                    return 2
    if args.phase == "build":
        return 0
    results = []
    for label, path in [("BASE (unmutated)", base)] + [(n, files[n]) for n in files]:
        e = dict(os.environ, I2_MOE_PREFILL_JIT=str(path))
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(TEST),
                "-x",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            env=e,
            capture_output=True,
            text=True,
        )
        out = r.stdout + r.stderr
        first = next(
            (
                ln
                for ln in out.splitlines()
                if ln.startswith("FAILED") or " failed" in ln
            ),
            "",
        )
        results.append((label, r.returncode, first))
        print(f"  {label:28s} pytest exit {r.returncode}  {first[:120]}", flush=True)
    base_ok = results[0][1] == 0
    survivors = [r[0] for r in results[1:] if r[1] == 0]
    print("unmutated file passes:", base_ok)
    print(f"mutants killed: {len(results) - 1 - len(survivors)} / {len(results) - 1}")
    if survivors:
        print("SURVIVORS:", ", ".join(survivors))
    return 0 if base_ok and not survivors else 1


if __name__ == "__main__":
    sys.exit(main())
