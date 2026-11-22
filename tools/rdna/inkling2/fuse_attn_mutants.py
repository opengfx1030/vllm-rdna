#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand mutants of ``csrc/rocm/inkling2_fuse_attn_rdna2.cu``: each fails the test.

    python tools/rdna/inkling2/fuse_attn_mutants.py --phase build  # CPU only
    python tools/rdna/inkling2/fuse_attn_mutants.py --phase test   # needs a GPU

Phase ``build`` compiles the unmutated file and every mutant with
``fuse_attn_jit.py`` (``--jobs N`` compiles in parallel; the build trees are shared
with the test through ``I2_JIT_DIR``), phase ``test`` runs
``tests/kernels/core/test_inkling2_fuse_attn_rdna2.py`` on each build with
``I2_FUSE_JIT`` set (``-x``: the first failing test ends a killed mutant).  The
unmutated file must pass; a mutant that passes is a hole of the test (or an
equivalent mutant) and makes the exit code 1.
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
SRC = REPO / "csrc" / "rocm" / "inkling2_fuse_attn_rdna2.cu"
TEST = REPO / "tests" / "kernels" / "core" / "test_inkling2_fuse_attn_rdna2.py"
JIT = REPO / "tools" / "rdna" / "inkling2" / "fuse_attn_jit.py"

# (name, what it breaks, old text (exactly once), new text, pytest -k selector or "")
MUTANTS = [
    (
        "conv_tap_order",
        "conv: tap / weight pairing of lane x",
        "o.x = tap[0].x * w0.w + tap[1].x * w0.z + tap[2].x * w0.y + tap[3].x * w0.x;",
        "o.x = tap[0].x * w0.x + tap[1].x * w0.z + tap[2].x * w0.y + tap[3].x * w0.w;",
        "",
    ),
    (
        "conv_inv_scale",
        "conv: the 1 / sqrt(5) scale of lane y dropped",
        "o.y = (o.y + tap[0].y) * inv;",
        "o.y = (o.y + tap[0].y);",
        "",
    ),
    (
        "conv_residual_tap",
        "conv: residual added is the previous token's input",
        "o.z = (o.z + tap[0].z) * inv;",
        "o.z = (o.z + tap[1].z) * inv;",
        "",
    ),
    (
        "state_write_skipped",
        "conv state is never written",
        "if (keep) write_ring(a.m, hp.blk_cur, tp.p, h, jch, a.o0, tap[0]);",
        "if (false) write_ring(a.m, hp.blk_cur, tp.p, h, jch, a.o0, tap[0]);",
        "",
    ),
    (
        "state_write_column",
        "conv state: stream column offset dropped in the write",
        "(int64_t)(p % m.ring) * m.cs2 + o0 + jch;",
        "(int64_t)(p % m.ring) * m.cs2 + jch;",
        "",
    ),
    (
        "history_ring_row",
        "conv history: ring row of a tap off by one",
        "(int64_t)(sp % m.ring) * m.cs2 + o0;",
        "(int64_t)((sp + 1) % m.ring) * m.cs2 + o0;",
        "",
    ),
    (
        "history_block",
        "conv history: block of a tap taken from the current position",
        "blk[lag - 1] = m.block_table[row + (max(sp, 0) >> m.blk_shift)];",
        "blk[lag - 1] = m.block_table[row + (max(tp.p, 0) >> m.blk_shift)];",
        "",
    ),
    (
        "token_position",
        "conv: position of the token off by one",
        "return {qs, p1 - (m.qsl[r + 1] - qs) + (t - qs), p1};",
        "return {qs, p1 - (m.qsl[r + 1] - qs) + (t - qs) + 1, p1};",
        "",
    ),
    (
        "null_block_write",
        "conv state: the null block (page 0) is written",
        "if (blk <= 0) return;",
        "if (blk < 0) return;",
        "null",
    ),
    (
        "pad_row_conv",
        "tokens of no request get a non-zero conv result",
        "float4 cv = z4;",
        "float4 cv = make_float4(1.f, 1.f, 1.f, 1.f);",
        "",
    ),
    (
        "x1_out_value",
        "x1_out receives zeros instead of the conv result",
        "if (a.x1_out != nullptr) store4(a.x1_out + t * a.x1o_t + 4 * v, cv);",
        "if (a.x1_out != nullptr) store4(a.x1_out + t * a.x1o_t + 4 * v, p);",
        "boundary",
    ),
    (
        "partial_drops_conv",
        "x0 + conv: lane x of the conv result not added",
        "p.x += cv.x;",
        "p.x += 0.f;",
        "",
    ),
    (
        "conv_only_lane",
        "conv alone: lane w of the partial state zeroed",
        "p = cv;",
        "p = make_float4(cv.x, cv.y, cv.z, 0.f);",
        "boundary",
    ),
    (
        "append_slot",
        "boundary: the appended state is stored one slot late",
        "store4(brow + (int64_t)a.n_blocks * a.bs_s + 4 * v, A[k]);",
        "store4(brow + (int64_t)(a.n_blocks + 1) * a.bs_s + 4 * v, A[k]);",
        "boundary",
    ),
    (
        "state_norm_eps",
        "attn_res: epsilon of the unweighted state norm doubled",
        "rsqrtf(tot[u][0] * a.inv_h + a.eps)",
        "rsqrtf(tot[u][0] * a.inv_h + 2.f * a.eps)",
        "",
    ),
    (
        "norm_weight_row",
        "attn_res: the three layer norms share the first weight row",
        "const int64_t wrow = (a.nw_rows == 1 ? 0 : q) * a.nw_s;",
        "const int64_t wrow = 0;",
        "",
    ),
    (
        "fill_query_layer",
        "messages: every layer uses the queries of layer 0",
        "const float* query = a.query + (int64_t)blockIdx.y * a.q_layer;",
        "const float* query = a.query;",
        "msgs_fill",
    ),
    (
        "fill_out_layer",
        "messages: every layer writes the output rows of layer 0",
        (
            "float* out = reinterpret_cast<float*>(a.out) + t * a.o_t +\n"
            "                 (int64_t)blockIdx.y * a.o_layer;"
        ),
        "float* out = reinterpret_cast<float*>(a.out) + t * a.o_t;",
        "msgs_fill",
    ),
    (
        "fill_scale",
        "host: attention scale H^-1/2 -> H^-1/4",
        "a.scale = (float)std::pow((double)H, -0.5);",
        "a.scale = (float)std::pow((double)H, -0.25);",
        "",
    ),
    (
        "softmax_wave_order",
        "attn_res: per-wave sums of the states added in the reverse order of the waves",
        "for (int w = 0; w < nwaves; ++w) acc += s_red[w][j * kQ1 + i];",
        "for (int w = nwaves - 1; w >= 0; --w) acc += s_red[w][j * kQ1 + i];",
        "",
    ),
    (
        "coef_lane",
        "attn_res: pass 2 takes the coefficient of the next state",
        "const float c = bcast_lane(coef[q][j >> 5], j & (kWave - 1));",
        "const float c = bcast_lane(coef[q][j >> 5], (j + 1) & (kWave - 1));",
        "",
    ),
    (
        "norm_weight_prefetch_hi",
        "attn_res: fp32 norm weights prefetched from the wrong half of the float4",
        "Wb[k] = load8(wbase + min(w_off + 8, (int64_t)a.H * esz - 8));",
        "Wb[k] = load8(wbase + min(w_off + 4, (int64_t)a.H * esz - 8));",
        "",
    ),
    (
        "norm_reduction_waves",
        "attn_res: the norm sum of squares skips the last wave",
        "for (int w = 0; w < nwaves; ++w) tot += s_nrm[w][q];",
        "for (int w = 0; w < nwaves - 1; ++w) tot += s_nrm[w][q];",
        "",
    ),
]


def _tag(path: pathlib.Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:10]


def _build(path: pathlib.Path, env: dict) -> tuple[str, int, str]:
    name = "i2_fuse_test_" + _tag(path)
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
    ap.add_argument("--phase", choices=("build", "test", "all"), default="all")
    ap.add_argument(
        "--work", default=os.environ.get("I2_JIT_DIR", "/tmp") + "/fuse_attn_mutants"
    )
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    text = SRC.read_text()
    chosen = [m for m in MUTANTS if not args.only or m[0] in args.only.split(",")]
    files = {}
    for name, _what, old, new, _k in chosen:
        assert text.count(old) == 1, f"{name}: pattern found {text.count(old)} times"
        p = work / f"mut_{name}.cu"
        p.write_text(text.replace(old, new))
        files[name] = p
    base = work / "base.cu"
    base.write_text(text)
    env = dict(os.environ)
    env["MAX_JOBS"] = "2"
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
    sel = {m[0]: m[4] for m in chosen}
    runs = [("BASE (unmutated)", base, "")] + [(n, files[n], sel[n]) for n in files]
    for label, path, k in runs:
        e = dict(os.environ, I2_FUSE_JIT=str(path))
        cmd = [sys.executable, "-m", "pytest", str(TEST), "-x", "-q",
               "-p", "no:cacheprovider"]  # fmt: skip
        if k:
            cmd += ["-k", k]
        r = subprocess.run(cmd, env=e, capture_output=True, text=True)
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
        print(f"  {label:24s} pytest exit {r.returncode}  {first[:130]}", flush=True)
    base_ok = results[0][1] == 0
    survivors = [r[0] for r in results[1:] if r[1] == 0]
    print("unmutated file passes:", base_ok)
    print(f"mutants killed: {len(results) - 1 - len(survivors)} / {len(results) - 1}")
    if survivors:
        print("SURVIVORS:", ", ".join(survivors))
    return 0 if base_ok and not survivors else 1


if __name__ == "__main__":
    sys.exit(main())
