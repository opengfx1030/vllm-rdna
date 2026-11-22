#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Oracle test of ``inkling2_qlinear_{grouped,lm_head}`` (weight-only int8 / int4 decode GEMV).

The reference is the fp64 matmul with the dequantized weights ``q * scale``: the exact
weights the kernel multiplies with, so the only difference left is the fp32 accumulation
order (relative error ~1e-6).  The test covers

* every element position of a chunk (unit-vector activations: bit-exact),
* random activations for K = 2304 / 3072, N tails, strided activations and outputs, M = 1..4,
* segment epilogues (sigmoid / tanh / softcap), the ``-inf`` vocabulary mask, fp16 outputs,
* the lm_head logits, the fused argmax partial with ties inside a wave, across waves and
  across the tiles of a lane (the first index wins) and ``col_base``,
* the production packs (``qlinear_pack``): attention input pack / o_proj / lm_head built from
  fp16 weights, the quantized path up to 4 rows and the fp16 path above.

    python qlinear_check.py                      # the installed ops, the production packs
    python qlinear_check.py --jit [--filter EXPR] # JIT build of the kernel, all instances
                                                 # the filter keeps (qlinear_jit.py)

The exit code is 1 when anything fails (``qlinear_mutants.py`` needs that).
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

FAILS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print("FAIL", msg, flush=True)


class JitBackend:
    def __init__(self, mod) -> None:
        self.mod = mod

    def grouped(
        self, xs, wq, sc, outs, seg_x, seg_n, acts, ps, valid, bits, group, R, cpi, wv
    ):
        fn = getattr(self.mod, f"grouped_b{bits}g{group}")
        fn(xs, wq, sc, outs, seg_x, seg_n, acts, ps, valid, bits, group, R, cpi, wv)

    def lm_head(
        self,
        x,
        wq,
        sc,
        logits,
        pv,
        pi,
        softcap,
        n_rows,
        n_valid,
        col_base,
        bits,
        group,
        cpi,
    ):
        fn = getattr(self.mod, f"lm_head_b{bits}g{group}")
        fn(
            x,
            wq,
            sc,
            logits,
            pv,
            pi,
            softcap,
            n_rows,
            n_valid,
            col_base,
            bits,
            group,
            cpi,
        )


class OpBackend:
    def grouped(self, *args):
        from vllm import _custom_ops as ops

        ops.inkling2_qlinear_grouped(*args)

    def lm_head(self, *args):
        from vllm import _custom_ops as ops

        ops.inkling2_qlinear_lm_head(*args)


def rand_w(n, K, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(n, K, device=dev, generator=g)
    w = w * (0.005 + 0.05 * torch.rand(n, 1, device=dev, generator=g))
    w[:, :: max(1, K // 7)] *= 6.0  # outlier columns: groups of very different scales
    return w.half()


def oracle(qp, x, w, bits, group, clip=False):
    q, s = qp.quantize_rows(w, bits, group, clip)
    return x.double() @ qp.dequantize_rows(q, s, group).double().t()


def check_grouped(qp, be, bits, group, R, cpi, wv, M, K, rows, seg_x, tag):
    dev = "cuda"
    ws = [rand_w(n, K, dev, 10 + i) for i, n in enumerate(rows)]
    packs = [qp.pack_weight(w, bits, group, R, cpi) for w in ws]
    wq = torch.cat([p[0] for p in packs]).contiguous()
    sc = torch.cat([p[1] for p in packs]).contiguous()
    g = torch.Generator(device=dev).manual_seed(3)
    ldx = K + 16  # activations: a view of a wider buffer
    bufs = [
        torch.randn(M, ldx, device=dev, generator=g).half()
        for _ in range(max(seg_x) + 1)
    ]
    xs = [b[:, :K] for b in bufs]
    # outputs with a row stride larger than n; the padding must stay untouched
    outs = [
        torch.full((M, n + 8), 7.0, dtype=torch.float32, device=dev)[:, :n]
        for n in rows
    ]
    be.grouped(xs, wq, sc, outs, seg_x, list(rows), [], [], [], bits, group, R, cpi, wv)
    torch.cuda.synchronize()
    worst = 0.0
    for i, (o, w) in enumerate(zip(outs, ws)):
        ref = oracle(qp, xs[seg_x[i]], w, bits, group)
        err = float((o.double() - ref).abs().max())
        rms = math.sqrt(float((ref**2).mean()))
        worst = max(worst, err / rms)
        if not err / rms < 2e-5:
            fail(f"{tag} seg {i}: err {err:.3e} rms {rms:.3e}")
        full = torch.as_strided(o, (M, rows[i] + 8), (o.stride(0), 1))
        if not bool((full[:, rows[i] :] == 7.0).all()):
            fail(f"{tag} seg {i}: wrote past the row end")
    return worst


def check_onehot(qp, be, bits, group, R, cpi, wv, M, K, N, tag, step):
    """X = unit vectors: y[m, n] must equal the dequantized weight element exactly."""
    dev = "cuda"
    w = rand_w(N, K, dev, 5)
    wq, sc = qp.pack_weight(w, bits, group, R, cpi)
    q, s = qp.quantize_rows(w, bits, group)
    deq = qp.dequantize_rows(q, s, group)
    bad = 0
    for k0 in range(0, K - M + 1, step * M):
        x = torch.zeros(M, K, dtype=torch.float16, device=dev)
        for m in range(M):
            x[m, k0 + m] = 1.0
        out = torch.zeros(M, N, dtype=torch.float32, device=dev)
        be.grouped([x], wq, sc, [out], [0], [N], [], [], [], bits, group, R, cpi, wv)
        want = torch.stack([deq[:, k0 + m] for m in range(M)])
        if not torch.equal(out, want):
            bad += 1
            if bad <= 3:
                fail(
                    f"{tag} one-hot k0={k0}: max diff {float((out - want).abs().max()):.3e}"
                )
    torch.cuda.synchronize()


def check_epilogue(qp, be, bits, group, R, cpi, wv, M, K, tag):
    dev = "cuda"
    rows, acts, ps, valid = (
        [200, 77, 130, 33],
        [1, 2, 3, 0],
        [1.0, 1.0, 5.0, 1.0],
        [200, 77, 130, 20],
    )
    ws = [rand_w(n, K, dev, 30 + i) for i, n in enumerate(rows)]
    packs = [qp.pack_weight(w, bits, group, R, cpi) for w in ws]
    wq = torch.cat([p[0] for p in packs]).contiguous()
    sc = torch.cat([p[1] for p in packs]).contiguous()
    g = torch.Generator(device=dev).manual_seed(4)
    x = torch.randn(M, K, device=dev, generator=g).half() * 0.5
    for dt, atol in ((torch.float32, 3e-5), (torch.float16, 2e-3)):
        outs = [torch.zeros(M, n, dtype=dt, device=dev) for n in rows]
        be.grouped(
            [x], wq, sc, outs, [0] * 4, rows, acts, ps, valid, bits, group, R, cpi, wv
        )
        torch.cuda.synchronize()
        for i in range(4):
            ref = oracle(qp, x, ws[i], bits, group)
            ref = (
                torch.sigmoid(ref),
                torch.tanh(ref),
                torch.tanh(ref / ps[i]) * ps[i],
                ref,
            )[[1, 2, 3, 0].index(acts[i])]
            ref[:, valid[i] :] = -math.inf
            o = outs[i].double()
            fin = torch.isfinite(ref)
            if not bool((torch.isinf(o) == torch.isinf(ref)).all()) or not bool(
                (o[~fin] == ref[~fin]).all()
            ):
                fail(f"{tag} epilogue seg {i} ({dt}): mask mismatch")
                continue
            err = float((o[fin] - ref[fin]).abs().max())
            if not err < atol * max(1.0, float(ref[fin].abs().max())):
                fail(f"{tag} epilogue seg {i} ({dt}): err {err:.3e}")


def check_lm_head(qp, be, bits, group, cpi, M, K, N, nparts, tag):
    dev = "cuda"
    w = rand_w(N, K, dev, 50)
    g = torch.Generator(device=dev).manual_seed(6)
    x = torch.randn(M, K, device=dev, generator=g).half()
    # ties: rows 5, 9 (another lane of the wave), 37 (another wave of the block) and
    # 5 + 512 * 32 (another tile of the same lane) are identical and dominate every logit:
    # the first index must win
    big = (x[0].float() / x[0].float().norm() * 0.9).half()
    tie_rows = [5, 9, 37] + ([5 + 512 * 32] if N > 5 + 512 * 32 else [])
    for r in tie_rows:
        w[r] = big
    n_valid = N - 1000
    w[N - 5] = (x[0].float() / x[0].float().norm() * 3.0).half()  # masked, larger
    wq, sc = qp.pack_weight(w, bits, group, 32, cpi)
    logits = torch.zeros(M, N, dtype=torch.float32, device=dev)
    be.lm_head(x, wq, sc, logits, None, None, 30.0, N, n_valid, 0, bits, group, cpi)
    torch.cuda.synchronize()
    ref = oracle(qp, x, w, bits, group)
    ref = torch.tanh(ref / 30.0) * 30.0
    ref[:, n_valid:] = -math.inf
    fin = torch.isfinite(ref)
    o = logits.double()
    if not bool((torch.isinf(o) == torch.isinf(ref)).all()):
        fail(f"{tag} lm_head: -inf mask mismatch")
    err = float((o[fin] - ref[fin]).abs().max())
    if not err < 1e-4:
        fail(f"{tag} lm_head: logit err {err:.3e}")
    for with_logits in (False, True):
        pv = torch.zeros(M, nparts, dtype=torch.float32, device=dev)
        pi = torch.zeros(M, nparts, dtype=torch.int32, device=dev)
        lg = torch.zeros(M, N, dtype=torch.float32, device=dev) if with_logits else None
        be.lm_head(x, wq, sc, lg, pv, pi, 30.0, N, n_valid, 1000, bits, group, cpi)
        torch.cuda.synchronize()
        for m in range(M):
            best = float(pv[m].max())
            cand = int(pi[m][pv[m] == best].min()) - 1000
            mx = logits[m].max()
            want = int((logits[m] == mx).nonzero().min())
            if cand != want or best != float(mx):
                fail(
                    f"{tag} lm_head argmax row {m} logits={with_logits}: got {cand} want {want}"
                )
        if with_logits and not torch.equal(lg, logits):
            fail(f"{tag} lm_head: logits differ when the partials are on")
        # the serving path reduces the partials with inkling2_linear_argmax_reduce
        from vllm import _custom_ops as ops

        am = torch.empty(M, dtype=torch.int64, device=dev)
        ops.inkling2_linear_argmax_reduce(pv, pi, am)
        torch.cuda.synchronize()
        for m in range(M):
            want = int((logits[m] == logits[m].max()).nonzero().min()) + 1000
            if int(am[m]) != want:
                fail(
                    f"{tag} lm_head argmax_reduce row {m} logits={with_logits}: got {int(am[m])} want {want}"
                )
    pv = torch.zeros(M, nparts, dtype=torch.float32, device=dev)
    pi = torch.zeros(M, nparts, dtype=torch.int32, device=dev)
    be.lm_head(x, wq, sc, None, pv, pi, 30.0, N, n_valid, 0, bits, group, cpi)
    torch.cuda.synchronize()
    if int(pi[0, 0]) != 5:  # block 0 holds rows 5 / 9 / 37 / 16389: row 5 wins the tie
        fail(f"{tag} lm_head: block 0 best {int(pi[0, 0])}, want 5 (tie)")


def check_production_packs(qp, be, M_list=(1, 2, 3, 4, 5, 8)):
    """The packs of the serving path against the fp64 oracle: the quantized kernels up to
    4 rows (dequantized weights) and the fp16 pack above (fp16 weights)."""
    from vllm.models.inkling2_loop.amd import linear_pack as lp

    dev = "cuda"
    H, nh, nkv, D, d_rel, ksz = 2304, 24, 8, 128, 16, 4
    for fmt, full_layer in itertools.product(("w8", "w4", "w4g64"), (False, True)):
        w = {
            "q": rand_w(nh * D, H, dev, 1),
            "gate": rand_w(nh * D, H, dev, 2),
            "rel": rand_w(nh * d_rel, H, dev, 3),
            "sink": rand_w(nh, H, dev, 4),
            "k": rand_w(nkv * D, H, dev, 5),
            "v": rand_w(nkv * D, H, dev, 6),
        }
        dyn = (
            (rand_w(nkv * ksz, H, dev, 7), rand_w(nkv * ksz, H, dev, 8))
            if full_layer
            else ()
        )
        pack = lp.inkling2_linear_pack_attn_in(
            w["q"], w["gate"], w["rel"], w["sink"], w["k"], w["v"], *dyn
        )
        o_w = rand_w(H, nh * D, dev, 9)
        opack = lp.inkling2_linear_pack_o_proj(o_w)
        bits, group, clip = qp.parse_format(fmt)
        hyb = qp.HybridLinearPack(
            pack, qp.quantize_linear_pack(pack, fmt, qp.ATTN_IN_CONFIG[bits])
        )
        ohyb = qp.HybridLinearPack(
            opack, qp.quantize_linear_pack(opack, fmt, qp.O_PROJ_CONFIG[bits])
        )
        for name, hp, weight_rows in (("attn_in", hyb, pack), ("o_proj", ohyb, opack)):
            K = weight_rows.weight.shape[1]
            for M in M_list:
                g = torch.Generator(device=dev).manual_seed(20 + M)
                xs = [
                    torch.randn(M, K, device=dev, generator=g).half()
                    for _ in range(weight_rows.num_inputs)
                ]
                outs = weight_rows.alloc_outputs(M, dev)
                hp(xs, outs)
                torch.cuda.synchronize()
                # the hybrid pack must run the quantized kernel up to MAX_ROWS rows and
                # the fp16 kernel above, bit for bit (the kernels are deterministic)
                want = weight_rows.alloc_outputs(M, dev)
                (hp.quantized if M <= qp.MAX_ROWS else hp.fp16)(xs, want)
                torch.cuda.synchronize()
                for i, (o, w_) in enumerate(zip(outs, want)):
                    if not torch.equal(o, w_):
                        fail(
                            f"pack {name} {fmt} M={M} seg {i}: wrong kernel for the row count"
                        )
                lo = 0
                worst = 0.0
                for i, n in enumerate(weight_rows.seg_n):
                    wi = weight_rows.weight[lo : lo + n]
                    lo += n
                    if M <= qp.MAX_ROWS:
                        ref = oracle(
                            qp, xs[weight_rows.seg_x[i]], wi, bits, group, clip
                        )
                    else:
                        ref = xs[weight_rows.seg_x[i]].double() @ wi.double().t()
                    # fp16 outputs: half an ulp is 4.9e-4 of a value, and a value is up
                    # to ~4.5 rms
                    tol = 6e-3
                    err = float((outs[i].double() - ref).abs().max())
                    rms = math.sqrt(float((ref**2).mean()))
                    worst = max(worst, err / rms)
                    if not err / rms < tol:
                        fail(
                            f"pack {name} {fmt} full={full_layer} M={M} seg {weight_rows.names[i]}: "
                            f"err {err:.3e} rms {rms:.3e}"
                        )
                print(
                    f"ok pack {name} {fmt} full_layer={full_layer} M={M}  worst rel err {worst:.2e}",
                    flush=True,
                )
    # lm_head: the hybrid head (quantized up to 4 rows) with the argmax partials
    N, K = 200058, 2304
    wh = rand_w(N, K, dev, 99)
    head = lp.LMHeadPack(wh, N, 0)
    for fmt in ("w8", "w4"):
        bits, group, clip = qp.parse_format(fmt)
        hh = qp.HybridLMHead(head, qp.quantize_lm_head(head, fmt))
        for M in (1, 3, 4, 5):
            g = torch.Generator(device=dev).manual_seed(40 + M)
            x = torch.randn(M, K, device=dev, generator=g).half()
            logits = torch.empty(M, N, dtype=torch.float32, device=dev)
            hh(x, logits, softcap=30.0)
            torch.cuda.synchronize()
            ref = torch.empty(M, N, dtype=torch.float64, device=dev)
            for lo in range(0, N, 16384):
                blk = wh[lo : lo + 16384]
                if M <= qp.MAX_ROWS:
                    q, s = qp.quantize_rows(blk, bits, group, clip)
                    blk = qp.dequantize_rows(q, s, group)
                ref[:, lo : lo + 16384] = x.double() @ blk.double().t()
            ref = torch.tanh(ref / 30.0) * 30.0
            err = float((logits.double() - ref).abs().max())
            if not err < 2e-4:
                fail(f"pack lm_head {fmt} M={M}: err {err:.3e}")
            print(f"ok pack lm_head {fmt} M={M}  max err {err:.2e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--jit", action="store_true", help="JIT build instead of the installed ops"
    )
    ap.add_argument("--source", help="kernel source for --jit (default: the repo's)")
    ap.add_argument("--name", default="i2q_check")
    ap.add_argument(
        "--filter", default="1", help="C++ filter of the instances (qlinear_jit.py)"
    )
    ap.add_argument("--formats", default="8:64,4:32,4:64")
    ap.add_argument("--rs", default="4,8,16,32")
    ap.add_argument("--cpis", default="4,8")
    ap.add_argument("--wvs", default="8")
    ap.add_argument("--ms", default="1,2,3,4")
    ap.add_argument("--quick", action="store_true", help="fewer shapes")
    ap.add_argument("--no-packs", action="store_true")
    args = ap.parse_args()
    torch.cuda.init()
    from vllm.models.inkling2_loop.amd import qlinear_pack as qp

    fmts = [tuple(int(v) for v in f.split(":")) for f in args.formats.split(",")]
    if args.jit:
        import qlinear_jit

        t0 = time.time()
        be = JitBackend(
            qlinear_jit.build(args.name, args.source, fmts, extra_filter=args.filter)
        )
        print(f"JIT build {time.time() - t0:.0f}s", flush=True)
    else:
        be = OpBackend()
    n_ok = 0
    rs = [int(v) for v in args.rs.split(",")]
    cpis = [int(v) for v in args.cpis.split(",")]
    wvs = [int(v) for v in args.wvs.split(",")]
    ms = [int(v) for v in args.ms.split(",")]

    def configs(bits):
        """(R, cpi, waves): the sweep of --jit, else the instances of the serving build."""
        if args.jit:
            return list(itertools.product(rs, cpis, wvs))
        return sorted({qp.ATTN_IN_CONFIG[bits], qp.O_PROJ_CONFIG[bits]})

    cases = [
        ((bits, group), R, cpi, wv, M)
        for bits, group in fmts
        for R, cpi, wv in configs(bits)
        for M in ms
    ]
    for (bits, group), R, cpi, wv, M in cases:
        ck = qp.CK[bits]
        gc = group // ck
        for K, rows, sx in (
            (2304, [300, 77, 24, 130], [0, 0, 1, 2]),
            (3072, [100], [0]),
        ):
            if K % (ck * (32 // R)) or (K // (ck * (32 // R))) % gc or cpi % gc:
                continue
            tag = f"w{bits}g{group} R={R} cpi={cpi} wv={wv} M={M} K={K}"
            try:
                w = check_grouped(qp, be, bits, group, R, cpi, wv, M, K, rows, sx, tag)
                check_onehot(
                    qp,
                    be,
                    bits,
                    group,
                    R,
                    cpi,
                    wv,
                    M,
                    K,
                    37,
                    tag,
                    17 if args.quick else 3,
                )
                if K == 2304:
                    check_epilogue(qp, be, bits, group, R, cpi, wv, M, K, tag)
            except Exception as e:  # noqa: BLE001
                fail(f"{tag}: exception {str(e)[:200]}")
                continue
            n_ok += 1
            print(f"ok {tag}  worst rel err {w:.2e}", flush=True)
    lm_cpis = cpis if args.jit else [qp.LM_HEAD_CPI]
    for (bits, group), cpi, M in itertools.product(fmts, lm_cpis, ms):
        ck = qp.CK[bits]
        gc = group // ck
        if 2304 % ck or (2304 // ck) % gc or cpi % gc:
            continue
        tag = f"lm_head w{bits}g{group} cpi={cpi} M={M}"
        try:
            check_lm_head(
                qp,
                be,
                bits,
                group,
                cpi,
                M,
                2304,
                40000 if args.quick else 200058,
                64,
                tag,
            )
        except Exception as e:  # noqa: BLE001
            fail(f"{tag}: exception {str(e)[:200]}")
            continue
        print(f"ok {tag}", flush=True)
    if not args.jit and not args.no_packs:
        try:
            check_production_packs(qp, be)
        except Exception as e:  # noqa: BLE001
            fail(f"production packs: exception {str(e)[:300]}")
    print(
        f"CHECK {'FAIL' if FAILS else 'PASS'}: {len(FAILS)} failures, {n_ok} grouped configs",
        flush=True,
    )
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
