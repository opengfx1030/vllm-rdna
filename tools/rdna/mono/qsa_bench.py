#!/usr/bin/env python3
"""E3 standalone: QSA decode indexer (qsa_select_paged_tokens) and sparse
attention (qsa_sparse_paged_attention) for one Flash-Next TP rank, today vs the
--monokernel QSA path (live-context column bound + all-padding tile skip).
Checks the selected index sets are identical and the attention output matches
(to summation-order noise), then times
both in a graph at decode / verify widths and 1k / 16k contexts.

  PYTHONPATH=<tree> python3 qsa_bench.py [--widths 1,3,8,24] [--ctx 1024,16384]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import after_bench as A  # noqa: E402

MAXLEN = 262144
RATIO, TOPK = 4, 2048
IH, ID = 4, 128          # indexer heads / dim
QH, KVH, HD = 6, 1, 256  # attention heads per rank, kv heads, head dim
PAGE = 64


def build(gen, dev, n_req, ctx):
    cols_cap = MAXLEN // RATIO
    cpages_per_req = cols_cap // PAGE
    kpages_per_req = MAXLEN // PAGE
    live_c = (ctx // RATIO + PAGE - 1) // PAGE
    live_k = (ctx + PAGE - 1) // PAGE
    nc = n_req * live_c + 1
    nk = n_req * live_k + 1
    ccache = (torch.randn(nc, PAGE, 1, ID, generator=gen, device=dev) * 0.5).half()
    kv = (torch.randn(nk, PAGE, 2, KVH, HD, generator=gen, device=dev) * 0.5).half()
    ctab = torch.full((n_req, cpages_per_req), -1, dtype=torch.int32, device=dev)
    ktab = torch.full((n_req, kpages_per_req), -1, dtype=torch.int32, device=dev)
    for r in range(n_req):
        ctab[r, :live_c] = torch.arange(1 + r * live_c, 1 + (r + 1) * live_c, device=dev)
        ktab[r, :live_k] = torch.arange(1 + r * live_k, 1 + (r + 1) * live_k, device=dev)
    return ccache, kv, ctab, ktab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1,3,8,24")
    ap.add_argument("--ctx", default="1024,16384")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()
    from torch.utils.cpp_extension import load

    from vllm.models.qwen4_exp.amd.mono.rdna2 import qsa as MQ
    from vllm.models.qwen4_exp.amd.ops import qsa as Q

    here = os.path.dirname(os.path.abspath(__file__))
    ext = load(name="qsa_decode_ext",
               sources=[os.path.join(here, "../../../csrc/rocm/mono_qsa_decode_rdna2.cu"),
                        os.path.join(here, "qsa_decode_binding.cpp")],
               extra_cuda_cflags=["-O3", "--offload-arch=gfx1030", "-ffp-contract=fast"] + os.environ.get("QSA_CFLAGS", "").split(),
               build_directory=os.environ.get("MONO_BUILD_DIR"), verbose=False)
    counters = torch.zeros(4096, dtype=torch.int32, device="cuda")

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(5)
    ok = True
    for ctx in [int(x) for x in a.ctx.split(",")]:
        for M in [int(x) for x in a.widths.split(",")]:
            T = 3 if M % 3 == 0 and M > 1 else 1  # verify rows per sequence
            n_req = M // T
            ccache, kv, ctab, ktab = build(gen, dev, n_req, ctx)
            seq = torch.full((n_req,), ctx, dtype=torch.int32, device=dev)
            t2r = torch.arange(n_req, device=dev, dtype=torch.int32).repeat_interleave(T)
            pos = (ctx - T + torch.arange(M, device=dev) % T).long()
            qi = (torch.randn(M, IH, ID, generator=gen, device=dev) * 0.5).half()
            qa = (torch.randn(M, QH, HD, generator=gen, device=dev) * 0.5).half()
            kc, vc = kv[:, :, 0], kv[:, :, 1]
            out_w = TOPK + RATIO - 1

            def select(bound, host_bound=True):
                # host_bound False: E3 inside a CUDA graph (worst-case columns,
                # the scoring kernel's device-side early exit only)
                Q.MONO_QSA = MQ if bound else None
                return Q.qsa_select_paged_tokens(
                    qi, ccache, ctab, t2r, pos, seq, TOPK, RATIO,
                    torch.empty(M, out_w, dtype=torch.int32, device=dev),
                    max_seq_len=ctx if bound and host_bound else None)

            def attend(skip, idx, out):
                Q.MONO_QSA = MQ if skip else None
                return Q.qsa_sparse_paged_attention(qa, kc, vc, idx, ktab, t2r, out)

            S = int(os.environ.get("QSA_S", 0)) or int(ext.splits(M, KVH, out_w))
            scratch = torch.empty(S * M * QH * (HD + 1), dtype=torch.float32, device=dev)

            def hip(idx, out):
                ext.decode(qa, kc, vc, idx, ktab, t2r, out, scratch, counters, S)
                return out

            i0, i1 = select(False), select(True)
            o0 = attend(False, i0, torch.empty_like(qa))
            o1 = attend(True, i1, torch.empty_like(qa))
            torch.cuda.synchronize()
            i2 = select(True, host_bound=False)
            same_i = bool(torch.equal(i0.sort(1).values, i1.sort(1).values)) and bool(
                torch.equal(i0.sort(1).values, i2.sort(1).values))
            d_o = (o0.float() - o1.float()).abs().max().item()
            oh = hip(i0, torch.empty_like(qa))
            torch.cuda.synchronize()
            d_h = (o0.float() - oh.float()).abs().max().item()
            rel_h = d_h / max(o0.float().abs().max().item(), 1e-6)
            clean = bool((counters == 0).all().item())
            # top-k may order the same set differently once columns differ:
            # the attention then sums in another order (fp16-rounding noise)
            good = same_i and d_o < 1e-3 and rel_h < 5e-3 and clean
            ok &= good
            nvalid = int((i0 >= 0).sum(1).float().mean().item())
            # R calls per graph: one graph replay alone costs ~12 us on V620
            R = a.reps

            def gt(f):
                return A.graph_time(lambda: [f() for _ in range(R)], a.iters) / R

            t_s0 = gt(lambda: select(False))
            t_s1 = gt(lambda: select(True))
            t_s2 = gt(lambda: select(True, host_bound=False))
            ob = torch.empty_like(qa)
            t_a0 = gt(lambda: attend(False, i0, ob))
            t_a1 = gt(lambda: attend(True, i1, ob))
            t_h = gt(lambda: hip(i1, ob))
            Q.MONO_QSA = None
            print(f"ctx {ctx:5d} M={M:2d} (T={T}) valid idx {nvalid:4d} | indices "
                  f"{'same' if same_i else 'DIFF'} out diff {d_o:.1e} | select today "
                  f"{t_s0:6.1f} us e3 {t_s1:6.1f} us x{t_s0 / t_s1:.2f} in-graph {t_s2:6.1f} us | attn today "
                  f"{t_a0:6.1f} us skip {t_a1:6.1f} us hip(S={S}) {t_h:6.1f} us x{t_a0 / t_h:.2f}"
                  f" (hip rel {rel_h:.1e}) {'ok' if good else 'BAD'}")
    print("OK" if ok else "BAD")


if __name__ == "__main__":
    main()
