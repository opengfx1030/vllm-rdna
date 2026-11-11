#!/usr/bin/env python3
"""MonoKernel 2a, milestone M1: the persistent FFN kernel against today's
Flash-Next decode FFN block, standalone on one V620.

Today's block (the reference) is the exact op sequence of a TP4/EP4 rank's
decode step between the attention all-reduce and the FFN all-reduce:
hc_combine_norm -> rdna_hc_mix -> rdna_shared_expert -> router GEMM ->
fused_topk -> moe_align_block_size + moe_gptq_gemm_rdna2 x2 (+ SwiGLU) -> add.
Weights are random in the checkpoint-derived layouts the kernels read
(symmetric int4 g128, exllama-shuffled, synthesized zeros); an fp32 torch
"golden" is computed from the unshuffled weights.

  python3 m1_bench.py [--widths 3,6,12,24] [--layers 8] [--wg 72] [--iters 20]

Correctness: each path vs the fp32 golden evaluated with that path's own
routing, plus the share of tokens whose top-10 set differs from the golden.
Timing: both paths captured in one hipGraph over --layers distinct layers
(per-rank weights ~334 MB per layer, so the 128 MB Infinity Cache cannot
hold them), replayed --iters times; reported per layer.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

H, HC, R, ND, E, EL, TOPK, I, ISE, GROUP = 2560, 4, 320, 336, 512, 128, 10, 640, 160, 128
EPS = 1e-6


def build_ext():
    from torch.utils.cpp_extension import load

    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")
    SRC = os.environ.get("MONO_SRC_DIR", os.path.join(TREE, "csrc/rocm"))
    return load(
        name="mono_ffn_ext",
        sources=[os.path.join(SRC, "mono_k2_rdna2.cu"), os.path.join(SRC, "rdna_allreduce.cu"),
                 os.path.join(HERE, "ext_binding.cpp")],
        extra_include_paths=[SRC],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx1030", "-ffp-contract=fast",
                           f"-DMONO_FFN_MIN_BLOCKS={os.environ.get('MONO_MIN_BLOCKS', '1')}"]
        + [f"-D{k}={v}" for k, v in os.environ.items() if k.startswith("MONO_FFN_DEF_")
           for k in [k.replace("MONO_FFN_DEF_", "MONO_FFN_")]],
        build_directory=os.environ.get("MONO_BUILD_DIR"),
        verbose=False,
    )


def make_layer(gen, dev, keep_golden=False):
    """Random weights; returns (kernel-layout dict, golden fp32 dict)."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        _synthesize_rdna3_qzeros,
    )

    def rn(*s, std):
        return (torch.randn(*s, generator=gen, device=dev) * std)

    L = {}
    L["norm_w"] = rn(HC * H, std=0.1).half()
    L["wd"] = rn(ND, HC * H, std=(HC * H) ** -0.5).half()
    L["wu"] = rn(HC * H, R, std=R ** -0.5 * 2).half()
    L["wr"] = rn(E, H, std=H ** -0.5 * 2).half()
    L["se_w1"] = rn(2 * ISE, H, std=H ** -0.5).half()
    L["se_w2"] = rn(H, ISE, std=ISE ** -0.5).half()
    L["se_wg"] = rn(H, std=H ** -0.5).half()

    def q4(K, N):
        q = torch.randint(0, 16, (EL, K, N), generator=gen, device=dev, dtype=torch.int32)
        s = (torch.rand(EL, K // GROUP, N, generator=gen, device=dev) * 0.01 + 0.005)
        s = s.half()
        packed = torch.zeros(EL, K // 8, N, dtype=torch.int32, device=dev)
        for j in range(8):
            packed |= (q[:, j::8, :] & 0xF) << (4 * j)
        for e in range(EL):
            pe = packed[e].contiguous()
            ops.gptq_shuffle(pe, 4)
            packed[e] = pe
        z = _synthesize_rdna3_qzeros(K // GROUP, N, dev)
        z = z.unsqueeze(0).expand(EL, -1, -1).contiguous()
        dense = None
        if keep_golden:
            dense = (q.float() - 8.0) * s.float().repeat_interleave(GROUP, dim=1)
        del q
        return packed, s.contiguous(), z, dense  # dense [EL, K, N]

    L["w13"], L["s13"], L["z13"], g13 = q4(H, 2 * I)
    L["w2"], L["s2"], L["z2"], g2 = q4(I, H)
    if not keep_golden:
        return L, None
    G = {k: v.float() for k, v in L.items() if v.dtype == torch.half and k not in ("s13", "s2")}
    G["w13"] = g13
    G["w2"] = g2
    return L, G


def golden(G, res, blk, inj, base, ids=None):
    res, blk, inj = res.float(), blk.float(), inj.float()
    M = res.shape[0]
    injs = 2 * torch.sigmoid(inj / HC)
    ro = res.view(M, HC, H) + blk[:, None, :] * injs[:, :, None]
    rr = torch.rsqrt(ro.pow(2).mean(-1, keepdim=True) + EPS)
    xn = (ro * rr) * (1 + G["norm_w"].view(HC, H))
    xn = xn.reshape(M, HC * H)
    dai = xn @ G["wd"].t()
    lora = F.silu(dai[:, :R] / HC)
    inj_out = dai[:, R:R + HC]
    gate = lora @ G["wu"].t()
    b = (torch.sigmoid(gate).view(M, HC, H) * xn.view(M, HC, H)).mean(1)
    logits = b @ G["wr"].t()
    p = torch.softmax(logits, -1)
    srt = logits.sort(-1, descending=True).values
    golden.last_gap = (srt[:, TOPK - 1] - srt[:, TOPK]).tolist()  # per token
    if ids is None:
        ids = p.topk(TOPK, -1).indices
    w = p.gather(1, ids.long())
    w = w / w.sum(-1, keepdim=True)
    se = torch.sigmoid(b @ G["se_wg"])[:, None] * (
        (F.silu(b @ G["se_w1"][:ISE].t()) * (b @ G["se_w1"][ISE:].t())) @ G["se_w2"].t())
    out = se.clone()
    for m in range(M):
        for k in range(TOPK):
            le = int(ids[m, k]) - base
            if 0 <= le < EL:
                gu = b[m] @ G["w13"][le]
                a = F.silu(gu[:I]) * gu[I:]
                out[m] += w[m, k] * (a @ G["w2"][le])
    return out, ro.reshape(M, HC * H), inj_out, ids


class Reference:
    """Today's decode FFN block on one EP rank (the op sequence production runs)."""

    def __init__(self, base, dev):
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
        self.fused_topk = fused_topk
        emap = torch.full((E,), -1, dtype=torch.int32, device=dev)
        emap[base:base + EL] = torch.arange(EL, dtype=torch.int32, device=dev)
        self.emap = emap

    def __call__(self, L, res, blk, inj):
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers import rdna_ops  # noqa: F401
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        from vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe import _resolve_block_size_m
        from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
        from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_norm

        M = res.shape[0]
        res_out, xn = hc_combine_norm(res, blk, inj, L["norm_w"], EPS, HC)
        b, dai = torch.ops.vllm.rdna_hc_mix(xn, L["wd"], None, None, L["wu"], None, None, R, HC)
        inj_out = dai[:, R:R + HC].contiguous()
        se = torch.ops.vllm.rdna_shared_expert(b, L["se_w1"], None, None, L["se_w2"], None, None, L["se_wg"])
        logits = rocm_unquantized_gemm_impl(b, L["wr"], None)
        tw, tids, _ = self.fused_topk(b, logits, TOPK, True)
        bsm = _resolve_block_size_m(M)
        sorted_ids, expert_ids, ntp = moe_align_block_size(
            tids, bsm, EL, self.emap, ignore_invalid_experts=True)
        w1_out = torch.zeros(M * TOPK, 2 * I, dtype=torch.half, device=res.device)
        twf = tw.reshape(-1).float()
        ops.moe_gptq_gemm_rdna2(b, w1_out, L["w13"], L["s13"], L["z13"], twf.new_empty(0),
                                sorted_ids, expert_ids, ntp, TOPK, bsm, False, 0)
        g, u = w1_out.chunk(2, dim=-1)
        act = F.silu(g) * u
        routed = torch.zeros(M, H, dtype=torch.half, device=res.device)
        ops.moe_gptq_gemm_rdna2(act, routed, L["w2"], L["s2"], L["z2"], twf,
                                sorted_ids, expert_ids, ntp, 1, bsm, True, TOPK)
        return routed + se, res_out, inj_out, tids


def stats(x, gold):
    d = (x.float() - gold).abs()
    ref = gold.abs().max().item()
    return {"max_abs": d.max().item(), "max_rel": d.max().item() / max(ref, 1e-9),
            "mean_abs": d.mean().item(), "gold_absmax": ref}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="3,6,12,24")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--wg", type=int, default=0, help="0: the residency cap")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--base", type=int, default=0, help="first local expert (EP rank * 128)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ext = build_ext()
    gen = torch.Generator(device=dev).manual_seed(1234)
    print(f"building {a.layers} layers of random weights ...", flush=True)
    layers = [make_layer(gen, dev, keep_golden=(n == 0)) for n in range(a.layers)]
    wsb = ext.workspace_bytes(H, R, ND, E, ISE, I)
    ws = torch.zeros(wsb, dtype=torch.uint8, device=dev)
    ref = Reference(a.base, dev)
    results = {"wg": a.wg, "layers": a.layers, "base": a.base, "widths": {}}

    def mono(L, res, blk, inj):
        M = res.shape[0]
        out = torch.empty(M, H, dtype=torch.half, device=dev)
        ro = torch.empty_like(res)
        io = torch.empty(M, HC, dtype=torch.half, device=dev)
        ext.mono_ffn(res, blk, inj, L["norm_w"], L["wd"], L["wu"], L["wr"], L["se_w1"],
                     L["se_w2"], L["se_wg"], L["w13"], L["s13"], L["z13"], L["w2"], L["s2"],
                     L["z2"], ro, io, out, ws, a.base, 1, a.wg, EPS, 2000.0)
        return out, ro, io

    for M in [int(w) for w in a.widths.split(",")]:
        row = {"correctness": {}, "timing": {}}
        for dist, blk_std in (("gdn_like", 0.5), ("qsa_like", 2.0)):
            L, G = layers[0]
            res = (torch.randn(M, HC * H, generator=gen, device=dev) * 1.0).half()
            blk = (torch.randn(M, H, generator=gen, device=dev) * blk_std).half()
            inj = torch.randn(M, HC, generator=gen, device=dev).half()
            o_ref, ro_ref, io_ref, ids_ref = ref(L, res, blk, inj)
            o_m, ro_m, io_m = mono(L, res, blk, inj)
            torch.cuda.synchronize()
            ids_m = ext.topk(ws)[:M * TOPK].view(M, TOPK).to(dev)
            err = ext.error(ws).tolist()
            gold, gro, gio, gids = golden(G, res, blk, inj, a.base)
            gold_r, *_ = golden(G, res, blk, inj, a.base, ids=ids_ref)
            gold_m, *_ = golden(G, res, blk, inj, a.base, ids=ids_m)
            diff_m = sum(set(ids_m[m].tolist()) != set(gids[m].tolist()) for m in range(M))
            # the kernel's routing is internal: score it against the golden routing
            diff_ref = sum(set(ids_ref[m].tolist()) != set(gids[m].tolist()) for m in range(M))
            gaps = golden.last_gap
            rowerr = (o_m.float() - gold).abs().max(1).values.tolist()
            bad = [(m, round(rowerr[m], 4), round(gaps[m], 6)) for m in range(M) if rowerr[m] > 0.01]
            row["correctness"][dist] = {
                "mono_bad_tokens(row, max_abs, gold top10-11 logit gap)": bad,
                "min_top10_gap": round(min(gaps), 6),
                "sticky_err": err,
                "ref_vs_gold(own routing)": stats(o_ref, gold_r),
                "mono_vs_gold(own routing)": stats(o_m, gold_m),
                "mono_vs_gold(gold routing)": stats(o_m, gold),
                "mono_routing_differs_from_gold": f"{diff_m}/{M}",
                "mono_vs_ref": stats(o_m, o_ref.float()),
                "res_out mono_vs_ref": stats(ro_m, ro_ref.float()),
                "inj_out mono_vs_gold": stats(io_m, gio),
                "inj_out ref_vs_gold": stats(io_ref, gio),
                "ref_routing_differs_from_gold": f"{diff_ref}/{M}",
            }
        # timing: one graph over all layers
        gt = torch.Generator(device=dev).manual_seed(4321 + M)  # same routing per variant
        ins = [((torch.randn(M, HC * H, generator=gt, device=dev)).half(),
                (torch.randn(M, H, generator=gt, device=dev)).half(),
                torch.randn(M, HC, generator=gt, device=dev).half()) for _ in layers]
        for name, fn in (("today_block", lambda L, x: ref(L, *x)),
                         ("mono_kernel", lambda L, x: mono(L, *x))):
            for (L, _), x in zip(layers, ins):
                fn(L, x)  # warm-up / JIT
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s):
                with torch.cuda.graph(g, stream=s):
                    for (L, _), x in zip(layers, ins):
                        fn(L, x)
            torch.cuda.synchronize()
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            ts = []
            for _ in range(a.iters):
                t0.record()
                g.replay()
                t1.record()
                t1.synchronize()
                ts.append(t0.elapsed_time(t1) * 1000 / len(layers))
            ts.sort()
            row["timing"][name] = {"median_us_per_layer": ts[len(ts) // 2],
                                   "min_us_per_layer": ts[0]}
            if name == "mono_kernel":
                t = ext.phase_ts(ws).tolist()  # last launch of the last replay
                ts, done = t[:8], t[8:]
                G = int(ext.error(ws).tolist()[3])
                row["timing"]["mono_phase_us"] = {
                    f"P{i + 1}": round((ts[i + 1] - ts[i]) / 100.0, 1) for i in range(7)}
                nused = int(ext.error(ws).tolist()[5])
                ph = row["timing"]["mono_phase_us"]
                wbytes = {  # weight bytes streamed per phase (per rank)
                    "P2": ND * HC * H * 2, "P3": HC * H * R * 2,
                    "P4": (E + 2 * ISE + 1) * H * 2,
                    "P5": nused * (H // 8) * 2 * I * 4 + H * ISE * 2,
                    "P6": nused * (I // 8) * H * 4}
                row["timing"]["local_experts_used"] = nused
                row["timing"]["mono_phase_GBps"] = {
                    k: round(v / (ph[k] * 1e-6) / 1e9, 0) for k, v in wbytes.items() if ph[k] > 0}
                # per phase: earliest / latest WG to finish its work, from phase start
                row["timing"]["mono_phase_wg_finish_us"] = {
                    f"P{i + 1}": [round((min(done[i * 256:i * 256 + G]) - ts[i]) / 100.0, 1),
                                  round((max(done[i * 256:i * 256 + G]) - ts[i]) / 100.0, 1)]
                    for i in range(6)}
            del g
        row["timing"]["speedup"] = (row["timing"]["today_block"]["median_us_per_layer"]
                                    / row["timing"]["mono_kernel"]["median_us_per_layer"])
        row["timing"]["sticky_err_after"] = ext.error(ws).tolist()
        results["widths"][M] = row
        print(f"== M={M}", json.dumps(row, indent=1), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
