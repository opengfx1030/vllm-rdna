#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check the FA-RDNA2 head_dim 256 decode path against an fp32 reference.

Drives ``RdnaAttentionImpl.forward`` (the production dispatch) on synthetic
paged KV in the production layout: pure decode, split decode with
``cu_query_lens`` (MTP verify rows, short extends) and mixed batches (decode
rows plus a prefill chunk), for the per-rank head counts of the TP1/TP2/TP4
27B recipes. ``VLLM_FA_RDNA2_GQA_DECODE`` is read once per process (a static in
``fa_rdna2.cu``), so run the script once per setting and diff the results:

    VLLM_FA_RDNA2_GQA_DECODE=0 python gqa_decode_check.py --tag off --out off.json
    VLLM_FA_RDNA2_GQA_DECODE=1 python gqa_decode_check.py --tag on --out on.json
    python gqa_decode_check.py --compare off.json on.json

Per case: max abs / rel error against the reference, NaN or Inf, rows beyond
the tolerance, whether ``--repeat`` repeats of one call are bitwise equal (a
race shows up as a run-to-run difference), and the sha1 of the output, which
``--compare`` uses to tell whether the two settings produce identical bits.
The ``gate`` column is ``rows x KV heads`` (the kernel's engagement gate is
``>= 4``). ``--bench`` replays one captured forward per case and prints
microseconds; ``--compare-bench off.json on.json`` prints the ratio.

Exit status is 1 when any case fails (NaN, error above ``--tol-abs``, repeats
that differ, or a non-zero padded row).
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

import torch

D = 256


def sha(t: torch.Tensor) -> str:
    return hashlib.sha1(t.contiguous().view(torch.uint8).cpu().numpy()).hexdigest()[:16]


def make_kv(num_blocks, h_kv, bs, pad, gen):
    x = h_kv * D * bs
    big = torch.empty(num_blocks, 2, x + pad, dtype=torch.float16, device="cuda")
    big.normal_(0.0, 1.0, generator=gen)
    return big[:, :, :x]


def split_kv(kv, h_kv):
    from vllm.v1.attention.backends.rdna_attn import _reinterpret_v_to_5d
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    key_cache, value_cache = PagedAttention.split_kv_cache(kv.transpose(0, 1), h_kv, D)
    return key_cache, value_cache, _reinterpret_v_to_5d(key_cache, value_cache, D)


def ref_rows(q, key_cache, value_cache4, bt_row, kvlens, h_q, h_kv, bs, scale):
    """Run fp32 attention for one sequence; query ``l`` sees ``kvlens[l]`` keys."""
    ql = q.shape[0]
    maxlen = int(max(kvlens))
    pos = torch.arange(maxlen, device=q.device)
    blk = bt_row[pos // bs].long()
    slot = pos % bs
    k = key_cache[blk, :, :, slot, :].reshape(maxlen, h_kv, D)
    v = value_cache4[blk, :, :, slot].reshape(maxlen, h_kv, D)
    g = h_q // h_kv
    kvl = torch.tensor(kvlens, device=q.device)
    masked = pos[None, None, :] >= kvl[:, None, None]
    out = torch.zeros(ql, h_q, D, dtype=torch.float32, device=q.device)
    for kvh in range(h_kv):
        qg = q[:, kvh * g : (kvh + 1) * g, :].float()
        s = torch.einsum("lgd,kd->lgk", qg, k[:, kvh, :].float()) * scale
        p = torch.softmax(s.masked_fill(masked, float("-inf")), dim=-1)
        out[:, kvh * g : (kvh + 1) * g, :] = torch.einsum(
            "lgk,kd->lgd", p, v[:, kvh, :].float()
        )
    return out


def build_case(c, seed, with_ref=True):
    """Build the impl, metadata, query, KV cache and fp32 reference of one case."""
    from vllm.v1.attention.backends.rdna_attn import (
        RdnaAttentionImpl,
        RdnaAttentionMetadata,
    )

    h_q, h_kv, bs = c["H_q"], c["H_kv"], c["bs"]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    reqs = list(c["dec"]) + list(c.get("pre", []))  # (q_len, kv_len), decode first
    n_pad = c.get("pad_rows", 0)
    n_dec = len(c["dec"])
    blocks_per = [(sl + bs - 1) // bs for _, sl in reqs]
    total_blocks = sum(blocks_per) + 3
    kv = make_kv(total_blocks, h_kv, bs, c.get("pad", 0), gen)
    key_cache, value_cache4, value_cache5 = split_kv(kv, h_kv)
    perm = torch.randperm(total_blocks, generator=torch.Generator().manual_seed(seed))
    bt = torch.zeros(len(reqs) + n_pad, max(blocks_per), dtype=torch.int32)
    k = 0
    for i, nb in enumerate(blocks_per):
        bt[i, :nb] = perm[k : k + nb].to(torch.int32)
        k += nb
    bt = bt.cuda()
    seq_lens = torch.tensor([sl for _, sl in reqs] + [0] * n_pad, dtype=torch.int32)
    seq_lens = seq_lens.cuda()
    qlens = [ql for ql, _ in reqs] + [1] * n_pad
    t_real = sum(ql for ql, _ in reqs)
    n_tok = t_real + n_pad
    qstart = torch.zeros(len(qlens) + 1, dtype=torch.int32)
    qstart[1:] = torch.cumsum(torch.tensor(qlens, dtype=torch.int32), 0)
    qstart = qstart.cuda()
    q = torch.randn(n_tok, h_q, D, device="cuda", dtype=torch.float32, generator=gen)
    q = (q * c.get("qscale", 1.0)).half()
    scale = D**-0.5
    nd_tok = sum(ql for ql, _ in c["dec"])
    common = dict(
        num_actual_tokens=n_tok,
        query_start_loc=qstart,
        max_seq_len=int(seq_lens.max()),
        seq_lens=seq_lens,
        block_table=bt,
        slot_mapping=torch.zeros(n_tok, dtype=torch.int64, device="cuda"),
        causal=True,
    )
    if c["kind"] == "decode":
        # The builder marks max_query_len <= 1 batches as all decode rows.
        meta = RdnaAttentionMetadata(
            max_query_len=1,
            prefill_query_start_loc=qstart,
            max_prefill_seq_len=int(seq_lens.max()),
            num_decodes=len(qlens),
            num_decode_tokens=n_tok,
            **common,
        )
    else:
        has_pre = n_dec < len(reqs)
        meta = RdnaAttentionMetadata(
            max_query_len=max(qlens),
            prefill_query_start_loc=qstart[n_dec:] - nd_tok if has_pre else qstart,
            max_prefill_seq_len=int(seq_lens[n_dec : len(reqs)].max())
            if has_pre
            else int(seq_lens.max()),
            num_decodes=n_dec,
            num_decode_tokens=nd_tok,
            decode_query_start_loc=qstart[: n_dec + 1],
            **common,
        )
    impl = RdnaAttentionImpl(h_q, D, scale, h_kv, None, None, "auto")
    refs = torch.zeros(n_tok, h_q, D, dtype=torch.float32, device="cuda")
    off = 0
    for i, (ql, sl) in enumerate(reqs if with_ref else []):
        kvlens = [sl - (ql - 1 - j) for j in range(ql)]
        refs[off : off + ql] = ref_rows(
            q[off : off + ql],
            key_cache,
            value_cache4,
            bt[i],
            kvlens,
            h_q,
            h_kv,
            bs,
            scale,
        )
        off += ql
    info = dict(
        n_dec=n_dec,
        nd_tok=nd_tok,
        gate=(n_tok if c["kind"] == "decode" else nd_tok) * h_kv,
        strides=dict(k4=key_cache.stride(4), v3=value_cache5.stride(3)),
    )
    return impl, meta, q, kv, refs, info, t_real


def run_case(c, seed, repeat):
    impl, meta, q, kv, refs, info, t_real = build_case(c, seed)
    n_tok = q.shape[0]
    shas = []
    out = None
    for _ in range(repeat):
        out = torch.zeros(n_tok, c["H_q"] * D, dtype=torch.float16, device="cuda")
        impl.forward(None, q, None, None, kv, meta, out)
        torch.cuda.current_stream().synchronize()
        shas.append(sha(out))
    o = out.view(n_tok, c["H_q"], D)
    diff = (o[:t_real].float() - refs[:t_real]).abs()
    return dict(
        name=c["name"],
        T=n_tok,
        gate=info["gate"],
        n_dec=info["n_dec"],
        strides=info["strides"],
        max_abs=float(diff.max()),
        max_rel=float((diff / (refs[:t_real].abs() + 1e-3)).max()),
        nan=bool(torch.isnan(o).any() or torch.isinf(o).any()),
        bad_rows=int((diff.amax(dim=(1, 2)) > 5e-3).sum()),
        pad_zero=bool(o[t_real:].abs().max() == 0.0) if n_tok > t_real else True,
        repeat_equal=len(set(shas)) == 1,
        sha=shas[0],
    )


def cases(quick):
    out = []
    rnd = random.Random(1234)

    def dec(name, h_q, h_kv, bs, lens, **kw):
        out.append(
            dict(
                name=name,
                kind="decode",
                H_q=h_q,
                H_kv=h_kv,
                bs=bs,
                dec=[(1, s) for s in lens],
                **kw,
            )
        )

    def split(name, h_q, h_kv, bs, qlen, lens, pre=None, **kw):
        out.append(
            dict(
                name=name,
                kind="split",
                H_q=h_q,
                H_kv=h_kv,
                bs=bs,
                dec=[(qlen, s) for s in lens],
                pre=pre or [],
                **kw,
            )
        )

    def lens(n, lo, hi):
        return [rnd.randint(lo, hi) for _ in range(n)]

    # 27B TP4 rank (H_q 6, H_kv 1, G 6): the gate needs >= 4 rows.
    for n in (1, 2, 3, 4, 5, 8, 12, 16):
        dec(f"27b_tp4_decode_n{n}_bs1024", 6, 1, 1024, lens(n, 33, 20000))
    for n in (4, 8):
        dec(f"27b_tp4_decode_n{n}_bs16", 6, 1, 16, lens(n, 33, 6000))
        dec(f"27b_tp4_decode_n{n}_bs784", 6, 1, 784, lens(n, 33, 12000))
    dec("27b_tp4_decode_n8_16k_equal", 6, 1, 1024, [16384] * 8)
    dec("27b_tp4_decode_n8_short", 6, 1, 1024, [1, 2, 7, 31, 32, 33, 63, 64])
    dec(
        "27b_tp4_decode_n8_pad_unaligned",
        6,
        1,
        1024,
        [1000, 1023, 1024, 1025, 2047, 2048, 2049, 4097],
    )
    dec(
        "27b_tp4_decode_n8_pad2rows",
        6,
        1,
        1024,
        [3000, 9000, 15000, 400, 12345, 777],
        pad_rows=2,
    )
    dec("27b_tp4_decode_n8_padstride", 6, 1, 1024, lens(8, 33, 9000), pad=64)
    dec("27b_tp4_decode_n8_peaky", 6, 1, 1024, lens(8, 33, 9000), qscale=4.0)
    # 27B TP2 rank (H_q 12, H_kv 2, G 6): the gate needs >= 2 rows.
    for n in (1, 2, 3, 4, 8):
        dec(f"27b_tp2_decode_n{n}_bs1024", 12, 2, 1024, lens(n, 33, 18000))
    # 27B TP1 (24/4) and a 16/4 model; other group sizes; G 12 is never grouped.
    for n in (1, 2, 8):
        dec(f"27b_tp1_decode_n{n}_bs1024", 24, 4, 1024, lens(n, 33, 12000))
        dec(f"h16_kv4_decode_n{n}_bs16", 16, 4, 16, lens(n, 33, 9000))
    for g in (2, 3, 4, 5, 7, 8, 12):
        dec(f"G{g}_hkv1_decode_n8_bs1024", g, 1, 1024, lens(8, 33, 8000))
    dec("flashnext_tp1_G12_decode_n8_bs1024", 24, 2, 1024, lens(8, 33, 8000))
    # MTP verify rows: every request has q_len 2..4 (split decode, cu_query_lens).
    for ql in (2, 3, 4):
        for n in (2, 4, 6, 8):
            split(
                f"27b_tp4_verify_q{ql}_n{n}_bs1024", 6, 1, 1024, ql, lens(n, 40, 16000)
            )
    for n in (2, 4):
        split(f"27b_tp2_verify_q3_n{n}_bs1024", 12, 2, 1024, 3, lens(n, 40, 16000))
    # Mixed batches: decode rows first, then a prefill chunk of a longer prompt.
    for n in (1, 2, 3, 4, 5, 8):
        split(
            f"27b_tp4_mixed_dec{n}_pre2048_bs1024",
            6,
            1,
            1024,
            1,
            lens(n, 200, 14000),
            pre=[(2048, 16384)],
        )
    for n in (2, 4, 7):
        split(
            f"27b_tp4_mixed_dec{n}_pre512_bs1024",
            6,
            1,
            1024,
            1,
            lens(n, 200, 14000),
            pre=[(512, 5000)],
        )
    for n in (2, 4):
        split(
            f"27b_tp2_mixed_dec{n}_pre2048_bs1024",
            12,
            2,
            1024,
            1,
            lens(n, 200, 14000),
            pre=[(2048, 16384)],
        )
    split(
        "27b_tp4_mixed_verify3_n4_pre1024_bs1024",
        6,
        1,
        1024,
        3,
        lens(4, 200, 14000),
        pre=[(1024, 9000)],
    )
    split(
        "27b_tp4_mixed_dec6_pre2x_bs1024",
        6,
        1,
        1024,
        1,
        lens(6, 200, 14000),
        pre=[(1024, 3000), (1024, 7000)],
    )
    split(
        "27b_tp4_mixed_dec8_pre2048_bs16",
        6,
        1,
        16,
        1,
        lens(8, 200, 3000),
        pre=[(2048, 4096)],
    )
    split("27b_tp4_extend_q8_n4_bs1024", 6, 1, 1024, 8, lens(4, 100, 12000))
    if quick:
        keep = ("n8", "mixed_dec4", "verify_q3_n4")
        out = [c for c in out if any(k in c["name"] for k in keep)]
    return out


def bench_cases():
    out = []
    for ctx in (1024, 4096, 16384, 32768):
        for n in (1, 2, 3, 4, 6, 8, 12, 16):
            out.append(
                dict(
                    name=f"27b_tp4_decode_n{n}_ctx{ctx}",
                    kind="decode",
                    H_q=6,
                    H_kv=1,
                    bs=1024,
                    dec=[(1, ctx)] * n,
                )
            )
    for ctx in (4096, 16384):
        for n in (2, 4, 8):
            out.append(
                dict(
                    name=f"27b_tp2_decode_n{n}_ctx{ctx}",
                    kind="decode",
                    H_q=12,
                    H_kv=2,
                    bs=1024,
                    dec=[(1, ctx)] * n,
                )
            )
    for n in (2, 4, 8):
        out.append(
            dict(
                name=f"27b_tp4_verify3_n{n}_ctx16384",
                kind="split",
                H_q=6,
                H_kv=1,
                bs=1024,
                dec=[(3, 16384)] * n,
                pre=[],
            )
        )
    return out


def bench_case(c, seed, iters):
    impl, meta, q, kv, _, info, _ = build_case(c, seed, with_ref=False)
    out = torch.zeros(q.shape[0], c["H_q"] * D, dtype=torch.float16, device="cuda")
    for _ in range(5):
        impl.forward(None, q, None, None, kv, meta, out)
    torch.cuda.current_stream().synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        impl.forward(None, q, None, None, kv, meta, out)
    for _ in range(10):
        graph.replay()
    torch.cuda.current_stream().synchronize()
    times = []
    for _ in range(5):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            graph.replay()
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1) * 1000.0 / iters)
    times.sort()
    return dict(
        name=c["name"],
        gate=info["gate"],
        us_median=times[2],
        us_min=times[0],
        us_max=times[-1],
    )


def load_results(path):
    with open(path) as f:
        return {r["name"]: r for r in json.load(f)["results"]}


def compare(path_a, path_b):
    ra, rb = load_results(path_a), load_results(path_b)
    same = diff = 0
    for name in ra:
        if name in rb:
            eq = ra[name]["sha"] == rb[name]["sha"]
            same += eq
            diff += not eq
            if not eq:
                print(
                    f"DIFF {name}: {ra[name]['sha']} vs {rb[name]['sha']} "
                    f"max_abs {ra[name]['max_abs']:.2e} vs {rb[name]['max_abs']:.2e}"
                )
    print(f"bitwise equal: {same}, different: {diff}")
    return 0


def compare_bench(path_off, path_on):
    ra, rb = load_results(path_off), load_results(path_on)
    print(f"{'case':36s} gate    off_us     on_us  off/on")
    for name in ra:
        if name in rb:
            gate = ra[name]["gate"]
            off, on = ra[name]["us_median"], rb[name]["us_median"]
            print(f"{name:36s} {gate:4d} {off:9.1f} {on:9.1f} {off / on:6.2f}x")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--repeat", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="a 28-case subset")
    ap.add_argument("--only", default="", help="substring of the case name")
    ap.add_argument("--out", default="", help="write the results as JSON")
    ap.add_argument("--tol-abs", type=float, default=5e-3)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--compare-bench", nargs=2, metavar=("OFF.json", "ON.json"))
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    if a.compare_bench:
        return compare_bench(*a.compare_bench)

    env = os.environ.get("VLLM_FA_RDNA2_GQA_DECODE", "<unset>")
    props = torch.cuda.get_device_properties(0)
    print(
        f"# tag={a.tag} VLLM_FA_RDNA2_GQA_DECODE={env} dev={props.name} "
        f"arch={props.gcnArchName} torch={torch.__version__}"
    )
    results = []
    bad = 0
    t0 = time.time()
    if a.bench:
        for c in bench_cases():
            if a.only in c["name"]:
                r = bench_case(c, a.seed, a.iters)
                results.append(r)
                print(
                    f"{r['name']:36s} gate={r['gate']:3d} median "
                    f"{r['us_median']:8.1f} us (min {r['us_min']:.1f} "
                    f"max {r['us_max']:.1f})",
                    flush=True,
                )
    else:
        for c in cases(a.quick):
            if a.only not in c["name"]:
                continue
            r = run_case(c, a.seed, a.repeat)
            r["ok"] = (
                not r["nan"]
                and r["max_abs"] <= a.tol_abs
                and r["repeat_equal"]
                and r["pad_zero"]
            )
            results.append(r)
            bad += not r["ok"]
            print(
                f"{'OK  ' if r['ok'] else 'FAIL'} {r['name']:44s} T={r['T']:4d} "
                f"gate={r['gate']:3d} max_abs={r['max_abs']:.2e} "
                f"rel={r['max_rel']:.2e} nan={int(r['nan'])} "
                f"bad_rows={r['bad_rows']} rep_eq={int(r['repeat_equal'])} "
                f"sha={r['sha']}",
                flush=True,
            )
        summary = dict(
            tag=a.tag,
            env=env,
            n=len(results),
            failed=bad,
            seconds=round(time.time() - t0, 1),
        )
        print("SUMMARY", json.dumps(summary))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(tag=a.tag, env=env, results=results), f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
