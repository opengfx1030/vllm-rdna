#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time one prefill-attention layer call on a gfx1030 at the 27B TP=2 shapes.

Qwen3.5/3.8 27B at TP=2 has 12 query heads and 2 KV heads of head_dim 256 per
rank (``--heads-q 12 --heads-kv 2``). One chunk of the chunked prefill is one
call per attention layer (16 per step). The script times, with HIP events:

* ``flash``  : the fork's paged varlen kernel
  (``fa_rdna2_prefill_paged_varlen`` -> ``fa_prefill_paged_varlen_gqa_kernel``),
* ``unfused``: the textbook attention written with two rocBLAS GEMMs and a
  softmax (the 6 query heads of a KV group are folded into the GEMM rows, the
  query rows are walked in blocks of ``--qblock`` and keys past the last row
  of a block are not computed), on K/V gathered from the same paged cache.
  This is a yardstick for what the GPU can do on these shapes, not a backend:
  it needs a ``[6 * qblock, ctx + q]`` fp16 score tile.

and reports ms per call and useful causal TFLOP/s
(``4 * H_q * D * (q * ctx + q * (q + 1) / 2)``). ``--check`` compares the two
outputs on a small shape first.

    HIP_VISIBLE_DEVICES=3 python bench_attn_prefill.py --q 2048 \
        --ctx 0,2048,6144,14336,30720

No technique is taken from outside code: the flash kernel is the fork's, the
unfused variant is plain softmax(Q K^T / sqrt(D)) V.
"""

import argparse
import json
import statistics
import sys

import torch

X_DIM = 8


def make_cache(
    tokens: int, h_kv: int, d: int, block_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nb = (tokens + block_size - 1) // block_size
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (nb, h_kv, d // X_DIM, block_size, X_DIM)
    kc = torch.randn(shape, device="cuda", dtype=torch.float16, generator=g)
    vc = torch.randn(shape, device="cuda", dtype=torch.float16, generator=g)
    table = torch.arange(nb, device="cuda", dtype=torch.int32)[None]
    return kc, vc, table


def gather_contiguous(cache: torch.Tensor, tokens: int) -> torch.Tensor:
    """[nb, H_kv, D/x, bs, x] paged cache -> [tokens, H_kv, D]."""
    nb, h_kv, dx, bs, x = cache.shape
    return cache.permute(0, 3, 1, 2, 4).reshape(nb * bs, h_kv, dx * x)[:tokens]


def unfused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: int,
    qblock: int,
    tri: torch.Tensor,
) -> torch.Tensor:
    """Causal attention of q rows at positions [ctx, ctx + n) over k/v[:ctx + n]."""
    n, h_q, d = q.shape
    h_kv = k.shape[1]
    g = h_q // h_kv
    out = torch.empty_like(q)
    qs = q * (d**-0.5)  # a power of two for d = 256: exact in fp16
    for h in range(h_kv):
        kh = k[:, h, :]
        vh = v[:, h, :]
        for lo in range(0, n, qblock):
            hi = min(lo + qblock, n)
            rows = hi - lo
            s_len = ctx + hi
            qb = qs[lo:hi, h * g : (h + 1) * g, :].permute(1, 0, 2).reshape(g * rows, d)
            s = torch.matmul(qb, kh[:s_len].T)  # [g * rows, s_len]
            s.view(g, rows, s_len)[:, :, s_len - rows :] += tri[:rows, :rows]
            p = torch.softmax(s, dim=-1)
            o = torch.matmul(p, vh[:s_len])  # [g * rows, d]
            out[lo:hi, h * g : (h + 1) * g, :] = o.view(g, rows, d).permute(1, 0, 2)
    return out


def time_ms(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    out = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--heads-q", type=int, default=12)
    ap.add_argument("--heads-kv", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--q", default="2048", help="query tokens of the chunk, comma list")
    ap.add_argument(
        "--ctx", default="0,2048,6144,14336", help="cached tokens before it, comma list"
    )
    ap.add_argument("--qblock", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-unfused", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    from vllm.v1.attention.ops import fa_rdna2_backend as fa

    dev = torch.accelerator.current_device_index()
    props = torch.cuda.get_device_properties(dev)
    print(
        f"device {dev} {props.name} {props.gcnArchName} pci_bus {props.pci_bus_id:#x}"
    )
    h_q, h_kv, d = a.heads_q, a.heads_kv, a.head_dim
    qs = [int(x) for x in a.q.split(",")]
    ctxs = [int(x) for x in a.ctx.split(",")]
    tri = torch.triu(
        torch.full((a.qblock, a.qblock), float("-inf"), device="cuda"), diagonal=1
    ).half()
    rows: list[dict] = []

    def run_case(n: int, ctx: int, check: bool) -> None:
        total = ctx + n
        kc, vc, table = make_cache(total, h_kv, d, a.block_size, seed=1)
        g = torch.Generator(device="cuda").manual_seed(2)
        q = torch.randn(n, h_q, d, device="cuda", dtype=torch.float16, generator=g)
        cu = torch.tensor([0, n], device="cuda", dtype=torch.int32)
        sl = torch.tensor([total], device="cuda", dtype=torch.int32)
        out = torch.empty_like(q)

        def flash() -> torch.Tensor:
            return fa.fa_rdna2_prefill_paged_varlen(
                q, kc, vc, table, cu, sl, a.block_size, True, 0, None, out
            )

        k = gather_contiguous(kc, total).contiguous()
        v = gather_contiguous(vc, total).contiguous()
        ref = None
        if not a.no_unfused or check:
            ref = unfused(q, k, v, ctx, a.qblock, tri)
        if check and ref is not None:
            o = flash().float()
            r = ref.float()
            print(
                f"check q={n} ctx={ctx}: max|flash-unfused| "
                f"{(o - r).abs().max().item():.4g}, rel L2 "
                f"{((o - r).norm() / r.norm()).item():.3g}"
            )
        flops = 4.0 * h_q * d * (n * ctx + n * (n + 1) / 2)
        f_ms = statistics.median(time_ms(flash, a.warmup, a.iters))
        row = {
            "q": n,
            "ctx": ctx,
            "flash_ms": f_ms,
            "flash_tflops": flops / f_ms / 1e9,
        }
        if not a.no_unfused:
            u_ms = statistics.median(
                time_ms(lambda: unfused(q, k, v, ctx, a.qblock, tri), a.warmup, a.iters)
            )
            row["unfused_ms"] = u_ms
            row["unfused_tflops"] = flops / u_ms / 1e9
            row["speedup"] = f_ms / u_ms
        rows.append(row)
        print(
            " ".join(
                f"{k_}={v_:.3f}" if isinstance(v_, float) else f"{k_}={v_}"
                for k_, v_ in row.items()
            ),
            flush=True,
        )

    if a.check:
        run_case(256, 512, True)
    for n in qs:
        for ctx in ctxs:
            run_case(n, ctx, False)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
