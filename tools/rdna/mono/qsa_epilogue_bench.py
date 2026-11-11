#!/usr/bin/env python3
"""QSA indexer epilogue after the qk projection, timed in a CUDA graph:

  A  norm_rope(q) + qsa_compress_groups + norm_rope_store3   (--monokernel now)
  B  mono_qsa_epilogue_rdna2                                  (one launch)

with the serve's layouts: ring = key + int64 RoPE tail in one fp16 page of
ratio * cdiv(ratio + nspec, ratio) slots, MRoPE positions, decode width 1
(MTP0) or 3 (MTP2).

  PYTHONPATH=<tree> python3 qsa_epilogue_bench.py [--rows 1,3,8,24]
"""

import argparse
from types import SimpleNamespace

import torch

FLASH_NEXT_ROPE = dict(
    head_size=256,
    max_position=262144,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
    },
)


def graph_us(fn, calls=100, reps=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / calls)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,3,8,24")
    a = ap.parse_args()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.models.qwen4_exp.amd.mono.rdna2 import qsa as MQ
    from vllm.models.qwen4_exp.amd.ops import qsa as Q
    from vllm.models.qwen4_exp.common.qsa_cache import canonical_qsa_rope_positions

    dev = torch.device("cuda")
    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(**FLASH_NEXT_ROPE, dtype=torch.float16).to(dev)
        qn = GemmaRMSNorm(128, eps=1e-6).to(device=dev, dtype=torch.float16)
        kn = GemmaRMSNorm(128, eps=1e-6).to(device=dev, dtype=torch.float16)
    ratio = 4
    print(f"{'width':>5} {'rows':>5} {'A today':>9} {'B epilogue':>11}   (us/call)")
    for width in (1, 3):
        ring = ratio * -(-(ratio + width - 1) // ratio)
        for n in [int(x) for x in a.rows.split(",")]:
            if n % width:
                continue
            g = torch.Generator(device=dev).manual_seed(n)
            n_req = n // width
            ctx = torch.randint(ratio + width, 20000, (n_req,), generator=g, device=dev)
            t2r = (torch.arange(n, device=dev) // width).int()
            off = width - 1 - (torch.arange(n, device=dev) % width)
            end = (ctx[t2r.long()] - 1 - off).long()
            qsl = torch.arange(0, n + 1, width, device=dev).int()
            blocks = n_req + 1
            st = torch.arange(1, blocks, device=dev).int().view(-1, 1)
            r_slots = st[t2r.long(), 0].long() * ring + end % ring
            c_slots = torch.randperm(256 * 64, generator=g, device=dev)[:n].long()
            c_slots[(end + 1) % ratio != 0] = -1
            pos = torch.stack((end, end, end))
            kv = torch.randn(blocks, ring, 1, 140, generator=g, device=dev).half()
            comp = torch.randn(256, 64, 1, 128, generator=g, device=dev).half()
            qk = torch.randn(n, 640, generator=g, device=dev).half()
            raw_md = SimpleNamespace(
                block_table=st,
                token_to_req=t2r,
                query_start_loc=qsl,
                logical_positions=end,
                slot_mapping=r_slots,
                max_query_len=width,
            )
            comp_md = SimpleNamespace(slot_mapping=c_slots)
            key_ring = kv[..., :128]
            rope_ring = kv[..., 128:].view(torch.int64)
            # the ring's RoPE tail holds real positions (indexes cos/sin)
            rope_ring.copy_(
                torch.randint(0, 20000, rope_ring.shape, generator=g, device=dev)
            )

            def today(
                qk=qk,
                pos=pos,
                end=end,
                st=st,
                t2r=t2r,
                qsl=qsl,
                c_slots=c_slots,
                r_slots=r_slots,
                key_ring=key_ring,
                rope_ring=rope_ring,
                comp=comp,
                n=n,
            ):
                MQ.norm_rope(qn, rope, pos, qk[:, :512].view(n, 4, 128))
                token_k = qk[:, 512:].reshape(n, 1, 128)
                rows = canonical_qsa_rope_positions(pos)
                pooled, first = Q.qsa_compress_groups_with_ratio(
                    token_k,
                    rows,
                    key_ring,
                    st,
                    t2r,
                    qsl,
                    end,
                    c_slots,
                    ratio,
                    rope_ring,
                )
                assert MQ.norm_rope_store3(
                    kn,
                    rope,
                    first,
                    pooled,
                    comp,
                    c_slots,
                    key_ring,
                    r_slots,
                    token_k,
                    rope_ring,
                    rows,
                )

            def one(
                qk=qk,
                pos=pos,
                raw_md=raw_md,
                comp_md=comp_md,
                comp=comp,
                key_ring=key_ring,
                rope_ring=rope_ring,
            ):
                assert (
                    MQ.epilogue(
                        qn,
                        kn,
                        rope,
                        qk,
                        pos,
                        raw_md,
                        comp_md,
                        ratio,
                        comp,
                        key_ring,
                        rope_ring,
                    )
                    is not None
                )

            print(f"{width:5d} {n:5d} {graph_us(today):9.2f} {graph_us(one):11.2f}")


if __name__ == "__main__":
    main()
