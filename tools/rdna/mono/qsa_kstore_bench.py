#!/usr/bin/env python3
"""QSA indexer compressed-key epilogue + cache stores, timed in a CUDA graph:

  A  norm_rope (Triton) + store_rows3 (Triton)      -- --monokernel today
  B  norm_rope (Triton) + store_rows3 (HIP)
  C  norm_rope_store3 (HIP, one launch)

with the serve's layouts (Flash-Next interleaved MRoPE, first positions [n, 3]
int64, rope rows int64 in a padded fp16 page, ~1/3 PAD compressed slots).

  PYTHONPATH=<tree> python3 qsa_kstore_bench.py [--rows 1,3,8,24,512,2048]
"""

import argparse

import torch

FLASH_NEXT_ROPE = dict(  # the main attention's rope (rotary_dim 64)
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
    ap.add_argument("--rows", default="1,3,8,24,512,2048")
    a = ap.parse_args()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.models.qwen4_exp.amd.mono.rdna2 import qsa as MQ

    dev = torch.device("cuda")
    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(**FLASH_NEXT_ROPE, dtype=torch.float16).to(dev)
        norm = GemmaRMSNorm(128, eps=1e-6).to(device=dev, dtype=torch.float16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(128, device=dev) * 0.1)
    print(
        f"{'rows':>6} {'A tri+tri':>10} {'B tri+HIP':>10} {'C fused HIP':>12}"
        "   (us/call, graph)"
    )
    for n in [int(x) for x in a.rows.split(",")]:
        g = torch.Generator(device=dev).manual_seed(n)
        pages, page = 256, 64
        comp = torch.randn(pages, page, 1, 128, generator=g, device=dev).half()
        raw = torch.randn(pages, page, 1, 128, generator=g, device=dev).half()
        rope_store = torch.zeros(pages, page, 1, 12, device=dev, dtype=torch.float16)
        rope_cache = rope_store.view(torch.int64)[..., :3]
        cs = torch.randperm(pages * page, generator=g, device=dev)[:n].long()
        cs[::3] = -1
        rs = torch.randperm(pages * page, generator=g, device=dev)[:n].long()
        pooled = torch.randn(n, 1, 128, generator=g, device=dev).half()
        first = torch.randint(0, 20000, (n, 3), generator=g, device=dev)
        rrow = torch.randn(n, 1, 128, generator=g, device=dev).half()
        prow = torch.randint(0, 1 << 40, (3, n), generator=g, device=dev).t()
        prow = prow.unsqueeze(1)

        caches = (comp, raw, rope_cache)
        rows = (pooled, first, cs, rs, rrow, prow)

        def two_launches(hip_store, caches, rows):
            comp, raw, rope_cache = caches
            pooled, first, cs, rs, rrow, prow = rows
            normalized = MQ.norm_rope(norm, rope, first.transpose(0, 1), pooled)
            orig = MQ._has_hip_store3
            MQ._has_hip_store3 = (lambda: True) if hip_store else (lambda: False)
            try:
                MQ.store_rows3(comp, cs, normalized, raw, rs, rrow, rope_cache, prow)
            finally:
                MQ._has_hip_store3 = orig

        def fused(caches, rows):
            comp, raw, rope_cache = caches
            pooled, first, cs, rs, rrow, prow = rows
            assert MQ.norm_rope_store3(
                norm, rope, first, pooled, comp, cs, raw, rs, rrow, rope_cache, prow
            )

        ta = graph_us(lambda c=caches, r=rows: two_launches(False, c, r))
        tb = graph_us(lambda c=caches, r=rows: two_launches(True, c, r))
        tc = graph_us(lambda c=caches, r=rows: fused(c, r))
        print(f"{n:6d} {ta:10.2f} {tb:10.2f} {tc:12.2f}")


if __name__ == "__main__":
    main()
