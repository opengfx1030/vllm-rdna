#!/usr/bin/env python3
"""E3 check: the indexer's fused GemmaRMSNorm + MRoPE (mono/rdna2/qsa.py norm_rope)
against today's apply_qsa_rmsnorm + apply_qsa_rope, with the Flash-Next rope
parameters, for the indexer q (4 heads) and the compressed k (1 head), plus the
time of both in a graph (8 calls per graph).

  PYTHONPATH=<tree> python3 qsa_normrope_check.py --config <model>/config.json
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import after_bench as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--widths", default="1,3,8,24,2048")
    a = ap.parse_args()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.models.qwen4_exp.amd.indexer_qsa import apply_qsa_rmsnorm, apply_qsa_rope
    from vllm.models.qwen4_exp.amd.mono.rdna2.qsa import norm_rope as apply_qsa_rmsnorm_rope

    cfg = json.load(open(a.config))
    t = cfg.get("text_config", cfg)
    dev = torch.device("cuda")
    torch.set_default_dtype(torch.float16)
    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(head_size=t["head_dim"], max_position=t["max_position_embeddings"],
                        rope_parameters=t["rope_parameters"]).to(dev)
        print("rope", type(rope).__name__, "rotary_dim", rope.rotary_dim, "neox",
              rope.is_neox_style, "mrope", getattr(rope, "mrope_section", None),
              "interleaved", getattr(rope, "mrope_interleaved", None))
        ok = True
        for heads in (4, 1):
            norm = GemmaRMSNorm(128, eps=t["rms_norm_eps"]).to(dev)
            with torch.no_grad():
                norm.weight.copy_(torch.randn(128, device=dev) * 0.1)
            for n in [int(x) for x in a.widths.split(",")]:
                x = torch.randn(n, heads, 128, device=dev, dtype=torch.float16)
                pos = torch.randint(0, 20000, (3, n), device=dev)
                if not getattr(rope, "mrope_section", None):
                    pos = pos[0]

                def ref():
                    y = apply_qsa_rmsnorm(norm, x.reshape(-1, 128)).reshape_as(x)
                    return apply_qsa_rope(rope, pos, y)

                def fused():
                    return apply_qsa_rmsnorm_rope(norm, rope, pos, x)

                r, f = ref(), fused()
                if f is None:
                    print("fused path not applicable"); return
                d = (r.float() - f.float()).abs().max().item()
                good = d < 4e-3
                ok &= good
                tr = A.graph_time(lambda: [ref() for _ in range(8)], 20) / 8
                tf = A.graph_time(lambda: [fused() for _ in range(8)], 20) / 8
                print(f"heads {heads} n {n:5d} max diff {d:.1e} | today {tr:5.1f} us fused "
                      f"{tf:5.1f} us {'ok' if good else 'BAD'}")
        print("OK" if ok else "BAD")


if __name__ == "__main__":
    main()
