#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Does a row's MoE result depend on how many rows the call has?

One GPU, real checkpoint tensors, the chain the expert servers run
(``ExpertServerModel.moe``, which is the hub's ``_rdna2_fused_moe`` too). The first
rows of a fixed batch are computed in calls of T = 1..48 rows and compared with
their result at T = 3 (the solo decode width). ``_rdna2_fused_moe`` picks its tile
height from T (1 row up to T = 4, else 4), and the Q32 deterministic mode
(``VLLM_RDNA2_MOE_DETERMINISTIC=1``) is bit-stable for a fixed tile height only, so
the concurrency probe's ``solo`` (T = 3) and ``together`` (T = 6) runs may differ
at the last ULP of every expert output even when the transport is exact. The
probe also checks repeatability and that the row order inside a call does not
matter.

    HIP_VISIBLE_DEVICES=2 VLLM_RDNA2_MOE_DETERMINISTIC=1 \\
        PYTHONPATH=<tree> python tools/rdna/hetero_moe/blockm_probe.py --model <ckpt>

Prints one line per T and ``blockm_probe: <what was found>``. Exit 0 always (it
measures); run it under the lease of the GPU.
"""

import argparse
import sys

import torch

WIDTHS = (1, 2, 3, 4, 5, 6, 8, 12, 24, 48)


def ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance of two fp16 tensors in units in the last place (sign-magnitude)."""

    def key(t: torch.Tensor) -> torch.Tensor:
        bits = t.view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (key(a) - key(b)).abs()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=24, help="experts held (>= 10)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.experts < 10:
        ap.error("--experts must be at least top_k = 10")

    from vllm import envs
    from vllm.distributed.hetero_moe.server_model import ExpertServerModel

    gen = torch.Generator().manual_seed(args.seed)
    held = torch.randperm(512, generator=gen)[: args.experts].sort().values
    sets = [[] for _ in range(args.layer)] + [[int(e) for e in held]]
    model = ExpertServerModel(
        args.model, sets, args.device, num_experts=512, top_k=10, max_capture_rows=48
    )
    model.load()
    dev = model.device
    rows = max(WIDTHS)
    x = (torch.randn(rows, model.hidden, generator=gen) * 0.5).to(torch.float16)
    ids = torch.stack(
        [held[torch.randperm(args.experts, generator=gen)[:10]] for _ in range(rows)]
    ).to(torch.int32)
    w = torch.softmax(torch.randn(rows, 10, generator=gen), dim=-1)

    def run(order: list[int]) -> torch.Tensor:
        sel = torch.tensor(order)
        out = model.moe(
            args.layer,
            x[sel].contiguous().to(dev),
            ids[sel].contiguous().to(dev),
            w[sel].contiguous().to(dev),
        )
        torch.accelerator.synchronize()
        return out.cpu()

    print(
        f"VLLM_RDNA2_MOE_DETERMINISTIC={int(envs.VLLM_RDNA2_MOE_DETERMINISTIC)} "
        f"hidden {model.hidden} layer {args.layer} experts {args.experts} on {dev}"
    )
    ref = run([0, 1, 2])
    findings: list[str] = []
    again = run([0, 1, 2])
    same = bool(torch.equal(ref, again))
    print(f"repeat T=3: {'bit-identical' if same else 'DIFFERENT'}")
    if not same:
        findings.append("the same call is not repeatable")
    for t in WIDTHS:
        out = run(list(range(t)))
        n = min(t, 3)
        diff = ulps(out[:n], ref[:n])
        bad = int((diff > 0).sum())
        print(
            f"T={t:2d} tile {1 if t <= 4 else 4}: rows 0..{n - 1} vs T=3: "
            f"{bad} of {out[:n].numel()} elements differ, max {int(diff.max())} ulp"
        )
        if bad:
            findings.append(f"T={t} differs from T=3")
    perm = run([2, 0, 1])
    d = ulps(perm[1:3], ref[:2])
    print(f"rows permuted in the call (T=3): {int((d > 0).sum())} elements differ")
    if int((d > 0).sum()):
        findings.append("row order inside a call matters")
    moved = run([3, 4, 5, 0, 1, 2])
    d = ulps(moved[3:6], ref)
    print(f"rows 0..2 moved to positions 3..5 (T=6): {int((d > 0).sum())} differ")
    if int((d > 0).sum()):
        findings.append("a row's position inside a call matters")
    print(
        "blockm_probe: "
        + ("; ".join(findings) if findings else "no result depends on T or order")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
