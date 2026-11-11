#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What can make a row's MoE result differ between two batches?

One GPU, real checkpoint tensors, the chain the expert servers run
(``ExpertServerModel.moe``, which is the hub's ``_rdna2_fused_moe`` too), in the
mode of ``VLLM_RDNA2_MOE_DETERMINISTIC``:

1. The call width. The first rows of a fixed batch are computed in calls of
   T = 1..48 rows (tile height 1 up to T = 4, else 4) and compared bit for bit with
   their result at T = 3 (the solo decode width), at other positions of a call and
   in another order. In Q32 mode they are all identical: a row's result does not
   depend on the batch it is in.
2. The grouping of a row's pairs over devices. A hetero layer sums the pairs of a
   row in parts (the hub's resident experts, one part per server); every part is
   rounded to fp16, the hub adds the parts in fp32 and rounds once more. The same
   pairs split another way give another fp16 result, and the cold copy of a pair is
   chosen by ``(row + layer) % n_cold``: the same token at another row of the batch
   is split another way. The probe splits the pairs of 48 rows two random ways in
   3 parts and compares the two sums with each other and with the one-call sum.

    HIP_VISIBLE_DEVICES=2 VLLM_RDNA2_MOE_DETERMINISTIC=1 \\
        PYTHONPATH=<tree> python tools/rdna/hetero_moe/blockm_probe.py --model <ckpt>

Prints one line per check and ``blockm_probe: <what was found>``. Exit 0 always (it
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

    def grouped(labels: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(rows, model.hidden, dtype=torch.float32)
        for g in range(3):
            part = torch.where(labels == g, ids, torch.full_like(ids, -1))
            out = model.moe(args.layer, x.to(dev), part.to(dev), w.to(dev))
            torch.accelerator.synchronize()
            total += out.cpu().float()
        return total.to(torch.float16)

    one = run(list(range(rows)))
    split_a = grouped(torch.randint(0, 3, ids.shape, generator=gen))
    split_b = grouped(torch.randint(0, 3, ids.shape, generator=gen))
    for what, a, b in (
        ("split A vs split B", split_a, split_b),
        ("split A vs one call", split_a, one),
    ):
        d = ulps(a, b)
        print(
            f"grouping, {what}: {int((d > 0).sum())} of {d.numel()} elements differ, "
            f"max {int(d.max())} ulp"
        )
        if int((d > 0).sum()):
            findings.append(f"{what}: the grouping of the pairs changes the result")
    print(
        "blockm_probe: "
        + ("; ".join(findings) if findings else "no result depends on T or order")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
