#!/usr/bin/env python3
"""E3 check: mono/rdna2/qsa.py store_rows3 (one launch) leaves the three indexer caches
bit-identical to three qsa_store_cache_rows calls, including PAD (-1) slots,
with the serve's layouts (rope rows int64 in a view of an fp16 page)."""

import torch


def main():
    from vllm.models.qwen4_exp.amd.mono.rdna2.qsa import store_rows3 as qsa_store_cache_rows3
    from vllm.models.qwen4_exp.amd.ops.qsa import qsa_store_cache_rows

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(3)
    ok = True
    for n in (1, 3, 24, 2048):
        pages, page = 64, 64
        comp = torch.randn(pages, page, 1, 128, generator=g, device=dev).half()
        raw = torch.randn(pages, page, 1, 128, generator=g, device=dev).half()
        # rope positions: int64 x3 stored in a padded fp16 page (stride 35 fp16-elements)
        rope_store = torch.zeros(pages, page, 1, 12, device=dev, dtype=torch.float16)
        rope = rope_store.view(torch.int64)[..., :3]
        c_slots = torch.randperm(pages * page, generator=g, device=dev)[:n].long()
        c_slots[::3] = -1  # most tokens close no group
        r_slots = torch.randperm(pages * page, generator=g, device=dev)[:n].long()
        crow = torch.randn(n, 1, 128, generator=g, device=dev).half()
        rrow = torch.randn(n, 1, 128, generator=g, device=dev).half()
        prow = torch.randint(0, 1 << 40, (3, n), generator=g, device=dev).t()  # transposed
        prow = prow.unsqueeze(1)
        a = [t.clone() for t in (comp, raw, rope_store)]
        qsa_store_cache_rows(a[0], c_slots, crow)
        qsa_store_cache_rows(a[1], r_slots, rrow)
        qsa_store_cache_rows(a[2].view(torch.int64)[..., :3], r_slots, prow)
        b = [t.clone() for t in (comp, raw, rope_store)]
        done = qsa_store_cache_rows3(b[0], c_slots, crow, b[1], r_slots, rrow,
                                     b[2].view(torch.int64)[..., :3], prow)
        torch.cuda.synchronize()
        # bitwise: int64 rows reinterpreted as fp16 can be NaN patterns
        a = [t.view(torch.int16) for t in a]
        b = [t.view(torch.int16) for t in b]
        same = done and all(torch.equal(x, y) for x, y in zip(a, b))
        ok &= same
        per = [bool(torch.equal(x, y)) for x, y in zip(a, b)]
        if not same:
            for i, (x, y) in enumerate(zip(a, b)):
                bad = (x != y).nonzero()
                if bad.numel():
                    print("   cache", i, "differs at", bad[:3].tolist(), "count", bad.shape[0])
        print(f"n {n:5d} fused {'yes' if done else 'NO'} identical {same} per-cache {per}")
    print("OK" if ok else "BAD")


if __name__ == "__main__":
    main()
