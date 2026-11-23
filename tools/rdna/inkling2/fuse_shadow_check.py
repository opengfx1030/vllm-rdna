#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wiring check of the fused attention-chain launches on the real checkpoint.

Boots Inkling-2-Mini eagerly (one GPU, ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so the
model lives in this process), then runs a few requests while every fused launch of
``vllm.models.inkling2_loop.amd.fuse_chain`` is shadowed by the shipped launches it
replaces, run on the *same inputs* (clones of the conv state / block states):

* ``MessageBatch.fill``: the batched message launch against one
  ``inkling2_glue_attn_res`` per layer;
* ``attn_res_conv``, if ``fuse_chain`` has a wiring of the fused conv (the model in
  the tree does not call it): the fused conv + attn_res against
  ``inkling2_sconv_static`` + ``inkling2_glue_attn_res``, comparing ``out``,
  ``partial_out``, ``x1_out``, the appended block state and the conv state.

Every comparison is **bit exact** and deterministic (these kernels have no atomics), so
unlike a greedy-output comparison it is not blurred by the run-to-run noise of the
expert GEMMs.  It proves that the model passes the right queries, layers, metadata and
state to the fused ops on real data; the kernels themselves are covered by
``tests/kernels/core/test_inkling2_fuse_attn_rdna2.py``.

    HIP_VISIBLE_DEVICES=2 python fuse_shadow_check.py --ckpt ~/hfcache/inkling2-mini
"""

from __future__ import annotations

import os

# the model must live in this process: set before vllm is imported
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import random
import sys
import time

import torch


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = a.contiguous().view(-1), b.contiguous().view(-1)
    iv = {4: torch.int32, 2: torch.int16}[a.element_size()]
    return bool(torch.equal(a.view(iv), b.view(iv)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--requests", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--prompt-len", type=int, default=22)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm import _custom_ops as ops
    from vllm.models.inkling2_loop.amd import fuse_chain

    stats = {"fill": 0, "conv": 0, "bad": []}

    orig_fill = fuse_chain.MessageBatch.fill

    def fill(self, v, n_prev, eps):
        orig_fill(self, v, n_prev, eps)
        ref = torch.empty_like(self.buf[: v.T])
        tmp = torch.empty(v.T, 4, ref.shape[-1], dtype=torch.float32, device=ref.device)
        for j in range(self.queries.shape[0]):
            ops.inkling2_glue_attn_res(
                v.prev_blk, n_prev, v.prev_x0, self.queries[j], tmp,
                x1=v.prev_x1, eps=eps,
            )  # fmt: skip
            ref[:, j] = tmp
        torch.accelerator.synchronize()
        stats["fill"] += 1
        if not same_bits(ref, self.buf[: v.T]):
            stats["bad"].append(f"fill T={v.T} n_prev={n_prev}")

    fuse_chain.MessageBatch.fill = fill

    orig_conv = getattr(fuse_chain, "attn_res_conv", None)
    if orig_conv is not None:

        def attn_res_conv(
            blocks, n_blocks, x0, x, conv, owner, md, stream, query, out, *,
            x1_out=None, partial_out=None, append=None, norm_w=None, eps,
            norm_eps=None,
        ):  # fmt: skip
            # the reference runs on a private copy of the pages the launch can address
            # (cloning the whole cache would read pages the layer never mapped)
            state = owner.state
            bt = md.block_table
            ids = torch.unique(bt.reshape(-1))
            ids = ids[(ids >= 0) & (ids < state.shape[0])].long()
            state_r = state[ids].clone()
            remap = torch.zeros(state.shape[0], dtype=bt.dtype, device=bt.device)
            remap[ids] = torch.arange(ids.numel(), dtype=bt.dtype, device=bt.device)
            bt_r = torch.where(
                (bt >= 0) & (bt < state.shape[0]),
                remap[bt.clamp(0, state.shape[0] - 1).long()],
                bt,
            )
            blocks_r = blocks.clone()
            c = torch.empty(
                x.shape[0], x.shape[1], dtype=torch.float32, device=x.device
            )
            out_r = torch.empty_like(out)
            po_r = None if partial_out is None else torch.empty_like(partial_out)
            ops.inkling2_sconv_static(
                x, conv.conv1d.weight, c, state_r, bt_r, md.seq_idx,
                md.query_start_loc, md.seq_lens, owner.stream_ranges[stream][0],
                owner.num_kv_heads, owner.block_size, True,
            )  # fmt: skip
            ops.inkling2_glue_attn_res(
                blocks_r, n_blocks, x0 if x0 is not None else c, query, out_r,
                x1=c if x0 is not None else None, partial_out=po_r, append=append,
                norm_w=norm_w, eps=eps, norm_eps=eps if norm_eps is None else norm_eps,
            )  # fmt: skip
            orig_conv(
                blocks, n_blocks, x0, x, conv, owner, md, stream, query, out,
                x1_out=x1_out, partial_out=partial_out, append=append,
                norm_w=norm_w, eps=eps, norm_eps=norm_eps,
            )  # fmt: skip
            torch.accelerator.synchronize()
            stats["conv"] += 1
            what = f"conv T={x.shape[0]} n_blocks={n_blocks} stream={stream}"
            if not same_bits(out, out_r):
                stats["bad"].append(what + " out")
            if po_r is not None and not same_bits(partial_out, po_r):
                stats["bad"].append(what + " partial_out")
            if x1_out is not None and not same_bits(x1_out, c):
                stats["bad"].append(what + " x1_out")
            if append is not None and not same_bits(blocks, blocks_r):
                stats["bad"].append(what + " appended block")
            if not same_bits(state[ids], state_r):
                stats["bad"].append(what + " conv state")

        fuse_chain.attn_res_conv = attn_res_conv

    t0 = time.time()
    llm = LLM(
        model=args.ckpt,
        dtype="float16",
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=max(args.requests, 2),
        skip_tokenizer_init=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
        limit_mm_per_prompt={"image": 0},
    )
    print(f"BOOT OK in {time.time() - t0:.1f}s", flush=True)
    rng = random.Random(7)
    prompts = [
        {
            "prompt_token_ids": [
                rng.randrange(1000, 50000) for _ in range(args.prompt_len)
            ]
        }
        for _ in range(args.requests)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=args.tokens, ignore_eos=True)
    llm.generate(prompts, sp, use_tqdm=False)
    llm.generate(prompts[:1], sp, use_tqdm=False)  # batch 1
    print(
        f"shadowed launches: fill {stats['fill']}, conv {stats['conv']}; "
        f"mismatches: {len(stats['bad'])}",
        flush=True,
    )
    for b in stats["bad"][:20]:
        print("MISMATCH", b)
    need_conv = orig_conv is not None and fuse_chain.FUSE_ATTN_CONV
    ok = not stats["bad"] and stats["fill"] > 0 and (stats["conv"] > 0 or not need_conv)
    print("SHADOW CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
