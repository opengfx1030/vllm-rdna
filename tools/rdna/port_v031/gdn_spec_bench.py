#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the GDN spec-decode core (conv update + recurrent update) per layer.

Qwen3.8-27B at TP=4 per rank: 4 k-heads, 12 v-heads, head dim 128, conv
dim 2560, kernel 4. Each sequence has 1 + num_spec query tokens.

    HIP_VISIBLE_DEVICES=6 python tools/rdna/port_v031/gdn_spec_bench.py \
        [--seqs 1,8] [--spec 2] [--state-dtype float16]
"""

import argparse
import functools

import torch


def timeit(fn, iters=50):
    """Mean us per call, replayed from a graph (no host launch cost)."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        with torch.cuda.graph(g, stream=stream):
            for _ in range(iters):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="1,8")
    ap.add_argument("--spec", type=int, default=2)
    ap.add_argument("--state-dtype", default="float16")
    args = ap.parse_args()
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.third_party.flash_linear_attention.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )

    dev = "cuda"
    h, hv, dk, dv, width = 4, 12, 128, 128, 4
    dim = 2 * h * dk + hv * dv
    sdt = getattr(torch, args.state_dtype)
    q_len = 1 + args.spec
    blocks = 64
    for n in [int(v) for v in args.seqs.split(",")]:
        t = n * q_len
        x = torch.randn(t, dim, dtype=torch.half, device=dev)
        conv_state = torch.randn(
            blocks, dim, width - 1 + args.spec, dtype=torch.half, device=dev
        )
        w = torch.randn(dim, width, dtype=torch.half, device=dev)
        bias = torch.randn(dim, dtype=torch.half, device=dev)
        idx = torch.arange(1, 1 + n * q_len, dtype=torch.int32, device=dev).view(
            n, q_len
        )
        acc = torch.full((n,), q_len, dtype=torch.int32, device=dev)
        qsl = torch.arange(0, t + 1, q_len, dtype=torch.int32, device=dev)
        ssm = torch.randn(blocks, hv, dv, dk, dtype=sdt, device=dev) * 0.01
        a = torch.randn(t, hv, dtype=torch.half, device=dev)
        b = torch.randn(t, hv, dtype=torch.half, device=dev)
        a_log = torch.randn(hv, dtype=torch.float32, device=dev)
        dt_bias = torch.randn(hv, dtype=torch.float32, device=dev)

        conv = functools.partial(
            causal_conv1d_update,
            x,
            conv_state,
            w,
            bias,
            "silu",
            conv_state_indices=idx[:, 0],
            num_accepted_tokens=acc,
            query_start_loc=qsl,
            max_query_len=q_len,
            validate_data=False,
        )

        y = conv()
        q = y[:, : h * dk].reshape(1, t, h, dk)
        k = y[:, h * dk : 2 * h * dk].reshape(1, t, h, dk)
        v = y[:, 2 * h * dk :].reshape(1, t, hv, dv)

        rec = functools.partial(
            fused_sigmoid_gating_delta_rule_update,
            A_log=a_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            initial_state=ssm,
            inplace_final_state=True,
            cu_seqlens=qsl,
            ssm_state_indices=idx,
            num_accepted_tokens=acc,
            use_qk_l2norm_in_kernel=True,
        )

        tc = timeit(conv)
        tr = timeit(rec)
        print(
            f"seqs={n} tokens={t}: conv_update {tc:7.1f} us  "
            f"recurrent {tr:7.1f} us  (x48 layers = {(tc + tr) * 48 / 1000:.2f} ms)"
        )


if __name__ == "__main__":
    main()
