# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA sparse paged attention prefill benchmark for gfx1030 (RDNA2).

Times ``qsa_sparse_paged_attention_rdna2`` at prefill-like row counts and
context lengths, sweeping the key tile size ``VLLM_RDNA_QSA_BLOCK_N`` and
the split-K count ``VLLM_RDNA_QSA_SPLITS``.

The kernel reads its tuning knobs from the environment on every call, so the
sweep assigns ``os.environ[...]`` between timed loops.

Page size 16 and head_dim 256 come from ``tests/kernels/rocm/qwen4_exp/
test_qsa_rdna2.py`` (production geometry: head_dim 256, partial rotary
factor 0.25). The expanded top-k width defaults to 128, matching the x4
``qsa_expand_indices`` expansion in ``vllm/models/qwen4_exp/amd/ops/
qsa_rdna2.py``.

The script refuses to run on devices 0 and 1 (reserved by the parent job)
and prints the device name it actually uses.

  python benchmarks/kernels/benchmark_rdna_qsa_prefill.py --device 2 --rows 1024
"""

import argparse
import os
import statistics
import sys

import torch


def resolve_device(device_str):
    """Resolve the torch device index and guard physical HIP devices 0 and 1.

    The torch index is mapped through ``HIP_VISIBLE_DEVICES`` when it is set
    (comma list; unset = identity). Only the physical index is refused.
    """
    if device_str.isdigit():
        torch_idx = int(device_str)
    else:
        d = torch.device(device_str)
        torch_idx = d.index if d.index is not None else 0
    hip = os.environ.get("HIP_VISIBLE_DEVICES", "")
    if hip:
        vis = [int(x.strip()) for x in hip.split(",") if x.strip()]
        if torch_idx >= len(vis):
            print(
                f"torch index {torch_idx} is out of range for HIP_VISIBLE_DEVICES={hip}"
            )
            sys.exit(1)
        phys = vis[torch_idx]
    else:
        phys = torch_idx
    if phys in (0, 1):
        print(
            f"Refusing physical device {phys} (torch index {torch_idx}): "
            "devices 0 and 1 are reserved."
        )
        sys.exit(1)
    name = torch.cuda.get_device_name(torch_idx)
    print(f"Running on torch device {torch_idx} (physical HIP {phys}): {name}")
    torch.accelerator.set_device_index(torch_idx)
    return torch_idx


def make_inputs(rows, ctx, topk, page_size, head_dim, device):
    num_blocks = ctx // page_size + 1
    q = torch.randn(rows, 6, head_dim, device=device, dtype=torch.float16)
    k_cache = torch.randn(
        num_blocks, page_size, 1, head_dim, device=device, dtype=torch.float16
    )
    v_cache = torch.randn(
        num_blocks, page_size, 1, head_dim, device=device, dtype=torch.float16
    )
    logical_indices = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
    start_pos = max(0, ctx - rows)
    for i in range(rows):
        p = start_pos + i
        w = min(p + 1, 128)
        logical_indices[i, :w] = torch.arange(p + 1 - w, p + 1, device=device)
        if topk > w:
            older_max = p + 1 - w
            if older_max > 0:
                older = torch.randint(0, older_max, (topk - w,), device=device)
                logical_indices[i, w:] = older
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, logical_indices, block_table, token_to_req


def run_op(*inputs):
    q, k_cache, v_cache, logical_indices, block_table, token_to_req = inputs
    out = torch.empty_like(q)
    torch.ops._rocm_C.qsa_sparse_paged_attention_rdna2(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, out
    )
    return out


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.accelerator.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    return statistics.median(times), min(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    dev_id = resolve_device(args.device)
    device = torch.device(f"cuda:{dev_id}")
    torch.manual_seed(0)

    page_size = 16
    head_dim = 256

    q, k_cache, v_cache, logical_indices, block_table, token_to_req = make_inputs(
        args.rows, args.ctx, args.topk, page_size, head_dim, device
    )

    configs = [
        (16, "auto"),
        (32, "auto"),
        (16, 1),
        (32, 1),
        (16, 2),
        (32, 2),
        (16, 4),
        (32, 4),
    ]

    default_block_n = 32
    default_splits = "auto"

    os.environ["VLLM_RDNA_QSA_BLOCK_N"] = str(default_block_n)
    os.environ["VLLM_RDNA_QSA_SPLITS"] = "0"
    default_out = run_op(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req
    )
    default_med, default_min = bench(
        lambda: run_op(q, k_cache, v_cache, logical_indices, block_table, token_to_req),
        5,
        args.iters,
    )
    print(
        f"Default BLOCK_N={default_block_n}, SPLITS={default_splits}: "
        f"{default_med:.2f} us/call (median), {default_min:.2f} us/call (min)"
    )
    print(
        "Config, Median us, Min us, Speedup vs default, Max abs diff"
        + (" (fp16 tol 1e-2)" if args.check else "")
    )

    for bn, sp in configs:
        if bn == default_block_n and sp == default_splits:
            continue
        os.environ["VLLM_RDNA_QSA_BLOCK_N"] = str(bn)
        os.environ["VLLM_RDNA_QSA_SPLITS"] = "0" if sp == "auto" else str(sp)
        out = run_op(q, k_cache, v_cache, logical_indices, block_table, token_to_req)
        diff = (
            (out.float() - default_out.float()).abs().max().item()
            if args.check
            else -1.0
        )
        t_med, t_min = bench(
            lambda: run_op(
                q, k_cache, v_cache, logical_indices, block_table, token_to_req
            ),
            5,
            args.iters,
        )
        speedup = default_med / t_med if t_med > 0 else 0.0
        diff_str = f"{diff:.2e}" if args.check else "n/a"
        print(
            f"BLOCK_N={bn}, SPLITS={sp}: {t_med:.2f} us/call, {t_min:.2f} us/call, "
            f"{speedup:.2f}x, diff={diff_str}"
        )


if __name__ == "__main__":
    main()
