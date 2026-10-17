#!/usr/bin/env python
"""QSA sparse paged attention: HIP vs the Triton splitk+merge pair.

The sparse-attention table in this PR had no bench in the tree, which is how it ended
up asserting both "HIP is faster at all nine served shapes" and "1.9-2.7x slower than
Triton" in the same body. This reuses the parity test's own input builder, so both arms
run shapes `test_qsa_rdna2.py` already verifies, and reports the same two numbers
`bench_qsa_rdna2.py` reports:

  call_us    wall time per call, CUDA events around a batched loop. What a
             non-cudagraph decode step pays; dominated by Python + Triton launch
             overhead (~20 us) rather than by the kernel.
  kernel_us  summed GPU kernel time from torch.profiler. What a cudagraph-captured
             step pays, and the number to compare kernels with.

The arms differ only in VLLM_RDNA_QSA_HIP, as in bench_qsa_rdna2.py.

    HIP_VISIBLE_DEVICES=1 .venv/bin/python tools/rdna2/bench_qsa_sparse_rdna2.py
"""

import sys

sys.path.insert(0, "tests/kernels/rocm/qwen4_exp")
sys.path.insert(0, "tools/rdna2")

import bench_qsa_rdna2 as B  # noqa: E402
import test_qsa_rdna2 as T  # noqa: E402

from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops  # noqa: E402

# Production geometry, TP4: main attention is num_attention_heads 24 / TP4 = 6 q
# heads, num_key_value_heads 2 -> 1 kv head, head_dim 256 (config.json). Do not use
# heads=4 here: that is the *indexer* MQA geometry (indexer_n_heads 4,
# indexer_head_dim 128), a different op benched by bench_qsa_rdna2.py.
HEADS, KV_HEADS, HEAD_DIM, TOPK, PAGE_SIZE = 6, 1, 256, 2048, 1024
NUM_PAGES, NUM_REQUESTS = 64, 4


def main():
    print(
        f"geometry heads={HEADS} kv_heads={KV_HEADS} head_dim={HEAD_DIM} "
        f"topk={TOPK} page_size={PAGE_SIZE} num_pages={NUM_PAGES} "
        f"reqs={NUM_REQUESTS}"
    )
    print(
        f"{'rows':>6}  {'call_us triton -> hip':>28}  {'kernel_us triton -> hip':>30}"
    )
    for rows in B.ROWS:
        q, k_cache, v_cache, logical, block_table, token_to_req = T._sparse_inputs(
            rows, HEADS, KV_HEADS, HEAD_DIM, TOPK, PAGE_SIZE, NUM_PAGES,
            NUM_REQUESTS, seed=rows,
        )

        # The gate is a process-wide env var, so each arm must be measured while
        # its own gate is set — do not hoist the lambdas and measure later.
        B.set_gate(False)
        ref = lambda: qsa_ops.qsa_sparse_paged_attention(
            q, k_cache, v_cache, logical, block_table, token_to_req
        )
        out_ref = ref()
        c0, k0 = B.call_us(ref), B.kernel_us(ref)

        B.set_gate(True)
        hip = lambda: qsa_ops.qsa_sparse_paged_attention(
            q, k_cache, v_cache, logical, block_table, token_to_req
        )
        out_hip = hip()
        c1, k1 = B.call_us(hip), B.kernel_us(hip)

        worst = (out_hip.float() - out_ref.float()).abs().max().item()

        print(
            f"{rows:>6}  {c0:11.2f} ->{c1:8.2f} {100 * (c0 - c1) / c0:+6.1f}%"
            f"  {k0:12.2f} ->{k1:8.2f} {100 * (k0 - k1) / k0:+7.1f}%"
            f"   maxdiff={worst:.5f}"
        )


if __name__ == "__main__":
    main()
