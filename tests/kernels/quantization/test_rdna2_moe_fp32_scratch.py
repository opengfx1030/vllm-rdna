# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fp32 scratch of the RDNA2 MoE GEMMs (``fp32_accum=True``) is per call.

Both ops accumulate into ``moe_fp32_scratch`` (csrc/rocm/moe_accum_rdna2.cuh).
It used to be cached per (rows, n, device) and never freed, so every new token
count left rows x n x 4 bytes behind (61,440 x T bytes per count T at the
Flash-Next shapes). It now comes from the caching allocator on every call;
under graph capture that is the graph's private pool.

Run ``pytest tests/kernels/quantization/test_rdna2_moe_fp32_scratch.py``.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 MoE kernels are ROCm-only", allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32,
)
from vllm.platforms.rocm import on_gfx10x  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (
        on_gfx10x()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna2")
        and hasattr(torch.ops._rocm_C, "moe_w4a8_gemm_rdna2")
    ),
    reason="Requires gfx1030 with both RDNA2 MoE GEMM ops",
)

device = "cuda"
E, K, N, TOP_K, GROUP, BLOCK_M = 8, 512, 256, 4, 128, 4
OPS = ["w4a16", "w4a8"]


def _weights():
    torch.manual_seed(0)
    q = torch.randint(0, 16, (E, K, N), dtype=torch.int32, device=device)
    packed = torch.zeros(E, K // 8, N, dtype=torch.int32, device=device)
    for i in range(8):
        packed |= (q[:, i::8, :] & 0xF) << (i * 4)
    for e in range(E):
        we = packed[e].contiguous()
        ops.gptq_shuffle(we, 4)
        packed[e] = we
    scales = (0.05 * torch.rand(E, K // GROUP, N, device=device) + 0.01).half()
    zeros = torch.full(
        (K // GROUP, N),
        scalar_types.uint4b8.bias - 1,
        dtype=torch.int32,
        device=device,
    )
    zeros = pack_quantized_values_into_int32(zeros, scalar_types.uint4b8, packed_dim=1)
    return packed, scales, zeros.unsqueeze(0).expand(E, -1, -1).contiguous()


def _inputs(tokens: int, seed: int):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(tokens, K, dtype=torch.float16, device=device, generator=gen)
    topk_ids = torch.randint(
        0, E, (tokens, TOP_K), dtype=torch.int32, device=device, generator=gen
    )
    out = torch.empty(tokens * TOP_K, N, dtype=torch.float16, device=device)
    return x, topk_ids, out


def _launch(op, weights, x, topk_ids, out):
    """Routing plus one fp32-accumulating MoE GEMM, graph-capturable."""
    w, s, z = weights
    si, ei, ntp = moe_align_block_size(topk_ids, BLOCK_M, E)
    no_weights = torch.empty(0, device=device)
    gemm = ops.moe_gptq_gemm_rdna2 if op == "w4a16" else ops.moe_w4a8_gemm_rdna2
    gemm(
        x,
        out,
        w,
        s,
        z,
        no_weights,
        si,
        ei,
        ntp,
        TOP_K,
        BLOCK_M,
        False,
        0,
        fp32_accum=True,
    )


def _scratch_bytes(tokens: int) -> int:
    return tokens * TOP_K * N * 4


# The old cache was shared by both ops, so each op takes its own token counts:
# with the same counts the second op would find the first op's buffers.
@pytest.mark.parametrize("op,first", [("w4a16", 1), ("w4a8", 41)])
@torch.inference_mode()
def test_eager_calls_keep_no_scratch(op, first):
    """Eager calls at 40 new token counts leave allocated memory unchanged."""
    weights = _weights()
    counts = range(first, first + 40)
    # Allocator and kernel warm-up, at a count outside both ranges.
    _launch(op, weights, *_inputs(200 + first, seed=0))
    torch.accelerator.synchronize()
    before = torch.accelerator.memory_allocated()
    for tokens in counts:
        _launch(op, weights, *_inputs(tokens, seed=tokens))
    torch.accelerator.synchronize()
    kept = torch.accelerator.memory_allocated() - before
    expected_leak = sum(_scratch_bytes(t) for t in counts)
    assert kept == 0, (
        f"{kept} bytes still allocated after eager calls at 40 token counts "
        f"(one never-freed scratch per count would be {expected_leak})"
    )


@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_graphs_in_one_pool_replay_after_eager_calls(op):
    """Graphs captured at two sizes in one pool (as vLLM captures its sizes)
    replay correctly, in any order, after eager calls at other sizes and with
    live tensors of every scratch size that a stale pointer would hit."""
    weights = _weights()
    sizes = (24, 12)  # vLLM captures the largest size first
    static = {t: _inputs(t, seed=t) for t in sizes}

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for t in sizes:
            _launch(op, weights, *static[t])
    torch.cuda.current_stream().wait_stream(side)
    graphs = {}
    pool = None
    for t in sizes:
        graphs[t] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[t], pool=pool):
            _launch(op, weights, *static[t])
        pool = graphs[t].pool()

    others = (1, 5, 12, 24, 100, 256)
    for t in others:
        _launch(op, weights, *_inputs(t, seed=100 + t))
    fillers = [
        torch.full((t * TOP_K, N), 7.0, dtype=torch.float32, device=device)
        for t in others
    ]

    for rep, t in enumerate((12, 24, 12, 24)):
        x, topk_ids, out = static[t]
        x_new, ids_new, expected = _inputs(t, seed=1000 + rep)
        _launch(op, weights, x_new, ids_new, expected)
        x.copy_(x_new)
        topk_ids.copy_(ids_new)
        out.fill_(float("nan"))
        graphs[t].replay()
        torch.accelerator.synchronize()
        assert torch.isfinite(out).all(), f"replay {rep} (T={t}) left non-finite rows"
        torch.testing.assert_close(out, expected, atol=2e-2, rtol=5e-2)
    for t, filler in zip(others, fillers):
        assert torch.all(filler == 7.0), f"a replay wrote into a live T={t} tensor"
