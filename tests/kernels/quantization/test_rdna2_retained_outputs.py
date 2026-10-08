# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA2 GEMM ops must return fresh output storage.

These ops used to return views of one process-wide persistent buffer, so a
later call overwrote (and zeroed) an output the caller still held. The same
bug in the rdna_ar all-reduce corrupted the layer-0 residual.
"""

import pytest
import torch

from vllm.platforms import current_platform


def _has_op(name: str) -> bool:
    if not current_platform.is_rocm():
        return False
    import vllm._custom_ops  # noqa: F401  (loads _rocm_C)

    schemas = torch._C._jit_get_all_schemas()
    return any(str(s).startswith(f"_rocm_C::{name}(") for s in schemas)


K = 512
N = 512
G = 4  # group size 128


def _weights(seed: int, n: int = N):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (K // 8, n),
        device="cuda",
        dtype=torch.int32,
        generator=g,
    )
    qzeros = torch.randint(
        -(2**31),
        2**31 - 1,
        (G, n // 8),
        device="cuda",
        dtype=torch.int32,
        generator=g,
    )
    scales = (
        torch.rand((G, n), device="cuda", generator=g, dtype=torch.float32) * 0.01
        + 0.001
    ).half()
    g_idx = torch.empty(0, device="cuda", dtype=torch.int32)
    return qweight, qzeros, scales, g_idx


def _call(name: str, x: torch.Tensor, w) -> torch.Tensor:
    op = getattr(torch.ops._rocm_C, name)
    return op(x, *w, False)


CASES = [
    pytest.param("gptq_gemm_rdna2", 1, id="decode-m1"),
    pytest.param("gptq_gemm_rdna2", 8, id="decode-m8"),
    pytest.param("gptq_gemm_rdna2_prefill", 64, id="prefill-m64"),
]


@pytest.mark.parametrize("name,m", CASES)
def test_output_survives_later_call(name: str, m: int):
    if not _has_op(name):
        pytest.skip(f"_rocm_C::{name} not built")
    w = _weights(0)
    x1 = torch.randn((m, K), device="cuda", dtype=torch.float16)
    x2 = torch.randn((m, K), device="cuda", dtype=torch.float16)

    first = _call(name, x1, w)
    expected = first.clone()
    second = _call(name, x2, w)
    torch.cuda.synchronize()

    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, expected, atol=0, rtol=0)


@pytest.mark.parametrize("name,m", CASES)
def test_output_fed_back_as_input(name: str, m: int):
    """An output reused as the next call's input (e.g. after an in-place norm
    in eager mode) must not be zeroed before the kernel reads it."""
    if not _has_op(name):
        pytest.skip(f"_rocm_C::{name} not built")
    w1, w2 = _weights(1), _weights(2)
    x = torch.randn((m, K), device="cuda", dtype=torch.float16)

    h = _call(name, x, w1)
    h.mul_(0.01)  # stand-in for an in-place op on the output
    out = _call(name, h, w2)
    ref = _call(name, h.clone(), w2)
    torch.cuda.synchronize()

    assert h.abs().sum() > 0
    # Separate calls: the kernel accumulates with fp16 atomics, so allow
    # order-dependent rounding.
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("m", [1, 4])
def test_decode_outputs_survive_graph_replay(m: int):
    name = "gptq_gemm_rdna2"
    if not _has_op(name):
        pytest.skip(f"_rocm_C::{name} not built")
    w = _weights(3)
    zero_w = (w[0], w[1], torch.zeros_like(w[2]), w[3])
    x = torch.randn((m, K), device="cuda", dtype=torch.float16)

    def projections():
        return _call(name, x, w), _call(name, x, zero_w)

    projections()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        first, second = projections()
    for scale in (1.0, 2.0, -0.5):
        x.copy_(torch.randn_like(x) * scale)
        graph.replay()
        expected = _call(name, x.clone(), w)
        torch.cuda.synchronize()
        torch.testing.assert_close(first, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(second, torch.zeros_like(second), atol=0, rtol=0)
