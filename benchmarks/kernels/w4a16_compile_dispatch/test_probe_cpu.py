# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the probe's parsers (no GPU, no vLLM engine).

.venv/bin/python -m pytest benchmarks/kernels/w4a16_compile_dispatch -q
"""

import pytest

from benchmarks.kernels.w4a16_compile_dispatch import probe

GRAPH = """
# File: rdna2_w4a16.py:291 in apply_weights, code: output = ops.gptq_gemm_rdna2(
gptq_gemm_rdna2_prefill = torch.ops._rocm_C.gptq_gemm_rdna2_prefill.default(x, w)
gptq_gemm_rdna2_prefill_1 = torch.ops._rocm_C.gptq_gemm_rdna2_prefill.default(y, w)
gptq_gemm_rdna2 = torch.ops._rocm_C.gptq_gemm_rdna2.default(z, w)
gptq_gemm = torch.ops._C.gptq_gemm.default(z, w, True, False, 4)
direct = torch.ops._rocm_C.gptq_gemm_rdna2_prefill_direct.default(z, w)
rdna2_w4a16_gemm = torch.ops.vllm.rdna2_w4a16_gemm.default(z, w, 6144, True, 4)
"""


def test_graph_ops_counts_calls_not_source_comments(tmp_path):
    graph = tmp_path / "torch_compile_cache" / "abc" / "rank_0_0" / "backbone"
    graph.mkdir(parents=True)
    (graph / "computation_graph.py").write_text(GRAPH)
    found = probe.count_graph_ops(tmp_path / "torch_compile_cache")
    assert found == {
        "abc/rank_0_0/backbone/computation_graph.py": {
            "gptq_gemm_rdna2_prefill": 2,
            "gptq_gemm_rdna2": 1,
            "gptq_gemm": 1,
            "rdna2_w4a16_gemm": 1,
        }
    }


def test_kernel_stats_group_by_launching_op(tmp_path):
    stats = tmp_path / "kernel_stats.csv"
    stats.write_text(
        '"Name","Calls","TotalDurationNs","AverageNs","Percentage"\n'
        '"void vllm::gptq_rdna2::gemm_q4_kernel_rdna2<__half, 1>()",10,2000000,1,1\n'
        '"vllm::gptq_rdna2_prefill::gemm_dynamic_kernel<A>()",30,9000000,1,1\n'
        '"vllm::gptq_rdna2_prefill::gemm_dynamic_kernel<B>()",5,1000000,1,1\n'
        '"gemm_half_q_half_gptq_4bit_kernel<true, 1>()",2,500000,1,1\n'
        '"fa_decode_paged_splitk_kernel_256<__half>()",7,700000,1,1\n'
    )
    assert probe.summarize_kernels(stats) == {
        "rdna2_decode": {"calls": 10, "total_ms": 2.0},
        "prefill": {"calls": 35, "total_ms": 10.0},
        "exllama": {"calls": 2, "total_ms": 0.5},
    }


def test_kernel_trace_counts_dispatches(tmp_path):
    trace = tmp_path / "prof_kernel_trace.csv"
    trace.write_text(
        '"Kind","Kernel_Name","Start_Timestamp","End_Timestamp"\n'
        '"KERNEL_DISPATCH","gemm_q4_kernel_rdna2<__half, 1>()",1000000,1072000\n'
        '"KERNEL_DISPATCH","gemm_q4_kernel_rdna2<__half, 1>()",2000000,2067000\n'
        '"KERNEL_DISPATCH","gemm_dynamic_kernel<A>()",3000000,3500000\n'
    )
    got = probe.summarize_kernels(trace)
    assert got["rdna2_decode"] == {"calls": 2, "total_ms": pytest.approx(0.139)}
    assert got["prefill"] == {"calls": 1, "total_ms": pytest.approx(0.5)}
