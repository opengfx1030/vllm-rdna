#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone JIT microbenchmark, bit-exact parity & VRAM accounting harness for
Inkling-2-Mini TP=1 DSpark optimizations on RDNA2 (gfx1030).

JIT-compiles ``csrc/rocm/inkling2_dspark_rdna2.cu`` (and its dense-GEMM companion
``csrc/rocm/rdna_hint_flex_dense_gemm_rdna2.cu``) via
``torch.utils.cpp_extension.load`` with ``--offload-arch=gfx1030 -O3``, then verifies:

1. **Bit-exact parity**:
   - ``inkling2_dspark_gemv`` (rows = 1, 2, 4, 8, 16, 32; fp16 and fp32 split-K) and
     ``inkling2_dspark_gemm`` (rows = 48, 64; designs 5 and 0) reading a 32,768-id
     subset in place from ``[200058, 2304]`` via ``w_row_ids`` vs materializing
     ``w[w_row_ids]``.
   - ``inkling2_dspark_argmax`` writing strided 2D columns ``out[:B, s]`` with fused
     ``token_map`` vs ``argmax + index_select + copy_`` at B = 1, 2, 4, 8, 16 across
     all 7 Markov steps (plus adversarial NaN / -inf / tie cases and
     ``inkling2_dspark_argmax_partial``).
2. **Latency microbenchmarks (us, HIP events + HIP graph replay)**:
   - Zero-copy ``w_row_ids`` gather GEMV/GEMM on ``[200058, 2304]`` vs contiguous
     ``[32768, 2304]`` ``ranked.weight`` (measuring L2/TLB impact).
   - 7-step fused strided ``argmax + token_map`` (7 launches) vs 3-op chain
     ``argmax + index_select + copy_`` (21 launches), eager and HIP-graphed.
   - Full Markov head pass at b=2: ``steps_for_reqs(2) = 2`` vs legacy 7 steps.
3. **Exact VRAM accounting (MiB)** for M1a, M1b, M1c, M1d on Inkling-2-Mini TP=1.

Usage on par1-cs25 (single V620, e.g. HIP 1):
    HIP_VISIBLE_DEVICES=1 python tools/rdna/inkling2/bench_tp1_dspark_micro.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import statistics
from typing import Any

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
CSRC_ROCM = REPO / "csrc" / "rocm"

VOCAB_FULL = 200_058
VOCAB_PAD_FULL = 200_064
DRAFT_VOCAB = 32_768
HIDDEN = 2_304
RANK = 256
STEPS = 7
MAX_REQS_TP1 = 16
OLD_HEAD_REQS = 32

_WRAPPER = r"""
#include <torch/extension.h>
#include <optional>

void inkling2_dspark_gemv_v2(
    torch::Tensor x, torch::Tensor w, torch::Tensor out,
    std::optional<torch::Tensor> x_index, int64_t num_rows, int64_t splits,
    int64_t variant, std::optional<torch::Tensor> w_row_ids);

void inkling2_dspark_gemm_v2(
    torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t num_rows,
    int64_t design, std::optional<torch::Tensor> w_row_ids);

void inkling2_dspark_argmax_v2(
    torch::Tensor logits, torch::Tensor out, int64_t num_rows, int64_t vocab,
    std::optional<torch::Tensor> token_map);

void inkling2_dspark_argmax_partial_v2(
    torch::Tensor logits, torch::Tensor part, int64_t num_rows, int64_t vocab,
    int64_t first_id, std::optional<torch::Tensor> token_map);

void inkling2_dspark_markov_step(
    torch::Tensor prev, int64_t prev_stride, torch::Tensor state,
    torch::Tensor w1, torch::Tensor wt, torch::Tensor bias,
    torch::Tensor ctx_h, torch::Tensor ctx_f, torch::Tensor latent,
    int64_t num_reqs, int64_t step, int64_t nsteps, bool first);

void inkling2_dspark_head_logits(
    torch::Tensor base, torch::Tensor latent, torch::Tensor fold_w,
    torch::Tensor fold_b, torch::Tensor logits, int64_t num_reqs,
    int64_t step, int64_t nsteps, double softcap);

void inkling2_dspark_fold_bias(
    torch::Tensor head, torch::Tensor b, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv", &inkling2_dspark_gemv_v2);
  m.def("gemm", &inkling2_dspark_gemm_v2);
  m.def("argmax", &inkling2_dspark_argmax_v2);
  m.def("argmax_partial", &inkling2_dspark_argmax_partial_v2);
  m.def("markov_step", &inkling2_dspark_markov_step);
  m.def("head_logits", &inkling2_dspark_head_logits);
  m.def("fold_bias", &inkling2_dspark_fold_bias);
}
"""


def build_jit(archs: str = "gfx1030", opt: str = "-O3", verbose: bool = False):
    """JIT-compile DSpark + dense-GEMM RDNA2 kernels via cpp_extension.load."""
    from torch.utils.cpp_extension import load

    cu_files = (
        "inkling2_dspark_rdna2.cu",
        "rdna_hint_flex_dense_gemm_rdna2.cu",
    )
    cuh_files = (
        "rdna_hint_flex_dense_gemm_kernel.cuh",
        "rdna_hint_flex_dense_gemm_rdna2.cuh",
        "rdna_hint_flex_dense_gemm_table.cuh",
    )
    h = hashlib.md5()
    h.update(_WRAPPER.encode())
    h.update(f"{archs}|{opt}".encode())
    for fname in (*cu_files, *cuh_files):
        h.update((CSRC_ROCM / fname).read_bytes())
    tag = h.hexdigest()[:12]
    name = f"i2_dspark_micro_{tag}"
    work = (
        pathlib.Path(
            os.environ.get("I2_JIT_DIR", "~/.cache/vllm_inkling2_jit")
        ).expanduser()
        / name
    )
    src_dir = work / "src"
    bld_dir = work / "build"
    src_dir.mkdir(parents=True, exist_ok=True)
    bld_dir.mkdir(parents=True, exist_ok=True)

    for fname in (*cu_files, *cuh_files):
        dst = src_dir / fname
        src_bytes = (CSRC_ROCM / fname).read_bytes()
        if not dst.exists() or dst.read_bytes() != src_bytes:
            shutil.copyfile(CSRC_ROCM / fname, dst)
    wrapper = src_dir / "wrapper.cpp"
    if not wrapper.exists() or wrapper.read_text() != _WRAPPER:
        wrapper.write_text(_WRAPPER)

    os.environ["PYTORCH_ROCM_ARCH"] = archs
    arch_flags = [f"--offload-arch={a}" for a in archs.split(";") if a]
    return load(
        name=name,
        sources=[str(wrapper), *(str(src_dir / f) for f in cu_files)],
        extra_include_paths=[str(src_dir), str(CSRC_ROCM)],
        extra_cuda_cflags=[opt, *arch_flags],
        build_directory=str(bld_dir),
        verbose=verbose,
    )


def _mib(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def vram_accounting() -> dict[str, Any]:
    """Exact persistent VRAM delta (MiB) on Inkling-2-Mini TP=1 + 32k draft_vocab."""
    # M1a: ranked.weight [32768, 2304] fp16 + ranked.ids_i32 [32768] int32
    #      vs ranked.local_idx [32768] int32 (zero fp16 copy).
    old_m1a = DRAFT_VOCAB * HIDDEN * 2 + DRAFT_VOCAB * 4
    new_m1a = DRAFT_VOCAB * 4
    save_m1a = old_m1a - new_m1a

    # M1b: full head.fold_w [200064, 256] fp16 + head.fold_b [200064] fp32
    #      + ranked.fold_w [32768, 256] fp16 + ranked.fold_b [32768] fp32
    #      vs subset-only ranked.fold_w + ranked.fold_b under greedy sampling.
    old_m1b = (
        VOCAB_PAD_FULL * RANK * 2
        + VOCAB_PAD_FULL * 4
        + DRAFT_VOCAB * RANK * 2
        + DRAFT_VOCAB * 4
    )
    new_m1b = DRAFT_VOCAB * RANK * 2 + DRAFT_VOCAB * 4
    save_m1b = old_m1b - new_m1b

    # M1c: _Scratch.base + _Scratch.logits (active vocab + SEQS=16 head_reqs cap).
    old_m1c = (OLD_HEAD_REQS * STEPS * VOCAB_PAD_FULL * 2) + (
        OLD_HEAD_REQS * VOCAB_PAD_FULL * 2
    )
    new_m1c = (MAX_REQS_TP1 * STEPS * DRAFT_VOCAB * 2) + (
        MAX_REQS_TP1 * DRAFT_VOCAB * 2
    )
    vocab_only_save_m1c = old_m1c - (
        (OLD_HEAD_REQS * STEPS * DRAFT_VOCAB * 2) + (OLD_HEAD_REQS * DRAFT_VOCAB * 2)
    )
    save_m1c = old_m1c - new_m1c

    # M1d: freed checkpoint parameters after derive() + attach_target():
    #      rnn_proj_w [768, 5120] fp16 + hidden_out_w [2304, 256] fp16
    #      + hidden_out_t [256, 2304] fp16 + hidden_out_b [2304] fp16
    old_m1d = (
        (3 * RANK) * (2 * RANK + 2 * HIDDEN) * 2
        + (HIDDEN * RANK * 2)
        + (RANK * HIDDEN * 2)
        + (HIDDEN * 2)
    )
    new_m1d = 0
    save_m1d = old_m1d - new_m1d

    total_old = old_m1a + old_m1b + old_m1c + old_m1d
    total_new = new_m1a + new_m1b + new_m1c + new_m1d
    total_save = total_old - total_new

    report = {
        "M1a_lm_head_zero_copy_mib": round(_mib(save_m1a), 3),
        "M1b_subset_fold_wb_mib": round(_mib(save_m1b), 3),
        "M1c_scratch_vocab_and_reqs_mib": round(_mib(save_m1c), 3),
        "M1c_vocab_width_only_mib": round(_mib(vocab_only_save_m1c), 3),
        "M1d_dead_params_freed_mib": round(_mib(save_m1d), 3),
        "total_before_mib": round(_mib(total_old), 3),
        "total_after_mib": round(_mib(total_new), 3),
        "total_reclaimed_mib": round(_mib(total_save), 3),
        "total_reclaimed_mb_decimal": round(total_save / 1e6, 3),
    }

    print("\n=== Inkling-2-Mini TP=1 (SEQS=16, 32k draft_vocab) VRAM Accounting ===")
    print(
        f"  M1a (zero-copy lm_head w_row_ids): {_mib(old_m1a):7.2f} -> "
        f"{_mib(new_m1a):6.2f} MiB "
        f"(-{_mib(save_m1a):.2f} MiB / -{save_m1a / 1e6:.1f} MB)"
    )
    print(
        f"  M1b (subset-only fold_w/fold_b):   {_mib(old_m1b):7.2f} -> "
        f"{_mib(new_m1b):6.2f} MiB "
        f"(-{_mib(save_m1b):.2f} MiB / -{save_m1b / 1e6:.1f} MB)"
    )
    print(
        f"  M1c (_Scratch.base + logits):      {_mib(old_m1c):7.2f} -> "
        f"{_mib(new_m1c):6.2f} MiB (-{_mib(save_m1c):.2f} MiB; "
        f"vocab-only -{_mib(vocab_only_save_m1c):.2f} MiB)"
    )
    print(
        f"  M1d (freed rnn_proj_w/hidden_out): {_mib(old_m1d):7.2f} -> "
        f"{_mib(new_m1d):6.2f} MiB "
        f"(-{_mib(save_m1d):.2f} MiB / -{save_m1d / 1e6:.1f} MB)"
    )
    print("  " + "-" * 72)
    print(
        f"  Total DSpark head/scratch buffers: {_mib(total_old):7.2f} -> "
        f"{_mib(total_new):6.2f} MiB (RECLAIMED -{_mib(total_save):.2f} MiB / "
        f"-{total_save / 1e6:.1f} MB)"
    )
    return report


def _time_us(
    fn, *, warmup: int = 15, iters: int = 80, graph: bool = False
) -> tuple[float, float]:
    """Return (median_us, iqr_us) of ``fn()`` eagerly or inside a HIP graph."""
    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
        run = g.replay
    else:
        run = fn

    for _ in range(warmup):
        run()
    torch.accelerator.synchronize()

    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        run()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1e3)
    samples.sort()
    med = statistics.median(samples)
    iqr = samples[(3 * len(samples)) // 4] - samples[len(samples) // 4]
    return med, iqr


@torch.no_grad()
def verify_and_bench_gpu(ext, iters: int = 80) -> dict[str, Any]:
    device = torch.device("cuda")
    rng = np.random.default_rng(42)
    ids_np = np.sort(rng.choice(VOCAB_FULL, size=DRAFT_VOCAB, replace=False)).astype(
        np.int64
    )
    w_row_ids_i32 = torch.from_numpy(ids_np.astype(np.int32)).to(device)
    w_row_ids_i64 = torch.from_numpy(ids_np).to(device)

    gen = torch.Generator(device="cuda").manual_seed(1030)
    w_full = (
        torch.randn(
            VOCAB_FULL, HIDDEN, device=device, generator=gen, dtype=torch.float32
        )
        * 0.05
    ).to(torch.float16)
    w_sub = w_full.index_select(0, w_row_ids_i64).contiguous()

    # 1. Bit-exact parity: gemv (rows <= 32) & gemm (rows > 32)
    print("\n=== 1. Bit-Exact Parity: gemv & gemm w_row_ids vs w[w_row_ids] ===")
    test_rows = (1, 2, 4, 8, 16, 32, 48, 64)
    x_max = (
        torch.randn(
            max(test_rows), HIDDEN, device=device, generator=gen, dtype=torch.float32
        )
        * 0.5
    ).to(torch.float16)
    out_idx = torch.empty(
        max(test_rows), DRAFT_VOCAB, dtype=torch.float16, device=device
    )
    out_mat = torch.empty(
        max(test_rows), DRAFT_VOCAB, dtype=torch.float16, device=device
    )

    for rows in test_rows:
        out_idx.fill_(float("nan"))
        out_mat.fill_(float("nan"))
        if rows <= 32:
            for ids_t, tag in ((w_row_ids_i32, "i32"), (w_row_ids_i64, "i64")):
                ext.gemv(x_max, w_full, out_idx, None, rows, 0, 0, ids_t)
                ext.gemv(x_max, w_sub, out_mat, None, rows, 0, 0, None)
                assert torch.equal(out_idx[:rows], out_mat[:rows]), (
                    f"GEMV fp16 mismatch at rows={rows} ({tag})"
                )
            part_idx = torch.empty(
                4, rows, DRAFT_VOCAB, dtype=torch.float32, device=device
            )
            part_mat = torch.empty(
                4, rows, DRAFT_VOCAB, dtype=torch.float32, device=device
            )
            ext.gemv(x_max, w_full, part_idx, None, rows, 4, 0, w_row_ids_i32)
            ext.gemv(x_max, w_sub, part_mat, None, rows, 4, 0, None)
            assert torch.equal(part_idx, part_mat), (
                f"GEMV split-K fp32 mismatch at rows={rows}"
            )
            print(
                f"  [PASS] rows={rows:2d} (gemv fp16 i32/i64 + split-K=4 fp32): "
                "bit-exact (0 ULP)"
            )
        else:
            for design in (5, 0):
                for ids_t, tag in ((w_row_ids_i32, "i32"), (w_row_ids_i64, "i64")):
                    ext.gemm(x_max, w_full, out_idx, rows, design, ids_t)
                    ext.gemm(x_max, w_sub, out_mat, rows, design, None)
                    assert torch.equal(out_idx[:rows], out_mat[:rows]), (
                        f"GEMM design={design} mismatch at rows={rows} ({tag})"
                    )
            print(
                f"  [PASS] rows={rows:2d} (gemm design=5 & design=0, i32/i64):  "
                "bit-exact (0 ULP)"
            )

    # 2. Bit-exact parity: strided argmax + token_map vs 3-op chain
    print("\n=== 2. Bit-Exact Parity: Fused Strided argmax + token_map ===")
    for b in (1, 2, 4, 8, 16):
        logits_steps = (
            torch.randn(
                STEPS, b, DRAFT_VOCAB, device=device, generator=gen, dtype=torch.float32
            )
            * 4.0
        ).to(torch.float16)
        logits_steps[0, 0, [5, 1024, 32760]] = 25.0
        if b >= 2:
            logits_steps[1, 1, :] = float("-inf")
            logits_steps[2, 1, :] = float("nan")
            logits_steps[2, 1, 777] = -2.5

        table_fused = torch.full(
            (MAX_REQS_TP1, STEPS), -99, dtype=torch.int64, device=device
        )
        table_ref = torch.full(
            (MAX_REQS_TP1, STEPS), -99, dtype=torch.int64, device=device
        )
        scratch_tok = torch.empty(MAX_REQS_TP1, dtype=torch.int64, device=device)
        scratch_tok2 = torch.empty(MAX_REQS_TP1, dtype=torch.int64, device=device)

        for s in range(STEPS):
            ext.argmax(
                logits_steps[s], table_fused[:b, s], b, DRAFT_VOCAB, w_row_ids_i64
            )
            ext.argmax(logits_steps[s], scratch_tok[:b], b, DRAFT_VOCAB, None)
            torch.index_select(w_row_ids_i64, 0, scratch_tok[:b], out=scratch_tok2[:b])
            table_ref[:b, s].copy_(scratch_tok2[:b])

        assert torch.equal(table_fused, table_ref), f"Fused argmax mismatch at B={b}"
        part_fused = torch.empty(2 * b, dtype=torch.float32, device=device)
        part_ref = torch.empty(2 * b, dtype=torch.float32, device=device)
        ext.argmax_partial(
            logits_steps[0], part_fused, b, DRAFT_VOCAB, 0, w_row_ids_i64
        )
        ext.argmax_partial(logits_steps[0], part_ref, b, DRAFT_VOCAB, 0, None)
        ref_i32 = w_row_ids_i32.index_select(
            0, part_ref[b : 2 * b].view(torch.int32).long()
        )
        part_ref[b : 2 * b].view(torch.int32).copy_(ref_i32)
        assert torch.equal(part_fused[:b], part_ref[:b])
        assert torch.equal(
            part_fused[b : 2 * b].view(torch.int32),
            part_ref[b : 2 * b].view(torch.int32),
        )
        print(
            f"  [PASS] B={b:2d} (all {STEPS} strided cols + ties/NaN/-inf + TP): "
            "bit-exact"
        )

    # 3. Latency Microbenchmarks
    print("\n=== 3. Latency: Zero-Copy w_row_ids Gather vs Contiguous Weight ===")
    hdr = (
        f"  {'rows':>4s}  {'kernel':>8s}  {'contig_e_us':>11s}  "
        f"{'index_e_us':>11s}  {'contig_g_us':>11s}  {'index_g_us':>11s}  "
        f"{'delta_us':>9s}"
    )
    print(hdr)
    gemm_bench: dict[str, Any] = {}
    bench_rows = (1, 2, 4, 7, 8, 14, 16, 32, 48, 64)
    x_bench = (
        torch.randn(
            max(bench_rows), HIDDEN, device=device, generator=gen, dtype=torch.float32
        )
        * 0.5
    ).to(torch.float16)
    out_bench = torch.empty(
        max(bench_rows), DRAFT_VOCAB, dtype=torch.float16, device=device
    )
    for rows in bench_rows:
        design = 5 if rows <= 64 else 0
        kname = "gemv" if rows <= 32 else f"gemm_d{design}"

        def fn_contig(r: int = rows, d: int = design) -> None:
            if r <= 32:
                ext.gemv(x_bench, w_sub, out_bench, None, r, 0, 0, None)
            else:
                ext.gemm(x_bench, w_sub, out_bench, r, d, None)

        def fn_index(r: int = rows, d: int = design) -> None:
            if r <= 32:
                ext.gemv(x_bench, w_full, out_bench, None, r, 0, 0, w_row_ids_i32)
            else:
                ext.gemm(x_bench, w_full, out_bench, r, d, w_row_ids_i32)

        ce, _ = _time_us(fn_contig, iters=iters, graph=False)
        ie, _ = _time_us(fn_index, iters=iters, graph=False)
        cg, _ = _time_us(fn_contig, iters=iters, graph=True)
        ig, _ = _time_us(fn_index, iters=iters, graph=True)
        gemm_bench[str(rows)] = {
            "kernel": kname,
            "contiguous_eager_us": round(ce, 2),
            "indexed_eager_us": round(ie, 2),
            "contiguous_graph_us": round(cg, 2),
            "indexed_graph_us": round(ig, 2),
            "graph_delta_us": round(ig - cg, 2),
        }
        print(
            f"  {rows:4d}  {kname:>8s}  {ce:11.2f}  {ie:11.2f}  "
            f"{cg:11.2f}  {ig:11.2f}  {ig - cg:+9.2f}"
        )

    print("\n=== 4. Latency: 7 Markov Steps Fused Strided argmax vs 3-Op Chain ===")
    print(
        f"  {'B':>4s}  {'3op_eager_us':>13s}  {'fused_e_us':>12s}  "
        f"{'3op_graph_us':>13s}  {'fused_g_us':>12s}  {'saved_us':>10s}"
    )
    argmax_bench: dict[str, Any] = {}
    logits_b = torch.randn(STEPS, 16, DRAFT_VOCAB, device=device, generator=gen).half()
    tok_out = torch.zeros(MAX_REQS_TP1, STEPS, dtype=torch.int64, device=device)
    s_tok = torch.empty(MAX_REQS_TP1, dtype=torch.int64, device=device)
    s_tok2 = torch.empty(MAX_REQS_TP1, dtype=torch.int64, device=device)

    for b in (1, 2, 4, 8, 16):

        def run_3op(b_cur: int = b) -> None:
            for s in range(STEPS):
                ext.argmax(logits_b[s], s_tok[:b_cur], b_cur, DRAFT_VOCAB, None)
                torch.index_select(w_row_ids_i64, 0, s_tok[:b_cur], out=s_tok2[:b_cur])
                tok_out[:b_cur, s].copy_(s_tok2[:b_cur])

        def run_fused(b_cur: int = b) -> None:
            for s in range(STEPS):
                ext.argmax(
                    logits_b[s],
                    tok_out[:b_cur, s],
                    b_cur,
                    DRAFT_VOCAB,
                    w_row_ids_i64,
                )

        e3, _ = _time_us(run_3op, iters=iters, graph=False)
        ef, _ = _time_us(run_fused, iters=iters, graph=False)
        g3, _ = _time_us(run_3op, iters=iters, graph=True)
        gf, _ = _time_us(run_fused, iters=iters, graph=True)
        argmax_bench[str(b)] = {
            "chain_3op_eager_us": round(e3, 2),
            "fused_1op_eager_us": round(ef, 2),
            "chain_3op_graph_us": round(g3, 2),
            "fused_1op_graph_us": round(gf, 2),
            "graph_saved_us": round(g3 - gf, 2),
        }
        print(
            f"  {b:4d}  {e3:13.2f}  {ef:12.2f}  {g3:13.2f}  {gf:12.2f}  {g3 - gf:10.2f}"
        )

    # 5. End-to-end b=2 Markov head pass: legacy 7 steps vs #6b 2 steps
    w1 = torch.randn(VOCAB_PAD_FULL, RANK, device=device, generator=gen).half()
    wt = torch.randn(2 * RANK, 3 * RANK, device=device, generator=gen).half()
    bias = torch.randn(3 * RANK, device=device, generator=gen)
    ctx_h = torch.randn(16 * STEPS, 3 * RANK, device=device, generator=gen)
    ctx_f = torch.randn(16, 3 * RANK, device=device, generator=gen)
    state = torch.zeros(16, RANK, device=device)
    latent = torch.zeros(16, RANK, device=device, dtype=torch.float16)
    fold_w_sub = torch.randn(DRAFT_VOCAB, RANK, device=device, generator=gen).half()
    fold_b_sub = torch.randn(DRAFT_VOCAB, device=device, generator=gen)
    logits_2d = torch.empty(16, DRAFT_VOCAB, device=device, dtype=torch.float16)
    anchors = torch.zeros(16 * 8, dtype=torch.int32, device=device)

    def head_b2_old() -> None:
        ext.gemv(x_bench, w_sub, out_bench, None, 2 * 7, 0, 0, None)
        for i in range(7):
            prev = anchors if i == 0 else tok_out[:, i - 1]
            stride = 8 if i == 0 else tok_out.stride(0)
            ext.markov_step(
                prev, stride, state, w1, wt, bias, ctx_h, ctx_f, latent, 2, i, 7, i == 0
            )
            ext.head_logits(
                out_bench, latent, fold_w_sub, fold_b_sub, logits_2d, 2, i, 7, 30.0
            )
            ext.argmax(logits_2d, s_tok[:2], 2, DRAFT_VOCAB, None)
            torch.index_select(w_row_ids_i64, 0, s_tok[:2], out=s_tok2[:2])
            tok_out[:2, i].copy_(s_tok2[:2])

    def head_b2_new() -> None:
        ext.gemv(x_bench, w_full, out_bench, None, 2 * 2, 0, 0, w_row_ids_i32)
        for i in range(2):
            prev = anchors if i == 0 else tok_out[:, i - 1]
            stride = 8 if i == 0 else tok_out.stride(0)
            ext.markov_step(
                prev, stride, state, w1, wt, bias, ctx_h, ctx_f, latent, 2, i, 2, i == 0
            )
            ext.head_logits(
                out_bench, latent, fold_w_sub, fold_b_sub, logits_2d, 2, i, 2, 30.0
            )
            ext.argmax(logits_2d, tok_out[:2, i], 2, DRAFT_VOCAB, w_row_ids_i64)

    b2_old_g, _ = _time_us(head_b2_old, iters=iters, graph=True)
    b2_new_g, _ = _time_us(head_b2_new, iters=iters, graph=True)
    ratio = b2_old_g / max(b2_new_g, 1e-9)
    print("\n=== 5. End-to-End Markov Head at b=2 (HIP Graph Replay) ===")
    print(f"  Legacy (K=7, 151MB weight, 3-op argmax): {b2_old_g:8.2f} us")
    print(
        f"  Optimized (K=2 #6b, w_row_ids, 1-op):    {b2_new_g:8.2f} us "
        f"(saved {b2_old_g - b2_new_g:.2f} us / {ratio:.2f}x)"
    )

    return {
        "gemv_gemm_us": gemm_bench,
        "argmax_7steps_us": argmax_bench,
        "b2_markov_head_graph_us": {
            "legacy_7steps_us": round(b2_old_g, 2),
            "optimized_2steps_us": round(b2_new_g, 2),
            "saved_us": round(b2_old_g - b2_new_g, 2),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--archs", default="gfx1030", help="ROCm offload arch(s) for JIT build"
    )
    ap.add_argument("--opt", default="-O3", help="HIP compiler optimization level")
    ap.add_argument(
        "--iters", type=int, default=80, help="Timed iterations per benchmark"
    )
    ap.add_argument(
        "--compile-only", action="store_true", help="Compile JIT extension only"
    )
    ap.add_argument(
        "--cpu-vram-only",
        action="store_true",
        help="Print VRAM accounting without GPU",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose ninja build output")
    ap.add_argument("--json", default="", help="Optional path to write JSON results")
    args = ap.parse_args()

    results: dict[str, Any] = {"vram": vram_accounting()}
    if args.cpu_vram_only:
        if args.json:
            pathlib.Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
        return

    ext = build_jit(archs=args.archs, opt=args.opt, verbose=args.verbose)
    print(f"\nJIT BUILD OK (archs={args.archs}, opt={args.opt})")
    if args.compile_only:
        return

    if not torch.accelerator.is_available():
        raise RuntimeError(
            "No ROCm/CUDA GPU visible; pass --cpu-vram-only on CPU hosts."
        )
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if "gfx1030" not in arch:
        raise RuntimeError(f"Expected gfx1030 device, got {arch!r}")

    results["gpu"] = verify_and_bench_gpu(ext, iters=args.iters)
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
