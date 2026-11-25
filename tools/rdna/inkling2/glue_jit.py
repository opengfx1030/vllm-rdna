#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT build of ``csrc/rocm/inkling2_glue_rdna2.cu`` as a stand-alone extension.

Used by the glue kernel test (``I2_GLUE_JIT=<path to a .cu>`` runs the gate / router
tests on that file instead of the installed ``_rocm_C``: that is how mutants of the
decode-latency kernels are run) and as a compile check:

    python tools/rdna/inkling2/glue_jit.py --archs "gfx1030;gfx1100;gfx90a;gfx942"

The extension exposes the ops the decode-latency variants change (``msg_gate_down``,
``msg_gate_mix``, ``moe_route``) with the positional signature of ``torch.ops._rocm_C``,
plus ``swiglu``.  ``VLLM_INKLING2_FUSE_GLUE`` is read by the kernel host code of the
build.  ``I2_JIT_DIR`` (default ``~/.cache/vllm_inkling2_jit``) holds the build trees.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil

_WRAPPER = """
#include <torch/extension.h>
#include <optional>

void inkling2_glue_attn_res(const at::Tensor& blocks, int64_t n_blocks,
                            const at::Tensor& x0,
                            const std::optional<at::Tensor>& x1,
                            const std::optional<at::Tensor>& partial_out,
                            const std::optional<at::Tensor>& append,
                            const at::Tensor& query,
                            const std::optional<at::Tensor>& norm_w,
                            const at::Tensor& out, double eps,
                            double norm_eps);
void inkling2_glue_msg_gate_down(const at::Tensor& h, const at::Tensor& m,
                                 const at::Tensor& wd, const at::Tensor& d,
                                 double eps);
void inkling2_glue_msg_gate_mix(const at::Tensor& h, const at::Tensor& m,
                                const at::Tensor& d, const at::Tensor& wf,
                                const at::Tensor& wg,
                                const std::optional<at::Tensor>& norm_w,
                                const at::Tensor& out,
                                const std::optional<at::Tensor>& scratch,
                                const std::optional<at::Tensor>& partials,
                                const std::optional<at::Tensor>& counters,
                                double norm_eps);
void inkling2_glue_moe_route(
    const at::Tensor& x, const at::Tensor& gate_w, const at::Tensor& bias,
    const at::Tensor& logits, const at::Tensor& topk_weights,
    const at::Tensor& topk_ids, const at::Tensor& sorted_ids,
    const at::Tensor& expert_ids, const at::Tensor& post_pad,
    const at::Tensor& counters, const std::optional<at::Tensor>& zero1,
    const std::optional<at::Tensor>& zero2, int64_t block_size_m,
    double route_scale);
void inkling2_glue_swiglu(const at::Tensor& gate_up, const at::Tensor& out,
                          double gate_cap, double up_cap);
void inkling2_glue_loopcd_blend(
    const torch::Tensor& x0, const std::optional<torch::Tensor>& x1,
    const torch::Tensor& prev_x0, const torch::Tensor& prev_x1,
    const std::optional<torch::Tensor>& blk,
    const std::optional<torch::Tensor>& prev_blk, const torch::Tensor& omega,
    const std::optional<torch::Tensor>& omega_blk);
void inkling2_glue_loopcd_readout(
    const torch::Tensor& blocks, int64_t n_blocks, const torch::Tensor& x0,
    const std::optional<torch::Tensor>& x1, const torch::Tensor& query,
    const torch::Tensor& norm_w, const torch::Tensor& out,
    const std::optional<torch::Tensor>& gate_flag, double eps,
    double norm_eps);
void inkling2_loopcd_prepare_hidden(
    const torch::Tensor& omega_out, const torch::Tensor& omega_blk_out,
    const torch::Tensor& idx_mapping, const torch::Tensor& query_start_loc,
    const torch::Tensor& seq_lens, const torch::Tensor& prefill_len,
    const torch::Tensor& req_alpha, const torch::Tensor& req_mode,
    const torch::Tensor& req_blend_blk, int64_t num_tokens_padded);
void inkling2_loopcd_logits(const torch::Tensor& z_final,
                            const torch::Tensor& z_first,
                            const torch::Tensor& omega,
                            const torch::Tensor& log_beta,
                            const torch::Tensor& adaptive,
                            const torch::Tensor& stats_scratch);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_res", &inkling2_glue_attn_res);
  m.def("msg_gate_down", &inkling2_glue_msg_gate_down);
  m.def("msg_gate_mix", &inkling2_glue_msg_gate_mix);
  m.def("moe_route", &inkling2_glue_moe_route);
  m.def("swiglu", &inkling2_glue_swiglu);
  m.def("loopcd_blend", &inkling2_glue_loopcd_blend);
  m.def("loopcd_readout", &inkling2_glue_loopcd_readout);
  m.def("loopcd_prepare_hidden", &inkling2_loopcd_prepare_hidden);
  m.def("loopcd_logits", &inkling2_loopcd_logits);
}
"""

REPO = pathlib.Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc" / "rocm" / "inkling2_glue_rdna2.cu"


def build(
    name: str,
    source: str | os.PathLike | None = None,
    archs: str = "gfx1030",
    verbose: bool = False,
):
    """Compile ``source`` (default: the repo's file) and return the extension module."""
    from torch.utils.cpp_extension import load

    work = (
        pathlib.Path(
            os.environ.get("I2_JIT_DIR", "~/.cache/vllm_inkling2_jit")
        ).expanduser()
        / name
    )
    (work / "src").mkdir(parents=True, exist_ok=True)
    (work / "build").mkdir(parents=True, exist_ok=True)
    cu = work / "src" / "inkling2_glue_rdna2.cu"
    shutil.copyfile(source or SOURCE, cu)
    wrapper = work / "src" / "wrapper.cpp"
    wrapper.write_text(_WRAPPER)
    os.environ["PYTORCH_ROCM_ARCH"] = archs
    return load(
        name=name,
        sources=[str(wrapper), str(cu)],
        extra_include_paths=[str(REPO / "csrc" / "rocm")],
        extra_cuda_cflags=["-O2"],
        build_directory=str(work / "build"),
        verbose=verbose,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="gfx1030")
    ap.add_argument("--source", default="")
    ap.add_argument("--name", default="i2_glue_jit")
    args = ap.parse_args()
    build(args.name, args.source or None, args.archs, verbose=True)
    print("BUILD OK", args.archs)


if __name__ == "__main__":
    main()
