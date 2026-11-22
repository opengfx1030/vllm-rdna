#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT build of ``csrc/rocm/moe_lowp_dot2_rdna2.cu`` as a stand-alone extension.

Used by the kernel test (``I2_MOE_JIT=<path to a .cu>`` runs the test on that file
instead of the installed ``_rocm_C``: that is how mutants are run), by
``moe_rows_bench.py --jit`` and, with several architectures, as the multi-arch compile
check:

    python tools/rdna/inkling2/moe_jit.py \
        --archs "gfx1030;gfx1100;gfx90a;gfx942;gfx950;gfx1200"

The extension exposes the two ops with the signature of ``vllm._custom_ops``.
``I2_JIT_DIR`` (default ``~/.cache/vllm_inkling2_jit``) holds the build trees.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil

_WRAPPER = """
#include <torch/extension.h>

void moe_nvfp4_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                          torch::Tensor b_q_weight, torch::Tensor b_scales,
                          torch::Tensor b_globals, torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t top_k,
                          int64_t block_size_m, bool mul_topk_weight,
                          int64_t output_topk);
void moe_mxfp8_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                          torch::Tensor b_q_weight, torch::Tensor b_scales,
                          torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t top_k,
                          int64_t block_size_m, bool mul_topk_weight,
                          int64_t output_topk);
void moe_nvfp4_gemm_swiglu_rdna2(
    torch::Tensor gate_up, torch::Tensor c, torch::Tensor b_q_weight,
    torch::Tensor b_scales, torch::Tensor b_globals,
    torch::Tensor topk_weights, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded,
    int64_t top_k, int64_t block_size_m, bool mul_topk_weight,
    int64_t output_topk, double gate_cap, double up_cap);
void moe_mxfp8_gemm_swiglu_rdna2(
    torch::Tensor gate_up, torch::Tensor c, torch::Tensor b_q_weight,
    torch::Tensor b_scales, torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded, int64_t top_k, int64_t block_size_m,
    bool mul_topk_weight, int64_t output_topk, double gate_cap,
    double up_cap);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_nvfp4_gemm_rdna2", &moe_nvfp4_gemm_rdna2);
  m.def("moe_mxfp8_gemm_rdna2", &moe_mxfp8_gemm_rdna2);
  m.def("moe_nvfp4_gemm_swiglu_rdna2", &moe_nvfp4_gemm_swiglu_rdna2);
  m.def("moe_mxfp8_gemm_swiglu_rdna2", &moe_mxfp8_gemm_swiglu_rdna2);
}
"""

REPO = pathlib.Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc" / "rocm" / "moe_lowp_dot2_rdna2.cu"


def build(
    name: str,
    source: str | os.PathLike | None = None,
    archs: str = "gfx1030",
    verbose: bool = False,
    resource_usage: bool = False,
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
    cu = work / "src" / "moe_lowp_dot2_rdna2.cu"
    shutil.copyfile(source or SOURCE, cu)
    wrapper = work / "src" / "wrapper.cpp"
    wrapper.write_text(_WRAPPER)
    os.environ["PYTORCH_ROCM_ARCH"] = archs
    flags = ["-O2"]
    if resource_usage:
        flags.append("-Rpass-analysis=kernel-resource-usage")
    return load(
        name=name,
        sources=[str(wrapper), str(cu)],
        extra_include_paths=[str(REPO / "csrc" / "rocm")],
        extra_cuda_cflags=flags,
        build_directory=str(work / "build"),
        verbose=verbose,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="gfx1030")
    ap.add_argument("--source", default="")
    ap.add_argument("--name", default="i2_moe_jit")
    ap.add_argument("--resource-usage", action="store_true")
    args = ap.parse_args()
    build(
        args.name,
        args.source or None,
        args.archs,
        verbose=True,
        resource_usage=args.resource_usage,
    )
    print("BUILD OK", args.archs)


if __name__ == "__main__":
    main()
