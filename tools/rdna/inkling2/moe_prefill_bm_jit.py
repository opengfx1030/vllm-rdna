#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT build of ``csrc/rocm/moe_prefill_bm_rdna2.cu`` as a stand-alone extension.

Used by the kernel test (``I2_MOE_PREFILL_JIT=<path to a .cu>`` runs the test on that
file instead of the installed ``_rocm_C``: that is how the mutants are run, see
``moe_prefill_bm_mutants.py``) and, with several architectures, as the multi-arch
compile check:

    python tools/rdna/inkling2/moe_prefill_bm_jit.py \
        --archs "gfx1030;gfx1100;gfx90a;gfx942;gfx950;gfx1200"

The extension exposes the three ops with the signature of ``vllm._custom_ops``.
``I2_JIT_DIR`` (default ``~/.cache/vllm_inkling2_jit``) holds the build trees.
"""

from __future__ import annotations

import argparse
import os
import pathlib

_WRAPPER = """
#include <torch/extension.h>

torch::Tensor moe_prefill_cta_rdna2(torch::Tensor expert_ids,
                                    torch::Tensor num_tokens_post_padded,
                                    int64_t num_experts, int64_t block_size_m);
void moe_nvfp4_bm_rdna2(torch::Tensor a, torch::Tensor c,
                        torch::Tensor b_q_weight, torch::Tensor b_scales,
                        torch::Tensor b_globals, torch::Tensor topk_weights,
                        torch::Tensor sorted_token_ids, torch::Tensor cta_info,
                        int64_t top_k, bool mul_topk_weight,
                        int64_t output_topk);
void moe_mxfp8_bm_rdna2(torch::Tensor a, torch::Tensor c,
                        torch::Tensor b_q_weight, torch::Tensor b_scales,
                        torch::Tensor topk_weights,
                        torch::Tensor sorted_token_ids, torch::Tensor cta_info,
                        int64_t top_k, bool mul_topk_weight,
                        int64_t output_topk);
std::tuple<torch::Tensor, torch::Tensor> moe_prefill_act_quant_rdna2(torch::Tensor a);
void moe_nvfp4_bm_i8_rdna2(torch::Tensor a8, torch::Tensor sa, torch::Tensor c,
                           torch::Tensor b_q_weight, torch::Tensor b_scales,
                           torch::Tensor b_globals, torch::Tensor topk_weights,
                           torch::Tensor sorted_token_ids, torch::Tensor cta_info,
                           int64_t top_k, bool mul_topk_weight, int64_t output_topk);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_prefill_cta_rdna2", &moe_prefill_cta_rdna2);
  m.def("moe_nvfp4_bm_rdna2", &moe_nvfp4_bm_rdna2);
  m.def("moe_mxfp8_bm_rdna2", &moe_mxfp8_bm_rdna2);
  m.def("moe_prefill_act_quant_rdna2", &moe_prefill_act_quant_rdna2);
  m.def("moe_nvfp4_bm_i8_rdna2", &moe_nvfp4_bm_i8_rdna2);
}
"""

REPO = pathlib.Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc" / "rocm" / "moe_prefill_bm_rdna2.cu"


def _sync(data: bytes, dst: pathlib.Path) -> None:
    """Write ``dst`` only when its content changes (an untouched file is kept)."""
    if not dst.exists() or dst.read_bytes() != data:
        dst.write_bytes(data)


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
    cu = work / "src" / "moe_prefill_bm_rdna2.cu"
    _sync(pathlib.Path(source or SOURCE).read_bytes(), cu)  # keep the mtime: ninja
    wrapper = work / "src" / "wrapper.cpp"
    _sync(_WRAPPER.encode(), wrapper)
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
    ap.add_argument("--name", default="i2_moe_prefill_jit")
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
