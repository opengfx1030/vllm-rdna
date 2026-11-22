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
plus ``swiglu``.  ``VLLM_INKLING2_FUSE_GLUE`` is read by the kernel host code of the build.
``I2_JIT_DIR`` (default ``~/.cache/vllm_inkling2_jit``) holds the build trees.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil

_WRAPPER = """
#include <torch/extension.h>
#include <optional>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("msg_gate_down", &inkling2_glue_msg_gate_down);
  m.def("msg_gate_mix", &inkling2_glue_msg_gate_mix);
  m.def("moe_route", &inkling2_glue_moe_route);
  m.def("swiglu", &inkling2_glue_swiglu);
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
