#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT build of ``csrc/rocm/inkling2_fuse_attn_rdna2.cu`` as a stand-alone extension.

Used by the fused attention-chain test (``I2_FUSE_JIT=<path to a .cu>`` runs the
tests on that file instead of the installed ``_rocm_C``: that is how the hand mutants
are run), by ``bench_fuse_attn.py`` and as a compile check:

    python tools/rdna/inkling2/fuse_attn_jit.py --archs "gfx1030;gfx1100;gfx90a;gfx942"

The module exposes ``attn_res_conv`` and ``msgs_fill`` with the positional signature
of ``torch.ops._rocm_C.inkling2_fuse_attn_res_conv`` / ``inkling2_fuse_msgs_fill``.
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

void inkling2_fuse_attn_res_conv(
    const at::Tensor& blocks, int64_t n_blocks,
    const std::optional<at::Tensor>& x0, const at::Tensor& conv_x,
    const at::Tensor& conv_w, const at::Tensor& conv_cache,
    const at::Tensor& block_table, const at::Tensor& seq_idx,
    const at::Tensor& query_start_loc, const at::Tensor& seq_lens,
    int64_t stream_offset, int64_t num_kv_heads, int64_t block_size,
    const std::optional<at::Tensor>& x1_out,
    const std::optional<at::Tensor>& partial_out,
    const std::optional<at::Tensor>& append, const at::Tensor& query,
    const std::optional<at::Tensor>& norm_w, const at::Tensor& out, double eps,
    double norm_eps);
void inkling2_fuse_msgs_fill(const at::Tensor& blocks, int64_t n_blocks,
                             const at::Tensor& x0, const at::Tensor& x1,
                             const at::Tensor& queries, const at::Tensor& out,
                             double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_res_conv", &inkling2_fuse_attn_res_conv);
  m.def("msgs_fill", &inkling2_fuse_msgs_fill);
}
"""

REPO = pathlib.Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc" / "rocm" / "inkling2_fuse_attn_rdna2.cu"


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
    cu = work / "src" / "inkling2_fuse_attn_rdna2.cu"
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
    ap.add_argument("--name", default="i2_fuse_jit")
    args = ap.parse_args()
    build(args.name, args.source or None, args.archs, verbose=True)
    print("BUILD OK", args.archs)


if __name__ == "__main__":
    main()
