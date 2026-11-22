#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT build of ``csrc/rocm/inkling2_qlinear_rdna2.cu`` as stand-alone extensions.

One translation unit per (bits, group) so that they compile in parallel; the instance
table of the kernel (``inst_at``) is filtered by a C++ expression over ``B`` (bits),
``G`` (chunks per scale group), ``R`` (rows per tile), ``K`` (K quarters), ``M`` (activation
rows), ``C`` (chunks per iteration), ``W`` (waves per block) and ``A`` (argmax partials),
so a build holds only the instances a test or a sweep needs.  Used by
``qlinear_check.py`` (the oracle test; ``qlinear_mutants.py`` runs it on mutated copies of
the kernel) and ``qlinear_bench.py`` (the sweep).  The serving build is the normal
``_rocm_C``.  ``I2_JIT_DIR`` (default ``~/.cache/vllm_inkling2_jit``) holds the build
trees.  The module exposes ``grouped_b{bits}g{group}`` and ``lm_head_b{bits}g{group}``
with the signature of ``vllm._custom_ops.inkling2_qlinear_*``.
"""

from __future__ import annotations

import os
import pathlib

CK = {4: 32, 8: 16}  # K elements per 16-byte chunk
REPO = pathlib.Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc" / "rocm" / "inkling2_qlinear_rdna2.cu"

_WRAP_HEAD = """
#include <torch/extension.h>
#include <optional>
#include <vector>
"""

_DECL = """
void inkling2_qlinear_grouped_{t}(
    const std::vector<torch::Tensor>& xs, torch::Tensor w, torch::Tensor sc,
    const std::vector<torch::Tensor>& outs, const std::vector<int64_t>& seg_x,
    const std::vector<int64_t>& seg_n, const std::vector<int64_t>& seg_act,
    const std::vector<double>& seg_p, const std::vector<int64_t>& seg_valid,
    int64_t bits, int64_t group, int64_t rows_per_tile, int64_t cpi,
    int64_t waves);
void inkling2_qlinear_lm_head_{t}(
    torch::Tensor x, torch::Tensor w, torch::Tensor sc,
    const std::optional<torch::Tensor>& logits,
    const std::optional<torch::Tensor>& part_val,
    const std::optional<torch::Tensor>& part_idx, double softcap,
    int64_t n_rows, int64_t n_valid, int64_t col_base, int64_t bits,
    int64_t group, int64_t cpi);
"""

_DEF = """
  m.def("grouped_{t}", &inkling2_qlinear_grouped_{t});
  m.def("lm_head_{t}", &inkling2_qlinear_lm_head_{t});
"""

_TU = """
#define I2Q_ENABLED(B, G, R, K, M, C, W, A) (({filt}))
#define inkling2_qlinear_grouped inkling2_qlinear_grouped_{t}
#define inkling2_qlinear_lm_head inkling2_qlinear_lm_head_{t}
"""


def build(
    name: str,
    source: str | os.PathLike | None,
    combos: list[tuple[int, int]],
    archs: str = "gfx1030",
    extra_filter: str = "1",
    extra_defs: str = "",
    extra_flags: list[str] | None = None,
    verbose: bool = False,
):
    """Compile ``source`` (default: the repo's kernel) for the (bits, group) ``combos``
    and return the extension module; ``extra_defs`` is C++ text put before the kernel
    (``#define I2Q_NONTEMPORAL``)."""
    from torch.utils.cpp_extension import load

    work = (
        pathlib.Path(
            os.environ.get("I2_JIT_DIR", "~/.cache/vllm_inkling2_jit")
        ).expanduser()
        / name
    )
    (work / "src").mkdir(parents=True, exist_ok=True)
    (work / "build").mkdir(parents=True, exist_ok=True)
    # every TU is a full copy of the kernel with its own filter: torch hipifies the
    # listed sources, not what they include
    body = pathlib.Path(source or SOURCE).read_text()
    wrapper = [_WRAP_HEAD]
    defs, sources = [], []
    for bits, group in combos:
        gc = group // CK[bits]
        tag = f"b{bits}g{group}"
        filt = f"(B) == {bits} && (G) == {gc} && ({extra_filter})"
        tu = work / "src" / f"tu_{tag}.cu"
        tu.write_text(extra_defs + _TU.format(filt=filt, t=tag) + body)
        sources.append(str(tu))
        wrapper.append(_DECL.format(t=tag))
        defs.append(_DEF.format(t=tag))
    wrapper.append("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
    wrapper.extend(defs)
    wrapper.append("}")
    wcpp = work / "src" / "wrapper.cpp"
    wcpp.write_text("\n".join(wrapper))
    os.environ["PYTORCH_ROCM_ARCH"] = archs
    os.environ.setdefault("MAX_JOBS", "12")
    flags = [
        "-O3",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-fno-gpu-rdc",
        "-Werror=unused-variable",
    ] + (extra_flags or [])
    return load(
        name=name,
        sources=[str(wcpp)] + sources,
        extra_cuda_cflags=flags,
        build_directory=str(work / "build"),
        verbose=verbose,
    )


def main() -> None:
    """Build only (no GPU needed): ``qlinear_mutants.py`` compiles its mutants in parallel
    with this, then runs the GPU checks against the cached builds."""
    import argparse

    ap = argparse.ArgumentParser(description="JIT build of the quantized GEMV kernel")
    ap.add_argument("--name", required=True)
    ap.add_argument("--source")
    ap.add_argument("--formats", default="8:64,4:32,4:64")
    ap.add_argument("--filter", default="1")
    ap.add_argument("--defs", default="")
    args = ap.parse_args()
    fmts = [tuple(int(v) for v in f.split(":")) for f in args.formats.split(",")]
    build(args.name, args.source, fmts, extra_filter=args.filter, extra_defs=args.defs)
    print(f"BUILD OK {args.name}", flush=True)


if __name__ == "__main__":
    main()
