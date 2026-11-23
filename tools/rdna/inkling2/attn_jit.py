#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT build of ``csrc/rocm/inkling2_attn_rdna2.cu`` as a stand-alone extension.

Used by the prefill-attention depth battery
(``tests/kernels/core/test_inkling2_attn_prefill_depth_rdna2.py``:
``I2_ATTN_VARIANT_JIT`` runs the battery on another file or on other ``-D`` macros,
which is how the hand mutants of ``attn_tile_mutants.py`` are run) and as a compile
check:

    python tools/rdna/inkling2/attn_jit.py --archs "gfx1030;gfx1100;gfx90a;gfx942"

The module exposes ``inkling2_attention`` with the positional signature of the
``torch.ops._rocm_C`` op and ``NARGS``, the number of its arguments (the trailing ones
of a newer signature are not in the 24 of the shipped op).  ``SPEC`` strings read
``path[@MACRO=value,MACRO=value]``.  ``I2_JIT_DIR`` (default
``~/.cache/vllm_inkling2_jit``) holds the build trees, ``I2_JIT_OPT`` the optimisation
level (default ``-O2``: the level the library is built with).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib

_TAIL = """

#include <torch/extension.h>
template <class R, class... A>
constexpr int i2_jit_nargs(R (*)(A...)) {
  return sizeof...(A);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inkling2_attention", &inkling2_attention);
  m.attr("NARGS") = i2_jit_nargs(&inkling2_attention);
}
"""

REPO = pathlib.Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc" / "rocm" / "inkling2_attn_rdna2.cu"


def parse_spec(spec: str | os.PathLike | None) -> tuple[pathlib.Path, list[str]]:
    """``path[@MACRO=value,...]`` -> (path, macros); an empty spec is the repo file."""
    path, _, macros = str(spec or "").partition("@")
    return (
        pathlib.Path(path) if path else SOURCE,
        [m for m in macros.split(",") if m],
    )


def build(
    spec: str | os.PathLike | None = None,
    archs: str = "gfx1030",
    verbose: bool = False,
):
    """Compile ``spec`` (default: the repo's file) and return the extension module.

    The build tree is named after the content hash of the source, the macros, the
    archs and the optimisation level, so the same inputs share one build.
    """
    from torch.utils.cpp_extension import load

    path, macros = parse_spec(spec)
    opt = os.environ.get("I2_JIT_OPT", "-O2")
    text = path.read_text() + _TAIL
    tag = hashlib.md5(
        "\0".join([text, ",".join(macros), archs, opt]).encode()
    ).hexdigest()[:12]
    name = f"i2_attn_jit_{tag}"
    work = (
        pathlib.Path(
            os.environ.get("I2_JIT_DIR", "~/.cache/vllm_inkling2_jit")
        ).expanduser()
        / name
    )
    (work / "src").mkdir(parents=True, exist_ok=True)
    (work / "build").mkdir(parents=True, exist_ok=True)
    cu = work / "src" / "inkling2_attn_rdna2.cu"
    if not cu.exists() or cu.read_text() != text:
        cu.write_text(text)
    os.environ["PYTORCH_ROCM_ARCH"] = archs
    return load(
        name=name,
        sources=[str(cu)],
        extra_include_paths=[str(REPO / "csrc" / "rocm")],
        extra_cuda_cflags=[opt, *(f"-D{m}" for m in macros)],
        build_directory=str(work / "build"),
        verbose=verbose,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="gfx1030")
    ap.add_argument("--spec", default="")
    args = ap.parse_args()
    mod = build(args.spec or None, args.archs, verbose=True)
    print("BUILD OK", args.archs, "NARGS", mod.NARGS)


if __name__ == "__main__":
    main()
