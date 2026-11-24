#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write the tiny Inkling-2-Mini checkpoint and quantize it to a tiny HI2 directory.

CPU only, no GPU and no lease.  This is the fixture of
``tests/models/inkling2_loop/test_hi2_reference.py`` and the fastest end-to-end
check of the quantizer's stacked output against this fork's loader.

    INKLING2_TOOLS=$TREE/tools/rdna/inkling2 \\
    PYTHONPATH=$HI2_SRC:$TFM_MAIN:$TREE \\
    python tools/inkling_hi2/make_tiny_hi2.py --out ~/w4a8_runs/inkling-hi2-runtime/tiny

``--tools`` / ``--hi2-src`` default to ``$INKLING2_TOOLS`` and ``$HI2_SRC`` (or
``$PYTHONPATH``).  The script imports ``torch`` before anything else: the vLLM
tree's compiled extensions must find torch's LLVM first on this image.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch


def _import(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make(
    out: Path,
    tools: Path,
    *,
    layers: int = 6,
    seed: int = 11,
    log=print,
) -> Path:
    """Write ``out/stock`` (tiny checkpoint) and ``out/hi2`` (its HI2 output)."""
    from hi2.calibrate import Config
    from hi2.pipeline import RunConfig, run

    make_tiny = _import("hi2_test_make_tiny", tools / "make_tiny_checkpoint.py")
    stock = out / "stock"
    if not (stock / "model.safetensors.index.json").exists():
        log(f"writing the tiny stock checkpoint in {stock}")
        spec = make_tiny.TinySpec(num_hidden_layers=layers, router_bias_scale=8.0)
        make_tiny.write_checkpoint(stock, spec, seed=seed)
    hi2_dir = out / "hi2"
    if (hi2_dir / "hi2_run.json").exists():
        log(f"reusing {hi2_dir}")
        return hi2_dir
    gen = torch.Generator().manual_seed(5)
    ids = torch.randint(0, 50, (4, 12), generator=gen)
    rc = RunConfig(
        src=stock,
        dst=hi2_dir,
        workdir=out / "work",
        include=r"\.mlp\.experts\.",
        devices=["cpu"],
        compute_dtype="float32",
        target_bpw=3.5,
        calib_ids=ids,
        calib_samples=4,
        seq_len=12,
        cfg=Config(effort="fast"),
    )
    summary = run(rc, log=lambda *a: None)
    log(
        f"quantized {summary.get('tensors', '?')} expert tensors to {hi2_dir} "
        f"({hi2_dir.stat().st_size if hi2_dir.exists() else 0} bytes of entries)"
    )
    return hi2_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tools", type=Path, default=None)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    tools = args.tools or (
        Path(os.environ["INKLING2_TOOLS"]) if os.environ.get("INKLING2_TOOLS") else None
    )
    if tools is None or not (tools / "make_tiny_checkpoint.py").exists():
        ap.error("pass --tools or set INKLING2_TOOLS to the fork's tools/rdna/inkling2")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        import hi2  # noqa: F401
    except ImportError:
        ap.error(
            "the quantizer is not importable: set PYTHONPATH to its tree (HI2_SRC)"
        )
    make(args.out, tools, layers=args.layers, seed=args.seed)
    print(args.out / "hi2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
