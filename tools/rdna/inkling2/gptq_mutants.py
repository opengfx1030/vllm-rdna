#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand mutants of the GPTQ solver (``gptq_quant.py``) and of the calibrated-weights path of
``qlinear_pack.py``: each one changes a line of the real source, the CPU tests of
``tests/kernels/quantization/test_inkling2_qlinear_rdna2.py`` (``-k "gptq or calibrated"``)
must FAIL on it, and pass on the unmodified control.  Needs no GPU.

    python gptq_mutants.py --work $HOME/scratch/gptq_mut
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[2]
GPTQ = HERE / "gptq_quant.py"
QPACK = TREE / "vllm/models/inkling2_loop/amd/qlinear_pack.py"
TEST = TREE / "tests/kernels/quantization/test_inkling2_qlinear_rdna2.py"

COMP = "W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)"
MUTANTS = {
    # name: (target, old, new)
    "g1_no_compensation": (GPTQ, COMP, "pass"),
    "g2_wrong_sign": (GPTQ, COMP, COMP.replace("-=", "+=")),
    "g3_no_block_propagation": (GPTQ, "W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]", "pass"),
    "g4_err_not_scaled": (
        GPTQ,
        "err = (w - q * s_col) / Hinv1[i, i]",
        "err = (w - q * s_col)",
    ),
    "g5_scale_not_searched": (
        GPTQ,
        "cands = CLIP_CANDIDATES if clip else (1.0,)",
        "cands = (1.0,)",
    ),
    "c1_file_ignored": (
        QPACK,
        'got = calibrated.get(f"{layer}.{name}") if calibrated else None',
        "got = None",
    ),
    "c2_key_without_layer": (
        QPACK,
        'calibrated.get(f"{layer}.{name}")',
        "calibrated.get(name)",
    ),
    "c3_no_format_check": (
        QPACK,
        'if (data["bits"], data["group"]) != (bits, group):',
        "if False:",
    ),
    "c4_no_shape_check": (
        QPACK,
        "if q.shape != (n, pack.weight.shape[1]):",
        "if False:",
    ),
    "c5_scales_not_used": (
        QPACK,
        's = got["s"].to(pack.weight.device)',
        's = got["s"].to(pack.weight.device) * 0 + 1',
    ),
}

RUNNER = """
import importlib.util, os, sys
import pytest
os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
import vllm.models.inkling2_loop.amd.linear_pack  # parent packages first
for name, path in (("gptq_quant", sys.argv[1]), ("vllm.models.inkling2_loop.amd.qlinear_pack", sys.argv[2])):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
sys.exit(pytest.main([sys.argv[3], "-q", "--noconftest", "-p", "no:cacheprovider", "-x",
                      "-k", "gptq or calibrated"]))
"""


def run(gptq: Path, qpack: Path) -> int:
    env = dict(os.environ, HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
    p = subprocess.run(
        [sys.executable, "-c", RUNNER, str(gptq), str(qpack), str(TEST)],
        env=env,
        capture_output=True,
        text=True,
    )
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ok = True
    rc = run(GPTQ, QPACK)
    print(f"control: exit {rc} ({'pass' if rc == 0 else 'FAIL'})", flush=True)
    ok &= rc == 0
    for name, (target, old, new) in MUTANTS.items():
        text = target.read_text()
        if text.count(old) != 1:
            print(f"{name}: pattern found {text.count(old)} times: MUTANT NOT APPLIED")
            ok = False
            continue
        g, q = GPTQ, QPACK
        path = work / f"{name}_{target.name}"
        path.write_text(text.replace(old, new))
        if target == GPTQ:
            g = path
        else:
            q = path
        rc = run(g, q)
        print(f"{name}: exit {rc} ({'killed' if rc != 0 else 'SURVIVED'})", flush=True)
        ok &= rc != 0
    print(
        "MUTANTS ALL KILLED, CONTROL PASSES"
        if ok
        else "MUTANT SURVIVED OR CONTROL FAILED"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
