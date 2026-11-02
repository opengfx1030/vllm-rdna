# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Executable half of the RidgeFill explore note. No third-party dependency.

    python3 docs/explore/ridgefill/reference_ridge.py

Prints the fp32 fit on a known map and the fp16 range decision for a
10240-wide input and a 2560-wide output. The numbers are arithmetic, not
V620 measurements. ``TESTPLAN.md`` holds the empty GPU cells.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MATH = ROOT / "vllm" / "model_executor" / "layers" / "ridgefill" / "math.py"


def _math():
    spec = importlib.util.spec_from_file_location("ridgefill_math", MATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(MATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ridgefill_math"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    math = _math()
    xs = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    ys = [(3.0 * x[0] + 0.5 * x[1] - 1.0,) for x in xs]
    weight, bias = math.fit_ridge(xs, ys, lam=0.0)
    pred = math.apply_map(xs, weight, bias)
    err = max(abs(pred[i][0] - ys[i][0]) for i in range(len(xs)))
    print(f"fit A={weight[0]} b={bias[0]:.6g} max abs err={err:.3e}")
    if err > 1e-8:
        print("fit did not recover the map", file=sys.stderr)
        return 1

    # Shapes from the Flash-Next release: din 10240, dout 2560. One row.
    din, dout = 10240, 2560
    x = [[0.5 if i % 2 == 0 else -0.25 for i in range(din)]]
    weight_big = [[0.0] * din for _ in range(dout)]
    weight_big[0][0] = 1.0
    pred, decision, in_max, out_max = math.apply_checked(
        x, weight_big, [0.0] * dout, "fp16"
    )
    print(
        f"flash-next shape in_max={in_max} out_max={out_max:.4g} "
        f"gemm={decision.gemm_dtype} "
        f"input={decision.input_dtype} output={decision.output_dtype} "
        f"pred0={pred[0][0]:.4g}"
    )
    huge = [[math.FP16_MAX + 1.0] + [0.0] * (din - 1)]
    _, over, in_max, out_max = math.apply_checked(
        huge, weight_big, [0.0] * dout, "fp16"
    )
    print(
        f"overflow in_max={in_max:.6g} out_max={out_max:.6g} "
        f"gemm={over.gemm_dtype} input_overflow={over.input_overflow}"
    )
    if not over.input_overflow or over.gemm_dtype != "bf16":
        print("overflow did not fall back", file=sys.stderr)
        return 1
    print("reference ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
