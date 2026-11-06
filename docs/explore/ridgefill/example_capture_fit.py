# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture activations and refit a RidgeFill map. CPU synthetic demo by default.

A real capture has to run in dest's torch 2.12 / ROCm 7.14 tree so the
activations match the served model. This process does not import torch
unless ``--require-torch`` is passed, and then it refuses any other pin.

The released Flash-Next maps were fitted on an MXFP4/FP8 checkpoint. Their
``ridgefill.json`` pins that checkpoint's vocab sha256 and per-tensor
encodings. Point this script at those pins and at a W4A16 / EXL3 / HI2 pin
file and it exits. That is the fail-closed behavior the server uses too.

    python3 docs/explore/ridgefill/example_capture_fit.py
    python3 docs/explore/ridgefill/example_capture_fit.py --show-pin-failure
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "vllm" / "model_executor" / "layers" / "ridgefill"

# Dest serve pins. Do not bump these from this explore draft.
DEST_TORCH = "2.12"
DEST_ROCM = "7.14"


def _load(name: str):
    path = PKG / f"{name}.py"
    mod_name = f"ridgefill_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def assert_dest_runtime(torch_version: str | None, rocm_version: str | None) -> None:
    """Refuse a capture that is not dest's torch 2.12 / ROCm 7.14."""
    if torch_version is None or not str(torch_version).startswith(DEST_TORCH):
        raise RuntimeError(
            f"capture torch must be {DEST_TORCH}.x (dest pin), got {torch_version}"
        )
    if rocm_version is None or not str(rocm_version).startswith(DEST_ROCM):
        raise RuntimeError(
            f"capture ROCm must be {DEST_ROCM}.x (dest pin), got {rocm_version}"
        )


def synthetic_capture(math):
    """One layer, din 4, dout 2, known map, fp32 rows."""
    weight = [[1.0, 0.0, -1.0, 0.5], [0.0, 2.0, 0.0, -0.5]]
    bias = [0.25, -0.25]
    xs = [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    ys = math.apply_map(xs, weight, bias)
    return xs, ys, weight, bias


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--require-torch",
        action="store_true",
        help="import torch and refuse unless it is dest's 2.12 / ROCm 7.14",
    )
    parser.add_argument(
        "--show-pin-failure",
        action="store_true",
        help="print the fail-closed result of an MXFP4 pin against a W4A16 pin",
    )
    args = parser.parse_args(argv)
    math = _load("math")
    if args.require_torch:
        import torch

        hip = getattr(torch.version, "hip", None)
        hip_s = hip if isinstance(hip, str) else None
        assert_dest_runtime(torch.__version__, hip_s)
        print(f"torch {torch.__version__} hip {hip}")

    xs, ys, true_w, true_b = synthetic_capture(math)
    fit_w, fit_b = math.fit_ridge(xs, ys, lam=0.0)
    print("refit bias", [round(v, 6) for v in fit_b])
    for row, truth in zip(fit_w, true_w):
        err = max(abs(a - b) for a, b in zip(row, truth))
        if err > 1e-6:
            print("refit error", err, file=sys.stderr)
            return 1
    print("synthetic refit matches the capture map")

    if args.show_pin_failure:
        pins = _load("loader")
        released = {
            "model": {
                "vocab_sha256": "mxfp4-vocab",
                "encodings": {"blk.24.ffn_down.weight": "fp8_e4m3*f32[1x128]"},
            }
        }
        try:
            pins.validate_pins(
                released,
                vocab_sha256="w4a16-vocab",
                encodings={"blk.24.ffn_down.weight": "int4_awq"},
            )
        except pins.RidgeFillPinError as exc:
            print("pin check refused the released map:", exc)
        else:
            print("pin check should have failed", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
