# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fit error of the RidgeFill maps with int8 weights and int8 activations.

CPU only. Reads the fp32 capture (split state and the 24 late-layer block
inputs) and the fitted fp16 maps, then per layer reports the relative RMS
error against the captured block input for:

* ``fp16``: the serve path today (fp16 inputs, fp32 accumulate);
* ``int8``: per-token absmax int8 activations x per-output-channel absmax
  int8 weights, int32 accumulate, fp32 rescale (the sdot4 kernel);
* ``int8_g128``: the same with one activation scale per 128-wide K group,
  the W4A8 MoE kernel's scheme, for comparison.

``q_vs_fp16`` is the int8 output's distance from the fp16 output, relative to
the fp16 output's RMS: the error the quantization alone adds.

    python3 int8_fit.py --capture ~/w4a8_runs/lanes/ridgefill/capture \
        --rows 32768 --threads 16 --json int8_fit.json
"""

import argparse
import json
import time
from pathlib import Path

import torch


def rel_rms(diff: torch.Tensor, ref: torch.Tensor) -> float:
    return float(diff.square().mean().sqrt() / ref.square().mean().sqrt())


def quant_rows(x: torch.Tensor, group: int = 0):
    """Symmetric absmax int8 per row (``group`` 0) or per row and K group."""
    if group:
        rows, cols = x.shape
        xg = x.view(rows, cols // group, group)
        scale = xg.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / 127.0
        q = torch.round(xg / scale).clamp(-127, 127)
        return q.view(rows, cols), scale.view(rows, cols // group)
    scale = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    q = torch.round(x / scale).clamp(-127, 127)
    return q, scale


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--rows", type=int, default=32768)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    t0 = time.time()
    cap = torch.load(args.capture / "capture.pt", map_location="cpu", mmap=True)
    maps = torch.load(args.capture / "maps.pt", map_location="cpu", mmap=True)
    xs = cap["split_state"][: args.rows].float()
    rows, din = xs.shape
    # The serve casts the split state to fp16 before the map GEMM.
    x16 = xs.half().float()
    xq, sx = quant_rows(x16)
    group = 128
    xg, sxg = quant_rows(x16, group)
    peak = x16.abs().amax(dim=1)
    rms = x16.square().mean(dim=1).sqrt()
    crest = peak / rms.clamp(min=1e-12)
    xg_err = rel_rms(
        (xg.view(rows, -1, group) * sxg.unsqueeze(2)).view(rows, din) - x16, x16
    )
    print(
        f"rows {rows} din {din}: |x| max {float(peak.max()):.4f}, "
        f"row crest (absmax/rms) median {float(crest.median()):.1f} "
        f"max {float(crest.max()):.1f}; act int8 rel err "
        f"{rel_rms(xq * sx - x16, x16):.5f} (per token), {xg_err:.5f}"
        f" (per 128 group); load {time.time() - t0:.1f}s",
        flush=True,
    )
    out = {"rows": rows, "din": din, "layers": {}}
    worst = {"fp16": 0.0, "int8": 0.0, "int8_g128": 0.0, "q_vs_fp16": 0.0}
    sums = dict.fromkeys(worst, 0.0)
    for layer in sorted(maps):
        a, b = maps[layer]
        a16 = a.float()
        b16 = b.float()
        ys = cap["layer_inputs"][layer][:rows].float()
        sa = a16.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
        aq = torch.round(a16 / sa).clamp(-127, 127)
        y16 = x16 @ a16.T + b16
        # int32 accumulate is exact; fp32 here differs by < 1e-7 relative.
        yq = (xq @ aq.T) * sx * sa.T + b16
        # Per-group activation scales: sum_g s_x[m, g] * (q_x[m, g] . q_a[n, g]).
        yg = torch.zeros_like(y16)
        for g in range(din // group):
            sl = slice(g * group, (g + 1) * group)
            yg += (xg[:, sl] @ aq[:, sl].T) * sxg[:, g : g + 1]
        yg = yg * sa.T + b16
        y16h = y16.half().float()
        yqh = yq.half().float()
        ygh = yg.half().float()
        e = {
            "fp16": rel_rms(y16h - ys, ys),
            "int8": rel_rms(yqh - ys, ys),
            "int8_g128": rel_rms(ygh - ys, ys),
            "q_vs_fp16": rel_rms(yqh - y16h, y16h),
            "q_max_abs": float((yqh - y16h).abs().max()),
        }
        out["layers"][int(layer)] = e
        for k in worst:
            worst[k] = max(worst[k], e[k])
            sums[k] += e[k]
        print(
            f"layer {layer}: fit rel_rms fp16 {e['fp16']:.5f} int8 {e['int8']:.5f} "
            f"int8_g128 {e['int8_g128']:.5f}; int8 vs fp16 {e['q_vs_fp16']:.5f} "
            f"(max abs {e['q_max_abs']:.4f}) [{time.time() - t0:.0f}s]",
            flush=True,
        )
    n = len(out["layers"])
    out["mean"] = {k: v / n for k, v in sums.items()}
    out["max"] = worst
    print("mean", json.dumps(out["mean"]), "max", json.dumps(worst))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
