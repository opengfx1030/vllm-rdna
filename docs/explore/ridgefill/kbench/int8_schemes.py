# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which side of the int8 map loses accuracy, and which scale layout fixes it.

CPU only, on the captured split state and the fitted maps. For each layer
asked for, the error is the relative RMS of the int8 output against the fp16
output (``q``), and the fit error against the captured block input (``fit``):

* ``w8_chan``: int8 weights, one scale per output row, fp16 activations;
* ``w8_g{G}``: int8 weights, one scale per output row and K group of G;
* ``a8_tok``: per-token int8 activations, fp16 weights;
* ``w8a8_chan_tok``: the plain sdot4 kernel (per row x per token);
* ``w8a8_g{G}``: per-group scales on both sides (W4A8-MoE style epilogue);
* ``smooth{alpha}``: SmoothQuant smoothing ``x / s``, ``w * s`` with
  ``s_k = max|x_k|^alpha / max|w_k|^(1 - alpha)``, then per row x per token;
* ``had{B}_chan_tok``: block-Hadamard rotation of K (``x H``, ``w H``, exact
  in real arithmetic since H is orthonormal), then per row x per token. The
  rotation spreads an outlier column over its block, so the plain sdot4
  epilogue survives; the activation side would run it in the quant kernel.

    python3 int8_schemes.py --capture ~/w4a8_runs/lanes/ridgefill/capture \
        --layers 24 25 27 31 39 47 --rows 8192
"""

import argparse
import time
from pathlib import Path

import torch


def rel(d: torch.Tensor, ref: torch.Tensor) -> float:
    return float(d.square().mean().sqrt() / ref.square().mean().sqrt())


def q_rows(t: torch.Tensor, group: int = 0):
    """Symmetric int8 along dim 1, per row or per row and group."""
    rows, cols = t.shape
    g = group or cols
    tg = t.view(rows, cols // g, g)
    s = tg.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / 127.0
    q = torch.round(tg / s).clamp(-127, 127)
    return q.view(rows, cols), s.view(rows, cols // g)


def deq(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    rows, cols = q.shape
    g = cols // s.shape[1]
    return (q.view(rows, -1, g) * s.unsqueeze(2)).view(rows, cols)


def hadamard(n: int) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix of size n (a power of two)."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / h.shape[0] ** 0.5


def rotate(t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    rows, cols = t.shape
    b = h.shape[0]
    return (t.view(rows, cols // b, b) @ h).view(rows, cols)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[24, 25, 27, 31, 39, 47])
    p.add_argument("--rows", type=int, default=8192)
    p.add_argument("--threads", type=int, default=16)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    cap = torch.load(args.capture / "capture.pt", map_location="cpu", mmap=True)
    maps = torch.load(args.capture / "maps.pt", map_location="cpu", mmap=True)
    # Spread the rows over all eight captured prompts.
    total = cap["split_state"].shape[0]
    idx = torch.linspace(0, total - 1, args.rows).long()
    x = cap["split_state"][idx].half().float()
    xmax = x.abs().amax(dim=0)
    xq_tok = deq(*q_rows(x))
    for layer in args.layers:
        a, b = maps[layer]
        w = a.float()
        bias = b.float()
        ys = cap["layer_inputs"][layer][idx].float()
        y16 = x @ w.T + bias
        wr = w.abs().amax(dim=1) / w.square().mean(dim=1).sqrt().clamp(min=1e-12)
        print(
            f"\nlayer {layer}: fit fp16 {rel(y16.half().float() - ys, ys):.5f}; "
            f"weight row crest median {float(wr.median()):.1f} max "
            f"{float(wr.max()):.1f} [{time.time() - t0:.0f}s]",
            flush=True,
        )
        arms = {}
        wq_chan = deq(*q_rows(w))
        arms["w8_chan"] = x @ wq_chan.T + bias
        for g in (128, 32):
            arms[f"w8_g{g}"] = x @ deq(*q_rows(w, g)).T + bias
        arms["a8_tok"] = xq_tok @ w.T + bias
        arms["w8a8_chan_tok"] = xq_tok @ wq_chan.T + bias
        for g in (128, 256, 512, 1024):
            arms[f"w8a8_g{g}"] = deq(*q_rows(x, g)) @ deq(*q_rows(w, g)).T + bias
        for blk in (128, 512, 2048):
            h = hadamard(blk)
            xr = deq(*q_rows(rotate(x, h)))
            wrot = deq(*q_rows(rotate(w, h)))
            arms[f"had{blk}_chan_tok"] = xr @ wrot.T + bias
        wmax = w.abs().amax(dim=0)
        for alpha in (0.5, 0.7):
            s = (xmax.clamp(min=1e-6) ** alpha) / (
                wmax.clamp(min=1e-6) ** (1.0 - alpha)
            )
            xs = deq(*q_rows(x / s))
            ws = deq(*q_rows(w * s))
            arms[f"smooth{alpha}"] = xs @ ws.T + bias
        for name, y in arms.items():
            yh = y.half().float()
            print(
                f"  {name:16s} q {rel(yh - y16, y16):.5f}  fit {rel(yh - ys, ys):.5f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
