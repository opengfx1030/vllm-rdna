#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline GPTQ of the attention input projections and ``o_proj`` for the
``inkling2_qlinear_*`` kernels (``docs/rdna2/inkling2-qlinear-kernels.md``).

Round-to-nearest int4 loses too much of the model (``decode_gate.py``); GPTQ rounds the
weights column by column and pushes the rounding error of each column onto the columns
still to come, weighted by the inverse Hessian ``H = sum x x^T`` of the inputs the
projection really sees, so the OUTPUT error ``|(W - Q) X|`` shrinks instead of the weight
error.  The result keeps the format the kernels read (symmetric integers, one fp16 scale
per group of K): only the integers and the scales differ from round-to-nearest.

One job: boot the real checkpoint eager (the quantization flags must be off), teacher-force
``--tokens`` tokens of ``--text`` (from ``--skip-tokens``) through the model with hooks on
the attention input packs (the three input streams hq / hk / hv) and the o_proj packs of
every layer visit (the loop calls each layer twice: both visits train the same weights),
accumulate the fp32 Hessians on the GPU, then quantize every segment and save

    {"bits": 4, "group": 32, "weights": {"<layer>.<segment>": {"q": int8 [N, K],
                                                                "s": fp16 [N, K // group]}}}

for ``VLLM_INKLING2_QUANT_FILE`` (``qlinear_pack.py``).  A second token range (``--val-*``) is
only used to report the output-error proxy ``tr((W - Q) H (W - Q)^T) / tr(W H W^T)`` of
round-to-nearest and GPTQ on inputs the GPTQ solve did not see.

    python gptq_quant.py --ckpt ~/hfcache/inkling2-mini --text nll_text.md \
        --skip-tokens 4000 --tokens 3000 --format w4g32 --out gptq_w4g32.pt
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

CLIP_CANDIDATES = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65)


@torch.no_grad()
def gptq(
    W: torch.Tensor,
    H: torch.Tensor,
    bits: int,
    group: int,
    clip: bool,
    percdamp: float = 0.01,
    blocksize: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``W [N, K]``, ``H [K, K]`` fp32 -> (``q`` int8 ``[N, K]``, fp16 scales ``[N, K // group]``).

    Columns are rounded left to right; the scale of a group is chosen (MSE clip search over
    ``CLIP_CANDIDATES`` when ``clip``) from the already error-compensated weights of the
    group when its first column is reached, rounded to fp16 before use."""
    N, K = W.shape
    assert K % group == 0 and blocksize % group == 0
    W = W.float().clone()
    H = H.float().clone()
    idx = torch.arange(K, device=W.device)
    dead = torch.diag(H) == 0
    H[idx[dead], idx[dead]] = 1.0
    W[:, dead] = 0.0
    H.diagonal().add_(percdamp * H.diagonal().mean())
    # factorized on the host in fp64: no GPU LAPACK needed and the Hessian is ill conditioned
    L = torch.linalg.cholesky(H.double().cpu())
    Hinv = (
        torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
        .float()
        .to(W.device)
    )
    qmax = 2 ** (bits - 1) - 1
    Q = torch.empty(N, K, dtype=torch.int8, device=W.device)
    S = torch.empty(N, K // group, dtype=torch.float16, device=W.device)
    cands = CLIP_CANDIDATES if clip else (1.0,)
    s_col = None
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if i % group == 0:
                g = W1[:, i : i + group]
                amax = g.abs().amax(1, keepdim=True)
                best_s = best_e = None
                for c in cands:
                    s = (amax * c / qmax).clamp_min(2.0**-14).half().float()
                    e = ((torch.round(g / s).clamp(-qmax - 1, qmax) * s - g) ** 2).sum(
                        1, keepdim=True
                    )
                    if best_s is None:
                        best_s, best_e = s, e
                    else:
                        take = e < best_e
                        best_s = torch.where(take, s, best_s)
                        best_e = torch.where(take, e, best_e)
                S[:, (i1 + i) // group] = best_s.squeeze(1).half()
                s_col = best_s.squeeze(1)
            w = W1[:, i]
            q = torch.round(w / s_col).clamp(-qmax - 1, qmax)
            Q[:, i1 + i] = q.to(torch.int8)
            err = (w - q * s_col) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q, S


def proxy_error(W: torch.Tensor, Wq: torch.Tensor, H: torch.Tensor) -> float:
    """``tr((W - Wq) H (W - Wq)^T) / tr(W H W^T)``: the relative output energy error."""
    D = (W - Wq).float()
    W = W.float()
    num = torch.einsum("nk,kj,nj->", D, H, D)
    den = torch.einsum("nk,kj,nj->", W, H, W)
    return float(num / den)


class Recorder:
    """Wraps a pack: accumulates the input Hessians ``x^T x`` of the active phase."""

    def __init__(self, inner, store: dict, layer: int, kind: str, phase: list) -> None:
        self.inner = inner
        self.store = store
        self.layer = layer
        self.kind = kind
        self.phase = phase

    def __call__(self, xs, outs, *a, **k):
        r = self.inner(xs, outs, *a, **k)
        if self.kind == "attn_in":
            streams = {"hq": xs[0], "hk": xs[1], "hv": xs[2]}
        else:
            streams = {"o": xs[0]}
        acc = self.store[self.phase[0]]
        done: dict[int, torch.Tensor] = {}  # the same tensor may feed several streams
        for name, x in streams.items():
            if id(x) not in done:
                xf = x.float()
                done[id(x)] = xf.t() @ xf
            key = (self.layer, name)
            if key in acc:
                acc[key] += done[id(x)]
            else:
                acc[key] = done[id(x)].clone()
        return r

    def __getattr__(self, name):
        return getattr(self.inner, name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument(
        "--skip-tokens", type=int, default=4000, help="first calibration token"
    )
    ap.add_argument("--tokens", type=int, default=3000, help="calibration tokens")
    ap.add_argument(
        "--val-tokens", type=int, default=1500, help="held-out tokens after them"
    )
    ap.add_argument("--format", default="w4g32", help="w4g32, w4g64 or w8g64")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--layers", default="", help="comma list (default: all)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.67)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for k in list(os.environ):
        if k.startswith("VLLM_INKLING2_") and k.endswith("_QUANT"):
            raise SystemExit(f"unset {k}: GPTQ reads the fp16 packs")

    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams
    from vllm.models.inkling2_loop.amd import attention as A
    from vllm.models.inkling2_loop.amd import qlinear_pack as qp

    bits, group, clip = qp.parse_format(args.format)
    clip = clip and not args.no_clip
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    ids = tok(Path(args.text).read_text(), add_special_tokens=False)["input_ids"]
    lo = args.skip_tokens
    calib_ids = ids[lo : lo + args.tokens]
    val_ids = ids[lo + args.tokens : lo + args.tokens + args.val_tokens]
    assert len(calib_ids) == args.tokens and val_ids, f"text has {len(ids)} tokens"
    t0 = time.time()
    llm = LLM(
        model=args.ckpt,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        max_num_batched_tokens=1024,
        max_num_seqs=2,
        skip_tokenizer_init=True,
        max_logprobs=8,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0},
    )
    print(f"BOOT OK {time.time() - t0:.1f}s", flush=True)
    holder: dict = {}
    llm.apply_model(lambda m: holder.__setitem__("m", m))
    model = holder["m"]
    mods = [m for m in model.modules() if isinstance(m, A.Inkling2Attention)]
    store: dict = {"calib": {}, "val": {}}
    phase = ["calib"]
    for m in mods:
        m._attn_in = Recorder(m._attn_in, store, m.layer_idx, "attn_in", phase)
        m._o_pack = Recorder(m._o_pack, store, m.layer_idx, "o", phase)
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    window = 1500  # one request per window: below max_model_len, chunks of 1024 tokens
    for name, tokens in (("calib", calib_ids), ("val", val_ids)):
        phase[0] = name
        # every window is one prompt (the loop visits each layer twice per chunk)
        t0 = time.time()
        for lo in range(0, len(tokens), window):
            prompt = {"prompt_token_ids": tokens[lo : lo + window]}
            llm.generate([prompt], sp, use_tqdm=False)
        print(f"{name}: {len(tokens)} tokens in {time.time() - t0:.1f}s", flush=True)
    wanted = {int(v) for v in args.layers.split(",")} if args.layers else None
    weights: dict = {}
    rows = []
    t0 = time.time()
    for m in mods:
        if wanted is not None and m.layer_idx not in wanted:
            continue
        for kind, pack in (("attn_in", m._attn_in.inner), ("o", m._o_pack.inner)):
            lo_row = 0
            groups: dict[str, list] = {}
            for name, n, sx in zip(pack.names, pack.seg_n, pack.seg_x):
                stream = ("hq", "hk", "hv")[sx] if kind == "attn_in" else "o"
                groups.setdefault(stream, []).append((name, lo_row, n))
                lo_row += n
            for stream, segs in groups.items():
                H = store["calib"][(m.layer_idx, stream)]
                Hv = store["val"][(m.layer_idx, stream)]
                W = torch.cat([pack.weight[a : a + n] for _, a, n in segs], 0).float()
                Q, S = gptq(W, H, bits, group, clip, args.percdamp)
                Wq = qp.dequantize_rows(Q, S, group)
                q0, s0 = qp.quantize_rows(W, bits, group, clip)
                W0 = qp.dequantize_rows(q0, s0, group)
                rows.append(
                    (
                        m.layer_idx,
                        stream,
                        proxy_error(W, W0, H),
                        proxy_error(W, Wq, H),
                        proxy_error(W, W0, Hv),
                        proxy_error(W, Wq, Hv),
                    )
                )
                r = 0
                for name, _, n in segs:
                    weights[f"{m.layer_idx}.{name}"] = {
                        "q": Q[r : r + n].cpu(),
                        "s": S[r : r + n].cpu(),
                    }
                    r += n
        print(
            f"layer {m.layer_idx}: "
            + " ".join(
                f"{s} rtn {a:.4f}/{c:.4f} gptq {b:.4f}/{d:.4f}"
                for (li, s, a, b, c, d) in rows
                if li == m.layer_idx
            )
            + f"  [{time.time() - t0:.0f}s]",
            flush=True,
        )
    import numpy as np

    a = np.array([[r[2], r[3], r[4], r[5]] for r in rows])
    print(
        "output-error proxy (mean over solves), calibration: rtn {:.4f} gptq {:.4f}; "
        "held-out: rtn {:.4f} gptq {:.4f}".format(*a.mean(0)),
        flush=True,
    )
    torch.save({"bits": bits, "group": group, "weights": weights}, args.out)
    print(f"saved {len(weights)} weights to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
