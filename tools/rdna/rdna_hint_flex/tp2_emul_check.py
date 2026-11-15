#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU model-level check of TP=2 rocBLAS rows: greedy tokens and logits.

The TunableOp rows of recipe ``27b-hi2-tp2`` are keyed by the per-rank GEMM shapes
(``tn_17408_2048_5120_ld_5120_5120_17408``). A TP=1 engine never issues those shapes,
and the TP=2 pair of GPUs is not available to this campaign. ``run`` therefore loads the
real checkpoint at TP=1 on ONE GPU and replaces, for M > 48 (the dequant + rocBLAS path
of ``rdna_hint_flex_linear_rdna2``), the single full-width GEMM of each hi2 linear by
the two GEMMs the two TP=2 ranks issue:

* column-parallel layer (qkv, in_proj_qkvz / ba, gate_up): rows [0, R/2) and [R/2, R) of
  the dequantized segment, each a contiguous ``[R/2, K]`` operand (ld = K), outputs
  concatenated;
* row-parallel layer (o_proj / out_proj, down): columns [0, K/2) and [K/2, K) of the
  transformed activation and the weight, made contiguous (ld = K/2), outputs summed in
  fp16 (what the 2-rank all-reduce does).

Every element of a column-parallel output depends on one weight row only, and the
kernel choice depends on (M, N, K) only, so splitting [0, R/2) | [R/2, R) instead of the
true rank slices does not change a single rounding. The activation transform and the
dequantization are the checkpoint's own ops, unchanged. Everything else (attention,
GDN, norms, sampler, the M <= 48 GEMV path, the dense LM head GEMV) runs as in serving.
The calls issue exactly the rocBLAS keys of the TP=2 ranks, so the same
``PYTORCH_TUNABLEOP_FILENAME`` rows apply; ``compare --rows`` counts the calls the
rows hit, so the report says how much of the prefill they cover.

  python tp2_emul_check.py run MODEL --out untuned.json          # rows off
  PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 \\
      PYTORCH_TUNABLEOP_FILENAME=<dir>/tunableop_results.csv \\
      python tp2_emul_check.py run MODEL --arm tuned --out tuned.json
  python tp2_emul_check.py compare untuned.json tuned.json --rows <dir>/results0.csv

Not covered: the MTP drafter's dense GEMMs and the LM head at M > 32 (kernel-level gate
only), and the all-reduce order beyond two operands. Run with the TunableOp env of the
arm set by the caller; torch 2.12 loads no rows under PYTORCH_TUNABLEOP_RECORD_UNTUNED=1
and ``tunable.enable(False)`` does not unload rows, hence two processes.
"""

import argparse
import collections
import json
import os
import sys
from pathlib import Path

DECODE_MAX_M = 48
LENGTHS = (1024, 4200, 3100)  # chunks [1023, 1], [2048, 2048, ...], [2048, 1024, 27, 1]
STEPS = 64
TOP = 20


def prompt_ids(tokenizer, tree, lengths):
    """Real technical text (repo docs), cut to exact token counts."""
    docs = sorted(Path(tree, "docs/rdna2").glob("*.md"))
    out = []
    for i, n in enumerate(lengths):
        ids: list[int] = []
        for doc in docs[i:] + docs[:i]:
            ids += tokenizer(doc.read_text(), add_special_tokens=False)["input_ids"]
            if len(ids) >= n:
                break
        if len(ids) < n:
            raise SystemExit(
                f"docs/rdna2 holds only {len(ids)} tokens for a {n} prompt"
            )
        out.append(ids[:n])
    return out


def install(rows_path):
    """Emulate the two TP=2 ranks inside RdnaHintFlexLinearMethod.apply (M > 48)."""
    import torch
    import torch.nn.functional as F

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.quantization import rdna_hint_flex as hf

    have = set()
    if rows_path:
        for line in Path(rows_path).read_text().splitlines():
            if line.startswith("GemmTunableOp_Half_TN,"):
                have.add(line.split(",")[1])
    stats = {"hit": 0, "miss": 0, "m": collections.Counter(), "keys": {}}
    orig = hf.RdnaHintFlexLinearMethod.apply

    def count(r, m, k):
        key = f"tn_{r}_{m}_{k}_ld_{k}_{k}_{r}"
        hit = key in have
        stats["hit" if hit else "miss"] += 1
        stats["keys"][key] = stats["keys"].get(key, 0) + 1

    def apply(self, layer, x, bias=None):
        x2 = x.reshape(-1, x.shape[-1])
        m = x2.shape[0]
        if m <= DECODE_MAX_M or not self.use_kernels or self.compute_mode == 2:
            return orig(self, layer, x, bias)
        stats["m"][m] += 1
        row_parallel = isinstance(layer, RowParallelLinear)
        outs = []
        for seg in layer._rdna_hint_flex_segments:
            k, r = x2.shape[1], seg["row_nbits"].numel()
            xt = x2.contiguous()
            if seg["transform"]:
                xt = torch.empty_like(x2)
                ops.rdna_hint_flex_act_transform_rdna2(
                    x2, seg["col_scale"], seg["had"], xt
                )
            wd = torch.empty(r, k, dtype=x2.dtype, device=x2.device)
            ops.rdna_hint_flex_dequant_rdna2(
                seg["codes"],
                seg["code_off"],
                seg["blk_params"],
                seg["param_off"],
                seg["row_consts"],
                seg["row_nbits"],
                seg["row_block"],
                wd,
            )
            if row_parallel:
                kh = k // 2
                y = None
                for h in range(2):
                    xh = xt[:, h * kh : (h + 1) * kh].contiguous()
                    wh = wd[:, h * kh : (h + 1) * kh].contiguous()
                    count(r, m, kh)
                    yh = F.linear(xh, wh)
                    y = yh if y is None else y + yh
            else:
                rh = r // 2
                count(rh, m, k)
                count(rh, m, k)
                y = torch.cat([F.linear(xt, wd[:rh]), F.linear(xt, wd[rh:])], dim=1)
            outs.append(y)
        y = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], y.shape[-1])

    hf.RdnaHintFlexLinearMethod.apply = apply
    return stats


def cmd_run(args) -> int:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    stats = install(args.rows)
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    tree = Path(__file__).resolve().parents[3]
    lengths = [int(n) for n in args.lengths.split(",")]
    ids = prompt_ids(AutoTokenizer.from_pretrained(args.model), tree, lengths)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="float16",
        block_size=1024,
        max_model_len=16384,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.92,
        kv_cache_memory_bytes=args.kv_gib << 30,
        enable_prefix_caching=True,
        language_model_only=True,
        attention_backend=AttentionBackendEnum.RDNA_ATTN,
        enforce_eager=True,
        logprobs_mode="raw_logits",
        seed=0,
    )
    params = SamplingParams(
        temperature=0.0, max_tokens=args.steps, logprobs=TOP, ignore_eos=True
    )
    outs = llm.generate([{"prompt_token_ids": p} for p in ids], params)
    result = {
        "arm": args.arm,
        "tunableop": {
            k: v for k, v in os.environ.items() if k.startswith("PYTORCH_TUNABLEOP")
        },
        "prompt_tokens": [len(p) for p in ids],
        "hit": stats["hit"],
        "miss": stats["miss"],
        "m_hist": {str(m): n for m, n in sorted(stats["m"].items())},
        "keys": dict(sorted(stats["keys"].items())),
        "completions": [],
    }
    for o in outs:
        c = o.outputs[0]
        result["completions"].append(
            {
                "tokens": list(c.token_ids),
                "chosen": [lp[t].logprob for t, lp in zip(c.token_ids, c.logprobs)],
                "top": [
                    {str(t): v.logprob for t, v in lp.items()} for lp in c.logprobs
                ],
            }
        )
    Path(args.out).write_text(json.dumps(result))
    print(
        f"{args.arm}: {len(outs)} prompts x {args.steps} tokens, emulated rocBLAS "
        f"calls hit={stats['hit']} miss={stats['miss']}, "
        f"M histogram {result['m_hist']}"
    )
    return 0


def cmd_compare(args) -> int:
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    worst = 0.0
    flips = 0
    for i, (ca, cb) in enumerate(zip(a["completions"], b["completions"])):
        ta, tb = ca["tokens"], cb["tokens"]
        div = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        end = len(ta) if div is None else div
        diffs, chosen = [], []
        for j in range(end + (0 if div is None else 1)):
            top_a, top_b = ca["top"][j], cb["top"][j]
            shared = [t for t in top_a if t in top_b]
            diffs.append(max(abs(top_a[t] - top_b[t]) for t in shared))
            if j < end:
                chosen.append(abs(ca["chosen"][j] - cb["chosen"][j]))
        flips += div is not None
        worst = max(worst, max(diffs))
        print(
            f"prompt {i} ({a['prompt_tokens'][i]} tokens): "
            + ("tokens identical" if div is None else f"FIRST DIVERGENCE at step {div}")
            + f"; steps compared {len(diffs)}; max |dlogit| over the top-{TOP} "
            f"{max(diffs):.4g}; chosen-token |dlogit| max {max(chosen, default=0):.4g} "
            f"mean {sum(chosen) / max(1, len(chosen)):.4g}"
        )
    summary = (
        f"{a['arm']} vs {b['arm']}: {flips} of {len(a['completions'])} prompts "
        f"diverge; worst top-{TOP} |dlogit| {worst:.4g} (raw fp16 logits)"
    )
    if args.rows:
        have = {
            line.split(",")[1]
            for line in Path(args.rows).read_text().splitlines()
            if line.startswith("GemmTunableOp_Half_TN,")
        }
        hit = sum(n for key, n in b["keys"].items() if key in have)
        total = sum(b["keys"].values())
        flop = {
            k: n * int(k.split("_")[1]) * int(k.split("_")[2]) * int(k.split("_")[3])
            for k, n in b["keys"].items()
        }
        hit_flop = sum(v for k, v in flop.items() if k in have)
        summary += (
            f"; rows hit {hit} of {total} emulated rocBLAS calls "
            f"({100 * hit_flop / sum(flop.values()):.0f} % of their FLOPs)"
        )
    print(summary)
    return 1 if flips else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("model")
    run.add_argument("--out", required=True)
    run.add_argument("--arm", default="untuned")
    run.add_argument("--rows", default="", help="rows file to count hits against")
    run.add_argument("--kv-gib", type=int, default=8)
    run.add_argument("--steps", type=int, default=STEPS)
    run.add_argument(
        "--lengths",
        default=",".join(map(str, LENGTHS)),
        help="prompt token counts (default: chunks [1023, 1], [2048, 2048, ..], "
        "[2048, 1024, 27, 1])",
    )
    run.set_defaults(func=cmd_run)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument(
        "--rows", default="", help="rows file: report the share of calls it hits"
    )
    cmp_.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
