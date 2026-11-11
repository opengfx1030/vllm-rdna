# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-quantization study for the I8 compute mode, on real activations.

Runs vLLM in-process (eager) on an rdna_hint_flex checkpoint and patches the
linear method so every quantized linear sees its real input after smoothing
and rotation (x' = H(x / s)):

  --mode stats  : output unchanged (F16 path); for each activation group size
                  G in --groups (64 = per 64 columns, 0 = per token) record the
                  activation error |x' - q(x')| / |x'| and the output error
                  |(x' - q(x')) W^T| / |x' W^T| with W the stored-basis
                  dequantized weight, per matrix and pooled.
  --mode fq     : the linear outputs become q_G(x') W^T in fp32 (q_G =
                  symmetric int8 with scale fp16(absmax/127) per G columns),
                  the math of the I8 path; teacher-forced perplexity follows.
  --mode none   : plain F16 path (reference perplexity).

Both use the held-out text protocol of tools/hi2 (samples x seq_len tokens
after skip_tokens).

  python actq_study.py --model DIR --mode fq --group 64 --out res.json
"""

import argparse
import json
import math
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402

STATS: dict = {}
CFG: dict = {}


def quant(x: torch.Tensor, g: int) -> torch.Tensor:
    """Symmetric int8 fake quantization with one fp16 scale per g columns
    (g = 0: per token), as the I8 activation pre-pass computes it."""
    m, k = x.shape
    g = k if g == 0 else g
    xg = x.reshape(m, k // g, g)
    sx = (xg.abs().amax(-1, keepdim=True) / 127.0).to(torch.float16).float()
    safe = torch.where(sx > 0, sx, torch.ones_like(sx))
    q = torch.where(sx > 0, torch.round(xg / safe).clamp(-127, 127), 0.0)
    return (q * sx).reshape(m, k)


def patch() -> None:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization import rdna_hint_flex as rhf
    from vllm.model_executor.layers.quantization.utils import (
        rdna_hint_flex_utils as hu,
    )

    orig = rhf.RdnaHintFlexLinearMethod.apply

    def weight(seg) -> torch.Tensor:
        r = seg["row_nbits"].numel()
        k = seg["had"].numel() * 16
        w = torch.empty(r, k, dtype=torch.float16, device=seg["codes"].device)
        ops.rdna_hint_flex_dequant_rdna2(
            seg["codes"], seg["code_off"], seg["blk_params"], seg["param_off"],
            seg["row_consts"], seg["row_nbits"], seg["row_block"], w,
        )  # fmt: skip
        return w.float()

    def apply(self, layer, x, bias=None):
        mode = CFG["mode"]
        if mode == "none":
            return orig(self, layer, x, bias)
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if mode == "stats":
            y = orig(self, layer, x, bias)
            name = getattr(layer, "prefix", "") or str(id(layer))
            for i, seg in enumerate(layer._rdna_hint_flex_segments):
                xt = hu.act_transform_f32(x2, seg)
                w = weight(seg)
                yr = xt @ w.T
                for g in CFG["groups"]:
                    d = xt - quant(xt, g)
                    st = STATS.setdefault(f"{name}#{i}", {}).setdefault(
                        str(g), [0.0, 0.0, 0.0, 0.0]
                    )
                    st[0] += float((d * d).sum())
                    st[1] += float((xt * xt).sum())
                    eo = d @ w.T
                    st[2] += float((eo * eo).sum())
                    st[3] += float((yr * yr).sum())
            return y
        outs = []
        for seg in layer._rdna_hint_flex_segments:
            xt = hu.act_transform_f32(x2, seg)
            outs.append((quant(xt, CFG["group"]) @ weight(seg).T).to(x.dtype))
        y = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        if bias is not None:
            y = y + bias
        return y.reshape(*shape[:-1], y.shape[-1])

    rhf.RdnaHintFlexLinearMethod.apply = apply


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["stats", "fq", "none"], required=True)
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--groups", default="16,32,64,0")
    p.add_argument("--text", default=os.path.expanduser(
        "~/w4a8_runs/hadamard-int2/study_text_all.txt"))  # fmt: skip
    p.add_argument("--skip-tokens", type=int, default=800000)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--language-model-only", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    CFG.update(mode=a.mode, group=a.group,
               groups=[int(g) for g in a.groups.split(",")])  # fmt: skip
    patch()
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(a.model)
    with open(a.text) as f:
        ids = tok(f.read(), add_special_tokens=False).input_ids[a.skip_tokens :]
    seqs = [ids[i * a.seq_len : (i + 1) * a.seq_len] for i in range(a.samples)]
    llm = LLM(model=a.model, dtype="float16", max_model_len=4096,
              gpu_memory_utilization=0.6, enforce_eager=True, seed=0,
              max_num_seqs=1, language_model_only=a.language_model_only,
              enable_prefix_caching=False)  # fmt: skip
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    nll, n, lps = 0.0, 0, []
    for q in seqs:
        o = llm.generate([TokensPrompt(prompt_token_ids=q)], sp)[0]
        for i, d in enumerate(o.prompt_logprobs):
            if i == 0:
                continue
            lp = d[q[i]].logprob
            lps.append(lp)
            nll -= lp
            n += 1
    res = {"model": a.model, "mode": a.mode, "group": a.group, "tokens": n,
           "ppl": math.exp(nll / n), "prompt_logprobs": lps}  # fmt: skip
    if a.mode == "stats":
        per = {}
        pooled = {}
        for name, d in STATS.items():
            for g, (ea, na, eo, no) in d.items():
                per.setdefault(g, []).append((math.sqrt(ea / na), math.sqrt(eo / no)))
                pp = pooled.setdefault(g, [0.0, 0.0, 0.0, 0.0])
                for j, v in enumerate((ea, na, eo, no)):
                    pp[j] += v
        res["act_out_error"] = {
            g: {
                "mean_act": sum(v[0] for v in vals) / len(vals),
                "mean_out": sum(v[1] for v in vals) / len(vals),
                "pooled_act": math.sqrt(pooled[g][0] / pooled[g][1]),
                "pooled_out": math.sqrt(pooled[g][2] / pooled[g][3]),
                "matrices": len(vals),
            }
            for g, vals in per.items()
        }
        res["per_input"] = {
            name: {
                g: {"act": math.sqrt(v[0] / v[1]), "out": math.sqrt(v[2] / v[3])}
                for g, v in d.items()
            }
            for name, d in STATS.items()
        }
    with open(a.out, "w") as f:
        json.dump(res, f)
    print(json.dumps({k: v for k, v in res.items() if k != "prompt_logprobs"}))


if __name__ == "__main__":
    main()
