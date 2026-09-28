# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G1: accuracy cost of int8 FFN activations, measured without a new kernel.

Forward pre-hooks fake-quantize the input of every ``mlp.gate_up_proj`` and
``mlp.down_proj`` to vLLM's per-token int8 (and back) before the unchanged
W4A16 GEMM runs. A W4A8 kernel computes the same thing up to fp rounding,
so the perplexity / GSM8K deltas measured here are what W4A8 would cost.
TP-sharded ``down_proj`` inputs are quantized per rank, as a real kernel
would. Fused MoE experts are not covered (their down input is internal).
``--act-group-size G`` models the ``*_ag`` kernel configs instead: one scale
per (token, G inputs), with G the checkpoint's weight group size.

Run on the V620 box with the environment you serve with; the engine is
eager because hooks do not run inside captured graphs::

    python -m benchmarks.kernels.w4a8_sdot4_explore.fakequant_eval \\
        --model /models/Qwen3.8-27B-AWQ-INT4 --tp 2 \\
        --ppl-file benchmarks/sonnet.txt --gsm8k 500 --json g1.json
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
from pathlib import Path

LAYERS = ("mlp.gate_up_proj", "mlp.down_proj")


def fake_quant_int8(x, group_size: int = 0, clip_frac: float = 1.0):
    """int8 round trip matching ``dynamic_scaled_int8_quant`` (per token), or
    with one scale per (token, ``group_size`` inputs) when it is non-zero.

    ``clip_frac < 1`` saturates at ``clip_frac * absmax`` (absolute clipping):
    the extreme values lose resolution so the rest of the row gains it.
    """
    import torch

    xf = x.float()
    if group_size and xf.shape[-1] % group_size:
        raise ValueError(
            f"activation group {group_size} does not divide K={xf.shape[-1]} "
            "(the per-rank sharded dim); pick a divisor of the weight group"
        )
    if group_size:
        xf = xf.unflatten(-1, (-1, group_size))
    thr = xf.abs().amax(dim=-1, keepdim=True) * clip_frac
    inv = torch.where(thr > 0, 127.0 / thr, torch.zeros_like(thr))
    q = torch.round(xf * inv).clamp_(-128, 127) * (thr / 127.0)
    return (q.flatten(-2) if group_size else q).to(x.dtype)


def _pre_hook(module, args):
    if not module._w4a8_fq:
        return None
    x = args[0]
    if x.numel() // x.shape[-1] < module._w4a8_min_rows:
        return None
    module._w4a8_calls += 1
    return (
        fake_quant_int8(x, module._w4a8_group_size, module._w4a8_clip),
        *args[1:],
    )


def install_hooks(
    model,
    layers: tuple[str, ...],
    min_rows: int,
    group_size: int = 0,
    clip_frac: float = 1.0,
) -> int:
    count = 0
    for name, module in model.named_modules():
        if name.endswith(layers):
            module._w4a8_fq = False
            module._w4a8_min_rows = min_rows
            module._w4a8_group_size = group_size
            module._w4a8_clip = clip_frac
            module._w4a8_calls = 0
            module.register_forward_pre_hook(_pre_hook)
            count += 1
    return count


def set_fake_quant(model, enabled: bool) -> int:
    """Toggles the hooks; returns and resets the fake-quant call count."""
    calls = 0
    for module in model.modules():
        if hasattr(module, "_w4a8_fq"):
            module._w4a8_fq = enabled
            calls += module._w4a8_calls
            module._w4a8_calls = 0
    return calls


def perplexity(llm, text: str, ctx: int, max_windows: int) -> tuple[float, int]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    ids = llm.get_tokenizer().encode(text)
    windows = [ids[i : i + ctx] for i in range(0, len(ids), ctx)][:max_windows]
    windows = [w for w in windows if len(w) > 1]
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=w) for w in windows], params, use_tqdm=False
    )
    nll, count = 0.0, 0
    for out, w in zip(outs, windows):
        for pos in range(1, len(w)):
            nll -= out.prompt_logprobs[pos][w[pos]].logprob
            count += 1
    return math.exp(nll / count), count


def evaluate(llm, args) -> dict:
    result: dict = {}
    if args.ppl_file:
        text = Path(args.ppl_file).read_text()
        result["ppl"], result["ppl_tokens"] = perplexity(
            llm, text, args.ppl_ctx, args.ppl_windows
        )
    if args.gsm8k:
        from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline

        gsm = evaluate_gsm8k_offline(llm, num_questions=args.gsm8k, num_shots=5)
        result["gsm8k"] = float(gsm["accuracy"])
        result["gsm8k_invalid"] = float(gsm["invalid_rate"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--llm-kwargs", default="{}", help="extra LLM(...) JSON")
    parser.add_argument("--layers", default=",".join(LAYERS))
    parser.add_argument(
        "--min-rows",
        type=int,
        default=0,
        help="only quantize inputs with at least this many tokens "
        "(e.g. 257 to model prefill-only W4A8)",
    )
    parser.add_argument(
        "--act-group-size",
        type=int,
        default=0,
        help="one int8 scale per (token, this many inputs), as the *_ag kernel "
        "configs do with the weight group size; 0 = per token",
    )
    parser.add_argument(
        "--clip-frac",
        type=float,
        default=1.0,
        help="saturate the int8 activation grid at clip_frac*absmax (1.0 = no "
        "clipping); clipping trades the extremes for a finer step on the rest",
    )
    parser.add_argument("--ppl-file", help="text for perplexity")
    parser.add_argument("--ppl-ctx", type=int, default=2048)
    parser.add_argument("--ppl-windows", type=int, default=16)
    parser.add_argument("--gsm8k", type=int, default=0, help="questions (0 = skip)")
    parser.add_argument("--max-ppl-rel", type=float, default=0.02)
    parser.add_argument("--max-gsm8k-drop", type=float, default=0.01)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not (args.ppl_file or args.gsm8k):
        parser.error("pass --ppl-file and/or --gsm8k")

    # apply_model ships the hook functions to TP workers by pickling.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "enforce_eager": True,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        **json.loads(args.llm_kwargs),
    }
    llm = LLM(**llm_kwargs)
    layers = tuple(args.layers.split(","))
    hooked = llm.apply_model(
        functools.partial(
            install_hooks,
            layers=layers,
            min_rows=args.min_rows,
            group_size=args.act_group_size,
            clip_frac=args.clip_frac,
        )
    )
    if not sum(hooked):
        raise SystemExit(f"no module name ends with {layers}")

    llm.apply_model(functools.partial(set_fake_quant, enabled=False))
    base = evaluate(llm, args)
    llm.apply_model(functools.partial(set_fake_quant, enabled=True))
    fq = evaluate(llm, args)
    calls = sum(llm.apply_model(functools.partial(set_fake_quant, enabled=False)))
    if calls == 0:
        raise SystemExit("hooks never fired: graphs or compile bypassed them")

    verdicts = []
    rows = [["hooked modules per rank", hooked[0], ""], ["fake-quant calls", calls, ""]]
    if "ppl" in base:
        rel = fq["ppl"] / base["ppl"] - 1
        verdicts.append(rel <= args.max_ppl_rel)
        rows.append(
            [
                "perplexity",
                f"{base['ppl']:.4f}",
                f"{fq['ppl']:.4f} ({rel:+.2%}, limit +{args.max_ppl_rel:.0%})",
            ]
        )
    if "gsm8k" in base:
        drop = base["gsm8k"] - fq["gsm8k"]
        verdicts.append(drop <= args.max_gsm8k_drop)
        rows.append(
            [
                "GSM8K 5-shot",
                f"{base['gsm8k']:.4f}",
                f"{fq['gsm8k']:.4f} ({-drop:+.4f}, limit -{args.max_gsm8k_drop:.3f})",
            ]
        )
    verdict = "PASS" if all(verdicts) else "FAIL"
    scales = f"G={args.act_group_size}" if args.act_group_size else "per token"
    if args.clip_frac != 1.0:
        scales += f", clip={args.clip_frac}"
    print(f"\n### G1 fake-quant A8 on {','.join(layers)} ({args.model})\n")
    print(f"| metric | W4A16 | W4A16 + int8 FFN inputs ({scales}) |")
    print("| --- | --- | --- |")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} |")
    print(f"\nG1: **{verdict}**")
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "gate": "G1",
                    "model": args.model,
                    "tp": args.tp,
                    "layers": layers,
                    "min_rows": args.min_rows,
                    "act_group_size": args.act_group_size,
                    "clip_frac": args.clip_frac,
                    "baseline": base,
                    "fake_quant": fq,
                    "calls": calls,
                    "verdict": verdict,
                },
                indent=1,
            )
        )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
