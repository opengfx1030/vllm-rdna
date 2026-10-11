# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-level A/B of the exact dequant: perplexity and GSM8K for one build.

Run with identical arguments on the default build and on a build compiled
with -DVLLM_RDNA2_W4A16_EXACT_DEQUANT=1, then compare the two records:

    M=benchmarks.kernels.w4a16_exact_dequant.eval_model
    python -m $M run --label baked --model /models/Qwen3.8-27B-AWQ-INT4 \\
        --tp 2 --ppl-file wiki.test.raw --gsm8k 500 --json eval-baked.json
    python -m $M run --label exact ... --json eval-exact.json
    python -m $M compare eval-baked.json eval-exact.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


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


def cmd_run(args) -> dict:
    from vllm import LLM

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **json.loads(args.llm_kwargs),
    )
    record: dict = {"label": args.label, "model": args.model, "tp": args.tp}
    if args.ppl_file:
        text = Path(args.ppl_file).read_text()
        record["ppl"], record["ppl_tokens"] = perplexity(
            llm, text, args.ppl_ctx, args.ppl_windows
        )
    if args.gsm8k:
        from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline

        gsm = evaluate_gsm8k_offline(llm, num_questions=args.gsm8k, num_shots=5)
        record["gsm8k"] = float(gsm["accuracy"])
        record["gsm8k_invalid"] = float(gsm["invalid_rate"])
    print(json.dumps(record, indent=1))
    return record


def cmd_compare(args) -> None:
    a, b = (json.loads(Path(p).read_text()) for p in (args.baseline, args.candidate))
    print(f"| metric | {a['label']} | {b['label']} | Δ |")
    print("| --- | ---: | ---: | ---: |")
    if "ppl" in a and "ppl" in b:
        rel = b["ppl"] / a["ppl"] - 1
        print(f"| perplexity | {a['ppl']:.4f} | {b['ppl']:.4f} | {rel:+.2%} |")
    if "gsm8k" in a and "gsm8k" in b:
        d = b["gsm8k"] - a["gsm8k"]
        print(f"| GSM8K 5-shot | {a['gsm8k']:.4f} | {b['gsm8k']:.4f} | {d:+.4f} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="perplexity and GSM8K on the installed build")
    run.add_argument("--label", required=True, help="e.g. baked or exact")
    run.add_argument("--model", required=True)
    run.add_argument("--tp", type=int, default=1)
    run.add_argument("--max-model-len", type=int, default=4096)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run.add_argument("--llm-kwargs", default="{}", help="extra LLM(...) JSON")
    run.add_argument("--ppl-file", help="text for perplexity (wikitext-2 test)")
    run.add_argument("--ppl-ctx", type=int, default=2048)
    run.add_argument("--ppl-windows", type=int, default=32)
    run.add_argument("--gsm8k", type=int, default=0, help="questions (0 = skip)")
    run.add_argument("--json", type=Path)
    cmp_ = sub.add_parser("compare", help="baseline record vs candidate record")
    cmp_.add_argument("baseline")
    cmp_.add_argument("candidate")
    args = parser.parse_args()
    if args.command == "run":
        if not (args.ppl_file or args.gsm8k):
            parser.error("pass --ppl-file and/or --gsm8k")
        record = cmd_run(args)
        if args.json:
            args.json.write_text(json.dumps(record, indent=1))
    else:
        cmd_compare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
