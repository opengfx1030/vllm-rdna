#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The hi2 bench suite with an agent system prompt in front of every chat prompt.

Agent traffic starts every request with a long system prompt, so even a short
task arrives as a long prefill: on this fork a prompt of 512+ tokens takes the
prefill-only kernels (W4A8 MoE from 512 tokens per call). This wrapper runs the
hi2 checkout's ``bench/run.py`` and ``bench/tfkld.py`` unchanged, after putting
``--system`` (placeholders filled) at the head of the first system message of
each chat task (a new system message when the task has none). Text-completion
prompts (``mc``) are left alone. The system prompt's hash joins each task's
fingerprint, so a results file never mixes prompts.

    python agentic_bench.py --hi2 HI2_CHECKOUT --system cline_act.md run \\
        --url http://127.0.0.1:18096 --model flash-next --tasks gsm8k,toolcall \\
        --limit 120 --workers 8 --out A/bench.json
    python agentic_bench.py --hi2 HI2_CHECKOUT --system cline_act.md tfkld score \\
        --url http://127.0.0.1:18096 --model flash-next \\
        --ref-results A/bench.json --out B/lp.json
    python agentic_bench.py --hi2 HI2_CHECKOUT tokens A/bench.json

``tokens`` prints the prompt and completion token counts per task from the
server's ``usage`` (record them next to the results).
"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

FILL = {
    "{{PLATFORM_NAME}}": "linux",
    "{{CURRENT_DATE}}": "2026-11-04",
    "{{IDE_NAME}}": "Visual Studio Code",
    "{{CWD}}": "/home/user/project",
    "{{CLINE_RULES}}": "",
    "{{CLINE_METADATA}}": "",
}


def load_system(path: Path) -> str:
    text = path.read_text()
    for key, value in FILL.items():
        text = text.replace(key, value)
    return text.rstrip() + "\n"


def prefix_tasks(registry, system: str) -> None:
    tag = hashlib.sha256(system.encode()).hexdigest()[:16]
    for name in registry.NAMES:
        mod = registry.get(name)
        base_prompt = mod.prompt

        def prompt(item, base_prompt=base_prompt):
            msgs = base_prompt(item)
            if isinstance(msgs, str):
                return msgs
            msgs = [dict(m) for m in msgs]
            if msgs and msgs[0]["role"] == "system":
                msgs[0]["content"] = system + "\n" + msgs[0]["content"]
            else:
                msgs.insert(0, {"role": "system", "content": system})
            return msgs

        base_fp = getattr(mod, "fingerprint", None)

        def fingerprint(base_fp=base_fp):
            return f"{base_fp() if base_fp else ''}+system:{tag}"

        mod.prompt = prompt
        mod.fingerprint = fingerprint


def tokens(path: Path) -> int:
    run = json.loads(path.read_text())
    print("| task | items | prompt tokens min / median / max | completion mean |")
    print("|---|---:|---|---:|")
    for name, rows in run["results"].items():
        usage = [r.get("usage") or {} for r in rows.values()]
        prompt = [u["prompt_tokens"] for u in usage if "prompt_tokens" in u]
        done = [u["completion_tokens"] for u in usage if "completion_tokens" in u]
        if not prompt:
            continue
        print(
            f"| {name} | {len(prompt)} | {min(prompt)} / "
            f"{int(statistics.median(prompt))} / {max(prompt)} | "
            f"{statistics.mean(done) if done else float('nan'):.0f} |"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hi2", type=Path, required=True, help="hi2 checkout root")
    ap.add_argument("--system", type=Path, help="system prompt file (run, tfkld)")
    ap.add_argument("cmd", choices=("run", "tfkld", "tokens"))
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    sys.path.insert(0, str(a.hi2.resolve()))
    if a.cmd == "tokens":
        return tokens(Path(a.rest[0]))
    if a.system is None:
        ap.error(f"{a.cmd} needs --system")
    from bench import tasks as registry

    prefix_tasks(registry, load_system(a.system))
    if a.cmd == "run":
        from bench import run

        return run.main(a.rest)
    from bench import tfkld

    return tfkld.main(a.rest)


if __name__ == "__main__":
    sys.exit(main())
