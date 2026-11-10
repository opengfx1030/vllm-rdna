#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only chunk-boundary simulator of the DisaggRework spike
(docs/rdna2/disagg-g0-2026-11-06.md).

For a prompt of N tokens it prints where the prefill chunks end for

* ``B``: the monolithic engine (no KV connector: lookahead hashes, no EAGLE block
  drop) prefilling the whole prompt; only the chunks that end by N-1 are compared
  because the decode side recomputes the last token;
* ``P``: the NIXL prefiller as it was (legacy hashes + EAGLE block drop, prompt cut to
  N-1 tokens);
* ``PM``: the prefiller with ``prefill_monolithic_split`` (same cut request, split as
  the whole N-token prompt);
* ``D``: the decode side's tail chunk (computed = N-1).

The real ``Scheduler._mamba_block_aligned_split``, ``replay_boundaries`` and
``get_mamba_prefill_checkpoint_position`` are extracted from the source tree with
``ast`` and executed on stubs, because ``import vllm`` is not possible without the
GPU stack (triton/LLVM option clash on the CPU target).

  g0r_sim_chunks.py TREE show 1025,2049,... [budget]
  g0r_sim_chunks.py TREE sweep LO HI STEP [budget ...]

``budget`` is the tokens per step (default 2048, ``max_num_scheduled_tokens`` of the
Flash-Next MTP2 recipe); the block is the recipe's 1024 for the mamba, scheduler and
hash block.
"""

from __future__ import annotations

import ast
import sys
from types import SimpleNamespace
from typing import Any

BLOCK = 1024


def extract(path: str, names: set[str], in_class: str | None = None) -> str:
    """Source of the module-level (or ``in_class``) functions called ``names``."""
    with open(path) as f:
        src = f.read()
    nodes: list[ast.stmt] = ast.parse(src).body
    if in_class:
        cls = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == in_class
        )
        nodes = cls.body
    out = []
    for n in nodes:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            seg = ast.get_source_segment(src, n) or ""
            if in_class:  # dedent one level so it execs as a plain function
                seg = "\n".join(
                    line[4:] if line.startswith("    ") else line
                    for line in seg.splitlines()
                )
            out.append(seg)
    return "\n\n".join(out)


def load_functions(tree: str) -> dict[str, Any]:
    ns: dict[str, Any] = {}
    exec("def cdiv(a, b):\n    return -(a // -b)\n", ns)
    exec(
        extract(
            f"{tree}/vllm/v1/kv_cache_interface.py",
            {
                "get_mamba_prefill_checkpoint_position",
                "is_mamba_prefill_checkpoint_valid",
            },
        ),
        ns,
    )
    exec(
        extract(f"{tree}/vllm/v1/core/kv_cache_coordinator.py", {"replay_boundaries"}),
        ns,
    )
    split = extract(
        f"{tree}/vllm/v1/core/sched/scheduler.py",
        {"_mamba_block_aligned_split"},
        "Scheduler",
    ).replace("def _mamba_block_aligned_split(", "def split_fn(")
    exec(split, ns)
    return ns


def make_stub(
    fns: dict[str, Any],
    *,
    eagle_drop: bool,
    lookahead: int,
    budget: int,
    monolithic_split: bool = False,
) -> SimpleNamespace:
    replay_boundaries = fns["replay_boundaries"]
    coordinator = SimpleNamespace(
        scheduler_block_size=BLOCK,
        get_replay_boundaries=lambda request: replay_boundaries(
            request.num_prompt_tokens, BLOCK, use_eagle=eagle_drop
        ),
    )
    return SimpleNamespace(
        block_size=BLOCK,
        cache_config=SimpleNamespace(block_size=BLOCK),
        use_eagle_block_drop=eagle_drop,
        num_hash_lookahead_tokens=lookahead,
        # what a connector engine records of the monolithic engine (Flash-Next MTP2)
        monolithic_eagle_block_drop=False,
        monolithic_hash_lookahead_tokens=1,
        prefill_monolithic_split=monolithic_split,
        max_num_scheduled_tokens=budget,
        mamba_state_block_size=BLOCK,
        kv_cache_manager=SimpleNamespace(coordinator=coordinator),
        _effective_lpt=lambda: 0,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        mamba_shared_prefix_checkpoint=False,
        hash_block_size=BLOCK,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_prefill_checkpoint_alignment=None,
    )


def chunks(
    fns: dict[str, Any],
    prompt_len: int,
    stub: SimpleNamespace,
    start: int = 0,
    params: dict[str, Any] | None = None,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    pos = start
    budget = stub.max_num_scheduled_tokens
    while pos < prompt_len and len(out) < 400:
        req = SimpleNamespace(
            num_computed_tokens=pos,
            num_prompt_tokens=prompt_len,
            num_tokens=prompt_len,
            shared_prefix_boundary=0,
            kv_transfer_params=params,
        )
        n = fns["split_fn"](stub, req, min(prompt_len - pos, budget))
        if n <= 0:
            out.append((pos, -1))
            break
        out.append((pos, pos + n))
        pos += n
    return out


def fmt(cs: list[tuple[int, int]]) -> str:
    return " ".join(f"[{a},{b})" for a, b in cs)


def variants(fns: dict[str, Any], budget: int) -> tuple[Any, Any, Any]:
    base = make_stub(fns, eagle_drop=False, lookahead=1, budget=budget)
    legacy = make_stub(fns, eagle_drop=True, lookahead=0, budget=budget)
    mono = make_stub(
        fns, eagle_drop=True, lookahead=0, budget=budget, monolithic_split=True
    )
    return base, legacy, mono


def run(fns: dict[str, Any], n: int, stubs: tuple[Any, Any, Any]) -> dict[str, Any]:
    base, legacy, mono = stubs
    cb = chunks(fns, n, base)
    cut = {"_p_side_truncated": True}
    return {
        "B": cb,
        "B_to_N-1": [(a, min(b, n - 1)) for a, b in cb if a < n - 1],
        "P": chunks(fns, n - 1, legacy, params=cut),
        "PM": chunks(fns, n - 1, mono, params={**cut, "_p_side_backoff": 1}),
        "D": chunks(fns, n, legacy, start=n - 1),
        "B_tail": [c for c in cb if c[0] >= n - 1],
    }


def main() -> int:
    tree, mode = sys.argv[1], sys.argv[2]
    fns = load_functions(tree)
    if mode == "show":
        lengths = [int(x) for x in sys.argv[3].split(",")]
        budget = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
        stubs = variants(fns, budget)
        for n in lengths:
            r = run(fns, n, stubs)
            print(f"N={n} budget={budget}")
            print(f"  B[:N-1]  {fmt(r['B_to_N-1'])}")
            print(f"  P        {fmt(r['P'])}   same_as_B={r['P'] == r['B_to_N-1']}")
            print(f"  PM       {fmt(r['PM'])}   same_as_B={r['PM'] == r['B_to_N-1']}")
            print(f"  D-tail   {fmt(r['D'])}   B-tail={fmt(r['B_tail'])}")
        return 0
    lo, hi, step = (int(x) for x in sys.argv[3:6])
    for budget in [int(x) for x in sys.argv[6:]] or [2048]:
        stubs = variants(fns, budget)
        total = bad_p = bad_pm = bad_tail = 0
        first_bad: list[int] = []
        for n in range(lo, hi + 1, step):
            r = run(fns, n, stubs)
            total += 1
            bad_p += r["P"] != r["B_to_N-1"]
            if r["PM"] != r["B_to_N-1"]:
                bad_pm += 1
                if len(first_bad) < 5:
                    first_bad.append(n)
            bad_tail += r["D"] != r["B_tail"]
        print(
            f"budget={budget} lengths={total} P!=B:{bad_p} PM!=B:{bad_pm} "
            f"D-tail!=B-tail:{bad_tail} first_PM_mismatch={first_bad}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
