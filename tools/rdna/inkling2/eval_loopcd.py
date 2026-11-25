#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench, decode-speed sweep, and self-verifying eval for Inkling-2 LoopCD.

Supports three composable modes (``--speed`` and ``--eval`` share ONE ``LLM`` boot):

* ``--microbench``: standalone GPU kernel latency benchmark (no checkpoint load needed)
  using ``torch.cuda.Event(enable_timing=True)`` (50 warmup + 200 timed reps, reporting
  median & p95 µs) at ``Inkling-2-Mini`` shapes (``H=2304``, ``P=1`` loop block,
  ``n_blocks=3``, ``V=200058``):
    - ``inkling2_glue_loopcd_blend`` at ``T in (1, 3, 8, 16)``
    - ``inkling2_glue_loopcd_readout`` at ``T in (1, 3, 8)`` (``gate_flag`` 0 vs 1)
    - ``inkling2_loopcd_logits`` at ``B in (1, 3, 8)``, ``V=200058``
    - ``inkling2_linear_lm_head`` via ``LMHeadPack`` (``[200058, 2304]`` fp16) at
      ``M in (1, 3, 8)`` for the exact ``LoopCD-Logits`` weak-head cost breakdown.
* ``--speed``: boots ``vllm.LLM`` ONCE (``dtype="float16"``, text-only ``limit_mm``,
  ``FULL_AND_PIECEWISE`` HIP graphs, optional ``--dspark`` with ``ADAPTIVE=1`` schedule
  ``[[1,1,7],[2,2,2],[3,16,0]]`` and ``--draft-vocab``), runs a 4-prompt ``alpha=0.0``
  sanity gate, then sweeps LoopCD arms across batch sizes, reporting decode ``ms/tok``,
  aggregate ``tok/s``, and DSpark acceptance rate & ``tok/step``.
* ``--eval``: evaluates 20 self-verifying tricky problems (12 multi-step Math /
  Combinatorics / Number Theory / Logic problems ending with ``\\boxed{N}`` + 8
  algorithmic Python coding problems verified against unit tests in an isolated
  subprocess with a timeout) across all requested LoopCD arms in ONE ``LLM`` boot.

Examples::

    HIP_VISIBLE_DEVICES=1 python tools/rdna/inkling2/eval_loopcd.py --microbench
    HIP_VISIBLE_DEVICES=1 python tools/rdna/inkling2/eval_loopcd.py \\
        --ckpt ~/hfcache/inkling2-mini --speed --eval --dspark --json loopcd.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_DSPARK_SCHEDULE = "[[1,1,7],[2,2,2],[3,16,0]]"

DEFAULT_ARMS: tuple[str, ...] = (
    "baseline",
    "hidden_a0.20",
    "hidden_a0.35",
    "hidden_noblock_a0.25",
    "logits_a0.25",
    "logits_a0.50",
    "logits_adap_a0.25",
    "logits_adap_a0.50",
    "logits_p0_adap_a0.35",
)

PRESET_ARMS: dict[str, dict[str, Any] | None] = {
    "baseline": None,
    "hidden_a0.20": {
        "loopcd_mode": "hidden",
        "loopcd_alpha": 0.20,
        "loopcd_blend_blocks": True,
    },
    "hidden_a0.35": {
        "loopcd_mode": "hidden",
        "loopcd_alpha": 0.35,
        "loopcd_blend_blocks": True,
    },
    "hidden_noblock_a0.25": {
        "loopcd_mode": "hidden",
        "loopcd_alpha": 0.25,
        "loopcd_blend_blocks": False,
    },
    "logits_a0.25": {
        "loopcd_mode": "logits",
        "loopcd_alpha": 0.25,
        "loopcd_adaptive": False,
        "loopcd_beta": 0.1,
        "loopcd_weak_pass": -1,
    },
    "logits_a0.50": {
        "loopcd_mode": "logits",
        "loopcd_alpha": 0.50,
        "loopcd_adaptive": False,
        "loopcd_beta": 0.1,
        "loopcd_weak_pass": -1,
    },
    "logits_adap_a0.25": {
        "loopcd_mode": "logits",
        "loopcd_alpha": 0.25,
        "loopcd_adaptive": True,
        "loopcd_beta": 0.1,
        "loopcd_weak_pass": -1,
    },
    "logits_adap_a0.50": {
        "loopcd_mode": "logits",
        "loopcd_alpha": 0.50,
        "loopcd_adaptive": True,
        "loopcd_beta": 0.1,
        "loopcd_weak_pass": -1,
    },
    "logits_p0_adap_a0.35": {
        "loopcd_mode": "logits",
        "loopcd_alpha": 0.35,
        "loopcd_adaptive": True,
        "loopcd_beta": 0.1,
        "loopcd_weak_pass": 0,
    },
}

SPEED_PROMPTS: tuple[str, ...] = (
    (
        "Explain how a B+ tree maintains balanced leaf depth during node splits and "
        "merges, and walk through inserting 8 keys into an order-3 B+ tree."
    ),
    (
        "Write a self-contained Python implementation of Tarjan's strongly connected "
        "components algorithm with complexity analysis and edge-case tests."
    ),
    (
        "Derive the closed-form generating function of the Fibonacci sequence and use "
        "partial fractions to recover Binet's formula step by step."
    ),
    (
        "Compare write-ahead logging (WAL) and shadow paging in database recovery, "
        "detailing the ARIES analysis, redo, and undo phases."
    ),
    (
        "A factory operates three machines with independent exponential failure rates. "
        "Set up the continuous-time Markov chain for one repair technician and find "
        "the steady-state availability."
    ),
    (
        "Implement a thread-safe bounded ring buffer in C++20 using condition "
        "variables, and explain why Mesa monitor semantics require a predicate loop."
    ),
    (
        "Prove that every finite integral domain is a field, then give an explicit "
        "construction of GF(9) as a quotient of GF(3)[x]."
    ),
    (
        "Design a rate limiter supporting burst capacity and steady refill across "
        "distributed API gateways; compare token bucket, leaky bucket, and sliding "
        "window log."
    ),
)


def resolve_arm(name: str) -> dict[str, Any] | None:
    """Resolve an arm name to ``SamplingParams.extra_args`` (``None`` for baseline).

    Accepts preset names as well as dynamic patterns such as
    ``hidden_[noblock_]a<float>`` and ``logits_[p<int>_][adap_]a<float>[_b<float>]``.
    """
    if name in PRESET_ARMS:
        preset = PRESET_ARMS[name]
        return None if preset is None else dict(preset)
    m_hid = re.fullmatch(r"hidden_(noblock_)?a([0-9]*\.?[0-9]+)", name)
    if m_hid:
        return {
            "loopcd_mode": "hidden",
            "loopcd_alpha": float(m_hid.group(2)),
            "loopcd_blend_blocks": m_hid.group(1) is None,
        }
    m_log = re.fullmatch(
        r"logits_(?:p(-?\d+)_)?(adap_)?a([0-9]*\.?[0-9]+)(?:_b([0-9]*\.?[0-9]+))?",
        name,
    )
    if m_log:
        return {
            "loopcd_mode": "logits",
            "loopcd_alpha": float(m_log.group(3)),
            "loopcd_adaptive": m_log.group(2) is not None,
            "loopcd_beta": float(m_log.group(4)) if m_log.group(4) is not None else 0.1,
            "loopcd_weak_pass": int(m_log.group(1))
            if m_log.group(1) is not None
            else -1,
        }
    raise ValueError(
        f"Unknown LoopCD arm {name!r}. Known presets: {', '.join(PRESET_ARMS)}"
    )


# ===========================================================================
# 20 Self-Verifying Reasoning & Coding Problems (12 Math + 8 Python Coding)
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class EvalProblem:
    id: str
    category: str  # "math" or "code"
    prompt: str
    expected_int: int | None = None
    entry_point: str | None = None
    test_harness: str | None = None
    reference_solution: str | None = None


MATH_SUFFIX = (
    "\n\nReason carefully step by step, and put your final integer answer inside "
    "\\boxed{N} at the very end of your response."
)

CODE_SUFFIX_TMPL = (
    "\n\nProvide your final implementation inside a single ```python ... ``` code "
    "block defining `{sig}` using only the Python standard library."
)

EVAL_PROBLEMS: tuple[EvalProblem, ...] = (
    # --- 12 Multi-Step Math / AIME-style / Combinatorics / Number Theory / Logic ---
    EvalProblem(
        id="math_01_constrained_derangement",
        category="math",
        prompt=(
            "How many permutations (pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7) of "
            "(1, 2, 3, 4, 5, 6, 7) have **exactly two** fixed points (indices i such "
            "that pi_i = i) AND satisfy pi_1 != 2?" + MATH_SUFFIX
        ),
        expected_int=814,
    ),
    EvalProblem(
        id="math_02_lcm_gcd_triples",
        category="math",
        prompt=(
            "Find the number of ordered triples of positive integers (a, b, c) such "
            "that lcm(a, b, c) = 1800 and gcd(a, b, c) = 1." + MATH_SUFFIX
        ),
        expected_int=2592,
    ),
    EvalProblem(
        id="math_03_lattice_avoid_two",
        category="math",
        prompt=(
            "Consider lattice paths in the plane from (0, 0) to (8, 6) that each step "
            "move either (+1, 0) or (0, +1). How many such paths pass through "
            "**neither** (3, 2) **nor** (5, 4)?" + MATH_SUFFIX
        ),
        expected_int=1083,
    ),
    EvalProblem(
        id="math_04_idempotent_mod_180",
        category="math",
        prompt=(
            "How many integers x with 1 <= x <= 2025 satisfy x^2 == x (mod 180)?"
            + MATH_SUFFIX
        ),
        expected_int=91,
    ),
    EvalProblem(
        id="math_05_factorial_base24_zeros",
        category="math",
        prompt=(
            "When 500! (500 factorial) is written in base 24, in how many consecutive "
            "trailing zeros does it end?" + MATH_SUFFIX
        ),
        expected_int=164,
    ),
    EvalProblem(
        id="math_06_ternary_no_adj_zeros_mod3",
        category="math",
        prompt=(
            "How many sequences of length 8 formed from the alphabet {0, 1, 2} have "
            "**no two adjacent 0s** AND have the sum of all 8 terms divisible by 3?"
            + MATH_SUFFIX
        ),
        expected_int=1114,
    ),
    EvalProblem(
        id="math_07_nondecreasing_middle_gap",
        category="math",
        prompt=(
            "How many 6-tuples of integers (x_1, x_2, x_3, x_4, x_5, x_6) satisfy "
            "1 <= x_1 <= x_2 <= x_3 < x_4 <= x_5 <= x_6 <= 10 (note the strict "
            "inequality x_3 < x_4 in the middle)?" + MATH_SUFFIX
        ),
        expected_int=3003,
    ),
    EvalProblem(
        id="math_08_cubic_roots_product",
        category="math",
        prompt=(
            "Let r_1, r_2, r_3 be the three complex roots of "
            "x^3 - 7x^2 + 15x - 23 = 0. "
            "Compute the exact integer value of (r_1^2 + 4)(r_2^2 + 4)(r_3^2 + 4)."
            + MATH_SUFFIX
        ),
        expected_int=509,
    ),
    EvalProblem(
        id="math_09_twelve_divisors_mult15",
        category="math",
        prompt=(
            "Find the sum of all positive integers n <= 500 that are divisible by 15 "
            "AND have **exactly 12** positive divisors." + MATH_SUFFIX
        ),
        expected_int=1110,
    ),
    EvalProblem(
        id="math_10_tournament_3cycles",
        category="math",
        prompt=(
            "In a 9-player round-robin tournament, every pair of distinct players "
            "plays exactly one game with no ties. The win counts (out-degrees) of the "
            "9 players are (1, 2, 3, 4, 4, 4, 5, 6, 7). How many directed 3-cycles "
            "(unordered triples {A, B, C} where A beats B, B beats C, and C beats A) "
            "exist in this tournament?" + MATH_SUFFIX
        ),
        expected_int=16,
    ),
    EvalProblem(
        id="math_11_digit_sum_two_carries",
        category="math",
        prompt=(
            "For a positive integer n, let S(n) denote the sum of its base-10 digits. "
            "How many integers n with 1 <= n <= 9999 satisfy S(n) = 20 and "
            "S(n + 37) = 12?" + MATH_SUFFIX
        ),
        expected_int=205,
    ),
    EvalProblem(
        id="math_12_wheel_graph_colorings",
        category="math",
        prompt=(
            "The wheel graph W_6 has 6 labeled vertices: a 5-cycle v_1 - v_2 - v_3 - "
            "v_4 - v_5 - v_1 and a central hub vertex u connected by an edge to each "
            "of v_1, v_2, v_3, v_4, v_5. How many proper vertex colorings of this "
            "labeled graph exist using at most 5 available colors (adjacent vertices "
            "must have different colors)?" + MATH_SUFFIX
        ),
        expected_int=1200,
    ),
    # --- 8 Algorithmic Python Coding Problems ---
    EvalProblem(
        id="code_01_min_groups_with_gap",
        category="code",
        entry_point="min_groups_with_gap",
        prompt=(
            "Write a Python function "
            "`min_groups_with_gap(intervals: list[tuple[int, int]], "
            "min_gap: int) -> int`.\n"
            "Each element of `intervals` is a closed integer interval `(start, end)` "
            "with `start <= end`, and `min_gap >= 1`. Partition all intervals into the "
            "minimum possible number of groups such that within any single group, "
            "every pair of distinct intervals `(s1, e1)` and `(s2, e2)` satisfies "
            "either `s2 >= e1 + min_gap` or `s1 >= e2 + min_gap`. Return `0` when "
            "`intervals` is empty."
            + CODE_SUFFIX_TMPL.format(
                sig=(
                    "def min_groups_with_gap("
                    "intervals: list[tuple[int, int]], min_gap: int) -> int"
                )
            )
        ),
        test_harness=textwrap.dedent(
            """
            assert min_groups_with_gap([], 3) == 0
            assert min_groups_with_gap([(1, 3), (5, 7), (9, 10)], 2) == 1
            assert min_groups_with_gap([(1, 3), (5, 7), (9, 10)], 3) == 2
            assert min_groups_with_gap([(1, 5), (2, 6), (6, 8), (9, 12)], 3) == 3
            assert min_groups_with_gap([(0, 0), (1, 1), (2, 2)], 1) == 1
            assert min_groups_with_gap([(0, 0), (1, 1), (2, 2)], 2) == 2
            assert min_groups_with_gap([(10, 20), (1, 5), (6, 9), (15, 18)], 2) == 2
            """
        ),
        reference_solution=textwrap.dedent(
            """
            import heapq

            def min_groups_with_gap(
                intervals: list[tuple[int, int]], min_gap: int
            ) -> int:
                if not intervals:
                    return 0
                heap: list[int] = []
                for s, e in sorted(intervals):
                    if heap and heap[0] + min_gap <= s:
                        heapq.heapreplace(heap, e)
                    else:
                        heapq.heappush(heap, e)
                return len(heap)
            """
        ),
    ),
    EvalProblem(
        id="code_02_longest_balanced_mod_subarray",
        category="code",
        entry_point="longest_balanced_modulo_subarray",
        prompt=(
            "Write a Python function "
            "`longest_balanced_modulo_subarray(nums: list[int], k: int) -> int`.\n"
            "Given a list of integers `nums` (which may contain negative integers, "
            "zeros, and positive integers; `len(nums)` up to `100_000`) and `k >= 1`, "
            "return the maximum length of a contiguous subarray `nums[i:j]` whose sum "
            "is divisible by `k` AND that contains **at least one** element `x` with "
            "`x % k != 0`. If no such subarray exists, return `0`."
            + CODE_SUFFIX_TMPL.format(
                sig=(
                    "def longest_balanced_modulo_subarray("
                    "nums: list[int], k: int) -> int"
                )
            )
        ),
        test_harness=textwrap.dedent(
            """
            assert longest_balanced_modulo_subarray([], 5) == 0
            assert longest_balanced_modulo_subarray([5, 10, -15, 0], 5) == 0
            assert longest_balanced_modulo_subarray([5, 2, 3, 10], 5) == 4
            assert longest_balanced_modulo_subarray([6, -2, -4, 12], 6) == 4
            assert longest_balanced_modulo_subarray([1, 2, 6, 12, 4, 2], 6) == 4
            assert longest_balanced_modulo_subarray([1, 5, 6, 12, 4, 2], 6) == 6
            assert longest_balanced_modulo_subarray([-7, 3, -2, 12, -6], 6) == 5
            assert longest_balanced_modulo_subarray([14, 3, 10, 7], 7) == 0
            assert longest_balanced_modulo_subarray([14, 3, 11, 7], 7) == 4
            """
        ),
        reference_solution=textwrap.dedent(
            """
            def longest_balanced_modulo_subarray(nums: list[int], k: int) -> int:
                earliest = {0: (0, 0)}
                pref = 0
                non_div = 0
                best = 0
                for j, x in enumerate(nums, 1):
                    pref = (pref + x) % k
                    if x % k != 0:
                        non_div += 1
                    if pref in earliest:
                        i, nd_i = earliest[pref]
                        if non_div > nd_i and j - i > best:
                            best = j - i
                    else:
                        earliest[pref] = (j, non_div)
                return best
            """
        ),
    ),
    EvalProblem(
        id="code_03_dag_longest_path_tolls",
        category="code",
        entry_point="longest_path_at_most_k_tolls",
        prompt=(
            "Write a Python function `longest_path_at_most_k_tolls(n: int, "
            "edges: list[tuple[int, int, int, bool]], src: int, dst: int, "
            "max_tolls: int) -> int | None`.\n"
            "A directed acyclic graph (DAG) has `n` vertices `0 .. n - 1`. Each "
            "directed edge `(u, v, weight, is_toll)` goes from `u` to `v` with "
            "integer `weight` (possibly negative, zero, or positive) and boolean "
            "`is_toll`. Return the **maximum** total weight of a directed path from "
            "`src` to `dst` using **at most** `max_tolls` edges where `is_toll` is "
            "`True`. Return `None` if no valid path exists (if `src == dst` and "
            "`max_tolls >= 0`, the empty path has weight `0`)."
            + CODE_SUFFIX_TMPL.format(
                sig=(
                    "def longest_path_at_most_k_tolls("
                    "n: int, edges: list[tuple[int, int, int, bool]], "
                    "src: int, dst: int, max_tolls: int) -> int | None"
                )
            )
        ),
        test_harness=textwrap.dedent(
            """
            e1 = [
                (0, 1, 10, True), (1, 3, 10, True),
                (0, 2, 5, False), (2, 3, 6, False),
            ]
            assert longest_path_at_most_k_tolls(4, e1, 0, 3, 1) == 11
            assert longest_path_at_most_k_tolls(4, e1, 0, 3, 2) == 20
            assert longest_path_at_most_k_tolls(4, e1, 0, 3, 0) == 11
            e2 = [(0, 1, -5, False), (1, 2, -3, True)]
            assert longest_path_at_most_k_tolls(3, e2, 0, 2, 1) == -8
            e3 = [(0, 1, -5, True), (1, 2, -3, True)]
            assert longest_path_at_most_k_tolls(3, e3, 0, 2, 1) is None
            assert longest_path_at_most_k_tolls(3, [(1, 2, 100, False)], 0, 0, 0) == 0
            assert longest_path_at_most_k_tolls(3, [(0, 1, 5, False)], 0, 2, 2) is None
            """
        ),
        reference_solution=textwrap.dedent(
            """
            import collections

            def longest_path_at_most_k_tolls(
                n: int,
                edges: list[tuple[int, int, int, bool]],
                src: int,
                dst: int,
                max_tolls: int,
            ) -> int | None:
                if max_tolls < 0:
                    return None
                adj = [[] for _ in range(n)]
                indeg = [0] * n
                for u, v, w, toll in edges:
                    adj[u].append((v, w, 1 if toll else 0))
                    indeg[v] += 1
                q = collections.deque(i for i in range(n) if indeg[i] == 0)
                topo = []
                while q:
                    u = q.popleft()
                    topo.append(u)
                    for v, _, _ in adj[u]:
                        indeg[v] -= 1
                        if indeg[v] == 0:
                            q.append(v)
                neg_inf = -10**18
                dp = [[neg_inf] * (max_tolls + 1) for _ in range(n)]
                dp[src][0] = 0
                for u in topo:
                    for t in range(max_tolls + 1):
                        if dp[u][t] == neg_inf:
                            continue
                        for v, w, c in adj[u]:
                            nt = t + c
                            if nt <= max_tolls and dp[u][t] + w > dp[v][nt]:
                                dp[v][nt] = dp[u][t] + w
                ans = max(dp[dst])
                return None if ans == neg_inf else ans
            """
        ),
    ),
    EvalProblem(
        id="code_04_osa_weighted_edit_distance",
        category="code",
        entry_point="osa_edit_distance",
        prompt=(
            "Write a Python function `osa_edit_distance(s: str, t: str, "
            "ins_cost: int = 1, del_cost: int = 1, sub_cost: int = 1, "
            "trans_cost: int = 1) -> int`.\n"
            "Compute the **Optimal String Alignment** (restricted Damerau-Levenshtein) "
            "distance from `s` to `t`, where no substring is edited more than once. "
            "The allowed operations are:\n"
            "- insert one character (`ins_cost`),\n"
            "- delete one character (`del_cost`),\n"
            "- substitute one character for a different character (`sub_cost`; equal "
            "characters cost `0`),\n"
            "- transpose two adjacent characters (`s[i-2] == t[j-1]` and "
            "`s[i-1] == t[j-2]` with `s[i-1] != s[i-2]`, transitioning from "
            "`(i-2, j-2)` to `(i, j)` at cost `trans_cost`)."
            + CODE_SUFFIX_TMPL.format(
                sig=(
                    "def osa_edit_distance(s: str, t: str, ins_cost: int = 1, "
                    "del_cost: int = 1, sub_cost: int = 1, trans_cost: int = 1) -> int"
                )
            )
        ),
        test_harness=textwrap.dedent(
            """
            assert osa_edit_distance("ca", "abc") == 3
            assert osa_edit_distance("abcd", "acbd", 5, 5, 5, 2) == 2
            assert osa_edit_distance("abcd", "acbd", 1, 1, 1, 10) == 2
            assert osa_edit_distance("", "xyz", 3, 2, 4, 1) == 9
            assert osa_edit_distance("xyz", "", 3, 2, 4, 1) == 6
            assert osa_edit_distance("kitten", "sitting", 2, 3, 1, 1) == 4
            assert osa_edit_distance("abba", "baab", 4, 4, 4, 1) == 2
            """
        ),
        reference_solution=textwrap.dedent(
            """
            def osa_edit_distance(
                s: str,
                t: str,
                ins_cost: int = 1,
                del_cost: int = 1,
                sub_cost: int = 1,
                trans_cost: int = 1,
            ) -> int:
                m, n = len(s), len(t)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                for i in range(1, m + 1):
                    dp[i][0] = i * del_cost
                for j in range(1, n + 1):
                    dp[0][j] = j * ins_cost
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        c = 0 if s[i - 1] == t[j - 1] else sub_cost
                        best = min(
                            dp[i - 1][j] + del_cost,
                            dp[i][j - 1] + ins_cost,
                            dp[i - 1][j - 1] + c,
                        )
                        if (
                            i >= 2
                            and j >= 2
                            and s[i - 1] == t[j - 2]
                            and s[i - 2] == t[j - 1]
                            and s[i - 1] != s[i - 2]
                        ):
                            best = min(best, dp[i - 2][j - 2] + trans_cost)
                        dp[i][j] = best
                return dp[m][n]
            """
        ),
    ),
    EvalProblem(
        id="code_05_int_expr_evaluator",
        category="code",
        entry_point="eval_int_expr",
        prompt=(
            "Write a Python function `eval_int_expr(expr: str) -> int` that evaluates "
            "an integer arithmetic expression supporting:\n"
            "- non-negative integer literals and arbitrary spaces,\n"
            "- parentheses `(` and `)`,\n"
            "- `+` and `-` (both binary addition/subtraction and unary `+` / `-`),\n"
            "- `*` and `//` (Python integer floor division, e.g. `-7 // 2 == -4`),\n"
            "- `**` (exponentiation: **right-associative** and binding **tighter** "
            "than unary `+`/`-`, like Python: `-2 ** 2 == -4`, `2 ** 3 ** 2 == 512`, "
            "`2 * -3 ** 2 == -18`; all exponents in tests are non-negative).\n"
            "Do NOT call `eval`, `exec`, `compile`, or `ast` (`eval`/`exec`/`compile` "
            "are disabled at test time)."
            + CODE_SUFFIX_TMPL.format(sig="def eval_int_expr(expr: str) -> int")
        ),
        test_harness=textwrap.dedent(
            """
            import builtins as _b
            for _fn in ("eval", "exec", "compile"):
                setattr(_b, _fn, None)
            assert eval_int_expr("-2 ** 2") == -4
            assert eval_int_expr("(-2) ** 2") == 4
            assert eval_int_expr("2 ** 3 ** 2") == 512
            assert eval_int_expr("(2 ** 3) ** 2") == 64
            assert eval_int_expr("-7 // 2 + 3 * -4") == -16
            assert eval_int_expr("100 - (3 + 2) * 2 ** (1 + 2) // 3") == 87
            assert eval_int_expr("--5 + -(-3 * 2 ** 2)") == 17
            """
        ),
        reference_solution=textwrap.dedent(
            """
            def eval_int_expr(expr: str) -> int:
                tokens = []
                i, n = 0, len(expr)
                while i < n:
                    if expr[i].isspace():
                        i += 1
                    elif expr[i].isdigit():
                        j = i
                        while j < n and expr[j].isdigit():
                            j += 1
                        tokens.append(int(expr[i:j]))
                        i = j
                    elif expr[i : i + 2] in ("**", "//"):
                        tokens.append(expr[i : i + 2])
                        i += 2
                    elif expr[i] in "+-*()":
                        tokens.append(expr[i])
                        i += 1
                    else:
                        raise ValueError(expr[i])
                pos = 0

                def parse_add() -> int:
                    nonlocal pos
                    val = parse_mul()
                    while pos < len(tokens) and tokens[pos] in ("+", "-"):
                        op = tokens[pos]
                        pos += 1
                        rhs = parse_mul()
                        val = val + rhs if op == "+" else val - rhs
                    return val

                def parse_mul() -> int:
                    nonlocal pos
                    val = parse_unary()
                    while pos < len(tokens) and tokens[pos] in ("*", "//"):
                        op = tokens[pos]
                        pos += 1
                        rhs = parse_unary()
                        val = val * rhs if op == "*" else val // rhs
                    return val

                def parse_unary() -> int:
                    nonlocal pos
                    if tokens[pos] == "-":
                        pos += 1
                        return -parse_unary()
                    if tokens[pos] == "+":
                        pos += 1
                        return parse_unary()
                    return parse_pow()

                def parse_pow() -> int:
                    nonlocal pos
                    base = parse_atom()
                    if pos < len(tokens) and tokens[pos] == "**":
                        pos += 1
                        exp = parse_unary()
                        return base ** exp
                    return base

                def parse_atom() -> int:
                    nonlocal pos
                    tok = tokens[pos]
                    if tok == "(":
                        pos += 1
                        val = parse_add()
                        assert tokens[pos] == ")"
                        pos += 1
                        return val
                    pos += 1
                    return int(tok)

                res = parse_add()
                assert pos == len(tokens)
                return res
            """
        ),
    ),
    EvalProblem(
        id="code_06_lfu_cache_simulation",
        category="code",
        entry_point="simulate_lfu_cache",
        prompt=(
            "Write a Python function `simulate_lfu_cache(capacity: int, "
            "ops: list[tuple]) -> list[int]`.\n"
            "Simulate an LFU (Least Frequently Used) cache with `capacity >= 0`:\n"
            "- `('get', key)`: if `key` is in the cache, increment its frequency by 1, "
            "mark it most recently used within its new frequency, and append its value "
            "to the returned list; otherwise append `-1`.\n"
            "- `('put', key, value)`: if `capacity == 0`, do nothing. If `key` is "
            "already present, update its value, increment its frequency by 1, and mark "
            "it most recently used within its new frequency. Otherwise, if the cache "
            "is full, evict the key with the lowest frequency (breaking ties by "
            "evicting the least recently used among keys of that lowest frequency), "
            "and insert `(key, value)` with frequency `1`."
            + CODE_SUFFIX_TMPL.format(
                sig=(
                    "def simulate_lfu_cache("
                    "capacity: int, ops: list[tuple]) -> list[int]"
                )
            )
        ),
        test_harness=textwrap.dedent(
            """
            assert simulate_lfu_cache(0, [("put", 1, 10), ("get", 1)]) == [-1]
            ops1 = [
                ("put", 1, 10), ("put", 2, 20), ("get", 1), ("put", 3, 30),
                ("get", 2), ("get", 3), ("put", 4, 40),
                ("get", 1), ("get", 3), ("get", 4),
            ]
            assert simulate_lfu_cache(2, ops1) == [10, -1, 30, -1, 30, 40]
            ops2 = [
                ("put", 1, 1), ("put", 1, 2), ("get", 1),
                ("put", 2, 3), ("get", 1), ("get", 2),
            ]
            assert simulate_lfu_cache(1, ops2) == [2, -1, 3]
            ops3 = [
                ("put", 1, 10), ("put", 2, 20), ("put", 3, 30),
                ("get", 1), ("get", 2), ("put", 2, 25),
                ("put", 4, 40), ("get", 3), ("get", 1), ("get", 2), ("get", 4),
            ]
            assert simulate_lfu_cache(3, ops3) == [10, 20, -1, 10, 25, 40]
            """
        ),
        reference_solution=textwrap.dedent(
            """
            import collections

            def simulate_lfu_cache(capacity: int, ops: list[tuple]) -> list[int]:
                out: list[int] = []
                vals: dict[int, int] = {}
                freqs: dict[int, int] = {}
                buckets: dict[int, collections.OrderedDict[int, None]] = (
                    collections.defaultdict(collections.OrderedDict)
                )
                min_freq = 0

                def touch(k: int) -> None:
                    nonlocal min_freq
                    f = freqs[k]
                    del buckets[f][k]
                    if not buckets[f] and min_freq == f:
                        min_freq += 1
                    freqs[k] = f + 1
                    buckets[f + 1][k] = None

                for op in ops:
                    if op[0] == "get":
                        k = op[1]
                        if k not in vals:
                            out.append(-1)
                        else:
                            touch(k)
                            out.append(vals[k])
                    else:
                        _, k, v = op
                        if capacity <= 0:
                            continue
                        if k in vals:
                            vals[k] = v
                            touch(k)
                        else:
                            if len(vals) == capacity:
                                ev, _ = buckets[min_freq].popitem(last=False)
                                del vals[ev]
                                del freqs[ev]
                            vals[k] = v
                            freqs[k] = 1
                            buckets[1][k] = None
                            min_freq = 1
                return out
            """
        ),
    ),
    EvalProblem(
        id="code_07_grid_bfs_wall_breaks",
        category="code",
        entry_point="shortest_grid_path_with_wall_breaks",
        prompt=(
            "Write a Python function "
            "`shortest_grid_path_with_wall_breaks(grid: list[str], "
            "max_breaks: int) -> int`.\n"
            "Given a non-empty `R x C` grid of `'.'` (open cell) and `'#'` (wall), "
            "with `(0, 0)` and `(R - 1, C - 1)` guaranteed to be `'.'`, find the "
            "minimum number of 4-directional steps from `(0, 0)` to `(R - 1, C - 1)` "
            "entering `'#'` cells at most `max_breaks` times in total. Return `-1` if "
            "no such path exists."
            + CODE_SUFFIX_TMPL.format(
                sig=(
                    "def shortest_grid_path_with_wall_breaks("
                    "grid: list[str], max_breaks: int) -> int"
                )
            )
        ),
        test_harness=textwrap.dedent(
            """
            assert shortest_grid_path_with_wall_breaks(["."], 0) == 0
            assert shortest_grid_path_with_wall_breaks([".#.", "###", ".#."], 1) == -1
            assert shortest_grid_path_with_wall_breaks([".#.", "###", ".#."], 2) == 4
            # Pruning trap: (0, 2) is reached in 2 steps with 0 breaks left, or 6 steps
            # with 1 break left; reaching (2, 5) requires breaking (0, 3)!
            trap = [
                ".#.#..",
                ".#.##.",
                "...##.",
            ]
            assert shortest_grid_path_with_wall_breaks(trap, 1) == 11
            assert shortest_grid_path_with_wall_breaks(trap, 2) == 7
            """
        ),
        reference_solution=textwrap.dedent(
            """
            import collections

            def shortest_grid_path_with_wall_breaks(
                grid: list[str], max_breaks: int
            ) -> int:
                R, C = len(grid), len(grid[0])
                if R == 1 and C == 1:
                    return 0
                best_rem = [[-1] * C for _ in range(R)]
                best_rem[0][0] = max_breaks
                q = collections.deque([(0, 0, max_breaks, 0)])
                while q:
                    r, c, rem, dist = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < R and 0 <= nc < C:
                            nrem = rem - (1 if grid[nr][nc] == "#" else 0)
                            if nrem < 0 or nrem <= best_rem[nr][nc]:
                                continue
                            if nr == R - 1 and nc == C - 1:
                                return dist + 1
                            best_rem[nr][nc] = nrem
                            q.append((nr, nc, nrem, dist + 1))
                return -1
            """
        ),
    ),
    EvalProblem(
        id="code_08_rectangles_union_area",
        category="code",
        entry_point="union_rectangles_area",
        prompt=(
            "Write a Python function `union_rectangles_area("
            "rects: list[tuple[int, int, int, int]]) -> int`.\n"
            "Each tuple `(x1, y1, x2, y2)` represents an axis-aligned rectangle with "
            "integer coordinates satisfying `x1 < x2` and `y1 < y2` (coordinates can "
            "be negative and up to `10**9` in magnitude; `len(rects) <= 200`). Return "
            "the exact total area covered by the union of the rectangles (overlapping "
            "regions counted once, `0` if `rects` is empty)."
            + CODE_SUFFIX_TMPL.format(
                sig=(
                    "def union_rectangles_area("
                    "rects: list[tuple[int, int, int, int]]) -> int"
                )
            )
        ),
        test_harness=textwrap.dedent(
            """
            assert union_rectangles_area([]) == 0
            r1 = [(0, 0, 2, 2), (1, 0, 2, 3), (1, 0, 3, 1)]
            assert union_rectangles_area(r1) == 6
            assert union_rectangles_area([(-5, -5, 5, 5), (-2, -2, 2, 2)]) == 100
            assert union_rectangles_area(
                [(0, 0, 10**9, 10**9), (5 * 10**8, 5 * 10**8, 15 * 10**8, 15 * 10**8)]
            ) == 175 * 10**16
            assert union_rectangles_area(
                [(-10, -10, 0, 0), (0, 0, 10, 10), (-5, -5, 5, 5)]
            ) == 250
            """
        ),
        reference_solution=textwrap.dedent(
            """
            def union_rectangles_area(rects: list[tuple[int, int, int, int]]) -> int:
                if not rects:
                    return 0
                xs = sorted({x for x1, _, x2, _ in rects for x in (x1, x2)})
                total = 0
                for i in range(len(xs) - 1):
                    xa, xb = xs[i], xs[i + 1]
                    segs = sorted(
                        (y1, y2) for x1, y1, x2, y2 in rects if x1 <= xa and xb <= x2
                    )
                    if not segs:
                        continue
                    cov = 0
                    cs, ce = segs[0]
                    for y1, y2 in segs[1:]:
                        if y1 <= ce:
                            if y2 > ce:
                                ce = y2
                        else:
                            cov += ce - cs
                            cs, ce = y1, y2
                    cov += ce - cs
                    total += (xb - xa) * cov
                return total
            """
        ),
    ),
)


# ===========================================================================
# Verification Helpers (Math \boxed{} & Sandboxed Python Subprocess)
# ===========================================================================


def _strip_thinking(text: str) -> str:
    """Return the final assistant answer after any reasoning end-marker."""
    for marker in ("</think>", "<|end_thought|>", "<|thought_end|>"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
    return text


def extract_boxed_int(text: str) -> int | None:
    """Extract the last ``\\boxed{...}`` integer from ``text``."""
    tail = _strip_thinking(text)
    for source in (tail, text):
        spans: list[str] = []
        idx = 0
        while True:
            pos = source.find("\\boxed{", idx)
            if pos == -1:
                break
            start = pos + len("\\boxed{")
            depth = 1
            cur = start
            while cur < len(source) and depth > 0:
                if source[cur] == "{":
                    depth += 1
                elif source[cur] == "}":
                    depth -= 1
                cur += 1
            if depth == 0:
                spans.append(source[start : cur - 1])
            idx = cur
        for raw in reversed(spans):
            cleaned = (
                raw.replace(",", "")
                .replace("\\!", "")
                .replace("\\,", "")
                .replace(" ", "")
                .strip("$")
            )
            m = re.fullmatch(r"(?:\\text\{)?([+-]?\d+)\}?", cleaned)
            if m:
                return int(m.group(1))
    return None


def extract_python_code(text: str) -> str | None:
    """Extract the last fenced Python code block from ``text``."""
    tail = _strip_thinking(text)
    for source in (tail, text):
        py_blocks = re.findall(r"```python\s*\n(.*?)```", source, flags=re.DOTALL)
        if py_blocks:
            return py_blocks[-1].strip()
        any_blocks = re.findall(r"```\w*\s*\n(.*?)```", source, flags=re.DOTALL)
        if any_blocks:
            return any_blocks[-1].strip()
    return None


def verify_problem_output(
    prob: EvalProblem, text: str, *, timeout_s: float = 5.0
) -> dict[str, Any]:
    if prob.category == "math":
        assert prob.expected_int is not None
        got = extract_boxed_int(text)
        ok = got == prob.expected_int
        return {
            "passed": ok,
            "extracted": got,
            "expected": prob.expected_int,
            "error": None if ok else f"got {got!r}, expected {prob.expected_int}",
        }

    assert prob.entry_point is not None and prob.test_harness is not None
    code = extract_python_code(text)
    if not code:
        return {
            "passed": False,
            "extracted": None,
            "expected": prob.entry_point,
            "error": "no ```python ... ``` block found",
        }
    runner = (
        f"{code}\n\n"
        f"assert callable(globals().get({prob.entry_point!r})), "
        f"'missing function {prob.entry_point}'\n"
        f"{prob.test_harness}\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", runner],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "extracted": prob.entry_point,
            "expected": prob.entry_point,
            "error": f"timeout after {timeout_s:.1f}s",
        }
    if proc.returncode == 0:
        return {
            "passed": True,
            "extracted": prob.entry_point,
            "expected": prob.entry_point,
            "error": None,
        }
    err_lines = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
    tail_err = err_lines[-1] if err_lines else f"exit code {proc.returncode}"
    return {
        "passed": False,
        "extracted": prob.entry_point,
        "expected": prob.entry_point,
        "error": tail_err[:240],
    }


def run_self_test() -> int:
    """Offline self-verification of all 20 eval problems and parser edge cases."""
    assert len(EVAL_PROBLEMS) == 20
    math_probs = [p for p in EVAL_PROBLEMS if p.category == "math"]
    code_probs = [p for p in EVAL_PROBLEMS if p.category == "code"]
    assert len(math_probs) == 12 and len(code_probs) == 8

    for p in math_probs:
        wrapped = (
            f"<think>Let me double check \\boxed{{999999}} first.</think>\n"
            f"Therefore the answer is \\boxed{{{p.expected_int}}}."
        )
        res = verify_problem_output(p, wrapped)
        assert res["passed"], (p.id, res)
        bad = verify_problem_output(p, f"\\boxed{{{int(p.expected_int or 0) + 1}}}")
        assert not bad["passed"], p.id

    for p in code_probs:
        assert p.reference_solution is not None
        wrapped = f"Here is the solution:\n```python\n{p.reference_solution}\n```\n"
        res = verify_problem_output(p, wrapped)
        assert res["passed"], (p.id, res)

    # Verify eval() is blocked in code_05_int_expr_evaluator
    p5 = next(p for p in code_probs if p.id == "code_05_int_expr_evaluator")
    cheat = verify_problem_output(
        p5,
        "```python\ndef eval_int_expr(expr: str) -> int:\n    return eval(expr)\n```",
    )
    assert not cheat["passed"], "eval() cheat must be rejected"

    # Verify infinite loops time out cleanly
    hang = verify_problem_output(
        code_probs[0],
        (
            "```python\n"
            "def min_groups_with_gap(intervals, min_gap):\n"
            "    while True: pass\n"
            "```"
        ),
        timeout_s=0.4,
    )
    assert not hang["passed"] and "timeout" in str(hang["error"])

    # Verify arm resolver
    for arm in DEFAULT_ARMS:
        resolve_arm(arm)
    print(
        "SELF-TEST OK: 12/12 math answers + 8/8 coding reference solutions + "
        "sandbox/timeout/arm checks passed."
    )
    return 0


# ===========================================================================
# Mode 1: Standalone GPU Kernel Microbenchmark (--microbench)
# ===========================================================================


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(math.ceil(q * len(sorted_vals))) - 1))
    return sorted_vals[idx]


def _time_cuda_op(
    fn: Callable[[int], None], warmup: int, reps: int
) -> dict[str, float]:
    import torch

    for i in range(warmup):
        fn(i)
    torch.accelerator.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        fn(i)
        ends[i].record()
    torch.accelerator.synchronize()
    us = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return {
        "median_us": round(statistics.median(us), 2),
        "p95_us": round(_percentile(us, 0.95), 2),
        "mean_us": round(statistics.mean(us), 2),
        "min_us": round(us[0], 2),
    }


def run_microbench(
    ckpt: str | None, warmup: int = 50, reps: int = 200
) -> dict[str, Any]:
    import torch

    from vllm import _custom_ops as ops
    from vllm.models.inkling2_loop.amd.linear_pack import inkling2_linear_pack_lm_head

    if not torch.cuda.is_available():
        raise RuntimeError("--microbench requires a ROCm/CUDA GPU")

    dev = torch.device("cuda")
    H, P, N_BLOCKS, V = 2304, 1, 3, 200058
    gen = torch.Generator(device=dev).manual_seed(20261124)
    rows: list[dict[str, Any]] = []

    # 1) inkling2_glue_loopcd_blend at T in (1, 3, 8, 16), H=2304, P=1
    ring = 8
    for T in (1, 3, 8, 16):
        x0_ring = [
            torch.randn(T, H, generator=gen, device=dev, dtype=torch.float32) * 1e-12
            for _ in range(ring)
        ]
        x1_ring = [
            torch.randn(T, H, generator=gen, device=dev, dtype=torch.float32) * 1e-12
            for _ in range(ring)
        ]
        prev_x0 = (
            torch.randn(T, H, generator=gen, device=dev, dtype=torch.float32) * 1e-12
        )
        prev_x1 = (
            torch.randn(T, H, generator=gen, device=dev, dtype=torch.float32) * 1e-12
        )
        blk_ring = [
            torch.randn(T, P, H, generator=gen, device=dev, dtype=torch.float32) * 1e-12
            for _ in range(ring)
        ]
        prev_blk = (
            torch.randn(T, P, H, generator=gen, device=dev, dtype=torch.float32) * 1e-12
        )

        cases = [
            (
                "omega=0.0 (fast exit)",
                torch.zeros(T, dtype=torch.float32, device=dev),
                torch.zeros(T, dtype=torch.float32, device=dev),
            ),
            (
                "omega=0.35, blend_blocks=True",
                torch.full((T,), 0.35, dtype=torch.float32, device=dev),
                torch.full((T,), 0.35, dtype=torch.float32, device=dev),
            ),
            (
                "omega=0.35, blend_blocks=False",
                torch.full((T,), 0.35, dtype=torch.float32, device=dev),
                torch.zeros(T, dtype=torch.float32, device=dev),
            ),
            (
                "mixed [0.0, 0.35, ...]",
                torch.tensor(
                    [0.0 if i % 2 == 0 else 0.35 for i in range(T)],
                    dtype=torch.float32,
                    device=dev,
                ),
                torch.tensor(
                    [0.0 if i % 2 == 0 else 0.35 for i in range(T)],
                    dtype=torch.float32,
                    device=dev,
                ),
            ),
        ]
        for label, w_vec, wb_vec in cases:

            def _run_blend(
                it: int,
                x0r=x0_ring,
                x1r=x1_ring,
                br=blk_ring,
                px0=prev_x0,
                px1=prev_x1,
                pb=prev_blk,
                w=w_vec,
                wb=wb_vec,
            ) -> None:
                ops.inkling2_glue_loopcd_blend(
                    x0r[it % ring],
                    px0,
                    px1,
                    w,
                    x1=x1r[it % ring],
                    blk=br[it % ring],
                    prev_blk=pb,
                    omega_blk=wb,
                )

            stats = _time_cuda_op(_run_blend, warmup, reps)
            rows.append(
                {
                    "op": "loopcd_blend",
                    "shape": f"T={T}, H={H}, P={P}",
                    "case": label,
                    **stats,
                }
            )

    # 2) inkling2_glue_loopcd_readout at T in (1, 3, 8), n_blocks=3, H=2304
    for T in (1, 3, 8):
        blocks = torch.randn(
            T, N_BLOCKS, H, generator=gen, device=dev, dtype=torch.float32
        )
        x0 = torch.randn(T, H, generator=gen, device=dev, dtype=torch.float32)
        x1 = torch.randn(T, H, generator=gen, device=dev, dtype=torch.float32)
        query = torch.randn(1, H, generator=gen, device=dev, dtype=torch.float32)
        norm_w = torch.ones(1, H, device=dev, dtype=torch.float16)
        out_buf = torch.empty(T, H, device=dev, dtype=torch.float16)
        out_view = out_buf.unsqueeze(1)
        for gv in (0, 1):
            flag = torch.tensor([gv], dtype=torch.int32, device=dev)

            def _run_readout(
                _it: int,
                b=blocks,
                x0_=x0,
                x1_=x1,
                q=query,
                nw=norm_w,
                ov=out_view,
                f=flag,
            ) -> None:
                ops.inkling2_glue_loopcd_readout(
                    b, N_BLOCKS, x0_, q, nw, ov, x1=x1_, gate_flag=f
                )

            stats = _time_cuda_op(_run_readout, warmup, reps)
            rows.append(
                {
                    "op": "loopcd_readout",
                    "shape": f"T={T}, n_blk={N_BLOCKS}, H={H}",
                    "case": f"gate_flag={gv}",
                    **stats,
                }
            )

    # 3) inkling2_loopcd_logits at B in (1, 3, 8), V=200058
    logits_ring = 4
    for B in (1, 3, 8):
        zf_ring = [
            torch.randn(B, V, generator=gen, device=dev, dtype=torch.float32) * 3.0
            for _ in range(logits_ring)
        ]
        zw = torch.randn(B, V, generator=gen, device=dev, dtype=torch.float32) * 3.0
        lb = torch.full((B,), math.log(0.1), dtype=torch.float32, device=dev)
        stats_scratch = torch.empty((B, 32, 2), dtype=torch.float32, device=dev)
        l_cases = [
            (
                "omega=0.0 (fast exit)",
                torch.zeros(B, dtype=torch.float32, device=dev),
                torch.zeros(B, dtype=torch.int32, device=dev),
            ),
            (
                "omega=0.25, fixed",
                torch.full((B,), 0.25, dtype=torch.float32, device=dev),
                torch.zeros(B, dtype=torch.int32, device=dev),
            ),
            (
                "omega=0.25, adaptive=True",
                torch.full((B,), 0.25, dtype=torch.float32, device=dev),
                torch.ones(B, dtype=torch.int32, device=dev),
            ),
        ]
        for label, w_vec, ad_vec in l_cases:
            # Refresh zf_ring so -inf masks from earlier cases do not persist.
            for buf in zf_ring:
                buf.normal_(0.0, 3.0, generator=gen)

            def _run_logits(
                it: int,
                zfr=zf_ring,
                zw_=zw,
                w=w_vec,
                lb_=lb,
                ad=ad_vec,
                sc=stats_scratch,
            ) -> None:
                ops.inkling2_loopcd_logits(
                    zfr[it % logits_ring], zw_, w, lb_, ad, stats_scratch=sc
                )

            stats = _time_cuda_op(_run_logits, warmup, reps)
            rows.append(
                {
                    "op": "loopcd_logits",
                    "shape": f"B={B}, V={V}",
                    "case": label,
                    **stats,
                }
            )

    # 4) inkling2_linear_lm_head via LMHeadPack ([200058, 2304] fp16) at M in (1, 3, 8)
    head_source = "random_fp16"
    lm_w: torch.Tensor | None = None
    if ckpt:
        ckpt_path = Path(ckpt).expanduser()
        idx_file = ckpt_path / "model.safetensors.index.json"
        if idx_file.is_file():
            from safetensors import safe_open

            index = json.loads(idx_file.read_text())
            shard = index["weight_map"].get("model.llm.unembed.weight")
            if shard and (ckpt_path / shard).is_file():
                with safe_open(str(ckpt_path / shard), framework="pt") as sf:
                    lm_w = sf.get_tensor("model.llm.unembed.weight")[:V].to(
                        device=dev, dtype=torch.float16
                    )
                head_source = "ckpt_unembed"
    if lm_w is None:
        lm_w = torch.randn(V, H, generator=gen, device=dev, dtype=torch.float16) * 0.02
    pack = inkling2_linear_pack_lm_head(lm_w, V)
    for M in (1, 3, 8):
        x_in = torch.randn(M, H, generator=gen, device=dev, dtype=torch.float16)
        logits_out = torch.empty(M, pack.rows, device=dev, dtype=torch.float32)
        stats = _time_cuda_op(
            lambda _it, p=pack, x=x_in, lo=logits_out: p(x, lo, softcap=30.0),
            warmup,
            reps,
        )
        rows.append(
            {
                "op": "linear_lm_head",
                "shape": f"M={M}, V={V}, H={H}",
                "case": f"weak_head ({head_source})",
                **stats,
            }
        )

    print("\n### LoopCD Kernel Microbenchmark (`Inkling-2-Mini` Shapes)\n")
    print("| Kernel | Shape | Case | Median (µs) | p95 (µs) |")
    print("|---|---|---|---:|---:|")
    for r in rows:
        print(
            f"| `{r['op']}` | `{r['shape']}` | {r['case']} | "
            f"{r['median_us']:.2f} | {r['p95_us']:.2f} |"
        )
    print(flush=True)
    return {"warmup": warmup, "reps": reps, "rows": rows}


# ===========================================================================
# Engine Boot, Speculative Counters & Prompt Rendering (--speed & --eval)
# ===========================================================================


def spec_counters(llm: Any, enabled: bool) -> dict[str, Any]:
    """Read cumulative vLLM spec-decode counters (matching ``dspark_bench.py``)."""
    out: dict[str, Any] = {}
    if not enabled:
        return out
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_drafts":
            out["drafts"] = m.value
        elif m.name == "vllm:spec_decode_num_draft_tokens":
            out["draft_tokens"] = m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens":
            out["accepted"] = m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            out["per_pos"] = list(m.values)
    return out


def spec_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if not after:
        return {}
    drafts = after.get("drafts", 0) - before.get("drafts", 0)
    draft_toks = after.get("draft_tokens", 0) - before.get("draft_tokens", 0)
    acc = after.get("accepted", 0) - before.get("accepted", 0)
    per_pos_a = before.get("per_pos") or [0] * len(after.get("per_pos", []))
    per_pos = [y - x for x, y in zip(per_pos_a, after.get("per_pos", []))]
    return {
        "drafts": drafts,
        "draft_tokens": draft_toks,
        "accepted": acc,
        "accept_rate": (acc / draft_toks) if draft_toks else None,
        "mean_accepted_per_step": (acc / drafts) if drafts else None,
        "tok_per_step": (1.0 + acc / drafts) if drafts else None,
        "per_pos_rate": [p / drafts for p in per_pos] if drafts else [],
    }


def render_chat_texts(
    ckpt: str,
    texts: list[str] | tuple[str, ...],
    *,
    template_file: str = "",
    reasoning_effort: str | None = None,
) -> tuple[Any, list[list[int]]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt)
    template = Path(template_file).read_text() if template_file else None
    effort = (
        None if reasoning_effort in (None, "", "none", "None") else reasoning_effort
    )
    extra = {} if effort is None else {"reasoning_effort": effort}
    rendered: list[list[int]] = []
    for text in texts:
        try:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=True,
                chat_template=template,
                **extra,
            )
        except TypeError:
            # Fallback if a custom chat template does not accept reasoning_effort
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=True,
                chat_template=template,
            )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        rendered.append(list(ids))
    return tok, rendered


def build_llm(args: argparse.Namespace) -> Any:
    from engine_args import apply_tp_env

    from vllm import LLM

    apply_tp_env(args.ar_backend, args.ar_max_kb)
    kwargs: dict[str, Any] = {
        "model": args.ckpt,
        "dtype": "float16",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "skip_tokenizer_init": True,
        "max_logprobs": 8,
        "enable_prefix_caching": not args.no_prefix_caching,
        "limit_mm_per_prompt": {"image": 0},
        "tensor_parallel_size": args.tensor_parallel_size,
        "seed": 0,
        "disable_log_stats": not args.dspark,
    }
    if args.cudagraph_mode == "eager":
        kwargs["enforce_eager"] = True
    else:
        kwargs["compilation_config"] = {"cudagraph_mode": args.cudagraph_mode}

    if args.dspark:
        spec_opts: dict[str, Any] = {}
        if not args.no_dspark_adaptive:
            spec_opts["adaptive_k"] = {}
        if args.draft_vocab:
            spec_opts["draft_vocab"] = str(Path(args.draft_vocab).expanduser())
        spec: dict[str, Any] = {
            "method": "dspark",
            "num_speculative_tokens": args.spec_k,
        }
        if args.schedule:
            spec["num_speculative_tokens_per_batch_size"] = json.loads(args.schedule)
        if spec_opts:
            spec["speculator_options"] = spec_opts
        kwargs["speculative_config"] = spec

    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    print(f"[boot] LLM ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return llm


# ===========================================================================
# Mode 2: Sanity Gate & Speed Benchmark (--speed)
# ===========================================================================


def run_zero_alpha_sanity_gate(
    llm: Any, prompts: list[list[int]], max_tokens: int = 128
) -> dict[str, Any]:
    """Compare ``extra_args=None`` vs ``alpha=0.0`` across ``hidden`` and ``logits``."""
    from vllm import SamplingParams

    gate_prompts = [{"prompt_token_ids": p} for p in prompts[:4]]
    cases: list[tuple[str, dict[str, Any] | None]] = [
        ("none", None),
        ("hidden_a0.0", {"loopcd_alpha": 0.0, "loopcd_mode": "hidden"}),
        ("logits_a0.0", {"loopcd_alpha": 0.0, "loopcd_mode": "logits"}),
    ]
    outputs: dict[str, list[dict[str, Any]]] = {}
    for label, extra in cases:
        llm.reset_prefix_cache()
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True,
            logprobs=2,
            extra_args=extra,
        )
        outs = llm.generate(gate_prompts, sp, use_tqdm=False)
        rows = []
        for o in outs:
            c = o.outputs[0]
            rows.append(
                {
                    "ids": list(c.token_ids),
                    "top": [
                        {int(t): float(lp.logprob) for t, lp in step.items()}
                        for step in (c.logprobs or [])
                    ],
                }
            )
        outputs[label] = rows

    report: dict[str, Any] = {}
    base = outputs["none"]
    for label in ("hidden_a0.0", "logits_a0.0"):
        cur = outputs[label]
        exact = 0
        divergences: list[dict[str, Any]] = []
        for idx, (a, b) in enumerate(zip(base, cur)):
            if a["ids"] == b["ids"]:
                exact += 1
                continue
            n = min(len(a["ids"]), len(b["ids"]))
            div = next((i for i in range(n) if a["ids"][i] != b["ids"][i]), n)
            gaps = []
            for side in (a, b):
                if div < len(side["top"]):
                    vals = sorted(side["top"][div].values(), reverse=True)
                    if len(vals) >= 2:
                        gaps.append(round(vals[0] - vals[1], 4))
            divergences.append(
                {
                    "prompt_idx": idx,
                    "first_divergence": div,
                    "top2_gaps": gaps,
                    "cas_near_tie": bool(gaps and all(g < 0.15 for g in gaps)),
                }
            )
        report[label] = {
            "prompts": len(base),
            "token_exact": exact,
            "divergences": divergences,
        }
        print(
            f"[sanity-gate] none vs {label}: {exact}/{len(base)} prompts token-exact"
            + (f" (divergences: {divergences})" if divergences else ""),
            flush=True,
        )
    return report


def run_speed(
    llm: Any,
    prompts: list[list[int]],
    arm_names: list[str],
    batches: list[int],
    tokens: int,
    reps: int,
    dspark_on: bool,
) -> dict[str, Any]:
    from vllm import SamplingParams

    sanity = run_zero_alpha_sanity_gate(llm, prompts, max_tokens=min(128, tokens))

    # Warm up decode & graph shapes once
    max_b = max(batches)
    warm_batch = [{"prompt_token_ids": prompts[i % len(prompts)]} for i in range(max_b)]
    llm.reset_prefix_cache()
    llm.generate(
        warm_batch,
        SamplingParams(temperature=0.0, max_tokens=min(16, tokens), ignore_eos=True),
        use_tqdm=False,
    )

    speed_rows: dict[str, dict[str, Any]] = {}
    for arm in arm_names:
        extra = resolve_arm(arm)
        sp1 = SamplingParams(
            temperature=0.0, max_tokens=1, ignore_eos=True, extra_args=extra
        )
        spn = SamplingParams(
            temperature=0.0, max_tokens=tokens, ignore_eos=True, extra_args=extra
        )
        arm_res: dict[str, Any] = {}
        for b in batches:
            batch_reqs = [
                {"prompt_token_ids": prompts[i % len(prompts)]} for i in range(b)
            ]
            t1_list: list[float] = []
            tn_list: list[float] = []
            spec_list: list[dict[str, Any]] = []
            for _ in range(reps):
                llm.reset_prefix_cache()
                t0 = time.perf_counter()
                llm.generate(batch_reqs, sp1, use_tqdm=False)
                t1_list.append(time.perf_counter() - t0)

                llm.reset_prefix_cache()
                mid = spec_counters(llm, dspark_on)
                t0 = time.perf_counter()
                outs = llm.generate(batch_reqs, spn, use_tqdm=False)
                tn_list.append(time.perf_counter() - t0)
                spec_list.append(spec_delta(mid, spec_counters(llm, dspark_on)))
                assert all(len(o.outputs[0].token_ids) == tokens for o in outs)

            t1 = statistics.median(t1_list)
            tn = statistics.median(tn_list)
            dt = max(1e-9, tn - t1)
            ms_per_tok = dt / (tokens - 1) * 1000.0
            agg_tok_s = b * (tokens - 1) / dt
            spec = spec_list[len(spec_list) // 2] if dspark_on else {}
            arm_res[str(b)] = {
                "t1_s": round(t1, 4),
                "tN_s": round(tn, 4),
                "decode_ms_per_tok": round(ms_per_tok, 3),
                "req_tok_s": round(1000.0 / ms_per_tok, 2),
                "agg_tok_s": round(agg_tok_s, 2),
                "spec": spec,
            }
            spec_msg = ""
            if spec and spec.get("tok_per_step") is not None:
                acc_pct = 100.0 * (spec.get("accept_rate") or 0.0)
                spec_msg = (
                    f" | DSpark {spec['tok_per_step']:.2f} tok/step "
                    f"(accept {acc_pct:.1f}%)"
                )
            print(
                f"[speed] {arm:22s} B={b}: {ms_per_tok:6.2f} ms/tok | "
                f"{agg_tok_s:6.1f} agg tok/s{spec_msg}",
                flush=True,
            )
        speed_rows[arm] = arm_res

    print("\n### LoopCD Decode Speed Summary\n")
    headers = ["Arm"]
    for b in batches:
        headers.append(f"B={b} ms/tok")
        headers.append(f"B={b} tok/s")
        if dspark_on:
            headers.append(f"B={b} tok/step")
            headers.append(f"B={b} accept%")
    print("| " + " | ".join(headers) + " |")
    print("|---" + "|---:" * (len(headers) - 1) + "|")
    for arm in arm_names:
        cells = [f"`{arm}`"]
        for b in batches:
            cell = speed_rows[arm][str(b)]
            cells.append(f"{cell['decode_ms_per_tok']:.2f}")
            cells.append(f"{cell['agg_tok_s']:.1f}")
            if dspark_on:
                sp_info = cell["spec"]
                tps = sp_info.get("tok_per_step")
                ar = sp_info.get("accept_rate")
                cells.append(f"{tps:.2f}" if tps is not None else "-")
                cells.append(f"{100.0 * ar:.1f}%" if ar is not None else "-")
        print("| " + " | ".join(cells) + " |")
    print(flush=True)
    return {"sanity_gate": sanity, "tokens": tokens, "reps": reps, "arms": speed_rows}


# ===========================================================================
# Mode 3: 20-Problem Self-Verifying Evaluation Suite (--eval)
# ===========================================================================


def run_eval(
    llm: Any,
    tok: Any,
    rendered_prompts: list[list[int]],
    problems: list[EvalProblem],
    arm_names: list[str],
    max_tokens: int,
    eval_batch_size: int,
    dspark_on: bool,
) -> dict[str, Any]:
    from vllm import SamplingParams

    eval_arms: dict[str, Any] = {}
    for arm in arm_names:
        extra = resolve_arm(arm)
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=False,
            extra_args=extra,
        )
        llm.reset_prefix_cache()
        before = spec_counters(llm, dspark_on)
        t0 = time.perf_counter()
        all_outs = []
        for start in range(0, len(rendered_prompts), eval_batch_size):
            chunk = [
                {"prompt_token_ids": p}
                for p in rendered_prompts[start : start + eval_batch_size]
            ]
            all_outs.extend(llm.generate(chunk, sp, use_tqdm=False))
        wall_s = time.perf_counter() - t0
        spec = spec_delta(before, spec_counters(llm, dspark_on))

        per_problem: list[dict[str, Any]] = []
        math_pass = code_pass = truncated = total_gen_tokens = 0
        for prob, out in zip(problems, all_outs):
            comp = out.outputs[0]
            token_ids = list(comp.token_ids)
            text = tok.decode(token_ids, skip_special_tokens=True)
            verdict = verify_problem_output(prob, text)
            is_trunc = comp.finish_reason == "length"
            truncated += int(is_trunc)
            total_gen_tokens += len(token_ids)
            if verdict["passed"]:
                if prob.category == "math":
                    math_pass += 1
                else:
                    code_pass += 1
            per_problem.append(
                {
                    "id": prob.id,
                    "category": prob.category,
                    "passed": verdict["passed"],
                    "extracted": verdict["extracted"],
                    "expected": verdict["expected"],
                    "error": verdict["error"],
                    "gen_tokens": len(token_ids),
                    "finish_reason": comp.finish_reason,
                }
            )

        n_math = sum(1 for p in problems if p.category == "math")
        n_code = sum(1 for p in problems if p.category == "code")
        total_pass = math_pass + code_pass
        summary = {
            "total_pass": total_pass,
            "total_count": len(problems),
            "accuracy": round(total_pass / max(1, len(problems)), 4),
            "math_pass": math_pass,
            "math_count": n_math,
            "code_pass": code_pass,
            "code_count": n_code,
            "truncated": truncated,
            "mean_gen_tokens": round(total_gen_tokens / max(1, len(problems)), 1),
            "total_gen_tokens": total_gen_tokens,
            "wall_s": round(wall_s, 2),
            "spec": spec,
            "problems": per_problem,
        }
        eval_arms[arm] = summary
        tps_str = (
            f", DSpark {spec['tok_per_step']:.2f} tok/step"
            if spec and spec.get("tok_per_step") is not None
            else ""
        )
        print(
            f"[eval] {arm:22s}: {total_pass:2d}/{len(problems)} "
            f"(math {math_pass:2d}/{n_math}, code {code_pass}/{n_code}) | "
            f"mean {summary['mean_gen_tokens']:.0f} tok, trunc {truncated}, "
            f"{wall_s:.1f}s{tps_str}",
            flush=True,
        )

    print("\n### LoopCD Reasoning & Coding Accuracy Summary\n")
    cols = ["Arm", "Total", "Math (12)", "Code (8)", "Mean Toks", "Trunc", "Wall (s)"]
    if dspark_on:
        cols.extend(["DSpark tok/step", "DSpark accept%"])
    print("| " + " | ".join(cols) + " |")
    print("|---" + "|---:" * (len(cols) - 1) + "|")
    for arm in arm_names:
        s = eval_arms[arm]
        row = [
            f"`{arm}`",
            f"**{s['total_pass']}/{s['total_count']}** ({100.0 * s['accuracy']:.1f}%)",
            f"{s['math_pass']}/{s['math_count']}",
            f"{s['code_pass']}/{s['code_count']}",
            f"{s['mean_gen_tokens']:.0f}",
            f"{s['truncated']}",
            f"{s['wall_s']:.1f}",
        ]
        if dspark_on:
            sp_info = s["spec"]
            tps = sp_info.get("tok_per_step")
            ar = sp_info.get("accept_rate")
            row.append(f"{tps:.2f}" if tps is not None else "-")
            row.append(f"{100.0 * ar:.1f}%" if ar is not None else "-")
        print("| " + " | ".join(row) + " |")
    print(flush=True)
    return {"max_tokens": max_tokens, "arms": eval_arms}


# ===========================================================================
# CLI Entry Point
# ===========================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=None, help="Path to Inkling-2-Mini checkpoint")
    ap.add_argument(
        "--microbench",
        action="store_true",
        help="Run standalone LoopCD GPU kernel latency microbenchmark",
    )
    ap.add_argument("--warmup", type=int, default=50, help="Microbench warmup iters")
    ap.add_argument(
        "--bench-iters", type=int, default=200, help="Microbench timed reps"
    )
    ap.add_argument(
        "--speed",
        action="store_true",
        help="Run 0-ULP sanity gate + in-process LLM decode speed sweep",
    )
    ap.add_argument(
        "--eval",
        action="store_true",
        help="Run 20-problem self-verifying Math + Python Coding evaluation suite",
    )
    ap.add_argument(
        "--self-test-eval",
        action="store_true",
        help="Run offline verification of the 20-problem bank & evaluators (no GPU)",
    )
    ap.add_argument(
        "--arms",
        default=",".join(DEFAULT_ARMS),
        help="Comma-separated LoopCD arm names to sweep",
    )
    ap.add_argument(
        "--batches", default="1,4", help="Comma-separated speed batch sizes"
    )
    ap.add_argument(
        "--tokens", type=int, default=256, help="Generated tokens in --speed"
    )
    ap.add_argument(
        "--reps", type=int, default=2, help="Timed repetitions per speed cell"
    )
    ap.add_argument(
        "--eval-max-tokens",
        type=int,
        default=2048,
        help="Max generated tokens per problem in --eval",
    )
    ap.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Concurrent requests per generate() call in --eval",
    )
    ap.add_argument(
        "--reasoning-effort",
        default="medium",
        help="Chat template reasoning_effort (e.g. 'low', 'medium', 'high', or 'none')",
    )
    ap.add_argument("--chat-template-file", default="")
    ap.add_argument("--cudagraph-mode", default="FULL_AND_PIECEWISE")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--ar-backend", choices=("rccl", "rdna"), default="rccl")
    ap.add_argument("--ar-max-kb", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-num-batched-tokens", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--dspark", action="store_true", help="Enable DSpark speculation")
    ap.add_argument("--spec-k", type=int, default=7, help="DSpark drafted tokens")
    ap.add_argument(
        "--schedule",
        default=DEFAULT_DSPARK_SCHEDULE,
        help="DSpark dynamic batch schedule JSON",
    )
    ap.add_argument(
        "--no-dspark-adaptive",
        action="store_true",
        help="Disable adaptive_k inside speculator_options when --dspark is set",
    )
    ap.add_argument(
        "--draft-vocab",
        default="",
        help="Optional .npy draft vocabulary path for DSpark",
    )
    ap.add_argument("--json", default="", help="Write full JSON report to this file")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test_eval:
        return run_self_test()

    if not (args.microbench or args.speed or args.eval):
        raise SystemExit(
            "Pass at least one of --microbench, --speed, --eval, or --self-test-eval"
        )

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arm_names:
        resolve_arm(arm)

    report: dict[str, Any] = {"args": vars(args)}
    if args.microbench:
        report["microbench"] = run_microbench(
            args.ckpt, warmup=args.warmup, reps=args.bench_iters
        )

    if args.speed or args.eval:
        if not args.ckpt:
            raise SystemExit("--ckpt is required when running --speed or --eval")
        speed_prompts: list[list[int]] = []
        eval_prompts: list[list[int]] = []
        tok: Any = None
        if args.speed:
            tok, speed_prompts = render_chat_texts(
                args.ckpt,
                SPEED_PROMPTS,
                template_file=args.chat_template_file,
                reasoning_effort=args.reasoning_effort,
            )
        if args.eval:
            tok, eval_prompts = render_chat_texts(
                args.ckpt,
                [p.prompt for p in EVAL_PROBLEMS],
                template_file=args.chat_template_file,
                reasoning_effort=args.reasoning_effort,
            )

        # Boot vllm.LLM ONCE for both --speed and --eval!
        llm = build_llm(args)
        if args.speed:
            batches = [int(b) for b in args.batches.split(",") if b.strip()]
            report["speed"] = run_speed(
                llm,
                speed_prompts,
                arm_names,
                batches,
                args.tokens,
                args.reps,
                args.dspark,
            )
        if args.eval:
            report["eval"] = run_eval(
                llm,
                tok,
                eval_prompts,
                list(EVAL_PROBLEMS),
                arm_names,
                args.eval_max_tokens,
                args.eval_batch_size,
                args.dspark,
            )

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"[out] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
