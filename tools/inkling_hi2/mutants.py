#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mutants of the HI2 loader / name map / TP slice / reference: tests kill each.

    python tools/inkling_hi2/mutants.py --list
    $VENV/bin/python tools/inkling_hi2/mutants.py --tree /path/to/a/tree

Each mutant is one edit in a file of the tree; the edit must occur exactly once.
The tree's test is run against the mutated file (``-x``: the first failure kills the
mutant) and the file is restored in a ``finally`` (and by the SIGINT/SIGTERM/atexit
handlers).  CPU only, no GPU, no checkpoint: the reference tests build their tiny HI2
fixture on first use (``INKLING2_TINY_OUT`` reuses one).

Exit 0 only when the baseline passes and every mutant is killed (the test fails on
the mutant and passes on the original).  A surviving mutant is a hole in the tests.

The tree must be a scratch copy, not the checkout under review; the reference test
also needs the quantizer on ``PYTHONPATH`` (see tools/inkling_hi2/README.md section 1).
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
from pathlib import Path

_L = "tests/models/inkling2_loop/test_hi2_loader.py::"
_R = "tests/models/inkling2_loop/test_hi2_reference.py::"

# (name, file relative to the tree, old text, new text, test node ids)
MUTANTS = [
    (
        "loader_embed_name",
        "vllm/models/inkling2_loop/amd/loader.py",
        'return f"{_SERVING_TEXT_PREFIX}.embed.weight"',
        'return f"{_SERVING_TEXT_PREFIX}.embed_tokens.weight"',
        [
            "tests/models/inkling2_loop/test_hi2_loader.py::test_hf_name_map_inverts_the_reference_map"
        ],
    ),
    (
        "loader_query_layer_off_by_one",
        "vllm/models/inkling2_loop/amd/loader.py",
        'return f"{_SERVING_TEXT_PREFIX}.attn_res.attn_queries.{layer - 1}"',
        'return f"{_SERVING_TEXT_PREFIX}.attn_res.attn_queries.{layer}"',
        [
            "tests/models/inkling2_loop/test_hi2_loader.py::test_hf_name_map_inverts_the_reference_map"
        ],
    ),
    (
        "loader_version_check_off",
        "vllm/models/inkling2_loop/amd/loader.py",
        "if version not in HI2_SPEC_VERSIONS:",
        "if False:",
        [_L + "test_a_directory_that_is_not_the_stacked_layout_is_refused"],
    ),
    (
        "loader_layout_check_off",
        "vllm/models/inkling2_loop/amd/loader.py",
        'if str(quant_config.get("moe_layout", "")) != "stacked":',
        "if False:",
        [_L + "test_a_directory_that_is_not_the_stacked_layout_is_refused"],
    ),
    (
        "hi2_hadamard_sign",
        "vllm/models/inkling2_loop/amd/hi2.py",
        "h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)",
        "h = torch.cat([torch.cat([h, h], 1), torch.cat([h, h], 1)], 0)",
        [_R + "test_decoded_experts_match_the_quantizer"],
    ),
    (
        "hi2_smoothing_direction",
        "vllm/models/inkling2_loop/amd/hi2.py",
        "w = w / scale[expert].double()",
        "w = w * scale[expert].double()",
        [_R + "test_decoded_experts_match_the_quantizer"],
    ),
    (
        "hi2_unpack_low_bits",
        "vllm/models/inkling2_loop/amd/hi2.py",
        "c = torch.stack([(w >> (nbits * i)) & mask for i in range(per_word)], dim=-1)",
        (
            "c = torch.stack([(w >> (nbits * i)) & (mask >> 1) "
            "for i in range(per_word)], dim=-1)"
        ),
        [_R + "test_decoded_experts_match_the_quantizer"],
    ),
    (
        "hi2_w13_slice_gate_only",
        "vllm/models/inkling2_loop/amd/hi2.py",
        "pieces.append(_take_rows(t, r0 + lo, r0 + hi))",
        "pieces.append(_take_rows(t, r0, r0 + hi - lo))",
        [_L + "test_tp_slice_of_the_fused_gate_up_keeps_the_units_rows"],
    ),
    (
        "hi2_w2_slice_first_cols",
        "vllm/models/inkling2_loop/amd/hi2.py",
        "piece = _select_cols(_take_rows(t, r0, r1), lo, hi)",
        "piece = _select_cols(_take_rows(t, r0, r1), 0, hi - lo)",
        [_L + "test_tp_slice_of_the_down_projection_takes_64_aligned_columns"],
    ),
    (
        "hi2_part_rows_extra_row",
        "vllm/models/inkling2_loop/amd/hi2.py",
        "return int(po[part]), int(po[part + 1])",
        "return int(po[part]), int(po[part + 1]) + 1",
        [_L + "test_tp_slice_of_the_fused_gate_up_keeps_the_units_rows"],
    ),
    (
        "hi2_validate_code_off",
        "vllm/models/inkling2_loop/amd/hi2.py",
        "if not torch.equal(co[1:] - co[:-1], kk * nb // 32):",
        "if False:",
        [_L + "test_a_valid_stack_is_accepted_and_a_bad_offset_is_refused"],
    ),
    (
        "hi2_parse_layer_plus_one",
        "vllm/models/inkling2_loop/amd/hi2.py",
        "return int(m.group(1)), m.group(2), m.group(3)",
        "return int(m.group(1)) + 1, m.group(2), m.group(3)",
        ["tests/models/inkling2_loop/test_hi2_loader.py::test_parse_tensor_name"],
    ),
    (
        "hi2_canonical_had_flat",
        "vllm/models/inkling2_loop/amd/hi2.py",
        '"had": t["had"].reshape(t["had"].shape[0], -1).to(torch.uint8),',
        '"had": t["had"].reshape(-1).to(torch.uint8),',
        [_L + "test_the_moe_collects_both_groups_and_runs_the_reference"],
    ),
    (
        "hi2_ref_swap_caps",
        "vllm/models/inkling2_loop/amd/hi2.py",
        "torch.tanh(gate / gate_softcap)",
        "torch.tanh(gate / up_softcap)",
        [
            (
                "tests/models/inkling2_loop/test_hi2_reference.py"
                "::test_routed_moe_matches_the_transformers_eval"
            )
        ],
    ),
    (
        "hi2_ref_drop_route_weight",
        "vllm/models/inkling2_loop/amd/hi2.py",
        "cur = cur * top_k_weights[token_idx, top_k_pos, None].float()",
        "cur = cur",
        [
            (
                "tests/models/inkling2_loop/test_hi2_reference.py"
                "::test_routed_moe_matches_the_transformers_eval"
            )
        ],
    ),
    (
        "hi2_ref_gate_up_swap",
        "vllm/models/inkling2_loop/amd/hi2.py",
        "gate = decode_part(w13, expert, 0, 2)",
        "gate = decode_part(w13, expert, 1, 2)",
        [
            (
                "tests/models/inkling2_loop/test_hi2_reference.py"
                "::test_routed_moe_matches_the_transformers_eval"
            )
        ],
    ),
    (
        "hi2_uneven_rows_check_off",
        "vllm/models/inkling2_loop/amd/hi2.py",
        "        if counts.numel() > 1 and bool((counts != counts[0]).any()):",
        "        if False:",
        [_L + "test_a_stack_with_uneven_expert_rows_is_refused"],
    ),
    (
        "glue_block64_no_round_up",
        "vllm/models/inkling2_loop/amd/glue.py",
        "n_pad64 = -(-n_pad // 64) * 64",
        "n_pad64 = n_pad",
        [_L + "test_the_hi2_block64_layout_is_padded_for_every_chunk"],
    ),
    (
        "model_lm_head_alias_off",
        "vllm/models/inkling2_loop/amd/model.py",
        '    if name == "lm_head.weight":\n        return "model.llm.unembed.weight"',
        '    if False:\n        return "model.llm.unembed.weight"',
        [_L + "test_the_unembed_name_is_aliased"],
    ),
    (
        "moe_gfx_guard_off",
        "vllm/models/inkling2_loop/amd/moe.py",
        "    if not on_gfx10x():",
        "    if False:",
        [_L + "test_the_hi2_kernel_paths_refuse_other_architectures"],
    ),
]

_BASELINE = [
    "tests/models/inkling2_loop/test_hi2_loader.py",
    "tests/models/inkling2_loop/test_hi2_reference.py",
]

_originals: dict[Path, bytes] = {}


def restore() -> None:
    for path, data in _originals.items():
        try:
            path.write_bytes(data)
        except OSError as exc:  # pragma: no cover - best effort on the way out
            print(f"mutants: could not restore {path}: {exc}", file=sys.stderr)


def _on_signal(signum, _frame):
    restore()
    raise SystemExit(128 + signum)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="a single mutant name")
    args = ap.parse_args()

    if args.list:
        for name, rel, _old, _new, nodes in MUTANTS:
            print(f"{name:<32} {rel} -> {len(nodes)} test(s)")
        return 0

    tree = Path(args.tree).resolve()
    for required in _BASELINE:
        if not (tree / required).is_file():
            print(
                f"mutants: {tree} is not the tree ({required} missing)", file=sys.stderr
            )
            return 2
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tree) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    def run(nodes: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-x", *nodes],
            cwd=tree,
            env=env,
            capture_output=True,
            text=True,
        )

    base = run(_BASELINE)
    tail = (base.stdout or "").strip().splitlines()[-1:] or ["(no output)"]
    if base.returncode != 0:
        print(f"mutants: baseline failed: {tail[0]}", file=sys.stderr)
        print(base.stdout[-4000:], base.stderr[-2000:], file=sys.stderr)
        return 2
    print(f"mutants: baseline passes ({tail[0]})")

    atexit.register(restore)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    selected = [m for m in MUTANTS if args.only is None or m[0] == args.only]
    if not selected:
        print(f"mutants: no mutant named {args.only!r}", file=sys.stderr)
        return 2
    survivors = []
    try:
        for name, rel, old, new, nodes in selected:
            path = tree / rel
            data = path.read_bytes()
            _originals.setdefault(path, data)
            text = data.decode()
            if text.count(old) != 1:
                raise SystemExit(
                    f"mutants: {name}: the old text occurs {text.count(old)} times "
                    f"in {rel}"
                )
            path.write_text(text.replace(old, new))
            try:
                proc = run(nodes)
            finally:
                path.write_bytes(data)
            killed = proc.returncode != 0
            print(f"mutants: {name:<32} {'KILLED' if killed else 'SURVIVED'}")
            if not killed:
                survivors.append(name)
    finally:
        restore()

    if survivors:
        print(f"mutants: SURVIVORS: {survivors}", file=sys.stderr)
        return 1
    print(f"mutants: all {len(selected)} mutants killed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
