# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check qualified FP16 row coverage before starting the V620 candidate."""

import argparse
import csv
import sys
from pathlib import Path

from tune_v620_fp16 import DENSE_SHAPES


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--rows-template", required=True, type=Path)
    parser.add_argument("--block-size", required=True, type=int)
    parser.add_argument("--max-num-batched-tokens", required=True, type=int)
    parser.add_argument(
        "--decode-batch",
        nargs="*",
        type=int,
        default=[],
        help="Also require non-Default solvers for these token counts.",
    )
    # Accept the complete launch argv so the final CLI overrides are checked.
    args, _ = parser.parse_known_args()
    block, budget = args.block_size, args.max_num_batched_tokens
    if min(block, budget) <= 0:
        parser.error("Block size and batch budget must be positive")
    chunks = list(range(block, budget + 1, block)) or [budget]
    decode = [m for m in args.decode_batch if m not in chunks]
    missing = []
    for rank in range(4):
        path = args.rows_template.with_name(
            args.rows_template.stem + str(rank) + args.rows_template.suffix
        )
        if not path.is_file():
            missing.append(f"rank {rank}: missing file {path}")
            continue
        solvers: dict[tuple[str, str], str] = {}
        with path.open() as source:
            for row in csv.reader(source):
                if len(row) >= 2 and row[0] != "Validator":
                    solvers[tuple(row[:2])] = row[2] if len(row) > 2 else ""
        for m in chunks:
            absent = sum(
                (
                    "GemmTunableOp_Half_TN",
                    f"tn_{n}_{m}_{k}_ld_{k}_{k}_{n}",
                )
                not in solvers
                for n, k in DENSE_SHAPES
            )
            if absent:
                missing.append(f"rank {rank}: {m} tokens ({absent} missing shapes)")
        for m in decode:
            absent = 0
            for n, k in DENSE_SHAPES:
                key = (
                    "GemmTunableOp_Half_TN",
                    f"tn_{n}_{m}_{k}_ld_{k}_{k}_{n}",
                )
                solver = solvers.get(key)
                if not solver or solver == "Default":
                    absent += 1
            if absent:
                missing.append(
                    f"rank {rank}: decode {m} tokens ({absent} missing or Default)"
                )
    if missing:
        message = (
            "WARNING: V620 FP16 tuning coverage is incomplete for the requested "
            "cache/batch configuration. These shapes use the default FP16 path "
            "and may be slower:\n" + "\n".join(missing) + "\n"
        )
        if args.strict:
            parser.exit(2, message)
        print(message, file=sys.stderr, end="")
        return
    print(f"FP16 tuning coverage verified on four ranks for token counts {chunks}")


if __name__ == "__main__":
    main()
