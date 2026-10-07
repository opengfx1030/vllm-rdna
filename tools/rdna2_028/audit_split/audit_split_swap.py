#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Swap the compute_split_k body in the scratch tree.

Replaces the region between the end of the LDS-budget expression and the
VLLM_RDNA2_PREFILL_DEBUG print with a prepared block, so the same source tree
can be rebuilt with the legacy (pre-W4A8) split search, the W4A8-era
enumeration, or the repair (legacy-when-valid) implementation.
"""

import argparse
import pathlib

ANCHOR = ": (32 * 1024);\n\n"
END = "  static const bool debug_split ="


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--block", required=True)
    args = ap.parse_args()

    p = pathlib.Path(args.file)
    txt = p.read_text()
    block = pathlib.Path(args.block).read_text()
    i = txt.index(ANCHOR) + len(ANCHOR)
    j = txt.index(END)
    p.write_text(txt[:i] + block + txt[j:])
    print(f"swapped {p} -> {pathlib.Path(args.block).name} ({j - i} -> {len(block)} bytes)")


if __name__ == "__main__":
    main()
