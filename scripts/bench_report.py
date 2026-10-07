# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report PP / TG / TTFT from a vLLM bench result directory.

Every gfx1030 benchmark must report prompt processing (PP), token generation
(TG) and TTFT — never TG alone. vLLM's "Total token throughput" mixes prompt
and output tokens and is not a substitute for either.

Usage:
    python scripts/bench_report.py <result-dir> [<result-dir> ...]
"""

import glob
import json
import os
import sys


def main(dirs: list[str]) -> int:
    for d in dirs:
        files = glob.glob(os.path.join(d, "*.json"))
        if not files:
            print(f"{d}: no result json")
            continue
        # Newest by mtime, not name: result dirs accumulate across sessions and
        # a lexical sort silently reports an older run's numbers.
        newest = max(files, key=os.path.getmtime)
        with open(newest) as fh:
            r = json.load(fh)
        total_input = r.get("total_input_tokens") or 0
        ttft_ms = r.get("mean_ttft_ms") or 0.0
        tg = r.get("output_throughput") or 0.0
        concurrency = r.get("max_concurrency") or 1
        pp = total_input / (ttft_ms / 1000.0) if ttft_ms else 0.0
        print(
            f"{os.path.basename(d):28s} "
            f"PP={pp:7.0f} tok/s agg {pp / concurrency:7.0f} per-req | "
            f"TG={tg:7.2f} tok/s agg {tg / concurrency:6.2f} per-req | "
            f"TTFT={ttft_ms / 1000:6.2f}s"
        )
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
