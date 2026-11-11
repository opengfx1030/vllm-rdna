#!/usr/bin/env python3
"""Four-metric table from serve_validate bench JSONs, grouped by recipe and arm
(directory names serve-m3-<mtp>-<arm>-<rep>), mean over reps, on/off delta.

  python3 ab_table.py ~/w4a8_runs/mono-layout/serve-ab-*
"""

import collections
import json
import os
import re
import statistics
import sys

CELLS = ["1024x512-c1", "1024x512-c8", "16384x1024-c1", "16384x1024-c8"]


def metrics(path):
    d = json.load(open(path))
    n = d.get("completed") or d.get("num_prompts") or 0
    ttft = d["mean_ttft_ms"] / 1000.0
    in_len = d["total_input_tokens"] / max(n, 1)
    out_len = d["total_output_tokens"] / max(n, 1)
    tpot = d.get("mean_tpot_ms") or 0.0
    return {
        "prefill_tok_s_req": in_len / ttft if ttft > 0 else 0.0,
        "decode_tok_s_req": 1000.0 / tpot if tpot > 0 else 0.0,
        "ttft_s": ttft,
        "itl_med_ms": d.get("median_itl_ms", 0.0),
        "itl_mean_ms": d.get("mean_itl_ms", 0.0),
        "out_tok_s": d.get("output_throughput", 0.0),
    }


def main():
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in sys.argv[1:]:
        # serve-m3-<mtp>-<arm>-<rep> (M3) or serve-<tag>-<mtp>-<arm> (run_specs)
        m = re.search(r"serve-(?:m3-)?(?:\w+?-)?(mtp\d)-(on|off)(?:-\w+)?$",
                      run.rstrip("/"))
        if not m:
            continue
        mtp, arm = m.groups()
        for cell in CELLS:
            f = os.path.join(run, f"bench-{cell}.json")
            if os.path.exists(f):
                rows[(mtp, cell)][arm].append(metrics(f))
    keys = ["prefill_tok_s_req", "decode_tok_s_req", "ttft_s", "itl_med_ms", "itl_mean_ms"]
    print("| recipe | cell | arm | n | prefill tok/s/req | decode tok/s/req | TTFT s | ITL med ms | ITL mean ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (mtp, cell), arms in sorted(rows.items()):
        mean = {}
        for arm in ("off", "on"):
            if arm not in arms:
                continue
            mean[arm] = {k: statistics.mean(r[k] for r in arms[arm]) for k in keys}
            v = mean[arm]
            print(f"| {mtp} | {cell} | {arm} | {len(arms[arm])} | {v['prefill_tok_s_req']:.1f} | "
                  f"{v['decode_tok_s_req']:.2f} | {v['ttft_s']:.2f} | {v['itl_med_ms']:.1f} | "
                  f"{v['itl_mean_ms']:.1f} |")
        if "on" in mean and "off" in mean:
            d = {k: 100.0 * (mean["on"][k] / mean["off"][k] - 1.0) if mean["off"][k] else 0.0
                 for k in keys}
            print(f"| {mtp} | {cell} | **on vs off** | | {d['prefill_tok_s_req']:+.1f}% | "
                  f"{d['decode_tok_s_req']:+.1f}% | {d['ttft_s']:+.1f}% | {d['itl_med_ms']:+.1f}% | "
                  f"{d['itl_mean_ms']:+.1f}% |")


if __name__ == "__main__":
    main()
