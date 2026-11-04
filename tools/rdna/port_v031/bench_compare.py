# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""G3 (gate): flag on versus flag off bench numbers (same recipe, same cells)."""

import glob
import json
import sys


def value(arm: str, cell: str, field: str) -> float | None:
    best = None
    for path in glob.glob(f"{arm}/bench-{cell}*.json"):
        with open(path) as f:
            data = json.load(f)
        found = data.get(field)
        if found is None:
            continue
        if isinstance(found, list):
            if not found:
                continue
            found = found[0]
        if best is None or found > best:
            best = found
    return best


def main() -> int:
    off, on = sys.argv[1], sys.argv[2]
    # Optional cell overrides so a milestone gate can compare the cells it
    # actually ran (M3 asks G0-G2 and G5, so G3 normally keeps the defaults).
    mixed = sys.argv[3] if len(sys.argv) > 3 else "16384x1024-c8"
    solo = sys.argv[4] if len(sys.argv) > 4 else "16384x128-c1"
    bad = 0

    def compare(label: str, cell: str, field: str, limit: float) -> None:
        nonlocal bad
        off_v = value(off, cell, field)
        on_v = value(on, cell, field)
        if off_v is None or on_v is None:
            print(f"FAIL G3: missing {cell} {field} (off={off_v} on={on_v})")
            bad = 1
        elif on_v <= limit * off_v:
            print(f"ok   G3: {label} (off={off_v:g} on={on_v:g})")
        else:
            print(f"FAIL G3: {label} (off={off_v:g} on={on_v:g})")
            bad = 1

    compare("mixed-load mean ITL drops sharply (<=0.7x off)", mixed, "mean_itl_ms", 0.7)
    compare(
        "mixed-load ITL p50 not worse beyond noise (<=1.1x off)",
        mixed,
        "median_itl_ms",
        1.1,
    )
    compare(
        "solo prefill TTFT within 5% (<=1.05x off)",
        solo,
        "median_ttft_ms",
        1.05,
    )

    off_pf = value(off, mixed, "request_throughput")
    on_pf = value(on, mixed, "request_throughput")
    print(
        f"     G3: {mixed} prefill off="
        f"{off_pf if off_pf is not None else 'n/a'} on="
        f"{on_pf if on_pf is not None else 'n/a'} tok/s/req"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
