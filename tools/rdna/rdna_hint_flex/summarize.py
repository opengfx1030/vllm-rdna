# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Markdown table of serve_validate.sh bench results: all four numbers per cell.

python summarize.py ROOT tag1 tag2 ...   (ROOT/serve-<tag>/bench-*.json)
"""

import json
import sys
from pathlib import Path

CELLS = [
    ("1k/512 c=1", "bench-1024x512-c1"),
    ("1k/512 c=8", "bench-1024x512-c8"),
    ("16k/1k c=1", "bench-16384x1024-c1"),
    ("16k/1k c=8", "bench-16384x1024-c8"),
]


def _parse_cell(name: str) -> tuple[int, int, int] | None:
    """bench-<in>x<out>-c<conc>.json -> (in, out, conc)."""
    if not (name.startswith("bench-") and name.endswith(".json")):
        return None
    io, _, conc = name[len("bench-") : -len(".json")].partition("-c")
    i, _, o = io.partition("x")
    if not (i.isdigit() and o.isdigit() and conc.isdigit()):
        return None
    return int(i), int(o), int(conc)


def _k(n: int) -> str:
    return f"{n // 1024}k" if n % 1024 == 0 else str(n)


def cells_present(root: Path, tags: list[str]) -> list[tuple[str, str]]:
    """The cells found under the arms' result dirs (e.g. 2k/512 at max
    context 4096), or the standard four when none is found."""
    seen = set()
    for tag in tags:
        for f in (root / f"serve-{tag}").glob("bench-*.json"):
            c = _parse_cell(f.name)
            if c:
                seen.add(c)
    if not seen:
        return CELLS
    return [
        (f"{_k(i)}/{_k(o) if o >= 1024 else o} c={c}", f"bench-{i}x{o}-c{c}")
        for i, o, c in sorted(seen)
    ]


def cell(path: Path) -> str:
    if not path.exists():
        return "| n/a | n/a | n/a | n/a |"
    d = json.loads(path.read_text())
    done = max(int(d.get("completed", 0)), 1)
    in_len = d.get("total_input_tokens", 0) / done
    ttft = d.get("mean_ttft_ms", 0.0) / 1000
    p99 = d.get("p99_ttft_ms", 0.0) / 1000
    wave = min(int(d.get("max_concurrency") or 1), done)
    pre = in_len / ttft if ttft else 0
    pre_agg = wave * in_len / p99 if p99 else 0
    tpot = d.get("mean_tpot_ms", 0.0)
    dec = 1000 / tpot if tpot else 0
    return (
        f"| {pre:.0f} ({pre_agg:.0f}) | {dec:.1f} ({d.get('output_throughput', 0):.1f})"
        f" | {ttft:.2f} | {d.get('median_itl_ms', 0):.1f} |"
    )


def main() -> None:
    root = Path(sys.argv[1])
    print("| cell | arm | prefill tok/s | decode tok/s | TTFT s | ITL ms (median) |")
    print("| --- | --- | --- | --- | --- | --- |")
    tags = sys.argv[2:]
    for label, name in cells_present(root, tags):
        for tag in tags:
            print(f"| {label} | {tag} " + cell(root / f"serve-{tag}" / f"{name}.json"))


if __name__ == "__main__":
    main()
