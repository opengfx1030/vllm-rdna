# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ITL gate: is a flag-on arm within noise of the flag-off arms?

    python -m tools.rdna.eplb.itl_gate --off A1 A2 --on B1 B2 [--min-rel 0.005]

Each argument is a ``serve_validate.sh`` result directory holding the
``bench-<in>x<out>-c<conc>.json`` files of ``vllm bench serve``. Per cell and per metric
(ITL p50 ``median_itl_ms``, ITL p99 ``p99_itl_ms``, TTFT p50 and output tok/s for
context) the report gives the mean over the off arms, the mean over the on arms, the
relative difference, and the noise: the spread of the off arms (``max - min`` over their
mean). An ABBA order (off, on, on, off) makes two off arms bracket the on arms, so a
drift of the host shows up in the noise. A cell passes when ``|on - off|`` is at most
``max(noise, min_rel)`` of the off mean; ``min_rel`` (default 0.5 %) is the floor for
the case where the two off arms happen to agree to the last digit. With a single off
arm there is no noise estimate and the verdict is ``unknown``: the number is still
printed. A directory whose ``summary.txt`` marks a cell INVALID (KFD queue evictions)
is refused for that cell: such a window is not a measurement.

The gate is the acceptance test of phase 0 (counters on: ITL p50 and p99 within noise
of off) and of phase 1 (a placement file: same). The exit status is 0 when no cell
failed.
"""

import argparse
import glob
import json
import os
import sys

import regex as re

GATED = ("median_itl_ms", "p99_itl_ms")
CONTEXT = ("median_ttft_ms", "output_throughput")
CELL = re.compile(r"bench-(\d+)x(\d+)-c(\d+)\.json$")


def invalid_cells(directory: str) -> set[str]:
    """Cell names ``bench-...`` that ``summary.txt`` marks INVALID."""
    path = os.path.join(directory, "summary.txt")
    if not os.path.exists(path):
        return set()
    bad: set[str] = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "INVALID" in line:
                bad.update(re.findall(r"bench-\d+x\d+-c\d+", line))
    return bad


def load_cells(directory: str) -> dict[str, dict]:
    """Cell name -> bench result of one arm directory (INVALID cells dropped)."""
    bad = invalid_cells(directory)
    cells = {}
    for path in sorted(glob.glob(os.path.join(directory, "bench-*.json"))):
        name = os.path.basename(path).removesuffix(".json")
        if not CELL.search(name + ".json") or name in bad:
            continue
        with open(path, encoding="utf-8") as f:
            cells[name] = json.load(f)
    return cells


def _cell_key(name: str) -> tuple[int, ...]:
    """``bench-1024x512-c8`` -> ``(1024, 512, 8)``: input, output, concurrency."""
    match = CELL.search(name + ".json")
    return tuple(int(x) for x in match.groups()) if match else (0, 0, 0)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def compare(
    off: list[dict[str, dict]], on: list[dict[str, dict]], min_rel: float = 0.005
) -> list[dict]:
    """One row per (cell, metric) present in every arm."""
    names = set.intersection(*(set(a) for a in off + on)) if off and on else set()
    rows = []
    for name in sorted(names, key=_cell_key):
        for metric in GATED + CONTEXT:
            vo = [a[name][metric] for a in off if metric in a[name]]
            vn = [a[name][metric] for a in on if metric in a[name]]
            if len(vo) != len(off) or len(vn) != len(on):
                continue
            m_off, m_on = _mean(vo), _mean(vn)
            rel = (m_on - m_off) / m_off if m_off else float("nan")
            noise = (max(vo) - min(vo)) / m_off if len(vo) > 1 and m_off else None
            if metric not in GATED:
                verdict = "-"
            elif noise is None:
                verdict = "unknown"
            else:
                verdict = "pass" if abs(rel) <= max(noise, min_rel) else "FAIL"
            rows.append(
                {
                    "cell": name.removeprefix("bench-"),
                    "metric": metric,
                    "off": m_off,
                    "on": m_on,
                    "rel": rel,
                    "noise": noise,
                    "verdict": verdict,
                }
            )
    return rows


def format_rows(rows: list[dict]) -> str:
    lines = [
        f"{'cell':>16} {'metric':>18} {'off':>10} {'on':>10} {'on/off-1':>9} "
        f"{'noise':>7} verdict"
    ]
    for r in rows:
        noise = "-" if r["noise"] is None else f"{100 * r['noise']:.2f}%"
        lines.append(
            f"{r['cell']:>16} {r['metric']:>18} {r['off']:>10.2f} {r['on']:>10.2f} "
            f"{100 * r['rel']:>+8.2f}% {noise:>7} {r['verdict']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--off", nargs="+", required=True, help="flag-off arm directories")
    ap.add_argument("--on", nargs="+", required=True, help="flag-on arm directories")
    ap.add_argument("--min-rel", type=float, default=0.005)
    ap.add_argument("--json", help="write the rows here")
    a = ap.parse_args(argv)
    rows = compare(
        [load_cells(d) for d in a.off], [load_cells(d) for d in a.on], a.min_rel
    )
    if not rows:
        print("no bench cell is present (and valid) in every arm", file=sys.stderr)
        return 3
    print(format_rows(rows))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
            f.write("\n")
    return 1 if any(r["verdict"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
