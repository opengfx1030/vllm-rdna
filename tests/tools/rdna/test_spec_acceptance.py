# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The acceptance length of a bench cell, from SpecDecoding log lines."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "spec_acceptance", ROOT / "tools/rdna/port_v031/spec_acceptance.py"
)
assert _spec is not None and _spec.loader is not None
sa = importlib.util.module_from_spec(_spec)
sys.modules["spec_acceptance"] = sa
_spec.loader.exec_module(sa)


def _line(hms: str, accepted: int, drafted: int) -> str:
    return (
        f"(APIServer pid=1) INFO 11-05 {hms} [metrics.py:120] SpecDecoding metrics: "
        f"Mean acceptance length: 9.99, Accepted throughput: 1.00 tokens/s, "
        f"Drafted throughput: 2.00 tokens/s, Accepted: {accepted} tokens, "
        f"Drafted: {drafted} tokens, Per-position acceptance rate: 0.723, 0.477, "
        "Avg Draft acceptance rate: 60.0%"
    )


LOG = [
    "(APIServer pid=1) INFO 11-05 19:59:20 [loggers.py:1] Avg prompt throughput",
    _line("19:59:29", 86, 120),  # interval 19:59:19-29: middle 19:59:24, before
    _line("19:59:39", 193, 452),  # middle 19:59:34
    _line("19:59:49", 60, 162),  # middle 19:59:44
    _line("19:59:59", 39, 190),  # middle 19:59:54, after the cell
]


def test_the_acceptance_length_counts_the_intervals_whose_middle_is_in_the_cell():
    records = sa.parse_log(LOG)
    assert [r[1:] for r in records] == [
        (86, 120, 2),
        (193, 452, 2),
        (60, 162, 2),
        (39, 190, 2),
    ]
    got = sa.acceptance(records, "19:59:30", "19:59:50")
    assert got["intervals"] == 2
    assert (got["accepted"], got["drafted"]) == (253, 614)
    # a step drafts 2 tokens: 307 steps, 253 of the 614 drafted accepted
    assert got["length"] == pytest.approx(1 + 2 * 253 / 614)
    assert sa.acceptance(records, "10:00:00", "10:01:00")["intervals"] == 0


def test_cli_prints_the_step_rate(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    log = tmp_path / "serve.log"
    log.write_text("\n".join(LOG) + "\n")
    bench = tmp_path / "bench.json"
    bench.write_text(
        json.dumps(
            {"output_throughput": 80.0, "mean_itl_ms": 23.5, "mean_tpot_ms": 11.5}
        )
    )
    argv = [str(log), "19:59:30", "19:59:50", "--bench", str(bench)]
    assert sa.main(argv) == 0
    out = capsys.readouterr().out
    length = 1 + 2 * 253 / 614
    assert f"acceptance length {length:.3f} (253 accepted, 614 drafted, 2 log" in out
    assert f"steps/s {80.0 / length:.2f}" in out
    assert "mean ITL 23.50 ms; mean TPOT 11.50 ms" in out
