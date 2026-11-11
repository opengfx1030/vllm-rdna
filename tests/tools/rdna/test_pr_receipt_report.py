# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Receipt logic of tools/rdna/pr_receipt.sh: cells, ABBA verdicts, gates.

Everything runs on synthetic boot directories (summary.txt and the JSON files
serve_validate.sh writes), so no server, GPU or network is involved.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rep = _load("pr_receipt_report", "tools/rdna/pr_receipt_report.py")
bm = _load("bench_metrics", "tools/rdna/port_v031/bench_metrics.py")

CELL_A = "bench-1024x512-c1"
CELL_B = "bench-1024x512-c4"
EVICT_OK = "evicted_ms Worker_TP0=+0 Worker_TP1=+0 -> ok"
EVICT_BAD = (
    "evicted_ms Worker_TP0=+312 Worker_TP1=+0 -> INVALID (KFD evicted a "
    "worker's GPU queues during this window; rerun it)"
)


# ------------------------------------------------------------ cells and rules


def test_default_cells_are_the_contract_matrix():
    cells = rep.parse_cells(rep.DEFAULT_CELLS)
    assert [c.label for c in cells] == [
        "1k/512 c=1",
        "1k/512 c=4",
        "1k/512 c=8",
        "16k/1k c=1",
        "16k/1k c=4",
        "16k/1k c=8",
        "32k/2k c=1",
        "32k/2k c=4",
    ]
    assert "32k/2k c=8" not in [c.label for c in cells]


def test_cell_names_and_request_counts_follow_serve_validate():
    script = (ROOT / "tools/rdna/port_v031/serve_validate.sh").read_text()
    # the report derives names and request counts the way the driver does
    assert "name=bench-${in}x${out}-c${conc}" in script
    assert "n=$(( conc * 4 > 8 ? conc * 4 : 8 ))" in script
    assert "(( conc == 1 )) && n=4" in script
    got = {c.conc: c.n for c in rep.parse_cells("1:1:1 1:1:2 1:1:4 1:1:8")}
    assert got == {1: 4, 2: 8, 4: 16, 8: 32}
    explicit = rep.parse_cells("16384:1024:8:5")[0]
    assert (explicit.n, explicit.name, explicit.spec) == (
        5,
        "bench-16384x1024-c8",
        "16384:1024:8:5",
    )


@pytest.mark.parametrize("spec", ["", "none", "  "])
def test_no_cells(spec):
    assert rep.parse_cells(spec) == []


@pytest.mark.parametrize(
    "spec",
    ["1024:512", "a:b:c", "0:512:1", "1024:512:1:2:3", "1024:512:1 1024:512:1:9"],
)
def test_bad_cells_are_refused(spec):
    with pytest.raises(ValueError):
        rep.parse_cells(spec)


@pytest.mark.parametrize(
    ("base", "head", "lower", "want"),
    [
        # lower is better (TTFT, ITL): regression needs both head runs above both base
        ([100, 102], [110, 111], True, "regression"),
        ([100, 102], [103, 104], True, "regression"),
        ([100, 102], [101, 111], True, "none"),  # one head run inside the base spread
        ([100, 102], [90, 99], True, "win"),
        ([100, 102], [95, 101], True, "none"),
        ([100, 102], [102, 110], True, "none"),  # a tie with a base run
        # higher is better (tok/s)
        ([100, 102], [90, 91], False, "regression"),
        ([100, 102], [99, 120], False, "none"),
        ([100, 102], [103, 110], False, "win"),
        ([100, 102], [102, 110], False, "none"),
        # the ABBA order: A and A' are the base runs, B and B' the head runs
        ([100, 120], [110, 111], False, "none"),
        ([100, 120], [90, 130], False, "none"),
        ([], [1], True, "n/a"),
    ],
)
def test_abba_verdict(base, head, lower, want):
    assert rep.abba_verdict(base, head, lower) == want


def test_abba_verdict_is_symmetric_between_the_two_directions():
    # a metric that is worse when higher flips with the sign of the test
    assert rep.abba_verdict([1, 2], [5, 6], True) == "regression"
    assert rep.abba_verdict([1, 2], [5, 6], False) == "win"


def test_single_run_direction_is_not_a_verdict():
    assert rep.ab_direction(10.0, 11.0, True) == "worse"
    assert rep.ab_direction(10.0, 9.0, True) == "better"
    assert rep.ab_direction(10.0, 11.0, False) == "better"
    assert rep.ab_direction(10.0, 10.0, False) == "equal"


def test_cell_roll_up():
    assert rep.roll_up({"a": "none", "b": "win", "c": "regression"}) == "regression"
    assert rep.roll_up({"a": "none", "b": "win"}) == "win"
    assert rep.roll_up({"a": "none"}) == "none"
    assert rep.roll_up_ab({"a": "equal", "b": "worse", "c": "better"}) == "worse"
    assert rep.roll_up_ab({"a": "equal", "b": "better"}) == "better"
    assert rep.roll_up_ab({"a": "equal"}) == "equal"


def test_delta_pct_uses_means():
    assert rep.delta_pct([100.0, 100.0], [110.0, 120.0]) == pytest.approx(15.0)
    assert rep.delta_pct([0.0], [1.0]) is None


# ----------------------------------------------------------- bench_metrics

GOLDEN_A = (
    "a                            prefill   1329.4 tok/s/req (agg  3383.0) | "
    "decode  44.64 tok/s/req (agg out  160.37) | TTFT    0.77 s | "
    "ITL med    21.9 ms mean    22.4 ms | ok 16"
)
GOLDEN_EMPTY = (
    "b                            prefill      0.0 tok/s/req (agg     0.0) | "
    "decode   0.00 tok/s/req (agg out    0.00) | TTFT    0.00 s | "
    "ITL med     0.0 ms mean     0.0 ms | ok 1"
)
BENCH_A = {
    "completed": 16,
    "total_input_tokens": 16384,
    "max_concurrency": 4,
    "mean_ttft_ms": 770.25,
    "median_ttft_ms": 701.5,
    "p99_ttft_ms": 1210.75,
    "mean_tpot_ms": 22.4,
    "median_itl_ms": 21.9,
    "mean_itl_ms": 22.4,
    "p99_itl_ms": 41.3,
    "output_throughput": 160.37,
}


def test_bench_metrics_line_is_unchanged(tmp_path):
    # outputs of the one-line summary before metrics() was split out of it
    (tmp_path / "a.json").write_text(json.dumps(BENCH_A))
    (tmp_path / "b.json").write_text(
        json.dumps({"completed": 0, "total_input_tokens": 0, "max_concurrency": None})
    )
    assert bm.line(str(tmp_path / "a.json")) == GOLDEN_A
    assert bm.line(str(tmp_path / "b.json")) == GOLDEN_EMPTY


def test_bench_metrics_dict():
    m = bm.metrics(BENCH_A)
    assert m["prefill_req_tps"] == pytest.approx(1024 / 0.77025)
    assert m["prefill_agg_tps"] == pytest.approx(4 * 1024 / 1.21075)
    assert m["decode_req_tps"] == pytest.approx(1000 / 22.4)
    assert (m["ttft_p50_ms"], m["ttft_p99_ms"]) == (701.5, 1210.75)
    assert (m["itl_p50_ms"], m["itl_p99_ms"], m["out_tps"]) == (21.9, 41.3, 160.37)


# --------------------------------------------------------------- summary.txt

SUMMARY = """\
[10:00:00] boot TREE=/t RECIPE=r MODEL=/m GPUS=2,3,4,5 PORT=1 overrides= bench_extra=
[10:04:02] READY in 242s served=flash-next
[10:04:10] PROBE PASS
[10:04:20] PPL PROBE PASS
[10:04:30] PREFIX PASS
[10:05:40] CONCURRENCY PASS
    RESULT PASS: 0 failing, 3 near-tie, 0 soft, 0 solo-noise, 45 identical of 48 checks
[10:05:41] quiet window open: /q/Me
[10:05:50] PREFIX RESET PASS before bench-1024x512-c1
[10:05:50] bench bench-1024x512-c1 (n=4)
    bench-1024x512-c1            prefill 1329.4 tok/s/req (agg 1329.4) | ok 4
    evicted_ms Worker_TP0=+0 Worker_TP1=+0 -> ok
[10:07:00] PREFIX RESET PASS before bench-1024x512-c4
[10:07:00] bench bench-1024x512-c4 (n=16)
    bench-1024x512-c4            prefill 900.1 tok/s/req (agg 3000.1) | ok 16
    evicted_ms Worker_TP0=+312 Worker_TP1=+0 -> INVALID (KFD evicted ...; rerun it)
[10:09:00] PREFIX RESET PASS before bench-16384x1024-c1
[10:09:00] bench bench-16384x1024-c1 (n=4)
    evicted_ms: no vLLM workers with a KFD context found
[10:09:01] SERVER DIED during bench-16384x1024-c1
[10:09:05] quiet window closed
[10:09:06] LOAD HOOK rc=0 PASS pr_receipt hook
[10:09:20] PROBE-AFTER-LOAD FAIL
[10:09:21] evicted_ms since READY: Worker_TP0=+312 Worker_TP1=+0 -> INVALID (x)
[10:09:21] markers:
      5 Using WNA16 MoE backend for flash-next
     12 Captured 51 graphs took 12.5 s
[10:09:22] fault markers in serve.log: 2
[10:09:23] new amdgpu dmesg faults: 0
n/a
[10:09:24] done
"""


def test_parse_summary():
    log = rep.parse_summary(SUMMARY)
    assert log.ready_s == 242
    assert log.marks == {
        "PROBE": "PASS",
        "PPL PROBE": "PASS",
        "PREFIX": "PASS",
        "CONCURRENCY": "PASS",
        "PROBE-AFTER-LOAD": "FAIL",
    }
    cells = log.cells
    assert (cells[CELL_A].n, cells[CELL_A].evicted, cells[CELL_A].start) == (
        4,
        "ok",
        "10:05:50",
    )
    assert cells[CELL_B].evicted == "INVALID"
    assert cells["bench-16384x1024-c1"].evicted == "unchecked"
    assert log.died_in == "bench-16384x1024-c1"
    assert log.hook == "rc=0 PASS pr_receipt hook"
    assert log.evicted_since_ready == "INVALID"
    assert (log.fault_markers, log.dmesg_faults, log.finished) == (2, "0", True)
    assert log.markers == [
        "5 Using WNA16 MoE backend for flash-next",
        "12 Captured 51 graphs took 12.5 s",
    ]
    assert not log.boot_failed


def test_parse_summary_boot_failure():
    log = rep.parse_summary("[10:00:00] boot X\n[10:30:00] BOOT FAILED after 1800s\n")
    assert log.boot_failed and log.ready_s is None and not log.finished


# ---------------------------------------------------------------- run fixtures


def bench(
    ttft=700.0,
    itl=20.0,
    out=160.0,
    n=4,
    conc=1,
    in_len=1024,
    tpot=22.0,
    prefill_p99=None,
    **extra,
):
    data = {
        "completed": n,
        "failed": 0,
        "duration": 20.0,
        "total_input_tokens": n * in_len,
        "max_concurrency": conc,
        "mean_ttft_ms": ttft,
        "median_ttft_ms": ttft,
        "p99_ttft_ms": prefill_p99 or ttft * 1.5,
        "mean_tpot_ms": tpot,
        "median_itl_ms": itl,
        "mean_itl_ms": itl,
        "p99_itl_ms": itl * 2,
        "output_throughput": out,
    }
    data.update(extra)
    return data


def summary_text(results, *, ready=120, hook=None, after="PASS", extra=""):
    lines = [
        "[10:00:00] boot TREE=/t RECIPE=r MODEL=/m GPUS=2,3,4,5 PORT=1 overrides=",
        f"[10:02:00] READY in {ready}s served=flash-next",
        "[10:02:10] PROBE PASS",
        "[10:02:20] PPL PROBE PASS",
        "[10:02:30] PREFIX PASS",
        "[10:03:30] CONCURRENCY PASS",
    ]
    for name, (n, evict) in results.items():
        lines += [
            f"[10:04:00] PREFIX RESET PASS before {name}",
            f"[10:04:00] bench {name} (n={n})",
            f"    {name}  prefill 1.0",
            f"    {evict}",
        ]
    if hook:
        lines.append(f"[10:08:00] LOAD HOOK rc=0 {hook}")
    lines += [
        f"[10:09:00] PROBE-AFTER-LOAD {after}",
        "[10:09:01] evicted_ms since READY: Worker_TP0=+0 -> ok",
        "[10:09:02] fault markers in serve.log: 0",
        "[10:09:03] new amdgpu dmesg faults: 0",
        "[10:09:04] done",
    ]
    return "\n".join(lines) + "\n" + extra


def conc_txt(failing=0):
    verdict = "PASS" if failing == 0 else "FAIL"
    return (
        f"RESULT {verdict}: {failing} failing, 2 near-tie, 0 soft, 0 solo-noise, "
        "46 identical of 48 checks (60.1s; tie <= 0.1, fail > 1.0 nats, lp drop)\n"
    )


def saved(tokens=("token_id:1", "token_id:2"), solo2=None):
    def one(toks):
        return {"tokens": list(toks), "text": "x"}

    return {
        "solo": {"0": one(tokens), "1": one(("token_id:9",))},
        "solo2": {"0": one(solo2 or tokens), "1": one(("token_id:9",))},
        "long_solo": {"0": one(("token_id:5", "token_id:6"))},
        "together": {},
    }


def write_boot(run, bid, results=None, **kwargs):
    d = run / "boots" / bid
    d.mkdir(parents=True, exist_ok=True)
    results = results or {}
    summary = {name: (data[1], data[2]) for name, data in results.items()}
    (d / "summary.txt").write_text(summary_text(summary, **kwargs))
    for name, (data, _n, _evict) in results.items():
        (d / f"{name}.json").write_text(json.dumps(data))
    (d / "concurrency.txt").write_text(conc_txt())
    return d


def manifest(run, settings, boots):
    flat = [f"{k}={v}" for k, v in settings.items()]
    assert rep.main(["set", str(run), *flat]) == 0
    for bid, role, kind in boots:
        argv = ["boot", str(run), bid, role, kind, f"boots/{bid}", "rc=0"]
        assert rep.main(argv) == 0


def cell(name, data, n=4, evict=EVICT_OK):
    return {name: (data, n, evict)}


def abba_run(tmp_path, a1, b1, b2, a2, *, cells="1024:512:1", mode="abba", **extra):
    """Four boots; each argument maps a cell name to (bench json, n, evict line)."""
    run = tmp_path / "run"
    for bid, results in (("A1", a1), ("B1", b1), ("B2", b2), ("A2", a2)):
        write_boot(run, bid, results)
    boots = [
        ("A1", "base", "speed"),
        ("B1", "head", "speed"),
        ("B2", "head", "speed"),
        ("A2", "base", "speed"),
    ]
    settings = {
        "mode": mode,
        "cells": cells,
        "exact": "0",
        "quality": "0",
        "head_sha": "h" * 40,
        "base_sha": "b" * 40,
        **extra,
    }
    manifest(run, settings, boots)
    return run


def speed_of(run):
    return rep.build_receipt(run)["speed"]


def test_abba_regression_win_and_no_difference_per_cell(tmp_path):
    def runs(ttft, out):
        return {
            **cell(CELL_A, bench(ttft=ttft, out=160.0)),
            **cell(CELL_B, bench(ttft=700.0, out=out, n=16, conc=4), n=16),
        }

    # cell A: head TTFT worse in both head runs; cell B: output tok/s beats both base
    run = abba_run(
        tmp_path,
        runs(700.0, 100.0),
        runs(800.0, 120.0),
        runs(810.0, 121.0),
        runs(702.0, 101.0),
        cells="1024:512:1 1024:512:4",
    )
    speed = speed_of(run)
    by_name = {c["name"]: c for c in speed["cells"]}
    a, b = by_name[CELL_A], by_name[CELL_B]
    assert a["metrics"]["ttft_p50_ms"]["verdict"] == "regression"
    assert a["metrics"]["ttft_p99_ms"]["verdict"] == "regression"
    assert a["metrics"]["out_tps"]["verdict"] == "none"  # identical in all four
    assert a["verdict"] == "regression"
    assert b["metrics"]["out_tps"]["verdict"] == "win"
    assert b["metrics"]["ttft_p50_ms"]["verdict"] == "none"
    assert b["verdict"] == "win"
    assert speed["counts"] == {"win": 1, "regression": 1, "none": 0, "invalid": 0}


def test_abba_overlap_is_no_measurable_difference(tmp_path):
    def one(ttft):
        return cell(CELL_A, bench(ttft=ttft))

    run = abba_run(tmp_path, one(700.0), one(705.0), one(690.0), one(710.0))
    speed = speed_of(run)
    assert speed["cells"][0]["verdict"] == "none"
    assert speed["counts"]["none"] == 1


def test_an_evicted_cell_is_invalid_and_has_no_verdict(tmp_path):
    good = cell(CELL_A, bench())
    # head run B2 slower, but its window was invalidated by a KFD eviction
    slow = cell(CELL_A, bench(ttft=2000.0), evict=EVICT_BAD)
    run = abba_run(tmp_path, good, cell(CELL_A, bench(ttft=900.0)), slow, good)
    speed = speed_of(run)
    entry = speed["cells"][0]
    assert entry["verdict"] == "invalid"
    assert "B2" in entry["invalid"] and "INVALID" in rep.speed_markdown(speed)
    assert "metrics" not in entry
    assert speed["counts"]["invalid"] == 1
    text = rep.speed_markdown(speed)
    assert 'CELLS="1024:512:1"' in text


def test_missing_result_and_failed_requests_are_invalid(tmp_path):
    good = cell(CELL_A, bench())
    run = abba_run(
        tmp_path,
        good,
        cell(CELL_A, bench(n=3, failed=1)),  # one of four requests failed
        good,
        {},  # the boot died before the cell
    )
    entry = speed_of(run)["cells"][0]
    assert entry["verdict"] == "invalid"
    assert entry["invalid"]["B1"].startswith("errors")
    assert entry["invalid"]["A2"].startswith("missing")


def test_ab_mode_shows_direction_not_a_verdict(tmp_path):
    run = tmp_path / "run"
    write_boot(run, "A1", cell(CELL_A, bench(ttft=700.0)))
    write_boot(run, "B1", cell(CELL_A, bench(ttft=900.0)))
    manifest(
        run,
        {"mode": "ab", "cells": "1024:512:1", "exact": "0", "quality": "0"},
        [("A1", "base", "speed"), ("B1", "head", "speed")],
    )
    speed = speed_of(run)
    entry = speed["cells"][0]
    assert entry["verdict"] == "worse"
    assert entry["metrics"]["ttft_p50_ms"]["direction"] == "worse"
    text = rep.speed_markdown(speed)
    assert "ABBA needed" in text and "REGRESSION" not in text
    assert "MODE=abba" in text


def test_speculative_acceptance_from_the_bench_result(tmp_path):
    run = tmp_path / "run"
    write_boot(run, "A1", cell(CELL_A, bench(spec_decode_acceptance_length=2.91, n=4)))
    write_boot(run, "B1", cell(CELL_A, bench(spec_decode_acceptance_length=2.88, n=4)))
    manifest(
        run,
        {"mode": "ab", "cells": "1024:512:1", "exact": "0", "quality": "0"},
        [("A1", "base", "speed"), ("B1", "head", "speed")],
    )
    text = rep.accept_markdown(speed_of(run))
    assert "2.910" in text and "2.880" in text and "bench" in text


def test_speculative_acceptance_falls_back_to_the_server_log(tmp_path):
    run = tmp_path / "run"
    boot = write_boot(run, "A1", cell(CELL_A, bench(duration=30.0)))
    line = (
        "(APIServer pid=1) INFO 11-05 10:04:09 [metrics.py:120] SpecDecoding "
        "metrics: Mean acceptance length: 9.99, Accepted throughput: 1.00 "
        "tokens/s, Drafted throughput: 2.00 tokens/s, Accepted: 150 tokens, "
        "Drafted: 200 tokens, Per-position acceptance rate: 0.8, 0.7, "
        "Avg Draft acceptance rate: 75.0%"
    )
    (boot / "serve.log").write_text(line + "\n")
    result = rep.load_cell(
        boot,
        rep.parse_cells("1024:512:1")[0],
        rep.parse_summary((boot / "summary.txt").read_text()),
    )
    assert result.accept_source == "server log"
    assert result.accept_len == pytest.approx(1 + 2 * 150 / 200)


# ---------------------------------------------------------------- exactness


def exact_run(tmp_path, base_saved, head_saved, head_failing=0, cell_texts=None):
    run = tmp_path / "run"
    for bid, data, failing in (
        ("XA", base_saved, 0),
        ("XB", head_saved, head_failing),
    ):
        d = write_boot(run, bid)
        (d / "concurrency.json").write_text(json.dumps(data))
        (d / "concurrency.txt").write_text(conc_txt(failing))
        if cell_texts:
            name = "bench-1024x128-c1"
            texts = cell_texts[bid]
            (d / f"{name}.json").write_text(json.dumps({"generated_texts": texts}))
    manifest(
        run,
        {"mode": "ab", "cells": "", "exact": "1", "quality": "0"},
        [("XA", "base", "exact"), ("XB", "head", "exact")],
    )
    return run


def test_exact_gate_passes_on_equal_hashes(tmp_path):
    run = exact_run(
        tmp_path,
        saved(),
        saved(),
        cell_texts={"XA": ["a", "b"], "XB": ["a", "b"]},
    )
    exact = rep.build_receipt(run)["exact"]
    assert exact["verdict"] == "PASS"
    # 2 solo + 2 solo2 + 1 long_solo saved outputs, plus 2 bench-cell texts
    assert (exact["compared"], exact["equal"]) == (7, 7)
    assert exact["differ"] == [] and exact["missing"] == []


def test_exact_gate_fails_on_a_different_token(tmp_path):
    head = saved()
    head["long_solo"]["0"]["tokens"][1] = "token_id:7"
    exact = rep.build_receipt(exact_run(tmp_path, saved(), head))["exact"]
    assert exact["verdict"] == "FAIL"
    assert exact["differ"] == ["long_solo/0"]


def test_exact_gate_fails_on_a_different_cell_text(tmp_path):
    run = exact_run(
        tmp_path, saved(), saved(), cell_texts={"XA": ["a", "b"], "XB": ["a", "c"]}
    )
    exact = rep.build_receipt(run)["exact"]
    assert exact["verdict"] == "FAIL"
    assert exact["differ"] == ["bench-1024x128-c1/1"]


def test_exact_gate_fails_when_a_boot_lacks_an_output(tmp_path):
    head = saved()
    del head["long_solo"]["0"]
    exact = rep.build_receipt(exact_run(tmp_path, saved(), head))["exact"]
    assert exact["verdict"] == "FAIL" and exact["missing"] == ["long_solo/0"]


def test_exact_gate_fails_on_a_concurrency_probe_failure(tmp_path):
    run = exact_run(tmp_path, saved(), saved(), head_failing=2)
    result = rep.build_receipt(run)
    assert result["exact"]["verdict"] == "FAIL"
    assert result["ok"] is False


def test_exact_gate_is_inconclusive_when_the_base_is_not_repeatable(tmp_path):
    flaky = saved(solo2=("token_id:1", "token_id:3"))
    exact = rep.build_receipt(exact_run(tmp_path, flaky, saved()))["exact"]
    assert exact["verdict"] == "INCONCLUSIVE"
    assert exact["unstable"]["base"] == ["solo/0"]


# ------------------------------------------------------------------- quality


def kl_entry(kl, kl_ci, top1, top1_ci):
    return {
        "windows": 8,
        "positions": 16376,
        "ppl_ref": 5.0,
        "ppl_cand": 5.001,
        "kl": kl,
        "kl_ci": kl_ci,
        "top1": top1,
        "top1_ci": top1_ci,
    }


def ppl_text(rel, verdict):
    return (
        "passage 0: PPL 5.000 -> 5.010 (+0.20%)\n"
        f"TOTAL PPL 5.0000 -> 5.0100 ({rel:+.3%})  mean|dlogp| 0.0100  "
        f"greedy exact 5/6  mean first divergence 40.0 tokens  {verdict}\n"
    )


def quality_run(tmp_path, head_kl=0.0011, head_top1=0.985, rel=0.002, verdict="PASS"):
    run = tmp_path / "run"
    boots = [("A1", "base", "speed"), ("B1", "head", "speed"), ("A2", "base", "floor")]
    for bid, role, kind in boots:
        write_boot(run, bid, hook="PASS pr_receipt hook")
    manifest(
        run,
        {"mode": "ab", "cells": "", "exact": "0", "quality": "1"},
        boots,
    )
    q = run / "quality"
    q.mkdir()
    floor = kl_entry(0.0010, [0.0008, 0.0012], 0.986, [0.984, 0.988])
    head = kl_entry(head_kl, [0.0009, 0.0013], head_top1, [0.983, 0.987])
    kl = {
        "pairs": {
            "floor": {"domains": {"docs": floor, "mix": floor}},
            "head": {"domains": {"docs": head, "mix": floor}},
        }
    }
    (q / "kl.json").write_text(json.dumps(kl))
    (q / "ppl-head.txt").write_text(ppl_text(rel, verdict))
    (q / "ppl-floor.txt").write_text(ppl_text(0.0001, "PASS"))
    return run


def test_quality_gate_passes_inside_the_floor(tmp_path):
    quality = rep.build_receipt(quality_run(tmp_path))["quality"]
    assert quality["verdict"] == "PASS"
    names = [c["name"] for c in quality["checks"]]
    assert "docs: head KL within floor" in names and "PPL rise, head" in names
    assert all(c["ok"] for c in quality["checks"])


def test_quality_gate_fails_above_the_kl_floor(tmp_path):
    quality = rep.build_receipt(quality_run(tmp_path, head_kl=0.0013))["quality"]
    assert quality["verdict"] == "FAIL"
    bad = [c["name"] for c in quality["checks"] if not c["ok"]]
    assert bad == ["docs: head KL within floor"]


def test_quality_gate_fails_below_the_top1_floor(tmp_path):
    quality = rep.build_receipt(quality_run(tmp_path, head_top1=0.980))["quality"]
    assert [c["name"] for c in quality["checks"] if not c["ok"]] == [
        "docs: head top-1 within floor"
    ]


def test_quality_gate_fails_on_ppl_rise(tmp_path):
    run = quality_run(tmp_path, rel=0.031, verdict="FAIL")
    quality = rep.build_receipt(run)["quality"]
    assert quality["verdict"] == "FAIL"
    assert [c["name"] for c in quality["checks"] if not c["ok"]] == ["PPL rise, head"]


def test_quality_gate_fails_without_the_compare_outputs(tmp_path):
    run = quality_run(tmp_path)
    (run / "quality" / "kl.json").unlink()
    assert rep.build_receipt(run)["quality"]["verdict"] == "FAIL"


def test_quality_gate_fails_when_the_collect_failed(tmp_path):
    run = quality_run(tmp_path)
    summary = run / "boots" / "B1" / "summary.txt"
    summary.write_text(summary.read_text().replace("LOAD HOOK rc=0", "LOAD HOOK rc=1"))
    quality = rep.build_receipt(run)["quality"]
    assert quality["verdict"] == "FAIL"
    assert "quality collect, B1" in [
        c["name"] for c in quality["checks"] if not c["ok"]
    ]


def test_quality_gate_fails_when_the_greedy_probe_after_load_fails(tmp_path):
    run = quality_run(tmp_path)
    summary = run / "boots" / "B1" / "summary.txt"
    summary.write_text(
        summary.read_text().replace("PROBE-AFTER-LOAD PASS", "PROBE-AFTER-LOAD FAIL")
    )
    assert rep.build_receipt(run)["quality"]["verdict"] == "FAIL"


def test_within_floor_boundaries():
    floor = kl_entry(0.0010, [0.0008, 0.0012], 0.986, [0.984, 0.988])
    assert rep.within_floor(kl_entry(0.0012, [0, 0], 0.984, [0, 0]), floor) == (
        True,
        True,
    )
    assert rep.within_floor(kl_entry(0.00121, [0, 0], 0.98399, [0, 0]), floor) == (
        False,
        False,
    )


def test_ppl_total_line_is_parsed():
    got = rep.parse_ppl_total(ppl_text(0.0123, "PASS"))
    assert got is not None
    assert (got["rel_pct"], got["verdict"], got["passages"]) == (1.23, "PASS", 6)
    assert rep.parse_ppl_total("passage 0: tokenization differs\n") is None


# ----------------------------------------------------------- the whole report


def receipt_run(tmp_path):
    """An ABBA run with one regression cell and one invalid cell."""
    base = {
        CELL_A: (bench(ttft=700.0), 4, EVICT_OK),
        CELL_B: (bench(ttft=700.0, n=16, conc=4), 16, EVICT_OK),
    }
    head_slow = {
        CELL_A: (bench(ttft=800.0), 4, EVICT_OK),
        CELL_B: (bench(ttft=700.0, n=16, conc=4), 16, EVICT_BAD),
    }
    return abba_run(
        tmp_path,
        base,
        head_slow,
        {k: (v[0], v[1], EVICT_OK) for k, v in head_slow.items()},
        base,
        cells="1024:512:1 1024:512:4",
        change_class="tool",
        recipe="flashnext-mtp2",
    )


def headings(text: str, level: int) -> list[str]:
    return re.findall(rf"^{'#' * level} (.+)$", text, flags=re.MULTILINE)


def test_report_writes_the_receipt(tmp_path, capsys):
    run = receipt_run(tmp_path)
    code = rep.main(["report", str(run)])
    out = capsys.readouterr().out
    assert code == 0  # exactness and quality were not asked for
    assert "speed (abba)" in out
    text = (run / "receipt.md").read_text()
    assert text.startswith("## RDNA receipt: hhhhhhhhhh against bbbbbbbbbb")
    assert headings(text, 3) == [
        "Environment",
        "Change class and reach",
        "Exactness",
        "Quality",
        "Speed",
        "Boots",
        "Raw results",
    ]
    assert "REGRESSION" in text  # cell A
    assert "INVALID" in text  # cell B: KFD eviction in head run B1
    assert "Change class: tool" in text
    data = json.loads((run / "receipt.json").read_text())
    assert data["gates"] == {"exact": "NOT RUN", "quality": "NOT RUN"}
    assert data["speed"]["counts"]["regression"] == 1
    assert data["speed"]["counts"]["invalid"] == 1


def test_pr_template_mirrors_the_receipt(tmp_path):
    """The RDNA block of the PR template has the receipt's sections, in order."""
    run = receipt_run(tmp_path)
    assert rep.main(["report", str(run)]) == 0
    receipt = (run / "receipt.md").read_text()
    template = (ROOT / ".github/PULL_REQUEST_TEMPLATE.md").read_text()
    block = template[template.index("## RDNA receipt") :].split("\n---\n")[0]
    assert headings(block, 3) == headings(receipt, 3)
    assert set(headings(block, 4)) <= {"Speculative acceptance length"}

    def gate_rows(text: str) -> list[str]:
        table = text.split("| gate | result |")[1].split("\n\n")[0]
        return [row.split("|")[1].strip() for row in table.splitlines()[2:]]

    assert (
        gate_rows(block)
        == gate_rows(receipt)
        == [
            "token-exact",
            "quality",
            "speed",
            "boots",
        ]
    )


def test_report_exit_code_follows_the_gates(tmp_path):
    run = quality_run(tmp_path, head_kl=0.0013)
    assert rep.main(["report", str(run)]) == 1
    run2 = tmp_path / "ok"
    run2.mkdir()
    assert rep.main(["report", str(run2)]) == 2  # no manifest


def test_a_failed_boot_fails_the_run(tmp_path):
    run = quality_run(tmp_path)
    assert rep.main(["boot", str(run), "B1", "head", "speed", "boots/B1", "rc=1"]) == 0
    result = rep.build_receipt(run)
    assert result["boot_failures"] == ["B1"] and result["ok"] is False


def test_cells_command_prints_the_default_list(capsys):
    assert rep.main(["cells"]) == 0
    assert capsys.readouterr().out.split() == rep.DEFAULT_CELLS.split()
    assert rep.main(["cells", "1024:512:1  16384:1024:4:6"]) == 0
    assert capsys.readouterr().out.split() == ["1024:512:1", "16384:1024:4:6"]
    assert rep.main(["cells", "none"]) == 0
    assert capsys.readouterr().out.strip() == "none"
    assert rep.main(["cells", "1024:512"]) == 2


def test_manifest_commands_replace_a_boot_with_the_same_id(tmp_path):
    run = tmp_path / "run"
    manifest(run, {"mode": "ab"}, [("A1", "base", "speed")])
    assert rep.main(["boot", str(run), "A1", "base", "speed", "boots/A1", "rc=3"]) == 0
    data = json.loads((run / "manifest.json").read_text())
    assert [b["rc"] for b in data["boots"]] == ["3"]
    assert data["settings"] == {"mode": "ab"}


def test_plan_diff_masks_tree_paths_before_comparing():
    base = "export PYTHONPATH='/b/tree'\ncmd /b/tree/py --flag 1\n"
    head = "export PYTHONPATH='/h/tree'\ncmd /h/tree/py --flag 1\n"
    masks = [("/b/tree", "<TREE>"), ("/h/tree", "<TREE>")]
    assert rep.plan_diff(base, head, masks) == []
    changed = head.replace("--flag 1", "--flag 2")
    assert rep.plan_diff(base, changed, masks) == [
        "-cmd <TREE>/py --flag 1",
        "+cmd <TREE>/py --flag 2",
    ]
    # the longer path is masked first, so a prefix cannot split it
    assert (
        rep.plan_diff("/b/tree/x", "/b/tree/x", [("/b", "<A>"), ("/b/tree", "<T>")])
        == []
    )


def test_plan_commands(tmp_path, capsys):
    plan = "x '--compilation-config' '{\"mode\":3}' y\n"
    (tmp_path / "a").write_text(plan)
    (tmp_path / "b").write_text(plan.replace("y", "z"))
    assert rep.main(["plan-field", str(tmp_path / "a"), "compilation"]) == 0
    assert capsys.readouterr().out.strip() == '{"mode":3}'
    out = tmp_path / "diff.txt"
    args = ["plan-diff", str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(out)]
    assert rep.main(args) == 0
    assert capsys.readouterr().out.strip() == "2 lines differ"
    assert "+x" in out.read_text()
    assert rep.main(["plan-diff", str(tmp_path / "a"), str(tmp_path / "none")]) == 0
    assert capsys.readouterr().out.strip() == "a launch plan is missing"
