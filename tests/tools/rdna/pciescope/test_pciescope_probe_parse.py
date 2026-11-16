# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``pciescope.probe.parse``.

Real data: the files under ``fixtures/probe/audit`` (captured on par1-cs25, see
the README there). The ``GRAMMAR_FIXTURE`` constants below are hand-written
format examples for the tags the legacy logs lack; their numbers are NOT
measurements.
"""

import dataclasses
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import pytest

_R = TypeVar("_R")
_RDNA = Path(__file__).resolve().parents[4] / "tools" / "rdna"
if str(_RDNA) not in sys.path:
    sys.path.insert(0, str(_RDNA))

from pciescope.probe import parse as P  # noqa: E402

AUDIT = Path(__file__).resolve().parent / "fixtures" / "probe" / "audit"
S1 = "s1_base.probe.log"
CUTS = (
    "w7_win1.head220.log",
    "w7_win3.head220.log",
    "w8_win4.head220.log",
    "w8_win5.head220.log",
)

# ---------------------------------------------------------------------------
# GRAMMAR_FIXTURE: format examples of the new tags, not measurements.
# ---------------------------------------------------------------------------

_STATS = "min_us=1 p50_us=1 p99_us=1 max_us=1 mean_us=1"

GRAMMAR_FIXTURE = """\
# hip=1,2 buses={0: 'ce', 1: 'D3'} A=[0] B=[1] extra=ignored
READY devices=2 tick_mhz=100.001,99.999 version=pciescope-probe/1 future=7
### version
VERSION pciescope-probe/1 modes=version,buses,hbw,curve future=1
END rc=0 ms=0.1
### hbw 0.5 1048576 128 h2d:0 bg:d2h_pg:1
HBW h2d:0 size=1048576 blocks=128 gbps=11.50 verify=ok secs=0.500 future=1
HBW bg:d2h_pg:1 size=1048576 blocks=128 gbps=3.25 verify=FAIL secs=0.501
HAGG size=1048576 fg_gbps=11.50 bg_gbps=3.25 n=2
END rc=0 ms=1012.5
### hlat 100 0 4096
HLAT dev=0 dir=h2d mem=pinned size=4096 n=100 min_us=1.000 p50_us=2.000 p99_us=3.000 max_us=4.000 mean_us=2.500
HLAT dev=0 dir=d2h mem=pageable size=4096 n=90 min_us=5.000 p50_us=6.000 p99_us=7.000 max_us=8.000 mean_us=6.500
END rc=0 ms=30.0
### hflag 50 0
HFLAG dev=0 n=50 rtt_min_us=10.000 rtt_p50_us=11.000 rtt_p99_us=12.000 rtt_max_us=13.000 rtt_mean_us=11.500
END rc=0 ms=5.0
### curve push 100 0 4096 0:1
CURVE mode=push src=0 dst=1 size=4096 blocks=1 n=100 min_us=1.100 p50_us=1.200 p99_us=1.300 max_us=1.400 mean_us=1.250 ev_us=1.260
CURVE mode=sdma src=1 dst=0 size=4096 blocks=0 n=20 min_us=9.000 p50_us=9.500 p99_us=10.000 max_us=11.000 mean_us=9.600
CURVE mode=pull src=0 dst=1 size=8 blocks=1 n=30 min_us=2.000 p50_us=2.100 p99_us=2.200 max_us=2.300 mean_us=2.150 ev_us=2.140 note=x
ERR timeout push 0:1 size=4096
ERR rc=1
END rc=1 ms=20.0
"""  # noqa: E501

# a legacy-style block without END that is followed by another block
LEGACY_ERR = """\
### bw 0.5 4096 128 p:0>1
ERR rc=3
### latc 2000 4096 0:1
LATC 0<->1 payload=4096 one_way_us=6.50
"""


def _read(name: str) -> str:
    return (AUDIT / name).read_text()


def _rows(log: P.ProbeLog, kind: type[_R]) -> list[_R]:
    return [r for b in log.blocks for r in b.rows if isinstance(r, kind)]


def _lat_values(rows: Sequence[P.Row]) -> list[float]:
    return [r.one_way_us for r in rows if isinstance(r, P.LatRow)]


def _stream(text: str) -> tuple[P.ProbeLog, list[P.Item]]:
    parser = P.StreamParser()
    items: list[P.Item] = []
    for line in text.split("\n"):
        items.extend(parser.feed(line))
    return parser.finish(), items


# ---------------------------------------------------------------------------
# Real data: s1_base.probe.log
# ---------------------------------------------------------------------------

BUSES = ["ce", "d3", "d8", "db", "0a", "0f", "14", "17"]


@pytest.fixture(scope="module")
def s1() -> P.ProbeLog:
    return P.parse_text(_read(S1))


def test_s1_devices(s1: P.ProbeLog) -> None:
    assert len(s1.devices) == 8
    assert s1.devices[0] == P.Device(
        idx=0, name="AMD Radeon Pro V620", arch="gfx1030", bdf="0000:ce:00.0"
    )
    assert [d.idx for d in s1.devices] == list(range(8))
    assert [d.bus for d in s1.devices] == BUSES
    assert {d.name for d in s1.devices} == {"AMD Radeon Pro V620"}
    assert {d.arch for d in s1.devices} == {"gfx1030"}


def test_s1_p2p(s1: P.ProbeLog) -> None:
    assert len(s1.p2p) == 56
    assert s1.p2p[0] == P.P2pAttr(0, 1, True, 2, True)
    assert s1.p2p[-1] == P.P2pAttr(7, 6, True, 2, True)
    assert {(p.src, p.dst) for p in s1.p2p} == {
        (a, b) for a in range(8) for b in range(8) if a != b
    }
    assert all(p.access and p.native_atomics and p.perf_rank == 2 for p in s1.p2p)


def test_s1_header(s1: P.ProbeLog) -> None:
    assert s1.header == {
        "hip": [1, 2, 3, 4, 6, 7, 8, 9],
        "buses": dict(enumerate(BUSES)),
        "A": [0, 1, 2, 3],
        "B": [4, 5, 6, 7],
    }
    assert s1.ready is None
    assert s1.version is None
    assert s1.unparsed == []


def test_s1_block_structure(s1: P.ProbeLog) -> None:
    # the audit log holds 66 commands: 48 bw, 9 latc, 9 a2a
    assert len(s1.blocks) == 66
    verbs = Counter(b.argv[0] for b in s1.blocks)
    assert verbs == {"bw": 48, "latc": 9, "a2a": 9}
    first = s1.blocks[0]
    assert first.cmd == "bw 0.5 65536 128 p:0>4"
    assert first.argv == ["bw", "0.5", "65536", "128", "p:0>4"]
    assert first.rc is None
    assert first.elapsed_ms is None
    assert all(b.complete for b in s1.blocks[:-1])
    assert all(b.rc is None and b.elapsed_ms is None for b in s1.blocks)


def test_s1_truncated_last_block(s1: P.ProbeLog) -> None:
    # legacy log without END: only a following block closes a block; the
    # last one is reported as not complete even though its row is there
    last = s1.blocks[-1]
    assert last.cmd == "a2a 2000 65536 0,1,2,3,4,5,6,7"
    assert last.complete is False
    assert last.rows == [P.A2aRow(65536, (0, 1, 2, 3, 4, 5, 6, 7), 62.58)]


def test_s1_row_counts(s1: P.ProbeLog) -> None:
    counts = Counter(type(r) for b in s1.blocks for r in b.rows)
    assert counts == {P.XferRow: 96, P.AggRow: 48, P.LatRow: 36, P.A2aRow: 9}
    assert all(r.concurrent for r in _rows(s1, P.LatRow))
    assert all(r.verify_ok for r in _rows(s1, P.XferRow))


def test_s1_exact_rows(s1: P.ProbeLog) -> None:
    b0 = s1.blocks[0]
    assert b0.rows == [
        P.XferRow("p:0>4", "p", 0, 4, False, 65536, 128, 25.19, True),
        P.AggRow(65536, 25.19, 0.0, 1),
    ]
    b1 = s1.blocks[1]
    assert b1.rows[0] == P.XferRow("l:0>4", "l", 0, 4, False, 65536, 128, 5.67, True)
    sdma = s1.blocks[4].rows[0]
    assert sdma == P.XferRow("s:0>4", "s", 0, 4, False, 65536, 128, 4.63, True)
    assert sdma.mode == "s"
    latc = next(b for b in s1.blocks if b.cmd == "latc 2000 4096 0:4 1:5 2:6 3:7")
    assert latc.rows[0] == P.LatRow(0, 4, 4096, 6.82, True)
    assert _lat_values(latc.rows) == [6.82, 6.76, 6.73, 6.66]
    a2a = next(b for b in s1.blocks if b.cmd == "a2a 2000 4096 0,1,2,3,4,5,6,7")
    assert a2a.rows == [P.A2aRow(4096, (0, 1, 2, 3, 4, 5, 6, 7), 16.28)]


def test_s1_eight_way_block(s1: P.ProbeLog) -> None:
    cmd = "bw 0.5 1048576 128 " + " ".join(
        f"p:{a}>{b}"
        for a, b in [(0, 4), (1, 5), (2, 6), (3, 7)]
        + [(4, 0), (5, 1)]
        + [(6, 2), (7, 3)]
    )
    block = next(b for b in s1.blocks if b.cmd == cmd)
    xs = [r for r in block.rows if isinstance(r, P.XferRow)]
    assert [(x.src, x.dst) for x in xs] == [
        (0, 4), (1, 5), (2, 6), (3, 7), (4, 0), (5, 1), (6, 2), (7, 3)
    ]  # fmt: skip
    assert xs[0].gbps == 4.36
    assert xs[6].gbps == 4.44
    assert block.rows[-1] == P.AggRow(1048576, 35.27, 0.0, 8)


# ---------------------------------------------------------------------------
# Real data: the four cut excerpts
# ---------------------------------------------------------------------------

# name: (devices, p2p, header hip, blocks, {row type: count}, last block rows)
CUT_FACTS = {
    "w7_win1.head220.log": (
        7, 42, [1, 2, 3, 4, 6, 7, 8], 18,
        {P.LatRow: 84, P.XferRow: 55, P.AggRow: 13}, 6,
    ),
    "w7_win3.head220.log": (
        7, 42, [1, 2, 3, 4, 6, 7, 8], 57,
        {P.XferRow: 57, P.AggRow: 56}, 1,
    ),
    "w8_win4.head220.log": (
        8, 56, [1, 2, 3, 4, 6, 7, 8, 9], 13,
        {P.LatRow: 112, P.XferRow: 22, P.AggRow: 8}, 4,
    ),
    "w8_win5.head220.log": (
        8, 56, [1, 2, 3, 4, 6, 7, 8, 9], 17,
        {P.LatRow: 112, P.XferRow: 13, P.AggRow: 13}, 2,
    ),
}  # fmt: skip


@pytest.mark.parametrize("name", CUTS)
def test_cut_excerpts(name: str) -> None:
    ndev, np2p, hip, nblocks, counts, last_rows = CUT_FACTS[name]
    log = P.parse_text(_read(name))
    assert len(log.devices) == ndev
    assert len(log.p2p) == np2p
    assert log.header is not None
    assert log.header["hip"] == hip
    assert log.header["buses"] == dict(enumerate(BUSES[:ndev]))
    assert len(log.blocks) == nblocks
    assert Counter(type(r) for b in log.blocks for r in b.rows) == counts
    assert log.unparsed == []
    assert log.blocks[-1].complete is False
    assert len(log.blocks[-1].rows) == last_rows
    assert all(b.complete for b in log.blocks[:-1])


def test_cut_truncated_block_is_partial() -> None:
    # the cut stops inside the last bw command: 6 of its 7 xfers, no AGG row
    log = P.parse_text(_read("w7_win1.head220.log"))
    last = log.blocks[-1]
    assert last.cmd == "bw 0.4 65536 128 p:0>1 p:1>2 p:2>3 p:3>0 p:4>5 p:5>6 p:6>4"
    assert [type(r) for r in last.rows] == [P.XferRow] * 6
    assert last.rows[-1] == P.XferRow(
        "p:5>6", "p", 5, 6, False, 65536, 128, 25.85, True
    )
    assert len(last.argv) == 4 + 7  # bw 0.4 65536 128 + 7 labels
    assert last.argv[:4] == ["bw", "0.4", "65536", "128"]


def test_cut_lat_values() -> None:
    w7 = P.parse_text(_read("w7_win1.head220.log"))
    lat = w7.blocks[0]
    assert lat.cmd.startswith("lat 2000 0 0:1 0:2")
    assert len(lat.rows) == 21
    assert lat.rows[0] == P.LatRow(0, 1, 0, 6.24, False)
    assert _lat_values(lat.rows[:3]) == [6.24, 6.27, 6.14]
    w8 = P.parse_text(_read("w8_win4.head220.log"))
    assert len(w8.blocks[0].rows) == 28
    assert w8.blocks[0].rows[0] == P.LatRow(0, 1, 0, 6.25, False)
    assert w8.blocks[0].rows[-1] == P.LatRow(6, 7, 0, 6.23, False)
    w8b = P.parse_text(_read("w8_win5.head220.log"))
    assert w8b.blocks[1].rows[-1] == P.LatRow(6, 7, 4096, 6.50, False)
    assert w8b.blocks[2].rows[1] == P.LatRow(0, 2, 16384, 7.01, False)


def test_cut_xfer_values() -> None:
    w7 = P.parse_text(_read("w7_win3.head220.log"))
    pull = next(r for r in _rows(w7, P.XferRow) if r.label == "l:0>4")
    assert pull == P.XferRow("l:0>4", "l", 0, 4, False, 4096, 128, 0.29, True)
    last = w7.blocks[-1].rows
    assert last == [P.XferRow("p:4>5", "p", 4, 5, False, 4096, 128, 8.94, True)]
    w8 = P.parse_text(_read("w8_win5.head220.log"))
    assert w8.blocks[-1].rows == [
        P.XferRow("p:6>7", "p", 6, 7, False, 65536, 128, 24.58, True),
        P.AggRow(65536, 24.58, 0.0, 1),
    ]


# ---------------------------------------------------------------------------
# StreamParser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [S1, *CUTS])
def test_stream_equals_parse_text(name: str) -> None:
    text = _read(name)
    log, items = _stream(text)
    assert log == P.parse_text(text)
    starts = [i for i in items if isinstance(i, P.BlockStart)]
    ends = [i for i in items if isinstance(i, P.BlockEnd)]
    assert len(starts) == len(log.blocks)
    # legacy log: every block but the last is ended by the next ###
    assert len(ends) == len(log.blocks) - 1
    assert [e.block for e in ends] == log.blocks[:-1]


def test_stream_incremental_items() -> None:
    parser = P.StreamParser()
    dev = parser.feed("DEV 0 AMD Radeon Pro V620 gfx1030 bus=0000:ce:00.0\n")
    assert dev == [P.Device(0, "AMD Radeon Pro V620", "gfx1030", "0000:ce:00.0")]
    assert parser.feed("P2P 0->1 access=1 perf_rank=2 native_atomics=1") == [
        P.P2pAttr(0, 1, True, 2, True)
    ]
    assert parser.feed("") == []
    assert parser.feed("# rep 4") == []
    assert parser.feed("garbage") == []
    assert parser.feed("### bw 0.5 65536 128 p:0>4") == [
        P.BlockStart("bw 0.5 65536 128 p:0>4", ("bw", "0.5", "65536", "128", "p:0>4"))
    ]
    x = P.XferRow("p:0>4", "p", 0, 4, False, 65536, 128, 25.19, True)
    assert parser.feed("X p:0>4 size=65536 blocks=128 gbps=25.19 verify=ok") == [x]
    agg = P.AggRow(65536, 25.19, 0.0, 1)
    assert parser.feed("AGG size=65536 fg_gbps=25.19 bg_gbps=0.00 n=1") == [agg]
    # the next ### completes the previous block (no END in legacy logs)
    out = parser.feed("### latc 2000 4096 0:4")
    assert len(out) == 2
    assert isinstance(out[0], P.BlockEnd)
    assert out[0].block.rows == [x, agg]
    assert out[0].block.complete is True
    assert out[1] == P.BlockStart("latc 2000 4096 0:4", ("latc", "2000", "4096", "0:4"))
    parser.feed("LATC 0<->4 payload=4096 one_way_us=6.82")
    log = parser.finish()
    assert log.blocks[-1].complete is False
    assert log.unparsed == ["garbage"]


def test_stream_end_line_closes_block() -> None:
    parser = P.StreamParser()
    parser.feed("### bw 0.5 4096 128 p:0>1")
    parser.feed("X p:0>1 size=4096 blocks=128 gbps=8.00 verify=ok")
    out = parser.feed("END rc=0 ms=12.5")
    assert len(out) == 1
    assert isinstance(out[0], P.BlockEnd)
    block = out[0].block
    assert (block.complete, block.rc, block.elapsed_ms) == (True, 0, 12.5)
    # the following ### does not emit a second BlockEnd
    nxt = parser.feed("### bw 0.5 4096 128 p:0>2")
    assert [type(i) for i in nxt] == [P.BlockStart]
    log = parser.finish()
    assert log.blocks[0] is block


def test_stream_finish_semantics() -> None:
    parser = P.StreamParser()
    parser.feed("### a")
    log = parser.finish()
    assert log.blocks[0].complete is False
    assert parser.finish() is log
    with pytest.raises(RuntimeError):
        parser.feed("### b")


# ---------------------------------------------------------------------------
# New grammar (format examples)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grammar() -> P.ProbeLog:
    return P.parse_text(GRAMMAR_FIXTURE)


def test_grammar_header_ready_version(grammar: P.ProbeLog) -> None:
    assert grammar.unparsed == []
    assert grammar.header == {
        "hip": [1, 2],
        "buses": {0: "ce", 1: "d3"},
        "A": [0],
        "B": [1],
    }
    assert grammar.ready == {
        "devices": "2",
        "tick_mhz": "100.001,99.999",
        "version": "pciescope-probe/1",
        "future": "7",
    }
    assert grammar.version == "pciescope-probe/1"
    assert [b.cmd for b in grammar.blocks] == [
        "version",
        "hbw 0.5 1048576 128 h2d:0 bg:d2h_pg:1",
        "hlat 100 0 4096",
        "hflag 50 0",
        "curve push 100 0 4096 0:1",
    ]
    assert all(b.complete for b in grammar.blocks)
    assert [b.rc for b in grammar.blocks] == [0, 0, 0, 0, 1]
    assert [b.elapsed_ms for b in grammar.blocks] == [0.1, 1012.5, 30.0, 5.0, 20.0]
    assert grammar.blocks[0].rows == []


def test_grammar_ready_and_version_items() -> None:
    _, items = _stream(GRAMMAR_FIXTURE)
    assert P.Ready(2, (100.001, 99.999), "pciescope-probe/1") in items
    assert P.Version("pciescope-probe/1", ("version", "buses", "hbw", "curve")) in items


def test_grammar_host_bandwidth(grammar: P.ProbeLog) -> None:
    assert grammar.blocks[1].rows == [
        P.HostXferRow("h2d:0", "h2d", 0, False, 1048576, 128, 11.5, True, 0.5),
        P.HostXferRow(
            "bg:d2h_pg:1", "d2h_pg", 1, True, 1048576, 128, 3.25, False, 0.501
        ),
        P.HostAggRow(1048576, 11.5, 3.25, 2),
    ]


def test_grammar_host_latency_and_flag(grammar: P.ProbeLog) -> None:
    assert grammar.blocks[2].rows == [
        P.HostLatRow(0, "h2d", "pinned", 4096, 100, 1.0, 2.0, 3.0, 4.0, 2.5),
        P.HostLatRow(0, "d2h", "pageable", 4096, 90, 5.0, 6.0, 7.0, 8.0, 6.5),
    ]
    assert grammar.blocks[3].rows == [
        P.HostFlagRow(0, 50, 10.0, 11.0, 12.0, 13.0, 11.5)
    ]


def test_grammar_curve_and_errors(grammar: P.ProbeLog) -> None:
    rows = grammar.blocks[4].rows
    assert rows[0] == P.CurveRow(
        "push", 0, 1, 4096, 1, 100, 1.1, 1.2, 1.3, 1.4, 1.25, 1.26
    )
    assert rows[1] == P.CurveRow(
        "sdma", 1, 0, 4096, 0, 20, 9.0, 9.5, 10.0, 11.0, 9.6, None
    )
    assert rows[2] == P.CurveRow("pull", 0, 1, 8, 1, 30, 2.0, 2.1, 2.2, 2.3, 2.15, 2.14)
    assert rows[3] == P.ErrRow(None, "timeout push 0:1 size=4096")
    assert rows[4] == P.ErrRow(1, "rc=1")


def test_grammar_stream_equals_parse_text(grammar: P.ProbeLog) -> None:
    log, items = _stream(GRAMMAR_FIXTURE)
    assert log == grammar
    ends = [i for i in items if isinstance(i, P.BlockEnd)]
    assert [e.block for e in ends] == grammar.blocks  # every block has END


def test_legacy_err_sets_rc_without_end() -> None:
    log = P.parse_text(LEGACY_ERR)
    assert [b.rc for b in log.blocks] == [3, None]
    assert [b.complete for b in log.blocks] == [True, False]
    assert log.blocks[0].rows == [P.ErrRow(3, "rc=3")]
    assert log.blocks[1].elapsed_ms is None


def test_end_overrides_err_rc() -> None:
    log = P.parse_text("### x\nERR rc=3\nEND rc=1 ms=2.0\n")
    assert log.blocks[0].rc == 1
    assert log.blocks[0].complete is True


def test_missing_ms_and_unknown_trailing_keys() -> None:
    log = P.parse_text("### x\nEND rc=0 future=1\n### y\nEND rc=2 ms=1.5 more=2\n")
    assert [(b.rc, b.elapsed_ms) for b in log.blocks] == [(0, None), (2, 1.5)]


# ---------------------------------------------------------------------------
# Garbage, empty input, leading rows, strictness
# ---------------------------------------------------------------------------


def test_empty_and_blank_input() -> None:
    for text in ("", "\n", "  \n\t\n\n"):
        log = P.parse_text(text)
        assert log == P.ProbeLog()


def test_garbage_goes_to_unparsed() -> None:
    text = (
        "Wall time: 2.02 seconds\n"
        "\x00\x01 binary\n"
        "UNKNOWN a=b\n"
        "# some other comment\n"
        "X\n"
        "LAT garbage\n"
        "DEV x y z\n"
        "P2P 0->1 access=2 perf_rank=1 native_atomics=0\n"
        "END\n"
        "VERSION modes=a\n"
        "# hip=broken\n"
    )
    log = P.parse_text(text)
    assert log.unparsed == [
        "Wall time: 2.02 seconds",
        "\x00\x01 binary",
        "UNKNOWN a=b",
        "# some other comment",
        "X",
        "LAT garbage",
        "DEV x y z",
        "P2P 0->1 access=2 perf_rank=1 native_atomics=0",
        "END",
        "VERSION modes=a",
        "# hip=broken",
    ]
    assert log.blocks == []
    assert log.devices == []
    assert log.header is None


def test_rows_before_first_block() -> None:
    log = P.parse_text(
        "LATC 0<->1 payload=64 one_way_us=1.50\n"
        "### a\n"
        "A2A payload=64 group=0,1 step_us=2.00\n"
    )
    assert [b.cmd for b in log.blocks] == ["", "a"]
    assert log.blocks[0].argv == []
    assert log.blocks[0].rows == [P.LatRow(0, 1, 64, 1.5, True)]
    assert log.blocks[0].complete is True
    assert log.blocks[1].complete is False


def test_end_without_block_opens_anonymous_block() -> None:
    log, items = _stream("END rc=0 ms=1.0\n")
    assert [b.cmd for b in log.blocks] == [""]
    assert log.blocks[0].complete is True
    assert [type(i) for i in items] == [P.BlockStart, P.BlockEnd]


def test_missing_end_between_blocks() -> None:
    log = P.parse_text(
        "### a\nAGG size=1 fg_gbps=1.00 bg_gbps=0.00 n=1\n"
        "### b\nAGG size=2 fg_gbps=2.00 bg_gbps=0.00 n=1\n"
    )
    assert [(b.cmd, b.complete, b.rc) for b in log.blocks] == [
        ("a", True, None),
        ("b", False, None),
    ]


def test_crlf_and_indentation() -> None:
    log = P.parse_text("### a\r\n  AGG size=1 fg_gbps=1.00 bg_gbps=0.00 n=1  \r\n")
    assert log.blocks[0].rows == [P.AggRow(1, 1.0, 0.0, 1)]
    assert log.unparsed == []


def test_bare_block_marker() -> None:
    log = P.parse_text("###\n### \n### x  y\n")
    assert [(b.cmd, b.argv) for b in log.blocks] == [
        ("", []),
        ("", []),
        ("x  y", ["x", "y"]),
    ]


def test_device_details() -> None:
    log = P.parse_text(
        "DEV 3 AMD Radeon Pro W7800 gfx1100 bus=0000:CB:00.0 extra=1\n"
        "DEV 3 AMD Radeon Pro W7800 gfx1100 bus=0000:CB:00.0\n"
        "DEV 4 Name gfx1030 bus=0000:0a:00.0\n"
    )
    assert log.devices == [
        P.Device(3, "AMD Radeon Pro W7800", "gfx1100", "0000:cb:00.0"),
        P.Device(4, "Name", "gfx1030", "0000:0a:00.0"),
    ]
    assert log.devices[0].bus == "cb"
    assert log.unparsed == []


@pytest.mark.parametrize(
    "line",
    [
        "DEV 0 AMD gfx1030 bus=0000:ce:00",
        "DEV 0 gfx1030 bus=0000:ce:00.0",
        "DEV 0 AMD Radeon gfx1030 bus=000:ce:00.0",
        "DEV 0 AMD Radeon gfx1030 bus=0000:ce:00.9",
        "P2P 0-1 access=1 perf_rank=2 native_atomics=1",
        "P2P 0->1 access=1 perf_rank=2",
        "LAT 0<->1 payload=4096",
        "LATC 0->1 payload=4096 one_way_us=1.0",
        "LAT 0<->1 payload=4x one_way_us=1.0",
        "A2A payload=4096 group= step_us=1.0",
        "A2A payload=4096 group=0,,1 step_us=1.0",
        "X p:0>1 size=1 blocks=1 gbps=1.0",
        "X q:0>1 size=1 blocks=1 gbps=1.0 verify=ok",
        "X p:0>1 size=1 blocks=1 gbps=1.0 verify=maybe",
        "X bg:bg:p:0>1 size=1 blocks=1 gbps=1.0 verify=ok",
        "AGG size=1 fg_gbps=1.0 n=1",
        "HBW h2d size=1 blocks=1 gbps=1.0 verify=ok secs=0.5",
        "HBW h2d:0 size=1 blocks=1 gbps=1.0 verify=ok",
        "HAGG size=1 fg_gbps=x bg_gbps=0 n=1",
        f"HLAT dev=0 dir=sideways mem=pinned size=1 n=1 {_STATS}",
        "HFLAG dev=0 n=1 rtt_min_us=1",
        f"CURVE mode=warp src=0 dst=1 size=1 blocks=1 n=1 {_STATS}",
        f"CURVE mode=push src=0 dst=1 size=1 blocks=1 n=1 {_STATS} ev_us=zz",
        "END ms=1.0",
        "END rc=x",
        "READY devices=2 version=v",
        "READY devices=2 tick_mhz=1,x version=v",
        "VERSION",
        "VERSION modes=a",
        "# hip=1 buses={0: 'zz'} A=[0] B=[1]",
        "# hip=1 buses={0 'ce'} A=[0] B=[1]",
        "# hip=1 buses={} A=[0]",
    ],
)
def test_malformed_record(line: str) -> None:
    log = P.parse_text(line + "\n")
    assert log.unparsed == [line]
    assert log.blocks == []
    with pytest.raises(P.ParseError, match="line 1"):
        P.parse_text(line + "\n", strict=True)


def test_strict_error_names_the_line() -> None:
    with pytest.raises(P.ParseError, match=r"line 3: .*missing gbps="):
        P.parse_text(
            "### a\nAGG size=1 fg_gbps=1.00 bg_gbps=0.00 n=1\n"
            "X p:0>1 size=1 blocks=1 verify=ok\n",
            strict=True,
        )
    assert issubclass(P.ParseError, ValueError)


def test_strict_tolerates_unknown_lines() -> None:
    log = P.parse_text("hello\n# rep 1\n### a\nEND rc=0\n", strict=True)
    assert log.unparsed == ["hello"]


def test_number_syntax_is_ascii_only() -> None:
    # str.isdigit() and float() accept these; the grammar does not
    for bad in ("size=\u0663", "size=1_0", "size=+", "size=1.5"):
        line = f"AGG {bad} fg_gbps=1.0 bg_gbps=0.0 n=1"
        assert P.parse_text(line).unparsed == [line]
    for bad in ("fg_gbps=1_0", "fg_gbps=\u0663.0", "fg_gbps=", "fg_gbps=1,5"):
        line = f"AGG size=1 {bad} bg_gbps=0.0 n=1"
        assert P.parse_text(line).unparsed == [line]
    ok = P.parse_text("AGG size=-1 fg_gbps=1e2 bg_gbps=.5 n=+3").blocks[0].rows[0]
    assert ok == P.AggRow(-1, 100.0, 0.5, 3)


# ---------------------------------------------------------------------------
# parse_header
# ---------------------------------------------------------------------------


def test_parse_header_variants() -> None:
    line = "# hip=1,2,3 buses={0: 'ce', 1: '0A', 2: 'db'} A=[0, 1] B=[2] tag=x"
    assert P.parse_header(line) == {
        "hip": [1, 2, 3],
        "buses": {0: "ce", 1: "0a", 2: "db"},
        "A": [0, 1],
        "B": [2],
    }
    assert P.parse_header("# hip= buses={} A=[] B=[]") == {
        "hip": [],
        "buses": {},
        "A": [],
        "B": [],
    }
    for not_header in ("", "hip=1", "# rep 3", "### x", "# hip=1 A=[0] B=[1]"):
        assert P.parse_header(not_header) is None


def test_rep_comment_is_ignored() -> None:
    log = P.parse_text("# rep 7\n# rep 8 note\n")
    assert log.unparsed == []
    assert log.header is None


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


def test_rows_are_frozen() -> None:
    row = P.AggRow(1, 1.0, 0.0, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.size = 2  # type: ignore[misc]
    dev = P.Device(0, "n", "gfx1030", "0000:ce:00.0")
    with pytest.raises(dataclasses.FrozenInstanceError):
        dev.idx = 1  # type: ignore[misc]


def test_parser_does_not_use_regular_expressions() -> None:
    source = (_RDNA / "pciescope" / "probe" / "parse.py").read_text()
    assert "import re" not in source
    assert "regex" not in source
