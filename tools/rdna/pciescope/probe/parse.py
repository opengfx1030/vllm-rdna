# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parser for the line-oriented output of ``p2p_probe`` (pure, no I/O).

The grammar is one record per line, ``TAG field...``. Legacy tags (``DEV``,
``P2P``, ``###``, ``LAT``, ``LATC``, ``A2A``, ``X``, ``AGG``, ``ERR``) and the
pciescope tags (``VERSION``, ``READY``, ``END``, ``HBW``, ``HAGG``, ``HLAT``,
``HFLAG``, ``CURVE``) are recognised; ``key=value`` tokens after the known
ones are ignored, so a newer binary does not break an older parser. Logs
written before ``END``/``READY`` existed parse too.

``parse_text`` parses a whole log; ``StreamParser`` takes one line at a time and
reports what each line completed, for live progress.

Lines that are not recognised are collected in ``ProbeLog.unparsed``. A line
with a recognised tag but a missing or malformed required field is collected
there as well, or raises ``ParseError`` with ``strict=True``.

Only string methods are used (no regular expressions): the parser runs in the
pciescope virtual environment, which has no third-party packages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from string import hexdigits


class ParseError(ValueError):
    """A recognised record has a missing or malformed required field."""


class _Bad(Exception):
    """Internal: the line being parsed is malformed (reason in args[0])."""


# ---------------------------------------------------------------------------
# Row types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Device:
    idx: int
    name: str
    arch: str
    bdf: str  # lower-case ``dddd:bb:dd.f``

    @property
    def bus(self) -> str:
        """Lower-case 2-digit PCI bus, e.g. ``ce``."""
        return self.bdf.split(":")[1]


@dataclass(frozen=True, slots=True)
class P2pAttr:
    src: int
    dst: int
    access: bool
    perf_rank: int
    native_atomics: bool


@dataclass(frozen=True, slots=True)
class LatRow:
    a: int
    b: int
    payload: int
    one_way_us: float
    concurrent: bool  # False for LAT, True for LATC


@dataclass(frozen=True, slots=True)
class A2aRow:
    payload: int
    group: tuple[int, ...]
    step_us: float


@dataclass(frozen=True, slots=True)
class XferRow:
    label: str  # as printed, e.g. ``bg:p:0>4``
    mode: str  # ``p`` (push), ``l`` (pull) or ``s`` (sdma)
    src: int
    dst: int
    bg: bool
    size: int
    blocks: int
    gbps: float
    verify_ok: bool


@dataclass(frozen=True, slots=True)
class AggRow:
    size: int
    fg_gbps: float
    bg_gbps: float
    n: int


@dataclass(frozen=True, slots=True)
class HostXferRow:
    label: str  # as printed, e.g. ``bg:d2h_pg:3``
    mode: str  # h2d, d2h, h2d_pg, d2h_pg, h2d_k, d2h_k
    dev: int
    bg: bool
    size: int
    blocks: int
    gbps: float
    verify_ok: bool
    secs: float


@dataclass(frozen=True, slots=True)
class HostAggRow:
    size: int
    fg_gbps: float
    bg_gbps: float
    n: int


@dataclass(frozen=True, slots=True)
class HostLatRow:
    dev: int
    direction: str  # ``h2d`` or ``d2h``
    mem: str  # ``pinned`` or ``pageable``
    size: int
    n: int
    min_us: float
    p50_us: float
    p99_us: float
    max_us: float
    mean_us: float


@dataclass(frozen=True, slots=True)
class HostFlagRow:
    dev: int
    n: int
    rtt_min_us: float
    rtt_p50_us: float
    rtt_p99_us: float
    rtt_max_us: float
    rtt_mean_us: float


@dataclass(frozen=True, slots=True)
class CurveRow:
    mode: str  # ``push``, ``pull`` or ``sdma``
    src: int
    dst: int
    size: int
    blocks: int
    n: int
    min_us: float
    p50_us: float
    p99_us: float
    max_us: float
    mean_us: float
    ev_us: float | None


@dataclass(frozen=True, slots=True)
class ErrRow:
    rc: int | None  # the ``rc=<n>`` value, None for ``ERR timeout ...`` and the like
    text: str  # the line after the ``ERR`` tag, e.g. ``rc=1``


Row = (
    LatRow
    | A2aRow
    | XferRow
    | AggRow
    | HostXferRow
    | HostAggRow
    | HostLatRow
    | HostFlagRow
    | CurveRow
    | ErrRow
)

_ROW_TYPES = (
    LatRow,
    A2aRow,
    XferRow,
    AggRow,
    HostXferRow,
    HostAggRow,
    HostLatRow,
    HostFlagRow,
    CurveRow,
    ErrRow,
)


# ---------------------------------------------------------------------------
# Structure types
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """The output of one command of a batch (or of a whole non-batch run)."""

    cmd: str  # text after ``### ``; empty for rows before the first ``###``
    argv: list[str]
    rows: list[Row] = field(default_factory=list)
    rc: int | None = None
    elapsed_ms: float | None = None
    complete: bool = False


@dataclass(frozen=True, slots=True)
class Ready:
    devices: int
    tick_mhz: tuple[float, ...]
    version: str


@dataclass(frozen=True, slots=True)
class Version:
    version: str
    modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BlockStart:
    cmd: str
    argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BlockEnd:
    block: Block


Item = Device | P2pAttr | BlockStart | Row | BlockEnd | Ready | Version


@dataclass
class ProbeLog:
    devices: list[Device] = field(default_factory=list)
    p2p: list[P2pAttr] = field(default_factory=list)
    header: dict[str, object] | None = None
    ready: dict[str, str] | None = None  # raw ``key=value`` pairs of READY
    version: str | None = None
    blocks: list[Block] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _End:
    rc: int
    ms: float | None


@dataclass(frozen=True, slots=True)
class _ReadyRecord:
    ready: Ready
    raw: dict[str, str]


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

_VERIFY = {"ok": True, "FAIL": False}
_DIRECTIONS = frozenset({"h2d", "d2h"})
_MEMORIES = frozenset({"pinned", "pageable"})
_CURVE_MODES = frozenset({"push", "pull", "sdma"})


def _is_uint(text: str) -> bool:
    """True for a non-empty run of ASCII digits (``str.isdigit`` alone accepts more)."""
    return text.isascii() and text.isdigit()


def _is_hex(text: str, length: int) -> bool:
    return len(text) == length and all(c in hexdigits for c in text)


def _parse_int(value: str, what: str) -> int:
    body = value[1:] if value.startswith(("+", "-")) else value
    if not _is_uint(body):
        raise _Bad(f"{what}: not an integer: {value!r}")
    return int(value)


def _parse_float(value: str, what: str) -> float:
    # float() also takes surrounding blanks, underscores and non-ASCII digits
    if not value or not value.isascii() or "_" in value or value != value.strip():
        raise _Bad(f"{what}: not a number: {value!r}")
    try:
        return float(value)
    except ValueError:
        raise _Bad(f"{what}: not a number: {value!r}") from None


def _pairs(tokens: list[str]) -> dict[str, str]:
    """Return the ``key=value`` tokens (the first occurrence of a key wins)."""
    kv: dict[str, str] = {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if sep and key:
            kv.setdefault(key, value)
    return kv


def _need(kv: dict[str, str], key: str) -> str:
    try:
        return kv[key]
    except KeyError:
        raise _Bad(f"missing {key}=") from None


def _kint(kv: dict[str, str], key: str) -> int:
    return _parse_int(_need(kv, key), key)


def _kfloat(kv: dict[str, str], key: str) -> float:
    return _parse_float(_need(kv, key), key)


def _kflag(kv: dict[str, str], key: str) -> bool:
    value = _need(kv, key)
    if value not in ("0", "1"):
        raise _Bad(f"{key}: expected 0 or 1: {value!r}")
    return value == "1"


def _kverify(kv: dict[str, str]) -> bool:
    value = _need(kv, "verify")
    try:
        return _VERIFY[value]
    except KeyError:
        raise _Bad(f"verify: expected ok or FAIL: {value!r}") from None


def _kchoice(kv: dict[str, str], key: str, choices: frozenset[str]) -> str:
    value = _need(kv, key)
    if value not in choices:
        raise _Bad(f"{key}: unexpected value {value!r}")
    return value


def _id_pair(token: str, sep: str) -> tuple[int, int]:
    """Parse ``<a><sep><b>`` with two unsigned integers."""
    left, found, right = token.partition(sep)
    if not found or not _is_uint(left) or not _is_uint(right):
        raise _Bad(f"expected <a>{sep}<b>, got {token!r}")
    return int(left), int(right)


def _first(rest: list[str], what: str) -> str:
    if not rest:
        raise _Bad(f"missing {what}")
    return rest[0]


# ---------------------------------------------------------------------------
# Record parsers: tokens after the tag -> parsed object
# ---------------------------------------------------------------------------


def _parse_bdf(token: str) -> str | None:
    """Return the lower-case ``dddd:bb:dd.f`` of ``token``, None when malformed."""
    parts = token.split(":")
    if len(parts) != 3:
        return None
    dom, bus, devfn = parts
    dev, dot, fn = devfn.partition(".")
    if not (_is_hex(dom, 4) and _is_hex(bus, 2) and _is_hex(dev, 2) and dot):
        return None
    if len(fn) != 1 or fn not in "01234567":
        return None
    return token.lower()


def _p_dev(rest: list[str]) -> Device:
    # DEV <idx> <name with spaces> <arch> bus=<dddd:bb:dd.f> [key=value ...]
    bad = _Bad("expected DEV <idx> <name> <arch> bus=<dddd:bb:dd.f>")
    pos = next((i for i, t in enumerate(rest) if t.startswith("bus=")), -1)
    if pos < 3 or not _is_uint(rest[0]):
        raise bad
    bdf = _parse_bdf(rest[pos][4:])
    if bdf is None:
        raise bad
    return Device(
        idx=int(rest[0]),
        name=" ".join(rest[1 : pos - 1]),
        arch=rest[pos - 1],
        bdf=bdf,
    )


def _p_p2p(rest: list[str]) -> P2pAttr:
    src, dst = _id_pair(_first(rest, "<a>-><b>"), "->")
    kv = _pairs(rest[1:])
    return P2pAttr(
        src=src,
        dst=dst,
        access=_kflag(kv, "access"),
        perf_rank=_kint(kv, "perf_rank"),
        native_atomics=_kflag(kv, "native_atomics"),
    )


def _lat(rest: list[str], *, concurrent: bool) -> LatRow:
    a, b = _id_pair(_first(rest, "<a><-><b>"), "<->")
    kv = _pairs(rest[1:])
    return LatRow(
        a=a,
        b=b,
        payload=_kint(kv, "payload"),
        one_way_us=_kfloat(kv, "one_way_us"),
        concurrent=concurrent,
    )


def _p_lat(rest: list[str]) -> LatRow:
    return _lat(rest, concurrent=False)


def _p_latc(rest: list[str]) -> LatRow:
    return _lat(rest, concurrent=True)


def _p_a2a(rest: list[str]) -> A2aRow:
    kv = _pairs(rest)
    group = _need(kv, "group").split(",")
    if not all(_is_uint(g) for g in group):
        raise _Bad(f"group: not a comma separated list of ids: {kv['group']!r}")
    return A2aRow(
        payload=_kint(kv, "payload"),
        group=tuple(int(g) for g in group),
        step_us=_kfloat(kv, "step_us"),
    )


def _p_x(rest: list[str]) -> XferRow:
    # label = [bg:]<p|l|s>:<src>><dst>
    label = _first(rest, "label")
    bg = label.startswith("bg:")
    mode, colon, pair = (label[3:] if bg else label).partition(":")
    if not colon or mode not in ("p", "l", "s"):
        raise _Bad(f"expected [bg:]<p|l|s>:<src>><dst>, got {label!r}")
    src, dst = _id_pair(pair, ">")
    kv = _pairs(rest[1:])
    return XferRow(
        label=label,
        mode=mode,
        src=src,
        dst=dst,
        bg=bg,
        size=_kint(kv, "size"),
        blocks=_kint(kv, "blocks"),
        gbps=_kfloat(kv, "gbps"),
        verify_ok=_kverify(kv),
    )


def _p_agg(rest: list[str]) -> AggRow:
    kv = _pairs(rest)
    return AggRow(
        size=_kint(kv, "size"),
        fg_gbps=_kfloat(kv, "fg_gbps"),
        bg_gbps=_kfloat(kv, "bg_gbps"),
        n=_kint(kv, "n"),
    )


def _p_hbw(rest: list[str]) -> HostXferRow:
    # label = [bg:]<mode>:<dev>
    label = _first(rest, "label")
    bg = label.startswith("bg:")
    mode, colon, dev = (label[3:] if bg else label).partition(":")
    if (
        not colon
        or not mode
        or not mode.isascii()
        or not mode.replace("_", "").isalnum()
        or not _is_uint(dev)
    ):
        raise _Bad(f"expected [bg:]<mode>:<dev>, got {label!r}")
    kv = _pairs(rest[1:])
    return HostXferRow(
        label=label,
        mode=mode,
        dev=int(dev),
        bg=bg,
        size=_kint(kv, "size"),
        blocks=_kint(kv, "blocks"),
        gbps=_kfloat(kv, "gbps"),
        verify_ok=_kverify(kv),
        secs=_kfloat(kv, "secs"),
    )


def _p_hagg(rest: list[str]) -> HostAggRow:
    kv = _pairs(rest)
    return HostAggRow(
        size=_kint(kv, "size"),
        fg_gbps=_kfloat(kv, "fg_gbps"),
        bg_gbps=_kfloat(kv, "bg_gbps"),
        n=_kint(kv, "n"),
    )


def _p_hlat(rest: list[str]) -> HostLatRow:
    kv = _pairs(rest)
    return HostLatRow(
        dev=_kint(kv, "dev"),
        direction=_kchoice(kv, "dir", _DIRECTIONS),
        mem=_kchoice(kv, "mem", _MEMORIES),
        size=_kint(kv, "size"),
        n=_kint(kv, "n"),
        min_us=_kfloat(kv, "min_us"),
        p50_us=_kfloat(kv, "p50_us"),
        p99_us=_kfloat(kv, "p99_us"),
        max_us=_kfloat(kv, "max_us"),
        mean_us=_kfloat(kv, "mean_us"),
    )


def _p_hflag(rest: list[str]) -> HostFlagRow:
    kv = _pairs(rest)
    return HostFlagRow(
        dev=_kint(kv, "dev"),
        n=_kint(kv, "n"),
        rtt_min_us=_kfloat(kv, "rtt_min_us"),
        rtt_p50_us=_kfloat(kv, "rtt_p50_us"),
        rtt_p99_us=_kfloat(kv, "rtt_p99_us"),
        rtt_max_us=_kfloat(kv, "rtt_max_us"),
        rtt_mean_us=_kfloat(kv, "rtt_mean_us"),
    )


def _p_curve(rest: list[str]) -> CurveRow:
    kv = _pairs(rest)
    return CurveRow(
        mode=_kchoice(kv, "mode", _CURVE_MODES),
        src=_kint(kv, "src"),
        dst=_kint(kv, "dst"),
        size=_kint(kv, "size"),
        blocks=_kint(kv, "blocks"),
        n=_kint(kv, "n"),
        min_us=_kfloat(kv, "min_us"),
        p50_us=_kfloat(kv, "p50_us"),
        p99_us=_kfloat(kv, "p99_us"),
        max_us=_kfloat(kv, "max_us"),
        mean_us=_kfloat(kv, "mean_us"),
        ev_us=_kfloat(kv, "ev_us") if "ev_us" in kv else None,
    )


def _p_err(rest: list[str]) -> ErrRow:
    rc: int | None = None
    for token in rest:
        if token.startswith("rc="):
            try:
                rc = _parse_int(token[3:], "rc")
            except _Bad:
                continue
            break
    return ErrRow(rc=rc, text=" ".join(rest))


def _p_end(rest: list[str]) -> _End:
    kv = _pairs(rest)
    return _End(
        rc=_kint(kv, "rc"),
        ms=_kfloat(kv, "ms") if "ms" in kv else None,
    )


def _p_ready(rest: list[str]) -> _ReadyRecord:
    kv = _pairs(rest)
    csv = _need(kv, "tick_mhz")
    ticks = tuple(_parse_float(v, "tick_mhz") for v in csv.split(",")) if csv else ()
    ready = Ready(
        devices=_kint(kv, "devices"),
        tick_mhz=ticks,
        version=_need(kv, "version"),
    )
    return _ReadyRecord(ready, kv)


def _p_version(rest: list[str]) -> Version:
    if not rest or "=" in rest[0]:
        raise _Bad("expected VERSION <string> [modes=<csv>]")
    modes = _pairs(rest[1:]).get("modes", "")
    return Version(
        version=rest[0],
        modes=tuple(m for m in modes.split(",") if m),
    )


_RECORDS: dict[str, Callable[[list[str]], object]] = {
    "DEV": _p_dev,
    "P2P": _p_p2p,
    "LAT": _p_lat,
    "LATC": _p_latc,
    "A2A": _p_a2a,
    "X": _p_x,
    "AGG": _p_agg,
    "ERR": _p_err,
    "VERSION": _p_version,
    "READY": _p_ready,
    "END": _p_end,
    "HBW": _p_hbw,
    "HAGG": _p_hagg,
    "HLAT": _p_hlat,
    "HFLAG": _p_hflag,
    "CURVE": _p_curve,
}

# ---------------------------------------------------------------------------
# Header line written by run_matrix.py
# ---------------------------------------------------------------------------


def _id_list(text: str) -> list[int] | None:
    parts = [p.strip() for p in text.split(",")] if text.strip() else []
    if not all(_is_uint(p) for p in parts):
        return None
    return [int(p) for p in parts]


def _bracketed(text: str, prefix: str, close: str) -> tuple[str, str] | None:
    """Split ``<prefix>inner<close>rest`` (leading blanks allowed) at ``close``.

    Returns ``(inner, rest)``; None when the prefix is absent, ``close`` is
    missing, or ``rest`` does not start with a blank (or is non-empty).
    """
    text = text.lstrip()
    if not text.startswith(prefix):
        return None
    inner, found, rest = text[len(prefix) :].partition(close)
    if not found or (rest and not rest[0].isspace()):
        return None
    return inner, rest


def parse_header(line: str) -> dict[str, object] | None:
    """Parse ``# hip=<csv> buses={0: 'ce', ...} A=[..] B=[..]``.

    Returns ``{"hip": [int], "buses": {int: str}, "A": [int], "B": [int]}`` with
    lower-case buses, or None when ``line`` is not such a header. Keys after
    ``B=[..]`` are ignored.
    """
    text = line.strip()
    if not text.startswith("#"):
        return None
    head = text[1:].split(None, 1)
    if len(head) != 2 or not head[0].startswith("hip="):
        return None
    hip = _id_list(head[0][4:])
    rest = head[1]
    got = _bracketed(rest, "buses={", "}")
    if got is None or hip is None:
        return None
    body, rest = got
    got = _bracketed(rest, "A=[", "]")
    if got is None:
        return None
    a_text, rest = got
    got = _bracketed(rest, "B=[", "]")
    if got is None:
        return None
    b_text = got[0]
    a = _id_list(a_text)
    b = _id_list(b_text)
    if a is None or b is None:
        return None
    buses: dict[int, str] = {}
    for entry in body.split(",") if body.strip() else []:
        key, colon, value = entry.partition(":")
        value = value.strip()
        ok = (
            colon
            and _is_uint(key.strip())
            and len(value) == 4
            and value[0] == value[-1] == "'"
            and _is_hex(value[1:-1], 2)
        )
        if not ok:
            return None
        buses[int(key)] = value[1:-1].lower()
    return {"hip": hip, "buses": buses, "A": a, "B": b}


def _is_rep_comment(text: str) -> bool:
    words = text[1:].split()
    return len(words) >= 2 and words[0] == "rep" and _is_uint(words[1])


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


class StreamParser:
    """Incremental parser: ``feed`` one line at a time, ``finish`` at the end.

    ``feed`` returns the items the line completed, in order. A ``BlockEnd`` is
    emitted when an ``END`` line arrives or when the next ``###`` starts a new
    block, whichever comes first; the ``###`` line then returns
    ``[BlockEnd(previous), BlockStart(next)]``. A row outside any block opens
    an anonymous block (``cmd=""``) first. ``DEV`` lines are added to
    ``ProbeLog.devices`` unless an identical device is already there (a batch
    ``buses`` command repeats them). Header, ``# rep`` comments, blank lines and
    unparsed lines return ``[]``.

    Args:
        strict: raise ``ParseError`` for a recognised record with a missing or
            malformed required field instead of collecting the line in
            ``ProbeLog.unparsed``.

    """

    def __init__(self, *, strict: bool = False) -> None:
        self._strict = strict
        self._log = ProbeLog()
        self._cur: Block | None = None
        self._lineno = 0
        self._finished = False

    def feed(self, line: str) -> list[Item]:
        """Consume one line (with or without its newline)."""
        if self._finished:
            raise RuntimeError("feed() after finish()")
        self._lineno += 1
        text = line.strip()
        if not text:
            return []
        if text == "###" or text.startswith("### "):
            return self._start_block(text[3:].strip())
        if text.startswith("#"):
            return self._comment(text)
        tag, *rest = text.split()
        parser = _RECORDS.get(tag)
        if parser is None:
            self._log.unparsed.append(text)
            return []
        try:
            parsed = parser(rest)
        except _Bad as exc:
            return self._reject(text, str(exc))
        return self._accept(parsed)

    def finish(self) -> ProbeLog:
        """Close a truncated last block (``complete=False``) and return the log."""
        if not self._finished:
            self._finished = True
            if self._cur is not None:
                self._close(self._cur, None, complete=False)
                self._cur = None
        return self._log

    def _reject(self, text: str, reason: str) -> list[Item]:
        if self._strict:
            raise ParseError(f"line {self._lineno}: {reason}: {text!r}")
        self._log.unparsed.append(text)
        return []

    def _comment(self, text: str) -> list[Item]:
        header = parse_header(text)
        if header is not None:
            self._log.header = header
        elif text[1:].lstrip().startswith("hip="):
            return self._reject(text, "malformed header")
        elif not _is_rep_comment(text):
            self._log.unparsed.append(text)
        return []

    def _start_block(self, cmd: str) -> list[Item]:
        items: list[Item] = []
        if self._cur is not None:
            items.append(self._end_block(None, complete=True))
        block = Block(cmd=cmd, argv=cmd.split())
        self._log.blocks.append(block)
        self._cur = block
        items.append(BlockStart(cmd, tuple(block.argv)))
        return items

    def _ensure_block(self, items: list[Item]) -> Block:
        if self._cur is None:
            block = Block(cmd="", argv=[])
            self._log.blocks.append(block)
            self._cur = block
            items.append(BlockStart("", ()))
        return self._cur

    def _end_block(self, end: _End | None, *, complete: bool) -> BlockEnd:
        block = self._cur
        assert block is not None
        self._close(block, end, complete=complete)
        self._cur = None
        return BlockEnd(block)

    @staticmethod
    def _close(block: Block, end: _End | None, *, complete: bool) -> None:
        block.complete = complete
        if end is not None:
            block.rc = end.rc
            block.elapsed_ms = end.ms
            return
        for row in reversed(block.rows):
            if isinstance(row, ErrRow) and row.rc is not None:
                block.rc = row.rc
                return

    def _accept(self, parsed: object) -> list[Item]:
        log = self._log
        if isinstance(parsed, Device):
            if parsed not in log.devices:
                log.devices.append(parsed)
            return [parsed]
        if isinstance(parsed, P2pAttr):
            log.p2p.append(parsed)
            return [parsed]
        if isinstance(parsed, Version):
            log.version = parsed.version
            return [parsed]
        if isinstance(parsed, _ReadyRecord):
            log.ready = parsed.raw
            return [parsed.ready]
        items: list[Item] = []
        block = self._ensure_block(items)
        if isinstance(parsed, _End):
            items.append(self._end_block(parsed, complete=True))
        elif isinstance(parsed, _ROW_TYPES):
            block.rows.append(parsed)
            items.append(parsed)
        return items


def parse_text(text: str, *, strict: bool = False) -> ProbeLog:
    """Parse a whole log (see the module docstring for the rules)."""
    parser = StreamParser(strict=strict)
    for line in text.split("\n"):
        parser.feed(line)
    return parser.finish()
