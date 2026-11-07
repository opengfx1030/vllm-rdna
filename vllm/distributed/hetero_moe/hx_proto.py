# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire protocol v1 of ``hx``, the hub <-> expert-server push protocol.

Design: docs/rdna2/hetero-moe-design-2026-11-07.md sections 6.4 and 9.4. The numbers
the design leaves open are fixed here and in ``csrc/rocm/hetero_moe/hx_core.cuh``.
This module is the host-side single source of truth: constants, sequence-number
arithmetic, the byte layouts of the server inbox and of the hub reply buffer, the
64-byte step header and the abort-record codec.

It imports the standard library only, so it loads on any host (no torch, no numpy,
no vllm). ``tests/distributed/hetero_moe/test_hx_proto.py`` keeps every
``#define HX_<NAME> <literal>`` of the header equal to the constant of the same name
here, so keep the names identical when adding a shared constant.

Abort record (u64, per process; the ``live`` word has the same layout with
``HX_REC_LIVE`` set)::

    bit  0      valid
    bit  1      live (only in the live word)
    bit  2      peer seen (reserved)
    bits 4-7    phase (HX_PH_*)
    bits 8-15   peer index (phase 5: the mismatching header field, HX_MM_*)
    bits 16-31  measured wait in ms, saturating at 65535
    bits 32-63  low 32 bits of the sequence number of the wait

The sequence number of a layer wait is ``epoch * 64 + layer + 1``; the begin phases
(``HX_PH_SRV_BEGIN``, ``HX_PH_HDR``) wait per step, not per layer, and record the raw
epoch they expected instead.
"""

import operator
import struct
from dataclasses import dataclass
from types import MappingProxyType

# ---------------------------------------------------------------------------
# Protocol constants (equal to hx_core.cuh; the names are shared)
# ---------------------------------------------------------------------------
HX_VERSION = 1
HX_MAGIC = 0x4858_5F50_524F_5431  # b"HX_PROT1"
HX_OPCODE_MOE = 1  # opcodes 2 and 3 are reserved
HX_MAX_LAYERS = 64
HX_MAX_SERVERS = 8
HX_MAX_COLD = 8
HX_MAX_SLOTS = 4
HX_MAX_EXPERTS = 512
HX_MAX_TOPK = 16
HX_PAGE = 4096  # the control page and every payload stride are 4 KiB aligned
HX_HDR_BYTES = 64
HX_CTX_WORDS = 128  # int64 words of a device ctx tensor
HX_FLAG_STRIDE = 64  # one flag per 64 B line

# Control-page offsets. The inbox holds the header at 0 and the request flags at
# HX_CTRL_REQ_FLAG; the reply buffer holds the reply flags at HX_CTRL_REPLY_FLAG.
# The host abort poke word lives at HX_CTRL_HOST_ABORT of the OWN control page.
HX_CTRL_REQ_FLAG = 0x100
HX_CTRL_REPLY_FLAG = 0
HX_CTRL_HOST_ABORT = 0xF00

# Step header field offsets (bytes inside the 64 B header; hdr_seq is stored last).
HX_HDR_OFF_MAGIC = 0
HX_HDR_OFF_EPOCH = 8
HX_HDR_OFF_T = 16
HX_HDR_OFF_BUCKET = 20
HX_HDR_OFF_OPCODE = 24
HX_HDR_OFF_VERSION = 26
HX_HDR_OFF_MODE = 28
HX_HDR_OFF_PLAN = 32
HX_HDR_OFF_HUB = 40
HX_HDR_OFF_SEQ = 56

# Abort record phases. 1 and 6 (arrival barriers) are reserved: no kernel produces
# them. 8 is the host poke written after the host saw a peer die.
HX_PH_DISPATCH = 1
HX_PH_COMBINE = 2
HX_PH_SRV_WAIT = 3
HX_PH_SRV_BEGIN = 4
HX_PH_HDR = 5
HX_PH_PUSH = 6
HX_PH_PLAN = 7
HX_PH_HOST = 8

# Header mismatch fields (the peer byte of a phase 5 record).
HX_MM_MAGIC = 1
HX_MM_PLAN = 2
HX_MM_EPOCH = 3
HX_MM_T = 4
HX_MM_BUCKET = 5
HX_MM_OPCODE = 6
HX_MM_VERSION = 7
HX_MM_SKIPPED = 8

# Record flag bits.
HX_REC_VALID = 1
HX_REC_LIVE = 2
HX_REC_PEER_SEEN = 4  # reserved

# Words of the hx_status vector.
HX_ST_RECORD = 0
HX_ST_EPOCH = 1
HX_ST_LIVE = 2
HX_ST_STEPS = 3
HX_ST_DISPATCHES = 4
HX_ST_COMBINES = 5
HX_ST_PAIRS_HUB = 6
HX_ST_PAIRS_REMOTE = 7
HX_STATUS_WORDS = 8

# Columns of the optional per-layer pair counters (cumulative, hub only).
HX_LSTAT_COLS = 4
HX_LSTAT_HUB = 0  # pairs kept on the hub
HX_LSTAT_COLD = 1  # sent to a server: expert not resident on the hub
HX_LSTAT_SPILL = 2  # sent to a server by the spill rule (resident, rank >= K)
HX_LSTAT_LOST = 3  # plan errors: owned expert with no hub copy and no server

# Defaults of the runtime parameters (everything else is carried in the ctx).
HX_DEFAULT_ROWS_CAP = 2048
HX_DEFAULT_TOPK = 10
HX_DEFAULT_HIDDEN = 2560
HX_DEFAULT_NSLOTS = 2
HX_DEFAULT_WAIT_MS = 2000
HX_DEFAULT_BEGIN_WAIT_MS = 600000

HX_NAN_F16 = 0x7E00  # output of an aborted combine, every element

U32_MAX = 0xFFFF_FFFF
U64_MAX = 0xFFFF_FFFF_FFFF_FFFF

HX_PHASE_NAMES = MappingProxyType(
    {
        HX_PH_DISPATCH: "dispatch arrival barrier",
        HX_PH_COMBINE: "combine wait for server",
        HX_PH_SRV_WAIT: "server request wait",
        HX_PH_SRV_BEGIN: "server begin header wait",
        HX_PH_HDR: "header mismatch",
        HX_PH_PUSH: "push arrival barrier",
        HX_PH_PLAN: "plan error",
        HX_PH_HOST: "host abort poke",
    }
)
# Which process writes each phase (None: either).
HX_PHASE_ROLES = MappingProxyType(
    {
        HX_PH_DISPATCH: "hub",
        HX_PH_COMBINE: "hub",
        HX_PH_SRV_WAIT: "server",
        HX_PH_SRV_BEGIN: "server",
        HX_PH_HDR: "server",
        HX_PH_PUSH: "server",
        HX_PH_PLAN: "hub",
        HX_PH_HOST: None,
    }
)
HX_MISMATCH_NAMES = MappingProxyType(
    {
        HX_MM_MAGIC: "magic",
        HX_MM_PLAN: "plan hash",
        HX_MM_EPOCH: "epoch",
        HX_MM_T: "T",
        HX_MM_BUCKET: "bucket",
        HX_MM_OPCODE: "opcode",
        HX_MM_VERSION: "version",
        HX_MM_SKIPPED: "skipped epoch",
    }
)
# Phases whose record carries the raw expected epoch instead of a layer sequence.
_RAW_EPOCH_PHASES = frozenset({HX_PH_SRV_BEGIN, HX_PH_HDR})
_HEX = frozenset("0123456789abcdefABCDEF")


def _int(name: str, value: int, lo: int, hi: int) -> int:
    """Coerce to int (numpy integers pass) and check ``lo <= value <= hi``."""
    try:
        v = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if not lo <= v <= hi:
        raise ValueError(f"{name}={v} is outside [{lo}, {hi}]")
    return v


def align_up(n: int, a: int) -> int:
    """Round ``n`` up to the next multiple of ``a`` (any ``a`` > 0, ``n`` >= 0)."""
    n = _int("n", n, 0, 1 << 128)
    a = _int("a", a, 1, 1 << 128)
    return (n + a - 1) // a * a


# ---------------------------------------------------------------------------
# Sequence numbers (u64, wrap-safe)
# ---------------------------------------------------------------------------
def seq_of(epoch: int, layer: int) -> int:
    """Flag value of a layer: ``epoch * 64 + layer + 1`` modulo 2**64."""
    epoch = _int("epoch", epoch, 0, U64_MAX)
    layer = _int("layer", layer, 0, HX_MAX_LAYERS - 1)
    return (epoch * HX_MAX_LAYERS + layer + 1) & U64_MAX


def seq_before(a: int, b: int) -> bool:
    """``int64(a - b) < 0`` in u64 arithmetic: ``a`` is a sequence before ``b``."""
    a = _int("a", a, 0, U64_MAX)
    b = _int("b", b, 0, U64_MAX)
    return ((a - b) & U64_MAX) >= 1 << 63


def seq_reached(flag: int, seq: int) -> bool:
    """A flag of value ``flag`` satisfies a wait for ``seq``."""
    return not seq_before(flag, seq)


def slot_of(layer: int, nslots: int) -> int:
    """Payload slot (and flag index) of ``layer``."""
    layer = _int("layer", layer, 0, HX_MAX_LAYERS - 1)
    return layer % _int("nslots", nslots, 1, HX_MAX_SLOTS)


def plan_hash(plan_id: str) -> int:
    """Header plan hash of a plan id: 16 hex characters read as one u64."""
    if (
        not isinstance(plan_id, str)
        or len(plan_id) != 16
        or not _HEX.issuperset(plan_id)
    ):
        raise ValueError(f"plan_id must be exactly 16 hex characters, got {plan_id!r}")
    return int(plan_id, 16)


# ---------------------------------------------------------------------------
# Byte layouts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InboxLayout:
    """Layout of one server inbox (server VRAM, written by the hub).

    Offsets are bytes from the start of the allocation unless the name says it is
    inside a slot. The control page holds the header, the request flags and the host
    abort word; payload slot ``j`` follows at ``control_bytes + j * slot_stride``.
    """

    rows_cap: int
    topk: int
    hidden: int
    nslots: int
    control_bytes: int
    hdr_off: int
    hdr_bytes: int
    req_flag_base: int
    flag_stride: int
    host_abort_off: int
    x_bytes: int
    ids_bytes: int
    w_bytes: int
    ids_off: int  # inside a slot (x is at +0)
    w_off: int  # inside a slot
    slot_stride: int
    total_bytes: int

    def _slot(self, j: int) -> int:
        return _int("slot", j, 0, self.nslots - 1)

    def req_flag_off(self, j: int) -> int:
        return self.req_flag_base + self.flag_stride * self._slot(j)

    def slot_off(self, j: int) -> int:
        return self.control_bytes + self._slot(j) * self.slot_stride


@dataclass(frozen=True)
class ReplyLayout:
    """Layout of the hub reply buffer (hub VRAM, written by all servers).

    Flag ``(s, j)`` and reply slot ``(s, j)`` are indexed ``s * nslots + j``.
    """

    rows_cap: int
    hidden: int
    nslots: int
    nsrv: int
    control_bytes: int
    flag_base: int
    flag_stride: int
    host_abort_off: int
    y_bytes: int
    y_stride: int
    total_bytes: int

    def _index(self, s: int, j: int) -> int:
        s = _int("server", s, 0, self.nsrv - 1)
        return s * self.nslots + _int("slot", j, 0, self.nslots - 1)

    def flag_off(self, s: int, j: int) -> int:
        return self.flag_base + self.flag_stride * self._index(s, j)

    def y_off(self, s: int, j: int) -> int:
        return self.control_bytes + self._index(s, j) * self.y_stride


def _geometry(rows_cap: int, hidden: int, nslots: int) -> tuple[int, int, int]:
    rows_cap = _int("rows_cap", rows_cap, 1, U32_MAX)
    hidden = _int("hidden", hidden, 8, U32_MAX)
    if hidden % 8:
        raise ValueError(f"hidden={hidden} must be a multiple of 8 (16 B rows)")
    return rows_cap, hidden, _int("nslots", nslots, 1, HX_MAX_SLOTS)


def inbox_layout(
    rows_cap: int = HX_DEFAULT_ROWS_CAP,
    topk: int = HX_DEFAULT_TOPK,
    hidden: int = HX_DEFAULT_HIDDEN,
    nslots: int = HX_DEFAULT_NSLOTS,
) -> InboxLayout:
    """Inbox of one server: x ``[rows_cap][hidden]`` fp16, ids ``[rows_cap][topk]``
    int16 and w ``[rows_cap][topk]`` fp32 per slot behind a 4 KiB control page."""
    rows_cap, hidden, nslots = _geometry(rows_cap, hidden, nslots)
    topk = _int("topk", topk, 1, HX_MAX_TOPK)
    x_bytes = rows_cap * hidden * 2
    ids_bytes = rows_cap * topk * 2
    w_bytes = rows_cap * topk * 4
    ids_off = x_bytes
    w_off = x_bytes + align_up(ids_bytes, 16)
    slot_stride = align_up(w_off + w_bytes, HX_PAGE)
    return InboxLayout(
        rows_cap=rows_cap,
        topk=topk,
        hidden=hidden,
        nslots=nslots,
        control_bytes=HX_PAGE,
        hdr_off=0,
        hdr_bytes=HX_HDR_BYTES,
        req_flag_base=HX_CTRL_REQ_FLAG,
        flag_stride=HX_FLAG_STRIDE,
        host_abort_off=HX_CTRL_HOST_ABORT,
        x_bytes=x_bytes,
        ids_bytes=ids_bytes,
        w_bytes=w_bytes,
        ids_off=ids_off,
        w_off=w_off,
        slot_stride=slot_stride,
        total_bytes=HX_PAGE + nslots * slot_stride,
    )


def reply_layout(rows_cap: int, hidden: int, nslots: int, nsrv: int) -> ReplyLayout:
    """Reply buffer of one hub for ``nsrv`` servers: a control page with one flag
    per (server, slot) followed by one fp16 ``[rows_cap][hidden]`` slot each."""
    rows_cap, hidden, nslots = _geometry(rows_cap, hidden, nslots)
    nsrv = _int("nsrv", nsrv, 1, HX_MAX_SERVERS)
    y_bytes = rows_cap * hidden * 2
    y_stride = align_up(y_bytes, HX_PAGE)
    return ReplyLayout(
        rows_cap=rows_cap,
        hidden=hidden,
        nslots=nslots,
        nsrv=nsrv,
        control_bytes=HX_PAGE,
        flag_base=HX_CTRL_REPLY_FLAG,
        flag_stride=HX_FLAG_STRIDE,
        host_abort_off=HX_CTRL_HOST_ABORT,
        y_bytes=y_bytes,
        y_stride=y_stride,
        total_bytes=HX_PAGE + nsrv * nslots * y_stride,
    )


# ---------------------------------------------------------------------------
# Step header (64 B at offset 0 of the inbox control page)
# ---------------------------------------------------------------------------
# magic, epoch_abs, T, bucket, opcode, version, mode, plan_hash, hub_index,
# rsv32, rsv64, hdr_seq (little endian, no padding)
_HDR = struct.Struct("<QQIIHHIQIIQQ")
assert _HDR.size == HX_HDR_BYTES


@dataclass(frozen=True)
class Hdr:
    """The fields of a step header (the reserved words are not carried)."""

    magic: int
    epoch: int
    T: int
    bucket: int
    opcode: int
    version: int
    mode: int
    plan_hash: int
    hub_index: int
    hdr_seq: int


def pack_hdr(
    epoch: int,
    T: int,
    bucket: int,
    plan: int,
    hub_index: int = 0,
    *,
    magic: int = HX_MAGIC,
    opcode: int = HX_OPCODE_MOE,
    version: int = HX_VERSION,
    mode: int | None = None,
    hdr_seq: int | None = None,
) -> bytes:
    """The 64 header bytes. ``mode`` defaults to 1 (eager) when ``bucket`` is 0 and 0
    (decode graph) otherwise; ``hdr_seq`` defaults to ``epoch`` (the value the hub
    stores last, with release semantics, to publish the header)."""
    epoch = _int("epoch", epoch, 0, U64_MAX)
    if mode is None:
        mode = 1 if bucket == 0 else 0
    return _HDR.pack(
        _int("magic", magic, 0, U64_MAX),
        epoch,
        _int("T", T, 0, U32_MAX),
        _int("bucket", bucket, 0, U32_MAX),
        _int("opcode", opcode, 0, 0xFFFF),
        _int("version", version, 0, 0xFFFF),
        _int("mode", mode, 0, U32_MAX),
        _int("plan", plan, 0, U64_MAX),
        _int("hub_index", hub_index, 0, U32_MAX),
        0,
        0,
        _int("hdr_seq", epoch if hdr_seq is None else hdr_seq, 0, U64_MAX),
    )


def unpack_hdr(buf: bytes | bytearray | memoryview) -> Hdr:
    """Parse the first 64 bytes of ``buf`` (a header, or a whole control page)."""
    if len(buf) < HX_HDR_BYTES:
        raise ValueError(f"header needs {HX_HDR_BYTES} bytes, got {len(buf)}")
    f = _HDR.unpack_from(buf)
    return Hdr(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[11])


# ---------------------------------------------------------------------------
# Abort record codec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Record:
    """A decoded abort-record word."""

    valid: bool
    live: bool
    phase: int
    peer: int
    ms: int
    seq_lo: int
    peer_seen: bool = False

    @property
    def saturated(self) -> bool:
        return self.ms == 0xFFFF

    @property
    def layer(self) -> int | None:
        """Layer of the waited sequence number (None for the begin phases)."""
        if self.phase in _RAW_EPOCH_PHASES:
            return None
        return (self.seq_lo - 1) & (HX_MAX_LAYERS - 1)

    @property
    def epoch(self) -> int:
        """Epoch of the wait: modulo 2**26 for layer sequences (only 26 bits of the
        epoch survive in 32 sequence bits), modulo 2**32 for the begin phases."""
        if self.phase in _RAW_EPOCH_PHASES:
            return self.seq_lo
        return ((self.seq_lo - 1) & U32_MAX) >> 6


def encode_record(phase: int, peer: int, seq: int, ms: int, bits: int = 0) -> int:
    """The u64 record word. ``seq`` is truncated to its low 32 bits, ``ms``
    saturates at 65535, ``bits`` may add ``HX_REC_LIVE`` / ``HX_REC_PEER_SEEN``
    (``HX_REC_VALID`` is always set)."""
    if phase not in HX_PHASE_NAMES:
        raise ValueError(f"phase={phase!r} is not one of {sorted(HX_PHASE_NAMES)}")
    peer = _int("peer", peer, 0, 0xFF)
    seq = _int("seq", seq, 0, U64_MAX)
    ms = min(_int("ms", ms, 0, U64_MAX), 0xFFFF)
    bits = _int("bits", bits, 0, U32_MAX)
    if bits & ~(HX_REC_LIVE | HX_REC_PEER_SEEN):
        raise ValueError(f"bits={bits:#x} may only hold the live and peer-seen flags")
    return (
        HX_REC_VALID | bits | phase << 4 | peer << 8 | ms << 16 | (seq & U32_MAX) << 32
    )


def decode_record(code: int) -> Record:
    """Decode a record word (an int64 view of the u64, as hx_status returns, works)."""
    code = _int("code", code, -(1 << 63), U64_MAX) & U64_MAX
    return Record(
        valid=bool(code & HX_REC_VALID),
        live=bool(code & HX_REC_LIVE),
        phase=(code >> 4) & 0xF,
        peer=(code >> 8) & 0xFF,
        ms=(code >> 16) & 0xFFFF,
        seq_lo=code >> 32,
        peer_seen=bool(code & HX_REC_PEER_SEEN),
    )


def describe_record(code: int, role: str) -> str:
    """One sentence for an abort (or live) record written by ``role``: the phase,
    the peer (or the mismatching header field), the waited time, the epoch and the
    layer."""
    if role not in ("hub", "server"):
        raise ValueError(f"role must be 'hub' or 'server', got {role!r}")
    rec = decode_record(code)
    if not rec.valid:
        return f"{role}: no abort recorded"
    name = HX_PHASE_NAMES.get(rec.phase, f"unknown phase {rec.phase}")
    peer_kind = "server" if role == "hub" else "hub"
    if rec.phase == HX_PH_HDR:
        who = f"field {HX_MISMATCH_NAMES.get(rec.peer, rec.peer)}"
    elif rec.phase == HX_PH_PLAN:
        who = "no peer"
    elif rec.phase in (HX_PH_DISPATCH, HX_PH_PUSH):
        who = f"peer {rec.peer}"
    else:
        who = f"peer {peer_kind} {rec.peer}"
    waited = "65535+ ms (saturated)" if rec.saturated else f"{rec.ms} ms"
    if rec.layer is None:
        where = f"epoch {rec.epoch}"
    else:
        where = f"epoch {rec.epoch} (mod 2^26), layer {rec.layer}"
    if rec.live:
        head = f"{role} is still waiting in {name} (phase {rec.phase})"
        waited += " so far"
    else:
        head = f"{role} aborted in {name} (phase {rec.phase})"
    note = ""
    if HX_PHASE_ROLES.get(rec.phase, role) not in (None, role):
        note = f" [phase {rec.phase} is not written by a {role}]"
    return f"{head}: {who}, waited {waited}, {where}.{note}"
