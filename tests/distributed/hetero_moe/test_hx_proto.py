# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests of the hx wire protocol (vllm/distributed/hetero_moe/hx_proto.py).

CPU only: the module is loaded by path, so the tests run without torch (run them
with ``--noconftest`` where the repo conftests cannot be imported). They pin the
byte layouts, the sequence-number arithmetic and the abort-record codec to the numbers
of the design (6.4, 9.4), and keep the constants equal to those of ``hx_core.cuh``.
"""

import ast
import importlib
import importlib.util
import itertools
import random
import sys
import types
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG = "vllm.distributed.hetero_moe"


def _load(*names):
    """Import hetero_moe modules by file path under their dotted names. They need
    the standard library (and numpy) only, but ``import vllm`` needs torch: where
    it fails, empty stand-ins for the parent packages are registered."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import vllm  # noqa: F401
    except Exception:
        for key in [k for k in sys.modules if k == "vllm" or k.startswith("vllm.")]:
            del sys.modules[key]
        parent = None
        for dotted in ("vllm", "vllm.distributed", PKG):
            stub = types.ModuleType(dotted)
            stub.__path__ = []
            sys.modules[dotted] = stub
            if parent is not None:
                setattr(parent, dotted.rpartition(".")[2], stub)
            parent = stub
    else:
        importlib.import_module(PKG)
    loaded = []
    for name in names:
        dotted = f"{PKG}.{name}"
        path = ROOT / "vllm" / "distributed" / "hetero_moe" / f"{name}.py"
        mod = sys.modules.get(dotted)
        if mod is None or getattr(mod, "__file__", None) != str(path):
            spec = importlib.util.spec_from_file_location(dotted, path)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            sys.modules[dotted] = mod
            spec.loader.exec_module(mod)
        setattr(sys.modules[PKG], name, mod)
        loaded.append(mod)
    return loaded[0] if len(loaded) == 1 else loaded


hx = _load("hx_proto")
U64 = (1 << 64) - 1

# ---------------------------------------------------------------------------
# Constants (typed in from the design, not read back from the module)
# ---------------------------------------------------------------------------
CONSTANTS = {
    "HX_VERSION": 1,
    "HX_MAGIC": 0x48585F50524F5431,
    "HX_OPCODE_MOE": 1,
    "HX_MAX_LAYERS": 64,
    "HX_MAX_SERVERS": 8,
    "HX_MAX_COLD": 8,
    "HX_MAX_SLOTS": 4,
    "HX_MAX_EXPERTS": 512,
    "HX_MAX_TOPK": 16,
    "HX_PAGE": 4096,
    "HX_HDR_BYTES": 64,
    "HX_CTX_WORDS": 128,
    "HX_FLAG_STRIDE": 64,
    "HX_CTRL_REQ_FLAG": 0x100,
    "HX_CTRL_REPLY_FLAG": 0x0,
    "HX_CTRL_HOST_ABORT": 0xF00,
    "HX_HDR_OFF_MAGIC": 0,
    "HX_HDR_OFF_EPOCH": 8,
    "HX_HDR_OFF_T": 16,
    "HX_HDR_OFF_BUCKET": 20,
    "HX_HDR_OFF_OPCODE": 24,
    "HX_HDR_OFF_VERSION": 26,
    "HX_HDR_OFF_MODE": 28,
    "HX_HDR_OFF_PLAN": 32,
    "HX_HDR_OFF_HUB": 40,
    "HX_HDR_OFF_SEQ": 56,
    "HX_PH_DISPATCH": 1,
    "HX_PH_COMBINE": 2,
    "HX_PH_SRV_WAIT": 3,
    "HX_PH_SRV_BEGIN": 4,
    "HX_PH_HDR": 5,
    "HX_PH_PUSH": 6,
    "HX_PH_PLAN": 7,
    "HX_PH_HOST": 8,
    "HX_PH_DIGEST": 9,
    "HX_MM_MAGIC": 1,
    "HX_MM_PLAN": 2,
    "HX_MM_EPOCH": 3,
    "HX_MM_T": 4,
    "HX_MM_BUCKET": 5,
    "HX_MM_OPCODE": 6,
    "HX_MM_VERSION": 7,
    "HX_MM_SKIPPED": 8,
    "HX_REC_VALID": 1,
    "HX_REC_LIVE": 2,
    "HX_REC_PEER_SEEN": 4,
    "HX_ST_RECORD": 0,
    "HX_ST_EPOCH": 1,
    "HX_ST_LIVE": 2,
    "HX_ST_STEPS": 3,
    "HX_ST_DISPATCHES": 4,
    "HX_ST_COMBINES": 5,
    "HX_ST_PAIRS_HUB": 6,
    "HX_ST_PAIRS_REMOTE": 7,
    "HX_STATUS_WORDS": 8,
    "HX_DIGEST": 256,
    "HX_DIG_OFF": 8,
    "HX_DIG_RETRIES": 5,
    "HX_DIG_DELAY_US": 4,
    "HX_DS_OK": 0,
    "HX_DS_BAD": 1,
    "HX_DS_HEALED": 2,
    "HX_DS_UNHEALED": 3,
    "HX_DS_LATE": 4,
    "HX_DS_LAST_A": 5,
    "HX_DS_LAST_B": 6,
    "HX_DS_RESCANS": 7,
    "HX_DS_WORDS": 8,
    "HX_DEFAULT_ROWS_CAP": 2048,
    "HX_DEFAULT_TOPK": 10,
    "HX_DEFAULT_HIDDEN": 2560,
    "HX_DEFAULT_NSLOTS": 2,
    "HX_DEFAULT_WAIT_MS": 2000,
    "HX_DEFAULT_BEGIN_WAIT_MS": 600000,
    "HX_NAN_F16": 0x7E00,
}


@pytest.mark.parametrize(("name", "value"), sorted(CONSTANTS.items()))
def test_constant_value(name, value):
    assert getattr(hx, name) == value


def test_magic_spells_hx_prot1():
    assert hx.HX_MAGIC.to_bytes(8, "big") == b"HX_PROT1"


def test_name_tables_cover_every_phase_and_field():
    phases = {v for k, v in CONSTANTS.items() if k.startswith("HX_PH_")}
    fields = {v for k, v in CONSTANTS.items() if k.startswith("HX_MM_")}
    assert set(hx.HX_PHASE_NAMES) == phases == set(range(1, 10))
    assert set(hx.HX_MISMATCH_NAMES) == fields == set(range(1, 9))
    assert len(set(hx.HX_PHASE_NAMES.values())) == 9
    assert len(set(hx.HX_MISMATCH_NAMES.values())) == 8
    assert set(hx.HX_PHASE_ROLES) == phases
    assert hx.HX_PHASE_ROLES[hx.HX_PH_COMBINE] == "hub"
    assert hx.HX_PHASE_ROLES[hx.HX_PH_SRV_WAIT] == "server"
    assert hx.HX_PHASE_ROLES[hx.HX_PH_HOST] is None
    assert hx.HX_PHASE_ROLES[hx.HX_PH_DIGEST] is None


def test_status_words_are_distinct_and_dense():
    words = sorted(v for k, v in CONSTANTS.items() if k.startswith("HX_ST_"))
    assert words == list(range(hx.HX_STATUS_WORDS))


def test_mutable_name_tables_are_read_only():
    with pytest.raises(TypeError):
        hx.HX_PHASE_NAMES[9] = "x"
    with pytest.raises(TypeError):
        hx.HX_MISMATCH_NAMES[9] = "x"


# ---------------------------------------------------------------------------
# Sequence numbers
# ---------------------------------------------------------------------------
def test_seq_of_values():
    assert hx.seq_of(0, 0) == 1
    assert hx.seq_of(1, 0) == 65
    assert hx.seq_of(1, 47) == 112
    assert hx.seq_of(5, 17) == 338
    assert hx.seq_of(1, 63) == 128
    assert hx.seq_of(2**58 - 1, 62) == U64
    assert hx.seq_of(2**58 - 1, 63) == 0  # u64 wrap
    assert hx.seq_of(2**58, 0) == 1


@pytest.mark.parametrize(
    ("epoch", "layer"), [(-1, 0), (0, -1), (0, 64), (2**64, 0), (1.5, 0), ("1", 0)]
)
def test_seq_of_rejects_bad_arguments(epoch, layer):
    with pytest.raises(ValueError):
        hx.seq_of(epoch, layer)


@pytest.mark.parametrize(
    ("a", "b", "before"),
    [
        (0, 0, False),
        (0, 1, True),
        (1, 0, False),
        (U64, 0, True),  # -1 is before 0
        (0, U64, False),
        (U64 - 1, 1, True),  # across the wrap
        (1, U64 - 1, False),
        (2**63 - 1, 0, False),  # distance 2**63 - 1: a is after b
        (0, 2**63 - 1, True),
        (2**63, 0, True),  # the antipode is "before" in both directions
        (0, 2**63, True),
    ],
)
def test_seq_before(a, b, before):
    assert hx.seq_before(a, b) is before
    assert hx.seq_reached(a, b) is (not before)


def test_seq_before_is_signed_distance_modulo_2_64():
    rng = random.Random(7)
    for _ in range(2000):
        base = rng.getrandbits(64)
        delta = rng.randint(-(2**63) + 1, 2**63 - 1)
        assert hx.seq_before((base + delta) & U64, base) is (delta < 0)
        assert hx.seq_before(base, (base + delta) & U64) is (delta > 0)


def test_sequence_order_survives_the_u64_wrap():
    """The program order of (epoch, layer) is strict across epoch 2**58."""
    pairs = [
        (e, layer) for e in range(2**58 - 3, 2**58 + 3) for layer in range(0, 64, 7)
    ]
    seqs = [hx.seq_of(e, layer) for e, layer in pairs]
    assert seqs[0] > seqs[-1]  # numerically it wraps: a plain compare would break
    for prev, nxt in itertools.pairwise(seqs):
        assert hx.seq_before(prev, nxt)
        assert not hx.seq_reached(prev, nxt)
        assert hx.seq_reached(nxt, prev)


def test_fresh_flag_has_not_reached_the_first_sequence():
    assert not hx.seq_reached(0, hx.seq_of(1, 0))
    assert hx.seq_reached(hx.seq_of(1, 0), hx.seq_of(1, 0))


@pytest.mark.parametrize(
    ("layer", "nslots", "slot"),
    [(0, 1, 0), (47, 1, 0), (5, 2, 1), (6, 2, 0), (7, 4, 3)],
)
def test_slot_of(layer, nslots, slot):
    assert hx.slot_of(layer, nslots) == slot


@pytest.mark.parametrize(("layer", "nslots"), [(64, 2), (-1, 2), (0, 0), (0, 5)])
def test_slot_of_rejects_bad_arguments(layer, nslots):
    with pytest.raises(ValueError):
        hx.slot_of(layer, nslots)


def test_align_up():
    assert hx.align_up(0, 16) == 0
    assert hx.align_up(1, 16) == 16
    assert hx.align_up(16, 16) == 16
    assert hx.align_up(4097, 4096) == 8192
    assert hx.align_up(10, 3) == 12
    with pytest.raises(ValueError):
        hx.align_up(1, 0)
    with pytest.raises(ValueError):
        hx.align_up(-1, 16)


def test_plan_hash():
    assert hx.plan_hash("0000000000000000") == 0
    assert hx.plan_hash("00000000000000a1") == 0xA1
    assert hx.plan_hash("ffffffffffffffff") == U64
    assert hx.plan_hash("0123456789ABCDEF") == 0x0123456789ABCDEF
    for bad in (
        "",
        "a1",
        "0" * 15,
        "0" * 17,
        "g" * 16,
        " " + "0" * 15,
        "0" * 15 + "\n",
    ):
        with pytest.raises(ValueError):
            hx.plan_hash(bad)
    with pytest.raises(ValueError):
        hx.plan_hash(0xA1)


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------
def test_inbox_layout_default_geometry():
    lay = hx.inbox_layout(2048, 10, 2560, 2)
    assert lay == hx.inbox_layout()
    assert (lay.rows_cap, lay.topk, lay.hidden, lay.nslots) == (2048, 10, 2560, 2)
    assert lay.x_bytes == 10_485_760
    assert lay.ids_bytes == 40_960
    assert lay.w_bytes == 81_920
    assert lay.ids_off == 10_485_760
    assert lay.w_off == 10_485_760 + 40_960 == 10_526_720
    assert lay.slot_stride == 10_608_640
    assert lay.total_bytes == 21_221_376
    assert (lay.control_bytes, lay.hdr_off, lay.hdr_bytes) == (4096, 0, 64)
    assert (lay.req_flag_base, lay.flag_stride, lay.host_abort_off) == (
        0x100,
        64,
        0xF00,
    )
    assert [lay.req_flag_off(j) for j in range(2)] == [0x100, 0x140]
    assert [lay.slot_off(j) for j in range(2)] == [4096, 4096 + 10_608_640]


def test_inbox_layout_four_slots():
    lay = hx.inbox_layout(2048, 10, 2560, 4)
    assert lay.slot_stride == 10_608_640
    assert lay.total_bytes == 4096 + 4 * 10_608_640 == 42_438_656
    assert lay.req_flag_off(3) == 0x100 + 3 * 64
    assert lay.slot_off(3) == 4096 + 3 * 10_608_640


def test_inbox_layout_pads_odd_geometries():
    lay = hx.inbox_layout(3, 5, 8, 2)  # ids are 30 B: padded to 32, w follows at 16 B
    assert (lay.x_bytes, lay.ids_bytes, lay.w_bytes) == (48, 30, 60)
    assert (lay.ids_off, lay.w_off) == (48, 80)
    assert lay.slot_stride == 4096
    assert lay.total_bytes == 4096 + 2 * 4096


def test_inbox_layout_invariants_on_random_geometries():
    rng = random.Random(3)
    for _ in range(300):
        rows = rng.randint(1, 5000)
        topk = rng.randint(1, 16)
        hidden = 8 * rng.randint(1, 600)
        nslots = rng.randint(1, 4)
        lay = hx.inbox_layout(rows, topk, hidden, nslots)
        x = rows * hidden * 2
        ids = -(-rows * topk * 2 // 16) * 16
        w = rows * topk * 4
        assert lay.slot_stride == -(-(x + ids + w) // 4096) * 4096
        assert lay.total_bytes == 4096 + nslots * lay.slot_stride
        assert lay.ids_off == x and lay.w_off == x + ids
        assert lay.ids_off % 16 == lay.w_off % 16 == 0
        assert lay.ids_off + lay.ids_bytes <= lay.w_off
        assert lay.w_off + lay.w_bytes <= lay.slot_stride
        assert lay.slot_stride - (lay.w_off + lay.w_bytes) < 4096


def test_inbox_control_page_regions_do_not_overlap():
    lay = hx.inbox_layout(8, 4, 16, 4)
    flags = [(lay.req_flag_off(j), lay.req_flag_off(j) + 8) for j in range(4)]
    regions = [(0, 64), *flags, (lay.host_abort_off, lay.host_abort_off + 8)]
    regions.sort()
    for (_, end), (start, _) in itertools.pairwise(regions):
        assert end <= start
    assert regions[-1][1] <= lay.control_bytes
    assert all(off % 64 == 0 for off, _ in flags)  # one flag per cache line


def test_reply_layout_default_geometry():
    rep = hx.reply_layout(2048, 2560, 2, 4)
    assert rep.y_bytes == 10_485_760
    assert rep.y_stride == 10_485_760
    assert rep.total_bytes == 83_890_176 == 4096 + 4 * 2 * 10_485_760
    assert (rep.control_bytes, rep.flag_base, rep.flag_stride) == (4096, 0, 64)
    assert rep.host_abort_off == 0xF00
    for s in range(4):
        for j in range(2):
            assert rep.flag_off(s, j) == 64 * (s * 2 + j)
            assert rep.y_off(s, j) == 4096 + (s * 2 + j) * 10_485_760


def test_reply_layout_flags_and_slots_are_distinct():
    for nsrv, nslots in itertools.product((1, 4, 8), (1, 2, 4)):
        rep = hx.reply_layout(7, 24, nslots, nsrv)
        flags = [rep.flag_off(s, j) for s in range(nsrv) for j in range(nslots)]
        slots = [rep.y_off(s, j) for s in range(nsrv) for j in range(nslots)]
        assert len(set(flags)) == len(flags) and len(set(slots)) == len(slots)
        assert max(flags) + 8 <= rep.host_abort_off < rep.control_bytes
        assert min(slots) == rep.control_bytes
        assert rep.y_stride % 4096 == 0 and rep.y_stride >= rep.y_bytes
        assert max(slots) + rep.y_bytes <= rep.total_bytes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows_cap": 0},
        {"rows_cap": -1},
        {"rows_cap": 2**32},
        {"topk": 0},
        {"topk": 17},
        {"hidden": 0},
        {"hidden": 12},
        {"hidden": 7},
        {"nslots": 0},
        {"nslots": 5},
        {"nslots": 1.5},
    ],
)
def test_inbox_layout_rejects_bad_geometry(kwargs):
    with pytest.raises(ValueError):
        hx.inbox_layout(**kwargs)


@pytest.mark.parametrize("nsrv", [0, 9, -1])
def test_reply_layout_rejects_bad_server_count(nsrv):
    with pytest.raises(ValueError):
        hx.reply_layout(8, 16, 2, nsrv)


def test_layout_accessors_check_their_indices():
    lay = hx.inbox_layout(8, 4, 16, 2)
    rep = hx.reply_layout(8, 16, 2, 4)
    for call in (
        lambda: lay.req_flag_off(2),
        lambda: lay.slot_off(2),
        lambda: lay.slot_off(-1),
        lambda: rep.flag_off(4, 0),
        lambda: rep.flag_off(0, 2),
        lambda: rep.y_off(-1, 0),
        lambda: rep.y_off(0, 2),
    ):
        with pytest.raises(ValueError):
            call()


# ---------------------------------------------------------------------------
# Step header
# ---------------------------------------------------------------------------
def test_pack_hdr_places_every_field_at_its_offset():
    raw = hx.pack_hdr(
        0x0102030405060708,
        0x11223344,
        0x55667788,
        0x99AABBCCDDEEFF00,
        0x0A0B0C0D,
        magic=0x1112131415161718,
        opcode=0x2122,
        version=0x3132,
        mode=0x41424344,
        hdr_seq=0x5152535455565758,
    )
    assert len(raw) == hx.HX_HDR_BYTES == 64

    def field(off, size):
        return int.from_bytes(raw[off : off + size], "little")

    assert field(hx.HX_HDR_OFF_MAGIC, 8) == 0x1112131415161718
    assert field(hx.HX_HDR_OFF_EPOCH, 8) == 0x0102030405060708
    assert field(hx.HX_HDR_OFF_T, 4) == 0x11223344
    assert field(hx.HX_HDR_OFF_BUCKET, 4) == 0x55667788
    assert field(hx.HX_HDR_OFF_OPCODE, 2) == 0x2122
    assert field(hx.HX_HDR_OFF_VERSION, 2) == 0x3132
    assert field(hx.HX_HDR_OFF_MODE, 4) == 0x41424344
    assert field(hx.HX_HDR_OFF_PLAN, 8) == 0x99AABBCCDDEEFF00
    assert field(hx.HX_HDR_OFF_HUB, 4) == 0x0A0B0C0D
    assert raw[44:56] == bytes(12)  # reserved words
    assert field(hx.HX_HDR_OFF_SEQ, 8) == 0x5152535455565758  # published last


def test_pack_hdr_defaults_and_mode():
    plan = hx.plan_hash("00000000000000a1")
    raw = hx.pack_hdr(5, 3, 3, plan)
    hdr = hx.unpack_hdr(raw)
    assert hdr == hx.Hdr(hx.HX_MAGIC, 5, 3, 3, 1, 1, 0, plan, 0, 5)
    assert hx.unpack_hdr(hx.pack_hdr(5, 3, 0, plan)).mode == 1  # eager: bucket 0
    assert hx.unpack_hdr(hx.pack_hdr(5, 3, 0, plan, mode=7)).mode == 7
    torn = hx.unpack_hdr(hx.pack_hdr(5, 3, 3, plan, hdr_seq=4))
    assert (torn.epoch, torn.hdr_seq) == (5, 4)


def test_unpack_hdr_reads_a_whole_control_page():
    raw = hx.pack_hdr(9, 2, 2, 77)
    page = bytearray(raw) + bytearray(4096 - 64)
    assert hx.unpack_hdr(page) == hx.unpack_hdr(raw)
    assert hx.unpack_hdr(memoryview(page)).epoch == 9
    with pytest.raises(ValueError):
        hx.unpack_hdr(raw[:63])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epoch": -1},
        {"epoch": 2**64},
        {"T": -1},
        {"T": 2**32},
        {"bucket": 2**32},
        {"plan": 2**64},
        {"plan": -1},
        {"hub_index": 2**32},
        {"magic": -1},
        {"opcode": 2**16},
        {"version": 2**16},
        {"mode": 2**32},
        {"hdr_seq": 2**64},
    ],
)
def test_pack_hdr_rejects_out_of_range_fields(kwargs):
    base = {"epoch": 1, "T": 1, "bucket": 1, "plan": 1}
    base.update(kwargs)
    with pytest.raises(ValueError):
        hx.pack_hdr(**base)


# ---------------------------------------------------------------------------
# Abort record codec
# ---------------------------------------------------------------------------
def test_record_bits_by_hand():
    seq = hx.seq_of(5, 17)
    assert seq == 338
    code = hx.encode_record(hx.HX_PH_COMBINE, 3, seq, 2000)
    assert code == 1 | (2 << 4) | (3 << 8) | (2000 << 16) | (338 << 32)
    live = hx.encode_record(hx.HX_PH_COMBINE, 3, seq, 2000, hx.HX_REC_LIVE)
    assert live == code | 2
    seen = hx.encode_record(hx.HX_PH_COMBINE, 3, seq, 2000, hx.HX_REC_PEER_SEEN)
    assert seen == code | 4
    both = hx.HX_REC_LIVE | hx.HX_REC_PEER_SEEN
    assert hx.encode_record(hx.HX_PH_COMBINE, 3, seq, 2000, both) == code | 6


def test_record_decode_by_hand():
    rec = hx.decode_record(1 | (3 << 4) | (0 << 8) | (2000 << 16) | (77 << 32))
    assert rec == hx.Record(True, False, 3, 0, 2000, 77)
    assert not rec.saturated and not rec.peer_seen
    assert (rec.layer, rec.epoch) == (12, 1)  # seq 77 = 1 * 64 + 12 + 1


def test_record_ms_saturates_at_65535():
    seq = hx.seq_of(1, 1)
    for ms in (65535, 65536, 70000, 600_000, 2**40):
        rec = hx.decode_record(hx.encode_record(hx.HX_PH_SRV_BEGIN, 0, seq, ms))
        assert rec.ms == 65535 and rec.saturated
    for ms in (0, 1, 2000, 65534):
        rec = hx.decode_record(hx.encode_record(hx.HX_PH_SRV_BEGIN, 0, seq, ms))
        assert rec.ms == ms and not rec.saturated
    # saturation does not spill into the sequence bits
    code = hx.encode_record(hx.HX_PH_SRV_BEGIN, 0, 0, 2**40)
    assert hx.decode_record(code).seq_lo == 0


def test_record_keeps_only_the_low_32_bits_of_the_sequence():
    for seq, low in (
        (0, 0),
        (2**32 - 1, 2**32 - 1),
        (2**32, 0),
        (2**40 + 7, 7),
        (U64, 2**32 - 1),
    ):
        rec = hx.decode_record(hx.encode_record(hx.HX_PH_COMBINE, 1, seq, 5))
        assert rec.seq_lo == low


def test_record_layer_and_epoch_follow_the_sequence():
    for epoch, layer in itertools.product((0, 1, 5, 2**26 - 1, 2**26 + 5), (0, 1, 63)):
        rec = hx.decode_record(
            hx.encode_record(hx.HX_PH_COMBINE, 0, hx.seq_of(epoch, layer), 0)
        )
        assert rec.layer == layer
        assert rec.epoch == epoch % 2**26
    # the sequence of (2**58 - 1, 63) wraps to 0 and still decodes to its epoch
    rec = hx.decode_record(
        hx.encode_record(hx.HX_PH_COMBINE, 0, hx.seq_of(2**58 - 1, 63), 0)
    )
    assert (rec.layer, rec.epoch) == (63, 2**26 - 1)


def test_begin_phases_carry_the_raw_epoch():
    for phase in (hx.HX_PH_SRV_BEGIN, hx.HX_PH_HDR):
        rec = hx.decode_record(hx.encode_record(phase, 0, 123456, 10))
        assert rec.layer is None and rec.epoch == 123456


def test_record_round_trip_over_the_field_ranges():
    for phase, peer, ms, bits in itertools.product(
        range(1, 10), (0, 1, 7, 255), (0, 1, 2000, 65535), (0, 2, 4, 6)
    ):
        seq = (phase * 0x01010101 + peer) & 0xFFFFFFFF
        rec = hx.decode_record(hx.encode_record(phase, peer, seq, ms, bits))
        assert (
            rec.valid and rec.live == bool(bits & 2) and rec.peer_seen == bool(bits & 4)
        )
        assert (rec.phase, rec.peer, rec.ms, rec.seq_lo) == (phase, peer, ms, seq)


def test_record_round_trip_of_random_words():
    rng = random.Random(11)
    for _ in range(3000):
        code = rng.getrandbits(64)
        code = (code & ~0xFF) | 1 | rng.choice((0, 2, 4, 6)) | rng.randint(1, 9) << 4
        rec = hx.decode_record(code)
        bits = (2 if rec.live else 0) | (4 if rec.peer_seen else 0)
        assert hx.encode_record(rec.phase, rec.peer, rec.seq_lo, rec.ms, bits) == code
        signed = code - (1 << 64) if code >> 63 else code  # the int64 view
        assert hx.decode_record(signed) == rec


def test_empty_record_is_not_valid():
    rec = hx.decode_record(0)
    assert not rec.valid and not rec.live
    assert hx.describe_record(0, "hub") == "hub: no abort recorded"
    assert hx.describe_record(0, "server") == "server: no abort recorded"


@pytest.mark.parametrize(
    ("phase", "peer", "seq", "ms", "bits"),
    [
        (0, 0, 0, 0, 0),
        (10, 0, 0, 0, 0),
        (16, 0, 0, 0, 0),
        (-1, 0, 0, 0, 0),
        (1, 256, 0, 0, 0),
        (1, -1, 0, 0, 0),
        (1, 0, -1, 0, 0),
        (1, 0, 2**64, 0, 0),
        (1, 0, 0, -1, 0),
        (1, 0, 0, 0, 1),  # the valid bit is implied
        (1, 0, 0, 0, 8),
        (1, 0, 0, 0, 0x10),
        (1.5, 0, 0, 0, 0),
    ],
)
def test_encode_record_rejects_bad_fields(phase, peer, seq, ms, bits):
    with pytest.raises(ValueError):
        hx.encode_record(phase, peer, seq, ms, bits)


@pytest.mark.parametrize("code", [2**64, -(2**63) - 1, 1.5, "1"])
def test_decode_record_rejects_bad_words(code):
    with pytest.raises(ValueError):
        hx.decode_record(code)


def test_describe_hub_combine_timeout():
    code = hx.encode_record(hx.HX_PH_COMBINE, 3, hx.seq_of(5, 17), 2000)
    assert hx.describe_record(code, "hub") == (
        "hub aborted in combine wait for server (phase 2): peer server 3, "
        "waited 2000 ms, epoch 5 (mod 2^26), layer 17."
    )


def test_describe_server_request_wait():
    code = hx.encode_record(hx.HX_PH_SRV_WAIT, 0, hx.seq_of(2**26 + 9, 63), 2000)
    assert hx.describe_record(code, "server") == (
        "server aborted in server request wait (phase 3): peer hub 0, "
        "waited 2000 ms, epoch 9 (mod 2^26), layer 63."
    )


def test_describe_begin_wait_with_saturated_time():
    code = hx.encode_record(hx.HX_PH_SRV_BEGIN, 0, 12, 600_000)
    assert hx.describe_record(code, "server") == (
        "server aborted in server begin header wait (phase 4): peer hub 0, "
        "waited 65535+ ms (saturated), epoch 12."
    )


@pytest.mark.parametrize("field", sorted(hx.HX_MISMATCH_NAMES))
def test_describe_header_mismatch_names_the_field(field):
    text = hx.describe_record(hx.encode_record(hx.HX_PH_HDR, field, 7, 0), "server")
    assert f"field {hx.HX_MISMATCH_NAMES[field]}," in text
    assert "(phase 5)" in text and "epoch 7." in text and "waited 0 ms" in text


def test_describe_unknown_header_field_and_phase():
    text = hx.describe_record(hx.encode_record(hx.HX_PH_HDR, 42, 7, 0), "server")
    assert "field 42," in text
    word = 1 | (10 << 4) | (1 << 8) | (5 << 16) | (65 << 32)  # phase 10 is not defined
    assert "unknown phase 10" in hx.describe_record(word, "hub")


def test_describe_plan_error_and_host_poke():
    seq = hx.seq_of(1, 4)
    plan = hx.describe_record(hx.encode_record(hx.HX_PH_PLAN, 0, seq, 0), "hub")
    assert plan == (
        "hub aborted in plan error (phase 7): no peer, waited 0 ms, "
        "epoch 1 (mod 2^26), layer 4."
    )
    poke = hx.encode_record(hx.HX_PH_HOST, 2, hx.seq_of(3, 0), 500)
    assert "peer server 2" in hx.describe_record(poke, "hub")
    assert "peer hub 2" in hx.describe_record(poke, "server")
    assert "[phase" not in hx.describe_record(poke, "hub")
    assert "[phase" not in hx.describe_record(poke, "server")


def test_describe_live_record_says_still_waiting():
    live = hx.encode_record(hx.HX_PH_COMBINE, 1, hx.seq_of(1, 2), 1000, hx.HX_REC_LIVE)
    assert hx.describe_record(live, "hub") == (
        "hub is still waiting in combine wait for server (phase 2): peer server 1, "
        "waited 1000 ms so far, epoch 1 (mod 2^26), layer 2."
    )


def test_describe_flags_a_record_of_the_other_role():
    code = hx.encode_record(hx.HX_PH_COMBINE, 1, hx.seq_of(1, 2), 5)
    assert hx.describe_record(code, "server").endswith(
        "[phase 2 is not written by a server]"
    )
    assert "[phase" not in hx.describe_record(code, "hub")


def test_describe_is_one_sentence_for_every_phase():
    for phase, role in itertools.product(range(1, 10), ("hub", "server")):
        text = hx.describe_record(hx.encode_record(phase, 1, 100, 10), role)
        assert "\n" not in text and role in text and f"(phase {phase})" in text
        assert text.rstrip("]").rstrip().endswith(".") or "[phase" in text


def test_describe_digest_records():
    seq = hx.seq_of(7, 12)
    mismatch = hx.encode_record(hx.HX_PH_DIGEST, 2, seq, 3)
    text = hx.describe_record(mismatch, "hub")
    assert "payload of server 2 did not match its digest" in text
    assert "3 rescans" in text and "epoch 7" in text and "layer 12" in text
    missing = hx.encode_record(hx.HX_PH_DIGEST, 0x80 | 1, seq, 2000)
    text = hx.describe_record(missing, "server")
    assert "digest word of hub 1 never arrived" in text and "2000 ms" in text
    assert "[phase" not in text  # either role writes the phase


def test_describe_dig_stats():
    zero = [0] * hx.HX_DS_WORDS
    assert hx.describe_dig_stats(zero, "hub") == (
        "hub digest: 0 ok, 0 bad at the first read, 0 healed, 0 unhealed, "
        "0 late digest words, 0 rescans"
    )
    words = list(zero)
    words[hx.HX_DS_OK] = 1000
    words[hx.HX_DS_BAD] = 2
    words[hx.HX_DS_HEALED] = 2
    words[hx.HX_DS_RESCANS] = 3
    words[hx.HX_DS_LAST_A] = 77 << 32 | 41 << 16 | 3 << 8 | 1
    words[hx.HX_DS_LAST_B] = 0xDEADBEEF << 32 | 0x0BADF00D
    text = hx.describe_dig_stats(words, "server")
    assert text.startswith("server digest: 1000 ok, 2 bad at the first read")
    assert "last mismatch: epoch 77, layer 41, peer 3, 1 rescans" in text
    assert "expected 0xdeadbeef, read 0x0badf00d" in text
    with pytest.raises(ValueError):
        hx.describe_dig_stats(words[:-1], "hub")
    with pytest.raises(ValueError):
        hx.describe_dig_stats(words, "client")


def test_describe_rejects_an_unknown_role():
    with pytest.raises(ValueError):
        hx.describe_record(1, "client")


# ---------------------------------------------------------------------------
# hx_core.cuh <-> hx_proto
# ---------------------------------------------------------------------------
CORE_NAMES = (
    "HX_VERSION",
    "HX_MAGIC",
    "HX_MAX_LAYERS",
    "HX_MAX_SERVERS",
    "HX_MAX_COLD",
    "HX_MAX_SLOTS",
    "HX_MAX_EXPERTS",
    "HX_MAX_TOPK",
    "HX_PAGE",
    "HX_OPCODE_MOE",
)
HEADER = ROOT / "csrc" / "rocm" / "hetero_moe" / "hx_core.cuh"
_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.LShift: lambda a, b: a << b,
    ast.RShift: lambda a, b: a >> b,
    ast.BitOr: lambda a, b: a | b,
    ast.BitAnd: lambda a, b: a & b,
    ast.BitXor: lambda a, b: a ^ b,
}


def _eval_int(node):
    if isinstance(node, ast.Expression):
        return _eval_int(node.body)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_int(node.left), _eval_int(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.Invert)):
        v = _eval_int(node.operand)
        return -v if isinstance(node.op, ast.USub) else ~v
    raise ValueError("not an integer constant expression")


def _c_int(body, known):
    """Value of the body of an object-like C macro made of integer literals (with
    u/l suffixes and ' separators), earlier macros, parentheses and + - * / << >> | &
    ^ ~; ValueError for anything else."""
    out = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch.isdigit() or ch == "_" or ch.isalpha():
            j = i
            while j < len(body) and (body[j].isalnum() or body[j] in "_'"):
                j += 1
            tok = body[i:j].replace("'", "")
            if tok[0].isdigit():
                tok = tok.rstrip("uUlL")
                if not tok.isalnum():
                    raise ValueError(f"bad literal {tok!r}")
            elif tok in known:
                tok = str(known[tok])
            else:
                raise ValueError(f"unknown name {tok!r}")
            out.append(tok)
            i = j
        else:
            out.append("//" if ch == "/" else ch)
            i += 1
    return _eval_int(ast.parse("".join(out), mode="eval"))


def _strip_comments(text):
    out = []
    i = 0
    while i < len(text):
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end < 0 else end + 2
            out.append(" ")
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def parse_defines(text):
    """``{name: value}`` of the object-like ``#define HX_<NAME> <integer>`` lines."""
    defs: dict[str, int] = {}
    for line in _strip_comments(text.replace("\\\n", " ")).splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        parts = line[1:].strip().split(None, 1)
        if len(parts) != 2 or parts[0] != "define":
            continue
        name, _, body = parts[1].partition(" ")
        body = body.strip()
        if not name.startswith("HX_") or "(" in name or not body:
            continue  # not an HX_ macro, function-like, or no value
        try:
            defs[name] = _c_int(body, defs)
        except (ValueError, SyntaxError):
            continue
    return defs


def mismatches(text):
    """Problems between the ``#define``s of a header text and ``hx_proto``."""
    defs = parse_defines(text)
    out = [f"{n}: not defined in the header" for n in CORE_NAMES if n not in defs]
    for name, value in defs.items():
        mine = getattr(hx, name, None)
        if type(mine) is int and mine != value:
            out.append(f"{name}: header {value:#x}, hx_proto {mine:#x}")
    return out


SAMPLE_HEADER = """
#ifndef HX_CORE_CUH_
#define HX_CORE_CUH_  // include guard: no value
#define HX_VERSION 1
#define HX_MAGIC 0x48585F50524F5431ull  /* "HX_PROT1" */
#define HX_PAGE 4096u
#define HX_CTX_WORDS (HX_PAGE / 32)
#define HX_FLAG(i) ((i) * 64)
#define HX_SHIFTED (1ULL << 40) | 0x10
#define NOT_HX 5
#define HX_TEXT "x"
#define HX_CONT 1 + \\
    2
#define HX_SEP 1'000'000
#define HX_NEG (-3)
#endif
"""


def test_define_parser():
    assert parse_defines(SAMPLE_HEADER) == {
        "HX_VERSION": 1,
        "HX_MAGIC": 0x48585F50524F5431,
        "HX_PAGE": 4096,
        "HX_CTX_WORDS": 128,
        "HX_SHIFTED": (1 << 40) | 0x10,
        "HX_CONT": 3,
        "HX_SEP": 1_000_000,
        "HX_NEG": -3,
    }


def test_sync_check_reports_a_wrong_value_and_a_missing_name():
    good = "\n".join(f"#define {name} {getattr(hx, name)}ull" for name in CORE_NAMES)
    assert mismatches(good) == []
    wrong = good.replace(
        f"#define HX_VERSION {hx.HX_VERSION}ull", "#define HX_VERSION 2"
    )
    assert mismatches(wrong) == ["HX_VERSION: header 0x2, hx_proto 0x1"]
    missing = good.replace(f"#define HX_PAGE {hx.HX_PAGE}ull", "")
    assert mismatches(missing) == ["HX_PAGE: not defined in the header"]
    # a name only the header has is not an error, a name of another type is ignored
    extra = good + "\n#define HX_ONLY_HERE 5\n#define HX_PHASE_NAMES 3\n"
    assert mismatches(extra) == []


def test_header_constants_equal_hx_proto():
    if not HEADER.is_file():
        pytest.skip(f"{HEADER.relative_to(ROOT)} is not in this tree")
    text = HEADER.read_text()
    assert mismatches(text) == []
    shared = [n for n in parse_defines(text) if type(getattr(hx, n, None)) is int]
    assert set(CORE_NAMES) <= set(shared)
