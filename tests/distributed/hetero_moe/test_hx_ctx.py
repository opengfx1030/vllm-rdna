# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the host-side ctx construction (``ctx.py``) against the fake hx
ops: Geometry, table upload, hub/server ctx images, allocation, IPC, clock."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from hx_fake_ops import (  # noqa: E402
    CTX_FIELDS,
    FAKE_TICKS_PER_MS,
    FakeOps,
    tiny_plan,
    tiny_tables,
)

from vllm.distributed.hetero_moe import ctx as C  # noqa: E402
from vllm.distributed.hetero_moe import hx_proto as P  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def few_cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(min(before, 4))
    yield
    torch.set_num_threads(before)


@pytest.fixture
def fake(monkeypatch):
    ops = FakeOps()
    monkeypatch.setattr(C, "_OPS", ops)
    return ops


def get(fake: FakeOps, image: torch.Tensor, name: str) -> int:
    """Field of an image as the unsigned value (the op returns the int64 bits)."""
    return fake.hx_ctx_get(image, name) & ((1 << (8 * CTX_FIELDS[name][1])) - 1)


SMALL = C.Geometry(rows_cap=8, topk=2, hidden=64, nslots=2)
PLAN_ID = "fedcba9876543210"  # plan hash above 2**63: exercises the signed path


# ---- _i64 --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (1, 1),
        ((1 << 63) - 1, (1 << 63) - 1),
        (1 << 63, -(1 << 63)),
        ((1 << 64) - 1, -1),
        (0xDEAD_BEEF_0000_0001, 0xDEAD_BEEF_0000_0001 - (1 << 64)),
        (-1, -1),
        (1 << 64, 0),
    ],
)
def test_i64_keeps_the_u64_bits(value, expected):
    got = C._i64(value)
    assert got == expected
    assert -(1 << 63) <= got < (1 << 63)
    assert got & P.U64_MAX == value & P.U64_MAX


# ---- Geometry ----------------------------------------------------------------------


def test_default_geometry_matches_the_protocol_layouts():
    g = C.Geometry()
    assert (g.rows_cap, g.topk, g.hidden, g.nslots) == (2048, 10, 2560, 2)
    inbox = g.inbox()
    assert inbox.slot_stride == 10_608_640
    assert inbox.total_bytes == 21_221_376
    reply = g.reply(4)
    assert reply.y_stride == 10_485_760
    assert reply.total_bytes == 83_890_176


@pytest.mark.parametrize(
    "geom",
    [
        C.Geometry(),
        C.Geometry(rows_cap=8, topk=2, hidden=64, nslots=3, num_experts=8, nlayers=4),
        C.Geometry(rows_cap=48, topk=10, hidden=2560, nslots=4),
    ],
)
@pytest.mark.parametrize("nsrv", [1, 4, 8])
def test_geometry_sizes_equal_the_hx_proto_layouts(geom, nsrv):
    assert geom.inbox() == P.inbox_layout(
        geom.rows_cap, geom.topk, geom.hidden, geom.nslots
    )
    assert geom.reply(nsrv) == P.reply_layout(
        geom.rows_cap, geom.hidden, geom.nslots, nsrv
    )


def test_geometry_dict_roundtrip_and_strictness():
    g = C.Geometry(rows_cap=8, topk=2, hidden=64, nslots=3, num_experts=8, nlayers=4)
    d = g.as_dict()
    assert d == {
        "rows_cap": 8,
        "topk": 2,
        "hidden": 64,
        "nslots": 3,
        "num_experts": 8,
        "nlayers": 4,
    }
    assert C.Geometry.from_dict(d) == g
    assert C.Geometry.from_dict({**d, "extra": 1}) == g  # unknown keys are ignored
    assert C.Geometry.from_dict({k: str(v) for k, v in d.items()}) == g  # json ints
    d.pop("nlayers")
    with pytest.raises(KeyError):
        C.Geometry.from_dict(d)
    with pytest.raises(AttributeError):
        g.rows_cap = 9  # frozen


# ---- upload_tables -----------------------------------------------------------------


def stacked_stub(layers=4, experts=8, copies=8, cold_max=1, slot_max=3, k=lambda t: 2):
    st = SimpleNamespace(
        slot_map=np.full((layers, experts), slot_max, np.int32),
        cold_srv=np.full((layers, experts, copies), cold_max, np.int32),
        n_cold=np.ones((layers, experts), np.int32),
        owned=np.ones((layers, experts), np.uint8),
    )
    return SimpleNamespace(stacked=lambda: st, k_for=k)


def test_upload_tables_dtypes_shapes_and_values_follow_the_island_tables():
    plan = tiny_plan(nservers=2)
    tables = tiny_tables(plan, spill_k={3: 2, 6: 3})
    rows_cap = 8
    up = C.upload_tables(tables, rows_cap, "cpu")
    st = tables.stacked()
    L, E = st.slot_map.shape
    assert (L, E) == (4, 8) and (up.nlayers, up.num_experts) == (L, E)

    assert up.slot_map.dtype == torch.int16 and up.slot_map.shape == (L, E)
    assert up.cold_srv.dtype == torch.int8
    assert up.cold_srv.shape == (L, E, P.HX_MAX_COLD)
    assert up.n_cold.dtype == torch.int8 and up.n_cold.shape == (L, E)
    assert up.owned.dtype == torch.uint8 and up.owned.shape == (L, E)
    assert up.ktab.dtype == torch.int16 and up.ktab.shape == (rows_cap + 1,)
    for t in (up.slot_map, up.cold_srv, up.n_cold, up.owned, up.ktab):
        assert t.device.type == "cpu" and t.is_contiguous()

    assert np.array_equal(up.slot_map.numpy(), st.slot_map)
    assert np.array_equal(up.cold_srv.numpy(), st.cold_srv)
    assert np.array_equal(up.n_cold.numpy(), st.n_cold)
    assert np.array_equal(up.owned.numpy(), st.owned)
    # the plan has hub residents, replicas, and padding to be checked
    assert (st.slot_map >= 0).any() and (st.slot_map < 0).any()
    assert (st.n_cold > 1).any() and (st.cold_srv < 0).any()

    ktab = up.ktab.tolist()
    assert ktab == [tables.k_for(max(1, t)) for t in range(rows_cap + 1)]
    assert ktab[0] == ktab[1] == 2  # T = 0 is unused: it takes the T = 1 value
    assert ktab[3] == 2 and ktab[4] == ktab[6] == 3
    assert ktab[7] == ktab[8] == tables.prefill_k  # above the widest bucket
    assert tables.prefill_k == 4


def test_upload_tables_accepts_a_torch_device_object():
    tables = tiny_tables(tiny_plan(nservers=3))
    up = C.upload_tables(tables, 4, torch.device("cpu"))
    assert up.slot_map.device.type == "cpu" and up.ktab.shape == (5,)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(layers=65), "tables have 65 layers x 8 experts"),
        (dict(experts=513), "x 513 experts"),
        (dict(copies=4), "cold_srv has 4 copies, not 8"),
        (dict(copies=9), "cold_srv has 9 copies, not 8"),
        (dict(cold_max=P.HX_MAX_SERVERS), "server index >= HX_MAX_SERVERS"),
        (dict(slot_max=P.HX_MAX_EXPERTS), "slot >= HX_MAX_EXPERTS"),
        (dict(k=lambda t: P.HX_MAX_EXPERTS + 1), "spill K outside"),
        (dict(k=lambda t: -1), "spill K outside"),
    ],
    ids=[
        "layers",
        "experts",
        "few-copies",
        "many-copies",
        "server-index",
        "slot",
        "k-high",
        "k-negative",
    ],
)
def test_upload_tables_rejects_what_the_ctx_cannot_hold(kwargs, match):
    with pytest.raises(ValueError, match=match):
        C.upload_tables(stacked_stub(**kwargs), 8, "cpu")


def test_upload_tables_accepts_the_limits():
    tables = stacked_stub(
        layers=P.HX_MAX_LAYERS,
        experts=P.HX_MAX_EXPERTS,
        cold_max=P.HX_MAX_SERVERS - 1,
        slot_max=P.HX_MAX_EXPERTS - 1,
        k=lambda t: P.HX_MAX_EXPERTS,
    )
    up = C.upload_tables(tables, 4, "cpu")
    assert up.nlayers == 64 and up.num_experts == 512
    assert up.ktab.tolist() == [P.HX_MAX_EXPERTS] * 5


# ---- hub_image ---------------------------------------------------------------------


@pytest.fixture
def hub_parts(fake):
    plan = tiny_plan(nservers=2)
    tables = C.upload_tables(tiny_tables(plan), SMALL.rows_cap, "cpu")
    reply = C.alloc_exported(SMALL.reply(2).total_bytes)
    return SimpleNamespace(plan=plan, tables=tables, reply=reply)


def make_hub_image(parts, **over):
    kw = dict(
        plan_id=PLAN_ID,
        hub_index=0,
        nsrv=2,
        reply=parts.reply,
        inbox_ptrs=[0x1000_0000, 0x2000_0000],
        tables=parts.tables,
        ticks_per_ms=25_000,
        wait_ms=500,
    )
    kw.update(over)
    return C.hub_image(SMALL, **kw)


def test_hub_image_fields(fake, hub_parts):
    p = hub_parts
    image = make_hub_image(p, hub_index=1, flush_mode=1)
    inbox, reply = SMALL.inbox(), SMALL.reply(2)
    expected = {
        "role": C.ROLE_HUB,
        "version": P.HX_VERSION,
        "plan_hash": int(PLAN_ID, 16),
        "index": 1,
        "nsrv": 2,
        "rows_cap": 8,
        "topk": 2,
        "hidden": 64,
        "nslots": 2,
        # the hub takes E and L from its tables, not from the (default) geometry
        "num_experts": p.tables.num_experts,
        "nlayers": p.tables.nlayers,
        "flush_mode": 1,
        "wait_ticks": 500 * 25_000,
        "begin_wait_ticks": 500 * 25_000,
        "ticks_per_ms": 25_000,
        "slot_stride": inbox.slot_stride,
        "off_ids": inbox.ids_off,
        "off_w": inbox.w_off,
        "y_stride": reply.y_stride,
        "last_epoch": 0,
        "own_base": p.reply.ptr,
        "hub_slot": p.tables.slot_map.data_ptr(),
        "cold_srv": p.tables.cold_srv.data_ptr(),
        "n_cold": p.tables.n_cold.data_ptr(),
        "owned": p.tables.owned.data_ptr(),
        "ktab": p.tables.ktab.data_ptr(),
        "peer_base0": 0x1000_0000,
        "peer_base1": 0x2000_0000,
        "peer_base2": 0,  # only nsrv peers are filled
        "ids32": 0,
        "x_ws": 0,  # server-only fields stay empty on a hub
        "w_ws": 0,
        "layer_stats": 0,  # counters off
    }
    assert (p.tables.num_experts, p.tables.nlayers) == (8, 4)
    for name, value in expected.items():
        assert get(fake, image, name) == value, name
    assert get(fake, image, "plan_hash") >= 1 << 63
    assert fake.hx_ctx_get(image, "plan_hash") < 0  # the op returns the int64 bits
    assert image.dtype == torch.int64 and image.shape == (P.HX_CTX_WORDS,)


def test_hub_image_layer_stats_pointer(fake, hub_parts):
    stats = torch.zeros(P.HX_MAX_LAYERS, P.HX_LSTAT_COLS, dtype=torch.int64)
    image = make_hub_image(hub_parts, layer_stats=stats)
    assert get(fake, image, "layer_stats") == stats.data_ptr() != 0
    assert get(fake, make_hub_image(hub_parts, layer_stats=None), "layer_stats") == 0
    C.register_ctx(image, "cpu")  # the counters do not disturb registration


def test_hx_lstat_columns():
    assert P.HX_LSTAT_COLS == 4
    cols = (P.HX_LSTAT_HUB, P.HX_LSTAT_COLD, P.HX_LSTAT_SPILL, P.HX_LSTAT_LOST)
    assert cols == (0, 1, 2, 3)


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(P.HX_MAX_LAYERS, P.HX_LSTAT_COLS, dtype=torch.int32),
        torch.zeros(P.HX_MAX_LAYERS, P.HX_LSTAT_COLS, dtype=torch.float64),
        torch.zeros(P.HX_MAX_LAYERS, P.HX_LSTAT_COLS - 1, dtype=torch.int64),
        torch.zeros(P.HX_MAX_LAYERS - 1, P.HX_LSTAT_COLS, dtype=torch.int64),
        torch.zeros(P.HX_MAX_LAYERS * P.HX_LSTAT_COLS, dtype=torch.int64),
        torch.zeros(P.HX_LSTAT_COLS, P.HX_MAX_LAYERS, dtype=torch.int64).t(),
    ],
    ids=["int32", "float64", "cols", "layers", "flat", "non-contiguous"],
)
def test_hub_image_rejects_a_wrong_layer_stats_tensor(fake, hub_parts, bad):
    with pytest.raises(ValueError, match="layer_stats must be a contiguous int64"):
        make_hub_image(hub_parts, layer_stats=bad)


def test_hub_image_default_flush_mode_is_zero_and_registers(fake, hub_parts):
    image = make_hub_image(hub_parts)
    assert get(fake, image, "flush_mode") == 0
    ctx = C.register_ctx(image, "cpu")  # the fake runs the C++ op's validation
    assert torch.equal(ctx, image)


def test_hub_image_wait_ticks_scale_with_ms_and_clock(fake, hub_parts):
    a = make_hub_image(hub_parts, ticks_per_ms=100, wait_ms=7)
    b = make_hub_image(hub_parts, ticks_per_ms=3, wait_ms=2000)
    assert get(fake, a, "wait_ticks") == 700 and get(fake, b, "wait_ticks") == 6000
    assert get(fake, b, "begin_wait_ticks") == 6000


def test_hub_image_checks_the_pointer_count_and_the_reply_size(fake, hub_parts):
    with pytest.raises(ValueError, match="need 2 server inbox pointers, got 1"):
        make_hub_image(hub_parts, inbox_ptrs=[0x1000])
    with pytest.raises(ValueError, match="need 2 server inbox pointers, got 3"):
        make_hub_image(hub_parts, inbox_ptrs=[1, 2, 3])
    small = C.alloc_exported(SMALL.reply(2).total_bytes - P.HX_PAGE)
    with pytest.raises(ValueError, match="reply region .* B < layout"):
        make_hub_image(hub_parts, reply=small)
    # exactly the layout size is enough, a bigger allocation too
    exact = C.alloc_exported(SMALL.reply(2).total_bytes)
    assert get(fake, make_hub_image(hub_parts, reply=exact), "own_base") == exact.ptr
    with pytest.raises(ValueError):  # zero servers: no reply layout
        make_hub_image(hub_parts, nsrv=0, inbox_ptrs=[])


# ---- server_image ------------------------------------------------------------------


def make_server_image(fake, ids32=None, inbox=None, x_ws=None, w_ws=None, **over):
    inbox = inbox or C.alloc_exported(SMALL.inbox().total_bytes)
    ids32 = ids32 if ids32 is not None else torch.zeros(16, dtype=torch.int32)
    x_ws = x_ws if x_ws is not None else torch.zeros(8, 64, dtype=torch.float16)
    w_ws = w_ws if w_ws is not None else torch.zeros(8, 2, dtype=torch.float32)
    geom = C.Geometry(rows_cap=8, topk=2, hidden=64, nslots=2, num_experts=8, nlayers=4)
    kw = dict(
        plan_id=PLAN_ID,
        server_index=1,
        nsrv=2,
        inbox=inbox,
        hub_reply_ptr=0x3000_0000,
        ids32=ids32,
        x_ws=x_ws,
        w_ws=w_ws,
        ticks_per_ms=25_000,
        wait_ms=500,
        begin_wait_ms=7_000,
        hub_nsrv=2,
    )
    kw.update(over)
    image = C.server_image(geom, **kw)
    return SimpleNamespace(image=image, inbox=inbox, ids32=ids32, x_ws=x_ws, w_ws=w_ws)


def test_server_image_fields(fake):
    r = make_server_image(fake, flush_mode=1)
    image, inbox, ids32 = r.image, r.inbox, r.ids32
    il, rl = SMALL.inbox(), SMALL.reply(2)
    expected = {
        "role": C.ROLE_SERVER,
        "version": P.HX_VERSION,
        "plan_hash": int(PLAN_ID, 16),
        "index": 1,
        "nsrv": 2,
        "rows_cap": 8,
        "topk": 2,
        "hidden": 64,
        "nslots": 2,
        "num_experts": 8,
        "nlayers": 4,
        "flush_mode": 1,
        "wait_ticks": 500 * 25_000,
        "begin_wait_ticks": 7_000 * 25_000,  # the begin wait has its own bound
        "ticks_per_ms": 25_000,
        "slot_stride": il.slot_stride,
        "off_ids": il.ids_off,
        "off_w": il.w_off,
        "y_stride": rl.y_stride,
        "own_base": inbox.ptr,
        "peer_base0": 0x3000_0000,
        "peer_base1": 0,
        "ids32": ids32.data_ptr(),
        "x_ws": r.x_ws.data_ptr(),
        "w_ws": r.w_ws.data_ptr(),
        "layer_stats": 0,
        "hub_slot": 0,
        "ktab": 0,
    }
    for name, value in expected.items():
        assert get(fake, image, name) == value, name
    assert get(fake, image, "plan_hash") >= 1 << 63


def test_server_image_registers_and_takes_a_larger_ids32(fake):
    big = torch.zeros(64, dtype=torch.int32)
    image = make_server_image(fake, ids32=big).image
    ctx = C.register_ctx(image, "cpu")
    assert get(fake, ctx, "ids32") == big.data_ptr()
    assert get(fake, ctx, "role") == C.ROLE_SERVER


def test_server_image_workspace_pointers_differ_and_register(fake):
    r = make_server_image(fake)
    ctx = C.register_ctx(r.image, "cpu")
    ptrs = {get(fake, ctx, n) for n in ("ids32", "x_ws", "w_ws")}
    assert len(ptrs) == 3 and 0 not in ptrs


def test_server_image_size_and_workspace_checks(fake):
    small = C.alloc_exported(SMALL.inbox().total_bytes - P.HX_PAGE)
    with pytest.raises(ValueError, match=r"inbox \d+ B < layout \d+ B"):
        make_server_image(fake, inbox=small)
    for bad in (
        torch.zeros(16, dtype=torch.int64),
        torch.zeros(15, dtype=torch.int32),  # one short of rows_cap * topk
        torch.zeros(16, dtype=torch.float32),
    ):
        with pytest.raises(ValueError, match="ids32 must be int32"):
            make_server_image(fake, ids32=bad)
    for bad in (
        torch.zeros(8, 64, dtype=torch.float32),
        torch.zeros(8, 64, dtype=torch.bfloat16),
        torch.zeros(8 * 64 - 1, dtype=torch.float16),  # one short of rows * hidden
    ):
        with pytest.raises(ValueError, match="x_ws must be fp16"):
            make_server_image(fake, x_ws=bad)
    for bad in (
        torch.zeros(8, 2, dtype=torch.float16),
        torch.zeros(8, 2, dtype=torch.int32),
        torch.zeros(15, dtype=torch.float32),  # one short of rows * topk
    ):
        with pytest.raises(ValueError, match="w_ws must be fp32"):
            make_server_image(fake, w_ws=bad)
    big = make_server_image(
        fake,
        x_ws=torch.zeros(9, 64, dtype=torch.float16),
        w_ws=torch.zeros(9, 2, dtype=torch.float32),
    )
    assert get(fake, big.image, "x_ws") == big.x_ws.data_ptr()
    with pytest.raises(ValueError):  # hub with no servers: no reply layout
        make_server_image(fake, hub_nsrv=0)


# ---- register_ctx / reregister_ctx -------------------------------------------------


def test_register_ctx_uploads_the_image_and_reregister_replaces_it(fake, hub_parts):
    image = make_hub_image(hub_parts)
    ctx = C.register_ctx(image, "cpu")
    assert ctx.dtype == torch.int64 and ctx.shape == (P.HX_CTX_WORDS,)
    assert ctx is not image and ctx.data_ptr() != image.data_ptr()
    assert torch.equal(ctx, image)
    assert fake.registered[ctx.data_ptr()] == 1

    changed = make_hub_image(hub_parts, inbox_ptrs=[0x5000, 0x6000], wait_ms=9)
    assert not torch.equal(changed, ctx)
    C.reregister_ctx(ctx, changed)
    assert torch.equal(ctx, changed)
    assert get(fake, ctx, "peer_base1") == 0x6000
    assert fake.registered[ctx.data_ptr()] == 2  # same ctx, registered again


def test_register_ctx_rejects_an_image_that_hx_ctx_new_did_not_make(fake):
    with pytest.raises(RuntimeError, match="not made by hx_ctx_new"):
        C.register_ctx(torch.zeros(P.HX_CTX_WORDS, dtype=torch.int64), "cpu")
    bare = fake.hx_ctx_new()  # no role, no geometry
    with pytest.raises(RuntimeError, match="bad role"):
        C.register_ctx(bare, "cpu")


def test_register_ctx_validates_what_the_hub_forgot(fake, hub_parts):
    image = make_hub_image(hub_parts)
    fake.hx_ctx_set(image, "peer_base1", 0)
    with pytest.raises(RuntimeError, match="peer_base1 not set"):
        C.register_ctx(image, "cpu")


# ---- allocation, IPC, clock --------------------------------------------------------


def test_alloc_exported_returns_the_fake_allocation(fake):
    buf = C.alloc_exported(3 * P.HX_PAGE)
    assert isinstance(buf, C.Exported)
    assert buf.nbytes == 3 * P.HX_PAGE
    assert buf.kind == 1 and buf.kind_name == "uncached"  # the default kind
    assert len(buf.handle) == 64
    assert buf.handle_hex() == buf.handle.hex() and len(buf.handle_hex()) == 128
    assert fake.allocs[buf.ptr]["handle"] == buf.handle
    assert C.alloc_exported(P.HX_PAGE, "coarse").kind_name == "coarse"
    assert C.alloc_exported(P.HX_PAGE, "finegrained").kind_name == "finegrained"
    assert C.alloc_exported(P.HX_PAGE, "uncached").kind_name == "uncached"
    for gone in ("auto", "bogus"):  # no silent fallback kind any more
        with pytest.raises(KeyError):
            C.alloc_exported(P.HX_PAGE, gone)
    with pytest.raises(RuntimeError, match="multiple of"):
        C.alloc_exported(P.HX_PAGE + 1)


def test_alloc_kinds_table_matches_the_op_contract():
    assert C.ALLOC_KINDS == {"coarse": 0, "uncached": 1, "finegrained": 2}
    assert C.ALLOC_NAMES == {0: "coarse", 1: "uncached", 2: "finegrained"}


def test_open_handle_maps_back_to_the_same_allocation(fake):
    buf = C.alloc_exported(2 * P.HX_PAGE)
    mapped = C.open_handle(buf.handle)
    assert isinstance(mapped, int) and mapped != buf.ptr  # a peer sees its own address
    assert fake.opened[mapped] == buf.ptr
    C.close_handle(mapped)
    assert mapped not in fake.opened
    with pytest.raises(RuntimeError, match="not an open mapping"):
        C.close_handle(mapped)


def test_open_handle_rejects_bad_handles(fake):
    with pytest.raises(ValueError, match="IPC handle must be 64 bytes, got 63"):
        C.open_handle(bytes(63))
    with pytest.raises(ValueError, match="got 65"):
        C.open_handle(bytes(65))
    with pytest.raises(RuntimeError, match="invalid IPC handle"):
        C.open_handle(bytes(64))
    buf = C.alloc_exported(P.HX_PAGE)
    C.free_exported(buf)
    with pytest.raises(RuntimeError, match="invalid IPC handle"):
        C.open_handle(buf.handle)  # the exporter freed it


def test_free_exported_and_leak_check(fake):
    a, b = C.alloc_exported(P.HX_PAGE), C.alloc_exported(P.HX_PAGE)
    with pytest.raises(AssertionError, match="leaked allocations"):
        fake.assert_clean()
    C.free_exported(a)
    C.free_exported(b)
    fake.assert_clean()
    with pytest.raises(RuntimeError, match="not a live allocation"):
        C.free_exported(a)


def test_calibrate_clock_rounds_the_measured_rate_and_passes_the_attribute(fake):
    assert C.calibrate_clock() == (FAKE_TICKS_PER_MS, float(FAKE_TICKS_PER_MS))
    assert ("hx_calibrate", 150) in fake.calls  # the default interval
    fake.calibration = (99_999.6, 100_250.0)
    assert C.calibrate_clock(40) == (100_000, 100_250.0)
    assert ("hx_calibrate", 40) in fake.calls
    assert isinstance(C.calibrate_clock()[0], int)


def test_ops_returns_the_installed_module(fake):
    assert C.ops() is fake
