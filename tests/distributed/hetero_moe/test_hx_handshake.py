# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the hx host handshake: ``HxHub.create`` in a thread against real
``ExpertServer`` objects (``publish``/``connect_hub``) over a rendezvous directory
in ``tmp_path``, with the fake hx ops. Process death is played by real
subprocesses and the pidfd watcher.

``ExpertServer.prepare`` needs a GPU and is not run: ``inject_prepared`` sets the
attributes it would set (allocation, ctx, workspace) from the fake ops.
"""

import contextlib
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import numpy as np  # noqa: E402
import regex as re  # noqa: E402
import torch  # noqa: E402
from hx_fake_ops import (  # noqa: E402
    CTX_FIELDS,
    FAKE_TICKS_PER_MS,
    FakeOps,
    inject_prepared,
    server_doc,
    tiny_plan,
    tiny_tables,
)

from vllm.distributed.hetero_moe import ctx as C  # noqa: E402
from vllm.distributed.hetero_moe import hx, rdv  # noqa: E402
from vllm.distributed.hetero_moe import hx_proto as P  # noqa: E402
from vllm.distributed.hetero_moe import server as S  # noqa: E402
from vllm.distributed.hetero_moe.plan import PlanError  # noqa: E402
from vllm.distributed.hetero_moe.ring import StepDesc, StepRing  # noqa: E402

HUB_KW = dict(
    max_rows=8,
    top_k=2,
    hidden=P.HX_DEFAULT_HIDDEN,
    wait_ms=500,
    connect_timeout_s=20.0,
    nslots=2,
    flush=0,
    alloc="uncached",
    device="cpu",
)
needs_pidfd = pytest.mark.skipif(
    not hasattr(os, "pidfd_open"), reason="os.pidfd_open is missing"
)


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


@pytest.fixture
def rdir(tmp_path, monkeypatch):
    """The rendezvous directory of every plan lives in ``tmp_path`` (not /dev/shm)."""
    root = tmp_path / "shm"

    def hx_dir(plan_id, root_arg=None):
        path = root / f"hx_{plan_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(rdv, "hx_dir", hx_dir)
    return hx_dir


@pytest.fixture
def procs():
    spawned: list[subprocess.Popen] = []
    yield spawned
    for p in spawned:
        if p.poll() is None:
            p.kill()
        p.wait(30)


class Job(threading.Thread):
    """Run ``fn`` on a thread; ``finish`` joins it (bounded) and returns self."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__(daemon=True)
        self._call = (fn, args, kwargs)
        self.result = None
        self.error: BaseException | None = None
        self.elapsed = 0.0

    def run(self):
        t0 = time.monotonic()
        try:
            fn, args, kwargs = self._call
            self.result = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001
            self.error = e
        finally:
            self.elapsed = time.monotonic() - t0

    def finish(self, timeout: float = 60.0) -> "Job":
        self.join(timeout)
        assert not self.is_alive(), "thread did not finish"
        return self


def wait_for(pred, timeout: float = 30.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


def spawn_sleeper(procs) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    procs.append(proc)
    return proc


def reaped_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(30)
    return proc.pid


def make_server(plan, device_id: str, **kw) -> S.ExpertServer:
    kw.setdefault("hub_timeout_s", 20.0)
    kw.setdefault("wait_ms", 500)
    server = S.ExpertServer(
        plan, device_id, None, buckets=[3], max_rows=8, synthetic=True, **kw
    )
    inject_prepared(server)
    return server


def hub_geometry(plan) -> C.Geometry:
    return C.Geometry(
        rows_cap=8,
        topk=plan.top_k,
        hidden=P.HX_DEFAULT_HIDDEN,
        nslots=2,
        num_experts=plan.num_experts,
        nlayers=48,
    )


def hub_doc(plan, reply: C.Exported, **overrides) -> dict:
    """A hub's rendezvous document as ``HxHub.create`` writes it."""
    nsrv = len(plan.experts(0))
    doc = {
        "format": hx.HUB_FORMAT,
        "plan_id": plan.plan_id,
        "device_id": "hub0",
        "hub_index": 0,
        "pid": os.getpid(),
        "geometry": hub_geometry(plan).as_dict(),
        "nsrv": nsrv,
        "servers": [plan.device_id(d) for d in plan.experts(0)],
        "ticks_per_ms": FAKE_TICKS_PER_MS,
        "reply": {
            "handle": reply.handle_hex(),
            "nbytes": reply.nbytes,
            "kind": reply.kind_name,
        },
    }
    doc.update(overrides)
    return doc


class World:
    """A connected island: one hub, ``n`` servers, all in this process."""

    def __init__(self, fake, rdir, nservers: int, **hub_kw):
        self.fake = fake
        self.hub_kw = dict(HUB_KW, **hub_kw)
        self.plan = tiny_plan(nservers)
        self.tables = tiny_tables(self.plan)
        self.d = rdir(self.plan.plan_id)
        self.servers = [make_server(self.plan, f"exp{i}") for i in range(nservers)]
        self.hub: hx.HxHub | None = None
        self._closed: set[int] = set()

    def connect(self) -> None:
        job = Job(hx.HxHub.create, self.plan, "hub0", tables=self.tables, **self.hub_kw)
        job.start()
        for s in self.servers:
            s.publish()
        for s in self.servers:
            s.connect_hub()
        job.finish()
        assert job.error is None, repr(job.error)
        self.hub = job.result

    def close_servers(self) -> None:
        for s in self.servers:
            if id(s) not in self._closed:
                self._closed.add(id(s))
                s.close()

    def close_all(self) -> None:
        if self.hub is not None:
            self.hub.close()
        self.close_servers()


@pytest.fixture
def world(fake, rdir):
    w = World(fake, rdir, nservers=2)
    w.connect()
    yield w
    w.close_all()


@pytest.fixture
def world_stats(fake, rdir):
    w = World(fake, rdir, nservers=2, layer_stats=True)
    w.connect()
    yield w
    w.close_all()


def get(fake: FakeOps, ctx, name: str) -> int:
    return fake.hx_ctx_get(ctx, name) & ((1 << (8 * CTX_FIELDS[name][1])) - 1)


# ---- the happy path ----------------------------------------------------------------


def test_path_helpers(tmp_path):
    assert hx.srv_path(tmp_path, "exp0") == tmp_path / "srv_exp0.json"
    assert hx.hub_path(tmp_path, "hub0") == tmp_path / "hub_hub0.json"
    assert hx.ack_path(tmp_path, "exp1", "hub0") == tmp_path / "ack_exp1_hub0.json"
    assert hx.ring_path(tmp_path, "hub0") == tmp_path / "ring_hub0.bin"


def test_hxhub_cannot_be_constructed_directly():
    with pytest.raises(TypeError, match="HxHub.create"):
        hx.HxHub()


def test_published_documents_have_the_documented_format(world):
    w, d = world, world.d
    for i, s in enumerate(w.servers):
        doc = rdv.read_json(hx.srv_path(d, f"exp{i}"))
        assert doc == {
            "format": "hx-srv/1",
            "plan_id": w.plan.plan_id,
            "device_id": f"exp{i}",
            "server_index": i,
            "island": 0,
            "pid": os.getpid(),
            "arch": "gfx1030",
            "bus": "ce",
            "geometry": s.geometry.as_dict(),
            "buckets": [3],
            "ticks_per_ms": FAKE_TICKS_PER_MS,
            "inbox": {
                "handle": s._inbox.handle_hex(),
                "nbytes": s._inbox.nbytes,
                "kind": "uncached",
            },
            "state": "ready",
        }
        assert s.geometry.nlayers == 4 and s.geometry.rows_cap == 8
    hub = w.hub
    doc = rdv.read_json(hx.hub_path(d, "hub0"))
    assert doc == {
        "format": "hx-hub/1",
        "plan_id": w.plan.plan_id,
        "device_id": "hub0",
        "hub_index": 0,
        "pid": os.getpid(),
        "geometry": hub.geometry.as_dict(),
        "nsrv": 2,
        "servers": ["exp0", "exp1"],
        "ticks_per_ms": FAKE_TICKS_PER_MS,
        "reply": {
            "handle": hub._reply.handle_hex(),
            "nbytes": hub.geometry.reply(2).total_bytes,
            "kind": "uncached",
        },
    }
    assert hub.geometry == hub_geometry(w.plan)
    for i in range(2):
        ack = rdv.read_json(hx.ack_path(d, f"exp{i}", "hub0"))
        assert ack == {"pid": os.getpid(), "hub": "hub0", "server": f"exp{i}"}
    assert hx.ring_path(d, "hub0").is_file()
    assert hub.plan_id == w.plan.plan_id and hub.hub_id == "hub0"
    assert (hub.hub_index, hub.nsrv, hub.server_ids) == (0, 2, ["exp0", "exp1"])


def test_hub_ctx_points_at_the_servers_inboxes_and_its_own_reply(world, fake):
    w = world
    ctx = w.hub.ctx
    assert fake.is_registered(ctx) and fake.registered[ctx.data_ptr()] == 1
    assert get(fake, ctx, "role") == C.ROLE_HUB
    assert get(fake, ctx, "nsrv") == 2 and get(fake, ctx, "index") == 0
    assert get(fake, ctx, "plan_hash") == int(w.plan.plan_id, 16)
    assert get(fake, ctx, "own_base") == w.hub._reply.ptr
    assert get(fake, ctx, "wait_ticks") == 500 * FAKE_TICKS_PER_MS
    for i, s in enumerate(w.servers):
        mapped = get(fake, ctx, f"peer_base{i}")
        assert fake.opened[mapped] == s._inbox.ptr  # opened THEIR handle
    assert get(fake, ctx, "peer_base2") == 0
    assert get(fake, ctx, "hub_slot") == w.hub._tables.slot_map.data_ptr()
    assert get(fake, ctx, "num_experts") == 8 and get(fake, ctx, "nlayers") == 4
    assert len(w.hub._opened) == 2


def test_server_ctx_is_reregistered_with_the_hubs_reply(world, fake):
    w = world
    for i, s in enumerate(w.servers):
        assert fake.registered[s.ctx.data_ptr()] == 2  # prepare + connect_hub
        mapped = get(fake, s.ctx, "peer_base0")
        assert fake.opened[mapped] == w.hub._reply.ptr
        assert s._opened == [mapped]
        assert get(fake, s.ctx, "role") == C.ROLE_SERVER
        assert get(fake, s.ctx, "index") == i
        assert get(fake, s.ctx, "own_base") == s._inbox.ptr
        assert s._hub_pid == os.getpid()
        assert not s.ring.writable and s.ring.depth == w.hub.ring.depth
        assert s.watcher.is_alive() and s.watcher.dead == []


def test_the_step_ring_carries_a_step_from_the_hub_to_every_server(world):
    w = world
    w.hub.post_step(3, 3)
    w.hub.post_step(5, 0)
    for s in w.servers:
        assert s._wait_desc(1) == StepDesc(1, 1, 3, 3, P.HX_OPCODE_MOE)
        assert s._wait_desc(2) == StepDesc(2, 2, 5, 0, P.HX_OPCODE_MOE)
    assert w.hub.check() is None  # healthy: nothing recorded, nobody dead


def test_hub_close_is_seen_by_the_servers_and_removes_its_files(world, fake):
    w = world
    w.hub.post_step(3, 3)
    ctx = w.hub.ctx
    reply_ptr = w.hub._reply.ptr
    w.hub.close()
    w.hub.close()  # twice is fine
    for s in w.servers:
        assert s.ring.closed
        assert s._wait_desc(1) is not None  # what was posted stays readable
        assert s._wait_desc(2) is None  # closed ring: the server leaves
    d = w.d
    assert not hx.hub_path(d, "hub0").exists()
    assert not hx.ring_path(d, "hub0").exists()
    for i in range(2):
        assert not hx.ack_path(d, f"exp{i}", "hub0").exists()
        assert hx.srv_path(d, f"exp{i}").exists()  # the servers' are theirs
    assert not fake.is_registered(ctx)
    assert reply_ptr not in fake.allocs and w.hub._reply is None
    assert w.hub._opened == []
    w.hub.watcher.join(30)
    assert not w.hub.watcher.is_alive()


def test_servers_close_releases_everything(world, fake):
    w = world
    w.hub.close()
    w.close_servers()
    for i, s in enumerate(w.servers):
        assert not hx.srv_path(w.d, f"exp{i}").exists()
        assert s._inbox is None and s._opened == []
        assert not fake.is_registered(s.ctx)
        s.watcher.join(30)
        assert not s.watcher.is_alive()
    fake.assert_clean()


def test_module_level_ops_forward_to_the_ctx(world, fake):
    hub = world.hub
    x = torch.zeros(3, 8, dtype=torch.float16)
    ids = torch.zeros(3, 2, dtype=torch.int32)
    w = torch.zeros(3, 2)
    out = torch.zeros(3, 8, dtype=torch.float16)
    hx.begin_step(hub, 3, 3)
    hx.dispatch(hub, x, ids, w, ids.clone(), 2)
    assert hx.combine(hub, x, out, 2) is out
    assert fake.calls[-3:] == [
        ("hx_begin_step", 3, 3),
        ("hx_dispatch", 2),
        ("hx_combine", 2),
    ]
    assert hx.status(hub).tolist() == [0] * 8
    assert hub.status_words() == [0] * 8


def test_check_raises_with_the_decoded_record(world, fake):
    hub = world.hub
    rec = P.encode_record(P.HX_PH_COMBINE, 1, P.seq_of(5, 2), 1500)
    fake.set_record(hub.ctx, rec)
    assert hub.status_words()[P.HX_ST_RECORD] == rec
    with pytest.raises(hx.HxAbort) as exc:
        hub.check()
    assert exc.value.record == rec
    assert str(exc.value) == P.describe_record(rec, "hub")
    assert "combine wait for server" in str(exc.value) and "1500 ms" in str(exc.value)
    # a record with the top bit set comes back as a negative int64: still decoded
    neg = P.encode_record(P.HX_PH_HDR, P.HX_MM_EPOCH, 7, 3) | (1 << 63)
    fake.set_record(hub.ctx, neg)
    with pytest.raises(hx.HxAbort) as exc:
        hub.check()
    assert exc.value.record == hub.status_words()[P.HX_ST_RECORD]
    fake.set_record(hub.ctx, 0)
    hub.check()  # the record is the only thing that failed it


# ---- layer statistics --------------------------------------------------------------


class FakeStream:
    created = 0

    def __init__(self, device=None):
        type(self).created += 1
        self.device = device
        self.synchronized = 0

    def synchronize(self):
        self.synchronized += 1


@pytest.fixture
def fake_streams(monkeypatch):
    """The side stream of ``layer_stats`` needs CUDA; a stand-in on the CPU."""
    FakeStream.created = 0
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    return FakeStream


def test_layer_stats_are_off_by_default(world, fake):
    hub = world.hub
    assert hub._lstats is None
    assert hub.layer_stats() is None
    hub.reset_layer_stats()  # nothing to reset
    assert hub.nlayers == 4  # from the plan tables, not the geometry's 48
    assert get(fake, hub.ctx, "layer_stats") == 0


def test_layer_stats_on_allocates_the_counters_and_points_the_ctx_at_them(
    world_stats, fake
):
    hub = world_stats.hub
    stats = hub._lstats
    assert stats.dtype == torch.int64 and stats.device.type == "cpu"
    assert tuple(stats.shape) == (P.HX_MAX_LAYERS, P.HX_LSTAT_COLS)
    assert not stats.any()
    assert get(fake, hub.ctx, "layer_stats") == stats.data_ptr()


def test_layer_stats_read_back_and_reset(world_stats, fake_streams):
    hub = world_stats.hub
    stats = hub._lstats
    for layer in range(hub.nlayers):
        stats[layer] = torch.tensor(
            [10 * layer + 1, 10 * layer + 2, 10 * layer + 3, 99]
        )
    stats[hub.nlayers + 5] = 7  # a row of a layer the plan does not have
    got = hub.layer_stats()
    assert got.dtype == np.int64 and got.shape == (hub.nlayers, 3)
    want = [[10 * layer + c for c in (1, 2, 3)] for layer in range(hub.nlayers)]
    assert got.tolist() == want  # hub, cold, spill; the lost column is not there
    got[:] = -1
    assert hub.layer_stats().tolist() == want  # a copy, not a view of the counters
    assert fake_streams.created == 1  # one side stream, made on first use
    hub.reset_layer_stats()
    assert not stats.any()
    assert hub.layer_stats().tolist() == [[0, 0, 0]] * hub.nlayers
    assert fake_streams.created == 1


def test_layer_stats_off_never_touches_a_stream(world, fake_streams):
    world.hub.layer_stats()
    world.hub.reset_layer_stats()
    assert fake_streams.created == 0


def test_hxabort_defaults_to_no_record():
    err = hx.HxAbort("boom")
    assert err.record == 0 and str(err) == "boom"
    assert isinstance(err, RuntimeError)


# ---- ready file --------------------------------------------------------------------


def test_publish_writes_the_ready_file_with_the_same_document(fake, rdir, tmp_path):
    plan = tiny_plan(2)
    ready = tmp_path / "logs" / "ready_exp0.json"
    ready.parent.mkdir()
    s = make_server(plan, "exp1", ready_file=str(ready))
    try:
        s.publish()
        doc = rdv.read_json(hx.srv_path(rdir(plan.plan_id), "exp1"))
        assert rdv.read_json(ready) == doc
        assert doc["server_index"] == 1 and doc["device_id"] == "exp1"
    finally:
        s.close()
    assert not hx.srv_path(rdir(plan.plan_id), "exp1").exists()
    assert ready.exists()  # the launcher's file is not the server's to remove
    fake.assert_clean()


# ---- hub side failures -------------------------------------------------------------


@pytest.fixture
def solo(fake, rdir):
    """One-server plan; the test plays the server by hand."""
    plan = tiny_plan(1)
    inbox = C.alloc_exported(hub_geometry(plan).inbox().total_bytes)
    return SimpleNamespace(
        plan=plan,
        tables=tiny_tables(plan),
        d=rdir(plan.plan_id),
        inbox=inbox,
        fake=fake,
    )


def start_hub(solo, **over) -> Job:
    kw = dict(HUB_KW, **over)
    job = Job(hx.HxHub.create, solo.plan, "hub0", tables=solo.tables, **kw)
    job.start()
    return job


def assert_hub_cleaned_up(solo) -> None:
    """A hub whose create() failed left nothing behind: files, mappings, VRAM."""
    assert not hx.hub_path(solo.d, "hub0").exists()
    assert not hx.ring_path(solo.d, "hub0").exists()
    assert not hx.ack_path(solo.d, "exp0", "hub0").exists()
    assert solo.fake.opened == {}
    assert set(solo.fake.allocs) == {solo.inbox.ptr}  # only the test's own inbox
    assert solo.fake.registered == {}


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"plan_id": "0" * 16}, r"hx: exp0 published format 'hx-srv/1' plan '0{16}'"),
        ({"format": "hx-srv/0"}, r"published format 'hx-srv/0'"),
        ({"format": None}, r"published format None"),
    ],
    ids=["wrong-plan", "wrong-format", "no-format"],
)
def test_hub_rejects_a_server_of_another_plan_or_format(solo, override, match):
    rdv.write_json(
        hx.srv_path(solo.d, "exp0"),
        server_doc(solo.plan, "exp0", solo.inbox, **override),
    )
    job = start_hub(solo).finish()
    assert isinstance(job.error, RuntimeError)
    assert re.search(match, str(job.error)), str(job.error)
    assert job.elapsed < 15.0  # failed at once, not after the connect timeout
    assert_hub_cleaned_up(solo)


@pytest.mark.parametrize(
    ("name", "value"),
    [("rows_cap", 16), ("topk", 4), ("hidden", 128), ("nslots", 4)],
)
def test_hub_rejects_a_server_with_another_geometry(solo, name, value):
    base = C.Geometry(
        rows_cap=8, topk=2, hidden=P.HX_DEFAULT_HIDDEN, nslots=2, num_experts=8
    ).as_dict()
    doc = server_doc(solo.plan, "exp0", solo.inbox, geometry={**base, name: value})
    rdv.write_json(hx.srv_path(solo.d, "exp0"), doc)
    job = start_hub(solo).finish()
    assert isinstance(job.error, RuntimeError)
    assert (
        f"geometry mismatch with exp0: {name} hub {base[name]} server {value}"
        in str(job.error)
    )
    assert_hub_cleaned_up(solo)


def test_hub_ignores_a_server_that_differs_only_in_experts_and_layers(solo):
    """``num_experts`` and ``nlayers`` are not part of the transport geometry."""
    base = C.Geometry(
        rows_cap=8, topk=2, hidden=P.HX_DEFAULT_HIDDEN, nslots=2, num_experts=64
    ).as_dict()
    base["nlayers"] = 20
    doc = server_doc(solo.plan, "exp0", solo.inbox, geometry=base)
    rdv.write_json(hx.srv_path(solo.d, "exp0"), doc)
    job = start_hub(solo)
    wait_for(lambda: hx.hub_path(solo.d, "hub0").exists(), what="hub document")
    rdv.write_json(hx.ack_path(solo.d, "exp0", "hub0"), {"pid": os.getpid()})
    job.finish()
    assert job.error is None, repr(job.error)
    assert job.result is not None
    job.result.close()


def test_hub_rejects_a_server_whose_pid_is_dead(solo):
    pid = reaped_pid()
    rdv.write_json(
        hx.srv_path(solo.d, "exp0"), server_doc(solo.plan, "exp0", solo.inbox, pid=pid)
    )
    job = start_hub(solo).finish()
    assert isinstance(job.error, RuntimeError)
    assert f"server exp0 pid {pid} is not alive" in str(job.error)
    assert job.elapsed < 15.0
    assert_hub_cleaned_up(solo)


def test_hub_times_out_when_the_server_never_publishes(solo):
    job = start_hub(solo, connect_timeout_s=0.4).finish()
    assert isinstance(job.error, TimeoutError)
    text = str(job.error)
    assert f"server exp0 (plan {solo.plan.plan_id})" in text
    assert "did not appear within 0.4 s" in text
    assert 0.35 <= job.elapsed < 15.0
    assert_hub_cleaned_up(solo)


def test_hub_times_out_when_the_server_never_acks(solo, procs):
    sleeper = spawn_sleeper(procs)  # alive for the whole wait
    rdv.write_json(
        hx.srv_path(solo.d, "exp0"),
        server_doc(solo.plan, "exp0", solo.inbox, pid=sleeper.pid),
    )
    job = start_hub(solo, connect_timeout_s=0.5).finish()
    assert isinstance(job.error, TimeoutError)
    assert "ack of server exp0 (waited 0.5 s)" in str(job.error)
    assert job.elapsed < 15.0
    assert_hub_cleaned_up(solo)


def test_hub_fails_fast_when_the_server_dies_before_it_acks(solo, procs):
    sleeper = spawn_sleeper(procs)
    rdv.write_json(
        hx.srv_path(solo.d, "exp0"),
        server_doc(solo.plan, "exp0", solo.inbox, pid=sleeper.pid),
    )
    job = start_hub(solo, connect_timeout_s=60.0)
    wait_for(lambda: hx.hub_path(solo.d, "hub0").exists(), what="hub document")
    sleeper.kill()
    sleeper.wait(30)
    job.finish(30)  # nowhere near the 60 s timeout
    assert isinstance(job.error, RuntimeError)
    assert f"server exp0 (pid {sleeper.pid}) died" in str(job.error)
    assert job.elapsed < 30.0
    assert_hub_cleaned_up(solo)


def test_hub_create_with_a_removed_alloc_kind_fails_before_waiting(solo):
    with pytest.raises(KeyError):  # "auto" and its silent fallback are gone
        hx.HxHub.create(
            solo.plan, "hub0", tables=solo.tables, **dict(HUB_KW, alloc="auto")
        )
    assert set(solo.fake.allocs) == {solo.inbox.ptr}
    assert solo.fake.opened == {}


def test_hub_create_with_an_unknown_or_non_hub_device_fails_cleanly(solo):
    for bad, exc in (("nope", PlanError), ("exp0", ValueError)):
        with pytest.raises(exc):
            hx.HxHub.create(solo.plan, bad, tables=solo.tables, **HUB_KW)
    assert set(solo.fake.allocs) == {solo.inbox.ptr}
    assert solo.fake.opened == {}


# ---- peer death --------------------------------------------------------------------


@pytest.fixture
def hub_with_peer(solo, procs):
    """A connected hub whose single server is a real, killable process."""
    sleeper = spawn_sleeper(procs)
    rdv.write_json(
        hx.srv_path(solo.d, "exp0"),
        server_doc(solo.plan, "exp0", solo.inbox, pid=sleeper.pid),
    )
    job = start_hub(solo)
    wait_for(lambda: hx.hub_path(solo.d, "hub0").exists(), what="hub document")
    rdv.write_json(
        hx.ack_path(solo.d, "exp0", "hub0"), {"pid": sleeper.pid, "server": "exp0"}
    )
    job.finish()
    assert job.error is None, repr(job.error)
    hub = job.result
    assert hub is not None
    yield SimpleNamespace(hub=hub, peer=sleeper, solo=solo)
    hub.close()


@needs_pidfd
def test_a_killed_server_pokes_the_abort_word_and_fails_post_step(hub_with_peer):
    hub, peer, solo = hub_with_peer.hub, hub_with_peer.peer, hub_with_peer.solo
    hub.post_step(3, 3)  # healthy so far
    hub.check()
    assert hub._dead is None and not solo.fake.aborted(hub.ctx)
    peer.kill()
    peer.wait(30)
    wait_for(lambda: hub._dead is not None, what="the watcher to see the death")
    assert hub._dead == (peer.pid, "exp0")
    wait_for(lambda: solo.fake.aborted(hub.ctx), what="the abort poke")
    assert solo.fake.aborts.count(hub.ctx.data_ptr()) == 1
    for call in (lambda: hub.post_step(3, 3), hub.check):
        with pytest.raises(hx.HxAbort) as exc:
            call()
        assert f"hx server exp0 (pid {peer.pid}) died" in str(exc.value)
        assert exc.value.record == 0
    assert hub.ring.head == 1  # the refused step was never posted
    # the device record, when there is one, is the better explanation
    rec = P.encode_record(P.HX_PH_HOST, 0, P.seq_of(1, 0), 40)
    solo.fake.set_record(hub.ctx, rec)
    with pytest.raises(hx.HxAbort) as exc:
        hub.check()
    assert exc.value.record == rec


@needs_pidfd
def test_a_server_killed_before_the_hub_connects_is_reported_at_once(solo, procs):
    """The pid is alive when the hub validates the document and dead when the
    watcher starts: ``add`` reports it synchronously."""
    sleeper = spawn_sleeper(procs)
    rdv.write_json(
        hx.srv_path(solo.d, "exp0"),
        server_doc(solo.plan, "exp0", solo.inbox, pid=sleeper.pid),
    )
    job = start_hub(solo)
    wait_for(lambda: hx.hub_path(solo.d, "hub0").exists(), what="hub document")
    sleeper.kill()
    sleeper.wait(30)
    rdv.write_json(hx.ack_path(solo.d, "exp0", "hub0"), {"pid": sleeper.pid})
    job.finish()
    # either the alive() check inside the ack wait or the watcher saw it
    if job.error is not None:
        assert "died" in str(job.error)
        assert_hub_cleaned_up(solo)
        return
    hub = job.result
    assert hub is not None
    try:
        assert hub._dead == (sleeper.pid, "exp0")
        with pytest.raises(hx.HxAbort):
            hub.post_step(3, 3)
    finally:
        hub.close()


# ---- server side failures ----------------------------------------------------------


@pytest.fixture
def lone(fake, rdir):
    """A prepared server of a 2-server plan and the means to play its hub."""
    plan = tiny_plan(2)
    d = rdir(plan.plan_id)
    server = make_server(plan, "exp1")
    reply = C.alloc_exported(hub_geometry(plan).reply(2).total_bytes)
    ring = StepRing.create(hx.ring_path(d, "hub0"), depth=8)
    ns = SimpleNamespace(plan=plan, d=d, server=server, reply=reply, ring=ring)
    ns.put = lambda **over: rdv.write_json(
        hx.hub_path(d, "hub0"), hub_doc(plan, reply, **over)
    )
    yield ns
    ring.close()
    server.close()
    C.free_exported(reply)


def test_server_times_out_when_the_hub_never_appears(fake, rdir):
    plan = tiny_plan(2)
    server = make_server(plan, "exp0", hub_timeout_s=0.4)
    try:
        t0 = time.monotonic()
        with pytest.raises(TimeoutError) as exc:
            server.connect_hub()
        assert 0.35 <= time.monotonic() - t0 < 15.0
        assert f"hub hub0 (plan {plan.plan_id})" in str(exc.value)
        assert "did not appear within 0.4 s" in str(exc.value)
        assert server.ring is None and server.watcher is None
        assert server._opened == []
    finally:
        server.close()
    fake.assert_clean()


def test_server_connects_to_a_hub_document_written_by_hand(lone, fake):
    lone.put()
    lone.server.connect_hub()
    s = lone.server
    assert fake.opened[get(fake, s.ctx, "peer_base0")] == lone.reply.ptr
    ack = rdv.read_json(hx.ack_path(lone.d, "exp1", "hub0"))
    assert ack == {"pid": os.getpid(), "hub": "hub0", "server": "exp1"}
    lone.ring.post(1, 3, 3)
    assert s._wait_desc(1) == StepDesc(1, 1, 3, 3, P.HX_OPCODE_MOE)


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"format": "hx-hub/0"}, r"hub published format 'hx-hub/0' plan"),
        ({"plan_id": "1" * 16}, r"hub published format 'hx-hub/1' plan '1{16}'"),
        ({"servers": ["exp0"]}, "hub hub0 does not list exp1"),
        (
            {"servers": ["exp1", "exp0"]},
            "hub lists exp1 at index 0, plan says 1",
        ),
    ],
    ids=["format", "plan", "unlisted", "index"],
)
def test_server_rejects_a_hub_document_that_does_not_fit(lone, over, match):
    lone.put(**over)
    with pytest.raises(S.ServerError) as exc:
        lone.server.connect_hub()
    assert re.search(match, str(exc.value)), str(exc.value)
    assert lone.server.ring is None and lone.server._opened == []  # nothing opened


@pytest.mark.parametrize(
    ("name", "value"),
    [("rows_cap", 16), ("topk", 4), ("hidden", 128), ("nslots", 4)],
)
def test_server_rejects_a_hub_with_another_geometry(lone, name, value):
    geom = {**hub_geometry(lone.plan).as_dict(), name: value}
    lone.put(geometry=geom)
    mine = getattr(lone.server.geometry, name)
    with pytest.raises(S.ServerError) as exc:
        lone.server.connect_hub()
    assert f"geometry mismatch with hub hub0: {name} hub {value} server {mine}" in str(
        exc.value
    )
    assert lone.server._opened == []


def test_server_rejects_a_hub_whose_pid_is_dead(lone):
    pid = reaped_pid()
    lone.put(pid=pid)
    with pytest.raises(S.ServerError, match=f"hub pid {pid} is not alive"):
        lone.server.connect_hub()


@needs_pidfd
def test_server_sees_the_hub_die_and_serve_leaves_with_abort(lone, fake, procs):
    hub = spawn_sleeper(procs)
    lone.put(pid=hub.pid)
    s = lone.server
    s.connect_hub()
    assert not s._hub_dead
    hub.kill()
    hub.wait(30)
    wait_for(lambda: s._hub_dead, what="the watcher to see the hub die")
    wait_for(lambda: fake.aborted(s.ctx), what="the abort poke")
    assert s.watcher.dead == [(hub.pid, "hub0")]
    assert s._wait_desc(1) is None  # nothing posted and the hub is gone
    s._next_no = 1
    assert s.serve() == S.EXIT_ABORT  # not a clean stop


@needs_pidfd
def test_a_stop_request_wins_over_a_dead_hub(lone, fake, procs):
    hub = spawn_sleeper(procs)
    lone.put(pid=hub.pid)
    s = lone.server
    s.connect_hub()
    hub.kill()
    hub.wait(30)
    wait_for(lambda: s._hub_dead, what="the watcher to see the hub die")
    s.request_stop()
    assert s.serve() == S.EXIT_OK


def test_server_close_is_complete_after_a_connect(lone, fake):
    lone.put()
    s = lone.server
    s.connect_hub()
    ring_ptr = get(fake, s.ctx, "peer_base0")
    s.close()
    assert ring_ptr not in fake.opened
    assert not fake.is_registered(s.ctx)
    assert not hx.srv_path(lone.d, "exp1").exists()
    assert not hx.ack_path(lone.d, "exp1", "hub0").exists()
    s.watcher.join(30)
    assert not s.watcher.is_alive()
    s.ring = None  # the fixture closes the ring it created, not the server's
    s._opened = []
    s.watcher = None
    s.ctx = None
