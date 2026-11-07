# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the host logic of the hx expert server (``server.py``): the
constructor checks, env defaults, the opcode registry, ``serve`` driven by a fake
ring and the fake ops, ``_wait_desc``, the abort paths and ``SyntheticModel``.
``prepare`` (CUDA) is not exercised; ``_start_signal_thread`` runs in a subprocess."""

import json
import multiprocessing
import os
import signal
import sys
import threading
import time
import types

import pytest

pytest.importorskip("torch")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from hx_fake_ops import (  # noqa: E402
    FakeOps,
    FakeRing,
    inject_prepared,
    tiny_plan,
    tiny_plan_doc,
)

from vllm.distributed.hetero_moe import ctx as C  # noqa: E402
from vllm.distributed.hetero_moe import hx  # noqa: E402
from vllm.distributed.hetero_moe import hx_proto as P  # noqa: E402
from vllm.distributed.hetero_moe import server as S  # noqa: E402
from vllm.distributed.hetero_moe.plan import PlanError, load_plan  # noqa: E402
from vllm.distributed.hetero_moe.ring import RingOverrun, StepDesc  # noqa: E402


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


def new_server(plan=None, device_id="exp0", **kw) -> S.ExpertServer:
    kw.setdefault("buckets", [3])
    kw.setdefault("max_rows", 8)
    kw.setdefault("synthetic", True)
    return S.ExpertServer(plan or tiny_plan(2), device_id, None, **kw)


@pytest.fixture
def server(fake, tmp_path):
    s = new_server()
    inject_prepared(s, dir=tmp_path)
    yield s
    s.close()
    fake.assert_clean()


# ---- constructor -------------------------------------------------------------------


def test_constructor_derives_the_server_identity_from_the_plan():
    plan = tiny_plan(3)
    s = new_server(plan, "exp2", buckets=[6, 3, 6, 3], max_rows=48, nslots=3)
    assert s.buckets == [3, 6]  # sorted, deduplicated
    assert (s.dev, s.island, s.server_index) == (3, 0, 2)
    assert s.hubs == ["hub0"] and s.hub_id == "hub0"
    assert s.layers == [0, 1, 2, 3] and s.nlayers == 4
    assert s.geometry == C.Geometry(
        rows_cap=48,
        topk=plan.top_k,
        hidden=P.HX_DEFAULT_HIDDEN,
        nslots=3,
        num_experts=8,
        nlayers=4,
    )
    assert s.ring is None and s.ctx is None and s._inbox is None
    assert not s._stop and not s._hub_dead and s._abort_exit is None
    assert s._next_no == 1 and s._opened == []
    assert new_server(plan, "exp0").server_index == 0


def test_constructor_keeps_the_caller_options():
    s = new_server(
        wait_ms=123,
        begin_wait_ms=456,
        alloc="coarse",
        flush=1,
        selftest=False,
        graphs=False,
        hub_timeout_s=9.5,
        ready_file="/x/ready.json",
        expect_bus="ce",
    )
    assert (s.wait_ms, s.begin_wait_ms, s.alloc, s.flush) == (123, 456, "coarse", 1)
    assert not s.selftest and not s.use_graphs and s.synthetic
    assert (s.hub_timeout_s, s.ready_file, s.expect_bus) == (9.5, "/x/ready.json", "ce")


def test_an_island_with_two_hubs_is_rejected():
    plan = tiny_plan(2, hubs=2)
    assert len(plan.hubs(0)) == 2
    with pytest.raises(S.ServerError, match="island 0 has 2 hubs"):
        new_server(plan, "exp0")


def test_a_device_that_is_not_an_expert_is_rejected():
    with pytest.raises(S.ServerError, match="device hub0 .* is not an expert"):
        new_server(device_id="hub0")


def test_an_unknown_device_is_a_plan_error():
    with pytest.raises(PlanError, match="'exp9' is not in plan"):
        new_server(device_id="exp9")


@pytest.mark.parametrize("layers", [[0, 1, 64], [64], [0, 100], []])
def test_plan_layers_outside_the_protocol_are_rejected(layers):
    plan = tiny_plan(2)
    plan.target_layers = layers
    with pytest.raises(S.ServerError, match="outside the protocol"):
        new_server(plan)


def test_the_highest_legal_layer_is_accepted():
    plan = tiny_plan(2)
    plan.target_layers = [0, P.HX_MAX_LAYERS - 1]
    assert new_server(plan).nlayers == P.HX_MAX_LAYERS


@pytest.mark.parametrize("buckets", [[3, 16], [9], [100, 3]])
def test_buckets_above_max_rows_are_rejected(buckets):
    with pytest.raises(S.ServerError, match=r"must be within 1\.\.8"):
        new_server(buckets=buckets, max_rows=8)


def test_empty_buckets_are_rejected():
    with pytest.raises(S.ServerError, match="must be within"):
        new_server(buckets=[])


def test_the_largest_bucket_may_equal_max_rows():
    assert new_server(buckets=[8], max_rows=8).buckets == [8]


@pytest.mark.parametrize("buckets", [[0], [0, 3], [-1, 3]])
def test_buckets_below_one_are_rejected(buckets):
    """The message says ``1..max_rows`` but only the largest bucket is checked."""
    with pytest.raises(S.ServerError, match="must be within"):
        new_server(buckets=buckets, max_rows=8)


# ---- env defaults ------------------------------------------------------------------


def test_apply_env_defaults_fills_an_empty_environment():
    env: dict[str, str] = {}
    S.apply_env_defaults(env)
    assert env == S.ENV_DEFAULTS
    assert all(isinstance(v, str) for v in env.values())


def test_apply_env_defaults_never_overrides_a_caller_variable():
    env = {key: f"mine-{i}" for i, key in enumerate(S.ENV_DEFAULTS)}
    before = dict(env)
    S.apply_env_defaults(env)
    assert env == before
    partial = {"HSA_ENABLE_SDMA": "1", "UNRELATED": "x"}
    S.apply_env_defaults(partial)
    assert partial["HSA_ENABLE_SDMA"] == "1" and partial["UNRELATED"] == "x"
    assert partial["GPU_MAX_HW_QUEUES"] == S.ENV_DEFAULTS["GPU_MAX_HW_QUEUES"]
    assert set(partial) == set(S.ENV_DEFAULTS) | {"UNRELATED"}


def test_apply_env_defaults_without_an_argument_uses_os_environ(monkeypatch):
    env = {"PATH": os.environ.get("PATH", ""), "VLLM_RDNA2_W4A8_BLOCK_M": "16"}
    monkeypatch.setattr(os, "environ", env)  # restored; the real env stays clean
    S.apply_env_defaults()
    assert env["VLLM_RDNA2_W4A8_BLOCK_M"] == "16"
    assert env["VLLM_RDNA2_W4A8_SDOT4"] == "1"
    assert set(S.ENV_DEFAULTS) <= set(env)


# ---- opcode registry ---------------------------------------------------------------


@pytest.fixture
def handlers(monkeypatch):
    table = dict(S._OP_HANDLERS)
    monkeypatch.setattr(S, "_OP_HANDLERS", table)
    return table


def test_moe_opcode_is_registered_and_a_duplicate_is_rejected(handlers):
    assert handlers[P.HX_OPCODE_MOE] is S._moe_op
    with pytest.raises(ValueError, match="opcode 1 already registered"):
        S.register_op(P.HX_OPCODE_MOE, lambda server, desc: None)
    assert handlers[P.HX_OPCODE_MOE] is S._moe_op  # the original stays


def test_a_new_opcode_registers_once(handlers):
    def handler(server, desc):
        return None

    S.register_op(2, handler)
    assert handlers[2] is handler
    with pytest.raises(ValueError, match="opcode 2 already registered"):
        S.register_op(2, handler)


# ---- serve -------------------------------------------------------------------------


class FakeEvent:
    made: list["FakeEvent"] = []

    def __init__(self):
        self.recorded = 0
        self.synchronized = 0
        FakeEvent.made.append(self)

    def record(self):
        self.recorded += 1

    def synchronize(self):
        self.synchronized += 1


@pytest.fixture
def events(monkeypatch):
    FakeEvent.made = []
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    return FakeEvent.made


def desc(n: int, T: int = 3, bucket: int = 3, opcode: int = P.HX_OPCODE_MOE):
    return StepDesc(n, 100 + n, T, bucket, opcode)


def test_serve_with_an_unknown_opcode_aborts_and_pokes(server, fake, handlers):
    server.ring = FakeRing([desc(1, opcode=9)])
    assert server.serve() == S.EXIT_ABORT
    assert fake.aborted(server.ctx)  # the device waits are released
    assert server.ring.gets == [1] and server._next_no == 1


def test_serve_dispatches_each_step_to_its_handler_in_order(
    server, fake, handlers, events
):
    seen = []
    S.register_op(2, lambda srv, d: seen.append(("two", d.step_no, srv is server)))
    handlers[P.HX_OPCODE_MOE] = lambda srv, d: seen.append(("moe", d.step_no, True))
    posted = [desc(1), desc(2, opcode=2), desc(3)]
    server.ring = FakeRing(posted, closed=True)
    assert server.serve() == S.EXIT_OK
    assert seen == [("moe", 1, True), ("two", 2, True), ("moe", 3, True)]
    assert server._next_no == 4
    assert server.ring.gets == [1, 2, 3, 4]
    assert len(events) == 3 and all(e.recorded == 1 for e in events)
    assert all(e.synchronized >= 1 for e in events)  # drained, one in flight at most
    assert not fake.aborts


def test_serve_leaves_cleanly_when_the_ring_is_closed_and_empty(server, events):
    server.ring = FakeRing([], closed=True)
    assert server.serve() == S.EXIT_OK
    assert events == []


def test_serve_stops_cleanly_after_a_stop_request_during_a_step(
    server, fake, handlers, events
):
    def handler(srv, d):
        srv.request_stop()

    handlers[P.HX_OPCODE_MOE] = handler
    server.ring = FakeRing([desc(1)])  # open ring: only the stop ends the loop
    assert server.serve() == S.EXIT_OK
    assert server._stop and fake.aborted(server.ctx)  # request_stop poked
    assert server._next_no == 2


def test_a_device_abort_after_a_step_ends_serve_with_abort(
    server, fake, handlers, events
):
    ran = []
    handlers[P.HX_OPCODE_MOE] = lambda srv, d: ran.append(d.step_no)
    rec = P.encode_record(P.HX_PH_SRV_WAIT, 0, P.seq_of(1, 2), 2000)
    fake.set_record(server.ctx, rec)
    server.ring = FakeRing([desc(1), desc(2)])
    assert server.serve() == S.EXIT_ABORT
    assert ran == [1]  # no second step after the abort showed up


def test_a_device_abort_is_ignored_while_stopping(server, fake, handlers, events):
    def handler(srv, d):
        srv._stop = True  # the signal thread got there first

    handlers[P.HX_OPCODE_MOE] = handler
    fake.set_record(server.ctx, P.encode_record(P.HX_PH_SRV_WAIT, 0, 5, 9))
    server.ring = FakeRing([desc(1)], closed=True)
    assert server.serve() == S.EXIT_OK


@pytest.mark.parametrize(
    "error",
    [RingOverrun("step 1 was overwritten"), S.ServerError("bad step")],
    ids=["overrun", "server-error"],
)
def test_ring_overrun_and_server_errors_abort_with_a_poke(server, fake, error):
    server.ring = FakeRing(raises={1: error})
    assert server.serve() == S.EXIT_ABORT
    assert fake.aborted(server.ctx)


def test_a_handler_server_error_aborts_with_a_poke(server, fake, handlers, events):
    def handler(srv, d):
        raise S.ServerError("no graph")

    handlers[P.HX_OPCODE_MOE] = handler
    server.ring = FakeRing([desc(1)])
    assert server.serve() == S.EXIT_ABORT
    assert fake.aborted(server.ctx) and server._next_no == 1
    assert events == []  # the failed step enqueued nothing to wait for


def test_serve_returns_abort_when_the_hub_died_and_ok_when_stopping(server, fake):
    server.ring = FakeRing([])
    server._hub_dead = True
    assert server.serve() == S.EXIT_ABORT
    server._stop = True
    assert server.serve() == S.EXIT_OK


# ---- the MoE step ------------------------------------------------------------------


class FakeGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


def test_eager_step_runs_the_chain_with_the_workspaces(server, fake, events):
    T, layers = 3, server.layers
    gen = torch.Generator().manual_seed(5)
    x = torch.randint(-8, 8, (T, server.geometry.hidden), generator=gen) / 4
    w = torch.randint(0, 8, (T, 2), generator=gen) / 8
    ids = torch.tensor([[1, 2], [-1, 5], [9, -1]], dtype=torch.int32)
    server.x_ws[:T] = x.to(torch.float16)
    server.w_ws[:T] = w
    server.ids32[: T * 2] = ids.reshape(-1)
    server.ring = FakeRing([StepDesc(1, 77, T, 0, P.HX_OPCODE_MOE)], closed=True)
    assert server.serve() == S.EXIT_OK

    assert fake.calls == [("hx_srv_post_desc", 77, T, 0), ("hx_srv_begin",)] + [
        c
        for layer in layers
        for c in (
            ("hx_srv_wait", layer, T),
            ("hx_srv_push", layer, T, (T, server.geometry.hidden), torch.float16),
        )
    ]
    # y = fp16(x * sum_k w * ((id & 7) + 1)) over the pairs with id >= 0
    scale = np.zeros(T)
    for r in range(T):
        for k in range(2):
            if ids[r, k] >= 0:
                scale[r] += float(w[r, k]) * ((int(ids[r, k]) & 7) + 1)
    want = (x.double().numpy() * scale[:, None]).astype(np.float16)
    assert [layer for layer, _ in fake.pushed] == layers
    for _, y in fake.pushed:
        assert np.array_equal(y.numpy(), want)
    assert len(events) == 1


def test_bucket_step_replays_the_captured_graph_and_skips_the_chain(
    server, fake, events
):
    server._graphs[3] = graph = FakeGraph()
    server.ring = FakeRing([desc(1, T=3, bucket=3), desc(2, T=3, bucket=3)], True)
    assert server.serve() == S.EXIT_OK
    assert graph.replays == 2
    assert fake.calls == [
        ("hx_srv_post_desc", 101, 3, 3),
        ("hx_srv_post_desc", 102, 3, 3),
    ]


@pytest.mark.parametrize(
    ("T", "bucket", "match", "posted"),
    [
        (0, 0, r"step T=0 outside 1\.\.8", False),
        (9, 0, r"step T=9 outside 1\.\.8", False),
        (3, 6, r"announced bucket 6 \(T=3\).*graphs for \[3\]", True),
        (4, 3, r"announced bucket 3 \(T=4\).*graphs for \[3\]", True),
    ],
    ids=["T-zero", "T-too-big", "unknown-bucket", "T-not-bucket"],
)
def test_run_moe_step_validates_the_announcement(
    server, fake, T, bucket, match, posted
):
    server._graphs[3] = FakeGraph()
    with pytest.raises(S.ServerError, match=match):
        server.run_moe_step(StepDesc(1, 1, T, bucket))
    assert server._graphs[3].replays == 0
    # the descriptor reaches the device only for a step that passed the T check
    assert bool(fake.calls) is posted


# ---- _wait_desc and the abort checks -----------------------------------------------


def test_wait_desc_returns_a_posted_step_even_when_stopping(server):
    server.ring = FakeRing([desc(1)], closed=True)
    server._stop = server._hub_dead = True
    assert server._wait_desc(1) == desc(1)


@pytest.mark.parametrize("flag", ["_stop", "_hub_dead", "closed"])
def test_wait_desc_returns_none_on_stop_hub_death_or_a_closed_ring(server, flag):
    server.ring = FakeRing([])
    if flag == "closed":
        server.ring.closed = True
    else:
        setattr(server, flag, True)
    t0 = time.monotonic()
    assert server._wait_desc(1) is None
    assert time.monotonic() - t0 < 5.0
    assert server._abort_exit is None


def test_wait_desc_waits_for_a_step_posted_later(server):
    ring = FakeRing([])
    server.ring = ring
    t = threading.Timer(0.15, ring.posted.append, args=(desc(1),))
    t.start()
    try:
        t0 = time.monotonic()
        assert server._wait_desc(1) == desc(1)
        assert time.monotonic() - t0 >= 0.14
    finally:
        t.join(30)


def test_wait_desc_checks_the_device_record_while_idle(server, fake):
    fake.set_record(server.ctx, P.encode_record(P.HX_PH_SRV_BEGIN, 0, 1, 600_000))
    server.ring = FakeRing([])  # open and empty: the server idles
    t0 = time.monotonic()
    assert server._wait_desc(1) is None
    assert 0.45 <= time.monotonic() - t0 < 20.0  # first check after 0.5 s
    assert server._abort_exit == S.EXIT_ABORT
    server._abort_exit = None
    server.ring = FakeRing([])
    assert server.serve() == S.EXIT_ABORT
    assert not fake.aborts  # idle abort: nothing to poke, the device already knows


def test_an_idle_abort_does_not_override_a_stop(server, fake):
    fake.set_record(server.ctx, P.encode_record(P.HX_PH_SRV_BEGIN, 0, 1, 600_000))
    server.ring = FakeRing([])
    server._abort_exit = S.EXIT_ABORT
    server._stop = True
    assert server.serve() == S.EXIT_OK


def test_check_abort_reads_the_record_word(server, fake):
    assert server._check_abort() is False
    fake.set_record(server.ctx, P.encode_record(P.HX_PH_HDR, P.HX_MM_EPOCH, 3, 1))
    assert server._check_abort() is True
    fake.set_record(server.ctx, 0)
    assert server._check_abort() is False


# ---- stop, hub death, close --------------------------------------------------------


def test_request_stop_sets_the_flag_and_pokes_the_device(server, fake):
    server.request_stop()
    assert server._stop and fake.aborted(server.ctx)


def test_hub_death_callback_flags_the_server_and_pokes(server, fake):
    server._on_hub_death(4242, "hub0")
    assert server._hub_dead and fake.aborted(server.ctx)


def test_a_failing_poke_is_swallowed(server, fake):
    fake.hx_ctx_unregister(server.ctx)  # hx_host_abort now raises "not registered"
    with pytest.raises(RuntimeError, match="not registered"):
        fake.hx_host_abort(server.ctx)
    server.request_stop()  # runs on a signal path: must not raise
    server._on_hub_death(1, "hub0")
    assert server._stop and server._hub_dead


def test_close_of_a_prepared_server_releases_everything(fake, tmp_path):
    s = new_server()
    inject_prepared(s, dir=tmp_path)
    s.publish()
    assert hx.srv_path(tmp_path, "exp0").exists()
    ctx = s.ctx
    s.close()
    assert s._inbox is None and not fake.is_registered(ctx)
    assert not hx.srv_path(tmp_path, "exp0").exists()
    fake.assert_clean()


def test_close_of_a_server_that_never_prepared_is_a_noop(fake):
    new_server().close()
    fake.assert_clean()


# ---- SyntheticModel ----------------------------------------------------------------


def plain_loop_moe(x, ids, w):
    rows, hidden = x.shape
    y = np.zeros((rows, hidden), np.float16)
    for r in range(rows):
        scale = 0.0
        for k in range(ids.shape[1]):
            if ids[r, k] >= 0:
                scale += float(w[r, k]) * ((int(ids[r, k]) & 7) + 1)
        for c in range(hidden):
            y[r, c] = np.float16(float(x[r, c]) * scale)
    return y


def test_synthetic_model_equals_a_plain_loop():
    gen = torch.Generator().manual_seed(11)
    T, H, K = 6, 16, 5
    # values on a coarse grid: every product and sum is exact, so the vectorised
    # torch result and the loop agree bit for bit
    x = (torch.randint(-8, 8, (T, H), generator=gen) / 4).to(torch.float16)
    ids = torch.randint(-1, 20, (T, K), generator=gen).to(torch.int32)
    ids[2] = -1  # a fully masked row
    ids[3, 0] = 7
    w = torch.randint(0, 8, (T, K), generator=gen).to(torch.float32) / 8
    model = S.SyntheticModel(4, H, K)
    y = model.moe(1, x, ids, w)
    assert y.dtype == torch.float16 and y.shape == (T, H)
    assert np.array_equal(y.numpy(), plain_loop_moe(x.numpy(), ids.numpy(), w.numpy()))
    assert not y[2].any()  # masked row: zeros
    assert model.moe(3, x, ids, w).equal(y)  # the layer does not matter


def test_synthetic_model_masks_negative_ids_and_uses_the_low_three_bits():
    model = S.SyntheticModel(1, 8, 3)
    x = torch.ones(1, 8, dtype=torch.float16)
    w = torch.ones(1, 3)
    ids = torch.tensor([[0, 7, 8]], dtype=torch.int32)  # scales 1, 8, 1
    assert model.moe(0, x, ids, w).tolist() == [[10.0] * 8]
    ids = torch.tensor([[-1, -5, 3]], dtype=torch.int32)  # only the last counts
    assert model.moe(0, x, ids, w).tolist() == [[4.0] * 8]


def test_synthetic_model_interface():
    model = S.SyntheticModel(5, 32, 10)
    assert (model.num_layers, model.hidden, model.top_k) == (5, 32, 10)
    assert model.expert_sets == [[0]] * 5
    assert model.known_answer_selftest(anything=1) == {"synthetic": True}
    assert model.memory_bytes() == 0


# ---- CLI ---------------------------------------------------------------------------


def test_parse_defaults_and_flags():
    a = S._parse(["--plan", "p.json", "--device-id", "exp0"])
    assert (a.plan, a.device_id, a.model, a.ready_file) == (
        "p.json",
        "exp0",
        None,
        None,
    )
    assert a.buckets == "3,6,12,24,48" and a.max_rows == P.HX_DEFAULT_ROWS_CAP
    assert (a.nslots, a.wait_ms) == (P.HX_DEFAULT_NSLOTS, P.HX_DEFAULT_WAIT_MS)
    assert a.begin_wait_ms == P.HX_DEFAULT_BEGIN_WAIT_MS and a.hub_timeout == 7200.0
    assert not (a.synthetic or a.no_graphs or a.no_selftest) and a.expect_bus is None
    b = S._parse(
        ["--plan", "p", "--device-id", "exp1", "--synthetic", "--no-graphs"]
        + ["--no-selftest", "--expect-bus", "d3", "--alloc", "coarse", "--flush", "1"]
        + ["--buckets", "3,6", "--hub-timeout", "5"]
    )
    assert b.synthetic and b.no_graphs and b.no_selftest
    assert (b.expect_bus, b.alloc, b.flush, b.buckets, b.hub_timeout) == (
        "d3",
        "coarse",
        1,
        "3,6",
        5.0,
    )
    with pytest.raises(SystemExit):
        S._parse(["--plan", "p", "--device-id", "exp0", "--alloc", "auto"])


@pytest.fixture
def clean_environ(monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))


def test_main_reports_a_setup_failure_for_an_unreadable_plan(clean_environ, tmp_path):
    argv = ["--plan", str(tmp_path / "missing.json"), "--device-id", "exp0"]
    assert S.main(argv) == S.EXIT_SETUP


def test_main_reports_a_setup_failure_for_a_device_that_is_not_an_expert(
    clean_environ, tmp_path
):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(tiny_plan_doc(2)))
    load_plan(path)  # the plan itself is fine: only the device is wrong
    assert S.main(["--plan", str(path), "--device-id", "hub0", "--synthetic"]) == 4
    assert S.main(["--plan", str(path), "--device-id", "nope", "--synthetic"]) == 4


def test_exit_codes_are_the_documented_ones():
    assert (S.EXIT_OK, S.EXIT_ABORT, S.EXIT_SETUP) == (0, 3, 4)
    assert issubclass(S.ServerError, RuntimeError)


# ---- signals ---------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "linux", reason="the servers run on Linux")
def test_sigterm_reaches_request_stop_despite_older_threads():
    """SIGTERM must end in ``request_stop()``; a thread that predates the handler
    must not take the default action (exit code -15: the T5 wrong-plan failure)."""
    mp = multiprocessing.get_context("fork")
    ready = mp.Event()

    def child() -> None:
        older = threading.Thread(target=time.sleep, args=(60,), daemon=True)
        older.start()  # created before the handlers, signals unblocked
        stopped = threading.Event()
        fake = types.SimpleNamespace(device_id="exp0", request_stop=stopped.set)
        S._start_signal_thread(fake)  # type: ignore[arg-type]
        ready.set()
        os._exit(0 if stopped.wait(30) else 7)

    proc = mp.Process(target=child)
    proc.start()
    try:
        assert proc.pid is not None and ready.wait(30)
        os.kill(proc.pid, signal.SIGTERM)
        proc.join(60)
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()


def test_connect_hub_returns_false_when_stopped_while_waiting(server):
    server.hub_timeout_s = 60.0
    timer = threading.Timer(0.3, server.request_stop)
    timer.start()
    t0 = time.monotonic()
    try:
        assert server.connect_hub() is False
    finally:
        timer.cancel()
    assert time.monotonic() - t0 < 20.0
    assert server.ring is None and server._opened == []
