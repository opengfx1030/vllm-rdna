# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session store: layout, append/read, live tailing, crash tolerance."""

from __future__ import annotations

import json
import threading

import pytest
from pciescope import model, store


def _sample(i: int, value: float = 1.0, entity: str = "gpu:ce") -> model.Sample:
    return model.Sample(1_000 + i, entity, "gpu.busy_pct", value, "%")


def test_create_layout_and_meta(tmp_path, fx_topo):
    s = store.Session.create(
        tmp_path,
        "unit test",
        topo=fx_topo,
        collectors=["aer", "kfd"],
        skipped={"plx": "no root"},
        interval_s=0.5,
        cmdline="pciescope watch",
    )
    assert s.path.parent == tmp_path
    assert s.path.name.endswith("-unit-test")
    assert (s.path / store.TOPO_FILE).exists()
    meta = json.loads((s.path / store.META_FILE).read_text())
    assert meta["status"] == "running"
    assert meta["host"] == "par1-cs25"
    assert meta["collectors"] == ["aer", "kfd"]
    assert meta["skipped"] == {"plx": "no root"}
    assert meta["interval_s"] == 0.5
    s.append(_sample(0))
    s.close()
    meta = json.loads((s.path / store.META_FILE).read_text())
    assert meta["status"] == "ok"
    assert meta["counts"] == {"sample": 1, "measurement": 0, "event": 0}
    assert meta["ended_ns"] >= meta["created_ns"]
    s.close()  # idempotent
    with pytest.raises(ValueError):
        s.append(_sample(1))


def test_names_do_not_collide(tmp_path):
    a = store.Session.create(tmp_path, "x", name="fixed")
    b = store.Session.create(tmp_path, "x", name="fixed")
    assert a.path.name == "fixed"
    assert b.path.name == "fixed-2"
    a.close()
    b.close()


def test_roundtrip_all_record_types(tmp_path, fx_topo):
    m = model.Measurement(
        kind="lat",
        src="host",
        dst="gpu:ce",
        direction="h2d",
        size_bytes=4,
        value=2.1,
        unit="us",
        stats={"p50": 2.0, "p99": 3.0, "n": 100.0},
        path_class="host",
        started_ns=5,
        ended_ns=9,
    )
    ev = model.Event(7, "gpu:ce", "thermal", "warn", "hot", {"junction_c": 91.0})
    with store.Session.create(tmp_path, "rt", topo=fx_topo) as w:
        w.append_many([_sample(0), _sample(1, 2.0), m, ev])
    r = store.Session.open(w.path)
    recs = list(r.records())
    assert [type(x).__name__ for x in recs] == [
        "Sample",
        "Sample",
        "Measurement",
        "Event",
    ]
    assert recs[2] == m
    assert recs[3] == ev
    assert [x.value for x in r.samples()] == [1.0, 2.0]
    assert r.measurements() == [m]
    assert r.events() == [ev]
    assert list(r.records(["event"])) == [ev]
    assert [x.ts_ns for x in r.records(["sample"], since_ns=1_001)] == [1_001]
    assert [x.ts_ns for x in r.records(["sample"], until_ns=1_000)] == [1_000]
    topo = r.topology()
    assert topo is not None
    assert topo.to_dict() == fx_topo.to_dict()
    assert r.meta["status"] == "ok"
    assert r.meta["counts"]["sample"] == 2


def test_non_finite_values_are_dropped_not_written(tmp_path):
    with store.Session.create(tmp_path, "nan") as w:
        w.append(_sample(0, float("nan")))
        w.append(_sample(1, float("inf")))
        w.append(_sample(2, 3.0))
    assert w.dropped == 2
    assert [x.value for x in store.Session.open(w.path).samples()] == [3.0]


def test_read_from_tails_a_live_trace(tmp_path):
    w = store.Session.create(tmp_path, "live")
    r = store.Session.open(w.path)
    assert r.read_from(0) == ([], 0)
    w.append_many([_sample(0), _sample(1)])
    recs, off = r.read_from(0)
    assert len(recs) == 2
    assert r.read_from(off) == ([], off)
    w.append(_sample(2))
    recs, off2 = r.read_from(off)
    assert [x.ts_ns for x in recs] == [1_002]
    assert off2 > off
    w.close()


def test_partial_last_line_is_not_consumed(tmp_path):
    w = store.Session.create(tmp_path, "partial")
    w.append(_sample(0))
    w.flush()
    line = json.dumps(_sample(1).to_dict(), separators=(",", ":"))
    with open(w.path / store.TRACE_FILE, "a") as fh:
        fh.write(line[:20])  # another writer is mid-line
    r = store.Session.open(w.path)
    recs, off = r.read_from(0)
    assert len(recs) == 1
    with open(w.path / store.TRACE_FILE, "a") as fh:
        fh.write(line[20:] + "\n")
    recs, _ = r.read_from(off)
    assert [x.ts_ns for x in recs] == [1_001]
    w.close()


def test_garbage_lines_are_skipped(tmp_path):
    w = store.Session.create(tmp_path, "garbage")
    w.close()
    with open(w.path / store.TRACE_FILE, "w") as fh:
        fh.write("not json\n")
        fh.write('{"type":"nonsense"}\n')
        fh.write("\n")
        fh.write(json.dumps(_sample(3).to_dict()) + "\n")
        fh.write('{"type":"sample","ts_ns":1}\n')  # missing fields
    assert [x.ts_ns for x in store.Session.open(w.path).records()] == [1_003]


def test_concurrent_appends_keep_lines_intact(tmp_path):
    w = store.Session.create(tmp_path, "threads")

    def work(k: int) -> None:
        for i in range(200):
            w.append(_sample(k * 1000 + i, float(i), entity=f"gpu:{k:02x}"))

    workers = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    w.close()
    r = store.Session.open(w.path)
    assert len(list(r.samples())) == 800
    assert w.meta["counts"]["sample"] == 800


def test_resolve_latest_name_and_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        store.resolve("latest", tmp_path)
    a = store.Session.create(tmp_path, "a", name="20260101-000000-a")
    b = store.Session.create(tmp_path, "b", name="20260102-000000-b")
    a.close()
    b.close()
    assert store.resolve("latest", tmp_path) == b.path
    assert store.resolve("20260101-000000-a", tmp_path) == a.path
    assert store.resolve(str(a.path)) == a.path
    assert store.list_sessions(tmp_path) == [a.path, b.path]
    with pytest.raises(FileNotFoundError):
        store.resolve("missing", tmp_path)
    d = store.describe(a.path)
    assert d["name"] == "20260101-000000-a"
    assert d["status"] == "ok"
    assert store.Session.open("latest", tmp_path).name == "20260102-000000-b"


def test_default_root_follows_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(store.ENV_HOME, str(tmp_path / "home"))
    assert store.default_root() == tmp_path / "home"
    s = store.Session.create(label="env")
    assert s.path.parent == tmp_path / "home"
    s.close()
    monkeypatch.delenv(store.ENV_HOME)
    assert store.default_root().name == "pciescope"


def test_attach_appends_to_a_running_session_without_touching_its_meta(tmp_path):
    owner = store.Session.create(tmp_path, "run", name="20260101-000000-run")
    owner.append(_sample(0))
    guest = store.Session.attach("20260101-000000-run", tmp_path)
    assert guest.writable and not guest.owns_meta
    guest.append(_sample(1, 2.0))
    guest.update_meta(note="ignored on disk")
    guest.close("exit 2")
    on_disk = json.loads((owner.path / store.META_FILE).read_text())
    assert on_disk["status"] == "running"  # still the owner's
    assert "note" not in on_disk
    owner.append(_sample(2, 3.0))
    owner.close()
    done = store.Session.open(owner.path)
    assert done.meta["status"] == "ok"
    assert sorted(x.value for x in done.samples()) == [1.0, 2.0, 3.0]
    with pytest.raises(FileNotFoundError):
        store.Session.attach("missing", tmp_path)
