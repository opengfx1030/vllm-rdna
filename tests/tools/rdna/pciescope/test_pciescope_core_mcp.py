# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MCP server: ``Service`` methods, and a stdio round trip with the SDK client."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from pciescope import mcp_server, model
from pciescope.env import Env

RDNA_DIR = Path(__file__).resolve().parents[4] / "tools" / "rdna"


@pytest.fixture()
def env(fx_root, tmp_path) -> Env:
    return Env(
        sysfs_root=str(fx_root / "sys"),
        proc_root=str(fx_root / "proc"),
        host="par1-cs25",
        store_root=str(tmp_path / "store"),
    )


@pytest.fixture()
def svc(env):
    s = mcp_server.Service(env)
    yield s
    s.shutdown()


def record(svc: mcp_server.Service, seconds: float = 0.4, **kw: Any) -> str:
    """Record a short watch; returns the session name."""
    w = svc.start_watch(interval_s=0.05, **kw)
    time.sleep(seconds)
    svc.stop_watch(w["watch_id"])
    return w["session"]


# -- topology / snapshot ------------------------------------------------------


def test_topology_summary_and_full(svc):
    out = svc.topology()
    assert "root complex c0 (switch A)" in out["tree"]
    assert len(out["summary"]["gpus"]) == 10
    assert out["summary"]["gpus"][0]["id"] == "gpu:cb"
    full = svc.topology("full")["topology"]
    assert len(full["gpus"]) == 10
    assert len(full["links"]) == 32
    with pytest.raises(ValueError, match="view must be"):
        svc.topology("nope")


def test_topology_without_gpus(tmp_path):
    s = mcp_server.Service(Env(sysfs_root=str(tmp_path)))
    with pytest.raises(ValueError, match="no AMD GPU found"):
        s.topology()


def test_snapshot_summary_and_raw(svc):
    doc = svc.snapshot()
    assert len(doc["gpus"]) == 10
    assert doc["aer"]["devices_checked"] == 66
    assert "samples" not in doc
    doc = svc.snapshot(collectors="amdgpu", metric=["gpu.temp"], entity=["gpu:ce"])
    assert doc["collectors"] == ["amdgpu"]
    assert {s["entity"] for s in doc["samples"]} == {"gpu:ce"}
    assert all(s["metric"].startswith("gpu.temp") for s in doc["samples"])
    with pytest.raises(ValueError, match="no collector available"):
        svc.snapshot(collectors="nope")


# -- watches ------------------------------------------------------------------


def test_watch_lifecycle(svc):
    w = svc.start_watch(interval_s=0.05, label="t")
    assert w["collectors"] == ["sysfs_link", "aer", "amdgpu", "kfd"]
    assert Path(w["path"]).is_dir()
    time.sleep(0.4)
    st = svc.watch_status()["watches"]
    assert [x["session"] for x in st] == [w["watch_id"]]
    assert st[0]["running"] and st[0]["ticks"] >= 3
    out = svc.stop_watch()  # the only running one
    assert out["status"] == "ok"
    assert out["ticks"] >= 3
    codes = [f["code"] for f in out["findings"] if f["code"] != "collector_cost"]
    assert codes == ["aer_nonzero_since_boot"]  # collector_cost depends on the host
    assert svc.watch_status() == {"watches": []}
    with pytest.raises(ValueError, match="0 watches running"):
        svc.stop_watch()
    with pytest.raises(ValueError, match="unknown watch_id"):
        svc.stop_watch("nope")
    rep = svc.report(w["session"], sections=["findings"])
    assert set(rep) == {"session", "findings"}


def test_watch_limits_and_validation(env):
    s = mcp_server.Service(env, max_watches=1)
    try:
        with pytest.raises(ValueError, match="interval_s"):
            s.start_watch(interval_s=0.001)
        a = s.start_watch(interval_s=0.05, label="a")
        with pytest.raises(ValueError, match="1 watches already running"):
            s.start_watch(interval_s=0.05, label="b")
        with pytest.raises(ValueError, match="no collector available"):
            mcp_server.Service(env).start_watch(collectors="nope")
        assert a["watch_id"] in s.stop_watch(a["watch_id"])["session"]
    finally:
        s.shutdown()


def test_stop_watch_needs_an_id_when_several_run(env):
    s = mcp_server.Service(env)
    try:
        a = s.start_watch(interval_s=0.05, label="a")
        b = s.start_watch(interval_s=0.05, label="b")
        with pytest.raises(ValueError, match="pass watch_id"):
            s.stop_watch()
        s.stop_watch(b["watch_id"])
        assert s.stop_watch()["session"] == a["watch_id"]
    finally:
        s.shutdown()


def test_shutdown_finalises_running_watches(env):
    s = mcp_server.Service(env)
    w = s.start_watch(interval_s=0.05)
    time.sleep(0.2)
    s.shutdown()
    meta = json.loads((Path(w["path"]) / "session.json").read_text())
    assert meta["status"] == "ok"
    assert s.watch_status() == {"watches": []}


# -- reading sessions ---------------------------------------------------------


def test_query_trace_kinds(svc):
    name = record(svc)
    series = svc.query_trace(name, kind="series", metric="link.speed")
    assert series["count"] == 22
    assert {r["metric"] for r in series["series"]} == {"link.speed_gts"}
    stats = svc.query_trace(name, kind="stats", metric="gpu.temp_edge", entity="gpu:ce")
    (row,) = stats["series"]
    assert row["entity"] == "gpu:ce"
    assert row["min"] <= row["mean"] <= row["max"]
    assert stats["span_s"] > 0
    head = svc.query_trace(name, kind="samples", metric="link.speed", limit=3)
    assert head["count"] == 22 and head["truncated"] and len(head["records"]) == 3
    tail = svc.query_trace(name, kind="samples", limit=2, tail=True)
    assert tail["records"][-1]["t_s"] >= tail["records"][0]["t_s"]
    assert len(tail["records"]) == 2
    ev = svc.query_trace(name, kind="events")
    assert [e["name"] for e in ev["records"]] == ["watch_start", "watch_stop"]
    assert svc.query_trace(name, kind="events", severity="error")["count"] == 0
    assert svc.query_trace(name, kind="measurements")["count"] == 0
    late = svc.query_trace(name, kind="samples", since_s=3600)
    assert late["count"] == 0 and late["records"] == []
    with pytest.raises(ValueError, match="kind must be"):
        svc.query_trace(name, kind="nope")
    with pytest.raises(FileNotFoundError):
        svc.query_trace("no-such-session")


def test_report_text_sections_and_sessions(svc):
    name = record(svc)
    assert svc.report("latest", format="text")["text"].startswith("pciescope report:")
    full = svc.report(name)
    assert len(full["links"]) == 22
    with pytest.raises(ValueError, match="unknown sections"):
        svc.report(name, sections=["nope"])
    sess = svc.sessions()
    assert [s["name"] for s in sess["sessions"]] == [name]
    out = svc.export_trace(name)
    assert Path(out["path"]).stat().st_size == out["bytes"] > 0


def test_metrics_lists_the_model(svc):
    names = {m["name"] for m in svc.metrics()["metrics"]}
    assert names == set(model.METRICS)


# -- measure ------------------------------------------------------------------


def test_measure_is_a_dry_run_by_default(svc, fake_probe):
    out = svc.measure(gpus="ce, cb")
    args, session = fake_probe.calls[0]
    assert args.gpus == ["ce", "cb"] and args.dry_run and args.quiet and args.json
    assert session is None  # a dry run records nothing
    assert out["exit_code"] == 0 and out["exit_meaning"] == "ok or dry-run"
    assert "session" not in out
    svc.measure(list_suites=True)
    assert fake_probe.calls[1][0].list_suites and fake_probe.calls[1][1] is None


def test_measure_real_run_records_into_a_session(svc, fake_probe):
    out = svc.measure(
        "standard",
        ["hip:6"],
        owner="me",
        dry_run=False,
        repeats=2,
        max_seconds=90,
        wait_quiet_s=5,
        force_noisy=True,
    )
    args, session = fake_probe.calls[0]
    assert (args.suite, args.gpus, args.owner) == ("standard", ["hip:6"], "me")
    assert (args.repeats, args.max_seconds, args.wait_quiet) == (2, 90.0, 5.0)
    assert args.force_noisy and not args.dry_run
    assert out["session"] == session.name
    meta = json.loads((Path(out["path"]) / "session.json").read_text())
    assert meta["status"] == "ok"
    refused = svc.measure(gpus="ce", owner="refuse", dry_run=False)
    assert refused["exit_code"] == 2 and "refused" in refused["exit_meaning"]
    meta = json.loads((Path(refused["path"]) / "session.json").read_text())
    assert meta["status"] == "exit 2"


def test_measure_in_situ_records(svc, fake_probe):
    out = svc.measure(in_situ_s=3, interval_s=0.5)
    args, session = fake_probe.calls[0]
    assert (args.in_situ, args.interval) == (3.0, 0.5)
    assert session is not None and out["session"] == session.name


def test_measure_without_the_probe_module(svc, no_probe):
    with pytest.raises(ValueError, match="probe module is not available"):
        svc.measure()


# -- the real stdio transport -------------------------------------------------


def _payload(res: Any) -> Any:
    data = getattr(res, "structured_content", None) or getattr(
        res, "structuredContent", None
    )
    return data if data is not None else json.loads(res.content[0].text)


def _is_error(res: Any) -> bool:
    return bool(getattr(res, "is_error", getattr(res, "isError", False)))


def test_stdio_round_trip(fx_root, tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "pciescope",
            "mcp",
            "--sysfs-root",
            str(fx_root / "sys"),
            "--proc-root",
            str(fx_root / "proc"),
            "--host",
            "par1-cs25",
            "--root",
            str(tmp_path / "store"),
        ],
        env={**os.environ, "PYTHONPATH": str(RDNA_DIR)},
    )

    async def session_flow() -> dict[str, Any]:
        seen: dict[str, Any] = {}
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as client,
        ):
            await client.initialize()
            seen["tools"] = sorted(t.name for t in (await client.list_tools()).tools)
            topo = _payload(await client.call_tool("topology", {}))
            seen["gpus"] = len(topo["summary"]["gpus"])
            snap = _payload(await client.call_tool("snapshot", {}))
            seen["snap_gpus"] = len(snap["gpus"])
            w = _payload(await client.call_tool("start_watch", {"interval_s": 0.1}))
            await asyncio.sleep(0.6)
            stopped = _payload(
                await client.call_tool("stop_watch", {"watch_id": w["watch_id"]})
            )
            seen["ticks"] = stopped["ticks"]
            rep = _payload(
                await client.call_tool(
                    "report", {"session": w["session"], "sections": ["findings"]}
                )
            )
            seen["findings"] = [
                f["code"] for f in rep["findings"] if f["code"] != "collector_cost"
            ]
            q = _payload(
                await client.call_tool(
                    "query_trace",
                    {
                        "session": w["session"],
                        "kind": "stats",
                        "metric": "gpu.temp_edge",
                        "entity": "gpu:ce",
                    },
                )
            )
            seen["stats"] = q["series"][0]["entity"]
            bad = await client.call_tool("query_trace", {"session": "nope"})
            seen["error"] = (_is_error(bad), bad.content[0].text)
        return seen

    seen = asyncio.run(asyncio.wait_for(session_flow(), timeout=90))
    assert seen["tools"] == [
        "export_trace",
        "measure",
        "metrics",
        "query_trace",
        "report",
        "sessions",
        "snapshot",
        "start_watch",
        "stop_watch",
        "topology",
        "watch_status",
    ]
    assert seen["gpus"] == seen["snap_gpus"] == 10
    assert seen["ticks"] >= 3
    assert seen["findings"] == ["aer_nonzero_since_boot"]
    assert seen["stats"] == "gpu:ce"
    is_err, text = seen["error"]
    assert is_err and "session not found: nope" in text
