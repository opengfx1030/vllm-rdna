# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic fixtures for the pciescope TUI tests (CPU only, no sysfs).

``box_topology`` mirrors par1-cs25 (two PEX88096 switches, W7800 HIP 0/5, V620
HIP 1-4 slot A and 6-9 slot B); ``synthetic_records`` is a short recorded-session
stand-in with every kind of record the views consume.
"""

import asyncio
import gzip
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools" / "rdna"))

import json  # noqa: E402

from pciescope.model import (  # noqa: E402
    Event,
    Measurement,
    Sample,
    Topology,
    port_entity,
)

T0 = 1_762_520_000 * 1_000_000_000  # 2025-11-07 ~13:33:20 UTC
NS = 1_000_000_000
FIXTURES = Path(__file__).parent / "fixtures"


def box_topology() -> Topology:
    """The real par1-cs25 topology, as ``topology.discover()`` captured it on the
    box on 2026-11-07 (two PEX88096 switches, W7800 HIP 0/5, V620 HIP 1-4/6-9)."""
    with gzip.open(FIXTURES / "tui_topo_par1-cs25.json.gz", "rt") as f:
        d = json.load(f)
    return Topology.from_dict(d)


def port_of(topo: Topology, bus: str) -> str:
    """Switch downstream port BDF of the GPU on ``bus``."""
    return topo.gpus[f"gpu:{bus}"].port


def synthetic_records(
    seconds: int = 30,
    *,
    plx: bool = False,
    errors: bool = True,
    probes: bool = True,
    seed: int = 7,
) -> list:
    """Passive samples at 1 Hz + probe matrix + events, as a recorded session."""
    rnd = random.Random(seed)
    topo = box_topology()
    recs: list = [Event(T0, "host", "session_start", "info", "fixture")]
    aer = {g: 0 for g in topo.gpus}
    for t in range(seconds):
        ts = T0 + t * NS
        for g in topo.gpu_list():
            e = g.id
            busy = (
                0.0
                if t < 8
                else min(
                    100.0,
                    40 + 60 * abs(((t - 8) % 12) / 6 - 1) * (1.0 if g.hip % 2 else 0.5),
                )
            )
            recs += [
                Sample(ts, e, "gpu.busy_pct", round(busy), "%"),
                Sample(ts, e, "gpu.mem_busy_pct", round(busy / 3), "%"),
                Sample(ts, e, "gpu.temp_junction_c", 38 + busy * 0.4 + g.hip % 3, "C"),
                Sample(
                    ts,
                    e,
                    "gpu.vram_used_bytes",
                    (0.5 + (2 if t > 5 else 0) * (g.hip % 4)) * 2**30,
                    "B",
                ),
                Sample(ts, e, "gpu.vram_total_bytes", g.vram_total_bytes, "B"),
                Sample(ts, e, "gpu.power_w", 30 + busy, "W"),
                Sample(ts, e, "pcie.link_speed_gts", 16.0, "GT/s"),
                Sample(ts, e, "pcie.link_width", 16, "lanes"),
                Sample(ts, g.link, "link.speed_gts", 16.0, "GT/s"),
                Sample(ts, g.link, "link.width", 16, "lanes"),
                Sample(ts, g.link, "link.degraded", 0, "bool"),
            ]
            if errors and g.id == "gpu:d3" and t == 20:
                aer[g.id] += 3
            recs.append(Sample(ts, e, "aer.correctable", aer[g.id], "count"))
            recs.append(Sample(ts, e, "aer.nonfatal", 0, "count"))
            recs.append(Sample(ts, e, "aer.fatal", 0, "count"))
            if errors and g.id == "gpu:d3" and t >= 20:
                recs.append(
                    Sample(
                        ts,
                        e,
                        "aer.error",
                        aer[g.id],
                        "count",
                        {"class": "correctable", "type": "BadTLP"},
                    )
                )
            if plx and g.port:
                tx = busy / 100 * rnd.uniform(18, 26)
                rx = busy / 100 * rnd.uniform(18, 26)
                pe = port_entity(g.port)
                recs += [
                    Sample(ts, pe, "plx.perf.in_gbps", tx, "GB/s"),
                    Sample(ts, pe, "plx.perf.out_gbps", rx, "GB/s"),
                ]
        if plx:
            for sw in topo.switches.values():
                pe = port_entity(sw.upstream)
                load = 0.0 if t < 8 else 12.0 + 6 * rnd.random()
                recs += [
                    Sample(ts, pe, "plx.perf.in_gbps", load, "GB/s"),
                    Sample(ts, pe, "plx.perf.out_gbps", load / 3, "GB/s"),
                    Sample(ts, pe, "plx.perf.in_util", load / 31.5, "ratio"),
                    Sample(ts, pe, "plx.perf.out_util", load / 94.5, "ratio"),
                ]
        if errors and t == 25:
            recs.append(
                Event(
                    ts,
                    "port:" + port_of(topo, "d3"),
                    "link_change",
                    "warn",
                    "speed 16.0 -> 8.0 GT/s",
                    {"speed": 8.0},
                )
            )
    if probes:
        recs += probe_measurements(topo, T0 + (seconds - 12) * NS)
    return recs


def probe_measurements(topo: Topology, start_ns: int) -> list:
    """One lat + bw (push, pull) result per ordered V620 pair, plus a sweep."""
    out: list = []
    gpus = [g for g in topo.gpu_list() if g.slot]
    t = start_ns
    for a in gpus:
        for b in gpus:
            if a is b:
                continue
            cross = topo.path_class(a.id, b.id) == "cross-root"
            cap = 31.5
            for direction, bw in (
                ("d2d-push", 25.0 if cross else 26.2),
                ("d2d-pull", 5.7 if cross else 24.6),
            ):
                bw += (a.hip * 7 + b.hip * 3) % 10 / 20
                out.append(
                    Measurement(
                        "bw",
                        a.id,
                        b.id,
                        direction,
                        64 * 2**20,
                        bw,
                        "GB/s",
                        {
                            "p50": bw,
                            "p99": bw * 1.02,
                            "min": bw * 0.97,
                            "max": bw * 1.03,
                            "n": 20,
                        },
                        "cross-root" if cross else "intra-switch",
                        cap,
                        bw / cap,
                        t,
                        t + NS // 50,
                        [a.id, b.id],
                    )
                )
                t += NS // 50
            lat = (6.9 if cross else 6.2) + (a.hip + b.hip) % 3 / 20
            out.append(
                Measurement(
                    "lat",
                    a.id,
                    b.id,
                    "d2d-push",
                    4096,
                    lat,
                    "us",
                    {
                        "p50": lat,
                        "p99": lat * 1.3,
                        "min": lat * 0.9,
                        "max": lat * 2,
                        "n": 2000,
                    },
                    "cross-root" if cross else "intra-switch",
                    None,
                    None,
                    t,
                    t + NS // 100,
                    [a.id, b.id],
                )
            )
            t += NS // 100
    return out


GOLDEN = Path(__file__).parent / "golden_tui"


def check_golden(name: str, text: str) -> None:
    """Compare ``text`` with golden_tui/<name>.txt (UPDATE_GOLDEN=1 rewrites)."""
    import os

    path = GOLDEN / f"{name}.txt"
    text = "\n".join(line.rstrip() for line in text.rstrip("\n").split("\n")) + "\n"
    if os.environ.get("PCIESCOPE_UPDATE_GOLDEN") == "1":
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(text)
        return
    assert path.exists(), f"missing golden {path} (PCIESCOPE_UPDATE_GOLDEN=1 to create)"
    assert text == path.read_text(), f"{name} differs from {path}"


# -- helpers shared by the app / session tests ----------------------------------


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def drive(app, steps, size=(80, 24)):
    """Run ``steps(pilot)`` inside the app; returns what steps returns."""

    async def main():
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            return await steps(pilot)

    return asyncio.run(main())


async def shown(pilot, *keys):
    for k in keys:
        await pilot.press(k)
    await pilot.pause()
    return pilot.app.screen_text()


class FakeCollector:
    name = "fake"

    def __init__(self):
        self.n = 0

    def available(self):
        return True, ""

    def collect(self):
        self.n += 1
        return [
            Sample(T0 + self.n * NS, "gpu:ce", "gpu.busy_pct", 10.0 * self.n, "%"),
            Sample(T0 + self.n * NS, "gpu:ce", "gpu.temp_junction_c", 50.0, "C"),
        ]


class ListSession:
    """Stands in for ``store.Session`` (append / flush / close)."""

    path = "/runs/20261107-live"

    def __init__(self):
        self.records, self.closed = [], False

    def append(self, rec):
        self.records.append(rec)

    def append_many(self, recs):
        self.records.extend(recs)

    def flush(self):
        pass

    def close(self, status="ok"):
        self.closed = True

    def metric(self, name):
        return [r.value for r in self.records if getattr(r, "metric", "") == name]


def make_live(collectors, skipped=(), topo=None, **kw):
    """A non-threaded LiveSource over Core's Runner (deduplicating)."""
    from pciescope.runner import Runner
    from pciescope.tui.sources import LiveSource

    topo = topo or box_topology()
    runner = Runner(topo, list(collectors), interval_s=60)
    return LiveSource(topo, runner, skipped, interval=60, threaded=False, **kw)
