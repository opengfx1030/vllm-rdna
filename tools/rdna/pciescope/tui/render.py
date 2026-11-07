# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure render functions: (topology, ViewState, ...) -> ``rich.text.Text``.

No textual, no I/O, no clocks: everything the screen shows is a function of its
arguments, so views are snapshot-testable against recorded sessions.  Widgets in
``app`` just call these with the current terminal width.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from rich.text import Text

from pciescope.model import (
    Gpu,
    Measurement,
    Topology,
    port_entity,
    switch_entity,
)
from pciescope.tui import fmt
from pciescope.tui.state import ViewState

RECENT_ERR_NS = 60 * 1_000_000_000
VIEWS = ("topology", "live", "matrix", "errors", "measure")
VIEW_TITLES = {
    "topology": "Topology",
    "live": "Live",
    "matrix": "Matrix",
    "errors": "Errors",
    "measure": "Measure",
}
# gpu.pcie_*_bytes_per_s only exists on ASICs exposing amdgpu pcie_bw.
PCIE_BW_REASON = "amdgpu pcie_bw not exposed on RDNA2/RDNA3"


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

_KNOWN_MODELS = ("V620", "W7800", "W7900", "W6800", "MI50", "MI100", "MI210", "MI300")


def gpu_model(g: Gpu) -> str:
    """Short model name from the marketing name (``AMD RADEON PRO V620 Azure``)."""
    name = (g.name or "").upper()
    for m in _KNOWN_MODELS:
        if m in name:
            return m
    words = [
        w
        for w in name.replace("[", " ").replace("]", " ").split()
        if any(c.isdigit() for c in w)
    ]
    return (words[0] if words else (g.arch or "?"))[:8]


def gpu_label(g: Gpu) -> str:
    """``H1:ce`` (HIP index : PCI bus)."""
    return f"H{g.hip}:{g.bus}" if g.hip is not None else f"--:{g.bus}"


def entity_label(topo: Topology | None, entity: str) -> str:
    """Short label for any entity id (gpu:ce -> H1:ce, host -> host)."""
    if topo is not None and entity.startswith("gpu:"):
        g = topo.gpus.get(entity)
        if g is not None:
            return gpu_label(g)
    return entity.split(":", 1)[1] if ":" in entity else entity


def link_text(speed: float | None, width: float | None) -> str:
    if not speed or not width:
        return "down"
    return f"{speed:g}Gx{int(width)}"


# ---------------------------------------------------------------------------
# header / footer
# ---------------------------------------------------------------------------


@dataclass
class HeaderInfo:
    mode: str = "live"  # live | replay
    host: str = ""
    view: str = "topology"
    ts_ns: int | None = None
    recording: bool = False
    session: str = ""
    paused: bool = False
    speed: float = 1.0
    pos_s: float | None = None  # replay position
    dur_s: float | None = None  # replay duration
    n_samples: int = 0
    note: str = ""


def render_tabs(width: int, view: str) -> Text:
    t = Text()
    for i, v in enumerate(VIEWS, 1):
        label = f" {i}:{VIEW_TITLES[v]} "
        t.append(label, "bold reverse" if v == view else "")
    t.truncate(width)
    return t


def render_header(h: HeaderInfo, width: int) -> Text:
    t = Text(no_wrap=True, overflow="ellipsis")
    t.append(" pciescope ", "bold white on dark_blue")
    if h.mode == "replay":
        t.append(" REPLAY ", "bold black on yellow")
        if h.pos_s is not None and h.dur_s is not None:
            t.append(f" {h.pos_s:6.1f}/{h.dur_s:.1f}s x{h.speed:g}")
        if h.paused:
            t.append(" ‖paused", "bold yellow")
    else:
        t.append(" LIVE ", "bold black on green")
        if h.paused:
            t.append(" ‖frozen", "bold yellow")
    if h.note:
        t.append(f" {h.note}", "bold yellow")
    t.append(f" {h.host}" if h.host else "")
    t.append(f"  {fmt.clock(h.ts_ns)}Z", "dim")
    if h.recording:
        t.append("  ● REC ", "bold red")
        t.append(h.session, "red")
    elif h.mode == "live":
        t.append("  ○ not recording", "dim")
    elif h.session:
        t.append(f"  {h.session}", "dim")
    t.truncate(width, overflow="ellipsis")
    return t


KEYS_COMMON = "1-5 view  space pause  e export  ? help  q quit"
KEYS_BY_VIEW = {
    "topology": "d detail",
    "live": "",
    "matrix": "m metric  [/] size  v direction",
    "errors": "c clear log",
    "measure": "↑↓ suite  s start  x cancel  g gpus",
}
KEYS_REPLAY = "←/→ seek  +/- speed"
KEYS_LIVE = "r rec on/off"


def render_keys(width: int, view: str, mode: str) -> Text:
    """Key hints; drops the least important groups until the line fits."""
    mode_keys = KEYS_REPLAY if mode == "replay" else KEYS_LIVE
    view_keys = KEYS_BY_VIEW.get(view, "")
    options = (
        [view_keys, mode_keys, KEYS_COMMON],
        [view_keys, KEYS_COMMON],
        [view_keys, "? help  q quit"],
        ["? help  q quit"],
    )
    line = ""
    for parts in options:
        line = " " + "  │  ".join(p for p in parts if p)
        if len(line) <= width:
            break
    t = Text(line, style="dim", no_wrap=True)
    t.truncate(width, overflow="ellipsis")
    return t


# ---------------------------------------------------------------------------
# link / device state
# ---------------------------------------------------------------------------


@dataclass
class LinkInfo:
    speed: float | None
    width: float | None
    max_speed: float | None
    max_width: float | None
    state: str  # ok | degraded | down | unknown

    @property
    def text(self) -> str:
        return link_text(self.speed, self.width)


def link_info(
    topo: Topology, state: ViewState, link_id: str | None, entity: str | None
) -> LinkInfo:
    """Current link state: live samples first, discovery snapshot as fallback.

    ``entity`` is the downstream device (``gpu:..`` or ``port:..``) whose own
    link end is read; ``link_id`` supplies the capability.
    """
    link = topo.links.get(link_id) if link_id else None
    max_s = link.max_speed_gts if link else None
    max_w = link.max_width if link else None
    speed = width = None
    if link_id:
        speed = state.value(link_id, "link.speed_gts")
        width = state.value(link_id, "link.width")
    if speed is None and entity:
        speed = state.value(entity, "pcie.link_speed_gts")
        if speed is None:
            speed = state.value(entity, "plx.link.speed")
    if width is None and entity:
        width = state.value(entity, "pcie.link_width")
        if width is None:
            width = state.value(entity, "plx.link.width")
    if speed is None and width is None and link is not None:
        node = topo.nodes.get(link.down)
        if node is not None:
            speed, width = node.link_speed_gts, node.link_width
    if speed is None and width is None:
        return LinkInfo(None, None, max_s, max_w, "unknown")
    if not speed or not width:
        return LinkInfo(speed, width, max_s, max_w, "down")
    degraded = (link_id and state.value(link_id, "link.degraded") == 1) or (
        (max_s and speed < max_s - 0.01) or (max_w and width < max_w)
    )
    return LinkInfo(speed, width, max_s, max_w, "degraded" if degraded else "ok")


STATE_STYLE = {
    "ok": "green",
    "degraded": "yellow",
    "down": "bold red",
    "unknown": "dim",
    "error": "bold red",
}
STATE_DOT = {"ok": "●", "degraded": "▲", "down": "✖", "unknown": "○", "error": "✖"}


def recent_error_entities(state: ViewState) -> set[str]:
    """Entities with a counter delta / warn event in the last minute."""
    if state.last_ts is None:
        return set()
    cutoff = state.last_ts - RECENT_ERR_NS
    return {e.entity for e in state.errors if e.ts_ns >= cutoff}


def _chain_up(topo: Topology, bdf: str, stop: str | None) -> list:
    """Nodes from ``bdf`` up to ``stop`` (inclusive), bottom first."""
    out = []
    node = topo.nodes.get(bdf)
    while node is not None:
        out.append(node)
        if node.bdf == stop or not node.parent:
            break
        node = topo.nodes.get(node.parent)
    return out


def _gpu_error_entities(topo: Topology, g: Gpu) -> set[str]:
    """Entities whose errors belong to a GPU: itself, its port, bridges, links."""
    ents = {g.id}
    if g.link:
        ents.add(g.link)
    for n in _chain_up(topo, g.bdf, g.port):
        ents.add(n.entity)
        above = topo.link_above(n.bdf)
        if above is not None:
            ents.add(above.id)
    return ents


# ---------------------------------------------------------------------------
# (a) topology
# ---------------------------------------------------------------------------


def _dot(t: Text, st: str) -> None:
    t.append(STATE_DOT[st] + " ", STATE_STYLE[st])


def _switch_of(topo: Topology, root):
    sw = root.switch
    if not sw:
        return None
    return topo.switches.get(sw) or topo.switches.get(switch_entity(sw))


def render_topology(
    topo: Topology, state: ViewState, width: int = 80, detail: bool = False
) -> Text:
    """Roots -> switches -> ports -> GPUs, one line per device.

    Colours: green = link at its maximum, yellow = degraded/downshifted, red =
    link down or errors in the last minute, dim = unused port.  ``detail`` adds
    the bridge chain between the switch upstream port and each GPU.
    """
    out = Text(no_wrap=True, overflow="ellipsis")
    wide = width >= 100
    err_ents = recent_error_entities(state)
    roots = sorted(topo.roots.values(), key=lambda r: r.bus)
    if not roots:
        out.append("no PCIe roots with a switch found in the topology\n", "red")
    for ri, root in enumerate(roots):
        sw = _switch_of(topo, root)
        gl = [topo.gpus[x] for x in root.gpus if x in topo.gpus]
        out.append(f"root:{root.bus}", "bold cyan")
        out.append(f"  root port {root.root_port}  {len(gl)} GPU\n", "dim")
        if sw is None:
            continue
        up_link = topo.link_above(sw.upstream)
        up_ent = port_entity(sw.upstream)
        li = link_info(topo, state, up_link.id if up_link else None, up_ent)
        slots = sorted({g.slot for g in gl if g.slot})
        out.append("└─ ", "dim")
        _dot(out, "error" if up_ent in err_ents else li.state)
        out.append(f"switch {sw.letter}", "bold")
        out.append(f"  {sw.model}  up {sw.upstream}  ")
        out.append(li.text, STATE_STYLE[li.state])
        if slots:
            out.append(f"  slot {'/'.join(slots)}", "magenta")
        out.append("\n")
        by_port = {g.port: g for g in gl if g.port}

        def order(p: str, by_port=by_port, sw=sw) -> tuple:
            if p in by_port:
                return (0, by_port[p].hip if by_port[p].hip is not None else 99, p)
            node = topo.nodes.get(p)
            return (1 if node and sw.mgmt in node.children else 2, 0, p)

        ports = sorted(sw.ports, key=order)
        for i, bdf in enumerate(ports):
            last = i == len(ports) - 1
            out.append("   " + ("└─ " if last else "├─ "), "dim")
            g = by_port.get(bdf)
            if g is None:
                _topology_unused_port(out, topo, state, bdf, sw)
            else:
                _topology_gpu(out, topo, state, g, err_ents, wide)
                if detail:
                    pad = "   " + ("   " if last else "│  ") + "      "
                    for line in _chain_lines(topo, state, g, sw.upstream):
                        out.append(pad + line + "\n", "dim")
        if ri != len(roots) - 1:
            out.append("\n")
    if topo.notes:
        out.append("\n" + "; ".join(topo.notes)[: width * 2] + "\n", "dim")
    return out


def _topology_unused_port(
    out: Text, topo: Topology, state: ViewState, bdf: str, sw
) -> None:
    node = topo.nodes.get(bdf)
    out.append(f"{bdf}  ", "dim")
    if node is not None and sw.mgmt and sw.mgmt in node.children:
        above = topo.link_above(sw.mgmt)
        li = link_info(topo, state, above.id if above else None, port_entity(bdf))
        out.append(f"mgmt endpoint {sw.mgmt}  {li.text}\n", "dim")
    else:
        out.append("unused (link down)\n", "dim")


def _topology_gpu(
    out: Text,
    topo: Topology,
    state: ViewState,
    g: Gpu,
    err_ents: set[str],
    wide: bool,
) -> None:
    li = link_info(topo, state, g.link, g.id)
    st = "error" if _gpu_error_entities(topo, g) & err_ents else li.state
    out.append(f"{g.port}  ", "dim")
    _dot(out, st)
    out.append(f"{gpu_label(g):<6}", "bold")
    out.append(f" {gpu_model(g):<6} {g.arch:<7}")
    out.append(f" kfd{g.kfd_node:<2}" if g.kfd_node is not None else " kfd? ")
    if wide and g.smi is not None:
        out.append(f" smi{g.smi}", "dim")
    out.append("  ")
    out.append(li.text, STATE_STYLE[li.state])
    if li.state == "degraded" and li.max_speed:
        out.append(f" (max {li.max_speed:g}Gx{int(li.max_width or 0)})", "yellow")
    if g.slot:
        out.append(f"  slot {g.slot}", "magenta")
    if wide and g.vram_total_bytes:
        out.append(f"  {fmt.human_bytes(g.vram_total_bytes)}", "dim")
    out.append("\n")


def _chain_lines(topo: Topology, state: ViewState, g: Gpu, upstream: str) -> list[str]:
    """Bridge chain between the switch upstream port and the GPU (detail mode)."""
    above = [n for n in _chain_up(topo, g.port, upstream)[1:] if n.bdf != upstream]
    card = _chain_up(topo, g.bdf, g.port)[:-1]
    lines = []
    if above:
        lines.append("switch " + " ▸ ".join(n.bdf for n in reversed(above)))
    if card:
        lines.append("card   " + " ▸ ".join(n.bdf for n in reversed(card)))
    return lines


# ---------------------------------------------------------------------------
# (b) live throughput / utilisation
# ---------------------------------------------------------------------------


def _na(t: Text, width: int, style: str = "dim") -> None:
    t.append("n/a".rjust(width), style)


def gpu_throughput(
    topo: Topology, state: ViewState, g: Gpu
) -> tuple[float | None, float | None, str]:
    """(rx, tx) GB/s of a GPU from the best source, plus the source name.

    rx = into the GPU, tx = out of the GPU.  Preference: amdgpu pcie_bw, then
    the switch port the card hangs off (``plx.perf.out_gbps`` is switch->GPU).
    """
    rx = state.value(g.id, "gpu.pcie_rx_bytes_per_s")
    tx = state.value(g.id, "gpu.pcie_tx_bytes_per_s")
    if rx is not None or tx is not None:
        return (
            None if rx is None else rx / 1e9,
            None if tx is None else tx / 1e9,
            "pcie_bw",
        )
    if g.port:
        pe = port_entity(g.port)
        out_g = state.value(pe, "plx.perf.out_gbps")
        in_g = state.value(pe, "plx.perf.in_gbps")
        if out_g is not None or in_g is not None:
            return out_g, in_g, "plx"
    return None, None, ""


def throughput_reason(state: ViewState) -> str:
    """Why the throughput columns are n/a (from collector availability)."""
    reasons = []
    if not state.has_metric("gpu.pcie_rx_bytes_per_s"):
        reasons.append(PCIE_BW_REASON)
    plx = state.unavailable.get("plx")
    if plx:
        reasons.append(f"plx: {plx}")
    elif "plx" in state.collectors and not state.has_metric("plx.perf.in_gbps"):
        reasons.append("plx: counters off")
    elif "plx" not in state.collectors:
        reasons.append("plx: collector not enabled")
    return "; ".join(reasons)


def _spark_series(
    topo: Topology, state: ViewState, g: Gpu
) -> tuple[list[float], float | None, str]:
    """Series + fixed scale + label for the sparkline column."""
    if g.port:
        pe = port_entity(g.port)
        tx = state.values(pe, "plx.perf.in_gbps")
        rx = state.values(pe, "plx.perf.out_gbps")
        if tx or rx:
            n = max(len(tx), len(rx))
            tx = [0.0] * (n - len(tx)) + tx
            rx = [0.0] * (n - len(rx)) + rx
            li = topo.links.get(g.link) if g.link else None
            return (
                [a + b for a, b in zip(tx, rx)],
                (li.max_gbs * 2 if li and li.max_gbs else None),
                "GB/s",
            )
    vals = state.values(g.id, "gpu.pcie_rx_bytes_per_s")
    if vals:
        return vals, None, "GB/s"
    return state.values(g.id, "gpu.busy_pct"), 100.0, "busy"


def _cap_gbs(topo: Topology, g: Gpu) -> float:
    link = topo.links.get(g.link) if g.link else None
    return (link.max_gbs if link and link.max_gbs else 0.0) or 31.5


def render_live(topo: Topology, state: ViewState, width: int = 80) -> Text:
    out = Text(no_wrap=True, overflow="ellipsis")
    fixed = 13 + 1 + 6 + 1 + 7 + 1 + 7 + 1 + 1 + 4 + 1 + 4 + 1 + 8
    spark_w = max(8, min(60, width - fixed))
    out.append(f"{'GPU':<13} {'link':<6} {'RX GB/s':>7} {'TX GB/s':>7} ", "bold")
    out.append(f"{'history':<{spark_w}}", "bold")
    out.append(f" {'busy':>4} {'temp':>4} {'VRAM GiB':>8}\n", "bold")
    any_tp = False
    last_sw = None
    for g in topo.gpu_list():
        if g.switch != last_sw:
            last_sw = g.switch
            out.append(
                f"─ switch {g.switch or '?'} ".ljust(width, "─")[:width] + "\n", "dim"
            )
        li = link_info(topo, state, g.link, g.id)
        rx, tx, src = gpu_throughput(topo, state, g)
        any_tp = any_tp or bool(src)
        series, vmax, label = _spark_series(topo, state, g)
        busy = state.value(g.id, "gpu.busy_pct")
        temp = state.value(g.id, "gpu.temp_junction_c")
        if temp is None:
            temp = state.value(g.id, "gpu.temp_edge_c")
        vu = state.value(g.id, "gpu.vram_used_bytes")
        vt = state.value(g.id, "gpu.vram_total_bytes") or (g.vram_total_bytes or None)
        out.append(f"{gpu_label(g):<6} {gpu_model(g):<6}", "bold")
        out.append(f" {li.text:<6} ", STATE_STYLE[li.state])
        cap = _cap_gbs(topo, g)
        for v in (rx, tx):
            if v is None:
                _na(out, 7)
            else:
                out.append(f"{v:7.2f}", fmt.util_style(v / cap))
            out.append(" ")
        out.append(
            fmt.sparkline(series, spark_w, vmax), "cyan" if label == "GB/s" else "blue"
        )
        out.append(" ")
        if busy is None:
            _na(out, 4)
        else:
            out.append(f"{busy:3.0f}%", fmt.util_style(busy / 100))
        out.append(" ")
        if temp is None:
            _na(out, 4)
        else:
            out.append(f"{temp:3.0f}C", fmt.temp_style(temp))
        out.append(" ")
        if vu is not None and vt:
            out.append(
                f"{vu / 2**30:4.1f}/{vt / 2**30:<3.0f}", "red" if vu / vt > 0.95 else ""
            )
        else:
            _na(out, 8)
        out.append("\n")
    out.append(
        "history: switch-port GB/s (in+out)\n"
        if any_tp
        else "history: GPU busy % (no throughput counters)\n",
        "dim",
    )
    out.append(_uplink_block(topo, state, width))
    reason = throughput_reason(state)
    if reason and not any_tp:
        out.append(f"RX/TX n/a: {reason}\n", "yellow")
    out.append(_collectors_line(state))
    return out


def _collectors_line(state: ViewState) -> Text:
    """Which collectors run (with their per-tick cost) and which are off."""
    cost = {
        lt.tags.get("collector", "?"): lt.value
        for _, _, lt in state.tagged("collector.cost_ms")
    }
    parts = [f"{c} {cost[c]:.1f}ms" if c in cost else c for c in state.collectors]
    out = Text(no_wrap=True, overflow="ellipsis")
    out.append("collectors: " + "  ".join(parts) + "\n", "dim")
    for name, why in sorted(state.unavailable.items()):
        out.append(f"  off: {name}: {why}\n", "dim yellow")
    return out


def _uplink_block(topo: Topology, state: ViewState, width: int) -> Text:
    """One line per switch uplink: capacity, h2d/d2h GB/s, history, utilisation.

    ``plx.perf.in`` on the upstream port is host->switch (h2d), ``out`` is d2h.
    """
    out = Text(no_wrap=True, overflow="ellipsis")
    spark_w = 10 if width < 100 else 20
    for sw in sorted(topo.switches.values(), key=lambda s: s.letter):
        link = topo.link_above(sw.upstream)
        pe = port_entity(sw.upstream)
        li = link_info(topo, state, link.id if link else None, pe)
        cap = link.max_gbs if link else None
        out.append(f"uplink {sw.letter} ", "bold")
        out.append(f"{li.text:<6}", STATE_STYLE[li.state])
        out.append(f" cap {cap:.1f}" if cap else " cap n/a")
        for label, metric in (
            ("h2d", "plx.perf.in_gbps"),
            ("d2h", "plx.perf.out_gbps"),
        ):
            v = state.value(pe, metric)
            out.append(f"  {label} ")
            if v is None:
                out.append(fmt.NA, "dim")
                continue
            out.append(f"{v:5.2f} ")
            out.append(fmt.sparkline(state.values(pe, metric), spark_w, cap), "cyan")
        util = [state.value(pe, m) for m in ("plx.perf.in_util", "plx.perf.out_util")]
        util = [u for u in util if u is not None]
        if util:
            out.append(f"  util {max(util) * 100:.0f}%", fmt.util_style(max(util)))
        out.append("\n")
    return out


# ---------------------------------------------------------------------------
# (c) latency / bandwidth matrix
# ---------------------------------------------------------------------------

MATRIX_VIEWS = ("bw", "eff", "lat")
MATRIX_TITLE = {
    "bw": "bandwidth GB/s",
    "eff": "efficiency % of line rate",
    "lat": "latency us",
}
DIR_ORDER = ("d2d-push", "d2d-pull", "d2d-sdma", "h2d", "d2h")


@dataclass
class MatrixSel:
    view: str = "bw"  # bw | eff | lat
    direction: str | None = None  # None = first available
    size_idx: int = -1  # index into the sorted sizes (-1 = largest)

    def cycle_view(self) -> None:
        self.view = MATRIX_VIEWS[(MATRIX_VIEWS.index(self.view) + 1) % 3]
        self.size_idx = 0 if self.view == "lat" else -1

    def cycle_size(self, step: int) -> None:
        self.size_idx += step

    def cycle_direction(self, directions: list[str]) -> None:
        if not directions:
            return
        cur = self.direction if self.direction in directions else directions[0]
        self.direction = directions[(directions.index(cur) + 1) % len(directions)]


def _kind(view: str) -> str:
    return "lat" if view == "lat" else "bw"


def latest_cells(
    ms: list[Measurement], kind: str
) -> dict[tuple[str, str, str, int], Measurement]:
    cells: dict[tuple[str, str, str, int], Measurement] = {}
    for m in ms:
        if m.kind != kind or not getattr(m, "ok", True):
            continue
        k = (m.src, m.dst, m.direction, m.size_bytes)
        if k not in cells or m.started_ns >= cells[k].started_ns:
            cells[k] = m
    return cells


def matrix_choices(state: ViewState, view: str) -> tuple[list[str], list[int]]:
    cells = latest_cells(state.measurements, _kind(view))
    dirs = [d for d in DIR_ORDER if any(k[2] == d for k in cells)]
    dirs += sorted({k[2] for k in cells} - set(dirs))
    sizes = sorted({k[3] for k in cells})
    return dirs, sizes


def _cell_text(
    m: Measurement | None, view: str, cols: tuple[float, float]
) -> tuple[str, str]:
    """(4-char text, style) for one matrix cell."""
    if m is None:
        return "   ·", "dim"
    if view == "lat":
        return f"{m.value:4.1f}", fmt.lat_style(m.value, cols[0], cols[1])
    if view == "eff":
        e = m.efficiency
        return (" n/a", "dim") if e is None else (f"{e * 100:3.0f}%", fmt.eff_style(e))
    v = f"{m.value:4.1f}" if m.value < 100 else f"{m.value:4.0f}"
    return v, fmt.eff_style(m.efficiency)


def render_matrix(
    topo: Topology, state: ViewState, width: int = 80, sel: MatrixSel | None = None
) -> Text:
    sel = sel or MatrixSel()
    out = Text(no_wrap=True, overflow="ellipsis")
    kind = _kind(sel.view)
    cells = latest_cells(state.measurements, kind)
    if not cells:
        out.append(f"No {MATRIX_TITLE[sel.view]} results yet.\n", "yellow")
        out.append(
            "Run a suite from the Measure view (5) or open a session with "
            "probe results (--replay).\n",
            "dim",
        )
        others = [m for m in state.measurements if m.kind != kind]
        if others:
            out.append(
                f"({len(others)} other measurements: "
                f"{', '.join(sorted({m.kind for m in others}))}; press m)\n",
                "dim",
            )
        return out
    dirs, sizes = matrix_choices(state, sel.view)
    direction = sel.direction if sel.direction in dirs else dirs[0]
    sel.direction = direction
    sel.size_idx = max(-len(sizes), min(len(sizes) - 1, sel.size_idx))
    size = sizes[sel.size_idx]
    cur = {
        (s, d): m
        for (s, d, di, sz), m in cells.items()
        if di == direction and sz == size
    }
    if not cur:  # direction has no cell at this size: take its largest size
        szs = sorted(k[3] for k in cells if k[2] == direction)
        size = szs[-1]
        cur = {
            (s, d): m
            for (s, d, di, sz), m in cells.items()
            if di == direction and sz == size
        }
    gpu_order = [g.id for g in topo.gpu_list()]
    ents = [e for e in gpu_order if any(e in k for k in cur)]
    if any("host" in k for k in cur):
        ents.append("host")
    for e in sorted({x for k in cur for x in k} - set(ents)):
        ents.append(e)
    out.append(
        f"{MATRIX_TITLE[sel.view]}  {direction}  {fmt.human_bytes(size)}"
        f"  (rows=src, cols=dst)\n",
        "bold",
    )
    vals = [m.value for m in cur.values() if m.kind == "lat"]
    bounds = (min(vals), max(vals)) if vals else (0.0, 0.0)
    cw = 6
    labw = 7
    # header: columns labelled with the HIP index (+ bus when it fits)
    gaps = _gaps(topo, ents)
    out.append(" " * labw)
    for i, e in enumerate(ents):
        if i in gaps:
            out.append("│")
        out.append(entity_label(topo, e).split(":")[0].rjust(cw), "bold")
    out.append("\n")
    for i, s in enumerate(ents):
        if i in gaps:
            out.append(
                ("─" * labw)
                + "".join(
                    ("┼" if j in gaps else "") + "─" * cw for j in range(len(ents))
                )
                + "\n",
                "dim",
            )
        out.append(entity_label(topo, s).ljust(labw), "bold")
        for j, d in enumerate(ents):
            if j in gaps:
                out.append("│", "dim")
            if s == d:
                out.append("╲".rjust(cw), "dim")
                continue
            m = cur.get((s, d))
            txt, style = _cell_text(m, sel.view, bounds)
            mark = " "
            if m is not None:
                mark = (
                    "×"
                    if m.path_class == "cross-root"
                    else ("h" if m.path_class == "host" else " ")
                )
            out.append((txt + mark).rjust(cw), style)
        out.append("\n")
    out.append(_matrix_summary(cur, sel.view, topo))
    out.append("mark: × cross-root, h host, none = intra-switch\n", "dim")
    out.append("colour: ", "dim")
    legend = (
        ("≥80%", "black on green"),
        ("≥50%", "black on yellow"),
        ("≥25%", "black on dark_orange"),
        ("<25%", "white on red"),
    )
    if sel.view == "lat":
        legend = (
            ("best", "black on green"),
            ("", "black on yellow"),
            ("", "black on dark_orange"),
            ("worst", "white on red"),
        )
    for txt, style in legend:
        out.append(f" {txt or '··'} ", style)
    out.append("\n")
    return out


def _gaps(topo: Topology, ents: list[str]) -> set[int]:
    """Indices where the root complex changes (visual separator)."""
    gaps = set()
    prev = None
    for i, e in enumerate(ents):
        g = topo.gpus.get(e)
        grp = g.root if g else "host"
        if prev is not None and grp != prev:
            gaps.add(i)
        prev = grp
    return gaps


def _matrix_summary(cur: dict, view: str, topo: Topology) -> Text:
    out = Text(no_wrap=True, overflow="ellipsis")
    by_class: dict[str, list[Measurement]] = {}
    for m in cur.values():
        by_class.setdefault(m.path_class or "?", []).append(m)
    for pc in ("intra-switch", "cross-root", "host"):
        ms = by_class.get(pc)
        if not ms:
            continue
        v = statistics.median(m.value for m in ms)
        out.append(f"{pc:<13}", "bold")
        out.append(f" n={len(ms):<3}")
        if view == "lat":
            out.append(f" median {v:5.1f} us  worst {max(m.value for m in ms):5.1f}")
        else:
            effs = [m.efficiency for m in ms if m.efficiency is not None]
            out.append(f" median {v:6.2f} GB/s  min {min(m.value for m in ms):6.2f}")
            if effs:
                e = statistics.median(effs)
                out.append(f"  eff {e * 100:3.0f}%", fmt.eff_style(e))
        out.append("\n")
    return out


def render_other_measurements(state: ViewState, width: int, limit: int = 6) -> Text:
    """Concurrency / curve results that do not fit a pair matrix."""
    out = Text(no_wrap=True, overflow="ellipsis")
    rows = [m for m in state.measurements if m.kind not in ("lat", "bw")][-limit:]
    for m in rows:
        out.append(
            f"{m.kind:<12} {m.direction:<9} {fmt.human_bytes(m.size_bytes):>9} "
            f"{m.value:8.2f} {m.unit:<5} {m.path_class}",
            "",
        )
        if m.efficiency is not None:
            out.append(f"  eff {m.efficiency * 100:.0f}%", fmt.eff_style(m.efficiency))
        out.append("\n")
    return out


# ---------------------------------------------------------------------------
# (d) errors
# ---------------------------------------------------------------------------


def render_errors(
    topo: Topology | None, state: ViewState, width: int = 80, limit: int = 60
) -> Text:
    out = Text(no_wrap=True, overflow="ellipsis")
    n = state.n_error_series()
    nz = state.counters_nonzero()
    if n == 0:
        out.append("no AER / switch error registers sampled", "yellow")
        out.append(" (collectors: aer, plx unavailable or not started)\n", "dim")
    else:
        style = "green" if not nz else "yellow"
        out.append(f"{n} error series watched, {len(nz)} non-zero ", style)
        out.append("(standing since boot / driver load)\n", "dim")
    for ent, what, val in nz[:12]:
        out.append(f"  {entity_label(topo, ent):<14} {what:<34} {val:>10g}\n", "yellow")
    if len(nz) > 12:
        out.append(f"  … {len(nz) - 12} more\n", "dim")
    out.append("\n")
    ww = 22 if width < 100 else 34
    out.append(
        f"{'time(UTC)':<9} {'entity':<9} {'what':<{ww}} {'delta':>6} detail\n", "bold"
    )
    rows = list(state.errors)[-limit:][::-1]
    if not rows:
        out.append("no new errors since the TUI started\n", "green")
    for r in rows:
        style = "bold red" if r.severity == "error" else "yellow"
        d = "" if r.delta is None else f"+{r.delta:g}"
        what = r.what if len(r.what) <= ww else r.what[: ww - 1] + "…"
        out.append(
            f"{fmt.clock(r.ts_ns):<9} {entity_label(topo, r.entity):<9} "
            f"{what:<{ww}} {d:>6} {r.text}\n",
            style,
        )
    return out


# ---------------------------------------------------------------------------
# (e) measure panel
# ---------------------------------------------------------------------------


@dataclass
class MeasureModel:
    suites: list = field(
        default_factory=list
    )  # SuiteInfo-like (name, description, est_seconds)
    sel: int = 0
    gpus: list = field(default_factory=list)  # gpu ids chosen
    lease_ok: bool = False
    lease_reason: str = ""
    lease_needed: list = field(default_factory=list)
    status: str = "idle"  # idle | running | done | refused | aborted | failed
    log: list = field(default_factory=list)
    i: int = 0
    n: int = 0
    est_s: float | None = None
    started: float | None = None
    owner: str = ""
    error: str = ""


def render_measure(
    topo: Topology | None,
    m: MeasureModel,
    width: int = 80,
    height: int = 20,
    elapsed_s: float | None = None,
) -> Text:
    out = Text(no_wrap=True, overflow="ellipsis")
    out.append("Probe suites ", "bold")
    out.append("(active: only on GPUs whose lease you hold)\n", "dim")
    if not m.suites:
        out.append("  probe module unavailable or lists no suites\n", "red")
    for i, s in enumerate(m.suites):
        cur = i == m.sel
        out.append(
            ("▶ " if cur else "  ") + f"{s.name:<10}", "bold reverse" if cur else "bold"
        )
        out.append(f" ~{s.est_seconds:>4}s  {s.description}\n", "" if cur else "dim")
    out.append("\nGPUs: ", "bold")
    if m.gpus:
        out.append(" ".join(entity_label(topo, g) for g in m.gpus))
    else:
        out.append("none selected (g to choose among leased GPUs)", "yellow")
    out.append("\nLease: ", "bold")
    if m.lease_ok:
        out.append(f"ok{(' (' + m.owner + ')') if m.owner else ''}\n", "green")
    else:
        out.append(m.lease_reason or "not checked", "red")
        out.append("\n")
        if m.lease_needed:
            out.append("  needs: " + ", ".join(m.lease_needed) + "\n", "red")
            out.append(
                "  (lease files live in ~/w4a8_runs/lanes/lease/; the lead "
                "grants them)\n",
                "dim",
            )
    out.append("Status: ", "bold")
    st_style = {"running": "bold cyan", "done": "green", "idle": "dim"}.get(
        m.status, "red"
    )
    out.append(m.status, st_style)
    if m.status == "running" and m.n:
        w = max(10, min(40, width - 40))
        out.append(f"  [{fmt.bar(m.i / m.n, w)}] {m.i}/{m.n}")
        if elapsed_s is not None:
            out.append(f"  {elapsed_s:.0f}s")
            if m.est_s:
                out.append(f"/~{m.est_s:.0f}s")
    if m.error:
        out.append(f"  {m.error}", "red")
    out.append("\n")
    room = max(3, height - (len(m.suites) + 9))
    for line in m.log[-room:]:
        out.append("  " + line + "\n", "dim")
    return out


# ---------------------------------------------------------------------------
# whole-screen text (tests, ssh capture)
# ---------------------------------------------------------------------------


def to_plain(t: Text, width: int = 80) -> str:
    """Render a Text at a fixed width to plain text (what a terminal shows)."""
    from rich.console import Console

    con = Console(
        width=width,
        record=True,
        file=_NullFile(),
        force_terminal=False,
        color_system=None,
        legacy_windows=False,
    )
    con.print(t, end="", overflow="ellipsis", no_wrap=True, soft_wrap=False)
    return con.export_text()


class _NullFile:
    def write(self, *_):
        return 0

    def flush(self):
        pass
