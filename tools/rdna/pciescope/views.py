# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact views of one sampling pass (``snap`` verb, MCP ``snapshot`` tool).

``snapshot_summary`` folds the raw samples of one tick into per-GPU, per-link and
error tables (JSON-ready); ``render_snapshot`` prints that summary for humans.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Any

from pciescope.model import Sample, Topology

_GPU_FIELDS = (
    ("gpu.busy_pct", "busy_pct"),
    ("gpu.mem_busy_pct", "mem_busy_pct"),
    ("gpu.temp_edge_c", "temp_edge_c"),
    ("gpu.temp_junction_c", "temp_junction_c"),
    ("gpu.temp_mem_c", "temp_mem_c"),
    ("gpu.power_w", "power_w"),
    ("gpu.vram_used_bytes", "vram_used_bytes"),
    ("gpu.vram_total_bytes", "vram_total_bytes"),
    ("gpu.gtt_used_bytes", "gtt_used_bytes"),
    ("gpu.throttle_status", "throttle_status"),
)
_LINK_FIELDS = (
    ("link.speed_gts", "speed_gts"),
    ("link.width", "width"),
    ("link.capacity_gbs", "capacity_gbs"),
    ("link.degraded", "degraded"),
)
_KNOWN = {m for m, _ in _GPU_FIELDS + _LINK_FIELDS} | {
    "gpu.link_speed_gts",
    "gpu.link_width",
    "gpu.pcie_dpm_speed_gts",
    "gpu.pcie_dpm_width",
    "pcie.link_speed_gts",
    "pcie.link_width",
    "kfd.evicted_ms",
    "collector.cost_ms",
    "collector.samples",
    "gpu.asleep",
    "pcie.asleep",
}


def _num(x: float | None) -> float | int | None:
    if x is None:
        return None
    return int(x) if float(x).is_integer() and abs(x) < 1e15 else round(x, 3)


def snapshot_summary(
    topo: Topology,
    samples: Iterable[Sample],
    *,
    costs_ms: Mapping[str, float] | None = None,
    collectors: Iterable[str] = (),
    skipped: Mapping[str, str] | None = None,
    ts_ns: int | None = None,
) -> dict[str, Any]:
    """Fold the samples of one tick into tables (see module docstring)."""
    plain: dict[tuple[str, str], float] = {}
    tagged: list[Sample] = []
    other: dict[str, int] = {}
    counts: dict[str, int] = {}
    for s in samples:
        counts[s.metric] = counts.get(s.metric, 0) + 1
        if s.tags:
            tagged.append(s)
        else:
            plain[(s.entity, s.metric)] = s.value
        if s.metric not in _KNOWN and not s.metric.startswith("aer."):
            other[s.metric] = other.get(s.metric, 0) + 1

    def val(entity: str, metric: str) -> float | int | None:
        return _num(plain.get((entity, metric)))

    kfd: dict[str, list[dict[str, Any]]] = {}
    for s in tagged:
        if s.metric == "kfd.evicted_ms":
            kfd.setdefault(s.entity, []).append(
                {
                    "pid": int(s.tags.get("pid", 0)),
                    "comm": s.tags.get("comm", ""),
                    "evicted_ms": _num(s.value),
                }
            )
    gpus = []
    for g in topo.gpu_list():
        row: dict[str, Any] = {
            "id": g.id,
            "hip": g.hip,
            "smi": g.smi,
            "arch": g.arch,
            "name": g.name,
            "switch": g.switch,
            "slot": g.slot,
        }
        for metric, key in _GPU_FIELDS:
            row[key] = val(g.id, metric)
        row["asleep"] = (
            bool(val(g.id, "gpu.asleep")) if (g.id, "gpu.asleep") in plain else None
        )
        row["gpu_link"] = {
            "speed_gts": val(g.id, "gpu.link_speed_gts"),
            "width": val(g.id, "gpu.link_width"),
            "dpm_speed_gts": val(g.id, "gpu.pcie_dpm_speed_gts"),
            "dpm_width": val(g.id, "gpu.pcie_dpm_width"),
        }
        row["card_link"] = g.link
        if g.id in kfd:
            row["kfd"] = sorted(kfd[g.id], key=lambda e: e["pid"])
        gpus.append(row)
    links = []
    for lk in topo.links_ordered(physical_only=True):
        item: dict[str, Any] = {
            "id": lk.id,
            "kind": lk.kind,
            "gpu": lk.gpu,
            "switch": lk.switch,
            "max_speed_gts": lk.max_speed_gts,
            "max_width": lk.max_width,
            "raw_gbs_at_max": lk.max_gbs,
        }
        for metric, key in _LINK_FIELDS:
            item[key] = val(lk.id, metric)
        if item["degraded"] is not None:
            item["degraded"] = bool(item["degraded"])
        links.append(item)
    aer = _aer_rows(topo, plain, tagged)
    asleep_ports = sorted(
        ent[5:]
        for (ent, metric), v in plain.items()
        if metric == "pcie.asleep" and v and ent.startswith("port:")
    )
    out: dict[str, Any] = {
        "host": topo.host,
        "ts_ns": ts_ns,
        "collectors": list(collectors),
        "skipped": dict(skipped or {}),
        "cost_ms": {k: round(v, 3) for k, v in (costs_ms or {}).items()},
        "n_samples": sum(counts.values()),
        "gpus": gpus,
        "links": links,
        "aer": aer,
        "asleep": {
            "ports": asleep_ports,
            "gpus": [g["id"] for g in gpus if g.get("asleep")],
        },
    }
    if other:
        out["other_metrics"] = other

    return out


def _aer_rows(
    topo: Topology,
    plain: dict[tuple[str, str], float],
    tagged: list[Sample],
) -> dict[str, Any]:
    """Non-zero AER counters (since boot) with the error types, plus the totals."""
    nonzero: list[dict[str, Any]] = []
    types: dict[tuple[str, str], dict[str, int]] = {}
    for s in tagged:
        if s.metric == "aer.error":
            key = (s.entity, s.tags.get("class", ""))
            types.setdefault(key, {})[s.tags.get("type", "?")] = int(s.value)
    n_devices = 0
    for (ent, metric), value in sorted(plain.items()):
        if not metric.startswith("aer."):
            continue
        n_devices += metric == "aer.correctable"
        if not value:
            continue
        cls = metric.split(".", 1)[1]
        label = ""
        if ent.startswith("port:"):
            node = topo.nodes.get(ent[5:])
            label = node.label if node else ""
        elif ent.startswith("gpu:"):
            g = topo.gpus.get(ent)
            label = g.name if g else ""
        row: dict[str, Any] = {
            "entity": ent,
            "label": label,
            "class": cls,
            "count": int(value),
        }
        if types.get((ent, cls)):
            row["types"] = types[(ent, cls)]
        nonzero.append(row)
    return {"devices_checked": n_devices, "nonzero": nonzero}


def _fmt(x: float | int | None, spec: str = "g", dash: str = "-") -> str:
    return dash if x is None else format(x, spec)


def render_snapshot(summary: Mapping[str, Any]) -> str:
    """Human-readable table form of ``snapshot_summary``."""
    when = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(summary["ts_ns"] / 1e9))
        if summary.get("ts_ns")
        else ""
    )
    cost = ", ".join(f"{k} {v:g} ms" for k, v in summary["cost_ms"].items())
    out = [
        f"{summary['host']} snapshot {when}  collectors: "
        f"{' '.join(summary['collectors'])}  ({summary['n_samples']} samples"
        + (f"; cost {cost}" if cost else "")
        + ")"
    ]
    for name, why in (summary.get("skipped") or {}).items():
        out.append(f"  skipped {name}: {why}")
    hdr = [
        "gpu", "hip", "smi", "arch", "sw", "slot", "busy%", "Tj C", "W",
        "VRAM GiB", "GPU link", "card link",
    ]  # fmt: skip
    rows = []
    link_by_id = {lk["id"]: lk for lk in summary["links"]}
    for g in summary["gpus"]:
        gl = g["gpu_link"]
        used, total = g.get("vram_used_bytes"), g.get("vram_total_bytes")
        vram = "-" if used is None else f"{used / 2**30:.1f}/{(total or 0) / 2**30:.0f}"
        card = link_by_id.get(g["card_link"] or "", {})
        card_txt = "-"
        if card.get("speed_gts") is not None:
            card_txt = f"{card['speed_gts']:g}GT/s x{_fmt(card['width'])}"
            if card.get("degraded"):
                card_txt += " DEGRADED"
        rows.append(
            [
                g["id"], _fmt(g["hip"]), _fmt(g["smi"]), g["arch"], g["switch"] or "-",
                g["slot"] or "-", _fmt(g["busy_pct"]), _fmt(g["temp_junction_c"]),
                _fmt(g["power_w"]), vram,
                "-" if gl["speed_gts"] is None
                else f"{gl['speed_gts']:g}GT/s x{_fmt(gl['width'])}",
                card_txt,
            ]
        )  # fmt: skip
    widths = [max(len(hdr[i]), *(len(r[i]) for r in rows)) for i in range(len(hdr))]
    line = "  ".join(f"{{:<{w}}}" for w in widths)
    out += [line.format(*hdr).rstrip(), line.format(*("-" * w for w in widths))]
    out += [line.format(*r).rstrip() for r in rows]
    degraded = [lk for lk in summary["links"] if lk.get("degraded")]
    n_ok = sum(1 for lk in summary["links"] if lk.get("degraded") is False)
    out.append("")
    out.append(
        f"links: {len(summary['links'])} physical, {n_ok} at their maximum"
        + (f", {len(degraded)} DEGRADED" if degraded else "")
    )
    for lk in degraded:
        out.append(
            f"  {lk['id']} ({lk['kind']}) {_fmt(lk['speed_gts'])} GT/s "
            f"x{_fmt(lk['width'])} = {_fmt(lk['capacity_gbs'])} GB/s, "
            f"max {lk['max_speed_gts']:g} GT/s x{lk['max_width']}"
        )
    asleep = summary.get("asleep") or {}
    if asleep.get("ports") or asleep.get("gpus"):
        ports = " ".join(asleep["ports"]) or "-"
        gpus = " ".join(asleep["gpus"]) or "-"
        out.append(
            f"asleep (runtime-suspended, attributes not read): ports {ports}; "
            f"GPUs {gpus}"
        )
    aer = summary["aer"]
    if aer["nonzero"]:
        out.append(
            f"AER: {len(aer['nonzero'])} non-zero counter(s) since boot "
            f"({aer['devices_checked']} devices checked)"
        )
        for e in aer["nonzero"]:
            types = f" {e['types']}" if e.get("types") else ""
            out.append(
                f"  {e['entity']:<12} {e['class']:<22} {e['count']}{types}  "
                f"{e['label']}"
            )
    else:
        out.append(f"AER: all zero ({aer['devices_checked']} devices checked)")
    kfd = [(g["id"], k) for g in summary["gpus"] for k in g.get("kfd", [])]
    if kfd:
        out.append("KFD queues:")
        for gid, k in kfd:
            out.append(
                f"  {gid} pid {k['pid']} {k['comm']} evicted {k['evicted_ms']} ms"
            )
    other = summary.get("other_metrics")
    if other:
        fam: dict[str, int] = {}
        for name, n in other.items():
            head = name.split(".", 1)[0]
            fam[head] = fam.get(head, 0) + n
        out.append(
            f"other metrics (see --json or `report`): {len(other)} metric(s), "
            + ", ".join(f"{h}.* {n} samples" for h, n in sorted(fam.items()))
        )
    return "\n".join(out)
