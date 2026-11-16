# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure formatting helpers for the TUI: sparklines, units, heat colours."""

from collections.abc import Sequence

SPARK = "▁▂▃▄▅▆▇█"
NA = "n/a"


def sparkline(values: Sequence[float], width: int, vmax: float | None = None) -> str:
    """Render the last ``width`` values as block characters.

    ``vmax`` fixes the scale (e.g. theoretical link bandwidth); without it the
    window maximum is used. Always returns exactly ``width`` characters, left
    padded with spaces when fewer samples exist.
    """
    if width <= 0:
        return ""
    vals = list(values)[-width:]
    if not vals:
        return " " * width
    top = vmax if vmax and vmax > 0 else max(vals)
    out = []
    for v in vals:
        if v is None or top <= 0:
            out.append(SPARK[0] if v is not None else " ")
            continue
        idx = int(min(max(v / top, 0.0), 1.0) * (len(SPARK) - 1) + 0.5)
        out.append(SPARK[idx])
    return "".join(out).rjust(width)


def bar(frac: float | None, width: int) -> str:
    """Horizontal bar of ``width`` cells for a 0..1 fraction (n/a -> dots)."""
    if frac is None:
        return "·" * width
    frac = min(max(frac, 0.0), 1.0)
    full = int(frac * width)
    rem = frac * width - full
    s = "█" * full
    if full < width:
        s += SPARK[int(rem * (len(SPARK) - 1))] if rem > 0.125 else " "
    return s.ljust(width)[:width]


def gbps(v: float | None, digits: int = 1) -> str:
    """Bytes/s -> 'x.y GB/s' (decimal GB, matching the brief's numbers)."""
    if v is None:
        return NA
    return f"{v / 1e9:.{digits}f}"


def human_bytes(n: float | None) -> str:
    if n is None:
        return NA
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TiB"


def pct(v: float | None, digits: int = 0) -> str:
    return NA if v is None else f"{v:.{digits}f}%"


def usec(v: float | None) -> str:
    return NA if v is None else f"{v:.1f}"


def clock(ts_ns: int | None) -> str:
    """Wall-clock HH:MM:SS from ns since epoch (UTC; stable for tests)."""
    if ts_ns is None:
        return "--:--:--"
    s = int(ts_ns // 1_000_000_000) % 86400
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def util_style(frac: float | None) -> str:
    """Rich style for a utilisation fraction (high = busy = bright)."""
    if frac is None:
        return "dim"
    if frac >= 0.75:
        return "bold green"
    if frac >= 0.25:
        return "green"
    if frac > 0.02:
        return "dark_green"
    return "dim"


def temp_style(c: float | None) -> str:
    """Thresholds follow box rules: 90 C mean / 95 C any = abort."""
    if c is None:
        return "dim"
    if c >= 95:
        return "bold white on red"
    if c >= 90:
        return "bold red"
    if c >= 80:
        return "yellow"
    return "green"


def eff_style(eff: float | None) -> str:
    """Rich style for an efficiency fraction (1.0 = theoretical)."""
    if eff is None:
        return "dim"
    if eff >= 0.80:
        return "black on green"
    if eff >= 0.50:
        return "black on yellow"
    if eff >= 0.25:
        return "black on dark_orange"
    return "white on red"


def lat_style(us: float | None, best: float | None, worst: float | None) -> str:
    """Latency heat: lowest = green, highest = red (relative to the matrix)."""
    if us is None or best is None or worst is None or worst <= best:
        return "dim" if us is None else "black on green"
    f = (us - best) / (worst - best)
    if f <= 0.25:
        return "black on green"
    if f <= 0.5:
        return "black on yellow"
    if f <= 0.75:
        return "black on dark_orange"
    return "white on red"
