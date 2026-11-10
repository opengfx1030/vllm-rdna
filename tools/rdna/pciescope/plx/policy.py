# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which switches may have their traffic counters switched on.

The PerformancePAK counters need register writes (``plx/helper.py``) and every
chip needs the user's approval.  Switch B of the pre-move layout had it
(2026-11-07: idle switch, lease held, health diff after every phase) and ran clean;
the SlimSAS card that carries all six GPUs since the 2026-11-08 case move is a new
chip and has none, so ``DEFAULT_WRITE_SWITCHES`` is empty and a plain
``--allow-switch-writes`` writes nothing.  Once the user approves, the environment
variable ``PCIESCOPE_PLX_WRITE_SWITCHES`` (comma list of letters, empty = none)
names the switches for one invocation.  The helper maps the BAR0 of every other
chip ``PROT_READ``, so the fence also holds below the collector.

Independent of the switch list, the switch port of a GPU in ``NEVER_WRITTEN_GPUS``
is never written: the list is merged into every exclusion list, and the root
helper refuses the whole station of that port itself (``regs.KnownChip.protected``).
Nothing is listed now: the only such GPU was the W7800 of HIP 5, which is gone.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

ENV_WRITE_SWITCHES = "PCIESCOPE_PLX_WRITE_SWITCHES"
DEFAULT_WRITE_SWITCHES: tuple[str, ...] = ()
#: GPUs whose switch port is never written, whatever the caller asks for; a GPU
#: listed here needs its station in ``regs.KnownChip.protected`` (the collector
#: refuses to start the counters otherwise)
NEVER_WRITTEN_GPUS: tuple[str, ...] = ()
_LETTERS = frozenset("AB")


def parse_switches(spec: str | Iterable[str]) -> frozenset[str]:
    """``"A,b"`` or ``["A", "b"]`` -> ``{"A", "B"}``; unknown letters raise."""
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    letters = frozenset(i.strip().upper() for i in items if i.strip())
    bad = sorted(letters - _LETTERS)
    if bad:
        raise ValueError(f"unknown switch letter(s) {bad}: use A and/or B")
    return letters


def write_switches(
    explicit: Iterable[str] | None = None, environ: Mapping[str, str] | None = None
) -> frozenset[str]:
    """Switches whose counters may be enabled: argument, else env, else default."""
    if explicit is not None:
        return parse_switches(explicit)
    env = os.environ if environ is None else environ
    if ENV_WRITE_SWITCHES in env:
        return parse_switches(env[ENV_WRITE_SWITCHES])
    return frozenset(DEFAULT_WRITE_SWITCHES)
