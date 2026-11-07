# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which switches may have their traffic counters switched on.

The PerformancePAK counters need register writes (``plx/helper.py``).  The user
approved them for switch B only (2026-11-07, conditions: idle switch, lease held,
health diff after every phase); switch A waits for a quiet window of the hetero
work.  A plain ``--allow-switch-writes`` therefore must not touch switch A:
``DEFAULT_WRITE_SWITCHES`` lists the approved switches, and the environment
variable ``PCIESCOPE_PLX_WRITE_SWITCHES`` (comma list of letters, empty = none)
overrides it for one invocation once more are approved.  The helper maps the
BAR0 of every other chip ``PROT_READ``, so the fence also holds below the
collector.

Independent of the switch list, the switch port of the W7800 of HIP 5
(``gpu:07``, lease key ``w7800-07``, the lead only) is never written:
``NEVER_WRITTEN_GPUS`` is merged into every exclusion list, and the root helper
refuses the whole station of that port itself (``regs.PROTECTED_STATIONS``).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

ENV_WRITE_SWITCHES = "PCIESCOPE_PLX_WRITE_SWITCHES"
DEFAULT_WRITE_SWITCHES = ("B",)
#: GPUs whose switch port is never written, whatever the caller asks for
NEVER_WRITTEN_GPUS = ("gpu:07",)
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
