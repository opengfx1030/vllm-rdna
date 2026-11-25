#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-touch a measured-window quiet file every ``--interval`` seconds.

One process with an in-process sleep (no ``sleep`` child), so the launcher's
``kill`` on exit cannot leave a stray sleeper behind; SIGTERM/SIGINT exit 0.

  quiet_touch.py --file ~/w4a8_runs/lanes/quiet-a2/inkling-hi2-smoke \
      [--interval 300] [--count 3]
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=300.0)
    ap.add_argument("--count", type=int, default=3, help="touches, 0 = forever")
    args = ap.parse_args(argv)

    def stop(*_args):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args.file.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(args.count or sys.maxsize):
        time.sleep(args.interval)
        args.file.touch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
