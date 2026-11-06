#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Junction watcher for one slot run (stdlib only).

  tempwatch.py BUS,BUS,BUS,BUS LOG1S LOGALL SERVE_PIDFILE ABORTFILE

LOG1S : slot GPUs every second  ("HH:MM:SS bus=degC ...")
LOGALL: every amdgpu every 15 s (same format as the merge smoke's temps.log)
Abort rule (task contract): any slot GPU's 5 s mean >= 90 C or any sample >= 95 C
-> write ABORTFILE and SIGTERM (then SIGKILL) the server's process group.
"""

import collections
import glob
import os
import signal
import sys
import time
from pathlib import Path


def sensors() -> dict[str, str]:
    out = {}
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if Path(f"{h}/name").read_text().strip() != "amdgpu":
                continue
            bus = os.path.basename(os.path.realpath(f"{h}/device")).removeprefix(
                "0000:"
            )
            out[bus] = f"{h}/temp2_input"
        except OSError:
            continue
    return out


def read(path: str) -> float:
    try:
        return int(Path(path).read_text()) / 1000.0
    except (OSError, ValueError):
        return float("nan")


def main() -> int:
    buses = [b.removeprefix("0000:") for b in sys.argv[1].split(",")]
    log1, logall, pidfile, abortfile = sys.argv[2:6]
    paths = sensors()
    missing = [b for b in buses if not any(k.startswith(b) for k in paths)]
    if missing:
        print("tempwatch: no sensor for", missing, file=sys.stderr)
        return 2
    key = {b: next(k for k in paths if k.startswith(b)) for b in buses}
    hist = {b: collections.deque(maxlen=5) for b in buses}
    tick = 0
    with open(log1, "a", buffering=1) as f1, open(logall, "a", buffering=1) as fa:
        while True:
            now = time.strftime("%H:%M:%S")
            line = now
            breach = None
            for b in buses:
                t = read(paths[key[b]])
                hist[b].append(t)
                line += f" {b}={t:.0f}"
                if t >= 95:
                    breach = f"{b} sample {t:.0f} C >= 95"
                if len(hist[b]) == 5 and sum(hist[b]) / 5 >= 90:
                    breach = f"{b} 5 s mean {sum(hist[b]) / 5:.1f} C >= 90"
            f1.write(line + "\n")
            if tick % 15 == 0:
                cols = "".join(f" {k}={read(paths[k]):.0f}" for k in sorted(paths))
                fa.write(now + cols + "\n")
            if breach:
                Path(abortfile).write_text(
                    f"{time.strftime('%F %T')} THERMAL ABORT: {breach}\n"
                )
                try:
                    pgid = int(Path(pidfile).read_text().strip())
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(15)
                    os.killpg(pgid, signal.SIGKILL)
                except (OSError, ValueError):
                    pass
                return 3
            tick += 1
            time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
