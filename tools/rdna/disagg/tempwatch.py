#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Junction watcher for one run (stdlib only).

  tempwatch.py BUS,BUS,... LOG1S LOGALL PGID_FILE ABORTFILE

LOG1S : the run's GPUs every second  ("HH:MM:SS bus=degC ...")
LOGALL: every amdgpu every 15 s
Abort rule (box rules): any watched GPU's 5 s mean >= 90 C or any sample >= 95 C
-> write ABORTFILE and SIGTERM (then SIGKILL) the process group in PGID_FILE.
"""

import collections
import contextlib
import os
import signal
import sys
import time
from pathlib import Path


def sensors() -> dict[str, str]:
    out = {}
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (h / "name").read_text().strip() != "amdgpu":
                continue
            bus = os.path.basename(os.path.realpath(h / "device"))
            out[bus.removeprefix("0000:")] = str(h / "temp2_input")
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
    key: dict[str, str] = {}
    for b in buses:
        match = next((k for k in paths if k.startswith(b)), None)
        if match is None:
            print("tempwatch: no sensor for", b, file=sys.stderr)
            return 2
        key[b] = paths[match]
    hist = {b: collections.deque(maxlen=5) for b in buses}
    tick = 0
    with open(log1, "a", buffering=1) as f1, open(logall, "a", buffering=1) as fa:
        while True:
            now = time.strftime("%H:%M:%S")
            line = now
            breach = None
            for b in buses:
                t = read(key[b])
                hist[b].append(t)
                line += f" {b}={t:.0f}"
                if t >= 95:
                    breach = f"{b} sample {t:.0f} C >= 95"
                if len(hist[b]) == 5 and sum(hist[b]) / 5 >= 90:
                    breach = f"{b} 5 s mean {sum(hist[b]) / 5:.1f} C >= 90"
            f1.write(line + "\n")
            if tick % 15 == 0:
                cols = "".join(f" {k}={read(v):.0f}" for k, v in sorted(paths.items()))
                fa.write(now + cols + "\n")
            if breach:
                Path(abortfile).write_text(
                    f"{time.strftime('%F %T')} THERMAL ABORT: {breach}\n"
                )
                with contextlib.suppress(OSError, ValueError):
                    pgid = int(Path(pidfile).read_text().strip())
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(15)
                    os.killpg(pgid, signal.SIGKILL)
                return 3
            tick += 1
            time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
