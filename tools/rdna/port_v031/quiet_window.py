# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wait for a quiet GPU window before a boot or weight load.

    python tools/rdna/port_v031/quiet_window.py --name TwoLaneFeat

A boot or a weight load evicts other processes' GPU queues (measured: a TP2
boot cost MonoW4A8's TP4 workers +7.1 s of KFD evictions). Before starting
one:

1. wait until no file in the quiet dir is newer than --min-age seconds,
2. touch ``<quiet dir>/<name>``,
3. re-check that no other file is newer than --min-age (a sibling may have
   started in between).

Then take the boot lock and boot. The own file is ignored by the checks.
"""

import argparse
import os
import subprocess
import sys
import time


def _newest_other(directory: str, name: str, since: float) -> tuple[float, str]:
    newest = (since, "")
    for entry in os.scandir(directory):
        if entry.name == name or not entry.is_file():
            continue
        mtime = entry.stat().st_mtime
        if mtime > newest[0]:
            newest = (mtime, entry.name)
    return newest


def _boot_lock_free(args: argparse.Namespace, start: float) -> bool:
    """Wait until `flock -n <boot-lock> true` succeeds (no boot in flight)."""
    while time.time() - start <= args.timeout:
        rc = subprocess.run(
            ["flock", "-n", os.path.expanduser(args.boot_lock), "true"],
            check=False,
        ).returncode
        if rc == 0:
            return True
        time.sleep(args.poll)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default=os.environ.get("W4A8_QUIET_DIR")
        or os.path.expanduser("~/w4a8_runs/lanes/quiet"),
    )
    ap.add_argument(
        "--name",
        default=os.environ.get("W4A8_QUIET_NAME") or "unowned",
    )
    ap.add_argument("--min-age", type=float, default=360.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--poll", type=float, default=10.0)
    # A booter can take the boot lock just before the quiet file appears, so
    # the measurer also waits until `flock -n <lock> true` succeeds and
    # re-checks that before each measured cell (a boot on another slot still
    # pushes host memory and evicts running queues).
    ap.add_argument("--boot-lock", default="")
    args = ap.parse_args()

    directory = os.path.abspath(os.path.expanduser(args.dir))
    os.makedirs(directory, exist_ok=True)
    start = time.time()
    while True:
        now = time.time()
        newest, owner = _newest_other(directory, args.name, now)
        if not owner or now - args.min_age >= newest:
            break
        if now - start > args.timeout:
            print(
                f"quiet window: {owner} still active after "
                f"{now - start:.0f}s; giving up",
                file=sys.stderr,
            )
            return 1
        time.sleep(args.poll)
    own = os.path.join(directory, args.name)
    with open(own, "a"):
        os.utime(own, None)
    # Re-check after taking the window: a sibling can start between the scan
    # and the touch.
    for _ in range(2):
        now = time.time()
        newest, owner = _newest_other(directory, args.name, now)
        if not owner or now - args.min_age >= newest:
            if args.boot_lock and not _boot_lock_free(args, start):
                print(
                    f"quiet window: {args.name} lost the boot-lock race",
                    file=sys.stderr,
                )
                return 1
            print(f"quiet window: clear, {args.name} owns it")
            return 0
        if now - start > args.timeout:
            break
        time.sleep(args.poll)
    print(
        f"quiet window: {owner} started under {args.name}; giving up", file=sys.stderr
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
