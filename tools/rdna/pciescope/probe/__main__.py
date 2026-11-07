# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``python -m pciescope.probe build|status``: manage the probe binary."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .binary import ARCHS, LOCK_WAIT_S, binary_status, build
from .runner import default_binary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pciescope.probe",
        description="Build or inspect the p2p_probe binary of `pciescope measure`.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser(
        "build",
        help="compile the fat binary under ~/w4a8_runs/audit/build.lock",
    )
    b.add_argument("--out", default=None, help="binary path (default: the stable path)")
    b.add_argument(
        "--archs",
        nargs="+",
        default=list(ARCHS),
        help="offload architectures (default: %(default)s)",
    )
    b.add_argument(
        "--lock-wait",
        type=int,
        default=LOCK_WAIT_S,
        help="seconds to wait for the build lock (default: %(default)s)",
    )
    s = sub.add_parser("status", help="path, version and build stamp of the binary")
    s.add_argument("--binary", default=None)
    for p in (b, s):
        p.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.cmd == "build":
        out = Path(args.out).expanduser() if args.out else default_binary()
        result = build(out=out, archs=args.archs, lock_wait_s=args.lock_wait)
        if args.json:
            print(json.dumps(result.to_dict()))
        else:
            print(f"{'built' if result.ok else 'FAILED'}: {result.message}")
            print(f"  binary: {result.binary}  ({result.seconds:.0f} s)")
            for line in result.log_tail if not result.ok else []:
                print(f"  | {line}")
        return 0 if result.ok else 1
    status = binary_status(Path(args.binary).expanduser() if args.binary else None)
    if args.json:
        print(json.dumps(status.to_dict()))
    else:
        print(f"binary:  {status.path}")
        print(f"exists:  {status.exists}   runnable: {status.runnable}")
        if status.version:
            print(f"version: {status.version}  modes: {','.join(status.modes)}")
        if status.stamp:
            print(f"stamp:   {json.dumps(status.stamp)}")
        if status.stale is not None:
            print(f"stale:   {status.stale} (source differs from the build)")
        if status.problem:
            print(f"problem: {status.problem}")
    return 0 if status.runnable else 1


if __name__ == "__main__":
    sys.exit(main())
