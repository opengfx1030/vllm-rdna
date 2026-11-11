#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Insert arm lines in front of the unread part of a running queue (python3).

    queue_insert.py QUEUE_FILE LINES_FILE

queue.sh reads QUEUE_FILE through one open descriptor, so the file must keep its inode
and every byte before that descriptor's offset (the lines already read, including the
arm that runs now). The script finds the reading process through /proc, takes the
offset from /proc/<pid>/fdinfo/<fd> and rewrites what follows in place: the new lines
come first, the old unread lines after them. The new lines run right after the
current arm. To edit lines that are still unread, read the tail, change it and write
it back the same way (queue_insert.py with LINES_FILE = the new tail and `--replace`).
"""

import os
import sys
from pathlib import Path


def reader_offset(qfile: Path) -> int:
    ino = qfile.stat().st_ino
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if b"queue.sh" not in Path(f"/proc/{pid}/cmdline").read_bytes():
                continue
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if os.stat(f"/proc/{pid}/fd/{fd}").st_ino != ino:
                    continue
                for line in Path(f"/proc/{pid}/fdinfo/{fd}").read_text().splitlines():
                    if line.startswith("pos:"):
                        return int(line.split()[1])
        except OSError:
            continue
    raise SystemExit(f"no queue.sh process has {qfile} open")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--replace"]
    replace = "--replace" in sys.argv[1:]
    qfile, lines_file = Path(args[0]), Path(args[1])
    new = lines_file.read_bytes()
    if not new.endswith(b"\n"):
        new += b"\n"
    pos = reader_offset(qfile)
    with open(qfile, "r+b") as f:
        f.seek(pos)
        tail = b"" if replace else f.read()
        f.seek(pos)
        f.write(new + tail)
        f.truncate()
    print(f"{qfile.name}: {new.count(bytes([10]))} line(s) written at byte {pos}")


if __name__ == "__main__":
    main()
