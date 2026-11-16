# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check every file:line citation in this explore README against the base tree.

The README cites specific lines in the fork's source.  Those lines move as
the tree moves, so this script resolves every citation against a base commit
and prints the cited source, which makes a stale reference obvious.

Usage:

    python3 docs/explore/kv-int8-native/check_citations.py

The base commit defaults to ``origin/rdna_extras``; override it with
``VLLM_BASE_REF``.  Exit status is 1 if a cited file or line is missing.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_REF = os.environ.get("VLLM_BASE_REF", "origin/rdna_extras")
DOC = Path(__file__).with_name("README.md")

#: Source file extensions a citation may point at.
EXTENSIONS = (".py", ".cu", ".cuh", ".cpp", ".h", ".md")


def base_lines(path: str) -> list[str]:
    """Return the lines of *path* at the base commit.

    Args:
        path: Repository-relative path to a file in the base tree.

    Returns:
        The file's lines, or an empty list if it does not exist there.

    """
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{BASE_REF}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return result.stdout.splitlines()


def citations(text: str) -> list[tuple[str, int, int]]:
    """Return every ``path:line`` citation in *text*.

    Only code spans are considered, so a citation in prose is not picked up,
    and only paths that end in a source file extension count.

    Args:
        text: The document to scan.

    Returns:
        One (path, first line, last line) tuple per citation, in the order
        they appear.

    """
    found: list[tuple[str, int, int]] = []
    for span in text.split("`")[1::2]:
        for token in span.replace("(", " ").replace(")", " ").split():
            path, _, rest = token.partition(":")
            if not path.endswith(EXTENSIONS) or not rest:
                continue
            head, _, tail = rest.partition("-")
            if not head.isdigit():
                continue
            start = int(head)
            digits = "".join(char for char in tail if char.isdigit())
            end = int(digits) if digits else start
            found.append((path, start, end))
    return found


def main() -> int:
    """Resolve every citation in the README and report the ones that fail.

    Returns:
        0 if every citation resolves, 1 otherwise.

    """
    text = DOC.read_text(encoding="utf-8")
    seen: set[tuple[str, int, int]] = set()
    missing = 0
    for path, start, end in citations(text):
        if (path, start, end) in seen:
            continue
        seen.add((path, start, end))
        lines = base_lines(path)
        if not lines:
            print(f"MISSING FILE  {path}")
            missing += 1
            continue
        if end > len(lines):
            print(f"MISSING LINE  {path}:{start}-{end} (file has {len(lines)})")
            missing += 1
            continue
        quoted = " / ".join(line.strip() for line in lines[start - 1 : end])
        print(f"{path}:{start}-{end}\n    {quoted[:400]}")
    print(f"\n{len(seen)} citations checked against {BASE_REF}, {missing} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
