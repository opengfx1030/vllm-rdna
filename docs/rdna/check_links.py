#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check relative links and ``#anchors`` in the repo's tracked markdown files.

Usage (from anywhere inside the checkout)::

    python3 docs/rdna/check_links.py              # every tracked *.md
    python3 docs/rdna/check_links.py --fork-only  # fork-owned docs only

Only tracked files count as link targets, so a link that resolves in a dirty
working tree but not in a fresh clone is reported. External URLs
(``http(s)://``, ``mailto:``) and links inside fenced or inline code are
skipped. Anchors are only validated for markdown targets and follow GitHub's
heading slug rules. Exits 1 when at least one link is broken.
"""

import argparse
import subprocess
import sys
from pathlib import PurePosixPath

import regex as re

# Paths (files or directory prefixes) owned by the RDNA fork.
FORK_PATHS = (
    "docs/rdna/",
    "docs/rdna2/",
    "docs/explore/",
    "README.md",
)
FORK_README_GLOB = "tools/"  # only README.md files below this prefix

FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
HTML_ANCHOR_RE = re.compile(r"""<a\s+[^>]*?(?:id|name)=["']([^"']+)["']""")
INLINE_CODE_RE = re.compile(r"(`+)(?:(?!\1).)+?\1")
# [text](target "title") -- target may be <wrapped> or contain balanced ().
INLINE_LINK_RE = re.compile(
    r"\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*"
    r"(<[^>]*>|(?:[^\s()]|\((?:[^\s()])*\))+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)
# [label]: target "optional title" -- prose such as "[INFERENCE]: the race ..."
# is not a definition because the tail is not a title.
REF_DEF_RE = re.compile(
    r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]*>|\S+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*$"
)
EXTERNAL_RE = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|//)")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def github_slug(heading: str) -> str:
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # links -> text
    text = re.sub(r"<[^>]+>", "", text)  # inline html
    text = text.replace("`", "").replace("*", "")
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def iter_prose_lines(text: str):
    """Yield (lineno, line) for lines outside fenced code blocks."""
    fence = None
    for lineno, line in enumerate(text.splitlines(), 1):
        m = FENCE_RE.match(line)
        if m:
            marker = m.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            continue
        if fence is None:
            yield lineno, line


def collect_anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for _, line in iter_prose_lines(text):
        m = HEADING_RE.match(line)
        if m:
            slug = github_slug(m.group(1))
            n = seen.get(slug, 0)
            seen[slug] = n + 1
            anchors.add(slug if n == 0 else f"{slug}-{n}")
        anchors.update(HTML_ANCHOR_RE.findall(line))
    return anchors


def is_fork_doc(path: str) -> bool:
    if path.startswith(FORK_PATHS[:3]) or path == FORK_PATHS[3]:
        return True
    return path.startswith(FORK_README_GLOB) and path.endswith("/README.md")


def extract_targets(text: str):
    for lineno, line in iter_prose_lines(text):
        stripped = INLINE_CODE_RE.sub("", line)
        for m in INLINE_LINK_RE.finditer(stripped):
            yield lineno, m.group(1)
        m = REF_DEF_RE.match(stripped)
        if m:
            yield lineno, m.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--fork-only",
        action="store_true",
        help="limit to docs/rdna, docs/rdna2, docs/explore, README.md and "
        "tools/**/README.md",
    )
    args = parser.parse_args()

    root = git("rev-parse", "--show-toplevel").strip()
    tracked = set(git("-C", root, "ls-files").split("\n")) - {""}
    dirs = {
        str(parent)
        for path in tracked
        for parent in PurePosixPath(path).parents
        if str(parent) != "."
    }
    md_files = sorted(p for p in tracked if p.endswith(".md"))
    if args.fork_only:
        md_files = [p for p in md_files if is_fork_doc(p)]

    texts: dict[str, str] = {}
    anchor_cache: dict[str, set[str]] = {}

    def read(path: str) -> str:
        if path not in texts:
            with open(f"{root}/{path}", encoding="utf-8", errors="replace") as f:
                texts[path] = f.read()
        return texts[path]

    failures: list[str] = []
    n_links = 0
    for src in md_files:
        for lineno, raw in extract_targets(read(src)):
            target = raw.strip("<>")
            if not target or EXTERNAL_RE.match(target):
                continue
            n_links += 1
            path_part, _, anchor = target.partition("#")
            path_part = path_part.partition("?")[0]
            if path_part.startswith("/"):
                resolved = PurePosixPath(path_part.lstrip("/"))
            elif path_part:
                resolved = PurePosixPath(src).parent / path_part
            else:
                resolved = PurePosixPath(src)
            norm = _normalize(resolved)
            where = f"{src}:{lineno}"
            if norm is None or (norm not in tracked and norm not in dirs):
                failures.append(f"{where}: missing target: {target}")
                continue
            if anchor and norm.endswith(".md") and norm in tracked:
                if norm not in anchor_cache:
                    anchor_cache[norm] = collect_anchors(read(norm))
                if anchor.lower() not in anchor_cache[norm]:
                    failures.append(f"{where}: missing anchor: {target}")

    for line in failures:
        print(line)
    print(
        f"checked {len(md_files)} files, {n_links} relative links, "
        f"{len(failures)} failures",
        file=sys.stderr,
    )
    return 1 if failures else 0


def _normalize(path: PurePosixPath) -> str | None:
    parts: list[str] = []
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None  # escapes the repository
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


if __name__ == "__main__":
    sys.exit(main())
