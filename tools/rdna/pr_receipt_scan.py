#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""List the added lines that match host-sync and kernel-launch patterns.

A reviewer aid for pr_receipt.sh; it never decides pass or fail.

    pr_receipt_scan.py BASE_TREE HEAD_TREE [--json PATH] [--md PATH]
                       [--max-lines N]

Both trees are directories (checkouts or built trees). The files compared by
content are ``vllm/**/*.py`` and ``csrc/**/*`` with a C, C++ or HIP suffix;
``__pycache__`` and dot-directories are skipped. An added line is a line of
the HEAD file that is not covered by a matching block of
``difflib.SequenceMatcher`` against the BASE file; a file that only exists in
HEAD is all added lines.

Python lines (after dropping a trailing comment) are matched against
``.synchronize(``, ``.item()``, ``.cpu()``, ``.tolist()``, ``.to("cpu")`` and
``non_blocking=False``. Docstring bodies are not recognised, so a docstring
line that spells one of these is listed too. C/C++/HIP lines are matched
against the blocking runtime calls; kernel launches (``<<<...>>>`` and
``hipLaunchKernelGGL``) are analysed on the whole HEAD file and reported when
one of their lines was added, for a missing stream (``null-stream``) and a
missing launch check (``no-launch-check``).

The markdown fragment has no heading and is also printed to stdout. Exit code
0 means the scan ran (findings or not), 2 means bad arguments or an
unreadable tree. Standard library and ``regex`` only.
"""

import argparse
import difflib
import json
import os
import sys
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import regex as re

C_SUFFIXES = frozenset(
    {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hip"}
)
SOURCE_ROOTS: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("vllm", "python", frozenset({".py"})),
    ("csrc", "csrc", C_SUFFIXES),
)

# vllm/ paths of the per-step path: model forward, attention/MoE/collective,
# sampler, scheduler, worker step. Every csrc/ file counts as per-step.
STEP_PATH_PREFIXES = (
    "vllm/model_executor/",
    "vllm/attention/",
    "vllm/v1/attention/",
    "vllm/v1/sample/",
    "vllm/v1/core/",
    "vllm/v1/worker/",
    "vllm/v1/spec_decode/",
    "vllm/v1/executor/",
    "vllm/distributed/",
    "vllm/forward_context.py",
    "vllm/compilation/",
)

PYTHON_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("synchronize", re.compile(r"\bsynchronize\s*\(")),
    ("item", re.compile(r"\.item\s*\(")),
    ("cpu", re.compile(r"\.cpu\s*\(")),
    ("tolist", re.compile(r"\.tolist\s*\(")),
    (
        "to-cpu",
        re.compile(
            r"""\.to\(\s*(?:device\s*=\s*)?"""
            r"""(?:["']cpu["']|torch\.device\(\s*["']cpu["']\s*\))"""
        ),
    ),
    ("blocking-copy", re.compile(r"non_blocking\s*=\s*False")),
)

C_HOST_SYNC = re.compile(
    r"\b(?:cuda|hip)(?:Device|Stream|Event)Synchronize\s*\("
    r"|\b(?:cuda|hip)Memcpy\s*\("
)
LAUNCH_CHECKS = (
    "C10_CUDA_KERNEL_LAUNCH_CHECK",
    "C10_HIP_KERNEL_LAUNCH_CHECK",
    "hipGetLastError",
    "cudaGetLastError",
    "C10_CUDA_CHECK(",
    "HIP_CHECK(",
    "CUDA_CHECK(",
)
CHEVRON_NULL_STREAMS = frozenset({"0", "nullptr", "NULL", "hipStreamDefault"})
GGL_NULL_STREAMS = frozenset({"0", "nullptr", "NULL"})
CHECK_WINDOW_LINES = 12
STATEMENT_WINDOW_LINES = 40
PAREN_SCAN_CHARS = 20000
TEXT_LIMIT = 160
DEFAULT_MAX_LINES = 80

_CHEVRON = re.compile(r"<<<(.*?)>>>", re.DOTALL)
_GGL = re.compile(r"\bhipLaunchKernelGGL\s*\(")
_C_TOKEN = re.compile(r"""//|/\*|"|'""")
_C_CHAR = re.compile(r"""'(?:\\[^\n]{1,10}?|[^'\\\n])'""")
_NOT_NEWLINE = re.compile(r"[^\n]")
_CONTINUATION = re.compile(r"\\\r?\n")


@dataclass(frozen=True)
class Finding:
    """One added line (or launch) that matches a pattern."""

    path: str
    line: int
    kind: str
    pattern: str
    text: str
    step: bool

    def sort_key(self) -> tuple[bool, str, int, str]:
        """Return the report order: per-step first, then path, line, pattern."""
        return (not self.step, self.path, self.line, self.pattern)


@dataclass(frozen=True)
class ScanResult:
    """Findings of one scan plus the number of changed files per tree root."""

    base: str
    head: str
    files_changed: dict[str, int]
    findings: list[Finding]


@dataclass(frozen=True)
class Launch:
    """A kernel launch: statement span (offsets into the file) and arguments."""

    start: int
    end: int
    args: list[str]
    null_stream: bool


# ----------------------------------------------------------------- text helpers


def split_lines(text: str) -> list[str]:
    """Split on newlines only, without a trailing empty entry."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def clip(text: str, limit: int = TEXT_LIMIT) -> str:
    """Cut ``text`` to at most ``limit`` characters."""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def added_lines(base: Sequence[str], head: Sequence[str]) -> list[int]:
    """Return the 1-based HEAD line numbers not covered by a matching block."""
    matcher = difflib.SequenceMatcher(None, base, head, autojunk=False)
    added: list[int] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            added.extend(range(j1 + 1, j2 + 1))
    return added


def is_step_path(path: str) -> bool:
    """Tell whether a ``vllm/`` or ``csrc/`` path is in the per-step path."""
    return path.startswith("csrc/") or path.startswith(STEP_PATH_PREFIXES)


# ----------------------------------------------------------------------- python


def strip_py_comment(line: str) -> str:
    """Drop a trailing ``# comment`` that is outside a string literal."""
    quote = ""
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if line.startswith(quote, i):
                i += len(quote)
                quote = ""
                continue
        elif ch == "#":
            return line[:i]
        elif ch in "\"'":
            quote = ch * 3 if line.startswith(ch * 3, i) else ch
            i += len(quote)
            continue
        i += 1
    return line


def python_patterns(line: str) -> list[str]:
    """Return the names of the Python patterns that match one source line."""
    code = strip_py_comment(line)
    if not code.strip():
        return []
    return [name for name, regex in PYTHON_PATTERNS if regex.search(code)]


def scan_python(
    lines: Sequence[str], added: Sequence[int]
) -> list[tuple[int, str, str]]:
    """Return ``(line, pattern, text)`` for the added Python lines that match."""
    found = []
    for number in added:
        line = lines[number - 1]
        for name in python_patterns(line):
            found.append((number, name, clip(line.strip())))
    return found


# -------------------------------------------------------------------------- C/C++


def _blank(segment: str) -> str:
    return _NOT_NEWLINE.sub(" ", segment)


def mask_c(text: str) -> str:
    """Blank comments and string contents, keeping offsets and newlines."""
    pieces: list[str] = []
    last = 0
    pos = 0
    n = len(text)
    while True:
        m = _C_TOKEN.search(text, pos)
        if m is None:
            break
        i = m.start()
        token = m.group()
        if token == "//":
            j = i
            while j < n:
                if text[j] == "\n":
                    k = j - 1
                    if k > i and text[k] == "\r":
                        k -= 1
                    if k >= i and text[k] == "\\":
                        j += 1
                        continue
                    break
                j += 1
            pieces.append(text[last:i])
            pieces.append(_blank(text[i:j]))
            last = pos = j
        elif token == "/*":
            close = text.find("*/", i + 2)
            j = n if close < 0 else close + 2
            pieces.append(text[last:i])
            pieces.append(_blank(text[i:j]))
            last = pos = j
        elif token == '"':
            j = i + 1
            while j < n and text[j] not in '"\n':
                j += 2 if text[j] == "\\" else 1
            j = min(j, n)
            pieces.append(text[last : i + 1])
            pieces.append(_blank(text[i + 1 : j]))
            last = j
            pos = j + 1
        else:
            literal = _C_CHAR.match(text, i)
            if literal is None:
                pos = i + 1
                continue
            pieces.append(text[last : i + 1])
            pieces.append(_blank(text[i + 1 : literal.end() - 1]))
            last = literal.end() - 1
            pos = literal.end()
    pieces.append(text[last:])
    return "".join(pieces)


def split_params(config: str) -> list[str]:
    """Split a launch configuration at its top-level commas.

    Brackets nest; a ``<`` opens a template angle level only when it directly
    follows an identifier character, and a ``>`` closes one only while one is
    open (``->``, ``>=`` and ``>>=`` never do).
    """
    params: list[str] = []
    current: list[str] = []
    saved: list[int] = []
    angle = 0
    n = len(config)
    for i, ch in enumerate(config):
        prev = config[i - 1] if i else ""
        nxt = config[i + 1] if i + 1 < n else ""
        if ch in "([{":
            saved.append(angle)
            angle = 0
        elif ch in ")]}":
            if saved:
                angle = saved.pop()
        elif ch == "<":
            if (prev.isalnum() or prev == "_") and nxt not in ("<", "="):
                angle += 1
        elif ch == ">":
            if angle and prev != "-" and nxt != "=":
                angle -= 1
        elif ch == "," and not saved and angle == 0:
            params.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    last = "".join(current).strip()
    if params or last:
        params.append(last)
    return params


class _LineIndex:
    """Offset to 1-based line number mapping for one text."""

    def __init__(self, text: str) -> None:
        self.size = len(text)
        self.starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                self.starts.append(i + 1)

    def line(self, offset: int) -> int:
        """Return the 1-based line holding ``offset``."""
        return bisect_right(self.starts, offset)

    def after_lines(self, line: int, count: int) -> int:
        """Return the offset where line ``line + count + 1`` starts."""
        index = line + count
        return self.starts[index] if index < len(self.starts) else self.size


def _launch_start(masked: str, pos: int) -> int:
    """Return the offset of the kernel name in front of ``<<<`` at ``pos``."""
    floor = max(0, pos - 300)
    i = pos
    while i > floor and masked[i - 1].isspace():
        i -= 1
    if i > floor and masked[i - 1] == ">":
        depth = 0
        j = i
        while j > floor:
            j -= 1
            if masked[j] == ">":
                depth += 1
            elif masked[j] == "<":
                depth -= 1
                if depth == 0:
                    break
        else:
            return pos
        i = j
    while i > floor and (masked[i - 1].isalnum() or masked[i - 1] in "_:"):
        i -= 1
    while i < pos and masked[i].isspace():
        i += 1
    return i


def _statement_end(masked: str, after: int, index: _LineIndex) -> int:
    """Return the offset past the first ``;`` outside parentheses, or ``after``."""
    limit = index.after_lines(index.line(max(after - 1, 0)), STATEMENT_WINDOW_LINES)
    depth = 0
    for i in range(after, limit):
        ch = masked[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == ";" and depth == 0:
            return i + 1
    return after


def _matching_paren(masked: str, open_at: int) -> int:
    """Return the offset of the bracket closing the one at ``open_at``, or -1."""
    depth = 0
    for i in range(open_at, min(len(masked), open_at + PAREN_SCAN_CHARS)):
        ch = masked[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def find_launches(masked: str, index: _LineIndex) -> list[Launch]:
    """Find the ``<<<...>>>`` and ``hipLaunchKernelGGL`` launches of a file."""
    launches = []
    for m in _CHEVRON.finditer(masked):
        args = split_params(m.group(1))
        null = len(args) < 4 or args[3] in CHEVRON_NULL_STREAMS
        end = _statement_end(masked, m.end(), index)
        launches.append(Launch(_launch_start(masked, m.start()), end, args, null))
    for m in _GGL.finditer(masked):
        close = _matching_paren(masked, m.end() - 1)
        if close < 0:
            continue
        args = split_params(masked[m.end() : close])
        null = len(args) < 5 or args[4] in GGL_NULL_STREAMS
        end = _statement_end(masked, close + 1, index)
        launches.append(Launch(m.start(), end, args, null))
    return launches


def scan_c(text: str, added: Sequence[int]) -> list[tuple[int, str, str]]:
    """Return ``(line, pattern, text)`` for the added C/C++/HIP lines that match."""
    lines = split_lines(text)
    masked = mask_c(text)
    masked_lines = masked.split("\n")
    added_set = set(added)
    found: list[tuple[int, str, str]] = []
    for number in sorted(added_set):
        if number <= len(masked_lines) and C_HOST_SYNC.search(masked_lines[number - 1]):
            found.append((number, "host-sync", clip(lines[number - 1].strip())))
    index = _LineIndex(masked)
    for launch in find_launches(masked, index):
        first = index.line(launch.start)
        last = index.line(launch.end - 1)
        if added_set.isdisjoint(range(first, last + 1)):
            continue
        statement = _CONTINUATION.sub(" ", text[launch.start : launch.end])
        shown = clip(" ".join(statement.split()))
        if launch.null_stream:
            found.append((first, "null-stream", shown))
        window = masked[launch.end : index.after_lines(last, CHECK_WINDOW_LINES)]
        if not any(check in window for check in LAUNCH_CHECKS):
            found.append((first, "no-launch-check", shown))
    return found


# -------------------------------------------------------------------- tree scan


def list_sources(tree: Path) -> dict[str, str]:
    """Map the relative posix path of each source file of ``tree`` to its kind."""
    found: dict[str, str] = {}
    for top, kind, suffixes in SOURCE_ROOTS:
        root = tree / top
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames if d != "__pycache__" and not d.startswith(".")
            )
            for name in filenames:
                if os.path.splitext(name)[1] in suffixes:
                    rel = Path(dirpath, name).relative_to(tree).as_posix()
                    found[rel] = kind
    return found


def _read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="replace")


def _same_bytes(a: Path, b: Path) -> bool:
    return a.stat().st_size == b.stat().st_size and a.read_bytes() == b.read_bytes()


def scan_trees(base: Path, head: Path) -> ScanResult:
    """Scan the added lines of HEAD against BASE; raise OSError if unreadable."""
    base_files = list_sources(base)
    head_files = list_sources(head)
    changed = {"vllm": 0, "csrc": 0}
    findings: list[Finding] = []
    for rel in sorted(base_files.keys() | head_files.keys()):
        kind = head_files.get(rel) or base_files[rel]
        top = rel.split("/", 1)[0]
        if rel not in head_files:
            changed[top] += 1
            continue
        base_lines: list[str] = []
        if rel in base_files:
            if _same_bytes(base / rel, head / rel):
                continue
            base_lines = split_lines(_read_text(base / rel))
        changed[top] += 1
        text = _read_text(head / rel)
        head_lines = split_lines(text)
        added = added_lines(base_lines, head_lines)
        if not added:
            continue
        if kind == "python":
            hits = scan_python(head_lines, added)
        else:
            hits = scan_c(text, added)
        step = is_step_path(rel)
        findings.extend(
            Finding(rel, n, kind, name, shown, step) for n, name, shown in hits
        )
    findings.sort(key=Finding.sort_key)
    return ScanResult(str(base), str(head), changed, findings)


# ----------------------------------------------------------------------- reports


def _summary(result: ScanResult) -> str:
    files = (
        f"({result.files_changed['vllm']} vllm/ and "
        f"{result.files_changed['csrc']} csrc/ files changed)"
    )
    if not result.findings:
        return f"No added line matches the host-sync or launch patterns {files}."
    lines = {(f.path, f.line) for f in result.findings}
    step_lines = {(f.path, f.line) for f in result.findings if f.step}
    counts = pattern_counts(result.findings)
    parts = ", ".join(
        f"{name} x {count}"
        for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    return (
        f"{len(lines)} added lines to review ({len(step_lines)} in the per-step "
        f"path): {parts} {files}."
    )


def pattern_counts(findings: Sequence[Finding]) -> dict[str, int]:
    """Count the findings per pattern name."""
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.pattern] = counts.get(f.pattern, 0) + 1
    return counts


def code_span(text: str) -> str:
    """Render ``text`` as a markdown code span that is safe inside a table cell."""
    text = " ".join(text.split()).replace("|", "\\|")
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def render_markdown(result: ScanResult, max_lines: int = DEFAULT_MAX_LINES) -> str:
    """Render the markdown fragment (no heading) for a scan result."""
    out = [_summary(result), ""]
    if result.findings:
        out += ["| where | pattern | code |", "| --- | --- | --- |"]
        for f in result.findings[:max_lines]:
            where = code_span(f"{f.path}:{f.line}")
            out.append(f"| {where} | {f.pattern} | {code_span(f.text)} |")
        hidden = len(result.findings) - max_lines
        if hidden > 0:
            out += ["", f"{hidden} more in the JSON file."]
    return "\n".join(out).rstrip("\n") + "\n"


def to_json(
    result: ScanResult, max_lines: int = DEFAULT_MAX_LINES
) -> dict[str, object]:
    """Return the JSON document of a scan result.

    ``step_count`` is the number of distinct per-step added lines with a
    finding; ``truncated`` is the number of findings the markdown leaves out.
    """
    return {
        "base": result.base,
        "head": result.head,
        "files_changed": result.files_changed,
        "findings": [asdict(f) for f in result.findings],
        "counts": pattern_counts(result.findings),
        "step_count": len({(f.path, f.line) for f in result.findings if f.step}),
        "truncated": max(0, len(result.findings) - max_lines),
    }


def _non_negative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scan; return 0 on success and 2 on bad input."""
    parser = argparse.ArgumentParser(
        description="List added lines that match host-sync and launch patterns."
    )
    parser.add_argument("base", metavar="BASE_TREE", type=Path)
    parser.add_argument("head", metavar="HEAD_TREE", type=Path)
    parser.add_argument("--json", dest="json_path", metavar="PATH", type=Path)
    parser.add_argument("--md", dest="md_path", metavar="PATH", type=Path)
    parser.add_argument(
        "--max-lines", type=_non_negative, default=DEFAULT_MAX_LINES, metavar="N"
    )
    args = parser.parse_args(argv)
    for tree in (args.base, args.head):
        if not tree.is_dir():
            print(f"pr_receipt_scan: not a directory: {tree}", file=sys.stderr)
            return 2
    try:
        result = scan_trees(args.base, args.head)
        markdown = render_markdown(result, args.max_lines)
        if args.json_path:
            document = json.dumps(to_json(result, args.max_lines), indent=2)
            args.json_path.write_text(document + "\n")
        if args.md_path:
            args.md_path.write_text(markdown)
    except OSError as exc:
        print(f"pr_receipt_scan: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
