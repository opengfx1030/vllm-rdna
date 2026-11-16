#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-arch disassembly diff of the device code of two builds of the same csrc files.

``tools/rdna/archdiff.sh`` builds the csrc translation units a change reaches for
every arch of the contract, once from BASE and once from HEAD (same tree path, same
build directory), and calls this helper for two steps:

  plan     which translation units to build: the csrc files that differ, plus every
           translation unit of the ROCm extension targets that includes one of them
           (textual include closure), and the ``__LINE__`` shifts the source diff
           allows inside assert stubs;
  compare  per object and arch: unbundle the device code object, disassemble it,
           normalise, classify every function, write archdiff.md / archdiff.json
           and return the verdict as the exit status (0 nothing unexpected differs,
           1 unexpected difference, 2 error or an arch that was not run).

Everything that decides a verdict (include closure, line shifts, normalisation,
classification, EXPECT / MAY_DIFFER) is plain Python over strings and is unit
tested without ROCm. clang-offload-bundler, llvm-objdump and llvm-cxxfilt (or
c++filt) are called through thin wrappers; the ROCm LLVM directory wins over PATH.
No GPU is touched.
"""

import argparse
import bisect
import dataclasses
import difflib
import json
import os
import posixpath
import shutil
import struct
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import regex as re

EXIT_OK = 0
EXIT_DIFFERS = 1
EXIT_ERROR = 2

DEFAULT_ARCHES = ("gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100", "gfx1200")
# Repo-relative roots whose files can change the device code of a translation unit.
TRACKED_ROOTS = ("csrc", "cmake", "CMakeLists.txt")
CSRC = "csrc"

IDENTICAL = "identical"
MASKED = "identical-after-masking"
LINE_ONLY = "line-constant-only"
CHANGED = "CHANGED"
ONLY_BASE = "only-in-base"
ONLY_HEAD = "only-in-head"
CLASSES = (IDENTICAL, MASKED, LINE_ONLY, CHANGED, ONLY_BASE, ONLY_HEAD)

STATUS_COMPARED = "compared"
STATUS_NO_CODE = "no-code-object"

# At most this many items are listed per group in the markdown report.
LIST_LIMIT = 12


class ToolError(RuntimeError):
    """A ROCm tool is missing or failed, or an input is unusable."""


# --------------------------------------------------------------------------- #
# Source trees and the textual include closure
# --------------------------------------------------------------------------- #

_SKIP_DIRS = frozenset({"__pycache__", ".git"})
_INCLUDE = re.compile(
    r'^[ \t]*#[ \t]*include[ \t]*(?:"([^"\n]+)"|<([^>\n]+)>)', re.MULTILINE
)


class Tree(Protocol):
    """A source tree as the include closure and the change detection read it."""

    def exists(self, rel: str) -> bool: ...

    def read(self, rel: str) -> str: ...

    def files(self, rel: str) -> list[str]: ...


class DirTree:
    """A source tree on disk; paths are repo-relative with forward slashes."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def exists(self, rel: str) -> bool:
        return (self.root / rel).is_file()

    def read(self, rel: str) -> str:
        data = (self.root / rel).read_bytes()
        return data.decode("utf-8", errors="surrogateescape")

    def files(self, rel: str) -> list[str]:
        """Regular files at or below ``rel``, sorted."""
        top = self.root / rel
        if top.is_file():
            return [rel]
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(top):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            here = Path(dirpath).relative_to(self.root).as_posix()
            found.extend(posixpath.join(here, name) for name in filenames)
        return sorted(found)


def parse_includes(text: str) -> list[str]:
    """Names in the ``#include "x"`` and ``#include <x>`` lines of ``text``."""
    return [m.group(1) or m.group(2) for m in _INCLUDE.finditer(text)]


def resolve_include(
    name: str, includer: str, exists: Callable[[str], bool]
) -> str | None:
    """Repo-relative path ``name`` refers to, or None for a system header.

    Quote and angle includes are both tried against the includer's directory and
    then against ``csrc/``; only files under ``csrc/`` count.
    """
    here = posixpath.dirname(includer)
    for base in (here, CSRC):
        candidate = posixpath.normpath(posixpath.join(base, name))
        if candidate.startswith(CSRC + "/") and exists(candidate):
            return candidate
    return None


class IncludeGraph:
    """The textual include graph of one tree, built lazily and cached."""

    def __init__(self, tree: Tree) -> None:
        self.tree = tree
        self._direct: dict[str, tuple[str, ...]] = {}

    def direct(self, rel: str) -> tuple[str, ...]:
        found = self._direct.get(rel)
        if found is None:
            resolved: list[str] = []
            if self.tree.exists(rel):
                for name in parse_includes(self.tree.read(rel)):
                    target = resolve_include(name, rel, self.tree.exists)
                    if target is not None and target not in resolved:
                        resolved.append(target)
            found = tuple(resolved)
            self._direct[rel] = found
        return found

    def closure(self, rel: str) -> set[str]:
        """``rel`` and every file it includes, directly or not."""
        seen = {rel}
        todo = [rel]
        while todo:
            for dep in self.direct(todo.pop()):
                if dep not in seen:
                    seen.add(dep)
                    todo.append(dep)
        return seen


def changed_files(
    base: Tree, head: Tree, roots: Sequence[str] = TRACKED_ROOTS
) -> list[str]:
    """Files under ``roots`` that differ, or exist on one side only, sorted."""
    changed: list[str] = []
    for root in roots:
        for rel in sorted(set(base.files(root)) | set(head.files(root))):
            before = base.read(rel) if base.exists(rel) else None
            after = head.read(rel) if head.exists(rel) else None
            if before != after:
                changed.append(rel)
    return changed


def is_cmake_input(rel: str) -> bool:
    """True for files that can change flags or targets, not a single TU."""
    return rel == "CMakeLists.txt" or rel.startswith("cmake/")


# --------------------------------------------------------------------------- #
# Translation units of the build
# --------------------------------------------------------------------------- #

_OBJECT = re.compile(r"^CMakeFiles/([^/]+)\.dir/")


@dataclasses.dataclass(frozen=True, order=True)
class CompileUnit:
    """One object of the build: the original csrc source and where it lands."""

    source: str  # repo-relative original source, e.g. csrc/rocm/skinny_gemms.cu
    object: str  # object path relative to the build directory
    target: str  # CMake target that owns it


def hip_name(source: str) -> str:
    """Path CMake's hipify step gives a ``.cu`` source (relative to the build dir)."""
    return re.sub(r"\.cu$", ".hip", source).replace("cuda", "hip")


def hip_sources(tree: Tree) -> dict[str, str]:
    """Map hipified build paths of the tree's ``.cu`` files back to the sources."""
    return {hip_name(rel): rel for rel in tree.files(CSRC) if rel.endswith(".cu")}


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _source_of(
    path: str, build_dir: str, tree_dir: str, hip_map: Mapping[str, str]
) -> str | None:
    if _inside(path, build_dir):
        return hip_map.get(os.path.relpath(path, build_dir).replace(os.sep, "/"))
    if _inside(path, tree_dir):
        return os.path.relpath(path, tree_dir).replace(os.sep, "/")
    return None


def units_from_compdb(
    entries: Iterable[Mapping[str, Any]],
    build_dir: str,
    tree_dir: str,
    hip_map: Mapping[str, str],
) -> list[CompileUnit]:
    """Compile units of the csrc sources in a ``ninja -t compdb`` listing."""
    build = os.path.realpath(build_dir)
    tree = os.path.realpath(tree_dir)
    units: set[CompileUnit] = set()
    for entry in entries:
        output = str(entry.get("output", ""))
        file = str(entry.get("file", ""))
        match = _OBJECT.match(output)
        if match is None or not output.endswith(".o") or file in ("", "."):
            continue
        directory = str(entry.get("directory", build))
        path = os.path.realpath(
            file if os.path.isabs(file) else os.path.join(directory, file)
        )
        source = _source_of(path, build, tree, hip_map)
        if source is not None and source.startswith(CSRC + "/"):
            units.add(CompileUnit(source, output, match.group(1)))
    return sorted(units)


def select_units(
    units: Sequence[CompileUnit],
    changed: Collection[str],
    closures: Mapping[str, Collection[str]],
    explicit: Collection[str] = (),
    everything: bool = False,
) -> dict[CompileUnit, list[str]]:
    """The units to build and why.

    ``everything`` takes every unit, ``explicit`` the units of those sources;
    otherwise a unit is taken when its source or anything it includes changed.
    """
    chosen: dict[CompileUnit, list[str]] = {}
    changed_set = set(changed)
    for unit in units:
        if everything:
            chosen[unit] = ["ALL=1"]
        elif explicit:
            if unit.source in explicit:
                chosen[unit] = ["TUS"]
        else:
            hit = sorted(set(closures.get(unit.source, ())) & changed_set)
            if hit:
                chosen[unit] = hit
    return chosen


# --------------------------------------------------------------------------- #
# Source diff and the __LINE__ shifts it allows
# --------------------------------------------------------------------------- #

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclasses.dataclass
class FileDiff:
    """Hunks of one file of a unified diff, reduced to what the report needs."""

    path: str
    hunks: int = 0
    added: int = 0
    removed: int = 0
    # Cumulative new-minus-old line offset after each run of changed lines
    # (zero offsets left out): what an unchanged line below that run moved by.
    shifts: list[int] = dataclasses.field(default_factory=list)


def _diff_path(header: str) -> str:
    path = header[4:].split("\t", 1)[0].strip()
    return path[2:] if path.startswith(("a/", "b/")) else path


def parse_unified_diff(diff: str) -> list[FileDiff]:
    """Parse a unified diff (any context size) into per-file shift data."""
    lines = diff.splitlines()
    files: list[FileDiff] = []
    current: FileDiff | None = None
    offset = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            line.startswith("--- ")
            and i + 1 < len(lines)
            and lines[i + 1].startswith("+++ ")
        ):
            new = _diff_path(lines[i + 1])
            current = FileDiff(_diff_path(line) if new == "/dev/null" else new)
            files.append(current)
            offset = 0
            i += 2
            continue
        hunk = _HUNK.match(line)
        if hunk is None or current is None:
            i += 1
            continue
        old_left = int(hunk.group(2) or 1)
        new_left = int(hunk.group(4) or 1)
        current.hunks += 1
        plus = minus = 0
        i += 1
        while i < len(lines) and (old_left > 0 or new_left > 0):
            body = lines[i]
            if body.startswith("\\"):
                i += 1
                continue
            kind = body[:1]
            if kind in ("", " "):
                if plus or minus:
                    offset += plus - minus
                    if offset:
                        current.shifts.append(offset)
                    plus = minus = 0
                old_left -= 1
                new_left -= 1
            elif kind == "-":
                minus += 1
                current.removed += 1
                old_left -= 1
            elif kind == "+":
                plus += 1
                current.added += 1
                new_left -= 1
            else:
                break
            i += 1
        if plus or minus:
            offset += plus - minus
            if offset:
                current.shifts.append(offset)
    return files


def line_shifts_from_unified_diff(diff: str) -> set[int]:
    """Deltas a ``__LINE__`` constant may have moved by, from a unified diff.

    A line below a run of changed lines moved by the cumulative new-minus-old
    line count of the runs above it. The union over all files is returned.
    """
    shifts: set[int] = set()
    for file in parse_unified_diff(diff):
        shifts.update(file.shifts)
    return shifts


def unified_diff_text(rel: str, old: str, new: str) -> str:
    """Zero-context unified diff of two file contents (one hunk per change run)."""
    a = old.splitlines()
    b = new.splitlines()
    head = 0
    while head < min(len(a), len(b)) and a[head] == b[head]:
        head += 1
    tail = 0
    while tail < min(len(a), len(b)) - head and a[-1 - tail] == b[-1 - tail]:
        tail += 1
    mid_a = a[head : len(a) - tail]
    mid_b = b[head : len(b) - tail]
    if not mid_a and not mid_b:
        return ""
    matcher = difflib.SequenceMatcher(None, mid_a, mid_b, autojunk=False)
    out = [f"--- a/{rel}", f"+++ b/{rel}"]
    for group in matcher.get_grouped_opcodes(0):
        i1, i2 = group[0][1], group[-1][2]
        j1, j2 = group[0][3], group[-1][4]
        old_start = head + i1 + (1 if i2 > i1 else 0)
        new_start = head + j1 + (1 if j2 > j1 else 0)
        out.append(f"@@ -{old_start},{i2 - i1} +{new_start},{j2 - j1} @@")
        for tag, a1, a2, b1, b2 in group:
            if tag in ("replace", "delete"):
                out.extend("-" + line for line in mid_a[a1:a2])
            if tag in ("replace", "insert"):
                out.extend("+" + line for line in mid_b[b1:b2])
    return "\n".join(out) + "\n"


def tree_diff(base: Tree, head: Tree, rels: Iterable[str]) -> str:
    """Unified diff of the listed files between two trees."""
    parts = []
    for rel in rels:
        before = base.read(rel) if base.exists(rel) else ""
        after = head.read(rel) if head.exists(rel) else ""
        parts.append(unified_diff_text(rel, before, after))
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Disassembly: parsing and normalisation
# --------------------------------------------------------------------------- #

_HDR = re.compile(r"^([0-9a-fA-F]+) <(.+)>:$")
_ADDR = re.compile(r"^\s*([0-9a-fA-F]+):")
_LITERAL = re.compile(r"(?<![\w.])-?0x[0-9a-fA-F]+\b")
_GETPC_DST = re.compile(r"^s_getpc_b64\s+s\[(\d+):(\d+)\]")
_ADD_LO = frozenset({"s_add_u32", "s_add_co_u32"})
_ADD_HI = frozenset({"s_addc_u32", "s_add_co_ci_u32"})
_MOV_IMM = re.compile(
    r"^(?:s_movk_i32|s_mov_b32|s_mov_b64|v_mov_b32(?:_e32|_e64)?|v_dual_mov_b32)$"
)
_EMPTY_OPS = frozenset({"s_endpgm", "s_nop", "s_code_end"})
_ASSERT_FAIL = re.compile(r"^__assert_fail(?:$|[.])")
# s_getpc_b64 is followed by its add pair within a few instructions (gfx12 puts
# s_wait_alu and s_sext_i32_i16 in between).
_PAIR_WINDOW = 8
_MASK32 = 0xFFFFFFFF


@dataclasses.dataclass
class Function:
    """One function symbol of a code object, as llvm-objdump prints it."""

    name: str
    addr: int
    insns: list[str] = dataclasses.field(default_factory=list)
    # Index of every s_getpc_b64 in insns, and the address of those that had one.
    getpc: list[int] = dataclasses.field(default_factory=list)
    pcs: dict[int, int] = dataclasses.field(default_factory=dict)
    end: int = 0

    @property
    def empty(self) -> bool:
        """True when the body holds nothing but s_endpgm / s_nop / s_code_end."""
        return all(i.partition(" ")[0] in _EMPTY_OPS for i in self.insns)


def parse_disassembly(lines: Iterable[str]) -> dict[str, Function]:
    """Functions of ``llvm-objdump -d --no-show-raw-insn`` output, by symbol name.

    Trailing ``// address: encoding`` comments are dropped (they carry addresses
    and branch annotations, which move with the layout), and so are the ``...``
    zero-fill markers. Symbols that occur twice keep their order as ``name#2``.
    """
    functions: dict[str, Function] = {}
    current: Function | None = None
    last_comment = ""

    def close() -> None:
        if current is not None:
            found = _ADDR.match(last_comment)
            last = int(found.group(1), 16) if found else current.addr
            current.end = max(last + 8, current.addr)

    for raw in lines:
        line = raw.rstrip("\n")
        header = _HDR.match(line)
        if header is not None:
            close()
            name = header.group(2)
            count = 2
            while name in functions:
                name = f"{header.group(2)}#{count}"
                count += 1
            current = Function(name, int(header.group(1), 16))
            functions[name] = current
            last_comment = ""
            continue
        if current is None or not line.startswith("\t"):
            continue
        body, _, comment = line.partition("//")
        text = body.strip()
        if not text or text == "...":
            continue
        last_comment = comment
        if text.startswith("s_getpc_b64"):
            current.getpc.append(len(current.insns))
            found = _ADDR.match(comment)
            if found is not None:
                current.pcs[len(current.insns)] = int(found.group(1), 16)
        current.insns.append(text)
    close()
    return functions


@dataclasses.dataclass(frozen=True)
class SymbolTable:
    """Function entry addresses of a code object, to resolve PC-relative targets."""

    starts: tuple[int, ...]
    names: tuple[str, ...]
    end: int

    @classmethod
    def from_functions(cls, functions: Mapping[str, Function]) -> "SymbolTable":
        ordered = sorted(functions.values(), key=lambda f: f.addr)
        return cls(
            tuple(f.addr for f in ordered),
            tuple(f.name for f in ordered),
            max((f.end for f in ordered), default=0),
        )

    def resolve(self, addr: int) -> str:
        """``name`` or ``name+off`` for an address inside the text, else ``data``."""
        if not self.starts or addr < self.starts[0] or addr >= self.end:
            return "data"
        k = bisect.bisect_right(self.starts, addr) - 1
        off = addr - self.starts[k]
        return self.names[k] if off == 0 else f"{self.names[k]}+{off}"


def _immediate(operand: str) -> int | None:
    operand = operand.strip()
    if operand.startswith("lit(") and operand.endswith(")"):
        operand = operand[4:-1].strip()
    try:
        return int(operand, 0)
    except ValueError:
        return None


def _operands(text: str) -> list[str]:
    return [op.strip() for op in text.partition(" ")[2].split(",")]


def _pair_after(insns: Sequence[str], i: int) -> tuple[int | None, int | None]:
    """Indices of the lo / hi add that complete the ``s_getpc_b64`` at ``i``."""
    dst = _GETPC_DST.match(insns[i])
    if dst is None:
        return None, None
    lo_reg, hi_reg = f"s{dst.group(1)}", f"s{dst.group(2)}"
    lo = hi = None
    for j in range(i + 1, min(i + 1 + _PAIR_WINDOW, len(insns))):
        text = insns[j]
        if text.startswith("s_getpc_b64"):
            break
        mnemonic = text.partition(" ")[0]
        ops = _operands(text)
        if len(ops) != 3 or ops[0] != ops[1]:
            continue
        if lo is None and mnemonic in _ADD_LO and ops[0] == lo_reg:
            lo = j
        elif hi is None and mnemonic in _ADD_HI and ops[0] == hi_reg:
            hi = j
        if lo is not None and hi is not None:
            break
    return lo, hi


def _replace_last_operand(text: str, token: str) -> str:
    head, sep, _ = text.rpartition(",")
    return f"{head}, {token}" if sep else text


def normalize_function(
    fn: Function, symbols: SymbolTable | None = None
) -> tuple[str, ...]:
    """Instruction text with PC-relative address pairs replaced by their target.

    ``s_getpc_b64`` plus the add pair behind it (``s_add_u32``/``s_addc_u32``, on
    gfx12 ``s_add_co_u32``/``s_add_co_ci_u32``) forms a PC-relative address. Its
    literals move whenever the layout around the function moves. The target is
    resolved to a function entry (``<pcrel:__assert_fail>``) when the address
    lands on one, so a changed callee still shows; any other target (string
    literals and constant data) is masked as ``<pcrel:data>``. Without addresses
    or symbols every pair is masked as ``<pcrel>``.
    """
    out = list(fn.insns)
    for i in fn.getpc:
        lo, hi = _pair_after(fn.insns, i)
        if lo is None and hi is None:
            continue
        token = "<pcrel>"
        if lo is not None and hi is not None and symbols is not None and i in fn.pcs:
            low = _immediate(_operands(fn.insns[lo])[2])
            high = _immediate(_operands(fn.insns[hi])[2])
            if low is not None and high is not None:
                offset = ((high & _MASK32) << 32) | (low & _MASK32)
                if offset >= 1 << 63:
                    offset -= 1 << 64
                # s_getpc_b64 is 4 bytes: the PC it reads is that of the next insn.
                token = f"<pcrel:{symbols.resolve(fn.pcs[i] + 4 + offset)}>"
        for j in (lo, hi):
            if j is not None:
                out[j] = _replace_last_operand(fn.insns[j], token)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class FunctionResult:
    """How one function compares between BASE and HEAD in one object and arch."""

    name: str
    klass: str
    base_insns: int | None
    head_insns: int | None
    demangled: str = ""
    deltas: list[int] = dataclasses.field(default_factory=list)
    detail: str = ""
    empty: bool = False

    @property
    def differs(self) -> bool:
        """CHANGED, or a function on one side only that holds real code."""
        if self.klass == CHANGED:
            return True
        return self.klass in (ONLY_BASE, ONLY_HEAD) and not self.empty


def _literal_owner_is_mov(text: str, index: int) -> bool:
    """True when the ``index``-th hex literal of ``text`` sits in a mov-immediate.

    A dual-issue line (``a :: b``) is judged by the half holding the literal.
    """
    seen = 0
    for part in text.split("::"):
        seen += len(_LITERAL.findall(part))
        if index < seen:
            mnemonic = part.strip().partition(" ")[0]
            return _MOV_IMM.match(mnemonic) is not None
    return False


def calls_assert_fail(name: str, normalized: Sequence[str]) -> bool:
    """True for ``__assert_fail`` itself and for functions that call it."""
    if _ASSERT_FAIL.match(name):
        return True
    return any("<pcrel:__assert_fail" in line for line in normalized)


def line_constant_diff(
    base: Sequence[str],
    head: Sequence[str],
    shifts: Collection[int],
    in_assert: bool,
) -> tuple[list[int] | None, str]:
    """Deltas when two normalised functions differ only in moved line constants.

    Returns ``(deltas, "")`` when every differing instruction pair differs in one
    hex literal that moved by an allowed shift, that literal is the operand of a
    mov-immediate, and the function is ``__assert_fail`` or calls it
    (``in_assert``). Otherwise ``(None, why)``: the first differing instruction
    and what is wrong with it, empty when the instruction counts differ.
    """
    if len(base) != len(head):
        return None, ""
    deltas: set[int] = set()
    for i, (x, y) in enumerate(zip(base, head)):
        if x == y:
            continue
        shown = f"instruction {i}: {x!r} -> {y!r}"
        if _LITERAL.sub("#", x) != _LITERAL.sub("#", y):
            return None, shown
        old, new = _LITERAL.findall(x), _LITERAL.findall(y)
        moved = [k for k, (p, q) in enumerate(zip(old, new)) if p != q]
        if len(moved) != 1:
            return None, shown
        delta = int(new[moved[0]], 16) - int(old[moved[0]], 16)
        if delta not in shifts:
            return None, f"{shown}; literal moved by {delta:+d}, not an allowed shift"
        if not in_assert:
            return None, f"{shown}; literal moved by {delta:+d} outside an assert stub"
        if not _literal_owner_is_mov(x, moved[0]):
            return None, f"{shown}; literal moved by {delta:+d} but not in a mov"
        deltas.add(delta)
    return sorted(deltas), ""


def classify_function(
    name: str,
    base: Function,
    head: Function,
    shifts: Collection[int],
    base_symbols: SymbolTable,
    head_symbols: SymbolTable,
) -> FunctionResult:
    """Class of one function present on both sides."""
    counts = (len(base.insns), len(head.insns))
    if base.insns == head.insns:
        return FunctionResult(name, IDENTICAL, *counts)
    norm_base = normalize_function(base, base_symbols)
    norm_head = normalize_function(head, head_symbols)
    if norm_base == norm_head:
        return FunctionResult(name, MASKED, *counts)
    in_assert = calls_assert_fail(name, norm_base) and calls_assert_fail(
        name, norm_head
    )
    deltas, why = line_constant_diff(norm_base, norm_head, shifts, in_assert)
    if deltas is not None:
        return FunctionResult(name, LINE_ONLY, *counts, deltas=deltas)
    return FunctionResult(name, CHANGED, *counts, detail=why)


def compare_functions(
    base: Mapping[str, Function],
    head: Mapping[str, Function],
    shifts: Collection[int] = (),
) -> list[FunctionResult]:
    """Classify every function of two code objects of the same arch, by name."""
    base_symbols = SymbolTable.from_functions(base)
    head_symbols = SymbolTable.from_functions(head)
    results: list[FunctionResult] = []
    for name in sorted(set(base) | set(head)):
        if name in base and name in head:
            results.append(
                classify_function(
                    name, base[name], head[name], shifts, base_symbols, head_symbols
                )
            )
        elif name in base:
            only = base[name]
            results.append(
                FunctionResult(name, ONLY_BASE, len(only.insns), None, empty=only.empty)
            )
        else:
            only = head[name]
            results.append(
                FunctionResult(name, ONLY_HEAD, None, len(only.insns), empty=only.empty)
            )
    return results


def family(demangled: str) -> str:
    """Function name without return type, template arguments and parameters."""
    text = demangled.replace("(anonymous namespace)", "{anon}")
    depth = 0
    cut = len(text)
    last_space = -1
    for i, char in enumerate(text):
        if char == "<":
            depth += 1
        elif char == ">":
            depth = max(depth - 1, 0)
        elif char == "(" and depth == 0:
            cut = i
            break
        elif char == " " and depth == 0:
            last_space = i
    name = text[last_space + 1 : cut].rstrip()
    if name.endswith(">"):
        depth = 0
        for i in range(len(name) - 1, -1, -1):
            if name[i] == ">":
                depth += 1
            elif name[i] == "<":
                depth -= 1
                if depth == 0:
                    name = name[:i]
                    break
    return (name or demangled).replace("{anon}", "(anonymous namespace)")


# --------------------------------------------------------------------------- #
# EXPECT / MAY_DIFFER verdict
# --------------------------------------------------------------------------- #


def split_list(value: str | None) -> list[str]:
    """Split a ``;``, ``,`` or blank separated list."""
    return [item for item in re.split(r"[;,\s]+", value or "") if item]


def is_expected(
    arch: str,
    demangled: str,
    expect: Any,
    may_differ: Collection[str] | None,
) -> bool:
    """Whether a difference of ``demangled`` on ``arch`` is allowed.

    ``expect`` is a compiled pattern or None. Nothing may differ unless EXPECT
    matches the demangled name; MAY_DIFFER, when given, further restricts which
    arches may differ at all: an arch outside it must be untouched.
    """
    if expect is None:
        return False
    if may_differ is not None and arch not in may_differ:
        return False
    return expect.search(demangled) is not None


# --------------------------------------------------------------------------- #
# ROCm tool wrappers
# --------------------------------------------------------------------------- #


def _rocm_bin_dirs() -> list[str]:
    dirs: list[str] = []
    for var in ("ROCM", "ROCM_PATH", "HIP_PATH", "ROCM_HOME"):
        root = os.environ.get(var)
        if root:
            dirs += [f"{root}/lib/llvm/bin", f"{root}/llvm/bin", f"{root}/bin"]
    dirs += ["/opt/rocm/lib/llvm/bin", "/opt/rocm/llvm/bin"]
    return list(dict.fromkeys(dirs))


def find_tool(*names: str) -> str | None:
    """First of ``names`` in the ROCm LLVM directories, else on PATH.

    ROCm wins over PATH: the disassembler has to be the one of the compiler that
    wrote the code objects, and a distribution llvm-objdump on PATH can be older.
    """
    for search in (os.pathsep.join(_rocm_bin_dirs()), None):
        for name in names:
            found = shutil.which(name, path=search)
            if found is not None:
                return found
    return None


def rocm_tool(*names: str) -> str:
    found = find_tool(*names)
    if found is None:
        raise ToolError(f"{' or '.join(names)} not found on PATH or in a ROCm LLVM dir")
    return found


def _run(cmd: Sequence[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        raise ToolError(f"{Path(cmd[0]).name} failed ({proc.returncode}): {tail}")
    return proc.stdout


def elf_section(data: bytes, name: str) -> bytes | None:
    """Contents of section ``name`` of a little-endian ELF64 file, or None."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ToolError("not a little-endian ELF64 file")
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    if shoff == 0:
        return None
    if shnum == 0:  # more than 0xff00 sections: the real count is in section 0
        (shnum,) = struct.unpack_from("<Q", data, shoff + 0x20)
    headers = [
        struct.unpack_from("<IIQQQQ", data, shoff + k * shentsize) for k in range(shnum)
    ]
    str_off, str_size = headers[shstrndx][4], headers[shstrndx][5]
    names = data[str_off : str_off + str_size]
    for name_off, _type, _flags, _addr, offset, size in headers:
        if names[name_off : names.index(b"\0", name_off)].decode() == name:
            return data[offset : offset + size]
    return None


def extract_code_objects(
    obj: Path, arches: Sequence[str], work: Path
) -> dict[str, Path] | None:
    """Unbundle the device code of ``arches`` from a HIP host object.

    Returns None for an object without device code (host-only), else a map from
    each requested arch the object carries to its code object file in ``work``.
    """
    section = elf_section(obj.read_bytes(), ".hip_fatbin")
    if not section:
        return None
    work.mkdir(parents=True, exist_ok=True)
    fatbin = work / "fatbin"
    fatbin.write_bytes(section)
    bundler = rocm_tool("clang-offload-bundler")
    listing = _run([bundler, "--list", "--type=o", f"--input={fatbin}"]).split()
    wanted: dict[str, str] = {}
    for target in listing:
        arch = target.rsplit("--", 1)[-1]
        for requested in arches:
            if requested in (arch, arch.split(":", 1)[0]):
                wanted.setdefault(requested, target)
    outputs = {arch: work / f"{arch}.co" for arch in wanted}
    if wanted:
        cmd = [
            bundler,
            "--unbundle",
            "--type=o",
            f"--input={fatbin}",
            "--targets=" + ",".join(wanted.values()),
        ]
        cmd += [f"--output={path}" for path in outputs.values()]
        _run(cmd)
    fatbin.unlink()
    return outputs


def disassemble(code_object: Path) -> dict[str, Function]:
    """Functions of a code object, from ``llvm-objdump -d --no-show-raw-insn``."""
    cmd = [rocm_tool("llvm-objdump"), "-d", "--no-show-raw-insn", str(code_object)]
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=errors, text=True, errors="replace"
        )
        assert proc.stdout is not None
        functions = parse_disassembly(proc.stdout)
        code = proc.wait()
        if code != 0:
            errors.seek(0)
            tail = errors.read().decode(errors="replace").strip().splitlines()[-3:]
            raise ToolError(f"llvm-objdump failed on {code_object.name}: {tail}")
    return functions


def tool_versions() -> dict[str, str]:
    """Path and version line of the ROCm tools that produced the disassembly."""
    found: dict[str, str] = {}
    for name in ("llvm-objdump", "clang-offload-bundler"):
        path = find_tool(name)
        if path is None:
            continue
        try:
            lines = _run([path, "--version"]).splitlines()
        except ToolError:
            lines = []
        version = next(
            (line.strip() for line in lines if "version" in line.lower()),
            lines[0].strip() if lines else "",
        )
        found[name] = f"{path} ({version})" if version else path
    return found


def demangle(names: Sequence[str]) -> dict[str, str]:
    """Demangled names (c++filt or llvm-cxxfilt); unchanged when neither exists."""
    tool = find_tool("c++filt", "llvm-cxxfilt")
    if tool is None or not names:
        return {name: name for name in names}
    proc = subprocess.run(
        [tool], input="\n".join(names) + "\n", capture_output=True, text=True
    )
    out = proc.stdout.splitlines()
    if proc.returncode != 0 or len(out) != len(names):
        return {name: name for name in names}
    return dict(zip(names, out))


# --------------------------------------------------------------------------- #
# Comparing the objects of both sides
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class ArchObject:
    """Result for one object on one arch."""

    object: str
    arch: str
    status: str
    results: list[FunctionResult] = dataclasses.field(default_factory=list)
    base_functions: int = 0
    head_functions: int = 0
    stub_only: bool = False


@dataclasses.dataclass
class ObjectAnalysis:
    """Result for one object on every requested arch."""

    object: str
    host_only: bool = False
    missing: str | None = None  # "base" or "head": the object exists on one side only
    cells: list[ArchObject] = dataclasses.field(default_factory=list)


def analyze_object(
    rel: str,
    base_obj: Path | None,
    head_obj: Path | None,
    arches: Sequence[str],
    shifts: Collection[int],
    work: Path,
) -> ObjectAnalysis:
    """Compare one object between BASE and HEAD on every requested arch.

    A side without the object, or whose object has no device code, counts as
    having no functions: everything on the other side is then one-sided.
    """
    analysis = ObjectAnalysis(rel)
    if base_obj is None and head_obj is None:
        raise ToolError(f"{rel}: not built on either side")
    if base_obj is None or head_obj is None:
        analysis.missing = "base" if base_obj is None else "head"
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        sides: dict[str, dict[str, Path] | None] = {}
        for side, path in (("base", base_obj), ("head", head_obj)):
            sides[side] = (
                None
                if path is None
                else extract_code_objects(path, arches, Path(tmp) / side)
            )
        if all(code is None for code in sides.values()):
            analysis.host_only = True
            return analysis
        for arch in arches:
            funcs: dict[str, dict[str, Function]] = {}
            absent = False
            for side, code in sides.items():
                if code is None:
                    funcs[side] = {}
                elif arch in code:
                    funcs[side] = disassemble(code[arch])
                else:
                    absent = True
            if absent:
                analysis.cells.append(ArchObject(rel, arch, STATUS_NO_CODE))
                continue
            results = compare_functions(funcs["base"], funcs["head"], shifts)
            names = demangle([r.name for r in results])
            for result in results:
                result.demangled = names.get(result.name, result.name)
            everything = [*funcs["base"].values(), *funcs["head"].values()]
            analysis.cells.append(
                ArchObject(
                    rel,
                    arch,
                    STATUS_COMPARED,
                    results,
                    len(funcs["base"]),
                    len(funcs["head"]),
                    stub_only=bool(everything) and all(f.empty for f in everything),
                )
            )
    return analysis


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _summarize_arch(
    arch: str,
    cells: Sequence[ArchObject],
    pattern: Any,
    may_differ: Collection[str] | None,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    differences: list[dict[str, Any]] = []
    empty_stubs: list[dict[str, Any]] = []
    line_families: Counter[str] = Counter()
    deltas: set[int] = set()
    objects: dict[str, Any] = {}
    not_run: list[str] = []
    for cell in sorted(cells, key=lambda c: c.object):
        if cell.status != STATUS_COMPARED:
            not_run.append(cell.object)
            objects[cell.object] = {"status": cell.status}
            continue
        per_class = Counter(r.klass for r in cell.results)
        counts.update(per_class)
        objects[cell.object] = {
            "status": cell.status,
            "stub_only": cell.stub_only,
            "functions_base": cell.base_functions,
            "functions_head": cell.head_functions,
            "counts": {k: per_class.get(k, 0) for k in CLASSES},
        }
        for result in cell.results:
            shown = result.demangled or result.name
            if result.klass == LINE_ONLY:
                line_families[family(shown)] += 1
                deltas.update(result.deltas)
            if result.klass not in (CHANGED, ONLY_BASE, ONLY_HEAD):
                continue
            item: dict[str, Any] = {
                "object": cell.object,
                "name": result.name,
                "demangled": shown,
                "family": family(shown),
                "class": result.klass,
                "base_insns": result.base_insns,
                "head_insns": result.head_insns,
                "detail": result.detail,
            }
            if result.differs:
                item["expected"] = is_expected(arch, shown, pattern, may_differ)
                differences.append(item)
            else:
                empty_stubs.append(item)
    differences.sort(key=lambda d: (d["family"], d["demangled"], d["object"]))
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in differences:
        families[item["family"]].append(item)
    expected = sum(1 for d in differences if d["expected"])
    if not_run:
        verdict = "NOT RUN"
    elif len(differences) > expected:
        verdict = "UNEXPECTED"
    elif expected:
        verdict = "expected only"
    else:
        verdict = "no difference"
    compared = [o for o, v in objects.items() if v["status"] == STATUS_COMPARED]
    return {
        "verdict": verdict,
        "objects": objects,
        "objects_compared": len(compared),
        "stub_only_objects": [o for o in compared if objects[o]["stub_only"]],
        "not_run_objects": not_run,
        "functions_base": sum(v.get("functions_base", 0) for v in objects.values()),
        "functions_head": sum(v.get("functions_head", 0) for v in objects.values()),
        "counts": {k: counts.get(k, 0) for k in CLASSES},
        "expected": expected,
        "unexpected": len(differences) - expected,
        "differences": differences,
        "families": [
            {
                "family": name,
                "class": ", ".join(sorted({i["class"] for i in items})),
                "functions": len(items),
                "base_insns": sum(i["base_insns"] or 0 for i in items),
                "head_insns": sum(i["head_insns"] or 0 for i in items),
                "expected": sum(1 for i in items if i["expected"]),
            }
            for name, items in families.items()
        ],
        "empty_stubs": empty_stubs,
        "line_constant_only": {
            "families": dict(sorted(line_families.items())),
            "deltas": sorted(deltas),
        },
    }


def build_report(
    analyses: Sequence[ObjectAnalysis],
    arches: Sequence[str],
    expect: str | None = None,
    may_differ: Sequence[str] | None = None,
    plan: Mapping[str, Any] | None = None,
    info: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Everything the markdown and JSON reports show, and the exit status."""
    plan = plan or {}
    pattern = re.compile(expect) if expect else None
    per_arch = {
        arch: _summarize_arch(
            arch,
            [c for a in analyses for c in a.cells if c.arch == arch],
            pattern,
            may_differ,
        )
        for arch in arches
    }
    not_run = [
        [obj, arch] for arch in arches for obj in per_arch[arch]["not_run_objects"]
    ]
    expected = sum(a["expected"] for a in per_arch.values())
    unexpected = sum(a["unexpected"] for a in per_arch.values())
    notes = list(plan.get("notes", []))
    if pattern is not None and expected == 0:
        notes.append(
            f"EXPECT {expect!r} matched no differing function on an arch that may "
            "differ: the intended change did not change any compared device code."
        )
    if may_differ is not None and pattern is None:
        notes.append(
            "MAY_DIFFER only narrows EXPECT: with EXPECT unset no function may "
            "differ on any arch."
        )
    if not_run:
        code = EXIT_ERROR
    elif unexpected:
        code = EXIT_DIFFERS
    else:
        code = EXIT_OK
    return {
        "tool": "archdiff",
        "base": plan.get("base", ""),
        "head": plan.get("head", ""),
        "arches": list(arches),
        "expect": expect or None,
        "may_differ": list(may_differ) if may_differ is not None else None,
        "line_shifts": list(plan.get("line_shifts", [])),
        "line_shifts_source": plan.get("line_shifts_source", ""),
        "changed_files": list(plan.get("changed_files", [])),
        "source_diff": dict(plan.get("source_diff", {})),
        "mode": plan.get("mode", ""),
        "units": list(plan.get("units", [])),
        "notes": notes,
        "info": dict(info or {}),
        "host_only": sorted(a.object for a in analyses if a.host_only),
        "one_sided": {a.object: a.missing for a in analyses if a.missing},
        "arch": per_arch,
        "expected": expected,
        "unexpected": unexpected,
        "not_run": not_run,
        "contract_not_run": [a for a in DEFAULT_ARCHES if a not in arches],
        "exit": code,
    }


def _cut(items: Sequence[Any], limit: int = LIST_LIMIT) -> tuple[Sequence[Any], int]:
    return items[:limit], max(len(items) - limit, 0)


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def result_line(report: Mapping[str, Any]) -> str:
    """One line: what the exit status means."""
    code = report["exit"]
    if code == EXIT_ERROR:
        parts = [f"{obj} on {arch}" for obj, arch in report["not_run"][:4]]
        line = f"ERROR (exit 2): not run (no code object): {'; '.join(parts)}"
    elif code == EXIT_DIFFERS:
        bad = [a for a in report["arches"] if report["arch"][a]["unexpected"]]
        line = (
            f"FAIL (exit 1): {report['unexpected']} unexpected differing "
            f"function(s) on {' '.join(bad)}"
        )
    elif report["expected"]:
        line = (
            f"PASS (exit 0): {report['expected']} differing function(s), all "
            "expected by EXPECT / MAY_DIFFER"
        )
    elif not report["units"]:
        line = "PASS (exit 0): no translation unit to compare"
    else:
        line = "PASS (exit 0): nothing differs"
    if report["contract_not_run"]:
        line += f"; contract arches NOT RUN: {' '.join(report['contract_not_run'])}"
    return line


def _signed(values: Iterable[int]) -> str:
    return ", ".join(f"{v:+d}" for v in values)


def _render_header(report: Mapping[str, Any]) -> list[str]:
    expect = f"`{report['expect']}`" if report["expect"] else "not set"
    may = report["may_differ"]
    shifts = _signed(report["line_shifts"]) or "none"
    source = report["line_shifts_source"]
    out = [
        "# archdiff: per-arch disassembly diff",
        "",
        f"- BASE: `{report['base']}`",
        f"- HEAD: `{report['head']}`",
        f"- Arches compiled and diffed: {' '.join(report['arches'])}",
        f"- EXPECT: {expect}",
        f"- MAY_DIFFER: {' '.join(may) if may is not None else 'not set'}",
        f"- Allowed `__LINE__` shifts: {shifts}" + (f" ({source})" if source else ""),
        "",
        f"**Result: {result_line(report)}**",
        "",
    ]
    for note in report["notes"]:
        out.append(f"> NOTE: {note}")
    if report["notes"]:
        out.append("")
    return out


def _render_sources(report: Mapping[str, Any]) -> list[str]:
    out = ["## Changed sources", ""]
    files = report["changed_files"]
    if not files:
        return [*out, "none under csrc/, cmake/ or CMakeLists.txt", ""]
    shown, more = _cut(files, 2 * LIST_LIMIT)
    for path in shown:
        diff = report["source_diff"].get(path)
        tail = (
            f" (+{diff['added']} -{diff['removed']}, {diff['hunks']} hunk(s))"
            if diff
            else ""
        )
        out.append(f"- `{path}`{tail}")
    if more:
        out.append(f"- ... and {more} more")
    return [*out, ""]


def _render_units(report: Mapping[str, Any]) -> list[str]:
    out = ["## Translation units", ""]
    if report["units"]:
        out += _table(
            ["object", "source", "why built"],
            (
                (
                    f"`{u['object']}`",
                    f"`{u['source']}`",
                    ", ".join(f"`{r}`" for r in u["reasons"][:3])
                    + (" ..." if len(u["reasons"]) > 3 else ""),
                )
                for u in report["units"]
            ),
        )
    else:
        out.append("none: no ROCm translation unit includes a changed file")
    if report["host_only"]:
        out += [
            "",
            "Host-only objects (no device code, not compared): "
            + ", ".join(f"`{o}`" for o in report["host_only"]),
        ]
    for obj, side in report["one_sided"].items():
        built = "HEAD" if side == "base" else "BASE"
        out += [
            "",
            f"`{obj}` exists on the {built} side only: every function on that side "
            f"is only-in-{built.lower()}.",
        ]
    return [*out, ""]


def _render_platforms(report: Mapping[str, Any]) -> list[str]:
    rows = []
    for arch in report["arches"]:
        a = report["arch"][arch]
        c = a["counts"]
        if a["objects_compared"]:
            run = f"{a['objects_compared']} object(s)"
        else:
            run = "NOT RUN" if a["not_run_objects"] else "none to compare"
        rows.append(
            (
                arch,
                run,
                f"{a['functions_base']} / {a['functions_head']}",
                c[IDENTICAL],
                c[MASKED],
                c[LINE_ONLY],
                c[CHANGED],
                c[ONLY_BASE],
                c[ONLY_HEAD],
                a["verdict"],
            )
        )
    return [
        "## Platforms",
        "",
        *_table(
            [
                "arch",
                "compiled + diffed",
                "functions base / head",
                "identical",
                "after masking",
                "line-constant-only",
                "CHANGED",
                "only in base",
                "only in head",
                "verdict",
            ],
            rows,
        ),
        "",
        f"Platforms run: {' '.join(report['arches'])}. "
        + (
            "Contract platforms not run: "
            f"{' '.join(report['contract_not_run'])} (not in ARCHES): not run is "
            "not a pass."
            if report["contract_not_run"]
            else "No contract platform was left out."
        ),
        "",
        "This report covers code generation only: no kernel was run and no GPU "
        "was touched, so execution on hardware is not covered.",
        "",
    ]


def _render_arch(arch: str, a: Mapping[str, Any]) -> list[str]:
    out = [f"### {arch}: {a['verdict']}", ""]
    if a["not_run_objects"]:
        out += [
            "NOT RUN (no code object for this arch): "
            + ", ".join(f"`{o}`" for o in a["not_run_objects"]),
            "",
        ]
    if a["stub_only_objects"]:
        out += [
            "Compiles to empty stubs only: "
            + ", ".join(f"`{o}`" for o in a["stub_only_objects"]),
            "",
        ]
    line = a["line_constant_only"]
    if line["families"]:
        families = ", ".join(f"{k}: {v}" for k, v in line["families"].items())
        out += [
            f"line-constant-only ({sum(line['families'].values())} function(s), "
            f"`__LINE__` of assert stubs moved by {_signed(line['deltas'])}; "
            f"everything else identical): {families}",
            "",
        ]
    if a["empty_stubs"]:
        shown, more = _cut(a["empty_stubs"])
        out += [
            f"Empty stub functions on one side only ({len(a['empty_stubs'])}; no "
            "code, not a difference): "
            + ", ".join(f"`{i['demangled'][:80]}`" for i in shown)
            + (f", ... and {more} more" if more else ""),
            "",
        ]
    if not a["differences"]:
        return [*out, "No differing function.", ""]
    for fam in a["families"]:
        if fam["expected"] == fam["functions"]:
            tag = "expected"
        else:
            tag = f"{fam['functions'] - fam['expected']} UNEXPECTED"
        out.append(
            f"- {fam['class']} family `{fam['family']}`: {fam['functions']} "
            f"function(s), instructions {fam['base_insns']} -> {fam['head_insns']}"
            f" [{tag}]"
        )
        members = [d for d in a["differences"] if d["family"] == fam["family"]]
        shown, more = _cut(members)
        for item in shown:
            mark = "" if item["expected"] else " **UNEXPECTED**"
            detail = f"; {item['detail']}" if item["detail"] else ""
            out.append(
                f"    - `{item['demangled'][:140]}`: {item['base_insns']} -> "
                f"{item['head_insns']} instrs{mark}{detail}"
            )
        if more:
            out.append(f"    - ... and {more} more")
    return [*out, ""]


def _render_method() -> list[str]:
    return [
        "## How functions are compared",
        "",
        "Per object and arch the device code object is unbundled, disassembled "
        "(`llvm-objdump -d`, trailing comments and `...` fill dropped) and "
        "matched by mangled name. EXPECT is searched in the demangled name; "
        "without it nothing may differ, and arches outside MAY_DIFFER may not "
        "differ at all.",
        "",
        f"- {IDENTICAL}: instruction text equal.",
        f"- {MASKED}: equal once each PC-relative address pair (`s_getpc_b64` + "
        "`s_add_u32`/`s_addc_u32`, on gfx12 `s_add_co_u32`/`s_add_co_ci_u32`) "
        "is replaced by its target: a function entry keeps its name "
        "(`<pcrel:__assert_fail>`, so a changed callee shows), any other target "
        "(string literals, constant data) is `<pcrel:data>`.",
        f"- {LINE_ONLY}: same length, and every remaining difference is one hex "
        "literal of a mov-immediate that moved by an allowed shift, inside "
        "`__assert_fail` or a function calling it. A shifted literal anywhere "
        "else, in another instruction, or by another amount is CHANGED.",
        f"- {CHANGED}: anything else, including a different instruction count.",
        f"- {ONLY_BASE} / {ONLY_HEAD}: symbol on one side only. A function that "
        "is nothing but `s_endpgm` / `s_nop` is an empty stub: listed, not "
        "counted as a difference.",
        "",
        "Not compared: host code, kernel descriptors and HSA metadata (register "
        "and LDS counts), and the contents of constant data read through "
        "PC-relative addresses.",
        "",
    ]


def render_markdown(report: Mapping[str, Any]) -> str:
    """The human report."""
    out = [
        *_render_header(report),
        *_render_sources(report),
        *_render_units(report),
        *_render_platforms(report),
        "## Per arch",
        "",
    ]
    for arch in report["arches"]:
        out += _render_arch(arch, report["arch"][arch])
    out += _render_method()
    if report["info"]:
        out += ["## Run", "", *(f"- {k}: {v}" for k, v in report["info"].items()), ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text())


def _parse_shifts(value: str) -> list[int]:
    try:
        return sorted({int(item) for item in split_list(value)})
    except ValueError as exc:
        raise ToolError(f"LINE_SHIFTS must be a list of integers: {value!r}") from exc


def _plan_units(
    args: argparse.Namespace,
    base: Tree,
    head: Tree,
    changed: Sequence[str],
    explicit: Sequence[str],
) -> tuple[dict[CompileUnit, list[str]], list[CompileUnit], list[str]]:
    """Units of the configured build that this phase has to build, and notes."""
    hip_map = hip_sources(DirTree(args.tree_dir))
    units = units_from_compdb(
        _read_json(args.compdb), args.build_dir, args.tree_dir, hip_map
    )
    graphs = (IncludeGraph(base), IncludeGraph(head))
    closures = {
        unit.source: set().union(*(g.closure(unit.source) for g in graphs))
        for unit in units
    }
    chosen = select_units(units, changed, closures, explicit, bool(args.all))
    notes: list[str] = []
    reached = set().union(*closures.values()) if closures else set()
    listed = {unit.source for unit in units}
    orphans = [
        rel
        for rel in changed
        if rel.startswith(CSRC + "/") and rel not in reached and rel not in listed
    ]
    if orphans:
        notes.append(
            "changed csrc files that no ROCm translation unit includes (not "
            f"compiled in this configuration): {', '.join(orphans[:6])}"
            + (" ..." if len(orphans) > 6 else "")
        )
    return chosen, units, notes


def cmd_plan(args: argparse.Namespace) -> int:
    """Decide which objects to build and which line shifts to allow."""
    base, head = DirTree(args.base_tree), DirTree(args.head_tree)
    changed = changed_files(base, head)
    csrc_changed = [rel for rel in changed if rel.startswith(CSRC + "/")]
    diff = tree_diff(base, head, csrc_changed)
    if args.diff_out:
        Path(args.diff_out).write_text(diff)
    if args.line_shifts:
        shifts, shifts_source = _parse_shifts(args.line_shifts), "LINE_SHIFTS"
    else:
        shifts = sorted(line_shifts_from_unified_diff(diff))
        shifts_source = "derived from the source diff"
    explicit = [posixpath.normpath(t) for t in split_list(args.tus)]
    if explicit and args.all:
        raise ToolError("TUS and ALL are exclusive")
    notes: list[str] = []
    cmake = [rel for rel in changed if is_cmake_input(rel)]
    if cmake:
        notes.append(
            f"{', '.join(cmake[:4])}{' ...' if len(cmake) > 4 else ''} changed: "
            "flag, arch and target-list changes cannot be attributed to "
            "translation units, so only units whose sources or headers changed "
            "are built. Pass TUS='csrc/a.cu csrc/b.cu' or ALL=1 to compare more."
        )
    plan: dict[str, Any] = {
        "base": args.base_label,
        "head": args.head_label,
        "mode": "ALL" if args.all else "TUS" if explicit else "auto",
        "changed_files": changed,
        "source_diff": {
            f.path: {"hunks": f.hunks, "added": f.added, "removed": f.removed}
            for f in parse_unified_diff(diff)
        },
        "line_shifts": shifts,
        "line_shifts_source": shifts_source,
        "tus_missing": [],
        "units": [],
        "notes": notes,
    }
    build: list[str] = []
    if args.compdb:
        chosen, units, unit_notes = _plan_units(args, base, head, changed, explicit)
        listed = {unit.source for unit in units}
        absent = [t for t in explicit if t not in listed]
        merged: dict[str, dict[str, Any]] = {}
        if args.previous:
            previous = _read_json(args.previous)
            gone = [t for t in absent if t in previous["tus_missing"]]
            if gone:
                raise ToolError(
                    f"{', '.join(gone)}: not a translation unit of the ROCm build "
                    "on either side"
                )
            merged = {u["object"]: u for u in previous["units"]}
            notes = [*previous["notes"], *notes]
            plan["changed_files"] = sorted({*previous["changed_files"], *changed})
        else:
            plan["tus_missing"] = absent
        for unit, reasons in chosen.items():
            old = merged.get(unit.object, {}).get("reasons", [])
            merged[unit.object] = {
                "source": unit.source,
                "object": unit.object,
                "target": unit.target,
                "reasons": sorted({*old, *reasons}),
            }
        plan["notes"] = list(dict.fromkeys([*notes, *unit_notes]))
        plan["units"] = [merged[o] for o in sorted(merged)]
        known = {unit.object for unit in units}
        build = [o for o in sorted(merged) if o in known]
    if args.plan_out:
        Path(args.plan_out).write_text(json.dumps(plan, indent=1) + "\n")
    if args.objects_out:
        Path(args.objects_out).write_text("".join(f"{o}\n" for o in build))
    print("graph" if changed or explicit or args.all else "empty")
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare the objects of the plan and write the reports."""
    plan: dict[str, Any] = _read_json(args.plan) if args.plan else {}
    arches = split_list(args.arches) or list(DEFAULT_ARCHES)
    may_differ = split_list(args.may_differ) if args.may_differ is not None else None
    if args.line_shifts:
        plan["line_shifts"] = _parse_shifts(args.line_shifts)
        plan["line_shifts_source"] = "LINE_SHIFTS"
    shifts = [int(s) for s in plan.get("line_shifts", [])]
    base_dir, head_dir = Path(args.base_dir), Path(args.head_dir)
    scratch = Path(args.tmp) if args.tmp else None
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)
    analyses: list[ObjectAnalysis] = []
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        for unit in plan.get("units", []):
            rel = unit["object"]
            base_obj = base_dir / rel if (base_dir / rel).is_file() else None
            head_obj = head_dir / rel if (head_dir / rel).is_file() else None
            analyses.append(
                analyze_object(rel, base_obj, head_obj, arches, shifts, Path(tmp))
            )
    info = dict(item.split("=", 1) for item in args.info or [] if "=" in item)
    info.update(tool_versions())
    report = build_report(analyses, arches, args.expect, may_differ, plan, info)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "archdiff.json").write_text(json.dumps(report, indent=1) + "\n")
    text = render_markdown(report)
    (out / "archdiff.md").write_text(text)
    print(text)
    return int(report["exit"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    plan = sub.add_parser("plan", help="pick the translation units to build")
    plan.add_argument("--base-tree", required=True)
    plan.add_argument("--head-tree", required=True)
    plan.add_argument("--base-label", default="")
    plan.add_argument("--head-label", default="")
    plan.add_argument("--compdb", help="`ninja -t compdb` output of the build")
    plan.add_argument("--build-dir", help="the configured build directory")
    plan.add_argument("--tree-dir", help="the configured source tree")
    plan.add_argument("--tus", default="", help="explicit csrc sources")
    plan.add_argument("--all", action="store_true", help="every unit of the build")
    plan.add_argument("--line-shifts", default="", help="override the derived shifts")
    plan.add_argument("--previous", help="plan.json of the BASE phase")
    plan.add_argument("--plan-out")
    plan.add_argument("--objects-out", help="objects to build, one per line")
    plan.add_argument("--diff-out", help="write the csrc unified diff here")
    plan.set_defaults(func=cmd_plan)
    compare = sub.add_parser("compare", help="diff the objects and report")
    compare.add_argument("--base-dir", required=True)
    compare.add_argument("--head-dir", required=True)
    compare.add_argument("--plan", help="plan.json")
    compare.add_argument("--arches", default=";".join(DEFAULT_ARCHES))
    compare.add_argument("--expect", default="")
    compare.add_argument("--may-differ", default=None)
    compare.add_argument("--line-shifts", default="", help="override the plan's")
    compare.add_argument("--out", required=True)
    compare.add_argument("--tmp", help="directory for temporary files")
    compare.add_argument("--info", action="append", help="key=value for the report")
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ToolError, OSError, ValueError, re.error, subprocess.SubprocessError) as e:
        print(f"archdiff: error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
