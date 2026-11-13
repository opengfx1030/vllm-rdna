# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests of tools/rdna/archdiff_report.py and tools/rdna/archdiff.sh.

Synthetic disassembly, synthetic ELF objects and fake ROCm tools: no ROCm, no
GPU, no network. The real tools are exercised by the smoke run in the PR text.
"""

import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "tools" / "rdna" / "archdiff_report.py"
DRIVER = ROOT / "tools" / "rdna" / "archdiff.sh"

_spec = importlib.util.spec_from_file_location("archdiff_report", HELPER)
assert _spec is not None and _spec.loader is not None
ar = importlib.util.module_from_spec(_spec)
sys.modules["archdiff_report"] = ar
_spec.loader.exec_module(ar)

DATA_BASE = 0x100000
PR54_SHIFTS = (11, 18, 24)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


class MemTree:
    """A source tree held in a dict of repo-relative path -> text."""

    def __init__(self, files: dict[str, str]) -> None:
        self.data = dict(files)

    def exists(self, rel: str) -> bool:
        return rel in self.data

    def read(self, rel: str) -> str:
        return self.data[rel]

    def files(self, rel: str) -> list[str]:
        prefix = rel.rstrip("/") + "/"
        return sorted(p for p in self.data if p == rel or p.startswith(prefix))


def _size(insn: str) -> int:
    return 8 if ("0x" in insn or "@lo" in insn or "@hi" in insn) else 4


def disassembly(funcs: Sequence[tuple[str, int, Sequence[str]]]) -> str:
    """Lay functions out the way llvm-objdump prints them.

    ``funcs`` is ``(name, start address, instructions)``. Instructions take 4
    bytes, 8 with a literal. ``@lo(X)`` / ``@hi(X)`` / ``@hilit(X)`` stand for
    the low / high half of the PC-relative offset from the preceding
    ``s_getpc_b64`` to X, a function name or ``dataN`` (N bytes into the data).
    """
    starts = {name: addr for name, addr, _ in funcs}

    def target(spec: str) -> int:
        if spec.startswith("data"):
            return DATA_BASE + int(spec[4:] or 0)
        return starts[spec]

    lines = ["", "Disassembly of section .text:", ""]
    for name, start, insns in funcs:
        lines += [f"{start:016x} <{name}>:"]
        addr = start
        getpc = start
        for insn in insns:
            if insn.startswith("s_getpc_b64"):
                getpc = addr
            for kind in ("lo", "hilit", "hi"):
                for found in re.findall(rf"@{kind}\(([\w.$]+)\)", insn):
                    offset = (target(found) - (getpc + 4)) & (2**64 - 1)
                    low, high = offset & 0xFFFFFFFF, offset >> 32
                    if kind == "lo":
                        text = f"0x{low:x}"
                    elif kind == "hilit":
                        text = f"lit(0x{high:x})" if high else "0"
                    else:
                        text = "-1" if high == 0xFFFFFFFF else f"{high}"
                    insn = insn.replace(f"@{kind}({found})", text)
            lines.append(f"\t{insn:<48}// {addr:012X}: DEADBEEF")
            addr += _size(insn)
    return "\n".join(lines) + "\n"


def parse(funcs: Sequence[tuple[str, int, Sequence[str]]]) -> dict[str, Any]:
    return ar.parse_disassembly(disassembly(funcs).splitlines())


def assert_stub(line: int, style: str = "gfx10") -> list[str]:
    """The body of a kernel that only calls __assert_fail, like an RDNA stub."""
    if style == "gfx12":
        return [
            "s_getpc_b64 s[0:1]",
            "s_wait_alu depctr_sa_sdst(0)",
            "s_sext_i32_i16 s1, s1",
            "s_add_co_u32 s0, s0, @lo(data0)",
            "s_wait_alu depctr_sa_sdst(0)",
            "s_add_co_ci_u32 s1, s1, @hilit(data0)",
            f"v_dual_mov_b32 v8, 0x{line:x} :: v_dual_mov_b32 v9, s0",
            "s_getpc_b64 s[2:3]",
            "s_add_co_u32 s2, s2, @lo(__assert_fail)",
            "s_wait_alu depctr_sa_sdst(0)",
            "s_add_co_ci_u32 s3, s3, @hilit(__assert_fail)",
            "s_swappc_b64 s[30:31], s[2:3]",
            "s_endpgm",
        ]
    return [
        "s_getpc_b64 s[4:5]",
        "s_add_u32 s4, s4, @lo(data0)",
        "s_addc_u32 s5, s5, @hi(data0)",
        f"v_mov_b32_e32 v0, 0x{line:x}",
        "v_mov_b32_e32 v1, s4",
        "v_mov_b32_e32 v2, s5",
        "s_getpc_b64 s[6:7]",
        "s_add_u32 s6, s6, @lo(__assert_fail)",
        "s_addc_u32 s7, s7, @hi(__assert_fail)",
        "s_swappc_b64 s[30:31], s[6:7]",
        "s_endpgm",
    ]


ASSERT_FAIL_BODY = ["s_mov_b32 s0, 0", "s_setpc_b64 s[30:31]"]


def compare(
    base: Sequence[tuple[str, int, Sequence[str]]],
    head: Sequence[tuple[str, int, Sequence[str]]],
    shifts: Sequence[int] = PR54_SHIFTS,
) -> dict[str, Any]:
    results = ar.compare_functions(parse(base), parse(head), shifts)
    return {r.name: r for r in results}


def make_elf(sections: dict[str, bytes]) -> bytes:
    """A minimal little-endian ELF64 object with the named sections."""
    names = [".shstrtab", *sections]
    strtab = b"\0" + b"".join(n.encode() + b"\0" for n in names)
    blobs: list[tuple[int, bytes]] = [(0, b"")]
    for index, data in enumerate(sections.values()):
        blobs.append((strtab.index(names[index + 1].encode()), data))
    blobs.append((strtab.index(b".shstrtab"), strtab))
    offset = 64
    placed = []
    for name_off, data in blobs:
        placed.append((name_off, offset, len(data)))
        offset += len(data)
    shoff = offset
    body = b"".join(data for _, data in blobs)
    headers = b"".join(
        struct.pack(
            "<IIQQQQIIQQ", name_off, 1 if size else 0, 0, 0, off, size, 0, 0, 1, 0
        )
        for name_off, off, size in placed
    )
    ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8)
    header = ident + struct.pack(
        "<HHIQQQIHHHHHH",
        1,
        0x3E,
        1,
        0,
        0,
        shoff,
        0,
        64,
        0,
        0,
        64,
        len(placed),
        len(placed) - 1,
    )
    assert len(header) == 64
    return header + body + headers


def fat_object(per_target: dict[str, str]) -> bytes:
    """An object whose .hip_fatbin the fake bundler understands (JSON)."""
    return make_elf({".text": b"\x90", ".hip_fatbin": json.dumps(per_target).encode()})


FAKE_BUNDLER = """\
#!{python} -S
import json, sys
opts, outs = {{}}, []
for arg in sys.argv[1:]:
    if arg == "--list":
        opts["list"] = True
    elif arg.startswith("--input="):
        opts["input"] = arg[8:]
    elif arg.startswith("--targets="):
        opts["targets"] = arg[10:].split(",")
    elif arg.startswith("--output="):
        outs.append(arg[9:])
data = json.load(open(opts["input"]))
if opts.get("list"):
    print("\\n".join(data))
    sys.exit(0)
for target, out in zip(opts["targets"], outs):
    open(out, "w").write(data[target])
"""

FAKE_OBJDUMP = """\
#!{python} -S
import sys
if sys.argv[1:] == ["--version"]:
    print("AMD LLVM (fake):")
    print("  LLVM version 23.0.0fake")
    sys.exit(0)
sys.stdout.write(open(sys.argv[-1]).read())
"""

FAKE_FILT = """\
#!{python} -S
import sys
names = {{
    "_Z10LLGemm1_kernelIN3c104HalfELi2EEvv": "void LLGemm1_kernel<c10::Half, 2>()",
    "_Z10LLGemm1_kernelIN3c104HalfELi4EEvv": "void LLGemm1_kernel<c10::Half, 4>()",
    "_Z4normv": "void norm()",
}}
for line in sys.stdin:
    line = line.rstrip("\\n")
    print(names.get(line, line))
"""


@pytest.fixture
def fake_rocm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A PATH directory with fake clang-offload-bundler, llvm-objdump, c++filt."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    for name, body in (
        ("clang-offload-bundler", FAKE_BUNDLER),
        ("llvm-objdump", FAKE_OBJDUMP),
        ("c++filt", FAKE_FILT),
    ):
        script = bindir / name
        script.write_text(body.format(python=sys.executable))
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    # No real ROCm LLVM directory may win over the fake tools on PATH.
    monkeypatch.setattr(ar, "_rocm_bin_dirs", lambda: [])
    return bindir


def target(arch: str) -> str:
    return f"hipv4-amdgcn-amd-amdhsa--{arch}"


# --------------------------------------------------------------------------- #
# Include closure and translation unit selection
# --------------------------------------------------------------------------- #


def test_parse_includes_quote_angle_and_spacing() -> None:
    text = textwrap.dedent(
        """\
        #include "a.h"
        #  include <b/c.h>
        \t#include\t"d.h"
        // #include "commented.h"
        #include_next <skip.h>
        int x;
        """
    )
    assert ar.parse_includes(text) == ["a.h", "b/c.h", "d.h"]


def test_resolve_include_against_includer_dir_then_csrc() -> None:
    files = {"csrc/rocm/local.h", "csrc/core/shared.h", "csrc/rocm/sub/x.h"}
    resolve = lambda name: ar.resolve_include(  # noqa: E731
        name, "csrc/rocm/k.cu", files.__contains__
    )
    assert resolve("local.h") == "csrc/rocm/local.h"
    assert resolve("core/shared.h") == "csrc/core/shared.h"
    assert resolve("sub/x.h") == "csrc/rocm/sub/x.h"
    assert resolve("../core/shared.h") == "csrc/core/shared.h"
    assert resolve("torch/all.h") is None  # system header
    assert resolve("../../outside.h") is None  # never leaves csrc/


def test_include_closure_is_transitive_cycle_safe_and_tolerates_missing() -> None:
    tree = MemTree(
        {
            "csrc/k.cu": '#include "a.h"\n#include <hip/hip_runtime.h>\n',
            "csrc/a.h": '#include "b.h"\n#include "missing.h"\n',
            "csrc/b.h": '#include "a.h"\n#include "c/deep.h"\n',
            "csrc/c/deep.h": "// leaf\n",
        }
    )
    graph = ar.IncludeGraph(tree)
    assert graph.closure("csrc/k.cu") == {
        "csrc/k.cu",
        "csrc/a.h",
        "csrc/b.h",
        "csrc/c/deep.h",
    }
    assert graph.closure("csrc/gone.cu") == {"csrc/gone.cu"}


def test_changed_files_modified_added_removed_and_cmake() -> None:
    base = MemTree(
        {
            "csrc/same.cu": "x",
            "csrc/mod.cu": "old",
            "csrc/gone.cu": "bye",
            "CMakeLists.txt": "project(a)",
            "cmake/utils.cmake": "same",
            "vllm/ignored.py": "a",
        }
    )
    head = MemTree(
        {
            "csrc/same.cu": "x",
            "csrc/mod.cu": "new",
            "csrc/new.cu": "hi",
            "CMakeLists.txt": "project(b)",
            "cmake/utils.cmake": "same",
            "vllm/ignored.py": "b",
        }
    )
    assert ar.changed_files(base, head) == [
        "csrc/gone.cu",
        "csrc/mod.cu",
        "csrc/new.cu",
        "CMakeLists.txt",
    ]
    assert ar.is_cmake_input("CMakeLists.txt")
    assert ar.is_cmake_input("cmake/utils.cmake")
    assert not ar.is_cmake_input("csrc/mod.cu")


def _units() -> list[Any]:
    return [
        ar.CompileUnit(
            "csrc/rocm/a.cu", "CMakeFiles/_rocm_C.dir/csrc/rocm/a.hip.o", "_rocm_C"
        ),
        ar.CompileUnit(
            "csrc/rocm/b.cu", "CMakeFiles/_rocm_C.dir/csrc/rocm/b.hip.o", "_rocm_C"
        ),
        ar.CompileUnit("csrc/host.cpp", "CMakeFiles/_C.dir/csrc/host.cpp.o", "_C"),
    ]


def test_select_units_by_closure_explicit_and_all() -> None:
    units = _units()
    closures = {
        "csrc/rocm/a.cu": {"csrc/rocm/a.cu", "csrc/rocm/common.h"},
        "csrc/rocm/b.cu": {"csrc/rocm/b.cu"},
        "csrc/host.cpp": {"csrc/host.cpp", "csrc/rocm/common.h"},
    }
    auto = ar.select_units(units, ["csrc/rocm/common.h"], closures)
    assert {u.source: r for u, r in auto.items()} == {
        "csrc/rocm/a.cu": ["csrc/rocm/common.h"],
        "csrc/host.cpp": ["csrc/rocm/common.h"],
    }
    own = ar.select_units(units, ["csrc/rocm/b.cu"], closures)
    assert [u.source for u in own] == ["csrc/rocm/b.cu"]
    assert ar.select_units(units, ["csrc/unrelated.h"], closures) == {}
    explicit = ar.select_units(units, [], closures, explicit=["csrc/rocm/b.cu"])
    assert {u.source: r for u, r in explicit.items()} == {"csrc/rocm/b.cu": ["TUS"]}
    everything = ar.select_units(units, [], closures, everything=True)
    assert len(everything) == 3
    assert all(r == ["ALL=1"] for r in everything.values())


def test_hip_name_matches_the_cmake_hipify_rule() -> None:
    assert ar.hip_name("csrc/rocm/skinny_gemms.cu") == "csrc/rocm/skinny_gemms.hip"
    assert ar.hip_name("csrc/cuda_utils_kernels.cu") == "csrc/hip_utils_kernels.hip"
    assert ar.hip_name("csrc/quantization/cuda/x.cu") == "csrc/quantization/hip/x.hip"


def test_units_from_compdb_maps_hipified_and_plain_sources(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    build = tmp_path / "build"
    (tree / "csrc" / "rocm").mkdir(parents=True)
    (tree / "csrc" / "rocm" / "skinny.cu").write_text("")
    (tree / "csrc" / "plain.cpp").write_text("")
    (build / "csrc" / "rocm").mkdir(parents=True)
    hip_map = {"csrc/rocm/skinny.hip": "csrc/rocm/skinny.cu"}
    entries = [
        {
            "directory": str(build),
            "file": str(build / "csrc/rocm/skinny.hip"),
            "output": "CMakeFiles/_rocm_C_skinny.dir/csrc/rocm/skinny.hip.o",
        },
        {
            "directory": str(build),
            "file": "../tree/csrc/plain.cpp",
            "output": "CMakeFiles/_C.dir/csrc/plain.cpp.o",
        },
        # not an object, no real source, not from csrc/, not hipified from csrc/
        {"directory": str(build), "file": "x.cpp", "output": "libfoo.so"},
        {"directory": str(build), "file": ".", "output": "CMakeFiles/_C.dir/.o"},
        {
            "directory": str(build),
            "file": "/usr/include/other.cpp",
            "output": "CMakeFiles/_C.dir/other.o",
        },
        {
            "directory": str(build),
            "file": str(build / "csrc/rocm/orphan.hip"),
            "output": "CMakeFiles/_C.dir/orphan.o",
        },
    ]
    units = ar.units_from_compdb(entries, str(build), str(tree), hip_map)
    assert [(u.source, u.object, u.target) for u in units] == [
        ("csrc/plain.cpp", "CMakeFiles/_C.dir/csrc/plain.cpp.o", "_C"),
        (
            "csrc/rocm/skinny.cu",
            "CMakeFiles/_rocm_C_skinny.dir/csrc/rocm/skinny.hip.o",
            "_rocm_C_skinny",
        ),
    ]


def test_hip_sources_lists_only_cu_files() -> None:
    tree = MemTree({"csrc/a/cuda_x.cu": "", "csrc/b.cpp": "", "csrc/c.hip": ""})
    assert ar.hip_sources(tree) == {"csrc/a/hip_x.hip": "csrc/a/cuda_x.cu"}


# --------------------------------------------------------------------------- #
# Source diff and the line shifts it allows
# --------------------------------------------------------------------------- #

PR54_DIFF = textwrap.dedent(
    """\
    diff --git a/csrc/rocm/skinny_gemms.cu b/csrc/rocm/skinny_gemms.cu
    index 1111111..2222222 100644
    --- a/csrc/rocm/skinny_gemms.cu
    +++ b/csrc/rocm/skinny_gemms.cu
    @@ -69,6 +69,17 @@ namespace x {
     ctx
     ctx
     ctx
    +a1
    +a2
    +a3
    +a4
    +a5
    +a6
    +a7
    +a8
    +a9
    +a10
    +a11
     ctx
     ctx
     ctx
    @@ -257,6 +268,13 @@ void y() {
     ctx
     ctx
     ctx
    +b1
    +b2
    +b3
    +b4
    +b5
    +b6
    +b7
     ctx
     ctx
     ctx
    @@ -1215,6 +1233,12 @@ void z() {
     ctx
     ctx
     ctx
    +c1
    +c2
    +c3
    +c4
    +c5
    +c6
     ctx
     ctx
     ctx
    """
)


def test_line_shifts_of_the_pr54_diff() -> None:
    assert ar.line_shifts_from_unified_diff(PR54_DIFF) == {11, 18, 24}


def test_line_shifts_deletion_replacement_and_zero_net() -> None:
    diff = textwrap.dedent(
        """\
        --- a/csrc/x.cu
        +++ b/csrc/x.cu
        @@ -10,5 +10,3 @@
         ctx
        -gone1
        -gone2
         ctx
        @@ -50,4 +48,4 @@
         ctx
        -old
        +new
         ctx
        @@ -90,3 +88,5 @@
         ctx
        +add1
        +add2
         ctx
        """
    )
    # -2 after the first run, unchanged by a 1:1 replacement, 0 after the insert
    assert ar.line_shifts_from_unified_diff(diff) == {-2}


def test_line_shifts_new_file_and_each_file_starts_at_zero() -> None:
    diff = textwrap.dedent(
        """\
        --- /dev/null
        +++ b/csrc/new.cu
        @@ -0,0 +1,3 @@
        +l1
        +l2
        +l3
        --- a/csrc/a.cu
        +++ b/csrc/a.cu
        @@ -5,1 +5,2 @@
         ctx
        +one
        --- a/csrc/b.cu
        +++ b/csrc/b.cu
        @@ -7,1 +7,3 @@
         ctx
        +one
        +two
        """
    )
    files = ar.parse_unified_diff(diff)
    assert [f.path for f in files] == ["csrc/new.cu", "csrc/a.cu", "csrc/b.cu"]
    assert [(f.hunks, f.added, f.removed) for f in files] == [
        (1, 3, 0),
        (1, 1, 0),
        (1, 2, 0),
    ]
    assert ar.line_shifts_from_unified_diff(diff) == {3, 1, 2}


def test_unified_diff_text_zero_context_round_trips_through_the_parser() -> None:
    old_lines = [f"line{i}" for i in range(1, 101)]
    new_lines = list(old_lines)
    new_lines[5] = "changed"  # replaces line 6: net zero
    new_lines[90:90] = [f"two{i}" for i in range(7)]  # after old line 90
    new_lines[72:72] = [f"ins{i}" for i in range(11)]  # after old line 72
    old = "\n".join(old_lines) + "\n"
    diff = ar.unified_diff_text("csrc/x.cu", old, "\n".join(new_lines) + "\n")
    assert diff.startswith("--- a/csrc/x.cu\n+++ b/csrc/x.cu\n")
    assert [line for line in diff.splitlines() if line.startswith("@@")] == [
        "@@ -6,1 +6,1 @@",
        "@@ -72,0 +73,11 @@",
        "@@ -90,0 +102,7 @@",
    ]
    (file,) = ar.parse_unified_diff(diff)
    assert (file.hunks, file.added, file.removed) == (3, 19, 1)
    assert ar.line_shifts_from_unified_diff(diff) == {11, 18}
    assert ar.unified_diff_text("csrc/x.cu", old, old) == ""


def test_tree_diff_covers_added_and_deleted_files() -> None:
    base = MemTree({"csrc/gone.cu": "a\nb\n"})
    head = MemTree({"csrc/new.cu": "x\ny\nz\n"})
    diff = ar.tree_diff(base, head, ["csrc/gone.cu", "csrc/new.cu"])
    files = {f.path: f for f in ar.parse_unified_diff(diff)}
    assert (files["csrc/gone.cu"].added, files["csrc/gone.cu"].removed) == (0, 2)
    assert (files["csrc/new.cu"].added, files["csrc/new.cu"].removed) == (3, 0)


# --------------------------------------------------------------------------- #
# Disassembly parsing and normalisation
# --------------------------------------------------------------------------- #


def test_parse_disassembly_strips_comments_fill_and_keeps_order() -> None:
    text = textwrap.dedent(
        """\

        Disassembly of section .text:

        0000000000001000 <_Z4normv>:
        \ts_load_dword s0, s[4:5], 0x0 // 000000001000: C0020002 00000000
        \tv_add_f32_e32 v0, v1, v0 // 000000001008: 02000101
        \ts_cbranch_scc1 BB0_2 // 000000001010: BF850001 <_Z4normv+0x18>
        \t\t...
        \ts_endpgm // 000000001018: BF810000
        0000000000002000 <_Z4normv>:
        \ts_endpgm // 000000002000: BF810000
        """
    )
    funcs = ar.parse_disassembly(text.splitlines())
    assert list(funcs) == ["_Z4normv", "_Z4normv#2"]
    first = funcs["_Z4normv"]
    assert first.insns == [
        "s_load_dword s0, s[4:5], 0x0",
        "v_add_f32_e32 v0, v1, v0",
        "s_cbranch_scc1 BB0_2",
        "s_endpgm",
    ]
    assert first.addr == 0x1000
    assert not first.empty
    assert funcs["_Z4normv#2"].empty


def test_empty_functions_are_endpgm_nop_and_code_end_only() -> None:
    fn = ar.Function("k", 0, ["s_endpgm", "s_nop 0", "s_code_end"])
    assert fn.empty
    assert not ar.Function("k", 0, ["s_endpgm", "v_mov_b32 v0, 0"]).empty
    assert ar.Function("k", 0, []).empty


def test_normalize_masks_data_and_names_the_call_target() -> None:
    funcs = parse(
        [
            ("kern", 0x1000, assert_stub(0x336)),
            ("__assert_fail", 0x4000, ASSERT_FAIL_BODY),
        ]
    )
    symbols = ar.SymbolTable.from_functions(funcs)
    norm = ar.normalize_function(funcs["kern"], symbols)
    assert norm[1] == "s_add_u32 s4, s4, <pcrel:data>"
    assert norm[2] == "s_addc_u32 s5, s5, <pcrel:data>"
    assert norm[7] == "s_add_u32 s6, s6, <pcrel:__assert_fail>"
    assert norm[8] == "s_addc_u32 s7, s7, <pcrel:__assert_fail>"
    assert norm[3] == "v_mov_b32_e32 v0, 0x336"  # line constant stays visible


def test_normalize_handles_gfx12_wait_alu_sext_and_lit() -> None:
    funcs = parse(
        [
            ("kern", 0x1000, assert_stub(0x721, "gfx12")),
            ("__assert_fail", 0x4000, ASSERT_FAIL_BODY),
        ]
    )
    norm = ar.normalize_function(funcs["kern"], ar.SymbolTable.from_functions(funcs))
    assert norm[3] == "s_add_co_u32 s0, s0, <pcrel:data>"
    assert norm[5] == "s_add_co_ci_u32 s1, s1, <pcrel:data>"
    assert norm[8] == "s_add_co_u32 s2, s2, <pcrel:__assert_fail>"
    assert norm[10] == "s_add_co_ci_u32 s3, s3, <pcrel:__assert_fail>"
    assert norm[6] == "v_dual_mov_b32 v8, 0x721 :: v_dual_mov_b32 v9, s0"


def test_normalize_resolves_a_target_inside_a_function() -> None:
    text = disassembly(
        [
            (
                "kern",
                0x1000,
                [
                    "s_getpc_b64 s[4:5]",
                    "s_add_u32 s4, s4, @lo(callee)",
                    "s_addc_u32 s5, s5, @hi(callee)",
                    "s_endpgm",
                ],
            ),
            ("callee", 0x2000, ["s_nop 0"] * 8 + ["s_endpgm"]),
        ]
    )
    funcs = ar.parse_disassembly(text.splitlines())
    kern = funcs["kern"]
    low = int(kern.insns[1].rsplit(" ", 1)[1], 16)
    kern.insns[1] = f"s_add_u32 s4, s4, 0x{low + 12:x}"  # 12 bytes into callee
    norm = ar.normalize_function(kern, ar.SymbolTable.from_functions(funcs))
    assert norm[1] == "s_add_u32 s4, s4, <pcrel:callee+12>"


def test_normalize_without_addresses_masks_the_pair_as_pcrel() -> None:
    fn = ar.Function(
        "k",
        0,
        [
            "s_getpc_b64 s[4:5]",
            "s_add_u32 s4, s4, 0x1234",
            "s_addc_u32 s5, s5, -1",
            "s_endpgm",
        ],
        getpc=[0],
    )
    assert ar.normalize_function(fn) == (
        "s_getpc_b64 s[4:5]",
        "s_add_u32 s4, s4, <pcrel>",
        "s_addc_u32 s5, s5, <pcrel>",
        "s_endpgm",
    )


def test_normalize_leaves_unrelated_adds_alone() -> None:
    fn = ar.Function(
        "k",
        0,
        [
            "s_getpc_b64 s[4:5]",
            "s_add_u32 s0, s0, 0x10",  # not the getpc destination
            "s_add_u32 s4, s4, 0xffff0010",
            "s_addc_u32 s5, s5, -1",
        ],
        getpc=[0],
        pcs={0: 0x100},
    )
    norm = ar.normalize_function(fn, ar.SymbolTable((0,), ("k",), 0x200))
    assert norm[1] == "s_add_u32 s0, s0, 0x10"
    assert norm[2].startswith("s_add_u32 s4, s4, <pcrel")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

KERNEL = [
    "s_load_dwordx2 s[0:1], s[4:5], 0x0",
    "v_mov_b32_e32 v0, 0x10",
    "v_add_f32_e32 v1, v0, v0",
    "s_endpgm",
]


def layout(kernels: dict[str, list[str]], assert_at: int = 0x4000) -> list[Any]:
    funcs: list[Any] = [
        (name, 0x1000 + 0x400 * i, body)
        for i, (name, body) in enumerate(kernels.items())
    ]
    funcs.append(("__assert_fail", assert_at, ASSERT_FAIL_BODY))
    return funcs


def test_identical_functions() -> None:
    res = compare(layout({"k": KERNEL}), layout({"k": KERNEL}))
    assert res["k"].klass == ar.IDENTICAL
    assert res["k"].base_insns == res["k"].head_insns == 4
    assert not res["k"].differs


def test_identical_after_masking_when_only_pcrel_literals_moved() -> None:
    res = compare(
        layout({"k": assert_stub(0x336)}),
        layout({"k": assert_stub(0x336)}, assert_at=0x4040),
    )
    assert res["k"].klass == ar.MASKED
    assert res["__assert_fail"].klass == ar.IDENTICAL


def test_line_constant_only_in_an_assert_stub() -> None:
    res = compare(
        layout({"k": assert_stub(0x336)}),
        layout({"k": assert_stub(0x348)}),  # +18
    )
    assert res["k"].klass == ar.LINE_ONLY
    assert res["k"].deltas == [18]
    assert not res["k"].differs


def test_line_constant_only_gfx12_dual_issue_mov() -> None:
    res = compare(
        layout({"k": assert_stub(0x721, "gfx12")}),
        layout({"k": assert_stub(0x739, "gfx12")}),  # +24
    )
    assert res["k"].klass == ar.LINE_ONLY
    assert res["k"].deltas == [24]


def test_assert_fail_itself_may_carry_a_hoisted_line_constant() -> None:
    body = ["v_mov_b32_e32 v8, 0x{:x}", "s_setpc_b64 s[30:31]"]
    base = [("__assert_fail", 0x4000, [body[0].format(0x721), body[1]])]
    head = [("__assert_fail", 0x4000, [body[0].format(0x739), body[1]])]
    res = compare(base, head)
    assert res["__assert_fail"].klass == ar.LINE_ONLY
    assert res["__assert_fail"].deltas == [24]


def test_literal_moved_by_a_delta_outside_the_allowed_set_is_changed() -> None:
    res = compare(
        layout({"k": assert_stub(0x336)}),
        layout({"k": assert_stub(0x340)}),  # +10: not in {11, 18, 24}
    )
    assert res["k"].klass == ar.CHANGED
    assert "+10" in res["k"].detail
    assert "not an allowed shift" in res["k"].detail


def test_allowed_delta_outside_an_assert_stub_is_changed() -> None:
    changed = [i.replace("0x10", "0x28") for i in KERNEL]  # +24, no assert call
    res = compare(layout({"k": KERNEL}), layout({"k": changed}))
    assert res["k"].klass == ar.CHANGED
    assert "outside an assert stub" in res["k"].detail


def test_allowed_delta_in_a_non_mov_instruction_of_a_stub_is_changed() -> None:
    base = [*assert_stub(0x336)[:-1], "s_add_u32 s0, s0, 0x10", "s_endpgm"]
    head = [*assert_stub(0x336)[:-1], "s_add_u32 s0, s0, 0x28", "s_endpgm"]
    res = compare(layout({"k": base}), layout({"k": head}))
    assert res["k"].klass == ar.CHANGED
    assert "not in a mov" in res["k"].detail


def test_two_literals_moving_in_one_instruction_is_changed() -> None:
    # Both halves are movs and both literals move by an allowed delta, but a
    # line constant is a single literal: two moving together is not one.
    dual = "v_dual_mov_b32 v8, 0x{:x} :: v_dual_mov_b32 v9, 0x{:x}"
    base = [*assert_stub(0x336)[:-1], dual.format(0x721, 0x100), "s_endpgm"]
    head = [*assert_stub(0x336)[:-1], dual.format(0x739, 0x118), "s_endpgm"]
    res = compare(layout({"k": base}), layout({"k": head}), shifts=[24])
    assert res["k"].klass == ar.CHANGED
    assert res["k"].detail.startswith("instruction 10:")


def test_different_opcode_or_register_is_changed_even_with_a_line_shift() -> None:
    other = assert_stub(0x348)
    other[3] = "v_mov_b32_e32 v5, 0x348"  # register differs as well
    res = compare(layout({"k": assert_stub(0x336)}), layout({"k": other}))
    assert res["k"].klass == ar.CHANGED
    assert res["k"].detail.startswith("instruction 3:")


def test_instruction_count_change_is_changed_and_counted() -> None:
    longer = [*KERNEL[:-1], "v_add_f32_e32 v2, v1, v1", "s_endpgm"]
    res = compare(layout({"k": KERNEL}), layout({"k": longer}))
    assert res["k"].klass == ar.CHANGED
    assert (res["k"].base_insns, res["k"].head_insns) == (4, 5)


def test_a_changed_callee_is_not_hidden_by_masking() -> None:
    base = layout({"k": assert_stub(0x336)})
    head = layout({"k": assert_stub(0x336)})
    # HEAD's stub calls kernel k2 instead of __assert_fail
    head[0] = (
        "k",
        0x1000,
        [i.replace("__assert_fail", "k2") for i in assert_stub(0x336)],
    )
    head.append(("k2", 0x6000, ASSERT_FAIL_BODY))
    res = compare(base, head)
    assert res["k"].klass == ar.CHANGED


def test_one_sided_functions_and_empty_stubs() -> None:
    res = compare(
        [("only_base", 0x1000, KERNEL), ("stub_base", 0x2000, ["s_endpgm"])],
        [("only_head", 0x1000, KERNEL), ("stub_head", 0x2000, ["s_endpgm", "s_nop 0"])],
    )
    assert res["only_base"].klass == ar.ONLY_BASE and res["only_base"].differs
    assert res["only_head"].klass == ar.ONLY_HEAD and res["only_head"].differs
    assert res["stub_base"].klass == ar.ONLY_BASE and not res["stub_base"].differs
    assert res["stub_head"].klass == ar.ONLY_HEAD and not res["stub_head"].differs


@pytest.mark.parametrize(
    ("demangled", "expected"),
    [
        (
            "void LLGemm1_kernel<c10::Half, 16>(c10::Half const*, int)",
            "LLGemm1_kernel",
        ),
        ("void wvSplitK_hf_<__half, 32, 1, 16, 8, 1, 1>(int, int)", "wvSplitK_hf_"),
        ("LLGemm1_kernel", "LLGemm1_kernel"),
        ("__assert_fail", "__assert_fail"),
        ("int ns::Foo<A<B, C>, 3>::bar<D>(int)", "ns::Foo<A<B, C>, 3>::bar"),
        (
            "void (anonymous namespace)::kern<float>(float*)",
            "(anonymous namespace)::kern",
        ),
        ("void vllm::fn(int)", "vllm::fn"),
        ("", ""),
    ],
)
def test_family(demangled: str, expected: str) -> None:
    assert ar.family(demangled) == expected


# --------------------------------------------------------------------------- #
# EXPECT / MAY_DIFFER
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("expect", "may_differ", "arch", "name", "allowed"),
    [
        (None, None, "gfx1030", "void LLGemm1_kernel<int>()", False),
        (None, ["gfx1030"], "gfx1030", "void LLGemm1_kernel<int>()", False),
        ("LLGemm1_kernel", None, "gfx90a", "void LLGemm1_kernel<int>()", True),
        ("LLGemm1_kernel", ["gfx1030"], "gfx1030", "void LLGemm1_kernel<int>()", True),
        ("LLGemm1_kernel", ["gfx1030"], "gfx90a", "void LLGemm1_kernel<int>()", False),
        ("LLGemm1_kernel", ["gfx1030"], "gfx1030", "void wvSplitK_hf_<int>()", False),
        ("LLGemm1_(kernel|other)$", [], "gfx1030", "LLGemm1_kernel", False),
        ("^void LLGemm1", None, "gfx1030", "LLGemm1_kernel", False),
    ],
)
def test_is_expected(
    expect: str | None,
    may_differ: list[str] | None,
    arch: str,
    name: str,
    allowed: bool,
) -> None:
    pattern = re.compile(expect) if expect else None
    assert ar.is_expected(arch, name, pattern, may_differ) is allowed


def test_split_list_accepts_semicolons_commas_and_blanks() -> None:
    assert ar.split_list("gfx1030;gfx1100, gfx1200  gfx90a") == [
        "gfx1030",
        "gfx1100",
        "gfx1200",
        "gfx90a",
    ]
    assert ar.split_list("") == []
    assert ar.split_list(None) == []


# --------------------------------------------------------------------------- #
# Verdict and report
# --------------------------------------------------------------------------- #

ARCHES6 = ("gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100", "gfx1200")
LLGEMM = [
    f"void LLGemm1_kernel<{t}, {n}>({t} const*, int)"
    for t in ("c10::Half", "c10::BFloat16")
    for n in (2, 4, 8, 16)
]


def fr(
    name: str, klass: str, base: int | None = 10, head: int | None = 12, **kw: Any
) -> Any:
    return ar.FunctionResult(name, klass, base, head, demangled=name, **kw)


def cell(obj: str, arch: str, results: list[Any], **kw: Any) -> Any:
    return ar.ArchObject(
        obj, arch, ar.STATUS_COMPARED, results, len(results), len(results), **kw
    )


def pr54_analysis() -> Any:
    """The shape of the PR 54 result: 8 CHANGED on three arches, line-only on others."""
    obj = "CMakeFiles/_rocm_C_skinny.dir/csrc/rocm/skinny_gemms.hip.o"
    analysis = ar.ObjectAnalysis(obj)
    for arch in ARCHES6:
        results = [
            fr("void wvSplitK_hf_<__half, 32>(int)", ar.LINE_ONLY, deltas=[24]),
            fr("__assert_fail", ar.IDENTICAL),
        ]
        if arch in ("gfx1030", "gfx1100", "gfx1200"):
            results += [fr(name, ar.CHANGED) for name in LLGEMM]
        analysis.cells.append(cell(obj, arch, results))
    return analysis


PLAN = {
    "base": "origin/rdna_extras (1136bb952b)",
    "head": "origin/port/faisal-skinny-wave32 (8efea0ee37)",
    "line_shifts": [11, 18, 24],
    "line_shifts_source": "derived from the source diff",
    "changed_files": ["csrc/rocm/skinny_gemms.cu"],
    "source_diff": {
        "csrc/rocm/skinny_gemms.cu": {"hunks": 3, "added": 24, "removed": 0}
    },
    "units": [
        {
            "source": "csrc/rocm/skinny_gemms.cu",
            "object": "CMakeFiles/_rocm_C_skinny.dir/csrc/rocm/skinny_gemms.hip.o",
            "target": "_rocm_C_skinny",
            "reasons": ["csrc/rocm/skinny_gemms.cu"],
        }
    ],
    "notes": [],
}


def report(analyses: list[Any], **kw: Any) -> dict[str, Any]:
    return ar.build_report(analyses, ARCHES6, plan=PLAN, **kw)


def test_pr54_shape_passes_with_expect_and_may_differ() -> None:
    rep = report(
        [pr54_analysis()],
        expect="LLGemm1_kernel",
        may_differ=["gfx1030", "gfx1100", "gfx1200"],
    )
    assert rep["exit"] == ar.EXIT_OK
    assert rep["expected"] == 24 and rep["unexpected"] == 0
    for arch in ("gfx1030", "gfx1100", "gfx1200"):
        assert rep["arch"][arch]["verdict"] == "expected only"
        assert rep["arch"][arch]["counts"][ar.CHANGED] == 8
        (family,) = rep["arch"][arch]["families"]
        assert family["family"] == "LLGemm1_kernel" and family["functions"] == 8
    for arch in ("gfx90a", "gfx942", "gfx950"):
        assert rep["arch"][arch]["verdict"] == "no difference"
        assert rep["arch"][arch]["counts"][ar.CHANGED] == 0


@pytest.mark.parametrize("may_differ", [None, ["gfx1030", "gfx1100", "gfx1200"]])
def test_pr54_shape_fails_without_expect(may_differ: list[str] | None) -> None:
    rep = report([pr54_analysis()], expect=None, may_differ=may_differ)
    assert rep["exit"] == ar.EXIT_DIFFERS
    assert rep["unexpected"] == 24


def test_may_differ_without_expect_is_called_out_in_the_report() -> None:
    lone = report([pr54_analysis()], expect=None, may_differ=["gfx1030"])
    assert any("MAY_DIFFER only narrows EXPECT" in n for n in lone["notes"])
    both = report([pr54_analysis()], expect="LLGemm1_kernel", may_differ=["gfx1030"])
    assert not any("MAY_DIFFER only narrows" in n for n in both["notes"])
    neither = report([pr54_analysis()])
    assert not any("MAY_DIFFER only narrows" in n for n in neither["notes"])


def test_arch_outside_may_differ_must_have_no_difference() -> None:
    rep = report(
        [pr54_analysis()],
        expect="LLGemm1_kernel",
        may_differ=["gfx1030", "gfx1100"],
    )
    assert rep["exit"] == ar.EXIT_DIFFERS
    assert rep["arch"]["gfx1200"]["verdict"] == "UNEXPECTED"
    assert rep["arch"]["gfx1030"]["verdict"] == "expected only"
    assert rep["unexpected"] == 8


def test_a_function_outside_expect_fails_even_on_a_may_differ_arch() -> None:
    analysis = pr54_analysis()
    analysis.cells[3].results.append(
        fr("void wvSplitK_hf_<__half, 64>(int)", ar.CHANGED)
    )
    rep = report(
        [analysis],
        expect="LLGemm1_kernel",
        may_differ=["gfx1030", "gfx1100", "gfx1200"],
    )
    assert rep["exit"] == ar.EXIT_DIFFERS
    assert rep["unexpected"] == 1
    bad = [d for d in rep["arch"]["gfx1030"]["differences"] if not d["expected"]]
    assert [d["demangled"] for d in bad] == ["void wvSplitK_hf_<__half, 64>(int)"]


def test_all_identical_passes_without_expect() -> None:
    obj = "CMakeFiles/_C.dir/csrc/sampler.hip.o"
    analysis = ar.ObjectAnalysis(obj)
    for arch in ARCHES6:
        analysis.cells.append(cell(obj, arch, [fr("k", ar.IDENTICAL)]))
    rep = report([analysis])
    assert rep["exit"] == ar.EXIT_OK
    assert rep["expected"] == rep["unexpected"] == 0
    assert "nothing differs" in ar.result_line(rep)


def test_contract_arches_left_out_of_arches_are_listed_as_not_run() -> None:
    obj = "CMakeFiles/_C.dir/csrc/sampler.hip.o"
    analysis = ar.ObjectAnalysis(obj)
    for arch in ("gfx1030", "gfx1100"):
        analysis.cells.append(cell(obj, arch, [fr("k", ar.IDENTICAL)]))
    rep = ar.build_report([analysis], ("gfx1030", "gfx1100"), plan=PLAN)
    assert rep["exit"] == ar.EXIT_OK
    assert rep["contract_not_run"] == ["gfx90a", "gfx942", "gfx950", "gfx1200"]
    assert ar.result_line(rep).endswith(
        "contract arches NOT RUN: gfx90a gfx942 gfx950 gfx1200"
    )
    text = ar.render_markdown(rep)
    assert "Platforms run: gfx1030 gfx1100." in text
    assert (
        "Contract platforms not run: gfx90a gfx942 gfx950 gfx1200 (not in ARCHES): "
        "not run is not a pass." in text
    )
    full = ar.build_report([], ARCHES6, plan=PLAN)
    assert full["contract_not_run"] == []
    assert "No contract platform was left out." in ar.render_markdown(full)


def test_no_translation_unit_is_a_pass_that_says_so() -> None:
    rep = ar.build_report([], ARCHES6, plan={"notes": ["n"], "units": []})
    assert rep["exit"] == ar.EXIT_OK
    assert "no translation unit" in ar.result_line(rep)
    assert all(rep["arch"][a]["objects_compared"] == 0 for a in ARCHES6)


def test_an_arch_without_a_code_object_is_not_a_pass() -> None:
    obj = "CMakeFiles/_rocm_C.dir/csrc/rocm/k.hip.o"
    analysis = ar.ObjectAnalysis(obj)
    for arch in ARCHES6:
        if arch == "gfx950":
            analysis.cells.append(ar.ArchObject(obj, arch, ar.STATUS_NO_CODE))
        else:
            analysis.cells.append(cell(obj, arch, [fr("k", ar.IDENTICAL)]))
    rep = report([analysis])
    assert rep["exit"] == ar.EXIT_ERROR
    assert rep["not_run"] == [[obj, "gfx950"]]
    assert rep["arch"]["gfx950"]["verdict"] == "NOT RUN"
    assert "ERROR" in ar.result_line(rep) and "gfx950" in ar.result_line(rep)
    # not run outranks a difference
    analysis.cells[0].results.append(fr("x", ar.CHANGED))
    assert report([analysis])["exit"] == ar.EXIT_ERROR


def test_one_sided_functions_count_but_empty_stubs_do_not() -> None:
    obj = "CMakeFiles/_rocm_C.dir/csrc/rocm/k.hip.o"
    analysis = ar.ObjectAnalysis(obj)
    for arch in ARCHES6:
        results = [
            fr("added_kernel", ar.ONLY_HEAD, None, 40),
            fr("stub_kernel", ar.ONLY_HEAD, None, 1, empty=True),
        ]
        analysis.cells.append(cell(obj, arch, results))
    rep = report([analysis])
    assert rep["exit"] == ar.EXIT_DIFFERS
    a = rep["arch"]["gfx1030"]
    assert [d["name"] for d in a["differences"]] == ["added_kernel"]
    assert [d["name"] for d in a["empty_stubs"]] == ["stub_kernel"]
    assert a["counts"][ar.ONLY_HEAD] == 2


def test_stub_only_objects_are_listed_per_arch() -> None:
    obj = "CMakeFiles/_rocm_C.dir/csrc/rocm/gfx9_only.hip.o"
    analysis = ar.ObjectAnalysis(obj)
    for arch in ARCHES6:
        analysis.cells.append(
            cell(obj, arch, [fr("k", ar.IDENTICAL)], stub_only=arch == "gfx1030")
        )
    rep = report([analysis])
    assert rep["arch"]["gfx1030"]["stub_only_objects"] == [obj]
    assert rep["arch"]["gfx90a"]["stub_only_objects"] == []
    assert "Compiles to empty stubs only" in ar.render_markdown(rep)


def test_expect_that_matches_nothing_adds_a_note() -> None:
    obj = "CMakeFiles/_C.dir/csrc/sampler.hip.o"
    analysis = ar.ObjectAnalysis(obj)
    for arch in ARCHES6:
        analysis.cells.append(cell(obj, arch, [fr("k", ar.IDENTICAL)]))
    rep = report([analysis], expect="LLGemm1_kernel")
    assert rep["exit"] == ar.EXIT_OK
    assert any("matched no differing function" in n for n in rep["notes"])


def test_host_only_and_one_sided_objects_are_reported() -> None:
    host = ar.ObjectAnalysis("CMakeFiles/_C.dir/csrc/host.cpp.o", host_only=True)
    added = ar.ObjectAnalysis("CMakeFiles/_C.dir/csrc/new.hip.o", missing="base")
    for arch in ARCHES6:
        added.cells.append(
            cell(added.object, arch, [fr("new_kernel", ar.ONLY_HEAD, None, 30)])
        )
    rep = report([host, added])
    assert rep["host_only"] == [host.object]
    assert rep["one_sided"] == {added.object: "base"}
    text = ar.render_markdown(rep)
    assert "Host-only objects" in text and "csrc/host.cpp.o" in text
    assert "exists on the HEAD side only" in text


def test_markdown_pr54_report_has_every_required_fact() -> None:
    rep = report(
        [pr54_analysis()],
        expect="LLGemm1_kernel",
        may_differ=["gfx1030", "gfx1100", "gfx1200"],
        info={"phases": "build-base=1s", "disk before cleanup": "3 MiB"},
    )
    text = ar.render_markdown(rep)
    assert "**Result: PASS (exit 0)" in text
    assert "- BASE: `origin/rdna_extras (1136bb952b)`" in text
    assert "- EXPECT: `LLGemm1_kernel`" in text
    assert "- MAY_DIFFER: gfx1030 gfx1100 gfx1200" in text
    assert (
        "Allowed `__LINE__` shifts: +11, +18, +24 (derived from the source diff)"
        in text
    )
    for arch in ARCHES6:
        assert f"### {arch}:" in text
        assert re.search(rf"^\| {arch} \| 1 object\(s\) \|", text, re.M)
    assert "- CHANGED family `LLGemm1_kernel`: 8 function(s)" in text
    assert "[expected]" in text and "UNEXPECTED" not in text.replace("EXPECT / MAY", "")
    assert "line-constant-only (1 function(s)" in text
    assert "skinny_gemms.cu` (+24 -0, 3 hunk(s))" in text
    assert "## How functions are compared" in text
    assert "- phases: build-base=1s" in text
    assert "no GPU was touched" in text


def test_markdown_marks_unexpected_functions_and_caps_long_lists() -> None:
    obj = "CMakeFiles/_rocm_C.dir/csrc/rocm/k.hip.o"
    analysis = ar.ObjectAnalysis(obj)
    for arch in ARCHES6:
        many = [
            fr(f"void fn<{i}>(int)", ar.CHANGED, detail="instruction 3: 'a' -> 'b'")
            for i in range(30)
        ]
        analysis.cells.append(cell(obj, arch, many))
    text = ar.render_markdown(report([analysis]))
    assert "**UNEXPECTED**" in text
    assert "... and 18 more" in text
    assert "30 UNEXPECTED" in text
    assert "instruction 3: 'a' -> 'b'" in text


# --------------------------------------------------------------------------- #
# ELF, tool wrappers and the compare command
# --------------------------------------------------------------------------- #


def test_elf_section_finds_named_sections_and_rejects_junk() -> None:
    blob = make_elf({".text": b"abc", ".hip_fatbin": b"PAYLOAD"})
    assert ar.elf_section(blob, ".hip_fatbin") == b"PAYLOAD"
    assert ar.elf_section(blob, ".text") == b"abc"
    assert ar.elf_section(blob, ".nope") is None
    with pytest.raises(ar.ToolError):
        ar.elf_section(b"not an elf at all", ".text")


def _write_objects(
    root: Path, rel: str, fat: bytes | None, plain: bytes | None = None
) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fat if fat is not None else (plain or make_elf({".text": b"x"})))


OBJ = "CMakeFiles/_rocm_C_skinny.dir/csrc/rocm/skinny_gemms.hip.o"


LLGEMM_MANGLED = "_Z10LLGemm1_kernelIN3c104HalfELi2EEvv"


def _pr54_like_objects(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Objects for gfx1030 and gfx90a; HEAD grows LLGemm1_kernel on gfx1030 only.

    The assert stub's __LINE__ moves by 18 on both arches.
    """
    base_dir, head_dir = tmp_path / "obj-base", tmp_path / "obj-head"
    kernel = ["v_mov_b32_e32 v0, 0x10", "v_add_f32_e32 v1, v0, v0", "s_endpgm"]
    grown = [*kernel[:-1], "v_mul_f32_e32 v2, v1, v1", "s_endpgm"]
    for root, line, gfx1030_kernel in (
        (base_dir, 0x336, kernel),
        (head_dir, 0x348, grown),
    ):
        stub = layout({"_Z4normv": assert_stub(line)})
        per_target = {
            target("gfx1030"): disassembly(
                stub + [(LLGEMM_MANGLED, 0x6000, gfx1030_kernel)]
            ),
            target("gfx90a"): disassembly(stub + [(LLGEMM_MANGLED, 0x6000, kernel)]),
            "host-x86_64-unknown-linux-gnu-": "",
        }
        _write_objects(root, OBJ, fat_object(per_target))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({**PLAN, "line_shifts": [18]}))
    return base_dir, head_dir, plan


def _run_compare(
    tmp_path: Path, plan: Path, *extra: str, arches: str = "gfx1030;gfx90a"
) -> int:
    return ar.main(
        [
            "compare",
            "--base-dir", str(tmp_path / "obj-base"),
            "--head-dir", str(tmp_path / "obj-head"),
            "--plan", str(plan),
            "--arches", arches,
            "--out", str(tmp_path / "out"),
            "--tmp", str(tmp_path / "scratch"),
            *extra,
        ]
    )  # fmt: skip


def test_compare_end_to_end_with_fake_tools(
    tmp_path: Path, fake_rocm: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, plan = _pr54_like_objects(tmp_path)
    code = _run_compare(
        tmp_path, plan, "--expect", "LLGemm1_kernel", "--may-differ", "gfx1030"
    )
    assert code == ar.EXIT_OK
    out = json.loads((tmp_path / "out" / "archdiff.json").read_text())
    g1030, g90a = out["arch"]["gfx1030"], out["arch"]["gfx90a"]
    assert g1030["counts"][ar.CHANGED] == 1
    assert g1030["counts"][ar.LINE_ONLY] == 1
    assert g1030["differences"][0]["demangled"] == "void LLGemm1_kernel<c10::Half, 2>()"
    assert g1030["differences"][0]["family"] == "LLGemm1_kernel"
    assert g90a["counts"][ar.CHANGED] == 0
    assert g90a["counts"][ar.LINE_ONLY] == 1
    assert "LLVM version 23.0.0fake" in out["info"]["llvm-objdump"]
    assert out["info"]["clang-offload-bundler"].endswith("clang-offload-bundler")
    md = (tmp_path / "out" / "archdiff.md").read_text()
    assert capsys.readouterr().out.rstrip("\n") == md.rstrip("\n")
    assert "PASS (exit 0)" in md
    assert not list((tmp_path / "scratch").iterdir())  # temp files cleaned up


def test_compare_exits_1_without_expect_and_for_arches_outside_may_differ(
    tmp_path: Path, fake_rocm: Path
) -> None:
    _, _, plan = _pr54_like_objects(tmp_path)
    assert _run_compare(tmp_path, plan) == ar.EXIT_DIFFERS
    assert _run_compare(tmp_path, plan, "--expect", "LLGemm1_kernel") == ar.EXIT_OK
    # gfx1030 may not differ although EXPECT matches
    assert (
        _run_compare(
            tmp_path, plan, "--expect", "LLGemm1_kernel", "--may-differ", "gfx90a"
        )
        == ar.EXIT_DIFFERS
    )
    # the line-shift list decides line-constant-only: without 18 the stub is CHANGED
    assert (
        _run_compare(tmp_path, plan, "--expect", "LLGemm1_kernel", "--line-shifts", "5")
        == ar.EXIT_DIFFERS
    )


def test_compare_exits_2_when_an_arch_is_missing_from_the_object(
    tmp_path: Path, fake_rocm: Path
) -> None:
    _, _, plan = _pr54_like_objects(tmp_path)
    code = _run_compare(
        tmp_path, plan, "--expect", "LLGemm1_kernel", arches="gfx1030;gfx950"
    )
    assert code == ar.EXIT_ERROR
    out = json.loads((tmp_path / "out" / "archdiff.json").read_text())
    assert out["not_run"] == [[OBJ, "gfx950"]]


def test_compare_treats_a_missing_object_as_all_one_sided(
    tmp_path: Path, fake_rocm: Path
) -> None:
    _, _, plan = _pr54_like_objects(tmp_path)
    (tmp_path / "obj-base" / OBJ).unlink()
    code = _run_compare(tmp_path, plan, arches="gfx1030")
    assert code == ar.EXIT_DIFFERS
    out = json.loads((tmp_path / "out" / "archdiff.json").read_text())
    assert out["one_sided"] == {OBJ: "base"}
    counts = out["arch"]["gfx1030"]["counts"]
    assert counts[ar.ONLY_HEAD] > 0 and counts[ar.CHANGED] == 0


def test_compare_lists_host_only_objects(tmp_path: Path, fake_rocm: Path) -> None:
    rel = "CMakeFiles/_C.dir/csrc/host.cpp.o"
    for side in ("obj-base", "obj-head"):
        _write_objects(tmp_path / side, rel, None)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({**PLAN, "units": [{"source": "csrc/host.cpp", "object": rel,
                                       "target": "_C", "reasons": ["csrc/host.cpp"]}]})
    )  # fmt: skip
    assert _run_compare(tmp_path, plan) == ar.EXIT_OK
    out = json.loads((tmp_path / "out" / "archdiff.json").read_text())
    assert out["host_only"] == [rel]


def test_compare_without_the_bundler_is_an_error_not_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, plan = _pr54_like_objects(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    monkeypatch.setattr(ar, "_rocm_bin_dirs", lambda: [str(tmp_path / "norocm")])
    assert _run_compare(tmp_path, plan) == ar.EXIT_ERROR


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_find_tool_prefers_the_rocm_llvm_dir_over_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rocm = tmp_path / "rocm"
    llvm = rocm / "lib" / "llvm" / "bin"
    in_rocm = _executable(llvm / "llvm-objdump")
    cxxfilt = _executable(llvm / "llvm-cxxfilt")
    for name in ("llvm-objdump", "llvm-cxxfilt", "c++filt"):  # distro copies
        _executable(tmp_path / "distro" / name)
    on_path = _executable(tmp_path / "distro" / "only-on-path")
    monkeypatch.setenv("PATH", str(tmp_path / "distro"))
    monkeypatch.setenv("ROCM", str(rocm))
    for var in ("ROCM_PATH", "HIP_PATH", "ROCM_HOME"):
        monkeypatch.delenv(var, raising=False)
    assert ar.find_tool("llvm-objdump") == str(in_rocm)
    # Every ROCm dir is searched before PATH, whatever the order of the names.
    assert ar.find_tool("c++filt", "llvm-cxxfilt") == str(cxxfilt)
    assert ar.find_tool("only-on-path") == str(on_path)
    assert ar.find_tool("no-such-tool-anywhere") is None


# --------------------------------------------------------------------------- #
# plan command
# --------------------------------------------------------------------------- #


def _write_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _compdb(
    build: Path, tree: Path, hip: Sequence[str], cpp: Sequence[str] = ()
) -> Path:
    entries = [
        {
            "directory": str(build),
            "file": str(build / name),
            "output": f"CMakeFiles/_rocm_C.dir/{name}.o",
        }
        for name in hip
    ] + [
        {
            "directory": str(build),
            "file": str(tree / name),
            "output": f"CMakeFiles/_C.dir/{name}.o",
        }
        for name in cpp
    ]
    path = build.parent / f"compdb-{build.name}.json"
    path.write_text(json.dumps(entries))
    return path


def _plan_args(
    tmp_path: Path, base: Path, head: Path, tree: Path, compdb: Path | None, *extra: str
) -> list[str]:
    args = [
        "plan",
        "--base-tree", str(base),
        "--head-tree", str(head),
        "--base-label", "B",
        "--head-label", "H",
        "--build-dir", str(tmp_path / "build"),
        "--tree-dir", str(tree),
        "--plan-out", str(tmp_path / "plan.json"),
        "--objects-out", str(tmp_path / "objects.txt"),
        "--diff-out", str(tmp_path / "source.diff"),
    ]  # fmt: skip
    if compdb is not None:
        args += ["--compdb", str(compdb)]
    return args + list(extra)


COMMON = {
    "csrc/rocm/a.cu": '#include "common.h"\n__global__ void a() {}\n',
    "csrc/rocm/b.cu": "__global__ void b() {}\n",
    "csrc/rocm/common.h": "#pragma once\nint x;\n",
    "csrc/host.cpp": '#include "rocm/common.h"\nint main() {}\n',
}
OBJ_A = "CMakeFiles/_rocm_C.dir/csrc/rocm/a.hip.o"
OBJ_B = "CMakeFiles/_rocm_C.dir/csrc/rocm/b.hip.o"
OBJ_H = "CMakeFiles/_C.dir/csrc/host.cpp.o"


def test_plan_selects_units_that_include_a_changed_header(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _write_tree(tmp_path / "base", COMMON)
    head = _write_tree(
        tmp_path / "head",
        {**COMMON, "csrc/rocm/common.h": "#pragma once\nint x;\nint y;\n"},
    )
    (tmp_path / "build").mkdir()
    compdb = _compdb(
        tmp_path / "build",
        base,
        ["csrc/rocm/a.hip", "csrc/rocm/b.hip"],
        ["csrc/host.cpp"],
    )
    code = ar.main(_plan_args(tmp_path, base, head, base, compdb))
    assert code == 0
    assert capsys.readouterr().out.strip() == "graph"
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["changed_files"] == ["csrc/rocm/common.h"]
    assert {u["object"] for u in plan["units"]} == {OBJ_A, OBJ_H}
    assert all(u["reasons"] == ["csrc/rocm/common.h"] for u in plan["units"])
    assert plan["line_shifts"] == [1]
    assert plan["mode"] == "auto"
    assert (tmp_path / "objects.txt").read_text().split() == sorted([OBJ_A, OBJ_H])
    assert "+int y;" in (tmp_path / "source.diff").read_text()


def test_plan_without_changes_prints_empty_and_needs_no_graph(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _write_tree(tmp_path / "base", COMMON)
    head = _write_tree(tmp_path / "head", {**COMMON, "README.md": "docs only\n"})
    code = ar.main(_plan_args(tmp_path, base, head, base, None))
    assert code == 0
    assert capsys.readouterr().out.strip() == "empty"
    assert json.loads((tmp_path / "plan.json").read_text())["units"] == []


def test_plan_explicit_tus_all_and_their_exclusion(tmp_path: Path) -> None:
    base = _write_tree(tmp_path / "base", COMMON)
    head = _write_tree(tmp_path / "head", COMMON)
    (tmp_path / "build").mkdir()
    compdb = _compdb(
        tmp_path / "build",
        base,
        ["csrc/rocm/a.hip", "csrc/rocm/b.hip"],
        ["csrc/host.cpp"],
    )
    assert (
        ar.main(
            _plan_args(tmp_path, base, head, base, compdb, "--tus", "./csrc/rocm/b.cu")
        )
        == 0
    )
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert [u["object"] for u in plan["units"]] == [OBJ_B]
    assert plan["mode"] == "TUS"
    assert ar.main(_plan_args(tmp_path, base, head, base, compdb, "--all")) == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert {u["object"] for u in plan["units"]} == {OBJ_A, OBJ_B, OBJ_H}
    both = _plan_args(
        tmp_path, base, head, base, compdb, "--all", "--tus", "csrc/rocm/a.cu"
    )
    assert ar.main(both) == ar.EXIT_ERROR


def test_plan_two_phase_merge_handles_added_and_removed_units(tmp_path: Path) -> None:
    base_files = {**COMMON, "csrc/rocm/gone.cu": "__global__ void g() {}\n"}
    head_files = {
        **COMMON,
        "csrc/rocm/a.cu": '#include "common.h"\n__global__ void a() { }\n',
        "csrc/rocm/new.cu": "__global__ void n() {}\n",
    }
    # gone.cu is deleted in HEAD, new.cu is added, a.cu is edited
    base = _write_tree(tmp_path / "base", base_files)
    head = _write_tree(tmp_path / "head", head_files)
    (tmp_path / "build").mkdir()
    obj_gone = "CMakeFiles/_rocm_C.dir/csrc/rocm/gone.hip.o"
    obj_new = "CMakeFiles/_rocm_C.dir/csrc/rocm/new.hip.o"
    # phase 1: the BASE graph has gone.cu and a.cu, no new.cu
    compdb = _compdb(
        tmp_path / "build", base, ["csrc/rocm/a.hip", "csrc/rocm/gone.hip"]
    )
    assert ar.main(_plan_args(tmp_path, base, head, base, compdb)) == 0
    first = json.loads((tmp_path / "plan.json").read_text())
    assert {u["object"] for u in first["units"]} == {OBJ_A, obj_gone}
    previous = tmp_path / "plan-base.json"
    previous.write_text(json.dumps(first))
    # phase 2: the HEAD graph has new.cu and a.cu, gone.cu was removed
    compdb = _compdb(tmp_path / "build", head, ["csrc/rocm/a.hip", "csrc/rocm/new.hip"])
    assert (
        ar.main(
            _plan_args(tmp_path, base, head, head, compdb, "--previous", str(previous))
        )
        == 0
    )
    merged = json.loads((tmp_path / "plan.json").read_text())
    assert {u["object"] for u in merged["units"]} == {OBJ_A, obj_gone, obj_new}
    # only what HEAD still has is built; gone.cu stays in the plan as base-only
    assert set((tmp_path / "objects.txt").read_text().split()) == {OBJ_A, obj_new}


def test_plan_explicit_tu_in_neither_graph_is_an_error(tmp_path: Path) -> None:
    base = _write_tree(tmp_path / "base", COMMON)
    head = _write_tree(tmp_path / "head", COMMON)
    (tmp_path / "build").mkdir()
    compdb = _compdb(tmp_path / "build", base, ["csrc/rocm/a.hip"])
    args = ["--tus", "csrc/rocm/typo.cu"]
    assert ar.main(_plan_args(tmp_path, base, head, base, compdb, *args)) == 0
    previous = tmp_path / "plan-base.json"
    previous.write_text((tmp_path / "plan.json").read_text())
    code = ar.main(
        _plan_args(
            tmp_path, base, head, head, compdb, *args, "--previous", str(previous)
        )
    )
    assert code == ar.EXIT_ERROR


def test_plan_notes_cmake_changes_and_unreached_csrc_files(tmp_path: Path) -> None:
    base = _write_tree(tmp_path / "base", {**COMMON, "CMakeLists.txt": "project(a)\n"})
    head = _write_tree(
        tmp_path / "head",
        {**COMMON, "CMakeLists.txt": "project(b)\n", "csrc/cuda_only.cu": "x\n"},
    )
    (tmp_path / "build").mkdir()
    compdb = _compdb(tmp_path / "build", base, ["csrc/rocm/a.hip"])
    assert ar.main(_plan_args(tmp_path, base, head, base, compdb)) == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    joined = "\n".join(plan["notes"])
    assert "CMakeLists.txt" in joined and "TUS=" in joined and "ALL=1" in joined
    assert "csrc/cuda_only.cu" in joined
    assert plan["units"] == []


def test_plan_line_shifts_override(tmp_path: Path) -> None:
    base = _write_tree(tmp_path / "base", COMMON)
    head = _write_tree(
        tmp_path / "head",
        {**COMMON, "csrc/rocm/b.cu": "\n\n" + COMMON["csrc/rocm/b.cu"]},
    )
    assert (
        ar.main(_plan_args(tmp_path, base, head, base, None, "--line-shifts", "3,7"))
        == 0
    )
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["line_shifts"] == [3, 7]
    assert plan["line_shifts_source"] == "LINE_SHIFTS"
    assert (
        ar.main(_plan_args(tmp_path, base, head, base, None, "--line-shifts", "x"))
        == ar.EXIT_ERROR
    )
    assert ar.main(_plan_args(tmp_path, base, head, base, None)) == 0
    assert json.loads((tmp_path / "plan.json").read_text())["line_shifts"] == [2]


# --------------------------------------------------------------------------- #
# The driver: usage paths that need no ROCm
# --------------------------------------------------------------------------- #

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")


def run_driver(
    tmp_path: Path, env: dict[str, str], *args: str, script: Path = DRIVER
) -> subprocess.CompletedProcess[str]:
    base_env = {
        "PATH": os.environ["PATH"],
        # user-site packages (regex) live under the real home
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "PYTHON": sys.executable,
        "WORK": str(tmp_path / "work"),
        "BUILD_LOCK": "",
    }
    return subprocess.run(
        ["bash", str(script), *args],
        env={**base_env, **env},
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=300,
    )


@needs_bash
def test_driver_help_documents_the_interface(tmp_path: Path) -> None:
    proc = run_driver(tmp_path, {}, "--help")
    assert proc.returncode == 0
    for word in (
        "BASE",
        "HEAD",
        "ARCHES",
        "TUS",
        "EXPECT",
        "MAY_DIFFER",
        "LINE_SHIFTS",
        "WORK",
        "OUT",
        "JOBS",
        "BUILD_LOCK",
        "VENV",
        "ROCM",
        "KEEP_BUILD",
    ):
        assert word in proc.stdout
    assert "Exit status: 0" in proc.stdout


@needs_bash
@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({}, "BASE and HEAD are required"),
        ({"BASE": "x"}, "BASE and HEAD are required"),
        ({"BASE": ".", "HEAD": ".", "JOBS": "9"}, "refusing JOBS=9"),
        ({"BASE": ".", "HEAD": ".", "JOBS": "0"}, "JOBS must be an integer in 1..8"),
        ({"BASE": ".", "HEAD": ".", "JOBS": "many"}, "JOBS must be an integer in 1..8"),
        ({"BASE": ".", "HEAD": ".", "ARCHES": "gfx1030;sm_90"}, "bad arch 'sm_90'"),
        (
            {"BASE": ".", "HEAD": ".", "TUS": "csrc/a.cu", "ALL": "1"},
            "TUS and ALL are exclusive",
        ),
        (
            {"BASE": ".", "HEAD": ".", "MIN_FREE_GB": "x"},
            "MIN_FREE_GB must be an integer",
        ),
    ],
)
def test_driver_usage_errors_exit_2(
    tmp_path: Path, env: dict[str, str], message: str
) -> None:
    proc = run_driver(tmp_path, env)
    assert proc.returncode == 2, proc.stderr
    assert message in proc.stderr
    assert not (tmp_path / "work").exists()  # nothing was created


@needs_bash
def test_driver_refuses_a_foreign_non_empty_work_dir(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "precious.txt").write_text("mine")
    proc = run_driver(tmp_path, {"BASE": str(tmp_path), "HEAD": str(tmp_path)})
    assert proc.returncode == 2
    assert "no .archdiff marker" in proc.stderr
    assert (work / "precious.txt").read_text() == "mine"


@needs_bash
def test_driver_unresolvable_rev_exits_2(tmp_path: Path) -> None:
    proc = run_driver(
        tmp_path, {"BASE": "no-such-rev-anywhere", "HEAD": "also-not-a-rev"}
    )
    assert proc.returncode == 2
    assert "cannot resolve 'no-such-rev-anywhere'" in proc.stderr


@needs_bash
def test_driver_without_csrc_changes_builds_nothing_and_passes(tmp_path: Path) -> None:
    base = _write_tree(tmp_path / "base", COMMON)
    head = _write_tree(tmp_path / "head", {**COMMON, "README.md": "docs only\n"})
    proc = run_driver(tmp_path, {"BASE": str(base), "HEAD": str(head)})
    assert proc.returncode == 0, proc.stderr
    out = tmp_path / "work" / "results"
    text = (out / "archdiff.md").read_text()
    assert "PASS (exit 0): no translation unit to compare" in text
    assert "none under csrc/, cmake/ or CMakeLists.txt" in text
    assert json.loads((out / "archdiff.json").read_text())["exit"] == 0
    assert "exit=0" in (out / "info.txt").read_text()
    assert (tmp_path / "work" / ".archdiff").exists()
    assert not (tmp_path / "work" / "build").exists()


@needs_bash
def test_driver_csrc_change_without_a_toolchain_is_an_error_not_a_pass(
    tmp_path: Path,
) -> None:
    base = _write_tree(tmp_path / "base", COMMON)
    head = _write_tree(
        tmp_path / "head", {**COMMON, "csrc/rocm/b.cu": "__global__ void b() { }\n"}
    )
    env = {
        "BASE": str(base),
        "HEAD": str(head),
        "ROCM": str(tmp_path / "no-rocm"),
        "VENV": str(tmp_path / "no-venv"),
    }
    proc = run_driver(tmp_path, env)
    assert proc.returncode == 2
    assert "missing:" in proc.stderr
    assert not (tmp_path / "work" / "results" / "archdiff.md").exists()


@needs_bash
def test_driver_hands_the_rocm_root_to_the_helper_even_when_defaulted(
    tmp_path: Path,
) -> None:
    """The ROCM default is a shell variable: the helper only sees it if passed."""
    work = tmp_path / "work"
    for sub in ("obj-base", "obj-head"):
        (work / sub).mkdir(parents=True)
    (work / ".archdiff").touch()
    (work / "plan.json").write_text("{}")
    seen = tmp_path / "seen-rocm"
    spy = tmp_path / "spy"
    spy.write_text("#!/bin/sh\nprintf '%s' \"${ROCM-unset}\" >" + str(seen) + "\n")
    spy.chmod(0o755)
    env = {
        "BASE": str(tmp_path),
        "HEAD": str(tmp_path),
        "COMPARE_ONLY": "1",
        "PYTHON": str(spy),
    }
    proc = run_driver(tmp_path, env)
    assert proc.returncode == 0, proc.stderr
    assert seen.read_text() == "/opt/rocm/core-7.14"
    proc = run_driver(tmp_path, {**env, "ROCM": "/somewhere/rocm"})
    assert proc.returncode == 0, proc.stderr
    assert seen.read_text() == "/somewhere/rocm"


@needs_bash
def test_driver_compare_only_takes_the_rocm_llvm_tools_before_path(
    tmp_path: Path,
) -> None:
    """COMPARE_ONLY re-judges kept objects; ROCM must reach the helper."""
    base_dir, head_dir, plan = _pr54_like_objects(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    (work / ".archdiff").touch()
    shutil.move(str(base_dir), str(work / "obj-base"))
    shutil.move(str(head_dir), str(work / "obj-head"))
    shutil.copy(plan, work / "plan.json")
    llvm = tmp_path / "rocm" / "lib" / "llvm" / "bin"
    llvm.mkdir(parents=True)
    for name, body in (
        ("clang-offload-bundler", FAKE_BUNDLER),
        ("llvm-objdump", FAKE_OBJDUMP),
        ("llvm-cxxfilt", FAKE_FILT),
    ):
        (llvm / name).write_text(body.format(python=sys.executable))
        (llvm / name).chmod(0o755)
    # Older distribution copies earlier on PATH must lose to the ROCm ones.
    distro = tmp_path / "distro"
    distro.mkdir()
    for name, body in (
        ("llvm-objdump", "#!/bin/sh\necho 'distro llvm-objdump used' >&2\nexit 1\n"),
        ("c++filt", "#!/bin/sh\nsed 's/.*/DECOY/'\n"),
    ):
        (distro / name).write_text(body)
        (distro / name).chmod(0o755)
    env = {
        "BASE": str(tmp_path),
        "HEAD": str(tmp_path),
        "COMPARE_ONLY": "1",
        "ARCHES": "gfx1030;gfx90a",
        "ROCM": str(tmp_path / "rocm"),
        "EXPECT": "LLGemm1_kernel",
        "MAY_DIFFER": "gfx1030",
        "PATH": f"{distro}{os.pathsep}{os.environ['PATH']}",
    }
    proc = run_driver(tmp_path, env)
    assert proc.returncode == 0, proc.stderr
    out = json.loads((work / "results" / "archdiff.json").read_text())
    assert out["info"]["llvm-objdump"].startswith(str(llvm))
    names = [d["demangled"] for d in out["arch"]["gfx1030"]["differences"]]
    assert names == ["void LLGemm1_kernel<c10::Half, 2>()"]


@needs_bash
@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
def test_driver_resolves_revs_in_the_repo_that_holds_the_script(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "tools" / "rdna").mkdir(parents=True)
    for src in (DRIVER, HELPER):
        shutil.copy(src, repo / "tools" / "rdna" / src.name)
    _write_tree(repo, COMMON)
    git_env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            env=git_env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    (repo / "README.md").write_text("docs only\n")
    git("add", "README.md")
    git("commit", "-q", "-m", "docs")
    script = repo / "tools" / "rdna" / "archdiff.sh"
    proc = run_driver(tmp_path, {"BASE": "HEAD~1", "HEAD": "main"}, script=script)
    assert proc.returncode == 0, proc.stderr
    text = (tmp_path / "work" / "results" / "archdiff.md").read_text()
    assert "- BASE: `HEAD~1 (" in text and "- HEAD: `main (" in text
    assert "no translation unit to compare" in text
    # a committed script keeps working when BASE and HEAD are the same commit
    env = {"BASE": "main", "HEAD": "main", "WORK": str(tmp_path / "work2")}
    same = run_driver(tmp_path, env, script=script)
    assert same.returncode == 0, same.stderr


def test_the_helper_cli_has_two_subcommands() -> None:
    parser = ar.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    assert (
        parser.parse_args(
            ["compare", "--base-dir", "a", "--head-dir", "b", "--out", "o"]
        ).cmd
        == "compare"
    )
