#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fingerprint the device code of one gfx arch inside HIP host objects.

Proves that a source change left the device code of one arch untouched. hipcc
derives a per-TU id (``__hip_cuid_``) from an MD5 of the source path and every
non-input command-line argument, and that id shifts section offsets and
PC-relative immediates, so base and patched objects are only comparable when
they were built from the *same tree path with the same command lines*: build the
base sources in a tree and build dir (``build_rocm_c.sh``), extract, rsync the
patched sources over the same tree, rebuild in place (ninja recompiles only what
changed), extract again:

    code_object_fingerprint.py extract --arch gfx1030 --root BUILD_DIR \
        --out base.json
    ... patch the sources, rebuild BUILD_DIR ...
    code_object_fingerprint.py extract --arch gfx1030 --root BUILD_DIR \
        --out new.json
    code_object_fingerprint.py compare base.json new.json

For every ``*.o`` the embedded fat binary (``.hip_fatbin``) is unbundled with
``clang-offload-bundler`` (it decompresses ``--offload-compress`` bundles), and
the AMDGPU ELF code object of the requested arch is reduced to the hashes of
``.text`` (machine code), ``.rodata`` (kernel descriptors) and ``.note`` (HSA
metadata: register counts, LDS, kernel arguments) plus the sorted kernel symbol
list. DWARF, ``.comment`` and anything that embeds paths or line numbers are left
out on purpose, so moving a kernel by an unrelated edit upstream of it does not
show up as a code change, while any change of an instruction does.

``compare`` classifies every object: IDENTICAL; STUB-CHURN (the code differs but
``.text`` is nothing but ``s_endpgm``/``s_code_end``/``s_nop`` words in both
builds, i.e. empty kernels that exist for symbol parity only, such as a kernel
whose body is compiled out for the arch); CHANGED (real instructions differ);
ONLY-IN-BASE / ONLY-IN-NEW; NEW-STUB (an object only the new build has whose
``.text`` is stubs only: a translation unit that is compiled out for this arch,
such as a gfx1100-only kernel file in a two-arch build). The exit status is 1
when anything is CHANGED or ONLY-IN-*, and also for STUB-CHURN and NEW-STUB with
``--strict``.

Only the Python standard library, ``objcopy`` and ``clang-offload-bundler`` are
needed; no GPU is used.
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROCM_LLVM_BIN = "/opt/rocm/core-7.14/lib/llvm/bin"
SECTIONS = (".text", ".rodata", ".note")
# s_endpgm, s_code_end, s_nop 0 and zero padding: an empty kernel's whole .text.
STUB_WORDS = frozenset({0xBF810000, 0xBF9F0000, 0xBF800000, 0})


def _stub_only(text: bytes) -> bool:
    """True when ``text`` holds no instruction besides the empty-kernel words."""
    usable = len(text) - len(text) % 4
    return {w for (w,) in struct.iter_unpack("<I", text[:usable])} <= STUB_WORDS


def _tool(name: str) -> str:
    found = shutil.which(name) or shutil.which(name, path=ROCM_LLVM_BIN)
    if found is None:
        sys.exit(f"{name} not found on PATH or in {ROCM_LLVM_BIN}")
    return found


def _read_elf_sections(data: bytes) -> dict[str, bytes]:
    """Return {section name: bytes} of a little-endian ELF64 file."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not a little-endian ELF64 file")
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    headers = []
    for i in range(shnum):
        off = shoff + i * shentsize
        name, _type, _flags, _addr, offset, size = struct.unpack_from(
            "<IIQQQQ", data, off
        )
        headers.append((name, offset, size))
    str_off, str_size = headers[shstrndx][1], headers[shstrndx][2]
    strtab = data[str_off : str_off + str_size]
    out = {}
    for name, offset, size in headers:
        end = strtab.index(b"\0", name)
        out[strtab[name:end].decode()] = data[offset : offset + size]
    return out


def _kernel_symbols(sections: dict[str, bytes]) -> list[str]:
    """Names of the defined function symbols in .dynsym (kernels + helpers)."""
    dynsym, dynstr = sections.get(".dynsym", b""), sections.get(".dynstr", b"")
    names = []
    for off in range(0, len(dynsym), 24):
        name, info, _other, shndx, _value, size = struct.unpack_from(
            "<IBBHQQ", dynsym, off
        )
        if shndx != 0 and (info & 0xF) == 2 and size:  # STT_FUNC, defined
            end = dynstr.index(b"\0", name)
            names.append(dynstr[name:end].decode())
    return sorted(names)


def _unbundle(fatbin: Path, arch: str, workdir: Path) -> bytes | None:
    bundler = _tool("clang-offload-bundler")
    listing = subprocess.run(
        [bundler, "--list", "--type=o", f"--input={fatbin}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    target = next((t for t in listing if t.split("--")[-1] == arch), None)
    if target is None:
        return None
    out = workdir / "device.co"
    subprocess.run(
        [
            bundler,
            "--unbundle",
            "--type=o",
            f"--input={fatbin}",
            f"--targets={target}",
            f"--output={out}",
        ],
        check=True,
        capture_output=True,
    )
    return out.read_bytes()


def fingerprint_object(obj: Path, arch: str) -> dict | None:
    """Fingerprint of ``arch`` device code in one host object, or None."""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or None) as tmp:
        work = Path(tmp)
        fatbin = work / "fatbin"
        copy = work / "copy.o"
        proc = subprocess.run(
            [
                _tool("objcopy"),
                f"--dump-section=.hip_fatbin={fatbin}",
                str(obj),
                str(copy),
            ],
            capture_output=True,
        )
        if proc.returncode != 0 or not fatbin.exists() or fatbin.stat().st_size == 0:
            return None
        code = _unbundle(fatbin, arch, work)
    if code is None:
        return None
    sections = _read_elf_sections(code)
    entry = {
        name.lstrip("."): hashlib.sha256(sections.get(name, b"")).hexdigest()
        for name in SECTIONS
    }
    text = sections.get(".text", b"")
    entry["text_bytes"] = len(text)
    entry["stub_only"] = _stub_only(text)
    entry["kernels"] = _kernel_symbols(sections)
    return entry


def cmd_extract(args: argparse.Namespace) -> None:
    root = Path(args.root)
    result = {}
    for obj in sorted(root.rglob("*.o")):
        entry = fingerprint_object(obj, args.arch)
        if entry is not None:
            result[obj.relative_to(root).as_posix()] = entry
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    kernels = sum(len(e["kernels"]) for e in result.values())
    print(f"{args.out}: {len(result)} objects with {args.arch} code, {kernels} symbols")


def cmd_compare(args: argparse.Namespace) -> None:
    base = json.loads(Path(args.base).read_text())
    new = json.loads(Path(args.new).read_text())
    only_base = sorted(set(base) - set(new))
    added = sorted(set(new) - set(base))
    only_new = [key for key in added if new[key].get("stub_only") is not True]
    new_stub = [key for key in added if new[key].get("stub_only") is True]
    churn, changed, same = [], [], 0
    for key in sorted(set(base) & set(new)):
        diff = [
            s
            for s in ("text", "rodata", "note", "kernels")
            if base[key][s] != new[key][s]
        ]
        if not diff:
            same += 1
        elif base[key].get("stub_only") is True and new[key].get("stub_only") is True:
            churn.append((key, diff))
        else:
            changed.append((key, diff))
    print(f"IDENTICAL objects: {same}")
    print(
        f"STUB-CHURN: {len(churn)}  NEW-STUB: {len(new_stub)}  "
        f"CHANGED: {len(changed)}  ONLY-IN-BASE: {len(only_base)}  "
        f"ONLY-IN-NEW: {len(only_new)}"
    )
    for key in only_base:
        print(f"  ONLY-IN-BASE {key}")
    for key in only_new:
        print(f"  ONLY-IN-NEW {key}")
    for key in new_stub:
        print(f"  NEW-STUB {key} ({len(new[key]['kernels'])} kernel symbols)")
    for key, diff in churn:
        grew = len(new[key]["kernels"]) - len(base[key]["kernels"])
        print(f"  STUB-CHURN {key}: {', '.join(diff)} (kernel symbols {grew:+d})")
    for key, diff in changed:
        print(f"  CHANGED {key}: {', '.join(diff)}")
    bad = changed or only_base or only_new or (args.strict and (churn or new_stub))
    sys.exit(1 if bad else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    extract = sub.add_parser("extract", help="fingerprint every *.o under --root")
    extract.add_argument("--arch", required=True, help="e.g. gfx1030")
    extract.add_argument("--root", required=True, help="build directory")
    extract.add_argument("--out", required=True, help="JSON output file")
    extract.set_defaults(func=cmd_extract)
    compare = sub.add_parser("compare", help="compare two extract outputs")
    compare.add_argument("base")
    compare.add_argument("new")
    compare.add_argument(
        "--strict", action="store_true", help="exit 1 on STUB-CHURN as well"
    )
    compare.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
