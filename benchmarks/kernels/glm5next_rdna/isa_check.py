# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1030 device-only compile of the Glm5Next HIP TUs, with register stats.

Runs the build's own hipify on ``csrc/rocm/rdna/glm5next/*.cu``, compiles each
result for gfx1030 with ``clang++ -x hip --cuda-device-only -nogpulib`` and
reads the amdhsa kernel metadata back: VGPR/SGPR counts, LDS, scratch, spills,
workgroup size and the resulting waves per SIMD. No GPU and no device
libraries are needed, only a clang with the AMDGPU backend, the HIP public
headers and a torch install for ``torch/all.h``.

With ROCm installed::

    python -m benchmarks.kernels.glm5next_rdna.isa_check \\
        --rocm-path /opt/rocm --clang /opt/rocm/llvm/bin/clang++

Without ROCm, point it at checkouts of the two header repositories and it
assembles a header-only tree (``hip_version.h`` is generated)::

    git clone --depth 1 --branch rocm-7.2.4 --filter=blob:none --sparse \\
        https://github.com/ROCm/HIP && git -C HIP sparse-checkout set include
    git clone --depth 1 --branch rocm-7.2.4 --filter=blob:none --sparse \\
        https://github.com/ROCm/clr && git -C clr sparse-checkout set hipamd/include
    python -m benchmarks.kernels.glm5next_rdna.isa_check \\
        --hip-src HIP --clr-src clr --hip-version 7.2.4 --markdown isa.md

When the torch install is not a ROCm build (no ``ATen/hip/``), the stand-ins
under ``shim/`` supply the stream and guard declarations the launch code
names; the device code is unaffected. Exit status is the number of TUs that
failed to compile.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
KERNEL_DIR = REPO / "csrc" / "rocm" / "rdna" / "glm5next"
SHIM = HERE / "shim"

# gfx1030: 1024 VGPRs per lane per SIMD, allocated in blocks of 8, at most 16
# waves per SIMD; the architectural ceiling per wave is 256.
VGPR_FILE = 1024
VGPR_GRANULE = 8
MAX_WAVES = 16


def waves_per_simd(vgprs: int) -> int:
    if vgprs <= 0:
        return MAX_WAVES
    alloc = -(-vgprs // VGPR_GRANULE) * VGPR_GRANULE
    return min(MAX_WAVES, VGPR_FILE // alloc)


def parse_kernels(asm: str) -> list[dict[str, str]]:
    """Return the ``amdhsa.kernels`` entries of a ``-S`` output as dicts."""
    _, _, meta = asm.partition("amdhsa.kernels:")
    if not meta:
        return []
    kernels: list[dict[str, str]] = []
    cur: dict[str, str] | None = None
    for raw in meta.splitlines():
        line = raw.strip()
        if line.startswith("amdhsa.target") or line.startswith("amdhsa.version"):
            break
        if line.startswith("- "):
            cur = {}
            kernels.append(cur)
            line = line[2:]
        if cur is None:
            continue
        m = re.match(r"\.([a-z_]+):\s+(.*)$", line)
        if m:
            cur[m.group(1)] = m.group(2).strip()
    return [k for k in kernels if "name" in k]


def short_name(mangled: str) -> str:
    """``_ZN12glm5_dsa_idx25glm5_dsa_first_key_kernelEv`` -> the kernel.

    Itanium names carry ``<length><identifier>`` pieces; the kernel is the
    piece that contains ``kernel``.
    """
    first = None
    pos = 0
    length = re.compile(r"\d+")
    while (m := length.search(mangled, pos)) is not None:
        n = int(m.group())
        ident = mangled[m.end() : m.end() + n]
        if len(ident) != n or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ident):
            pos = m.end()
            continue
        if "kernel" in ident:
            return ident
        first = first or ident
        pos = m.end() + n
    return first or mangled


def row(tu: str, k: dict[str, str]) -> dict[str, int | str]:
    get = lambda key: int(k.get(key, 0))  # noqa: E731
    vgpr = get("vgpr_count")
    return {
        "tu": tu,
        "kernel": short_name(k["name"]),
        "vgpr": vgpr,
        "sgpr": get("sgpr_count"),
        "lds": get("group_segment_fixed_size"),
        "scratch": get("private_segment_fixed_size"),
        "vgpr_spill": get("vgpr_spill_count"),
        "sgpr_spill": get("sgpr_spill_count"),
        "workgroup": get("max_flat_workgroup_size"),
        "wave": get("wavefront_size") or 32,
        "waves_per_simd": waves_per_simd(vgpr),
    }


def markdown(rows: list[dict[str, int | str]]) -> str:
    lines = [
        "| TU | kernel | VGPR | SGPR | LDS B | scratch B | spills V/S | "
        "max WG | waves/SIMD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['tu']}` | `{r['kernel']}` | {r['vgpr']} | {r['sgpr']} | "
            f"{r['lds']} | {r['scratch']} | {r['vgpr_spill']}/{r['sgpr_spill']} | "
            f"{r['workgroup']} | {r['waves_per_simd']} |"
        )
    return "\n".join(lines)


def assemble_header_tree(hip_src: Path, clr_src: Path, version: str, out: Path) -> Path:
    """Merge the two HIP header checkouts into ``<out>/include/hip`` with a
    generated ``hip_version.h``, so ``--rocm-path`` can point at it."""
    inc = out / "include" / "hip"
    inc.mkdir(parents=True, exist_ok=True)
    (out / "lib").mkdir(exist_ok=True)
    for src in (hip_src / "include" / "hip", clr_src / "hipamd" / "include" / "hip"):
        if not src.is_dir():
            raise SystemExit(f"{src} is not a directory")
        for entry in sorted(src.iterdir()):
            dest = inc / entry.name
            if entry.is_dir():
                dest.mkdir(exist_ok=True)
                for sub in sorted(entry.iterdir()):
                    target = dest / sub.name
                    if not target.exists():
                        target.symlink_to(sub)
            elif not dest.exists():
                dest.symlink_to(entry)
    major, minor, patch = (int(x) for x in version.split("."))
    (inc / "hip_version.h").write_text(
        "#pragma once\n"
        f"#define HIP_VERSION_MAJOR {major}\n"
        f"#define HIP_VERSION_MINOR {minor}\n"
        f"#define HIP_VERSION_PATCH {patch}\n"
        '#define HIP_VERSION_GITHASH "isa-check"\n'
        "#define HIP_VERSION_BUILD_ID 0\n"
        '#define HIP_VERSION_BUILD_NAME ""\n'
        "#define HIP_VERSION (HIP_VERSION_MAJOR * 10000000 + "
        "HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)\n"
    )
    return out


def torch_include_dirs() -> list[Path]:
    import torch

    base = Path(torch.__file__).resolve().parent / "include"
    return [base, base / "torch" / "csrc" / "api" / "include"]


def hipify(sources: list[Path], work: Path) -> list[Path]:
    """Run ``cmake/hipify.py`` on copies of the sources; return the .hip files."""
    src_dir = work / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    for extra in KERNEL_DIR.glob("*.cuh"):
        shutil.copy2(extra, src_dir / extra.name)
    copies = []
    for s in sources:
        dest = src_dir / s.name
        shutil.copy2(s, dest)
        copies.append(dest)
    out_dir = work / "hipified"
    cmd = [
        sys.executable,
        str(REPO / "cmake" / "hipify.py"),
        "-p",
        str(src_dir),
        "-o",
        str(out_dir),
        *map(str, copies),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # hipify.py prints its own progress before the one-path-per-source list.
    hipified = [
        Path(line.strip())
        for line in proc.stdout.splitlines()
        if line.strip().endswith((".hip", ".cu")) and Path(line.strip()).exists()
    ]
    if len(hipified) != len(copies):
        raise SystemExit(
            f"hipify returned {len(hipified)} files for {len(copies)} sources:\n"
            + proc.stdout
        )
    return hipified


def compile_one(
    clang: str,
    source: Path,
    asm_out: Path,
    rocm_path: Path,
    include_dirs: list[Path],
    extra_includes: list[Path],
    arch: str,
) -> tuple[bool, str]:
    resource = subprocess.run(
        [clang, "-print-resource-dir"], capture_output=True, text=True, check=True
    ).stdout.strip()
    cmd = [
        clang,
        "-x",
        "hip",
        f"--offload-arch={arch}",
        "--cuda-device-only",
        "-nogpulib",
        "-O3",
        "-std=c++17",
        # The flags the build gives these TUs (cmake/utils.cmake, plus the
        # NDEBUG of a Release/RelWithDebInfo CMake build). NDEBUG matters:
        # without it the HIP 7 `__shfl_xor_sync` mask check stays in and
        # alone pushes the KDA kernels from ~78 to 128 VGPRs with spills.
        "-DNDEBUG",
        "-DUSE_ROCM",
        "-DENABLE_FP8",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-fno-gpu-rdc",
        "-D__HIP_PLATFORM_AMD__=1",
        "-Wno-unknown-warning-option",
        "-Wno-unused-result",
        f"--rocm-path={rocm_path}",
        # Without device libraries clang does not add the wrapper itself.
        "-include",
        f"{resource}/include/__clang_hip_runtime_wrapper.h",
    ]
    for inc in extra_includes:
        cmd += ["-include", str(inc)]
    for d in include_dirs:
        cmd += ["-I", str(d)]
    cmd += ["-S", "-o", str(asm_out), str(source)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, proc.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clang", default=None, help="clang++ with the AMDGPU backend")
    parser.add_argument("--rocm-path", type=Path, default=None)
    parser.add_argument("--hip-src", type=Path, help="ROCm/HIP checkout (include/hip)")
    parser.add_argument(
        "--clr-src", type=Path, help="ROCm/clr checkout (hipamd/include)"
    )
    parser.add_argument("--hip-version", default="7.2.4")
    parser.add_argument("--arch", default="gfx1030")
    parser.add_argument("--markdown", type=Path, help="write the table here too")
    parser.add_argument("--asm-dir", type=Path, help="keep the .s files here")
    parser.add_argument("--work-dir", type=Path, help="scratch dir (default: temp)")
    parser.add_argument(
        "--only", nargs="*", default=None, help="TU basenames (default: all)"
    )
    args = parser.parse_args()

    clang = args.clang
    if clang is None:
        for candidate in (
            os.environ.get("ROCM_PATH", "/opt/rocm") + "/llvm/bin/clang++",
            shutil.which("clang++"),
        ):
            if candidate and Path(candidate).exists():
                clang = candidate
                break
    if clang is None:
        raise SystemExit("no clang++ found; pass --clang")

    work = args.work_dir or Path(tempfile.mkdtemp(prefix="glm5next-isa-"))
    work.mkdir(parents=True, exist_ok=True)

    rocm_path = args.rocm_path
    if rocm_path is None:
        if args.hip_src and args.clr_src:
            rocm_path = assemble_header_tree(
                args.hip_src, args.clr_src, args.hip_version, work / "rocm"
            )
        else:
            env = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
            if (env / "include" / "hip" / "hip_runtime.h").exists():
                rocm_path = env
            else:
                raise SystemExit(
                    "no HIP headers: pass --rocm-path, or --hip-src and --clr-src"
                )

    include_dirs = torch_include_dirs()
    extra_includes = [SHIM / "isa_compat.h"]
    if (
        not (include_dirs[0] / "ATen" / "hip" / "HIPContext.h").exists()
        or not (rocm_path / "include" / "thrust" / "complex.h").exists()
    ):
        include_dirs.insert(0, SHIM)
    include_dirs.append(KERNEL_DIR)
    include_dirs.append(REPO / "csrc")

    sources = sorted(KERNEL_DIR.glob("*.cu"))
    if args.only:
        sources = [s for s in sources if s.stem in set(args.only)]
    hipified = hipify(sources, work)

    asm_dir = args.asm_dir or (work / "asm")
    asm_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | str]] = []
    failures = 0
    for src in hipified:
        asm = asm_dir / (src.stem + ".s")
        ok, err = compile_one(
            clang, src, asm, rocm_path, include_dirs, extra_includes, args.arch
        )
        if not ok:
            failures += 1
            errors = [line for line in err.splitlines() if "error:" in line]
            print(f"FAIL {src.stem}: {len(errors)} errors", file=sys.stderr)
            for line in errors[:5]:
                print("  " + line, file=sys.stderr)
            continue
        rows.extend(row(src.stem, k) for k in parse_kernels(asm.read_text()))

    table = markdown(rows)
    print(table)
    print(f"\n{len(hipified) - failures}/{len(hipified)} TUs compiled for {args.arch}")
    if args.markdown:
        args.markdown.write_text(table + "\n")
    return failures


if __name__ == "__main__":
    sys.exit(main())
