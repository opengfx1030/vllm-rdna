# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile the W4A8 explore device code for gfx1030 and audit its ISA.

Needs only a clang with the AMDGPU backend (no ROCm, no GPU), e.g. the
distro clang-18. On the V620 box, pass ROCm's clang to audit the compiler
the harness .so is actually built with.

For every sweep config (``W4A8_EXPLORE_CONFIGS`` in w4a8_sdot4.cuh) and
group size it checks the main-loop block against the design budget:

* exactly ``2 * M_TILE * NPT * G / 8`` ``v_dot4`` per group (ConfigH guard);
* no scratch, no SGPR spills (``v_readlane``/``v_writelane``);
* no quarter-rate integer multiplies in the loop;
* A comes from LDS (kLds) or scalar loads (kSmem), as advertised.

Usage (from the repo root)::

    .venv/bin/python -m benchmarks.kernels.w4a8_sdot4_explore.isa_check
    .venv/bin/python -m benchmarks.kernels.w4a8_sdot4_explore.isa_check \\
        --clang /opt/rocm/llvm/bin/clang --markdown isa.md --save-asm isa.s
"""

from __future__ import annotations

import argparse
import collections
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import regex as re

from benchmarks.kernels.w4a8_sdot4_explore import reference as ref

ROOT = Path(__file__).resolve().parents[3]
EXPLORE = ROOT / "csrc" / "rocm" / "explore"
HEADER = EXPLORE / "w4a8_sdot4.cuh"
GROUPS = (32, 64, 128)
PROBE_CHAINS = 8
QUARTER_RATE = ("v_mul_lo_u32", "v_mul_lo_i32", "v_mul_hi_u32", "v_mul_hi_i32")
FMA = ("v_fma_f32", "v_fmac_f32", "v_fma_mix_f32", "v_fmac_legacy_f32")


@dataclass(frozen=True)
class SweepConfig:
    id: int
    name: str
    threads: int
    npt: int
    k_step: int
    m_tile: int
    a_src: str


def sweep_configs(header: Path = HEADER) -> list[SweepConfig]:
    """Parses the ``W4A8_EXPLORE_CONFIGS`` X-macro list from the header."""
    pat = re.compile(
        r'X\((\d+),\s*"(\w+)",\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(k\w+)\)'
    )
    configs = [
        SweepConfig(int(i), n, int(t), int(p), int(k), int(m), s)
        for i, n, t, p, k, m, s in pat.findall(header.read_text())
    ]
    if not configs:
        raise RuntimeError(f"no W4A8_EXPLORE_CONFIGS entries in {header}")
    return configs


def translation_unit() -> str:
    lines = [
        '#include "w4a8_sdot4_isa_shim.h"',
        '#include "w4a8_sdot4.cuh"',
        "namespace ex = vllm::explore_w4a8;",
        "#define W4A8_ISA_X(id, name, th, npt, ks, mt, src) \\",
    ]
    lines += [
        "  W4A8_EXPLORE_INSTANTIATE_GEMM("
        f"ex::Cfg<th, npt, ks, mt, {g}, ex::ASrc::src>) \\"
        for g in GROUPS
    ]
    lines += [
        "",
        "W4A8_EXPLORE_CONFIGS(W4A8_ISA_X)",
    ]
    lines += [
        f"template __global__ void ex::w4a8_act_quant_kernel<256, {mt}>("
        "const ex::f16_t*, int64_t, int8_t*, float*, int32_t*, int, int, int);"
        for mt in (8, 16, 32)
    ]
    lines += [
        f"template __global__ void ex::w4a8_probe_kernel<ex::Probe::{p}, "
        f"{PROBE_CHAINS}>(int, uint32_t, uint32_t*);"
        for p in ("kSdot4", "kFdot2", "kFmaF32")
    ]
    return "\n".join(lines) + "\n"


# Just enough of the HIP host API for w4a8_sdot4_capi.cu to parse.
HIP_STUB = """#pragma once
#include <cstddef>
#include "w4a8_sdot4_isa_shim.h"
typedef struct ihipStream_t* hipStream_t;
enum hipError_t { hipSuccess = 0 };
struct dim3 {
  unsigned x, y, z;
  constexpr dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1)
      : x(a), y(b), z(c) {}
};
struct hipDeviceProp_t { char gcnArchName[256]; };
hipError_t hipGetDevice(int*);
hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int);
hipError_t hipMemsetAsync(void*, int, size_t, hipStream_t);
hipError_t hipGetLastError();
const char* hipGetErrorString(hipError_t);
extern "C" hipError_t __hipPushCallConfiguration(dim3, dim3, size_t = 0,
                                                 hipStream_t = 0);
extern "C" hipError_t __hipPopCallConfiguration(dim3*, dim3*, size_t*,
                                                hipStream_t*);
extern "C" hipError_t hipLaunchKernel(const void*, dim3, dim3, void**, size_t,
                                      hipStream_t);
"""


def check_capi(clang: str) -> list[str]:
    """Compiles the C ABI glue (host and device passes) against a HIP stub.

    Returns the kernel kinds the device pass emitted, e.g. 21 gemm kernels.
    """
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "hip").mkdir()
        (Path(tmp) / "hip" / "hip_runtime.h").write_text(HIP_STUB)
        base = [
            clang,
            "-x",
            "hip",
            "--offload-arch=gfx1030",
            "-nogpulib",
            "-nogpuinc",
            "-std=c++17",
            f"-I{tmp}",
            f"-I{EXPLORE}",
        ]
        capi = str(EXPLORE / "w4a8_sdot4_capi.cu")
        subprocess.run([*base, "--cuda-host-only", "-fsyntax-only", capi], check=True)
        out = Path(tmp) / "capi.s"
        subprocess.run(
            [*base, "--cuda-device-only", "-O3", "-S", "-o", str(out), capi],
            check=True,
        )
        names = re.findall(r"^\s+\.amdhsa_kernel (\S+)", out.read_text(), re.M)
    return [
        next(k for k in ("gemm", "act_quant", "probe") if f"w4a8_{k}_kernel" in n)
        for n in names
    ]


def compile_asm(clang: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tu = Path(tmp) / "w4a8_isa_tu.hip"
        out = Path(tmp) / "w4a8_isa_tu.s"
        tu.write_text(translation_unit())
        cmd = [
            clang,
            "-x",
            "hip",
            "--cuda-device-only",
            "--offload-arch=gfx1030",
            "-nogpulib",
            "-nogpuinc",
            "-O3",
            "-std=c++17",
            f"-I{EXPLORE}",
            "-S",
            "-o",
            str(out),
            str(tu),
        ]
        subprocess.run(cmd, check=True)
        return out.read_text()


@dataclass
class Kernel:
    name: str
    blocks: dict[str, list[str]] = field(default_factory=dict)
    info: dict[str, int] = field(default_factory=dict)

    def main_block(self, prefix: str) -> list[str]:
        """The block with the most instructions starting with ``prefix``."""
        return max(
            self.blocks.values(),
            key=lambda ins: sum(i.startswith(prefix) for i in ins),
        )


def parse_asm(asm: str) -> list[Kernel]:
    kernels: list[Kernel] = []
    by_name: dict[str, Kernel] = {}
    cur: Kernel | None = None
    block = "entry"
    info_for: Kernel | None = None
    for raw in asm.splitlines():
        line = raw.strip()
        head = re.match(r"^(_Z\S+):", raw)
        if head:
            cur = Kernel(head.group(1))
            kernels.append(cur)
            by_name[cur.name] = cur
            block = "entry"
            cur.blocks[block] = []
            continue
        size = re.match(r"^\.size\s+(_Z\S+),", line)
        if size:
            info_for = by_name.get(size.group(1))
            cur = None
            continue
        if cur is not None:
            label = re.match(r"^(\.LBB\d+_\d+):", raw)
            if label:
                block = label.group(1)
                cur.blocks[block] = []
            elif line and not line.startswith((";", ".")):
                cur.blocks[block].append(line.split()[0])
            continue
        kv = re.match(r"^; (NumVgprs|NumSgprs|ScratchSize|Occupancy):\s*(\d+)", line)
        if kv and info_for is not None:
            info_for.info.setdefault(kv.group(1), int(kv.group(2)))
    return kernels


GEMM_RE = re.compile(r"CfgILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELNS\d+_4ASrcE(\d)E")


@dataclass
class Row:
    config: str
    group: int
    info: dict[str, int]
    counts: collections.Counter
    dot4: int
    non_dot_valu: int
    budget: ref.LoopBudget | None
    failures: list[str]

    @property
    def isa_ratio(self) -> float:
        """W4A16 budget / measured W4A8 VALU per group (VALU-bound ceiling)."""
        assert self.budget is not None
        return self.budget.w4a16 / max(1, self.dot4 + self.non_dot_valu)


def audit_gemm(k: Kernel, cfg: SweepConfig, group: int) -> Row:
    main = k.main_block("v_dot4")
    counts = collections.Counter(main)
    dot4 = sum(v for i, v in counts.items() if i.startswith("v_dot4"))
    valu = sum(v for i, v in counts.items() if i.startswith("v_"))
    budget = ref.loop_budget(cfg.m_tile, cfg.npt, group)
    expected = budget.sdot4
    failures = []
    if dot4 == 0 or dot4 % expected:
        failures.append(f"{dot4} v_dot4 in main block, want k*{expected}")
    unroll = max(1, dot4 // expected)
    if k.info.get("ScratchSize", 0):
        failures.append(f"scratch {k.info['ScratchSize']} B")
    lane = counts["v_readlane_b32"] + counts["v_writelane_b32"]
    if lane:
        failures.append(f"{lane} SGPR spill lane ops")
    slow = sum(counts[i] for i in QUARTER_RATE)
    if slow:
        failures.append(f"{slow} quarter-rate multiplies")
    lds = sum(v for i, v in counts.items() if i.startswith("ds_read"))
    smem = sum(v for i, v in counts.items() if i.startswith("s_load"))
    if cfg.a_src == "kLds" and lds == 0:
        failures.append("kLds config does not read A from LDS")
    if cfg.a_src == "kSmem" and (lds or not smem):
        failures.append("kSmem config does not read A via scalar loads")
    return Row(
        config=cfg.name,
        group=group,
        info=k.info,
        counts=counts,
        dot4=dot4 // unroll,
        non_dot_valu=(valu - dot4) // unroll,
        budget=budget,
        failures=failures,
    )


def audit(asm: str) -> tuple[list[Row], list[str]]:
    configs = {
        (c.threads, c.npt, c.k_step, c.m_tile, c.a_src): c for c in sweep_configs()
    }
    rows: list[Row] = []
    notes: list[str] = []
    for k in parse_asm(asm):
        m = GEMM_RE.search(k.name)
        if m and "w4a8_gemm_kernel" in k.name:
            th, npt, ks, mt, g, src = (int(x) for x in m.groups())
            cfg = configs[(th, npt, ks, mt, "kLds" if src == 0 else "kSmem")]
            rows.append(audit_gemm(k, cfg, g))
        elif "w4a8_probe_kernel" in k.name:
            want = {"0": "v_dot4", "1": "v_dot2", "2": "v_fma"}
            kind = re.search(r"ProbeE(\d)", k.name).group(1)
            prefix = want[kind]
            n = sum(i.startswith(prefix) for i in k.main_block(prefix))
            ok = n and n % PROBE_CHAINS == 0
            notes.append(
                f"probe {prefix}: {n} per loop block, VGPR "
                f"{k.info.get('NumVgprs')}{'' if ok else '  <-- FAIL'}"
            )
            if not ok:
                failure = f"probe loop has {n} {prefix}, want k*{PROBE_CHAINS}"
                rows.append(
                    Row(prefix, 0, k.info, collections.Counter(), 0, 0, None, [failure])
                )
        elif "w4a8_act_quant_kernel" in k.name:
            mt = re.search(r"act_quant_kernelILi\d+ELi(\d+)E", k.name).group(1)
            notes.append(
                f"act_quant MT={mt}: "
                "VGPR {NumVgprs}, SGPR {NumSgprs}, scratch {ScratchSize}, "
                "occupancy {Occupancy}".format(**k.info)
            )
    return rows, notes


def to_markdown(rows: list[Row], notes: list[str], compiler: str) -> str:
    out = [
        f"ISA audit (`{compiler}`, gfx1030, per thread per weight group)",
        "",
        "| config | G | VGPR | SGPR | occ | v_dot4 (want) | other VALU "
        "(budget) | v_mov | ds_read | s_load | W4A16/W4A8 VALU | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | --- |",
    ]
    for r in rows:
        if r.budget is None:
            continue
        c = r.counts
        out.append(
            f"| {r.config} | {r.group} | {r.info.get('NumVgprs')} | "
            f"{r.info.get('NumSgprs')} | {r.info.get('Occupancy')} | "
            f"{r.dot4} ({r.budget.sdot4}) | {r.non_dot_valu} "
            f"({r.budget.unpack + r.budget.flush}) | "
            f"{c['v_mov_b32_e32'] + c['v_mov_b32']} | "
            f"{sum(v for i, v in c.items() if i.startswith('ds_read'))} | "
            f"{sum(v for i, v in c.items() if i.startswith('s_load'))} | "
            f"{r.isa_ratio:.2f} (model {r.budget.ratio:.2f}) | "
            f"{'; '.join(r.failures) or 'ok'} |"
        )
    out += ["", *[f"- {n}" for n in notes]]
    return "\n".join(out) + "\n"


def find_clang(explicit: str | None) -> str:
    candidates = [explicit, os.environ.get("W4A8_CLANG")]
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    candidates += [f"{rocm}/llvm/bin/clang", shutil.which("clang")]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise SystemExit("no clang found; pass --clang")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clang", help="clang with the AMDGPU backend")
    parser.add_argument("--markdown", type=Path, help="write the table here")
    parser.add_argument("--save-asm", type=Path, help="keep the assembly")
    args = parser.parse_args()

    clang = find_clang(args.clang)
    kinds = collections.Counter(check_capi(clang))
    print(f"C ABI glue compiles (host + device passes): {dict(kinds)}")
    asm = compile_asm(clang)
    if args.save_asm:
        args.save_asm.write_text(asm)
    rows, notes = audit(asm)
    version = subprocess.run(
        [clang, "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    table = to_markdown(rows, notes, version)
    print(table)
    if args.markdown:
        args.markdown.write_text(table)
    failed = [r for r in rows if r.failures]
    for r in failed:
        print(f"FAIL {r.config} G={r.group}: {'; '.join(r.failures)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
