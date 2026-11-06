# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate ``dotrate.hip``: sustained dot-instruction and LDS rates on gfx1030.

Every VALU pattern is 64 instructions of one opcode inside an inline-asm block
on fixed registers, looped for about half a second over the whole GPU. The
operands are chosen by VGPR bank (bank = register index % 4) so the table
shows what a same-bank read costs and whether reusing the previous
instruction's operand hides it. The LDS patterns read ``ds_read_b128`` /
``b64`` with unique, 8-lane broadcast and wave broadcast addresses.

    python3 gen_dotrate.py > dotrate.hip
    hipcc --offload-arch=gfx1030 -O3 -o dotrate dotrate.hip
    HIP_VISIBLE_DEVICES=3 ./dotrate
"""

import sys


def reg(bank: int, idx: int) -> int:
    """VGPR ``idx`` of a bank: v[4 * idx + bank]."""
    return 4 * idx + bank


# Accumulators: bank 0, v0..v60 (16 registers, dependency distance 16).
# Operand pools: bank 0 idx 16..31, banks 1-3 idx 0..31. Nothing below writes
# an operand register, so the only dependencies are the accumulator chains.
def acc(t: int) -> int:
    return reg(0, t % 16)


PATTERNS = {
    # name: (description, fn(t) -> (d, a, b))
    "free": (
        "D, A, B in three banks, no reuse",
        lambda t: (acc(t), reg(1, (7 * t + 1) % 32), reg(2, (11 * t + 3) % 32)),
    ),
    "ab": (
        "A and B in one bank, no reuse",
        lambda t: (acc(t), reg(1, (7 * t) % 16), reg(1, 16 + (5 * t) % 16)),
    ),
    "da": (
        "D and A in one bank, no reuse",
        lambda t: (acc(t), reg(0, 16 + (7 * t) % 16), reg(2, (11 * t) % 32)),
    ),
    "dab": (
        "D, A, B in one bank, no reuse",
        lambda t: (acc(t), reg(0, 16 + (7 * t) % 8), reg(0, 24 + (5 * t) % 8)),
    ),
    "free_ra": (
        "three banks, A repeats the previous instruction's A",
        lambda t: (acc(t), reg(1, (t // 2) % 32), reg(2, (7 * t) % 32)),
    ),
    "ab_ra": (
        "A and B in one bank, A repeats",
        lambda t: (acc(t), reg(1, (t // 2) % 16), reg(1, 16 + (7 * t) % 16)),
    ),
    "ab_rb": (
        "A and B in one bank, B repeats",
        lambda t: (acc(t), reg(1, (7 * t) % 16), reg(1, 16 + (t // 2) % 16)),
    ),
    "da_ra": (
        "D and A in one bank, A repeats",
        lambda t: (acc(t), reg(0, 16 + (t // 2) % 16), reg(2, (7 * t) % 32)),
    ),
    "db_ra": (
        "D and B in one bank, A repeats (B and D fresh)",
        lambda t: (acc(t), reg(1, (t // 2) % 32), reg(0, 16 + (7 * t) % 16)),
    ),
}

# opcode: (asm template, MACs per lane per instruction, patterns to run)
OPCODES = {
    "v_dot2c_f32_f16": ("v_dot2c_f32_f16 v{d}, v{a}, v{b}", 2, list(PATTERNS)),
    "v_dot2_f32_f16": (
        "v_dot2_f32_f16 v{d}, v{a}, v{b}, v{d}",
        2,
        ["free", "ab", "dab", "free_ra", "ab_ra"],
    ),
    "v_dot4c_i32_i8": (
        "v_dot4c_i32_i8 v{d}, v{a}, v{b}",
        4,
        ["free", "ab", "da", "dab", "ab_ra", "db_ra"],
    ),
    "v_dot4_i32_i8": (
        "v_dot4_i32_i8 v{d}, v{a}, v{b}, v{d}",
        4,
        ["free", "ab", "dab"],
    ),
    "v_fmac_f32": ("v_fmac_f32 v{d}, v{a}, v{b}", 1, ["free", "ab", "dab"]),
    "v_fma_mix_f32": (
        "v_fma_mix_f32 v{d}, v{a}, v{b}, v{d} op_sel_hi:[1,1,0]",
        1,
        ["free", "ab"],
    ),
}

# LDS patterns: (description, instruction, bytes per lane, address expression)
LDS = {
    "b128_unique": ("ds_read_b128, 32 distinct 16 B", "ds_read_b128", 16, "lane*16"),
    "b128_bcast8": (
        "ds_read_b128, 8 lanes share an address",
        "ds_read_b128",
        16,
        "(lane/8)*16",
    ),
    "b128_bcast32": (
        "ds_read_b128, whole wave one address",
        "ds_read_b128",
        16,
        "0",
    ),
    "b128_conf2": (
        "ds_read_b128, 2-way bank conflict inside each 8-lane pass",
        "ds_read_b128",
        16,
        "(lane%4)*16+(lane/4)*128",
    ),
    "b64_unique": ("ds_read_b64, 32 distinct 8 B", "ds_read_b64", 8, "lane*8"),
    "b32_unique": ("ds_read_b32, 32 distinct 4 B", "ds_read_b32", 4, "lane*4"),
}

CLOBBER = ", ".join(f'"v{i}"' for i in range(128))


def valu_kernel(name: str, template: str, pattern: str) -> str:
    fn = PATTERNS[pattern][1]
    body = []
    for t in range(64):
        d, a, b = fn(t)
        assert a != d and b != d, (name, pattern, t)
        body.append(template.format(d=d, a=a, b=b))
    asm = "\\n\\t".join(body)
    init = "\\n\\t".join(
        f"v_mov_b32 v{i}, {'0' if i % 4 == 0 and i < 64 else '0x00010001'}"
        for i in range(128)
    )
    kname = f"k_{name}_{pattern}"
    return f"""
extern "C" __global__ void __launch_bounds__(256) {kname}(int iters,
                                                   unsigned long long* t) {{
  asm volatile("{init}" ::: {CLOBBER});
  const unsigned long long c0 = __builtin_amdgcn_s_memtime();
  const unsigned long long r0 = __builtin_amdgcn_s_memrealtime();
  for (int it = 0; it < iters; ++it) {{
    asm volatile("{asm}" ::: {CLOBBER});
  }}
  const unsigned long long c1 = __builtin_amdgcn_s_memtime();
  const unsigned long long r1 = __builtin_amdgcn_s_memrealtime();
  if (threadIdx.x == 0 && blockIdx.x == 0) {{
    t[0] = c1 - c0;
    t[1] = r1 - r0;
  }}
}}
"""


def lds_kernel(name: str, instr: str, width: int, addr: str) -> str:
    regs = width // 4
    lines = []
    for r in range(16):
        dst = f"v[{r * regs}:{r * regs + regs - 1}]" if regs > 1 else f"v{r}"
        lines.append(f"{instr} {dst}, %0 offset:{r * 1024}")
    lines.append("s_waitcnt lgkmcnt(0)")
    asm = "\\n\\t".join(lines)
    kname = f"k_lds_{name}"
    return f"""
extern "C" __global__ void __launch_bounds__(256) {kname}(int iters,
                                                   unsigned long long* t) {{
  __shared__ uint4 pool[16 * 1024 / 16 + 64];
  const int lane = threadIdx.x % 32;
  for (int i = threadIdx.x; i < 16 * 1024 / 16 + 64; i += 256)
    pool[i] = make_uint4(i, i, i, i);
  __syncthreads();
  const unsigned addr =
      (unsigned)(size_t)(&pool[0]) + (unsigned)({addr});
  const unsigned long long c0 = __builtin_amdgcn_s_memtime();
  const unsigned long long r0 = __builtin_amdgcn_s_memrealtime();
  for (int it = 0; it < iters; ++it) {{
    asm volatile("{asm}" :: "v"(addr) : {CLOBBER});
  }}
  const unsigned long long c1 = __builtin_amdgcn_s_memtime();
  const unsigned long long r1 = __builtin_amdgcn_s_memrealtime();
  if (threadIdx.x == 0 && blockIdx.x == 0) {{
    t[0] = c1 - c0;
    t[1] = r1 - r0;
  }}
}}
"""


def main() -> None:
    out = [
        "// Generated by gen_dotrate.py. Do not edit.",
        "#include <hip/hip_runtime.h>",
        "#include <cstdio>",
        "#include <cstdlib>",
        "#include <string>",
        "",
        "#define HIPCHK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \\",
        '  printf("HIP error %s at %d\\n", hipGetErrorString(e_), __LINE__); \\',
        "  return 1; } } while (0)",
    ]
    runs = []
    for op, (tmpl, macs, pats) in OPCODES.items():
        for p in pats:
            out.append(valu_kernel(op, tmpl, p))
            runs.append((f"k_{op}_{p}", op, p, PATTERNS[p][0], macs, 64, 0))
    for name, (desc, instr, width, addr) in LDS.items():
        out.append(lds_kernel(name, instr, width, addr))
        runs.append((f"k_lds_{name}", instr, name, desc, 0, 16, width))
    out.append("typedef void (*kfn)(int, unsigned long long*);")
    out.append("struct Run { kfn f; const char* op; const char* pat;")
    out.append("  const char* desc; int macs; int per_iter; int width; };")
    out.append("static const Run kRuns[] = {")
    for k, op, p, desc, macs, per_iter, width in runs:
        out.append(f'  {{{k}, "{op}", "{p}", "{desc}", {macs}, {per_iter}, {width}}},')
    out.append("};")
    out.append(HOST)
    sys.stdout.write("\n".join(out) + "\n")


HOST = r"""
int main(int argc, char** argv) {
  hipDeviceProp_t prop;
  HIPCHK(hipGetDeviceProperties(&prop, 0));
  const int units = prop.multiProcessorCount;
  // One resident round: blocks <= capacity, so no tail wave skews the rate.
  const int per_unit = argc > 2 ? atoi(argv[2]) : 1;
  const int blocks = units * per_unit;
  const double target_s = argc > 1 ? atof(argv[1]) : 0.5;
  printf("device %s, %d compute units reported, %d blocks x 256 threads\n",
         prop.gcnArchName, units, blocks);
  unsigned long long* dt;
  HIPCHK(hipMalloc(&dt, 2 * sizeof(unsigned long long)));
  hipEvent_t e0, e1;
  HIPCHK(hipEventCreate(&e0));
  HIPCHK(hipEventCreate(&e1));
  printf("%-16s %-12s %8s %8s %9s %9s %10s  %s\n", "op", "pattern", "ms",
         "MHz", "inst/clk", "T-MAC/s", "B/clk/CU", "description");
  for (const Run& r : kRuns) {
    int iters = 256;
    float ms = 0.f;
    for (int pass = 0; pass < 3; ++pass) {
      hipLaunchKernelGGL(r.f, dim3(blocks), dim3(256), 0, 0, iters, dt);
      HIPCHK(hipEventRecord(e0, 0));
      hipLaunchKernelGGL(r.f, dim3(blocks), dim3(256), 0, 0, iters, dt);
      HIPCHK(hipEventRecord(e1, 0));
      HIPCHK(hipEventSynchronize(e1));
      HIPCHK(hipEventElapsedTime(&ms, e0, e1));
      if (ms > 0.5 * target_s * 1e3) break;
      iters = (int)(iters * (target_s * 1e3 / (ms > 0.01f ? ms : 0.01f)));
    }
    unsigned long long h[2];
    HIPCHK(hipMemcpy(h, dt, sizeof(h), hipMemcpyDeviceToHost));
    // s_memrealtime ticks at 100 MHz; s_memtime at the shader clock.
    const double mhz = h[1] ? 100.0 * (double)h[0] / (double)h[1] : 0.0;
    const double waves = (double)blocks * 8.0;
    const double inst = waves * (double)iters * r.per_iter;  // wave instrs
    const double sec = ms * 1e-3;
    const double simds = units * 2.0;  // two SIMD32 per CU
    const double clk = mhz * 1e6;
    const double per_clk = inst / (simds * sec * clk);
    const double tmac = r.macs ? inst * 32.0 * r.macs / sec / 1e12 : 0.0;
    const double bpc =
        r.width ? inst * 32.0 * r.width / (units * sec * clk) : 0.0;
    printf("%-16s %-12s %8.1f %8.0f %9.3f %9.2f %10.1f  %s\n", r.op, r.pat,
           ms, mhz, per_clk, tmac, bpc, r.desc);
  }
  return 0;
}
"""


if __name__ == "__main__":
    main()
