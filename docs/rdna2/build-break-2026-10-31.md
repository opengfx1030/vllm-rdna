# `rdna_extras` build break at `b740289353` (2026-10-31)

**Status:** fixed — `_rocm_C` builds clean for `gfx1030;gfx1100`; see
[Resolution](#resolution) at the end of this document.
**Reported by:** `bernie setup` on the 8× V620 / 2× W7800 reference box (`par1-cs25`).
**First failing commit:** the merge at the tip, `b740289353`.

```text
$ pip install --no-build-isolation --no-deps -e .
...
[2026-10-31T11:07:01Z]   Building editable for vllm (pyproject.toml): started
[2026-10-31T11:07:01Z]   FAILED: CMakeFiles/_rocm_C.dir/csrc/rocm/q_gemm_rdna2_prefill.hip.o
...
Failed to build vllm
ERROR: Failed to build installable wheels for some pyproject.toml based projects (vllm)
```

The build stopped after **20m39s** on a 48-core EPYC with `MAX_JOBS=16`; 50 objects of
`_rocm_C` had already compiled cleanly, so these three translation units are the only
blockers.

## How to read the paths below

The vLLM build copies `csrc/rocm/*.cu` to `*.hip` inside its temporary build directory
and compiles the copies, so the compiler names `/tmp/tmp*.build-temp/csrc/rocm/foo.hip`
for the repository file `csrc/rocm/foo.cu`. **Line numbers are identical.**

## Build command

```bash
VLLM_TARGET_DEVICE=rocm VLLM_USE_PRECOMPILED=0 ENABLE_CK=0 \
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
PYTORCH_ROCM_ARCH='gfx1100;gfx1030' \
ROCM_HOME=/opt/rocm-7.2.0/core-7.14 \
CMAKE_HIP_COMPILER=/opt/rocm-7.2.0/core-7.14/bin/hipcc \
HIP_ROOT_DIR=/opt/rocm-7.2.0/core-7.14 \
CMAKE_BUILD_TYPE=RelWithDebInfo MAX_JOBS=16 \
python -m pip install --no-build-isolation --no-deps -e .
```

Toolchain: HIP 7.14.60850 (`/opt/rocm-7.2.0/core-7.14`), torch 2.12.0+rocm7.14.0,
triton 3.7.1+git0263a6a6, Python 3.12.14.

## Fast reproduction of a single unit

The full wheel is not needed to see the errors:

```bash
cd csrc/rocm
/opt/rocm-7.2.0/core-7.14/bin/hipcc -x hip --offload-arch=gfx1030 \
  -c q_gemm_rdna2_prefill.cu -o /tmp/q.o -I. -I/usr/include/python3.12
```

---

## 1. `csrc/rocm/q_gemm_rdna2_prefill.cu` — the non-RDNA2 stubs drifted from the launcher

Four call sites, lines **417**, **422**, **427** (`gemm_static_kernel`) and **432**
(`gemm_dynamic_kernel`):

```text
csrc/rocm/q_gemm_rdna2_prefill.cu:417:6: error: no matching function for call to 'gemm_static_kernel'
csrc/rocm/q_gemm_rdna2_prefill.cu:373:52: note: candidate function template not viable:
    requires 12 arguments, but 13 were provided
```

The file has two branches. The real kernels are defined in the RDNA2 device pass; the
`#else` branch holds stubs whose comment promises *"same signatures so the launcher
compiles on any gfx target"*. The stubs were not updated when the real signature gained
its 13th parameter, `long long* fixed`:

```cuda
// src, RDNA2 branch, line 218 — 13 parameters, last one is long long*
template <typename Config>
__global__ __launch_bounds__(Config::THREADS) void gemm_dynamic_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_qzeros, const half* __restrict__ b_scales,
    half* __restrict__ c, const int size_m, const int size_n, const int size_k,
    const int groups, const int zero_offset, const int* __restrict__ b_q_perm,
    const int split_k, long long* fixed) { ... }

// stubs, line 370 — 12 parameters, `fixed` missing
template <typename Config, int K_PER_SPLIT>
__global__ __launch_bounds__(Config::THREADS) void gemm_static_kernel(
    const half*, const uint32_t*, const uint32_t*, const half*, half*,
    const int, const int, const int, const int, const int, const int*,
    const int) {}

template <typename Config>
__global__ __launch_bounds__(Config::THREADS) void gemm_dynamic_kernel(
    const half*, const uint32_t*, const uint32_t*, const half*, half*,
    const int, const int, const int, const int, const int, const int*,
    const int) {}
```

The launcher passes 13 arguments:

```cuda
gemm_static_kernel<Config, 256><<<grid, block, shmem, stream>>>(
    a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
    zero_offset, b_q_perm, split_k, fixed);
```

**Severity: every target.** Because the failure is in the `#else` stub branch, it breaks
the build for gfx9, gfx11 and the host-only compilation pass alike — not only gfx1030.

**Fix:** add `long long*` as the 13th parameter of both stubs. `gemm_dynamic_kernel`'s
stub also needs its template parameter list checked against the real kernel.

**Introduced by:** `2436df2335` — *[ROCm] Add graph-safe RDNA2 MoE expert offloading*
(2026-10-28).

---

## 2. `csrc/rocm/rdna_hint_flex_rdna2.cu:1773` — `kQG` is not found through the namespace alias

```text
csrc/rocm/rdna_hint_flex_rdna2.cu:1773:34: error: no member named 'kQG' in namespace 'vllm::rdna_hint_flex'
```

The use site, inside `launch_act_quant64` (anonymous namespace, `hf = vllm::rdna_hint_flex`
at line 1425):

```cuda
const int64_t n = M * (K / hf::kQG);
if (n == 0) return;
const int threads = 128;
hf::act_quant64_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(...);
```

`kQG` is declared at line **1077**, and that line sits between the opening of
`namespace vllm { namespace rdna_hint_flex {` (58–59) and its close (1417–1418):

```cuda
constexpr int kQG = 64;  // activation quantization group (columns)
```

They share the namespace but **not** the preprocessor branch: line 1077 sits inside
`#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)` (73 → 1350 → 1415),
while the host launchers start *after* that `#endif`. `__HIP__RDNA2__` is only defined
for `__gfx1030__` (line 54), so the **gfx1100 device pass** compiles the launcher with no
`kQG` in scope — the reverse of the intent: the non-RDNA2 branch deliberately stubs out
the kernels for symbol parity, but the constants the host launchers need were left inside
the guard. Only the second arch in `PYTORCH_ROCM_ARCH` exposes it, which is why a
single-arch `gfx1030` build never saw it.

**Fix:** declare `kQG` with the other shared constants above the guard (same value, one
definition).

Do not paper over it with a literal `64`: `n` is the grid size of `act_quant64_kernel`, so
a value that disagrees with the kernel's own group width under-launches the grid and
produces wrong output rather than a compile error.

**Introduced by:** `10cf5bbe52` — *perf(rdna2): grouped prefill path for rdna_hint_flex
routed experts* (2026-10-26).

---

## 3. `csrc/rocm/mono_k2_rdna2.cu:2123` — `s_memtime` missing on gfx11

```text
csrc/rocm/mono_k2_rdna2.cu:2123:32: error: '__builtin_amdgcn_s_memtime' needs target feature s-memtime-inst
```

```cuda
__global__ void mono_mark_kernel(unsigned long long* stats, int kind,
                                 int wtype) {
  ...
  const unsigned long long t = wall_clock64();
  const unsigned long long c = __builtin_amdgcn_s_memtime();  // shader clock
```

Measured on this box's HIP 7.14.60850 (`hipcc -x hip --offload-arch=<arch> -c`) the bare
call compiles for **gfx906, gfx90a and gfx1030** and fails only for **gfx1100** — so the
monokernel's own target (gfx1030) never lost the shader clock, and `s-memtime-inst` is
not "gfx9 only". No `-mattr` can add it for gfx11, so the call has to be guarded, and the
guard must be the builtin itself: **this clang does not define `__gfx9__` at all** (probed
with `#error` for gfx906/gfx90a), so an arch-name guard would have silently disabled the
shader clock on gfx1030 too.

```cuda
#if __has_builtin(__builtin_amdgcn_s_memtime)
  const unsigned long long c = __builtin_amdgcn_s_memtime();
#else
  const unsigned long long c = 0ull;   // consumers read 0 as "unavailable"
#endif
```

`c` feeds only the sclk columns (28/30 numerator, 29/31 denominator), which the
reporters already guard (`if r[29]`, `max(r[31], 1)`), so 0 leaves those columns empty
instead of mixing wall-clock nanoseconds into a cycle count.

**Introduced by:** `2436df2335` — same commit as issue 1.

---

## 4. `csrc/rocm/moe_q_gemm_rdna2.cu:294` — a second drifted stub (found while fixing 1-3)

Not in the original report: that build died before this unit was reached.

```text
csrc/rocm/moe_q_gemm_rdna2.hip:294:2: error: no matching function for call to 'moe_gemm_q4_kernel_rdna2'
... note: candidate function template not viable: requires 19 arguments, but 20 were provided
```

`6f5edc41a` (*native W4A16 MoE for asymmetric compressed-tensors checkpoints*, 2026-10-13)
appended `zero_offset` to the real kernel and to `launch_moe_gemm_q4`, but not to the
`#else  // non-RDNA2` stub. Same class and same blast radius as issue 1: the stub is the
non-RDNA2 branch, so gfx9/gfx11 fail as well.

Arity was not the only defect. Restoring the count by appending one `const int` still left
the type *order* different from the real kernel (9 ints, `bool`, `int` instead of
8 ints, `bool`, `int`, `int`), which is invisible to the compiler as long as the arity
matches — but it changes the mangled name, so the non-RDNA2 device image exports a symbol
the host launch does not reference:

```text
real: _Z24moe_gemm_q4_kernel_rdna2I6__halfS0_Li1EE...SB_iiiiiiiibii   (8 int, bool, int, int)
stub: _Z24moe_gemm_q4_kernel_rdna2I6__halfS0_Li1EE...SB_iiiiiiiiibi   (9 int, bool, int)
```

Observed at runtime, not at build time: a binary carrying that stub crashed on the first
launch of the op on a **gfx1100** card with

```text
Cannot find Symbol with name: _Z24moe_gemm_q4_kernel_rdna2I6__halfS0_Li1EE...SB_iiiiiiiibii
```

**Fix:** mirror the parameter list in the same order (9 pointers, 8 `const int`,
`bool`, `const int`, `const int`), and say in the comment why the order matters.

---

## Other observations (not build failures)

- **Deprecation.** HIP 7.14 warns on `hipDevicePrimaryCtxRetain(hipCtx_t*, hipDevice_t)`
  (`amd_hip_runtime.h:6663`). Harmless today; it will become an error in a later ROCm.
- **Two-arch cost.** Building `PYTORCH_ROCM_ARCH='gfx1100;gfx1030'` roughly doubles the
  compile time of an editable install with no runtime benefit on a host that serves one
  architecture. Callers should scope the list to the cards the deployment actually uses.
- **`pip install -e .` is destructive on failure.** pip uninstalls the previous package
  before building the replacement, so this compile error left the target venv with no
  `vllm` at all. Installers that drive this build should snapshot and restore the previous
  install when the build fails.

---

## Resolution

Four files, one class of bug in three of them (an arch-gated stub or constant that drifted
from the code the host launcher uses) and one mis-scoped preprocessor guard:

| # | File | Change |
|---|---|---|
| 1 | `csrc/rocm/q_gemm_rdna2_prefill.cu` | non-RDNA2 stubs: added the 13th parameter `long long* fixed` |
| 2 | `csrc/rocm/rdna_hint_flex_rdna2.cu` | `kQG` moved out of the `__HIP__RDNA2__ \|\| !__HIP_DEVICE_COMPILE__` guard, next to `kGroup`/`kThreads` |
| 3 | `csrc/rocm/mono_k2_rdna2.cu` | `__builtin_amdgcn_s_memtime` behind `__has_builtin`, `0` where absent |
| 4 | `csrc/rocm/moe_q_gemm_rdna2.cu` | non-RDNA2 stub: parameter list restored to 20 parameters **in the launcher's order** |

`docs/rdna2/build-break-2026-10-31.md` (this file) keeps the original diagnosis; the two
places where it guessed wrong are corrected inline above (issue 2's cause, issue 3's
"s-memtime is gfx9-only").

### Verification (same box, HIP 7.14.60850, torch 2.12.0+rocm7.14.0)

- **Build.** `python setup.py build_ext --inplace` (`--inplace` rather than `pip install
  -e .` so a failure cannot empty the venv), `PYTORCH_ROCM_ARCH='gfx1030;gfx1100'`,
  `MAX_JOBS=48`: `_rocm_C` links, 0 `error:` lines, `_rocm_C.abi3.so` 415 MB. The
  gfx1100 device pass is what exercises all three stub branches; the gfx1030 pass
  exercises the real kernels.
- **Ops registered after load.** 162 `_rocm_C::` schemas, including
  `moe_gptq_gemm_rdna2`, `gptq_gemm_rdna2_prefill`, `mono_k2_rdna2_mark` and the 12
  `rdna_hint_flex_*` ops.
- **Kernels.** `tests/kernels/quantization/test_rdna_hint_flex_rdna2.py` — 209 passed
  (gfx1030); `tests/kernels/quantization/test_rdna2_moe_w4a16.py` — 90 passed, 49 xfailed
  (the bf16 cases, xfail by design).
- **Mono kernel end to end.** `tools/rdna/mono/check_op.py` on gfx1030: all widths ok,
  graph replay 9.8e-04, fail-stop NaN path holds, `sclk over region MHz 2002` — i.e. the
  shader clock is still live on the monokernel's own target after the `__has_builtin`
  change.
- **gfx11 is now launchable, not just compilable.** On a W7800 (gfx1100):
  `moe_gptq_gemm_rdna2`, `gptq_gemm_rdna2_prefill` and the mono marks all launch (the
  stub body writes nothing, so `out` stays zero), and the mark kernel reports
  `sclk` columns as 0. Before the order fix in item 4 this was a hard crash —
  `Cannot find Symbol with name: ..._iiiiiiiibii` — so symbol parity of the stubs is
  worth checking, not only parameter counts. Note the reference box enumerates both
  architectures (HIP devices 0-1 = W7800/gfx1100, 2-9 = V620/gfx1030); a "gfx1030" run
  that does not pin `HIP_VISIBLE_DEVICES` may land on gfx1100.
