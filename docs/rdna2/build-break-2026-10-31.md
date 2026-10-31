# `rdna_extras` does not compile (2026-10-31)

**Status:** open — blocks every source build of the branch, on every architecture.
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

So on inspection the declaration and the use site share the namespace, yet qualified
lookup fails. Whoever takes this should re-run the repro and confirm the enclosing
preprocessor scope: the observed message is authoritative, but the declaration appears
reachable, so the cause is likely a guard, an inline-namespace, or an ordering detail this
report could not settle from the source alone.

Do not paper over it with a literal `64`: `n` is the grid size of `act_quant64_kernel`, so
a value that disagrees with the kernel's own group width under-launches the grid and
produces wrong output rather than a compile error.

**Introduced by:** `10cf5bbe52` — *perf(rdna2): grouped prefill path for rdna_hint_flex
routed experts* (2026-10-26).

---

## 3. `csrc/rocm/mono_k2_rdna2.cu:2123` — gfx9-only builtin in a gfx10/11 file

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

`s-memtime-inst` exists on **gfx9 only**. gfx1030 and gfx1100 do not implement the
instruction, so no `-mattr` can enable it; the call has to be guarded:

```cuda
#if defined(__gfx9__)
  const unsigned long long c = __builtin_amdgcn_s_memtime();
#else
  const unsigned long long c = t;   // wall_clock64() is already computed above
#endif
```

**Introduced by:** `2436df2335` — same commit as issue 1.

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
