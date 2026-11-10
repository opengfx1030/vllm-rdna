# Flash-Next W4A16 on gfx1100 (Radeon PRO W7800): RDNA3 port, 2026-11-07

Branch `feat/rdna3-flashnext`, base `origin/rdna_extras` `27e883f3ce8b0d7ee84f184210756923abc8896`.
This is slice S1 / W2 of `docs/rdna2/hetero-moe-design-2026-11-07.md` (branch
`feat/hetero-moe`, section 9.5): Qwen3.8-Flash-Next-AWQ-W4A16 with MTP=2 on gfx1100,
TP2 across the two W7800 (HIP 0 and 5, PCIe buses `cb` and `07`), EP2, no offload. It is
also the numerics and performance baseline of the `hub` role in the hetero design.

**Nothing here was run on a GPU.** Every runtime claim below is either read from code
(file and line given), measured on the CPU of `par1-cs25` (builds, code-object
comparison, CPU unit tests) or marked `[INFERENCE]`. The GPU test plan (section 12) says
exactly what has to be measured.

## 1. What changed, in one paragraph

The Flash-Next serve path (Qwen4Exp AMD path: MTP2, QSA, GDN, PLE, HC, compiled
`FULL_AND_PIECEWISE`, fp16) selected its HIP kernels with `on_rdna2()`, `on_gfx10x()` or
`_GCN_ARCH == "gfx1030"` tests. On gfx1100 that meant: Triton/torch everywhere, one gate
(`hyperconnection._rdna_fused_ok`) that was true on every ROCm arch and selected a kernel
that was not compiled for gfx1100, a gfx1100-only `_rocm_C` that did not load (4 undefined
symbols), a wrong-arch hazard on the mixed V620 + W7800 box (amdsmi read the first handle),
and a gfx1100 W4A16 MoE kernel that had no determinism mode, no block-size control and no
hub contract audit. This branch makes the path buildable and loadable for gfx1100, adds
per-family opt-in of the fork's HIP kernels (`VLLM_RDNA3_HIP_OPS`), hardens the MoE
kernel for the hub role, gates `rdna_ar` for gfx11 behind an opt-in, widens the per-op GPU
tests so they run on gfx1100, and ships the tools to prove that gfx1030 is untouched and to
validate gfx1100 on a GPU.

## 2. Dispatch policy

gfx1100 defaults to the Triton/torch/upstream routes plus the native gfx1100 W4A16 MoE
kernel (`moe_gptq_gemm_rdna3`, `vllm/model_executor/layers/fused_moe/experts/rdna3_moe.py`).
The fork's HIP families are opt-in, one family at a time, until their per-op test and
serve gates pass on a W7800. Defaults flip only after that (open decision 1).

`VLLM_RDNA3_HIP_OPS` is a comma list of family names or `all` (`vllm/platforms/rdna.py`,
`RDNA3_HIP_FAMILIES`, parser `_rdna3_hip_families` at `:99`, predicate `on_rdna3_hip` at
`:141`, `on_rdna2_hip(family)` at `:152`).

| Family | Enables on gfx1100 | Probe op in `_rocm_C` | Call sites (file:line) |
|---|---|---|---|
| `hc` | fused HyperConnection glue and HC mix ops | checked at the call site | `hyperconnection.py:68`, `hc_rdna2.py:37`, `rdna_ops.py:47,165` |
| `gdn` | GDN decode/prefill kernels, conv1d update, fused sigmoid gating | `gdn_decode_rdna2` | `qwen_gdn_linear_attn.py:318,835,1995,2734`, `causal_conv1d.py:645,1189,1228`, `fused_sigmoid_gating.py:261` |
| `qsa` | QSA MQA decode, QSA metadata builder | `qsa_mqa_paged_rdna2` | `qsa_rdna2.py:41,62`, `qsa_cache.py:651` |
| `se` | shared-expert gate/up/silu and down/gated | checked at the call site | `qwen2_moe.py:81` |
| `dense` | `gemv_f16_rdna2` dense GEMV, int8 shadow | checked at the call site | `utils.py:345`, `rdna_dense_int8.py:42` |
| `moeu` | fp16 unquantized-MoE HIP kernels (BF16 MTP experts) | `moe_gate_up_silu_unquantized_rdna2` | `oracle/unquantized.py:72`, `rdna2_unquantized_moe.py:72-80` |
| `ar` | `rdna_ar` one-shot all-reduce (also needs `VLLM_RDNA_AR=1`) | checked in `rdna_all_reduce` | `cuda_communicator.py:203` |
| `v1` | V1 worker ops (`apply_write_rdna2` and friends) | `apply_write_rdna2` | module-level `_ON_RDNA2` of the V1 modules |

Rules of the parser, all enforced and unit-tested (`tests/platforms/test_rdna3_dispatch.py`):

- An unknown family name raises `ValueError`.
- A family named explicitly whose probe op is missing from `_rocm_C` raises
  `RuntimeError` naming the op and the fix (`PYTORCH_ROCM_ARCH='gfx1030;gfx1100'`).
- `all` drops families whose op is missing and logs a warning.
- The result is computed once per process.

Why gfx10x cannot change: every widened gate keeps its old term first and appends
`or on_rdna3_hip("<family>")`, and `on_rdna3_hip` is `False` unless the arch is gfx1100.
The only gate that was not arch-limited before (`_rdna_fused_ok`) keeps its old value on
gfx10x for the same reason. Section 6 has the full argument.

## 3. Per-op status, before and after

"Before" is `origin/rdna_extras` on gfx1100. "After" is this branch. "Default" is the
route on gfx1100 with `VLLM_RDNA3_HIP_OPS` unset.

| # | Op (serve path) | Before on gfx1100 | After | Default route | Opt-in |
|---|---|---|---|---|---|
| 1 | HC decode glue (`rdna_gemv_act`, `rdna_hc_up_gate_mix`) | gate true on every ROCm arch, kernel TU built for gfx1030 only, registration unconditional: a gfx1100-only `_rocm_C` failed to load | TU built for gfx1030 and gfx1100 (`CMakeLists.txt`), registration under `VLLM_ROCM_RDNA_GLUE` (`torch_bindings.cpp:26,102,250`), gate `on_gfx10x() or on_rdna3_hip("hc")` | torch/Triton HC | `hc` |
| 2 | HC prefill / mix (`hc_rdna2`) | gfx10x only | `hc_use_rdna2` widened (`hc_rdna2.py:37`) | Triton | `hc` |
| 3 | GDN decode, prefill chain, conv1d update, sigmoid gating | gfx10x only (no way to select the HIP kernels) | gates widened, op presence checked | Triton | `gdn` |
| 4 | QSA decode and metadata | gfx10x only | `qsa_use_rdna2`, `_resolve_qsa_metadata_builder` widened | Triton / upstream metadata | `qsa` |
| 5 | Shared expert | gfx1030 only (`_GCN_ARCH == "gfx1030"`) | `_rdna_shared_expert_ok` widened (`qwen2_moe.py:81`) | torch | `se` |
| 6 | Dense GEMV, int8 shadow | gfx10x only | widened (`utils.py:345`, `rdna_dense_int8.py:42`); upstream `wvSplitK` (n up to 5) and `LLMM1` unchanged | `wvSplitK` / hipBLASLt | `dense` |
| 7 | Routed experts, main model W4A16 | native kernel present, no determinism, no block-size control, `-1` masking not audited | Q32 deterministic mode, `VLLM_RDNA3_MOE_BLOCK_M`, `ignore_invalid_experts`, hub contract audit (section 7) | `moe_gptq_gemm_rdna3` default mode (bit-identical to before) | `VLLM_RDNA3_MOE_DETERMINISTIC=1` |
| 8 | MTP drafter experts | BF16 checkpoint: Triton `fused_moe` | unchanged by default; fp16 HIP kernels selectable; derived W4A16-MTP checkpoint uses row 7 | Triton `fused_moe` | `moeu` |
| 9 | Top-k decode (`top_k_per_row_decode`, QSA indexer top-2048) | insertion path on gfx11 (radix only on gfx10) | radix default on gfx10 and gfx11, `VLLM_TOPK_DECODE_ALGO=radix` or `insertion` forces either (`sampler.cu:1232-1238`); gfx11 default is `[INFERENCE]` | radix | env |
| 10 | All-reduce | RCCL (rdna_ar gated to gfx10x) | `rdna_ar` startable on gfx1100, group-wide arch agreement, per-arch wedge marker (section 8) | RCCL | `ar` plus `VLLM_RDNA_AR=1` |
| 11 | V1 worker ops | gfx10x only | `on_rdna2_hip()` widened | torch/Triton | `v1` |
| 12 | PLE conv, n-gram sidecar | Triton, arch-neutral | unchanged (the HIP `ple_short_conv_*` ops have no Python caller) | Triton | none |
| 13 | Arch detection (`_get_gcn_arch` in `rocm.py`, `rocm_visible.py`) | first amdsmi handle wins, whatever the visible-device list | no list set: unchanged (first amdsmi handle, one query); `HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES` set: the arch of the first visible GPU from the KFD topology files, no amdsmi call and no HIP context, torch only if the files cannot say | the same arch string on a single-arch box, the visible GPU's arch on a mixed box | none |
| 14 | `rms_norm` IR priority, `LLMM1`, router/lm_head/embedding GEMMs | upstream gfx11 behaviour | unchanged, listed as A/B items in section 12 | upstream | none |

Not changed and not usable on gfx1100: `w4a8_gemm_rdna2` (unreachable), `wvSplitKQ` (fp8)
is a stub-zero on gfx1100, `gptq_gemm_rdna3` rejects AWQ zero points and the checkpoint is
symmetric anyway (dense layers are not quantized), `gptq_gemm_rdna3_wmma` has no Python caller.

## 4. Build and device-code analysis (CPU only, `par1-cs25`)

Toolchain `/opt/rocm/core-7.14`, torch 2.12.0+rocm7.14.0. All builds ran under the shared
build lock, `MAX_JOBS` 8, `nice 19`, no GPU touched.

Command (one tree, one arch list, own build directory, status file, load check):

```bash
bash tools/rdna/rdna3/build_rocm_c.sh <tree> 'gfx1030;gfx1100' 8 [ninja-target]
```

The HIP families need the fat `PYTORCH_ROCM_ARCH='gfx1030;gfx1100'` build, because the
fork's kernels live in the `VLLM_ROCM_GFX1030` block of `torch_bindings.cpp` (from about
`:313` to the end) and in TUs that are empty stubs on gfx1100 builds. A gfx1100-only
build loads and runs the default path (native MoE kernel, upstream skinny GEMMs) but has no
`gdn`, `qsa`, `moeu`, `v1` ops: a named family then fails at startup with the message above.

| Build | Arch list | Time | `_rocm_C.abi3.so` | Load check (`rocm_c_load_check.py`) |
|---|---|---|---|---|
| base, untouched | gfx1030 | 10.5 min | 91.7 MB | loads |
| base, untouched | gfx1030 and gfx1100 | 18.7 min | 116 MB | loads |
| base, untouched | gfx1100 | 8.7 min | 57.9 MB | 4 undefined project symbols (the registration bug fixed here) |
| this branch, in place over the base tree | gfx1030 and gfx1100 | incremental: 5 ninja steps, then 9 after clang-format touched `ops.h` | 116,286,056 B | loads, 177 `_rocm_C` ops, no undefined project symbols |
| this branch, fresh tree | gfx1100 | 9.6 min | 60,171,208 B | loads, 44 `_rocm_C` ops, no undefined project symbols |
| this branch with S1b (`b964e29f2d`), in place | gfx1030 and gfx1100 | 127 s incremental (7 ninja steps) | 117,066,008 B | loads, 178 `_rocm_C` ops, no undefined project symbols |
| this branch with S1b, gfx1100-only tree | gfx1100 | incremental (5 ninja steps) | 60,873,792 B | loads, 45 `_rocm_C` ops, no undefined project symbols |

The load check lists the undefined dynamic symbols that look like project symbols; for
the 2-arch build it reports none after filtering libtorch/libstdc++ names (the unfiltered
list was 8 libtorch/c10/libstdc++ symbols). The authoritative result is the `dlopen`.
The fat build of phase 2 (`tools/rdna/rdna3/build_fat.sh`, 1,620 s at 8 jobs under the build
lock) compiled every other extension for both arches too: `_C`, `_C_stable_libtorch`,
`_moe_C_stable_libtorch`, `cumem_allocator`, `fs_io_C` and `spinloop` next to `_rocm_C`. All
of them loaded on the W7800 in the G0 to G8 run (section 16).

### 4.1 Which TUs are stubs on which arch

Measured from the code objects (`.text` bytes per object, word-level stub test in
`code_object_fingerprint.py`):

- Stub-like on gfx1100: `moe_q_gemm_rdna2`, `moe_w4a8_rdna2`, `w4a8_sdot4_rdna2`,
  `q_gemm_rdna2`, `q_gemm_rdna2_prefill`, `rdna_hint_flex_rdna2`, `mxfp4_dot2_dense`,
  `mxfp4_dot2_moe`.
- Stub on gfx1030: `q_gemm_rdna3`, `q_gemm_rdna3_wmma`, `moe_q_gemm_rdna3`,
  `mxfp4_w4a8_gemv_rdna3`.
- Real on both: GDN, conv1d, `hc_*`, QSA, mono, `fa_rdna2`, layernorm, mrope, V1,
  `rdna_allreduce`, `rdna_fused_glue`, exl3, `skinny_gemms_int4`, `moe_unquantized_rdna2`,
  `moe_w8a16_rdna2`.

### 4.2 The same-path oracle (why base and patched are built in one tree)

hipcc embeds a per-TU CUID, the MD5 of the source path and of the whole command line,
in every code object. A different path or flags rewrites PC-relative immediates even in
kernels whose source did not change, so "same source, different build dir" differs.
`-fuse-cuid=none` does not work (every TU then defines the same `__hip_cuid_` symbol and
the link fails). The oracle therefore rebuilds the base tree in place, with the patched
sources rsynced over it (checksum mode, so only changed files get new mtimes), and compares
the new objects with the untouched ones kept as JSON fingerprints
(`tools/rdna/rdna3/code_object_fingerprint.py extract` then `compare`).

`compare` classifies each object: IDENTICAL, STUB-CHURN (only stub words changed:
`s_endpgm`, `s_trap`, `s_nop`, zero), CHANGED, ONLY-IN-BASE, ONLY-IN-NEW. Exit status is 1
on CHANGED or ONLY-IN-*; `--strict` also fails on STUB-CHURN.

| Arch | IDENTICAL | STUB-CHURN | CHANGED | ONLY-IN-BASE | ONLY-IN-NEW |
|---|---|---|---|---|---|
| gfx1030 (70 objects, 6691 symbols) | 69 | 1 (`moe_q_gemm_rdna3`) | 0 | 0 | 0 |
| gfx1100 (70 objects, 6691 symbols) | 69 | 0 | 1 (`moe_q_gemm_rdna3`, new kernels) | 0 | 0 |

The gfx1030 STUB-CHURN is `moe_q_gemm_rdna3.hip.o` only: the file gained 13 four-byte
empty stubs (kernel symbols +5) for symbol parity, and nothing on gfx1030 can reach them,
because the op is registered in the GFX1100 block. The gfx1100 CHANGED object is the same
file with its new Q32 kernels; its 8 default kernels (half and bf16, block size 1/2/4/8)
have byte-identical `.text` and identical VGPR/SGPR/LDS/scratch/kernarg metadata (only the
mangled names change).

## 5. Mixed-arch box (V620 gfx1030 next to W7800 gfx1100)

One process never mixes archs (TP2 over HIP 0 and 5; the V620 offload role is a separate
process). The hazard was detection: `_query_gcn_arch_from_amdsmi` returned the arch of the
first handle whatever the process was allowed to use, and amdsmi lists every GPU of the
machine, ignores the visible-device variables and powers up the idle ones. A first version
of this port asked every handle (to raise on mixed archs), which wakes every parked GPU,
including the ones a launcher had left out: at 16:50 the box-rules owner saw the W7800
that must stay parked resume when pinned workers of a venv that ships amdsmi started, and
traced it to that loop ([INFERENCE]: not isolated from the platform plugin's own amdsmi
probe); venv-7.14.0 ships no amdsmi, so its runs never took that path.

`_get_gcn_arch` now follows two rules:

- No visible-device list set (`HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`,
  `ROCR_VISIBLE_DEVICES`; empty counts as unset): the base behaviour, the first amdsmi
  handle, one query, no HIP context.
- A list set: no amdsmi call. The arch comes from the KFD topology
  (`/sys/class/kfd/kfd/topology/nodes/*/properties`, `gfx_target_version`; helpers in
  `rocm_visible.py`), static files that need no GPU access and create no HIP context. If
  every GPU of the machine has one arch, that is the answer. On a mixed machine the first
  visible GPU is picked in HIP's device order (the KFD node order; the ROCR list filters
  first, the HIP list indexes what is left). That order is the HIP-to-bus map of the box
  rules: on the box after the case move the nodes 1 to 6 are the buses d1, d4, d9, dc, df
  and e2, which are HIP 0 to 5; on the box before it, HIP 0 and 5 were the W7800 nodes 1 and
  6. Only when the files cannot say (no KFD, a UUID list on a mixed machine, an index
  outside the GPUs) does torch answer with
  `torch.cuda.get_device_properties("cuda").gcnArchName`, which creates a HIP context at
  import.

Why torch is not the rule for a pinned process: a HIP context in the parent process also
makes vLLM fall back from `fork` to `spawn` (`_maybe_force_spawn`; the comment of
`get_device_total_memory` avoids exactly that) and holds device memory in a process that
never computes. The rest of the platform still calls amdsmi (the plugin probe in
`vllm/platforms/__init__.py`, `get_device_name`, memory and link queries); making a pinned
process free of amdsmi altogether, and putting `ROCR_VISIBLE_DEVICES` in the child
environment, is an open decision for the launcher owner (the box rules accept the
enumeration wake of every ROCm process: `hsa_init` binds all KFD nodes whatever
`HIP_VISIBLE_DEVICES` says). A worker-init warning for an arch that differs from the
engine's was not written.

CPU rule that follows: every CPU run of vLLM on that box goes through
`tools/rdna/rdna3/cpu_pytest.sh`, which blocks `sys.modules["amdsmi"]`, unsets
`ROCR_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`, sets `HIP_VISIBLE_DEVICES=` and puts
`tools/rdna/rdna3/cpu_stubs/rocminfo` first on `PATH`: aiter (imported through
`flash_attn` by the Qwen4Exp AMD modules) runs `rocminfo` at import, which enumerates every
GPU of the box and, on the V620-only box after the case move, reports a gfx1030 first, an
arch aiter does not list (`KeyError: Unknown GPU architecture`; 35 dispatch tests failed
on it before the stand-in, all of them in `test_rdna3_dispatch.py`). The stand-in reports
one gfx1100 agent and asks no GPU. The helper tools (`build_rocm_c.sh`,
`rocm_c_load_check.py`) block amdsmi and hide the GPUs the same way.

## 6. Why gfx1030 behaviour does not change

By construction:

1. Python gates: each widened gate is `old_term or on_rdna3_hip(family)`. The old term is
   evaluated first and `on_rdna3_hip` is `False` on every arch except gfx1100, so the
   boolean is the old one on gfx10x. `tests/platforms/test_rdna3_dispatch.py` has the truth
   tables for every call site on faked gfx1030, gfx1100 and gfx942.
2. `_rdna_fused_ok` (was true on all ROCm archs) keeps `on_gfx10x()` as its first term.
3. C++ and CUDA: all new device code is inside `defined(__gfx1100__)` guards, and the
   gfx1030 pass sees empty stubs. New op registrations are inside the GFX1100 block;
   `rdna_fused_glue` registrations are under `VLLM_ROCM_RDNA_GLUE`, which is defined for
   gfx1030 builds exactly as before (`torch_bindings.cpp:26`). `CMakeLists.txt` only adds
   gfx1100 to the arch test of the glue TU, so gfx1030 builds compile the same TU list
   with the same flags (no new compile definition, CUIDs stay stable).
4. `sampler.cu`: host-side only (`VLLM_TOPK_DECODE_ALGO` is read per call, the gfx10 default
   is still radix, `VLLM_RDNA3_MOE_DETERMINISTIC` only adds to the tie-break condition).
5. `rocm.py`: the same arch string on a single-arch box, and no HIP context at import in
   either rule unless the last resort of section 5 (torch) is taken. The one difference: a
   process with a visible-device list set reads the KFD topology files and no longer calls
   amdsmi for the arch (without a list it is still one query of the first handle).

By measurement: the same-path oracle of section 4.2 (gfx1030 code objects identical
except the stub churn of `moe_q_gemm_rdna3`), and the CPU truth tables of section 11.

Not provable without a GPU: that no gfx1030 runtime behaviour differs. Nothing in the diff
reaches gfx1030 except the items above, but no V620 run was made.

## 7. MoE kernel and the hub contract (Rdna3MoeDet)

All in `csrc/rocm/moe_q_gemm_rdna3.cu`, `rdna3_moe.py`, `_custom_ops.py`. Default mode is
bit-identical to the old kernel (8 default kernels byte-identical on gfx1100).

- Q32 deterministic mode, `VLLM_RDNA3_MOE_DETERMINISTIC=1` (default off, fp16 only; bf16
  with Q32 raises `ValueError`; envs at `envs.py:388`). The accumulator is an int64 tensor
  `[rows, N+1]` (the extra column is a sticky row status) written by fixed-point int64
  atomics with the `fixed_accum_row` logic of `q_gemm_rdna2_common.cuh`. New op
  `moe_fixedpoint_to_half_rdna3(Tensor accumulator, Tensor! output)` converts
  int64 to fp32 to fp16 (two roundings, as RDNA2) and writes NaN when the row status is set
  or the result is not finite (`torch_bindings.cpp:1330`).
  Results are bit-stable across runs and routing order for a fixed block size, not across
  block sizes. The hub contract (design section 9.4) requires Q32.
- `ignore_invalid_experts=True` in `moe_align_block_size`, and `apply` asserts that
  `expert_map` is an int32 contiguous device tensor with `numel >= global_num_experts`.
- `VLLM_RDNA3_MOE_BLOCK_M` (`rdna3_moe.py:51`): 0 (automatic: 1 if M is at most 4, else
  4) or one of 1, 2, 4, 8; anything else raises `ValueError`.
- Host side: `C10_CUDA_KERNEL_LAUNCH_CHECK()` after the launch, early return when there are
  no token blocks, `TORCH_CHECK` on the dtype of `c`.

Audit of the contract (evidence in the sub-agent report, summarized):

- `-1` expert ids are skipped end to end: `moe_align_sum_kernels.cu:120-132, 178-180,
  274-276, 321-324, 352-355`, and in the GEMM at `moe_q_gemm_rdna3.cu:179, 181-182, 472`.
  `num_experts` must be the global count when a map is passed.
- `expert_map` is read through a raw device pointer: in-place updates are seen on graph
  replay, rebinding a new tensor is not.
- `apply` has no `.item()`, `.tolist()`, `.cpu()` or synchronize, and no data-dependent
  shape (checked under `FakeTensorMode` at M = 1, 3, 5, 24, 48 in both modes).
- Sizes at the hub shapes (K 2560, N 640, top-k 10, 512 global experts): workspace2
  19,200 elements at M=1 and 921,600 (1.84 MB fp16) at M=48; Q32 scratch at M=48 is
  4,919,040 B (GEMM1) plus 983,424 B (GEMM2), at M=2048 209,879,040 B plus 41,959,424 B;
  at M=16384 about 1.68 GB plus 0.34 GB. Q32 limits: 2^30/10 (GEMM1), 2^30/30 (GEMM2).

Risks of this part: Q32 performance is unmeasured (int64 atomics, `torch.zeros` of the
scratch on every call); Q32 returns NaN, not inf, at or above 65520; the
int64-to-fp32-to-fp16 path is not "exact sum rounded once"; Q32 supports fp16 only.
Outside this branch's scope: `rdna2_w4a16_moe.py:213-219` and `rdna2_mxfp4_moe.py:223-229`
(gfx1030 paths) pass `local_num_experts` together with `expert_map` and
`ignore_invalid_experts=True`, which under expert parallelism would drop global ids at or
above the local count; Marlin and Triton pass the global count.

### 7.1 S1b: WMMA W4A16 prefill kernel (Rdna3Wmma, opt-in)

`moe_gptq_gemm_rdna3_wmma` (`csrc/rocm/moe_q_gemm_rdna3_wmma.cu`, a translation unit of its
own, as `q_gemm_rdna3_wmma.cu` is, because a TU shared with the scalar kernels once
miscompiled the scalar path) runs the experts of a token block on
`v_wmma_f32_16x16x16_{f16,bf16}_w32` and reduces the whole K in registers: fp32
accumulators, no split-K, no atomics on the GEMM result. The scalar `moe_gptq_gemm_rdna3`
stays the decode kernel and the fallback.

- Contract: the weights and the routing tensors of the scalar op (`b_q_weight [E, K/8, N]`
  GPTQ-shuffled, scales `[E, groups, N]`, zeros `[E, groups, N/8]`, `sorted_token_ids`,
  `expert_ids`, `num_tokens_post_padded`); `-1` expert blocks and padding rows are never
  touched. Schema `moe_gptq_gemm_rdna3_wmma(a, c!, b_q_weight, b_scales, b_qzeros,
  topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, int top_k, int
  block_size_m, bool mul_topk_weight, int output_topk=0)` (GFX1100 block of
  `torch_bindings.cpp`, wrapper plus `register_fake` in `_custom_ops.py`). The dtype of
  `c` picks the output: fp16 or bf16 `[pairs, N]` with `output_topk == 0` stores each
  scheduled pair row exactly once as `round(acc * topk_weight)`; an int64 `[rows, N+1]`
  (fp16 only) is the Q32 accumulator of the scalar kernel, one int64 atomic add per
  output element and pair, and `output_topk > 0` folds pair `p` into row `p /
  output_topk`.
- Tiles (`block_size_m` 16, 32, 64, `TILE_N` 256): BM16 is 4 waves with 4 fragments per
  wave and a K chunk of 128; BM32 is 8 waves, 2 fragments, K chunk 128; BM64 is 8 waves,
  2 fragments, K chunk 64.
- Dispatch (`rdna3_moe.py`): `VLLM_RDNA3_MOE_WMMA=1` (default 0) sends batches of at least
  `VLLM_RDNA3_MOE_WMMA_MIN_M` tokens (default 256) down GEMM 1 (direct store of the pair
  rows), the activation, GEMM 2 into the Q32 scratch with `output_topk = top_k`, then
  `moe_fixedpoint_to_half_rdna3`. That path is bit-deterministic whatever
  `VLLM_RDNA3_MOE_DETERMINISTIC` says; smaller batches (decode, verify) keep the scalar
  kernel. fp16 only: any other dtype raises `ValueError`, and a build without the op
  raises `RuntimeError` at construction. The tile is `VLLM_RDNA3_MOE_WMMA_BLOCK_M` (0 =
  automatic; 16, 32, 64) or, automatically, from the mean pair rows per expert
  `M * top_k / E`: below 20 tile 16, below 80 tile 32, else 64. That rule is a guess to
  tune on the GPU.
- Device code of the gfx1100 object (`llvm-objdump` and `llvm-readelf --notes`, confirmed
  on the object of the `b964e29f2d` build, nothing run). All 9 instantiations have 0
  scratch and 0 spills and, per kernel, 32 WMMA, 2 `s_barrier`, 6 `ds_store`:

  | Instantiation | VGPR | LDS (bytes) |
  |---|---|---|
  | fp16 BM16, direct or Q32 | 190 | 8,832 |
  | fp16 BM32, direct or Q32 | 148 | 17,664 |
  | fp16 BM64, direct or Q32 | 212 | 18,944 |
  | bf16 BM16 (direct) | 233 | 8,832 |
  | bf16 BM32 (direct) | 170 | 17,664 |
  | bf16 BM64 (direct) | 232 | 18,944 |

  Every tile is above the 128-VGPR occupancy target of the design: the occupancy this
  costs is the first thing to read off the first GPU run.
- Tests, all CPU so far: `test_rdna3_moe_wmma_model.py` (a Python emulation of the kernel
  mapping, index for index, that also parses the tile table of the `.cu`; 31 mutants times
  3 tiles are caught), `test_rdna3_moe_modes.py` (11 new dispatch tests: flow, threshold,
  override, fp16 rule, determinism over arrival order, hub contract, partial-sum identity,
  no host read), `test_rdna3_compile_guards.py` (2 new). Host-emulated builds of the real
  TU passed a 7-scenario matrix under ASan, UBSan and TSan, and 8 real-source mutants were
  caught by the value check or a sanitizer. The GPU test is
  `tests/kernels/quantization/test_rdna3_moe_wmma.py` (first the smoke subset, then `-k
  "e8-gate_up-m16 or e8-down-m16"`, then the whole file); it has not run on a GPU.
- Estimates from the model of the kernel ([INFERENCE], unmeasured): roughly 38, 46 and 42
  useful TFLOPS for BM16, BM32 and BM64, GEMM 1 about 2.9 to 3.5 ms and GEMM 2 about 1.5
  to 1.8 ms at M = 2048, BM32 best at 80 pairs per expert.

## 8. `rdna_ar` on gfx11 (Rdna3ArGfx11)

Changes: the gate in `cuda_communicator.py:203` is `on_gfx10x() or on_rdna3_hip("ar")`
(outer condition and the `VLLM_RDNA_AR` check unchanged, so the default stays RCCL);
`rdna_all_reduce.py` puts a group-wide agreement into the existing first
`all_gather_object` (collective count unchanged): each rank sends
`(dev, arch, allowed, marker text)` and all ranks disable `rdna_ar` together if the archs
are mixed, any rank lacks the opt-in, or any rank has a wedge marker. gfx1100 gets its
own marker (`rdna_ar_wedged_gfx1100`); the gfx10x marker path and log lines are
byte-identical (gfx1100 only appends `; arch gfx1100`).

gfx11 review, from disassembly of `rdna_allreduce.hip.o` built unmodified for both archs
(21 kernels on both; evidence under `~/w4a8_runs/rdna3_port/ar_isa/` on the box):

| Item | Verdict |
|---|---|
| `s_sleep`, `s_barrier`, fences, `atomicAdd`, `atomicCAS` (`rdna_allreduce.cuh`) | same instructions on both archs; gfx11 only adds `s_delay_alu` and `s_waitcnt_depctr` hints |
| `__hip_atomic_*` loads/stores | `glc dlc` on gfx1030, `glc` on gfx1100; visibility in practice UNKNOWN |
| `wall_clock64` | `s_sendmsg_rtn_b64 REALTIME` on gfx1100 versus `s_memrealtime`; the tick rate is UNKNOWN (the 100000 ticks per ms default at `rdna_allreduce.cu:54` and the abort record depend on it) |
| `hipExtMallocWithFlags(Uncached)` (`rdna_allreduce.cu:150`) | UNKNOWN whether the pool is exposed on gfx1100 and peer writes are coherent |
| peer access / IPC | API fine; cross-root W7800 P2P UNKNOWN (only V620 measured) |
| `oneshot<half>` resources | LDS 8 B, no scratch, vgpr 42, sgpr 75, wave32, same rsrc1 on both |
| hier kernels | not applicable at TP2 |

No `.cu` edit was needed. The default one-shot gate is 64 KiB, so 120 KiB (c8) messages need
`VLLM_RDNA_AR_MAX_KB` and `VLLM_RDNA_AR_ONESHOT_KB` of at least 128.
`tools/rdna/rdna_ar_stress.py` has its own marker constant and clears only the gfx10x marker.

## 9. Checkpoint and memory facts (read-only on the box)

`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` (snapshot `0939125b92...`): compressed-tensors
pack-quantized, W4A16 symmetric group 128, targets `Linear`. The `ignore` list keeps
`lm_head`, `embed_tokens`, `mtp.*`, `.ple.`, `visual`, the router `.gate`,
`hyper_connection`, `indexer`, `.linear_attn.`, `.self_attn.`, `shared_expert` unquantized.
So only the routed experts are W4A16; every dense layer is bf16 in the file and fp16 at
runtime. The MTP drafter is entirely BF16: `model_mtp.safetensors` is 4.86 GiB, of which
4.69 GiB are the 512 fused experts, 2.34 GiB per W7800 at EP2. `tools/rdna/quant_mtp_experts.py
<src> <dst>` derives a W4A16-MTP checkpoint (0.65 GiB per rank), which the hetero design
assumes. The recipe uses `ATTN=none` (QSA binds its own backend; `RDNA_ATTN` is a gfx10x
backend), `TUNABLEOP=0` and `TUNED_CONFIG=0` (the tuned MoE configs and the TunableOp rows
are gfx1030 artefacts; a row is an arch-specific solver index).

Memory per W7800 at TP2 (`[INFERENCE]` from sizes, to be checked at boot): experts 29.4
GiB + dense 5.3 GiB + MTP 0.65 GiB (W4A16) or 2.34 GiB (BF16) + KV 3.75 GiB, against 44.98
GiB. At TP2 the QSA decode group size is 12, which the legacy kernel handles through
`head_iters`.

## 10. Expected W7800 numbers (model, not measurements)

`tools/rdna/rdna3/perop_expectations.py` (stdlib; `--scenario pessimistic|central|optimistic`,
`--json`). Inputs: 864 GB/s peak with eta 0.8, kappa 1.0 (the HIP chain, as in the design
doc), `LATENCY_SCALE` 0.611 (design doc c1 value), V620 anchors from
`docs/rdna/monokernel/p0-results.md`. W is the number of rows per step: MTP2 gives 3 rows
per sequence, so W = 3, 24, 48 is c1, c8, c16. The model shares constants with the design
doc, so it is not an independent check: it matches at c1 (18.5 against 18.1 ms) and is
10 percent and 14 percent lower at W = 24 and 48 through the attention-core term.

Central target-forward sum without all-reduce, gaps and drafter: 18.5, 33.1, 46.9 ms at
W = 3, 24, 48 (pessimistic 23.9, 41.9, 61.8; optimistic 15.3, 27.6, 38.2).

| Op (per call, TP2, central) | W=3 us | W=24 us | W=48 us |
|---|---|---|---|
| HC per preamble | 38.2 | 38.2 | 42.5 |
| router | 23.4 | 23.4 | 24.3 |
| shared expert | 14.8 | 14.8 | 16.4 |
| attention core (mean over the 36:12 GDN:QSA layers) | 149.6 | 149.6 | 167.8 |
| GDN state read+write floor per GDN layer | 4.6 | 37.1 | 74.3 |
| QSA KV read floor per QSA layer | 9.1 | 72.8 | 145.6 |
| routed W4A16 experts per layer per rank (3.67 us per local expert) | 57.6 | 322.2 | 520.7 |
| same at the 5 us per expert gate | 77 | 438 | 709 |
| lm_head shard per pass | 919.7 | 919.7 | 1125.1 |

The design doc states, for W2 at c1: step 24.7 ms, with per layer HC 92 us, attention core
178, router 25, shared 18, experts 57, all-reduce 36; c8 is 45.7 ms.

Measured counterparts of the HC, shared-expert, GDN decode, lm_head and top-k rows are in
section 16.2 (G7, G8).

## 11. Verification done on the CPU

All Python ran through `tools/rdna/rdna3/cpu_pytest.sh` (amdsmi blocked, no GPU):

```bash
cd ~/vllm-rdna3-port-cpu && TREE=$PWD ~/w4a8_runs/rdna3_port/cpu_pytest.sh -q -p no:cacheprovider \
  tests/platforms/test_rdna3_dispatch.py tests/kernels/quantization/test_rdna3_moe_modes.py \
  tests/distributed/test_rdna_ar.py tests/distributed/test_rdna_ar_gate.py tests/tools/rdna \
  tests/platforms/test_rocm_amdsmi_index.py tests/platforms/test_rocm_capability_alias_config.py
```

Result on the final tree: 640 passed, 0 failed, 15 warnings (torch.jit deprecation and the
flash_attn Triton fallback) in 752 s, torch 2.12.0+rocm7.14.0.

Per suite: `test_rdna3_dispatch.py` (mocked-arch truth tables of every call site, the
family parser, the probe errors, the arch-detection cases; 123 tests at the time),
`test_rdna3_moe_modes.py` (115 tests, 17 mutations of `rdna3_moe.py` each killed by at
least one test), `test_rdna_ar_gate.py` (63 new tests, 41 of the gate/agreement tests fail
on the base sources; 128 passed together with the 65 existing `test_rdna_ar.py` tests),
`tests/tools/rdna` (recipe, fingerprint classifier, per-op model, existing serve tests).
`ruff`, `ruff format`, `typos`, `clang-format`, `shellcheck`, `mypy`, the tethering check and
the `torch.cuda` call check pass on every changed file (`pre-commit run --files`).

GPU test bodies were only dry-run on the CPU against fake ops (MoeDet: 30 passed, 9
deselected because they need hardware; ArGfx11: a 2-rank gloo dry run of `ar_probe.py`).
Collection check of all widened GPU test files (before the three V1 files below were
reverted): 1798 items, 0 collection errors.

Not widened (not done): `tests/v1/worker/test_sample_ops_rdna2.py`,
`test_mamba_rope_struct_ops_rdna2.py`, `test_spec_decode_ops_rdna2.py`. They carry
pre-existing lint and type debt (31 mypy errors, `torch.cuda.synchronize`, ruff findings)
that the pre-commit hooks flag as soon as the files are touched; they stay gfx10x-gated and
the `v1` family has no gfx1100 parity test for those three op groups until that debt is paid.

Phase 2, on the same CPU tree with the real `_custom_ops.py` wrapper and no shim:
`test_rdna3_moe_modes.py` 150 passed in 1,042 s (11 new WMMA dispatch tests);
`test_rdna3_moe_wmma_model.py` with `test_rdna3_compile_guards.py` 244 passed, 9 skipped
(checks of non-gfx1100 builds); `tests/tools/rdna` 346 passed, 9 failed in 1,545 s, all nine
120 s timeouts of `test_mlock_pinned_proof.py` at a box load of about 100 (rerun alone at a
load of 11: 39 passed in 261 s); `test_gpu_g0_g7.py` 11 passed and `test_trunc_checkpoint.py`
26 passed, off the box too. `ruff`, `ruff format`, `typos`, `clang-format`, `shellcheck`,
`mypy` and the tethering check pass on every file of this phase (`pre-commit run --files`).

## 12. GPU test plan (W7800 pair, HIP 0 and 5)

Rules: both W7800 idle, the box lease and `boot.lock` rules of `AGENTS.md`, never mix a
V620 into the same process, `ROCR_VISIBLE_DEVICES` unset, `HIP_VISIBLE_DEVICES=0` for
single-GPU steps and `0,5` for TP2. Stop at the first failing step.

Steps G0 to G8 and G16 run from `tools/rdna/rdna3/gpu_g0_g7.sh` (HIP 0 only, lease, quiet
file, thermal and KFD-eviction watch; `--from`, `--to`, `--only`, `--keep-going`,
`--labels`, `--dry-run`, `--list`) with `~/Apps/vllm/venv-7.14.0` and an exact-SHA tree;
section 16 has the results. The W7800 half of G13 is the runner's last step, outside the
default range (`--only G13`, see the note below the table). The rest of G9 to G15 needs
HIP 5 or a serve and has not run.

| Step | What | Command and gate |
|---|---|---|
| G0 | Arch and family parse | `HIP_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName)"` prints `gfx1100`; `HIP_VISIBLE_DEVICES=0 python -c "from vllm.platforms import rocm; print(rocm._GCN_ARCH)"` prints the same, read from the KFD topology (no HIP context before the first device use) |
| G1 | Build | the command of section 4 with the 2-arch list; `ops-*.json` shows `loaded: true`, 177 ops |
| G2 | GDN parity, first the two kernels with inline `v_dot2c_f32_f16` | `pytest -q -x tests/kernels/quantization/test_gdn_prefill_prep_rdna2.py tests/kernels/quantization/test_gdn_prefill_kkt_rdna2.py`, then `test_gdn_prefill_solve_wy_rdna2.py`, `test_gdn_prefill_delta_h_rdna2.py`, `test_gdn_prefill_o_rdna2.py`, `test_gdn_decode_rdna2.py`, `tests/kernels/mamba/test_causal_conv1d_update_rdna2.py`, `test_causal_conv1d_null_block.py`. The solve_wy tolerance (1e-3) may be marginal where the Triton reference lowers to WMMA; report, do not loosen |
| G3 | Dense, QSA, HC, SE parity | `tests/kernels/quantization/test_gemv_f16_rdna2.py`, `tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py`, `test_qsa_rdna2_int4.py`, `test_qsa_metadata_null_block_rdna2.py`, `tests/models/qwen4_exp/test_qsa_metadata_rdna2.py`, `test_qsa_amd.py`, `test_hc_ops_rdna2.py`, `test_hc_prefill_rdna2.py`, `test_se_rdna2.py` (the mrope tests of `test_qsa_rdna2.py` need `VLLM_RDNA3_HIP_OPS=all`, which the runner sets for every pytest group; the new fixtures of the other files patch the family set) |
| G4 | MoE | `tests/kernels/quantization/test_rdna3_moe_w4a16.py` (existing tests: default path unchanged; 10 new GPU tests: hub shapes, `-1` masking, Q32 bit identity over 50 repeats including M=2048, Q32 against default within fp16 noise, block-size sweep, split-expert partial sums, NaN poisoning, cast reference, malformed accumulators, graph capture and replay), `test_rdna3_compile_guards.py`, `tests/kernels/moe/test_rdna2_unquantized_moe.py` |
| G5 | Upstream skinny GEMMs and top-k | `tests/kernels/quantization/test_rocm_skinny_gemms.py` (wvSplitK n up to 5 on gfx11, LLMM1); `tests/kernels/test_top_k_per_row.py -k decode` runs for `auto`, `radix`, `insertion` |
| G6 | V1 ops that were widened | `tests/kernels/test_mrope_rdna2.py`, `tests/kernels/core/test_vit_pos_embed_rdna2.py`, `tests/v1/worker/test_input_batch_ops_rdna2.py`, `test_worker_ops_rdna2.py`, `test_gpu_block_table.py`, `tests/v1/sample/test_topk_topp_ops_rdna2.py` |
| G7 | Per-op latency against the model | graph-timed at W = 3, 24, 48 on the TP2 and TP1 hub shapes, one runner cell per route: `tools/rdna/port_v031/decode_ops_bench.py` (HC; SE with `--i-se 320` or `640`, the fused op or, with `--torch`, the composition the model runs with the `se` family off), `tools/rdna/rdna3/gdn_decode_bench.py --geometry flashnext-tp2` or `flashnext-tp1` (conv update and recurrent step, Triton against the `gdn` family), `lm_head_bench.py --n 124160` or `248320 --k 2560`, `gemv_crossover_bench.py --cold` (without `--cold` it times eager launches). Compare with `python tools/rdna/rdna3/perop_expectations.py --scenario central`. A family is worth enabling only if it is at least as fast as the default and the model is within the pessimistic-to-optimistic band |
| G8 | Top-k algo timing | `tools/rdna/rdna3/topk_decode_bench.py` times `top_k_per_row_decode` (graph-replayed; `auto`, `radix`, `insertion`; the selected indices are compared) at 1k to 131k live columns, and the runner repeats the decode tests with `VLLM_TOPK_DECODE_ALGO=insertion`. Keep the faster default |
| G9 | `rdna_ar` | `HIP_VISIBLE_DEVICES=0,5 HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=2 torchrun --standalone --nproc_per_node=2 tools/rdna/rdna3/ar_probe.py --out ar_probe_w7800.json --timeout-test` with `VLLM_RDNA3_HIP_OPS=ar`. Gates: zero mismatches against the fp32 rank-order reference at 8, 64, 120, 256 KiB and 10 MiB; p50 at most 20 us per collective at 8 KiB fp16 in graph mode (the design doc's TP2 figure for RCCL is 36 us); no abort record or marker; the timeout phase aborts within 0.8 to 1.5 times 1500 ms |
| G10 | TP2 serve, safe baseline | `bash tools/rdna/serve_rdna.sh RECIPE=flashnext-w2-mtp2 PCI_BUSES="cb 07" MODEL=<checkpoint> VENV=<fat-build venv>` with `VLLM_RDNA3_HIP_OPS` empty, then `tools/rdna/port_v031/serve_validate.sh` against port 18097. Design gates (section 9.5): probes 3 of 3, PPL 4.75 to 4.85, MTP acceptance 1.84 and 2.01, c1 at most 28 ms per step, c8 at most 52 ms, memory within 44.98 GiB; 16k TTFT at most 4.5 s and 16k chunk at most 0.60 s need the WMMA prefill kernel (S1b) and are expected to fail with the scalar kernel |
| G11 | Family ladder | repeat G10 with `VLLM_RDNA3_HIP_OPS=hc`, then `hc,se`, `hc,se,qsa`, `hc,se,qsa,gdn`, then add `dense`, `moeu` (BF16 MTP only), `v1`, and last `ar` with `VLLM_RDNA_AR=1`. Promote a family to default only if quality gates hold and step time improves at c1 or c8; otherwise keep it opt-in |
| G12 | Determinism | `VLLM_RDNA3_MOE_DETERMINISTIC=1`: 50 repeated greedy requests give identical logprobs; measure the Q32 step-time cost at c1 and c8 |
| G13 | Cross-arch numerics | same truncated-layer prompts through V620 (gfx1030) and W7800 (gfx1100) separately, compare per-layer hidden states and the PPL probe; expect fp16-level agreement, the first layer that diverges names the op. `tools/rdna/rdna3/trunc_checkpoint.py` cuts the real checkpoint to its first four layers (3 GDN, 1 QSA), `trunc_e2e.py run` dumps the per-layer states, `trunc_e2e.py compare` reports them |
| G14 | Prefix-cache NaN | the `FULL_AND_PIECEWISE` plus prefix-caching repro of `docs/rdna2/flash-next-prefix-cache-corruption.md` |
| G15 | A/B items | `rms_norm` IR priority (`rocm.py` near `:1640`), `LLMM1` against `wvSplitK`, `gemv_f16_rdna2` against `wvSplitK` on non-padded layers (`maybe_pad_weight_avoid_cache_aliasing` pads rows when `K * elem % 2048 == 0`, which fails the contiguity gate of `gemv_f16_rdna2`), int8 shadow, W7800 TunableOp rows (`tools/rdna/prof/tune_missing_gemms.sh`, TunableOp ordinals of the W7800) |
| G16 | Tools | print the per-layer QSA route and `rotary_emb.mrope_interleaved` at startup; confirm gfx1100 rocBLAS and hipBLASLt Tensile libraries exist |

G13 on the W7800 is `gpu_g0_g7.sh --only G13`: three engine starts on the truncated
checkpoint (`~/w4a8_runs/rdna3_port/trunc/ckpt-L4`), `G13-base` (the variables of the
W2 recipe, no family), `G13-all` (`VLLM_RDNA3_HIP_OPS=all`) and `G13-wmma` (all plus
`VLLM_RDNA3_MOE_WMMA=1`), then `G13-compare-all` (base against all) and
`G13-compare-wmma` (all against wmma). Six prompts of 20 to 5000 tokens (the two longest
give the WMMA kernel its 32 and 64 row tiles) and the PPL probe; the dumps `e2e-*.pt` stay
in the run directory, so a V620 run of `trunc_e2e.py run` can be compared with
`trunc_e2e.py compare` against `e2e-base.pt`. The step refuses to start with less than
130 GiB of available host memory (the pinned 95 GiB table).

The decode cells added after the first run (hetero stage 1, section 16.7) are in the same
G7 list: `G7-se-tp1-op-48` and `G7-se-tp2-op-48` (shared-expert row cap raised to 48),
`G7-qsa-tp1-default`, `-hip`, `G7-qsa-tp2-default`, `-hip`, `G7-qsa-tp1-hip-tight` and
`G7-qsa-tp1-hip-insertion` (`qsa_decode_bench.py`), `G7-moe-decode`, `-hub`, `-h1`, `-k-h1`,
`-k-h2`, `-block-m`,
`G7-moe-prefill`, `-hub` and `-block-m` (`moe_decode_bench.py`, the scalar kernel against the
S1b WMMA kernel at 256 to 2048 tokens), and `G7-lm_head-tp1-tune`, `-tuned`,
`G7-lm_head-tp2-tune`, `-tuned` (PyTorch TunableOp). A cell takes several `NAME=value` words
as its extra environment.

## 13. Optimization plan

Ordered by expected gain on the W2 baseline; each item names the evidence it needs.

1. WMMA W4A16 MoE prefill kernel (design slice S1b): the scalar kernel at an effective
   0.14 to 0.10 of hub throughput gives a W2 chunk of 1.0 to 1.4 s in the design model,
   against the 0.60 s gate. WMMA fp16 peak is 90.4 TFLOPS per W7800.
2. W8A16 or int8 dense weights: about 3.9 ms at c1 according to the design doc, from the
   4.64 GiB per rank dense read (7.2 ms floor at 691 GB/s).
3. lm_head: three passes cost 2.8 ms per step in the model (shard 919.7 us per pass); an
   argmax-only draft pass or W8A16 shrinks it.
4. HC family: measured on the W7800 (section 16.2), 32.9, 88.5 and 90.4 us on the torch route
   against 33.2, 41.7 and 59.8 us on the HIP chain at W = 3, 24, 48 (model: 38.2, 38.2, 42.5).
5. `rdna_ar` against RCCL, 1 to 1.5 ms per step at c1 if the 20 us probe gate holds.
6. `rms_norm` IR priority, W7800 TunableOp rows; top-k: `VLLM_TOPK_DECODE_ALGO=insertion` is
   1.3 to 2 times faster than the radix default at 4k to 8k live columns (section 16.2).
7. WMMA QSA prefill (backlog branch `origin/backlog/rdna3-attn-tiles`).
8. Shared expert beyond 8 rows: the fused op takes 23.6 us at W = 24 (TP2) against the 14.8 us
   of the model and falls back to the 62 us torch composition above 32 rows
   (`_SE_DECODE_MAX`), which is where W = 48 sits.
   The tile pair now takes up to 64 rows (a host-side check, section 16.7); the cap stays 32
   by default and `VLLM_RDNA_SE_DECODE_MAX=48` opts in. No GPU number yet.
9. GDN decode on gfx1100: the HIP `gdn` route wins only at 1 to 2 sequences; a switch to Triton
   from 3 sequences on, or a HIP recurrent kernel tuned for 8 to 16 sequences, is worth 0.45 ms
   (TP2, c8) to 2.3 ms (TP1, c16) per step (section 16.2).
10. lm_head at W = 24 and 48: rocBLAS needs 1.47 and 1.21 ms per pass at TP2 against the 0.92 ms
    of the model, and `gemv_f16_rdna2` is 2.4 times slower than rocBLAS at 24 rows.
    TunableOp cells for both vocabulary shards (`G7-lm_head-tp*-tune`) are in the runner; a
    WMMA skinny kernel is not written (section 16.7).

## 14. Risks and open decisions

Risks:

- The per-op G0 to G8 and G16 run passed on the W7800 (section 16), but no serve has run
  (G9 to G15): the HIP families are validated op by op, not end to end, and the WMMA kernel
  of section 7.1 has never executed on a GPU. The inline `v_dot2c_f32_f16` of
  `gdn_prefill_prep_rdna2.cu` (`:75-79`) and `gdn_prefill_kkt_rdna2.cu` (`:78-82`) gives
  correct results on gfx1100 (77 parity tests passed).
- Q32 performance and NaN semantics (section 7); `rdna_ar` coherence, wall-clock rate and
  cross-root P2P on gfx1100 (section 8); asymmetric opt-in under Ray or mixed hosts can
  leave a rank waiting in the first exchange (same class of hazard as the old gate).
- The sampler `auto` select is radix on gfx11 and 1.3 to 2 times slower than insertion at 4k to
  8k live columns (section 16.2); the default was left alone.
- `moe_q_gemm_rdna3.cu` clang-format and pre-commit were applied after the first oracle
  run; the oracle was repeated on the formatted sources (section 4.2 table).
- The G-run used the phase-1 `.so` files of `30b6a47eca` plus a Python overlay; the tree was
  rebuilt with S1b afterwards (section 16.5) and has not run on a GPU since.

Open decisions for the user:

1. Flip defaults of the HIP families after G7 and G11, per family, or keep them opt-in.
2. MTP checkpoint: derive the W4A16-MTP checkpoint (0.65 GiB per rank, assumed by the
   hetero design) or run the BF16 drafter (2.34 GiB per rank, Triton `fused_moe` or `moeu`).
3. Start S1b (WMMA W4A16 MoE prefill) now; the prefill gates of W2 depend on it.
4. Accept `VLLM_RDNA3_MOE_DETERMINISTIC=1` as the hub default if G12 shows an acceptable
   cost; the hetero contract requires it.
5. Pay the lint and type debt of the three V1 op test files so they can be widened.
6. Flip the family defaults on the G7 data: `hc` and `se` are faster at every decode width
   they apply to, `gdn` only at 1 to 2 sequences; `dense`, `v1` and `moeu` have no timing yet.
7. Put `VLLM_TOPK_DECODE_ALGO=insertion` into the W7800 recipe (section 16.2).
8. The pairwise `v_new` tolerance of the delta-h test was widened from 1e-3 to 2e-3 on the
   evidence of the fp64 oracle (section 16.3): keep that, or hold 1e-3 and xfail the case.
9. `VLLM_RDNA_SE_DECODE_MAX=48` in the hub recipe, if the `G7-se-tp*-op-48` cells beat the
   torch composition (section 16.7); the default of 32 leaves gfx1030 as it was.

## 15. Files

- Source: `vllm/platforms/rdna.py`, `vllm/platforms/rocm.py`, `vllm/envs.py`,
  `CMakeLists.txt`, `csrc/rocm/torch_bindings.cpp`, `csrc/rocm/ops.h`,
  `csrc/rocm/moe_q_gemm_rdna3.cu`, `csrc/sampler.cu`, `vllm/_custom_ops.py`, the call sites
  of section 2, `vllm/distributed/device_communicators/cuda_communicator.py` and
  `rdna_all_reduce.py`.
- Tools: `tools/rdna/rdna3/` (`build_rocm_c.sh`, `rocm_c_load_check.py`,
  `code_object_fingerprint.py`, `cpu_pytest.sh`, `perop_expectations.py`, `ar_probe.py`),
  `tools/rdna/recipes/flashnext-w2-mtp2.env`, `tools/rdna/serve_rdna.sh` (new env keys).
- Tests: `tests/platforms/test_rdna3_dispatch.py`, `tests/kernels/quantization/test_rdna3_moe_modes.py`,
  `tests/distributed/test_rdna_ar_gate.py`, `tests/tools/rdna/test_flashnext_w2_recipe.py`,
  `test_code_object_fingerprint.py`, `test_perop_expectations.py`, plus the widened gfx1100
  gates of 25 existing per-op GPU test files and the 10 new tests of
  `test_rdna3_moe_w4a16.py`.
- Phase 2 source: `csrc/rocm/moe_q_gemm_rdna3_wmma.cu`, `CMakeLists.txt`, `csrc/rocm/ops.h`,
  `torch_bindings.cpp` (GFX1100 block), `vllm/_custom_ops.py`, `rdna3_moe.py` (WMMA
  dispatch), `vllm/envs.py` (three variables), `vllm/platforms/rocm.py` (amdsmi stand-ins).
- Phase 2 tools: `tools/rdna/rdna3/gpu_g0_g7.sh`, `gpu_preflight.py`, `build_fat.sh`,
  `gdn_decode_bench.py`, `gdn_delta_h_oracle.py`, `topk_decode_bench.py`, `tensile_probe.py`,
  `trunc_checkpoint.py`, `trunc_e2e.py`; `tools/rdna/port_v031/decode_ops_bench.py` (`--i-se`,
  `--torch`) and `gemv_crossover_bench.py` (gemv only up to 32 rows).
- Phase 2 tests: `tests/kernels/quantization/test_rdna3_moe_wmma.py` (GPU),
  `test_rdna3_moe_wmma_model.py` (CPU), `tests/tools/rdna/test_gpu_g0_g7.py`,
  `test_trunc_checkpoint.py`, `tests/platforms/test_rocm_amdsmi_index.py` (one new case); the
  per-op GPU tests are widened to the TP1 and TP2 head counts, and
  `test_gdn_prefill_delta_h_rdna2.py` gained an fp64 oracle test.
- Hetero stage 1: `tools/rdna/rdna3/qsa_decode_bench.py`, `moe_decode_bench.py`, the new G7
  cells of `gpu_g0_g7.sh`, `csrc/rocm/hc_prefill_rdna2.cu` (the row limit of the shared-expert
  tile GEMMs), `vllm/model_executor/layers/rdna_ops.py` (`_SE_DECODE_MAX` clamp and ticket
  check), `vllm/_custom_ops.py` (docstrings), and the new cases of
  `tests/platforms/test_rdna3_dispatch.py`, `tests/models/qwen4_exp/test_se_rdna2.py` and
  `tests/tools/rdna/test_gpu_g0_g7.py`.

## 16. Phase 2: S1b, and G0 to G8 plus G16 on the W7800 (2026-11-07)

### 16.1 Setup

- GPU: HIP 0, the W7800 on bus cb, and nothing else (HIP 5 was never opened). Venv
  `~/Apps/vllm/venv-7.14.0` (torch 2.12.0+rocm7.14.0 with gfx1100 device code; the other venvs
  of the box are gfx1030-only) with `PYTHONPATH=~/vllm-rdna3-flashnext`, the fat tree of
  `30b6a47eca` plus a Python overlay of the phase-2 tests and tools, `vllm/platforms/rocm.py`,
  `vllm/envs.py` and `rdna3_moe.py` (the WMMA dispatch, off by default). `overlay.txt` in the
  tree lists the sha256 of every overlaid file; the `.so` files are the phase-1 build.
- `tools/rdna/rdna3/gpu_g0_g7.sh --tree <tree> --venv <venv>` takes the lease
  `~/w4a8_runs/lanes/lease/Rdna3Port.w7800cb`, checks that HIP 0 is the W7800 of bus cb with
  under 2 GiB in use, runs every GPU command with `HIP_VISIBLE_DEVICES=0` and without
  `ROCR_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES` and `VLLM_RDNA3_HIP_OPS` (pytest groups
  set `VLLM_RDNA3_HIP_OPS=all`, benchmark cells their own family), samples the junction
  temperature every second (abort at a 5 s mean of 90 C or any sample of 95 C) and runs each
  benchmark cell in a measured window: quiet file touched, `boot.lock` free, no docker
  buildx, KFD eviction counters unchanged (a cell is retried once; exit 123 marks an invalid
  window). Logs and `summary.tsv` go to `~/w4a8_runs/rdna3_port/gpu/<timestamp>/`.
  `tests/tools/rdna/test_gpu_g0_g7.py` checks the plan on any machine: HIP 0 only, no
  inherited family variable, every named test file exists, the G7 cells pair their routes.
- Five runs on 2026-11-07: 14:12 (stopped at the first failure), 14:35 (`--from G2
  --keep-going`), 15:44 and 15:55 (reruns of the failed or fixed commands with `--labels`),
  16:00 (G8 over the long columns). Junction temperature never passed 57 C.

### 16.2 Results

| Step | Result |
|---|---|
| G0 | pass: W7800 gfx1100, 44.98 GiB, 8 families, 177 `_rocm_C` ops |
| G1 | pass: sha, overlay list, load of every extension |
| G2 | prep and kkt 77 passed (the inline `v_dot2c_f32_f16` is right on gfx1100); solve_wy, delta_h and o 722 passed, 176 skipped (state dtype without an initial state), 1 failed (the delta-h outlier, 16.3); decode and conv1d 140 passed; delta_h alone with the 18 oracle cases 628 passed, 176 skipped, 1 failed, the same element |
| G3 | gemv 90 passed; QSA 633 passed, 4 failed (mrope without the family variable; 4 passed with it); metadata and qsa_amd 91; HC, HC prefill, SE 354 |
| G4 | compile guards and W4A16 MoE 169 passed, 7 skipped (checks of non-gfx1100 builds); unquantized MoE 20 passed (1 failed before `b8b99e3b73`, 16.3) |
| G5 | skinny GEMMs 168 passed, 16 xfailed, 2484 skipped (184 failures before the test changes, 16.3); top-k decode 102 passed, 22 skipped (gfx950 launch configs) |
| G6 | mrope and vit 80 passed; V1 worker ops 109 passed (21 failures before the test fixes, 16.3); top-k/top-p 22 passed |
| G7 | the cells below |
| G8 | the table below; the decode tests on `VLLM_TOPK_DECODE_ALGO=insertion`: 102 passed, 22 skipped |
| G16 | 155 BLAS device-code files for gfx1100 (`blas_lib_gfx1100.kpack`, rocBLAS Tensile, hipBLASLt); rocBLAS and hipBLASLt fp16 and bf16 GEMMs within tolerance from (M, N, K) = (1, 2560, 2560) to (48, 248320, 2560) |

G7, graph-timed microseconds per call at W = 3 / 24 / 48 (the model is the central scenario
of section 10):

| Op | Model | Measured |
|---|---|---|
| HC preamble, torch route | | 32.9 / 88.5 / 90.4 |
| HC preamble, HIP `hc` | 38.2 / 38.2 / 42.5 | 33.2 / 41.7 / 59.8 |
| Shared expert TP2, fused op | 14.8 / 14.8 / 16.4 | 15.3 / 23.6 / 61.6 (torch fallback above 32 rows) |
| Shared expert TP2, torch composition | | 59.1 / 62.2 / 62.0 |
| Shared expert TP1, fused op | | 22.8 / 35.3 / 77.2 |
| Shared expert TP1, torch composition | | 74.2 / 73.8 / 77.6 |
| GDN decode TP2, Triton: conv update + recurrent | recurrent floor 4.6 / 37.1 / 74.3 | 6.0 + 26.4 / 4.6 + 33.2 / 5.3 + 52.8 |
| GDN decode TP2, HIP `gdn` | | 6.3 + 9.0 / 8.8 + 41.6 / 13.4 + 87.0 |
| GDN decode TP1, Triton | | 4.1 + 19.3 / 5.3 + 50.7 / 7.0 + 114.1 |
| GDN decode TP1, HIP `gdn` | | 6.1 + 11.6 / 13.5 + 88.3 / 22.3 + 162.6 |
| lm_head TP2 (N 124,160), `F.linear`, ms | 0.92 / 0.92 / 1.13 | 0.898 / 1.466 / 1.208 |
| lm_head TP2, `gemv_f16_rdna2`, ms | | 0.888 / 3.474 / not defined above 32 rows |
| lm_head TP1 (N 248,320), `F.linear`, ms | | 1.769 / 2.429 / 2.823 |
| lm_head TP1, `gemv_f16_rdna2`, ms | | 1.757 / 6.907 / not defined |

GDN decode recurrent step, Triton against HIP, microseconds per layer at 1, 2, 3, 4, 6, 8, 12,
16 sequences (3 tokens each, so W = 3 to 48): TP2 26.4/9.0, 27.6/11.6, 22.0/19.7, 23.5/23.6,
29.0/32.8, 33.2/41.6, 41.2/60.6, 52.8/87.0; TP1 19.3/11.6, 23.5/23.6, 28.9/32.9, 33.3/40.4,
41.1/62.6, 50.7/88.3, 65.5/126.1, 114.1/162.6. The HIP conv update is slower than Triton from
3 sequences on (TP2: 6.4 against 4.2 us at 3, 13.4 against 5.3 us at 16).

G8, `top_k_per_row_decode` with top-k 2048, microseconds (`auto` is radix on gfx1100):

| Live columns | rows 3 / 24 / 48, radix | rows 3 / 24 / 48, insertion |
|---|---|---|
| 1,024 | 5.9 / 5.9 / 5.9 | 5.9 / 5.9 / 5.9 |
| 4,096 | 15.1 / 15.2 / 18.6 | 8.9 / 9.0 / 9.3 |
| 8,192 | 16.0 / 16.2 / 19.8 | 12.4 / 12.0 / 12.8 |
| 16,384 to 131,072 | 17.1 to 63.1 | the same within 1.5 us |

### 16.3 Failures and their causes

1. `amdsmi`. venv-7.14.0 has no `amdsmi` module. The `except ImportError` branch of
   `vllm/platforms/rocm.py` defined no `amdsmi_init`, so `RocmPlatform.get_device_name()`
   raised `NameError` on the startup path (`startup_plan.py:65`, the Triton MoE and SSM
   config lookups, `arg_utils`); G4-2 hit it first. Fixed as `b8b99e3b73` (stand-ins that
   behave like an amdsmi without GPUs; test
   `test_missing_amdsmi_module_acts_like_an_amdsmi_without_gpus`). A run from venv-7.14.0
   needs `rocm.py` at that commit or later.
2. Delta-h outlier (G2), `parity[True-False-seq_lens2-48-16]`: 1 of 3,932,160 `v_new`
   elements, index (0, 360, 18, 88), differs by 1.22e-3 (tolerance 1.18e-3 there).
   `gdn_delta_h_oracle.py` compares HIP and Triton with an exact fp64 recurrence on the same
   inputs. For that case: h, HIP max error 1.163e-3 and rms 1.298e-4, Triton identical;
   `v_new`, HIP max 2.216e-3 and rms 2.510e-4, Triton identical; final state, HIP max
   6.32e-4 and rms 1.064e-4, Triton 7.26e-4 and 1.062e-4. At the element: HIP -0.182861,
   Triton -0.181641, exact -0.182237, so they are 6.2e-4 and 6.0e-4 (about 5 fp16 ulp) from
   the exact value, on opposite sides of it. Over the grid of 48 cases (H, Hg in (12, 4),
   (4, 4), (24, 8), (48, 16); 63, 64, 100, 137, 192, 640 tokens; with and without an initial
   state) and 3 outputs, 144 rows: one row outside the pairwise tolerance (this one) and none
   where the HIP rms error exceeds 1.25 times Triton's (the ratio reads 1.00 on every row).
   Verdict: HIP and Triton are equally accurate; the mismatch is the tail of two correlated
   fp16 roundings, not a gfx1100 defect. The pairwise `v_new` atol went from 1e-3 to 2e-3 and
   `test_gdn_prefill_delta_h_rdna2_vs_fp64_oracle` (18 cases, rms against fp64 at most 1.25
   times Triton's) guards the accuracy itself. The failing inputs are saved by the runner as
   `delta_h_failing_case.pt`; `gdn_delta_h_oracle.py --replay` on a V620 compares the
   gfx1030 kernel bit for bit (no V620 slot was free).
3. Stale tests, not products: `test_worker_ops_rdna2.py` called the Triton gather kernel
   without its four new arguments (`num_computed_tokens_ptr`, `sliding_window_reach_ptr`,
   `block_sizes_ptr`, `MASK_SLIDING_WINDOW`; the HIP op takes the same), the slot kernel and
   op without `kernel_block_sizes`, and the write kernel with a 1-D grid although it is 2-D
   since one program covers a 1,024-element chunk: the five `content_len >= 1025` cases
   compared a truncated Triton write. `_apply_write_case` now has a torch oracle: the HIP ops
   matched it at every length (1,025 and 4,096 included) before the fix, and the Triton
   kernel matches it with the production grid. Neither side is a gfx1100 defect. The mrope
   tests need `VLLM_RDNA3_HIP_OPS=all` (gfx1100 opts in per family; the runner sets it).
   `test_rocm_skinny_gemms.py`: the fp8 skinny test is skipped on gfx11 (`supports_fp8()` is
   true on RDNA but its reference `torch._scaled_mm` needs MI300+), and LLMM1 is an xfail at
   K above 4096 on gfx11 (item 4).
4. Behaviours worth knowing: `LLMM1` returns wrong results at K = 6144 and 8192 on gfx1100
   (100 percent of the elements; K up to 4096 is right). `rocm_unquantized_gemm_impl` never
   reaches it (`use_wvsplitk` takes every n of 1 to 5 with m above 8 first), so no model
   path is affected, a changed dispatch or a direct caller would be. The upstream top-k test
   file is 30 minutes on this box because the batch-2048 cases time a per-row `torch.topk`
   reference; the runner deselects those (the serving batches stay).

### 16.4 What the data says about the families

- `hc`: on. Equal at W = 3, 2.1 times faster at 24, 1.5 times at 48 than the torch route.
- `se`: on. The fused op is 3.9 and 3.3 times faster than the torch composition at W = 3
  (TP2, TP1), 2.6 and 2.1 times at 24, and the same at 48, where it falls back to torch.
  The op does not read `VLLM_RDNA3_HIP_OPS`; the model-level gate
  (`_rdna_shared_expert_ok`) does. At W = 3 that is 2.1 ms of the 18.5 ms step (48 calls, TP2).
- `gdn`: off beyond 2 sequences. HIP wins at 1 sequence (TP2: 17.1 us per layer, 0.62 ms per
  step; TP1: 5.7 us, 0.21 ms) and at 2 on TP2, ties at 2 on TP1, and loses from 3 to 4
  sequences on, by 0.45 ms (TP2) or 1.65 ms (TP1) per step at c8 and 1.5 or 2.3 ms at c16.
- lm_head: `gemv_f16_rdna2` equals rocBLAS at 3 rows (716 to 723 GB/s) and loses at 24; the
  model's 0.92 ms is right at W = 3, and the measured pass at 24 rows is 60 percent above it.
- Top-k: set `VLLM_TOPK_DECODE_ALGO=insertion` on gfx1100 (never slower in this range, the
  decode tests pass on it); the gain is 45 to 110 us per step over the 12 QSA layers.
- The model and the measurements agree at W = 3 (HC -13 percent, SE +3, lm_head -2), which is
  where the central scenario was anchored; they diverge at W = 24 and 48 for the ops whose
  kernels were tuned for gfx1030 (HC +9 and +41 percent, SE +59 percent and the torch
  fallback, GDN, lm_head).
- Not timed here: `dense` (the cold `gemv_crossover_bench.py` cells of the runner, TP1 and
  TP2 shapes, were added after this run), `v1`, `moeu`, `ar`, and the whole serve.

### 16.5 S1b build and the gfx1030 oracle (CPU only, 2026-11-07 16:10 to 16:15)

- `build_fat.sh` on `~/vllm-rdna3-flashnext` at `b964e29f2d`: 127 s incremental (three
  relinks of `_C`, `_C_stable_libtorch` and `_moe_C_stable_libtorch`, `torch_bindings.cpp`,
  the new WMMA TU, the `_rocm_C` link), `build_exit=0`, `load_check=0`, 178 `_rocm_C` ops
  (177 before S1b), 0 undefined project symbols. The gfx1100-only build of
  `~/vllm-rdna3-g1100` loads with 45 ops (44 before).
- Code-object oracle against the untouched base (`origin/rdna_extras` at `27e883f3ce`,
  same-path in-place rebuild, `code_object_fingerprint.py compare`). gfx1030: 69 objects
  IDENTICAL, 1 STUB-CHURN (`moe_q_gemm_rdna3.hip.o`, +5 kernel symbols of its stub, as in
  phase 1), 1 NEW-STUB (the WMMA TU, 9 stub kernel symbols), 0 CHANGED. gfx1100: 69
  IDENTICAL, 1 CHANGED (`moe_q_gemm_rdna3.hip.o`, as in phase 1), 1 new object (the WMMA
  TU: 9 kernels, 207,012 bytes of text, not a stub). S1b adds no gfx1030 device code and
  changes no other object on either arch.
- The resources of the 9 gfx1100 kernels, re-read from this build with
  `clang-offload-bundler` and `llvm-readelf --notes`, equal the table of section 7.1.

### 16.6 Not done, and what each needs

- The WMMA kernel on a GPU (section 7.1): `gpu_g0_g7.sh --only G4 --labels
  G4-wmma-1,G4-wmma-2,G4-wmma-3` (smoke, one shape per stage, the whole file; without
  `--keep-going` the first time), then its timing at 256 to 2048 tokens (the
  `G7-moe-prefill*` cells, section 16.7).
- The delta-h group rerun after the tolerance change, the cold gemv cells, and the shared
  expert and GDN rows once more on the rebuilt tree: one W7800 window.
- `gpu_g0_g7.sh --only G13` on the W7800 (see the note below the table of section 12): the
  first four layers of the real checkpoint, 8.7 GiB plus the 95 GiB PLE table pinned in
  host memory. The V620 side (`trunc_e2e.py run` on a V620 slot, then `compare` against the
  W7800 `e2e-base.pt`) needs a slot.
- The rest of G9 to G15 (HIP 5, or a serve).

### 16.7 Hetero stage-1 asks: code in the tree, no GPU numbers yet

`docs/rdna2/hetero-moe-design-2026-11-07.md` (branch `feat/hetero-moe`) asks for four more
figures on the hub shapes (one W7800 as TP1, or the pair as TP2; 3, 24 and 48 rows). What
exists, per ask:

1. Shared expert above 32 rows. The two tile GEMMs of the fused op (`rdna_se_gate_up_tile`,
   `rdna_se_down_tile`) were limited to 32 rows by a `TORCH_CHECK`, but the kernel under them
   (`gemm_tn_splitk_k`) already tiles the rows in blocks of 32 (a row-tile grid axis, tail
   rows masked). The check now says 64; no device code changed. `_SE_DECODE_MAX` (variable
   `VLLM_RDNA_SE_DECODE_MAX`, default 32) is clamped to 64. The wave kernels still stop at 32
   rows, so above 32 the op runs only when both halves run as tiles: fp16 weights,
   `VLLM_RDNA_SE_TILE_MIN` and `VLLM_RDNA_SE_DOWN_TILE` on, shapes divisible by 32, and the
   output tiles of the gate_up kernel fit its 128 arrival tickets,
   `(2I / 32 + 1) x ceil(rows / 32) <= 128` (I = 640 and 320 pass at 64 rows; an I of 1152
   passes up to 32 rows only). Anything else is the torch composition, as before. Up to 32
   rows the routes are those of before (twelve CPU cases in `test_rdna3_dispatch.py`; the GPU
   tests of `test_se_rdna2.py` cover two row tiles, 64 rows, the ticket refusal and a graph
   replay at 48 rows). Cells `G7-se-tp1-op-48` and `G7-se-tp2-op-48` (`--m 24,32,48`).
   [INFERENCE] The second 32-row tile streams the weights again, but they are 9.8 MB at TP1
   and fit the 64 MB MALL, so 48 rows should cost well under the 77 us of the torch
   composition.
2. Attention core of the QSA layers (the GDN conv and recurrent steps are the `gdn` cells).
   `tools/rdna/rdna3/qsa_decode_bench.py` graph-times, per layer and over three cold KV
   banks, `qsa_store_cache_rows`, `qsa_compress_groups_with_ratio`, `qsa_select_paged_tokens`
   (paged MQA scoring of the 4 x 128 indexer heads, top-k of 512 blocks, index expansion) and
   `qsa_sparse_paged_attention` (2048 selected tokens, head_dim 256, 24 q and 2 kv heads at
   TP1, 12 and 1 at TP2) for 1, 8 and 16 sequences of 3 rows each at 1k, 2k, 8k and 32k
   context (1k and 2k are the contexts of the TP4 anchors). Every line states the KV length
   (`kv_len`), the live compressed columns the scoring sees (`kv_len / 4`) and the page-table
   columns (`cap`); the sparse attention reads `min(2048, kv_len)` tokens per row and the
   rest of the index row is the -1 sentinel. The route is the product dispatch: run once
   with the default and once with `VLLM_RDNA3_HIP_OPS=qsa`. The cells use a page table of
   262144 tokens, which is what a captured decode graph has at the recipe's maximum model
   length (the scoring kernels leave early on invisible columns), and `G7-qsa-tp1-hip-tight`
   repeats the select with a page table as wide as the live context, which shows what the
   empty blocks cost. The projections around it (qkv, o_proj) are the `gemv-tp1` and
   `gemv-tp2` cold cells, which had not run yet. Cells `G7-qsa-tp1-default`, `-hip`,
   `G7-qsa-tp2-default`, `-hip`, `G7-qsa-tp1-hip-tight`, and `G7-qsa-tp1-hip-insertion` (the
   select with `VLLM_TOPK_DECODE_ALGO=insertion`, the algorithm of the G8 timing).
3. Router and routed experts at M = 1, 3, 8, 24 and 48. `tools/rdna/rdna3/moe_decode_bench.py`
   times the gate GEMM with `topk_softmax`, and the shipping `Rdna3WNA16Experts.apply`
   (align, GEMM 1, silu*mul, GEMM 2 with the fused top-k reduction) in the default and the Q32
   mode over eight cold banks of 192 random int4 experts (group 128), and, in `wmma` mode, the
   S1b kernel from 256 tokens up. It prints the distinct experts touched, microseconds per
   call and per distinct expert, GB/s and TFLOPS of the resident pairs.
   `--global-experts 512` routes over 512 experts of which 192 are resident (the rest is
   dropped through the expert map as on the hub), `--block-m` sweeps
   `VLLM_RDNA3_MOE_BLOCK_M` (`VLLM_RDNA3_MOE_WMMA_BLOCK_M` in `wmma` mode). Cells
   `G7-moe-decode`, `-hub`, `-block-m` for 1 to 48 tokens, `G7-moe-prefill`, `-hub`,
   `-block-m` for 256 to 2048 tokens (default against `wmma`, two cold banks).
   `G7-moe-decode-h1` is the H1 hub's residency: 130 slots per layer, and 264 routed experts
   so that the resident experts a batch touches match the T10 routing statistics (about
   14 / 79 / 110 at 3 / 24 / 48 rows, against the 16.5 / 81 / 109 measured); the microseconds
   per distinct expert it prints are the figure the hetero model carries as 3.7 us.
   `G7-moe-decode-k-h1` and `-k-h2` force exactly K distinct experts live per width, the
   hub tables of the design (H1 10 / 21 / 37 / 63 / 98 and H2 7 / 13 / 22 / 37 / 58 at
   3 / 6 / 12 / 24 / 48 rows; below 10 a row holds K ids and the rest is the -1 mask).
4. lm_head at 24 and 48 rows. No variant is written. `F.linear` reaches 434 to 526 GB/s at 24
   and 48 rows (TP1 and TP2) against 720 GB/s at 3 rows, which costs 0.3 to 1.0 ms per pass
   against a 700 GB/s roofline. The first experiment is PyTorch TunableOp (a rocBLAS and
   hipBLASLt solution search), which needs no code: `G7-lm_head-tp1-tune` and `-tp2-tune`
   tune on the shape and write `$RUN/tunableop_lm_head_tp*.csv`, the `-tuned` cells read the
   table back with tuning off (the deployable form is that file plus
   `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_TUNING=0` and
   `PYTORCH_TUNABLEOP_FILENAME`). A WMMA skinny kernel is the fallback if tuning finds
   nothing, and has not been started.

`qsa_decode_bench.py` and `moe_decode_bench.py` time a graph as the best of three batches of
20 replays after 0.3 s of warm-up replays (so the clocks have ramped) and print the spread
of the three batches. The first window timed one batch without warm-up, and the MoE
block-size sweep then showed identical configurations apart: block size 0 (1 up to 4
tokens, 4 above) against the same size given explicitly was 6, 19 and 12% slower for the
explicit one at 3, 8 and 24 tokens (equal at 48), so those rows are good to about 20%
only and the sweep is to be re-timed.

One window runs all of it (the labels are the cells named above plus the two cold dense
cells):

```bash
bash tools/rdna/rdna3/gpu_g0_g7.sh --tree ~/vllm-rdna3-flashnext \
    --venv ~/Apps/vllm/venv-7.14.0 --only G7 --keep-going --labels \
    G7-gemv-tp1,G7-gemv-tp2,G7-se-tp1-op-48,G7-se-tp2-op-48,G7-qsa-tp1-default,\
G7-qsa-tp1-hip,G7-qsa-tp2-default,G7-qsa-tp2-hip,G7-qsa-tp1-hip-tight,\
G7-qsa-tp1-hip-insertion,\
G7-moe-decode,G7-moe-decode-hub,G7-moe-decode-h1,G7-moe-decode-k-h1,\
G7-moe-decode-k-h2,G7-moe-decode-block-m,\
G7-moe-prefill,\
G7-moe-prefill-hub,G7-moe-prefill-block-m,G7-lm_head-tp1-tune,\
G7-lm_head-tp1-tuned,G7-lm_head-tp2-tune,G7-lm_head-tp2-tuned
```
