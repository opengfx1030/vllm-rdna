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
| 13 | Arch detection (`_query_gcn_arch_from_amdsmi`, `rocm.py:195`) | first amdsmi handle wins | collects all handles, raises on mixed archs so `_get_gcn_arch` falls back to the torch device property | torch property on a mixed box | none |
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

The load check lists the undefined dynamic symbols that look like project symbols; for
the 2-arch build it reports none after filtering libtorch/libstdc++ names (the unfiltered
list was 8 libtorch/c10/libstdc++ symbols). The authoritative result is the `dlopen`.
Not built here: the upstream `_C` and `_moe_C` targets for gfx1100 (`[INFERENCE]`: upstream
supports gfx11); only `_rocm_C` and the two sampler objects were compiled.

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
first handle, and importing `vllm.platforms` calls amdsmi, which can also wake idle GPUs.
Now it collects every handle's arch and raises `RuntimeError("amdsmi reports mixed GPU
archs: [...]")`, so `_get_gcn_arch` falls back to
`torch.cuda.get_device_properties("cuda").gcnArchName`, which honours
`HIP_VISIBLE_DEVICES`. On a single-arch box the result is the old one. A worker-init
warning for an arch that differs from the engine's was not written.

CPU rule that follows: every CPU run of vLLM on that box goes through
`tools/rdna/rdna3/cpu_pytest.sh`, which blocks `sys.modules["amdsmi"]`, unsets
`ROCR_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` and sets `HIP_VISIBLE_DEVICES=`. The
helper tools (`build_rocm_c.sh`, `rocm_c_load_check.py`) do the same.

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
5. `rocm.py`: identical result on a single-arch box.

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
family parser, the probe errors, the amdsmi mixed-arch cases; 123 tests),
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

## 12. GPU test plan (W7800 pair, HIP 0 and 5)

Rules: both W7800 idle, the box lease and `boot.lock` rules of `AGENTS.md`, never mix a
V620 into the same process, `ROCR_VISIBLE_DEVICES` unset, `HIP_VISIBLE_DEVICES=0` for
single-GPU steps and `0,5` for TP2. Stop at the first failing step.

| Step | What | Command and gate |
|---|---|---|
| G0 | Arch and family parse | `HIP_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName)"` prints `gfx1100`; with the mixed box visible to amdsmi the log shows the mixed-arch fallback once |
| G1 | Build | the command of section 4 with the 2-arch list; `ops-*.json` shows `loaded: true`, 177 ops |
| G2 | GDN parity, first the two kernels with inline `v_dot2c_f32_f16` | `pytest -q -x tests/kernels/quantization/test_gdn_prefill_prep_rdna2.py tests/kernels/quantization/test_gdn_prefill_kkt_rdna2.py`, then `test_gdn_prefill_solve_wy_rdna2.py`, `test_gdn_prefill_delta_h_rdna2.py`, `test_gdn_prefill_o_rdna2.py`, `test_gdn_decode_rdna2.py`, `tests/kernels/mamba/test_causal_conv1d_update_rdna2.py`, `test_causal_conv1d_null_block.py`. The solve_wy tolerance (1e-3) may be marginal where the Triton reference lowers to WMMA; report, do not loosen |
| G3 | Dense, QSA, HC, SE parity | `tests/kernels/quantization/test_gemv_f16_rdna2.py`, `tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py`, `test_qsa_rdna2_int4.py`, `test_qsa_metadata_null_block_rdna2.py`, `tests/models/qwen4_exp/test_qsa_metadata_rdna2.py`, `test_qsa_amd.py`, `test_hc_ops_rdna2.py`, `test_hc_prefill_rdna2.py`, `test_se_rdna2.py` (the new fixtures patch the family set, no env needed) |
| G4 | MoE | `tests/kernels/quantization/test_rdna3_moe_w4a16.py` (existing tests: default path unchanged; 10 new GPU tests: hub shapes, `-1` masking, Q32 bit identity over 50 repeats including M=2048, Q32 against default within fp16 noise, block-size sweep, split-expert partial sums, NaN poisoning, cast reference, malformed accumulators, graph capture and replay), `test_rdna3_compile_guards.py`, `tests/kernels/moe/test_rdna2_unquantized_moe.py` |
| G5 | Upstream skinny GEMMs and top-k | `tests/kernels/quantization/test_rocm_skinny_gemms.py` (wvSplitK n up to 5 on gfx11, LLMM1); `tests/kernels/test_top_k_per_row.py -k decode` runs for `auto`, `radix`, `insertion` |
| G6 | V1 ops that were widened | `tests/kernels/test_mrope_rdna2.py`, `tests/kernels/core/test_vit_pos_embed_rdna2.py`, `tests/v1/worker/test_input_batch_ops_rdna2.py`, `test_worker_ops_rdna2.py`, `test_gpu_block_table.py`, `tests/v1/sample/test_topk_topp_ops_rdna2.py` |
| G7 | Per-op latency against the model | graph-timed at W = 3, 24, 48: `tools/rdna/port_v031/decode_ops_bench.py` (HC and SE; its `I_SE` constant is the TP4 value 160, TP2 needs 320), `gdn_spec_bench.py` (constants are the 27B TP4 shape; Flash-Next TP2 is 8 k-heads, 24 v-heads, 128, 128, conv width 4, 36 GDN layers), `lm_head_bench.py --n 124160 --k 2560`, `gemv_crossover_bench.py`. Run each with `VLLM_RDNA3_HIP_OPS=` (empty) and with the family set, compare with `python tools/rdna/rdna3/perop_expectations.py --scenario central`. A family is worth enabling only if it is at least as fast as the default and the model is within the pessimistic-to-optimistic band |
| G8 | Top-k algo timing | time `top_k_per_row_decode` with `VLLM_TOPK_DECODE_ALGO=radix` and `insertion` at 1k, 4k, 8k live columns (a 20-line script, not written; the tests cover correctness only) and keep the faster default |
| G9 | `rdna_ar` | `HIP_VISIBLE_DEVICES=0,5 HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=2 torchrun --standalone --nproc_per_node=2 tools/rdna/rdna3/ar_probe.py --out ar_probe_w7800.json --timeout-test` with `VLLM_RDNA3_HIP_OPS=ar`. Gates: zero mismatches against the fp32 rank-order reference at 8, 64, 120, 256 KiB and 10 MiB; p50 at most 20 us per collective at 8 KiB fp16 in graph mode (the design doc's TP2 figure for RCCL is 36 us); no abort record or marker; the timeout phase aborts within 0.8 to 1.5 times 1500 ms |
| G10 | TP2 serve, safe baseline | `bash tools/rdna/serve_rdna.sh RECIPE=flashnext-w2-mtp2 PCI_BUSES="cb 07" MODEL=<checkpoint> VENV=<fat-build venv>` with `VLLM_RDNA3_HIP_OPS` empty, then `tools/rdna/port_v031/serve_validate.sh` against port 18097. Design gates (section 9.5): probes 3 of 3, PPL 4.75 to 4.85, MTP acceptance 1.84 and 2.01, c1 at most 28 ms per step, c8 at most 52 ms, memory within 44.98 GiB; 16k TTFT at most 4.5 s and 16k chunk at most 0.60 s need the WMMA prefill kernel (S1b) and are expected to fail with the scalar kernel |
| G11 | Family ladder | repeat G10 with `VLLM_RDNA3_HIP_OPS=hc`, then `hc,se`, `hc,se,qsa`, `hc,se,qsa,gdn`, then add `dense`, `moeu` (BF16 MTP only), `v1`, and last `ar` with `VLLM_RDNA_AR=1`. Promote a family to default only if quality gates hold and step time improves at c1 or c8; otherwise keep it opt-in |
| G12 | Determinism | `VLLM_RDNA3_MOE_DETERMINISTIC=1`: 50 repeated greedy requests give identical logprobs; measure the Q32 step-time cost at c1 and c8 |
| G13 | Cross-arch numerics | same truncated-layer prompts through V620 (gfx1030) and W7800 (gfx1100) separately, compare per-layer hidden states and the PPL probe; expect fp16-level agreement, the first layer that diverges names the op |
| G14 | Prefix-cache NaN | the `FULL_AND_PIECEWISE` plus prefix-caching repro of `docs/rdna2/flash-next-prefix-cache-corruption.md` |
| G15 | A/B items | `rms_norm` IR priority (`rocm.py` near `:1640`), `LLMM1` against `wvSplitK`, `gemv_f16_rdna2` against `wvSplitK` on non-padded layers (`maybe_pad_weight_avoid_cache_aliasing` pads rows when `K * elem % 2048 == 0`, which fails the contiguity gate of `gemv_f16_rdna2`), int8 shadow, W7800 TunableOp rows (`tools/rdna/prof/tune_missing_gemms.sh`, TunableOp ordinals of the W7800) |
| G16 | Tools | print the per-layer QSA route and `rotary_emb.mrope_interleaved` at startup; confirm gfx1100 rocBLAS and hipBLASLt Tensile libraries exist |

## 13. Optimization plan

Ordered by expected gain on the W2 baseline; each item names the evidence it needs.

1. WMMA W4A16 MoE prefill kernel (design slice S1b): the scalar kernel at an effective
   0.14 to 0.10 of hub throughput gives a W2 chunk of 1.0 to 1.4 s in the design model,
   against the 0.60 s gate. WMMA fp16 peak is 90.4 TFLOPS per W7800.
2. W8A16 or int8 dense weights: about 3.9 ms at c1 according to the design doc, from the
   4.64 GiB per rank dense read (7.2 ms floor at 691 GB/s).
3. lm_head: three passes cost 2.8 ms per step in the model (shard 919.7 us per pass); an
   argmax-only draft pass or W8A16 shrinks it.
4. HC family A/B (G7, G11): HC is 38 us per preamble centrally against 92 us in the
   design doc; the HIP chain is the assumption of kappa 1.0.
5. `rdna_ar` against RCCL, 1 to 1.5 ms per step at c1 if the 20 us probe gate holds.
6. `rms_norm` IR priority, W7800 TunableOp rows, top-k radix against insertion.
7. WMMA QSA prefill (backlog branch `origin/backlog/rdna3-attn-tiles`).

## 14. Risks and open decisions

Risks:

- Nothing has run on a GPU: every HIP kernel on gfx1100 beyond the existing native MoE
  kernel is unvalidated. `gdn_prefill_prep_rdna2.cu` (`:75-79`) and
  `gdn_prefill_kkt_rdna2.cu` (`:78-82`) use inline asm `v_dot2c_f32_f16` for every arch
  except gfx1200 and gfx1201; the fat build assembled them for gfx1100, whether gfx11 needs
  `v_dot2acc_f32_f16` for correct results is `[INFERENCE]` and G2 tests it first.
- Q32 performance and NaN semantics (section 7); `rdna_ar` coherence, wall-clock rate and
  cross-root P2P on gfx1100 (section 8); asymmetric opt-in under Ray or mixed hosts can
  leave a rank waiting in the first exchange (same class of hazard as the old gate).
- The sampler radix default on gfx11 is `[INFERENCE]` (G8).
- `moe_q_gemm_rdna3.cu` clang-format and pre-commit were applied after the first oracle
  run; the oracle was repeated on the formatted sources (section 4.2 table).
- `_C` and `_moe_C` were not built for gfx1100 on the CPU box.

Open decisions for the user:

1. Flip defaults of the HIP families after G7 and G11, per family, or keep them opt-in.
2. MTP checkpoint: derive the W4A16-MTP checkpoint (0.65 GiB per rank, assumed by the
   hetero design) or run the BF16 drafter (2.34 GiB per rank, Triton `fused_moe` or `moeu`).
3. Start S1b (WMMA W4A16 MoE prefill) now; the prefill gates of W2 depend on it.
4. Accept `VLLM_RDNA3_MOE_DETERMINISTIC=1` as the hub default if G12 shows an acceptable
   cost; the hetero contract requires it.
5. Pay the lint and type debt of the three V1 op test files so they can be widened.

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
