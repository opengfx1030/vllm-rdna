# NVIDIA capability checks on gfx1030: audit of `vllm/` (2026-11-05)

Base: `rdna_extras` at `1154318cf0`. Every `file:line` in this document is that SHA's line, under `vllm/` unless a path
starts with `tests/` or `tools/`. Fixes are on branch `fix/rocm-capability-family`: `a5bb516a83`, `e2f6560c4b`,
`25cf389991` (code plus red→green tests) and `38ab281ea1` (three test files skip on gfx1030). "gfx1030 gets" is the value
the check takes for capability (10, 3) (`to_int()` = 103). Statements that were not read in source, run, or taken from a log
are marked `[INFERENCE]`.

## 1. Why gfx1030 matches NVIDIA checks

`RocmPlatform.get_device_capability()` (`platforms/rocm.py:1096-1107`) parses the GCN arch string with `_capability_from_gcn_arch`
(`platforms/rocm.py:276`, which mirrors how HIP fills `hipDeviceProp_t.major/.minor`) and falls back to `torch.cuda.get_device_capability()`
only when the string cannot be parsed: gfx1030 is (10, 3). The helpers in `platforms/interface.py:531-588` then compare that number as
if it were an NVIDIA SM version:

| Check | Definition | gfx1030 (103) | Meant for |
|---|---|---|---|
| `has_device_capability(N)` | `to_int() >= N` | True for N <= 103, so 75, 80, 89, 90, 100 | NVIDIA SM >= N |
| `is_device_capability(N)` | `to_int() == N` | True only for 103 | one SM version |
| `is_device_capability_family(N)` | `to_int() // 10 == N // 10` | True for 100 (and 101..109) | SM100/SM103 (Blackwell DC) |
| `capability.major ==/>=/in` | major = 10 | `>= 10`, `== 10`, `in (10, 11)`, `in [9, 10]` True; `== 9`, `== 12` False | SM major |
| `(10, 3) in {(10, 0), (10, 3)}` | tuple set | True (`fused_moe/flashinfer_moe_ep.py:40`) | SM100/SM103 |
| `to_int() in {100, 103}` | int set | True (`layers/hpc/gated_mla.py:30`, `layers/hpc/hpc_ihc.py:45`) | SM100/SM103 |
| `"10.3"` string key | `as_version_str()` | True: `CUSTOM_ALL_REDUCE_MAX_SIZES`, `SYMM_MEM_ALL_REDUCE_MAX_SIZES` have a B300 entry (`distributed/device_communicators/all_reduce_utils.py:44,71`) | SM103 |

The fix pattern is `current_platform.is_cuda() and <NVIDIA check>` (or `if not is_cuda(): return False` first). The fork already
used it in places: `model_executor/warmup/kernel_warmup.py:219-224`, `layers/fused_moe/oracle/fp8.py:113-125`,
`layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py:41-62` (comment: "Platform-aware to avoid ROCm capability aliasing"),
`v1/sample/ops/topk_topp_sampler.py:164`.

The same parser gives the other AMD architectures these values. Measured on the box with the real `RocmPlatform` helpers and no GPU
(`~/w4a8_runs/audit/CapFamily/probe_arch_cap.py`, log `logs/arch_cap_probe.log`, tree `38ab281ea1`):

| Arch | Capability | has(90) | has(100) | is(90) | is(103) | family(90) | family(100) | family(120) |
|---|---|---|---|---|---|---|---|---|
| gfx1030 | (10, 3) | True | True | False | True | False | True | False |
| gfx90a | (9, 0) | True | False | True | False | True | False | False |
| gfx942 / gfx950 | (9, 4) / (9, 5) | True | False | False | False | True | False | False |
| gfx1100 / gfx1151 | (11, 0) / (11, 5) | True | True | False | False | False | False | False |
| gfx1201 | (12, 0) | True | True | False | False | False | False | True |

So `has(100)` is also True on every gfx11xx/gfx12xx part, `is(90)` and `family(90)` hit CDNA, and `family(120)` hits gfx12xx. `is_cuda()` is
architecture independent, so the guards added here close the sites they touch for those parts too. This audit read reachability only for
gfx1030; for the other parts the sites that are not `is_cuda()`-guarded are `[INFERENCE]`, not audited.

## 2. Result

428 sites in 220 files: 535 matching source lines at `1154318cf0`, grouped by statement into 448 groups, then merged to 428 sites
(helper definitions, and statements split across two groups, count once). The scan also hit 14 lines on the bare names `cap`/`cc` that are
not device capabilities (`cap = 1 << ...`, `cc.block_size`, ...); they were read and are not sites.

| Verdict | Sites |
|---|---|
| wrong on gfx1030, **fixed here** | 12 |
| wrong on gfx1030, masked by a later gate (not fixed) | 25 |
| wrong on gfx1030, reachable only by explicit selection (not fixed) | 16 |
| correct on gfx1030 (False is intended, or a min-capability threshold it meets) | 60 |
| unreachable on ROCm (behind `is_cuda()`, a CUDA-only class list, a platform dispatch, or an absent package) | 305 |
| definition of the helpers / not a capability check (comment, docstring, message) | 6 / 4 |

53 sites are wrong on gfx1030. Of those, 12 are fixed. By reading the code, the other 41 do not change behaviour on a stock ROCm run today (a later
gate is always False, or they run only with an explicit `--attention-backend` / quant choice that then fails late); a few rows carry an
`[INFERENCE]`. They are listed in section 6 with the reason, and the proposal for the largest group needs a decision from Main.

## 3. Production verdict: the capture-size ceiling does not change either recipe

`config/vllm.py:2486` (`_set_cudagraph_sizes`) set the default ceiling `1024 if family(100) else 512`, so gfx1030 got 1024 instead of 512.
The ceiling is read whenever `max_cudagraph_capture_size` is `None` (L2482), which it is in both recipes (they pass only the sizes list; log
line 13 below). There it only bounds `min(max_num_seqs * decode_query_len * 2, ceiling)` (L2490-2492) and, with speculative decoding, the
request-count grid (`min(max_reqs, ceiling)`, L2512). So it binds only above 256 sequences at query length 1, from 86 sequences at MTP 2
(3 query tokens), or above 512 sequences for the speculative grid.

Both recipes serve 8 sequences, so it cannot bind:

* Arithmetic: `8 * 3 * 2 = 48` (flashnext-mtp2, MTP 2) and `8 * 1 * 2 = 16` (27b-awq) are far below 512, and `request_counts(8)` is 8 either
  way, so 512 and 1024 give the same result.
* Explicit sizes then replace the generated list (`config/vllm.py:2582-2593`). `tools/rdna/serve_rdna.sh:164,335` passes
  `cudagraph_capture_sizes` (`CG_CAPTURE_SIZES` defaults to 1) from `rdna_mtp_args` (`tools/rdna/rdna_launcher_common.sh:127-141`): MTP 0 gives
  `[1,2,4,8]`, MTP 2 gives `[3,6,12,24]`; `--max-num-seqs "$SEQS"` with `SEQS` defaulting to 8 (`tools/rdna/serve_rdna.sh:124,355`).
* flashnext-mtp2 engine log `~/w4a8_runs/determinism/arms/P0r/serve.log` (box): `cudagraph_capture_sizes': [3, 6, 12, 24]`,
  `max_cudagraph_capture_size': 24` (the echoed user config on log line 13 has `None`; the resolved config on line 42 has 24),
  `Graph capturing finished in 2 secs, took 0.36 GiB`, attention backend RDNA_ATTN.
* 27b-awq engine log `~/w4a8_runs/m27b_m0/serve.log` (box, an older run, not re-run for this audit): `[1, 2, 4, 8]`,
  `max_cudagraph_capture_size': 8` (line 42; `None` on line 12), `Graph capturing finished in 4 secs, took 2.50 GiB`.
* `tests/platforms/test_rocm_capability_alias_config.py::test_recipe_capture_sizes_do_not_depend_on_the_platform_ceiling` runs
  `VllmConfig._set_cudagraph_sizes` on the gfx1030 fake with `max_cudagraph_capture_size=None`, each recipe's explicit sizes and 8 sequences, and
  asserts the resolved list is exactly the explicit one (max 24 / max 8). It passes on `1154318cf0` and on the fix.

So the fix changes no recipe behaviour and no TP4 measurement (boot time, graph memory, decode ITL, 16k TTFT) is needed; none was run.
The only user-visible change is for a gfx1030 run that sets no explicit sizes and whose `max_num_seqs * decode_query_len * 2` exceeds 512: its
ceiling drops from 1024 to 512.

## 4. Sites the recipes actually execute

Rows that, by code reading, run in a flashnext-mtp2 or 27b-awq process on ROCm (no live process was traced). None changes an output today; the
fixed ones are marked.

| Site | What gfx1030 gets | Effect |
|---|---|---|
| `config/vllm.py:2486` (fixed) | family(100) True -> 1024 | read, but `min(48 or 16, 512 or 1024)` is the same; explicit sizes then replace the list (section 3) |
| `model_executor/warmup/kernel_warmup.py:230` (fixed) | has(90) True | base entered `_warmup_ll_bf16_router_gemm` at every warmup; it returns early because the NVIDIA CuTe DSL is absent: a no-op both ways |
| `kernels/oink_ops.py:18-19` (fixed) | has(100) True, then `hasattr(torch.ops, "oink")` | `OINK_AVAILABLE` was True on gfx1030 with a GPU visible; `has_oink_op("rmsnorm")` stayed False (no `torch.ops.oink.rmsnorm`), so no op was dispatched. With no GPU visible it made `import vllm.kernels` raise `RuntimeError: No CUDA GPUs are available` |
| `platforms/rocm.py:845` | (10, 3) | backend selection by name/gfx; RDNA_ATTN (P0r log) |
| `platforms/rocm.py:1527` | has(80) True | bf16 allowed; both recipes pass `--dtype float16` (`tools/rdna/serve_rdna.sh:350`) |
| `config/vllm.py:948` | 103 >= min | compressed-tensors minimum passes, as intended |
| `layers/quantization/compressed_tensors/compressed_tensors.py:395-398` | 103 | WNA16 passes; the exact-match `_sm90`/`_sm100` schemes are False |
| `model_executor/kernels/linear/__init__.py:676,930,1257` | 103 passed to `is_supported` | no kernel in a ROCm list compares it |
| `layers/fused_moe/router/gate_linear.py:75-76` | family(100) True | every consumer combines it with `is_cuda()` by `and` (L84, L123, L135) |
| `layers/mamba/gdn/qwen_gdn_linear_attn.py:469-470, 1131-1134` | not reached / has(80) True | prefill backend returns before any capability test; the fused-decode gate also needs bf16, and the P0r log shows "Falling back to the Triton GDN spec decode path" |

## 5. Fixed sites

| Commit | Site (`vllm/`) | What it decided on gfx1030 | Red→green test |
|---|---|---|---|
| `a5bb516a83` | `config/vllm.py:2486` | `_set_cudagraph_sizes` default ceiling `1024 if family(100) else 512` | `test_default_capture_ceiling_is_blackwell_only[gfx1030]` |
| `a5bb516a83` | `utils/deep_gemm.py:51-52` | `should_auto_disable_deep_gemm` Blackwell exclusion table | `test_deepgemm_blackwell_exclusion_is_cuda_only[qwen3_5-gfx1030]`, `[qwen3_5_moe-gfx1030]` |
| `e2f6560c4b` | `v1/attention/ops/triton_unified_attention.py:972-975` | `tuned_large_head` (B200 BLOCK_M retune) | `test_unified_attention_large_head_keeps_decode_tiling_on_gfx1030` |
| `e2f6560c4b` | `model_executor/layers/mamba/ops/replayssm_config.py:19` | `_is_blackwell` ReplaySSM launch config | `test_replayssm_blackwell_launch_config_is_cuda_only[gfx1030]` |
| `e2f6560c4b` | `model_executor/layers/mamba/mamba_mixer2.py:573` | `is_blackwell` (same expression) | none (constructing MambaMixer2 needs a distributed group) |
| `25cf389991` | `kernels/oink_ops.py:18-19` | `OINK_AVAILABLE` (import-time) | `test_importing_vllm_kernels_needs_no_gpu_on_rocm` |
| `25cf389991` | `model_executor/warmup/kernel_warmup.py:230` | SM90+ low-latency BF16 router GEMM warmup | none (`kernel_warmup(worker)` needs a live worker) |
| `25cf389991` | `models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py:254-266` | `tma_aligned_scales=capability.major >= 10` | `test_fused_inv_rope_warmup_tma_scales_are_cuda_only[gfx1030]` |
| `25cf389991` | `models/flint/model.py:79` | `_flint_kernels` SM100 gate | `test_flint_kernels_are_sm100_cuda_only[gfx1030]` |
| `25cf389991` | `models/deepseek_v41/attention.py:138` | `_use_v41_mxfp8_kv_record` | `test_deepseek_v41_mxfp8_kv_record_is_sm100_cuda_only[gfx1030]` |
| `25cf389991` | `model_executor/layers/quantization/online/nvfp4.py:100` | SM100 `ValueError` | `test_nvfp4_per_token_online_moe_rejects_gfx1030` |
| `25cf389991` | `model_executor/layers/fused_moe/flashinfer_moe_ep.py:44-46` | `supports_current_device` table `{(10,0),(10,3)}` | `test_flashinfer_moe_ep_supports_current_device_is_cuda_only[gfx1030]` |

`tests/platforms/conftest.py` provides the `gfx1030` (ROCm, (10, 3)) and `sm100` (CUDA, (10, 0)) fake platforms; the tests need no GPU.
The three test files `tests/kernels/test_fused_inv_rope_fp8_quant.py`, `tests/distributed/test_custom_all_reduce.py` and
`tests/kernels/attention/test_deepgemm_attention.py` (`38ab281ea1`) now skip on gfx1030 with a reason (section 7).

## 6. Wrong on gfx1030 but not fixed

### Masked by a later gate

The check is wrong on gfx1030, but a later, correct gate is always False on ROCm (package/op absent, backend name, `is_cuda()` combined by `and` at the use, multi-node only, or an `except` fallback).

| Site | gfx1030 gets | Why it does not misfire today |
|---|---|---|
| `compilation/passes/fusion/rms_quant_fusion.py:788-789` | has(100)=True | has(100) is True on gfx1030, but `_FLASHINFER_NVFP4_RMS_QUANT_OP is not None` precedes it (L788) and is None without flashinfer (no ROCm build) |
| `config/vllm.py:948-951` | 103 >= quant_config.get_min_capability() -> accepted | executed: compressed-tensors minimum is below 103 so it passes, as intended. For NVIDIA-only methods with min 100 (e.g. NVFP4) gfx1030 also passes (validation gap); real failure comes at kernel selection (ROCm NVFP4 list = Emulation only) |
| `distributed/device_communicators/custom_all_reduce.py:80-81` | has(100)=True | reached only for multi-node groups (`not same_node and ...`); then `_has_local_multicast_support` is the real gate. Single-node TP4 recipes never call it |
| `distributed/device_communicators/custom_all_reduce.py:108-110` | family(100)=True | `_has_local_multicast_support(device)` follows and is the real gate |
| `distributed/device_communicators/flashinfer_all_reduce.py:63-67` | (10, 3) -> key (103, world_size, nodes) | FlashInfer allreduce workspace needs flashinfer + NVLink; not constructible on ROCm [INFERENCE for the flashinfer-less construction path] |
| `distributed/device_communicators/flashinfer_all_reduce.py:164` | has(100)=True | FlashInfer allreduce workspace needs flashinfer + NVLink; not constructible on ROCm [INFERENCE for the flashinfer-less construction path] |
| `model_executor/layers/attention/mla_attention.py:1916` | family(100)=True | MLA models only; after the check the backend name must be FLASHINFER / TRTLLM_RAGGED (L1924), which ROCm never selects (ROCM_AITER_FA / FLASH_ATTN) |
| `model_executor/layers/attention/sparse_mla_attention.py:108` | family(100)=True | DeepSeek-V3.2-style DSA models only; a model-dimension tuple check follows (L110) |
| `model_executor/layers/fused_moe/router/gate_linear.py:75-76` | is((9,0)) -> False; family(100) -> True (unused: combined with is_cuda() by `and`) | executed wherever a GateLinear is built (MoE router): `is_blackwell` is True on gfx1030 but every consumer ANDs `current_platform.is_cuda()` (L84, L123, L135), so the value is unused on ROCm |
| `model_executor/layers/quantization/compressed_tensors/transform/linear.py:52-53` | has(100)=True | has(100) True on gfx1030, but `is_qutlass_fp4_scheme(...)` (NVFP4 scheme + transforms) must hold first and qutlass is CUDA-only |
| `model_executor/layers/quantization/utils/mxfp8_utils.py:195` | has(100)=True | has(100) True on gfx1030 but `has_flashinfer()` False on ROCm |
| `model_executor/model_loader/weight_cache/warmup_cache.py:73-75` | has(90)=True | `not has_flashinfer()` returns first |
| `model_executor/warmup/flashinfer_autotune_cache.py:139` | has(90)=True | `has_flashinfer()` False on ROCm |
| `model_executor/warmup/kernel_warmup.py:285` | has(90)=True | `has_flashinfer()` False on ROCm |
| `models/inkling/amd/moe.py:69-72` | has(90)=True | has(90) is True on gfx1030; the next gate `ll_bf16 is_available()` is False without the NVIDIA CuTe DSL (CPU-only probe with a fake (10,3) platform) |
| `utils/flashinfer.py:182,185` | 103 | returns False earlier when flashinfer.mm_bf16 is missing (no flashinfer on ROCm); then passes 103 to flashinfer |
| `utils/flashinfer.py:1190-1191` | is(90) -> False | `has_flashinfer()` False; is(90) False anyway |
| `utils/flashinfer.py:1252` | has(100)=True | has(100) is True on gfx1030, but FLASHINFER must then be in RocmPlatform.get_supported_vit_attn_backends() and it is not (rocm.py:976-982) -> returns False |
| `v1/attention/backends/fa_utils.py:232-233` | major in (10, 11)=True | True on gfx1030; consumed by FLASH_ATTN-backend warmup keys / FA4 fallback checks only (fa_utils L178 is after the ROCm return; flash_attn.py:178 warmup is FLASH_ATTN only) |
| `v1/attention/backends/flash_attn.py:478-479` | (10,3) >= (9,0) -> True | gfx1030 passes the numeric gate; supports_sink() is False on ROCm (flash_attn_supports_sinks -> get_flash_attn_version() None) |
| `v1/attention/backends/mla/prefill/flash_attn.py:195-197` | family(100)=True; family(120)=False; is(90)=False | feed the FA3/FA4-specific branches, which require vllm_flash_attn_version 3/4 (None on ROCm) |
| `v1/attention/backends/mla/prefill/flash_attn.py:400` | [0] in (10,11) -> True | True on gfx1030 but `self.vllm_flash_attn_version == 4 and self._is_vllm_fa` precede and are False on ROCm |
| `v1/attention/ops/triton_reshape_and_cache_flash.py:26-28` | has(89) -> True; has(80) -> True | gfx1030 passes `has(89)` although it has no native FP8: only an assert at L418, hit with explicit --kv-cache-dtype fp8; the recipes use auto/int4 KV. Not fixed: needs a design decision for the ROCm FP8-KV predicate |
| `v1/pool/flash_maxsim/_common.py:41-45` | major>=9 -> 232_448 (fallback) | wrong for gfx1030 (64 KiB LDS), but it is the `except` fallback of the Triton driver query `get_device_properties(...)['max_shared_mem']` (L33-36); pooling/late-interaction maxsim only, in neither recipe. Not fixed: the ROCm fallback value is a design choice |

### Reachable only by explicit selection

The check is wrong on gfx1030, and the backend or option is not selected on ROCm unless the user forces it.

| Site | gfx1030 gets | Why it does not misfire today |
|---|---|---|
| `models/deepseek_v4/sparse_mla.py:96` | major in [9,10] -> True | DeepSeek-V4 only (neither recipe). `DeepseekV4SparseMLABackend` has no impl class (`get_impl_cls` raises; attention runs through `DeepseekV4Attention`), so the hook matters only if backend selection is called for it on ROCm [INFERENCE: selection path not traced] |
| `models/deepseek_v41/sparse_mla.py:151` | major in [9, 10]=True | explicit selection only; needs FlashMLA/DeepGEMM ops (absent on ROCm) -> fails late |
| `models/deepseek_v41/sparse_mla.py:306` | major==10=True | explicit selection only; needs FlashMLA/DeepGEMM ops (absent on ROCm) -> fails late |
| `utils/flashinfer.py:691-693,697` | family(100) -> True (after has_nvidia_artifactory()) | returns True for family(100) when the NVIDIA artifactory is reachable or cubins present; consumers are the FLASHINFER attention backend only (`can_use_trtllm_attention`); that backend needs flashinfer which has no ROCm build |
| `v1/attention/backends/flash_attn.py:156-157` | major = 10 -> `major in (10, 11)` True | True on gfx1030 -> FA4 hd256 warmup keys for FLASH_ATTN on ROCm. FLASH_ATTN is not the recipes' backend (RDNA_ATTN) |
| `v1/attention/backends/flashinfer.py:702-703` | cap>=(8,0)=True | True on gfx1030; FLASHINFER selectable explicitly, then needs flashinfer (no ROCm build) -> fails late |
| `v1/attention/backends/mla/cutlass_mla.py:66` | major==10=True | True on gfx1030; MLA backend reachable on ROCm only by explicit selection; each needs a CUDA-only op/package (cutlass_mla/flashmla extensions, flashinfer, tokenspeed) -> fails late |
| `v1/attention/backends/mla/flashinfer_mla.py:221` | major==10=True | True on gfx1030; MLA backend reachable on ROCm only by explicit selection; each needs a CUDA-only op/package (cutlass_mla/flashmla extensions, flashinfer, tokenspeed) -> fails late |
| `v1/attention/backends/mla/flashinfer_mla_sparse.py:106` | major==10=True | True on gfx1030; explicit selection only; needs flashinfer |
| `v1/attention/backends/mla/flashmla.py:74` | major in [9, 10]=True | True on gfx1030; MLA backend reachable on ROCm only by explicit selection; each needs a CUDA-only op/package (cutlass_mla/flashmla extensions, flashinfer, tokenspeed) -> fails late |
| `v1/attention/backends/mla/flashmla_sparse.py:204` | major in [9, 10]=True | True on gfx1030; explicit selection only; needs FlashMLA extension |
| `v1/attention/backends/mla/prefill/flashinfer.py:58` | major==10=True | True on gfx1030; ROCm uses RocmPlatform.get_mla_prefill_backend_cls (ROCM_AITER_FA / FLASH_ATTN) unless mla_prefill_backend is set explicitly |
| `v1/attention/backends/mla/prefill/tokenspeed_mla.py:42` | major==10=True | True on gfx1030; explicit selection only |
| `v1/attention/backends/mla/prefill/trtllm_ragged.py:61` | major==10=True | True on gfx1030; explicit selection only |
| `v1/attention/backends/mla/tokenspeed_mla.py:126` | major==10=True | True on gfx1030; MLA backend reachable on ROCm only by explicit selection; each needs a CUDA-only op/package (cutlass_mla/flashmla extensions, flashinfer, tokenspeed) -> fails late |

`fbgemm_fp8.py:55` (`has_device_capability(89)` as "native FP8") and `triton_reshape_and_cache_flash.py:26` (the same threshold for an FP8 KV cache) treat gfx1030 as having native FP8. Neither is on a recipe path (compressed-tensors W4A16, `auto`/int4 KV). A fix needs a ROCm FP8 predicate (for example `current_platform.supports_fp8()`), which is a design choice, so it is listed rather than patched.

### Proposal (needs a decision): a ROCm allowlist for attention backends

The 16 "explicit selection only" rows are `supports_compute_capability` hooks and backend-internal checks of NVIDIA-only attention backends
(CUTLASS/FlashInfer/FlashMLA/TokenSpeed/TRT-LLM MLA, FlashInfer, the FA4 warmup keys of FLASH_ATTN, `supports_trtllm_attention`). They are
True on (10, 3), but ROCm auto-selection never lists those backends, and each needs a CUDA-only package or op, so a user who forces one
fails late, with an import or op error instead of a capability error. Fixing each hook with `is_cuda()` would be 16 more edits with
no recipe effect. The alternative is one change in `RocmPlatform.get_attn_backend_cls` / `_get_invalid_reasons`: reject any backend that
is not in the ROCm set (RDNA_ATTN, ROCM_ATTN, ROCM_AITER_*, TRITON_ATTN, TRITON_MLA, FLASH_ATTN, ...). That changes error behaviour
for forced backends, so it was not done without Main.

## 7. Evidence

CPU-only (every GPU hidden: `HIP_VISIBLE_DEVICES=` empty, `ROCR_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` unset), exact-SHA trees on the box,
`venv-7.14.0_0.32.0` (torch 2.12.0+rocm7.14.0):

```bash
TREE=~/vllm-audit-capfamily-1154318cf0-redtests bash ~/w4a8_runs/audit/CapFamily/run_cpu_tests.sh \
  tests/platforms/test_rocm_capability_alias_{config,triton_tuning,nvidia_only}.py
```

| Tree | Result |
|---|---|
| red: `1154318cf0` sources + the four new test files from `38ab281ea1` | 11 failed, 20 passed in 18.19 s |
| green: `38ab281ea1` | 31 passed in 17.94 s |

Red failures, by file: `..._config.py` 3 of 14 (capture ceiling gfx1030, DeepGemm exclusion x2), `..._triton_tuning.py` 2 of 4 (unified attention,
ReplaySSM), `..._nvidia_only.py` 6 of 13 (fused_inv_rope, flint, dsv41, online NVFP4, flashinfer_moe_ep, `import vllm.kernels` with no GPU).
All sm100/sm103/sm120/sm90 cases and the recipe-invariance cases pass on both.

GPU 9 (V620, gfx1030), the three test files, via `run_tests.sh`:

Command (`hip9_demo.sh <tree> <log>` runs `TREE=<tree> GPU=9 bash ~/w4a8_runs/audit/moe/run_tests.sh` on
`tests/kernels/test_fused_inv_rope_fp8_quant.py`, `tests/kernels/attention/test_deepgemm_attention.py` and
`tests/distributed/test_custom_all_reduce.py::test_all_reduce_mhc_matches_unfused_path`; the other tests of the all-reduce file were not run).
`ray` is not installed in the audit venv, so a stand-in `ray/` package (`~/w4a8_runs/audit/CapFamily/stubs/ray`) is copied into each tree only so that
the all-reduce file can be collected.

| Tree | Result | Log |
|---|---|---|
| base `1154318cf0` | 8 failed, 77 passed, 13 skipped in 41.03 s | `logs/hip9_before_1154318cf0.log` |
| fix `38ab281ea1` | 77 passed, 21 skipped in 37.35 s | `logs/hip9_after_38ab281ea1.log` |

* Before: the 8 failures are `test_dsv41_fused_wo_a_matches_deepgemm[1-2, 1-4, 6-2, 6-4, 18-2, 18-4, 32-2, 32-4]`, all
  `ModuleNotFoundError: No module named 'flashinfer'` at `tests/kernels/test_fused_inv_rope_fp8_quant.py:48` (the SM100 reference path ran because
  gfx1030 passed the SM100 check). Skips: `[5] test_fused_inv_rope_fp8_quant.py:857 DeepGEMM not supported on this platform`,
  `[3]/[3]/[1] test_deepgemm_attention.py:116/224/351 DeepGEMM not available`,
  `[1] test_custom_all_reduce.py:181 Requires four GPUs with NVLink multicast`.
* After: those 8 skip with `[8] test_fused_inv_rope_fp8_quant.py:50 Requires a CUDA SM100/SM103 GPU`; the others are
  `[5] test_fused_inv_rope_fp8_quant.py:864 DeepGEMM not supported on this platform`,
  `[3]/[3]/[1] test_deepgemm_attention.py:122/228/353 SM90 and SM100 only`,
  `[1] test_custom_all_reduce.py:185 Requires a CUDA SM100/SM103 GPU`.
* `test_deepgemm_attention.py` already skipped on gfx1030 before ("DeepGEMM not available"); only the reason is now the right one.
* An earlier run on dev trees assembled before the exact-SHA trees existed gave the same counts (8 failed, 77 passed, 13 skipped / 77 passed, 21 skipped).

The MHC test (`test_all_reduce_mhc_matches_unfused_path`) skips either way on one visible GPU. With four or more visible GPUs the base guard
would pass on gfx1030 and the test would run its NVLink-multicast path [INFERENCE: needs a TP4 slot, not run].

### Corrections to earlier commit messages on this branch

* `a5bb516a83` says "4 of its 17 cases fail on 1154318cf0". Measured: 14 cases, 3 fail (already corrected in `e2f6560c4b`).
* `25cf389991` says the oink query "evaluated False there already". Measured by a CPU probe with a (10, 3) platform: `OINK_AVAILABLE` was True
  on gfx1030 with a GPU visible and only masked by the missing `torch.ops.oink.rmsnorm` (`has_oink_op("rmsnorm")` False). It also says oink is the
  only recipe-path site; `kernel_warmup.py:230` is executed too (section 4).
* `25cf389991` says "the 6 gfx1030 cases and the import case fail". Measured: 6 of the 13 cases in `..._nvidia_only.py` fail on `1154318cf0` (5 gfx1030
  cases and the import case).

## 8. Audit table

One row per site; consecutive lines of one statement are one site, and sites of one file or model with the same check and reason share a row
(the lines are listed). Verdicts: **wrong - fixed**; "wrong, masked" (a later gate is always False on ROCm); "wrong, explicit selection only";
"correct"; "unreachable on ROCm"; "definition"; "not a capability check". Impact columns are by code reading, except where a log is cited.
Repeated reasons are tagged R1-R4:

* **R1**: kernel class is only in the CUDA lists of kernels/linear/`__init__.py` (ROCm lists: RDNA2/3 W4A16, Triton/Conch/Exllama, Aiter*, Emulation*) and its is_supported() checks is_cuda() first where present
* **R2**: `_supports_current_device()` starts with `p.is_cuda()` (checked per class) or the class needs DeepGEMM/FlashInfer/CUTLASS ops that do not exist on ROCm; ROCm selects TritonExperts / RDNA2_HIP
* **R3**: vllm/models/<model>/`__init__.py` imports .amd when is_rocm() and .nvidia only otherwise; flashnext (qwen4_exp) resolves to qwen4_exp/amd/*
* **R4**: MLA backend reachable on ROCm only by explicit selection; each needs a CUDA-only op/package (cutlass_mla/flashmla extensions, flashinfer, tokenspeed) -> fails late

### Platform: capability source and helpers

| Site (`vllm/` omitted) | Check | gfx1030 gets | Intended | Verdict | Impact on flashnext-mtp2 / 27b-awq |
|---|---|---|---|---|---|
| `platforms/cuda.py:97,134,160,173,258,261,282,426,522,552,588,640-641,693-696,741,764,769,774-775,785,1010-1012` | 19 sites, first `if device_capability.major == 10:` | if reached: major==10=True; major==12=False; major==9=False; has(80)=True; ... | CudaPlatform queries | unreachable on ROCm | current_platform is RocmPlatform on ROCm; CudaPlatform methods are not called |
| `platforms/interface.py:93-134,517-521,531-551,554-574,577-588` | `class DeviceCapability (NamedTuple): major/minor, comparisons, to_int() = major*10+minor` (+4 more) | n/a (definition) | generic helpers; the aliasing is in these definitions | definition | all call sites go through these |
| `platforms/interface.py:458,462` | 2 sites, first `device_capability = cls.get_device_capability()` | if reached: major==10=True | overridden in RocmPlatform | unreachable on ROCm | RocmPlatform overrides the MLA prefill-backend selection (rocm.py:1050) |
| `platforms/rocm.py:278` | `HIP derives hipDeviceProp_t.major / .minor.` | n/a (comment) | comment | not a capability check | - |
| `platforms/rocm.py:845` | `device_capability = cls.get_device_capability()` | (10, 3) | ROCm backend selection by name/gfx; capability is only passed to validation | correct | executed; RDNA_ATTN selected (P0r serve.log attention_backend RDNA_ATTN) |
| `platforms/rocm.py:1050` | `device_capability = cls.get_device_capability()` | (10, 3) | MLA prefill backend selection (ROCM_AITER_FA / FLASH_ATTN) | correct | MLA models only |
| `platforms/rocm.py:1096-1107` | `RocmPlatform.get_device_capability: torch.cuda.get_device_capability -> DeviceCapability` | n/a (definition) | generic helpers; the aliasing is in these definitions | definition | source of the alias: gfx1030 -> (10, 3) |
| `platforms/rocm.py:1527-1528` | `if not cls.has_device_capability(80): capability = cls.get_device_capability()` | has(80)=True | bf16 needs capability >= 8.0 (gfx1030 passes) | correct | executed in the dtype check; both recipes serve float16 |
| `platforms/xpu.py:174,319-320` | 2 sites, first `device_capability=cls.get_device_capability(), # type: ignore[arg-type]` | - | XPU platform | unreachable on ROCm | XPU platform class not instantiated on ROCm |

### Config, engine, compilation

| Site (`vllm/` omitted) | Check | gfx1030 gets | Intended | Verdict | Impact on flashnext-mtp2 / 27b-awq |
|---|---|---|---|---|---|
| `compilation/passes/fusion/allreduce_rms_fusion.py:220-221,1084` | 2 sites, first `curr_device = current_platform.get_device_capability() device_capability = curr…` | if reached: 103 (to_int) used as table key / compared with 100; has(100)=True | FlashInfer allreduce+RMSNorm fusion (SM100 NVFP4 patterns / capability-keyed workspace) | unreachable on ROCm | AllReduceFusionPass is imported only under `if current_platform.is_cuda()` (pass_manager.py:50-51); ROCm gets RocmAiterAllReduceFusionPass instead |
| `compilation/passes/fusion/rms_quant_fusion.py:788-789` | `if _FLASHINFER_NVFP4_RMS_QUANT_OP is not None and ( current_platform.has_device_capability(100)` | has(100)=True | NVFP4 RMSNorm+quant fusion patterns need SM100 (has(100)) | wrong, masked (not fixed) | has(100) is True on gfx1030, but `_FLASHINFER_NVFP4_RMS_QUANT_OP is not None` precedes it (L788) and is None without flashinfer (no ROCm build) |
| `config/compilation.py:206,209` | `capability = current_platform.get_device_capability()` (+1 more) | not reached on ROCm | FlashInfer allreduce fusion size table keyed by to_int() | correct | `if not is_cuda(): return {}` precedes (L204) |
| `config/mamba.py:95-96` | `self.backend == MambaBackendEnum.TRITON and not current_platform.is_device_capability_family(100)` | not reached on ROCm | Mamba TRITON backend requires SM100 family | unreachable on ROCm | L88 `if not is_cuda(): raise ValueError` precedes it |
| `config/model.py:2449` | `device_capability = current_platform.get_device_capability()` | (10, 3) (message text only) | device capability only quoted in the dtype-fallback warning | correct | executed only when the dtype is unsupported; text only |
| `config/vllm.py:205-209` | `and current_platform.is_cuda() and has_flashinfer() and ( current_platform.is_device_capability_family(100) or cur…` | if reached: family(100)=True; is(90)=False | FlashInfer allreduce fusion default (SM100/SM90) | unreachable on ROCm | `is_cuda() and has_flashinfer()` precede |
| `config/vllm.py:846` | `current_platform.get_device_capability()` | not reached on ROCm | batch-invariant tuned-matmul table keyed by arch family (CUDA only) | unreachable on ROCm | `not current_platform.is_cuda()` returns False at L842 before the capability is read |
| `config/vllm.py:948-951` | `capability = get_device_capability().to_int(); quant min-capability gate` | 103 >= quant_config.get_min_capability() -> accepted | quant min-capability validation: capability < quant_config.get_min_capability() raises | wrong, masked (not fixed) | executed: compressed-tensors minimum is below 103 so it passes, as intended. For NVIDIA-only methods with min 100 (e.g. NVFP4) gfx1030 also passes (validation gap); real failure comes at kernel selection (ROCm NVFP4 list = Emulation only) |
| `config/vllm.py:1777-1780` | `self.model_config is not None and self.scheduler_config.enable_chunked_prefill and self.model_config.dtype == torc…` | == (7,5) -> False | Turing (7,5) fp32 chunked-prefill workaround | correct | False on gfx1030: correct |
| `config/vllm.py:2486` | `1024 if current_platform.is_device_capability_family(100) else 512` | family(100)=True | SM100/SM103 ceiling 1024, others 512 | **WRONG - fixed** (`a5bb516a83`) | executed (default ceiling), but cannot bind for either recipe: `min(max_num_seqs * decode_query_len * 2, ceiling)` = 48 (flashnext-mtp2) / 16 (27b-awq), far below 512, and the explicit capture sizes then replace the list (P0r serve.log: max_cudagraph_capture_size 24) |
| `engine/arg_utils.py:2974` | `is_gb10 = current_platform.is_device_capability((12, 1))` | is((12,1)) -> False | GB10 (12,1) exact | correct | (10,3) != (12,1): False, correct |

### Distributed communicators

| Site (`vllm/` omitted) | Check | gfx1030 gets | Intended | Verdict | Impact on flashnext-mtp2 / 27b-awq |
|---|---|---|---|---|---|
| `distributed/device_communicators/custom_all_reduce.py:80-81` | `device_index is not None and current_platform.has_device_capability(100, device_index)` | has(100)=True | MNNVL multi-node attempt needs Blackwell (has(100)) | wrong, masked (not fixed) | reached only for multi-node groups (`not same_node and ...`); then `_has_local_multicast_support` is the real gate. Single-node TP4 recipes never call it |
| `distributed/device_communicators/custom_all_reduce.py:108-110` | `world_size in _MNNVL_MULTIMEM_REDUCE_SCATTER_WORLD_SIZES and device.index is not None and current_platform.is_devi…` | family(100)=True | MNNVL multimem reduce-scatter on SM100 family | wrong, masked (not fixed) | `_has_local_multicast_support(device)` follows and is the real gate |
| `distributed/device_communicators/custom_all_reduce.py:239` | `device_capability = current_platform.get_device_capability()` | if reached: as_version_str() == '10.3' would hit the B300 entry, but is_cuda() is False | per-arch max-size tables keyed by 'major.minor' ("10.3" exists: B300) | unreachable on ROCm | `current_platform.is_cuda()` precedes the table lookup (L241) |
| `distributed/device_communicators/flashinfer_all_reduce.py:63-67` | `capability = get_device_capability(); FI_MNNVL_ALLREDUCE_MAX_SIZE_MB.get((capability.to_int(), world_size, nodes))` | (10, 3) -> key (103, world_size, nodes) | FlashInfer MNNVL/TRTLLM allreduce backends (SM100) | wrong, masked (not fixed) | FlashInfer allreduce workspace needs flashinfer + NVLink; not constructible on ROCm [INFERENCE for the flashinfer-less construction path] |
| `distributed/device_communicators/flashinfer_all_reduce.py:164` | `elif current_platform.has_device_capability(100):` | has(100)=True | FlashInfer MNNVL/TRTLLM allreduce backends (SM100) | wrong, masked (not fixed) | FlashInfer allreduce workspace needs flashinfer + NVLink; not constructible on ROCm [INFERENCE for the flashinfer-less construction path] |
| `distributed/device_communicators/symm_mem.py:58` | `capability = current_platform.get_device_capability()` | if reached: as_version_str() == '10.3' would hit the B300 entry, but is_cuda() is False | symm-mem size tables keyed by 'major.minor' ("10.3" exists: B300) | unreachable on ROCm | `__init__` returns at `if not current_platform.is_cuda()` (L46) before reading capability |

### model_executor/kernels (linear kernel classes)

| Site (`vllm/` omitted) | Check | gfx1030 gets | Intended | Verdict | Impact on flashnext-mtp2 / 27b-awq |
|---|---|---|---|---|---|
| `model_executor/kernels/attention/dsa/sparse_mqa_logits.py:46-48` | `importlib.util.find_spec("vllm._deepselect_C") is not None and current_platform.is_cuda() and current_platform.is_…` | if reached: family(100)=True | DSA sparse MQA logits (SM100) | unreachable on ROCm | `is_cuda() and family(100)` |
| `model_executor/kernels/linear/__init__.py:676,930,1257-1258` | `_cc = current_platform.get_device_capability()` (+2 more) | compute_capability = 103 | pass the device capability (103) to kernel.is_supported / get_min_capability | correct | executed for every linear layer; no kernel in a ROCm list compares it (AiterInt8 is in no list); the ROCm W4A16 kernels' minimum is below 103 |
| `model_executor/kernels/linear/base.py:133-134` | `if compute_capability and compute_capability < 90: return False, "Requires compute capability >= 9.0"` | n/a (docstring) | docstring example | not a capability check | - |
| `model_executor/kernels/linear/cute_dsl/gemm_rs_ar.py:1124` | `elif not current_platform.is_device_capability_family(100):` | if reached: family(100)=True | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/cute_dsl/ll_bf16.py:84,89` | 2 sites, first `if current_platform.is_device_capability_family(100):` | if reached: family(100)=True; is(90)=False | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/mixed_precision/cutlass.py:36` | `if not current_platform.is_device_capability(90):` | if reached: is(90)=False | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/mixed_precision/machete.py:33` | `if not current_platform.is_device_capability(90):` | if reached: is(90)=False | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/mxfp4/b12x.py:59` | `if not current_platform.is_device_capability_family(120):` | if reached: family(120)=False | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/mxfp4/flashinfer.py:28` | `if current_platform.has_device_capability(100) and has_flashinfer_cutedsl():` | if reached: has(100)=True | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/mxfp4/humming.py:37` | `if not current_platform.has_device_capability(75):` | if reached: has(75)=True | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/mxfp6/humming.py:34` | `if not current_platform.has_device_capability(75):` | if reached: has(75)=True | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/mxfp8/b12x.py:56` | `if not current_platform.is_device_capability_family(120):` | if reached: family(120)=False | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/mxfp8/deep_gemm.py:34` | `if not current_platform.is_device_capability_family(100):` | if reached: family(100)=True | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/mxfp8/flashinfer.py:51,64,136-137,209-210` | 4 sites, first `current_platform.is_cuda() and current_platform.has_device_capability(100)` | if reached: has(100)=True; family(120)=False; family(100)=True | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/mxfp8/humming.py:34` | `if not current_platform.has_device_capability(75):` | if reached: has(75)=True | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/mxfp8/pytorch.py:32` | `if current_platform.is_cuda() and current_platform.has_device_capability(90):` | if reached: has(90)=True | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/nvfp4/b12x.py:62` | `if not current_platform.is_device_capability_family(120):` | if reached: family(120)=False | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/nvfp4/flashinfer.py:45,48,51,121,203-204,405` | 6 sites, first `capability = current_platform.get_device_capability()` | if reached: cc>=120=False; cc<130=True; family(100)=True; has(100)=True; ... | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/nvfp4/humming.py:37` | `if not current_platform.has_device_capability(75):` | if reached: has(75)=True | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/nvfp4/pytorch.py:30` | `if not current_platform.is_device_capability_family(100):` | if reached: family(100)=True | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/scaled_mm/aiter.py:46` | `if compute_capability is not None and compute_capability < 90:` | if reached: cc<90 -> False (passes the capability gate) | AITER INT8 needs gfx9 (cc >= 90) | unreachable on ROCm | AiterInt8ScaledMMLinearKernel is in no _POSSIBLE_* list (ROCm INT8 list is empty); also `import aiter` / is_linear_enabled() fail on gfx1030. Intended gfx9 gate would pass 103 [wrong in principle] |
| `model_executor/kernels/linear/scaled_mm/b12x.py:60,210` | 2 sites, first `if not current_platform.is_device_capability_family(120):` | if reached: family(120)=False | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/scaled_mm/flashinfer.py:50` | `if compute_capability is not None and compute_capability < 100:` | if reached: cc<100=False | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/scaled_mm/humming.py:40,131` | 2 sites, first `if not current_platform.has_device_capability(75):` | if reached: has(75)=True | NVIDIA-only linear kernel (CUTLASS/Machete/B12x/FlashInfer/Humming/DeepGEMM/CuTe-DSL) | unreachable on ROCm | R1: CUDA-only linear kernel; in no ROCm list |
| `model_executor/kernels/linear/scaled_mm/pytorch.py:303-305` | `compute_capability == 90 if compute_capability is not None else current_platform.is_device_capability(90)` | if reached: is(90)=False; cc==90=False | torch._scaled_mm block scaling needs SM90 | unreachable on ROCm | inside `if current_platform.is_cuda()` (L301); on ROCm the branch is skipped |
| `model_executor/kernels/mhc/tilelang_kernels.py:76` | `return current_platform.is_cuda() and current_platform.has_device_capability(100)` | if reached: has(100)=True | tilelang MHC kernels (SM100) | unreachable on ROCm | `is_cuda() and has(100)` |

### model_executor/layers/fused_moe

| Site (`vllm/` omitted) | Check | gfx1030 gets | Intended | Verdict | Impact on flashnext-mtp2 / 27b-awq |
|---|---|---|---|---|---|
| `model_executor/layers/fused_moe/b12x.py:501-502` | `current_platform.is_cuda() and current_platform.is_device_capability_family(120)` | if reached: family(120)=False | B12x MoE (SM120) | unreachable on ROCm | `is_cuda() and family(120)` |
| `model_executor/layers/fused_moe/deep_gemm_mega_moe.py:198-199` | `if not current_platform.is_device_capability_family( 100, device_id=device.index or 0` | if reached: family(100)=True | DeepGEMM mega-MoE (SM100) | unreachable on ROCm | DeepGEMM unavailable on ROCm: is_deep_gemm_supported() = platform.support_deep_gemm() (False on ROCm) and package present |
| `model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py:215,355-357` | 2 sites, first `if current_platform.is_cuda() and current_platform.has_device_capability(80):` | if reached: has(80)=True; family(100)=True; family(120)=False | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/cutlass_moe.py:718-721,1036,1038-1040,1357-1358` | 4 sites, first `return p.is_cuda() and ( p.is_device_capability_family(100) or p.is_device_capa…` | if reached: family(100)=True; family(110)=False; family(120)=False; is(90)=False | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/deep_gemm_moe.py:187` | `return current_platform.is_device_capability_family(100)` | if reached: family(100)=True | DeepGEMM MXFP8 MoE scheme (SM100 family) | unreachable on ROCm | class is excluded earlier: `_supports_current_device()` needs is_deep_gemm_supported() (platform.support_deep_gemm(), False on ROCm) |
| `model_executor/layers/fused_moe/experts/deep_gemm_moe.py:362-367,477-479,644` | 3 sites, first `use_psum_layout = ( not self.mxfp8 and ( current_platform.is_device_capability_…` | if reached: family(90)=False; family(100)=True; family(120)=False | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:168-169` | `p.is_cuda() and p.is_device_capability_family(120)` | if reached: family(120)=False | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/flashinfer_cutedsl_batched_moe.py:68-69` | `p.is_cuda() and p.is_device_capability_family(100)` | if reached: family(100)=True | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py:91-92` | `p.is_cuda() and p.is_device_capability_family(100)` | if reached: family(100)=True | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:131-136,160-161,169-170,178-179` | 4 sites, first `p.is_cuda() and ( p.is_device_capability(90) or p.is_device_capability_family(1…` | if reached: family(100)=True; family(120)=False; is(90)=False; has(90)=True; ... | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/fused_batched_moe.py:828` | `p.is_cuda() and p.has_device_capability((8, 9))` | if reached: has((8,9))=True | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/fused_humming_moe.py:406-408` | `has_humming() and platform.is_cuda() and platform.has_device_capability((7, 5))` | if reached: has((7,5))=True | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py:48,53,67-68` | `cap = p.get_device_capability()` (+2 more) | is_cuda() False -> on_gfx9() or on_gfx1x() -> False on gfx1030 | OAI Triton MoE device gate: capability range on CUDA, gfx-string helpers on ROCm | correct | already ROCm-aware by design (existing comment L43-45 on capability aliasing): capability read only under `p.is_cuda()`; ROCm uses on_gfx9() or on_gfx1x(), which excludes gfx10xx |
| `model_executor/layers/fused_moe/experts/marlin_moe.py:594` | `return p.is_cuda() and p.has_device_capability((7, 5))` | if reached: has((7,5))=True | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/triton_cutlass_moe.py:27` | `self.is_sm100 = current_platform.has_device_capability(100)` | if reached: has(100)=True | SM100 small-M path | unreachable on ROCm | built only after CutlassExpertsFp8 support (is_cuda() and family(100/110/120)) [reading of cutlass_moe.py:718] |
| `model_executor/layers/fused_moe/experts/triton_moe.py:144-145` | `current_platform.is_cuda() and current_platform.has_device_capability((7, 5))` | CUDA arm not evaluated on ROCm | TritonExperts: is_cuda() and cap >= (7,5) [or ROCm] | correct | `is_cuda() and has((7,5))` is the CUDA arm of the support test; ROCm takes the other arm |
| `model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py:99-100` | `p.is_cuda() and p.is_device_capability_family(100)` | if reached: family(100)=True | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py:169-170` | `p.is_cuda() and p.is_device_capability_family(100)` | if reached: family(100)=True | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/trtllm_lora_moe.py:149-150` | `p.is_cuda() and p.is_device_capability_family(100)` | if reached: family(100)=True | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py:123` | `return p.is_cuda() and p.is_device_capability_family(100) and has_flashinfer()` | if reached: family(100)=True | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/trtllm_mxint4_moe.py:52-53` | `p.is_cuda() and p.is_device_capability_family(100)` | if reached: family(100)=True | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py:208-209` | `p.is_cuda() and p.is_device_capability_family(100)` | if reached: family(100)=True | NVIDIA-only MoE expert backend (CUTLASS/FlashInfer/TRT-LLM/DeepGEMM/Marlin/Humming/GPT-OSS triton_kernels) | unreachable on ROCm | R2: CUDA-only MoE expert backend; ROCm uses Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/flashinfer_moe_ep.py:44-46` | `capability = current_platform.get_device_capability() return capability is not None and ( (capability.major, capab…` | (10, 3) in {(10,0),(10,3)} -> True | FlashInfer MoE-EP needs SM100/SM103 ((10,0),(10,3)) | **WRONG - fixed** (`25cf389991`) | FlashInfer MoE-EP only; recipes use Triton/RDNA2_HIP experts |
| `model_executor/layers/fused_moe/hpc_moe.py:65` | `return p.is_cuda() and p.is_device_capability(90) and has_hpc()` | if reached: is(90)=False | HPC MoE (SM90) | unreachable on ROCm | `is_cuda() and is(90) and has_hpc()` |
| `model_executor/layers/fused_moe/moe_fused_mul_sum.py:85-86,89` | 2 sites, first `capability = current_platform.get_device_capability(device_index) if capability…` | if reached: cap>=(7,5)=True | tile config for >= (7,5) NVIDIA | unreachable on ROCm | `and current_platform.is_cuda()` precedes (L84) |
| `model_executor/layers/fused_moe/oracle/fp8.py:114-115,124-125` | 2 sites, first `current_platform.is_cuda() and current_platform.is_device_capability_family(100)` | if reached: family(100)=True; is(90)=False | FP8 MoE oracle: SM100 / SM90 backend choice | unreachable on ROCm | `is_cuda() and ...` (existing fork guard) |
| `model_executor/layers/fused_moe/oracle/mxfp4.py:438` | `if current_platform.is_cuda() and current_platform.is_device_capability_family(120):` | if reached: family(120)=False | MXFP4 MoE oracle SM120 | unreachable on ROCm | `is_cuda() and family(120)` |
| `model_executor/layers/fused_moe/oracle/mxfp4.py:532-534` | `activation_key == kMxfp8Dynamic and _user_moe_activation_override() is None and current_platform.is_device_capabil…` | if reached: family(90) -> False | MXFP4/MXFP8 MoE oracle on Hopper | unreachable on ROCm | MXFP4 oracle (NVIDIA MXFP4 / MXFP8 activation); `family(90)` is False on gfx1030 anyway |
| `model_executor/layers/fused_moe/oracle/unquantized.py:87` | `if current_platform.is_device_capability_family(90):` | if reached: family(90)=False | Hopper reorder of FlashInfer backends | unreachable on ROCm | inside `elif current_platform.is_cuda():` branch (L77); ROCm branch above it |
| `model_executor/layers/fused_moe/router/gate_linear.py:75-76` | `is_hopper = current_platform.is_device_capability((9, 0)) is_blackwell = current_platform.is_device_capability_fam…` | is((9,0)) -> False; family(100) -> True (unused: combined with is_cuda() by `and`) | router GEMM: Hopper (9,0) / Blackwell family(100) specialized kernels (CUDA) | wrong, masked (not fixed) | executed wherever a GateLinear is built (MoE router): `is_blackwell` is True on gfx1030 but every consumer ANDs `current_platform.is_cuda()` (L84, L123, L135), so the value is unused on ROCm |
| `model_executor/layers/fused_moe/utils.py:707-708` | `return ( current_platform.is_device_capability(90)` | is(90) -> False | swap-AB tile choice on Hopper | correct | is(90) False on gfx1030: correct |
| `model_executor/layers/fused_moe/utils.py:726,732` | 2 sites, first `unsupported there. So this gates on a blanket ''has_device_capability(100)''` | if reached: has(100)=True | TD (gather4) path needs SM100+ | unreachable on ROCm | `if current_platform.is_cuda():` precedes (L731); ROCm returns False |

### model_executor/layers (non-MoE) and models/config

| Site (`vllm/` omitted) | Check | gfx1030 gets | Intended | Verdict | Impact on flashnext-mtp2 / 27b-awq |
|---|---|---|---|---|---|
| `model_executor/layers/attention/mla_attention.py:1916` | `if not current_platform.is_device_capability_family(100):` | family(100)=True | FP8 MLA prefill query quant needs SM100 (GB200) | wrong, masked (not fixed) | MLA models only; after the check the backend name must be FLASHINFER / TRTLLM_RAGGED (L1924), which ROCm never selects (ROCM_AITER_FA / FLASH_ATTN) |
| `model_executor/layers/attention/sparse_mla_attention.py:108` | `if not current_platform.is_device_capability_family(100):` | family(100)=True | masked MHA for sparse MLA needs SM100 family | wrong, masked (not fixed) | DeepSeek-V3.2-style DSA models only; a model-dimension tuple check follows (L110) |
| `model_executor/layers/fusion/fused_act_quant.py:180` | `if current_platform.is_cuda() and current_platform.has_device_capability(90):` | if reached: has(90)=True | fused act-quant Hopper path | unreachable on ROCm | `is_cuda() and has(90)` |
| `model_executor/layers/hpc/gated_mla.py:56-57` | `capability = current_platform.get_device_capability() if capability is None or capability.to_int() not in _HPC_GAT…` | if reached: to_int()=103 IS in {100,103}; is_cuda() is False first | HPC gated-MLA GEMM (sm100/sm103 set {100,103}: contains 103!) | unreachable on ROCm | `has_hpc()` then `is_cuda()` (L52) precede the set test |
| `model_executor/layers/hpc/hpc_ihc.py:66-67` | `capability = current_platform.get_device_capability() if capability is None or capability.to_int() not in _SUPPORT…` | if reached: to_int()=103 IS in {100,103}; is_cuda() is False first | HPC iHC kernels (sm100/sm103 set {100,103}: contains 103!) | unreachable on ROCm | `has_hpc()` then `is_cuda()` (L62) precede the set test |
| `model_executor/layers/indexer_topk.py:156-157,160-162` | 2 sites, first `self._has_deep_select = self._is_cuda and ( current_platform.is_device_capabili…` | if reached: family(100)=True; family(120)=False; has(90)=True | DSA indexer top-k on SM100/Hopper | unreachable on ROCm | `self._is_cuda and (...)` |
| `model_executor/layers/lightning_attn.py:411-412,415` | 2 sites, first `capability = torch.cuda.get_device_capability() if capability[0] < 8:` | if reached: [0]<8=False | lightning-attention requires SM80+ | unreachable on ROCm | inside `if current_platform.is_cuda():` (L410) |
| `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:475,477-478,484-485,533` | 4 sites, first `if current_platform.is_device_capability(90):` | if reached: is(90)=False; family(100)=True; family(120)=False | GDN prefill backend choice (flashinfer on SM90/100/120) | unreachable on ROCm | `if not current_platform.is_cuda(): return backend, 'triton'` at L469-470 precedes; L533 log line only when active backend == flashinfer |
| `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1131-1134` | `or vllm_config.model_config.dtype != torch.bfloat16 or conv_state_dtype != torch.bfloat16 or recurrent_state_dtype…` | has(80) -> True | fused GDN decode needs bf16 + cc >= 80 | correct | executed by both recipes (GDN): has(80) True is correct; the recipes serve fp16 so the dtype term already rejects the fused kernel (log: falls back to the Triton GDN spec-decode path) |
| `model_executor/layers/mamba/mamba_mixer2.py:573` | `self.is_blackwell = current_platform.is_device_capability_family(100)` | family(100)=True | Blackwell Mamba2 path | **WRONG - fixed** (`e2f6560c4b`) | MambaMixer2 is built only for Mamba2 models (nemotron_h, falcon_h1, ...); recipes use GDN (the module is imported by qwen_gdn_linear_attn but this line is in `__init__`) |
| `model_executor/layers/mamba/ops/replayssm_config.py:19` | `return current_platform.is_device_capability_family(100)` | family(100)=True | Blackwell ReplaySSM launch config | **WRONG - fixed** (`e2f6560c4b`) | ReplaySSM Mamba2 kernel only; not GDN |
| `model_executor/layers/mamba/ops/ssd_chunk_scan.py:30` | `if not (current_platform.is_cuda() and current_platform.is_device_capability(90)):` | if reached: is(90)=False | Hopper chunk-scan config | unreachable on ROCm | `not (is_cuda() and is(90))` |
| `model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:395,398` | `capability_tuple = current_platform.get_device_capability()` (+1 more) | 103 >= min -> supported; ==90/==100 exact -> False | min-capability / exact-capability scheme gate | correct | executed by both recipes (compressed-tensors): WNA16 min is below 103 so supported; the exact-match gates (_is_fp8_w*_sm90/_sm100 use match_exact) are False for 103: correct |
| `model_executor/layers/quantization/compressed_tensors/transform/linear.py:52-53` | `quant_scheme, input_tfms ) and current_platform.has_device_capability(100):` | has(100)=True | qutlass NVFP4 transform method needs SM100 | wrong, masked (not fixed) | has(100) True on gfx1030, but `is_qutlass_fp4_scheme(...)` (NVFP4 scheme + transforms) must hold first and qutlass is CUDA-only |
| `model_executor/layers/quantization/fbgemm_fp8.py:55` | `self.use_marlin = not current_platform.has_device_capability(89)` | has(89) -> True | marlin fallback when no native FP8 (< 89) | correct | explicit fbgemm_fp8 quant only; gfx1030 says 'native FP8' (103 >= 89) but has none: benign for the recipes |
| `model_executor/layers/quantization/moe_wna16.py:85-87,173-175` | `capability_tuple = current_platform.get_device_capability() device_capability = ( -1 if capability_tuple is None e…` (+1 more) | 103 >= 75 -> True | AWQ MoE-WNA16 minimum capability (75) | correct | executed only for AWQ/GPTQ MoE checkpoints under moe_wna16; gfx1030 passes, as intended |
| `model_executor/layers/quantization/online/nvfp4.py:100` | `if not current_platform.is_device_capability_family(100):` | family(100)=True | online NVFP4 needs SM100/SM103 | **WRONG - fixed** (`25cf389991`) | online NVFP4 only; recipes are compressed-tensors W4A16 |
| `model_executor/layers/quantization/quark/quark.py:351,354` | `capability_tuple = current_platform.get_device_capability()` (+1 more) | 103 >= min -> True | Quark scheme minimum capability | correct | Quark quant only; threshold pass |
| `model_executor/layers/quantization/torchao.py:105,118-120,126-127` | `require GPU compute capability >= 8.9 (Ada Lovelace / Hopper) on NVIDIA,` (+2 more) | n/a (message) | capability quoted in an error after supports_fp8() | correct | text only |
| `model_executor/layers/quantization/utils/flashinfer_mxint4_moe.py:27-30` | `envs.VLLM_USE_FLASHINFER_MOE_INT4 and has_flashinfer_trtllm_fused_moe() and current_platform.is_cuda() and current…` | if reached: family(100)=True | FlashInfer MXINT4 MoE | unreachable on ROCm | `is_cuda()` precedes |
| `model_executor/layers/quantization/utils/humming/priority.py:21-22` | `cc = current_platform.get_device_capability() compute_capability = cc.to_int() if cc is not None else None` | - | Humming (CUDA-only) capability / schema | unreachable on ROCm | Humming kernels are CUDA-only; every is_supported() checks is_cuda() first |
| `model_executor/layers/quantization/utils/humming/schema.py:113,159,269-271` | 3 sites, first `if not current_platform.has_device_capability(80):` | if reached: has(80)=True | Humming (CUDA-only) capability / schema | unreachable on ROCm | Humming kernels are CUDA-only; every is_supported() checks is_cuda() first |
| `model_executor/layers/quantization/utils/marlin_utils.py:61-63,66` | 2 sites, first `capability_tuple = current_platform.get_device_capability() device_capability =…` | - | Marlin capability >= 75 | unreachable on ROCm | inside `if not current_platform.is_rocm():` (L59) |
| `model_executor/layers/quantization/utils/marlin_utils.py:126-128` | `capability_tuple = current_platform.get_device_capability() device_capability = ( -1 if capability_tuple is None e…` | 103 passed; numeric gate skipped on ROCm | Marlin support query: capability passed down | correct | 103 is passed to query_marlin_supported_quant_types, which skips the numeric gate on ROCm (L59); Marlin ops are not built for ROCm |
| `model_executor/layers/quantization/utils/marlin_utils.py:641-642,702-703` | 2 sites, first `device_capability = torch.cuda.get_device_capability(device) if device_capabili…` | if reached: [0]<9=False | Marlin bf16 atomic-add advice (< SM90) | unreachable on ROCm | Marlin only; log message |
| `model_executor/layers/quantization/utils/marlin_utils.py:726-728` | `if not current_platform.is_device_capability( 89 ) and not current_platform.is_device_capability_family(120):` | is(89) -> False, family(120) -> False -> raises | Marlin W4A8-FP8 only on SM89/SM12x | correct | raises on gfx1030 (is(89) False, family(120) False): correct, and Marlin is CUDA-only |
| `model_executor/layers/quantization/utils/marlin_utils_fp4.py:35` | `return current_platform.is_cuda() and current_platform.has_device_capability(75)` | if reached: has(75)=True | Marlin FP4 support | unreachable on ROCm | `is_cuda() and has(75)` |
| `model_executor/layers/quantization/utils/marlin_utils_fp8.py:31` | `return current_platform.has_device_capability(75)` | has(75) -> True | Marlin FP8 weight-only needs >= 75 | correct | explicit fp8 Marlin only; threshold pass |
| `model_executor/layers/quantization/utils/mxfp4_utils.py:82-83,129,140-142` | 3 sites, first `current_platform.is_cuda() and current_platform.is_device_capability(90)` | if reached: is(90)=False; family(100)=True; family(120)=False | MXFP4 triton_kernels layout tweaks (SM90/100/120) | unreachable on ROCm | inside `current_platform.is_cuda()` / `is_cuda() and is(90)` |
| `model_executor/layers/quantization/utils/mxfp8_utils.py:195` | `if current_platform.has_device_capability(100) and has_flashinfer():` | has(100)=True | FlashInfer MXFP8 quantize (SM100) | wrong, masked (not fixed) | has(100) True on gfx1030 but `has_flashinfer()` False on ROCm |
| `model_executor/layers/quantization/utils/nvfp4_utils.py:56-57` | `capability_tuple = current_platform.get_device_capability() capability = -1 if capability_tuple is None else capab…` | - | NVFP4 capability helpers | unreachable on ROCm | `is_cuda()`-guarded callers (nvfp4 is not on the ROCm kernel list) |
| `model_executor/layers/quantization/utils/w8a8_utils.py:15-16,25-26,35-36` | 3 sites, first `capability_tuple = current_platform.get_device_capability() capability = -1 if…` | - | CUTLASS FP8 support | unreachable on ROCm | `if not current_platform.is_cuda(): return False` first |
| `model_executor/layers/sparse_mqa_indexer.py:100-101` | `current_platform.is_cuda() and current_platform.is_device_capability_family(100)` | if reached: family(100)=True | sparse MQA indexer SM100 | unreachable on ROCm | `is_cuda() and family(100)` |
| `model_executor/layers/utils.py:118` | `current_platform.is_cuda() and current_platform.is_device_capability_family(100)` | if reached: family(100)=True | SM100 router GEMM | unreachable on ROCm | `is_cuda() and family(100)` |
| `model_executor/models/config.py:96-98` | `cache_config.cache_dtype == "auto" and current_platform.is_cuda() and current_platform.is_device_capability_family…` | if reached: family(100)=True | SM100 KV dtype default | unreachable on ROCm | `is_cuda()` precedes |
| `model_executor/models/config.py:126-128` | `cache_config.cache_dtype != "auto" or not current_platform.is_cuda() or not current_platform.is_device_capability_…` | if reached: family(100)=True | SM100 KV dtype default | unreachable on ROCm | `not is_cuda()` short-circuits |
| `model_executor/models/config.py:340-341` | `use_per_layer_fa = current_platform.is_device_capability_family( 90` | family(90) -> False | Gemma4 per-layer FA3/FA4 on SM90 (FP8 KV) | correct | family(90) False on gfx1030: correct |
| `model_executor/models/deepseek_v2.py:976-980` | `and self.weight.shape[1] == 7168 and current_platform.is_cuda() and ( current_platform.is_device_capability(90) or…` | if reached: family(100)=True; is(90)=False | DeepSeek-V2 SM90/SM100 router GEMM | unreachable on ROCm | `is_cuda()` precedes |

### Warmup, determinism, loader, kernels, lora, third_party, vllm_flash_attn

| Site (`vllm/` omitted) | Check | gfx1030 gets | Intended | Verdict | Impact on flashnext-mtp2 / 27b-awq |
|---|---|---|---|---|---|
| `kernels/oink_ops.py:18-19` | `OINK_AVAILABLE = current_platform.has_device_capability(100) and hasattr( torch.ops, "oink"` | has(100)=True | Oink kernels exist on SM100+ (NVIDIA) | **WRONG - fixed** (`25cf389991`) | import-time constant, evaluated in recipe processes: True on base, but torch.ops.oink.rmsnorm is absent so has_oink_op() stayed False and no op dispatched. On base, `import vllm.kernels` also raised with no visible GPU |
| `lora/ops/triton_ops/lora_expand_op.py:237` | `if current_platform.is_cuda() and current_platform.is_device_capability(90):` | if reached: is(90)=False | Hopper+ LoRA kernel tuning | unreachable on ROCm | `is_cuda()` first in the same expression |
| `lora/ops/triton_ops/utils.py:321-322,330` | 2 sites, first `current_platform.is_cuda() and current_platform.has_device_capability(90)` | if reached: has(90)=True | Hopper+ LoRA kernel tuning | unreachable on ROCm | `is_cuda()` first in the same expression |
| `model_executor/determinism/batch_invariant.py:1318-1319` | `if current_platform.is_rocm() or current_platform.is_device_capability_family( 80` | is_rocm() -> True | batch-invariant GEMM setup: ROCm or Ampere | correct | `is_rocm() or ...`: ROCm short-circuits first (only with VLLM_BATCH_INVARIANT=1) |
| `model_executor/determinism/batch_invariant_configs.py:1230-1236` | `if capability.major == 12: return "sm120" if capability.major == 10: return "blackwell" if capability.major == 9:…` | not reached on ROCm | tuned matmul arch family (sm120/blackwell/hopper/ada) | unreachable on ROCm | caller passes capability only `if is_cuda() else None` (L1250): None on ROCm |
| `model_executor/determinism/batch_invariant_configs.py:1250` | `current_platform.get_device_capability() if current_platform.is_cuda() else None` | None | capability fetched only on CUDA | correct | `is_cuda()` ternary: None on ROCm |
| `model_executor/model_loader/weight_cache/warmup_cache.py:73-75` | `vllm_config.kernel_config.enable_flashinfer_autotune is False or not has_flashinfer() or not current_platform.has_…` | has(90)=True | FlashInfer autotune warmup cache needs SM90+ | wrong, masked (not fixed) | `not has_flashinfer()` returns first |
| `model_executor/warmup/b12x_warmup.py:54` | `if not current_platform.is_device_capability_family(120):` | if reached: family(120) -> False -> return | B12x warmup (SM120) | unreachable on ROCm | B12x kernels are SM120 only; `not family(120)` -> return, False on gfx1030 too |
| `model_executor/warmup/flashinfer_autotune_cache.py:139` | `if has_flashinfer() and current_platform.has_device_capability(90):` | has(90)=True | FlashInfer autotune cache | wrong, masked (not fixed) | `has_flashinfer()` False on ROCm |
| `model_executor/warmup/flashinfer_sparse_mla_warmup.py:134` | `if not has_flashinfer() or not current_platform.is_device_capability_family(120):` | if reached: family(120)=False | FlashInfer sparse-MLA warmup (SM120) | unreachable on ROCm | `not has_flashinfer() or not family(120)` -> return |
| `model_executor/warmup/kernel_warmup.py:221-223` | `enable_jit_warmup and current_platform.is_cuda() and current_platform.is_device_capability_family(100)` | if reached: family(100)=True | DeepGEMM JIT warmup on SM100 | unreachable on ROCm | `is_cuda() and family(100)` (existing fork comment L219) |
| `model_executor/warmup/kernel_warmup.py:230` | `if current_platform.has_device_capability(90):` | has(90)=True | SM90+ warmup of the low-latency BF16 router GEMM (NVIDIA) | **WRONG - fixed** (`25cf389991`) | executed at every warmup of both recipes: base entered the NVIDIA-only warmup function (early return, so no-op) |
| `model_executor/warmup/kernel_warmup.py:285` | `elif has_flashinfer() and current_platform.has_device_capability(90):` | has(90)=True | FlashInfer autotune warmup (SM90+) | wrong, masked (not fixed) | `has_flashinfer()` False on ROCm |
| `model_executor/warmup/minimax_m3_msa_warmup.py:33` | `current_platform.is_cuda() and current_platform.is_device_capability_family(100)` | if reached: family(100)=True | MiniMax-M3 MSA warmup SM100 | unreachable on ROCm | `is_cuda() and family(100)` |
| `third_party/flash_linear_attention/ops/fused_recurrent.py:444-446` | `K == V == 128 and current_platform.is_cuda() and current_platform.is_device_capability_family(100, device_id=dev.i…` | if reached: family(100)=True | FLA fused recurrent SM100 path | unreachable on ROCm | `is_cuda() and family(100)` |
| `third_party/flash_linear_attention/ops/utils.py:152-153` | `"NVIDIA H" in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability()[0] >= 9` | - | FLA Hopper detection | unreachable on ROCm | `is_nvidia and (...)` (L151): FLA's own platform probe says 'amd' on ROCm |
| `vllm_flash_attn/flash_attn_interface.py:57-58,67,77-80,460-461` | 4 sites, first `if not current_platform.has_device_capability(80): return False, "FA2 is only s…` | if reached: has(80)=True; family(90)=False; family(100)=True; family(110)=False | vllm_flash_attn FA2/FA3/FA4 capability gates | unreachable on ROCm | each `_is_fa*_supported()` returns first on `FA*_AVAILABLE` False; the exact-SHA tree has no `_vllm_fa*` extension (vllm/vllm_flash_attn holds only the two .py files) |

### models/* (amd, common, flint, ... non-nvidia)

| Site (`vllm/` omitted) | Check | gfx1030 gets | Intended | Verdict | Impact on flashnext-mtp2 / 27b-awq |
|---|---|---|---|---|---|
| `models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py:254-266` | `capability = get_device_capability(); ... tma_aligned_scales=capability.major >= 10` | (10, 3): major >= 10 -> True | TMA-aligned scale layout on SM100+ | **WRONG - fixed** (`25cf389991`) | DeepSeek-V4 only |
| `models/deepseek_v4/sparse_mla.py:96` | `return capability.major in [9, 10]` | major in [9,10] -> True | backend hook: sparse MLA on SM90/SM100 (major in [9,10]) | wrong, explicit selection only (not fixed) | DeepSeek-V4 only (neither recipe). `DeepseekV4SparseMLABackend` has no impl class (`get_impl_cls` raises; attention runs through `DeepseekV4Attention`), so the hook matters only if backend selection is called for it on ROCm [INFERENCE: selection path not traced] |
| `models/deepseek_v4/sparse_mla.py:335-336` | `capability = current_platform.get_device_capability() if capability is not None and capability.major == 12:` | major==12 -> False | SM12x decode buffer layout (major == 12) | correct | False on gfx1030: correct |
| `models/deepseek_v41/attention.py:138` | `return current_platform.is_device_capability_family(100)` | family(100)=True | MXFP8 KV record needs SM100/SM103 | **WRONG - fixed** (`25cf389991`) | DeepSeek-V4.1 only |
| `models/deepseek_v41/common/engram.py:94-95` | `if current_platform.is_cuda() and current_platform.is_device_capability_family( 100, device.index or 0` | if reached: family(100)=True | Engram SM100 path | unreachable on ROCm | `is_cuda() and family(100)` |
| `models/deepseek_v41/sparse_mla.py:116` | `return [64 if current_platform.is_device_capability_family(90) else 128]` | family(90) -> False | KV block size 64 on SM90, else 128 | correct | False on gfx1030 -> 128: correct |
| `models/deepseek_v41/sparse_mla.py:151` | `return capability.major in [9, 10]` | major in [9, 10]=True | backend hook: sparse MLA on SM90/SM100 (major in [9,10] / ==10), block-size choice for SM90 | wrong, explicit selection only (not fixed) | explicit selection only; needs FlashMLA/DeepGEMM ops (absent on ROCm) -> fails late |
| `models/deepseek_v41/sparse_mla.py:306` | `return capability.major == 10` | major==10=True | backend hook: sparse MLA on SM90/SM100 (major in [9,10] / ==10), block-size choice for SM90 | wrong, explicit selection only (not fixed) | explicit selection only; needs FlashMLA/DeepGEMM ops (absent on ROCm) -> fails late |
| `models/flint/model.py:79` | `if not current_platform.is_device_capability_family(100):` | family(100)=True | Flint kernels exist on SM100/SM103 only | **WRONG - fixed** (`25cf389991`) | Flint model only |
| `models/glm5next/common/kda.py:146,148-150` | 2 sites, first `capability = current_platform.get_device_capability()` | if reached: major in (9, 10, 12)=True | KDA kernels on SM90/100/120 | unreachable on ROCm | `is_cuda() and ...` (L148) |
| `models/inkling/amd/moe.py:69-72` | `and flat.is_contiguous() and weight.is_contiguous() and flat.shape[1] % 8 == 0 and current_platform.has_device_cap…` | has(90)=True | Inkling HIP MoE fast-path gate has(90) | wrong, masked (not fixed) | has(90) is True on gfx1030; the next gate `ll_bf16 is_available()` is False without the NVIDIA CuTe DSL (CPU-only probe with a fake (10,3) platform) |
| `models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py:531` | `if current_platform.is_cuda() and current_platform.has_device_capability(80)` | if reached: has(80)=True | KDA chunk-intra on SM80+ | unreachable on ROCm | `is_cuda() and has(80)` |
| `models/minimax_m3/common/indexer.py:544` | `is_sm100 = is_cuda and current_platform.is_device_capability_family(100)` | if reached: family(100)=True | MiniMax-M3 MSA indexer on SM100 | unreachable on ROCm | `is_cuda and family(100)` |
| `models/minimax_m3/common/ops/index_topk.py:712` | `current_platform.is_cuda() and current_platform.is_device_capability((12, 0))` | if reached: is((12,0))=False | MiniMax-M3 index top-k SM120 | unreachable on ROCm | `is_cuda() and is((12,0))` |
| `models/minimax_m3/common/sparse_attention.py:546-547` | `current_platform.is_cuda() and current_platform.is_device_capability_family(100)` | if reached: family(100)=True | MiniMax-M3 sparse attention SM100 | unreachable on ROCm | `is_cuda() and family(100)` |
| `models/muse_spark/common/ops/fused_rmsnorm.py:521` | `if D == 8192 and T >= 256 and current_platform.is_device_capability(100):` | is(100) -> False | SM100 exact fused RMSNorm tile (is(100)) | correct | exact match: 103 != 100 -> False, correct |
| `models/qwen4_exp/amd/ops/qsa.py:896-900` | `current_platform.is_cuda() and blocks.shape[0] <= 32 and logits.stride(0) % 4 == 0 and current_platform.has_device…` | not reached on ROCm | QSA top-k on SM90 not SM120 | unreachable on ROCm | `current_platform.is_cuda() and ...` is the first term (L896); flashnext's qsa path (qwen4_exp/amd) is ROCm so the term is False |

### models/*/nvidia/** (NVIDIA model implementations)

| Site (`vllm/` omitted) | Check | gfx1030 gets | Intended | Verdict | Impact on flashnext-mtp2 / 27b-awq |
|---|---|---|---|---|---|
| `models/beam/nvidia/**`: `allreduce.py:45`; `attention_ops.py:48-49`; `projection.py:24-26` | 3 sites in 3 files (NVIDIA model code) | if reached: family(100)=True | NVIDIA model implementation (model has no amd/ dir) | unreachable on ROCm | model has no ROCm implementation (unsupported on ROCm; the nvidia/ code would run if forced) [INFERENCE] |
| `models/deepseek_v32/nvidia/**`: `glm52_low_latency_gemm.py:213-215`; `ops/fused_q_cutedsl.py:36-37` | 2 sites in 2 files (NVIDIA model code) | if reached: is((10,7))=False; is((10,3))=True; has(100)=True | NVIDIA model implementation | unreachable on ROCm | R3: NVIDIA model code; the model `__init__` imports `.amd` on ROCm |
| `models/deepseek_v4/nvidia/**`: `flashinfer_sparse.py:141,156,165`; `model.py:546,557,566`; `ops/o_proj.py:28-31` | 7 sites in 3 files (NVIDIA model code) | if reached: major in [10, 12]=True; major==10=True; major==12=False; major<=9=False; ... | NVIDIA model implementation | unreachable on ROCm | R3: NVIDIA model code; the model `__init__` imports `.amd` on ROCm |
| `models/deepseek_v41/nvidia/**`: `flash_mla_mega_attn.py:98`; `flashinfer_sparse.py:138,153,162`; `flashmla.py:72`; `model.py:222,236,248,277-278`; `ops/engram_peer.py:134-135`; `ops/mega_mhc.py:20-21`; `ops/mhc.py:49-50`; `ops/o_proj.py:35-36`; `specialized_decode/__init__.py:83`; `specialized_decode/attention/mxfp8.py:66,71,139-140`; `specialized_decode/boundary/publishing_gemm.py:582` | 18 sites in 11 files (NVIDIA model code) | if reached: family(100)=True; major in [10, 12]=True; major==10=True; major==12=False; ... | NVIDIA model implementation | unreachable on ROCm | R3: NVIDIA model code; the model `__init__` imports `.amd` on ROCm |
| `models/hy_v4/nvidia/**`: `attention.py:598` | 1 site in 1 file (NVIDIA model code) | - | NVIDIA model implementation | unreachable on ROCm | R3: NVIDIA model code; the model `__init__` imports `.amd` on ROCm |
| `models/inkling/nvidia/**`: `moe.py:68-71`; `ops/fa4_rel_attention.py:32-33,81-82`; `ops/lamport.py:521` | 4 sites in 3 files (NVIDIA model code) | if reached: has(90)=True; major in (10, 11)=True; major==9=False | NVIDIA model implementation | unreachable on ROCm | R3: NVIDIA model code; the model `__init__` imports `.amd` on ROCm |
| `models/kimi_k3/nvidia/**`: `kda.py:207-210,225,228-229,253-254,371,373-374,390,393`; `latent_moe_runner.py:77-78`; `low_latency_gemm.py:75-77,663,667,675-677`; `ops/attn_res.py:196-199`; `ops/cute_dsl/kda_skinny_gemm.py:134`; `ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py:1066-1069`; `ops/latent_moe_tail.py:120`; `ops/third_party/kda/chunk_intra.py:573`; `recoverssm/kernels.py:52`; `specialized_decode/cutedsl/common/host.py:29`; `specialized_decode/cutedsl/mla/dcp.py:307`; `specialized_decode/runtime/__init__.py:80` | 22 sites in 12 files (NVIDIA model code) | if reached: family(100)=True; family(120)=False; is(90)=False; major in (9, 10, 12)=True; ... | NVIDIA model implementation | unreachable on ROCm | R3: NVIDIA model code; the model `__init__` imports `.amd` on ROCm |
| `models/minimax_m3/nvidia/**`: `msa_cutlass_sparse_decode.py:244-245,267-269` | 2 sites in 1 file (NVIDIA model code) | if reached: is((10,3))=True; family(100)=True | NVIDIA model implementation | unreachable on ROCm | R3: NVIDIA model code; the model `__init__` imports `.amd` on ROCm |
| `models/muse_spark/nvidia/**`: `model.py:514-516`; `muse_moe_runner.py:403-404,577-578,749`; `ops/gemm_rs.py:67-69`; `ops/mnnvl_fusions.py:36-37` | 6 sites in 4 files (NVIDIA model code) | if reached: has(100)=True; family(100)=True; family(90)=False; is(103)=True | NVIDIA model implementation | unreachable on ROCm | R3: NVIDIA model code; the model `__init__` imports `.amd` on ROCm |
| `models/qwen4_exp/nvidia/**`: `hyperconnection.py:113-115`; `low_latency_gemm.py:357,388`; `ops/cute_dsl/hc_down_silu.py:212`; `ops/qsa.py:22,64,70,826`; `ops/qsa_indexer.py:488-491`; `qsa.py:303,504-505` | 11 sites in 6 files (NVIDIA model code) | if reached: has(90)=True; has(89)=True; family(120)=False | NVIDIA model implementation | unreachable on ROCm | R3: NVIDIA model code; the model `__init__` imports `.amd` on ROCm |
| `models/step5/nvidia/**`: `flashinfer_sparse.py:46`; `mega_moe.py:341`; `model.py:1026,1263-1266`; `ops/sparse_indexer.py:595`; `sparse_attention.py:292,898-900` | 7 sites in 5 files (NVIDIA model code) | if reached: is(103)=True; family(100)=True | NVIDIA model implementation (model has no amd/ dir) | unreachable on ROCm | model has no ROCm implementation (unsupported on ROCm; the nvidia/ code would run if forced) [INFERENCE] |

### v1/attention (backends and ops)

| Site (`vllm/` omitted) | Check | gfx1030 gets | Intended | Verdict | Impact on flashnext-mtp2 / 27b-awq |
|---|---|---|---|---|---|
| `v1/attention/backends/b12x.py:219` | `return (capability.major, capability.minor) in ((12, 0), (12, 1))` | (10,3) not in ((12,0),(12,1)) -> False | B12X backend: (12,0)/(12,1) | correct | False on gfx1030: correct |
| `v1/attention/backends/composite.py:413-414` | `return (device_major is None or capability.major == device_major) and all( backend.supports_compute_capability(cap…` | n/a (generic) | composite backend: sub-backend device-major match | correct | generic comparison of the real capability, no hardcoded arch |
| `v1/attention/backends/fa_utils.py:94,99,102,120,141-142,195-196` | 6 sites, first `device_capability = current_platform.get_device_capability()` | not reached on ROCm | FA version choice (FA3 on SM90, FA4 on SM100, Blackwell head-size limits) | unreachable on ROCm | get_flash_attn_version() returns None for ROCm at L85-87 before reading the capability |
| `v1/attention/backends/fa_utils.py:232-233` | `capability = current_platform.get_device_capability() return capability is not None and capability.major in (10, 1…` | major in (10, 11)=True | uses_fa4_hd256_kernel: major in (10, 11) | wrong, masked (not fixed) | True on gfx1030; consumed by FLASH_ATTN-backend warmup keys / FA4 fallback checks only (fa_utils L178 is after the ROCm return; flash_attn.py:178 warmup is FLASH_ATTN only) |
| `v1/attention/backends/fa_utils.py:306` | `is_sm90 = current_platform.is_device_capability_family(90)` | family(90)=False | FP8-KV support for FA on SM90 / FA4 SM100 | correct | fa_version is None on ROCm so the result is False: correct |
| `v1/attention/backends/fa_utils.py:311` | `fa_version == 4 and current_platform.is_device_capability_family(100)` | family(100)=True | FP8-KV support for FA on SM90 / FA4 SM100 | correct | fa_version is None on ROCm so the result is False: correct |
| `v1/attention/backends/fa_utils.py:335-336` | `3 ) and current_platform.is_device_capability_family(90)` | if reached: family(90)=False | flash_attn_supports_mla on Hopper | unreachable on ROCm | inside `if current_platform.is_cuda()` (L328) |
| `v1/attention/backends/flash_attn.py:156-157` | `capability = current_platform.get_device_capability() major = capability.major if capability is not None else None` | major = 10 -> `major in (10, 11)` True | FA4 warmup keys: major in (10, 11) with hd256 pages | wrong, explicit selection only (not fixed) | True on gfx1030 -> FA4 hd256 warmup keys for FLASH_ATTN on ROCm. FLASH_ATTN is not the recipes' backend (RDNA_ATTN) |
| `v1/attention/backends/flash_attn.py:322-323` | `if ( current_platform.is_device_capability_family(90)` | family(90) -> False | SM90 FP8-KV page size (FA4 head 512) | correct | family(90) False on gfx1030: correct |
| `v1/attention/backends/flash_attn.py:463` | `return capability >= DeviceCapability(8, 0)` | (10,3) >= (8,0) -> True | FLASH_ATTN supports capability >= (8,0) | correct | by design: ROCm uses upstream flash-attn through this backend |
| `v1/attention/backends/flash_attn.py:478-479` | `if has_sink and device_capability < DeviceCapability(9, 0): return "sink not supported on compute capability < 9.0"` | (10,3) >= (9,0) -> True | attention sinks need capability >= (9,0) | wrong, masked (not fixed) | gfx1030 passes the numeric gate; supports_sink() is False on ROCm (flash_attn_supports_sinks -> get_flash_attn_version() None) |
| `v1/attention/backends/flash_attn.py:481-484` | `use_mm_prefix and kv_cache_dtype is not None and is_quantized_kv_cache(kv_cache_dtype) and device_capability == De…` | == (9,0) -> False | SM90 FP8 KV with mm_prefix | correct | == (9,0) False on gfx1030: correct |
| `v1/attention/backends/flash_attn.py:1163-1164` | `self.vllm_flash_attn_version == 4 and current_platform.is_device_capability_family(90)` | family(90) -> False | SM90 FA4 FP8 KV dequant | correct | family(90) False: correct |
| `v1/attention/backends/flashinfer.py:115,120,597-599,619,659-660,680-682,716-718,721,730-731,997-998,1001-1002,1099-1100,1146-1148,1163-1165,1196-1198,1213-1214,1303-1307,1397-1399,1432-1434,2080-2082,2137-2139` | 21 sites, first `return any(current_platform.is_device_capability_family(f) for f in (80, 90, 12…` | if reached: has(90)=True; family(100)=True; major in (8, 9, 12)=False; family(120)=False; ... | FlashInfer backend internals (SM90/SM100/SM12x kernel choice) | unreachable on ROCm | FLASHINFER backend only (needs flashinfer; ROCm has no flashinfer build) |
| `v1/attention/backends/flashinfer.py:702-703` | `return capability >= DeviceCapability(8, 0) and capability <= DeviceCapability( 12, 1` | cap>=(8,0)=True | FlashInfer backend hook: capability in [(8,0), (12,x)] | wrong, explicit selection only (not fixed) | True on gfx1030; FLASHINFER selectable explicitly, then needs flashinfer (no ROCm build) -> fails late |
| `v1/attention/backends/hpc_attn.py:312` | `return capability == DeviceCapability(9, 0)` | == (9,0) -> False | HPC attention only on (9,0) | correct | (10,3) != (9,0): False, correct (HPC also needs the hpc package) |
| `v1/attention/backends/mla/cutlass_mla.py:66` | `return capability.major == 10` | major==10=True | backend hook CUTLASS MLA: major == 10 | wrong, explicit selection only (not fixed) | R4: MLA backend that ROCm never auto-selects; forced selection fails late |
| `v1/attention/backends/mla/flashattn_mla.py:70` | `return capability.major == 9` | major==9 -> False | FA3-MLA: major == 9 | correct | False on gfx1030: correct |
| `v1/attention/backends/mla/flashattn_mla_sparse.py:83` | `return capability.major == 9` | major==9 -> False | FA3 sparse MLA: major == 9 | correct | False on gfx1030: correct |
| `v1/attention/backends/mla/flashinfer_mla.py:221` | `return capability.major == 10` | major==10=True | backend hook FlashInfer MLA: major == 10 | wrong, explicit selection only (not fixed) | R4: MLA backend that ROCm never auto-selects; forced selection fails late |
| `v1/attention/backends/mla/flashinfer_mla_sparse.py:106` | `return capability.major == 10` | major==10=True | backend hook: FlashInfer sparse MLA major == 10 | wrong, explicit selection only (not fixed) | True on gfx1030; explicit selection only; needs flashinfer |
| `v1/attention/backends/mla/flashinfer_mla_sparse.py:193` | `return capability.major == 12` | major==12 -> False | FlashInfer sparse MLA SM120 hook: major == 12 | correct | False on gfx1030: correct |
| `v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py:190` | `return capability.major == 9` | major==9 -> False | FlashInfer sparse MLA SM90 hook: major == 9 | correct | False on gfx1030: correct |
| `v1/attention/backends/mla/flashmla.py:74` | `return capability.major in [9, 10]` | major in [9, 10]=True | backend hook FlashMLA: major in [9, 10] | wrong, explicit selection only (not fixed) | R4: MLA backend that ROCm never auto-selects; forced selection fails late |
| `v1/attention/backends/mla/flashmla_sparse.py:160-162,171-172,227-228,250,366,419` | 6 sites, first `isinstance(kv_cache_spec, AttentionSpec) and kv_cache_spec.kv_quant_mode != KVQ…` | if reached: family(100)=True; major==9=False; major!=10=False | FlashMLA sparse internals (SM90 / SM100 / nvfp4 ds_mla) | unreachable on ROCm | FlashMLA sparse backend only (CUDA extension) |
| `v1/attention/backends/mla/flashmla_sparse.py:204` | `return capability.major in [9, 10]` | major in [9, 10]=True | FlashMLA sparse hook: major in [9, 10] | wrong, explicit selection only (not fixed) | True on gfx1030; explicit selection only; needs FlashMLA extension |
| `v1/attention/backends/mla/indexer.py:94` | `if use_fp4 and not current_platform.is_device_capability_family(100):` | if reached: family(100)=True | indexer mxfp4 KV needs SM100 family | unreachable on ROCm | reached only after the preceding `if is_rocm()` branch returned or raised (L86-93) |
| `v1/attention/backends/mla/indexer.py:367,932-933,940-941,965-967` | 4 sites, first `return [64 if current_platform.is_device_capability_family(90) else 128]` | if reached: family(90)=False; family(100)=True | DSA indexer: SM90/SM100 DeepGEMM paths | unreachable on ROCm | `is_cuda() and ... and has_deep_gemm()` / block-size list (DeepGEMM only) |
| `v1/attention/backends/mla/prefill/base.py:106` | `f"compute capability {device_capability.major}."` | n/a (message) | error message text | not a capability check | - |
| `v1/attention/backends/mla/prefill/flash_attn.py:195-197` | `is_sm90 = current_platform.is_device_capability(90) is_sm100_family = current_platform.is_device_capability_family…` | family(100)=True; family(120)=False; is(90)=False | MLA FA prefill: SM90/SM100/SM120 flags | wrong, masked (not fixed) | feed the FA3/FA4-specific branches, which require vllm_flash_attn_version 3/4 (None on ROCm) |
| `v1/attention/backends/mla/prefill/flash_attn.py:386,389-391` | `device_capability = current_platform.get_device_capability()` (+1 more) | [0]==9 -> False | V padding unless FA3 on SM90 or FA4 | correct | [0]==9 False: correct (requires padding on gfx1030) |
| `v1/attention/backends/mla/prefill/flash_attn.py:400` | `device_capability = current_platform.get_device_capability()` | [0] in (10,11) -> True | FA4 quant-output support: major in (10, 11) | wrong, masked (not fixed) | True on gfx1030 but `self.vllm_flash_attn_version == 4 and self._is_vllm_fa` precede and are False on ROCm |
| `v1/attention/backends/mla/prefill/flashinfer.py:58` | `return device_capability is not None and device_capability.major == 10` | major==10=True | MLA prefill hook: major == 10 (FlashInfer) | wrong, explicit selection only (not fixed) | True on gfx1030; ROCm uses RocmPlatform.get_mla_prefill_backend_cls (ROCM_AITER_FA / FLASH_ATTN) unless mla_prefill_backend is set explicitly |
| `v1/attention/backends/mla/prefill/selector.py:93` | `device_capability = current_platform.get_device_capability()` | n/a (passes (10,3)) | fetches capability only to validate an explicitly chosen prefill backend | correct | explicit selection only |
| `v1/attention/backends/mla/prefill/tokenspeed_mla.py:42` | `return device_capability is not None and device_capability.major == 10` | major==10=True | MLA prefill hook: major == 10 (TokenSpeed) | wrong, explicit selection only (not fixed) | True on gfx1030; explicit selection only |
| `v1/attention/backends/mla/prefill/trtllm_ragged.py:61` | `return device_capability is not None and device_capability.major == 10` | major==10=True | MLA prefill hook: major == 10 (TRT-LLM ragged) | wrong, explicit selection only (not fixed) | True on gfx1030; explicit selection only |
| `v1/attention/backends/mla/sparse_indexer.py:117-118` | `current_platform.is_cuda() and current_platform.is_device_capability_family(100)` | if reached: family(100)=True | sparse indexer requires SM100-class | unreachable on ROCm | `is_cuda() and family(100)` (L117-118) |
| `v1/attention/backends/mla/sparse_swa.py:933-936` | `num_decode_tokens == 0 or current_platform.is_rocm() or current_platform.is_xpu() or current_platform.is_device_ca…` | not reached on ROCm (is_rocm() True first) | SM12x skip of get_mla_metadata | correct | `current_platform.is_rocm()` precedes it in the same `or` chain (L934): returns early on ROCm |
| `v1/attention/backends/mla/tokenspeed_mla.py:126` | `return capability.major == 10` | major==10=True | backend hook TokenSpeed MLA: major == 10 | wrong, explicit selection only (not fixed) | R4: MLA backend that ROCm never auto-selects; forced selection fails late |
| `v1/attention/backends/mla/triton_mla.py:200,203-204,207,215-216` | 4 sites, first `cap = current_platform.get_device_capability()` | if reached: has(89)=True; has(80)=True | Triton MLA fp8/bf16 KV capability errors | unreachable on ROCm | inside `if current_platform.is_cuda():` (L199) |
| `v1/attention/backends/triton_attn.py:176-177` | `if current_platform.is_cuda() and current_platform.is_device_capability( (12, 0)` | if reached: is((12,0))=False | Triton attn SM-specific tweak | unreachable on ROCm | `is_cuda() and is(...)` (guarded) |
| `v1/attention/backends/triton_attn.py:547,550-551,554,561-562` | 4 sites, first `cap = current_platform.get_device_capability()` | if reached: has(89)=True; has(80)=True | Triton attn fp8/bf16 KV capability errors | unreachable on ROCm | inside `if current_platform.is_cuda():` (L546) |
| `v1/attention/backends/triton_flash_attn.py:25` | `return capability.major == 9 and super().supports_compute_capability(capability)` | major==9 -> False | FA half of the TRITON_FLASH_ATTN composite: Hopper only (major == 9) | correct | False on gfx1030: correct (a gfx90a (9,0) would match [INFERENCE: not run]) |
| `v1/attention/ops/flashmla.py:56,66-68` | 2 sites, first `if not current_platform.is_device_capability_family(90):` | if reached: family(90) -> False; family(100) -> True (after the extension check) | FlashMLA dense/sparse support (Hopper / Blackwell) | unreachable on ROCm | `_is_flashmla_available()` (CUDA extension) returns first; family(90)/(100) also False/True here |
| `v1/attention/ops/int4_per_token_head.py:970-971` | `current_platform.is_cuda() and current_platform.has_device_capability(80)` | if reached: has(80)=True | Hadacore needs CUDA sm_80+ | unreachable on ROCm | `is_cuda() and has(80) and hasattr(torch.ops._C, 'hadacore_transform')` |
| `v1/attention/ops/prefix_prefill.py:15,19` | `BASE_BLOCK = 128 if current_platform.has_device_capability(80) else 64` (+1 more) | has(80) -> True; ==(7,5) -> False | block size 128 on >= SM80 / Turing flag | correct | threshold: 103 >= 80 -> BASE_BLOCK 128; IS_TURING False. Not an SM100-meant check |
| `v1/attention/ops/triton_prefill_attention.py:218-219` | `elif current_platform.is_cuda_alike() and current_platform.has_device_capability( 80` | has(80) -> True | block size 128 on >= SM80 | correct | threshold pass, not SM100-meant |
| `v1/attention/ops/triton_reshape_and_cache_flash.py:26-28` | `return current_platform.has_device_capability(89) or current_platform.is_xpu() if kv_cache_dtype == "bfloat16": re…` | has(89) -> True; has(80) -> True | native fp8e4nv KV needs SM89+ (assert message) | wrong, masked (not fixed) | gfx1030 passes `has(89)` although it has no native FP8: only an assert at L418, hit with explicit --kv-cache-dtype fp8; the recipes use auto/int4 KV. Not fixed: needs a design decision for the ROCm FP8-KV predicate |
| `v1/attention/ops/triton_reshape_and_cache_flash.py:445` | `if torch.cuda.get_device_capability(key.device)[0] < 9:` | - | TILE_SIZE cap on pre-Hopper CUDA | unreachable on ROCm | in the `else: # cuda` arm (L442); `is_rocm() or is_xpu()` takes the first arm (L439) |
| `v1/attention/ops/triton_turboquant_decode.py:31-32` | `cap = torch.cuda.get_device_capability(device) _FP8_E4B15[device] = 1 if cap < (8, 9) else 0` | (10,3) < (8,9) -> False -> flag 0 | TurboQuant fp8e4b15 flag: cap < (8,9) | correct | threshold; TurboQuant KV only |
| `v1/attention/ops/triton_unified_attention.py:972-975` | `head_size == 256 and max_seqlen_q > 1 and num_queries_per_kv <= 16 and current_platform.is_device_capability_famil…` | family(100)=True | Blackwell launch tuning (BLOCK_M widening) | **WRONG - fixed** (`e2f6560c4b`) | TRITON_ATTN / ROCM_ATTN only; RDNA_ATTN (both recipes) never calls triton_unified_attention (grep of rdna_attn.py) |
| `v1/attention/ops/turboquant_soa/triton_turboquant_decode.py:31-32` | `cap = torch.cuda.get_device_capability(device) _FP8_E4B15[device] = 1 if cap < (8, 9) else 0` | (10,3) < (8,9) -> False -> flag 0 | TurboQuant fp8e4b15 flag: cap < (8,9) | correct | threshold; TurboQuant KV only |

### utils (deep_gemm, flashinfer)

| Site (`vllm/` omitted) | Check | gfx1030 gets | Intended | Verdict | Impact on flashnext-mtp2 / 27b-awq |
|---|---|---|---|---|---|
| `utils/deep_gemm.py:40` | `return (64,) if current_platform.is_device_capability_family(120) else (32, 64)` | if reached: family(120) -> False | DeepGEMM paged-MQA page sizes (SM120: only 64) | unreachable on ROCm | DeepGEMM only; is_deep_gemm_supported() False on ROCm |
| `utils/deep_gemm.py:51-52` | `return any( current_platform.is_device_capability_family(family)` | family(100)=True and family(120)=False; qwen3_5_text -> excluded (True) | Blackwell-only DeepGemm exclusion list (qwen3_5_text, qwen3_5_moe_text, kolibri1) | **WRONG - fixed** (`a5bb516a83`) | 27b-awq's text model_type is qwen3_5_text so base returned True, but the caller needs a quant config with use_deep_gemm (compressed-tensors AWQ has none): not hit |
| `utils/deep_gemm.py:86-88` | `if ( current_platform.is_device_capability_family(100) or current_platform.is_device_capability_family(120)` | if reached: family(100)=True; family(120)=False | DeepGEMM UE8M0 scale format on SM100/SM120 | unreachable on ROCm | `use_e8m0` requires is_deep_gemm_supported() (platform.support_deep_gemm(): False on ROCm) before reaching it |
| `utils/deep_gemm.py:604-605` | `return current_platform.is_cuda() and current_platform.is_device_capability_family( 100` | if reached: family(100)=True | DeepGEMM MX logits on SM100 | unreachable on ROCm | `is_cuda() and family(100)` |
| `utils/deep_gemm.py:690,702` | 2 sites, first `if current_platform.is_device_capability_family(90):` | if reached: family(90) -> False | SM90 paged-MQA next_n / multicast | unreachable on ROCm | DeepGEMM only (family(90) False on gfx1030 anyway) |
| `utils/flashinfer.py:127-128` | `current_platform.is_cuda() and current_platform.has_device_capability(90)` | if reached: has(90)=True | FlashInfer kernel-missing warning on Hopper+ | unreachable on ROCm | `is_cuda() and has(90) and has_flashinfer()` |
| `utils/flashinfer.py:182,185` | `device_capability = current_platform.get_device_capability()` (+1 more) | 103 | FlashInfer bf16 mm backend support query | wrong, masked (not fixed) | returns False earlier when flashinfer.mm_bf16 is missing (no flashinfer on ROCm); then passes 103 to flashinfer |
| `utils/flashinfer.py:691-693,697` | `if current_platform.is_device_capability( 90 ) or current_platform.is_device_capability_family(120):` (+1 more) | family(100) -> True (after has_nvidia_artifactory()) | supports_trtllm_attention(): SM100 family -> TRT-LLM prefill+decode | wrong, explicit selection only (not fixed) | returns True for family(100) when the NVIDIA artifactory is reachable or cubins present; consumers are the FLASHINFER attention backend only (`can_use_trtllm_attention`); that backend needs flashinfer which has no ROCm build |
| `utils/flashinfer.py:794-796` | `elif ( current_platform.is_device_capability(90) or current_platform.is_device_capability_family(120)` | if reached: is(90)/family(120) -> False | FlashInfer attention: XQA on SM90/SM12x with FP8 KV | unreachable on ROCm | inside use_trtllm_attention(), FlashInfer backend only |
| `utils/flashinfer.py:1190-1191` | `has_flashinfer() and current_platform.is_device_capability(90)` | is(90) -> False | FlashInfer FP8 block-scale GEMM (SM90) | wrong, masked (not fixed) | `has_flashinfer()` False; is(90) False anyway |
| `utils/flashinfer.py:1244` | `''prop.major >= 10''; on Hopper it raises a misleading` | n/a (docstring) | docstring | not a capability check | - |
| `utils/flashinfer.py:1252` | `if not current_platform.has_device_capability(100):` | has(100)=True | cuDNN FP8 SDPA needs Blackwell (has(100)) | wrong, masked (not fixed) | has(100) is True on gfx1030, but FLASHINFER must then be in RocmPlatform.get_supported_vit_attn_backends() and it is not (rocm.py:976-982) -> returns False |

### v1 other (sampler, worker, pool)

| Site (`vllm/` omitted) | Check | gfx1030 gets | Intended | Verdict | Impact on flashnext-mtp2 / 27b-awq |
|---|---|---|---|---|---|
| `v1/pool/flash_maxsim/_common.py:41-45` | `cap = current_platform.get_device_capability() if cap is not None: if cap.major >= 9: return 232_448 # Hopper+: 22…` | major>=9 -> 232_448 (fallback) | shared-memory budget fallback by capability major (Hopper+ 227 KiB, SM8x 99 KiB) | wrong, masked (not fixed) | wrong for gfx1030 (64 KiB LDS), but it is the `except` fallback of the Triton driver query `get_device_properties(...)['max_shared_mem']` (L33-36); pooling/late-interaction maxsim only, in neither recipe. Not fixed: the ROCm fallback value is a design choice |
| `v1/sample/ops/topk_topp_sampler.py:139,174` | 2 sites, first `CompilationContext._normalize_cuda_arch(capability.major, capability.minor)` | - | sampler triton top-k arch | unreachable on ROCm | `is_cuda()` guard at L164 precedes (existing fork guard) |
| `v1/worker/startup_plan.py:60` | `capability = current_platform.get_device_capability()` | '(10, 3)' string | device capability is a hash factor of the startup plan | correct | text/hash only |

## 9. Method and limits

* Scan: every `vllm/**/*.py` at `1154318cf0` (3,052 files) for `is_device_capability_family`, `has_device_capability`, `is_device_capability`,
  `get_device_capability`, `DeviceCapability`, raw `torch.cuda.get_device_capability`, `.major`/`.minor`/`to_int()`/`as_version_str()` and numeric
  `cc >= 90` style comparisons, plus string/tuple/int tables keyed by a capability (sections 1 and 8). `third_party/pynvml.py` is excluded.
* Reachability: `models/*/nvidia/**` is imported only when not ROCm (checked in each model's `__init__.py`: kimi_k3, deepseek_v41, qwen4_exp,
  deepseek_v4, inkling, muse_spark, minimax_m3, hy_v4, deepseek_v32); `step5`, `beam` and `dots3_note` have no `amd/` directory at all.
  Linear-kernel reachability comes from the `_POSSIBLE_*` per-platform lists in `model_executor/kernels/linear/__init__.py`.
* Not measured: no recipe was booted for this audit (no TP4 slot was requested because nothing the recipes execute changed); no other GPU architecture
  was run; model-level behaviour of the not-fixed sites is by reading.
