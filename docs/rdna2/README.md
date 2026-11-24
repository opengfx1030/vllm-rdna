# RDNA2 (gfx1030) documents: index

The notes in this directory are the engineering record of the `opengfx1030/vllm-rdna` fork. This page is
the only index; it lists every file, grouped by topic. Files keep their names, so links from other
repositories and from `bench_results/` stay valid.

Operator-facing entry points live outside this directory:

- [`docs/rdna/serving.md`](../rdna/serving.md): launcher, recipes and what each recipe key does.
- [`docs/rdna/monokernel/`](../rdna/monokernel/README.md): the Flash-Next MonoKernel.
- [`docs/explore/`](../explore/kv-int4-fp16-scales/README.md): exploration notes that have not become a shipped feature.

Each entry carries a kind:

- **guide**: kept current for a feature, a tool or a procedure. When code and a guide disagree, the guide is
  the bug.
- **dated record**: a measurement, plan, investigation, status or receipt as of the date it states (in its
  file name or in its first lines). It is not updated afterwards, so its numbers, branch names and commit
  hashes describe the tree it was written against. Read its code and recipe statements against the current
  recipes and [tuning guide](tuning.md).

The one-line descriptions come from each file's first heading and summary.

## Serving and recipes

| File | Kind | What it is |
| --- | --- | --- |
| [`gguf-flashnext-2026-11-16.md`](gguf-flashnext-2026-11-16.md) | guide | Qwen3.8 Flash-Next from a GGUF checkpoint (UD-Q4_K_XL): serve command, plugin fork, measurements. |
| [`V620-FULL-AND-PIECEWISE.md`](V620-FULL-AND-PIECEWISE.md) | guide | What `FULL_AND_PIECEWISE` captures and replays, and the stride fixes piecewise capture still needs. |
| [`tp8-pp2-flashnext-2026-11-06.md`](tp8-pp2-flashnext-2026-11-06.md) | dated record | Flash-Next on all eight V620s: TP8 and TP4 x PP2, memory table and comparison. |
| [`bench_27b_awq_matrix.md`](bench_27b_awq_matrix.md) | dated record | 27B AWQ bench matrix (gfx1030, TP=4). |
| [`V620-FP16-PERFORMANCE.md`](V620-FP16-PERFORMANCE.md) | dated record | V620 FP16 performance results of 2026-09-13 (MTP2 fastest across four valid 16k/32k trials). |
| [`V620-PIECEWISE-CUDAGRAPHS.md`](V620-PIECEWISE-CUDAGRAPHS.md) | dated record | Compiled piecewise CUDA graphs on the V620 stack, 2026-09-23. |
| [`V620-GIT-FAST-SERVICE.md`](V620-GIT-FAST-SERVICE.md) | dated record | How the V620 TP4 fast service of September 2026 was reproduced from Git. |

## Tuning

| File | Kind | What it is |
| --- | --- | --- |
| [`tuning.md`](tuning.md) | guide | RDNA tuning guide: fork knobs, modes, TunableOp and the measurements behind the defaults. |
| [`moe-epilogue-accum.md`](moe-epilogue-accum.md) | guide | MoE epilogue accumulation modes (`VLLM_RDNA2_MOE_FP32_ACCUM`, `VLLM_RDNA2_MOE_DETERMINISTIC`). |
| [`mixed-batch-stall-probe.md`](mixed-batch-stall-probe.md) | guide | `stall_probe.py`: catch, measure and reproduce prefill/decode stalls; the decode-stall cap. |
| [`tunableop-rank-agreement-2026-11-16.md`](tunableop-rank-agreement-2026-11-16.md) | dated record | Why TunableOp rank files must agree per key, how to check it and what the rewrite cost. |
| [`hwq-abba-2026-11-06.md`](hwq-abba-2026-11-06.md) | dated record | `GPU_MAX_HW_QUEUES` 4 versus 2 for `flashnext-mtp2`: ABBA run and decision. |
| [`V620-QSA-LIVE-PREFILL.md`](V620-QSA-LIVE-PREFILL.md) | dated record | QSA prefill scoring bounded to the live prompt context. |

## Inkling-2-Mini

Serving, switches and limits are in `inkling2-serving.md`; the recipes are `inkling2-mini-*` under
[`tools/rdna/recipes/`](../../tools/rdna/recipes/).

| File | Kind | What it is |
| --- | --- | --- |
| [`inkling2-serving.md`](inkling2-serving.md) | guide | Serving at TP=1, TP=2 and TP=4 with HIP graphs: recipes, switches, memory, long context. |
| [`inkling2-tensor-parallel.md`](inkling2-tensor-parallel.md) | guide | How the model shards for TP = 1, 2, 4 on the HIP serving path. |
| [`inkling2-dspark.md`](inkling2-dspark.md) | guide | DSpark draft (`mtp.safetensors`): native HIP port, adaptive K, draft vocabulary, tables. |
| [`inkling2-vision.md`](inkling2-vision.md) | guide | Image input on gfx1030. |
| [`inkling2-attn-kernels.md`](inkling2-attn-kernels.md) | guide | Attention and short-conv kernels (`inkling2_attention`, `inkling2_sconv_*`, `inkling2_kv_prep`). |
| [`inkling2-linear-kernels.md`](inkling2-linear-kernels.md) | guide | Dense-projection kernels (`inkling2_linear_*`). |
| [`inkling2-glue-kernels.md`](inkling2-glue-kernels.md) | guide | Glue kernels for everything else in a layer visit. |
| [`inkling2-moe-prefill-bm.md`](inkling2-moe-prefill-bm.md) | guide | Prefill expert GEMMs for prefill-sized chunks (`moe_prefill_bm_rdna2.cu`). |
| [`inkling2-fuse-attn-chain.md`](inkling2-fuse-attn-chain.md) | guide | `VLLM_INKLING2_FUSE_ATTN_CHAIN`: the fused attention-side decode chain and graph node count. |
| [`inkling2-baseline-2026-11-24.md`](inkling2-baseline-2026-11-24.md) | dated record | Full TP=1, TP=2 and TP=4 serving baseline (DSpark off and on) on `rdna_extras` `36a09895bb`. |
| [`inkling2-benchmarks.md`](inkling2-benchmarks.md) | dated record | Serving benchmarks at TP=1 on one V620. |
| [`inkling2-decode-glue.md`](inkling2-decode-glue.md) | dated record | b=1 decode: glue launches and speculation cost. |
| [`inkling2-decode-moe-profile.md`](inkling2-decode-moe-profile.md) | dated record | Decode at b = 1..4: expert kernels, Infinity Cache, prefetch, attention decode. |
| [`inkling2-prefill-profile.md`](inkling2-prefill-profile.md) | dated record | Prefill profile at TP=1. |
| [`inkling2-speculation-profile.md`](inkling2-speculation-profile.md) | dated record | b=1 speculation profile that led to the DSpark options. |
| [`inkling2-prefill-tile-qualification.md`](inkling2-prefill-tile-qualification.md) | dated record | Pipelined prefill attention tile: qualification and archdiff receipt. |
| [`inkling2-archdiff-receipt.md`](inkling2-archdiff-receipt.md) | dated record | Code-generation diff of the csrc change of the merge branch. |
| [`inkling2-pr-receipts.md`](inkling2-pr-receipts.md) | dated record | Per-PR test receipts for the merge branch (planner differential test and others). |
| [`inkling2-port-spec.md`](inkling2-port-spec.md) | dated record | Serving port spec written before the port: checkpoint layout and what the port has to provide. |

## HI2 and Qwen3.8-27B

HI2 is the user-facing name of the HadamardINT2-FP16 weight format (`rdna_hint_flex`).

| File | Kind | What it is |
| --- | --- | --- |
| [`hi2-27b-tp2-serving.md`](hi2-27b-tp2-serving.md) | guide | Serving Qwen3.8-27B-HI2-3.0bpw at TP=2: switch table, memory, limits, results. |
| [`rdna-hint-flex-weights.md`](rdna-hint-flex-weights.md) | guide | `rdna_hint_flex` weights (spec v3.2) on RDNA2. |
| [`hi2-scoreboard-2026-11-14.md`](hi2-scoreboard-2026-11-14.md) | guide | The scoreboard driver: one JSON and one markdown per arm, used to judge every HI2 change. |
| [`hi2-27b-tp2-2026-11-14.md`](hi2-27b-tp2-2026-11-14.md) | dated record | Recipe, probes and the decode kernel-time breakdown on two V620. |
| [`hi2-27b-mtp-2026-11-14.md`](hi2-27b-mtp-2026-11-14.md) | dated record | MTP speculative decoding with HI2 at TP2 (CPU-only slice). |
| [`hi2-27b-tunableop-2026-11-15.md`](hi2-27b-tunableop-2026-11-15.md) | dated record | TunableOp rows for the rocBLAS prefill GEMMs at TP=2. |
| [`hi2-attn-prefill-2026-11-14.md`](hi2-attn-prefill-2026-11-14.md) | dated record | Prefill attention on `RDNA_ATTN`: measurement and an opt-in GEMM path. |
| [`hi2-gdn-glue-2026-11-15.md`](hi2-gdn-glue-2026-11-15.md) | dated record | GDN decode glue: state arenas, every launch, in-place paged-state decode. |
| [`hi2-prefill-gemm-2026-11-15.md`](hi2-prefill-gemm-2026-11-15.md) | dated record | Can a custom dense fp16 prefill GEMM beat rocBLAS on the V620? |
| [`hi2-prefill-gemm2-2026-11-15.md`](hi2-prefill-gemm2-2026-11-15.md) | dated record | Round 2 of the same kernel: wired behind a switch, then optimized. |
| [`hi2-residual-direct-2026-11-15.md`](hi2-residual-direct-2026-11-15.md) | dated record | Gemma residual path: cached scale and in-place fused add-norm. |

## LeapQuant (INT8 GDN state)

| File | Kind | What it is |
| --- | --- | --- |
| [`leapquant-gdn-2026-11-13.md`](leapquant-gdn-2026-11-13.md) | guide | LeapQuant INT8 GDN recurrent state: opt-in port and how to use it. |
| [`leapquant-config-guards-2026-11-14.md`](leapquant-config-guards-2026-11-14.md) | guide | One flag (`--replayssm-state-dtype leapquant`) and one validator. |
| [`leapquant-gdn-numerics-and-isolation-2026-11-14.md`](leapquant-gdn-numerics-and-isolation-2026-11-14.md) | dated record | Numerics, determinism and state isolation checks. |
| [`leapquant-page-packing-2026-11-14.md`](leapquant-page-packing-2026-11-14.md) | dated record | Page packing for fewer GDN groups per request (draft, default off). |
| [`leapquant-pr170-gpu-validation-2026-11-15.md`](leapquant-pr170-gpu-validation-2026-11-15.md) | dated record | GPU validation of PR 170: 27B AWQ at TP2 and Flash-Next at TP4. |

## All-reduce (`rdna_ar`)

| File | Kind | What it is |
| --- | --- | --- |
| [`rdna-ar-twoshot.md`](rdna-ar-twoshot.md) | guide | Operational notes of the RDNA TP all-reduce: switches, gates, the wedge marker. |
| [`rdna-ar-hier-2026-11-06.md`](rdna-ar-hier-2026-11-06.md) | dated record | Hierarchical `rdna_ar` for two PCIe switch islands (TP8); shipped, default `auto`. |
| [`allreduce-vector-finisher-2026-11-02.md`](allreduce-vector-finisher-2026-11-02.md) | dated record | One-shot all-reduce latency floor, variants and the vector finisher. |
| [`tp4-allreduce-sizes-2026-10-10.md`](tp4-allreduce-sizes-2026-10-10.md) | dated record | TP=4 all-reduce at serving sizes. |
| [`rdna-ar-soak-2026-11-06.md`](rdna-ar-soak-2026-11-06.md) | dated record | Soak of `rdna_ar` on one slot next to boots on the other. |
| [`rdna-ar-wedge-2026-11-04.md`](rdna-ar-wedge-2026-11-04.md) | dated record | Investigation of a production TP4 wedge: KFD userptr evictions of the pinned PLE table. |
| [`rdna-ars-allreduce-plan.md`](rdna-ars-allreduce-plan.md) | dated record | Reduce-scatter plus allgather all-reduce: plan for a future PR (deferred). |
| [`explore-pr5-wvsplitk-rdna-ar.md`](explore-pr5-wvsplitk-rdna-ar.md) | dated record | Explore: unintegrated PR 5 wvSplitK and `rdna_ar` pieces. |

## MoE offload and EPLB

| File | Kind | What it is |
| --- | --- | --- |
| [`moe-offloading-target.md`](moe-offloading-target.md) | dated record | AWQ INT4 expert and KV offloading target (2026-10-28). |
| [`moe-offloading-improvements.md`](moe-offloading-improvements.md) | dated record | First optimization pass over the RDNA2 MoE offloading. |
| [`moe-offload-topology-2026-11-02.md`](moe-offload-topology-2026-11-02.md) | dated record | Expert-offload topology (TP2/TP4), HIP collectives and the v0.33 graph fix. |
| [`eplb-lean-design-2026-11-16.md`](eplb-lean-design-2026-11-16.md) | dated record | Lean EPLB for the expert-parallel MoE: design, phase-0 tooling and numbers. |
| [`expert-stats-2026-11-07.md`](expert-stats-2026-11-07.md) | dated record | Expert routing statistics and the placement solver (hetero MoE). |
| [`expert-stats-t10-2026-11-07.md`](expert-stats-t10-2026-11-07.md) | dated record | Test T10: expert statistics on the production server. |
| [`a2a-dbo-feasibility-2026-11-16.md`](a2a-dbo-feasibility-2026-11-16.md) | dated record | All-to-all, DBO and EP on gfx1030: feasibility, go/no-go and plan. |
| [`V620-CONCURRENT-MOE-20260927.md`](V620-CONCURRENT-MOE-20260927.md) | dated record | Concurrent MTP decode with the sequential W4A16 MoE (`VLLM_ROCM_MOE_SKINNY_MAX_M`). |

## PCIe, host and box audits

| File | Kind | What it is |
| --- | --- | --- |
| [`pcie-iommu-audit-2026-11-06.md`](pcie-iommu-audit-2026-11-06.md) | dated record | PCIe / PLX / IOMMU audit and P2P matrix after a re-cabling. |
| [`pciescope-2026-11-07.md`](pciescope-2026-11-07.md) | dated record | pciescope: PCIe fabric monitor and probe. |
| [`pinned-host-evictions-2026-11-05.md`](pinned-host-evictions-2026-11-05.md) | dated record | Pinned host memory and KFD queue evictions: the other pinned buffers. |
| [`disagg-g0-2026-11-06.md`](disagg-g0-2026-11-06.md) | dated record | P/D disaggregation go/no-go spikes G0a and G0b on Flash-Next TP4. |

## W4A8 and the MonoKernel

| File | Kind | What it is |
| --- | --- | --- |
| [`w4a8-wiring.md`](w4a8-wiring.md) | dated record | Dense W4A8 sdot4 prefill wiring (opt-in); written on the historical `w4a8-wiring` branch. |
| [`w4a8-moe-wiring.md`](w4a8-moe-wiring.md) | dated record | MoE W4A8 wiring; written on the historical `w4a8-wiring` branch. |
| [`w4a8-quality-2026-11-04.md`](w4a8-quality-2026-11-04.md) | dated record | W4A8 MoE quality against W4A16 on Flash-Next TP4. |
| [`mono-w4a8-2026-11-05.md`](mono-w4a8-2026-11-05.md) | dated record | `--monokernel` together with `VLLM_RDNA2_W4A8_SDOT4=1`. |
| [`qwen4/monokernel-design.md`](qwen4/monokernel-design.md) | dated record | Qwen4 monokernel design for gfx1030, written before any Qwen4 weight existed. |
| [`qwen4/readiness.md`](qwen4/readiness.md) | dated record | Qwen4 readiness and backport map (research only). |
| [`qwen4/PROGRESS-Qwen4MonoDesign.md`](qwen4/PROGRESS-Qwen4MonoDesign.md) | dated record | Progress note of the Qwen4 monokernel design turn. |

## Validation and receipts

| File | Kind | What it is |
| --- | --- | --- |
| [`regression-suite-d0b0c2b9af-2026-11-23.md`](regression-suite-d0b0c2b9af-2026-11-23.md) | dated record | No-regression suite on `rdna_extras` after PRs 197 to 200, and the spec-decode NaN-count crash PR 200 fixed. |
| [`fa-gqa-decode-validation-2026-11-11.md`](fa-gqa-decode-validation-2026-11-11.md) | dated record | FA-RDNA2 GQA decode kernel validation and the opt-in decision. |
| [`determinism-2026-11-04.md`](determinism-2026-11-04.md) | dated record | Run-to-run determinism of Flash-Next on gfx1030. |
| [`concurrency-probe-noise-2026-11-12.md`](concurrency-probe-noise-2026-11-12.md) | dated record | Concurrency-probe FAILs on `flashnext-mtp2`: noise, not corruption. |
| [`async-stall-profile-2026-11-02.md`](async-stall-profile-2026-11-02.md) | dated record | rocprofv3 serve profile: prefill/decode asynchrony, batching and stalls. |
| [`topk-ci-2026-11-06.md`](topk-ci-2026-11-06.md) | dated record | ROCm coverage for the radix top-k tests. |
| [`capability-audit-2026-11-05.md`](capability-audit-2026-11-05.md) | dated record | NVIDIA capability checks on gfx1030: audit of `vllm/`. |

## Investigations and bring-up

| File | Kind | What it is |
| --- | --- | --- |
| [`flash-next-ple-ipc.md`](flash-next-ple-ipc.md) | dated record | Flash-Next PLE offload: CUDA-IPC failure and fix. |
| [`flash-next-prefix-cache-corruption.md`](flash-next-prefix-cache-corruption.md) | dated record | Flash-Next prefix-caching correctness bug. |
| [`gdn-decode-investigation.md`](gdn-decode-investigation.md) | dated record | GDN decode kernel investigation. |
| [`qwen38_flash_next_triton_status.md`](qwen38_flash_next_triton_status.md) | dated record | Qwen3.8-Flash-Next Triton path status of 2026-09-15. |
| [`qwen4_exp_hip_path.md`](qwen4_exp_hip_path.md) | dated record | Qwen4Exp HIP path: opt-in scaffolding shipped 2026-09-14. |
| [`qwen4_exp_hip_tests.md`](qwen4_exp_hip_tests.md) | dated record | Test plan for the Qwen4Exp HIP path. |
| [`qwen4_exp_hip_tracker.md`](qwen4_exp_hip_tracker.md) | dated record | Work tracker for what to test and optimize on the HIP path. |
| [`dsv4-fp16-attention-plan.md`](dsv4-fp16-attention-plan.md) | dated record | DeepSeek-V4-Flash fp16 attention on RDNA (plan and status). |
| [`V620-BASELINE-PORT-20260922.md`](V620-BASELINE-PORT-20260922.md) | dated record | V620 baseline before porting the fast service. |
| [`V620-CANDIDATE.md`](V620-CANDIDATE.md) | dated record | V620 candidate integration of 2026-09-13. |
| [`V620-UPSTREAM-FINDINGS.md`](V620-UPSTREAM-FINDINGS.md) | dated record | V620 findings against `rdna_extras`, September 15. |

## Ports, rebases and build breaks

| File | Kind | What it is |
| --- | --- | --- |
| [`v0.31.0-port-notes.md`](v0.31.0-port-notes.md) | dated record | `rdna_extras` to v0.31.0 port: decisions, open questions, test plan. |
| [`rebase-v0.32.0-inventory.md`](rebase-v0.32.0-inventory.md) | dated record | Fork feature inventory before the v0.32.0 move. |
| [`build-break-2026-10-31.md`](build-break-2026-10-31.md) | dated record | `rdna_extras` build break at one commit and its fix. |
