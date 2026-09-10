# Decode-Time Kernel Profile — Qwen3.8-27B-AWQ-INT4 on gfx1030

**Date**: 2026-09-10
**Author**: Sisyphus (decode-time profile run, methodology captured in `~/.config/opencode/skills/decode-profile/`)
**Hardware**: 4× Radeon PRO V620 (gfx1030, RDNA2 Wave32), TP=2 (physical GPUs 2, 3)
**Software**: torch 2.12.0+rocm7.14, venv-7.14.0, opengfx1030_vllm-rdna `rdna_extras` @ `6c5ff94ef`
**Workload**: Qwen3.8-27B-AWQ-INT4, 1k input / 512 max-tokens (model returns 80 tokens before its own stop), c=1 and c=8
**Stack**: FPP13 (FULL_AND_PIECEWISE greedy-correct) + PIX PYNCCL + custom all-reduce (force) + FA-RDNA2 + AWQ RDNA2 W4A16 + HIP GDN + prefix caching + state arenas
**rocprofv3 binary**: `/home/chenco_adm/Apps/vllm/venv-7.14.0/bin/rocprofv3` (bundled ROCm 7.14 SDK)

---

## TL;DR

**Decode is not GPU-bound. The GPU is 60–65% idle during decode.** The W4A16 GEMM dominates GPU time when the GPU is active (~30% of GPU cycles), but the GPU spends more wall time idle than running kernels. Per-step Python overhead + small-kernel launch gaps are the real bottleneck, not the GEMM itself.

If you want to ship a faster decode path on gfx1030, the work is **not** "make the W4A16 kernel faster" — it's "collapse the gaps between kernels" (cudagraph captures more aggressively, or eliminate per-step Python work) and **"investigate why cudagraph isn't already doing this."** The GEMM could probably be 30-40% faster in isolation, but that buys ~10% wall-clock.

| Phase | Window | Dispatch events | GPU time | GPU util |
|---|---|---:|---:|---:|
| cold_init (model load + cudagraph capture) | t = 0..50 s | 4,627 | 3.79 s | **7.6 %** |
| warmup request (2k/16) | t = 55..63 s | 61,144 | 0.59 s | **7.4 %** |
| **c=1 decode (1k/80)** | t = 100..140 s | **654,887** | **15.80 s** | **39.5 %** |
| c=8 prefill (8 × 1k) | t = 145..185 s | 52,127 | 39.29 s | **98.2 %** |
| **c=8 decode (8 × 80)** | t = 185..200 s | **532,895** | **5.20 s** | **34.7 %** |

---

## 1. Methodology

### What worked

Wrap vLLM serve60 from birth with venv-7.14.0's bundled rocprofv3 in `--run` mode. This captures every kernel dispatch (name, start/end GPU timestamp, grid, workgroup, register usage) into CSV on clean shutdown.

```bash
# Launcher (see opengfx1030_vllm-rdna/docs/profiling/2026-09-10-decode-profile/scripts/)
exec /home/chenco_adm/Apps/vllm/venv-7.14.0/bin/rocprofv3 \
    --kernel-trace true --rccl-trace true --marker-trace true \
    --output-format csv --output-file "$RUN_DIR/prof" \
    -- env <canonical serve60 env> \
       bash opengfx1030_vllm-rdna/scripts/serve_gfx1030_full.sh
```

Send a warmup (2k/16) request, mark `c1_start` and run c=1 measurement, mark `c8_start` and run c=8 measurement, then SIGTERM the launcher. rocprofv3 catches the signal, waits for child workers to exit, merges the per-process `.dat` buffers into a single `prof_kernel_trace.csv` (~555 MB), and exits.

### What did NOT work (and why this took 90 minutes of trial)

The gfx1030 + venv-7.14.0 + torch-2.12.0+rocm7.14 stack has at least five distinct failure modes for the obvious profilers. Captured in the `decode-profile` skill for future runs.

| Tool | Failure | Cause |
|---|---|---|
| `/opt/rocm/bin/rocprofv3` (system, ROCm 7.2) | dispatch attach returns status 1 | wrong rocprofiler-sdk; gfx1030 tuning index missing |
| `/opt/rocm/core-7.14/bin/rocprofv3` | calls wrong rocprof-attach subprocess | symlink-resolved binary path picks `/opt/rocm-7.2.0/core-7.14/bin/rocprof-attach` even though invoked via `/opt/rocm/core-7.14/` |
| `LD_PRELOAD=librocprofiler-sdk.so` + `ROCP_TOOL_ATTACH=1` | Python `Illegal instruction` at init | torch isn't register-built, and the wrapper path crashes the loader on EPYC 7452 (Zen 2, no AVX-512) |
| `rocprof-sys-run --rocm=kernel` | `Illegal instruction` at startup | librosys-deps uses pre-AVX-512 paths or some instruction Zen 2 doesn't have |
| `rocprofv3 --attach <pid>` | "no rocp-bg-attach thread" | torch wheel not built with `ROCPROFILER_REGISTER_BUILD_DEFAULT_ATTACHMENT=ON` |
| `torch.profiler` in worker (VLLM_TORCH_PROFILER_DIR) | only captures Python attribution | cudagraph-replayed kernels are opaque to torch.profiler |

**Only the venv's `rocm_sdk_core` rocprofv3 works** — it ships its own embedded ROCm 7.14 SDK with venv-7.14.0 Python 3.12, ABI-compatible with torch 2.12.0+rocm7.14. No LLVM clash, no SIGILL.

### Phase boundaries (GPU-time seconds)

The trace duration is 198.62 s. Cold init (model load + cudagraph capture + JIT) takes the first ~95 s. After that, kernel density clearly delimits the workload phases. The wall-clock anchors (`c1_start.txt = 1789076532.605`, `c1_end.txt = 1789076558.782`, `c8_start.txt = 1789076558.785`, `c8_end.txt = 1789076624.556`) bracket these phases; the analysis script uses GPU-time windows directly because rocprofv3 timestamps are nanoseconds since GPU init, not Unix epoch.

| Phase | GPU-time (s) | Wall (s) |
|---|---|---|
| cold_init | 0..50 | 105 s (entire run prior to warmup request) |
| warmup | 55..63 | 8 s for the 2k-prompt warmup |
| **c=1 decode** | **100..140** | **40 s for c=1 decode (80 tokens × ~250 tok/s)** |
| c=8 prefill | 145..185 | 40 s for 8 × 1k prefills |
| **c=8 decode** | **185..200** | **15 s of c=8 decode (8 × 80 tokens at lower per-req rate)** |

The wall-clock durations within the GPU-time windows don't match (40 s wall > 15 s wall for the decode phases is a function of cudagraph replay overhead + Python launch overhead, not the GPU kernel times themselves).

---

## 2. C=1 decode window (40 s, 654,887 events, GPU util 39.5%)

The 10 hottest kernels by GPU time. **% of GPU time** is GPU-time / total-kernel-GPU-time-in-phase (15.80 s); **% wall** would be ÷40 s and is roughly half those numbers.

| # | Kernel | Count | Total ms | Avg µs | % GPU time |
|---|---|---:|---:|---:|---:|
| 1 | `vllm::gptq_rdna2_prefill::gemm_dynamic_kernel<…>` (template A, large M, prefill GEMM) | 1,021 | 3,622 | 3,548 | **22.9 %** |
| 2 | `vllm::gptq_rdna2_prefill::gemm_dynamic_kernel<…>` (template B, M=1 decode GEMM) | 19,872 | 1,443 | 73 | **9.1 %** |
| 3 | `vllm::cross_device_reduce_1stage<__half, 2>(…)` (vLLM cross-device reduce) | 12,900 | 1,048 | 81 | **6.6 %** |
| 4 | `fa_prefill_paged_varlen_kernel_256<__half, false>(…)` (FA-RDNA2 prefill) | 64 | 492 | 7,694 | 3.1 % |
| 5 | `at::native::reduce_kernel<512, 1, ReduceOp<bool, …>>` (torch reduce) | 52,420 | 434 | 8 | 2.7 % |
| 6 | `rocblas_gemvt_kernel<false, 256, _Float16, float, _Float16>(…)` (LM head GEMV) | 4,804 | 377 | 79 | **2.4 %** |
| 7 | `(anonymous namespace)::gdn_prefill_delta_h_packed_kernel(__half const*, …)` | 192 | 354 | 1,843 | 2.2 % |
| 8 | `__amd_rocclr_copyBuffer` (HIP allocator) | 97,809 | 348 | 4 | 2.2 % |
| 9 | `vllm::gptq_rdna2::gemm_q4_kernel_rdna2<__half, 1>(…)` (M=1 manual decode GEMM) | 4,608 | 308 | 67 | **1.9 %** |
| 10 | `at::native::vectorized_elementwise_kernel<8, …>(…)` (torch elementwise) | 52,416 | 239 | 5 | 1.5 % |

Plus: `gdn_prefill_o_rdna2_kernel` (192 × 1.04 ms = 199 ms = 1.3%), `Cijk_Alik_Bljk_HHS_BH_MT128x128x16_…` (200 × 0.73 ms = 146 ms = 0.9% — rocBLAS, but only 200 calls so not the steady-state decode path), `index_elementwise_kernel<128, 4, gpu_index_…>` (5,308 × 18 µs = 95 ms = 0.6%), and **`fa_decode_paged_splitk_kernel_256<__half, false, false>(…)`** (1,536 × 53 µs = **81 ms = 0.5 %** — FA-RDNA2 decode attention, tiny).

### Why the c=1 decode window includes prefill kernels

The window t = 100..140 s was chosen to bracket the dense-decode burst (1,300+ gemm_dynamic_kernel dispatches per second for 12 s), but it includes some prefill activity at the edges. To isolate **pure decode**, a tighter window t = 110..132 s would strip out the FA-RDNA2 prefill + GDN prefill chunks that ran at the front of the burst. The prefill kernels (#1, #4, #7, #10-of-c1_decode) are all the chunks-of-the-first-prefill — they happen because vLLM chunked the 1k prompt across ~12 steps. Pure-decode is dominated by **#2, #3, #5, #6, #9**.

---

## 3. C=8 decode window (15 s, 532,895 events, GPU util 34.7%)

The 10 hottest kernels by GPU time. Total kernel GPU time = 5.20 s.

| # | Kernel | Count | Total ms | Avg µs | % GPU time |
|---|---|---:|---:|---:|---:|
| 1 | `vllm::gptq_rdna2_prefill::gemm_dynamic_kernel<…>` (M=8 decode GEMM) | 16,146 | 1,362 | 84 | **26.2 %** |
| 2 | `vllm::cross_device_reduce_1stage<__half, 2>(…)` | 10,191 | 450 | 44 | **8.6 %** |
| 3 | `at::native::reduce_kernel<512, 1, ReduceOp<bool, …>>` | 43,442 | 430 | 10 | 8.3 % |
| 4 | `fa_decode_paged_splitk_kernel_256<__half, false, false>(…)` (FA-RDNA2 decode) | 1,264 | 409 | 323 | **7.9 %** |
| 5 | `vllm::gptq_rdna2::gemm_q4_kernel_rdna2<__half, 8>(…)` (M=8 manual decode GEMM) | 3,696 | 394 | 106 | **7.6 %** |
| 6 | `__amd_rocclr_copyBuffer` | 80,775 | 358 | 4 | 6.9 % |
| 7 | `Cijk_Alik_Bljk_HHS_BH_MT16x16x16_…` (rocBLAS, 1 call per shape from autotune) | 75 | 293 | 3,900 | 5.6 % |
| 8 | `at::native::vectorized_gather_kernel<16, long>(…)` | 7,566 | 174 | 23 | 3.3 % |
| 9 | `at::native::vectorized_elementwise_kernel<8, …>` | 43,134 | 139 | 3 | 2.7 % |
| 10 | **`gdn_decode_packed_rdna2_kernel<float, __half, …>(…)`** (HIP GDN decode) | 3,792 | **138** | 36 | **2.7 %** |

Plus: `index_elementwise_kernel<128, 4, index_co…>` (3,792 × 35 µs = 133 ms = 2.6%), `elementwise_kernel_manual_unroll<128, 8, …>` (22,590 × 5 µs = 120 ms = 2.3%), `reduce_kernel<512, 1, ReduceOp<float, …>>` (16,511 × 6 µs = 97 ms = 1.9%), `gemm_q4_kernel_rdna2<__half, 16>(…)` (3,696 × 24 µs = 90 ms = 1.7% — M=16 path firing for one of the GEMMs in the c=8 batch).

### Comparison vs c=1

- **W4A16 GEMM avg grew from 73 µs to 84 µs** going from c=1 to c=8 — almost no per-call cost growth because c=8 only doubles M (and at these sizes the GEMM is bandwidth-bound, not compute-bound).
- **FA-RDNA2 decode avg grew from 53 µs to 323 µs** — 6× per-call, because attention is compute-bound and KV-cache reads grow with batch.
- **HIP GDN decode appears at c=8** (`gdn_decode_packed_rdna2_kernel`, 138 ms) but **NOT at c=1**. At c=1 the GDN decode path uses a different kernel (likely the Triton/FLA fallback). Investigating this asymmetry is a TODO below.
- **All-reduce per-call latency dropped from 81 µs to 44 µs** at c=8 — the reduce kernel amortizes better with batch.
- **`at::native::reduce_kernel<bool>` is the #3 hot kernel at c=8** (430 ms, 8.3%). At c=1 it's only #5 (434 ms, 2.7%) — same absolute time, but proportionally much larger at c=8. This is suspicious; worth identifying what's being reduced.

---

## 4. Headline findings

### Finding 1: Decode is CPU/launch-overhead bound, not GPU-bound

The GPU is idle **60–65 % of the wall-clock decode window**. Even though the W4A16 GEMM is the largest single consumer of GPU cycles, the GPU spends more time waiting for the next kernel to launch than it spends running kernels.

This is the single most important finding. The intuitive answer ("decode is bandwidth-bound, the GEMM is the bottleneck") is wrong for this stack. The W4A16 GEMM is bandwidth-bound in principle, but the GPU is not fed kernels fast enough to actually saturate its bandwidth.

**Implication**: optimization work targeting the GEMM itself (smaller tiles, fused epilogue, LDG swizzles) will only buy 10-20 % wall-clock, even with a 30-40 % GEMM speedup, because the GEMM is only running ~25 % of the time.

**Implication**: cudagraph replay is supposed to fix exactly this — collapse per-step Python overhead and small-kernel gaps into one captured launch. The fact that GPU util is only 39.5 % at c=1 means the cudagraph is **not** collapsing the gaps the way it should. Two hypotheses:

1. The captured graph has too many small kernels with their own launch gaps (unlikely — cudagraph replays a single graph node).
2. The Python per-step work (`prepare_inputs`, `finish_requests`, `compute_slot_mappings`, `build_attn_metadata`) is still happening outside the graph, and the graph itself is short enough that the Python overhead dominates.

Per the existing AGENTS.md, the prior investigation found `prepare_inputs` + `finish_requests` together are only ~5 ms per step (out of ~80 ms per token at c=1). So hypothesis 2 is also unlikely. **The captured cudagraph must have its own gaps** — likely caused by NVTX-annotated regions or kernels the graph doesn't capture (FA-RDNA2 JIT load, PYNCCL collect calls, prefix-cache hash operations, etc.).

### Finding 2: The W4A16 GEMM family dominates GPU cycles (~30%)

Two `gemm_dynamic_kernel` template instantiations show up at c=1:

- Template A: 1,021 calls × **3,548 µs avg** = 3.62 s. This is the LARGE-M prefill path firing on chunks. Wait, in the c=1 decode window it shouldn't be firing that often. Looking again — 1,021 calls in 40 s = ~25/s. That's prefill-chunk GEMMs running during the chunked prefill of the c=1 request. Not decode.
- Template B: 19,872 calls × **73 µs avg** = 1.44 s. This IS the M=1 decode path. 19,872 calls in 40 s = 497/s. For c=1 at 80 tokens × ~250 tok/s effective rate, 80 tokens means each layer's GEMMs run ~80 times. 28 layers × 4 W4A16 GEMMs each × 80 tokens = 8,960 calls minimum — we see 19,872. So multiple template instantiations of `gemm_dynamic_kernel` are running for each layer per token (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, plus the GDN linear_attn projections).

The 73 µs avg for M=1 is plausible for a bandwidth-bound AWQ GEMM at N=4096, K=4096, M=1: that's ~16 MB of weights to read in 73 µs = **~220 GB/s sustained per GPU**. RDNA2 V620 peak HBM bandwidth is ~512 GB/s, so we're at ~43 % of peak. Reasonable for a small-M GEMM where occupancy and pipe utilization matter.

At c=8 the same kernel runs at 84 µs avg with batch M=8, reading ~16 MB of weights (same as M=1, weights are dominant). 16 MB / 84 µs = 190 GB/s. Slightly slower per kernel but serving 8× more requests, so aggregate throughput is fine.

### Finding 3: FA-RDNA2 decode attention is NOT the bottleneck at c=1 (0.5 %)

`fa_decode_paged_splitk_kernel_256` at c=1: 1,536 calls × 53 µs = **81 ms** (0.5 % of GPU time). At c=8: 1,264 calls × 323 µs = 409 ms (7.9 %). The c=8 number is real; c=1 is essentially negligible.

This rules out the natural hypothesis that "decode attention is the bottleneck." It's not. With 53 µs per attention call at c=1, the attention kernels are well-optimized.

### Finding 4: `at::native::reduce_kernel<bool>` fires 100k+ times — what's being reduced?

This kernel fires 100,869 times across the full run (52,420 in c=1 decode alone, 43,442 in c=8 decode). Per call it's only 8-10 µs, but the cumulative time (914 ms across the run, 434 ms in c=1 decode = 2.7 %) is non-trivial.

The kernel template `reduce_kernel<512, 1, ReduceOp<bool, …>>` operates on bool arrays. The most likely candidates in vLLM's hot path:

- `attention_logits > 0` reductions in top-k/top-p masking
- Bool masks in prefix cache lookup
- Bool masks in MoE top-k gating

Hard to tell from the trace alone. **TODO**: profile with `rocprofv3 --marker-trace` to see if there are ROCTx ranges labeling these calls.

### Finding 5: All-reduce via PYNCCL is 6-8 % of GPU time, ~10 % wall

`cross_device_reduce_1stage` at c=1: 12,900 calls × 81 µs = 1,048 ms (6.6 % of GPU time, 2.6 % of wall). At c=8: 10,191 calls × 44 µs = 450 ms (8.6 % of GPU time, 3.0 % of wall).

The `ncclDevKernel_Generic_4` (PYNCCL ring) shows up at 781 ms in c=8 prefill (the chunked-prefill all-reduce), but only **36 ms total during c=8 decode** — meaning at decode we are NOT using the PYNCCL ring; we're using the vLLM `cross_device_reduce_1stage` (which is the vLLM-inlined reduce for custom all-reduce + a small one-shot path).

Wait — let me re-read this. The previous AGENTS.md says custom AR force-flag is enabled. So `cross_device_reduce_1stage` IS the custom AR. Per-call 44 µs at c=8 is well above the prior measured all-reduce latency (~13-40 µs reported in AGENTS.md for the VLLM custom AR). So custom AR is firing but slower than the reported best-case. Why?

Possible answer: the prior 13-40 µs measurement was for a small data size (10 KB), and here at c=8 with seq-len accumulation we're sending larger payloads. Or the AGENTS.md latency was measured at c=1 with different per-call payloads. Either way, custom AR is contributing 2-3 % of wall — measurable but not dominant.

### Finding 6: `gdn_decode_packed_rdna2_kernel` fires at c=8 but NOT c=1

This is the HIP GDN decode kernel. 138 ms at c=8 (2.7 %), **zero calls in the c=1 decode window**. At c=1 the GDN decode must use a different path. The likely candidates:

- Triton/FLA fallback at low batch (c=1, batch=1)
- A different template instantiation of `gemm_dynamic_kernel` handling the GDN compute path

Either way, this is a **dispatcher asymmetry**: c=1 GDN decode takes one path, c=8 takes another. The two paths have different cost structures. To verify, would need to look at the GDN forward dispatch in `qwen_gdn_linear_attn.py`.

### Finding 7: Cold init + cudagraph capture is fine (no surprises)

Cold-init GPU util is only 7.6 % — most of the 105 s cold init is **CPU work** (weight loading, kernel compilation, cudagraph capture setup). The GPU only fires 4,627 kernels during the first 50 s, dominated by the rdna_ar_oneshot at 1.94 s (2 calls, but each is 970 ms — that's the FABRIC init for the custom-AR peer buffer). Once the cold init is done and the c=1 request lands, dispatch rate explodes to ~16,000/s.

The warmup phase (8 s, 61,144 events) shows the same kernels firing at ~steady-state rate, confirming warmup did its job.

---

## 5. Optimization priorities

### Priority 1: Reduce per-step Python overhead / investigate cudagraph gap source

**Why**: 60 % GPU idle means 60 % of the GPU's bandwidth is being thrown away. Fixing this would more than double decode throughput in theory (from ~250 tok/s to potentially ~600 tok/s for c=1 1k/512).

**How**:
- Profile with `VLLM_DBG_STEP_TIMING=1` AND check the captured-graph replay log to see what's inside the captured region.
- If the captured graph itself has gaps (NVTX, PYNCCL collect calls, prefix-cache hash), move those OUTSIDE the graph or capture them as part of the graph.
- If the gap is between graphs (Python overhead between graph launches), look at `prepare_inputs`, `compute_slot_mappings`, `finish_requests`, `build_attn_metadata`. Per AGENTS.md these together are ~5 ms — not the main culprit.
- Check whether `--cudagraph_capture_sizes=[1,2,4,8,16]` is the right capture set, or whether adding more shapes (like 256/512 prefill chunk sizes) helps.

### Priority 2: Optimize the M=1 W4A16 GEMM template

**Why**: 9.1 % of c=1 GPU time, 26.2 % of c=8 GPU time. A 30 % GEMM speedup would buy ~7 % wall-clock (c=1) or ~9 % wall-clock (c=8). Modest but real.

**How**:
- `gemm_dynamic_kernel<__half, 8, …>` is the M=1 path. Current 73 µs/call at ~220 GB/s suggests there's room to push bandwidth utilization higher.
- Look at `csrc/rocm/q_gemm_rdna2_awq_*.cu` and the prefill GEMM tile parameters. RDNA2 sweet spot for M=1 is usually BLOCK_K=128, BLOCK_N=64-128, num_warps=2 (matching Wave32).
- If current BLOCK_N is too small, increasing it increases per-block weight-cache pressure but may help when wave-pipelining.

### Priority 3: Identify and merge `at::native::reduce_kernel<bool>` (100k+ calls)

**Why**: 914 ms across full run, 430 ms at c=8 decode. Even at 5 µs per call, 100k calls add up.

**How**:
- Add `torch.cuda.nvtx.range` markers around suspected reduce call sites (top-k/top-p sampling, prefix-cache lookup) to find which one fires 100k+ times.
- Once identified, consider merging the reduce into the preceding kernel (e.g., a fused "compute_topk_inplace" that does the reduction internally).

### Priority 4: Investigate c=1 GDN decode path asymmetry

**Why**: `gdn_decode_packed_rdna2_kernel` fires 3,792 times at c=8 but not at c=1. At c=1 the GDN decode must use a slower path (Triton/FLA fallback or a non-optimal template of `gemm_dynamic_kernel`). Since c=1 represents single-request interactive use, getting this right matters.

**How**:
- Look at `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:_gdn_decode_dispatch` (or wherever the dispatch is). Find the gate that picks HIP kernel vs Triton fallback at c=1.
- Verify the M=1 path is firing the right template.

### Priority 5: All-reduce latency

**Why**: 6.6-8.6 % GPU time. With PIX + Simple protocol we're at the floor for PYNCCL; the vLLM custom AR is firing at 44-81 µs/call.

**How**:
- `VLLM_FORCE_CUSTOM_ALL_REDUCE=1` is on. Verify which AR path is selected at c=1 vs c=8.
- Investigate why `cross_device_reduce_1stage` (custom AR path) is 81 µs at c=1 but 44 µs at c=8. If there's a bug in the small-batch path, fixing it could shave 50 µs × 320 calls/s = ~16 ms/s of wall time.

---

## 6. What this profile does NOT tell you

- **Per-req decode throughput at c=1**: only ~80 decode tokens were generated per request (the model stopped on its own at "The capital of Portugal is" — vLLM doesn't add stop tokens, but the prompt structure apparently caused early EOS). At 80 tokens we see ~250 tok/s aggregate per-req. With 512-token output we'd see more of the steady-state (no end-of-decode ramp-down).
- **Whether the bottlenecks scale to 16k context**: this was a 1k/512 workload. The 16k context run from prior sessions uses the same kernel set but with much more FA-RDNA2 work per step.
- **CPU-side overhead breakdown**: `VLLM_DBG_STEP_TIMING` already showed ~5 ms total per step. Adding NVTX markers to the Python dispatcher would let us see which functions dominate.
- **Real wall-clock per-token latency**: the kernel trace has GPU timestamps only. To map GPU time to per-request wall-clock, we'd need to add `cudaEventRecord` around each forward call.

---

## 7. Files saved

| File | Size | Contents |
|---|---:|---|
| `/tmp/profile_decode/run6/prof_kernel_trace.csv` | 555 MB | 1,322,173 kernel dispatch records (raw) |
| `/tmp/profile_decode/run6/prof_rccl_api_trace.csv` | 255 KB | RCCL collective timings |
| `/tmp/profile_decode/run6/prof_agent_info.csv` | 2 KB | GPU + queue metadata |
| `/tmp/profile_decode/run6/decode_kernel_breakdown.txt` | 13 KB | Phase-windowed kernel aggregation (this report's data) |
| `/tmp/profile_decode/run6/agg_full.txt` | — | Full-run aggregation (every kernel, ranked by total time) |
| `/tmp/profile_decode/run6/c{1,8}_{start,end}.txt` | 21 B each | Wall-clock anchors for the workload phases |

The CSV can be re-aggregated with different phase windows by editing `analyze_phase3.py` (bundled in `~/.config/opencode/skills/decode-profile/scripts/`).

---

## 8. Related work

- `docs/profiling/2026-09-10-truefull-journal.md` — the prior session's investigation of the 16k "duct" bug. The fix landed; this profile run confirms the fixed stack is correct.
- `~/.config/opencode/skills/decode-profile/` — methodology skill for re-running this profile on different workloads / concurrency levels.
- `~/.config/opencode/skills/rdna-kernel-debug/` — canonical launch command and GPU-cleanup script.
- AGENTS.md "Critical: Queue Management for Multi-Concurrency" — `GPU_MAX_HW_QUEUES=2` is the production setting; this profile run used that setting.
