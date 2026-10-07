# AWQ vs GPTQ Prefill Kernel Review — Qwen3.8-27B-AWQ-INT4 on gfx1030

**Date**: 2026-09-10
**Context**: Decode profile (`docs/profiling/2026-09-10-decode-kernel-profile.md`) showed `gptq_rdna2_prefill::gemm_dynamic_kernel<Config<512,4,32,8,8>>` at 22.9% of c=1 decode GPU time (3.5 ms/call avg). User asked: "should we have AWQ kernel not GPTQ for prefill?" and "unify code if we can, but review whether AWQ kernel is optimal."

---

## TL;DR

1. **The AWQ prefill kernel exists and is compiled, but never fires.** `_custom_ops.py` lacks the Python wrapper for `awq_gemm_rdna2_prefill`, so the dispatcher falls through to `gptq_gemm_rdna2_prefill` with `use_v2_format=True`. Output is numerically correct.
2. **The GPTQ prefill kernel already handles AWQ correctly** via the `use_v2_format` flag (`zero_offset=0` for AWQ, `zero_offset=1` for GPTQv1). No correctness issue.
3. **The separate AWQ kernel (`q_gemm_rdna2_awq_prefill.cu`) is NOT optimal** for large-M prefill: fixed BLOCK_M=16, BLOCK_N=64, THREADS=128 gives worse wave utilization and more L2 pressure than the GPTQ kernel's ConfigV1 (M_TILE=8, N_TILE=2048, THREADS=512).
4. **The real issue is config selection for large M.** The GPTQ kernel's `select_config` only handles M <= 32; for M > 256 it always returns ConfigV1 (M_TILE=8), which creates 78-256 M-blocks with split_k=16 atomic contention. ConfigA (M_TILE=16) or ConfigC (M_TILE=16) would be better.
5. **Recommendation: unify on the GPTQ prefill kernel** and extend `select_config` for large M. Remove or deprecate the separate AWQ kernel. Add the missing `_custom_ops.py` wrapper for API parity.

---

## 1. Why the AWQ kernel doesn't fire

### Dispatch flow

```
rdna2_w4a16.py:_rdna2_w4a16_select_kernel(m, k, n, is_awq=True)
  → M > 256: returns "awq_prefill" if _awq_prefill_available() else "prefill"
  → _awq_prefill_available() checks torch.ops._rocm_C.awq_gemm_rdna2_prefill → True
  → kernel_name = "awq_prefill"

apply():
  if kernel_name == "awq_prefill" and hasattr(ops, "awq_gemm_rdna2_prefill"):
    # ops = vllm._custom_ops
    # hasattr(ops, "awq_gemm_rdna2_prefill") → FALSE (no Python wrapper!)
    ...
  else:
    if hasattr(ops, "awq_gemm_rdna2_prefill") and use_v2_format:
      # Still False
      ...
    elif hasattr(ops, "gptq_gemm_rdna2_prefill"):
      # True → calls GPTQ prefill kernel with use_v2_format=True
      output = ops.gptq_gemm_rdna2_prefill(x, w_q, w_zp, w_s, w_g_idx, True)
```

### Evidence

- `_custom_ops.py` has `awq_dequantize` and `awq_gemm` (line 513, 547) but **no `awq_gemm_rdna2_prefill`**.
- `torch_bindings.cpp` registers `awq_gemm_rdna2_prefill` (line 111-116).
- `.so` contains `awq_gemm_rdna2_prefill` symbols (`nm -D` confirms).
- `torch.ops._rocm_C.awq_gemm_rdna2_prefill` is callable from Python (verified).

**Conclusion**: The AWQ kernel is compiled, registered, and callable — but the Python wrapper in `_custom_ops.py` is missing, so the dispatcher can't reach it.

---

## 2. Kernel comparison

### GPTQ prefill kernel (`q_gemm_rdna2_prefill.cu`)

| Aspect | Detail |
|---|---|
| Namespace | `vllm::gptq_rdna2_prefill` |
| Configs | ConfigV1 (512,4,32,8,8), ConfigA (256,4,32,16,0), ConfigC (128,4,32,16,0) |
| Config selection | `select_config(M, N, K)` — dynamic, but only handles M <= 32 |
| Kernel variants | `gemm_static_kernel<Config, K_PER_SPLIT>` (static LDS) or `gemm_dynamic_kernel<Config>` (dynamic) |
| Epilogue | Packed-fp16 atomic add (`atomic_add_pk4_f16`) |
| AWQ support | Yes, via `use_v2_format` → `zero_offset=0` |
| GPTQv1 support | Yes, via `use_v2_format` → `zero_offset=1` |

### AWQ prefill kernel (`q_gemm_rdna2_awq_prefill.cu`)

| Aspect | Detail |
|---|---|
| Namespace | `vllm::gptq_rdna2_awq_prefill` |
| Config | Fixed: BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, THREADS=128, LDS_PAD=8 |
| Config selection | None — single tile size for all shapes |
| Kernel variants | `gemm_awq_prefill_kernel` (single kernel) |
| Epilogue | fp32 partials + deterministic reduce (multi-split) OR packed-fp16 atomic add (single-split) |
| AWQ support | AWQ-only (`use_v2_format` must be True, `zero_offset=0`) |
| GPTQv1 support | No — explicitly AWQ-only |

### Performance analysis for M=624-2048, N=6144, K=2560 (per-rank TP=2)

| Metric | GPTQ ConfigV1 | AWQ kernel | GPTQ ConfigA (proposed) |
|---|---:|---:|---:|
| M_TILE | 8 | 16 | 16 |
| N_TILE | 2048 | 64 | 1024 |
| THREADS | 512 | 128 | 256 |
| M-blocks (M=624) | 78 | 39 | 39 |
| N-blocks (N=6144) | 3 | 96 | 6 |
| Blocks per K-split | 234 | 3,744 | 234 |
| split_k | 16 | 16 | 16 |
| Total blocks | 3,744 | 59,904 | 3,744 |
| Waves per block | 16 | 4 | 8 |
| LDS per block | 4 KiB | 1.3 KiB | 20 KiB |
| Atomic contention | 16× per tile | 16× per tile (fp32 partials, no atomic) | 16× per tile |
| Est. occupancy | 1 block/CU (16 waves) | 2 blocks/CU (8 waves) | 1 block/CU (8 waves) |
| Est. L2 pressure | Low (3 N-blocks) | High (96 N-blocks) | Medium (6 N-blocks) |

**Key observations:**

1. **AWQ kernel has 59,904 total blocks** vs GPTQ ConfigV1's 3,744. That's 16× more blocks, each with 128 threads instead of 512. HSA dispatch overhead alone is significant.

2. **AWQ kernel has 96 N-blocks** vs GPTQ's 3. Each N-block reads the same K-slice of weights from global memory. With 96 N-blocks, the same weight is read 96 times from L2/HBM instead of 3 times. That's 32× more L2/HBM traffic for weights.

3. **AWQ kernel has only 4 waves per block** (128 threads) vs GPTQ's 16 waves (512 threads). On RDNA2 with 80 CUs, each CU can run 1-2 workgroups. With 128-thread blocks, each CU runs 2 blocks = 256 threads = 8 waves. With 512-thread blocks, each CU runs 1 block = 16 waves. The GPTQ kernel has better wave utilization per CU.

4. **AWQ kernel uses fp32 partials + deterministic reduce** for multi-split, which is more accurate but slower (extra kernel launch + more memory traffic). GPTQ uses packed-fp16 atomic add, which is faster but non-deterministic.

5. **ConfigV1 is suboptimal for large M** (M > 64). It was designed for M <= 64 (comment in source: "original v1 tile, wins for small M (M <= 64)"). For M=624, M_TILE=8 gives 78 M-blocks, each doing 8 rows × 2048 cols × 160 K = 2.6M FLOPs. ConfigA with M_TILE=16 gives 39 M-blocks, each doing 16 rows × 1024 cols × 160 K = 2.6M FLOPs — same FLOPs per block, but fewer blocks and less atomic contention.

---

## 3. Numerical correctness

### GPTQ prefill kernel with AWQ

The GPTQ prefill kernel handles AWQ via `use_v2_format`:
- `use_v2_format=True` → `zero_offset=0` (AWQ: literal zeros, no +1 bias)
- `use_v2_format=False` → `zero_offset=1` (GPTQv1: stored as zero-1, add +1)

The kernel reads `b_qzeros` and applies the dequant formula:
```cpp
// In refresh_group / dequant_4bit_8_fp16:
// z1z16 = (zero + zero_offset) * scale
// y1y16 = scale
// dequantized = (w - z1z16) * y1y16
```

For AWQ (`zero_offset=0`), this recovers the original weight exactly as the AWQ checkpoint intends. For GPTQv1 (`zero_offset=1`), it adds the +1 bias to recover the original zero.

**Verified**: The profiled run produced coherent output ("The capital of France is Paris..." → correct). The 16k c=8 validation passed 8/8. The GPTQ kernel with AWQ is numerically correct.

### AWQ kernel

The AWQ kernel is explicitly AWQ-only (`zero_offset=0` always). It would also be numerically correct for AWQ models. But it doesn't support GPTQv1 models.

---

## 4. Unification assessment

### Should we keep two separate kernels?

**No.** The GPTQ prefill kernel already handles both AWQ and GPTQv1 via `use_v2_format`. Maintaining a separate AWQ kernel is code duplication with no benefit.

### Should we use the AWQ kernel instead of GPTQ ConfigV1 for large M?

**No.** The AWQ kernel has worse tile parameters for large-M prefill (small N_TILE=64, few threads=128, fixed M_TILE=16). The GPTQ kernel with ConfigA (M_TILE=16, N_TILE=1024, THREADS=256) would be better.

### Should we unify on the GPTQ prefill kernel?

**Yes.** The GPTQ prefill kernel is the right base:
- Already handles both AWQ and GPTQv1
- Has a config system that can be extended
- Better tile parameters (larger N_TILE, more threads)
- Single codebase to maintain

### What needs to change

1. **Extend `select_config` for large M** (M > 256):
   ```cpp
   if (size_m > 256) {
     if (size_n >= 4096) return ConfigId_A;  // M_TILE=16, N_TILE=1024
     return ConfigId_C;                       // M_TILE=16, N_TILE=512
   }
   ```

2. **Add the missing `_custom_ops.py` wrapper** for `awq_gemm_rdna2_prefill` (for API parity, in case anyone calls it directly).

3. **Remove or deprecate the separate AWQ prefill kernel** (`q_gemm_rdna2_awq_prefill.cu`). Keep it as a fallback but route the dispatcher through the unified GPTQ kernel.

4. **Optionally add a new ConfigD for very large M** (M > 1024):
   ```cpp
   using ConfigD = Config<256, 4, 32, 32, 0>;  // M_TILE=32
   ```

5. **Review `compute_split_k` for large M**:
   - Current logic: split until LDS <= 32 KiB, then push to split_k=16 for parallelism
   - For large M, split_k=16 creates 16× atomic contention. Consider capping at 8 or 4 for M > 256.
   - The trace shows split_k=16 at 2.2 ms avg vs split_k=4 at 6.3 ms avg vs split_k=8 at 8.6 ms avg — but these are different shapes, not a controlled comparison.

---

## 5. Proposed optimization plan

### Phase 1: Fix config selection (immediate, low risk)

**Change**: Extend `select_config` in `q_gemm_rdna2_prefill.cu` to handle M > 256.

```cpp
inline int select_config(int size_m, int size_n, int size_k) {
  // Large-M prefill: use ConfigA (M_TILE=16) for better LDS reuse
  if (size_m > 256) {
    if (size_n >= 4096) return ConfigId_A;
    return ConfigId_C;
  }
  // Existing small-M logic (unchanged)
  if (size_m < 4 && size_n > 4096) return ConfigId_A;
  if (4 <= size_m && size_m <= 8 && size_n >= 2048 && size_k >= 1024)
    return ConfigId_C;
  if (size_m == 12 && size_n >= 2560 && size_n <= 8192 && size_k >= 512)
    return ConfigId_C;
  if (size_m == 32 && size_n >= 5120 && size_n <= 6144 && size_k >= 1536)
    return ConfigId_C;
  return ConfigId_V1;
}
```

**Expected gain**: 10-20% faster prefill GEMM for M > 256 (fewer M-blocks, better LDS reuse, less atomic contention).

**Quality gate**: Numerical parity test — compare output of ConfigV1 vs ConfigA for the same input. Must match within 1e-3 relative error (fp16 precision).

### Phase 2: Tune split_k for large M (medium risk)

**Change**: Cap split_k at 8 for M > 256 to reduce atomic contention.

```cpp
// In compute_split_k:
if (size_m > 256) {
  // Cap split_k at 8 for large M — 16× atomic contention is too high
  while (split_k > 8 && (size_k / split_k) % K_STEP != 0) split_k /= 2;
}
```

**Expected gain**: 5-15% faster prefill GEMM (less atomic contention, better L2 reuse).

**Quality gate**: Same numerical parity test.

### Phase 3: Add ConfigD for very large M (low risk, future)

**Change**: Add `ConfigD = Config<256, 4, 32, 32, 0>` (M_TILE=32) for M > 1024.

**Expected gain**: 5-10% faster for M > 1024 (fewer blocks, better LDS reuse).

**Quality gate**: Numerical parity + end-to-end output check.

### Phase 4: Remove separate AWQ kernel (cleanup, low risk)

**Change**: Remove `q_gemm_rdna2_awq_prefill.cu` and its registration. Add `_custom_ops.py` wrapper for API parity.

**Expected gain**: Code reduction, single codebase.

**Quality gate**: End-to-end output check (the GPTQ kernel already handles AWQ).

---

## 6. Files to change

| File | Change | Priority |
|---|---|---|
| `csrc/rocm/q_gemm_rdna2_prefill.cu` | Extend `select_config` for M > 256 | P1 |
| `csrc/rocm/q_gemm_rdna2_prefill.cu` | Cap split_k at 8 for M > 256 | P2 |
| `csrc/rocm/q_gemm_rdna2_prefill.cu` | Add ConfigD for M > 1024 | P3 |
| `vllm/_custom_ops.py` | Add `awq_gemm_rdna2_prefill` wrapper | P1 |
| `csrc/rocm/q_gemm_rdna2_awq_prefill.cu` | Remove or deprecate | P4 |
| `csrc/rocm/torch_bindings.cpp` | Remove `awq_gemm_rdna2_prefill` registration | P4 |
| `tests/kernels/quantization/test_rdna2_w4a16.py` | Add numerical parity tests for ConfigA/C | P1 |

---

## 7. Quality preservation gates

Every change must pass:

1. **Numerical parity test**: For a random [M, K] input and AWQ weights, compare the output of the new config against the old config. Relative error must be < 1e-3 (fp16 precision).
2. **End-to-end output check**: Serve the model and run the greedy probe (`tools/probe_greedy_correctness.py`). Must return "Paris", "2", "Berlin" etc.
3. **Prefill throughput regression test**: Run `vllm bench throughput` with 1k/512 c=1 and c=8. Must not regress more than 5% from baseline.

---

## 8. Summary

| Question | Answer |
|---|---|
| Should we use the AWQ kernel for prefill? | No — it's compiled but never fires (missing Python wrapper), and its tile parameters are worse than GPTQ ConfigV1 for large M. |
| Is the GPTQ kernel correct for AWQ? | Yes — `use_v2_format` handles AWQ's literal zeros correctly. |
| Should we unify the kernels? | Yes — the GPTQ prefill kernel already handles both AWQ and GPTQv1. |
| What's the real bottleneck? | ConfigV1 (M_TILE=8) is suboptimal for M > 256. ConfigA (M_TILE=16) would be better. |
| What's the expected gain? | 10-20% faster prefill GEMM for M > 256 with ConfigA + split_k tuning. |
| What's the risk? | Low — the change is config selection only, not kernel logic. Numerical parity test ensures correctness. |
