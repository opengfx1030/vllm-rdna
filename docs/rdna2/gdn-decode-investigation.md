# GDN decode kernel investigation (gfx1030)

Date: 2026-09-15. Kernel: `csrc/rocm/gdn_decode_rdna2.cu`
(`gdn_decode_packed_rdna2_kernel`). Target: Qwen3.8-Flash-Next-AWQ-W4A16,
TP=4, V1 + FULL_AND_PIECEWISE + prefix caching.

## Question

After enabling the HIP GDN decode path for the Flash-Next (fp16 state), an
earlier bench suggested a ~13% TPOT regression vs Triton. Is the HIP kernel
actually slow, and if so why?

## Method

1. Standalone microbenchmark of the kernel at production shapes
   (H=6, HV=12, K=V=128, grid `(V/32, B*HV)`, 256 threads/block).
2. ISA inspection (`hipcc -Rpass-analysis=kernel-resource-usage` + `.s` dump).
3. Empty-kernel launch-overhead baseline with the same grid/block.
4. Live A/B on the same build: `VLLM_GDN_DECODE_RDNA2=1` vs `=0`,
   1k/512 c=8, 32 prompts.

## Findings

### 1. The kernel is launch-bound, not compute-bound

| B | grid | per-call |
|---:|---|---:|
| 1 | (4, 12) = 48 | 6.1 µs |
| 2 | (4, 24) = 96 | 6.1 µs |
| 4 | (4, 48) = 192 | 6.1 µs |
| 8 | (4, 96) = 384 | 6.6 µs |
| 16 | (4, 192) = 768 | 8.3 µs |

A 16× grid increase (48 → 768 workgroups) costs only ~36% more time. The
empty-kernel launch baseline at the same grid/block is 3.3–3.8 µs:

| grid | empty-kernel launch |
|---:|---:|
| 48 | 3.27 µs |
| 384 | 3.83 µs |
| 768 | 4.46 µs |

So the GDN decode's **actual execution is only ~2.8 µs**; the rest is launch
overhead. Under cudagraph the launch is amortized, so the in-graph cost is
closer to ~2.8 µs.

### 2. The kernel body is already efficient

- VGPRs 62, SGPRs 46, **no spills**.
- State loads are vectorized (`global_load_dwordx4` ×4 = the 16 states; with
  fp16 state it is 4× fewer bytes).
- 391 `v_` instructions, 25 `s_waitcnt`.

Remaining micro-inefficiencies (immaterial given the size):
- Occupancy 4 waves/SIMD (capped by `amdgpu_waves_per_eu(2, 4)` + 62 VGPRs).
- 49 fp16→fp32 conversions (from the new fp16-state path; fp32 state needs 0).
- Scalar `global_load_ushort` ×9 and one `global_store_short` (a/b/q/k/out
  not fully vectorized).

### 3. It is not a meaningful share of decode time

48 GDN layers × ~6.1 µs ≈ **0.29 ms/step**. At c=8 the Flash-Next TPOT is
~47 ms, so the GDN decode is ~0.6% of the step.

### 4. The HIP decode is FASTER than the Triton decode

Same build, 1k/512 c=8, 32 prompts:

| GDN decode | out tok/s | TPOT |
|---|---:|---:|
| HIP `gdn_decode_rdna2` | **150.46** | **47.15 ms** |
| Triton (`VLLM_GDN_DECODE_RDNA2=0`) | 142.97 | 50.96 ms |

The earlier "13% regression" was a build/cache artifact (the two runs were on
different `.so` builds and cache states), not the GDN kernel.

## Conclusion

Nothing is meaningfully slowing the GDN decode. The kernel is launch-bound
with a fast body (~2.8 µs), it is ~0.6% of the decode step, and the HIP path
beats Triton. The fp16-state change is a net win, not a regression.

## If further optimization is wanted

The leverage is in **launch count**, not the kernel body:

1. **Fuse `causal_conv1d_update` + `gdn_decode_rdna2`** into one kernel. The
   conv1d is a separate launch per GDN layer, so this halves the GDN-path
   launches (48/step saved).
2. **Raise occupancy** (`amdgpu_waves_per_eu(2,4)` → `(4,8)`): may shave the
   ~2.8 µs execution. Small.
3. **Vectorize the scalar a/b/q/k/out accesses**: minor.

The real decode bottleneck is elsewhere — the MoE dominates (~0.98 ms/layer
vs ~3 µs for GDN). Use the `decode-profile` skill (rocprofv3) to get the
per-kernel breakdown of a live decode window.
