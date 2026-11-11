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

## 2026-11-02: the glue around the kernel

With MTP the spec-verify path launched about eight small kernels per GDN layer
around `gdn_spec_verify_rdna2`: `rearrange_mixed_qkv` (one cat and three
contiguous copies), one copy each for `a` and `b`, one for `z`, and the copy
of the kernel output into `core_attn_out`. The launch-site attribution of the
unmodified tree puts those at about 260 kernels and 1.3 ms per decode step.

`VLLM_GDN_STRIDED_INPUTS=1` (default) removes them. The spec-verify kernel
reads q/k/v and `a`/`b` at their token strides, as views into the conv output
and the projections, and writes `core_attn_out` itself when the batch is
pure-spec. The gated RMSNorm reads `z` in place at its token stride.
Mixed prefill+decode batches and every non-dense layout still take the copies.
The results are bitwise identical to the copy path
(`test_gdn_decode_rdna2.py`, `test_gdn_output_projection.py`).

Flash-Next MTP2 TP4, AR gate 256 KiB, 1k/512 (16k/1024 for the last row), HC
and shared-expert batch already in, ITL p50 in ms, one run per cell:

| cell | before | after |
|---|---:|---:|
| c1 | 24.9 | 23.8 |
| c6 | 49.6 | 47.9 |
| c8 | 52.9 | 51.8 |
| 16k c8 | 54.6 | 53.8 |

That is 1.5-4.4 % of the step. Mean accepted length moves by 0.1 or more
between runs of the same build, so decode tokens/s is within noise; prefill
tokens/s and solo TTFT did not change (1k c8 2057 to 2075 tok/s, 4k / 16k solo
1.86 / 7.49 s to 1.82 / 7.33 s).
