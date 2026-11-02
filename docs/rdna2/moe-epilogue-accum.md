# MoE epilogue accumulation mode for gfx1030 (`VLLM_RDNA2_MOE_FP32_ACCUM`)

Branch: `rdna_extras`. Applies to the fused RDNA2 MoE kernels
`moe_gptq_gemm_rdna2` (W4A16) and `moe_w4a8_gemm_rdna2` (W4A8) on gfx1030.
The knob selects how the per-expert partial sums are accumulated into the
output `c`.

The product default is the fp32 scratch. Opt out with
`VLLM_RDNA2_MOE_FP32_ACCUM=0` for the packed-fp16 CAS-64 epilogue.

## The two modes

### Default — fp32 scratch (`fp32_accum=True`)

Partials accumulate with `atomicAdd` on a cached fp32 scratch, then one
fp32→fp16 cast writes `c`.

On gfx1030 that add is not `v_global_atomic_add_f32`. That opcode is gfx11+.
`atomicAdd(float*)` lowers to a 32-bit `global_atomic_cmpswap` retry loop.
`moe_accum_row<float>` issues four of them, one per N column (offsets 0, 4,
8, 12). The fp16 result is still a single rounding at the cast: there is no
intermediate fp16 rounding. The scratch is a persistent per-(rows, n, device)
allocation created before graph capture; the per-call zero is
`hipMemsetAsync`.

Four CAS-32 loops per row can be slower than the single CAS-64 opt-out under
MoE decode contention. This page does not assign a tok/s delta to that. A
V620 check is still required before treating the default as validated (see
below).

### Opt-out — packed-fp16 CAS-64 (`VLLM_RDNA2_MOE_FP32_ACCUM=0`)

Each partial is rounded to fp16 and accumulated with `atomic_add_pk4_f16`:
one `atomicCAS` on a 64-bit word (`global_atomic_cmpswap_b64`) into the
caller's pre-zeroed fp16 output. gfx1030 has no
`v_global_atomic_pk_add_f16`. The add is order-dependent at fp16 precision,
so concurrent expert blocks can flip the last 1–2 fp16 ULPs between runs.

## Knobs

| Knob | Value | Effect |
|---|---|---|
| env `VLLM_RDNA2_MOE_FP32_ACCUM` | unset / `1` / `true` | **Default:** fp32 scratch. |
| env `VLLM_RDNA2_MOE_FP32_ACCUM` | `0` / `false` | Packed-fp16 CAS-64 epilogue. |
| per-call `fp32_accum=` (in `_custom_ops.py`) | `True` / `False` | Overrides the env-resolved default for that call. |

The wrapper reads the env var **once at import** into
`_custom_ops._RDNA2_MOE_FP32_ACCUM` (no env access in traced/captured code)
and passes the resolved `fp32_accum` per call. The per-call override wins
over the env default. Both `moe_gptq_gemm_rdna2` and `moe_w4a8_gemm_rdna2`
share the flag. The torch op schema default is already `fp32_accum=True`;
serving goes through the wrapper, so the env is what selects the mode.
Recipes and `tools/rdna/serve_rdna.sh` do not set this variable.

```bash
# fp32 scratch (default): leave unset, or
export VLLM_RDNA2_MOE_FP32_ACCUM=1

# previous packed-fp16 CAS-64 epilogue
export VLLM_RDNA2_MOE_FP32_ACCUM=0
```

Set the variable in the process environment before vLLM imports. A value
exported only inside an already-running engine is not reread.

## Historical measurement (not a validation of this default)

Source: `bench_results/2026-09-30_fp32-accum-matrix/SUMMARY.md` (4× Radeon PRO
V620, TP=4, FULL_AND_PIECEWISE cudagraphs, prefix caching, FA-RDNA2, frozen
TunableOp rows, Flash-Next AWQ-W4A16). `Δ` = CAS relative to fp32, **positive
= CAS faster**. Recorded when the two modes were A/B'd. It is not a sign-off
for shipping fp32 as the default, and it is not a new number from this
change.

| Arm / cell | out tok/s fp32 | out tok/s CAS | Δ out | prefill fp32→CAS | Δ prefill |
|---|---:|---:|---:|---|---:|
| MTP=0 16k/1k c=1 | 30.04 | 30.78 | **+2.5 %** | 1677.9→1745.1 | +4.0 % |
| MTP=0 1k/512 c=1 | 40.34 | 41.38 | **+2.6 %** | 1572.3→1671.8 | +6.3 % |
| MTP=2 16k/1k c=1 | 39.91 | 44.60 | **+11.8 %** | 1608.0→1684.7 | +4.8 % |
| MTP=2 1k/512 c=1 | 66.71 | 75.65 | **+13.4 %** | 1571.7→1698.5 | +8.1 % |

That matrix also reported c=8 at parity (16k/1k c=8: 66.64 vs 66.08 out
tok/s) and no correctness flags on either side. Do not treat those deltas as
the expected cost of the new default.

## V620 check still required

Before treating the new default as validated, run a MoE decode comparison on
V620 **under contention** (many experts writing the same output rows: small
batch, top-k > 1), against the opt-out:

```bash
# default (fp32 CAS-32 loops)
unset VLLM_RDNA2_MOE_FP32_ACCUM

# previous path (packed-fp16 CAS-64)
export VLLM_RDNA2_MOE_FP32_ACCUM=0
```

Report tok/s and a correctness check (greedy or the existing MoE numeric
tests) for both. The fp32 path may be slower because each row does four
CAS-32 retry loops instead of one CAS-64. No speed number is claimed here.

## Tests

* `test_rdna2_moe_fp32_accum_defaults_on` — CPU. Unset / `1` / `true` parse
  as on; `0` / `false` parse as off.
* `test_fp32_accum_run_to_run_stable` — gfx1030. Explicit fp32 stays inside
  the fp16-noise budget across two runs.
* `test_fp32_accum_rel_l2_vs_cas` — gfx1030. fp32 and CAS agree within 5 %
  rel-L2.
* `test_fp32_accum_cudagraph_capture_stable` — gfx1030. fp32 scratch survives
  a graph capture-replay cycle.
* `test_fp32_accum_disabled_matches_explicit_cas` — gfx1030. With the module
  flag forced off, `fp32_accum=None` is byte-identical to explicit `False`.
* `test_fp32_accum_env_on_enables_fp32_scratch` — gfx1030. With the module
  flag forced on, `fp32_accum=None` stays inside the noise budget on a
  contention shape.

GPU tests live in `tests/kernels/quantization/test_rdna2_moe_w4a16.py`. The
env parser test is in `tests/test_envs.py`.
