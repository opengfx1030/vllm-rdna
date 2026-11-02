# MoE epilogue accumulation mode for gfx1030 (`VLLM_RDNA2_MOE_FP32_ACCUM`)

Branch: `rdna_extras`. Applies to the fused RDNA2 MoE kernels
`moe_gptq_gemm_rdna2` (W4A16) and `moe_w4a8_gemm_rdna2` (W4A8).
The knob selects how the per-expert partial sums are accumulated into the
output `c`.

## The two modes

### Default ON — fp32 scratch accumulator (`fp32_accum=True`)

Partial sums accumulate in an **fp32 scratch** via HIP `atomicAdd(float*)`,
then a single fp32→fp16 cast. The source does not call
`v_global_atomic_add_f32`.

* **gfx1030 lowers that atomic to a compare-and-swap loop.** RDNA2 has no
  native global f32 atomic add. `v_global_atomic_add_f32` exists on gfx11;
  a gfx1100 compile of the same helper may emit it. The device body of
  these kernels is instantiated for gfx1030 (the gfx1100 image is the stub).
* **One rounding step at the end.** There is no intermediate fp16 rounding.
  Residual fp32 atomic reordering stays inside a few fp16 ULPs and does not
  flip an argmax on any plausible threshold (run-to-run stable in practice).
* **cudagraph-safe** — the fp32 scratch is a persistent per-(rows, n, device)
  allocation allocated eagerly before graph capture; the per-call zero is a
  capture-legal `hipMemsetAsync`.
* **May be slower than packed-fp16 CAS-64 on gfx1030**, because the fp32
  atomic is itself a CAS loop. See the 2026-09-30 V620 numbers below.

### Opt-out OFF — packed-fp16 CAS-64 (`fp32_accum=False`)

Set `VLLM_RDNA2_MOE_FP32_ACCUM=0`. Each partial is accumulated with
`atomic_add_pk4_f16` (a `global_atomic_cmpswap_b64` retry loop; gfx1030 has
no native `v_global_atomic_pk_add_f16`) directly into the pre-zeroed fp16
output.

* **Faster on the 2026-09-30 V620 matrix** — +2.5 % (MTP=0) and **+12–13 %**
  (MTP=2) at c=1 versus the fp32 path (see the numbers below).
* **Order-dependent** — the packed-fp16 atomic add is not associative, so the
  accumulation order across concurrent expert blocks can differ between runs,
  which can flip the last 1–2 fp16 ULPs and, on close-valued cells, an argmax.

## Knobs

| Knob | Value | Effect |
|---|---|---|
| env `VLLM_RDNA2_MOE_FP32_ACCUM` | unset / `1` | **Default:** fp32 scratch accumulator. |
| env `VLLM_RDNA2_MOE_FP32_ACCUM` | `0` | Packed-fp16 CAS-64 epilogue. |
| per-call `fp32_accum=` (in `_custom_ops.py` wrapper) | `True` / `False` | Overrides the env-resolved default for a single call. |

The pybind schemas for `moe_gptq_gemm_rdna2` and `moe_w4a8_gemm_rdna2` also
default `fp32_accum=True`. Serving goes through the Python wrapper, which
reads the env var **once at import** into `_custom_ops._RDNA2_MOE_FP32_ACCUM`
(no env access in traced/captured code) and passes the resolved `fp32_accum`
per call. The per-call override wins over the env default. Both kernels share
the same flag. Recipes and `tools/rdna/serve_rdna.sh` do not set it, so a
launch follows this default. Revert a launch with `VLLM_RDNA2_MOE_FP32_ACCUM=0`.

```bash
# Default: fp32 scratch (leave unset, or set explicitly)
export VLLM_RDNA2_MOE_FP32_ACCUM=1

# Opt out: packed-fp16 CAS-64
export VLLM_RDNA2_MOE_FP32_ACCUM=0
```

## Measured trade-off

Source: `bench_results/2026-09-30_fp32-accum-matrix/SUMMARY.md` (4× Radeon PRO
V620, TP=4, FULL_AND_PIECEWISE cudagraphs, prefix caching, FA-RDNA2, frozen
TunableOp rows, Flash-Next AWQ-W4A16). `Δ` = CAS relative to fp32, **positive =
CAS faster**. These cells predate the default flip; they are the cost of
leaving fp32 on.

| Arm / cell | out tok/s fp32 | out tok/s CAS | Δ out | prefill fp32→CAS | Δ prefill |
|---|---:|---:|---:|---|---:|
| MTP=0 16k/1k c=1 | 30.04 | 30.78 | **+2.5 %** | 1677.9→1745.1 | +4.0 % |
| MTP=0 1k/512 c=1 | 40.34 | 41.38 | **+2.6 %** | 1572.3→1671.8 | +6.3 % |
| MTP=2 16k/1k c=1 | 39.91 | 44.60 | **+11.8 %** | 1608.0→1684.7 | +4.8 % |
| MTP=2 1k/512 c=1 | 66.71 | 75.65 | **+13.4 %** | 1571.7→1698.5 | +8.1 % |

* CAS was 2.5 % faster at MTP=0 and 12–13 % faster at MTP=2, all at c=1.
* At c=8 the two modes were at parity (16k/1k c=8: 66.64 vs 66.08 out tok/s).
* No correctness regression on either side — greedy outputs coherent on both
  (France→Paris, 2+2→4), no garbage flags, no PCI SERR.
* Re-measure decode tok/s and worst-case ITL on V620 before treating those
  deltas as the current tree. Revert with `VLLM_RDNA2_MOE_FP32_ACCUM=0`.

## When to use which

* **fp32 (default)** — the production epilogue. One fp16 rounding at the end,
  in exchange for the CAS-loop cost on gfx1030.
* **CAS (`=0`)** — opt out when that throughput matters more than the last
  fp16 ULPs, especially MTP-2 at low concurrency.

## Tests

* `test_rdna2_moe_fp32_accum_defaults_on` (`tests/test_envs.py`) — unset env
  resolves to fp32; `=0` resolves to CAS. No GPU.
* `test_fp32_accum_run_to_run_stable` — fp32 is run-to-run stable within the
  fp16-noise budget.
* `test_fp32_accum_rel_l2_vs_cas` — fp32 and CAS agree within 5 % rel-L2.
* `test_fp32_accum_cudagraph_capture_stable` — fp32 scratch survives a graph
  capture-replay cycle.
* `test_fp32_accum_flag_off_byte_identical_to_cas` — with the module flag
  forced off, `fp32_accum=None` is byte-identical to explicit CAS.
* `test_fp32_accum_env_on_enables_fp32_scratch` — the module flag on (the
  unset default) resolves `fp32_accum=None` to the fp32 scratch path
  (run-to-run stable on a contention shape).

GPU tests are in `tests/kernels/quantization/test_rdna2_moe_w4a16.py`.
