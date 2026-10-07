# MoE epilogue accumulation mode for gfx1030 (`VLLM_RDNA2_MOE_FP32_ACCUM`)

Branch: `rdna_extras`. Applies to the fused RDNA2 MoE kernels
`moe_gptq_gemm_rdna2` (W4A16) and `moe_w4a8_gemm_rdna2` (W4A8) on gfx1030.
The knob selects how the per-expert partial sums are accumulated into the
pre-zeroed output `c`.

## The two modes

### Default OFF — legacy packed-fp16 CAS-64 (`fp32_accum=False`)

Each partial sum is accumulated with the native packed-fp16 CAS atomic
(`atomic_add_pk4_f16`) directly into the fp16 output. This is the historical
behaviour and the **production default**.

* **Faster** — measured +2.5 % (MTP=0) and **+12–13 %** (MTP=2) at c=1 vs the
  fp32 path (see the numbers below).
* **Order-dependent** — the packed-fp16 atomic add is not associative, so the
  accumulation order across concurrent expert blocks can differ between runs,
  which can flip the last 1–2 fp16 ULPs and, on close-valued cells, an argmax.
  This is the run-to-run non-determinism the fp32 mode exists to remove.

### Opt-in ON — fp32 scratch accumulator (`fp32_accum=True`)

Partial sums accumulate in an **fp32 scratch** via native `global_atomic_add_f32`
(no CAS), with a single fp32→fp16 cast at the end.

* **Run-to-run deterministic in practice** — one rounding step at the end; the
  residual fp32 atomic re-ordering is bounded to a few fp16 ULPs and does not
  flip an argmax on any plausible threshold.
* **cudagraph-safe** — the fp32 scratch is a persistent per-(rows, n, device)
  allocation allocated eagerly before graph capture; the per-call zero is a
  capture-legal `hipMemsetAsync`.
* **Costs a little at c=1** — see the numbers below; at c=8 the two modes are
  at parity.

## Knobs

| Knob | Value | Effect |
|---|---|---|
| env `VLLM_RDNA2_MOE_FP32_ACCUM` | unset / `0` | **Default:** CAS packed-fp16 epilogue. |
| env `VLLM_RDNA2_MOE_FP32_ACCUM` | `1` | fp32 scratch accumulator. |
| per-call `fp32_accum=` (in `_custom_ops.py` wrapper) | `True` / `False` | Overrides the env-resolved default for a single call. |

The wrapper reads the env var **once at import** into
`_custom_ops._RDNA2_MOE_FP32_ACCUM` (no env access in traced/captured code) and
passes the resolved `fp32_accum` per call. The per-call override wins over the
env default. Both `moe_gptq_gemm_rdna2` and `moe_w4a8_gemm_rdna2` share the
same flag.

```bash
# Production: CAS (default)
export VLLM_RDNA2_MOE_FP32_ACCUM=0        # or leave unset

# Reproducible debugging / A-B: fp32
export VLLM_RDNA2_MOE_FP32_ACCUM=1
```

## Measured trade-off

Source: `bench_results/2026-09-30_fp32-accum-matrix/SUMMARY.md` (4× Radeon PRO
V620, TP=4, FULL_AND_PIECEWISE cudagraphs, prefix caching, FA-RDNA2, frozen
TunableOp rows, Flash-Next AWQ-W4A16). `Δ` = CAS relative to fp32, **positive =
CAS faster**.

| Arm / cell | out tok/s fp32 | out tok/s CAS | Δ out | prefill fp32→CAS | Δ prefill |
|---|---:|---:|---:|---|---:|
| MTP=0 16k/1k c=1 | 30.04 | 30.78 | **+2.5 %** | 1677.9→1745.1 | +4.0 % |
| MTP=0 1k/512 c=1 | 40.34 | 41.38 | **+2.6 %** | 1572.3→1671.8 | +6.3 % |
| MTP=2 16k/1k c=1 | 39.91 | 44.60 | **+11.8 %** | 1608.0→1684.7 | +4.8 % |
| MTP=2 1k/512 c=1 | 66.71 | 75.65 | **+13.4 %** | 1571.7→1698.5 | +8.1 % |

* The fp32 path is **not free**: CAS is 2.5 % faster at MTP=0 and 12–13 %
  faster at MTP=2, all at c=1.
* At c=8 the two modes are at parity (16k/1k c=8: 66.64 vs 66.08 out tok/s).
* No correctness regression on either side — greedy outputs coherent on both
  (France→Paris, 2+2→4), no garbage flags, no PCI SERR.

## When to use which

* **CAS (default)** — production serving. Higher throughput, especially for
  MTP-2 at low concurrency. Accept the tiny run-to-run output variation.
* **fp32 (`=1`)** — reproducible debugging and A/B measurement where you need
  bitwise-stable output across runs, or a long cudagraph session where
  accumulated ULP noise must not drift into an argmax flip.

## Tests

* `test_fp32_accum_run_to_run_stable` — fp32 is run-to-run stable within the
  fp16-noise budget.
* `test_fp32_accum_rel_l2_vs_cas` — fp32 and CAS agree within 5 % rel-L2.
* `test_fp32_accum_cudagraph_capture_stable` — fp32 scratch survives a graph
  capture-replay cycle.
* `test_fp32_accum_default_off_byte_identical_to_cas` — the **default** (unset
  env) resolves to CAS and is byte-identical to an explicit `fp32_accum=False`.
* `test_fp32_accum_env_on_enables_fp32_scratch` — `=1` resolves the default to
  the fp32 scratch path (run-to-run stable on a contention shape).

All in `tests/kernels/quantization/test_rdna2_moe_w4a16.py`.
