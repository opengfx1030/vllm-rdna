# MoE epilogue accumulation modes for gfx1030 (`VLLM_RDNA2_MOE_FP32_ACCUM`, `VLLM_RDNA2_MOE_DETERMINISTIC`)

Branch: `rdna_extras`. Applies to the fused RDNA2 MoE kernels
`moe_gptq_gemm_rdna2` (W4A16) and `moe_w4a8_gemm_rdna2` (W4A8) on gfx1030.
The knobs select how the per-expert partial sums are accumulated into the
pre-zeroed output `c`.

## The modes

### Default OFF — legacy packed-fp16 CAS-64 (`fp32_accum=False`)

Each partial sum is accumulated with a packed-fp16 atomic add emulated by one
64-bit CAS retry loop per row (`atomic_add_pk4_f16`; gfx1030 has no native
packed-fp16 atomic) directly into the fp16 output. This is the historical
behaviour and the **production default**.

* **Faster** — measured +2.5 % (MTP=0) and **+12–13 %** (MTP=2) at c=1 vs the
  fp32 path (see the numbers below).
* **Order-dependent** — the packed-fp16 atomic add is not associative, so the
  accumulation order across concurrent expert blocks can differ between runs.
  On Flash-Next TP4 this is the first op of a 2048-token prefill whose output
  differs between two identical requests: 2.0–2.3 % of the layer-0 MoE output
  elements differ, most by 1–2 fp16 ULPs
  ([determinism-2026-11-04.md](determinism-2026-11-04.md)). The fp32 mode
  below does not remove this; `VLLM_RDNA2_MOE_DETERMINISTIC=1` does.

### Opt-in ON — fp32 scratch accumulator (`fp32_accum=True`)

Partial sums accumulate in an **fp32 scratch** via `atomicAdd`, with a single
fp32→fp16 cast at the end. gfx1030 has no native fp32 global atomic add: the
built library compiles this to a load + `v_add_f32` +
`global_atomic_cmpswap` retry loop per column, four loops per row, against one
CAS-64 loop per row in the fp16 mode (read from the disassembly of the
gfx1030 code object, 2026-11-03). The extra retry loops are on top of the
scratch zero and cast passes and are consistent with the throughput cost
measured below; the split between them was not isolated.

* **Not run-to-run deterministic** — the fp32 adds still happen in arrival
  order. The single cast at the end hides most of the re-ordering, not all of
  it. On a Flash-Next-sized layer, 0.001 % of the outputs differ between
  two runs at 2048 tokens (in every run) and 0.2–0.3 % at 1–3 tokens
  ([determinism-2026-11-04.md](determinism-2026-11-04.md)). The model
  amplifies a single ULP in a single element into logprob moves of 10 nats,
  so this is enough to move single logprobs by as much as CAS does.
* **cudagraph-safe, bounded memory** — each call takes its scratch from the
  caching allocator and releases it on return, so eager calls keep nothing,
  whatever their token count. Under graph capture the allocator serves it from
  the graph's private pool, which keeps the address the graph baked in. The
  per-call zero is a capture-legal `hipMemsetAsync`. (Before 2026-11-04 the
  scratch was cached per (rows, n, device) and never freed: every new token
  count added a rows × n × 4-byte buffer.)
* **Costs a little at c=1** — see the numbers below; at c=8 the two modes are
  at parity.

### Opt-in — run-to-run deterministic (`VLLM_RDNA2_MOE_DETERMINISTIC=1`)

Set by the compressed-tensors RDNA2 MoE method (`_rdna2_fused_moe`), not by
the kernels. It removes the order dependence instead of making it smaller:

* **W4A8** steps run w2 with `output_topk = 0`. Each (token, expert) row is
  stored once, with no atomics, and `moe_sum` adds a token's local rows in
  fp32 in top-k order. This needs the single-K-block w1
  (`VLLM_RDNA_W4A8_KBLOCK=-1`, as in `flashnext-mtp2`). With any other
  `KBLOCK` the step runs W4A16 instead.
* **W4A16** steps accumulate w1 and w2 in Q32 fixed point: `int64` atomic adds
  into the expert-cache epilogue, which are associative.
* **Bit-identical repeats:** on Flash-Next TP4 every hooked op of a prefill
  step matches across repeats, and 32 of 32 repeated prompt-logprob requests
  return identical results with CUDA graphs.
* **Cost** on `flashnext-mtp2`: +0.8 % ITL at c=1 and +0.4–1.3 % at c=8.
  Prefill is unchanged within noise. Per MoE call it is +1.7 % at 2048 tokens
  and +38–52 % at 1–3 tokens (see the
  [determinism report](determinism-2026-11-04.md#the-knob-and-what-it-costs)).
* **Not covered:** the modular `RDNA2W4A16MoEExperts` path
  (`rdna2_w4a16_moe.py`).

## Knobs

| Knob | Value | Effect |
|---|---|---|
| env `VLLM_RDNA2_MOE_FP32_ACCUM` | unset / `0` | **Default:** CAS packed-fp16 epilogue. |
| env `VLLM_RDNA2_MOE_FP32_ACCUM` | `1` | fp32 scratch accumulator (still order-dependent). |
| env `VLLM_RDNA2_MOE_DETERMINISTIC` | `1` | Order-independent epilogue (above); takes precedence over the accumulation mode for routed experts. |
| per-call `fp32_accum=` (in `_custom_ops.py` wrapper) | `True` / `False` | Overrides the env-resolved default for a single call. |

The wrapper reads the env var **once at import** into
`_custom_ops._RDNA2_MOE_FP32_ACCUM` (no env access in traced/captured code) and
passes the resolved `fp32_accum` per call. The per-call override wins over the
env default. Both `moe_gptq_gemm_rdna2` and `moe_w4a8_gemm_rdna2` share the
same flag.

```bash
# Production: CAS (default)
export VLLM_RDNA2_MOE_FP32_ACCUM=0        # or leave unset

# Lower rounding error, still order-dependent: fp32
export VLLM_RDNA2_MOE_FP32_ACCUM=1

# Bit-identical repeats (prompt scoring, A/B of numerics)
export VLLM_RDNA2_MOE_DETERMINISTIC=1
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
  MTP-2 at low concurrency. Identical requests do not return identical
  logprobs. On Flash-Next TP4 the mean move is 0.1 nats per position, and
  single positions move by up to 19 nats.
* **fp32 (`VLLM_RDNA2_MOE_FP32_ACCUM=1`)** — lower rounding error than CAS at
  12–13 % MTP-2 c=1 throughput. It is **not** bitwise-stable across runs; do
  not use it for reproducibility.
* **Deterministic (`VLLM_RDNA2_MOE_DETERMINISTIC=1`)** — bit-identical
  repeats, e.g. for prompt scoring, quality A/Bs that compare single
  positions, and regression baselines. It costs about 1 % ITL on
  `flashnext-mtp2`.

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

These are in `tests/kernels/quantization/test_rdna2_moe_w4a16.py`.
`tests/kernels/quantization/test_rdna2_moe_fp32_scratch.py` checks, for both
ops, that eager calls at many token counts keep no device memory, and that
graphs captured at two sizes in one memory pool replay correctly after eager
calls at other sizes. The
deterministic mode is covered by
`tests/kernels/quantization/test_rdna2_moe_deterministic.py`. It checks:

* bit-identical W4A16 repeats at 1–600 tokens;
* bit-identical W4A8 repeats at 1024 tokens, equal to an explicit k-ordered
  fp32 sum of the kernel's own rows.

Both run with and without expert parallelism. It also checks that the split-K
W4A8 fallback matches the W4A16 deterministic path, and the env/`KBLOCK`
gating.
