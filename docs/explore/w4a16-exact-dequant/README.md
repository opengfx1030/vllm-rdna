# Explore: exact fp16 dequant for the RDNA2 W4A16 kernels

**Status**: explore. It is not known yet whether the bias below matters for
model quality on the V620, or what removing it costs. Nothing changes by
default: the exact form is compiled in only with
`-DVLLM_RDNA2_W4A16_EXACT_DEQUANT=1`.

## The claim

The dense W4A16 kernels, `gptq_gemm_rdna2` (decode) and
`gptq_gemm_rdna2_prefill`, dequantize with the exllama bit trick in
`csrc/rocm/qdq_4_rdna2.cuh`:

- `prep_zero_scale_fp16` stores `s·(−1024 − z)` and `s·(−64 − z)` **as
  fp16**;
- `dequant_4bit_8_fp16` computes each weight in one fp16 FMA,
  `(1024 + q)·s + z1`.

The rounding of `z1` is up to half an fp16 ulp of about `1032·s`: 0.008 at
`s = 0.02`. It is the same for every weight of a (group, column) on four of
the eight K offsets, so it adds up along K rather than averaging out.

Evidence so far:

- **JartX, gfx1100** ([`dafcde3`](https://github.com/JartX/vllm/commit/dafcde3f8bb9da96e5ca24adcf8f54c9ced413d7)):
  on the four linear shapes of a 27B G32 model at TP4, max abs error was
  0.34–1.26 baked against 0.004–0.013 exact (64–142×), "a ~3% typical
  perturbation".
- **This repo already fixed it for MoE.** f125afc ("Fix RDNA2 MoE FP16 zero
  dequantization", 2026-09-22) says "even quantized zero then becomes a
  nonzero weight". It forms `q − z` first and scales after, and adds
  `test_quantized_zero_stays_zero`. The dense kernels still bake.
- **The dense test already tolerates it.**
  `tests/kernels/quantization/test_rdna2_w4a16.py` allows 5e-2 rel-L2
  ("the exllamav2 bit-trick; allow ~3% relative noise").
- **CPU model.** `reference.py` reproduces both forms bit for bit. On that
  test's own data distribution (G=32/128, AWQ-like and symmetric zeros) the
  baked output is 2.0–2.9 % rel-L2 away from exact; the exact form is
  1.9–2.1e-4.

## Hypotheses and kill criteria

Thresholds are proposals; the owner confirms them before the run.

| # | Hypothesis | Check | Holds if | If not |
| --- | --- | --- | --- | --- |
| H1 | The production dense ops carry the bias on the V620 | default build: `check_ops run`, `test_rdna2_w4a16_quantized_zero_stays_zero` (B1) | verdict `baked` for `rdna2_decode` and `prefill`; q == z gives a nonzero output at s = 0.007 / 0.01 | stop: nothing to fix on this path |
| H2 | Removing it matters for the model | `eval_model` on both builds, two runs each (B2) | perplexity improves by more than 2× the spread between repeat runs of one build; GSM8K (all 1319) not worse | stop, unless H3 comes out free |
| H3 | The exact form costs little | `check_ops compare` and the serve matrix (B3) | geomean op time ≤ +3 % for decode and for prefill cells; serve tok/s within −2 % | try E1 below for the kernel that regressed |

## Design options

| Option | Loop cost per dword per column | Status |
| --- | --- | --- |
| **E3**: `fp16((q − z)·s)`. Subtract the exact fp16 integers `1024 + z` or `1024 + 16z`, then scale: one rounding per weight | +4 packed ops (`v_pk_add_f16` and `v_pk_mul_f16` replace `v_pk_fma_f16`) | in this PR, behind the macro |
| **E1** (JartX): `Σ a·w = y·Σ a·(1024 + q) + z·Σ a`, with `y`, `z` applied once per group in f32 | −4 packed ops (no per-weight FMA); plus `Σ a` dots shared by the thread's columns, a per-group f32 correction per (row, column), and more accumulators | follow-up if E3's cost shows in H3 |
| **MoE form** (f125afc): prep with scale 1, then `hmul2` by the scale | same as E3 | already in `moe_q_gemm_rdna2.cu` |

Compiled with clang 18 for gfx1030 (`isa_check.py`, the real headers in a
micro-kernel with `gemm_q4_kernel_rdna2`'s loop body), per iteration over
4 columns × 1 dword:

| M | Baked VALU | Exact (E3) VALU | Δ |
| ---: | ---: | ---: | ---: |
| 1 | 62 | 78 | +26 % |
| 8 | 230 | 246 | +7 % |
| 16 | 422 | 438 | +3.8 % |

The whole delta is 16 `v_pk_fma_f16` replaced by 16 `v_pk_add_f16` plus 16
`v_pk_mul_f16`; everything else is identical. Decode at M=1 was
latency-bound in the 2026-09-10 profile (220 GB/s of ~512, GPU 60–65 %
idle), so the +26 % may not show; that is what H3 measures. On gfx1100
JartX found E1 1.14–1.35× *faster* than baked at M=1, neutral at M=2–4 and
0.79–1.05× at M=8.

## Scope and gaps

- **Covered by the macro:** `gptq_gemm_rdna2`, `gptq_gemm_rdna2_prefill`
  (all configs) and `moe_q_gemm_rdna2`. MoE is already exact and computes
  the same values, with up to 2 more packed ops per dword per column on the
  high pairs.
- **Not covered:** exllama `gptq_gemm` (`csrc/libtorch_stable/quantization/gptq/qdq_4.cuh`,
  shared with other GPUs). RDNA2 uses it for GPTQ at M > 256, and at
  32 < M ≤ 256 when N ≥ 3072, so GPTQ prefill stays baked in the exact
  build; `check_ops` reports that op as a control.
- The 27B AWQ path uses only the decode and prefill ops, so it is fully
  covered.
- **Codegen.** With the macro at 0 the preprocessed code is unchanged: only
  comments and the `#define` are new. In the exact build, check VGPRs and
  spills of the RDNA2 kernels (B0).

## Run plan (V620)

Record the git SHA, torch and ROCm versions, board and power cap for every
run. Results go in `RESULTS-<date>.md` next to this file.

### B0. Two builds

```bash
# default build: baseline arm
VLLM_TARGET_DEVICE=rocm pip install -e . --no-build-isolation
# exact arm (separate venv or rebuild), or flip the #define in qdq_4_rdna2.cuh
CMAKE_ARGS="-DCMAKE_HIP_FLAGS=-DVLLM_RDNA2_W4A16_EXACT_DEQUANT=1" \
    VLLM_TARGET_DEVICE=rocm pip install -e . --no-build-isolation
```

For each build, record the VGPR count, scratch and spills of
`gemm_q4_kernel_rdna2`, `gemm_dynamic_kernel` and `moe_gemm_q4_kernel_rdna2`
from the code object metadata.

### B1. H1: which form the ops compute

```bash
M=benchmarks.kernels.w4a16_exact_dequant.check_ops
python -m pytest tests/kernels/quantization/test_rdna2_w4a16.py -q        # default
python -m $M run --json baked.json
# exact build: the tests switch to the exact expectations and a 5e-3 bound
VLLM_RDNA2_W4A16_EXACT_DEQUANT=1 python -m pytest tests/kernels/quantization/test_rdna2_w4a16.py -q
python -m $M run --json exact.json
```

`check_ops run` first checks `gptq_shuffle` against the shuffle model. It
then runs every op on M ∈ {1, 4, 8, 16, 32, 624, 2048} × (N, K) ∈
{(2560, 8704), (6144, 2560), (8704, 2560)}, for AWQ G=32 and GPTQ G=128.
Each output is compared with exact, baked and exact-fp16 references
through the same fp32 GEMM, and gets a verdict.

| Build | `rdna2_decode` verdicts | `prefill` verdicts | `exllama` verdicts | rel-L2 vs exact (median) | quantized-zero test |
| --- | --- | --- | --- | --- | --- |
| default | | | | | |
| exact | | | | | |

### B2. H2: does it matter

```bash
E=benchmarks.kernels.w4a16_exact_dequant.eval_model
python -m $E run --label baked --model /models/Qwen3.8-27B-AWQ-INT4 --tp 2 \
    --ppl-file <wikitext-2 test> --gsm8k 1319 --json eval-baked-1.json
# repeat once per build (eval-*-2.json), then:
python -m $E compare eval-baked-1.json eval-exact-1.json
```

| Run | PPL baked | PPL exact | Δ | spread (repeat) | GSM8K baked | GSM8K exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | | | | | | |
| 2 | | | | | | |

### B3. H3: what it costs

```bash
python -m $M compare baked.json exact.json   # per op and cell; geomean by phase
```

Then run the `docs/rdna2/bench_27b_awq_matrix.md` matrix (1k/512 and
16k/1k, c=1/4/8) on both builds.

| Op, phase | geomean time ratio (exact / baked) |
| --- | ---: |
| `rdna2_decode`, decode | |
| `prefill`, decode | |
| `prefill`, prefill | |

| Cell | tok/s baked | tok/s exact | Δ |
| --- | ---: | ---: | ---: |
| 1k/512 c=1 | | | |
| 1k/512 c=8 | | | |
| 16k/1k c=1 | | | |

## Agent-side checks (no GPU)

```bash
.venv/bin/python -m pytest benchmarks/kernels/w4a16_exact_dequant -q
.venv/bin/python -m benchmarks.kernels.w4a16_exact_dequant.isa_check
```

The pytest run covers:

- the bit patterns of the exact zero constants (z = 0..16);
- that the exact path rounds each weight once;
- that the baked error concentrates on the low K offsets;
- the quantized-zero predictions the GPU test relies on (0.007 fails
  baked, 2^-7 passes);
- baked vs exact output error on the dense test's distribution;
- the shuffle model against the kernel read order;
- that `check_ops` packs what the kernels read (needs torch and vLLM's
  Python).

## If it graduates

Flip the default. Make the dense test's 5e-3 bound the only bound for the
RDNA2 ops, and keep `test_rdna2_w4a16_quantized_zero_stays_zero` strict.
Then decide separately about exllama's `qdq_4.cuh`, which is shared with
other GPUs.
