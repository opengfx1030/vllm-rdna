# Test plan: INT8 KV cache on the RDNA2 backend

**Status**: explore. Nothing has been measured. Every V620 cell in this
document is empty on purpose, and a number only goes in a cell when it comes
from a run that is recorded in the PR body.

The design under test is specified in `README.md`. The two things this plan
separates deliberately:

- the **wiring** (can the RDNA2 backend serve the `int8_per_token_head` dtype
  that the Triton backend already serves, correctly?), and
- the **read path** (what is the cheapest way to read an INT8 cache on
  gfx1030?), and
- the **format** (is the shipped per-token-head scale the right scale?).

The order matters: the wiring has to be correct before the read path can be
measured, and the read path has to be measured before a new format can be
judged against it.

## What has to be true

1. **H1 - the writer and the reader agree.** The legacy path's scales were
   written to the wrong address (`csrc/rocm/fa_rdna2.cu:4681`, `:4683`) and
   read back with a view that reinterpreted the slot dimension as bytes. Until
   a writer-to-reader test exists, every number this path has ever produced is
   uninterpretable. This is the first step and it is not optional.
2. **H2 - the read path is faster than the shipped one.** The fork's INT8
   decode branch is a per-element scalar dequantise into an FP16 shared tile,
   which the ISA study measured in the 0.65x-0.94x band of the FP16 ceiling
   (v0 0.654x at D=128, v1 0.877x at D=256). The study's own implementation
   choice (`v3_128`: packed dword load, VGPR dequant, `fdot2`, no FP16 tile)
   is the target.
3. **H3 - a native INT8 PV product does not cost accuracy.** The PV update is
   scalar in every variant the ISA study measured; quantising P to int8 and
   using `v_dot4c_i32_i8` is the one open native-dot item. It trades
   probability precision for instructions.
4. **H4 - the FP16 block scale buys accuracy at no memory cost.** At `k=64`
   the block scale costs exactly what the shipped format costs (8.250
   bits/value) and the synthetic gap is +3.01 dB on the K side. The real gap
   is unknown.
5. **H5 - the INT8 zero-point is not worth 0.25 bits/value.** The INT4 packed
   zero-point trick does not port to INT8 (an 8-bit zero-point does not fit in
   an FP16 mantissa next to a usable scale), so the format has to decide
   whether to spend its own bytes on an asymmetric quantiser.

If H1 fails, nothing else in this document matters. If H2 fails, the INT8
dtype stays on the Triton backend and this PR becomes a wiring note.

## Step 1 - writer-to-reader correctness (no model needed)

The cheapest step, and the one that has never been done.

- **What**: write K/V through `reshape_and_cache_int8_rdna2`, read the FP32
  scale back out of the packed cache with the same view the kernel will use,
  and assert it equals the per-(token, head) `amax/127` the writer computed.
- **Then**: assert the attention output against an FP16 reference for the
  same inputs, at the tolerance the Triton path already meets (see
  `tests/quantization/test_per_token_kv_cache.py` for the reference
  tolerance).
- **Where**: `tests/quantization/test_per_token_kv_cache.py` is the existing
  home for per-token-head KV tests; the FA-RDNA2-specific writer needs its
  own case there or in a new file next to it. Kernel perf work does not go in
  `tests/` (`benchmarks/kernels/` is where that belongs).

| Check | Model | Shape | Result |
| --- | --- | --- | --- |
| scale round-trip (K) | n/a | D=128, bs=16 | |
| scale round-trip (V) | n/a | D=128, bs=16 | |
| scale round-trip (K) | n/a | D=256, bs=16 | |
| scale round-trip (V) | n/a | D=256, bs=16 | |
| attention output vs FP16 | n/a | D=128, bs=16 | |
| attention output vs FP16 | n/a | D=256, bs=16 | |

## Step 2 - the read path (V620)

Two variants against the same baseline. Both are the *same* INT8 format; only
the read path differs, which is the point.

- **A (shipped today)**: per-element dequantise into an FP16 shared tile.
- **B (`v3_128`)**: packed dword load, dequantise in VGPRs, `fdot2`, no FP16
  shared tile.

Baselines: the FP16 path with the identical kernel structure, and the Triton
INT8 path end to end.

| Variant | Model | Head dim | Context | c=1 tok/s | vs FP16 | vs A |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 (baseline) | | 128 | 4k | | 1.00x | |
| A (shipped INT8 pattern) | | 128 | 4k | | | 1.00x |
| B (v3_128 pattern) | | 128 | 4k | | | |
| FP16 (baseline) | | 128 | 16k | | 1.00x | |
| A (shipped INT8 pattern) | | 128 | 16k | | | 1.00x |
| B (v3_128 pattern) | | 128 | 16k | | | |
| FP16 (baseline) | | 256 | 16k | | 1.00x | |
| A (shipped INT8 pattern) | | 256 | 16k | | | 1.00x |
| B (v3_128 pattern) | | 256 | 16k | | | |

The ISA study's own caveat applies to these cells: it was single-thread and
compute-bound, so its ratios are ISA efficiency, not memory saving, and it
expected the production ratio to land around 1.1x rather than 2.2x.

## Step 3 - native INT8 PV (V620)

Only if step 2 lands and the INT8 path is worth more than one more iteration.

- **What**: quantise P to int8 per block against the running softmax maximum,
  fold the per-(token, head) V scale into P normalised by the block maximum,
  accumulate the PV product in int32 with `v_dot4c_i32_i8`.
- **Correctness**: attention output error against the FP16 reference, against
  the FP16-P INT8 path, and against the FP16 path.

| Variant | Model | Head dim | Context | tok/s | attn output err (vs FP16) |
| --- | --- | ---: | ---: | ---: | ---: |
| FP16-P (step 2 winner) | | 128 | 16k | | |
| int8-P native PV | | 128 | 16k | | |
| FP16-P (step 2 winner) | | 256 | 16k | | |
| int8-P native PV | | 256 | 16k | | |

## Step 4 - the format (no GPU for the codec, V620 for the acceptance)

The synthetic numbers are in `README.md`; they come from PR #134's
`reference_codec.py`. What they do not settle is the acceptance criterion.

- **Capture**: real K/V activations from a V620 decode run, per layer, per
  head, at 16k tokens. Same capture procedure as PR #134's test plan.
- **Metric**: **attention output error**, not round-trip SNR. The synthetic
  rows say the FP16 block scale at `k=64` is worth +3.01 dB on the K side over
  the shipped scale at the same 8.250 bits/value; the real number is
  unknown, and SNR is not an acceptable substitute (a clipped K value perturbs
  a whole attention logit row, which a row-RMS metric barely sees).
- **Zero-point (H5)**: extend the codec with an asymmetric INT8 block variant
  and compare. This needs no GPU and no capture, and it decides the format
  before any kernel work.

| Scheme | bits/value | real K err | real V err | attn output err | PPL delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `int8_per_token_head` (shipped) | 8.250 | | | | |
| `int8_fp16_block` k=128 | 8.125 | | | | |
| `int8_fp16_block` k=64 | 8.250 | | | | |
| `int8_fp16_block` k=32 | 8.500 | | | | |
| `int8_fp16_block` k=64, asymmetric | 8.500 | | | | |

## Step 5 - serving checks (V620)

The end-to-end cells that decide whether the dtype is usable at all.

| Check | Model | Context | Result |
| --- | --- | ---: | --- |
| NIAH | | 16k/1k | |
| NIAH | | 32k/1k | |
| PPL vs FP16 baseline | | 16k | |
| PPL vs FP16 baseline | | 32k | |
| KV capacity at a fixed budget | | | |
| output-token parity vs the Triton INT8 path | | 16k | |

The parity row is the one that matters most: the Triton path is the reference
implementation of this dtype, and the RDNA2 path has to agree with it, not
merely with itself.

## Commands

Nothing below has been run for this PR. They are the commands the
implementation steps will use.

```bash
# correctness (step 1)
.venv/bin/python -m pytest tests/quantization/test_per_token_kv_cache.py -v

# serving checks (step 5)
python -m vllm.entrypoints.cli.main serve <model> \
  --kv-cache-dtype int8_per_token_head --kv-cache-dtype <variant>

# throughput sweep (step 2 / 3)
python -m vllm.entrypoints.cli.main bench throughput \
  --kv-cache-dtype <variant> --input-len 16384 --output-len 128 \
  --num-prompts 8 --concurrency 1
```

## Reproducing the synthetic numbers

See the *Reproducing the numbers* section of `README.md`. The codec lives in
PR #134's branch; the snippet in the README reproduces every INT8 row in this
document and checks itself against PR #134's published table.
