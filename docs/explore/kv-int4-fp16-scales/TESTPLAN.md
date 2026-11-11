# Test plan: INT4 + FP16 block scales for the RDNA2 KV cache

**Status**: explore. Nothing has been measured. Every cell in this document is
empty on purpose, and a number only goes in a cell when it comes from a run
that is recorded in the PR body.

The format under test is specified in `README.md`. The reference codec is
`reference_codec.py`; it needs no GPU and its output is already in the README.

## What has to be true for this format to be worth building

1. **H1 - accuracy.** Block-granular FP16 scales with a packed zero-point
   must beat the shipped `int4_per_token_head` scale on *real* KV activations,
   not on synthetic rows, and it must be measured on **attention output
   error**, not on round-trip SNR. The synthetic gap is 5.73 dB on
   K-side rows and 1.78 dB on the V side; the real gap is unknown.
   SNR alone is not an acceptable metric here: the OCP E8M0 floor rule
   scores *higher* SNR than the safe round-up rule while clipping 17 of 256
   blocks, because a rare clipped element barely moves a row's RMS error
   while a clipped K value perturbs a whole attention logit row.
2. **H2 - compute path.** Dequantising to FP16 must put the dot on
   `v_dot2_f32_f16` and not on the scalar FMA path, and that must be visible
   in decode throughput.
3. **H3 - it has to be reachable.** `RdnaAttentionBackend.supported_kv_cache_dtypes`
   is `["auto", "float16"]` today, and the backend rejects every quantised KV
   dtype. Either the RDNA2 backend grows the dtype - which means reviving the
   legacy INT8 FA-RDNA2 dispatch (`3baecdb516`, see the README's *Prior work*
   section) and swapping the format inside it - or the design is re-scoped to
   the Triton path, where the dot cannot use `v_dot2_f32_f16` at all.
4. **H4 - no global scale.** If the global scale turns out to matter on real
   activations, the format needs a per-tensor side buffer and the change
   stops being a kernel-sized change.
5. **H5 - the zero-point really is free.** The packed zero-point costs about
   half a dB against a zero-point that spends its own 0.125 bits/value. If
   real activations make that gap larger than 0.5 dB, the zero-point has to
   be stored separately and the format costs 4.625 bits/value.

If H1 or H2 fails, the format is not worth a new dtype and the shipped
`int4_per_token_head` stays.

## Step 1 - capture real KV activations

The synthetic rows in the README are a guess. The capture is the part that
decides H1.

- **Where**: the RDNA2 attention kernel, after the K and V projections and
  before the cache write. `vllm/v1/attention/ops/int4_per_token_head.py`
  already has the pre-quantisation tensors in hand at the reshape kernel.
- **What**: a sample of real K and V rows, per layer, per head, with the
  layer id and head id kept, at a context long enough for the outliers to
  show up (16k tokens is the fork's standard prompt length).
- **Size**: enough rows to see the tail - a few hundred rows per layer, not a
  few. The format question is entirely about what happens in the worst
  channel of a row.
- **Storage**: raw FP16 dumps, so the reference codec can be re-run offline
  against them without a GPU.

| Capture | Model | Layers | Rows/layer | Context | File |
| --- | --- | --- | --- | --- | --- |
| K | Qwen3.8-Flash-Next | | | 16k | |
| V | Qwen3.8-Flash-Next | | | 16k | |
| K | DeepSeek-V4-Flash | | | 16k | |
| V | DeepSeek-V4-Flash | | | 16k | |

## Step 2 - offline codec comparison (no GPU)

Run the reference codec over the captured rows. This is the H1 check and it
needs no GPU, only the dumps from Step 1.

| Scheme | SNR dB (real K) | SNR dB (real V) | attn output err | bits/value |
| --- | --- | --- | --- | --- |
| `int4_per_token_head` (shipped) | | | | 4.250 |
| `int4_fp16_block` k=64, symmetric | | | | 4.250 |
| `int4_fp16_block` k=64, packed zp | | | | 4.250 |
| `int4_fp16_block` k=32, symmetric | | | | 4.500 |
| **`int4_fp16_block` k=32, packed zp (proposal)** | | | | **4.500** |
| `int4_fp16_block` k=32, asymmetric, separate zp | | | | 4.625 |
| `int4_fp16_block` k=16, packed zp | | | | 5.000 |
| `int4_e4m3_block` k=32, no global | | | | 4.250 |
| `int4_e4m3_block` k=16, no global | | | | 4.500 |
| `int4_e8m0_block` k=32, round-up | | | | 4.250 |
| `int4_e8m0_block` k=32, OCP floor | | | | 4.250 |
| FP8 e4m3, per-tensor scale | | | | 8.000 |
| `int8_e8m0_block` k=32 | | | | 8.250 |

Acceptance for H1: the `k=32` packed row beats the shipped row by at least
2 dB of attention output error on both K and V, and the packed row stays
within 0.5 dB of the row that spends its own 4 bits on the zero-point
(H5). If it does not, stop here.

Acceptance for H5: the packed row is within 0.5 dB of the separate-zero-point
row. If it is not, the format grows to 4.625 bits/value.

## Step 3 - memory budget check (no GPU)

Byte budget per token per layer, `head_size=128`:

| Scheme | Element bytes | Scale bytes | Total bytes | bits/value |
| --- | --- | --- | --- | --- |
| `int4_per_token_head` | 64 | 4 (zp in the scale) | 68 | 4.250 |
| `int4_fp16_block` k=32, symmetric | 64 | 8 | 72 | 4.500 |
| `int4_fp16_block` k=32, packed zp | 64 | 8 (zp inside) | 72 | 4.500 |
| `int4_e4m3_block` k=32 | 64 | 4 | 68 | 4.250 |
| `int8_e8m0_block` k=32 | 128 | 4 | 132 | 8.250 |

The scale bytes sit beside the data in the same page, so the page size has to
grow by 4 bytes per head per token for the proposal. That is the whole
memory argument and it is small; the argument for the format is accuracy and
the compute path, not the bytes.

## Step 4 - kernel, then decode A/B (GPU)

Only after Steps 1-3 hold does a kernel get written.

- **Kernel**: a reshape (write) kernel and a split-dot (read) kernel next to
  `vllm/v1/attention/ops/int4_per_token_head.py`. Write path: FP16 block
  scale per `k` values with the zero-point packed into the scale's low
  mantissa bits, nibble-packed elements. Read path: unpack to FP16 and dot
  with `v_dot2_f32_f16`.
- **Two kernel decisions the README derives, both to be confirmed by the
  measurement rather than assumed**:
    - apply the block scale **per element**, not per partial dot. On gfx1030
      the per-partial-dot form costs `head_size^2 / k` accumulator ops against
      `head_size` for the per-element form - 8x worse at `head_size=256,
      k=32`.
    - hoist the zero-point correction **out of the element loop**: on the QK
      side it is a rank-1 correction against the query's own block sums (the
      same trick the shipped Triton kernel uses at
      `int4_per_token_head.py:570-572`); on the PV side it rides the per-block
      probability mass that per-block value scales already require.
- **Correctness**: round-trip against the reference codec, and the existing
  per-token-head test suite extended to the new mode
  (`tests/quantization/test_per_token_kv_cache.py`).
- **Wave32**: `num_warps` <= 4 and RDNA-sized tiles, per the fork's RDNA code
  rules.
- **Launch**: current stream, per the fork's RDNA code rules.
- **Cudagraphs**: correct under `FULL_AND_PIECEWISE` and under
  `--enforce-eager`.

Serve through the in-tree launcher; do not hand-write a `vllm serve` argv:

```bash
MODEL=/path/to/checkpoint VENV=/path/to/venv \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext
```

Decode matrix, one cell per row, all four numbers per cell as the fork
requires:

| Arm | KV dtype | prefill tok/s | decode tok/s per req | TTFT (s) | ITL (ms) | 16k/1k c=1 | 16k/1k c=8 | 1k/512 c=1 | 1k/512 c=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | fp16 | | | | | | | | |
| B | `int4_per_token_head` | | | | | | | | |
| C | `int4_fp16_block` k=32, packed zp | | | | | | | | |

Acceptance for H2: arm C's decode tok/s per request beats arm B's at c=1 and
c=8, with prefill within noise and identical greedy output.

An arm D for `int4_e4m3_block` k=32 is only worth building if Step 2's
real-activation numbers put the one-byte scale within 0.5 dB of the proposal.
At the same 4.500 bits/value it is 0.32 dB behind on the K side and 0.48 dB
behind on the V side, and at 4.250 bits/value it saves 0.25 bits/value but
loses 2.14 dB on the K side, so the default answer is no.

## Step 5 - output quality

If the format changes the numbers the model produces, it needs a quality
result, not just a speed result. The fork's standard is output-token parity
against the FP16 baseline, and `tools/rdna/port_v031/ppl_probe.py` for a
perplexity number on the pruned DeepSeek-V4 checkpoint.

| Arm | Perplexity / probe | Parity vs fp16 | Notes |
| --- | --- | --- | --- |
| A (fp16) | | - | baseline |
| B (`int4_per_token_head`) | | | |
| C (`int4_fp16_block` k=32, packed zp) | | | |
| D (`fp8`, per-tensor e4m3) | | | shipped today; the 8-bit reference point |

Arm D is not optional. FP8 KV is the only KV cache dtype with a V620
measurement (about 2x the memory saved, +0.1 PPL), and it is the reference
that says what a dB of SNR is worth in perplexity. With arms D and C the
capture has two points on the SNR-to-PPL curve instead of one, which is the
only way to tell whether the 12.67 dB between FP8 and the proposal is worth
the 1.78x memory.

Without arm D the only measured KV data point in the fork sits at 32.968 dB
synthetic SNR and +0.1 PPL, and the proposal's 20.302 dB has to be
extrapolated from it with no second point. That extrapolation is not
available, because the table's own FP8 row shows the SNR ordering and the
PPL ordering do not have to agree.

## Records

Nothing goes in a cell above without a run log next to it. When a cell is
filled, the PR body gets the command, the box, the model, the context length
and the raw output.

| Run | Date | Box | Recipe | Log |
| --- | --- | --- | --- | --- |
| | | | | |
