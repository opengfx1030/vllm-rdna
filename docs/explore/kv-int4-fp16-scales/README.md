# Explore: INT4 + FP16 block scales for the RDNA2 KV cache

**Status**: explore. This is a format contract plus a reference codec, not an
implementation. No dtype is added, no backend changes, nothing changes by
default. The kernels, the backend hook and the measurement on the V620 are all
still to be written.

The question this document answers: *if the fork wants its own 4-bit KV cache
on gfx1030, what should the format be, and what does it cost?* The short
version: the scale format the fork already ships has the wrong granularity and
the wrong arithmetic path; the zero-point can be carried inside the scale for
free; a one-byte scale (E8M0 or E4M3) is not the accuracy optimum at the
memory budget that matters; and the second (global) level NVFP4 carries buys
nothing for activation data, so it can be dropped.

## The question

`--kv-cache-dtype int4_per_token_head` already exists in this fork
(`vllm/config/cache.py:53`). It is the only 4-bit KV cache dtype here, and on
gfx1030 it is the wrong shape for two independent reasons.

**Its granularity is a whole head row.** The scale is `(max - min) / 15` over
all `head_size` values of a row, with a 4-bit zero-point hidden in the low
mantissa bits of the FP32 scale (`vllm/v1/attention/ops/int4_per_token_head.py:158`,
write side; `:550`, read side). One channel with a large magnitude therefore
sets the step size for every other channel in the row. The reference codec
measures the cost of that on synthetic rows with two channels scaled 40x:
**14.570 dB**, against **20.035 dB** for the same rows without outliers.

**Its dequant lands on the wrong hardware path.** The cached values are
unpacked to FP32 and multiplied against an FP32 query
(`vllm/v1/attention/ops/int4_per_token_head.py:440`, `:527`, and the dot at `:571`). gfx1030 has no
FP32 matrix instruction, so an FP32 `tl.dot` cannot use the packed path. The
established RDNA2 inner product in this fork is `v_dot2_f32_f16` with FP32
accumulate (`docs/rdna/monokernel/README.md:104`). An FP32 dot is the
scalar-`v_fma_f32` fallback, which that document describes as the path kept
only for deployments this fork does not target.

## The proposal

A new KV dtype, `int4_fp16_block`, asymmetric, with the zero-point carried
inside the scale:

| Field | Value |
| --- | --- |
| Element | INT4, unsigned, `q` in `[0, 15]` |
| Packing | 2 elements per byte, low nibble first (same as `int4_per_token_head`) |
| Block scale | FP16 (E5M10), one per `k` values along the head dimension |
| Zero-point | 4 bits, packed into the low 4 bits of the scale's mantissa |
| Global scale | none (see *Two levels* below) |
| Bits/value | `4 + 16/k`; at `k=32` that is **4.500** |
| Dequant | `x_hat = (q - zp) * scale`, in FP16 |

The scale cache would be `[num_blocks, block_size, num_kv_heads, head_size/k]`
FP16, mirroring the shape of the existing
`[num_blocks, block_size, num_kv_heads]` FP32 scale cache
(`vllm/v1/attention/backends/triton_attn.py:405`, `_ensure_scale_caches`).

`k=32` is the proposal's baseline because it is the granularity KIVI measured
as best for the value tensor, because it costs exactly the same 4.5 bits/value
as NVFP4, and because it is the block size the MX family standardised on.

### The zero-point is free

The shipped format already hides a 4-bit zero-point in the low mantissa bits of
its FP32 scale. The same trick works with an FP16 scale, and it is worth more
than it costs:

- An FP16 scale has 10 mantissa bits. Stealing the low 4 leaves 6 bits of
  mantissa for the scale itself, so the scale carries up to 1.5 % relative
  error. That is well below the INT4 quantisation error and costs about half a
  dB.
- In exchange the block gets **16 codes instead of 8**, because the
  quantiser is asymmetric: `q = round(x / scale) + zp` spans the block's
  `[min, max]` rather than `[-absmax, +absmax]`.

The reference codec measures both halves. At `k=32`:

| Variant | SNR dB (K-side) | SNR dB (no outliers) | bits/value |
| --- | --- | --- | --- |
| symmetric FP16 scale, no zero-point | 18.326 | 20.341 | 4.500 |
| asymmetric, zero-point in its own 4 bits | 20.341 | 21.867 | 4.625 |
| **asymmetric, zero-point packed in the scale** | **20.302** | **21.815** | **4.500** |

The packed variant gives up 0.04 dB against the version that spends its own
0.125 bits/value on the zero-point, and gives up nothing else. **The
zero-point is free**, and the block gets its second level of range for it.

The scale is snapped *up* onto the 6-bit-mantissa grid before use, so the
block's range always fits in 16 codes and nothing saturates: the reference
codec counts 0 clipped elements in 512 blocks.

### Two levels

NVFP4 is a two-level format: an E4M3 scale per 16 elements, times an FP32
scale per tensor. The global is folded into the stored block scale at write
time and multiplied back at read time
(`csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`,
`cvt_warp_fp16_to_fp4` in `csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh`).

That second level is not free of consequences. E4M3 is a narrow format: its
normal range runs from `2^-6` to 448, a ratio of about 2.9e4, so a block scale
has to fit inside that once the global has been divided out. FP16's normal
range runs from `2^-14` to 65504, a ratio of about 1.1e9 - roughly 37000x
wider.

The reference codec measures what the second level is worth. The global is
modelled at its most favourable value, the tensor maximum:

| | FP16 block scale | E4M3 block scale |
| --- | --- | --- |
| SNR, no global (K-side) | 18.326 dB | 19.984 dB |
| SNR, with a global (K-side) | 18.326 dB | 20.043 dB |
| SNR, no global (no outliers) | 20.341 dB | 21.331 dB |
| SNR, with a global (no outliers) | 20.341 dB | 21.369 dB |
| Blocks whose stored scale went subnormal, of 512 (with a global) | 0 | 15 |

So for an FP16 block scale the global scale **buys nothing**: the SNR is the
same to three decimals with and without it, and no block scale underflows.
For an E4M3 block scale it buys 0.06 dB, and it is what pushes 15 of the 512
block scales into E4M3's subnormal range, where they lose mantissa bits.

The reason NVFP4 carries a global at all is that its global is a *checkpoint*
constant (`layer._k_scale`), not the tensor maximum. Dividing by a calibration
constant that is far from the tensor maximum is exactly the case where a
narrow E4M3 block scale underflows, and where the second level earns its
keep. A KV cache has no checkpoint scale. Its rows are activations, and the
natural global - a running maximum - would invalidate every block already
written, because the blocks were divided by the old value.

That is the concrete reason this proposal drops the second level: not because
two-level scaling is wrong, but because for activation data it has nothing to
calibrate against, and a single FP16 block scale has the range to not need it.

## What the reference codec says

`reference_codec.py` in this directory implements all of the above in pure
Python and prints the table below. The numbers are arithmetic on synthetic
rows - 64 rows of `N(0, 0.5)`, `head_size=128`. **They are not V620
measurements.**

The harness runs every scheme against two outlier modes, because the two
tensors behave differently and the difference decides the design:

- **K-side** - two *channels* of every row are scaled 40x. This is KIVI's
  observation about K: a few channels carry much larger values, consistently
  across tokens.
- **V-side** - two whole *tokens* are scaled 40x. This is KIVI's observation
  about V: whole tokens occasionally carry much larger values.

| Scheme | K-side dB | V-side dB | bits/value |
| --- | --- | --- | --- |
| `int4_per_token_head` (shipped) | 14.570 | 20.035 | 4.250 |
| `int4_fp16_block` k=64, symmetric | 15.475 | 19.495 | 4.250 |
| **`int4_fp16_block` k=64, packed zp** | **17.429** | **20.882** | **4.250** |
| `int4_e8m0_block` k=32 | 16.938 | 16.934 | 4.250 |
| `int4_e4m3_block` k=32 | 18.158 | 20.273 | 4.250 |
| `int4_fp16_block` k=32, symmetric | 18.326 | 20.341 | 4.500 |
| `int4_e8m0_block` k=16 | 18.281 | 17.800 | 4.500 |
| `int4_e4m3_block` k=16 | 19.984 | 21.331 | 4.500 |
| **`int4_fp16_block` k=32, packed zp (proposal)** | **20.302** | **21.814** | **4.500** |
| `int4_fp16_block` k=32, token-axis groups | **21.778** | 4.829 | 4.500 |
| `int4_fp16_block` k=32, asymmetric, separate zp | 20.341 | 21.867 | 4.625 |
| `int4_fp32_block` k=32, packed zp | 20.342 | 21.867 | 5.000 |
| NVFP4 (E2M1 + E4M3/16 + global) | 21.724 | 20.334 | 4.500 |
| `int4_fp16_block` k=16, symmetric | 20.322 | 21.442 | 5.000 |
| **`int4_fp16_block` k=16, packed zp** | **22.694** | **23.237** | **5.000** |
| FP8 e4m3, per-tensor scale (shipped, measured) | 32.968 | 31.569 | 8.000 |
| `int8_e8m0_block` k=32 (MXINT8) | 36.587 | 41.913 | 8.250 |
| `int8_e4m3_block` k=32 | 35.952 | 35.304 | 8.250 |
| `int8_fp16_block` k=32 | 39.475 | 45.594 | 8.500 |
| `int8_fp32_block` k=32 | 39.476 | 45.593 | 9.000 |

Reading it:

- **The proposal beats the shipped format by 5.73 dB on the K side** and
  1.78 dB on the V side, for 5.9 % more memory (4.500 vs 4.250 bits/value).
  It is a replacement candidate for `int4_per_token_head` on RDNA2, not an
  additional dtype. The shipped format's zero-point is hidden in the low
  mantissa bits of its FP32 scale, so it costs no extra bytes and its true
  cost is 4.250.
- **The shipped format is 5.5 dB weaker on K than on V**, and that is exactly
  the channel-outlier failure: one row-level scale, one loud channel. The
  proposal is nearly symmetric (20.302 vs 21.814), which is what a block
  scale is supposed to buy.
- **At the same 4.500 bits/value, the proposal beats every one-byte scale.**
  Against `int4_e4m3_block` k=16 it is +0.32 dB (K) and +0.48 dB (V);
  against `int4_e8m0_block` k=16, +2.02 dB and +4.01 dB.
- **`k=64` with a packed zero-point costs exactly what ships today**
  (4.250 bits/value) and buys +2.86 dB on the K side and +0.85 dB on the V
  side over the shipped format.
- **NVFP4's own element format beats the proposal on the K side** (21.724 vs
  20.302 dB at the same 4.5 bits/value): E2M1's non-uniform levels - 0.5, 1,
  1.5, 2, 3, 4, 6 - spend their precision near zero, which is where a
  zero-centred activation distribution lives. On the V side the ordering
  reverses and the proposal is ahead of NVFP4 (21.814 vs 20.334 dB). Whether
  the element type should be INT4 or E2M1 is therefore an open question, not
  a settled one.
- **Below 8 bits/value the proposal is the best format in the table**, and
  above it FP8 already wins - see *Why INT8 is not the answer at this budget*.

### Grouping: the axis matters more than the scale format

There are two axes a scale can be shared along, and the table above measures
both:

- **Head-dimension grouping** (what the proposal uses): one scale per `k`
  contiguous values inside one token's row. The scale count per token per head
  is `head_size / k`.
- **Token-axis grouping** (KIVI's axis): one scale per `k` tokens inside one
  channel. The scale count per token per head is also `head_size / k`.

**They cost the same memory.** What differs is which outliers they isolate,
and the crossover is sharp:

| Grouping | K-side dB | V-side dB |
| --- | --- | --- |
| head-dimension, `k=32`, packed zp | 20.302 | 21.814 |
| token-axis, `k=32` tokens | **21.778** | 4.829 |

Token-axis grouping is **1.48 dB better on K** and **catastrophic on V**. The
reason is mechanical: a token outlier scales a whole row by 40x, and in
token-axis grouping that one token's values set the scale shared with the
other 31 tokens in its group, so they are crushed to zero. Head-dimension
grouping gives every token its own scale and is immune.

KIVI's observation therefore maps onto the measurement exactly: **K wants the
token axis, V wants the head-dimension axis**, and the same scale layout
cannot serve both.

The proposal keeps head-dimension grouping for both anyway, for a reason that
is not about accuracy: **a token-axis scale spans `k` tokens, so it cannot be
finalised until all `k` of them have been written.** A paged KV cache with
chunked prefill can land a chunk boundary inside a page, so the scale would
have to be recomputed after the fact and the already-quantised data
re-quantised - and the original FP16 values are gone by then. Head-dimension
grouping has no such problem: a token's row is complete the moment it is
written. That is why every per-token-head format upstream is
head-dimension granular, and it is why 1.48 dB is left on the table.

Making the K side use the token axis is therefore a real option, but it is a
*cache-management* change, not a kernel change: the group must not span pages,
and the partial-page problem needs an answer. It is the most promising
follow-up to this document, not part of this proposal.

### Do we need FP32 scales?

No, and the measurement is unambiguous:

| Elements and layout | Scale | bits/value | K-side dB | V-side dB |
| --- | --- | --- | --- | --- |
| INT4 `k=32`, symmetric | FP16 | 4.500 | 18.326 | 20.341 |
| INT4 `k=32`, symmetric | FP32 | 5.000 | 18.326 | 20.341 |
| INT4 `k=32`, packed zp | FP16 | 4.500 | 20.302 | 21.814 |
| INT4 `k=32`, packed zp | FP32 | 5.000 | 20.342 | 21.867 |
| INT8 `k=32` | FP16 | 8.500 | 39.475 | 45.594 |
| INT8 `k=32` | FP32 | 9.000 | 39.476 | 45.593 |

**FP32 buys between 0.000 and 0.05 dB and costs 0.5 bits/value - 11 % more
memory.** The shipped format's FP32 scale is not there for precision. It is
there because an FP32 scale has 23 mantissa bits, so the 4-bit zero-point can
hide inside them for free. A 6-bit-mantissa FP16 scale hides the same
zero-point for 0.04 dB, at half the bytes. So the zero-point trick removes
the only reason to prefer FP32, and **an FP32-scale variant should not be
added**.

### The INT8 family

If INT8 is on the table at all, the scale format is forced:

| INT8 format | bits/value | K-side dB | V-side dB | block scales subnormal (of 256) |
| --- | --- | --- | --- | --- |
| `int8_fp16_block` k=32 | 8.500 | **39.475** | **45.594** | 0 |
| `int8_fp32_block` k=32 | 9.000 | 39.476 | 45.593 | 0 |
| `int8_e8m0_block` k=32 | 8.250 | 36.587 | 41.913 | 0 |
| `int8_e4m3_block` k=32 | 8.250 | 35.952 | 35.304 | 190 K / 247 V |

- **INT8 + FP16 is the best INT8 format measured**, and it needs no second
  level: FP16's exponent range is wide enough that no block scale underflows.
- **E4M3 cannot hold an INT8 block scale.** The scale is `absmax / 127`,
  which lands in E4M3's subnormal range for 190 of 256 blocks on the K side
  and 247 of 256 on the V side, and the SNR falls off a cliff on the V side
  (35.304 dB). This is the same narrow-range
  problem that forces NVFP4's global scale, and it is worse for INT8 than for
  INT4 because the scale itself is smaller.
- **E8M0 costs 2.9 to 3.7 dB against FP16** for 3 % less memory, for the
  same power-of-two reason as at 4 bits.
- **FP32 is again pointless**: 0.001 dB for 0.5 bits/value.

And then the budget question decides whether any of this is worth building -
see the next section.

## What we lose

The honest trade, at the same memory budget. "Loss" is measured against the
proposal at `k=32`, 4.500 bits/value, 20.302 dB on the K side and 21.814 dB
on the V side:

| Against | We lose | We gain |
| --- | --- | --- |
| FP8 e4m3 per-tensor (8.000 bits/value) | 12.67 dB on the K side, 9.75 dB on the V side | 1.78x the context length at a fixed budget |
| INT8 + E8M0 (MXINT8, 8.250 bits/value) | 16.29 dB on the K side, 20.10 dB on the V side | 1.83x the context length at a fixed budget |
| INT8 + FP16 (8.500 bits/value) | 19.17 dB / 23.78 dB | 1.89x the context length at a fixed budget |
| E4M3 scale at the same 4.500 bits/value | 0.32 dB / 0.48 dB | no scale conversion on the read path, 10 mantissa bits, no subnormal risk |
| E8M0 scale at the same 4.500 bits/value | 2.02 dB / 4.01 dB | no global scale ever needed, one byte per block |
| Token-axis grouping instead of head-dimension, on the K side | nothing on the K side, **17.0 dB on the V side** | 1.48 dB on the K side |

Two further costs that do not show up in an SNR table:

- **Arithmetic per byte goes up, not down.** INT8's proven gfx1030 pattern
  (`v_cvt_f32_i32` -> `v_mul_f32` -> `v_cvt_f16_f32`) is about 3 ops per
  element on 1 byte per element. INT4 is about 2.5 ops per element but on
  0.5 byte per element, because unpacking a nibble pair costs a shift and a
  mask. Per *byte* that is roughly twice the arithmetic. Counting the whole
  read path - the conversion, the dot and the probability pass - this
  proposal's INT4 read is 8.1 instructions per element on 1.125 bytes, i.e.
  7.2 instructions per byte, against 8.5 on 2.063 bytes for INT8, i.e. 4.1 per
  byte: **1.75x the arithmetic per byte**, against 1.8x less traffic. On a
  bandwidth-bound decode that is the right trade - the V620's 512 GB/s is the
  limit - but on a compute-bound prefill it may not be, and that is a
  measurement, not an argument. The break-even moves with it: the narrow
  format has to keep about 7 compute units busy before its byte saving beats
  its instruction count, against about 8 for INT8.
- **The scale costs 2 bytes per block, not 1.** At `k=32` that is 0.25
  bits/value more than a one-byte scale, i.e. 5.6 % more KV DRAM traffic than
  `int4_e8m0_block` k=32 and 0.25 bits/value less than the shipped format's
  row-level FP32 scale at `k=64`.

### Why INT8 is not the answer at this budget

The 8-bit end of the table is worth reading carefully, because it is where
the fork's evidence is strongest and where a new format is least needed.

FP8 e4m3 with a per-tensor scale costs **8.000 bits/value**. It is
`--kv-cache-dtype fp8`, it needs no new kernel, and it is the only KV cache
dtype with a measurement on the V620: about 2x the memory saved and +0.1 PPL.
The format itself is documented in
`docs/features/quantization/quantized_kvcache.md`; the PPL figure comes from
a V620 measurement kept outside this tree, so it is the one number here a
reviewer cannot open from the repository and should be treated as a
recollection rather than a citation.

`int8_e8m0_block` - the OCP MXINT8 geometry - costs **8.250 bits/value**. It
is 3 % *more* memory than the FP8 that already ships, it needs a new element
width and a new scale format, and it needs a kernel. Its advantage is 3.6 dB
of synthetic SNR over FP8.

That advantage is not a reason to build it. The one measured data point we
have says the SNR ordering does not predict the PPL ordering: FP8 scores
32.968 dB on the synthetic rows and costs +0.1 PPL in reality, and the table
puts INT8 3.6 dB above it. A format that is bigger, slower to land, and
unmeasured should not displace one that is smaller, already shipping and
already measured on the strength of a synthetic dB figure.

So the honest summary of the whole table:

- **At the 8-bit budget, FP8 already wins** and the proposal should not
  compete there.
- **Below 8 bits/value, the proposal is the best format in the table** at
  every budget it can occupy, and it is the only one that buys a second
  factor of two.

For Qwen3.8-27B (65,536 B/token at FP16) at a 128k context:

| Format | bits/value | B/token | 128k KV cache |
| --- | --- | --- | --- |
| FP16 | 16.000 | 65,536 | 8.00 GiB |
| FP8 e4m3 per-tensor (shipped, measured) | 8.000 | 32,768 | 4.00 GiB |
| `int8_e8m0_block` k=32 | 8.250 | 33,792 | 4.12 GiB |
| **`int4_fp16_block` k=32, packed zp (proposal)** | **4.500** | **18,432** | **2.25 GiB** |
| `int4_per_token_head` (shipped) | 4.250 | 17,408 | 2.12 GiB |

That is the whole case for a 4-bit KV cache on RDNA2: it is the second factor
of two, on top of the FP8 that already exists. Whether that is worth the PPL
is the question H1 has to answer on captured activations, and the answer has
to be a perplexity number, not an SNR number.

## The E8M0 question

E8M0 is the OCP MX shared scale: an unsigned biased exponent, 8 bits, no
sign, no mantissa, no zero and no infinity. The code is the FP32 exponent
field with bias 127, so code 0 is `2^-127`, code 254 is `2^127` and code 255
is the one reserved NaN. It is stored as one raw byte per block, and it is
already in the fork for MXFP8 weights and activations
(`vllm/model_executor/layers/quantization/utils/mxfp8_utils.py`,
`vllm/scalar_type.py:330`).

It is attractive for a KV cache for three reasons: it is one byte, its
exponent range never needs a global scale, and decoding it to FP32 is a shift
(`code << 23`) rather than a conversion. The reference codec models both
rounding rules:

| Rule | Behaviour |
| --- | --- |
| OCP MX v1.0 "floor" | `shared_exp = floor(log2(amax)) - emax_elem`. The block maximum lands in `[4, 8)` and **saturates** at 7 whenever the fractional part of `log2(amax)` exceeds `log2(7/4)`. |
| Round-up ("rceil") | The smallest power of two with `7 * scale >= amax`. Nothing saturates. This is what the fork's own MXFP8 code does (`vllm/model_executor/layers/quantization/utils/mxfp8_utils.py:62`, `vllm/model_executor/layers/quantization/input_quant_fp8.py:249`, `vllm/kernels/helion/ops/per_token_group_fp8_quant.py:155`). |

Measured: the floor rule clipped 17 of 256 blocks at `k=32` and 33 of 256 at
`k=16`. The round-up rule never clips, and costs about 3 dB against an FP16
scale of the same block size - a power-of-two scale has no mantissa, so its
step is up to twice the ideal one.

That is the reason E8M0 is not the proposal here:

- **E4M3 is the same one byte and strictly better.** E4M3 has 3 mantissa
  bits, so its step is essentially exact; E8M0 has none. At 4.500
  bits/value, E4M3 k=16 scores 19.984 dB against E8M0's 18.281 dB on
  K-side rows, with the same byte count. E8M0's only advantages over E4M3
  are the wider exponent range (so no global scale) and the cheaper read
  path.
- **E8M0's power-of-two-ness cannot be repaired by a second level.** A
  per-row global multiplies every block's scale by the same constant, so the
  per-block rounding error stays in `[1, 2)`. Nor can it be repaired by
  halving the block size: the rounding loss is scale-invariant, so a finer
  block does not buy back what the power-of-two step costs.
- **The floor rule's clipping is invisible to the metric.** The floor rule
  scores *higher* SNR than the safe round-up rule (17.341 vs 16.938 dB at
  `k=32`) while clipping 17 blocks, because row-RMS SNR barely punishes one
  clipped element out of 32 while the round-up rule degrades every value.
  A KV cache cannot be accepted on SNR alone: a clipped K value perturbs an
  entire attention logit row. H1 therefore has to be measured on attention
  output error, not on round-trip SNR.

If the fork later wants a one-byte scale, E4M3 at `k=32` is the better buy
than E8M0, and it is the one the codec already models.

## How it would run on gfx1030

The format is only worth having if the kernel is native. Three decisions
follow from the hardware, and none of them are the MX defaults.

**Apply the block scale per element, because on this path there is no
per-partial-dot option.** The MX idiom is to promote the block scale to the
matrix-multiplication boundary and scale each partial accumulator. That is a
tensor-core idiom, and gfx1030 has no matrix instruction - and the
`head_size^2 / k` figure that goes with it describes only a *split-dot*
implementation, which needs a quantised query. This path has an FP16 query
and one FP16 dot over the whole head dimension in a single inner loop, so
there is no partial accumulator to scale. A scale that varies inside the
head therefore cannot be folded into the per-token score, and it costs `1/k`
of an instruction per element - 0.031 slots at `k=32`, 1.5 % - against the
`head_size^2 / k` accumulator operations an MX-shaped implementation would
carry (512 ops at `head_size=128, k=32`; 2048 at `head_size=256`). For the
4-bit formats the scale is free on top of that, because it rides inside the
dequantise that had to happen anyway. The dequantised values are converted to
FP16 in registers and fed straight into `v_dot2_f32_f16`, exactly like the
proven INT8 pattern in `csrc/rocm/fa_rdna2.cu` - a packed dword load, a
sign-extend, a scale multiply and an FP16 convert, with the dequant living
in VGPRs and no materialised FP16 workspace.

**Hoist the zero-point out of the element loop.** With `x_hat = (q - zp) * s`
the subtraction can be taken out of the inner loop entirely:

- QK side: `QK_b = s_b * (dot(Q_b, c_b) - zp_b * sum(Q_b))`. The correction
  term uses the sum of the *query* values in the block, a per-block scalar
  over the query rather than over the cache, so it costs one FMA per
  (query, block) and nothing per element. This is the same correction the
  shipped Triton kernel already applies
  (`vllm/v1/attention/ops/int4_per_token_head.py:570-572`).
- PV side: `out_b = s_b * (dot(P_b, c_b) - zp_b * sum(P_b))`. The per-block
  probability mass is needed anyway, because per-block value scales already
  force a segmented PV accumulation.

So the per-element work is unpack, convert and one multiply - about 2.5 ops
per element - and the zero-point is free.

**Keep the dot in FP16.** The scale is applied to the dequantised values
before the dot, so the inner product stays `v_dot2_f32_f16` with FP32
accumulate, not the scalar-FMA fallback the shipped format lands on.

## Hypotheses and kill criteria

| # | Hypothesis | Check | Holds if | If not |
| --- | --- | --- | --- | --- |
| H1 | Block-granular scales with a packed zero-point beat the shipped per-row scale on real KV | capture real K/V activations from a V620 decode run, run the reference codec over them, and score **attention output error**, not round-trip SNR | attention output error improves against `int4_per_token_head`, and the packed-zero-point variant is within 0.5 dB of the variant that spends its own 4 bits on the zero-point | stop: the format is not worth a new dtype, keep the shipped one |
| H2 | Dequantising to FP16 puts the dot on `v_dot2_f32_f16` instead of the scalar FMA path | A/B the decode kernel on the V620, FP16 operands against FP32 operands | decode tok/s at c=1 improves, prefill within noise | keep FP16 for the memory and accuracy, drop the speed claim |
| H3 | The format lands in the RDNA2 backend at all | revive the INT8 FA-RDNA2 dispatch (see *The INT8 backport*), then swap the format inside it | a decode run completes with the new dtype and the cache size matches the byte-budget arithmetic | stop: the Triton path is the only viable target and the design must be re-scoped for it |
| H4 | An FP16 block scale does not need a global scale | replay the H1 capture with the global on and off | SNR identical to within 0.01 dB | the global becomes mandatory and the format needs a per-tensor side buffer, which is a much larger change |
| H5 | The zero-point really is free | compare the packed variant against the variant that spends 0.125 bits/value on its own zero-point, on captured activations | within 0.5 dB | store the zero-point separately and pay 0.125 bits/value for it |

## Where it would plug in

Nothing below exists yet. This is the surface a future implementation would
have to touch.

| Piece | File | What changes |
| --- | --- | --- |
| dtype literal | `vllm/config/cache.py:39` | add `"int4_fp16_block"` to `CacheDType` |
| dtype to torch dtype | `vllm/utils/torch_utils.py:43` | add the `torch.uint8` entry |
| quantised-predicate | `vllm/utils/torch_utils.py:78` | `is_quantized_kv_cache` already returns True for any `*per_token_head`; a new dtype needs its own branch |
| mode enum | `vllm/v1/kv_cache_interface.py:39` | add a `KVQuantMode` value and map it in `get_kv_quant_mode` (`:84`) |
| page size | `vllm/v1/kv_cache_interface.py:86` | the scale bytes are part of the page, so the spec has to account for `head_size/k` FP16 scales per head |
| attention ops | `vllm/v1/attention/ops/` | a new module next to `int4_per_token_head.py`, owning nibble pack/unpack, the reshape (write) kernel and the split-dot (read) kernel |
| backend opt-in | `vllm/v1/attention/backends/rdna_attn.py:601` | `supported_kv_cache_dtypes` |

The Triton backend already lists `int4_per_token_head`
(`vllm/v1/attention/backends/triton_attn.py:320`), so H3 is the fork's own
question: whether the RDNA2 backend is the target, or whether the Triton path
is.

## The INT8 backport: what it actually costs

"Classic INT8 KV support" means reviving the INT8 FA-RDNA2 path that was
built in the `humanwork-legacy` tree and never landed. The port was mapped
against this tree, and the headline is better than expected: **the HIP
kernels are already here.** `origin/rdna_extras` carries
`fa_prefill_paged_varlen_int8`, `fa_rdna2_decode_paged_int8`,
`reshape_and_cache_int8_rdna2` and the `int8` split-K prefill kernels with
byte-identical C++ signatures. The port is Python and binding wiring, not
kernel work.

What is missing, and what is wrong:

1. **The ops are not registered.** `fa_rdna2_prefill_paged_varlen_int8` and
   `fa_rdna2_decode_paged_int8` are not in `csrc/rocm/torch_bindings.cpp`.
   Only the writer is, and its comment says why. The legacy tree reached the
   kernels through a `load_inline` JIT build; this tree loads them through
   `torch.ops._rocm_C`, so the registration and the
   `vllm/v1/attention/ops/fa_rdna2_backend.py` shim both have to change.
2. **The writer has a real defect, and it is in this tree already.**
   `csrc/rocm/fa_rdna2.cu:4681` writes the four bytes of a slot's FP32 scale
   at `HEAD_DIM * block_size + slot_in_block + d`. With the declared
   `[num_blocks, H_kv, HEAD_DIM+4, block_size]` layout and
   `stride_d = block_size`, that address is *not* `(HEAD_DIM + d) *
   block_size + slot_in_block`. Two consequences: slot `s`'s byte `d` lands
   on slot `s+d`'s byte 0, so the four bytes of one slot are spread across
   four slots and race with them; and the scale planes `HEAD_DIM+1 ..
   HEAD_DIM+3` are **never written at all**. The correct offset is
   `(HEAD_DIM + d) * block_size + slot_in_block`.
3. **The legacy Python reader is not the inverse of the writer.** It does
   `kv_cache[0, :, :, hs:hs+4, :].view(torch.float32)`, which reinterprets
   the *slot* dimension as bytes, not the byte dimension. Even with the
   writer fixed, the legacy split does not recover the scales.
4. **The scale index convention differs three ways.** The kernels index the
   scale by *logical sequence position*; the legacy Python produced
   *physical* `(block, slot)`; the canonical Triton layout in this tree is
   *physical* `(block, slot, head)`. A port has to pick one and make the
   writer, the reader and the cache spec agree.
5. **The legacy end-to-end evidence is a throughput number.**
   `.omo/plans/rdna2-int8-isa-benchmark/e2e-int8-bench.log` reports 2.93
   output tok/s against a 2.91 tok/s FP16 baseline at 5.89 GiB of KV cache.
   That measurement cannot distinguish a correct scale from a corrupted one,
   and no test in the legacy tree exercises the writer-to-reader path. So
   the legacy path's *numerical* correctness was never established, and the
   port has to be treated as new work with a new correctness test, not as a
   re-application of something that worked.
6. Smaller items the port has to handle: the int8 ops return a `Tensor`
   while every other `fa_rdna2` op writes into `out`; the writer takes an
   int32 `slot_mapping` where vLLM passes int64; `csrc/rocm/ops.h`
   declares `reshape_and_cache_int8_rdna2` twice; and the int8 work is a
   seven-commit chain, not one commit.

The practical conclusion: reviving INT8 support is a **Python-and-wiring
job of maybe a day, plus a kernel fix and a correctness test that never
existed**. That is the cheapest path to a quantised KV cache in the RDNA2
backend, and it is the host this proposal assumes for H3 - but it should be
landed and measured on its own before this format rides on it.

## Prior work, and why this is not a duplicate

Two things already exist in this area and neither is superseded by this
document:

- **The shipped `int4_per_token_head`** (`vllm/v1/attention/ops/int4_per_token_head.py`).
  This document argues for replacing it on RDNA2, not for adding a second
  4-bit dtype beside it. Its FP32 row-level scale and FP32 dot are the two
  things this proposal changes.
- **A FA-RDNA2 INT8 KV cache**, drafted as
  [PR #71](https://github.com/opengfx1030/vllm-rdna/pull/71) and implemented
  in the `humanwork-legacy` tree at commit `3baecdb516` ("feat(rocm): fp8 +
  int8 KV cache decode and prefill for fa_rdna2"). It is a *different* format:
  symmetric INT8 with a per-(token, head) **FP32** scale, int8-storage prefill
  and decode kernels in `csrc/rocm/fa_rdna2.cu` - the cache is int8, but the
  dot is FP16 in both of them: the decode path dequantises each element into an
  FP16 shared tile and the prefill path dequantises in VGPRs, and neither
  uses an integer dot instruction - and an INT8 writer
  (`reshape_and_cache_int8_rdna2`). In this tree it is not reachable:
  `fa_rdna2_prefill_paged_varlen_int8` and `fa_rdna2_decode_paged_int8` are
  not registered in `csrc/rocm/torch_bindings.cpp`, and
  `RdnaAttentionBackend._can_run_fa_rdna2` rejects every quantised KV dtype
  (`vllm/v1/attention/backends/rdna_attn.py:283`).

  That work is this proposal's most likely *host*, and the port is mapped in
  *The INT8 backport* above. It is also this proposal's clearest benchmark:
  the INT8 path's own measurements are in
  `.omo/plans/rdna2-int8-isa-benchmark/` on the parent project (2.93 output
  tok/s at 5.89 GiB of KV cache, against a 2.91 tok/s FP16 baseline), and
  the ISA study in `isa-report.md` is where the gfx1030 dequant patterns
  come from - the packed-dword load, the sign-extend, the scale multiply
  and the FP16 convert that the v3 variant beats every other on.

## What this PR contains

- `docs/explore/kv-int4-fp16-scales/README.md` - this document.
- `docs/explore/kv-int4-fp16-scales/reference_codec.py` - the reference codec
  and comparison harness. Pure Python, no dependencies, self-checking.
- `docs/explore/kv-int4-fp16-scales/TESTPLAN.md` - the measurement plan, with
  the V620 cells left empty.

## What this PR does not contain

- No dtype, no backend change, no kernel. Nothing in the serving path
  changes.
- No performance claim. `TESTPLAN.md` has empty cells; the only numbers in
  this document come from `reference_codec.py` and are labelled as arithmetic
  on synthetic data.
- No element-format decision. INT4 and E2M1 are both in the table above and
  the choice is open.
- No claim about a one-byte scale. E8M0 and E4M3 are measured and rejected
  at the 4.500 bits/value budget, not dismissed.

## Reproducing the numbers

```bash
python3 docs/explore/kv-int4-fp16-scales/reference_codec.py
```

The script runs its own self-check first (FP16, E4M3 and E8M0 round-trip,
INT4 packing, the zero-point packing, and the scale-format invariants) and
prints the three tables above.

## AI assistance

This draft was written with AI assistance. Every file:line reference above was
checked against the tree, and the numbers come from the committed script
rather than from the author. No hardware measurement was taken for this PR:
the V620 cells in `TESTPLAN.md` are empty on purpose.
