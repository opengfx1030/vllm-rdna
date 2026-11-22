# Inkling-2-Mini weight-only quantized decode linears on RDNA2 (`inkling2_qlinear_*`)

Native HIP GEMV kernels for the dense fp16 projections of the decode step with
int8 or int4 weights (one fp16 scale per group of K), fp16 activations and fp32
accumulation on gfx1030 (V620: wave32, `V_DOT2_F32_F16`, no int8 / fp8 / fp4
matrix hardware). They exist because the b = 1 decode step of
`Inkling2LoopForConditionalGeneration` reads about 4.6 GB per token, 3.5 GB of
it fp16 dense weights (attention input projections, `o_proj`, `lm_head`) that
the fp16 kernels of [`inkling2-linear-kernels.md`](inkling2-linear-kernels.md)
already stream at the DRAM roofline: the only way to be faster is to read fewer
bytes. Up to 4 activation rows are served (decode, the DSpark verification
step); more rows, and the whole prefill, keep the fp16 kernels. Everything is
HIP-graph capturable and needs no Triton, rocBLAS or torch in the serving path.

The levers are OFF by default (section 5). The numbers of this document are
measured on the box of the runs (one V620, HIP 2, TP = 1, graphs on); logs are
listed in section 9.

| What | Where |
| --- | --- |
| Kernels | `csrc/rocm/inkling2_qlinear_rdna2.cu` |
| Registration | `csrc/rocm/ops.h`, `csrc/rocm/torch_bindings.cpp`, `CMakeLists.txt` |
| Python ops | `vllm/_custom_ops.py` (`inkling2_qlinear_grouped`, `inkling2_qlinear_lm_head`) |
| Offline quantizer, packer, serving packs, flags | `vllm/models/inkling2_loop/amd/qlinear_pack.py` |
| Wiring | `attention.py` (`prepare_hip`), `model.py` (`_prepare_serving`) |
| Tests | `tests/kernels/quantization/test_inkling2_qlinear_rdna2.py` |
| Oracle check, mutants, benchmark, JIT build | `tools/rdna/inkling2/qlinear_check.py`, `qlinear_mutants.py`, `qlinear_bench.py`, `qlinear_jit.py` |
| Decode-path quality / speed gate | `tools/rdna/inkling2/decode_gate.py` |

## 1. Why bytes, and which

The b = 1 TP = 1 graph decode step before this work (`profile_decode.py`, one steady-state
decode step of `logs/prof_base2.log`: 783 kernels, 17.84 ms of GPU time, 19.88 ms of clean
wall per step; the first profiled step after the prefill is 2 to 3 ms slower and is not used):

| Family | Launches | Time (per launch) | Weights read | GB/s |
| --- | ---: | ---: | ---: | ---: |
| attention input pack (`q gate rel sink k v`, + dynamic taps on layers 5 11 17 23) | 48 | 4.48 ms (93.3 us) | 39.6 MB per visit, 1.90 GB | 424 |
| `o_proj` | 48 | 1.66 ms (34.6 us) | 14.2 MB per visit, 0.68 GB | 409 |
| `lm_head` (+ softcap, fused argmax partials) | 1 | 2.09 ms | 921.9 MB | 440 |
| NVFP4 / MXFP8 MoE experts | 96 | 5.40 ms | 1.84 GB | 341 average |
| glue and small kernels (router, gate mix, conv, norms, attention) | 590 | 4.20 ms | small | latency bound |

The dense fp16 matmuls stream at 409 to 440 GB/s of the 512 GB/s of the card (the
microbenchmarks of section 6 read 350 to 443 GB/s, they include the graph-node gap), so
their time is `bytes / bandwidth`; halving (int8) or quartering (int4) the bytes is the
lever. These three families are 3.5 GB of the step's dense fp16 weights; the rest (router,
gate mix, convolutions) is a few tens of MB. Whether the weights tolerate it is a quality
question, answered in section 7.

## 2. Formats and the quantizer

Symmetric round-to-nearest, no zero point per group, `q = round(w / s)` clamped to
`[-qmax - 1, qmax]`, `s` an fp16 per-group scale computed from the group absmax
(rounded to fp16 BEFORE the division, so the dequantized weight the kernel
multiplies with is exactly `q * s`); `s >= 2^-14`.

| Spec | Weight | Scale group | Bytes per fp16 weight pair | Notes |
| --- | --- | ---: | ---: | --- |
| `w8` / `w8g64` | int8 | 64 | 1.031 / 2 | default int8 |
| `w4` / `w4g32` | int4 | 32 | 0.5625 / 2 | MSE clip search of the scale (candidates 1.0 .. 0.65 of the absmax) |
| `w4g64` | int4 | 64 | 0.5313 / 2 | same clip search |
| `w4g32n` | int4 | 32 | 0.5625 / 2 | no clip search |

The kernel and the packer are generic in `bits in {4, 8}` and the group (a
multiple of the chunk, 16 int8 / 32 int4 K elements); the serving build
instantiates only the three formats above.

## 3. Kernel design

A TILE is `R` weight rows. A wave32 holds `32 / R` K quarters (`KQ`) of `R` rows:
lane `l` owns row `l % R` and the quarter `l / R`, and its 16-byte weight chunk `c`
sits at `((tile * nc + c) * 32 + l) * 16`: the 32 lanes of one load read 512
contiguous bytes, no lane ever shares a row with another lane of the same quarter,
and the only cross-lane reduction is the final `KQ`-way one. A chunk is `CK` = 16
(int8) or 32 (int4) consecutive K elements of the lane's row. Scales are laid out
`[tile][iteration][lane][SV]` (`SV` groups per iteration, one 32-bit load per
group pair). `WV` waves form a block; a block owns `WV` tiles; the activations
`[M, K]` are staged once per block in LDS (fp16) while the first weight loads are
already in flight, and the next iteration's chunks are prefetched into registers
while the current ones are consumed (`CPI` chunks per iteration).

Dequantization is exact and integer only. A byte `u` in the low mantissa bits of an
fp16 (`0x6400 | u`) is the number `1024 + u`; one packed subtract of `1024 + zero
point` leaves the signed weight as an fp16 value (`v_perm_b32` selects the byte
pairs; for int4 the two nibbles of a byte are taken in place, the high one as
`16 u` and corrected by 1/16 after the dot product). `v_dot2_f32_f16` multiplies
two weights with two activations into an fp32 accumulator, and the group scale
multiplies the fp32 partial sum of the group (`fma`).

`inkling2_qlinear_lm_head` is the same kernel with `R = 32`, the final softcap, the
vocabulary mask (`-inf` for rows `>= n_valid`) and, for the greedy path, the fused
per-block argmax partial (first index wins ties inside a lane, across lanes, across
waves and across blocks; NaN counts as `-inf`): `inkling2_linear_argmax_reduce` of
the fp16 head finishes it, so the sampler code is unchanged.

Instances are a compile-time table (`inst_at()` over bits, scale group, `R`, `KQ`,
activation rows `MT in {1, 2, 4}`, `CPI`, `WV`, argmax), filtered by
`I2Q_ENABLED(...)`. The serving build compiles the measured fastest of the sweep
(section 6): int8 `R = 8 CPI = 8`, int4 `R = 8 / 4 CPI = 4` (attention input /
`o_proj`), `R = 32 CPI = 8` (`lm_head`), `WV = 8` everywhere. The JIT build
(`tools/rdna/inkling2/qlinear_jit.py`) takes any filter for sweeps and mutants.

Resource use of the serving instances (`-Rpass-analysis=kernel-resource-usage`):
no scratch and no spills in any of the 22; 64 to 168 VGPRs, 5 to 16 waves per
SIMD (the largest: the int4 g32 `lm_head` with 4 activation rows).

## 4. Op reference

```text
inkling2_qlinear_grouped(Tensor[] xs, Tensor w, Tensor sc, Tensor(a!)[] outs,
                         int[] seg_x, int[] seg_n, int[] seg_act, float[] seg_p,
                         int[] seg_valid, int bits, int group, int rows_per_tile,
                         int cpi, int waves)
inkling2_qlinear_lm_head(Tensor x, Tensor w, Tensor sc, Tensor!? logits,
                         Tensor!? part_val, Tensor!? part_idx, float softcap,
                         int n_rows, int n_valid, int col_base, int bits,
                         int group, int cpi)
```

The segment model, activation / output strides, epilogues (`0` none, `1` sigmoid,
`2` tanh, `3` softcap `tanh(v / p) * p`), `seg_valid` and the `-inf` mask are those
of `inkling2_linear_grouped` / `inkling2_linear_lm_head`. `w` is uint8
`[tiles, nc, 32, 16]` and `sc` fp16 `[tiles, niter, 32, SV]` as produced by
`qlinear_pack.pack_weight` (every segment padded to whole tiles of `R` rows);
`1 <= M <= 4` (`M = 3` runs the 4-row instance with a zero row); activations fp16,
outputs fp16 or fp32, `softcap == 0` means no softcap. Wrong shapes, an instance
that is not built, or `M > 4` raise `RuntimeError` on the host (never inside a
graph).

## 5. Serving: flags, packs, memory

All off by default (`""`, `0`, `off`, `none`, `false`). Values: `w8`, `w8g64`, `w4`,
`w4g32`, `w4g64`, `w4g32n`.

| Variable | Replaces |
| --- | --- |
| `VLLM_INKLING2_ATTN_IN_QUANT` | attention input pack (`q gate rel sink k [dyn_k] v [dyn_v]`), 24 layers |
| `VLLM_INKLING2_O_PROJ_QUANT` | `o_proj`, 24 layers |
| `VLLM_INKLING2_ATTN_QUANT` | default of the two above |
| `VLLM_INKLING2_LM_HEAD_QUANT` | `lm_head` (logits and the fused greedy argmax), TP = 1 |

`prepare_hip` and `_prepare_serving` quantize at weight load (the fp16 weights are
already on the device: one pass of `pack_weight` per pack; no measurable boot time, the
engine boots in 37.3 to 38.6 s with a flag and 38.3 s without, 29.5 s with and without
on a warm page cache) and wrap the fp16 pack in `HybridLinearPack` / `HybridLMHead`: up to
4 token rows run the quantized kernel, larger chunks and the prefill the fp16 kernels
(which stay the owners of segment names, sizes and outputs, so the model code is
unchanged). The quantized copy is EXTRA memory next to the fp16 weights
(section 6.5). Tensor parallelism ignores the flags. DSpark verification
(`K + 1` rows, K <= 3) takes the quantized path.

## 6. Measurements

### 6.1 Kernels: cold weights, HIP-graph timed

`tools/rdna/inkling2/qlinear_bench.py`: every call reads DRAM (a rotating set of copies of
the weight, more than 700 MB, so the L2 never helps), ONE HIP graph over all copies,
microseconds per call INCLUDING the graph-node gap of about 4 us (the method of
`docs/rdna2/hi2-prefill-gemm-2026-11-15.md`). The fp16 baseline is the `inkling2_linear_*`
kernel the model runs. The error is `max |y - y_ref| / rms(y_ref)` against the fp64 matmul
with the dequantized weights (`4e-7 .. 9e-7` for every instance: fp32 accumulation order only).
Production instances, 1 activation row:

| Family | fp16 | int8 g64 | int4 g32 | int4 g64 |
| --- | --- | --- | --- | --- |
| attention input pack, 8600 rows x 2304 | 39.6 MB, 98.8 us, 401 GB/s | 20.5 MB, 55.3 us, 371 GB/s, 1.79x | 11.3 MB, 33.9 us, 333 GB/s, 2.92x | 10.6 MB, 32.5 us, 326 GB/s, 3.04x |
| `o_proj`, 2304 x 3072 | 14.2 MB, 40.5 us, 350 GB/s | 7.3 MB, 25.5 us, 287 GB/s, 1.59x | 4.0 MB, 12.7 us, 313 GB/s, 3.18x | 3.8 MB, 12.2 us, 307 GB/s, 3.31x |
| `lm_head` + softcap + argmax partials, 200058 x 2304 | 921.9 MB, 2079.5 us, 443 GB/s | 475.4 MB, 1144.0 us, 416 GB/s, 1.82x | 259.3 MB, 609.7 us, 425 GB/s, 3.41x | 244.9 MB, 585.6 us, 418 GB/s, 3.55x |

Four activation rows (the DSpark verification of K = 3 and decode of 4 requests) cost 3 to 12%
more: attention input pack 56.9 / 37.3 / 36.2 us, `o_proj` 27.7 / 15.3 / 14.8 us,
`lm_head` 1128 / 666 / 624 us (int8 g64 / int4 g32 / int4 g64), against 102.2 / 41.9 / 2093.8 us
for fp16. (`logs/bench_prod.log`. The first quantized case of that run, attention input
pack int8 g64 with one row, read 62.0 us; the same instance read 54.8 to 55.3 us in
`bench1.log`, `bench2.log` and `floor1.log`, the value of the table. [INFERENCE] a clock /
power transient after the 2 ms fp16 `lm_head` cases.)

### 6.2 The instance sweep is flat

`R` in {4, 8, 16, 32}, `CPI` in {4, 8}, `WV` in {2, 4, 8} (`bench2.log`): attention input pack
int8 g64 54.8 to 59.7 us, int4 g64 32.3 to 35.0 us; the serving instances are the fastest of
each family, the spread is at most 9%. There is no tuning left in these parameters.

### 6.3 What limits them: not the arithmetic

A scratch copy of the kernel with the dequantization and the dot products replaced by an
XOR of the loaded words (same loads, same layout, same launch; `logs/floor1.log`) takes
55.0 us (attention input pack int8 g64), 32.3 us (int4 g64), 25.2 us (`o_proj` int8 g64)
and 12.3 us (`o_proj` int4 g64): the same as the real kernels (55.3, 32.5, 25.5, 12.2 us). For
`lm_head` the stripped copy is slower (1271.6 us int8, 666.1 us int4 against 1144.0 and
609.7 us). The kernels sit on the memory path (about 330 to 430 GB/s per call, of which
about 4 us per call is the graph-node gap and the ramp of a 10 to 50 us kernel), so more
kernel work does not help: only fewer bytes or fewer launches do.

### 6.4 End to end: the decode step

Kernel time of one steady-state b = 1 graph decode step (`profile_decode.py`, TP = 1,
`FULL_AND_PIECEWISE`; `prof_base2` against `prof_all8`, both with the same tool, box and
hour; 783 kernels in both):

| Per step | fp16 | `LM_HEAD` + `ATTN` int8 | saved |
| --- | ---: | ---: | ---: |
| attention input pack, 48 launches | 4.480 ms (93.3 us each) | 2.409 ms (50.2 us) | 2.07 ms |
| `o_proj`, 48 launches | 1.660 ms (34.6 us) | 0.954 ms (19.9 us) | 0.71 ms |
| `lm_head` | 2.094 ms | 1.115 ms | 0.98 ms |
| the three families | 8.234 ms | 4.478 ms | 3.76 ms |
| everything else (MoE 5.40, glue 4.20) | 9.603 ms | 9.551 ms | 0.05 ms |
| GPU kernel time | 17.837 ms | 14.029 ms | 3.81 ms |
| clean step wall of that tool (median of 8 steps) | 19.88 ms | 16.67 ms | 3.21 ms |

The kernels run at 408 GB/s (attention input, int8), 367 GB/s (`o_proj`) and 426 GB/s
(`lm_head`) inside the real step. The first profiled step after a prefill is 2 to 3 ms slower
than the next ones for both builds (clock / cache warm-up); the 0.4 ms of slow first
attention-input visits seen in the first baseline profile belonged to that step.

Per token of ONE request, the gate engine (`decode_gate.py`: `FULL_AND_PIECEWISE` graphs,
TP = 1, 8 groups of distinct prompts, `max_num_batched_tokens` 4, `max_num_seqs` 4):

| Levers (environment) | b = 1 ms/token | tok/s | saved | b = 4 ms/step |
| --- | ---: | ---: | ---: | ---: |
| none (two runs) | 19.05 / 19.10 | 52.5 / 52.4 | - | 37.11 / 37.19 |
| `LM_HEAD_QUANT=w8` | 18.09 | 55.3 | 0.96 | 35.84 |
| `LM_HEAD_QUANT=w4` | 17.63 | 56.7 | 1.42 | 35.10 |
| `O_PROJ_QUANT=w8` | 18.44 | 54.2 | 0.61 | 36.74 |
| `ATTN_IN_QUANT=w8` | 17.07 | 58.6 | 1.98 | 34.67 |
| `ATTN_QUANT=w8` (attention input + `o_proj`) | 16.27 | 61.5 | 2.78 | 33.48 |
| `ATTN_QUANT=w8 LM_HEAD_QUANT=w8` | 15.28 | 65.5 | 3.77 | 31.99 |
| `ATTN_QUANT=w4 LM_HEAD_QUANT=w4` (round to nearest) | 13.49 | 74.2 | 5.56 | 29.68 |
| `ATTN_QUANT=w4 LM_HEAD_QUANT=w4` + GPTQ attention file | 13.39 | 74.7 | 5.66 | 29.58 |
| `ATTN_QUANT=w8 LM_HEAD_QUANT=w4` | 14.78 | 67.7 | 4.27 | 31.31 |

The savings add up (`lm_head` 0.96 + attention 2.78 = 3.74 against 3.77 measured together)
and equal the kernel-level deltas of the microbenchmarks (section 6.1: `lm_head` 0.97 ms,
attention 2.8 ms). A step of four requests gains 5.1 ms with everything int8 (37.11 to
31.99 ms) and 7.4 ms with everything int4 (29.68 ms): the weights are read once for all four
rows. Whether each row passes the quality gate is section 7.

### 6.5 Memory

The quantized copy lives NEXT TO the fp16 weights (the prefill and chunks above 4 rows
need them). Engine log, `Model loading took` (GiB):

| Levers | Weights | Extra |
| --- | ---: | ---: |
| none | 18.35 | - |
| `O_PROJ_QUANT=w8` | 18.52 | +0.17 |
| `LM_HEAD_QUANT=w8` | 18.79 | +0.44 (453 MB) |
| `ATTN_IN_QUANT=w8` | 18.81 | +0.46 |
| `ATTN_QUANT=w8` | 18.98 | +0.63 |
| `ATTN_QUANT=w8 LM_HEAD_QUANT=w8` | 19.42 | +1.07 |
| `ATTN_QUANT=w4 LM_HEAD_QUANT=w4` | 18.94 | +0.59 (`lm_head` 247 MB) |

At 32 KiB of KV cache per token (`docs/rdna2/inkling2-serving.md`, "Known limits") the
all-int8 build holds about 35,000 tokens less context at TP = 1 (1.07 GiB / 32 KiB) and the
all-int4 build about 19,000 less [derived, not booted at the limit]. Dropping the fp16
copies would need a dequantizing GEMM for the prefill; it is not built.

## 7. Quality

### 7.1 The formats on the real weights (round to nearest)

`qstudy.py` on the real checkpoint: relative rms error of the layer OUTPUT (mean over the 24
layers, the real hidden states of a 700-token chat prompt, fp32 accumulate, error against the
fp16-weight result):

| Segment | int8 g64 | int4 g32 + clip | int4 g32 + clip, max error / output rms |
| --- | ---: | ---: | ---: |
| `q` | 5.3e-3 | 7.7e-2 | 0.42 |
| `gate` | 2.9e-3 | 4.3e-2 | 0.26 |
| `rel` | 4.0e-3 | 5.9e-2 | 0.31 |
| `sink` | 8e-4 | 2.1e-2 | 0.07 |
| `k` | 4.6e-3 | 6.8e-2 | 0.38 |
| `v` | 4.8e-3 | 7.0e-2 | 0.45 |
| `dyn_k`, `dyn_v` (4 layers) | 3.5e-3, 4.1e-3 | 3.6e-2, 4.1e-2 | 0.16, 0.25 |
| `o_proj` | 7.8e-3 | 1.13e-1 | 0.87 |

`lm_head` on 701 real final hidden states (softcap included; "target" = the next token of the
text):

| Format | top-1 agrees with fp16 | fp16 top-1 in top-5 | mean abs change of the top-10 logits | mean abs change of log p(target) |
| --- | ---: | ---: | ---: | ---: |
| int8 g64 | 0.9986 | 1.0000 | 0.0079 | 0.0155 |
| int4 g32 + clip (`w4`) | 0.9700 | 1.0000 | 0.166 | 0.229 |
| int4 g32 | 0.9643 | 1.0000 | 0.129 | 0.228 |
| int4 g64 | 0.9558 | 1.0000 | 0.160 | 0.296 |

### 7.2 The real-model gate: teacher-forced NLL, 4-token chunks

`tools/rdna/inkling2/decode_gate.py` boots the engine with `max_num_batched_tokens` 4, so the
text is scored in chunks of four tokens: every token of it goes through the quantized kernels
(`real_check.py` NLL mode scores one big chunk, which the fp16 kernels run, and could not see
a decode lever). It scores 799 positions of `runs/nll_text.md` (`inkling2-port-spec.md`) against
the TP = 1 eager reference dump `tp1_a`; the gate is the mean `|dlogp|` below 0.1027 (twice the
0.0463 between the two reference dumps `tp1_a` and `tp1_b`, plus 0.01). The engine
without any lever scores 0.0397 to 0.0409 against `tp1_a` and 0.0396 against itself: that is the
run-to-run noise of the logits at near-ties (the serving doc, "Greedy output is not bit-stable"),
so the gate catches a break, and a lever has to be judged by the SIGNED paired shift (mean
`dlogp` of the same positions, with its standard error) and by the longer run below.

| Levers | mean abs `dlogp` against `tp1_a` (limit 0.1027) | against the lever-free run | paired shift against it (mean +- se) | mean log p |
| --- | ---: | ---: | ---: | ---: |
| none | 0.0409 | - | - | -1.9834 |
| none, second run | 0.0397 | 0.0396 | -0.0022 +- 0.0025 | -1.9856 |
| `LM_HEAD_QUANT=w8` | 0.0421 | 0.0384 | +0.0019 (se not printed) | -1.9815 |
| `LM_HEAD_QUANT=w4` | 0.0759 | 0.0745 | -0.0039 +- 0.0046 | -1.9873 |
| `O_PROJ_QUANT=w8` | 0.0541 | 0.0576 | +0.0041 +- 0.0038 | -1.9793 |
| `ATTN_IN_QUANT=w8` | 0.0548 | 0.0552 | -0.0072 +- 0.0036 | -1.9906 |
| `ATTN_QUANT=w8` | 0.0558 | 0.0585 | -0.0076 +- 0.0037 | -1.9910 |
| `ATTN_QUANT=w8 LM_HEAD_QUANT=w8` | 0.0558 | 0.0575 | -0.0084 +- 0.0038 | -1.9918 |
| `ATTN_QUANT=w4 LM_HEAD_QUANT=w4` (round to nearest) | **0.1700 FAIL** | 0.1685 | -0.0204 +- 0.0108 | -2.0038 |
| `ATTN_QUANT=w4 LM_HEAD_QUANT=w4` + GPTQ attention file | **0.1424 FAIL** | 0.1434 | -0.0037 +- 0.0093 | -1.9872 |
| `ATTN_QUANT=w8 LM_HEAD_QUANT=w4` | 0.0780 | 0.0818 | -0.0156 +- 0.0050 | -1.9991 |

The reference mean log p is -1.9824. Every int8 lever and the int4 `lm_head` pass; the
all-int4 round-to-nearest build fails (p99 `|dlogp|` 1.15, max 2.8). Over 3999 positions
(`h_*` runs, the same text, tokens 0 to 4000): the lever-free engine twice 0.0428 (p99 0.342) and
a paired shift +0.0003 +- 0.0014; everything int8 against the lever-free run 0.0513 (p99
0.410) and a shift -0.0007 +- 0.0016, no measurable NLL cost: the -0.008 of the first 800
positions (2 se) is not reproduced at five times the length.

The greedy generations of the 30 prompts (24 tokens) equal the lever-free run's for 21 prompts
with everything int8, 17 with everything int4, and 20 for a second lever-free run: the
differences are inside the noise of identical engines.

### 7.3 Which part breaks at int4

The `lm_head` alone passes at int4 (0.0759) and saves 0.46 ms more than int8; the failure of
the all-int4 build is therefore the attention packs: their round-to-nearest output error is 4
to 11% rms (table 7.1) against 0.3 to 0.8% for int8. The single-lever int4 attention runs were
not made (they would have cost two more 4-minute engine runs); the offline GPTQ solve below
attacks the attention error directly.

### 7.4 GPTQ attention: evidence, not a fix

`gptq_quant.py` solved the attention packs offline (corpus: 118k tokens of the vLLM docs, NOT the
text of the gate; 8000 calibration + 2000 held-out tokens, both visits of the loop; 976 s, 24
layers x 4 solves; `logs/gptq_w4g32.log`, `out/gptq_w4g32.pt`). The output-error proxy
`tr((W-Q) H (W-Q)^T) / tr(W H W^T)` on the HELD-OUT tokens falls from 0.0062 (round to nearest)
to 0.0025 (mean over the solves); on the calibration tokens to 0.0013.

The gate with that file (`VLLM_INKLING2_QUANT_FILE`, `ATTN_QUANT=w4 LM_HEAD_QUANT=w4`,
`logs/g_gptq_all4.log`): 13.39 ms/token (74.7 tok/s), mean abs `dlogp` 0.1424 against `tp1_a`
(limit 0.1027) = **FAIL** (p99 1.06, max 2.24), paired shift -0.0048 +- 0.0093. GPTQ narrows
round to nearest (0.1700) but does not pass. The configuration that passes at int4 speed is
`ATTN_QUANT=w8 LM_HEAD_QUANT=w4`: 14.78 ms/token (67.7 tok/s, b = 4: 31.31 ms), 0.0780 against
`tp1_a`, paired shift -0.0167 +- 0.0048 (`logs/g_a8l4.log`).

## Status (resume notes)

Prototype, OFF by default, nothing runs unless a flag is set: the model loads the native
bf16 -> fp16 weights untouched. The levers quantize AT LOAD from those weights (about a second,
no per-token cost), which the lead rejected as the shipping path (a quantized tier must be an
offline checkpoint); no offline tool or checkpoint format was written. What exists and is
verified is the kernels (section 8) and the measurements (sections 6 and 7). To resume: the
patch `inkling2-qlinear.patch` (header of this doc, section 9) applies to the merged tree.

## 8. Verification

### 8.1 Kernel oracle (`qlinear_check.py`)

Reference: the fp64 matmul with the dequantized weights `q * scale` (the weights the
kernels multiply with); tolerance 2e-5 of the output rms, observed 4e-7 to 9e-7. Covered:
every serving instance and the instance sweep, 1 to 4 rows, multi-input segment sets with
row tails, inactive segments (`seg_act`, `seg_valid`), the epilogues, shared and distinct
input streams, `lm_head` with softcap, the vocabulary mask, `col_base` and exact ties of the
fused argmax (the smaller index wins), the production packs against the dequantized
weights, the hybrid dispatch (up to 4 rows the quantized kernel, beyond it the fp16 one,
bit-exact outputs) and `argmax_reduce` through the real op.

| Run | Result |
| --- | --- |
| JIT sweeps, `logs/qcheck1.log`, `qcheck2.log` | `CHECK PASS: 0 failures`, python exit 0 |
| repo tool, JIT kernels, `logs/chk_tool_jit.log` | `CHECK PASS: 0 failures, 24 grouped configs`, exit 0 |
| repo tool, installed ops and production packs, `logs/chk_ops.log` | `CHECK PASS: 0 failures, 36 grouped configs` (+ `lm_head` w8 / w4, ties), exit 0 |

### 8.2 Hand mutants (a green check that cannot fail is no evidence)

`qlinear_mutants.py` changes one line of the real kernel source per mutant, rebuilds it and
runs the oracle: it must FAIL. Twelve mutants: the int4 nibble map, the int4 zero point, the
int8 byte permutation, the scale pair, the block tie, the lane tie, the wave tie, the vocabulary
mask, the K-quarter reduction, the softcap, the int4 high-half scale, the tail-row scale. All
twelve are killed, the control passes (`logs/qmut.log`, `MUTANTS ALL KILLED, CONTROL
PASSES`, exit 0).

`gptq_mutants.py` does the same on the CPU for the GPTQ solver and the calibrated-weights path
(no error compensation, wrong sign, no block propagation, error not scaled, no scale search,
file ignored, key without the layer, no format check, no shape check, scales not used): ten
mutants, ten killed, control passes (`logs/gptq_mut2.log`). The first run left "no error
compensation" ALIVE (the cross-block propagation alone still beat round to nearest by the
test's margin); the block-size independence test was added and kills it.

### 8.3 Tests, build, resources

- `pytest tests/kernels/quantization/test_inkling2_qlinear_rdna2.py -q --noconftest`:
  120 passed, 7 skipped ("shape not tiled by this instance") in 12.1 s on HIP 2
  (`logs/pytest_q2.log`, python exit 0; 110 passed before the 10 GPTQ / calibrated-pack tests
  were added, `logs/pytest_q.log`).
- `_rocm_C` registers `inkling2_qlinear_grouped` and `inkling2_qlinear_lm_head` (schemas
  checked on the installed library); the source is in the CMake whitelist and the `.hip`
  copy is built with `-O3`; build log `logs/build2.log`, exit 0 after 553 s.
- The 22 serving instances use 64 to 168 VGPRs, no spills and no scratch, 5 to 16 waves per
  SIMD (`logs/qres.log`).

### 8.4 What the real-model gate can and cannot see

It CAN fail: the all-int4 round-to-nearest build scored 0.1700 against the limit 0.1027
(section 7.2). It cannot resolve a change of the mean `|dlogp|` below about 0.03 (the
noise of two identical engines is 0.04): the signed paired shift and the 3999-position run
carry the fine resolution, and they show no cost of the int8 levers at the 1.6e-3 nats
level (se). One text (the port spec), greedy-decode kernels only, TP = 1: sampling with
temperature, TP > 1 (the flags are ignored there) and speculative verification were not
run end to end (the kernels take 1 to 4 rows and the oracle covers them).

## 9. Logs and reproduction

Everything ran on the box under `~/w4a8_runs/inkling2-work/Inkling2DecodeQuant/`
(`D` below) on HIP 2, through the lead's `gpu_sub.sh`.

| What | Log / output in `D` |
| --- | --- |
| baseline and int8 step profiles | `logs/base_prof.log`, `prof_base2.log`, `prof_all8.log`; `out/prof_*/` |
| weight error study | `out/qstudy.json` (`qstudy.py`, activations `out/acts.pt` from `dump_acts.py`) |
| microbenchmarks | `logs/bench1.log`, `bench2.log`, `bench_prod.log`, `floor1.log` |
| oracle, mutants, resources | `logs/qcheck1.log`, `qcheck2.log`, `chk_tool_jit.log`, `chk_ops.log`, `qmut.log`, `gptq_mut2.log`, `qres.log` |
| gate runs | `logs/g_*.log` and `out/g_*.json` (800 tokens), `logs/h_*.log`, `out/h_*.json` (4000 tokens) |
| GPTQ solve | `logs/gptq_w4g32.log`, `out/gptq_w4g32.pt`; corpus `calib/calib_text.md` |

Reproduce (from the repository root, on a gfx1030 box with the ops built):

```
# microbenchmark, cold weights, graph-timed, production instances
python tools/rdna/inkling2/qlinear_bench.py --families attn_in,o_proj,lm_head --ms 1,2,3,4
# oracle on the installed ops / on JIT-built kernels, and the hand mutants
python tools/rdna/inkling2/qlinear_check.py
python tools/rdna/inkling2/qlinear_check.py --jit
python tools/rdna/inkling2/qlinear_mutants.py --work "$SCRATCH/qmut"
# the real-model gate (one boot: NLL in 4-token chunks, generations, step time)
VLLM_INKLING2_ATTN_QUANT=w8 VLLM_INKLING2_LM_HEAD_QUANT=w8 \
  python tools/rdna/inkling2/decode_gate.py --ckpt "$CKPT" --chat-template-file "$CKPT/chat_template.jinja" \
    --nll-file "$NLL_TEXT" --nll-tokens 800 --nll-against "$TP1_A" --nll-floor "$TP1_B" \
    --cudagraph-mode FULL_AND_PIECEWISE --json out.json
# GPTQ int4 for the attention packs (flags off), then use it
python tools/rdna/inkling2/gptq_quant.py --ckpt "$CKPT" --text "$CALIB_TEXT" --skip-tokens 0 \
  --tokens 8000 --val-tokens 2000 --format w4g32 --out gptq_w4g32.pt
VLLM_INKLING2_QUANT_FILE=gptq_w4g32.pt VLLM_INKLING2_ATTN_QUANT=w4 ...
```
