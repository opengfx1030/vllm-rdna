# Inkling-2-Mini prefill expert GEMMs on RDNA2 (gfx1030)

`csrc/rocm/moe_prefill_bm_rdna2.cu` holds the expert GEMMs for **prefill-sized chunks** (an
expert receives tens to hundreds of routed rows): fp16 `v_dot2_f32_f16` kernels for NVFP4 and
MXFP8 experts that own up to 32 routed rows of one expert and the whole K range, and an
**experimental, opt-in** W4A8 NVFP4 kernel (int8 activations, `v_dot4_i32_i8`). The
decode-shaped kernels of `moe_lowp_dot2_rdna2.cu` keep every chunk below
`VLLM_INKLING2_PREFILL_BM_MIN_TOKENS` tokens, the graph-captured steps (1 .. 32 tokens)
included.

**The fp16 dot2 path is the default.** It reads the native checkpoint tensors exactly as the
shipped kernels do (the packed NVFP4 words and e4m3 group scales, the MXFP8 bytes and ue8m0
scales, in the layout the loader already produces: no requantisation, no calibration, no new
weights) and dequantises them in registers inside the GEMM. `VLLM_INKLING2_PREFILL_BM=0`
restores the shipped kernels (A/B and rollback). The W4A8 path derives int8 activations, so it
changes the numerics: it is opt-in (`VLLM_INKLING2_PREFILL_INT8=1`), off by default, and not
used by any recipe.

Measured on one V620 (HIP 5, TP = 1, 175 W cap), real checkpoint, 1000-token prompt, clean
step wall (`Inkling2PrefillProfile/scripts/profile_prefill.py`):

| path | clean ms | tok/s |
| --- | ---: | ---: |
| shipped kernels (`VLLM_INKLING2_PREFILL_BM=0`) | 873.6 / 880.4 / 883.4 | 1130 |
| **fp16 32-row kernels (default)** | 487.6 / 487.3 / 488.1 | **2051** |
| W4A8 NVFP4 + fp16 MXFP8 (`VLLM_INKLING2_PREFILL_INT8=1`, experimental) | 452.5 / 452.1 / 453.3 | 2210 |

The shipped row is the profile agent's baseline of the same checkpoint and prompt; the W4A8
row was measured with the same kernel source JIT-built before the `_rocm_C` build carried it.
Chunked prefill (`--mnbt 4096`, every chunk at depth 0), clean ms (tok/s), measured the same
way:

| chunk tokens | shipped | fp16 32-row (default) | W4A8 (experimental) |
| ---: | ---: | ---: | ---: |
| 184 | 215.7 (853) | 137.0 (1343) | 130.5 (1410) |
| 512 | 478.3 (1070) | 268.9 (1904) | 249.4 (2053) |
| 1024 | 905.0 (1131) | 496.9 (2061) | 460.6 (2223) |
| 2048 | 1836 (1115) | 1013.4 (2021) | 940.8 (2177) |
| 4096 | 3745 (1094) | 2078.3 (1971) | 1931 (2121) |

The MoE runs in calls of at most `ws.moe_chunk` = 1024 tokens, so a 4096-token chunk is four
MoE calls per layer visit. The default holds about 2 k tok/s at every chunk size. Decode is
untouched: the kernels only see chunks of at least 128 tokens, and prefill chunks are not
graph-captured (the capture grid ends at 32 tokens).

## 1. Switches

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_INKLING2_PREFILL_BM` | `1` | fp16 dot2 32-row kernels for every NVFP4 / MXFP8 layer whose chunk has at least `..._MIN_TOKENS` tokens. `0`: the shipped decode-shaped kernels (A/B and rollback). |
| `VLLM_INKLING2_PREFILL_INT8` | `0` | **Experimental.** W4A8 int8 kernel for the NVFP4 layers (0-19) instead of the fp16 one; MXFP8 layers stay fp16. Only meaningful while the dot2 path is on. Changes the numerics, see section 6.4. |
| `VLLM_INKLING2_PREFILL_BM_MIN_TOKENS` | `128` | The chunk size from which the 32-row kernels are used; smaller chunks (every decode and graph step) keep the decode-shaped kernels, see section 7.3. |

* The server logs the state: `grep "Inkling-2 MoE prefill kernels" serve.log` shows at load
  `... dot2 32-row on, W4A8 off, chunks of >= 128 tokens ...` and, at the first chunk that
  takes the path, `... kernels ACTIVE: fp16 dot2 32-row on a chunk of at least 128 tokens`.
  With `VLLM_INKLING2_PREFILL_BM=0` the load line says `off` and there is no `ACTIVE` line.
* A layer whose K does not fit falls back to the decode-shaped kernels: K of both stages
  must be a multiple of 128 (NVFP4) / 32 (MXFP8). At TP = 2 / 4 the intermediate size is
  sharded to 576 / 288, so the NVFP4 layers keep the shipped kernels and only the MXFP8
  layers (K = 288) change.
* The router, the workspace and the decode path are unchanged; no new weights are loaded
  or derived by the default path.

## 2. Launches per MoE layer visit

Chunk of at least 128 tokens, one launch per line (`ws.moe_chunk` = 1024 tokens per MoE
call):

| step | shipped (`..._BM=0`) | **default (fp16 dot2 32-row)** | W4A8 (experimental, NVFP4 layers) |
| ---: | --- | --- | --- |
| 1 | `inkling2_glue_moe_route`: router, top-k, 8-row grouping, zero fill of `gate_up` and of the output | `inkling2_glue_moe_route`: the same, **without** the `gate_up` zero fill (the w13 kernel writes every routed row) | as default |
| 2 | | `moe_prefill_cta_rdna2`: 32-row work list from the router's layout (one workgroup, the extra grouping op) | as default |
| 3 | | | `moe_prefill_act_quant_rdna2`: w13 input fp16 -> int8 + scales |
| 4 | `moe_{nvfp4,mxfp8}_gemm_rdna2` (w13) | `moe_{nvfp4,mxfp8}_bm_rdna2` (w13) | `moe_nvfp4_bm_i8_rdna2` (w13) |
| 5 | `inkling2_glue_swiglu` | `inkling2_glue_swiglu` | `inkling2_glue_swiglu` |
| 6 | | | `moe_prefill_act_quant_rdna2`: w2 input fp16 -> int8 + scales |
| 7 | `moe_{nvfp4,mxfp8}_gemm_rdna2` (w2) | `moe_{nvfp4,mxfp8}_bm_rdna2` (w2) | `moe_nvfp4_bm_i8_rdna2` (w2) |
| | **4 launches** | **5 launches** | **7 launches** (the MXFP8 layers: 5) |

* **The default path has no dequantisation, repack or quantisation launch in front of or
  behind the GEMMs.** Both stages read the packed checkpoint tensors directly and dequantise
  them in registers (NVFP4: one shift + mask per two weights, the e4m3 group scale applied
  with one `v_pk_mul_f16`; MXFP8: byte -> fp16 with the ue8m0 scale), the activation tile
  goes from the fp16 input to LDS inside the kernel. The only launch added to the shipped
  sequence is the work-list builder (step 2): one workgroup; the removed `gate_up` zero fill is part of the router launch, so it costs nothing; the removed `gate_up` zero fill is
  part of the router launch.
* The W4A8 path keeps its two activation-quantise passes (steps 3 and 6, fp16 -> int8 with
  one fp32 scale per 32-K window): about 38 us per NVFP4 layer visit at 1000 - 1024 tokens (11 us for the w13 input `[1024, 2304]`, 27 us for the w2 input `[8192, 1152]`, standalone, `bm_bench.py`), 1.7 ms of the step over the 44 NVFP4 visits [derived]. That is 0.4 % of the 452 ms step; the
  reason the path is opt-in is that it quantises the activations, which changes the numerics
  (section 6.4), not the cost of the passes. The weights are still dequantised inside the
  GEMM there (e2m1 code -> int8 through three `v_perm_b32`).

## 3. Why the decode-shaped kernel is slow at prefill

`moe_nvfp4_gemm_rdna2` / `moe_mxfp8_gemm_rdna2` run `block_size_m <= 8` routed rows per
expert block, split K across blocks and add the partial sums with fp16 atomics, and every
block dequantises every weight word it touches. At 1000 tokens an expert has ~62 rows
(median 31, maximum 568 - 999 on the real routing of layers 6 / 17): ~8x the dequantisation
work a 62-row GEMM needs, a zeroed `gate_up` buffer and CAS atomics on the w13 output. It
reaches 9 - 10 TFLOP/s on a part whose fp16 dot2 ceiling is ~25 TFLOP/s (31.2 TFLOP/s at
1824 MHz, ~25 at the 175 W cap).

## 4. The 32-row kernels

* **Block** = 128 threads x 4 columns, up to 32 rows of ONE expert, the whole K range (no K
  split: no atomics for w13, no zero fill of `gate_up`, one dequantisation of every weight
  word per block). `acc[32][4]` fp32, one `v_dot2` chain per (row, column).
* **Work list** (`moe_prefill_cta_rdna2(expert_ids, num_tokens_post_padded, E,
  block_size_m)` -> int32 `[blocks, 4]` = {expert, first sorted row, valid rows, 0}): groups
  up to `32 / block_size_m` consecutive blocks of one expert of the **router's own layout**
  (`inkling2_glue_moe_route`, `block_size_m = 8`), so the router, the workspace views and the
  decode path are unchanged. Full 32-row blocks come first (expert order), the partial last
  block of every expert after them, longest first (the tail of the grid is made of the
  cheapest blocks). Inside a block the kernel skips the 8-row groups past the last valid row.
* **NVFP4** (`b_q_weight [E, K/8, N]` uint32, `b_scales [E, K/16, N]` e4m3, `K % 128 == 0`):
  nibble j and j + 4 of a word share one `half2` after the activation tile is stored with
  the K order `(0,4,1,5,2,6,3,7)` inside every 8-K word (one shift + mask dequantises two
  weights); weights are prefetched two 16-K groups ahead in registers over the whole K range,
  the activation tile (32 x 128 K) is double-buffered in LDS (next chunk loaded into
  registers before the compute of the current one, stored after it, one barrier per chunk).
* **MXFP8** (`b_q_weight [E, K/4, N]`, `b_scales [E, K/32, N]` ue8m0, `K % 32 == 0`): 32 K
  per step (one scale byte per column), activation tile 32 x 256 K, natural K order.
* Contract of the shipped kernels, kept: w13 `top_k = 8, output_topk = 0` (plain stores, one
  row per routed id), w2 `top_k = 1, mul_topk_weight, output_topk = 8` (router weight, fp16
  CAS add of the 8 rows of a token into a zeroed output).
* gfx1030 only: every other arch compiles empty stubs of the same ops (section 9).

## 5. Quality of the default path

The default kernels must be numerically equal to or better than the shipped ones: the
dequantisation is the shipped one (one fp16 rounding of a dequantised weight, the same group
scales), the dot products accumulate in fp32 over the whole K range (the shipped kernels
split K across blocks and add fp16 partial sums), the fp16 results are rounded once.

### 5.1 Against an fp64 oracle, same inputs, real shapes

Relative L2 error against fp64 (the dequantised weights without the fp16 rounding, 24 sampled
tokens), T = 1024, uniform routing, random weights and activations, the same inputs for every
column (`bm_bench.py --check`):

| format | stage | shipped decode-shaped | **fp16 32-row (default)** | W4A8 (experimental) |
| --- | --- | ---: | ---: | ---: |
| NVFP4 | w13 | 6.09e-4 | **3.70e-4** | 5.36e-3 (2.07e-4 against the int8-activation oracle) |
| NVFP4 | w2 | 1.06e-3 | **5.91e-4** | 5.38e-3 (5.00e-4 against the int8-activation oracle) |
| MXFP8 | w13 | 5.26e-4 | **2.07e-4** | |
| MXFP8 | w2 | 1.02e-3 | **5.00e-4** | |

The default path is 1.6 - 2.5x closer to the oracle than the shipped kernels in every row (the
shipped kernels add fp16 partial sums of the split K). The W4A8 error against the fp16
activations is the activation quantisation (about 9x the fp16 kernel's); the kernel itself
matches an oracle that multiplies the int8 activations to 2e-4 / 5e-4.

Small chunks on the real checkpoint layers (`small_t_oracle.py`: the engine's own `repack_nvfp4` /
`repack_mxfp8` on the checkpoint tensors of layers 0, 6, 17 (NVFP4) and 20 - 23 (MXFP8); T = 33,
40, 70, 97 and 128 with uniform routing and T = 70 and 128 with a skewed routing (up to 39 rows
of one expert: full 32-row blocks plus partial tails); synthetic activations; 24 evenly spread
tokens, the last one included; the w2 stage includes the fp16 CAS accumulation; mean of 3
repeats), relative L2 against fp64, 98 cases:

| format | stage | shipped (mean) | 32-row (mean) | worst ratio 32-row / shipped |
| --- | --- | ---: | ---: | ---: |
| NVFP4 (layers 0, 6, 17) | w13 | 5.95e-4 | 3.45e-4 | 0.59 |
| NVFP4 (layers 0, 6, 17) | w2 | 1.06e-3 | 5.75e-4 | 0.55 |
| MXFP8 (layers 20 - 23) | w13 | 5.27e-4 | 2.08e-4 | 0.40 |
| MXFP8 (layers 20 - 23) | w2 | 1.02e-3 | 5.01e-4 | 0.50 |

The 32-row kernel is closer to fp64 than the shipped one in all 98 cases (also in the worst of
the 3 repeats, and in max |error| / rms: worst ratio 0.65). At the shapes of the tiny test models
(K 256 / 128, N 256, 8 experts, top-2; 36 format x stage x token cases, T = 1 to 128) both
kernels sit on the fp16 output rounding floor (NVFP4 3.7e-4 / 4.3e-4, MXFP8 2.1e-4 / 3.0e-4)
and are equal to within 0.1 % (largest 32-row / shipped ratio 1.0007).

### 5.2 Teacher-forced NLL on the real checkpoint

`tools/rdna/inkling2/real_check.py --nll-file docs/rdna2/inkling2-port-spec.md --nll-tokens
800` (799 positions) against the shipped kernels' run A; the floor is a second run B of the
shipped kernels (the run-to-run noise of the shipped path); gate: mean |dlogp| <= 2 x floor +
0.01 = 0.1054.

| run | mean \|dlogp\| | p99 | max | mean logp |
| --- | ---: | ---: | ---: | ---: |
| B (shipped, run-to-run floor) | 0.0477 | 0.354 | 0.882 | -1.9801 |
| **default config, `_rocm_C` build of this patch** | 0.0458 | 0.386 | 0.645 | -1.9829 |
| fp16 32-row, kernels JIT-built from the same source | 0.0470 | 0.323 | 0.483 | -1.9826 |
| W4A8, kernels JIT-built from the same source (experimental) | 0.0574 | 0.382 | 1.210 | -1.9784 |

(A mean logp: -1.9833.) The default is inside the floor of the shipped path. One text of 799
positions is a thin basis; the greedy-equality and long-prompt checks are in section 5.4.

### 5.3 Determinism

Six launches on the same inputs (real stage shapes, uniform routing, 128 experts), every
launch compared bitwise with the first (`det_check.py`, fraction of fp16 output elements that
differ):

| stage | format | tokens | **32-row kernel** | shipped decode-shaped | W4A8 |
| --- | --- | ---: | ---: | ---: | ---: |
| w13 | NVFP4 | 1000 / 4096 | **0** | 0 | 0 |
| w13 | MXFP8 | 1000 / 4096 | **0** | 0 | |
| w2 | NVFP4 | 1000 | 17.5 % | 4.9 % | |
| w2 | NVFP4 | 4096 | 2.1 % | 0.057 % | |
| w2 | MXFP8 | 1000 | 26.7 % | 5.8 % | |
| w2 | MXFP8 | 4096 | 3.2 % | 0.090 % | |

The first stage is bitwise repeatable (`test_up_stage_is_bitwise_repeatable`). The second stage
adds the top-k rows of a token with fp16 CAS atomics in the order the blocks finish, exactly as
the shipped kernel does, so repeat runs differ in the last fp16 bits for both; the 32-row
kernel's blocks differ more in duration (the work list puts the partial blocks last), so more
elements see another add order. Every difference is one fp16 rounding step of a sum of eight
terms (magnitudes below). A bitwise repeatable w2 needs a fixed-order reduction of the eight
rows (see section 10).

Magnitude (`det_check.py` v2, five repeats against the first launch; `|diff|` relative to the rms
of the sampled output): the largest difference is the same size for both kernels (NVFP4 w2 at
T = 1000: 8.8e-3 for both, at 4096: 8.8e-3 against 4.4e-3; MXFP8: 5.7e-3 against 8.5e-3 and
5.6e-3 for both), one fp16 rounding step of a few elements; the mean difference is 1.1e-4
(32-row) against 3.9e-5 (shipped) of the rms at T = 1000 and 1.3e-5 against 4.0e-7 at
T = 4096. The fp64 error of the same outputs is lower for the 32-row kernel at both token
counts (NVFP4 5.8e-4 against 1.05e-3, MXFP8 5.0e-4 against 1.02e-3). On the tiny models of the
model-level test the repeat runs are bitwise identical for both kernels.

### 5.4 Greedy equality

Served through `serve_rdna.sh RECIPE=inkling2-mini-tp1` on HIP 5, a 1000-token prompt of counting
numbers (low entropy), greedy: the 32 generated tokens equal the shipped kernels' (cold and
warm), the 96 tokens of a short decode equal, the largest logprob difference of the sampled
tokens is 6.2e-3; the prefix cache hits 960 of 1000 tokens, graph capture takes 2 s and
0.70 GiB, decode 20.2 ms per token, all identical with `VLLM_INKLING2_PREFILL_BM=0`. A
high-entropy prompt (the head of `inkling2-port-spec.md`) flips at the first token (logprob
-0.695 against -1.186, a near tie) in a run where the shipped kernels themselves answered two
identical 16-token requests differently (the run-to-run noise of the shipped path).

Long prompts (`long_req.py`: the head of the concatenated `docs/rdna2/*.md`, 16384 and 32768
tokens, 48 greedy tokens; `serve_long.sh` boots on HIP 5; two boots of each configuration, so
the floor is a pair of shipped boots). `off` = `VLLM_INKLING2_PREFILL_BM=0`, `on` = default.
The 32768-token request reuses the 16384-token prefix from the prefix cache.

| length | pair | tokens equal | first divergence | mean / max \|dlogprob\| before it | logprobs of the two choices |
| --- | --- | ---: | ---: | ---: | ---: |
| 16384 | off A vs off B (floor) | 10 / 48 | 10 | 5.2e-2 / 1.9e-1 | -1.884 / -1.853 |
| 16384 | on C vs on D | 38 / 48 | 38 | 3.7e-2 / 2.8e-1 | -1.494 / -1.510 |
| 16384 | on C vs off A | 10 / 48 | 10 | 2.2e-2 / 6.8e-2 | -1.901 / -1.884 |
| 16384 | on C vs off B | 38 / 48 | 38 | 2.7e-2 / 2.0e-1 | -1.494 / -1.470 |
| 16384 | on D vs off A | 10 / 48 | 10 | 5.2e-2 / 2.7e-1 | -1.885 / -1.884 |
| 16384 | on D vs off B | 48 / 48 | none | 2.9e-2 / 2.4e-1 | |
| 32768 | off A vs off B (floor) | 22 / 48 | 20 | 2.6e-2 / 9.5e-2 | -1.604 / -1.553 |
| 32768 | on C vs on D | 48 / 48 | none | 2.0e-2 / 9.5e-2 | |
| 32768 | on C vs off A | 33 / 48 | 33 | 2.2e-2 / 1.8e-1 | -0.698 / -0.704 |
| 32768 | on C vs off B | 21 / 48 | 20 | 1.9e-2 / 9.5e-2 | -1.579 / -1.553 |
| 32768 | on D vs off A | 33 / 48 | 33 | 2.4e-2 / 1.1e-1 | -0.710 / -0.704 |
| 32768 | on D vs off B | 21 / 48 | 20 | 2.6e-2 / 1.0e-1 | -1.647 / -1.553 |

Every divergence is a near tie (the two chosen tokens differ by 0.001 to 0.094 nats), the
earliest divergence of the default kernels from the shipped ones (token 10 at 16384, 20 at
32768) is the earliest divergence of the two shipped boots from each other (10 and 20), and
the default kernels' own two boots agree for 38 and 48 tokens: within the floor of the shipped
path. The four serve logs have no evict, KFD, fault or hang line. Wall time of a request
(prompt + 48 tokens): 16384 tokens 16.3 / 16.4 s shipped against 10.2 / 10.2 s default; the
32768-token request (second half computed) 18.7 / 18.8 s against 12.5 / 12.6 s.

## 6. W4A8 prototype (experimental, `VLLM_INKLING2_PREFILL_INT8=1`, off by default)

gfx1030 has no fp4 / fp8 hardware but `v_dot4_i32_i8` runs at twice the dot2 rate.

* `moe_prefill_act_quant_rdna2(a)`: fp16 `[rows, K]` -> int8 `[rows, K]` + fp32 scales
  `[rows, K/32]`, scale = absmax / 127 of the (row, 32 K) window, round to nearest even, K
  order inside every 8-K word `(0,2,4,6 | 1,3,5,7)`.
* `moe_nvfp4_bm_i8_rdna2`: the same blocking. A weight word gives `E = w & 0x0F0F0F0F`
  (nibbles 0,2,4,6) and `O = (w >> 4) & 0x0F0F0F0F` (1,3,5,7); three `v_perm_b32` (positive
  table, negative table, select by the sign bit) turn 4 e2m1 codes into 4 int8 = 2 x value.
  `dot4(E, A_even) + dot4(O, A_odd)` over a 16-K group is the integer `P_g`. The e4m3 scale of
  a group is `M * 2^(E - 10)`; over the two groups of a 32-K window (the activation scale
  window) the multipliers are aligned to the larger exponent, `Mm = (M << 6) >> (E_max - E)`
  (exact up to a gap of 2 octaves, truncated below), `T = Mm_0 P_0 + Mm_1 P_1`
  (`v_mul_i32_i24` + `v_mad_i32_i24`), and one fp32 flush per (row, column, window):
  `acc += float(T) * (sa_row * global * 2^(E_max - 17))`. The first dot of every chain is the
  untied VOP3P `v_dot4_i32_i8 ..., 0` (inline asm; the compiler's tied `v_dot4c` needs a
  `v_mov 0` first: -13 % instructions).
* MXFP8 layers (20-23) keep the fp16 kernel: an e4m3 weight needs more than 8 bits once the
  per-element exponent is kept.
* **Numerics change.** Random-data error against fp64: 5.4e-3 relative L2 (w13 and w2) against 3.7e-4 / 5.9e-4 for the fp16 kernels (section 5.1); NLL
  0.0574 against the floor of 0.0477 (section 5.2: inside the gate, but the activations are
  quantised, so it is a quality change). A NaN / Inf in an activation window is not
  propagated by the quantiser (`fmaxf` ignores a NaN, the clamp maps it to -127); the fp16
  kernels propagate it. Measured gain over the fp16 default: 1.08x at the 1000-token step
  (487.6 -> 452.5 ms), 1.19x on the expert GEMMs. Re-run the NLL gate on the serving prompts
  before enabling it.

## 7. Kernel measurements

Microbenchmark (`Inkling2PrefillInt8/bm_bench.py`, JIT build of the production variants:
fp16 NVFP4 `32:100:8`, MXFP8 `32:0:8`, W4A8 `32:303:8`): graph-free launches timed with
events, **cold rotating weights (> 700 MB of experts)**, the launch cost of an empty kernel
subtracted, min of 5 interleaved rounds x 15 iterations (the run-to-run drift of the GPU
state is +-10 %, compare only inside one table). FLOP = 2 x 8 T x K x N (w13: K = 2304, N =
2304; w2: K = 1152, N = 2304).

### 7.1 NVFP4, microseconds per launch (TFLOP/s)

| stage | T | routing | shipped BM8 | fp16 32-row | W4A8 | W4A8 / fp16 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| w13 | 256 | uniform | 2178 (10.0) | 1190 (18.3) | 1152 (18.9) | 1.03 |
| w13 | 1024 | uniform | 9739 (8.9) | 4316 (20.2) | 3642 (23.9) | 1.19 |
| w13 | 4096 | uniform | 37450 (9.3) | 15187 (22.9) | 12740 (27.3) | 1.19 |
| w13 | 1024 | skew 2 | 9241 (9.4) | 4323 (20.1) | 3641 (23.9) | 1.19 |
| w13 | 4096 | skew 2 | 37528 (9.3) | 15220 (22.9) | 12747 (27.3) | 1.19 |
| w13 | 1000 | real layer 6 | 8988 (9.5) | 4202 (20.2) | 3536 (24.0) | 1.19 |
| w13 | 1000 | real layer 17 | 8700 (9.8) | 4023 (21.1) | 3389 (25.1) | 1.19 |
| w13 | 1000 | real layer 0 | 8982 (9.5) | 4232 (20.1) | 3565 (23.8) | 1.19 |
| w13 | 1000 | real layer 44 | 8842 (9.6) | 4152 (20.5) | 3503 (24.3) | 1.19 |
| w2 | 256 | uniform | 1395 (7.8) | 746 (14.6) | 656 (16.6) | 1.14 |
| w2 | 1024 | uniform | 4925 (8.8) | 2226 (19.5) | 1922 (22.6) | 1.16 |
| w2 | 4096 | uniform | 18722 (9.3) | 8074 (21.5) | 6817 (25.5) | 1.18 |
| w2 | 1024 | skew 2 | 4904 (8.9) | 2218 (19.6) | 1913 (22.7) | 1.16 |
| w2 | 4096 | skew 2 | 18875 (9.2) | 8096 (21.5) | 6849 (25.4) | 1.18 |
| w2 | 1000 | real layer 6 | 4455 (9.5) | 2161 (19.7) | 1853 (22.9) | 1.17 |
| w2 | 1000 | real layer 17 | 4308 (9.9) | 2074 (20.5) | 1781 (23.8) | 1.16 |
| w2 | 1000 | real layer 0 | 4452 (9.5) | 2172 (19.6) | 1866 (22.8) | 1.16 |
| w2 | 1000 | real layer 44 | 4393 (9.7) | 2134 (19.9) | 1833 (23.2) | 1.16 |

Real routing = the top-8 ids captured by the profile agent on the 1000-token prompt
(`Inkling2PrefillProfile/out/prefill_a/moe_capture.pt`): layer 6 has 568 rows in its largest
expert and 11 empty experts (median 31), layer 17 has 999 / 49 / 2, layer 0 482 / 0 / 42,
layer 44 609 / 13 / 20. The kernels are insensitive to the distribution (the work list
removes the padding of the router layout). TFLOP/s counts the 2 x MAC of the
bf16-equivalent GEMM, also for W4A8. Per layer visit at 1000 tokens (mean of the four real
layers): shipped 8878 + 4402 = 13280 us, fp16 4152 + 2135 = 6287 us, W4A8 3498 + 1833 =
5331 us; times 44 NVFP4 visits: 584 / 277 / 235 ms per forward.

### 7.2 MXFP8 (fp16 kernel), microseconds per launch (TFLOP/s), uniform routing

| stage | T | shipped BM8 | fp16 32-row |
| --- | ---: | ---: | ---: |
| w13 | 256 | 1896 (11.5) | 1796 (12.1) |
| w13 | 1024 | 8156 (10.7) | 4781 (18.2) |
| w13 | 4096 | 30601 (11.4) | 16220 (21.5) |
| w2 | 256 | 1130 (9.6) | 971 (11.2) |
| w2 | 1024 | 4093 (10.6) | 2423 (18.0) |
| w2 | 4096 | 15375 (11.3) | 8605 (20.2) |

### 7.3 Small chunks (where the 32-row kernels start to pay off)

Uniform routing, microseconds per launch; "shipped" is the best of `block_size_m` 1 / 2 / 4
/ 8 (the router's table picks 4 below 64 tokens for NVFP4, 2 below 48 for MXFP8, else 8):

NVFP4:

| tokens | w13 shipped (best `block_size_m`) | w13 32-row | speed-up | w2 shipped | w2 32-row | speed-up |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 683 (BM4) | 681 | 1.00 | 329 (BM4) | 351 | 0.94 |
| 32 | 892 (BM4) | 879 | 1.01 | 439 (BM4) | 455 | 0.97 |
| 48 | 1031 (BM4) | 962 | 1.07 | 504 (BM4) | 492 | 1.02 |
| 64 | 1242 (BM4) | 1013 | 1.23 | 595 (BM4) | 517 | 1.15 |
| 80 | 1384 (BM8) | 1012 | 1.37 | 688 (BM8) | 523 | 1.31 |
| 96 | 1438 (BM8) | 1033 | 1.39 | 740 (BM8) | 530 | 1.39 |
| 128 | 1802 (BM8) | 1072 | 1.68 | 910 (BM8) | 550 | 1.65 |
| 192 | 2423 (BM8) | 1251 | 1.94 | 1222 (BM8) | 641 | 1.91 |
| 256 | 2958 (BM8) | 1447 | 2.04 | 1478 (BM8) | 731 | 2.02 |

MXFP8:

| tokens | w13 shipped (best `block_size_m`) | w13 32-row | speed-up | w2 shipped | w2 32-row | speed-up |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1116 (BM2) | 1138 | 0.98 | 542 (BM1) | 568 | 0.95 |
| 32 | 1458 (BM2) | 1490 | 0.98 | 732 (BM2) | 768 | 0.95 |
| 48 | 1631 (BM2) | 1628 | 1.00 | 784 (BM2) | 804 | 0.97 |
| 64 | 1672 (BM8) | 1653 | 1.01 | 832 (BM8) | 849 | 0.98 |
| 80 | 1706 (BM8) | 1674 | 1.02 | 841 (BM8) | 856 | 0.98 |
| 96 | 1750 (BM8) | 1700 | 1.03 | 865 (BM8) | 881 | 0.98 |
| 128 | 1888 (BM8) | 1772 | 1.07 | 915 (BM8) | 911 | 1.00 |
| 192 | 2357 (BM8) | 1841 | 1.28 | 1165 (BM8) | 957 | 1.22 |
| 256 | 2797 (BM8) | 1925 | 1.45 | 1412 (BM8) | 996 | 1.42 |

NVFP4 breaks even at about 48 tokens and wins by 1.2x from 64; MXFP8 is equal at 128 tokens
and wins from 192. `VLLM_INKLING2_PREFILL_BM_MIN_TOKENS` = 128 keeps every chunk below the
MXFP8 break-even and every graph-captured step on the shipped kernels; NVFP4 chunks of 64 -
127 tokens give up up to 1.65x on a layer visit that is a small part of a prefill.

### 7.4 Resources and instruction mix (gfx1030, `-O2`; `-O3` gives the same code)

| kernel | VGPR | SGPR | LDS B | waves/SIMD | spills | innermost loop |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `nvfp4_prefill_kernel<32,128,128>` | 231 | 38 | 16512 | 4 | 0 | 2631 instr / 2 groups of 16 K, 1920 `v_dot2c` = 73 % of instr, 83 % of VALU |
| `mxfp8_prefill_kernel<32,128,256>` | 243 | 52 | 16512 | 4 | 0 | 2703 instr / 32 K, 1920 `v_dot2c` = 71 % of instr, 81 % of VALU |
| `nvfp4_i8_prefill_kernel<32,128,256>` | 234 | 49 | 9344 | 4 | 0 | 2180 instr / 32 K window: 1024 dot4 (47 %), 640 flush ops (mul24, mad24, cvt, mul, fmac), 356 dequant / scale decode, 64 LDS reads |
| `moe_prefill_act_quant_kernel` | 38 | 18 | 0 | 16 | 0 | |
| `moe_prefill_cta_kernel` | 24 | 42 | 5152 | 16 | 0 | one workgroup |

The kernels are **instruction-issue bound** (the DRAM floor of w13 is ~1.2 ms per visit at
382 MB of weights, the dot2 issue floor ~3.2 ms): the W4A8 kernel's measured gain (1.19x)
follows its instruction count, not the 2x dot rate. Variants measured and dropped (T = 1024
uniform, w13 / w2 us, medians):

| variant | w13 | w2 | verdict |
| --- | ---: | ---: | --- |
| shipped BM8 | 8895 | 4846 | |
| 32 rows, BM-padded layout, v2 kernel | 4381 | 2283 | |
| 32 rows, work list, v2 | 4300 | 2251 | |
| 32 rows, work list, v3 (2-group prefetch + double-buffered tile) | 4198 | 2186 | **shipped** |
| 16 rows | 4784 | 2456 | dequantisation 2x per row |
| 96 columns per block | 4621 | 2425 | |
| 64 rows | -- | -- | 256 accumulators: spills |
| v3 with K chunk 256 | -- | -- | 255 VGPR + scratch |
| W4A8 untied `v_dot4c` (MODE 0) | 3800 | 1996 | +13 % instr (`v_mov 0` per chain) |
| W4A8 VOP3P first dot (MODE 1, shipped) | 3569 | 1877 | **shipped** |
| W4A8 + shared nibble prep (MODE 3) / K chunk 512 | 3558 / 3548 | 1882 / 1872 | no gain |

### 7.5 Where the rest of the 1000-token step goes (W4A8 profile, GPU ms)

NVFP4 experts 211.5 (w13 138.6, w2 72.9), attention input pack 67.9, attention 34.2, o_proj
27.8, message gates 24.6, MXFP8 experts 26.0, router 12.3, attention residual 11.5, swiglu
4.8; the 21 ms between the GPU kernel sum and the step wall is host time. The goal of 2.5 k
tok/s (400 ms) is not reached by the expert kernels alone: the dense GEMMs run at 26 TFLOP/s
(83 - 86 % of the dot2 peak) and the glue families are separate work.

### 7.6 End to end, default config (the `_rocm_C` build of this patch)

`profile_prefill.py` of the prefill profile on HIP 5 (TP = 1, graphs on, prefill chunks eager,
3 runs each, clean = without the profiler):

| prompt / chunk | ms per forward (3 runs) | tokens per second | kernels per step |
| --- | ---: | ---: | ---: |
| 1000 tokens, text | 488.3 / 488.2 / 488.4 | 2048 | 1075 |
| 1000 tokens, random ids | 476.1 / 475.0 / 475.0 | 2105 | |
| chunk 184 | 135.5 / 138.6 / 137.6 | 1338 | 1074 |
| chunk 512 | 268.7 / 269.9 / 269.8 | 1898 | |
| chunk 1024 | 497.9 / 497.4 / 498.2 | 2057 | 1074 |
| chunk 2048 | 1015.2 / 1014.7 / 1016.0 | 2017 | 1315 |
| chunk 4096 | 2079.8 / 2083.2 / 2085.8 | 1966 | 1794 |

The shipped kernels take 873 - 883 ms for the 1000-token prompt on the same harness (the
prefill profile's measurement; not repeated in this run). The ops this path adds to the shipped
four cost (`launch_cost.py`): the work list 9.4 us per layer visit at T = 1000 (9.5 us at 1024);
the activation quantiser of the experimental W4A8 path 10.0 us (w13 input [1000, 2304]) + 26.3 us
(w2 input [8000, 1152]) = 36.3 us per NVFP4 visit.

## 8. Tests and mutants

`tests/kernels/core/test_inkling2_moe_prefill_bm_rdna2.py` (304 tests, gfx1030 only):
fp64 oracle for both formats, both stages, uniform and skewed routing (three experts take
every token: many 32-row blocks + a partial one; four experts get none), 1 .. 4096 tokens
(counts that are no multiple of 8: padded rows inside the last group; one row past a block;
a sweep of 39 token counts), a K tail (MXFP8: K = 288) and an N tail (N = 1056), canary rows
behind the output (a padded row that leaks into the output is caught), the W4A8 kernel
against an oracle that multiplies the dequantised int8 activations the quantiser produced,
the quantiser against an fp32 reference (scale exactly, int8 within 1 LSB on < 0.1 % of the
values), the work list against a Python reference (block sizes 1 .. 16), repeat-run
determinism and the argument checks. `tests/models/inkling2_loop/test_moe_prefill_bm_gpu.py`
runs the serving model on router-stable tiny checkpoints with the kernels off / on / W4A8 and
proves the path with op call counters. It compares the error against the fp32 transformers
reference PAIRED over 8 checkpoints x 6 prompts per variant: the median and the one-sided
99 % bound of the error ratio (dot2 prefill / decode kernels) of the pairs whose router
decisions agree must stay below 1.04 and 1.06 (measured: tiny1 0.996 / 1.008, tiny8 1.005 /
1.027). A pair whose top-k expert set differs in one token is a discrete router event, not an
accuracy pair (5 of 120 pairs over 10 checkpoints, error ratio 1.7 to 7, the router
amplifying one rounding difference); at most a fifth of the pairs may flip (measured 1 and 4
of 48). A single-seed bound cannot separate these cases: a seed-0 pair measured +6 % while the
kernels differ by 0.1 % at those shapes. `test_moe_prefill_flags.py` is the CPU gate of the
switches (defaults, rollback flag, token gate, K eligibility).

```bash
python -m pytest tests/kernels/core/test_inkling2_moe_prefill_bm_rdna2.py -q   # needs _rocm_C with the ops
I2_MOE_PREFILL_JIT=csrc/rocm/moe_prefill_bm_rdna2.cu \
    python -m pytest tests/kernels/core/test_inkling2_moe_prefill_bm_rdna2.py -q  # JIT build of the file
python tools/rdna/inkling2/moe_prefill_bm_mutants.py --jobs 4                    # all mutants must be killed
python tools/rdna/inkling2/moe_prefill_bm_jit.py --archs "gfx1030;gfx1100;gfx90a;gfx942;gfx950;gfx1200"
```

Hand mutants of the kernel source (`moe_prefill_bm_mutants.py`; each is built with the JIT
helper and run against the kernel test; the unmutated file passes):

| mutant | what it breaks |
| --- | --- |
| `perm_k8_selector` | activation K permutation of a 8-K word (NVFP4) |
| `nibble_sign_shift` | sign bit of a dequantised nibble goes to bit 14 |
| `global_scale` | per-expert global scale of NVFP4 halved |
| `e4m3_scale_mask` | e4m3 scale mantissa bit dropped |
| `weight_prefetch_group` | second register set prefetches the wrong group |
| `k_chunk_count` | the last K chunk is never computed |
| `mxfp8_block_factor` | UE8M0 block scale off by 2 |
| `mxfp8_lds_row_offset` | activation window of the 32-K step |
| `mxfp8_sign` | e4m3 sign bit of the second byte |
| `padding_row_valid` | the padding id (token == size_m) is accepted as a row |
| `router_weight` | w2: router weight not applied |
| `topk_reduction_row` | w2: output row of a routed id |
| `atomic_add_is_store` | w2: the top-k rows overwrite instead of adding |
| `row_groups_counted` | only the first 8-row group of a block is computed |
| `work_list_row_offset` | first row of the second full work block of an expert |
| `work_list_partial_order` | partial blocks shortest first |
| `work_list_expert_search` | expert of a full block off by one |
| `quant_pair_order` | act quantiser: K order inside an 8-K word |
| `quant_inv_scale` | act quantiser: 126 / amax instead of 127 / amax |
| `lut_neg_table` | e2m1 LUT: one entry of the negative table |
| `lut_sign_select` | e2m1 LUT: sign bit position of the select |
| `lut_odd_nibbles` | odd nibbles dequantised like the even ones |
| `scale_align_mm1` | second group aligned by the first group's exponent gap |
| `scale_align_mm0` | first group multiplier off by a factor 2 |
| `scale_exponent` | window scale 2^(Em - 16) instead of 2^(Em - 17) |
| `window_scale_index` | activation scale of window wi of a chunk |
| `dot_chain_init` | second dot chain starts from the first one's sum |
| `act_tile_offset` | int8 activation tile: store offset of a 16-K piece |

28 mutants. All 28 are killed on the committed (clang-formatted) kernel source: the final
re-run (`moe_prefill_bm_mutants.py --phase test`, 29 results) passes the unmutated file and
fails every mutant (17 of the kernel / work-list / common code, 11 of the W4A8 kernel; the
runner asserts that every pattern matches the committed file exactly once). On HIP 5 against
the `_rocm_C` build of this patch: the kernel file 304 passed in 114 s, the model-level file
30 passed in 27 s.

## 9. Six-arch build and archdiff

`moe_prefill_bm_jit.py --archs "gfx1030;gfx1100;gfx90a;gfx942;gfx950;gfx1200"` compiles the file for
all six arches (`BUILD OK`); the `_rocm_C` build of this patch compiles it with the other ROCm
sources (1 new object, the objects that include `ops.h` rebuilt).

Receipt of `tools/rdna/archdiff.sh` (BASE `origin/rdna_extras` 9031c79a04, HEAD the merge head
plus this change; `EXPECT='_kernel|inkling2|_rdna2|moe_prefill_bm'`, `MAY_DIFFER` = every arch;
12 objects compiled and disassembled per arch; 844 s; exit 0, "PASS: 262 differing functions,
all expected"):

| arch | functions base / head | identical | changed | only in base | only in head | verdict |
|---|---|---|---|---|---|---|
| gfx90a | 121 / 313 | 121 | 0 | 0 | 192 | expected only |
| gfx942 | 121 / 313 | 121 | 0 | 0 | 192 | expected only |
| gfx950 | 121 / 313 | 121 | 0 | 0 | 192 | expected only |
| gfx1030 | 121 / 373 | 121 | 0 | 0 | 252 | expected only |
| gfx1100 | 121 / 313 | 121 | 0 | 0 | 192 | expected only |
| gfx1200 | 121 / 313 | 121 | 0 | 0 | 192 | expected only |

`origin/rdna_extras` predates the Inkling-2 port, so every Inkling-2 function (this change and
the attention, glue, linear, vit, dspark and decode-expert files) is "only in head"; the 121
functions the two trees share are bit-identical on all six arches (0 changed, 0 after
masking). The functions of this change:

| function | gfx1030 | gfx90a / gfx942 / gfx950 / gfx1100 / gfx1200 |
|---|---|---|
| `nvfp4_prefill_kernel<32,128,128>` | 5898 instructions | empty stub (no code) |
| `mxfp8_prefill_kernel<32,128,256>` | 5769 | empty stub |
| `nvfp4_i8_prefill_kernel<32,128,256>` | 5159 | empty stub |
| `moe_prefill_cta_kernel` | 584 | 678 to 845 (plain C++, compiled everywhere) |
| `moe_prefill_act_quant_kernel` | 255 | 265 to 294 (plain C++, compiled everywhere) |

The two helper kernels use no RDNA2 instruction, so they are real code on every arch and only
the three dot2 / dot4 GEMM kernels are stubs elsewhere. The report covers code generation
only: no kernel ran and no GPU was touched.

## 10. Limits, risks, open items

* The work-list kernel needs `expert_ids` ascending (true for the glue router and
  `moe_align_block_size`) and `num_experts <= 256`.
* The w2 stage adds the top-k rows of a token with fp16 CAS atomics like the shipped kernel:
  repeat runs on the same inputs are not bitwise equal (section 5.3). A deterministic
  variant follows the repo's `VLLM_RDNA2_MOE_DETERMINISTIC=1` (`docs/rdna2/determinism-2026-11-04.md`:
  w2 with `output_topk = 0`, one fp16 row per (token, expert) pair with plain stores, then a
  fixed-order fp32 sum of the rows in top-k order); it needs a second launch and is not part of
  this change.
* The kernels are not wired for graph capture of prefill (decode chunks are below the
  threshold); the ops allocate their outputs (`torch::empty`) like the glue router.
* Open: dense `attn_in` / `o_proj` (96 ms of the 1000-token step, fp16 `[N, K]` weights at
  25 - 28 TFLOP/s) are not touched by this change.
