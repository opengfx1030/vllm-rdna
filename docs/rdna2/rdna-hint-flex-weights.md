# rdna_hint_flex weights on RDNA2 (HadamardINT2-FP16 spec v3.2)

`quant_method: rdna_hint_flex` (alias `hadamard_int2`) loads checkpoints
written by the separate `tools/hi2` quantizer (spec:
`docs/rdna2/hadamard-int2-v3.2-spec.md` on the `feat/hadamard-int2-weights`
branch). This page covers the vLLM side: the loader, the gfx1030 kernels,
the decisions taken where the spec is open, and how it was checked.

## Name

The canonical `quant_method` is `rdna_hint_flex`; `hadamard_int2` (the name
the first quantizer releases write) is an accepted alias. Both live in
`vllm/model_executor/layers/quantization/rdna_hint_flex_names.py`, the only
place to change them. The checkpoint tensor prefix stays `hi2_`.

## Format, as read

Each quantized `<name>.weight` is replaced by flat per-row buffers
`<name>.hi2_<field>` (spec 2.4):

| field | dtype | shape |
| --- | --- | --- |
| `codes` | int32 | sum over rows of `K * nbits / 32` words, codes LSB-first |
| `code_off` | int32 | `R + 1`, first word of each row |
| `blk_params` | uint8 | `2 * sum K / B`, `(lo_u8, step_u8)` per block |
| `param_off` | int32 | `R + 1`, first block of each row |
| `row_consts` | fp16 | `[R, 3]`: `lo_min, lo_scale, step_scale` |
| `row_nbits` | uint8 | `R`: 2, 4 or 8 |
| `row_block` | uint8 | `R`: 16, 32 or 64 |
| `had` | uint8 | `K / 16` Hadamard flags |
| `col_scale` | fp16 | `K`, optional smoothing vector |

Decode (spec 2.2): `lo = fp16(fma_f32(lo_u8, lo_scale, lo_min))`,
`step = fp16(step_u8 * step_scale)`, `w = fp16(lo + code * step)`. The stored
basis is `H (W * s)`, so `y = H(x / s) . W_stored^T` (spec 2.3).

Which modules are quantized is read from the checkpoint index: a module stored
as `.hi2_*` uses this format, one stored as `.weight` stays dense (vision
tower, embeddings, head, GDN `conv1d`, ...). No module list is needed in
`quantization_config`.

## Decisions where the spec is open (for the spec to match)

1. **Offsets.** `code_off` and `param_off` are **int32** (the spec table says
   uint32), each with `R + 1` entries starting at 0. `code_off` counts
   **32-bit words** of `codes`, `param_off` counts **blocks** (one block is two
   bytes of `blk_params`, so the byte offset is `2 * param_off`). The loader
   checks `code_off[r+1] - code_off[r] == K * nbits_r / 32`,
   `param_off[r+1] - param_off[r] == K / B_r` and that the last entries match
   the buffer lengths.
2. **INT8 rows** (spec 11.6) are accepted with **any** block size of 16, 32 or
   64 (the spec lists 32 or 64). Four codes per word, one byte each, LSB first,
   same ladder and parameters as INT2/INT4.
3. **INT8 rows in I8 mode.** Codes 0..255 do not fit the signed operand of
   `v_dot4_i32_i8` (and `v_dot4_i32_iu8` does not exist on gfx1030, spec 7):
   they enter the dot as `code - 128` and `128 * sumq` is added back to the
   integer sum, which is exact.
4. **I8 activation.** Per token and 16-column group (spec 5, A = 16), from the
   fp32 transformed activation (not its fp16 rounding): `sx = fp16(absmax /
   127)`, `xq = clamp(rint(x / sx), -127, 127)` with rint rounding half to even,
   `sumq = sum xq` (stored int32). A group whose absmax is 0 gets `sx = 0` and
   `xq = 0`.
5. **F16 activation.** The transform (`x / s` then the order-16 Hadamard) runs
   in fp32 and is rounded once to fp16 before `v_dot2_f32_f16`.
6. **Routed experts.** Each expert is its own tensor with its own smoothing
   vector and flags; **gate and up of one expert must share them** (they see
   the same tokens). A checkpoint where they differ is rejected.
7. **Row-parallel TP** needs the per-rank K to be a multiple of 64 (no word,
   block or Hadamard group is cut).

## Files

| file | content |
| --- | --- |
| `csrc/rocm/rdna_hint_flex_rdna2.cu` | kernels and host wrappers (gfx1030 build only) |
| `vllm/model_executor/layers/quantization/rdna_hint_flex.py` | `RdnaHintFlexConfig`, `RdnaHintFlexLinearMethod` |
| `vllm/model_executor/layers/quantization/rdna_hint_flex_moe.py` | `RdnaHintFlexMoEMethod`, dense MoE reference |
| `vllm/model_executor/layers/quantization/rdna_hint_flex_names.py` | the name and its aliases |
| `vllm/model_executor/layers/quantization/utils/rdna_hint_flex_utils.py` | host helpers: validate, slice rows/columns, concatenate, PyTorch dequantize, transform and the I8 reference |
| `tests/kernels/quantization/test_rdna_hint_flex_rdna2.py` | reference port of `tools/hi2/hi2/format.py`, CPU and gfx1030 tests |
| `tools/rdna/rdna_hint_flex/` | env, sync/build, test runner, end-to-end check, kernel timing, serving benchmark |

Outside the new files: op declarations (`csrc/rocm/ops.h`), registration in
the `VLLM_ROCM_GFX1030` block of `csrc/rocm/torch_bindings.cpp`, the gfx1030
source list in `CMakeLists.txt`, wrappers and fake kernels in
`vllm/_custom_ops.py`, the registry in `quantization/__init__.py` (it also
appends the names to `RocmPlatform.supported_quantization`), and two hooks in
`fused_moe/routed_experts.py` (expert `hi2_*` tensors go to the `w13_weight` /
`w2_weight` loader and to the method's `absorb()`, as for EXL3). Nothing in
`fa_rdna2` or the attention backends changed.

## Compute modes

`compute_mode` in `quantization_config`, overridden by the
`VLLM_RDNA_HINT_FLEX_COMPUTE_MODE` environment variable (read when the config
is built, so it reaches the workers with it):

* `f16` (default, spec 6.1): fp16 activations, `v_dot2_f32_f16`.
* `i8` (spec 6.2): int8 activations and `v_dot4_i32_i8` for M <= 48 (decode),
  the f16 prefill path above.
* `i8_all`: the I8 kernel for every M (slow prefill; for accuracy runs).

Routed experts follow the same mode (I8 for batches up to 48 tokens with
`i8`, always with `i8_all`).

## Kernels

* `rdna_hint_flex_act_transform_rdna2`: `x' = H16(x / s)` on the flagged
  groups. One thread per (token, group).
* `rdna_hint_flex_gemv_rdna2` (F16 decode, M <= 48): one wave32 per row (or 2
  to 4 waves per row with an LDS reduction for tensors under 8192 rows), lanes
  take consecutive words, two in flight per lane. A word never crosses a block:
  one `lo`/`step` per word, code to fp16 by `half(0x6400 | c) - 1024` (exact up
  to 1023, so INT8 too), one `v_pk_fma_f16`, `v_dot2_f32_f16`, fp32
  accumulation, wave reduction.
* `rdna_hint_flex_gemv2_rdna2` (F16 decode GEMV v2, M <= 48): a wave owns 2 or 4
  rows and lane `l` owns the 16-column group `it * 32 + l` of each of them, so
  one activation load serves all of the wave's rows (see "GEMV v2" under
  Performance). `rdna_hint_flex_linear_rdna2` launches it when
  `RDNA_HINT_FLEX_GEMV` asks for it (default `v1`: unchanged); the op itself
  takes an explicit `(mt, rpw, wpb)` and is what the tests and
  `bench_tp2.py` call. `rdna_hint_flex_act_transform_rdna2` takes
  `pair_interleaved`, the activation order v2 reads.
* `rdna_hint_flex_dequant_rdna2`: stored-basis fp16 `W [R, K]`, one 16-column
  group per thread, 16-byte stores.
* `rdna_hint_flex_act_quant_rdna2` + `rdna_hint_flex_gemv_i8_rdna2` (I8):
  transform and quantization in one thread per (token, group); the GEMV spreads
  one group's codes into four byte-packed dwords and takes four
  `v_dot4_i32_i8`, then `sx * (step * S + lo * sumq)` in fp32.
* `rdna_hint_flex_linear_rdna2`: transform, then GEMV for M <= 48, otherwise
  dequant into scratch and a rocBLAS GEMM. Chosen inside the op at execution
  time, so torch.compile cannot fix it to the warmup shape.
* `rdna_hint_flex_moe_rdna2` (routed experts, F16): experts stacked row-wise
  into one ragged tensor per stage with `[E, K]` / `[E, K/16]` transform tables;
  a transform per (token, expert) pair, a GEMV per pair for gate/up, SiLU and
  mul, and the down GEMV with the routing-weighted top-k sum inside the kernel
  (no atomics, fixed shapes, cudagraph-capturable). Pairs whose expert is not
  on this rank (`-1`) add nothing. Used for batches up to 16 tokens
  (`RDNA_HINT_FLEX_MOE_GROUPED_MIN`) and for I8.
* Grouped routed-expert prefill (F16, batches over 16 tokens): the pairs are
  sorted by expert into 64-row blocks (`moe_align_block_size`),
  `rdna_hint_flex_moe_transform_rdna2` writes each pair's transformed
  activation with its expert's `col_scale`/`had`, and
  `rdna_hint_flex_moe_gemm_rdna2` runs the fused decode-in-GEMM tile kernel
  (`gemm2_kernel` in MOE mode: each 64-row block reads one expert's rows, padded
  slots are skipped) for gate/up and again for down after SiLU and mul; the
  routing-weighted top-k sum is a torch reduction. Each expert's weights are
  read once per 64 routed tokens instead of once per token.

All launches use the current torch stream and `C10_CUDA_KERNEL_LAUNCH_CHECK`.
The kernels trust the offsets; the loader validates them on the host first.

Tried and dropped (slower on V620, measured with `bench_kernel.py`): staging
the activation in LDS, lanes owning 16-column groups and reusing the activation
in registers across 2-8 rows (GEMV v2 below is that structure: it is not faster
at M=1 either, and wins from M=2 on), and fusing the transform into the GEMV
(VGPRs 43 to 126, occupancy 16 to 8 waves/SIMD; 1.8x to 3x slower at M=1).

## Loader and tensor parallelism

The loader collects each checkpoint tensor per output shard (`None`, an int, a
tuple such as Qwen3.5's `in_proj_qkv -> (0, 1, 2)`, or `q`/`k`/`v`),
splits fused checkpoint tensors into the layer's partitions, slices rows for
column-parallel layers (GQA kv replication included) and 64-aligned columns for
row-parallel layers, validates, and concatenates consecutive partitions that
share `col_scale` and `had` into one tensor, so fused q/k/v and gate/up are one
kernel call (in the tool's checkpoints they always share them). Experts slice
gate/up rows and down columns the same way.

Where the kernels are not built (not gfx10x, or not ROCm), dense layers are
dequantized once to fp16 and run through `F.linear` after the PyTorch transform;
routed experts need the kernels.

Activations must be fp16 (`--dtype float16`): RDNA2 has no bf16 dot.

## Quantizing routed experts (tool side)

The tool now writes MoE checkpoints (OLMoE-1B-7B, see "Routed experts:
OLMoE-1B-7B"); the points below were the requirements and still hold:

* Per-expert 2-D tensors (`mlp.experts.E.{gate,up,down}_proj.weight`) are
  already eligible (2-D, K a multiple of 64) and the router is excluded by
  `DEFAULT_EXCLUDE`; they are written as `mlp.experts.E.<proj>.hi2_*`, which is
  what vLLM loads.
* Checkpoints that store experts fused in 3-D (`experts.gate_up_proj [E, 2N, K]`)
  are skipped by `is_eligible` (2-D only); the writer must split them per expert
  under the names above.
* Gate and up of one expert must get the same smoothing vector and flags
  (decision 6); deriving `s` from the expert's routed-token statistics, shared
  by both, gives that.
* Calibration per expert: rarely routed experts see few tokens (the spec's
  data-free fallback applies and must be logged), one Hessian per expert input
  (gate/up share it) is `E * K^2` floats per layer, and the sensitivity probe's
  one forward per tensor multiplies by the expert count; probing experts in
  groups is needed to keep it practical.

## Checks

Unit tests: `bash tools/rdna/rdna_hint_flex/run_tests.sh <hip_device>`,
**1156 passed** on a V620 (209 before GEMV v2; its tests are listed under "GEMV
v2"). The dequant kernel and the I8 activation quantization
are bit-identical to the reference; `rdna_hint_flex_linear_rdna2` matches
`reference_linear` (float64) within 2e-3 relative error for M = 1..37 (F16),
the I8 GEMV matches the integer-exact I8 reference within 1e-3; the synthetic
4-expert MoE matches the dense reference within 1e-2 (F16) and 3e-2 (I8) for 1 to 300 tokens (per-pair and grouped paths); the fused prefill GEMM matches reference_linear within 2e-3.

End to end (`tools/rdna/rdna_hint_flex/e2e_check.py`: teacher-forced log-probs
on a fixed 137-token text and 4 greedy prompts of 48 tokens, against the
checkpoint's `hi2 export-dense`; 3.0 bpw checkpoints, sensitivity allocation):

| model | mode | ppl ours / dense | mean / max abs log-prob gap | greedy identical | weights in GPU |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | F16, eager | 24.454 / 24.471 | 0.005 / 0.031 | 4/4 | 0.39 vs 0.93 GiB |
| Qwen2.5-0.5B-Instruct | F16, F&P graphs | 24.457 / 24.471 | 0.005 / 0.034 | 4/4 | 0.39 GiB |
| Qwen3.5-0.8B | F16, eager | 27.954 / 27.947 | 0.004 / 0.017 | 4/4 | 0.78 vs 1.53 GiB |
| Qwen3.5-4B | F16, F&P graphs | 10.410 / 10.405 | 0.003 / 0.024 | 4/4 | 3.42 vs 8.81 GiB |
| Qwen3.5-4B | F16, F&P graphs, **TP=2** vs TP=1 | 10.417 / 10.410 | 0.003 / 0.019 | 4/4 | 1.79 GiB per GPU |

I8 accuracy against F16 (same checkpoints, eager; `i8_all` runs the I8 kernel
in prefill too, so the teacher-forced numbers measure it; `i8` only changes
decode, so its teacher-forced numbers equal F16 and only greedy can move):

| model | i8_all ppl vs F16 | mean / max abs log-prob gap vs F16 | greedy identical vs F16 (`i8_all` / `i8`) |
| --- | --- | --- | --- |
| Qwen2.5-0.5B | 24.479 vs 24.454 (+0.10%) | 0.015 / 0.081 | 3/4 (one prompt from token 4) / 4/4 |
| Qwen3.5-0.8B | 27.919 vs 27.954 (-0.13%) | 0.013 / 0.087 | 3/4 (one from token 36) / 3/4 (from token 36) |
| Qwen3.5-4B | 10.401 vs 10.410 (-0.08%) | 0.009 / 0.045 | 4/4 / 4/4 |

So I8 costs about three times the log-prob noise of F16 (F16 against the dense
export: 0.003-0.005) and no measurable perplexity on this short text.

## Performance

### Prefill: fused decode-in-GEMM against dequant + rocBLAS (decided: keep dequant + GEMM)

`rdna_hint_flex_gemm_rdna2` (128 x 128 tiles, 64-column K steps, the next
step's activation, weight words and block parameters prefetched into registers,
the ladder decoded on the LDS store, `v_dot2_f32_f16` on 8 x 8 micro-tiles) is
built and tested, and selectable with `RDNA_HINT_FLEX_PREFILL=1`. It does not
beat rocBLAS at prefill sizes, so the default stays dequant + rocBLAS GEMM.

Isolated (`tools/rdna/rdna_hint_flex/bench_prefill.py`, one V620, tuned
TunableOp rows), time relative to the dense fp16 `F.linear` of the same shape:

| shape (R x K) | path | M=128 | M=512 | M=2048 |
| --- | --- | --- | --- | --- |
| gate_up 18432 x 2560 | dequant + GEMM | 1.15 | 1.11 | 1.03 |
| gate_up 18432 x 2560 | fused | **0.71** | 1.17 | 1.16 |
| down 2560 x 9216 | dequant + GEMM | 1.08 | 1.08 | 1.06 |
| down 2560 x 9216 | fused | **0.83** | 1.09 | 1.19 |
| in_proj_qkvz 12288 x 2560 | dequant + GEMM | 1.04 | 1.12 | 1.03 |
| in_proj_qkvz 12288 x 2560 | fused | **0.25** | 1.17 | 1.21 |
| out_proj 2560 x 4096 | dequant + GEMM | 1.45 | 0.99 | 1.06 |
| out_proj 2560 x 4096 | fused | 1.14 | 1.20 | 1.09 |

The fused kernel reaches 19-21 TF/s at M = 2048 against rocBLAS's 21-24
TF/s; it only wins at M around 128, where rocBLAS has no good solution (a size
that serving rarely produces: decode goes to the GEMV, prefill chunks are 512 to
2048 tokens). Earlier variants (no prefetch, 32-column K steps, 64 x 64 and
128 x 64 tiles) were slower still (13-17 TF/s). In serving (Qwen3.5-4B, TP=1,
tuned rows), the fused path lost: 1k/512 c=1 prefill 1809 vs 1932 tok/s, 16k/1k
c=1 1334 vs 1407 tok/s (full table under "Serving").

### I8 for prefill (not worth it on gfx1030 with this activation grouping)

The spec's per-16-column activation scale (spec 5) forces a float rescale every
16 columns: per 16 multiply-adds an I8 GEMM issues 4 `v_dot4_i32_i8` plus a
convert, a multiply and two FMAs (8 instructions), the same as F16's 8
`v_dot2_f32_f16`; it would not be faster than the F16 GEMM, only less accurate.
The existing `i8_all` mode runs prefill through the I8 GEMV in 8-token tiles and
reaches 6 TF/s (3.4-4x slower than dense at M = 512-2048,
`bench_prefill.py --mode 2`); it is kept for accuracy evaluation. A per-token
activation scale would let the int32 sum run over a whole weight block (16-64
columns) and could roughly halve the instruction count, but spec 5 measured it
costing about 9% more output error on INT4 rows; that is a spec decision, not
done here.

### Kernel timing (decode)

`tools/rdna/rdna_hint_flex/bench_kernel.py`, one V620, CUDA graph of 50 calls
of one weight set (so a 19-32 MB tensor sits in the 128 MiB Infinity Cache
after the first call and the GB/s column is not a DRAM rate; cold-weight
numbers are under "GEMV v2" below); random tensors with 30% INT4 rows and
mixed blocks; microseconds per call including the activation transform:

| shape (R x K) | F16 M=1 | I8 M=1 | F16 M=3 | F16 M=8 | F16 M=32 |
| --- | --- | --- | --- | --- | --- |
| gate_up 18432 x 2560 | 40.1 (475 GB/s) | 33.9 | 71.5 | 138.5 | 535.9 |
| down 2560 x 9216 | 28.1 (338 GB/s) | 18.5 | 41.5 | 74.8 | 294.3 |
| in_proj_qkvz 12288 x 2560 | 30.5 (415 GB/s) | 28.6 | 47.4 | 93.5 | 366.9 |
| out_proj 2560 x 4096 | 15.6 (273 GB/s) | 12.6 | 27.9 | 36.8 | 127.0 |

### GEMV v2: decode at M = 2..48 (Qwen3.8-27B HI2, TP=2 and TP=4)

Branch `feat/hi2-kernel-opt`; kernel `gemv2_kernel` in
`csrc/rocm/rdna_hint_flex_rdna2.cu`, timing tool
`tools/rdna/rdna_hint_flex/bench_tp2.py`. `RDNA_HINT_FLEX_GEMV` (read once per
process) picks the decode GEMV that `rdna_hint_flex_linear_rdna2` runs
(F16 mode, M <= 48):

| value | M = 1 | M = 2 .. 48 |
| --- | --- | --- |
| unset, `v1` | v1 | v1: the kernel described above; every existing op gives the same bits as before this change |
| `auto` | v1 | v2 |
| `v2` | v2 | v2 |

A call goes to v2 when R >= 1024 (the 48-row `in_proj_ba` keeps v1, which
splits K over waves) and its code words are 16-byte aligned; anything else runs
v1. v2 reads a pair-interleaved activation that the transform writes
(`pair_interleaved`), so the two kernels of the op are always chosen together.
**The default stays `v1`**: `auto` is recommended once it is verified end to
end (see "End to end" below: it is not yet).

Batch invariance: v1 gives a token the same bits whatever M is. `v2` does too
(the eight pairs of a group are summed in the order 0..7 for every width and
every (MT, RPW, WPB), tested bitwise). `auto` does not: M = 1 (v1) and M >= 2
(v2) differ in the last fp32 bits before the one fp16 rounding, so a token
decoded alone and in a verify batch can differ by an fp16 ulp in a logit. Use
`v2` where that matters; it costs about 3% at M = 1 (0.4 ms per step at c=1).

#### What a decode step reads

Per rank at TP=2, from `tools/rdna/rdna_hint_flex/tp_layout.py MODEL --tp 2`
(checkpoint headers plus the loader's partitioning; one kernel segment per
layer, so one transform and one GEMV launch per linear):

| slot | layers | R x K | % INT4 rows | MB per layer | MB per step |
| --- | --- | --- | --- | --- | --- |
| in_proj_qkvz | 48 | 8192 x 5120 | 26.2 | 15.12 | 726.0 |
| in_proj_ba | 48 | 48 x 5120 | 99.9 | 0.15 | 7.1 |
| out_proj | 48 | 5120 x 3072 | 71.6 | 7.44 | 357.3 |
| gate_up_proj | 64 | 17408 x 5120 | 29.5 | 32.44 | 2076.1 |
| down_proj | 64 | 5120 x 8704 | 36.8 | 17.04 | 1090.4 |
| qkv_proj | 16 | 7168 x 5120 | 26.8 | 13.26 | 212.1 |
| o_proj | 16 | 5120 x 3072 | 75.7 | 7.55 | 120.8 |

That is 4.590 GB per rank per token for the 304 hi2 linears (no INT8 rows),
plus the `lm_head` (124160 x 5120 fp16, 1.271 GB) and, per draft step, the dense
MTP layer (0.425 GB). At 512 GB/s: 8.96 ms for the hi2 linears, 2.48 ms for
the head, 11.45 ms per token. At TP=4 the rank reads 2.297 GB (R x K: qkvz
4096 x 5120, in_proj_ba 24 x 5120, out_proj 5120 x 1536, gate_up 8704 x 5120,
down 5120 x 4352, qkv 3584 x 5120, o_proj 5120 x 1536) plus 0.636 GB of head.

#### How the numbers were taken

One V620 (HIP 4, bus df), ROCm 7.14, torch 2.12. `bench_tp2.py MODEL --tp 2
--m 1,2,3,4,8 --variants gemv,g2p:{mt}:2:4,... --repeats 5`: for each slot the
real layers of the checkpoint (both layer kinds, in a fixed low-discrepancy
order) are loaded as the loader would and replayed in one CUDA graph until the
weight working set is 640 MiB (5x the Infinity Cache), so every call reads cold
weights as a decode step does. 5 repeats per cell, min and median, GB/s on the
tensors' real bytes, `% floor` against 512 GB/s. No `docker buildx build` ran
(checked at the start and end of each run); a timing window is rerun when the
process's KFD eviction counter moved during it (at most one window per run was
hit); junction temperature stayed under 70 C (5 s mean). Cells vary up
to ~10% between processes (gate_up M=4, `4:4:8`: 92 and 102 us in two runs;
the M=2, 3, 8 cells agree to ~1%), ratios inside one run are stable to ~1%:
compare within a run. `{mt}` is the token tile the v2 instantiations offer at
that M; `g2p:MT:RPW:WPB` is the explicit-configuration op on the
pair-interleaved activation.

Streaming rate reachable on this card (`bench_tp2.py --stream`: the dense fp16
GEMV of the `lm_head` over cold weights): 311 / 272 / 353 / 388 / 413 / 432 /
437 GB/s at 3.9 / 7.9 / 16 / 33 / 67 / 268 / 1332 MB per call. So a 7-32 MB
kernel cannot beat ~350-390 GB/s; the 512 GB/s floor is not reachable for
kernels of this size.

#### Baseline (v1) and what it leaves

TP=2, v1 (`gemv`), min / median us (GB/s, share of the 512 GB/s floor):

| slot (R x K, MB per call) | M=1 | M=2 | M=3 | M=4 | M=8 |
| --- | --- | --- | --- | --- | --- |
| gate_up_proj (17408 x 5120, 31.6) | 82.5 / 82.5 (383, 75%) | 118.8 / 124.2 (266, 52%) | 146.2 / 149.5 (216, 42%) | 172.2 / 173.0 (183, 36%) | 268.6 / 271.2 (117, 23%) |
| down_proj (5120 x 8704, 16.8) | 50.1 / 50.2 (336, 66%) | 67.7 / 71.3 (249, 49%) | 81.8 / 85.9 (206, 40%) | 96.4 / 99.4 (175, 34%) | 150.3 / 153.2 (112, 22%) |
| in_proj_qkvz (8192 x 5120, 15.2) | 45.0 / 45.0 (338, 66%) | 61.6 / 64.0 (247, 48%) | 73.9 / 76.5 (205, 40%) | 86.1 / 88.2 (176, 34%) | 134.2 / 136.1 (113, 22%) |
| out_proj (5120 x 3072, 7.5) | 27.1 / 27.1 (275, 54%) | 33.5 / 34.1 (222, 43%) | 38.6 / 41.0 (193, 38%) | 43.1 / 45.5 (173, 34%) | 64.1 / 66.3 (116, 23%) |
| qkv_proj (7168 x 5120, 13.2) | 41.0 / 41.1 (321, 63%) | 56.0 / 59.4 (235, 46%) | 67.0 / 70.7 (197, 38%) | 77.6 / 81.0 (170, 33%) | 120.5 / 123.9 (110, 21%) |
| o_proj (5120 x 3072, 7.6) | 27.4 / 27.4 (276, 54%) | 33.6 / 34.2 (225, 44%) | 38.5 / 40.9 (197, 38%) | 43.2 / 45.5 (175, 34%) | 64.5 / 66.7 (117, 23%) |
| in_proj_ba (48 x 5120, 0.1) | 4.4 / 4.4 | 5.7 / 5.7 | 6.5 / 6.5 | 7.1 / 7.1 | 10.0 / 10.0 |

At M=1 v1 is at the streaming rate of its size (gate_up 383 GB/s against 388
for 33 MB; down 336 against 353 for 16 MB): nothing to win there on the weight
side. From M=2 the time grows by 26-36 us per token on gate_up (118.8, 146.2,
172.2 us at M = 2, 3, 4) because v1 reloads the 16 activation values of a word
for every word of every row: the GEMV falls to 23% of the floor at M=8. That
slope is what a verify batch (MTP) or c >= 2 pays 304 times per step. The
sum of the 304 v1 launches is 13.25 / 18.21 / 22.00 / 25.66 / 39.77 ms at
M = 1 / 2 / 3 / 4 / 8 (HiTp2Bring's rocprof of the TP=2 server, f16 mode,
taken while the image build ran, has 12.33 / 16.14 / 22.49 / 36.25 ms at
c = 1 / 2 / 4 / 8 and 82.9 / 105.6 / 148.8 / 244.4 us for the gate_up GEMV at
c = 1 / 2 / 4 / 8: the same shape of curve).

#### What v2 does

* A wave owns RPW rows (2 or 4) and lane `l` owns the 16-column group
  `it * 32 + l` of each of them: the group's activation (MT tokens x 16
  columns) is loaded once into registers (one iteration ahead for MT <= 4) and
  serves all RPW rows, so a weight word costs its load, about four VALU
  instructions to decode a column pair and MT `v_dot2_f32_f16`.
* Pairing: the dot runs over the pairs (x_i, x_{i+8}) of the 16-column group
  because those codes already sit in the two 16-bit halves of a dword (INT2:
  `(w >> 2i) & 0x00030003`; INT4: the two words are cut into 16-bit halves with
  one `v_perm_b32` each; INT8: one `v_perm_b32` per pair), then `| 0x64006400`,
  `-1024`, `v_pk_fma_f16(step, lo)`: the spec decode `w = fp16(lo + c * step)`
  with one rounding, identical to v1. The transform kernel writes the
  activation in that pair order (stored[2i] = x[i], stored[2i+1] = x[i+8]) at
  no cost. The accumulation order differs from v1 (other pairs, DPP wave sum:
  four row steps and `v_permlanex16` instead of the `ds_bpermute` tree).
* Rows of different width in one wave (the checkpoint's late layers have
  2000-3000 INT2/INT4 runs, so 8-28% of waves straddle one): one instruction
  stream whose decode is driven by wave-uniform SGPR controls (two word
  selectors, shift and mask per pair), no divergence and no scratch. A branchy
  version spilled (256 VGPRs, 84 B scratch at MT=8, RPW=4). Uniform waves keep
  the specialised loops (the mixed stream is +26% on pure INT2). VGPRs are
  73-237 per configuration, scratch 0.
* An INT8 row in a wave falls back to one row at a time (none in this
  checkpoint; covered by tests).

Mixed-width effect, standalone one-kernel harness on the real row layouts of
four gate_up layers, M=8, one process, **taken while the image build ran: only
the ratios mean anything**: per-row mixed fallback 107.1 / 186.1 / 172.0 /
142.5 us against the unified loop 104.7 / 133.1 / 133.3 / 136.4 us on layers
0 (pure INT2), 41, 52 and 63 (89% to 100% INT4).

#### Results

TP=2, v2 with the `auto` configuration, min / median us (speedup over the v1
min; config MT:RPW:WPB). M=1 is shown for reference: `auto` keeps v1 there.

| slot | M=1 | M=2 | M=3 | M=4 | M=8 |
| --- | --- | --- | --- | --- | --- |
| gate_up_proj | 85.4 / 85.8 (x0.97; 1:2:4) | 89.0 / 89.6 (x1.33; 2:2:4) | 93.4 / 93.6 (x1.57; 3:4:4) | 102.9 / 104.5 (x1.67; 4:4:4) | 134.5 / 142.5 (x2.00; 8:4:4) |
| down_proj | 52.2 / 52.3 (x0.96; 1:2:4) | 53.1 / 53.1 (x1.28; 2:2:4) | 54.9 / 55.0 (x1.49; 3:4:4) | 57.5 / 57.7 (x1.68; 4:4:4) | 76.0 / 76.6 (x1.98; 8:4:4) |
| in_proj_qkvz | 46.9 / 47.1 (x0.96; 1:2:4) | 47.8 / 47.9 (x1.29; 2:2:4) | 49.4 / 49.6 (x1.50; 3:4:4) | 51.9 / 51.9 (x1.66; 4:4:4) | 70.4 / 71.6 (x1.91; 8:4:4) |
| out_proj | 27.6 / 27.6 (x0.98; 1:2:4) | 27.8 / 27.9 (x1.21; 2:2:4) | 28.5 / 28.5 (x1.36; 3:4:4) | 29.3 / 29.3 (x1.47; 4:4:4) | 36.4 / 36.4 (x1.76; 8:4:4) |
| qkv_proj | 41.8 / 41.8 (x0.98; 1:2:4) | 43.2 / 43.3 (x1.30; 2:2:4) | 43.9 / 44.0 (x1.53; 3:4:4) | 46.3 / 46.3 (x1.68; 4:4:4) | 59.9 / 61.3 (x2.01; 8:4:4) |
| o_proj | 28.1 / 28.2 (x0.97; 1:2:4) | 28.4 / 28.5 (x1.18; 2:2:4) | 29.0 / 29.0 (x1.33; 3:4:4) | 29.7 / 29.7 (x1.45; 4:4:4) | 36.0 / 36.0 (x1.79; 8:4:4) |

Whole decode step, sum of the GEMV launches (ms, min times, launches per step
x min per call; in_proj_ba is v1 in both columns):

| M | TP=2 v1 | TP=2 v2 | v1 / v2 | TP=4 v1 | TP=4 v2 | v1 / v2 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 13.25 | 13.71 | x0.97 | 7.83 | 7.81 | x1.00 |
| 2 | 18.21 | 14.14 | x1.29 | 10.51 | 7.97 | x1.32 |
| 3 | 22.00 | 14.71 | x1.50 | 12.46 | 8.19 | x1.52 |
| 4 | 25.66 | 15.72 | x1.63 | 14.17 | 8.57 | x1.65 |
| 8 | 39.77 | 20.62 | x1.93 | 21.57 | 11.64 | x1.85 |

(TP=4 sums have the six large slots only: `in_proj_ba` at R=24 was not run.
The M=1 "v2" column is not what `auto` runs, it is v2 for reference.) At M = 1
v2 is 3% slower at TP=2 and level at TP=4 (v1 is weak at K = 1536: out_proj
and o_proj are 22-26% faster with v2 there, in_proj_qkvz and gate_up are 5%
slower), so `auto` does not touch M = 1 and a TP=4 `auto` run is exactly the
v1 baseline at c = 1.

TP=4, v2 `auto` configuration against v1 per slot (speedup over the v1 min;
rank-0 shapes):

| slot (R x K) | M=2 | M=3 | M=4 | M=8 |
| --- | --- | --- | --- | --- |
| gate_up_proj (8704 x 5120) | x1.28 | x1.51 | x1.66 | x1.94 |
| down_proj (5120 x 4352) | x1.28 | x1.50 | x1.60 | x1.84 |
| in_proj_qkvz (4096 x 5120) | x1.21 | x1.41 | x1.57 | x1.78 |
| out_proj (5120 x 1536) | x1.74 | x1.77 | x1.86 | x1.65 |
| qkv_proj (3584 x 5120) | x1.30 | x1.57 | x1.71 | x2.05 |
| o_proj (5120 x 1536) | x1.72 | x1.77 | x1.84 | x1.69 |

Larger batches, TP=2 (verify batches of MTP at c >= 4, c = 12..48), whole-step
sum over the six large slots, v1 / v2 (`8:4:4`): M=12 49.71 / 33.96 ms
(x1.46), M=16 72.25 / 35.36 (x2.04), M=24 105.60 / 51.59 (x2.05), M=32 138.71
/ 68.67 (x2.02), M=48 205.04 / 103.51 (x1.98). v2 is faster than v1 for every
slot at every M from 2 to 48. M = 5 and 6 use the 6-token tile (whole-step sum
16.15 / 17.11 ms against 18.68 / 18.99 ms with the 8-token tile; v1 25.79 /
33.04 ms), M = 7 the 8-token tile (19.83 ms; the 6-token tile in two passes
27.41; v1 36.35).

The whole op, transform plus GEMV (`linear` against `l2:` in one process, six
large slots, ms per step): v1 `linear` 14.01 / 18.12 / 21.36 / 24.65 / 38.55 at
M = 1 / 2 / 3 / 4 / 8; v2 14.85 (M=2, `2:2:8`) / 15.25 (`3:4:4`) / 16.02
(`4:4:4`) / 20.92 (`8:4:4`): x1.22 / 1.40 / 1.54 / 1.84. The activation
transform alone is 3.6-3.7 us per launch (0.93 ms per step for these 256
launches, 3.5 us for `in_proj_ba`'s 48): 304 launches, about 1.1 ms of a
27 ms c=1 step. Writing the pair-interleaved order costs nothing
(`transform_pi`, same tool: 3.5-3.8 us per launch for every slot at M = 1, 3,
8; 0.93 / 0.93 / 0.94 ms per step, natural 0.93 / 0.93 / 0.94 ms).

The `auto` configuration (`gemv2_auto`): MT = M up to 4, 6 for M = 5-6, 8
above (larger M runs MT = 8 along `grid.y`); 2 rows per wave at M <= 2, 4 from
M = 3 on (2 when R < 2048: [INFERENCE] nothing that short was measured); 4
waves per block. Per M, this single choice is within 1% of the best
per-tensor choice in whole-step time (4% in one of three runs, where 8 waves per
block were 10% faster on gate_up at M=4; 1% in the other two).

Warm weights, for contrast (`--layers 1`: one layer replayed from the Infinity
Cache; same process): gate_up layer 0, M=1, v1 51.5 us against v2 38.1 us,
M=8 158.7 against 94.7 us. v2 is the more efficient kernel per instruction at
every M; at M=1 on cold weights both wait for DRAM and v1's access pattern is
3-4% better, so that is where it stays.

#### lm_head and the other launches

The `lm_head` is 1.271 GB per rank and 21.7% of the bytes of a TP=2 token:
fp16 `gemv_f16_rdna2` 2911.5 us at M=1 (436.7 GB/s; 2915.7 / 2920.8 / 2939.1 /
2995.0 us at M = 2 / 3 / 4 / 8), the T45 int8 shadow (`VLLM_RDNA_DENSE_INT8=1`,
`gemv_i8_rdna2`) 1460.2 us at M=1 (435.5 GB/s; 1463.7 / 1468.4 / 1474.8 at
M = 2 / 3 / 4, 2141.1 us at M = 8). T45 halves the head's time at M <= 4
(-1.45 ms per step, 5% of a c=1 step). **Its accuracy on the HI2 27B is not
validated**: per-row int8 of a vocabulary-sized fp16 matrix is lossy by
construction; run `tools/rdna/port_v031/decode_kl.py` against the fp16 head
before using it. Nothing was added to the quantization code for this.

Launch counts, not kernels, dominate the rest of a c=1 step (HiTp2Bring's
profile: 2451 launches, 1.9 us gap each, 7.2 ms outside the GEMVs):
`gemma_fused_add_rms_norm` (~770 launches per step) and the GDN glue (~17 per
layer) are outside the hi2 files; so is folding the 304 activation transforms
(~1.1 ms) into the preceding norm, which needs model-code plumbing.

Tried and dropped while building v2 (standalone harness, ratios only): fusing
the transform into the GEMV (redundant in-register transform adds ~23% VALU;
LDS staging collapses occupancy), a v2 for the I8 mode, a factorised decode
(not bit-identical), row permutation, `waves_per_eu` limits, non-temporal
weight loads, persistent scheduling (~2%), an in-loop INT4 branch (spills).

#### Correctness

* Flag off: with `RDNA_HINT_FLEX_GEMV` unset the outputs of
  `rdna_hint_flex_linear_rdna2` (modes 0 and 1), `rdna_hint_flex_gemv_rdna2`
  and `rdna_hint_flex_act_transform_rdna2` on real tensors (7 slots x 3 layers
  x M in {1, 2, 3, 4, 5, 8, 13, 24, 48, 64}: 210 output sets) hash identically
  between the base `_rocm_C` and `_custom_ops.py` and this branch's.
* Real tensors (`bench_tp2.py --check "0:0:0;1:4:4;2:2:4;3:4:4;4:4:4;8:4:4;8:2:4"
  --check-layers 0,3,30,31,41,47,63`, 588 cells, final binary): v2 against v1
  differs in 0.18% of the outputs, by at most 3.9e-3 (3.4e-4 of max |y|: one
  fp16 ulp of the largest outputs); against the float64 decode of the same
  rows the error is identical for v1 and v2 in every cell (v2 never worse);
  the pair-interleaved and the natural activation give the same bits.
* `tests/kernels/quantization/test_rdna_hint_flex_rdna2.py`: 1156 passed (209
  before v2; the 947 added cover v2 against the float64 reference for each row
  width, mixed INT2/INT4/INT8 run patterns with K = 512, 1088, 1600, bitwise
  equality across (MT, RPW, WPB) and activation layouts, the pair-interleaved
  transform, and the production op at M = 1..48). With `RDNA_HINT_FLEX_GEMV=v2`
  and `=auto`, the 131 tests that go through the production op pass.

#### End to end

Verified end to end: **nothing of v2/auto yet.** The v1 baseline of the 27B
TP=2 recipe (`27b-hi2-tp2`, from HiTp2Bring's branch) was booted and probed
with an earlier build of this branch and the switch unset: READY in 180 s, greedy 3/3,
PPL probe 5.73 (mean NLL 1.746 over 261 tokens), concurrency probe 47 of 48
identical (1 near-tie), 0 KFD evictions; the prefix probe failed only because
the default prompt is short (hit granularity 2048 tokens; use
`PREFIX_WORDS=3000`). To A/B v2 on the same binary, only the environment
differs (`serve_rdna.sh` passes the calling environment through):

```text
export RDNA_HINT_FLEX_GEMV=auto          # unset for the v1 baseline
RECIPE=27b-hi2-tp2 MODEL=<snapshot> GPUS=4,5 PORT=18231 TAG=hi2k-auto \
  PREFIX_WORDS=3000 CELLS="1024:512:1 1024:512:8 16384:1024:8:32" \
  BENCH_EXTRA="--temperature 0" SERVE_TREE=<tree> VENV=<venv> \
  bash tools/rdna/port_v031/serve_validate.sh
# then: decode_kl.py collect against each server, compare auto and v1
```

Expected [INFERENCE]: c=1 unchanged (M=1 is v1); at c=8 the GEMV time of a
step falls by about 19 ms (39.77 to 20.62 ms by the table above; the step
length itself was not measured here); the MTP verify batch at c=1 (M = 3) pays
14.71 instead of 22.00 ms of GEMV. The MTP doc's cost model used v1's growth
with M (gate_up 1.77x of M=1 at M=3); with v2 it is 1.13x (93.4 against 82.5 us).

Remaining: the head (T45 or smaller), norm and transform launches, M = 1 (at
the streaming rate), and the FA prefill attention (about 1.05-1.10 TFLOP/s,
HiTp2Bring's branch `feat/hi2-attn-prefill`, not touched here).

### TunableOp

The rocBLAS paths (prefill GEMM after the dequant, the dense arm's layers,
`lm_head` and the remaining dense layers) were tuned per the profile policy:
census during the bench cells (`bench_serve.sh CENSUS_DIR`), then
`tools/rdna/rdna_hint_flex/tunableop_campaign.sh` (tune, measure heuristic /
current / new, curate at 3 %, additive merge into all four rank files of
`tunableop/rocm7.14-rocblas5.5/`, 1860 -> 2594 rows, lookup hits 2500/2500).
605 Qwen3.5-4B and 136 Qwen2.5-0.5B rows were adopted (median 1.60x and 1.26x
over the heuristic per shape, 1.21x for the 4B's M >= 512 prefill shapes).
Details: `tunableop/rocm7.14-rocblas5.5/README.md` and `provenance.json`.

### Serving

`tools/rdna/rdna_hint_flex/bench_serve.sh` (vLLM OpenAI server, `RECIPE=full` =
FULL_AND_PIECEWISE cudagraphs, `vllm bench serve` random dataset with
`--ignore-eos`, n = 4 / 32 / 4 / 16 requests). Qwen3.5-4B, 3.0 bpw
rdna_hint_flex against its dense fp16 export, TP=1 on HIP 2, TP=2 on HIP 4-5,
TP=4 on HIP 2-5. Prefill and decode per request (aggregate in brackets).

With the tuned rows:

| cell | arm | prefill tok/s | decode tok/s | TTFT s | ITL ms (median) |
| --- | --- | --- | --- | --- | --- |
| 1k/512 c=1 | TP=1 ours | 1932 (1910) | 78.4 (72.6) | 0.53 | 12.8 |
| 1k/512 c=1 | TP=1 dense | 1961 (1941) | 38.1 (36.7) | 0.52 | 26.3 |
| 1k/512 c=1 | TP=2 ours | 3436 (3399) | 102.7 (97.1) | 0.30 | 9.7 |
| 1k/512 c=1 | TP=2 dense | 3418 (3374) | 61.4 (59.4) | 0.30 | 16.3 |
| 1k/512 c=1 | TP=4 ours | 6541 (6529) | 113.9 (110.2) | 0.16 | 8.8 |
| 1k/512 c=1 | TP=4 dense | 6686 (6631) | 93.3 (90.9) | 0.15 | 10.7 |
| 1k/512 c=8 | TP=1 ours | 410 (2537) | 35.4 (242.0) | 2.50 | 26.8 |
| 1k/512 c=8 | TP=1 dense | 416 (2570) | 27.8 (196.2) | 2.46 | 34.6 |
| 1k/512 c=8 | TP=2 ours | 759 (4639) | 55.7 (389.2) | 1.35 | 17.2 |
| 1k/512 c=8 | TP=2 dense | 788 (5064) | 46.4 (332.3) | 1.30 | 20.7 |
| 1k/512 c=8 | TP=4 ours | 1492 (7518) | 71.3 (521.3) | 0.69 | 13.2 |
| 1k/512 c=8 | TP=4 dense | 1514 (7629) | 70.2 (514.1) | 0.68 | 13.5 |
| 16k/1k c=1 | TP=1 ours | 1407 (1402) | 57.8 (34.9) | 11.65 | 17.3 |
| 16k/1k c=1 | TP=1 dense | 1416 (1411) | 32.6 (23.9) | 11.57 | 30.7 |
| 16k/1k c=1 | TP=2 ours | 2647 (2626) | 79.0 (53.5) | 6.19 | 12.7 |
| 16k/1k c=1 | TP=2 dense | 2647 (2633) | 52.3 (39.8) | 6.19 | 19.1 |
| 16k/1k c=1 | TP=4 ours | 4724 (4705) | 98.5 (73.9) | 3.47 | 10.2 |
| 16k/1k c=1 | TP=4 dense | 4756 (4729) | 82.4 (64.6) | 3.45 | 12.2 |
| 16k/1k c=8 | TP=1 ours | 583 (2668) | 10.1 (63.3) | 28.10 | 57.6 |
| 16k/1k c=8 | TP=1 dense | 589 (2697) | 9.5 (60.0) | 27.83 | 65.2 |
| 16k/1k c=8 | TP=2 ours | 1149 (5170) | 18.3 (116.1) | 14.26 | 33.1 |
| 16k/1k c=8 | TP=2 dense | 1135 (5211) | 17.6 (112.0) | 14.44 | 36.0 |
| 16k/1k c=8 | TP=4 ours | 1528 (4795) | 27.0 (167.8) | 10.72 | 21.0 |
| 16k/1k c=8 | TP=4 dense | 1569 (4838) | 27.0 (168.6) | 10.44 | 21.0 |

Before tuning (the profile's 1860 rows, none for these shapes):

UN| cell | arm | prefill tok/s | decode tok/s | TTFT s | ITL ms (median) |
| --- | --- | --- | --- | --- | --- |
| 1k/512 c=1 | TP=1 ours | 1932 (1910) | 78.4 (72.6) | 0.53 | 12.8 |
| 1k/512 c=1 | TP=1 dense | 1961 (1941) | 38.1 (36.7) | 0.52 | 26.3 |
| 1k/512 c=1 | TP=2 ours | 3436 (3399) | 102.7 (97.1) | 0.30 | 9.7 |
| 1k/512 c=1 | TP=2 dense | 3418 (3374) | 61.4 (59.4) | 0.30 | 16.3 |
| 1k/512 c=1 | TP=4 ours | 6541 (6529) | 113.9 (110.2) | 0.16 | 8.8 |
| 1k/512 c=1 | TP=4 dense | 6686 (6631) | 93.3 (90.9) | 0.15 | 10.7 |
| 1k/512 c=8 | TP=1 ours | 410 (2537) | 35.4 (242.0) | 2.50 | 26.8 |
| 1k/512 c=8 | TP=1 dense | 416 (2570) | 27.8 (196.2) | 2.46 | 34.6 |
| 1k/512 c=8 | TP=2 ours | 759 (4639) | 55.7 (389.2) | 1.35 | 17.2 |
| 1k/512 c=8 | TP=2 dense | 788 (5064) | 46.4 (332.3) | 1.30 | 20.7 |
| 1k/512 c=8 | TP=4 ours | 1492 (7518) | 71.3 (521.3) | 0.69 | 13.2 |
| 1k/512 c=8 | TP=4 dense | 1514 (7629) | 70.2 (514.1) | 0.68 | 13.5 |
| 16k/1k c=1 | TP=1 ours | 1407 (1402) | 57.8 (34.9) | 11.65 | 17.3 |
| 16k/1k c=1 | TP=1 dense | 1416 (1411) | 32.6 (23.9) | 11.57 | 30.7 |
| 16k/1k c=1 | TP=2 ours | 2647 (2626) | 79.0 (53.5) | 6.19 | 12.7 |
| 16k/1k c=1 | TP=2 dense | 2647 (2633) | 52.3 (39.8) | 6.19 | 19.1 |
| 16k/1k c=1 | TP=4 ours | 4724 (4705) | 98.5 (73.9) | 3.47 | 10.2 |
| 16k/1k c=1 | TP=4 dense | 4756 (4729) | 82.4 (64.6) | 3.45 | 12.2 |
| 16k/1k c=8 | TP=1 ours | 583 (2668) | 10.1 (63.3) | 28.10 | 57.6 |
| 16k/1k c=8 | TP=1 dense | 589 (2697) | 9.5 (60.0) | 27.83 | 65.2 |
| 16k/1k c=8 | TP=2 ours | 1149 (5170) | 18.3 (116.1) | 14.26 | 33.1 |
| 16k/1k c=8 | TP=2 dense | 1135 (5211) | 17.6 (112.0) | 14.44 | 36.0 |
| 16k/1k c=8 | TP=4 ours | 1528 (4795) | 27.0 (167.8) | 10.72 | 21.0 |
| 16k/1k c=8 | TP=4 dense | 1569 (4838) | 27.0 (168.6) | 10.44 | 21.0 |

Tuned against untuned, per request: prefill +13% (ours) / +11% (dense) at
1k/512 c=1 TP=1 (TTFT 0.60 -> 0.53 s), +105% / +98% at TP=2 (TTFT 0.61 -> 0.30
s), +4% at TP=4; 16k/1k prefill and decode move by less than the run-to-run
spread (the TP=1/TP=2 tuned arms ran while other benches used the box's other
GPUs). Decode is unchanged (the decode path is the GEMV, not rocBLAS).

Ours against dense (tuned): decode per request 2.06x at TP=1 (1k/512 c=1),
1.67x at TP=2, 1.22x at TP=4; at c=8 1.27x / 1.20x / 1.02x; prefill within 4%
of dense everywhere (the same rocBLAS GEMM after the dequant). Every Qwen3.5-4B
arm passed the greedy, perplexity and concurrency probes with no fault markers.

Qwen2.5-0.5B-Instruct, TP=1, tuned rows, against its dense fp16 export and the
existing AWQ 4-bit g128 checkpoint (the AWQ arm uses the fork's RDNA2 W4A16
kernel). Head size 64 is not supported by RDNA_ATTN, so these arms run with
`VLLM_USE_RDNA2_FA=0` (Triton attention), which dominates the 16k cells:

| cell | arm | prefill tok/s | decode tok/s | TTFT s | ITL ms (median) |
| --- | --- | --- | --- | --- | --- |
| 1k/512 c=1 | ours 3.0 bpw | 13505 (13032) | 176.7 (172.5) | 0.08 | 5.7 |
| 1k/512 c=1 | dense fp16 | 13784 (13387) | 163.7 (160.2) | 0.07 | 6.1 |
| 1k/512 c=1 | AWQ 4-bit g128 | 14139 (13572) | 114.8 (113.1) | 0.07 | 8.7 |
| 1k/512 c=8 | ours 3.0 bpw | 3151 (14754) | 139.7 (1027.6) | 0.32 | 6.8 |
| 1k/512 c=8 | dense fp16 | 3002 (14600) | 149.5 (1088.8) | 0.34 | 6.3 |
| 1k/512 c=8 | AWQ 4-bit g128 | 3133 (15604) | 109.2 (817.4) | 0.33 | 8.8 |
| 16k/1k c=1 | ours 3.0 bpw | 3045 (3035) | 25.4 (22.4) | 5.38 | 39.4 |
| 16k/1k c=1 | dense fp16 | 3053 (3045) | 25.1 (22.2) | 5.37 | 39.8 |
| 16k/1k c=1 | AWQ 4-bit g128 | 3067 (3058) | 23.6 (21.0) | 5.34 | 42.4 |
| 16k/1k c=8 | ours 3.0 bpw | 934 (4201) | 15.7 (98.2) | 17.54 | 40.7 |
| 16k/1k c=8 | dense fp16 | 947 (4343) | 15.7 (98.8) | 17.30 | 40.0 |
| 16k/1k c=8 | AWQ 4-bit g128 | 942 (4258) | 15.3 (96.3) | 17.39 | 42.7 |

The 0.5B model fails the exact-answer greedy probe in its 3.0 bpw and dense
forms (model quality; AWQ passes). The tuned dense arm failed one of 48
concurrency checks (one prompt diverged at token 15 at c=8 with a 2.3-nat
margin; the untuned dense arm and all other arms passed); not investigated.

## Checks: held-out perplexity

`tools/rdna/rdna_hint_flex/acc_long.sh` / `acc_tp.sh`: 8 x 2048 tokens of
`study_text_all.txt` after token 800000 (the tools/hi2 eval protocol; the
unquantized ppl matches the spec: 10.49 / 10.87 / 7.067), eager unless noted:

| model | unquantized | dense export | ours F16 | ours I8 (`i8_all`) | I8 / F16 | mean / max abs log-prob gap I8 vs F16 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B (3.0 bpw) | 10.4932 | 12.8013 | 12.8018 | 12.8097 | 1.00061 | 0.0128 / 0.267 |
| Qwen3.5-0.8B (3.0 bpw) | 10.8736 | 13.1276 | 13.1283 | 13.1327 | 1.00033 | 0.0114 / 0.279 |
| Qwen3.5-4B (3.0 bpw) | 7.0671 | 7.7916 | 7.7911 | 7.8001 | 1.00115 | 0.0095 / 5.46 |

AWQ 4-bit g128 Qwen2.5-0.5B: 12.0334 (4.25 bpw against our 3.0).

Tensor parallelism (Qwen3.5-4B, F16, full cudagraphs): ppl 7.7913 (TP=1),
7.7915 (TP=2), 7.7912 (TP=4); log-prob gap to TP=1 mean 0.0025 (TP=2) and
0.0026 (TP=4), the same as TP=1 graphs against TP=1 eager (0.0025).

## Routed experts: OLMoE-1B-7B

`allenai/OLMoE-1B-7B-0924` (16 layers, 64 experts top-8, SiLU-gated, expert
`[1024, 2048]` gate/up and `[2048, 1024]` down, max context 4096), quantized by
`tools/hi2` into two checkpoints on the box
(`~/w4a8_runs/hadamard-int2/runsmoe/`):

| checkpoint | avg bpw (experts / rest) | rows at INT2 / INT4 | size |
| --- | --- | --- | --- |
| `olmoe-r4` | 3.54 (3.49 / 4.67) | 40 % / 60 % | 3.2 GB |
| `olmoe-b3.0` | 3.00 (2.93 / 4.57) | 67 % / 33 % | 2.8 GB |

**Loader.** The writer's names (`model.layers.N.mlp.experts.E.{gate,up,down}_proj.hi2_*`,
attention `hi2_*`, router and norms plain) load without any change. There is
no mismatch: gate and up of each expert share `col_scale` and `had`, as
decision 6 requires, so the loader stacks them into one gate/up tensor per
layer.

**bf16 to fp16.** The checkpoints and the original are bf16 and run as fp16
(`--dtype float16`). The held-out perplexity of the original in fp16 is finite
and matches the tool's numbers (table below), and no NaN or inf showed up in
any run. Nothing overflows.

### Correctness

Held-out teacher-forced perplexity over 16,376 tokens (`acc_tp.sh`, TP=1,
FULL_AND_PIECEWISE). `study` is the held-out text and `mix` is the calibration
mix (`eval_mix.txt`). Each checkpoint is compared with its `hi2 export-dense`
and with the quantizer's own HF evaluation:

| model | study ppl | mix ppl | mean / max abs log-prob gap vs dense export |
| --- | --- | --- | --- |
| original, fp16 | 6.6331 | 3.9233 | - |
| `olmoe-r4`, dense export | 6.7079 | 4.1071 | - |
| `olmoe-r4`, rdna_hint_flex (tool HF: 6.707 / 4.108) | **6.7071** | **4.1063** | 0.0072 / 0.62 (study), 0.0083 / 0.63 (mix) |
| `olmoe-b3.0`, dense export | 6.7724 | 4.2867 | - |
| `olmoe-b3.0`, rdna_hint_flex (tool HF: 6.773 / 4.287) | **6.7729** | **4.2870** | 0.0062 / 0.60 (study), 0.0084 / 0.57 (mix) |

The kernels reproduce the dense export to within 0.03 % perplexity.
`olmoe-r4` costs +1.1 % (study) / +4.7 % (mix) over the original, and
`olmoe-b3.0` +2.1 % / +9.3 %. The grouped prefill path and the per-pair path
differ by a mean of 0.0056 nats per token (study ppl 6.7071 vs 6.7076).
Greedy probes were coherent on both checkpoints in eager mode and under
graphs ("Paris", factorial code, "42", sky colour). Every serving boot passed:

* the probe (3/3) and its PPL probe;
* the concurrency probe: 48/48 identical for r4 and b3.0; an earlier b3.0 boot gave 47/48 plus one near-tie;
* the probe after load.

No boot hit a fault (0 serve-log markers, 0 new amdgpu dmesg faults).

### Serving

The setup for every arm is TP=1 on one V620, FULL_AND_PIECEWISE, RDNA_ATTN, and
TunableOp lookup on the merged rows (2707 per rank, lookup-hit proof 2613/2613 FP16 keys). The cells are 1k/512 and
2k/512, because OLMoE's max context is 4096, so the standard 16k/1k cells do
not fit. Each cell has the usual four numbers:

* prefill tok/s per request, with the wave aggregate in brackets;
* decode tok/s per request (1000 / TPOT), with aggregate output tok/s in brackets;
* mean TTFT;
* median ITL.

Dense fp16 runs once on the fork's default unquantized MoE backend and once on
`--moe-backend triton`. `r4 pair-GEMV` is `olmoe-r4` with the grouped prefill
path disabled (`RDNA_HINT_FLEX_MOE_GROUPED_MIN=1000000`). Every arm had the GPU
to itself: `gpu-pids.log` shows one process per arm. Results are in
`~/w4a8_runs/hi2-support/olmoe-clean/`.

| cell | arm | prefill tok/s | decode tok/s | TTFT s | ITL ms |
| --- | --- | --- | --- | --- | --- |
| 1k/512 c=1 | `olmoe-r4` | **5096** (4974) | **209.1** (193.5) | **0.20** | 4.8 |
| 1k/512 c=1 | `olmoe-b3.0` | 4989 (4864) | 206.4 (190.9) | 0.21 | **4.6** |
| 1k/512 c=1 | dense fp16 | 1371 (1364) | 98.8 (86.5) | 0.75 | 10.1 |
| 1k/512 c=1 | dense fp16, Triton MoE | 3917 (3839) | 47.6 (46.5) | 0.26 | 21.0 |
| 1k/512 c=1 | r4 pair-GEMV | 1157 (1146) | 205.9 (152.0) | 0.88 | 4.8 |
| 1k/512 c=8 | `olmoe-r4` | **1552** (7104) | 59.2 (440.4) | **0.66** | 15.6 |
| 1k/512 c=8 | `olmoe-b3.0` | 1421 (6467) | **60.9** (449.1) | 0.72 | **15.2** |
| 1k/512 c=8 | dense fp16 | 369 (1594) | 36.2 (242.2) | 2.77 | 22.8 |
| 1k/512 c=8 | dense fp16, Triton MoE | 968 (3526) | 24.5 (187.0) | 1.06 | 38.6 |
| 1k/512 c=8 | r4 pair-GEMV | 310 (1359) | 46.2 (285.3) | 3.30 | 15.8 |
| 2k/512 c=1 | `olmoe-r4` | **8391** (8186) | **189.7** (174.2) | **0.24** | 5.2 |
| 2k/512 c=1 | `olmoe-b3.0` | 8336 (8169) | 188.2 (172.9) | 0.25 | **5.1** |
| 2k/512 c=1 | dense fp16 | 2588 (2582) | 93.8 (82.0) | 0.79 | 10.5 |
| 2k/512 c=1 | dense fp16, Triton MoE | 6783 (6718) | 46.5 (45.3) | 0.30 | 21.4 |
| 2k/512 c=1 | r4 pair-GEMV | 2195 (2177) | 189.0 (140.8) | 0.93 | 5.3 |
| 2k/512 c=8 | `olmoe-r4` | 2420 (8645) | 48.9 (362.3) | 0.85 | 19.1 |
| 2k/512 c=8 | `olmoe-b3.0` | **2587** (9729) | **50.0** (371.4) | **0.79** | **18.7** |
| 2k/512 c=8 | dense fp16 | 733 (2531) | 32.2 (219.3) | 2.79 | 25.5 |
| 2k/512 c=8 | dense fp16, Triton MoE | 1923 (7423) | 23.3 (177.9) | 1.06 | 41.5 |
| 2k/512 c=8 | r4 pair-GEMV | 766 (2948) | 37.9 (253.2) | 2.67 | 19.5 |

**Decode.** Both checkpoints decode about 2.1x faster per request than dense
fp16 on its best backend, which is the default one (209 vs 99 tok/s at 1k
c=1). At c=8 the gain is 1.6x per request and 1.8x in aggregate. Decode reads
3.5 or 3.0 bpw expert weights instead of 16. The two checkpoints are within 2 %
of each other, so at this batch size the per-token cost is mostly not the
weight bytes. The Triton MoE backend decodes at half the speed of the default
backend.

**Prefill.** HI2 prefill is ahead of both dense backends: 1.2x Triton and
3.2x the default backend at 2k c=1 (1.3x and 3.7x at 1k). HI2 prefill is not behind dense, so this
round needed no prefill fix. The fix that mattered was the grouped path
(above, unit-tested). Before it, routed-expert prefill ran the per-pair GEMV,
which re-reads each expert for every routed token. Against the per-pair path,
the grouped path is:

* 4.4x faster prefill at 1k c=1 (5096 vs 1157) and 3.8x at 2k c=1;
* 3x (2k) to 5x (1k) lower TTFT at c=8;
* 28 % faster at c=8 decode, which interleaves prefill chunks.

**TunableOp.** The first (untuned) runs of r4, b3.0 and both dense arms
recorded the GEMM shapes: 113 FP16 keys. These are attention, router
and lm_head GEMMs; routed experts never reach rocBLAS on any arm. With the
M=1..8 extensions the census gave 145 shapes. Curation adopted 113 rows
(median 1.57x over the rocBLAS heuristic, best 3.55x for the 64x512x2048
router GEMM) and left 32 on the heuristic. The rows were appended to the
shared profile.

**Runs not used.**

* The first untuned runs overlapped another job's sweep on HIP 3/4.
* The first tuned campaign shared HIP 2 with a QAD quantize job (started 14:01,
  after this campaign's lock file).
* A second QAD job (`q-bud-es`) started on HIP 4 at 15:06 while that GPU's
  lock file was held.

The numbers from these runs (r4/b3.0 c=8 swinging 2x, decode at 2k down by a
quarter) are kept under `~/w4a8_runs/hi2-support/serve-olmoe-*` but not
reported. `bench_serve.sh` now logs the processes on its GPUs (`gpu-pids.log`)
so a shared run shows up.

### Format notes for MoE (spec not edited)

* **Tensor count.** Every expert projection is its own HI2 tensor with nine
  arrays, so the checkpoint has 28,307 tensors and a 3.3 MiB safetensors
  header. The loader stacks 64 experts x 3 projections per layer into one
  ragged tensor per stage. That stacking means `torch.cat` over 64 parts
  per array, plus 64-entry `[E, K]` / `[E, K/16]` transform tables. On a
  quiet host model loading takes about 95 s (r4 96 s, b3.0 92 s; dense
  fp16 loads in 22 s). With host load around 150 (other jobs' quantizers) it took
  350-560 s.
* **Metadata size (`olmoe-r4` experts).**
    * Codes are 2412 MiB (3.14 bpw) and block params 235 MiB (0.31 bpw).
    * Row metadata (`code_off`, `param_off`, `row_consts`, `row_nbits`,
      `row_block`) is 64 MiB (0.08 bpw).
    * `col_scale` + `had` are 10.3 MiB.
    * Per-expert transform tables are cheap. The per-row offsets are 8 of the
      16 bytes per row and are absolute within each tensor, so stacking has to
      rebase them, which is a host pass over 4 M rows.
* **Stacked layout.** The tool now writes this layout (`--moe-layout
  stacked`), and the loader reads it. See "Stacked expert layout" below.
* **Rarely routed experts.** The quantizer fell back to data-free
  calibration for 3 tensors in each checkpoint (layer 7 expert 51,
  gate/up/down). That is visible in `hi2_report.json`
  (`experts.coverage.*.fallback`), and the minimum per-expert token count is
  in the same table.

## Stacked expert layout

`tools/hi2 quantize --moe-layout stacked` and `hi2 restack` write one group
per layer and stage: `<pre>.gate_up.hi2_*` and `<pre>.down.hi2_*`, or `w13` /
`w2` for Mixtral names. A `hi2_part_off` array marks the group. The writer's
description is in `docs/rdna2/hi2-moe.md`, "Stacked expert layout", on
`feat/hadamard-int2-weights`. The arrays are:

* the seven row arrays of one flat container, offsets already rebased;
* `had [E, K/16]`;
* `col_scale [E, K]`, all or none;
* `part_off [E*P + 1]`.

Rows are expert-major with gate before up. That is exactly the
`_rdna_hint_flex_w13` / `_w2` layout the MoE method builds in memory.

**Loader.** `RoutedExperts.load_weights` sends `gate_up|down|w13|w2.hi2_<field>`
names to `RdnaHintFlexMoEMethod.absorb_stack`, which collects the arrays per
stage. `process_weights_after_loading` then picks one of three paths per stage:

* **Stacked, TP=1, every expert local.** It checks the arrays with
  `validate_stack`:
    * one valid container with `K = had.shape[1] * 16`;
    * `part_off` covering the rows;
    * every part with the layer's row count;
    * the expert count equal to the layer's.

  It then copies the arrays to the GPU as they are, with no per-expert
  stacking and no offset rebase. An absent `col_scale` becomes the empty tensor.
* **Stacked, TP>1 or expert parallel.** `split_stack` cuts each local expert's
  parts out by `part_off`. Under EP the local experts come from
  `_map_global_expert_id_to_local_expert_id`, in local order. The parts are views
  and byte for byte the per-expert tensors. They then take the per-expert path:
  `select_rows` gives the rank's gate/up rows, `select_cols` gives the down
  columns, and the result is restacked.
* **Per-expert.** This path is unchanged. It covers older checkpoints, and any
  group the writer left per expert inside a stacked checkpoint (the forms can mix
  per stage).

The config's module scan needs no change: the routed-expert method is chosen by
the experts prefix.

**Checks.**

* **Unit tests:** `test_moe_stacked_layout_equals_per_expert` checks that the
  stage tensors and transform flags are equal to the per-expert load (dtypes and
  `torch.equal`). It covers six cases: TP=1, with smoothed and unsmoothed down;
  TP=2 at ranks 0 and 1; and EP with experts {2,3} local, plus a non-monotonic
  local order at TP=2. Two more tests: `test_moe_stacked_and_per_expert_mixed`
  (w13 stacked, w2 per expert with mixed smoothing) and
  `test_moe_stacked_layout_rejects_bad_stacks`.
* **Real checkpoints:** `tools/rdna/rdna_hint_flex/compare_layouts.py` loads all
  16 layers of `olmoe-r4` and `olmoe-r4-stacked` through the method on the CPU,
  at TP=1 and at TP=2 rank 1. Every stage tensor is equal: `RESULT PASS` both
  times.
* **Serve probes:** the probe, PPL-probe and probe-after-load outputs are
  byte-identical between the layouts on all five boot pairs:
    * r4 and b3.0, cold and warm;
    * r4 for the serving cells.

  The greedy replies are "Paris", "2, 2+2=4, ..." and "Berlin". The
  concurrency probe gave 48/48 identical for stacked r4 and 47/48 plus one
  near-tie for per-expert r4.
* **Faults:** none in any boot.

**Perplexity through the fork** (teacher-forced, 16,376 tokens, TP=1,
FULL_AND_PIECEWISE):

| checkpoint | per-expert study / mix | stacked study / mix |
| --- | --- | --- |
| `olmoe-r4` | 6.7071 / 4.1072 | 6.7071 / 4.1072 |
| `olmoe-b3.0` | 6.7727 / 4.2871 | 6.7727 / 4.2871 |

With the engine in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), the per-token
log-probs are bit-identical across layouts. That holds for four r4 study runs,
alternating layouts, and for the r4 mix pair. With the default engine process,
the log-probs vary between runs **of either layout**, independently of the
layout:

* 2 of 5 stacked graph runs differed;
* eager mode gave two outcomes (study ppl 6.7081 and 6.7067), each produced by
  both layouts.

The divergent runs differ from the first token of every prompt, by a mean of
0.0055 nats. This is batch composition: requests reach the separate engine-core
process at timing-dependent moments, and the GEMMs are not batch-invariant. It is
not the weights. Use the in-process engine for bit-exact A/B. (The 4.1063 mix
figure in the OLMoE section came from the per-pair path; the grouped path gives
4.1072.)

**Model load time** ("Model loading took" in the serve log, one V620, TP=1). A
cold load has the checkpoint's pages evicted with `posix_fadvise(DONTNEED)`
just before the boot. A warm load is the next boot. The host's 1-minute load
average at the time was 11 to 52, out of 96 threads.

| checkpoint | layout | tensors | cold | warm |
| --- | --- | ---: | ---: | ---: |
| `olmoe-r4` | per-expert | 28,307 | 94.2 s | 84.6 s |
| `olmoe-r4` | stacked | 979 | **16.1 s** | **9.2 s** |
| `olmoe-b3.0` | per-expert | 28,307 | 73.9 s | 72.4 s |
| `olmoe-b3.0` | stacked | 979 | **9.9 s** | **8.8 s** |

Stacked loads 8-9x faster warm and 6-7x faster cold. It is now faster than the
dense fp16 model (22 s for 12.9 GiB), because it reads 3.2 GiB with no
per-expert work. The cold-cache penalty is only the disk read, 1-7 s.

**Serving, no regression** (TP=1, FULL_AND_PIECEWISE, tuned rows, the same
session, each arm alone on the GPU):

| cell | layout | prefill tok/s | decode tok/s | TTFT s | ITL ms |
| --- | --- | --- | --- | --- | --- |
| 1k/512 c=1 | per-expert | 4996 (4848) | 203.9 (188.8) | 0.20 | 4.8 |
| 1k/512 c=1 | stacked | 5015 (4931) | 206.5 (191.1) | 0.20 | 4.8 |
| 1k/512 c=8 | per-expert | 1305 (6203) | 59.8 (438.5) | 0.78 | 15.7 |
| 1k/512 c=8 | stacked | 1315 (5425) | 60.0 (440.4) | 0.78 | 15.6 |

The layouts are within noise: the kernels get identical tensors.

Drivers: `tools/rdna/rdna_hint_flex/olmoe_stacked.sh` (load times, probes,
cells, perplexity) and `compare_layouts.py`. Results:
`~/w4a8_runs/hi2-support/stacked/` and `acc_long/olmoe_*_layout|rep*|eager*|inproc*`.

## I8 prefill: parked

Stopped at a clean point on request; nothing here is wired into a serving
path (F16 stays the default, the `i8`/`i8_all` modes are unchanged).

**What exists (opt-in ops, tested, not benchmarked):**

* `rdna_hint_flex_act_quant64_rdna2`: one thread per (token, 64-column group):
  `x / s` and the order-16 Hadamard in fp32, `sx = fp16(absmax / 127)` per 64
  columns, `xq` int8, and `sq16 = sx * sum(xq)` per 16-column chunk.
* `rdna_hint_flex_i8gemm_rdna2` (128 x 128 or 64 x 128 tiles, one 64-column
  group per K step, register prefetch): per 16-column chunk four
  `v_dot4_i32_i8`, `S1 += u_chunk * d` in int32, `acc += lo'_chunk * sq16`;
  per group `acc += sx * step_scale * float(S1)`. The integer epilogue uses the
  unrounded `step_b = step_scale * u_b` and `lo_b = lo_min + lo_scale * lou_b`
  (linear in the stored bytes; the spec's fp16 rounding of lo/step is dropped);
  INT2/INT4 codes are used as int8 directly, INT8 codes as `code - 128` with
  `128 * step_b` folded into `lo_b`.
* `act_quant64` / `unrounded_weight` / `i8g64_reference` in
  `rdna_hint_flex_utils.py`; 34 tests pass (bit-exact pre-pass; the GEMM within
  1e-3 of the integer-exact reference for INT2/INT4/INT8 and blocks 16/32/64).
* Not measured: kernel time against tuned rocBLAS (the dense bench shapes at
  M = 16..4096 were tuned into a scratch rows file,
  `~/w4a8_runs/hi2-support/tunable-bench/rows_bench.csv`, but the benchmark was
  not run), the A=32 kernel variant, and the row sort by block size that would
  let a tile drop the per-chunk integer multiply and bias (generic path: about 27
  instructions per 64 multiply-adds against 32 for the F16 GEMM; B=64-uniform
  tiles would need about 21).

**Activation-group measurements** (`tools/rdna/rdna_hint_flex/actq_study.py`,
real linear inputs after smoothing and rotation, 8 x 2048 held-out tokens;
output error is `|(x' - q(x')) W^T| / |x' W^T|`; perplexity through the math of
the I8 path in fp32):

| model | A | act error mean / pooled | output error mean / pooled | ppl | ppl / F16 |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | F16 | | | 13.1279 | 1 |
| Qwen3.5-0.8B | 16 | 0.454% / 0.469% | 0.323% / 0.239% | 13.1292 | 1.00010 |
| Qwen3.5-0.8B | 32 | 0.526% / 0.534% | 0.375% / 0.271% | 13.1283 | 1.00004 |
| Qwen3.5-0.8B | 64 | 0.598% / 0.599% | 0.426% / 0.303% | 13.1298 | 1.00014 |
| Qwen3.5-0.8B | per token | 1.088% / 0.934% | 0.781% / 0.453% | 13.1387 | 1.00082 |
| Qwen3.5-4B | F16 | | | 7.7919 | 1 |
| Qwen3.5-4B | 16 | 0.450% / 0.466% | 0.317% / 0.249% | 7.7923 | 1.00006 |
| Qwen3.5-4B | 32 | 0.523% / 0.533% | 0.368% / 0.284% | 7.7903 | 0.99980 |
| Qwen3.5-4B | 64 | 0.593% / 0.594% | 0.418% / 0.317% | 7.7918 | 0.99999 |
| Qwen3.5-4B | per token | 1.298% / 1.113% | 0.893% / 0.541% | 7.8063 | 1.00186 |

A = 64 passes the ppl gate (<= 1.003) on both models; so does per-token. The
differences between 16, 32 and 64 are below the run-to-run noise of the
perplexity.

Per input at A = 64: 85 of 114 (0.8B) and 118 of 152 (4B) inputs exceed 0.3%
output error from the activation quantization alone (mean 0.42%). The worst are
the attention output projections (`self_attn.o_proj`), then GDN `out_proj` and
some `down_proj`:

| model | input | A=16 | A=32 | A=64 | per token |
| --- | --- | --- | --- | --- | --- |
| 0.8B | layers.11.self_attn.o_proj | 0.842% | 0.968% | 1.082% | 1.781% |
| 0.8B | layers.7.self_attn.o_proj | 0.769% | 0.890% | 1.002% | 1.663% |
| 0.8B | layers.15.self_attn.o_proj | 0.556% | 0.645% | 0.727% | 1.312% |
| 0.8B | layers.8.linear_attn.out_proj | 0.479% | 0.565% | 0.646% | 1.329% |
| 0.8B | layers.3.self_attn.o_proj | 0.471% | 0.557% | 0.644% | 1.243% |
| 0.8B | layers.10.linear_attn.out_proj | 0.483% | 0.560% | 0.632% | 1.220% |
| 0.8B | layers.9.linear_attn.out_proj | 0.453% | 0.535% | 0.612% | 1.290% |
| 0.8B | layers.12.linear_attn.out_proj | 0.446% | 0.526% | 0.606% | 1.353% |
| 0.8B | layers.8.mlp.down_proj | 0.426% | 0.519% | 0.606% | 1.196% |
| 0.8B | layers.17.mlp.down_proj | 0.427% | 0.516% | 0.606% | 1.339% |
| 4B | layers.11.self_attn.o_proj | 0.658% | 0.758% | 0.852% | 1.579% |
| 4B | layers.15.self_attn.o_proj | 0.614% | 0.708% | 0.793% | 1.550% |
| 4B | layers.7.self_attn.o_proj | 0.544% | 0.631% | 0.713% | 1.528% |
| 4B | layers.19.self_attn.o_proj | 0.481% | 0.556% | 0.628% | 1.376% |
| 4B | layers.0.mlp.gate_up_proj | 0.488% | 0.569% | 0.613% | 0.877% |
| 4B | layers.12.linear_attn.out_proj | 0.438% | 0.517% | 0.595% | 1.328% |
| 4B | layers.23.self_attn.o_proj | 0.447% | 0.520% | 0.589% | 1.346% |
| 4B | layers.13.linear_attn.out_proj | 0.431% | 0.509% | 0.584% | 1.459% |
| 4B | layers.28.mlp.down_proj | 0.399% | 0.478% | 0.563% | 1.568% |
| 4B | layers.9.linear_attn.out_proj | 0.414% | 0.489% | 0.562% | 1.353% |

No input is an outlier by an order of magnitude (worst about 2.5x the mean);
with the end-to-end perplexity unchanged, a per-input group or an F16-only flag
is not justified by these two models.

**Format implications (as far as measured):** no checkpoint layout change is
needed. The 8-bit block parameters are linear in the stored bytes with per-row
fp16 constants, smoothing and rotation fold into the activation pre-pass, and
INT2/INT4 codes are valid int8 operands. Only the spec's activation group A (16
-> 64, with B dividing 64) and the decision to use unrounded `lo`/`step` in the
integer epilogue change. A row sort by block size, to make tiles uniform, would
be a load-time repack (a permutation applied to the rows plus an output index
map), not a checkpoint change.

## Not done

* A prefill kernel faster than dequant + rocBLAS (the fused kernel is opt-in;
  see above). I8 prefill is parked (see above): kernel built and tested, not
  benchmarked or wired.
* OLMoE at TP=2 (no two GPUs were free together); I8 for the grouped MoE
  prefill path (F16 only; I8 batches over 16 tokens use the per-pair path).
* Stacked groups for models whose vLLM `load_weights` maps expert names itself
  instead of passing them to `RoutedExperts.load_weights` (OLMoE goes through
  `AutoWeightsLoader` and works; other MoE models are untested with stacks).
* The tuned dense Qwen2.5-0.5B concurrency failure was not investigated.
