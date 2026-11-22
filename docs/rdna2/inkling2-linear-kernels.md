# Inkling-2-Mini dense linears on RDNA2 (`inkling2_linear_*`)

Native HIP kernels for every dense projection of
`Inkling2LoopForConditionalGeneration` on gfx1030 (V620: wave32, no bf16 / fp8 /
fp4 dot hardware, `V_DOT2_F32_F16` is the matrix primitive). They replace the
`F.linear` (rocBLAS) calls of the paged text model, are HIP-graph capturable
(no host sync, no allocation, every pointer and shape is a kernel argument) and
need no Triton, rocBLAS, hipBLAS or CK in the serving path.

| What | Where |
| --- | --- |
| Kernels | `csrc/rocm/inkling2_linear_rdna2.cu` |
| Registration | `csrc/rocm/ops.h`, `csrc/rocm/torch_bindings.cpp`, `CMakeLists.txt` |
| Python ops | `vllm/_custom_ops.py` (`inkling2_linear_*`) |
| Weight packing | `vllm/models/inkling2_loop/amd/linear_pack.py` |
| Tests | `tests/kernels/quantization/test_inkling2_linear_rdna2.py` |
| Benchmark | `tools/rdna/inkling2/bench_linear.py` |

## 1. Coverage

Activations are fp16, accumulation is fp32 (`v_dot2_f32_f16`, or fp32 FMA for the
fp32 gate weights). Block states and the residual stream stay fp32 and are not
touched by these ops.

| Model call | Pack helper | Op | N x K at TP=1 |
| --- | --- | --- | --- |
| `q_proj` `gate_proj` `rel_proj` `sink_proj` | `inkling2_linear_pack_attn_in` | `grouped` | 3072, 3072, 384, 24 x 2304 |
| `k_proj` `v_proj` | same pack | `grouped` | 1024, 1024 x 2304 |
| dynamic-conv taps `k` `v` (layers 5, 11, 17, 23) | same pack | `grouped` | 32, 32 x 2304 |
| `o_proj.linear` | `inkling2_linear_pack_o_proj` | `grouped` | 2304 x 3072 |
| `MessageGate.down_proj` | `inkling2_linear_pack_down` | `grouped` | 64 x 4608 |
| `MessageGate.forget_gate_proj` + `message_gate_proj` | `inkling2_linear_pack_controller` | `f32w` | 2 x 2304 x 64 (fp32 W) |
| MoE router `gate.weight` | `inkling2_linear_pack_router` | `f32w` | 128 x 2304 (fp32 W) |
| `lm_head` + final softcap + vocab mask | `inkling2_linear_pack_lm_head` | `lm_head` | 200058 x 2304 |
| greedy argmax of the logits | - | `argmax_partial` / `argmax_reduce` | - |

The MoE experts, the four short convolutions, the attention itself and the
norms / residual glue belong to other kernels (`moe_*`, `inkling2_attn_*`,
`inkling2_glue_*`).

## 2. Op reference

All ops launch on the current stream, allocate nothing, never synchronise and
read every pointer from kernel arguments, so a captured graph replays them with
whatever the tensors hold at replay time. All tensors are CUDA/HIP tensors of
one device.

### 2.1 `inkling2_linear_grouped`

```text
inkling2_linear_grouped(Tensor[] xs, Tensor w, Tensor(a!)[] outs,
                        int[] seg_x, int[] seg_n, int[] seg_act=[],
                        float[] seg_p=[], int[] seg_valid=[], int variant=-1)
```

One launch computes up to 8 segments. Segment `i` multiplies the activations
`xs[seg_x[i]]` (`[M, K]`) with rows `[sum(seg_n[:i]), sum(seg_n[:i+1]))` of the
packed weight `w` (`[sum(seg_n), K]`), applies the epilogue and writes
`outs[i]` (`[M, seg_n[i]]`).

| Argument | Contract |
| --- | --- |
| `xs[j]` | fp16 `[M, K]`; unit column stride, row stride a multiple of 8 elements and `>= K`; 16-byte aligned base. All inputs share `M` and `K`. Strided views (for example `hidden[:, off:off+K]`) are fine. |
| `w` | fp16, contiguous, `[N_total, K]`, 16-byte aligned. `K % 8 == 0`, `seg_n[i] * K < 2^30`. |
| `outs[i]` | fp16 or fp32 `[M, seg_n[i]]`, unit column stride, row stride `>= seg_n[i]`. fp32 gives the exact fp32 accumulator (the TP o_proj partial). |
| `seg_act[i]` | `0` none, `1` sigmoid, `2` tanh, `3` softcap: `tanh(v / seg_p[i]) * seg_p[i]`. Empty means all `0`. |
| `seg_valid[i]` | columns `>= seg_valid[i]` of the segment are written as `-inf` (vocab padding). Empty means `seg_n[i]`. |
| `variant` | `-1` auto, see section 3. |

`M <= 16` takes the weight-streaming GEMV, `M >= 17` the tiled GEMM; the result
of both is the same fp32-accumulated dot product (the summation order differs,
so the two paths agree to a few fp32 ulp before the fp16 rounding).

### 2.2 `inkling2_linear_f32w`

```text
inkling2_linear_f32w(Tensor[] xs, Tensor w, Tensor(a!)[] outs,
                     int[] seg_x, int[] seg_n, int[] seg_act=[], int variant=-1)
```

Same segment model with an fp32 weight `[N_total, K]` and fp32 outputs. The
activations are fp16 (the reference rounds the `down_proj` output to fp16 and the
router input is the fp16 hidden state, so the fp16 values are exact fp32 inputs
of the reference). Used for the router (`[128, 2304]`) and the two controller
gate weights (`[2304, 64]` each; `act = 2` tanh).

### 2.3 `inkling2_linear_lm_head`

```text
inkling2_linear_lm_head(Tensor x, Tensor w, Tensor!? logits, Tensor!? part_val,
                        Tensor!? part_idx, float softcap, int n_valid,
                        int col_base, int variant=-1)
```

`logits[m, r] = softcap * tanh((x @ w^T)[m, r] / softcap)` for `r < n_valid`,
`-inf` for `n_valid <= r < N`; `softcap == 0` means no softcap. `logits` is fp32
`[M, N]` (the layout the V2 sampler reads; leave `N` as the real number of rows
of `w`, the output row stride may be larger). With `part_val` / `part_idx`
(fp32 / int32 `[M, nparts]`) the op also writes one argmax partial per row and
block: value and global index (`col_base + row`), first index wins ties, NaN
counts as `-inf`. `logits` may be `None` (partials only, `M <= 16`: the fused
greedy path never writes the 800 KB logits row per token). Finish the argmax
with `inkling2_linear_argmax_reduce`.

### 2.4 `inkling2_linear_argmax_partial` / `inkling2_linear_argmax_reduce`

```text
inkling2_linear_argmax_partial(Tensor logits, Tensor! part_val,
                               Tensor! part_idx, int n, int col_base)
inkling2_linear_argmax_reduce(Tensor part_val, Tensor part_idx, Tensor! out)
```

`argmax_partial` splits each row of fp32 `logits[M, >= n]` into `nparts =
part_val.size(1)` column slices (one block each) and stores the best value and
index (`+ col_base`) per slice; `argmax_reduce` merges the partials into the
int64 `out[M]`. Same tie and NaN rules as the fused path (and as the sampler's
`_argmax_partial`). Use them for `M > 16` or after a GEMM `lm_head`.

## 3. Variants

`variant = -1` picks by shape. A forced variant is for tests and tuning; an id
that does not fit the shape raises `RuntimeError`.

| Id | Kernel | Shape |
| --- | --- | --- |
| 0 | GEMV, 1 row / wave, 4 loads in flight | `R=1 U=4 KS=1` |
| 1 | GEMV, 2 rows / wave | `R=2 U=3 KS=1` |
| 2 | GEMV, 4 rows / wave | `R=4 U=2 KS=1` |
| 3 | GEMV, K split in 2 waves | `R=1 U=4 KS=2` |
| 4 | GEMV, K split in 4 waves | `R=1 U=4 KS=4` |
| 5 | GEMV, 2 rows, K split in 4 | `R=2 U=2 KS=4` |
| `cfg + 10 s`, `s = 1..5` | GEMV `cfg` with 1, 4, 8, 16, 32 persistent blocks per multiprocessor instead of 2 | grid size only |
| 100 | GEMM 128 x 128 tile | `BK=32`, 128 threads |
| 101 | GEMM 256 x 128 tile | `BK=32` |
| 102 | GEMM 128 x 64 tile | `BK=64` |
| 103 | GEMM 64 x 128 tile | `BK=64` |
| 104 | GEMM 32 x 128 tile | `BK=64` |
| 105 | GEMM 64 x 64 tile | `BK=64` |

GEMV variants take `M <= 16` (`R * MT <= 32`, with `MT` the activation rows per
pass: 1, 2, 4, 8 or 16 for `M` up to 1, 2, 4, 8, 16); the GEMM variants take any
`M`.

Auto choice (`variant = -1`):

- GEMV (`M <= 16`): `K <= 256` (the controller gates) variant 41 (2 rows per
  wave, 16 blocks per multiprocessor); `K < 2048` (o_proj at TP 2 and 4)
  variant 0, because the K split costs one sync per group for a few KB of
  weights; otherwise variant 4 for `M <= 8` or fewer than 512 weight rows, and
  variant 41 for `M` 9 to 16 (16 activation rows take two LDS chunks at
  K = 2304; many resident blocks hide the restaging).
- GEMM (`M >= 17`): `M` up to 32 variant 104, up to 64 variant 103, up to 192
  variant 102; below 1024 variant 102 when the weight has at most 4096 rows
  (o_proj) and 100 above; from 1024 variant 101.

Section 9 has the sweeps behind these choices.

## 4. Packing helpers (`linear_pack.py`)

A `LinearPack` holds the concatenated weight and the segment tables, allocates
the outputs and calls the right op:

```python
pack = inkling2_linear_pack_attn_in(q, gate, rel, sink, k, v, dyn_k, dyn_v)
outs = pack.alloc_outputs(max_tokens)        # once, graph-static
pack([hq, hk, hv], [o[:T] for o in outs])    # per step; row slices stay valid
```

| Helper | Row order (the segment order of every call) | Inputs `xs` | Output dtypes |
| --- | --- | --- | --- |
| `inkling2_linear_pack_attn_in` | `q`, `gate`, `rel`, `sink`, `k`, [`dyn_k`], `v`, [`dyn_v`] | `xs[0]=hq` for `q gate rel sink`, `xs[1]=hk` for `k dyn_k`, `xs[2]=hv` for `v dyn_v` | all fp16 |
| `inkling2_linear_pack_o_proj(w, out_dtype)` | `o` | `xs[0]` | fp16 or fp32 |
| `inkling2_linear_pack_down` | `down` | `xs[0]` (`[M, 4608]`) | fp16 |
| `inkling2_linear_pack_router` | `router` | `xs[0]` | fp32 |
| `inkling2_linear_pack_controller` | `forget`, `gain` (tanh) | `xs[0]` (`down` output, fp16) | fp32, fp32 |
| `inkling2_linear_pack_lm_head(w, vocab, tp, rank)` | the `rank` shard | - | fp32 logits |

Pass the same tensor three times for `xs` when a layer has no per-stream
messages. The `gate` output is the raw logit (the attention epilogue applies the
sigmoid); `sink` is `[T, nh]`; `dyn_k` / `dyn_v` are `[T, n_kv * 4]`.

### Tensor parallel

| | TP=1 | TP=2 | TP=4 |
| --- | --- | --- | --- |
| q heads / kv heads | 24 / 8 | 12 / 4 | 6 / 2 |
| `q`, `gate` rows | 3072 | 1536 | 768 |
| `rel` rows (`16 * nh`) | 384 | 192 | 96 |
| `sink` rows | 24 | 12 | 6 |
| `k`, `v` rows | 1024 | 512 | 256 |
| dyn taps rows (`4 * n_kv`) | 32 | 16 | 8 |
| `o_proj` K (row-parallel) | 3072 | 1536 | 768 |
| `lm_head` rows per rank | 200058 | 100032 | 50016 |

`o_proj` is K-sharded: every rank multiplies its heads' attention output by its
`[2304, nh_local * 128]` slice and writes the fp32 partial (`out_dtype=float32`);
the all-reduce stays outside the op (`TensorParallel.all_reduce` of the fp32 `[T, H]`
partial, `amd/parallel.py`) and its fp32 sum feeds `inkling2_sconv_static`, which
takes fp32 or fp16 input. `lm_head` is vocab-sharded: the padded vocabulary (200058
-> 200064, a multiple of 64) is divided by the TP size, rank `r` owns rows `[r *
rows, (r + 1) * rows)` (the last rank's tail is zero padding), `LMHeadPack.n_valid`
masks the padding to `-inf` and `LMHeadPack.col_base = r * rows` offsets the
argmax indices (`greedy_token_ids` gathers the per-rank partial maxima and ids,
`compute_logits` the fp32 logits shards). The K and N tails of every shape (`K % 8
== 0` is the only requirement) are handled in the kernels.

## 5. Kernel design

### 5.1 GEMV (`M <= 16`, weight streaming)

A block is 8 waves. A *group* is `RB = (8 / KS) * R` weight rows; wave `w` takes
`R` rows of the group and the `w % KS`-th slice of K. Each lane streams 16-byte
weight vectors (8 halves) with `U` loads in flight per row, applies every vector
to all `MT` activation rows (held in LDS in 16-byte vectors: the weights are read
from DRAM exactly once whatever `M`) and accumulates with `v_dot2_f32_f16`. The
32 lanes are reduced by a reduce-scatter (`log2(R*MT)` exchange steps, then a
butterfly), the `KS` slices meet in LDS, the epilogue (act, `-inf` mask) writes
fp16 or fp32. The grid is `bpc * multiProcessorCount` persistent blocks (`bpc = 2`
and `multiProcessorCount = 36` on the V620: 72 blocks; the `cfg + 10 s` variants
and the auto choice for `M` 9 to 16 use more), each owning `gpb` contiguous
groups. The activations are staged once per block and segment (48 KB of LDS);
for `MT = 16` with `K = 2304` (or `K = 4608`) they take two (three) chunks that
are re-staged for every group, which is why `M` 9 to 16 moves fewer weight bytes
per second than `M <= 8` (section 9). The fused argmax keeps the running best
of each lane across its groups and merges the waves at the end of the block
(`nparts` blocks exactly; `8 * multiProcessorCount` = 288 for `M` above 4, see
section 9).

### 5.2 GEMM (`M >= 17`)

Register-tiled dot2 GEMM adapted from the tuned dense prefill kernel
(`rdna_hint_flex_dense_gemm_kernel.cuh`): global loads through buffer descriptors
(out-of-range rows and the K tail read as zero), a swizzled LDS tile, and a
per-design thread tile. Tiles of all segments are laid out back to back in one
launch.

### 5.3 f32w

Router and controller gates: a 64 x 128 fp32 FMA tile (K in steps of 16) for
`M >= 17`, the GEMV with an fp32 weight vector (4 elements per lane load) for
`M <= 16`.

## 6. Graph capture and scratch

Nothing is allocated or synchronised; the launch parameters (pointers, strides,
group tables) are passed by value. For a captured graph allocate the outputs of
`alloc_outputs` once for the largest batch and pass row slices `out[:m]`
(contiguous, so every row stride stays valid). `M` is a launch parameter: a graph
is captured per padded batch size, as the model runner does. There is no scratch
tensor: the argmax partials are caller-provided (`[M, nparts]` fp32 / int32).

## 7. Tests

`tests/kernels/quantization/test_inkling2_linear_rdna2.py` (677 tests, gfx1030
only; other architectures skip) compares every kernel with an fp64 reference
computed from the same fp16 / fp32 inputs. Tolerance: 1 fp16 ulp plus
`5e-6 * max|ref|` for fp16 outputs, `5e-6 * max|ref|` for fp32 outputs, and
`1e-6 * max|pre-activation|` for fp32 outputs behind a tanh (the accumulation
noise passes through a tanh near 0 one to one, so it scales with the
accumulator, not with the bounded output); the `-inf` masks compare exactly.
Coverage: every model shape at TP 1 / 2 / 4, `M` in {1, 2, 3, 4, 8, 9, 16, 64,
257, 1024}, every forced variant that fits, K and N tails (the activation rows
are followed by NaN, so a read past `K` poisons the result), strided
activations and outputs, the epilogues and the `-inf` mask, the vocab shards
(including the padded last rank), the fused argmax (exact ties inside one
thread, across threads and across partials) and a captured-graph replay. A
V620 takes about 3.5 minutes for the 607 tests outside the lm_head and argmax
group and 26 seconds for the other 70.

```bash
python -m pytest tests/kernels/quantization/test_inkling2_linear_rdna2.py -q
```

## 8. Wiring into the model

`LinearPack` objects are built once after the weights are on the device (fp16
`[N, K]`; fp32 for the router and the controller gates) and called with
preallocated outputs. Every site below replaces a `F.linear` / `nn.Linear` call
of the current `amd/*.py`.

| Site (current code) | Replacement |
| --- | --- |
| `attention.py` `q_proj.linear(hq)`, `rb.linear(hq)`, `sink_proj(hq)`, `o_proj.gate_proj(hq)`, `_kv_stream` `proj.linear(h)`, `DynamicSconv.weight_proj(h)` | ONE `inkling2_linear_pack_attn_in(...)` call per layer |
| `attention.py` `o_proj.linear(attn)` | `inkling2_linear_pack_o_proj(weight, out_dtype)` |
| `blocks.py` `MessageGate`: `down_proj`, `forget_gate_proj`, `message_gate_proj` | `inkling2_linear_pack_down` then `inkling2_linear_pack_controller` |
| `moe.py` `Inkling2LoopTopKRouter.forward` `F.linear(flat, weight)` | `inkling2_linear_pack_router` |
| `model.py` `compute_logits` `F.linear(...)`, `.float()`, `tanh(x / 30) * 30` | `inkling2_linear_pack_lm_head` |

```python
from vllm.models.inkling2_loop.amd import linear_pack as lp

# Inkling2Attention, once after load_weights (weights already fp16 on device):
dyn = (None, None) if self.is_sliding else (
    self.k_proj.sconv.weight_proj.weight, self.v_proj.sconv.weight_proj.weight)
self._attn_in = lp.inkling2_linear_pack_attn_in(
    self.q_proj.linear.weight, self.o_proj.gate_proj.weight,
    self.relative_position_bias.linear.weight, self.sink_proj.weight,
    self.k_proj.linear.weight, self.v_proj.linear.weight, *dyn)
self._attn_in_outs = self._attn_in.alloc_outputs(max_num_batched_tokens)
self._o_pack = lp.inkling2_linear_pack_o_proj(self.o_proj.linear.weight)

# Inkling2Attention.forward: one launch replaces six matmuls (+ 2 dyn taps)
outs = [o[:T] for o in self._attn_in_outs]  # q, gate, rel, sink, k, [dyn_k], v, [dyn_v]
self._attn_in([hq, hk, hv], outs)
```

The outputs are exactly the tensors `inkling2_kv_prep` / `inkling2_attention`
consume (all fp16, `gate` is the raw logit and `sink` is `[T, n_heads]`). The
per-layer dense weights are freed after packing (the pack owns the only copy), so
the pack costs no extra memory. The controller and router follow the same
pattern: `pack(xs, outs)` with the `[T, 64]` fp16 `gate_hidden` or the fp16
hidden state as input and fp32 outputs (`ACT_TANH` is applied in the kernel for
the two gates).

```python
# model.py: logits [T, vocab] fp32, softcap and padding mask in the GEMM epilogue
self._head = lp.inkling2_linear_pack_lm_head(
    self.lm_head.weight.data, cfg.vocab_size, tp_size, tp_rank)

def compute_logits(self, hidden_states):
    logits = hidden_states.new_empty(
        hidden_states.shape[0], self._head.rows, dtype=torch.float32)
    self._head(hidden_states, logits, softcap=self._softcap)
    return logits
```

For greedy decoding the fused argmax removes the second pass over the logits:
pass `part_val` `[M, P]` fp32 and `part_idx` `[M, P]` int32 (`P >= 1`; the tests
use 1, 7, 61 and 64). For `M <= 16` the partial is produced in the GEMV epilogue
and `logits=None` skips the logits store; for larger `M` the GEMM writes the
logits and `inkling2_linear_argmax_partial` reduces them. Then
`ops.inkling2_linear_argmax_reduce(part_val, part_idx, out)` writes the int64
token ids `[M]` (the first index wins ties, NaN counts as `-inf`; the indices
already include `col_base`).

## 9. Measurements

`tools/rdna/inkling2/bench_linear.py` on one Radeon Pro V620 (gfx1030,
`multiProcessorCount = 36`, 512 GB/s DRAM peak from the spec sheet). Each case
replays a HIP graph over a rotating set of weight copies (at least 700 MB; the
Infinity Cache holds 128 MB), so every call streams its weights from DRAM. The
implementations run interleaved in rotating order, 4 rounds of 3 replays,
median. `rocBLAS+torch` is what the model runs today: one `F.linear` per
projection (plus `.float()` and `tanh(x / 30) * 30` for lm_head, `x.float()` for
the router, two `F.linear` and `tanh` for the controller). `skinny` is the
in-repo `gemv_f16_rdna2` (`M <= 32`, fp16 output, no epilogue, one launch per
projection). The same kernel varies by up to about 5 % (decode) and 10 %
(prefill) between runs. `attn_in` is the fused q, gate, rel, sink, k, v launch
(six `F.linear` in the baseline).

### 9.1 Decode, TP 1 (microseconds per call)

| op | M | rocBLAS+torch | skinny | native auto | GB/s | x rocBLAS | best fixed variant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attn_in | 1 | 504.8 | 120.5 | 99.3 | 399 | 5.1 | v50 (99.0) |
| attn_in | 4 | 506.7 | 123.3 | 103.2 | 384 | 4.9 | v51 (101.0) |
| attn_in | 16 | 531.7 | 312.3 | 121.6 | 326 | 4.4 | v41 (123.5) |
| o_proj | 1 | 169.9 | 42.1 | 40.4 | 351 | 4.2 | v4 (40.4) |
| o_proj | 4 | 164.3 | 48.5 | 42.1 | 336 | 3.9 | v4 (42.1) |
| o_proj | 16 | 157.4 | 73.8 | 66.2 | 214 | 2.4 | v50 (64.7) |
| down | 1 | 120.6 | 5.9 | 5.5 | 108 | 22.0 | v4 (5.5) |
| down | 4 | 120.8 | 6.3 | 6.8 | 87 | 17.8 | v5 (6.8) |
| down | 16 | 138.4 | 42.4 | 13.1 | 45 | 10.6 | v4 (13.1) |
| router (fp32) | 1 | 126.5 | - | 6.7 | 177 | 18.9 | v0 (6.6) |
| router (fp32) | 4 | 127.8 | - | 7.6 | 155 | 16.8 | v3 (7.6) |
| router (fp32) | 16 | 130.9 | - | 13.6 | 87 | 9.7 | v5 (12.3) |
| controller (fp32) | 1 | 23.1 | - | 6.4 | 183 | 3.6 | v31 (6.4) |
| controller (fp32) | 4 | 23.2 | - | 7.2 | 164 | 3.2 | v41 (7.2) |
| controller (fp32) | 16 | 23.8 | - | 15.4 | 77 | 1.5 | v1 (12.9) |
| lm_head | 1 | 8266.2 | 2100.8 | 2100.8 | 439 | 3.9 | v4 (2102.1) |
| lm_head | 4 | 8284.6 | 2118.3 | 2132.8 | 432 | 3.9 | v4 (2129.0) |
| lm_head | 16 | 8276.9 | 4312.9 | 2519.6 | 366 | 3.3 | v51 (2479.5) |
| lm_head + argmax | 1 | 8189.9 | - | 2095.7 | 440 | 3.9 | v4 (2080.3) |
| lm_head + argmax | 4 | 8331.2 | - | 2160.8 | 427 | 3.9 | v4 (2095.1) |
| lm_head + argmax | 16 | 8490.9 | - | 3102.4 | 297 | 2.7 | v31 (2927.3) |

The weight-streaming ops (lm_head, attn_in, o_proj) reach 351 to 440 GB/s at
`M <= 4`, against 512 GB/s of peak; lm_head at 440 GB/s is the best of the two
kernels we have (the `skinny` kernel also takes 2100.8 us). The small ops
(down, router, controller) are launch and latency bound: 5.5 to 7 us against
120 to 126 us for the rocBLAS composition (23.1 us for the controller). The
fused argmax costs what the lm_head alone costs for `M <= 4` (2095.7 us against
2100.8 us at `M = 1`, 2160.8 against 2132.8 us at `M = 4`) and replaces the
softcap, the `argmax` over 200058 logits and their launches. Fused argmax with
`nparts = 64` is latency bound at `M >= 8`: 2395 us (`M = 8`) and 3102 us
(`M = 16`); with `nparts = 288` (`8 * multiProcessorCount`) 2192 us and 2485 us
(371 GB/s).

### 9.2 Tensor parallel

Decode (microseconds per call; rank-local shapes):

| op | TP | M | rocBLAS+torch | skinny | native auto | GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| attn_in | 2 | 1 | 383.9 | 76.8 | 53.8 | 369 |
| attn_in | 2 | 16 | 500.1 | 224.3 | 73.0 | 272 |
| attn_in | 4 | 1 | 508.4 | 44.2 | 31.5 | 314 |
| attn_in | 4 | 16 | 513.0 | 203.4 | 48.0 | 206 |
| lm_head | 2 | 1 | 4046.8 | 1056.7 | 1057.1 | 436 |
| lm_head | 2 | 16 | 4198.9 | 2260.9 | 1257.0 | 367 |
| lm_head | 4 | 1 | 2015.5 | 532.7 | 533.7 | 432 |
| lm_head | 4 | 16 | 2109.2 | 1131.8 | 613.2 | 376 |
| lm_head + argmax | 2 | 1 | 4088.0 | - | 1063.2 | 434 |
| lm_head + argmax | 4 | 1 | 2058.1 | - | 538.6 | 428 |
| o_proj (K sharded) | 2 | 1 | 49.2 | 24.7 | 25.4 | 278 |
| o_proj (K sharded) | 4 | 1 | 45.7 | 11.3 | 24.6 | 144 |

The o_proj rows at TP 4 exposed a slowdown against `skinny` (24.6 us against
11.3 us): splitting K = 768 over four waves costs a sync per group. The sweep
of forced variants below (same bench, `--variants 0,1,2,3,4,5,...`) led to the
current auto choice (variant 0 for `K < 2048`); the numbers are the forced
variant 0 against the variant 4 that the auto choice used before:

| o_proj | M | rocBLAS+torch | skinny | variant 4 | variant 0 |
| --- | --- | --- | --- | --- | --- |
| TP 2 (K 1536) | 1 | 49.1 | 24.7 | 25.4 | 24.7 |
| TP 2 (K 1536) | 4 | 50.8 | 24.9 | 26.9 | 25.4 |
| TP 2 (K 1536) | 8 | 56.1 | 25.7 | 28.4 | 26.7 |
| TP 2 (K 1536) | 16 | 95.4 | 41.6 | 36.7 | 30.4 |
| TP 4 (K 768) | 1 | 45.6 | 11.3 | 24.6 | 11.8 |
| TP 4 (K 768) | 4 | 45.9 | 12.1 | 24.6 | 12.6 |
| TP 4 (K 768) | 8 | 45.9 | 13.1 | 24.7 | 13.5 |
| TP 4 (K 768) | 16 | 46.0 | 24.7 | 27.1 | 16.8 |

Prefill at `M = 1024` (TFLOP/s, rocBLAS+torch against native auto): attn_in TP 2
13.4 / 23.4, TP 4 11.6 / 23.0; o_proj TP 2 20.7 / 24.5, TP 4 22.0 / 23.4;
lm_head TP 2 17.8 / 24.0, TP 4 17.7 / 23.9; with the dyn taps attn_in TP 1 17.2 /
24.1, TP 2 12.3 / 21.5, TP 4 10.0 / 18.5 (the narrow TP 4 segments fill fewer
tiles).

### 9.3 Prefill (microseconds per call, TFLOP/s in parentheses)

| op | M | rocBLAS+torch | native auto | x rocBLAS | best fixed design |
| --- | --- | --- | --- | --- | --- |
| attn_in | 17 | 597 (1.1) | 129 (5.2) | 4.64 | v104 (131, 5.1) |
| attn_in | 64 | 687 (3.7) | 138 (18.4) | 4.97 | v103 (156, 16.3) |
| attn_in | 256 | 1025 (9.9) | 416 (24.4) | 2.46 | v100 (461, 22.0) |
| attn_in | 1024 | 2203 (18.4) | 1626 (25.0) | 1.36 | v101 (1606, 25.3) |
| attn_in | 4096 | 8503 (19.1) | 6436 (25.2) | 1.32 | v101 (6436, 25.2) |
| attn_in | 8192 | 16312 (19.9) | 12948 (25.1) | 1.26 | v101 (12963, 25.0) |
| o_proj | 17 | 220 (1.1) | 97 (2.5) | 2.27 | v105 (94, 2.6) |
| o_proj | 64 | 186 (4.9) | 119 (7.6) | 1.56 | v105 (97, 9.3) |
| o_proj | 256 | 236 (15.3) | 229 (15.8) | 1.03 | v102 (193, 18.8) |
| o_proj | 1024 | 705 (20.6) | 598 (24.3) | 1.18 | v101 (597, 24.3) |
| o_proj | 4096 | 2656 (21.8) | 2310 (25.1) | 1.15 | v101 (2305, 25.1) |
| o_proj | 8192 | 5028 (23.1) | 4591 (25.3) | 1.10 | v101 (4593, 25.2) |
| lm_head | 17 | 8406 (1.9) | 2500 (6.3) | 3.36 | v104 (2902, 5.4) |
| lm_head | 64 | 9131 (6.5) | 3732 (15.8) | 2.45 | v105 (3798, 15.5) |
| lm_head | 256 | 13186 (17.9) | 10941 (21.6) | 1.21 | v101 (10596, 22.3) |
| lm_head | 1024 | 54802 (17.2) | 39043 (24.2) | 1.40 | v101 (39847, 23.7) |
| lm_head | 4096 | 228681 (16.5) | 159506 (23.7) | 1.43 | v101 (160311, 23.6) |

The lm_head rows at `M` 1024 and 4096 come from an earlier window with the same
design choice for `M >= 1024`.

For `N` of 8664 rows and more (attn_in, lm_head) the GEMM sustains 24 to 25.6
TFLOP/s from `M = 256` on; the dot2 peak is 18.43 TFLOP/s per GHz, so about 68 %
at an assumed 2.0 GHz [INFERENCE: the shader clock under load was not
measured]. rocBLAS reaches 17 to 23 TFLOP/s on the same shapes. o_proj
(`N = 2304`) fills the CUs worse for `M` 192 to 768: the 128 x 128 tile gave 229
us at `M = 256` (15.8 TFLOP/s, 1.03x rocBLAS) where the forced 128 x 64 tile
takes 193 us (18.8 TFLOP/s); the forced 128 x 64 numbers for o_proj are 271 us
(`M = 384`, 20.0 TFLOP/s), 335 us (512, 21.6) and 490 us (768, 22.2). The auto
choice now selects the 128 x 64 tile up to 4096 rows (section 3). The per
projection bench ops `q_proj` and `kv_proj` (`--ops q_proj,kv_proj`) time the
3072 x 2304 and 1024 x 2304 projections alone.

### 9.4 Maximum error against the fp64 reference

`errscan` (random fp16 inputs, `K` as in the model) reports the worst element
of every op; fp16 outputs are limited by their own rounding (half an ulp is
2.4e-4 to 4.9e-4 of the largest value), fp32 outputs show the accumulation noise
only.

| op | M = 1 | M = 16 | M = 1024 |
| --- | --- | --- | --- |
| attn_in, 8 segments, fp16 | 9.8e-4 (2.7e-4) | 1.9e-3 (4.7e-4) | 2.0e-3 (4.1e-4) |
| o_proj fp16 | 9.6e-4 (2.8e-4) | 1.5e-3 (3.6e-4) | 1.9e-3 (3.9e-4) |
| o_proj fp32 | 4.7e-7 (1.4e-7) | 6.0e-7 (1.4e-7) | 8.5e-6 (1.7e-6) |
| down fp16 | 9.8e-4 (3.9e-4) | 9.7e-4 (2.5e-4) | 1.8e-3 (3.9e-4) |
| controller forget, tanh | 3.4e-7 | 6.0e-7 | 1.7e-6 |
| controller gain, tanh | 5.8e-7 | 5.3e-7 | 1.5e-6 |
| router fp32 | 4.0e-7 (1.3e-7) | 4.5e-7 (1.4e-7) | 7.5e-6 (1.4e-6) |
| lm_head softcap 30, fp32 | 7.5e-7 (1.6e-7) | 8.9e-7 (1.7e-7) | M = 64: 7.7e-6 (1.4e-6) |

Absolute error, relative to the largest output in parentheses.

### 9.5 What the sweeps decided

- More resident blocks for `M` 9 to 16: attn_in 147 us (2 blocks per
  multiprocessor) against 123 us (16); lm_head 2765 against 2520 us; flat for
  `M <= 8`.
- `K <= 256` (controller gates, `K = 64`): the K split cost 32.7 us, whole rows
  with 16 blocks per multiprocessor 6.4 us (rocBLAS composition 23.1 us).
- `K < 2048`: the o_proj table of section 9.2.
- Fused argmax: `nparts = 288` for `M` above 4.
- GEMM: the 256 x 128 tile from `M = 1024` (25.0 to 25.6 TFLOP/s); from 193 to
  1023 rows the 128 x 128 tile for wide outputs and the 128 x 64 tile up to 4096
  rows; smaller tiles below.

## 10. Mutation checks

Hand-run mutations: a JIT copy of the hipified TU carries a runtime switch
`g_mut`; the harness redirects `torch.ops._rocm_C.inkling2_linear_*` to it and
runs the pytest selection that must fail. With no mutation active every
selected test passes; each mutation below made at least one test fail.

| Id | Mutation | Killed by |
| --- | --- | --- |
| M1 | GEMV K slice per wave rounds down instead of up | `test_tails` |
| M2 | GEMV second LDS chunk re-reads the first (chunk offset dropped) | `test_single_segment` |
| M16 | GEMV ignores the activation row stride (hunk of M2) | `test_strided_activations` |
| M3 | GEMV `-inf` mask off by one column | `test_epilogues_and_mask` |
| M4 | softcap `* p` dropped | `test_epilogues_and_mask`, `test_lm_head` |
| M5 | GEMM activation K-tail predicate admits 8 extra elements | `test_tails` (NaN-padded rows) |
| M6 | GEMM `-inf` mask off by one column | `test_epilogues_and_mask` |
| M7 | GEMM B-tile word rotation dropped | `test_single_segment` |
| M17 | GEMM ignores `lda` | `test_strided_activations` |
| M8 | f32w B-tile store writes the wrong half | `test_f32w_router` |
| M9 | f32w A-tile uses the low half twice | `test_f32w_controller` |
| M10 | tanh epilogue dropped | `test_f32w_controller` |
| M11 | fused GEMV argmax drops `col_base` | `test_lm_head` |
| M12 | fused GEMV argmax wave-reduce tie takes the highest index | `test_lm_head_argmax_ties` |
| M13 | `argmax_partial` per-thread tie takes the highest index | `test_argmax_partial_ties` |
| M14 | `argmax_reduce` per-thread tie takes the highest index | `test_argmax_reduce_ties` |
| M15 | `argmax_partial` drops `col_base` | `test_argmax_partial_reduce` |
| P1 | pack: the `v` stream reads the `k` input (`seg_x`) | `test_attn_in_pack`, `test_pack_attn_in_order_and_shapes` |
| P2 | pack: lm_head `col_base + 1` | `test_pack_lm_head_shards`, `test_lm_head` |

M5, M13 and M14 first survived: the K tail test read finite data past `K`
(garbage times a zero B chunk is zero), and no test put two equal maxima in one
thread. The activation rows of `test_tails` are now followed by NaN, and the two
tie tests were added.

## 11. Open risks and not covered

- Not covered by a native op in this slice: the MoE expert GEMMs
  (`moe_lowp_dot2_rdna2.cu`), the attention-residual block-state matmul and the
  `rel_logits` einsum (glue / attention kernels of other agents), the vision
  tower linears and the DSpark draft head (their own tasks). Every dense linear
  of the text model is covered; the text model has no dense MLP and no shared
  experts.
- The model is not wired yet: `model.py`, `attention.py`, `blocks.py` and
  `moe.py` still call `F.linear` until the patches of section 8 are applied, so
  only per-op numbers exist; the end-to-end decode step was not measured.
- GEMV for `M` 9 to 16 reaches 326 to 366 GB/s (attn_in, lm_head) against
  400 to 440 GB/s for `M <= 8`: the activation chunks are re-staged per group.
  Next step: prefetch the next chunk into registers. Greedy argmax fused at
  `M = 16` needs `nparts` of at least 288 (section 9).
- GEMM sustains 25 to 26 TFLOP/s, 68 to 70 % of the dot2 peak at an assumed
  2.0 GHz (36.9 TFLOP/s) [INFERENCE: the shader clock under load was not
  measured]. The `M` thresholds of the auto choice sit inside the run-to-run
  noise of the bench (up to 10 % between runs of the same kernel).
- Operands: `K` a multiple of 8, activations 16-byte aligned with a row stride
  that is a multiple of 8 halves (checked), `N` and `M` arbitrary.
- TP is tested per rank on one GPU with sliced weights; the all-reduce of the
  fp32 o_proj partials and the gather of the vocab shards are the caller's.
- The device code exists for gfx1030 only: the other architectures compile empty
  kernels (compiled for gfx90a, gfx942, gfx950, gfx1100 and gfx1200, not run).
