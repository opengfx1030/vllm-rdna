# Inkling-2 decode: fused attention-side chain (graph node count)

Flag `VLLM_INKLING2_FUSE_ATTN_CHAIN` (default **on**; `0` or empty restores the separate
launches).  Every fusion is **bit-identical** to the launches it replaces: same fp32 /
fp16 expressions in the same order, same state writes (oracle tests compare with the shipped
kernels bit for bit; `tools/rdna/inkling2/fuse_shadow_check.py` does the same on the real model).
Files: `csrc/rocm/inkling2_fuse_attn_rdna2.cu`, `vllm/models/inkling2_loop/amd/fuse_chain.py`,
`tests/kernels/core/test_inkling2_fuse_attn_rdna2.py`,
`tools/rdna/inkling2/{bench_fuse_attn,fuse_attn_jit,fuse_attn_mutants,fuse_shadow_check}.py`.

**Status:** the message batching is wired into the model and on by default.  The kernel file
also holds `inkling2_fuse_attn_res_conv`, tested bit exact, mutation tested and benchmarked
below, but **not called by the model**: wired into the layer visits it removes 91 more nodes
per step and gains only 0.06 ms of a 17.95 ms step (see *End to end*).

## What is fused

| fusion | op | launches before -> after | applies to |
|---|---|---|---|
| message `attn_res` of every layer of a loop pass | `inkling2_fuse_msgs_fill` | 12 -> 1 per pass (22 fewer per step) | decode and verification batches of at most 16 tokens, any TP |
| short conv + the `attn_res` that reads it | `inkling2_fuse_attn_res_conv` | 2 -> 1 | batches of single-token requests at TP = 1 (`attention.conv_writes_in_op`); **not called by the model** |

The messages of a pass depend only on the block states of the *previous* pass, which are
fixed while the pass runs: the layers' `attn_res(prev_blocks, prev_x0 + prev_x1, query_j)` are
independent, so one grid `(T, layers)` computes them all into `[T, L, 4, H]` fp32 and the
message gates read the rows of their layer (`MessageBatch.select`).  Larger chunks (prefill)
keep the per-layer launch.

The conv op (not called by the model) puts the depthwise conv of a stream inside the
`attn_res` that consumes it:
the post-attention conv inside the MLP-input `attn_res` of the same visit, the post-MLP conv
of a visit inside the attention-input `attn_res` of the next visit of the pass (the conv of
the last visit of a pass stays separate: the pass-boundary `copy_states` needs its output).
Prefill, verification chunks (`T > 1` per request: the state write of one token races the tap reads of the others, which is why `conv_writes_in_op` needs one token per request), PIECEWISE graph segments and TP > 1 keep the separate conv launches.

## Why a removed node is worth ~2.3 us, and when it is not

Measured on gfx1030 (V620), graph replays, no host involvement:

* an **empty 1-block kernel in a 1500-node graph chain costs 3.07 us per node**; code size up
  to 8 KB does not change it, execution time grows with the executed instructions only
  (~0.6 ns each).  I-cache / code size is not the limit.
* an in-kernel `wall_clock64` timeline of a chain of the real kernels (99.7 ticks/us) puts the
  **gap between the last instruction of one node and the first of the next at 2.3 us**: that is
  what a removed node saves *if the fused body is not longer than the sum of the two bodies*.
  (A profiler shows ~4.4 us "gaps": that figure is inflated by the profiler.)
* the first fused conv kernel was **0.3 us slower** than the two launches (body 10.4 us).  Its
  body waited for a long dependent chain (`seq_idx -> query_start_loc -> seq_lens -> block_table`
  for the ring write and the three history taps) behind `s_waitcnt vmcnt(0)` drains that
  convert-on-load code forces, and the wave reductions went through `__shfl_xor` (LDS
  crossbar, ~15 serial rounds).  What took the body to 4.5 us: raw 16-byte loads converted at
  use (no drains), every independent load (stored states, `A` / `P` rows, norm weights, conv
  weights) issued first, the metadata chain on the scalar unit while they are in flight, the
  four `block_table` entries (current page + three history lags) loaded together, the
  ring-write page hoisted out of the lane loop, one barrier + a redundant per-wave softmax, and
  DPP / `permlanex16` butterflies instead of `__shfl_xor` (the same additions in the same
  order, so the same rounding).

## Measurements (V620, T = 1, 2 stored states, graph-timed, weights and states rotated cold over 1500 units)

| chain (per layer visit) | unfused (sconv + attn_res) | fused | saved |
|---|---|---|---|
| attention input, 1 query + norm (sliding layer) | 12.19 us | 6.82 us | **5.37 us** |
| attention input, 3 queries + norm (full-attention layer) | 15.66 us | 11.05 us | **4.61 us** |
| attention input, raw (layer with messages) | 9.73 us | 6.02 us | **3.70 us** |
| MLP input, norm | 12.16 us | 6.83 us | **5.33 us** |
| MLP input, norm, block boundary (append) | 12.35 us | 6.95 us | **5.40 us** |
| MLP input, raw (layer with messages) | 9.73 us | 6.03 us | **3.70 us** |

(warm 16 MB-flush regime: 2.7 - 3.8 us.)  Batched messages, 12 layers per pass, T = 1:
93.55 us -> 16.85 us, **76.7 us per pass**.

**The conv rows did not carry over to the real decode graph** (next section): there the fused
kernel takes as long as the two launches it replaces, so the step gains the node gap only.
The microbenchmark (2 stored states, one token, hot conv weights) is not a predictor of the
in-graph cost of the fused conv; the batched messages did carry over (predicted 0.153 ms per
step, kernel time measured -0.169 ms, step -0.20 ms).

## End to end (V620 HIP 4, TP = 1, FULL_AND_PIECEWISE graphs, gate engine with chunk 4)

| config | batch 1 | batch 4 | kernels / decode step | kernel time / step |
|---|---|---|---|---|
| `CHAIN=0` (two runs) | 18.15 / 18.17 ms/token | 35.40 / 35.34 ms | 735 | 16.788 ms |
| `CHAIN=1` (messages batched, this change) | **17.95 ms/token** (-0.20) | 34.93 ms (-0.47) | **713** (-22) | 16.619 ms (-0.169) |
| `CHAIN=1` + conv fused into `attn_res` (not in the tree) | 17.89 ms/token (-0.06 more) | 34.91 ms | **622** (-91 more) | 16.734 ms (+0.115) |

Kernel counts are decode-only (kineto traces of 65 and 33 generated tokens, difference / 32 steps).  The messages batch pays what the microbenchmark says (22 nodes: 0.153 ms predicted, 0.169 ms of kernel time and 0.20 ms of step time measured).  NLL gate (`real_check.py`, 799 positions, limit 0.1026 = 2 x floor 0.0463 + 0.01): 0.0405, PASS.  Greedy (30 generations): OFF vs OFF 22/30 identical (the run-to-run floor of the expert GEMMs' fp16 atomics), OFF vs messages 20/30, divergences at near-ties only; the bit-exactness evidence is the oracle and the shadow check.

## Correctness

* `tests/kernels/core/test_inkling2_fuse_attn_rdna2.py`: 384 tests, every fused output
  (`partial_out`, `out`, `x1_out`, the conv state, the appended block) **bit-identical** to the
  shipped op chain over token counts, hidden sizes (ITER 1..4), stack depths up to the
  register limit and beyond, block boundaries, norm / raw, 1 and 3 queries, fp16 and fp32 norm
  weights, padded tokens, the null cache block, page crossings of the conv history and ring
  rows; plus fp64 torch oracles of the conv and of the attention residual.
* `tools/rdna/inkling2/fuse_attn_mutants.py`: 23 hand mutants (conv tap order / scale /
  residual tap, state write skipped / wrong column / null block, history ring row / block /
  token position / start of sequence, padded rows, `x1_out` value, partial sum lane, appended
  slot, norm epsilon, norm weight row / prefetch half / reduction waves, message query layer /
  output layer, attention scale, softmax wave order, coefficient lane):
  22 of 23 killed on the first run; the survivor (`norm_weight_prefetch_hi`, fp32 norm weights of the one-query kernel) was a test gap (the dtype test only ran three queries); the test now runs both and kills it: **23 / 23**, the unmutated file passes.
* `tools/rdna/inkling2/fuse_shadow_check.py`: boots the real checkpoint eagerly and runs every
  fused launch next to the shipped launches on the same inputs (clones of the state), bit
  exact and deterministic (no atomics, unlike a greedy comparison, which has the run-to-run
  floor of the expert GEMMs): 118 batched message launches, each compared with 12 per-layer launches, **0 mismatches** (`SHADOW CHECK PASS`).
* the kernel file compiles for `gfx1030;gfx1100;gfx90a;gfx942` (the RDNA2 body is guarded).

## Limits and next steps

* `inkling2_fuse_attn_res_conv` is not called by the model: in the graph the fused kernel
  takes as long as the two launches it replaces (9.6 - 10.0 us against 3.5 + 5.8 - 7.1 us),
  so the 91 nodes it removes gain 0.06 ms per step.  It needs one token per request
  (`conv_writes_in_op`).
* The messages batch serves decode and verification batches of at most 16 tokens
  (`MSGS_BATCH_TOKENS`); larger chunks keep one launch per layer.
* Kernel: 192 threads per token, 164 VGPRs, latency bound; the DPP / `permlanex16` butterflies are gfx10+ wave32 code behind `__HIP__RDNA2__` (other archs compile the generic `__shfl_xor` path, checked for gfx1100, gfx90a, gfx942).
* The kernels are latency bound (one 192-thread block per token).  The same lessons apply to
  the shipped `attn_res` / `sconv` kernels (norm weights loaded at the very end: a dependent
  round trip; `__shfl_xor` reductions): plain `attn_res` variants with prefetched weights and
  DPP reductions would save ~1 us per node; the five pass-boundary `copy_states`, the
  `kv_prep` and the message gates are separate nodes that could join their neighbours.
