# Silicon: LSA on gfx1030 (V620)

**Status: untested explore draft.** No kernel in this PR runs on a GPU.
The card figures below are the ones this fork already uses: 32 GB, 512 GB/s,
no FP8, no bf16 dot, fp16 `v_dot2_f32_f16`, int8 `v_dot4_i32_i8`. Bandwidth
arithmetic uses 512×10^9 bytes/s and is a floor, not a timed kernel.

## What the upstream kernels actually are

Paper Appendix F.1 and `src/lsa/core/kernels/dispatch.py` at commit
`906be195c1052aaa74f95852ad1917ce1926ba64`:

- CuTe DSL is **Leave** on gfx1030. `dispatch.py` selects it only for
  compute capability (9, 0) and (10, 3), named there as H200 and B300.
  `cutedsl/hopper/bucket_scan.py` at `906be195` contains `tma` (17),
  `PipelineTma` (1), `cpasync` (6), and `warpgroup` (13).   `tile.py` and
  `ward.py` in the same directory also contain `warpgroup`. That is
  the CuTe spelling of the paper's TMA plus WGMMA schedule. The literal
  token `wgmma` was not in those files. The path does not exist on
  gfx1030.
- The Triton indexer is the port candidate. `dispatch.py` says every
  step has a Triton kernel, and `kernels: triton` runs them all. It is
  still not a drop-in on this pin. Details below.
- The paper's H200 bucket scan splits warps: one warp loads centroids with
  TMA, a warpgroup multiplies with WGMMA. The same paragraph says Triton
  emits those instructions on that GPU and picks the schedule itself.
  That is a statement about the H200 compiler output, not about Hopper
  intrinsics written in the Triton source.
- The Triton files under `src/lsa/core/kernels/triton` at that commit do
  not contain the strings `TMA`, `wgmma`, or `warp_specialize`. They are
  still not a drop-in:
  - `scan_select.py` and `shape.py` import `triton.language.extra.libdevice`.
  - `shape.py` names `bfloat16` operands. gfx1030 has no bf16 dot.
  - `top_k.py` loops with `tl.range(..., num_stages=STAGES)`. Whether
    ROCm Triton `3.7.1+git0263a6a6` accepts that keyword is unverified.
  - `expansion.py` launches `num_warps=16` on the decode path.
    `top_k.py` and `shape.py` launch `num_warps=8`. This fork sizes
    RDNA kernels at `num_warps <= 4` (wave32).
- This tree pins `triton==3.7.1+git0263a6a6.rocm7.14.0` and
  `torch==2.12.0+rocm7.14.0` (`requirements/build/rock.txt`). Triton 3.9+
  drops RDNA2, so the pin stays. Do not bump either package to chase
  the upstream kernels.

The path that can run here is a HIP selector in fp16, sketched in
`sketches/selector_fdot2.cu.txt`. It is not compiled.

## Centroid GEMV

Paper Table 2 fixes `d_idx = 64` for D-LSA. Scoring one query is a dot
against each unit centroid, 64 fp16 lanes, accumulated in fp32. That is
`v_dot2_f32_f16` (two fp16 lanes per instruction, fp32 accumulator).
The dot is the small kernel. Top-k or beam search, then the gather of
the selected pages into paged attention, is the real work, and RDNA_ATTN
has no sparse gather.

Centroid bytes per layer, two-byte elements, one centroid per 8 keys
(paper Appendix F: `2 * 64 * T / 8`):

| T | Centroids, one layer |
| --- | --- |
| 262144 | 4 MiB |
| 1048576 | 16 MiB |
| 10 × 2^20 | 160 MiB |

The 160 MiB cell is the paper's own product, checked by
`centroid_bytes` in `reference_lsa.py`. The paper's full flat index at
that length is 246 MiB per layer, centroids plus the id table. Scaling
that 246 MiB linearly is an assumption (one id per key is linear) and
is not a measurement on this card. At 262144 tokens it would be 6.15 MiB
per layer. Twelve Flash-Next softmax layers, replicated on every rank
the way the paper replicates the bucket cache, are about 74 MiB. That
is noise next to the KV cache. fp16 centroids are the same width as the
paper's bf16 centroids. The paper keeps centroids in bf16 because an
fp8 trial changed the selected set; we have no fp8 unit, and bf16 is
the wrong math type on gfx1030.

A bandwidth floor for one decode query reading 4 MiB of centroids once,
at 512 GB/s, is 4×2^20 / 512×10^9 ≈ 8.2 µs per layer. Twelve layers are
under 0.1 ms. Unverified: no kernel was timed. Full KV for Flash-Next
at 262144 tokens is 3 GiB per GPU (below), a floor of about 6.3 ms if
every byte is read once. The GEMV is not where a V620 spends the step.

## KV still grows with context

Paper Section 5: LSA reduces the cost of finding keys, not of storing
them. The KV cache stays linear in context. A 4×32 GB box does not hold
1M–10M of the caches this fork stores, except the cramped edge cases
below. The 10M and 72.4× figures are the paper's H200 numbers (abstract
and Section 4.3, Qwen-3.8-27B, one attention layer, prefill, versus full
attention). They are not a V620 forecast.

Bytes per token per GPU use the same head split as the Qwen modules:
`num_kv_heads = max(1, total_kv_heads // tp)`. fp16 K and V, head 256.

Flash-Next (`Qwen/Qwen3.8-Flash-Next`, fetched 2026-11-03): 12 softmax
layers, 2 KV heads, TP=4, so 1 head stored per rank (the two heads are
replicated, because TP does not divide 2 the other way: the code
requires `tp % total_kv_heads == 0` and then stores one head).

`12 * 2 * 1 * 256 * 2 = 12288` bytes/token.

| Context | Attention KV on one GPU |
| --- | --- |
| 131072 | 1.5 GiB |
| 262144 (checkpoint `max_position_embeddings`) | 3 GiB |
| 1048576 | 12 GiB |
| 10 × 2^20 | 120 GiB |

`tools/rdna/recipes/flashnext.env` sets `KV=7000000000`. If that entire
pool were this attention cache it would hold 7×10^9 / 12288 = 569661
tokens. GDN state also draws on the hybrid cache, so 569661 is an upper
bound, not a measured token cap. The split is unverified. The
checkpoint window is 262144, which is 3 GiB of attention KV and does
fit beside the MoE weights. 1M is 12 GiB of KV before weights, graphs,
and the GDN state. 10M does not fit.

Qwen3.6-27B and Qwen3.8-27B (same Hub shapes): 16 softmax layers, 4 KV
heads, TP=4, so 1 head per rank, partitioned.

`16 * 2 * 1 * 256 * 2 = 16384` bytes/token.

| Context | Attention KV on one GPU |
| --- | --- |
| 131072 | 2 GiB |
| 262144 | 4 GiB |
| 1048576 | 16 GiB |
| 10 × 2^20 | 160 GiB |

`27b-awq.env` sets `KV=8000000000`, an upper bound of 8×10^9 / 16384 =
488281 tokens if the pool were all attention KV. Again the hybrid cache
also holds GDN state. 1M tokens are 16 GiB of attention KV on a 32 GB
card, before weights. That is the edge of the card, not a serving
point, and it is unverified. 2M tokens are 32 GiB of KV alone.

DeepSeek-V4-Flash, from `dsv4-flash.env`: the recipe's pool is
`KV=6000000000` (5.59 GiB) and the comment says that holds 107892
tokens, with `MAXLEN=32768`. The checkpoint's `max_position_embeddings`
is 1048576. The cache that comment describes is the compressed fp8
layout, not a full fp16 KV. LSA does not shrink it. 1M–10M is outside
that pool.

## Where the index cost would matter

Paper Section 4.3: below 1M tokens the gain over a block indexer is
smaller, 2.1× at 512k on one GLM layer on an H200, because scanning
every block is still cheap. The steep part of that curve, and the
72.4× versus full attention, is at 10M.

On these cards the byte floor says the opposite of "the GEMV is the
bottleneck":

- Against full attention, attending to k=4096 plus a window of 512 is
  already less KV traffic than reading T once T is past a few thousand
  tokens. The missing piece is the gather, not the 64-wide dot.
- Against Flash-Next's QSA indexer (head 128, compress 4), index traffic
  is `(T/4) * 128 * 2 = 64T` bytes per layer. LSA centroids are `16T`
  bytes. That is a 4× smaller index at every length, not a 72×.
- The paper's large ratios need contexts this VRAM does not hold.

So the honest break-even is: sparsity would pay on bytes as soon as a
sparse gather exists, including inside 128k–256k. Replacing a trained
QSA or DeepSeek indexer with LSA does not pay inside the pool these
recipes actually run, and the paper's own 2.1× point (512k, H200, one
layer, already-sparse baseline) is at or past our KV limit. No tok/s
number is claimed here.

## Graphs

Paper Appendix F.1: prefill is served without CUDA graphs. A chunk is
milliseconds, and the selection reads list sizes back on the host.
Decode is captured, one graph per context range, up to `1.125 * 2^e`
for `e = 17…23` (the paper says from 144k to 9M), plus one graph for
the longest context. Clustering of a finished block of 512 keys runs
outside the graph.

HIP graphs on this fork are the same capture API (`FULL_AND_PIECEWISE`
in the recipes). A host readback cannot sit inside a captured prefill.
The decode plan, if a selector is ever real, is one capture per length
bucket and an eager prefill. The stub does not capture anything and
does not change the recipe's graph mode.
