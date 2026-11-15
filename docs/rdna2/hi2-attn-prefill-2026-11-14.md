# Prefill attention on RDNA_ATTN: measurement and an opt-in GEMM path (HI2 27B, TP=2, 2026-11-14)

Branch `feat/hi2-attn-prefill` (from `feat/hi2-27b-opt`). Code:
`vllm/v1/attention/ops/rdna_unfused_prefill.py`, routing in
`vllm/v1/attention/backends/rdna_attn.py`, tests
`tests/kernels/attention/test_fa_rdna2_unfused_prefill.py`, microbenchmark
`tools/rdna/rdna_hint_flex/bench_attn_prefill.py`. Labels: **[measured]** = produced by a
command listed here, **[derived]** = arithmetic on measured values, **[inference]** =
reasoning a run must confirm.

**Scope.** Measured on HI2 27B at TP=2 (12 query / 2 KV heads of head_dim 256 per rank),
box `par1-cs25`, HIP 2,3 (microbenchmark and unit tests on HIP 3). Flash-Next and 27B
AWQ use the same backend, and the TP=1 geometry (24q/4kv) is tested for correctness
only: **their speed with this path is unmeasured**. head_dim 128 is not enabled. Default
**off**, switched by an environment variable for the A/B period.
**Measurement windows.** The throughput numbers below are single requests from two
validation boots, one with the path on and one off, taken while the box load average was
25 to 35 and with the image-build client running for at least the first of them: **not
for comparison**. The quiet-window comparison is the scoreboard arms at the end.

## Why

A prefill chunk of the 27B at TP=2 spends 21% of its kernel time in
`fa_prefill_paged_varlen_gqa_kernel<256,2,8>` and 60% in rocBLAS GEMMs running at 24-26
TFLOP/s ([`hi2-27b-tp2-2026-11-14.md`](hi2-27b-tp2-2026-11-14.md), prefill chunk profile).
`csrc/rocm/fa_rdna2.cu` runs it at 3.4-3.6 TFLOP/s while a rocBLAS GEMM pair on the same
shapes reaches 11-15 TFLOP/s, and the attention time grows with the cached context:

| Chunk (q rows) | Cached context | GQA flash kernel ms/layer | TFLOP/s | GEMM path ms/layer | TFLOP/s | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 0 | 7.28 | 3.54 | 2.33 | 11.1 | 3.1x |
| 2048 | 2048 | 21.58 | 3.58 | 5.41 | 14.3 | 4.0x |
| 2048 | 6144 | 53.47 | 3.37 | 11.97 | 15.1 | 4.5x |
| 2048 | 14336 | 112.38 | 3.44 | 26.17 | 14.8 | 4.3x |
| 1024 | 2048 | 9.07 | 3.55 | 2.28 | 14.1 | 4.0x |
| 1024 | 512 | 3.78 | 3.41 | 1.18 | 10.9 | 3.2x |
| 256 | 2048 | 2.23 | 3.07 | 0.61 | 11.2 | 3.7x |
| 256 | 512 | 0.69 | 2.91 | 0.37 | 5.4 | 1.9x |

[measured, `HIP_VISIBLE_DEVICES=3 python tools/rdna/rdna_hint_flex/bench_attn_prefill.py
--check --q 2048 --ctx 0,2048,6144,14336` and `--q 1024,256 --ctx 2048,512`, median of 3,
logs `~/w4a8_runs/hi2-tp2/attn/gqa.log`; K/V in the paged layout with 1024-token pages;
the gather is outside the timed region, the engine timings below include it.] TFLOP/s
counts the useful causal FLOPs, `4 * H_q * D * (q * ctx + q * (q + 1) / 2)`. **Correction:**
the first version of this microbenchmark timed `fa_rdna2_prefill_paged_varlen` (the
non-GQA kernel) instead of `fa_rdna2_prefill_paged_varlen_gqa` that the backend calls, and
reported ratios of 10 to 16x; those numbers are withdrawn, the table above times the
kernel the engine runs. The engine kernel durations of the prefill profile (43.0 and
28.2 ms per layer for 2048 and 1024 rows) are consistent with it at about 4800 and 7300
tokens of cached context [derived].

## The path

`softmax(Q K^T * scale + causal mask) V` as two rocBLAS GEMMs and one softmax per block of
query rows, on K/V gathered from the paged cache (`index_select` over the sequence's
blocks, K packed by x and V unpacked as the backend's views describe them). The 6 query
heads of a GQA group are folded into the GEMM rows, the keys right of a block's last row
are never computed, and the diagonal block carries an additive `-inf` mask. Scores are
fp16 (the GEMM accumulates in fp32), the softmax accumulates in fp32 and is done in place
(`aten._softmax.out` with `out=` the input, bit-identical to the out-of-place result on
widths 64 to 16384), P is fp16, the PV GEMM accumulates in fp32. A `torch.mm` with an fp32
output exists on this stack but runs the QK GEMM 2.7x slower (2.17 vs 0.79 ms for
3072 x 256 x 16384 on HIP 3), so fp32 scores were not used. The technique is plain
softmax attention (the eager-attention recipe of Hugging Face transformers); no outside
code was taken.

Routing (`RdnaAttentionImpl._forward_prefill`): a row takes the GEMM path when
`VLLM_FA_RDNA2_UNFUSED_PREFILL=1`, head_dim 256, fp16 query and KV cache, causal, no sliding
window, its query length is at least `..._MIN_Q` (64) and the planner finds a block that
fits the scratch bound. Everything else keeps the flash kernel: decode and MTP-verify rows
(1 to 3 tokens, never routed here), short chunks, bf16, int4 KV, non-causal, sliding
window and sequences above the bound. A mixed prefill batch is split into runs, so the
flash kernel still serves the short rows around a long one. The per-row lengths come from
the host copies the runner already keeps (`query_start_loc_cpu`,
`seq_lens_cpu_upper_bound`, exact for prefill rows), copied only on steps whose longest
query reaches `MIN_Q`: decode steps and CUDA-graph captures never reach the new code. An
out-of-memory error inside the path logs once, switches it off for the process and reruns
the call on the flash kernel. Flag off, the flash kernel call sequence is unchanged.

The switch is an environment variable because the backend has no config route for its
other toggles either (`VLLM_FA_RDNA2_GQA_MODE`, `VLLM_FA_RDNA2_SPLIT_DECODE` are plain
`os.environ` reads); that is meant for the A/B period.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_FA_RDNA2_UNFUSED_PREFILL` | 0 | 1 enables the path (head_dim 256) |
| `VLLM_FA_RDNA2_UNFUSED_PREFILL_MIN_Q` | 64 | shortest query run that takes the path |
| `VLLM_FA_RDNA2_UNFUSED_PREFILL_SCRATCH_MB` | 768 | scratch bound per call: gathered K and V, one score tile, q/o staging |
| `VLLM_FA_RDNA2_UNFUSED_PREFILL_QBLOCK` | 512 | most query rows per GEMM block |

Scratch for a 2048-row chunk at the default bound (`scratch_bytes_required`, MiB, fp16;
the planner takes the largest block of at most 512 rows, a multiple of 32, that fits):

| Geometry | kv_len | Query block | K+V | Score tile | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP=2, 12q/2kv | 16384 | 512 | 32.0 | 96.0 | 131.5 |
| TP=2 | 32768 | 512 | 64.0 | 192.0 | 259.5 |
| TP=2 | 65536 | 512 | 128.0 | 384.0 | 515.5 |
| TP=2 | 131072 | 320 | 256.0 | 480.0 | 738.1 |
| TP=2 | 262144 | 64 | 512.0 | 192.0 | 704.4 |
| TP=1, 24q/4kv | 32768 | 512 | 128.0 | 192.0 | 323.5 |
| TP=1 | 131072 | 160 | 512.0 | 240.0 | 753.0 |
| TP=1 | 262144 | none fits: flash kernel (K+V alone is 1024) | 1024.0 | | |

The allocation goes through the torch caching allocator and stays in its pool; the
profile run does not execute the attention, so the memory profiler does not see it. The
27B recipe has 3.5 GiB free per rank at rest with MTP=2 (2.4 GiB at the boot peak), so the
default bound fits; lower the bound on a recipe with less headroom.

## Correctness [measured]

`pytest -s tests/kernels/attention/test_fa_rdna2_unfused_prefill.py` on the box: 33 tests.
First run on HIP 3: the 7 GPU tests and 24 of the 26 CPU tests passed; 2 planner tests
failed on a test bug (a chunk longer than the context), fixed, and the CPU-only rerun
(no GPU visible) gave 26 passed, 7 skipped, so the GPU tests skip cleanly off-GPU. CPU
tests (float32 against a float64 reference, max-abs error below 2e-5): TP=2 and TP=1
geometries and D=128,
causal offset into the cache (prefix hits), partial last pages, shuffled block tables,
several query blocks per sequence, the K/V gather against the engine's cache layouts,
the in-place softmax, the planner bound. GPU tests (skipped off gfx103x): the production
kernel against the new path against float64, and the backend routing on a mixed batch of
five rows (short rows byte-identical to the flash kernel, long rows within the error
below). Error against float64 on fp16 inputs:

| Case | max abs, flash / GEMM path | relative L2, flash / GEMM path |
| --- | ---: | ---: |
| 12q/2kv, n=300, ctx 0, 16-token pages | 9.6e-4 / 1.5e-3 | 2.0e-4 / 4.2e-4 |
| 12q/2kv, n=512, ctx 1536, 1024-token pages | 6.1e-5 / 3.3e-4 | 2.1e-4 / 5.5e-4 |
| 12q/2kv, varlen, n=200, ctx 1500 | 6.1e-5 / 2.8e-4 | 2.1e-4 / 5.4e-4 |
| 24q/4kv, n=257, ctx 700 | 1.2e-4 / 5.4e-4 | 2.1e-4 / 5.4e-4 |
| 12q/2kv, n=256, ctx 2048, inputs x4 (logits to about 30) | 1.9e-3 / 3.5e-2 | 1.9e-4 / 2.5e-3 |

On unit-variance inputs the new path is 2 to 3x the flash kernel's error (relative L2 about
5e-4, the size of one fp16 rounding of the output); with large logits the fp16 score
rounding shows (13x the kernel's error), so a model with very large attention logits
would want the flash kernel; the end-to-end check is the scoreboard's quality section.

## Engine validation [measured, not for comparison]

Two boots of `RECIPE=27b-hi2-tp2` (TP=2, HIP 2,3, f16, MTP 0, FULL_AND_PIECEWISE, 3000-word
prefix probe) through `tools/rdna/port_v031/serve_validate.sh`, one with
`VLLM_FA_RDNA2_UNFUSED_PREFILL=1` and one with `=0` (`~/w4a8_runs/hi2-tp2/unfused_ab.sh`,
results `serve-un1/`, `serve-un0/`), cells `8192:16:1:1 16384:16:1:1`, `--temperature 0`:

| | path on | path off |
| --- | ---: | ---: |
| READY | 181 s | 180 s |
| log line "unfused prefill attention is serving long chunks" | 1 | 0 |
| greedy probe / PPL probe | 3/3 / 5.73 (NLL 1.746) | 3/3 / 5.73 (NLL 1.746) |
| prefix probe (cold / warm TTFT, hit rate) | 4.52 s / 0.40 s, 96.2% | 4.73 s / 0.42 s, 96.2% |
| concurrency probe (48 checks) | 48 identical, 0 failing | 47 identical, 1 near-tie, 0 failing |
| KFD evictions | +0 / +0 | +0 / +0 |
| TTFT, 8192-token prompt (1 request) | 11.52 s (711 tok/s) | 12.99 s (631 tok/s) |
| TTFT, 16384-token prompt (1 request; the first 8192 tokens hit the cache of the previous cell) | 13.80 s (1187 tok/s) | 18.52 s (885 tok/s) |
| decode tok/s, 8k / 16k request | 33.7 / 30.6 | 34.5 / 31.6 |

The FULL_AND_PIECEWISE capture ran with the path on (decode graphs never reach it). The
TTFT difference matches the microbenchmark: summing the table's per-layer savings over the
chunks (contexts 0, 2048, 4096, 6144 for the 8k prompt, 8192 to 14336 for the cached 16k
one, linear interpolation between the measured contexts) gives 1.46 s and 4.45 s over 16
layers against 1.47 s and 4.72 s measured [derived]. For a 16k prompt from scratch the
same sums give about 5.9 s saved out of roughly 29 s [derived]. One request per cell and one
boot per arm: the boot-to-boot spread is not known, which is what the scoreboard's floor
arm measures.

## Not done

The quiet-window old/new comparison (scoreboard arms `base`, `attn`, floor `base2`: 1k/512
and 16k/1k at c=1 and c=8, 32k/2k), quality against the baseline with its noise floor,
long-context (64k and more) timing, TunableOp rows for the new GEMM shapes (they vary with
the context, so rows cannot cover them), TP=1 and TP=4 end to end, head_dim 128, and the
other recipes. The path costs a gather of K and V per chunk and fp16 scores (error above);
a fused kernel that keeps the 3.4-3.6 TFLOP/s kernel's memory behaviour but reaches GEMM
rates would remove both [inference].
