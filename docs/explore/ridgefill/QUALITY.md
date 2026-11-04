# RidgeFill quality plan: the HIP fill within +0.03 PPL

**Status: plan, partly measured.** Gate: long-window PPL within +0.03 of
exact (64 PG-19 windows of 16384 tokens, the last 2047 scored) and 16K solo
TTFT at least 3 % better than exact (5.741 s on a413af6b52, so 0.172 s). The
int8 HIP map kernel is the fast path; the fp16 kernel is the reference.

## What is already exact, and what is not

The map kernels match rocBLAS: in the compiled TP4 A/B on a413af6b52 the
fp16 kernel is -0.007 [-0.015, +0.000] PPL and the int8 kernel +0.001
[-0.007, +0.009] against the rocBLAS fill ([TESTPLAN.md](TESTPLAN.md),
item 4). Making the HIP fill exact is therefore a question of two other
things:

1. **The maps.** Every fill arm is +0.72 PPL against exact, and the legacy
   diagnostic (stock attention on the filled rows) still +0.66. Those maps
   were fitted on placeholder text: `capture.sh` ran
   `capture_activations.py --prompts 8 --tokens 4096` without
   `--prompts-file`, so `_read_prompts` returned its placeholder, one
   ~60-token sentence repeated to 4096 tokens in 8 near-identical prompts.
2. **The cache-only writers.** With the same maps, the cache-only fill is
   0.065 [0.039, 0.090] PPL worse than the legacy fill. That is twice the
   whole gate.

The fit error of the placeholder maps against their own training rows, by
token position (CPU,
`~/w4a8_runs/lanes/ridgefill/study/RidgeFill3/pos_err.py`, rel RMS):

| Layer | pos 0 | 1-3 | 4-15 | 16-63 | 64-511 | 512+ |
| --- | --- | --- | --- | --- | --- | --- |
| 24 | 0.003 | 0.094 | 0.095 | 0.088 | 0.109 | 0.101 |
| 31 | 0.007 | 0.196 | 0.235 | 0.232 | 0.396 | 0.430 |
| 47 | 0.009 | 0.160 | 0.184 | 0.205 | 0.385 | 0.373 |

Position 0 is the same token in all 8 prompts. Past position 64, where the
placeholder sentence starts to repeat, the deep layers' error doubles: their
inputs there follow the repetition through attention, which a per-token map
cannot see [INFERENCE on the mechanism]. These rows say nothing about
natural text, which is why the first step is a capture of it.

## 16K TTFT model

From the a413af6b52 A/B: the skipped late-layer work is 3.96 (int8 arm),
3.95 (rocBLAS) and 3.84 (fp16) us per filled row per predicted layer, each
being the arm's TTFT gain over exact plus its measured map cost. The map
costs 0.983 us per row-layer on int8 (48.3 ms per 2048-row chunk), 2.03 on
fp16 and 2.47 on rocBLAS. With k predicted layers and F filled rows per
request, the int8 fill saves about k F 2.93 us, so the gate needs
k F >= 58.7 k row-layers (fp16 91 k, rocBLAS 119 k). Projected, assuming
every late layer costs the same [INFERENCE]:

| Config | k | F (16K) | int8 saving | vs gate |
| --- | --- | --- | --- | --- |
| split 24, tail 4096 (measured 0.878 s) | 24 | 12288 | 0.865 s, 15.1 % | 5.0x |
| split 24, tail 8192 | 24 | 8192 | 0.577 s, 10.0 % | 3.4x |
| split 24, tail 12288 | 24 | 4096 | 0.288 s, 5.0 % | 1.7x |
| split 36, tail 4096 | 12 | 12288 | 0.433 s, 7.5 % | 2.5x |
| split 36, tail 8192 | 12 | 8192 | 0.288 s, 5.0 % | 1.7x |
| split 36, tail 12288 | 12 | 4096 | 0.144 s, 2.5 % | fails |

The cache-only cut is aligned down to 4096 rows (block 1024 x indexer ratio
4), so a 16K request fills 12288, 8192 or 4096 rows; the legacy cut is
unaligned.

## Options, ranked

Ranked by expected PPL gain per TTFT given up. The PPL column is what the
plan measures. No option's PPL is measured yet.

1. **Real-text capture** (no TTFT cost). 64 windows of 16384 tokens from
   PG-19 books disjoint from the eval books, captured as sufficient
   statistics (n, sum x, X^T X, and X^T Y per layer, fp64); 8 held-out
   windows kept raw for the held-out error, a lambda sweep and a learning
   curve (8, 16, 32, 64 windows). The learning curve answers "more data or
   more capacity": if the held-out error still falls at 64 windows, capture
   more; if it is flat, the linear map is the limit.
2. **Fix the cache-only writers** (no TTFT cost). The dump below pins the
   writer; the fix uses the stock writer's kernel or matches its numerics.
3. **The paper's released maps** (no TTFT cost). The Flash-Next i8 maps
   from `TheCacheIsALie`, fitted on the MXFP4/FP8 checkpoint, converted to
   our bundle. One arm prices how much a large capture buys, before ours
   lands.
4. **Longer exact tail**: tail 8192 keeps 10.0 %, tail 12288 keeps 5.0 % of
   the 15.1 %. The filled rows are then at least 8K or 12K tokens from the
   scored ones.
5. **Later split**: split 36 predicts only layers 36-47 (7.5 % at tail
   4096). The maps bridge 0-11 layers instead of 0-23.
6. **Richer maps**: per-layer lambda falls out of the sweep. Low-rank maps
   are a measured dead end (rel RMS 0.52-0.59 at rank 256-1024 on layer
   24). A nonlinear map would need new kernel work; none is planned unless
   the learning curve says capacity, not data, is the limit.

Depth-limited fill as "predict the shallow late layers, run the deep ones
exact" cannot be built: an exact deep layer needs the filled rows' exact
residual stream, which needs every layer before it, MoE included. Its
buildable forms are the later split (predict only the deepest layers) and
the exact tail.

An exact head (never fill the first rows, the attention sinks) costs
nothing in TTFT. It waits on the held-out position profile: if position 0's
error on real text is not an outlier it buys nothing.

## Workstream A: the cache-only defect

Planned harness: `cache_dump.py` with the `dump_probe.py` worker extension.
In one TP4 process, for the first filled forwards and every late QSA layer,
it snapshots the filled rows' main K/V, indexer raw-key, compressed-key and
position cache entries after the cache-only writes (A), after the stock
attention call on the same predicted inputs (B), and after a second stock
call (B2, the kernel-nondeterminism floor). A writer is wrong where
|A - B| is well above |B - B2|.

## Workstream C: kernel

The plan keeps linear maps, so the int8 kernel serves every arm unchanged.
The op takes the map count as `layers` and checks only the per-layer
`dout` and K (`csrc/rocm/ridgefill_map_rdna2.cu`, `check_shapes` and the
int8 K check), so split 36's 12 maps (N = 30720) need no change; not run
at 12 maps yet. Kernel work only returns if the learning curve says
capacity, not data, is the limit.

## Corpus

Eval: `corpus/pg19_w64_l16384.json`, 64 windows from 64 books of
emozilla/pg19-test, book IDs 0-3, 5-14, 16-27, 29-41, 43-56, 59-66 and
68-70. Fit and held-out windows come only from books 4, 15, 28, 42, 57, 58,
67 and 71-99.

## GPU plan

`slot_q.sh` on HIP 2-5, unattended once `GO_Q` exists, behind the lease
queue. In order:

1. Capture: 64 fit windows (statistics) and 8 held-out windows (raw), split
   24 and split 36; then the CPU fit runs in the background.
2. The cache-only defect dump (A, B, B2).
3. exact (TTFT and PPL, the floor); the paper's maps: legacy (PPL) and i8
   (TTFT and PPL).
4. After the fit: the PG-19 maps, legacy and i8; the tail sweep (8192,
   12288) on whichever map set has the lower legacy PPL; split 36, legacy;
   exact2 (the TTFT noise floor).

Legacy arms carry the maps' quality without the open cache-only defect.
i8 arms carry the TTFT and, until the writers are fixed, the defect.
