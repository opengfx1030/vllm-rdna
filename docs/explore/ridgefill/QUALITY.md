# RidgeFill quality plan: the HIP fill within +0.03 PPL

**Status: measured; no arm meets the gate.** Gate: long-window PPL
within +0.03 of exact (64 PG-19 windows of 16384 tokens, the last 2047
scored) and 16K solo TTFT at least 3 % better than exact. The int8 HIP map
kernel is the fast path; the fp16 kernel is the reference. On c772b7b76a
the int8 cache-only fill matches the stock attention path on both map
sets, and the PG-19 fit cuts the paper maps' PPL loss by 23-30 % at every
tail. The best arm still misses both halves of the gate, narrowly: tail
12288 gives +0.038 PPL at -2.9 % 16K TTFT against exact on the same GPUs.
The fit's learning curve is flat: the linear map, not the data, is the
limit.

## What is already exact, and what was not

The map kernels match rocBLAS: in the compiled TP4 A/B on a413af6b52 the
fp16 kernel is -0.007 [-0.015, +0.000] PPL and the int8 kernel +0.001
[-0.007, +0.009] against the rocBLAS fill ([TESTPLAN.md](TESTPLAN.md),
item 4). The +0.72 PPL of that A/B comes from three defects outside the
kernels, all found on 2026-11-04:

1. **The maps were fitted on placeholder text.** `capture.sh` ran
   `capture_activations.py --prompts 8 --tokens 4096` without
   `--prompts-file`, so `_read_prompts` returned its placeholder, one
   ~60-token sentence repeated to 4096 tokens in 8 near-identical prompts.
   A real capture now requires a corpus.
2. **The split state missed layer 23's MoE output** (fixed in ea1bb06a47).
   Each layer's MoE output stays pending in `(block_output, injection)`
   until the next layer's combine adds it, and both the runtime and
   `capture_probe.py` took `x_S` before layer 24's combine. On all 32768
   captured rows, layer 24's HC mix of the captured `x_S` is 0.171 rel RMS
   (mean cosine 0.986, minimum 0.841) off the captured block input; with
   the combine materialized it is 0.0062 (cosine 0.99998). The missing
   term is 9.0 % of the state norm. The paper's fit requires that
   identity (`provenance/fits/flashnext/fit-big-s24.log` in
   ridgefill-data v1.8: "HC identity at layer 24 reproduces
   block_input_24: cos 1.00000").
3. **The cache-only path ran on the layers before the split** (fixed in
   ea1bb06a47). The decoder's `cache_only_here` had no split guard, so the
   six full-attention layers 3-23 also took the cache-only op, which
   zeroes the filled rows' attention output: their residual, which there
   becomes `x_S`, lost six attention terms, and their K/V in layers 7-23,
   which the exact rows attend to, came from those corrupted states. The
   cache-only fill lost 0.065 [0.039, 0.090] PPL against the legacy fill
   on a413af6b52; on c772b7b76a the pair agrees to +0.000 [-0.006, +0.007]
   (Measured, below). [INFERENCE: this fix closed it. Defect 2 changes both
   paths alike and the solo A/B built no mixed forward; no arm isolated
   it.] It also inflated the measured cache-only TTFT gain (below).

A fourth defect was latent in that A/B, which never built a mixed forward:

1. **A mixed forward's exact rows got the wrong indexer metadata** (fixed
   in d2c0272a0e). A forward that holds filled and exact rows (a cut inside
   a chunk, or another request's rows, decode included, batched with a
   filled chunk) ran its exact rows compacted through the stock QSA
   forward. That forward narrowed the main attention's metadata to the
   rows but not the indexer's, which reads the full-batch metadata, so the
   exact rows' indexer keys were stored, compressed and selected against
   other rows' slots. The solo A/B's aligned cut falls on a chunk boundary,
   so its forwards were fully filled or fully exact. Mixed forwards now run
   the stock call on every row; only fully filled forwards take the
   cache-only writes. The op test fails on a413af6b52 and passes on
   d2c0272a0e, CPU-only.

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

## Measured on c772b7b76a, 2026-11-05

`slot_q.sh`, compiled TP4 on HIP 2-5, recipe `flashnext-mtp2` with MTP and
prefix caching off, one boot per arm, 08:32-10:12 box time. TTFT: median of
16 solo requests per cell after a 16K warm request, every cell window
eviction-clean. PPL: the 64 eval windows, deltas paired per window with a
95 % interval. `q_summary.py` over `~/w4a8_runs/lanes/ridgefill/q_c772b7b76a/ab`.

| Arm | Exact tail | 16K TTFT | vs exact | PPL | vs exact [95 % CI] |
| --- | --- | --- | --- | --- | --- |
| exact | all | 5.766 s | | 7.5515 | |
| exact2 | all | 5.774 s | +0.1 % | | |
| paper maps, i8 | 4096 | 5.178 s | -10.2 % | 7.8174 | +0.266 [+0.240, +0.292] |
| paper maps, legacy | 4096 | | | 7.8173 | +0.266 [+0.241, +0.291] |
| paper maps, legacy | 8192 | | | 7.6662 | +0.115 [+0.095, +0.135] |
| paper maps, legacy | 12288 | | | 7.5986 | +0.047 [+0.034, +0.061] |

4K TTFT: exact 1.416 s, paper i8 1.423 s (a 4096-token request fills no
row at tail 4096). Noise floors: exact2 - exact +0.008 s (+0.1 %) in this
slot, 0.063 s (1.1 %) on a413af6b52; PPL across boots -0.005 [-0.012,
+0.002] (a413af6b52).

- **The fill path is exact.** The int8 cache-only fill and the stock
  attention path give the same PPL on the same maps: +0.000 [-0.006,
  +0.007], 31 of 64 windows worse. On a413af6b52 the same pair was +0.065
  [+0.039, +0.092]. The cache dump agrees (Workstream A).
- **The TTFT gate passes** at tail 4096: -0.588 s, -10.2 %, against the
  model's projected 5.14 s (-10.4 %).
- **The paper's maps fail the PPL gate at every tail**, and no legacy arm
  came within +0.04 of exact, so no longer-tail i8 arm ran. Per filled row
  the loss falls the farther the filled rows end from the scored tokens:
  21.6, 14.0 and 11.5 x 1e-6 PPL per row at 12288, 8192 and 4096 filled
  rows. Tail 12288 also leaves only 2.9 % of TTFT (measured with the PG-19
  maps, below).

## Measured with the PG-19 maps, 2026-11-05

`slot_q2.sh` on slot A (HIP 1-4: the V620s on PCI buses ce, d3, d8 and db
after that day's topology change), 22:08-01:18 box time. The capture ran on
ed2786988b in 14 min, the CPU fit in 40 min (16 threads, nice 19), and the
arms on c772b7b76a as in the run above, every TTFT cell in a quiet window
and eviction-clean. TTFT compares with exact3, served on the same GPUs; this
slot's exact is 2.5 % faster than the HIP 2-5 one above. PPL pairs with the
exact arm above: same tree, other GPUs. The cross-boot floor is -0.005
[-0.012, +0.002]; a cross-GPU floor was not measured.

| Arm | Split | Exact tail | 16K TTFT | vs exact3 | PPL | vs exact [95 % CI] |
| --- | --- | --- | --- | --- | --- | --- |
| exact3 | | all | 5.624 s | | | |
| PG-19 maps, i8 | 24 | 4096 | 4.992 s | -11.2 % | 7.7449 | +0.193 [+0.171, +0.215] |
| PG-19 maps, legacy | 24 | 4096 | | | 7.7382 | +0.187 [+0.165, +0.208] |
| PG-19 maps, legacy | 24 | 8192 | | | 7.6371 | +0.086 [+0.069, +0.103] |
| PG-19 maps, legacy | 24 | 12288 | | | 7.5877 | +0.036 [+0.024, +0.049] |
| PG-19 maps, i8 | 24 | 12288 | 5.462 s | -2.9 % | 7.5895 | +0.038 [+0.026, +0.051] |
| PG-19 maps, legacy | 36 | 4096 | | | 7.6564 | +0.105 [+0.092, +0.118] |

4K TTFT, where neither tail fills a row: exact3 1.372 s, i8 at tails 4096
and 12288 1.369 s and 1.389 s, a 1.5 % spread. Map sha256: 80495577 (split
24), dcb19f20 (split 36).

- **No arm passes.** Tail 12288 misses both halves by less than their
  noise: +0.038 PPL against +0.03 (the interval spans the gate) and -2.9 %
  TTFT against -3 % (0.1 point, under the 1.5 % 4K spread). No tail between
  the measured ones meets both: PPL reaches +0.03 near tail 12800, where the
  TTFT gain is about 2.5 %, and the gain reaches 3 % near tail 12200, where
  PPL is about +0.04 [INFERENCE, linear between the measured tails].
- **The fill path is exact on these maps too**: i8 against legacy +0.007
  [-0.001, +0.014] at tail 4096 and +0.002 [-0.005, +0.008] at 12288.
- **The PG-19 fit beats the paper's maps at every tail**: +0.187, +0.086
  and +0.036 against +0.266, +0.115 and +0.047. At tail 4096 it lands on the
  paper's own gap at our scale (+0.20, below).
- **The TTFT model holds**: -11.2 % at tail 4096 (projected -10.4 %),
  -2.9 % at tail 12288 (projected -3.5 %).
- **Split 36 is dominated**: +0.105 at tail 4096 is worse than split 24 at
  tail 8192 (+0.086), whose projected TTFT gain is larger (6.9 % against
  5.2 % [INFERENCE, neither measured]). No split-36 i8 arm ran.

### The fit

`fit_stats.py` (ed2786988b) on the capture: one fp64 eigendecomposition per
split and snapshot, lambda 0.003-0.3 per layer. The error is
`||y_hat - y||_F / ||y||_F` as served (fp16 maps, fp32 accumulate) on 32768
rows of the 8 held-out windows (every 4th position).

| Fit windows | Split 24 train | Split 24 held-out | Split 36 train | Split 36 held-out |
| --- | --- | --- | --- | --- |
| 8 | 0.592 | 0.687 | 0.480 | 0.586 |
| 16 | 0.612 | 0.673 | 0.498 | 0.572 |
| 32 | 0.619 | 0.663 | 0.509 | 0.561 |
| 64 | 0.625 | 0.657 | 0.513 | 0.555 |

- **The learning curve is flat.** Each doubling of the fit windows buys
  less (-0.014, -0.010, -0.006 at split 24) and the train error rises to
  meet the held-out one. By option 1's test, capacity, not data, is the
  limit; another doubling would buy about 0.005 [INFERENCE].
- **The paper's maps are nearly as good on the same rows**: 0.687 held-out
  against 0.657 (`study/RidgeFill3/score_maps.py`, the fit's own scorer).
  That 0.03 is worth 0.08 PPL at tail 4096. The placeholder maps score
  1.229, worse than zeros.
- **The error grows with depth past the split**: 0.30 at layer 24 and
  0.70-0.77 at layers 28-36 (split 24); 0.23 at layer 36 and 0.72 at 47
  (split 36). Per-layer lambda: 0.01-0.1 at split 24, 0.003-0.03 at 36.
- **Position 0 is not an outlier**: below its layer's mean in 22 of 24
  layers at split 24. Positions 1-15 are worse, up to 0.12 above the mean
  at layers 40-47, but they are 15 of each window's 16384 positions.

## The paper's own fit, for scale

From the paper (OpenReview `ymLpDkJIGs`, appendix G, section 5.3) and
ridgefill-data v1.8 (`provenance/fits/flashnext/report-big-s24.json`):
the Flash-Next maps were fitted on 425,789 rows (every 8th position) of
151 documents up to about 20K tokens plus chat-template copies of 102 of
them, on the MXFP4/FP8 checkpoint, with lambda 0.03 in the same
convention as `math.py` (picked from 0.001-0.3 by held-out cosine). Their
held-out block-input cosine is 0.96 at layer 24, 0.906 at 25, and
0.68-0.81 from layer 27 on (mean 0.752). With those maps, split 24 and tail
2048, the paper reports a novel-token perplexity gap of +2.65 % [-0.02,
4.93] on 90 documents up to 32K tokens (last 512 scored), and 1.54x TTFT at
16K. On our scale that gap is about +0.20 PPL, about 7x the gate: even
paper-quality maps are not expected to pass at the full fill [INFERENCE].
The longer exact tail and the later split below were where the gate could
be met, if anywhere; measured, neither meets it.

## 16K TTFT model

From the a413af6b52 A/B, corrected for defect 3. The legacy fill (5.862 s)
is exact (5.741 s) minus the late MoE skip plus the rocBLAS map (0.727 s),
so the MoE skip is 0.606 s for 24 layers x 12288 filled rows (2.06 us per
row-layer). The rocBLAS cache-only fill was 0.560 s faster than legacy:
the QSA attention it skipped for the filled rows, on 12 layers instead of
the 6 late ones. Keeping the late half (equal per-layer cost
[INFERENCE]), the fixed fill skips 0.886 s, 3.00 us per row-layer, and the
int8 map costs 0.983 us per row-layer (48.3 ms per 2048-row chunk), so the
fixed int8 fill nets 2.02 us per row-layer: 16K TTFT about 5.14 s, -10.4 %
[INFERENCE]. With k predicted layers and F filled rows per request the gate
needs k F >= 85 k row-layers. Projected, assuming every late layer costs
the same [INFERENCE]:

| Config | k | F (16K) | int8 saving | vs gate |
| --- | --- | --- | --- | --- |
| split 24, tail 4096 | 24 | 12288 | 0.596 s, 10.4 % | 3.5x |
| split 24, tail 8192 | 24 | 8192 | 0.398 s, 6.9 % | 2.3x |
| split 24, tail 12288 | 24 | 4096 | 0.199 s, 3.5 % | 1.15x |
| split 36, tail 4096 | 12 | 12288 | 0.298 s, 5.2 % | 1.7x |
| split 36, tail 8192 | 12 | 8192 | 0.199 s, 3.5 % | 1.15x |
| split 36, tail 12288 | 12 | 4096 | 0.099 s, 1.7 % | fails |

Measured on c772b7b76a (paper maps, split 24, tail 4096): 5.178 s against
exact 5.766 s, -10.2 %.

The cache-only cut is aligned down to 4096 rows (block 1024 x indexer ratio
4), so a 16K request fills 12288, 8192 or 4096 rows; the legacy cut is
unaligned. The fp16 kernel (2.03 us per row-layer) nets 0.97 us, so it
clears the gate only at split 24 with tail 4096 (1.7x) or tail 8192 (1.1x)
[INFERENCE].

## Options, ranked

Ranked by expected PPL gain per TTFT given up. The PPL column is what the
plan measures. Defects 2-4 are fixed in every arm below. Options 1-4 are
measured, all short of the gate; option 5 is what is left.

1. **Real-text capture** (no TTFT cost). 64 windows of 16384 tokens from
   PG-19 books disjoint from the eval books, captured as sufficient
   statistics (n, sum x, X^T X, and X^T Y per layer, fp64) at splits 24
   and 36; 8 held-out windows kept raw for the held-out error, a lambda
   sweep and a learning curve (8, 16, 32, 64 windows). The learning curve
   answers "more data or more capacity": if the held-out error still falls
   limit. The capture takes the materialized split state (defect 2).
   Measured: the curve is flat, the linear map is the limit (The fit).
2. **The paper's released maps** (no TTFT cost). The Flash-Next i8 maps
   from `TheCacheIsALie`, fitted on the MXFP4/FP8 checkpoint, dequantized
   exactly (int8 codes times one bf16 scale per 128 columns) into our
   bundle; the stream order is the identity (every other permutation of
   the four streams fits worse). One arm prices what a large capture buys
   before ours lands. Their fit on our placeholder capture (0.34 rel RMS
   at layer 24, 0.64-0.79 from layer 27) is out of distribution twice
   over and says nothing yet.
3. **Longer exact tail**: tail 8192 keeps 6.9 %, tail 12288 keeps 3.5 % of
   the TTFT (projected). The filled rows are then at least 8K or 12K tokens
   from the scored ones; the paper found exact rows at the end worth more
   than the same number spread out (appendix I.1). Measured: tail 12288 is
   the closest arm, +0.038 PPL at -2.9 % TTFT.
4. **Later split**: split 36 predicts only layers 36-47 (5.2 % at tail
   4096, 3.5 % at tail 8192). The maps bridge 0-11 layers instead of 0-23.
   Measured: +0.105 at tail 4096, dominated by split 24 at tail 8192.
5. **Richer maps**: per-layer lambda falls out of the sweep. Low-rank maps
   are a measured dead end (rel RMS 0.52-0.59 at rank 256-1024 on layer
   24). A nonlinear map would need new kernel work. The learning curve now
   says capacity is the limit, so it is the one lever left; nothing is
   planned without a decision.

Depth-limited fill as "predict the shallow late layers, run the deep ones
exact" cannot be built: an exact deep layer needs the filled rows' exact
residual stream, which needs every layer before it, MoE included. Its
buildable forms are the later split (predict only the deepest layers) and
the exact tail.

An exact head (never fill the first rows, the attention sinks) costs
nothing in TTFT and, by the held-out position profile, buys nothing: position
0's error is not an outlier on real text (The fit). The paper excludes row
0 from its fit weighting as the sink (`code/kva/wfit.py`) but keeps it
filled.

## Workstream A: the cache-only defect

Measured closed on c772b7b76a: the paper maps' legacy and i8 cache-only
arms agree to +0.000 [-0.006, +0.007] (a413af6b52: +0.065 [+0.039,
+0.092]), inside the cross-boot floor (-0.005 [-0.012, +0.002]).

The dump checks the remaining suspects, the late-layer cache writers:
`cache_dump.py` with the `dump_probe.py` worker extension (afcf4b2fe0). In
one TP4 process, for the first filled forwards and every late QSA layer,
it snapshots the filled rows' main K/V, indexer raw-key, compressed-key
and position cache entries after the cache-only writes (A), after the
stock attention call on the same predicted inputs (B), and after a second
stock call (B2, the kernel-nondeterminism floor). A writer is wrong where
|A - B| is well above |B - B2|. Run on c772b7b76a (08:39-08:48 box time,
eval window 0, split 24, tail 2048, the placeholder-text maps: the writers
do not depend on the maps' quality): 12 records per rank over 6 filled
forwards, layers 27-47, 4 ranks; `cache_dump.py compare` flags no cache
row at 10x the floor (`q_c772b7b76a/dump/dump/compare.json`).

## Workstream C: kernel

The plan keeps linear maps, so the int8 kernel serves every arm unchanged.
The op takes the map count as `layers` and checks only the per-layer
`dout` and K (`csrc/rocm/ridgefill_map_rdna2.cu`, `check_shapes` and the
int8 K check), so split 36's 12 maps (N = 30720) need no change; not run
at 12 maps yet. The runtime indexes the stacked output by `layer - split`,
so a bundle must match the configured split: `check_map_layers` refuses a
manifest split that differs or a layer set other than split-47
(176ae4416c). Kernel work only returns if the learning curve says
capacity, not data, is the limit.

## Corpus

Every window is 16384 tokens, cut at 2048 + k x 16384 tokens into its book
with the eval's tokenizer. `make_fit_corpus.py` built the fit and held-out
files (`~/w4a8_runs/lanes/ridgefill/study/CaptureV2/`, log
`make_fit_corpus.log`).

- **Eval:** `corpus/pg19_w64_l16384.json`, one window from each of 64
  emozilla/pg19-test books (`test-00000-of-00001-29a571947c0b5ccc.parquet`,
  sha256 9aae5ddf035760257458cff08d2575d78a15f84eff867af7a87eff0681b01bfc):
  book IDs 0-3, 5-14, 16-27, 29-41, 43-56, 59-66 and 68-70. The builder
  re-tokenized these books and reproduced all 64 window shas, so both files
  use the same conventions.
- **Fit:** `pg19_fit_w64_l16384.json` (sha256 dbca1e98ef21…), 64 windows
  from 26 other books of the same parquet, at most 3 per book: IDs 71-77,
  79-96 and 99 (General Nelson's Scout, by Byron A. Dunn, to K, by Mary
  Roberts Rinehart). The remaining spare books (4, 15, 28, 42, 57, 58, 67,
  78, 97, 98) are shorter than 18432 tokens and give no window.
- **Held-out:** `pg19_held_w8_l16384.json` (sha256 8c89e496a893…), one
  window from each of PG-19 validation books 1-8
  (`validation-00000-of-00001-0f92e2337f79aeac.parquet`, sha256
  81680529564d4ead1c0e3859509a62d86c7126c32afc95dce6bd98e729e491ef), from
  The Atlantic Monthly Vol. 7 No. 42 to Blackwood's Edinburgh Magazine Vol.
  56 No. 348. The spare test books hold only 65 windows at 3 per book, short
  of 72.

Disjointness proof: the builder asserts that eval, fit and held-out share
no book (parquet sha256 and index), no title and no window token sha. Its
log reads "common books [] titles [] window shas []" for each of the three
pairs, then "disjointness: PASS". Each window in the files carries its
book index, title and sha.

## GPU plan

`slot_q.sh` on HIP 2-5, unattended once `GO_Q` exists, behind the lease
queue. Every arm runs one exact-SHA tree. In order:

0. GPU tests on HIP 3: the fill tests import the model, which queries the
   device at import, so they cannot run CPU-only. A failure stops the slot
   before any arm.
1. Capture: 64 fit windows (statistics) and 8 held-out windows (raw), split
   24 and split 36; then the CPU fit runs in the background.
2. The cache-only writer dump (A, B, B2).
3. exact (TTFT and PPL, the floor); the paper's maps: legacy (PPL) and i8
   (TTFT and PPL).
4. After the fit: the PG-19 maps, legacy and i8; the tail sweep (8192,
   12288) on whichever map set has the lower legacy PPL; split 36, legacy.
   When the 4096 tail fails, the most aggressive longer tail whose legacy
   arm is within +0.04 of exact (the gate plus the cross-boot noise) gets
   its i8 arm, for its TTFT; so does split 36 when its legacy arm is.
   Last, exact2 (the TTFT noise floor) and `q_summary.py`: one table of
   TTFT, PPL and paired deltas per arm.

Legacy arms give each map set's quality on the stock attention path. i8
arms give the TTFT, and their PPL against the legacy arm of the same maps
measures the fixed cache-only path. The slot needs about 2.6 h of GPU
time, up to 3 h with the gate-driven i8 arms [INFERENCE: capture 25-35
min from CaptureV2's estimate, the arms from the a413af6b52 A/B's
per-boot times].

**The 2026-11-05 run.** `slot_q.sh` took HIP 2-5 at 08:32 box time on
c772b7b76a and its GPU tests passed (74). The capture died at its first
fold, 6.5 min in: the probe's hooks create the statistics sums during the
worker's forward, which runs in inference mode, and the fold RPC updated
them in place outside it, which torch refuses ("Inplace update to inference
tensor outside InferenceMode"). The CPU tests ran their forwards under
`no_grad` and missed it. ed2786988b folds and snapshots inside inference
mode; its test runs the forwards in inference mode, as the worker does, and
fails on c772b7b76a with the GPU run's error. Without a fit, the slot ran
steps 2 and 3, and step 4's tail sweep on the paper's maps.

**The q2 run.** The box's 2026-11-05 power change renumbered the GPUs, so
`slot_q2.sh` resolves its slot's PCI buses to HIP indices and refuses any
other GPU. It took slot A at 22:08, after the EvictFix proof pair. The
capture on ed2786988b passed (8.2 s per window), the fit followed, and the
arms ran in order: the PG-19 maps, legacy and i8 at tail 4096; the legacy
tails 8192 and 12288; the i8 arm of tail 12288, whose legacy arm came
within +0.04; split 36, legacy (+0.105, so no i8 arm); exact3.
