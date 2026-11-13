# LeapQuant INT8 GDN recurrent state (opt-in, 2026-11-13)

Port of LeapQuant ([arXiv 2609.38166](https://arxiv.org/abs/2609.38166),
requested in [vllm#64725](https://github.com/vllm-project/vllm/issues/64725))
to the RDNA2 Gated DeltaNet path. Opt-in with `VLLM_RDNA_GDN_LEAPQUANT=1`
(gfx10x only; the flag raises elsewhere). Default off: the change is lossy.

## What it stores

The FP32 state of one value head (`[V, K] = [128, 128]`, 64 KiB) becomes one
28,768-byte blob (2.28x smaller):

| section | dtype | shape | bytes | offset |
|---|---|---|---:|---:|
| `z` INT8 residual | int8 | `[V, K]` | 16384 | 0 |
| `bscale` per-row scale | fp32 | `[V]` | 512 | 16384 |
| `csmooth` per-column smoothing | fp32 | `[K]` | 512 | 16896 |
| `comp_k` compensator keys | fp16 | `[r, K]` | 1024 | 17408 |
| `comp_u` compensator values | fp16 | `[r, V]` | 1024 | 18432 |
| `rec_k` token records | fp16 | `[cap, K]` | 4608 | 19456 |
| `rec_u` token records | fp16 | `[cap, V]` | 4608 | 24064 |
| `rec_g` decays | fp32 | `[cap]` | 72 | 28672 |
| header `[cursor, pending, 0, 0]` | int32 | `[4]` | 16 | 28752 |

`r = 4` compensator tokens, window 16, `cap = window + num_spec = 18` for MTP2.
The state is `replay(cursor records, Z*B*C + comp_u^T comp_k)`: the boundary
state (INT8 + rank-4 FP16 compensator of its top subspace, found by 2 power
iterations) plus the exact FP16 records `S_j = exp(g_j) S_{j-1} + u_j k_j^T`
since the boundary. Every 16 committed tokens the records are folded into a
new boundary state, so quantization error enters once per window instead of
once per token. The FP32 oracle is
`vllm/model_executor/layers/mamba/ops/leapquant_gdn.py`; the kernels are
`csrc/rocm/gdn_leapquant_rdna2.cu`.

## How it fits the serving path (MTP2, cudagraphs, prefix caching)

* **Slots are record counts.** The FP32 layout keeps one state per speculative
  token (`spec_state_indices[n, t]`). The blob lives at the home block
  `spec_state_indices[n, 0]`; "slot `t`" is "commit `t + 1` pending records".
  A decode/verify step appends its records and sets `pending`;
  `gdn_lq_normalize_rdna2` (first op of the next step) commits
  `num_accepted` of them (rejected drafts are dropped by never committing) and
  folds once `cursor >= window`. MTP verify needs no per-token state copies.
* **Readers of slot 0** (chunked prefill, non-spec decodes) commit
  `min(1, pending)`, the same state the FP32 path reads.
* **Prefill** materializes the blob to an FP32 `[N, HV, V, K]` initial state,
  runs the unchanged HIP prefill chain (`VLLM_GDN_HIP_PREFILL=1`) and stores
  the final state back (`gdn_lq_store_rdna2`). Inside a chunk nothing
  changes; LeapQuant error enters at chunk boundaries (MAXBAT 2048).
* **Prefix caching (`mamba-align`, V2 runner):** the align precopy/postprocess
  HIP kernels (`v1_mamba_rdna2.cu`, `copy_mamba_state_block`) copy the whole
  blob from the home block and write `cursor + min(bias + 1, pending)` into
  the copy's header, which is exactly the FP32 "copy slot `bias`" rule. The V1
  runner's memcpy specs and the Triton precopy fallback raise instead of
  silently copying wrong bytes.
* **MonoKernel (`EXTRA_ARGS=--monokernel`):** pure decode and pure verify
  batches run `gdn_lq_normalize_rdna2` and then `mono_gdn_lq_rdna2`, the
  `mono_gdn_rdna2` fused conv + delta rule + gated norm launch with the
  LeapQuant load/append in place of the FP32 state load/store (each 32-row
  tile rebuilds its rows from the blob; records are written per tile). Batches
  with prefill take the layer's path as with FP32 state.
* **HIP only.** Every LeapQuant read and write is one of the kernels above;
  the FP32-state Triton decode warmup is skipped under the flag (it never
  runs). Paths the flag does not change (conv update, gated norm outside the
  MonoKernel) are the recipe's, identical with the flag off.
* **Cudagraphs:** every op is a fixed-shape launch on the current stream that
  reads its metadata from device tensors, so FULL_AND_PIECEWISE captures it
  like the FP32 kernels.
* **Refused combinations** (raise at init): fused CUDA spec decode,
  AITER/FlashInfer GDN, align mode on the V1 runner, head dims other than 128.

## Use

```bash
bash tools/rdna/serve_rdna.sh RECIPE=flashnext-mtp2 VLLM_RDNA_GDN_LEAPQUANT=1
```

or uncomment `VLLM_RDNA_GDN_LEAPQUANT=1` in
`tools/rdna/recipes/flashnext-mtp2.env`. The worker log prints
`GDN recurrent state: LeapQuant INT8 (...)`.

## Results (flashnext-mtp2, 4x V620 TP4, 2026-11-13)

Base = flag off, head = flag on, same build (`_rocm_C` md5 identical; the
trees differ only in the recipe line). Receipts:
`~/w4a8_runs/leapquant/speed-20261113-023315` (MonoKernel off) and
`speed-20261113-062600-mono` (MonoKernel on, `EXTRA_ARGS=--monokernel`), each
`pr_receipt.sh` ABBA over 1k/512, 16k/1k, 32k/2k x c=1,4,8, plus the quality
boots `quality-20261113-005043` (on/off/off2) and
`quality-20261113-013837-mono` (on/off).

### Quality: gated checks inside the floor; decode-path KL needs more samples

| check (MonoKernel off) | off -> on | floor (off -> off2) |
|---|---|---|
| `topk_kl` docs, 64 windows: KL / top-1 | 0.01198, 0.01202 / 0.967 | 0.01168 [upper CI 0.01394] / 0.968 |
| long-context PPL (6 x ~15k tokens) | -0.01 %, +0.15 % | +0.10 % |
| decode-path KL (`decode_kl.py`), 32 x 512 tokens | 4.19e-3; 6.94e-3 | 4.77e-3 |
| decode-path KL, 8 x 2048 tokens at 15-22k context | 8.2e-3; 1.31e-2 | 7.3e-3 |
| concurrency probe / greedy / prefix cache | 0 FAIL / 3/3 / 3.1x | same |

The two decode-path KL figures per row are two on/off boot pairs of the same
build (`quality-20261113-005043`, `quality-20261113-101622`). The second
pair had no off2 boot, so it has no floor of its own. The metric averages
only the ~800 (512) / ~170 (2048) positions before the greedy paths
diverge, so it is heavy-tailed (p99 0.14-0.18 nats). That one build gives
4.2e-3 and 6.9e-3 means that a single pair does not resolve a 1.5x
difference. In the second pair: long-context PPL -0.25 %; gsm8k / needle /
humaneval (160 each) 96.2 / 100 / 96.2 % -> 96.2 / 100 / 96.9 %
(pooled 97.5 -> 97.7 %); PPL probe 4.81 -> 4.83; concurrency 0 FAIL.

MonoKernel on, off -> on: decode-path KL 4.84e-3 (512) and 7.7e-3 (2048),
long-context PPL +0.23 %, gsm8k / needle / humaneval (160 each, pooled 480)
97.7 % -> 97.7 %, concurrency 0 FAIL, prefix cache 3.2x. The worker log shows
`mono gdn_rdna2: verify N rows -> fused kernel` with the flag on.

### Memory: no KV capacity gain on this recipe

KV capacity is 266,945 tokens with the flag on and off. With
`BLOCK_SIZE=1024` the attention page sets the unified page size: vLLM pads
the GDN page by 29.13 % (FP32 state) or by 182.78 % (LeapQuant) to match it,
so the smaller state becomes padding and frees no KV blocks. One block, one
GDN layer, one rank:

| content | FP32 state | LeapQuant |
|---|---:|---:|
| recurrent state, 12 heads | 768 KiB | 337.1 KiB |
| conv window (2560 ch x 5 taps, fp16) | 25 KiB | 25 KiB |
| padding to the attention page | 231 KiB | 661.9 KiB |
| total (= attention page, 1024 tokens) | 1024 KiB | 1024 KiB |

The attention bytes per token stay constant when its page and block size
shrink together. The fixed per-request Mamba reservation does not: smaller
pages reduce its byte cost. LeapQuant makes a 512-token page feasible; the
793 KiB FP32 state cannot fit it.

`MambaSpec.max_memory_usage_bytes` reserves `2 + num_speculative_blocks +
num_prefill_checkpoint_blocks` pages in align mode. This HIP GDN path has
no internal prefill checkpoint exporter (`_checkpoint_alignment=None`), so
the reservation is **4 pages per GDN group**, not 5 (MTP2, checkpoint
blocks=0). For the 3 groups that is 12 pool blocks per request. An actual
CPU-only `MambaSpec` calculation gives 4 MiB per layer at block 1024 and
2 MiB at block 512; state contents are 812,032 and 370,816 bytes.
The LeapQuant kernel suite passes 13 tests, including grouped normalize
on 512 KiB and 1 MiB padded pages. These cases check equality with dense
per-layer normalization and preservation of neighboring conv/padding bytes.

LeapQuant recurrence and conv update use the home block
(`spec_state_indices[:, 0]`). Removing speculative blocks is separate work:
verify lengths currently depend on the index width, and V2 requires all
Mamba groups to share speculative-block scheduling parameters. Neither is
changed by the 512 experiment.

The 512-token experiment is included in this PR (user decision 2026-11-13),
with MTP2 and FULL_AND_PIECEWISE unchanged. The existing launcher accepts
`BLOCK_SIZE=512`; runtime bring-up and ABBA validation are pending.
Packing two LeapQuant layers into one page is reserved for a second PR.
That could reduce cached-prefix state bytes without shortening the
checkpoint interval, but needs allocator changes.

### Speed (first build): no gain

MonoKernel off, head vs base mean (ABBA verdict in parentheses where both
head boots fall on one side of both base boots):

| cell | TTFT p50 | prefill aggregate | ITL p50 | ITL p99 | output aggregate |
|---|---:|---:|---:|---:|---:|
| 1k/512 c=1 | +0.5 % | -0.2 % | -0.9 % (win) | +5.7 % (regr.) | -1.0 % |
| 1k/512 c=4 | -0.5 % | +0.0 % | -0.1 % | -72.2 % | +1.1 % |
| 1k/512 c=8 | -0.1 % | -0.1 % | -0.6 % | +0.1 % | -0.9 % |
| 16k/1k c=1 | -0.2 % | +0.1 % | -0.4 % | +5.9 % (regr.) | -2.0 % |
| 16k/1k c=4 | -0.9 % | -0.0 % | +1.3 % (regr.) | +0.4 % | +0.9 % |
| 16k/1k c=8 | +24.9 % (regr.) | -0.1 % | +1.4 % (regr.) | +0.1 % | +4.8 % (win) |
| 32k/2k c=1 | -0.2 % (win) | +0.3 % (win) | -1.2 % (win) | +5.7 % (regr.) | -3.4 % |
| 32k/2k c=4 | +0.4 % | -0.1 % | -0.8 % (win) | +0.0 % | -1.0 % |
| 32k/2k c=8 | -6.2 % (win) | -2.3 % (regr.) | +1.2 % (regr.) | +0.3 % | +1.2 % |

MonoKernel on, same columns:

| cell | TTFT p50 | prefill aggregate | ITL p50 | ITL p99 | output aggregate |
|---|---:|---:|---:|---:|---:|
| 1k/512 c=1 | -0.4 % | +1.0 % | +0.1 % | +6.5 % (regr.) | -3.5 % |
| 1k/512 c=4 | -0.7 % (win) | -0.2 % (regr.) | +0.0 % | -71.9 % | -1.2 % |
| 1k/512 c=8 | +0.0 % | -0.4 % | +0.3 % | +0.1 % | -4.1 % (regr.) |
| 16k/1k c=1 | -0.1 % (win) | -0.0 % | +0.3 % | +6.5 % (regr.) | +1.1 % |
| 16k/1k c=4 | -2.7 % (win) | -0.0 % | -0.5 % | +0.3 % (regr.) | -3.1 % (regr.) |
| 16k/1k c=8 | -9.3 % | -0.1 % | -0.5 % | +0.0 % | -2.7 % |
| 32k/2k c=1 | -0.2 % (win) | +0.2 % (win) | +0.3 % (regr.) | +6.1 % (regr.) | -4.4 % (regr.) |
| 32k/2k c=4 | +0.1 % | -0.0 % | +0.5 % | +0.3 % (regr.) | +3.5 % |
| 32k/2k c=8 | +2.7 % (regr.) | -9.5 % (regr.) | +0.9 % (regr.) | +0.1 % (regr.) | -3.9 % |

* Output aggregate follows MTP acceptance (1.68-2.60 per boot), not the
  flag.
* **c=1 ITL p99 +6 %** (23.6 -> 25.0 ms) is the fold: all 36 layers
  refold on the same step every ~16 committed tokens, one layer after the
  other.
* 16k/1k c=8 TTFT p50 is bimodal per boot (MonoKernel-on base boots: 11.7 s
  and 7.0 s); prefill aggregate does not move.

### Efficiency pass: one normalize launch per KV-cache group

Per decode step on one V620 (36 layers, HIP-graph replay,
`~/w4a8_runs/leapquant/mbench_mono.py`), the first build spent, at n=1:
~114 us on 36 normalize launches (~3 us each, mostly launch gaps), ~100 us
on folds (40 us per fold workgroup, one layer after the other) and ~55 us on
replaying the window's records.

* **Grouped normalize.** The GDN layers of one KV-cache group share a block
  table and one attention metadata, so the group's first LeapQuant layer
  normalizes all of its layers' blobs in one launch (`layer_offsets`, grid.y
  = layer) and the others skip. The groups are built at KV-cache bind
  (`share_leapquant_normalize_groups`, next to
  `share_replayssm_ring_trackers`). Flash-Next TP4 has 3 GDN groups of 12
  layers: 3 launches per step instead of 36, and each group's folds run
  concurrently instead of back to back. Decode arenas (non-spec pure
  decode) stay per layer. Same bytes as per-layer normalize
  (`test_grouped_normalize_matches_per_layer`,
  `test_normalize_groups_follow_kv_cache_groups`).
* **Replay loads.** The record replay loops unroll by 2 so two records'
  loads issue together (unroll 4 was slower at n=8). Same arithmetic order.
* Tried and dropped: LDS instead of shuffle reductions in the fold (40 us ->
  39 us at n=1, 96 us -> 110 us at n=8). Power iterations 2 -> 0 saves 20-40
  us per step but changes the compensators; not taken, quality not measured.

| step | FP32 MonoKernel | LeapQuant, first build | LeapQuant, efficiency pass |
|---|---:|---:|---:|
| decode n=1 | 341 us | 550 us | 368 us |
| MTP verify n=1 (acc 2) | 520 us | 838 us | 589 us |
| decode n=8 | 1503 us | 1290 us | 1135 us |
| MTP verify n=8 (acc 2) | 3289 us | 2147 us | 1857 us |

(`GROUP=0` vs `GROUP=3`. One launch over all 36 layers was 5-8 us faster,
but it is wrong: see below.)

Prefill adds materialize + store at chunk boundaries: 1.9 ms per
2048-token chunk (`mbench_prefill.py`).

#### Bug found in the first efficiency build: one group across KV-cache groups

The first version (`8ba674ad8b`) grouped every GDN layer of the model under
the first one. vLLM's hybrid allocator spreads Flash-Next's 36 GDN layers
over 3 KV-cache groups (`kv cache group sizes (1024, 1024, 1024, 1024, 8,
1024)`), each with its own block table, so one request sits at different
state slots per group. The leader committed the other groups' records at
its own group's slots, which belong to other requests in those groups.
Their own slots were never committed.

* Symptom: decode fell into repetition loops, which MTP accepts easily.
  The MonoKernel-on ABBA receipt showed fake speed "wins" (c=1 output
  +37 %, ITL p50 -4 to -8 % at c=4/8). Outputs whose last 600 chars compress
  below 0.12 (zlib), out of 32 at c=8: base boots 2-7, flag-on boots 10-21.
  On the first build, the flag-on boots matched base (1-6).
* The gated checks missed it. Prompt KL and teacher-forced PPL do not run
  decode (head KL 0.01178/0.01172 vs floor CI 0.01413, PPL +0.47/+0.15 %).
  The concurrency probe hinted at it: 1 FAIL and 10 soft divergences, vs
  0-5 soft on every other boot.
* Isolation: the tree3 and tree2 kernels are bitwise identical over 200
  MTP steps (decode, normalize, MonoKernel). Disabling only the Python
  grouping restores base behaviour.
* Fix: `d879da8c93` builds one group per KV-cache group.
  `test_normalize_groups_follow_kv_cache_groups` uses two groups with
  different slots per request.

### Speed (efficiency build `d879da8c93`, MonoKernel on)

ABBA receipt `~/w4a8_runs/leapquant/speed-20261113-152500-mono`, same
columns and rule as above:

| cell | TTFT p50 | prefill aggregate | ITL p50 | ITL p99 | output aggregate |
|---|---:|---:|---:|---:|---:|
| 1k/512 c=1 | +0.7 % (regr.) | +0.1 % | -1.0 % (win) | -0.6 % (win) | -19.3 % (regr.) |
| 1k/512 c=4 | -0.8 % (win) | -0.1 % | +0.1 % | +258.3 % | +6.2 % (win) |
| 1k/512 c=8 | -1.7 % (win) | +0.3 % (win) | -5.4 % (win) | -0.3 % (win) | -1.3 % |
| 16k/1k c=1 | -0.1 % | +0.1 % (win) | -1.2 % (win) | -0.3 % | -6.8 % |
| 16k/1k c=4 | +0.7 % | +0.1 % | -0.8 % | +0.0 % | +3.1 % |
| 16k/1k c=8 | -2.3 % | +0.1 % | -1.4 % (win) | -0.2 % (win) | -1.4 % |
| 32k/2k c=1 | -0.2 % (win) | +0.2 % (win) | -0.1 % | +0.3 % (regr.) | +7.0 % (win) |
| 32k/2k c=4 | +1.5 % (regr.) | +0.0 % | -1.0 % | -0.1 % | +1.4 % |
| 32k/2k c=8 | +1.7 % | +2.1 % (win) | +0.1 % | -0.1 % (win) | +4.6 % (win) |

* **The c=1 ITL p99 fold spike is gone** (+6 % in the first build, now
  -0.6 / -0.3 / +0.3 %): each KV-cache group's folds run in one launch.
* ITL p50 moves -0.1 to -1.4 %, and -5.4 % at 1k/512 c=8 (45.2 vs 47.8 ms).
* At 1k/512 c=1 (4 requests), acceptance length from each benchmark JSON
  is A1=2.034, B1=1.774, B2=1.622, A2=2.269 tokens per step. Both flag-on
  boots have lower acceptance and output throughput despite similar ITL.
  The -19.3 % throughput result remains a regression to investigate.
  Other cells move in both directions; pooling their acceptance logs
  cannot establish parity for this cell.
* 1k/512 c=4 ITL p99 +258 % is one boot (B2 380.9 ms vs 60.9-62.6 ms on the
  others). The first build had the same stall on a base boot (-72 %).
* At 32k/2k c=8 every arm logs `Running: 5-6 reqs, Waiting: 2-3 reqs`
  (TTFT p50 ~85 s). This is consistent with KV admission pressure
  [INFERENCE]; the 512 experiment will check admission and throughput.
* Gates: quality PASS (head KL 0.01158 / 0.01178 vs floor 0.01376, top-1
  0.967 vs 0.964, PPL +0.055 / -0.077 %), concurrency probe PASS with 0
  failing on both flag-on boots, stall PASS. Repetition loops (`loopscan.py`,
  out of 32 per c=8 cell): base 0-5, flag on 1-5.

### Triton census

`VLLM_TRITON_LAUNCH_COUNT=1` over the quality boots: the flag-on and
flag-off servers launch the same Triton kernels (MonoKernel off: 10,
MonoKernel on: 12). LeapQuant adds none; every LeapQuant op is HIP. The
baseline ones are the recipe's (`_causal_conv1d_update/_fwd` because the
recipe sets `VLLM_CAUSAL_CONV1D_RDNA2_*=0`, `_mamba_align_block_table_kernel`,
`_qsa_mqa_paged_kernel`, the PLE kernels, and under the MonoKernel
`_fused_qk_rmsnorm_rope_gate_kernel` and `_num_nans_kernel`).
