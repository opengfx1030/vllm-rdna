# Qwen3.8-27B HI2 3.0 bpw at TP=2: TunableOp rows for the rocBLAS prefill GEMMs (2026-11-15)

Branch `feat/hi2-tunableop` (from `feat/hi2-integration`, which holds recipe
`27b-hi2-tp2`). Checkpoint `BlivionIaG/Qwen3.8-27B-HI2-3.0bpw` (`rdna_hint_flex`,
snapshot `db2a3aa3cd5efb0f92403cd4864e1f0d99378149`), box `par1-cs25`, tree
`~/w4a8_runs/leapquant/tree-hi2t`, venv `venv-7.14.0_0.32.0`. Tuning and every
measurement ran on **HIP 4** (AMD Radeon Pro V620, gfx1030, PCI bus 0xdf, 29.98 GiB,
175 W package cap) inside `flock ~/w4a8_runs/hi2-gpu4.lock`, so no other job ran on the
card during a measurement; HIP 2,3 (the TP=2 pair) were not touched. Labels:
**[MEASURED]** = produced by a command listed here, **[CODE-READ]** = read from the
source at the cited place, **[DERIVED]** = arithmetic on measured values,
**[INFERENCE]** = a model or expectation a TP=2 run must confirm.

> **Status (2026-11-15, after the scoreboard queue).** Gate A and gate B ran on HIP 2,3 and the
> rows ship as the default of recipes `27b-hi2-tp2` and `27b-hi2-tp2-mtp2`. The deterministic
> concurrency probe fails 0 of 480 checks with the rows (arm `conc-on`) and 0 of 480 without
> them (`conc-off`). Arm `tunable` against `base`: TTFT 1k/512 c=1 1.51 to 1.41 s, 16k/1k c=1
> 30.54 to 28.68 s, 32k/2k c=1 75.75 to 72.09 s, output x1.01 to x1.03 at c=1 and within noise
> at c=8; long-context PPL 3.6633 to 3.6636, prompt KL 7.3e-6. Sections 1 to 6 were measured
> on HIP 4 before that: where they say "not run" or "not accepted", this status supersedes
> them. Tables and paths: [`hi2-27b-tp2-serving.md`](hi2-27b-tp2-serving.md).

## Summary

* The TP=2 serving path of this model issues rocBLAS GEMMs for every prefill chunk
  (the hi2 dequant + `mm_out` path above 48 tokens) and, with MTP, for the drafter's
  dense layers. The canonical rows (`rocm7.14-rocblas5.5`, 3330 per rank) hold **no key
  of this model**, so all of them ran on the rocBLAS heuristic.
* The M values that reach rocBLAS are **not** just the chunk sizes 1024 and 2048. A
  scheduler census with the production `Scheduler` on the five scoreboard cells shows
  M = 1023 (the last chunk of every block-multiple prompt), 1025..1031 (a chunk plus
  the other sequences' decode rows) and 2046 / 2047 at c=8. Rows for 1024 and 2048 alone
  would hit 0 % of the GEMM FLOPs of three of the five cells (section 1.3).
* **Result ([MEASURED], HIP 4).** 89 candidate rows were tuned; **36 rows per rank are
  adopted** at a 3 % margin (derived profile `rocm7.14-rocblas5.5-hi2-27b-tp2`, 3330 + 36 =
  3366 rows per rank, `auto: false`, selected only by recipe `27b-hi2-tp2`). They sit at
  the M at or just below a multiple of 1024 (1023, 1024, 2046, 2047, 2048) of `gate_up`,
  `in_proj_qkvz`, `qkv` and `out / o_proj`, plus the 48-row `in_proj_ba` at every census
  M. Kernel gain over the rocBLAS heuristic (cold weights, graph replay): `gate_up`
  +15.7 .. +20.1 %, `qkvz` and `qkv` +4 .. +5 % at 1023 / 1024 and +11 % at 2046 .. 2048,
  `out / o` +9.1 % at 1024 and +3.3 .. +3.7 % at 2046 .. 2048, `ba` 2.4 .. 2.9x (section 3).
* **No row helps an odd M** (1025 .. 1031, 1033 .. 1045). The tuner returns the `Default`
  solver or one under 3 % faster, while the heuristic takes 7 .. 17 % longer at M = 1025 ..
  1031 than at 1024 (`gate_up` 10.43 ms at 1024, 11.14 .. 11.25 ms there; `qkvz` and
  `qkv` +13 .. 15 %, `down` +17 %): an M a few rows past a tile multiple costs a whole
  extra tile row, which no rocBLAS solution avoids [INFERENCE: tile quantization]. The
  c=8 scoreboard cells and the agentic
  mix run at those M, so rows barely touch them: modelled target-GEMM time saving 9.5 ..
  9.7 % for the c=1 cells (1k, 16k, 32k), 1.6 % (1k c=8), 0.6 % (16k c=8), 0.4 % (agentic
  mix) [INFERENCE: scheduler census x kernel table, section 3.5]. Splitting such a chunk
  into its tile-aligned part and a GEMV tail would take 13 % (heuristic) to 21 % (with
  these rows) off the GEMM time of an M = 1029 forward [DERIVED from real-op timings,
  section 3.4]; that is a change of the op, not a row, and is not part of this branch.
* Model level, one GPU, the TP=2 GEMMs emulated (3 prompts x 64 greedy tokens, rows off
  vs on): tokens identical, max |dlogit| over the top-20 raw fp16 logits 0.094 with the
  three prompts batched and 0.043 one at a time (section 4.2). The LM head ships no row
  (section 3.3). The deterministic concurrency probe on HIP 2,3 (section 7) later failed 0 of
  480 checks with the rows and 0 of 480 without them (status note above).

## 1. Shape census

### 1.1 Per-rank GEMM shapes

TP=2 shards every projection to one rank: column-parallel layers keep half the rows,
row-parallel layers (`down_proj`, GDN `out_proj`, attention `o_proj`) half the
columns. The key of one call is `tn_{R}_{M}_{K}_ld_{K}_{K}_{R}` (weight `R x K`, `M`
tokens; `ld = K` on gfx1030 because the cache-aliasing weight pad only applies on gfx11).

| slot | layers | per-rank R x K | share of the target GEMM FLOPs |
| --- | ---: | --- | ---: |
| `gate_up_proj` | 64 | 17408 x 5120 | 47 % |
| `down_proj` | 64 | 5120 x 8704 | 23 % |
| `in_proj_qkvz` (GDN) | 48 | 8192 x 5120 | 17 % |
| `out_proj` (GDN) and `o_proj` (attention): the same key | 48 + 16 | 5120 x 3072 | 8 % |
| `qkv_proj` (attention) | 16 | 7168 x 5120 | 5 % |
| `in_proj_ba` (GDN) | 48 | 48 x 5120 | 0.1 % |
| `lm_head` (dense fp16) | 1 | 124160 x 5120 | only at M > 32 |
| MTP `fc` (dense fp16, MTP=2) | 1 | 2560 x 10240 | drafter pass |
| MTP decoder layer (dense fp16, MTP=2) | 1 | the `qkv` / `o` / `gate_up` / `down` shapes above | drafter pass |

[DERIVED] from the config (`hidden 5120`, `intermediate 17408`, 64 layers = 48 GDN +
16 attention, `vocab 248320`) by `tunable_hi2.py rank_shapes`, which mirrors
`tools/rdna/rdna_hint_flex/tp_layout.py`. [MEASURED] `tunable_hi2.py record` runs the
real ops on the checkpoint's own per-rank segments under
`PYTORCH_TUNABLEOP_RECORD_UNTUNED=1`: the 60 keys torch records equal the formula keys
exactly (no extra key, no missing key) and every real segment's R and K equals the
formula's.

### 1.2 Which calls reach rocBLAS

* hi2 layers [CODE-READ `csrc/rocm/rdna_hint_flex_rdna2.cu`
  `rdna_hint_flex_linear_rdna2`]: M <= 48 is the GEMV; above, transform -> dequantize
  to an fp16 `[R, K]` scratch -> `at::mm_out(out, xt, wd.t())` = rocBLAS (unless
  `RDNA_HINT_FLEX_PREFILL` selects the hand-written GEMM; the recipe leaves it 0).
  Both kernels' time is in the prefill step; only the second has a TunableOp key.
* dense layers [CODE-READ `vllm/model_executor/layers/utils.py`
  `rocm_unquantized_gemm_impl`]: up to 8 rows `gemv_f16_rdna2` (the LM head up to
  `VLLM_RDNA_LM_HEAD_GEMV_MAX_M`, 32 in the recipe); 9..32 rows `rdna_gemm_smallm`
  only when `VLLM_RDNA_SMALLM_GEMM_MAX_BYTES > 0` (default 0); otherwise `F.linear`
  = rocBLAS.
* The MTP drafter's first pass runs on **all** tokens of the step [CODE-READ
  `TargetDependentARSpeculator.propose` of the V2 runner and `llm_base_proposer.py`],
  so with MTP=2 the drafter layer and `fc` run at the chunk M of every prefill step and
  at the padded decode M (capture sizes `[3, 6, 12, 24]`); the drafter's dense shapes
  share their keys with the target's hi2 shapes (`d_gate_up` = `gate_up`, ...).
* At MTP=0 decode never reaches rocBLAS (M <= 8 everywhere, LM head included).
* GDN prefill and decode attention use custom kernels, not rocBLAS [CODE-READ
  `qwen_gdn_full_forward` / `RDNA_ATTN`]. The only rocBLAS GEMMs inside the
  attention are those of the opt-in unfused prefill path (section 1.4).

### 1.3 Which M reach rocBLAS: scheduler census

`tools/rdna/rdna_hint_flex/sched_census.py` [MEASURED, CPU only] drives the production
`Scheduler` + `KVCacheManager` with the recipe's configuration (block size 1024,
`max_num_seqs 8`, `max_num_batched_tokens 2048`, prefix caching, mamba align mode, MTP
off and MTP=2 with 2 and 1 accepted tokens per step) on the real prompt lengths of the
five scoreboard cells (`vllm bench serve --dataset-name random --seed 12345`: the
`RandomDataset` lengths are exact, 0 retokenize mismatches), the concurrency probe, a
warm-repeat prefix-cache workload, a mixed agentic workload and a `prompt_logprobs`
workload. Every request is checked: prefix hit + sum of its prefill chunks = prompt
length, and every chunk ends on a 1024-token block boundary or at the prompt end
(`~/w4a8_runs/hi2-tunable/sched/census.{txt,json}`). Share of the GEMM FLOPs
(M > 48 steps, weighted by M) per M, MTP=0:

| workload | M (FLOP share) |
| --- | --- |
| 1k/512 c=1 | **1023 (100 %)**: a 1024-token prompt is chunk 1023 + a 1-token last chunk |
| 1k/512 c=8 | 1029 (47 %), 1030 (12 %), 2046 (12 %), 1025..1028 (6 % each), 1023 (6 %) |
| 16k/1k c=1 | 2048 (87 %), 1023 (7 %), 1024 (6 %) |
| 16k/1k c=8 | **1031 (50 %)**, 1030 (9 %), 1029 (9 %), 1025..1028 (6 % each), 2048 (6 %), 2047 (1 %) |
| 32k/2k c=1 | 2048 (93 %), 1023 (4 %), 1024 (3 %) |
| agentic mix | 1031 (76 %), 1030 (8 %), the rest are arbitrary tails |
| concurrency probe | 2048 (53 %), 1015 (13 %), 2021 (13 %), 1024 (7 %), 1018 (7 %), 834 (5 %), 172 |
| warm repeat (prefix hit) | 1024 / 2048 plus the tail: 475 (1500-token prompt), 27 (3100), ... |
| `prompt_logprobs` | 2048 (40 %), 1027 (40 %), 903 / 906 / 297 |

c=8 steps are a prefill chunk plus the decode rows of the other running sequences
(1024 + 7 = 1031, 1023 + 7 = 1030, ...). With MTP=2 each decode sequence adds 3 rows, so
the c=8 values become 1027 .. 1045 (1045 = 50 % of 16k c=8, 68-71 % of the agentic mix).
After a prefix-cache hit the remaining prompt is again `[... , 1023, 1]`.

FLOP share of the M values a row set covers (exact keys), per workload, MTP=0:

| row set | 1k c=1 | 1k c=8 | 16k c=1 | 16k c=8 | 32k c=1 | agentic mix | probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M = 1024, 2048 only | 0 % | 0 % | 93 % | 1 % | 96 % | 0 % | 60 % |
| + 1023, 1025..1031, 2046, 2047 (this work) | 100 % | 100 % | 100 % | 100 % | 100 % | 89 % | 60 % |

With MTP=2 the set `1033, 1036, 1039, 1042, 1045` lifts the c=8 cells from 14-25 % to
90-94 % (agentic mix 4 % -> 85 %).

### 1.4 Decisions: bucketing and exclusions

* **No bucketing in model code.** Padding the activation to a row bucket would change
  the hi2 op or `rocm_unquantized_gemm_impl` for every model; exact-key rows for the
  census M values reach 100 % of the scoreboard cells' GEMM FLOPs without it.
* **Excluded: arbitrary chunk tails.** A prompt whose length is not a block multiple
  ends in a chunk of any size (probe 834 / 1015 / 1018 / 172, warm-repeat 475 / 27,
  `prompt_logprobs` 903 / 906 / 297): one chunk per request, each a distinct key. They
  keep the heuristic. Odd M is not a problem as such: 1023 and 1025..1031 are odd and
  get rows (section 3).
* **Excluded: the unfused prefill attention GEMMs** (`VLLM_FA_RDNA2_UNFUSED_PREFILL`,
  opt-in, off in `27b-hi2-tp2`): `torch.mm` of `[group * rows, 256] x [256, s_len]` and
  `[group * rows, s_len] x [s_len, 256]` where `s_len` is the running KV length, a new
  key at every chunk of every context length [CODE-READ
  `vllm/v1/attention/ops/rdna_unfused_prefill.py`]. Bucketing `s_len` would pad the
  softmax range. They stay on the heuristic; an arm with that flag is not covered by
  these rows.
* **Excluded: `down_proj` and MTP `fc`.** Tuned vs the heuristic they are within noise
  at 1024 and 2048 (section 3), so no row is adopted; the census M of those shapes are
  timed only (heuristic) for the expected-gain model.
* **LM head:** the recipe's GEMV cap of 32 keeps rocBLAS out of the 9..32-row verify
  step; rows for it are insurance for a cap-8 run only (section 3.3).
* **The GDN `in_proj_ba` rows** (R = 48) are adopted: they are 3x faster than the
  heuristic but 0.1 % of the FLOPs.

## 2. Method

All GPU steps are phases of `tools/rdna/rdna_hint_flex/tunableop_hi2.sh`; each GPU
process runs under `flock ~/w4a8_runs/hi2-gpu4.lock` with `HIP_VISIBLE_DEVICES=4`
(verified through `torch.cuda.get_device_properties`: V620, gfx1030, bus 0xdf).

1. **Shape groups.** `shapes` writes the census keys per group from the config
   (`tunable_hi2.py shapes`): `prefill` (the seven slots at M = 1024 and 2048),
   `sched0` / `sched1` (M = 1023, 1029, 1030, 1031, 2046 and 1025..1028, 2047 for
   `gate_up`, `in_proj_qkvz`, `in_proj_ba`, `out_proj / o_proj`, `qkv`), `sched2`
   (MTP=2 mixed steps 1033, 1036, 1039, 1042, 1045), `small` (the MTP drafter's dense
   layers at M = 9..24), `lmhead_small` and `lmhead_pl` (LM head). `down_proj` and
   `fc` are only timed at the scheduler M (`heuronly`).
2. **Tuning.** `tune GROUP` runs the fork's offline tuner
   (`tools/rdna/prof/tune_missing_gemms.sh tune`, i.e. `tunable.tune_gemm_in_file`) with
   `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1
   PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 TORCH_BLAS_PREFER_HIPBLASLT=0`, TunableOp's
   timing defaults ([MEASURED] on HIP 4: 30 ms and at most 100 iterations per candidate,
   4 MiB rotating buffer) and its numerical check
   (a candidate whose result differs from the rocBLAS default's by more than
   `atol = rtol = 2e-3` is never selected). Keys already in the canonical rank-0 table
   are skipped. A group is split into shards, one lock hold each (`prefill`: 4 shards of
   3 .. 4 keys; `sched0` .. `sched2`: 5 shards of 5 keys). Wall time per group, lock waits
   included: 6.6, 12.7, 11.6 and 23.7 min (1.6 .. 4.7 min per shard), so no other job
   shared the card during a tuning or a measurement and the other agents' short jobs
   waited at most one shard. One tuning serves both TP
   ranks (same arch, same build); the rows are written identically to every rank file.
3. **Measurement.** `measure heur|new` times `F.linear` of every key on **cold rotating
   weights** (>= 640 MB of distinct weight copies, so no weight is cache-resident from
   the previous call), CUDA-graph replayed, 5 repeats, median (`tunable_hi2.py gemm`, the
   protocol of
   `tools/rdna/rdna_hint_flex/bench_tp2.timed` and of the HiPrefillGemm study), once
   with TunableOp off (the heuristic) and once with the new rows looked up. The result
   is a per-key time and TF/s.
4. **Numerics gate per row.** `check` runs `tools/rdna/tp8pp2/rowcheck.py` in two
   processes (the heuristic, and the rows loaded the way a worker loads them; torch
   2.12 `tunable.enable(False)` does not unload rows, so one process cannot compare
   them): each key on a gaussian and an outlier-channel activation set, eager and from
   a HIP graph, against the fp64 matmul of the same fp16 inputs. A row is rejected when
   its error exceeds 1.5x the heuristic's (max error over max |ref|, or relative L2),
   when its graph result differs bitwise from its eager result, or when three runs
   differ.
5. **Curation.** `curate` (`tools/rdna2_028/curate_tunableop_rows.py`) keeps a row only
   when it is at least **3 %** faster than the heuristic in the cold-weight timing;
   rows whose solver is `Default` and rows below the margin are dropped (an absent row
   and a `Default` row behave the same).
6. **Harvest.** `tools/rdna/prof/tunableop_missing.py harvest --reject <rowcheck
   rejects>` appends the curated rows to every rank file of the derived profile and
   updates its `provenance.json` and the registry's `rows_per_rank` (section 5).
7. **Model-level check.** `tp2_emul_check.py` (section 4.2) runs the real checkpoint on
   one GPU with the TP=2 per-rank GEMM keys, untuned and tuned, and compares greedy
   tokens and logits.

Not run here, because it needs the TP=2 GPUs (HIP 2,3 belong to the scoreboard queue):
the **deterministic concurrency probe** (flat all-reduce; the hierarchical one cannot
apply to a TP=2 group, section 7) that decides whether the row set is accepted.

## 3. Kernel-level results (HIP 4)

HIP 4 was power-limited during every measurement. In the 2 s `rocm-smi` log of the
alternating passes (155 samples at >= 150 W) the junction sat at 76 .. 81 C, the package
at 169 .. 176 W against a 175 W cap and sclk between 1230 and 1955 MHz (median 1595), so
absolute TF/s are lower than on a cool card and move between passes. The gain of a row
is the ratio of two kernels measured back to back on the same card:
gain = heuristic time / tuned time - 1. A row is adopted when its tuned time is under
97 % of the heuristic's (gain above 3.1 %). The 34 keys whose single-pass gain was close
to that margin (9 `qkvz`, 9 `qkv`, 11 `out / o`, 5 `gate_up`) were measured again in
alternating heuristic / tuned passes (heuristic, tuned, tuned, heuristic, heuristic,
tuned) and decided on the median of four passes per condition. For those keys the four
pair gains differ by a median of 1.0 percentage point (maximum 2.6); the median of four
sits -0.3 .. +2.1 points (mean +0.6) from the first pass, and six rows pass the margin
only on the median (`out / o` 2046, 2047, 2048, `qkv` 1023, `qkvz` 1023 and 1024; a
first-pass curation would have adopted 25 rows instead of 31). An earlier pilot of the 14
chunk-size keys on a cooler card gave `gate_up` +14.7 % / +10.9 % at 1024 / 2048, `qkvz`
4.9 / 6.6 %, `qkv` 5.9 / 8.5 %, `out` 7.3 / 6.5 %.

### 3.1 The chunk sizes (M = 1024, 2048)

| slot | M | heuristic ms (TF/s) | tuned ms (TF/s) | gain | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `gate_up` | 1024 | 10.430 (17.5) | 8.686 (21.0) | +20.1 % | adopted |
| `gate_up` | 2048 | 20.958 (17.4) | 18.120 (20.1) | +15.7 % | adopted |
| `in_proj_qkvz` | 1024 | 4.102 (20.9) | 3.942 (21.8) | +4.1 % | adopted |
| `in_proj_qkvz` | 2048 | 8.829 (19.5) | 7.917 (21.7) | +11.5 % | adopted |
| `qkv` | 1024 | 3.631 (20.7) | 3.469 (21.7) | +4.7 % | adopted |
| `qkv` | 2048 | 7.743 (19.4) | 6.992 (21.5) | +10.7 % | adopted |
| `out / o` | 1024 | 1.659 (19.4) | 1.521 (21.2) | +9.1 % | adopted |
| `out / o` | 2048 | 3.079 (20.9) | 2.976 (21.6) | +3.5 % | adopted |
| `in_proj_ba` (R = 48) | 1024 | 0.224 | 0.077 | 2.9x | adopted |
| `in_proj_ba` (R = 48) | 2048 | 0.263 | 0.108 | 2.4x | adopted |
| `down` | 1024 | 4.279 (21.3) | 4.334 (21.1) | -1.3 % | not adopted |
| `down` | 2048 | 8.535 (21.4) | 8.553 (21.3) | -0.2 % | not adopted |
| MTP `fc` | 1024 | 2.788 (19.3) | 2.798 (19.2) | -0.4 % | not adopted |
| MTP `fc` | 2048 | 5.002 (21.5) | 4.997 (21.5) | +0.1 % | not adopted |

### 3.2 The scheduler's M

Gain in % (`*` = adopted; `in_proj_ba`: speed-up factor). A value in parentheses has no
row behind it: `down` and the MTP `fc` were tuned only at M = 1024 and 2048, so at the
other M both columns are the heuristic kernel measured twice and the value is the
measurement noise. Times, TF/s, solver, error and the number of passes of every key (119
keys): `tunableop/rocm7.14-rocblas5.5-hi2-27b-tp2/kernel_table.csv`.

| slot | 1023 | 1024 | 1025 | 1026 | 1027 | 1028 | 1029 | 1030 | 1031 | 1033 | 1036 | 1039 | 1042 | 1045 | 2046 | 2047 | 2048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gate_up` | +19.4* | +20.1* | -0.0 | -0.3 | -0.0 | -0.2 | +0.2 | -0.0 | -0.1 | -1.0 | -0.8 | -0.7 | -1.0 | -0.7 | +18.7* | +18.8* | +15.7* |
| `in_proj_qkvz` | +4.8* | +4.1* | -1.1 | +1.7 | +2.1 | +2.0 | +1.9 | +2.1 | +1.8 | +1.0 | +1.2 | +1.2 | -1.4 | +1.1 | +10.9* | +11.4* | +11.5* |
| `qkv` | +5.2* | +4.7* | -1.0 | -0.8 | +2.8 | +2.7 | -0.5 | -0.5 | -0.7 | -1.4 | -1.2 | +2.6 | +2.6 | -1.5 | +11.0* | +10.7* | +10.7* |
| `out / o` | -0.1 | +9.1* | -0.5 | -0.4 | -0.4 | -0.7 | -0.4 | -0.4 | -0.4 | -1.0 | -0.9 | -1.1 | -0.7 | -1.3 | +3.7* | +3.3* | +3.5* |
| `in_proj_ba` | 2.9x* | 2.9x* | 2.5x* | 2.5x* | 2.5x* | 2.5x* | 2.9x* | 2.9x* | 2.9x* | 2.6x* | 2.9x* | 2.9x* | 2.9x* | 2.9x* | 2.4x* | 2.4x* | 2.4x* |
| `down` | (-1.4) | -1.3 | (-0.7) | (-1.0) | (-0.4) | (-0.7) | (-0.6) | (-0.7) | (-0.3) | (-0.6) | (-0.5) | (-0.1) | (-0.5) | (-0.7) | (-1.3) | (-0.7) | -0.2 |
| MTP `fc` | (-0.2) | -0.4 | (-0.6) | (+0.2) | (-0.2) | (+0.3) | (+0.5) | (+0.1) | (+0.5) | (+0.0) | (-0.0) | (+0.1) | (-0.3) | (-0.4) | (+0.1) | (-0.1) | +0.1 |

M = 1033 .. 1045 are the MTP=2 mixed steps (one pass, 25 keys); the other columns are
the median of four passes for the 34 re-measured keys and one pass for the rest. The
gains that were not adopted reach +2.1 % (`qkvz` 1027, 1028, 1030) and +2.8 % (`qkv`
1027, 1028, 1039, 1042): below the 3.1 % margin, but within the pass-to-pass spread seen
on the repeated keys (median 1.0, maximum 2.6 points). The `down` and `fc` values in
parentheses give the noise floor of the method, -1.4 .. +0.5 %.

### 3.3 LM head and the MTP drafter

The dense LM head (124160 x 5120 per rank) runs on rocBLAS only for 9 .. 32 rows
without the recipe's `VLLM_RDNA_LM_HEAD_GEMV_MAX_M=32` (default cap 8). Real dense
dispatch, median of CUDA-event windows, ms, HIP 4:

| rows | rocBLAS heuristic | GEMV (cap 32) | rocBLAS with a tuned row |
| ---: | ---: | ---: | ---: |
| 9 | 11.28 | 4.57 | 3.99 |
| 12 | 11.42 | 5.12 | 4.86 |
| 15 | 11.50 | 5.53 | 4.90 |
| 18 | 11.58 | 6.35 | 5.57 |
| 21 | 11.64 | 6.80 | 5.62 |
| 24 | 11.65 | 7.41 | 5.65 |

At 12 rows (a c=4 MTP=2 verify step) the heuristic costs 11.4 ms, the recipe's GEMV 5.1 ms
and a tuned row 4.9 ms: 5 % less time than the GEMV (24 rows: 24 % less, 5.65 against
7.41 ms). **No LM head row is shipped.** With the cap at 32 the rows are never looked up
(the GEMV takes every M <= 32); using them needs the cap lowered to 8 and a row for every
M from 9 to 32 (24 keys), because a step whose row count has no row would run the 11+ ms
heuristic, plus the numerics gate and the concurrency probe for a decode-path change. The
six rows tuned here (M = 9, 12, 15, 18, 21, 24) are in `~/w4a8_runs/hi2-tunable/run/lm/`
on the box. The MTP drafter's dense layers at 9 .. 24 rows (`small` group, 30 keys) were
not tuned in this campaign.

### 3.4 Why odd M gains nothing, and what would

On the heuristic (cold weights) a chunk of M = 1029 takes longer than one of M = 1024 for
0.5 % more rows: `gate_up` +7.8 % (10.43 -> 11.24 ms, 17.5 -> 16.3 TF/s), `qkvz` +14.5 %,
`qkv` +14.6 %, `out / o` +9.2 %, `down` +17.0 %, MTP `fc` +8.0 % [MEASURED,
`kernel_table.csv`]. The tuned rows recover only the aligned sizes. A chunk of
M = 1024 + t (t <= 48 decode rows) could instead run its first 1024 rows through the
aligned GEMM and the t-row tail through the GEMV. Real-op timings (dequantize + GEMM, or
the GEMV below 49 rows; HIP 4, CUDA-event median, `time` phase), ms:

| slot (layers) | M = 1029 heuristic | M = 1024 heuristic / rows | 5-row GEMV | split, heuristic | split, rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gate_up` (64) | 10.829 | 9.041 / 7.682 | 0.128 | 9.169 (-15.3 %) | 7.809 (-27.9 %) |
| `in_proj_qkvz` (48) | 4.714 | 4.044 / 3.860 | 0.142 | 4.185 (-11.2 %) | 4.001 (-15.1 %) |
| `qkv` (16) | 4.163 | 3.597 / 3.421 | 0.104 | 3.701 (-11.1 %) | 3.524 (-15.4 %) |
| `out / o` (64) | 1.755 | 1.583 / 1.461 | 0.078 | 1.662 (-5.3 %) | 1.539 (-12.3 %) |
| `down` (64) | 4.982 | 4.268 / 4.230 | 0.102 | 4.370 (-12.3 %) | 4.332 (-13.0 %) |

Layer-weighted over the five slots a forward at M = 1029 takes 1417 ms on the heuristic,
1233 ms split (-13.0 %) and 1124 ms split with the rows (-20.7 %) [DERIVED; the split
itself was not implemented or run]. This is the lever for the c=8 cells and the agentic
mix, outside what TunableOp rows can do, and a candidate for a follow-up on
`rdna_hint_flex_linear_rdna2`.

### 3.5 Expected GEMM-time saving per workload

`tunable_report.py --census census.json` weights the measured kernel times by the
scheduler census: the steps with M > 48 of each scoreboard workload at MTP=0 (cell labels
are prompt tokens : output tokens : concurrency : prompts; the times are sums over all
prompts of the workload). Target layers only; an M that was not measured is scaled from
the mean TF/s over the measured keys and counts as no saving [INFERENCE]:

| workload (MTP=0) | steps M > 48 | rows hit (share of GEMM FLOPs) | GEMM time, heuristic | GEMM time, rows | saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| cells 1024:512:1:16 | 16 | 68 % | 21.02 s | 18.98 s | 9.7 % |
| cells 1024:512:8:16 | 15 | 10 % | 23.02 s | 22.65 s | 1.6 % |
| cells 16384:1024:1:8 | 64 | 76 % | 149.30 s | 135.00 s | 9.6 % |
| cells 16384:1024:8:16 | 239 | 1 % | 350.51 s | 348.40 s | 0.6 % |
| cells 32768:2048:1:4 | 52 | 76 % | 128.22 s | 115.98 s | 9.5 % |
| concurrency probe | 12 | 46 % | 20.05 s | 18.90 s | 5.7 % |
| warm repeat 1500 | 3 | 40 % | 2.58 s | 2.44 s | 5.3 % |
| warm repeat 3100 | 2 | 77 % | 3.96 s | 3.57 s | 9.8 % |
| warm repeat 4200 | 4 | 73 % | 5.56 s | 5.06 s | 9.0 % |
| warm repeat 16000 | 10 | 71 % | 21.53 s | 19.63 s | 8.8 % |
| agentic mix | 334 | 0 % | 464.03 s | 462.08 s | 0.4 % |
| prompt logprobs | 9 | 30 % | 13.90 s | 13.37 s | 3.8 % |

With MTP=2 the mixed steps (M = 1033 .. 1045) get only the `ba` rows (section 3.2); the
model above was not run for MTP=2.

## 4. Numerics

### 4.1 Per row (`rowcheck.py`, 89 candidate rows)

Against the fp64 matmul of the same fp16 inputs, on a gaussian and an outlier-channel
activation set, eager and from a HIP graph: **0 rows rejected**, worst error ratio
(tuned / heuristic) 1.00. Max error over max |ref| is 2.1e-4 .. 4.8e-4 and relative L2
2.05e-4 .. 2.09e-4 over both arms and both input sets, the fp16 rounding floor. The tuned
bits **equal the heuristic's for 32 rows and differ for 57**: those 57 change a
rounding, which is why the concurrency probe is the acceptance gate (section 7), not this
check.

### 4.2 Model level, one GPU (`tp2_emul_check.py`)

The real checkpoint at TP=1 on HIP 4 with the two TP=2 ranks' GEMMs issued in place of
each full-width one (section 2, item 7), 3 prompts (1024, 4200 and 3100 tokens of repo
text) x 64 greedy tokens, `--max-num-batched-tokens 2048`, raw fp16 logits, TunableOp off
against the derived profile's rank-0 file:

| arm pair (rows off / on) | tokens | max abs dlogit over the top-20 | chosen-token dlogit max (mean) | calls on a row |
| --- | --- | --- | --- | --- |
| 3 prompts as one batch, 3361-row profile (the 31 stage-one rows) | identical (0 of 3 diverge) | 0.094 / 0.023 / 0.023 | 0.031 (0.0059) / 0.016 (0.0034) / 0.016 (0.0040) | 1440 of 4864 (30 % of FLOPs) |
| the same 3 prompts one at a time, final 3366-row profile | identical (0 of 3 diverge) | 0.043 / 0.031 / 0.023 | 0.031 (0.0044) / 0.031 (0.0039) / 0.016 (0.0026) | 2752 of 4256 (75 % of FLOPs) |

Each cell lists the three prompts in order. The batch runs at M = 1025, 1026, 1128, 2047
and 2048 (the scheduler splits the three prompts across steps), so it exercises the rows
only at 2047 / 2048 and `ba`; one at a time the chunks are M = 103, 1023, 1024 and 2048,
so every M with rows is exercised. A `record` arm over the batched forward lists every
rocBLAS key it issues: **26 keys**, the six hi2 slots x the four M, plus a 16 x 16 x 16
`TN` and `NN` GEMM from the framework's start-up. No GDN, attention, sampler or LM head
call reaches rocBLAS at these shapes, so the six hi2 slots are the whole prefill
population (`down` has no row).

## 5. Wire-up

* `tunableop/rocm7.14-rocblas5.5-hi2-27b-tp2/`: rank files 0-3 (the canonical rows
  byte-for-byte + 36 rows), `provenance.json`, `README.md`, `kernel_table.csv`.
  `tunableop/profiles.json` registers it with the canonical `lib_sha256`, `"auto": false`
  and `"derived_from"`.
* `tools/rdna/recipes/27b-hi2-tp2.env`: `TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2`.
  The helper (`tools/rdna2_028/tunableop_env.sh`) uses a preferred profile only when it is
  registered for the loaded `librocblas.so.5` hash and otherwise warns and auto-selects, so
  a launch never fails on it. `TUNABLEOP_PROFILE` (explicit, validated) still wins;
  `TUNABLEOP_PREFER=` is the rows-off arm. Other recipes do not set it: their rank files
  stay the canonical ones (CPU test).
* Boot log, with the profile of this branch, on `par1-cs25` (`TP=2`) [MEASURED]:

```text
TunableOp profile rocm7.14-rocblas5.5-hi2-27b-tp2 preferred (TUNABLEOP_PREFER) for rocBLAS build f30bb442e9b5 (rows: <tree>/tunableop/rocm7.14-rocblas5.5-hi2-27b-tp2); tuning off.
TunableOp rows per rank (a worker on device ordinal r reads tunableop_results<r>.csv): 0=3366 1=3366
```

  With `TUNABLEOP_PREFER=` it auto-selects `rocm7.14-rocblas5.5` and logs `0=3330 1=3330`.
  The lookup itself was proven on HIP 4: `tunableop_hi2.sh hit` found **36 of 36** new
  keys in the final profile folder (rank 0 file, rows=3366, missing=0).

* Tests (CPU only): `tests/tools/test_tunableop_env.py` (selection logic),
  `tests/tools/rdna/test_hi2_tunableop_profile.py` (registry entry, rank files = canonical
  rows + identical new rows on every rank, no solver changed, only the 27B recipe prefers
  the profile, PRINT output of the launcher), `tests/tools/rdna/test_tunable_hi2.py`
  (the key formulas against the config).

## 6. Provenance

* rocBLAS `5.5.0.cd957402` (`librocblas.so.5` sha256 `f30bb442e9b5...`), torch
  `2.12.0+rocm7.14.0`, ROCm `7.14.60850`, venv `venv-7.14.0_0.32.0`; TunableOp defaults on
  HIP 4: 30 ms and 100 iterations per candidate, 4 MiB rotating buffer [MEASURED].
* Tuned 2026-11-15 on HIP 4 (V620, bus 0xdf, `tree-hi2t`) with
  `PYTORCH_TUNABLEOP_ENABLED=1 TUNING=1 HIPBLASLT_ENABLED=0 TORCH_BLAS_PREFER_HIPBLASLT=0`;
  groups `prefill` (14 keys), `sched0`, `sched1`, `sched2` (25 keys each), in 4 and 5
  shards, one `flock` hold per shard. Curated at a 3 % margin (section 2), checked with
  `rowcheck.py`, harvested with `tunableop_missing.py harvest`. The 36 keys are listed in
  `provenance.json` (`missing_gemms.shape_list`).
* Run data on the box: `~/w4a8_runs/hi2-tunable/{run,abab,s2}/` (shards, measurements,
  rowcheck, curated rows, model-level JSON).

## 7. TP=2 gates (need HIP 2,3)

What the single-GPU checks cannot see: the real rank slices, the 2-rank all-reduce, the
scheduler's batch composition under concurrency, and the MTP drafter inside a TP=2 step.
A tuned row can change one rounding and flip a near-tie token (`tn_2048_12_2560` did on
2026-11-06), so the row set was **not accepted until the gates below passed**; they were
not run on HIP 4 because HIP 2,3 belonged to the scoreboard queue, and they ran there
afterwards (status note at the top).

**Preparation** (on `par1-cs25`). The arms need no model code, only the selection logic,
the recipe line, the rows and the gate-A hook, so a tree that already runs the
integration branch (for example `tree-hi2j`) takes them with one `rsync` from a tree or
checkout of this branch (`tree-hi2t` holds them; run from there):

```bash
cd ~/w4a8_runs/leapquant/tree-hi2t
rsync -a -R tools/rdna2_028/tunableop_env.sh tools/rdna/serve_rdna.sh \
    tools/rdna/rdna_launcher_common.sh tools/rdna/recipes/27b-hi2-tp2.env \
    tools/rdna/rdna_hint_flex/conc_repeat_hook.sh tunableop/profiles.json \
    tunableop/rocm7.14-rocblas5.5-hi2-27b-tp2 ~/w4a8_runs/leapquant/tree-hi2j/
cd ~/w4a8_runs/leapquant/tree-hi2j      # or stay in tree-hi2t
export SERVE_TREE=$PWD OUT_ROOT=$HOME/w4a8_runs/hi2-tunable/tp2 TREE_SHA=$(git -C ~/w4a8_runs/leapquant/tree-hi2t rev-parse HEAD 2>/dev/null || echo feat/hi2-tunableop)
# scoreboard.sh checks that HIP 2,3 are PCI buses d9 dc before it boots anything; a tree
# without .git takes TREE_SHA for the arm's record
```

The recipe default now loads the rows (`TUNABLEOP_PREFER` in `27b-hi2-tp2.env`); the
boot log shows `TunableOp profile rocm7.14-rocblas5.5-hi2-27b-tp2 preferred ...` and
`TunableOp rows per rank ...: 0=<n> 1=<n>`. `TUNABLEOP_PREFER=` (empty) is the rows-off
control of the same tree.

**Gate A: deterministic concurrency probe, rows on vs rows off** (the acceptance test of
`docs/rdna2/tp8-pp2-flashnext-2026-11-06.md`; ten runs with `--ignore-eos` and six with
`--no-ignore-eos` per arm; one boot per arm, about 15 minutes each):

```bash
export QUALITY=0 CELLS=none CONC_REPEAT=10 CONC_REPEAT_NOEOS=6
export LOAD_HOOK=$PWD/tools/rdna/rdna_hint_flex/conc_repeat_hook.sh
bash tools/rdna/rdna_hint_flex/scoreboard.sh conc-on VLLM_RDNA_AR_HIER=0
bash tools/rdna/rdna_hint_flex/scoreboard.sh conc-off VLLM_RDNA_AR_HIER=0 TUNABLEOP_PREFER=
# optional hier pair (TP=2 cannot apply it, see below; it must log the warning and run flat):
#   bash tools/rdna/rdna_hint_flex/scoreboard.sh conc-on-hier VLLM_RDNA_AR_HIER=1
#   bash tools/rdna/rdna_hint_flex/scoreboard.sh conc-off-hier VLLM_RDNA_AR_HIER=1 TUNABLEOP_PREFER=
$HOME/Apps/vllm/venv-7.14.0_0.32.0/bin/python tools/rdna/tp8pp2/conctally.py \
    "$OUT_ROOT" on=conc-on off=conc-off \
    on_noeos=conc-on:'noeos*.txt' off_noeos=conc-off:'noeos*.txt'
```

Accept when rows-on fails no run that rows-off passes (the tally's Fisher test gives the
p-value; one flip in ten runs is not a verdict, greedy output is not bitwise reproducible
across boots), and no check fails in all ten runs. **Flat only:** the hierarchical
all-reduce applies to two equal islands of 2..4 ranks (`rdna_all_reduce.py` docstring,
[CODE-READ]); a TP=2 group is two islands of one rank, so `VLLM_RDNA_AR_HIER=1` cannot
apply and warns. The flat/hier pair of the TP=8 note therefore collapses to the flat arm
here. For an RCCL arm add `P2P=0` to both boots.

**Gate B: scoreboard cells and quality** (BASELINE / FLOOR are the earlier `base` /
`base2` arms under `~/w4a8_runs/scoreboard`; the same-tree control is `tunable-off`):

```bash
unset QUALITY CELLS LOAD_HOOK CONC_REPEAT CONC_REPEAT_NOEOS
export OUT_ROOT=$HOME/w4a8_runs/scoreboard
BASELINE=base FLOOR=base2 bash tools/rdna/rdna_hint_flex/scoreboard.sh tunable \
    TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2
BASELINE=base FLOOR=base2 bash tools/rdna/rdna_hint_flex/scoreboard.sh tunable-off TUNABLEOP_PREFER=
# MTP=2: only the ba rows apply at the mixed-step M; the drafter's dense GEMMs have no rows
BASELINE=base FLOOR=base2 bash tools/rdna/rdna_hint_flex/scoreboard.sh tunable-mtp2 MTP=2
BASELINE=base FLOOR=base2 bash tools/rdna/rdna_hint_flex/scoreboard.sh tunable-mtp2-off MTP=2 TUNABLEOP_PREFER=
```

Read `scoreboard.md` of `tunable` against `tunable-off`: prefill throughput and TTFT per
cell, decode tok/s (must not move), PPL / top-k KL / decode KL against `base` with the
`base2` floor.

### Expected metrics at TP=2 [INFERENCE]

Rows do not touch decode (M <= 48 is the GEMV, the LM head is unchanged), so decode tok/s
and ITL should not move. Prefill: GEMM is about 60 % of the prefill kernel time
(HiPrefillGemm's study), so the 9.5 .. 9.7 % GEMM saving of the c=1 cells is about
**5.7 .. 5.8 %** of prefill time: TTFT of the `base` arm 30.5 s at 16k c=1 -> about
28.8 s, 1.51 s at 1k c=1 -> about 1.42 s; 32k c=1 (75.8 s, attention has a larger share)
at most 5.7 % lower. These use the gains measured on the power-limited HIP 4 (1595 MHz
median sclk); the pilot on a cooler card gave smaller `gate_up` gains (+14.7 % / +10.9 %
against +20.1 % / +15.7 %), and the TP=2 cards run cooler (55-58 C, about 1955 MHz), so
the TTFT gain there may sit below these figures. The c=8 cells (1.6 %, 0.6 % of GEMM
time) and the agentic mix (0.4 %) stay within noise. PPL / top-k KL / decode KL against
`base` should stay inside the `base2` floor: 57 of the 89 rows change a rounding, the
model-level check saw at most 0.094 in a raw logit.

### Gate status

1. Gate A, the deterministic concurrency probe, rows on vs off: **passed** (arms `conc-on`
   and `conc-off`: 0 of 480 checks failing each, 0 of 288 in the `--no-ignore-eos` runs).
2. Gate B, scoreboard cells and quality: arm `tunable` against `base` and `base2` (TTFT and
   output in the status note; PPL 3.6636, prompt KL 7.33e-6, decode KL 6.44e-6 against a
   floor of 2.37e-6). The same-tree `tunable-off` arm was not run: `base` is the reference.
3. Gate B with `MTP=2`: the pair `tunable-mtp2` and `tunable-mtp2-off` was not run;
   `final-mtp2` (the rows and the other switches, MTP=2) has PPL 3.6635, prompt KL 7.73e-6
   and passes the acceptance probe at c=1 and c=4.
4. Rollback is `TUNABLEOP_PREFER=`.

## Files

* `tools/rdna/rdna_hint_flex/tunable_hi2.py`: key formulas (`shapes`), real-op record
  and timing (`record`, `time`), cold-weight graph timing of F.linear keys (`gemm`).
* `tools/rdna/rdna_hint_flex/tunableop_hi2.sh`: the campaign phases (`shapes`, `record`,
  `tune`, `gather`, `measure`, `check`, `curate`, `time`, `hit`, `emul`).
* `tools/rdna/rdna_hint_flex/sched_census.py`: the scheduler M census.
* `tools/rdna/rdna_hint_flex/tunable_report.py`: the kernel table and the expected-gain
  model of this note.
* `tools/rdna/rdna_hint_flex/tp2_emul_check.py`: the single-GPU model-level check.
* `tools/rdna/rdna_hint_flex/conc_repeat_hook.sh`: repeated deterministic concurrency
  probes for the TP=2 gate.
* `tunableop/rocm7.14-rocblas5.5-hi2-27b-tp2/` and `tunableop/profiles.json`: the rows.
* `tools/rdna2_028/tunableop_env.sh`, `tools/rdna/serve_rdna.sh`,
  `tools/rdna/rdna_launcher_common.sh`, `tools/rdna/recipes/27b-hi2-tp2.env`: selection.
* Tests: `tests/tools/test_tunableop_env.py`, `tests/tools/rdna/test_tunable_hi2.py`,
  `tests/tools/rdna/test_hi2_tunableop_profile.py` (CPU only).
