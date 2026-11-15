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
* RESULT_SUMMARY

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
   are skipped. A group is split into 5 shards; each shard is one lock hold of about
   2.5 minutes, so no other job shared the card during a tuning or a measurement and
   the other agents' short jobs waited at most one shard. One tuning serves both TP
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
the **deterministic concurrency probe on the flat and the hierarchical all-reduce**
that decides whether the row set is accepted (section 7).

RESULTS_SECTIONS

## 7. TP=2 gates (need HIP 2,3)

What the single-GPU checks cannot see: the real rank slices, the 2-rank all-reduce, the
scheduler's batch composition under concurrency, and the MTP drafter inside a TP=2 step.
A tuned row can change one rounding and flip a near-tie token (`tn_2048_12_2560` did on
2026-11-06), so the row set is **not accepted until the gates below pass**; they were
not run here because HIP 2,3 belong to the scoreboard queue.

**Preparation** (on `par1-cs25`; `tree-hi2t` already holds this branch's files; any other
tree needs the branch synced, including `tunableop/` and the recipe):

```bash
cd ~/w4a8_runs/leapquant/tree-hi2t
export SERVE_TREE=$PWD OUT_ROOT=$HOME/w4a8_runs/hi2-tunable/tp2 TREE_SHA=<branch sha>
# scoreboard.sh checks that HIP 2,3 are PCI buses d9 dc before it boots anything
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
BASELINE=base FLOOR=base2 bash tools/rdna/rdna_hint_flex/scoreboard.sh tunable
BASELINE=base FLOOR=base2 bash tools/rdna/rdna_hint_flex/scoreboard.sh tunable-off TUNABLEOP_PREFER=
# MTP=2 (needs the sched2 / small rows)
BASELINE=base FLOOR=base2 bash tools/rdna/rdna_hint_flex/scoreboard.sh tunable-mtp2 MTP=2
BASELINE=base FLOOR=base2 bash tools/rdna/rdna_hint_flex/scoreboard.sh tunable-mtp2-off MTP=2 TUNABLEOP_PREFER=
```

Read `scoreboard.md` of `tunable` against `tunable-off`: prefill throughput and TTFT per
cell, decode tok/s (must not move), PPL / top-k KL / decode KL against `base` with the
`base2` floor.

EXPECTED_AND_GATES

## Files

* `tools/rdna/rdna_hint_flex/tunable_hi2.py`: key formulas (`shapes`), real-op record
  and timing (`record`, `time`), cold-weight graph timing of F.linear keys (`gemm`).
* `tools/rdna/rdna_hint_flex/tunableop_hi2.sh`: the campaign phases (`shapes`, `record`,
  `tune`, `gather`, `measure`, `check`, `curate`, `time`, `hit`).
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
