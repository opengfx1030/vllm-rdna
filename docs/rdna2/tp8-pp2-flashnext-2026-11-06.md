# Flash-Next on all 8 V620s of par1-cs25: TP8 and TP4 x PP2 (2026-11-06)

Date: 2026-11-06. Box: par1-cs25 (two PEX880xx switches, roots `c0` = slot A and `00` = slot B, four V620 each).
Model: `Qwen3.8-Flash-Next-AWQ-W4A16`, MTP=2, production compiled mode (`COMPILE_MODE=3`, `FULL_AND_PIECEWISE`,
`MAXBAT=2048`, `SEQS=8`) on every arm. Hardware audit this doc builds on:
[pcie-iommu-audit-2026-11-06.md](pcie-iommu-audit-2026-11-06.md). Run directory on the box:
`~/w4a8_runs/tp8pp2_2026-11-06/` (arms in `arms/<arm>/`). Claims without a measurement are marked [INFERENCE].

## Status

Phase 1 (done, reported): TP8 and TP4 x PP2 boot, serve and pass every smoke in production compiled mode; no crash, hang or
hardware error; one memory waste fixed (`lm_head` on PP stage 0, merged as `251917a5a1`); a silent TunableOp gap found (ranks
4-7 had no rows) and a 593-row campaign run; the all-reduce numbers at world size 8; the memory audit; the measured
comparison (one fresh boot per arm, no repeats).

Phase 2 (2026-11-06 afternoon and evening, closed): the harness is committed (`tools/rdna/tp8pp2/`, see Recipes below); the two
recipes are committed with `KV=13958643712` and passed recipe-native compiled boots with smoke and cells (`PCI_BUSES` resolved to
the right cards, probes 3/3, PPL 4.75-4.85, two boots each). `KV=auto GMEM=0.94` leaves too little at both widths, so the recipes
keep the fixed 13 GiB (peak driver 28.24 GiB with the 16k c8 cells). **Rows: 592 of the 593 new TunableOp rows are on
`rdna_extras`** (data `006916e58f`, tools `f40b8c290f`; 3330 rows per rank on ordinals 0-7). They beat the shipped rows at both
layouts (ABBA, 593 rows: ITL p50 at c8 -3.5 % at TP8 and -10 % at PP2, output +2.5 to +9 %; 592 rows on the hier tree, boots
A A B B: -3.7 % and -4.1 %). The one row left out, `tn_2048_12_2560` (the 12-token decode GEMM), is an enabler of a reproducible
flip in the deterministic concurrency probe on the flat all-reduce (593 rows: 7 of 10 runs on the old tree, 5 of 10 on the hier
tree with hier off; 0 of 10 with hier on, and 0 of 10 without the row on either path), see "Rows" below. The row is exactly as
accurate as the rocBLAS default and rounds differently in 0.3 % of its outputs; the earlier numerics check that passed it
compared the tuned path with itself and is void, a two-process gate replaces it. Hierarchical all-reduce (`dc35df271d`) is on by
default at TP8: ITL p50 at c8 -11 % (final cells below). Not run: 2 x TP4 replicas, PP partitions 26,22 / 28,20, the PLE sidecar
under TP8 or PP2, a 592-row re-measure at PP2.

## Summary

| Question | Verdict |
|---|---|
| TP8 (HIP 1,2,3,4,6,7,8,9) boots and serves in production mode | Yes. READY 190-291 s, probes 3/3, PPL 4.84, concurrency PASS, 0 dmesg/KFD/AER/IOMMU lines. |
| TP4 x PP2 (stage 0 = slot A, stage 1 = slot B) | Yes, no code fix needed to run. MTP2, QSA (both stages), PLE (stage 0), `rdna_ar` (TP, EP and PP groups) all work. |
| Crashes / hangs / faults | None. One waste bug (below), one silent tuning gap, harness-only bugs. |
| `rdna_ar` at world size 8 | Active (handles 0 and 1 = TP and EP groups, devices 0-7). One-shot beats RCCL up to ~155 KiB; RCCL graph costs 134-178 us per call up to 240 KiB. |
| Memory | TP8 weights 11.23 GiB/GPU, 12.5 GiB free after a 3.75 GiB KV boot. PP2 fits 1.86x the tokens per GiB of KV (7 KiB vs 13 KiB per token and GPU) [INFERENCE for the per-token split]. |
| Fastest layout | Prefill-heavy (16k): PP2, TTFT 3.40 s vs 5.52 s TP4 (-38 %), 16k c8 output 161 vs 116 tok/s (+39 %). Decode latency: TP8, step 19.6 ms vs 22.4 ms TP4 (-12 %). TP8 prefill is only -4 % vs TP4. |
| TunableOp | Ranks 4-7 had no rows (TP8 and PP2 stage 1 ran untuned, silently). 592 of 593 new rows ship (3330 per rank, hit proof 692/693: the miss is the excluded `tn_2048_12_2560`): c8 ITL p50 -3.4 % (ABBA) and -3.7 / -4.1 % (hier tree) at TP8, -10 % at PP2. With the 593rd row the deterministic concurrency probe flips one token reproducibly on the flat all-reduce (7/10 and 5/10 runs); that row is as accurate as the rocBLAS default, so the cause is not a defective solution [INFERENCE: a rounding pattern]. |

## Method

* GPUs: slot A = HIP 1-4 = V620 buses `ce d3 d8 db`; slot B = HIP 6-9 = buses `0a 0f 14 17`. HIP 0 and HIP 5 are the W7800s
  and are never used. `rocm-smi` numbers GPUs in KFD order, so every launch resolved HIP ids from the PCI bus first
  (`slots.sh verify_slot`: bus, name and 30 GB per GPU).
* TP8 = `TP=8` with expert parallel (EP8), ranks in HIP order `1 2 3 4 6 7 8 9`. TP4 x PP2 = `TP=4 PP=2`, stage 0 = slot A,
  stage 1 = slot B, `PP_PARTITION=24,24`. Single-slot TP4 = the shipped `flashnext-mtp2` recipe on slot A.
* Every arm is a fresh boot of the `flashnext-mtp2` recipe with the TP/PP overrides, from an exact-SHA tree (`git archive`
  plus the matching `_rocm_C` build, fresh `VLLM_CACHE_ROOT` per configuration, no `rdna_ar_wedged` marker), fixed KV
  `4026531840` bytes (3.75 GiB), `GPU_MAX_HW_QUEUES=4`, one-shot gate 256 KiB (the recipe value). Tree `608d5df287` is the
  base; tree `2697797c78` adds the `lm_head` fix below. The fix is merged into `rdna_extras` as `251917a5a1` (cherry-picked onto
  `1ce3de8045`). Branch `fix/tp8-pp2-flashnext` was rebased onto `1ce3de8045` after the runs; the SHAs in this doc are the
  pre-rebase ones the trees were built from (`2697797c78` = `d57b8ca745`, `23f1d01305` = `ae6c68ffd6`, `1ea03e6456` =
  `fc0442cb53`, `a1721065db` = `70ce829213` on the branch).
* Smoke per arm: three probes (Paris / 2 / Berlin), PPL probe (reference 4.78-4.88), concurrency probe, then four measured
  cells (1k/512 and 16k/1k, c1 and c8; 4 requests at c1, 32 at c8; fixed seeds, greedy), then a long-context quality collect
  (6 passages, 21,874 tokens). A cell with a KFD `evicted_ms` jump would be invalid and rerun: none occurred. Junction
  temperature is logged at 1 Hz (abort rule 90 C mean / 95 C peak: never reached, peaks 76-83 C).
* Every arm: `dmesg` and BMC SEL diffed before and after (0 new lines of any kind in every arm of this doc; the SEL is 100 %
  full and cannot log new events, see Open items). Raw logs: `arms/<arm>/{serve.log,summary.txt,concurrency.txt,cells/,
  cells.tsv,quality.json,torchmem/,temps1s.log,dmesg_summary.txt}`.
* Kernel: every arm and gate run in this doc ran in a single kernel boot (`7.1.7-100.p2p_post7.fc43`, booted 2026-11-05 14:04,
  `iommu=pt`, `amd_iommu=on` ignored by the kernel, no `pcie_ports`). The PCIe audit changed the command line afterwards
  (`amd_iommu=on` dropped, `pcie_ports=native` added; boots from 2026-11-07 01:43). The cells were not re-measured on it;
  PcieAudit reports the TP8 smoke unchanged.
* Phase 1 numbers are one boot per arm, without spread; phase 2 (recipe trees) runs ABBA pairs and a difference counts only when
  it is larger than the spread inside both pairs (`compare.py`). MTP acceptance length varies by up to 0.5 between boots on the
  same prompts (4 requests per c1 cell), which moves output tok/s; ITL p50 (time per verify step) and TTFT are the robust columns.

## Bring-up evidence

| Arm | Tree | Config | READY | Probes | PPL | Concurrency | PPL_LONG | Notes |
|---|---|---|---:|---|---:|---|---:|---|
| `tp8-a` | 608d5df287 | TP8, RECORD_UNTUNED census | 291 s | 3/3 | 4.84 | PASS | - | census only (rows disabled) |
| `tp8-b` | 2697797c78 | TP8, shipped rows | 290 s | 3/3 | 4.84 | PASS | 2.8386 | temps 78-83 C |
| `tp8-census` | 2697797c78 | TP8, RECORD_UNTUNED + cells | 190 s | 3/3 | 4.84 | PASS | - | S0 reference |
| `tp8-c` | 2697797c78 | TP8, new rows (S2) | 190 s | 3/3 | 4.81 | **FAIL** | 2.8475 | see TunableOp |
| `pp2-a` | 608d5df287 | TP4 x PP2 | 270 s | 3/3 | 4.84 | PASS (8 near-tie warns) | 2.8422 | cells lost to a harness bug |
| `pp2-b` | 2697797c78 | TP4 x PP2 | 261 s | 3/3 | 4.82 | PASS | 2.8530 | temps 76-81 C |
| `tp4a-b` | 2697797c78 | TP4 slot A | 301 s | 3/3 | 4.84 | PASS | 2.8586 | temps 79-81 C |
| `kvauto-tp8` | 2697797c78 | TP8, `KV=auto GMEM=0.94` | 190 s | 3/3 | pass | PASS | - | 14.96 GiB KV, 1.3 GiB free: too tight |
| `kvauto-pp2` | 2697797c78 | TP4 x PP2, `KV=auto GMEM=0.94` | 190 s | 3/3 | pass | PASS | - | stage 1 0.74 GiB free: too tight |
| `rtp8-1` | d2eea35872 | recipe `flashnext-tp8-mtp2`, `PCI_BUSES` | 310 s | 3/3 | 4.85 | PASS; repeat 3/3 | collected | cards resolved from the buses, matched the slot |
| `rpp2-1` | d2eea35872 | recipe `flashnext-tp4pp2-mtp2` | 271 s | 3/3 | 4.79 | PASS; repeat 2/3 | collected | the one FAIL is the haiku prompt past the stop token (below) |

PP2 rank layout: workers `Worker_PP{0,1}_TP{0..3}_EP{0..3}`; the log reports "rank 0 in world size 8 is assigned as DP rank 0,
PP rank 0, TP rank 0, EP rank 0". Feature matrix under PP2:

| Feature | Result |
|---|---|
| MTP=2 | Works; acceptance length 2.04-2.40 in the cells (TP4 1.84-2.70, TP8 1.97-2.51). The drafter lives on stage 1. |
| QSA | Works on both stages (`qsa_rdna2: indexer scoring/epilogue (HIP)` lines from PP0 and PP1 workers). |
| PLE (bf16, pinned) | Only stage-0 workers pin the table: 23.84 GiB exact, mlocked, x4. |
| PLE sidecar (`PLE=int4-disk`) | Not booted. `gpu_worker.py:737-740` connects only the first stage [INFERENCE: should work]. |
| `rdna_ar` | One-shot active on the TP and EP groups of 4 (devices [0-3] and [4-7]) and on the PP pair groups ([0,4] [1,5] [2,6] [3,7]); backends `['RDNA_ONESHOT','PYNCCL']`. |
| Refusals | None. |

## Crashes, hangs, faults and fixes

No crash, hang, amdgpu fault, KFD eviction, AER/GHES line, SERR or IOMMU fault in any compiled boot, in the all-reduce
microbench windows or in the TunableOp shards. Findings, all fixed or recorded:

| # | Finding | Root cause (file:line) | Fix / status |
|---|---|---|---|
| 1 | PP stage 0 loads and keeps the unused target `lm_head` (0.30 GiB per GPU; 12 "Attempted to load weight lm_head.weight" warnings in the pp2-a log) | `vllm/models/qwen4_exp/amd/model.py:797` built `ParallelLMHead` on every stage | `2697797c78`: only the last stage builds it, others get `PPMissingLayer` (import at `:65`). Safe: the drafter only shares the head by assignment (`llm_base_proposer.py:1570-1588`), `AutoWeightsLoader` skips `PPMissingLayer`, `compute_logits` runs on the last rank. Proof: `pp2-b` (compiled; stage 0 weights 10.09 -> 9.79 GiB, no `lm_head` load on PP0, probes, concurrency, cells, quality pass) and non-PP boots `tp8-b`, `tp4a-b` on the same tree. The NVIDIA variant (`vllm/models/qwen4_exp/nvidia/model.py:730`) has the same unconditional head and is untouched. |
| 2 | TP8 and PP2 stage 1 run every GEMM untuned, silently | `tools/rdna2_028/tunableop_env.sh:110` checks rank files 0-3 only; the profile ships `tunableop_results0..3.csv`; torch opens `tunableop_results<ordinal>.csv`, so ordinals 4-7 print `could not open` and load 0 rows. A new rank file would also have been dropped by git: `.gitignore:216` ignores `*.csv`. | `1ea03e6456`: warning for ranks 4..TP*PP-1 without rows, `TUNABLEOP_RANKS=TP*PP` from the launcher, `tunableop_missing.py extend`. `.gitignore` exception for `tunableop/**/*.csv`. Rows: see TunableOp. |
| 3 | `tune_missing_gemms.sh verify` checked nothing | Its script set `PYTORCH_TUNABLEOP_RECORD_UNTUNED=1`, which makes torch 2.12 skip the lookup (every GEMM is recorded), and read `untuned_verify.csv` while torch writes `untuned_verify0.csv`: it always printed PASS (full 286-key census against a table missing 241 of them: PASS) | `a1721065db`: loads each rank file in its own process, lookup only, compares census keys. Controls: 40 in-table keys PASS (40/40 on ranks 0-3); full census FAIL (45 hit, 237 missing). |
| 4 | Harness only: the memory sampler's `PYTHONPATH` leaked into the bench client | `sitecustomize` imports torch; the client died with `LLVM ERROR: ... 'spirv-expand-step' registered more than once` | `hook8.sh` unsets `PYTHONPATH` for the client (box-local harness). |
| 5 | `benchmark_rdna_allreduce.py` needs `amdsmi` | the venv has none | `23f1d01305`: bench calls `set_custom_all_reduce(current_platform.use_custom_allreduce())`. `rdna_ar: PIX topology query failed (amdsmi_init)` stays a log-only warning. |

## All-reduce at world size 8 (decode and prefill sizes)

`rdna_ar` (`VLLM_RDNA_AR=1`, one-shot) is active at world size 8: `rdna_ar: one-shot all-reduce active (handle 0 and 1, rank
r/8, devices [0..7], up to 256 KB, RCCL above)` on every worker (TP and EP groups), the self-test passes and sums are exact
in the microbench. Microbench: `benchmarks/kernels/benchmark_rdna_allreduce.py` on the shipped venv, hidden 2560, fp16,
graph replay and eager, one process per rank (`ar/tables.md` in the run directory).

Per call in us at TP8 (HIP 1,2,3,4,6,7,8,9), CUDA-graph replay:

| tokens x 2560 | KiB | RCCL | one-shot | note |
|---|---:|---:|---:|---|
| 1 | 5 | 163 | 29 (auto 20) | decode, c1 without MTP |
| 3 | 15 | 164 | 33 | MTP2 verify, 1 seq (c1) |
| 6 | 30 | 164 | 46 | 2 seqs |
| 12 | 60 | 178 | 62 | 4 seqs |
| 24 | 120 | 134 | 107 | 8 seqs (largest decode graph) |
| 32 | 160 | 142 | 136 | crossover |
| 48 | 240 | 144 | 196 | RCCL wins |
| 1024 | 5120 | 391 | - | 2048-token prefill chunk (23.5 GB/s bus bandwidth) |
| 2048 | 10240 | 724 | - | |

* The one-shot crossover is 150-160 KiB (slope 0.72 us/KiB). The decode graphs of `SEQS=8` stop at 24 tokens = 120 KiB, so
  every decode all-reduce stays below it; a gate of 128 KiB is the TP8 setting the data supports (the TP8 arms here ran the
  recipe's 256 KiB; the serving effect of 128 is not measured). TP4 keeps 256: its one-shot curve is flatter (TP4: 120 KiB
  43 us, 240 KiB 67 us, RCCL 65 / 74 us).
* RCCL at TP8 costs 134-178 us per call for anything up to 240 KiB, 2.6x the TP4 RCCL cost (65-75 us) and 4-5x the one-shot
  cost at decode sizes. A 5 MiB prefill all-reduce takes 391 us at TP8 and 319 us at TP4 (23.5 vs 24.6 GB/s bus bandwidth:
  the ring is bound by one PCIe x16 link per hop, so TP8 does not speed up prefill all-reduces; this is why TP8 prefill
  gains only 4 % over TP4).
* TP4 with two GPUs on each root (HIP 1,2,6,7) is identical to four GPUs on one root (HIP 1,2,3,4) at every size: the
  cross-root hop costs nothing at 4 ranks.
* RCCL 2.30.4 (`NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,TUNING`): TP8 builds one ring in HIP order
  `0 1 2 3 4 5 6 7` (Pattern 4, bandwidth 24/24 GB/s, 2 channels, two cross-root hops, every link `P2P/direct pointer`) and a
  tree `0 -> 1`. AllReduce picks RING with LL up to about 30 KiB and SIMPLE from 60 KiB; Tree and LL128 are never chosen.
  `NCCL_P2P_LEVEL` pxb, sys and phb give the same rings and timings (PcieAudit), so no RCCL variable is needed.
* TP4 x PP2 (`arms/pp2-a/nccl.*.log`): five communicators per rank: two 4-rank (TP and EP; ring `0 1 2 3`, type PXB/PIX,
  2 channels) and three 2-rank (the PP pairs; ring `0 1`, type PHB/PIX, 2 channels).
* The hierarchical all-reduce (HierAr, `rdna_extras` `dc35df271d`) is on by default at TP8 from 20 KiB up to 1 MiB; its per-call
  numbers at TP8 (hier vs RCCL: 64 KiB 37 vs 178 us, 128 KiB 36 vs 136, 256 KiB 50 vs 146, 512 KiB 77 vs 166, 1 MiB 144 vs 171,
  tie at 1.5 MiB) are in [rdna-ar-hier-2026-11-06.md](rdna-ar-hier-2026-11-06.md). Measured in the recipe (section "Final cells on
  the hier tree"): ITL p50 at c8 -11 % (41.81 to 37.17 ms), c1 unchanged. An MTP2 c8 step issues about 100 all-reduces at 120 KiB
  [INFERENCE: not counted], which makes the 4.6 ms about 46 us per call.

## TunableOp

How rows are chosen: `configure_tunableop` selects profile `rocm7.14-rocblas5.5` from the sha256 of the loaded
`librocblas.so.5` and sets `PYTORCH_TUNABLEOP_FILENAME=<dir>/tunableop_results.csv`; torch inserts the worker's device
ordinal (`tunableop_results<ordinal>.csv`). Lookup is exact-shape.

Findings (proven on the box with `tunetest.sh` / `tunetest2.py`, one process per ordinal):

* The shipped profile has rank files 0-3 only. Ordinals 0-3 load 2738 rows; ordinals 4-7 print `could not open ...
  tunableop_results4.csv` and load 0 rows. A view directory with `tunableop_results{4..7}.csv` symlinked to `{0..3}` loads
  2738 rows on 4-7. TP8 and the PP2 stage-1 workers (ordinals 4-7) therefore ran the rocBLAS default for every GEMM.
* With `PYTORCH_TUNABLEOP_RECORD_UNTUNED=1` torch 2.12 reads no rows at all (rows_loaded = 0 on every ordinal) and records
  every GEMM it runs, not only the misses. A census arm is thus always an untuned arm and never a measured one. Census
  files are named `untuned<ordinal>.csv`.
* TP8 census (`arms/tp8-a` boot + probes: 286 keys; `arms/tp8-census` boot + probes + the four cells: 640 keys per ordinal;
  union 697 unique): the shipped rank-0 table hits 100; 597 are missing (593 tunable, 2 noise 16x16x16, 2 strided-batched fp32).
  The missing keys are dominated by the token-count dimension: 433 `TN K=2560` keys with 141 distinct token counts (9 to
  8192; weight dimension 12, 160, 320, 512, 640, 2048, 12800 or 31040), 59 + 59 `TN K=768` / `K=80` keys (59 token counts
  each), 36 `NN K=2560 M=1` keys (data-dependent lengths). Prefill steps run eagerly (the graph sizes are only 3, 6, 12, 24),
  so every mixed prefill + decode step produces another token count: a fresh run will not hit 100 % even after tuning
  [INFERENCE: 20 % of the boot-only keys of `tp8-a` are absent from the later census].
* The PP2 stage shapes are the TP4 shapes; a PP2 census arm was not run, so the stage-1 hit rate against ranks 0-3 is not
  measured [INFERENCE: covered].

Campaign (`tunable/` in the run directory; `camp8.sh census|tune|harvest|hit`): 593 shapes tuned in 3.5 minutes on 8 GPUs
(one shard of 74-75 keys per GPU, `tune_missing_gemms.sh tune --shard i/8`), harvested into rank files 0-7 (ranks 4-7 start
as copies of ranks 0-3): **3331 rows per rank** (2738 + 593), of which 3330 ship (`tn_2048_12_2560` is left out, see "Rows"
below). Lookup-hit proof (`tunehit.py`, torch loads each rank file for its ordinal): the 3331-row files hit `census_keys=693
hit=693 missing=0` on ordinals 0-7; the shipped 3330-row files hit 692 and miss 1 on all eight ordinals (`tunable/hit592.txt`; the
miss is the excluded key). Numerics: see "The numerics of the culprit row, and the gate that missed it" below; the first check
(`rowcheck.py` v1: 557 `TN` keys, "none worse than 3x the default's error", `tunable/rowcheck_rank0.txt`) compared the tuned path
with itself and is void.

Measured (fix tree, fixed KV, single boot per arm; S0 = no rows, S1 = shipped rows = ranks 0-3 only, S2 = new rows 0-7):

| Cell | Metric | S0 `tp8-census` | S1 `tp8-b` | S2 `tp8-c` | S2 vs S1 |
|---|---|---:|---:|---:|---:|
| 1k c1 | ITL p50 ms | 19.54 | 19.59 | 19.53 | -0.3 % |
| 1k c8 | ITL p50 ms | 41.66 | 41.56 | 40.19 | -3.3 % |
| 16k c1 | TTFT s | 5.28 | 5.28 | 5.17 | -2.1 % |
| 16k c8 | ITL p50 ms | 43.34 | 43.55 | 42.23 | -3.0 % |
| 1k c8 | TTFT mean s | 0.80 | 0.81 | 0.77 | -5.5 % |
| 16k c8 | TTFT mean s | 11.96 | 12.49 | 11.73 | -6.1 % |
| 16k c8 | prefill aggregate tok/s | 2938 | 2937 | 3042 | +3.6 % |

S1 equals S0 within 1 %: the shipped TP4 rows do nothing at TP8 while ranks 4-7 are untuned (the slowest rank sets the
step). S2 is a 2-6 % improvement in one pair of boots, which is inside the spread seen between TP4 boots of different days
[INFERENCE], so it is not shown to be a win. **The S2 arm failed the concurrency probe:** `FAIL together 1: token 25/64,
margin 2.375` (`arms/tp8-c/concurrency.txt`): the solo run ends its haiku with the stop token, the concurrent run continues
with `user\nWrite a haiku about the ocean ...` (the 8-request batch picked a different special token at the end of the turn).
All other prompts matched or were near-ties; PPL probe 4.81 and long PPL 2.8475 (S1 2.8386, TP4 2.8586) are normal. The
`solo` vs `solo2` pass of the same server already differs in 5 of 8 prompts at S2 (3 of 8 at S1), so run-to-run noise exists;
whether a margin of 2.4 is that noise (stop-token logits sit on rank 4's vocab shard, where ordinal 4 was untuned in S1) or a
real effect of a tuned solution is **open**. Next step for whoever continues: boot S1 and S2 once each and run the probe 10x
against each server, then bisect the 593 rows by shard if S2 fails more often than S1.

Repeat test (phase 2; harness `hook.sh` with `CONC_REPEAT=10 CONC_REPEAT_NOEOS=6`, tree `2697797c78`, one boot per state,
`conctally.py`; the probe's own pass criterion decides, default run = `--ignore-eos`, 64 tokens each):

| State | Rows | Boot | Default probe: failing runs | failing checks | `--no-ignore-eos`: failing runs |
|---|---|---|---:|---:|---:|
| S1 | shipped (ranks 0-3) | `tp8-s1r` | 0 / 10 | 0 / 480 | 0 / 6 |
| S1b | shipped rows copied to ranks 4-7 | `tp8-s1bp` | 0 / 10 | 0 / 480 | 0 / 6 |
| S2 | 593 new rows, ranks 0-7 | `tp8-s2p` | 1 / 10 | 1 / 480 | 0 / 6 |

Fisher exact, failing runs S2 vs S1: p = 1.0 (1/10 vs 0/10); with the `tp8-c` / `tp8-b` single runs added the counts are 2/11
vs 0/11 (p = 0.48 [INFERENCE: not computed on the box]). No evidence that the rows change the failure rate. The one failing
check (`tp8-s2p` run 8, `churn` prompt 1, token 29/64, margin 2.437, continuation mean lp -0.30 vs solo -0.28) is the same
prompt and regime as `tp8-c`: the haiku ends and the run is forced past the stop token, where one run goes on with
`assistant<think>...` and the other stops at a newline. [INFERENCE: this is the noise of the post-stop-token regime; no run
fails when requests stop where serving stops (`--no-ignore-eos`, 18 runs).] The first harness arm (`tp8-s1p`) is void: the
repeat posted to the base URL instead of `/v1/completions` (404 in all 10 runs); fixed in `0fa6bd1c6e`, rerun as `tp8-s1r`.

All default-probe runs (the boot probe plus the repeats, `--ignore-eos`) over every arm of this doc before the long samples: S1
(shipped rows) 1 failing of 37 runs (`rpp2-1`), S1b 0 of 11, S2 (new rows) 3 of 28 (`tp8-c`, `tp8-s2p` run 8, the boot probe of
`m-pp2-b1`); S2 against S1 + S1b: 3/28 against 1/48, Fisher two-sided p = 0.139. Long samples, TP8, one boot each, 40 runs of
the default probe (`tp8-s1l`, `tp8-s2l`): S1 0 of 40 failing runs (0 / 1920 checks), S2 2 of 40 (2 / 1920; `run18` haiku token 29,
margin 2.09; `run33` prompt 0 token 28, margin 1.03, continuation mean lp -0.13 against solo -0.15), Fisher p = 0.494. Pooled over
every arm: S2 5 of 68 runs (7.4 %), S1 + S1b 1 of 88 (1.1 %), p = 0.086. Every failing check is a short answer continued past its
stop token (haiku prompt tokens 25-29, margin 1.5-2.6, continuation mean lp -0.04 to -0.39 against solo -0.16 to -0.32), also with
the shipped rows (`rpp2-1`); the runs that stop where serving stops (`--no-ignore-eos`) never fail (18 of 18). The default-MoE
probe could not separate the states (its noise comes from the order-dependent fp16 atomics of the default MoE kernel); the 40-run
no-EOS samples, the deterministic-MoE pair, the bisect and the culprit row follow in the next section, and the final verdict
(592 of the 593 rows) is in the Status paragraph.

## Rows: no-EOS samples, deterministic MoE, bisect, the culprit row and the numerics gate

**No-EOS samples (default MoE, tree `2697797c78`, `CFG=tp8`, one boot per state, 40 runs each).** `tp8-s1n` (shipped rows) 0 of 40
failing runs, `tp8-s2n` (593 rows) 0 of 40, Fisher p = 1.0; with the 18 earlier no-EOS runs both states have 0 failing runs of 58.
Runs that stop where serving stops never fail, so the default-MoE probe cannot tell the states apart in the regime that matters.

**Deterministic MoE.** `VLLM_RDNA2_MOE_DETERMINISTIC=1` makes the routed MoE run-to-run bit-identical (Q32 fixed-point
accumulation, fixed-order top-k, deterministic radix top-k ties; `vllm/envs.py`), so the default MoE kernel's order-dependent fp16
atomics no longer cover a difference between the solo run and the batched run: what remains is batch-shape-dependent kernels,
tuned GEMMs included. Same tree and config, one boot per state, 10 default-probe runs and 6 no-EOS runs each (`conctally.py`):

| State | Arm | Default probe: failing runs | `--no-ignore-eos`: failing runs | near-tie flips tolerated (of 480) |
|---|---|---:|---:|---:|
| S1D shipped rows | `tp8-s1d` | 0 / 10 | 0 / 6 | 46 |
| S2D 593 rows | `tp8-s2d` | **7 / 10** | **3 / 6** | 82 |
| NCL 592 rows (S2D without `tn_2048_12_2560`) | `bis-ncl` | 0 / 10 | 0 / 6 | 94 |

Fisher on the default probe: S1D vs S2D p = 0.003 and S2D vs NCL p = 0.003; no-EOS 0/6 against 3/6 p = 0.18 (six runs cannot
separate). Every S2D failure is the same check, `FAIL together 0 prompt 0: token 28/64: margin 1.234, continuation mean lp -0.14 vs
solo -0.14`: the solo run continues the answer with " Lisbon", the batched run emits "\n\n<think>" at that token. It is a flip
inside an answer (not past a stop token) with a 1.2-nat margin, far above fp16 rounding, the same token in all ten failing runs:
with the noise removed the 593-row set fails reproducibly where the shipped rows and the 592-row set do not. (Near-tie flips of at
most 0.2 nats are tolerated by the probe; NCL has the most of them and no failure, so they are not the signal.)

**Bisect (deterministic MoE, 4 repeats per arm plus the boot probe; `bisectrows.py` builds the row trees `tunable/bis/<arm>`).** Z =
the shipped 2738 rows copied to all eight ranks. The new rows are 36 `NN` (M = 1), 59 `TN` with n <= 64, 124 `TN` with 65 <= n <=
256 and 374 `TN` with n > 256.

| Stage | Arm | Rows per rank | Failing runs of 4 |
|---|---|---|---:|
| 1 | `bis-z` | Z (2738) | 0 |
| 1 | `bis-x1` | Z + 36 `NN` + the 124 `TN` 65-256 (2898) | 0 |
| 1 | `bis-x2` | Z + the 59 `TN` n <= 64 + the 124 `TN` 65-256 (2921) | 3 |
| 2 | `bis-s0` .. `bis-s5` | Z + the 124 `TN` 65-256 + the 59 `TN` n <= 64 rows (sorted by key) whose index has bit b set (2889-2891) | s1 4, s4 3, s0 s2 s3 s5 0 |
| 3 | `bis-cu` | Z + row 18 alone (2739) | 0 |
| 3 | `bis-nc` | all 593 minus row 18 (3330) | 0 |

Stage 2 decodes to index 18 (bits 1 and 4 and no other: the only row that is in both failing arms s1 and s4 and in none of the
four passing ones):
`GemmTunableOp_Half_TN,tn_2048_12_2560_ld_2560_2560_2048,Gemm_Rocblas_-1086326157,0.0518068`: a 2048 x 2560 weight against 12
activation rows (4 requests x 3 MTP2 tokens, the captured graph size 12); the shipped rows have no row for it. The single-defect
assumption was wrong: row 18 alone passes (`bis-cu`), every new row except row 18 passes (`bis-nc`, `bis-ncl`), and X1 (the 124
`TN` 65-256 rows without row 18) passes. The failure needs row 18 together with other new rows: with the 124 `TN` 65-256 rows and
half (`bis-s1`, `bis-s4`) or all (`bis-x2`) of the 59 `TN` n <= 64 rows it fails; a smaller combination was not booted.

**The same arms on the hier tree (`38ab87ed3a`, deterministic MoE, `CFG=tp8`, 10 default-probe and 6 no-EOS runs each).** This tree
differs from `2697797c78` only in the hierarchical all-reduce, which sums island partials for 20 KiB to 1 MiB (the batched 60 KiB
step, not the 15 KiB solo step) and so changes the summation order of the batched step; `VLLM_RDNA_AR_HIER=0` gives the flat
one-shot (no `hier islands` line in that arm's `serve.log`).

| Arm | Rows | All-reduce | Default probe: failing runs | `--no-ignore-eos`: failing runs |
|---|---|---|---:|---:|
| `h-s1d` | shipped | hier | 0 / 10 | 0 / 6 |
| `h-s2d` | 593 | hier | 0 / 10 | 0 / 6 |
| `h-s2d-flat` | 593 | flat (`VLLM_RDNA_AR_HIER=0`) | **5 / 10** | **3 / 6** |
| `h-ncd` | 592 (the shipped set) | hier | 0 / 10 | 0 / 6 |

`h-s2d-flat` against `h-s2d`: Fisher p = 0.033 (default probe) and p = 0.036 (no-EOS, 3 of 6 against 0 of 6). All eight failing
checks of `h-s2d-flat` are the check that fails in `tp8-s2d`, `bis-x2`, `bis-s1` and `bis-s4`: `together 0 prompt 0: token 28/64:
margin 1.234`, the same token and the same margin to three decimals in five arms with different row sets. So the 593-row flip is not
a property of the old tree: it returns on the new tree as soon as the all-reduce is flat, and the hier all-reduce hides it as
leaving row 18 out does. The 592-row set and the shipped rows pass on both all-reduce paths (`bis-ncl` and `h-ncd`, `tp8-s1d` and
`h-s1d`). The all-reduce abort and late-peer lines are absent from every failing arm's log, and an abort fails the engine step.

**The numerics of the culprit row, and the gate that missed it.** The first numerics check of this campaign was void.
`rowcheck.py` v1 (and a first `rowprobe.py`) took the "default" result with `tunable.enable(False)` in the process that had loaded
the rows (by a first GEMM). In torch 2.12 on this box that does not bypass the loaded rows: digests of the `F.linear` output of the
557 new `TN` keys in three processes (TunableOp never enabled / rows loaded / rows loaded, then `enable(False)`) show the rows
changing the bits of 393 keys against the process that never enabled TunableOp, and all 557 keys after the toggle equal the tuned
result. So "tuned against default, error ratio 1.00, none worse than 3x" (557 keys, this row included) compared the tuned path
with itself, and the first probe's "tuned equals default bit for bit" for row 18 was the same artifact. The corrected gate
(`rowcheck.py` v3, `rowcheck3/` in the run directory) runs two processes on one V620: one never enables TunableOp (the rocBLAS
heuristic), one loads the rows (the loaded rows and the looked-up solution are checked); inputs are seeded by the key; two
activation sets (gaussian, and gaussian with four channels at 60x); the reference is the fp64 matmul of the same fp16 inputs; eager
and HIP graph.

| | 593-row file | 592-row file (shipped) |
|---|---:|---:|
| `TN` keys checked | 557 | 556 |
| tuned bits equal to the default's | 165 | 165 |
| tuned bits differ from the default's | 392 | 391 |
| max-error ratio tuned / default (1114 key-set pairs) | 0.991 to 1.005, median 1.000 | same keys |
| L2-error ratio | 1.000 for every pair | same keys |
| graph replay equal to eager; three eager repeats identical | all | all |
| rejected (ratio above 1.5) | 0 | 0 |

Row 18, `tn_2048_12_2560` (solution `Gemm_Rocblas_-1086326157`, 2048 x 2560 weight, 12 activation rows): error against fp64 (max
over max |ref|; L2) with gaussian activations 3.683e-04; 2.073e-04 for the default and for the row alike, with the outlier set
2.692e-04; 2.080e-04 for both. Against the default's own output (a dump of both processes, `rowcheck3/row18_dump.py` and
`row18_cmp.py`, not committed): 73 of 24576 elements differ on the gaussian set (0.30 %, largest absolute difference 1.95e-03 at an
output scale of 10.6) and 58 on the outlier set (0.24 %, 7.8e-03 at a scale of 28.9), and the largest absolute error against fp64
is identical for the default and the row (3.892e-03; 7.785e-03). The row rounds differently in about 0.3 % of its outputs and is
exactly as accurate as the default. Its result is bit-identical over 20 eager runs, in a HIP graph, and after 450 other tuned
shapes ran in between (a one-off probe on the tuned path only, valid despite the toggle problem; the tool is not kept).

Should the campaign's gate have caught it? Not by accuracy: row 18 is at ratio 1.00 like the other 556 rows (0.991 to 1.005), so
no accuracy threshold separates it from them. A bit-exact gate would reject it, together with 391 other rows
(70 %), and the tuner's own check shows the price: `tune_missing_gemms.sh tune` on four shapes (`rowcheck3/smoke`): at
`TUNE_ATOL=TUNE_RTOL=2e-3` the winners are the fastest solvers (`tn_160_1042_2560` -1086326128 at 0.0751 ms), at 1e-9 (bit-exact
to the default) the tuner takes slower ones (-1086326053 at 0.0973 ms, +29 %; `tn_12800_1037_2560` 2.643 to 2.842 ms, +7.5 %).
A first test on eight shapes with four settings (no check, 1e-2, 2e-3, 1e-9) ran without errors. The failure was caught by one
thing: the deterministic concurrency probe (solo against four requests), and only on the flat all-reduce.

What was changed: `tools/rdna/tp8pp2/rowcheck.py` is the two-process gate above (rejects a key whose max or L2 error exceeds 1.5x
the default's, with no absolute slack, whose graph replay differs from its eager result, or whose eager repeats differ;
`--reject-out FILE` writes the `op,key` lines); `camp.sh check` runs it between `tune` and `harvest`;
`tunableop_missing.py harvest --reject FILE` leaves the listed keys out (unit test); `tune_missing_gemms.sh tune` runs the tuner
with a numerical check (`TUNE_ATOL`, `TUNE_RTOL`, default 2e-3; it had none, `tune_prod_shapes.py` used 1e-2). Not covered: the 36
`NN` M=1 rows, an interaction between rows (the culprit passes alone) and schedule effects. The acceptance test for a row set is
the deterministic probe (at least 10 default and 6 no-EOS runs, with a control arm) on the flat and on the hier all-reduce.

What the flip is: not shown. Observed: the same logits at that token in five failing arms with different row sets, so it is
bit-reproducible (a data race is unlikely to repeat the margin to three decimals [INFERENCE]); it needs row 18 and other new rows; it follows the
all-reduce summation order; the row is as accurate as the default. [INFERENCE] A rounding pattern tips a decision with a 1.2-nat
margin one token after "The capital of Portugal is" (an MoE routing near-tie or the linear-attention state would amplify a
one-ulp change); it is not a defective solution. Not run: the real activations of that GEMM replayed with row 18 on and off at
the failing step, and the arm Z + row 18 + the 124 `TN` rows of 65 to 256 tokens.

**Verdict: ship 592 rows (`bis-nc`), leave out `tn_2048_12_2560`.** The final configuration (hier tree, 592 rows, deterministic
MoE) is clean (`h-ncd` 0 of 10 and 0 of 6), as is the 592-row set on the flat all-reduce (`bis-ncl`); the 593-row set fails on
the flat path (`tp8-s2d` 7 of 10, `h-s2d-flat` 5 of 10). Leaving the row out costs nothing measurable [INFERENCE: the trees differ]:
the 592 rows give -3.7 % and -4.1 % ITL p50 at c8 on the hier tree against -3.4 % for the 593 rows on the old tree.

## Memory audit (GiB per GPU, fixed KV 3.75 GiB, production mode)

| | TP4 (slot A) | TP8 | PP2 stage 0 | PP2 stage 1 |
|---|---:|---:|---:|---:|
| Weights ("Model loading took") | 20.56 | 11.23 | 9.79 (10.09 before fix 1) | 11.12 |
| KV allocated by torch | 3.75 | 3.75 | 3.12 (trimmed to the block count of stage 1) | 3.75 |
| Torch allocated / reserved | 24.72 / 25.17 | 15.38-15.45 / 15.80 | 13.16-13.22 / 13.56 | 15.33 / 15.86 |
| Peak allocated / reserved | 26.29 / 26.64 | 16.87 / 17.23 | 13.52 / 13.76 | 16.88 / 17.29 |
| Non-torch (driver used - reserved) | 1.63-1.68 | 1.60-1.68 | 1.46-1.48 | 1.43-1.52 |
| Driver used at READY + 30 s | 26.80-26.85 | 17.41-17.49 | 15.02-15.04 | 17.30-17.38 |
| Peak driver used | 28.27-28.32 | 18.83-18.91 | 15.22-15.24 | 18.72-18.80 |
| Free VRAM after boot / at peak (of 29.98) | 3.14-3.19 / 1.7 | 12.50-12.58 / 11.1 | 14.94-14.97 / 14.7 | 12.61-12.69 / 11.2 |
| CUDA-graph pool ("took") | 0.37-0.58 | 0.37-0.54 | 0.02-0.19 | 0.20-0.29 |
| KV capacity at 3.75 GiB | 266,945 tokens (1.02x) | 266,945 (1.02x) | 495,480 tokens (1.89x), set by stage 1 | |

Sources: `arms/<arm>/torchmem/*.jsonl` (per-worker allocator sampler, 2 s), `memsum.py`, sysfs VRAM per bus, `serve.log`.
rocm-smi was not used for VRAM (KFD order); VRAM was read per PCI bus.

* **Fixed costs do not scale with TP.** Non-torch memory is 1.4-1.7 GiB per process at any width [INFERENCE: HIP context,
  code objects, RCCL and `rdna_ar` buffers]. The activation peak (peak allocated - allocated) is 1.47 GiB at TP8, 1.57 at
  TP4 and 1.55 on PP2 stage 1 but only 0.34 on stage 0: it lives on the last stage [INFERENCE: logits/sampler/drafter; not
  attributed]. The CUDA-graph pool is 0.4-0.5 GiB at TP4 and TP8 alike (an earlier 0.04 GiB TP4 figure from the 2026-11-05
  logs was not reproduced in this harness).
* **Replicated weights:** hyper-connection tensors (1.17 GiB) and routers (0.12 GiB) are replicated on every rank: 1.29 GiB
  of TP8's 11.23 GiB per GPU, 10.3 GiB over the machine (checkpoint composition: PLE tables 95.4 GiB, packed int4 experts
  56.25 + 1.76 scales, MTP experts bf16 4.69, linear attention 3.87, HC 1.17, router 0.12, embeddings 1.18 + head 1.18).
* **KV:** one KV head per rank at TP4 and TP8 (replicated), so TP8 does not raise tokens per GiB; PP2 stage 1 holds 6 QSA
  layers plus the drafter's, stage 0 holds 6, so PP2 gives 1.86x the tokens per GiB. The scheduler's block count is the
  minimum over the stages, so stage 0 allocates 3.12 of the requested 3.75 GiB. Any partition keeps 7 KiB per token on the
  limiting stage [INFERENCE]. Decode context parallel would remove the replication at TP8 (unsupported on this backend
  [INFERENCE]).
* **Host RAM:** the PLE table is pinned and mlocked per worker and is sharded by the TP width. Measured: PP2 stage-0 workers lock
  24.3 GiB each (24,925 MiB, x4 = 97 GiB; stage-1 workers lock 0.6 GiB), TP8 workers lock 12.4 GiB each (12,744 MiB, x8 = 99.6
  GiB); TP4 shards 4-way like PP2 stage 0 [INFERENCE: ~97 GiB, not sampled in `tp4a-b`]. The total is ~100 GiB in every layout.
  Worker RSS after load: 27.6 GiB (PP2 stage 0), 15.8 GiB (TP8), 3.7 GiB (PP2 stage 1); load transient up to 48-50.5 GiB per
  stage-0 worker and 31.5-33.8 GiB per TP8 worker (mmapped safetensors in the page cache). `MemAvailable` minimum 319 GiB of
  455 GiB during `pp2-a`. The int4 sidecar (30 GB table through the page cache) was not booted [INFERENCE: ~65 GiB less pinned RAM].
* **Fixed:** `lm_head` on stage 0 (-0.30 GiB). **Proposed, not done:** a partition that balances the stages (26,22 gives about
  10.45 GiB on both [INFERENCE from 0.33 GiB per layer]) frees ~0.66 GiB on stage 1 = ~87k tokens; size the fixed KV for 8
  GPUs: ~12 GiB leaves 3.1 GiB free at the peak (TP4 production leaves 1.7), giving ~850k tokens at TP8 and ~1.58M at PP2
  [INFERENCE: linear in KV bytes].

KV sizing (phase 2; `KV=auto` at `GMEM=0.94` against the fixed 13 GiB of the recipes; smoke only, no cells; free = `total - driver used`
after the smoke, peak = largest `driver used` sample of the whole boot):

| Arm | KV per GPU | engine KV tokens | driver used after smoke (GiB) | free (GiB) | peak driver (GiB) |
|---|---:|---:|---:|---:|---:|
| `kvauto-tp8` | 14.96 auto | 1,061,059 (4.05x of 262k) | 28.63-28.73 | 1.25-1.36 | 28.63-28.73 |
| `kvauto-pp2` stage 0 / stage 1 | 16.17 auto (stage 0 line) | 2,079,867 (7.93x) | 25.12-25.15 / 29.21-29.25 | 4.84-4.86 / 0.73-0.77 | 25.29-25.31 / 29.33-29.36 |
| `rtp8-1` | 13.0 fixed | 924,705 (3.53x) | 26.62-26.72 | 3.27-3.36 | 28.07-28.17 |
| `rpp2-1` stage 0 / stage 1 | 13.0 fixed | 1,717,859 (6.55x) | 22.78-22.80 / 26.50-26.54 | 7.18-7.20 / 3.45-3.48 | 23.00-23.04 / 28.04-28.07 |

At `GMEM=0.94` the engine's estimate (profile-based) is 1.96 GiB above the fixed 13 GiB and leaves 1.3 GiB free at TP8 and 0.74 GiB
on PP2 stage 1 (sysfs VRAM peak 29,970-30,006 MiB: the card is full); the 16k c8 prefill added 1.4 GiB at the peak in the 3.75 GiB
arms, so 0.94 is too high for both layouts. The fixed 13 GiB is about `GMEM=0.875` and leaves 3.3 GiB after boot and 1.8-1.9 GiB at
the boot peak (the `flashnext-mtp2` margin is 1.7 GiB); the cell phase of the recipe arms in the measured section confirms it.
PP2 stage 0 keeps 7.2 GiB free that the minimum-over-stages block count cannot use (a partition that moves layers to stage 0
would use it: 26,22 and 28,20 are queued).

## Measured comparison (S1 rows, fix tree, fixed KV 3.75 GiB, one fresh boot per arm)

| Cell | Metric | TP4 slot A | TP8 | TP4 x PP2 |
|---|---|---:|---:|---:|
| 1k/512 c1 | TTFT mean / p99 s | 0.39 / 0.46 | 0.36 / 0.43 | 0.38 / 0.41 |
| | ITL p50 / p99 ms | 22.38 / 23.00 | 19.59 / 20.07 | 23.08 / 23.69 |
| | output tok/s (MTP acc len) | 80.3 (1.84) | 98.2 (2.00) | 95.6 (2.39) |
| 1k/512 c8 | TTFT mean / p99 s | 0.90 / 2.92 | 0.81 / 2.68 | 0.70 / 1.99 |
| | ITL p50 / p99 ms | 43.97 / 370.8 | 41.56 / 324.6 | 47.23 / 368.2 |
| | output tok/s (acc len) | 262.6 (2.01) | 290.3 (1.97) | 272.7 (2.04) |
| | prefill aggregate tok/s | 2806 | 3054 | 4110 |
| 16k/1k c1 | TTFT s | 5.52 | 5.28 | 3.40 |
| | prefill tok/s per request | 2966 | 3100 | 4823 |
| | ITL p50 / p99 ms | 22.92 / 23.40 | 20.00 / 20.94 | 23.51 / 24.34 |
| | output tok/s (acc len) | 72.1 (2.70) | 76.3 (2.51) | 78.6 (2.40) |
| 16k/1k c8 | TTFT mean / p50 / p99 s | 13.11 / 7.25 / 48.7 | 12.49 / 7.31 / 46.2 | 7.40 / 4.07 / 28.1 |
| | ITL p50 / p99 ms | 45.73 / 402.9 | 43.55 / 383.1 | 49.37 / 450.3 |
| | output tok/s (acc len) | 116.3 (2.17) | 126.7 (2.44) | 161.5 (2.32) |
| | prefill aggregate tok/s | 2789 | 2937 | 4838 |
| KV capacity at 3.75 GiB | tokens (max concurrency at 262k) | 266,945 (1.02x) | 266,945 (1.02x) | 495,480 (1.89x) |

* **TP8 vs TP4:** decode step -12 % at c1 (19.6 vs 22.4 ms; 20.0 vs 22.9 at 16k), -5 % at c8; output +10 % at c8 (290 vs 263,
  127 vs 116); prefill and TTFT only -4 %: eight GPUs buy a faster decode step but the prefill all-reduce is link-bound
  (see above), and TP8 gives no extra KV capacity. It uses 8 GPUs for +10 % throughput.
* **PP2 vs TP4:** prefill is the win: 16k solo TTFT -38 %, 16k c8 TTFT mean -44 % and output +39 %, 1k c8 TTFT mean -22 %,
  1.86x tokens per GiB of KV. Decode costs a little: c1 step +3 % (23.1 vs 22.4 ms: both stages in series), c8 step +7 %. The
  c8 1k output (+4 %) is inside the noise.
* **PP2 vs TP8:** PP2 wins every prefill-heavy cell (16k c8 output +27 %, TTFT -41 %); TP8 wins decode step time (-15 % at c1,
  -12 % at c8) and leads 1k c8 output by 6 % (290 vs 273 tok/s, inside the noise).
* **Against two TP4 replicas** (not measured): one replica gets the TP4 column, so two get about twice the aggregate output
  at saturating load [INFERENCE]; PP2's +39 % and TP8's +10 % are far below that. Both layouts win on latency for a *single*
  request stream (TTFT for PP2, step time for TP8), not on aggregate throughput per GPU.
* Junction temperature peaks 76-83 C in every arm; 0 KFD eviction, 0 dmesg lines.

## Measured: recipe cells and the new rows (ABBA, recipe trees, KV 13 GiB)

Tree `d2eea35872`, recipe-native boots, order A B B A per layout (A = shipped rows, B = the 593 new rows on ranks 0-7), four cells
each, `compare.py ... --abba`; "better" = larger than the spread inside both pairs. Means of the two boots:

| Layout / cell | Metric | A | B | B vs A | spread A / B | verdict |
|---|---|---:|---:|---:|---|---|
| TP8 1k c1 | TTFT mean s | 0.36 | 0.35 | -3.4 % | 0.2 / 0.5 % | better |
| TP8 1k c1 | ITL p50 ms | 19.55 | 19.56 | +0.1 % | 0.0 / 0.2 % | within spread |
| TP8 1k c8 | ITL p50 ms | 41.81 | 40.40 | -3.4 % | 0.1 / 0.2 % | better |
| TP8 1k c8 | out tok/s | 295.0 | 311.4 | +5.5 % | 2.1 / 4.5 % | better |
| TP8 16k c1 | TTFT s | 5.28 | 5.17 | -2.0 % | 0.0 / 0.2 % | better |
| TP8 16k c8 | ITL p50 ms | 43.61 | 42.07 | -3.5 % | 0.4 / 0.6 % | better |
| TP8 16k c8 | out tok/s | 126.7 | 129.9 | +2.5 % | 1.5 / 1.7 % | better |
| PP2 1k c1 | TTFT mean s | 0.40 | 0.38 | -3.4 % | 0.2 / 1.2 % | better |
| PP2 1k c1 | ITL p50 ms | 23.38 | 23.18 | -0.8 % | 0.1 / 0.8 % | better |
| PP2 1k c8 | ITL p50 ms | 48.18 | 43.24 | -10.2 % | 2.0 / 0.8 % | better |
| PP2 1k c8 | out tok/s | 262.6 | 285.3 | +8.7 % | 3.2 / 1.3 % | better |
| PP2 16k c1 | TTFT s | 3.64 | 3.56 | -2.3 % | 0.2 / 0.0 % | better |
| PP2 16k c8 | ITL p50 ms | 49.24 | 44.50 | -9.6 % | 0.1 / 0.1 % | better |
| PP2 16k c8 | out tok/s | 156.3 | 164.1 | +5.0 % | 0.2 / 1.6 % | better |

Not in the table: TP8 1k c1 output is -7.5 % with B (95.0 against 102.8 tok/s) while the step time is equal; the MTP acceptance
length moved (2.10 against 2.05; 4 requests per cell), so this is acceptance noise [INFERENCE]; PP2 1k c1 output varies 85-97 tok/s
for the same reason. TTFT means at c8 are within the spread (queueing). The A pair is the recipe's own spread: ITL p50 repeats
within 0.1 % at c1 and 0.4 % (TP8) / 2.0 % (PP2) at c8. Same arms: peak driver memory over the whole arm including the 16k c8
cells 28.13-28.24 GiB at TP8 and 23.1 / 28.0 GiB per stage at PP2 (1.74-1.85 GiB free at TP8, 1.9 GiB on PP2 stage 1: the KV
choice holds); PPL probe 4.75-4.85 and `PPL_LONG` 3.221-3.235 in every arm (this harness's passage set of 21,261 tokens; the
phase-1 `PPL_LONG` numbers use another set and are not comparable); 0 KFD evictions, 0 dmesg lines. Concurrency repeats: 0 of 12
failing runs per layout (3 per boot, four boots); the boot probe of `m-pp2-b1` failed on the haiku check above.

## Final cells on the hier tree (`38ab87ed3a`, recipe-native, KV 13 GiB)

Tree `38ab87ed3a` is `rdna_extras` with HierAr's hierarchical all-reduce (`dc35df271d`): the exact-SHA tree is the HierAr tree
`~/vllm-hier-9b96ecfa7a` with `git archive 38ab87ed3a` over it and its `_rocm_C.abi3.so` (`csrc/` and `vllm/` of the build match
the tip). The launch log shows `rdna_ar: one-shot all-reduce active (handle 0, rank r/8, devices [0..7], ... up to 1024 KB, RCCL
above; ... hier islands [[0, 1, 2, 3], [4, 5, 6, 7]] ...)` on all 16 TP and EP workers, so hier is on by default at TP8 (auto: a
world-8 group that spans two PCIe-switch islands; hier from 20 KiB up to 1024 KiB, the flat one-shot below 20 KiB, so every c1
size and the c1 step are untouched). PP2 (world 4) is not affected: hier auto never applies to four ranks. Hier adds about 30 MiB
of uncached staging per GPU (16 MiB per instance), negligible against the 1.8 GiB headroom of the 13 GiB KV. The
`PIX topology query failed (name 'amdsmi_init' is not defined)` warning (16 per boot) is pre-existing and also appears on the
older tree.

Boots `h-tp8-a1`, `h-tp8-a2` (A = shipped rows) and `h-tp8-b1`, `h-tp8-b2` (B = the 592 rows, 3330 per rank), four cells each,
`compare.py ... --abba`. The order was A A ... B B (1.5 h apart), not ABBA, so a slow drift cannot be excluded; the old-tree ABBA
above (A B B A, 593 rows) gave the same gain at TP8. "Pre-hier A" is the A pair of the ABBA table above (tree `d2eea35872`, flat
one-shot). Means of the two boots:

| Cell | Metric | Pre-hier A | Hier A | Hier A vs pre-hier A | Hier B | B vs A (hier) | spread A / B |
|---|---|---:|---:|---:|---:|---:|---|
| 1k c1 | TTFT mean s | 0.36 | 0.36 | 0 | 0.35 | -2.1 % better | 1.3 / 0.3 % |
| 1k c1 | ITL p50 ms | 19.55 | 19.57 | +0.1 % | 19.57 | 0.0 % | 0.0 / 0.0 % |
| 1k c8 | TTFT mean s | | 0.82 | | 0.76 | -7.8 % better | 3.9 / 2.3 % |
| 1k c8 | ITL p50 ms | 41.81 | 37.17 | **-11.1 %** | 35.80 | -3.7 % better | 0.7 / 0.3 % |
| 1k c8 | out tok/s | 295.0 | 326.8 | **+10.8 %** | 327.6 | +0.3 % within spread | 2.9 / 2.5 % |
| 16k c1 | TTFT s | 5.28 | 5.27 | -0.2 % | 5.18 | -1.8 % better | 0.4 / 0.3 % |
| 16k c1 | ITL p50 ms | 19.99 | 19.96 | -0.2 % | 19.98 | +0.1 % | 0.3 / 0.4 % |
| 16k c8 | TTFT mean s | | 12.68 | | 12.45 | -1.9 % within spread | 8.5 / 1.3 % |
| 16k c8 | ITL p50 ms | 43.61 | 39.28 | **-9.9 %** | 37.67 | -4.1 % better | 0.0 / 0.4 % |
| 16k c8 | out tok/s | 126.7 | 128.4 | +1.3 % within spread | 133.4 | +3.9 % better | 1.8 / 0.6 % |

Per-boot values are in `arms/h-tp8-{a1,a2,b1,b2}/cells.tsv`. The 1k c1 and 16k c1 output tok/s (96.7 / 101.7 and 68.5 / 75.3 for A /
B) are MTP-acceptance noise (acceptance length 1.79-2.66 across the four boots, spread 2-21 %), not a rate change; the step time is
equal. PPL probe 4.80-4.83 in all four boots, probes 3/3, 0 KFD evictions, 0 dmesg lines. Hier moves the whole c8 column: the
decode step carries 24 tokens = 120 KiB per all-reduce, which hier serves with island partials and at most four cross-root flows of
a quarter of the payload instead of 16 whole-payload flows ([rdna-ar-hier-2026-11-06.md](rdna-ar-hier-2026-11-06.md)). The new
rows add a further 3.7 to 4.1 % on top (B against A on the same tree).

Correctness on the hier tree (`CFG=rtp8`, the recipe itself, run before the A/B pairs; `hook.sh` with `CONC_REPEAT=20
CONC_REPEAT_NOEOS=10`, default MoE):
`h-tp8-pa` (shipped rows) 1 of 20 failing default-probe runs (haiku past its stop token, token 27, margin 1.844) and 0 of 10
no-EOS runs; `h-tp8-pb` (593 rows) 1 of 20 (token 29, margin 1.219) and 0 of 10; Fisher p = 1.0 on both. `h-pp2-s` (the PP2 recipe
on this tree, shipped rows): READY, probes 3/3, PPL 4.80, concurrency PASS. The deterministic-MoE arms on this tree are in the rows
section above.

## Recipes (boot-validated on `PCI_BUSES`; cells in the measured section)

`flashnext-tp8-mtp2` (port 18098) and `flashnext-tp4pp2-mtp2` (port 18099, `PP_PARTITION=24,24`) are in
`tools/rdna/recipes/` (the `flashnext-pp3-mtp2` naming convention). Both set `PCI_BUSES="ce d3 d8 db 0a 0f 14 17"` (the
launcher key; print-plan tests in `tests/tools/rdna/test_serve_pci_buses.py`), `KV=13958643712` (13 GiB), `SEQS=8`,
`MAXBAT=2048`, capture sizes unchanged (`[3, 6, 12, 24]`), `GPU_MAX_HW_QUEUES=4`, no RCCL variables, and a one-shot gate of
128 KiB (TP8: every captured decode size is at most 120 KiB, so serving dispatch equals the 256 KiB gate; on a tree with the hier
all-reduce every size from 20 KiB up to 1 MiB takes the hier path, so 128 KiB then gates only the flat sizes below 20 KiB and the
other collectives) or 256 KiB (PP2).
KV from the memory audit: the peak driver memory is 18.9 GiB (TP8) and 18.8 GiB (PP2 stage 1) at 3.75 GiB of KV, so 13 GiB
leaves about 1.8 GiB at the activation peak, the margin `flashnext-mtp2` runs with (1.7 GiB). A fixed KV skips the memory
profile, so the launcher does not check it against `GMEM`. Both recipes ship TunableOp rows for ordinals 0-7 (3330 per rank).

Recipe-native compiled boots with smoke (`rtp8-1`, `rpp2-1`, tree `d2eea35872`, `RECIPE_RESOLVES_GPUS=1`: no `HIP_VISIBLE_DEVICES`
is passed, `serve_wrap.sh` compares what `PCI_BUSES` resolved to with the verified slot list): both resolved to HIP
`1,2,3,4,6,7,8,9`, probes 3/3, PPL 4.85 and 4.79, one concurrency probe PASS each; the launcher printed the TunableOp warning
for ranks 4-7. Three more concurrency runs: TP8 3/3 PASS; PP2 2/3, the failing check `together 1: token 27/64, margin 2.563,
continuation mean lp -0.04 vs solo -0.32` (the haiku prompt past the stop token again, here with the shipped rows, so the flip is
not specific to the new rows). `torchmem` (memhook on `torch.accelerator`) recorded no errors. `KV=13958643712` printed
"reserved 13.0 GiB for KV Cache ... skipped memory profiling".

Tools (`tools/rdna/tp8pp2/`, committed on `fix/tp8-pp2-flashnext`; the phase-1 box-local names `driver8.sh`, `hook8.sh`,
`slots8.sh`, `camp8.sh`, `ar8.sh` are `driver.sh`, `hook.sh`, `slots.sh`, `camp.sh`, `ar.sh`): `slots.sh` (GPU slots by PCI
bus, leases), `driver.sh` (one arm: lease and memory checks, exact-SHA tree, `CFG=tp8|pp2|tp4a|tp4b|dp2|rtp8|rpp2`,
`serve_validate.sh`, evidence), `hook.sh` (probes, `CONC_REPEAT` / `CONC_REPEAT_NOEOS`, quality, cells under the quiet-file
protocol), `queue.sh` (an arm queue that holds the leases and releases them at a boot boundary when `queue.pause` appears),
`queue_insert.py` (rewrites the unread tail of an open queue file in place, same inode), `conctally.py` (failing-run rate and
Fisher test), `compare.py` (cell tables, ABBA), `memsum.py`, `extract.sh`, `tunehit.py` (lookup-hit proof), `bisectrows.py` (row
trees for a bisect: none, all, `nn`, `tn_s`, `tn_m`, `tn_b`, bit and index subsets of a group) and `rowcheck.py` (the two-process
numerics gate, see the rows section; `camp.sh check` runs it before `harvest`). One arm: `export
RUN_DIR=$HOME/w4a8_runs/tp8pp2_2026-11-06; OWNER=<you> SHA=<40 hex> CFG=tp8 ARM=<name> HOOK=1 [CONC_REPEAT=10 CELLS_SPEC=none
SERVE_ENV=TUNABLEOP_ROOT=<dir>] bash tools/rdna/tp8pp2/driver.sh`; a queue: `OWNER=<you> SHA=<40 hex> bash
tools/rdna/tp8pp2/queue.sh <queue file>` (one `KEY=value` line per arm; append only while it runs: the file stays open).

## Open items and proposals

* Rows: shipped as 592 of 593 (data `006916e58f`, tools `f40b8c290f`). Not run: the arm Z + row 18 + the 124 `TN` rows of 65 to
  256 tokens (the smallest failing combination is unknown), a replay of the real activations of the 2048 x 2560 layer with row 18
  on and off at the failing step (the flip's mechanism is open), and a 592-row re-measure at PP2 (the PP2 cells are from the
  593-row ABBA).
* Row acceptance for the next campaign: the deterministic concurrency probe (`VLLM_RDNA2_MOE_DETERMINISTIC=1`, 10 default and 6
  no-EOS runs) on the flat and the hier all-reduce, plus the accuracy gate (`camp.sh check`). The gate covers `TN` keys only (the
  36 `NN` M=1 rows have none) and cannot see interactions; `TUNE_ATOL=1e-9` (bit-exact solvers) costs 7 to 29 % on the shapes
  where the fast solver rounds differently and is not the default.
* Recipes: the ABBA arms ran the cells at KV 13 GiB with at most 28.24 GiB peak driver memory, so the KV stays.
* PLE sidecar under TP8 and PP2; PP partitions 26,22 / 28,20 (decode and prefill balance of the drafter stage); 2 x TP4
  replicas (`CFG=dp2`, both servers loaded at c4 each): not run.
* The BMC SEL is 100 % full (last add 2026-11-05 14:01): hardware errors can no longer be logged there. Proposal for the
  user: `ipmitool sel clear` (not done).
* Memory proposals above (balanced partition, 12 GiB KV, decode context parallel at TP8).
