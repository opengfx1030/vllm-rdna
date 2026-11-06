# Flash-Next on all 8 V620s of par1-cs25: TP8 and TP4 x PP2 (2026-11-06)

Date: 2026-11-06. Box: par1-cs25 (two PEX880xx switches, roots `c0` = slot A and `00` = slot B, four V620 each).
Model: `Qwen3.8-Flash-Next-AWQ-W4A16`, MTP=2, production compiled mode (`COMPILE_MODE=3`, `FULL_AND_PIECEWISE`,
`MAXBAT=2048`, `SEQS=8`) on every arm. Hardware audit this doc builds on:
[pcie-iommu-audit-2026-11-06.md](pcie-iommu-audit-2026-11-06.md). Run directory on the box:
`~/w4a8_runs/tp8pp2_2026-11-06/` (arms in `arms/<arm>/`). Claims without a measurement are marked [INFERENCE].

## Status

The run was stopped at its request budget. Done and proven: TP8 and TP4 x PP2 boot, serve and pass every smoke in production
compiled mode; no crash, hang or hardware error; one memory waste fixed (`lm_head` on PP stage 0); a silent TunableOp gap
found (ranks 4-7 had no rows) and a 593-row campaign run; the all-reduce numbers at world size 8; the memory audit; the
measured comparison (one fresh boot per arm, no repeats). **Not done:** the two new recipes were not boot-validated, so they
are not shipped (drafts only); the TunableOp ABBA was not completed and the new-rows arm failed the concurrency probe once
(open, see TunableOp); the PLE sidecar was not booted under TP8 or PP2; no `KV=auto` memory arms; no 2 x TP4 replica
reference; the HierAr hand-over (final TP8 recipe) is therefore missing.

## Summary

| Question | Verdict |
|---|---|
| TP8 (HIP 1,2,3,4,6,7,8,9) boots and serves in production mode | Yes. READY 190-291 s, probes 3/3, PPL 4.84, concurrency PASS, 0 dmesg/KFD/AER/IOMMU lines. |
| TP4 x PP2 (stage 0 = slot A, stage 1 = slot B) | Yes, no code fix needed to run. MTP2, QSA (both stages), PLE (stage 0), `rdna_ar` (TP, EP and PP groups) all work. |
| Crashes / hangs / faults | None. One waste bug (below), one silent tuning gap, harness-only bugs. |
| `rdna_ar` at world size 8 | Active (handles 0 and 1 = TP and EP groups, devices 0-7). One-shot beats RCCL up to ~155 KiB; RCCL graph costs 134-178 us per call up to 240 KiB. |
| Memory | TP8 weights 11.23 GiB/GPU, 12.5 GiB free after a 3.75 GiB KV boot. PP2 fits 1.86x the tokens per GiB of KV (7 KiB vs 13 KiB per token and GPU) [INFERENCE for the per-token split]. |
| Fastest layout | Prefill-heavy (16k): PP2, TTFT 3.40 s vs 5.52 s TP4 (-38 %), 16k c8 output 161 vs 116 tok/s (+39 %). Decode latency: TP8, step 19.6 ms vs 22.4 ms TP4 (-12 %). TP8 prefill is only -4 % vs TP4. |
| TunableOp | Ranks 4-7 had no rows (TP8 and PP2 stage 1 ran untuned, silently). 593 new rows tuned, hit proof 693/693 on ordinals 0-7. First arm with them: -3 % ITL, -2..-6 % TTFT (single pair) but the concurrency probe FAILED once: not shippable yet. |

## Method

* GPUs: slot A = HIP 1-4 = V620 buses `ce d3 d8 db`; slot B = HIP 6-9 = buses `0a 0f 14 17`. HIP 0 and HIP 5 are the W7800s
  and are never used. `rocm-smi` numbers GPUs in KFD order, so every launch resolved HIP ids from the PCI bus first
  (`slots8.sh verify_slot`: bus, name and 30 GB per GPU).
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
* All numbers are one boot per arm: no run-to-run spread was measured for TP8 or PP2, so differences below ~5 % are not
  evidence. MTP acceptance length varies by up to 0.5 between arms on the same prompts (4 requests per c1 cell), which moves
  output tok/s; ITL p50 (time per verify step) and TTFT are the robust columns.

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
* The hierarchical all-reduce (HierAr, branch `feat/rdna-ar-hier`) is a separate change; its measured per-call numbers at
  TP8 (hier vs RCCL: 64 KiB 37 vs 178 us, 128 KiB 36 vs 136, 256 KiB 50 vs 146, 512 KiB 77 vs 166, 1 MiB 144 vs 171, tie at
  1.5 MiB) are reported in its own document. At TP8 an MTP2 c8 step issues about 100 all-reduces at 120 KiB, so a 65 us saving
  per call would be a few ms of a 40 ms step [INFERENCE, not measured here].

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
as copies of ranks 0-3): **3331 rows per rank** (2738 + 593). Lookup-hit proof (`tunehit.py`, torch loads each rank file
for its ordinal): ordinals 0-7 all `census_keys=693 hit=693 missing=0`. Numerics (`rowcheck.py`, rank-0 file, one V620):
557 `TN` keys of the 593 compared with an fp32 reference, tuned vs default: none worse than 3x the default's error
(`tunable/rowcheck_rank0.txt`).

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

## Recipes (not shipped: no recipe-native compiled boot was run)

Drafts (`flashnext-tp8-mtp2.env`, `flashnext-tp4pp2-mtp2.env`; the `flashnext-pp3-mtp2` naming convention) were prepared under
`tools/rdna/recipes/` but are **not committed**: a recipe ships only after a compiled boot, smoke and bench on it. Copies are in
`~/w4a8_runs/tp8pp2_2026-11-06/recipes_draft/` on the box. Settings the data supports: `PCI_BUSES="ce d3 d8 db 0a 0f 14 17"`
(the launcher key of `23f1d01305`, not yet exercised on a real launch), `TP=8` or `TP=4 PP=2 PP_PARTITION=24,24`, `KV` about
12 GiB, one-shot gate 128 KiB (TP8) or 256 KiB (PP2), `SEQS=8`, `MAXBAT=2048`, capture sizes unchanged (`[3, 6, 12, 24]`),
`GPU_MAX_HW_QUEUES=4`, no RCCL variables, TunableOp rows for ranks 0-7 once S2 is resolved.

Arm invocation used here (box-local harness in the run directory): `SHA=2697797c787e0401d4cfd261b9cbbd73b5ccf8b1 CFG=tp8|pp2|tp4a
ARM=<name> LOAD_HOOK=$PWD/hook8.sh bash driver8.sh`; S2: `SERVE_ENV="TUNABLEOP_ROOT=$PWD/tunable/rows"`; census:
`RECORD_UNTUNED=1`; `python3 compare.py <tree> arms label=arm ... [--abba A=a1,a2 B=b1,b2]` prints the tables above.

## Open items and proposals

* Resolve the S2 concurrency FAIL (procedure above) before shipping the 593 rows; then complete the S1/S2/S2/S1 ABBA at TP8
  and S1 vs S2 at PP2. The rows are the separate branch commit `667d0dce25` and must not be merged before that.
* Validate the two recipes (compiled boot, smoke, bench) and then size `KV` from the `KV=auto GMEM=0.90` arms (not run).
* PLE sidecar under TP8 and PP2; PP partition 26,22 / 28,20 (decode and prefill balance of the drafter stage); a 2 x TP4 replica
  reference with both servers loaded at c4 each; the 128 vs 256 KiB gate at TP8 in serving; repeat boots for spread.
* The BMC SEL is 100 % full (last add 2026-11-05 14:01): hardware errors can no longer be logged there. Proposal for the
  user: `ipmitool sel clear` (not done).
* Memory proposals above (balanced partition, 12 GiB KV, decode context parallel at TP8).
* HierAr needs the final TP8 invocation: use the box harness line above (tree `2697797c78`); no validated recipe exists yet.
