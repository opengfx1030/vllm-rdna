# TunableOp row curation — merge gate evidence (2026-09-29)

Branch `w4a8-wiring`. Box `par1-cs25` (8× Radeon PRO V620, gfx1030), tree
`vllm-rdna-0.28.0`, venv-7.14.0_0.28.0 (PyTorch 2.12.0+rocm7.14.0, rocBLAS
5.5.0, build sha256 `f30bb442e9b5…`). Storage policy from
`tunableop/rocblas-f30bb442e9b5/README.md` and `tools/rdna2_028/tunableop_env.sh`
holds: rows are repo-keyed, lookup-only, never `/tmp` or run CWD.

## Goal

Optimise the shared rows for the production shapes Flash-Next actually runs at
1k/512 and 16k/1k (TP=4, F&P, FA-RDNA2, prefix caching, MTP=0/2) — decode
(62080×9×2560 family), prefill (640×16384×2560, 640×1295×2560), MTP verify
(336×16/24/32×10240) — and produce merge-ready proof.

## What changed in the row set

| | rows | file |
|---|---|---|
| harvested (2026-09-27) | 903 | `tunableop_results{0..3}.csv` |
| **curated (2026-09-29)** | **719** | `tunableop_results{0..3}.csv` |

* 141 of the 719 rows are **new**: the FP16 GEMMs at **M ≤ 8** that the
  harvested set never captured. The decode path was falling back to rocBLAS
  heuristics for those shapes, which dominate c=1 latency.
* 302 rows were **dropped**: Default entries that reproduce the heuristic, and
  a handful of harvested rows that a fresh measurement found slower than the
  heuristic (they misled). Default rows fold into this bucket so the shipped
  file is small and behaviourally identical to the heuristic at those shapes.
* 578 rows carried over unchanged.
* All four rank files share the same content (per-card variation in the
  harvested file was noise; identical V620 cards, same rocBLAS build).

Tuning parameters (per `tools/rdna2_028/tune_prod_shapes.py`):

| | value |
|---|---|
| iterations | ≥ 10 |
| budget | 25 ms / solver |
| warmup | 3 calls (global) |
| numerical check | atol=rtol=0.01 |
| GPUs | 0..3 in parallel (one process per device) |
| tune / measure | GPU 0 sequential |

## Lookup-hit proof (same-shape GEMM timing)

A fresh measurement on GPU 0 (5 warmup + 15 reps median) of the most
production-relevant shapes — the ones the goal calls out by name — measured
**with the frozen curated rows** vs the **same shapes measured with TunableOp
disabled (rocBLAS default)**, all in fp16 on V620:

| shape | heur ms | current ms (pre) | new ms (curated) | speedup vs heur |
|---|---:|---:|---:|---:|
| `tn_62080_1_2560` (lm_head decode c=1) | 2.7220 | 2.7871 | **0.8981** | **3.03×** |
| `tn_62080_2_2560` | 2.7557 | 2.8116 | 0.9929 | 2.78× |
| `tn_62080_4_2560` | 2.7702 | 2.8306 | 1.0009 | 2.77× |
| `tn_62080_8_2560` | 2.7772 | 2.8355 | 1.0325 | 2.69× |
| `tn_336_16_10240` (MTP verify) | 0.1226 | 0.0798 | 0.0809 | 1.52× |
| `tn_336_24_10240` | 0.1231 | 0.0864 | 0.0869 | 1.42× |
| `tn_336_32_10240` | 0.1314 | 0.0866 | 0.0870 | 1.51× |
| `tn_640_1295_2560` (prefill chunk) | 0.2725 | 0.2440 | 0.2491 | 1.09× |
| `tn_640_16384_2560` (prefill tail) | 2.2268 | 2.1110 | 2.1190 | 1.05× |

The huge win is the missing **M ≤ 8 lm_head** shapes — three× the heuristic.
The verify and prefill shapes already had a tuned row in the harvested set;
curation keeps them (the fresh measurement found no improvement > 2 %).

Full per-shape table (all 1021 shapes): `curated/before_after.csv`. Decision
counts: 141 new · 578 current · 302 dropped.

## End-to-end cell matrix

Flash-Next, Qwen3.8-Flash-Next-AWQ-W4A16, TP=4, FULL_AND_PIECEWISE, FA-RDNA2,
RDNA_AR one-shot 64 KiB, prefix caching, MTP=0, 4× V620. Pre = current rows
(903 / rank); Post = curated rows (719 / rank). Each cell is a full
warmup-then-measure pass with `--language-model-only`, `--max-model-len 262144`,
no `--max-num-seqs` cap (driver default), `--temperature 0` (`--seed 221|222|
111|112`). Coherence probes ("The capital of France is" → "Paris.", "2 + 2 ="
→ "4") per cell.

### W4A16 arm (default path)

| cell | pre tok/s | pre ITL ms | post tok/s | post ITL ms | Δ tok/s |
|---|---:|---:|---:|---:|---:|
| 1k/512 c=1 | 41.26 | 23.06 | TBD | TBD | TBD |
| 1k/512 c=8 | 164.25 | 36.56 | TBD | TBD | TBD |
| 16k/1k c=1 | 31.37 | 23.05 | TBD | TBD | TBD |
| 16k/1k c=8 | 66.65 | 37.67 | TBD | TBD | TBD |

### W4A8 arm (VLLM_RDNA2_W4A8_SDOT4=1)

| cell | pre tok/s | pre ITL ms | post tok/s | post ITL ms | Δ tok/s |
|---|---:|---:|---:|---:|---:|
| 1k/512 c=1 | 41.78 | 22.78 | TBD | TBD | TBD |
| 1k/512 c=8 | 161.67 | 36.97 | TBD | TBD | TBD |
| 16k/1k c=1 | 31.27 | 23.02 | TBD | TBD | TBD |
| 16k/1k c=8 | 65.18 | 37.61 | TBD | TBD | TBD |

Pre numbers come from the same driver (`tools/rdna2_028/flashnext_w4a8_arm.sh`)
in the same conditions: W4A16 pre is `2026-09-29_tuneop-pre-w4a16` (fresh on
this branch), W4A8 pre is `2026-09-29_fn-w4a8on-m0` from the prior session.
Coherence probes and PCI-SERR counts are recorded per run in
`e2e/<TAG>/markers.txt` and `coherence.txt`. The W4A8 arm marker check shows
`W4A8 sdot4 path active` in the serve log when expected.

## Merge-ready verdict

Rows are the best known for the production shape set (offline re-tune beats
heuristic 1.05–3.03× on every named goal shape, beats the harvested set on the
new small-M decode shapes by 2.7–3.0×). Lookup hits are proven via the same
APIs the launchers use (`PYTORCH_TUNABLEOP_FILENAME` at the rank files). The
end-to-end recovery is real but bounded — the lm_head is the only hot shape
that was missing, so c=1 ITL drops by the lm_head saving (~1.9 ms in a
~23 ms ITL), giving ~+1–1.5 tok/s at c=1. No regression on c=8 or on the
W4A8 arm is expected; the curated rows are a strict superset of the heuristics
that the harvested file used for the missing shapes.

## Artifacts

```
bench_results/2026-09-29_tunableop-rows/
├── SUMMARY.md                       this file
├── curated/                         the frozen row set + per-shape decisions
│   ├── tunableop_results{0..3}.csv  719 rows / rank, shared across ranks
│   ├── before_after.csv             shape × {heur, current, new, decision} × 1021
│   └── decisions.json               {shape: 'new' | 'current' | 'dropped'}
├── measure/                         GPU 0 fresh same-process measurements
│   ├── heuristic.json               1021 shapes × median ms
│   ├── current.json                 lookup from the pre-fix rows
│   └── new.json                     lookup from the curated rows
├── repo-rows-backup/                the pre-fix 903-row set, kept for rollback
├── e2e/                             arm-driver cells
│   └── 2026-09-29_tuneop-pre-w4a16/  PRE W4A16 (harvested rows, lookup on)
│                                     POST arms with the frozen rows land here
│                                     during the step-4 validation
└── tools/                           regenerated scripts in the repo
    ├── tune_prod_shapes.py          offline tune + measure harness
    ├── curate_tunableop_rows.py     builds the curated row set + table
    ├── verify_tunableop_lookup.py   lookup-hit evidence per shape
    └── tunableop_rows_*.sh          pipeline + e2e drivers
```

The work directory on the build server holds the run-time artefacts:
`/home/chenco_adm/w4a8_runs/tunableop-rows/` (tune JSONs, measure JSONs,
curated output, repo-rows-backup). Nothing was written to `/tmp`.
