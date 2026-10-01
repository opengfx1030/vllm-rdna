# FP16 TunableOp rows — rocblas-f30bb442e9b5

Build-specific rocBLAS solver rows for the serving stack on 4× V620 (gfx1030),
captured 2026-09-29 by re-tuning the production Flash-Next shapes with the
0.28.0 tree. The directory hash is the first 12 hex chars of the `librocblas.so.5`
sha256 (see `provenance.json`).

## What changed vs the harvested 903-row set

| | rows | file |
|---|---|---|
| harvested (2026-09-27) | 903 | `tunableop_results{0..3}.csv` |
| **curated (2026-09-29)** | **719** | `tunableop_results{0..3}.csv` |

* The harvested set was missing every FP16 GEMM with **M ≤ 8** — exactly the
  cudagraph capture sizes for MTP-0 `[1,2,4,8]` and MTP-2 `[3,6,12,24]`. The
  decode path was therefore falling back to rocBLAS heuristics for those
  shapes, which dominate c=1 latency.
* 141 of the 719 rows are new (the M ≤ 8 decode projections for every `(N, K)`
  pair already seen at M ≤ 32).
* 302 rows were dropped: Default entries that reproduce the heuristic, and a
  handful of harvested rows that a fresh measurement found slower than the
  heuristic (they misled).
* 578 rows carried over unchanged: the existing solver still beat the heuristic
  by more than 3 % under the new measurement.
* All four rank files share the same content; the per-rank variation in the
  harvested file was noise (identical V620 cards, same rocBLAS build). The
  helper still requires four rank files, so we ship four copies of the same
  719 rows.

The biggest single win is the lm_head at M ≤ 8: the heuristic / current row
takes ~2.77 ms per call, the tuned row takes ~0.90 ms (3.0×) for M=1 and
~2.7–2.8× for M=2..8. That lands on every decode step in c=1, which is the
recovery the earlier `/tmp` storage leak made invisible.

## Storage policy (mandatory)

* **Never** store rows in `/tmp` or the run CWD. Both are wiped / vary between
  runs, which silently drops c=1 and small-batch shapes back to rocBLAS
  heuristics.
* **This directory is the shared source of truth.** It ships with the fork
  (`opengfx1030/vllm-rdna`) so every user gets the same tuned rows.
* **Per-user fallback:** `$HOME/.cache/tunableop/tunableop_results.csv`. The
  helper uses it automatically when this build has no rows, and it is where
  online tuning writes.

## Use

Consumption is a one-liner — source the helper and point it at this build:

```bash
source tools/rdna2_028/tunableop_env.sh
configure_tunableop "$HOME/Apps/vllm/venv-7.14.0_0.28.0/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/librocblas.so.5" \
                    "$PWD/tunableop"
```

It selects `tunableop/rocblas-<hash>/` from the loaded rocBLAS build and sets
a **lookup-only** environment (`PYTORCH_TUNABLEOP_ENABLED=1`, `TUNING=0`,
`PYTORCH_TUNABLEOP_FILENAME` at the rank files). The launchers
(`scripts/serve_gfx1030_flashnext_mtp.sh`, `scripts/serve_gfx1030_27b_dense.sh`,
`scripts/serve_gfx1030_flashnext.sh`, `scripts/serve_gfx1030_full.sh`) already
do this. A healthy start logs:

```
TunableOp lookup enabled for rocBLAS build f30bb442e9b5 (rows: .../tunableop/rocblas-f30bb442e9b5); tuning off ...
```

* `PYTORCH_TUNABLEOP_TUNING=1` re-tunes shapes missing from the table (writes
  into the selected rows path; harvest into this directory when validated).
* If the rocBLAS hash differs, the helper warns and falls back to
  `~/.cache/tunableop/`; serving continues with default FP16 algorithms.

## Measured (4× V620, TP=4, Qwen3.8-Flash-Next-AWQ-W4A16)

See [`bench_results/2026-09-29_tunableop-rows/SUMMARY.md`](../../../bench_results/2026-09-29_tunableop-rows/SUMMARY.md)
for the per-shape before/after table. The lookup-vs-heuristic A/B is the shipped
evidence: the missing M ≤ 8 lm_head shapes go from ~2.77 ms (heuristic) to
~0.90 ms (3.0×) at M=1, 2.7–2.8× for M=2..8, which is the c=1 decode recovery.
End-to-end cells with these frozen rows are recorded under
`bench_results/2026-09-29_tunableop-rows/e2e/`.

## Caveats

* Solution IDs are build-specific: never reuse across a different rocBLAS build
  (compare the library sha256 in `provenance.json`).
* Tuned algorithms may change the floating-point reduction order. Outputs are
  not bit-identical and greedy acceptance shifts slightly.
* Cold tuning is impractical online (>20 min of inline tuning per workload);
  ship these rows instead.

## Regeneration

```bash
WORK=/home/chenco_adm/w4a8_runs/tunableop-rows
bash /home/chenco_adm/vllm-rdna-0.28.0/tools/rdna2_028/tunableop_rows_pipeline.sh all
python3 /home/chenco_adm/vllm-rdna-0.28.0/tools/rdna2_028/curate_tunableop_rows.py \
  --current-rows /home/chenco_adm/vllm-rdna-0.28.0/tunableop/rocblas-f30bb442e9b5 \
  --scratch-rows $WORK/scratch \
  --heur $WORK/measure/heuristic.json \
  --cur $WORK/measure/current.json \
  --new $WORK/measure/new.json \
  --out $WORK/curated --adopt-margin 0.02 --drop-margin 0.03
```

The historical leak this directory fixes wrote to `/tmp/tunableop_results{0..3}.csv`
(the PyTorch default name) from drivers that enabled TunableOp without setting
`PYTORCH_TUNABLEOP_FILENAME`. All launch paths now source the helper, so no
script can leave rows in `/tmp` or the CWD.
