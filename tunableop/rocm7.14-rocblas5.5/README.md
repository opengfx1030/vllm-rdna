# FP16 TunableOp rows — profile rocm7.14-rocblas5.5

Build-specific rocBLAS solver rows for the serving stack on 4× V620 (gfx1030),
captured 2026-09-29 by re-tuning the production Flash-Next shapes with the
0.28.0 tree. The profile is registered in `../profiles.json`; its
`lib_sha256` is the `librocblas.so.5` sha256 (see `provenance.json`), and
`configure_tunableop` matches it against the loaded build.

## What changed vs the harvested 903-row set

| | rows | file |
|---|---|---|
| harvested (2026-09-27) | 903 | `tunableop_results{0..3}.csv` |
| curated (2026-09-29) | 719 | `tunableop_results{0..3}.csv` |
| campaign-3 expansion (2026-09-30) | 760 | `tunableop_results{0..3}.csv` |
| EXL3 27B expansion (2026-10-01) | 783 | `tunableop_results{0..3}.csv` |
| Flash-Next mixed-batch expansion (2026-10-10) | 979 | `tunableop_results{0..3}.csv` |
| **Coverage: Qwen3.6, DSV4, EXL3 MTP=2, AWQ W4A8 (2026-10-10)** | **1601** | `tunableop_results{0..3}.csv` |

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

## Coverage expansion (2026-10-10)

Census over the four bench cells of Qwen3.6-35B-A3B GPTQ (MTP=0/2),
DeepSeek-V4-Flash, 27B EXL3 MTP=2 and 27B AWQ W4A8=1: 856 unique keys, 846
novel. Tuned/curated through the usual pipeline: 622 rows adopted (>=3 % over
both the current row and the heuristic; per-shape median 1.79x, best 15x),
appended to the 979 shipped rows. New rows per config: Qwen3.6 MTP=0 99,
MTP=2 160, DSV4 90, EXL3 MTP=2 176, AWQ W4A8 47, plus 82 small-M extensions.
The merge is additive: six shipped rows the fresh measurement found 3-10 %
slower than the heuristic are kept and listed in `provenance.json`
(`campaign.curate.existing_measured_slower`). Lookup-hit proof 1601/1601.

## Flash-Next mixed-batch expansion (2026-10-10)

Census (`PYTORCH_TUNABLEOP_RECORD_UNTUNED=1`) over the mixed prefill/decode
scenario (`stall_probe.py` inject/periodic/reverse with 6 decoders, plus the
four bench cells) and the hyper-connection GEMMs at every row count
M = 64..2048 step 64, which `rdna_ops._rdna_hc_mix` now pads to
(`VLLM_RDNA_HC_PAD_M`, default 64): a mixed step has an arbitrary row count
(chunk + decode tokens), and off a tuned row rocBLAS runs the skinny HC down
GEMM (N=336, K=10240) at ~4 TF/s (0.89 ms at M=512 vs 0.229 ms tuned).
Tuned/curated through the usual pipeline (>=3 % over current and heuristic):
198 adopted, 2 dropped, 104 census keys left on the heuristic. Lookup-hit
proof 979/979. Details in `provenance.json` (`campaign`).

## EXL3 27B expansion (2026-10-01)

The canonical set also covers `Qwen3.8-27B-exl3-3.00bpw` (mul1, TP=4, FA-RDNA2,
FULL_AND_PIECEWISE). 29 new rows were adopted (21 novel EXL3 K = 5120/1536/4352
shapes + 8 small-M decode extensions, each ≥ 3 % faster than both the current row
and the rocBLAS heuristic; best 2.35×) and 6 previously-shipped rows were dropped
(fresh measurement 4–7 % slower than the heuristic). EXL3 fp16 GEMM shapes
collide with none of the Flash-Next K = 2560/320/10240 set (0 collisions, 34
novel captured). MTP=2 was **not** captured — the EXL3 MTP head is not wired in
this tree, so the validated EXL3 serving config is MTP=0. Lookup-hit proof:
**783/783**. Prior set preserved at
`bench_results/2026-10-01_exl3-tunableop/repo-rows-premerge/`.

## Storage policy (mandatory)

* **One visible folder per rocBLAS build**, registered by name in
  `../profiles.json`. This directory (profile `rocm7.14-rocblas5.5`) is **the**
  canonical set for the fork's serving build (`5.5.0.cd957402` →
  `f30bb442e9b5`) and covers Flash-Next, 27B AWQ dense and EXL3 27B. The
  foreign-build table lives in the sibling profile `../rocm10-rocblas5.6/`.

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
# $VENV = the venv that loads this rocBLAS build; <tree> = this checkout.
source <tree>/tools/rdna2_028/tunableop_env.sh
configure_tunableop "$VENV/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/librocblas.so.5" \
                    "<tree>/tunableop"
```

It auto-selects profile `rocm7.14-rocblas5.5` from the loaded rocBLAS build
(matching `lib_sha256`) and sets a **lookup-only** environment
(`PYTORCH_TUNABLEOP_ENABLED=1`, `TUNING=0`, `PYTORCH_TUNABLEOP_FILENAME` at the
rank files). The launchers
(`scripts/serve_gfx1030_flashnext_mtp.sh`, `scripts/serve_gfx1030_27b_dense.sh`,
`scripts/serve_gfx1030_flashnext.sh`, `scripts/serve_gfx1030_full.sh`) already
do this. A healthy start logs:

```
TunableOp profile rocm7.14-rocblas5.5 auto-selected for rocBLAS build f30bb442e9b5.
TunableOp lookup enabled for profile rocm7.14-rocblas5.5 (rocBLAS f30bb442e9b5; rows: .../tunableop/rocm7.14-rocblas5.5); tuning off ...
```

Set `TUNABLEOP_PROFILE=rocm7.14-rocblas5.5` to force this profile explicitly;
the helper validates its `lib_sha256` against the loaded library and refuses to
boot on a mismatch (unless `TUNABLEOP_ALLOW_MISMATCH=1`, experiments only).

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
# <tree> = this checkout; WORK = a persistent scratch dir (never /tmp).
export VLLM_TREE=<tree> VENV=/path/to/venv WORK=$HOME/tunableop-rows
bash "$VLLM_TREE/tools/rdna2_028/tunableop_rows_pipeline.sh" all
python3 "$VLLM_TREE/tools/rdna2_028/curate_tunableop_rows.py" \
  --current-rows "$VLLM_TREE/tunableop/rocm7.14-rocblas5.5" \
  --scratch-rows "$WORK/scratch" \
  --heur "$WORK/measure/heuristic.json" \
  --cur "$WORK/measure/current.json" \
  --new "$WORK/measure/new.json" \
  --out "$WORK/curated" --adopt-margin 0.02 --drop-margin 0.03
```

The historical leak this directory fixes wrote to `/tmp/tunableop_results{0..3}.csv`
(the PyTorch default name) from drivers that enabled TunableOp without setting
`PYTORCH_TUNABLEOP_FILENAME`. All launch paths now source the helper, so no
script can leave rows in `/tmp` or the CWD.
