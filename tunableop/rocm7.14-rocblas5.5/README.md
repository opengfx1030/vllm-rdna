# TunableOp rows — profile rocm7.14-rocblas5.5

Build-specific rocBLAS solver rows for the serving stack on 4× V620 (gfx1030),
captured 2026-09-29 by re-tuning the production Flash-Next shapes with the
0.28.0 tree. The profile is registered in `../profiles.json`; its
`lib_sha256` is the `librocblas.so.5` sha256 (see `provenance.json`), and
`configure_tunableop` matches it against the loaded build.

## rdna_hint_flex / dense OLMoE-1B-7B (2026-10-26)

Census during the four bench cells (1k/512 and 2k/512 at c=1/c=8, max context
4096) of OLMoE-1B-7B-0924 as rdna_hint_flex `olmoe-r4` and `olmoe-b3.0` and as
dense fp16 on the RDNA2 HIP and Triton MoE backends, TP=1: 113 unique FP16
keys (attention, router and lm_head; routed experts do not reach rocBLAS on
any arm). Tuned with the M=1..8 extensions (145 shapes) on one V620 and curated
at the usual 3 % margin: 113 rows adopted (median 1.57x over the heuristic,
best 3.55x), 32 left on the heuristic, appended to each rank file:
**2707 rows/rank**. Lookup-hit proof 2613/2613 FP16 keys. Details in
`provenance.json` (`olmoe_hi2`).

## rdna_hint_flex / dense Qwen3.5-4B and Qwen2.5-0.5B (2026-10-25)

Census during the four bench cells of `tools/rdna/rdna_hint_flex/bench_serve.sh`
(Qwen3.5-4B rdna_hint_flex 3.0 bpw and its dense fp16 export at TP=1/2/4;
Qwen2.5-0.5B-Instruct rdna_hint_flex, dense export and AWQ at TP=1): 798 + 182
unique FP16 keys, tuned with the M=1..8 extensions (966 + 222 shapes) and
curated at the usual 3 % margin by `tools/rdna/rdna_hint_flex/tunableop_campaign.sh`:
605 + 136 rows adopted (median 1.60x / 1.26x over the heuristic, 1.21x for the
M >= 512 prefill shapes of the 4B), 734 appended to each rank file (7 adopted keys
already had a shipped row and keep it): **2594 rows/rank**. Lookup-hit proof
2500/2500 FP16 keys. Details in `provenance.json` (`campaign`).

## DFlash2 BF16 draft (2026-10-20)

The Flash-Next DFlash2 recipe keeps its draft in BF16 on gfx1030 because an
FP16 draft produced NaN candidates. A four-rank TunableOp census during 1k/512
and 16k/1k requests found 58 regular BF16 GEMM keys plus four strided-batched
keys per rank. The regular keys were tuned on a V620 with the profile's exact
rocBLAS build and added to all four rank files, taking each file from 1766 to
1824 rows. The strided-batched keys still use rocBLAS defaults.

All 58 regular shapes beat both the previous lookup and the heuristic by more
than 3% in isolated timing (median previous/new ratio 3.03); a lookup replay
confirmed 58/58 solver hits on each of the four V620s. The capture and tuning
files are under
`~/w4a8_runs/dflash2/{shape-census1,bf16-shapes-tune}/` on `par1-cs25`.

The final recipe validation passed greedy, perplexity (4.85), prefix, and
concurrency probes, with no GPU faults. For 1k/512 requests, output was
65.91 tok/s at c=1 and 223.81 aggregate tok/s at c=8. Two earlier untuned
c=8 runs gave 153.92 and 168.96 tok/s; two tuned c=8 runs gave 229.95 and
223.81. At 16k/1k c=8, untuned runs ranged 37.98–49.23 and tuned runs
39.54–46.38 tok/s. This range does not establish a long-context gain.

## DFlash2 two-token graph expansion (2026-10-20)

Added 36 missing BF16 TN keys per rank to this existing profile, bringing it
to **1860 rows/rank: 1766 FP16 + 94 BF16**. All previous rows and validators
are preserved byte-for-byte. Each rank uses its own newly tuned solvers;
the original rows remain shared. The separate ROCm 10 profile is unchanged.

The query widths are M=3/6/12/24 and vocab-head/selector widths M=2/4/8/16.
Across all four V620s, the 144 isolated measurements beat both canonical
lookup and the heuristic by more than 3% (minimum 2.354x, median 3.417x,
maximum 5.347x). Each GPU passed numerical checks and 36/36 lookup hits.
See the [shape inputs, measurements and reproduction commands](../../../bench_results/2026-10-20_dflash2-spec2-tunableop/README.md).

The full same-boot old1824/new1860 serving A/B passed all four standard
cells (80 benchmark requests per arm), greedy/PPL/prefix/normal-EOS/after-load
gates, and all eight prefix resets, with zero recorded GPU fault markers.
Aggregate output improved 57.7% / 20.1% at 1k/512 c=1/c=8 and 28.8% / 10.8%
at 16k/1k c=1/c=8; median ITL fell 37.2% / 22.6% and 36.1% / 21.0%.
The linked evidence reports all four metrics per cell, acceptance, and the
single-pair precision limitation for the modest long-c8 throughput gain.
Variable auxiliary-FC/mixed-prefill context widths and strided-batched BF16
GEMMs are outside this tuning set. The recipe keeps its existing depth default.

## What changed vs the harvested 903-row set

| | rows | file |
| --- | --- | --- |
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
(`tools/rdna/serve_gfx1030_flashnext_mtp.sh`,
`tools/rdna/serve_gfx1030_27b_dense.sh`,
`tools/rdna/serve_gfx1030_flashnext.sh`, `tools/rdna/serve_gfx1030_full.sh`)
already do this. A healthy start logs:

```text
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
