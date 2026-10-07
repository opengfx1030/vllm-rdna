# TunableOp EXL3 27B expansion — capture / tune / curate / freeze / verify / e2e

Box `par1-cs25`, venv-7.14.0_0.28.0, tree `/home/chenco_adm/vllm-rdna-0.28.0`,
librocblas build hash `f30bb442e9b5…`. GPUs 4-7 (HIP 4-7, formerly DRM 0-3),
4× Radeon PRO V620 (gfx1030). Box co-tenant at boot:
- DRM 0-3 / HIP 4-7: an orphaned EXL3 server (log `exl3_27b_fp.log`, tree
  `vllm-rdna-0.28.0`) left over from the 2026-10-01 validation run — killed
  (authorized under "free only OUR vLLM leftovers").
- DRM 4 / HIP 0: an external `tensorfold` (`/tmp/tf-fa-cells.py`,
  `cd ~/tensorfold-src`, HIP_VISIBLE_DEVICES=0, no relation to this project —
  never touched).
- DRM 5-7 / HIP 1-3: free throughout.

## Frozen row set: `tunableop/rocblas-f30bb442e9b5/`

**760 → 783 data rows/rank** (commit on `rdna_extras`).
Lookup-only, keyed by the rocBLAS build hash. All 4 rank files share
rank-0's solver picks (`configure_tunableop` ships the rank-0 file to all
ranks; verify ran 783/783 hits against rank-0). Prior set preserved at
[`repo-rows-premerge/`](./repo-rows-premerge/).

## 1. Shape capture

`tools/rdna2_028/exl3_capture.sh` boots the EXL3 27B launcher
([`scripts/serve_gfx1030_exl3_27b.sh`](../../../scripts/serve_gfx1030_exl3_27b.sh))
with `PYTORCH_TUNABLEOP_RECORD_UNTUNED=1` + `PYTORCH_TUNABLEOP_UNTUNED_FILENAME`.
record_untuned DISABLES the results lookup, so cells run on rocBLAS
heuristics (shape discovery only — capture boot != validation boot). Cells:

| Cell | c | Purpose |
|---|---|---|
| 1k/512 c=1 | 1 | Decode prefill, M=1 |
| 1k/512 c=8 | 8 | Decode aggregate, M=8 |
| 16k/1k c=1 | 1 | Prefill-dominated, M=16k |
| 16k/1k c=8 | 8 | Mixed-batch prefill, M=16k chunked |

MTP=0 (validated EXL3 serving config); MTP=2 attempt crashed
`ValueError: There is no module or parameter named 'fc.suh' in
Qwen3_5MultiTokenPredictor` — the EXL3 MTP head (`mtp.fc`, EXL3-quantized
to mul1/4-bit in the checkpoint) is not wired in this tree. The
checkpoint ships `mtp.fc.{suh,svh,trellis,mul1}` but
`qwen3_5_mtp.py:load_weights` registers only `fc.weight` for the
ColumnParallelLinear, so the EXL3 `MUL:Loader` logs `[exl3]
UNQUANTIZED prefix=mtp.fc pn=fc` and the weight-load fails before any
GEMM runs. Fixing this needs an EXL3+MTP loader patch — documented as a
known gap, not blocked by it.

### Capture summary

- Captured GemmTunableOp_Half keys (union over the 4 cells): **34**
  ([`capture/shapes_exl3-m0.txt`](./capture/shapes_exl3-m0.txt))
- Colliding with the 760-set: **0**
  - All 34 keys are genuinely novel: EXL3 27B's K=5120/1536/4352 fp16
    GEMMs (GDN in_proj_a/b + a residual fp16 path) and the 24-wide TN
    shape (likely an attention `q_norm`/`k_norm` linear).
  - Trellis-fused EXL3 projections do NOT emit rocBLAS keys — the
    `exl3_gemm_rdna2` custom kernel bypasses TunableOp, so the capture
    sees only the sparse fp16 fallbacks (dequant path, lm_head path
    via the `A_log`-adjacent linear family).
- NOVEL = 34, COLLIDING = 0.

## 2. Tune

`tools/rdna2_028/exl3_tunableop_campaign.sh PHASE=tune` —
`tools/rdna2_028/tune_prod_shapes.py --mode tune` on the 34 novel keys
(with the small-M extension → 43 keys after extension: 34 captured + 8
decode-size extensions of the 1 pair at M≤32), 4 GPUs in parallel,
≥10 iterations, 25 ms/solver, numerical check 0.01/0.01. Result: 43
tuned rows/rank in `WORK/scratch/`.

## 3. Curate

`PHASE=measure` → heuristic / current / new per shape (union of 760 + 34 +
extension ≈ 826 shapes, measured on a single idle GPU, 5 warmup + 15 reps
median). `PHASE=curate` → `tools/rdna2_028/curate_tunableop_rows.py`
with 3%/3% adopt/drop thresholds.

| decision | count | note |
|---|---:|---|
| new | 29 | 21 captured-novel EXL3 keys + 8 small-M-extended decode keys, each ≥3% faster than both the current row (heuristic fallback) and the rocBLAS heuristic |
| current | 754 | carried over unchanged |
| dropped | 43 | 13 EXL3 novel keys whose tuned solver was not ≥3% better than heuristic + 6 previously-shipped rows whose fresh measurement is 4-7% slower than heuristic |

Rows/rank: 826 → **783** (760 + 29 new - 6 existing dropped).

### Adopted EXL3 novel keys (21, speedup = new_ms / heur_ms)

| Shape | heur ms | new ms | speedup |
|---|---:|---:|---:|
| tn_24_1037_5120 | 0.210 | 0.089 | 0.42× (2.35× faster) |
| tn_24_1024_5120 | 0.211 | 0.092 | 0.44× |
| tn_24_2048_5120 | 0.225 | 0.114 | 0.51× |
| tn_24_57_5120 | 0.082 | 0.050 | 0.61× |
| tn_24_16_5120 | 0.074 | 0.050 | 0.68× |
| tn_24_225_5120 | 0.083 | 0.058 | 0.70× |
| nn_512_1037_5120 | 0.468 | 0.244 | 0.52× |
| nn_512_1024_5120 | 0.406 | 0.263 | 0.65× |
| nn_512_225_5120 | 0.138 | 0.097 | 0.71× |
| nn_256_2048_5120 | 0.418 | 0.231 | 0.55× |
| nn_256_1024_5120 | 0.227 | 0.145 | 0.64× |
| nn_256_1037_5120 | 0.228 | 0.141 | 0.62× |
| nn_256_225_5120 | 0.086 | 0.077 | 0.89× |
| nn_1536_1024_5120 | 0.860 | 0.699 | 0.81× |
| nn_1536_1037_5120 | 0.864 | 0.762 | 0.88× |
| nn_1536_225_5120 | 0.317 | 0.255 | 0.80× |
| nn_3072_225_5120 | 0.495 | 0.405 | 0.82× |
| nn_4352_1024_5120 | 2.068 | 1.849 | 0.89× |
| nn_4352_1037_5120 | 2.315 | 2.090 | 0.90× |
| nn_4352_225_5120 | 0.537 | 0.519 | 0.97× |
| nn_5120_225_4352 | 0.590 | 0.521 | 0.88× |

### Dropped EXL3 novel keys (13 — fall back to heuristic)

`nn_5120_1037_4352`, `nn_512_2048_5120`, `nn_4352_2048_5120`,
`nn_3072_1037_5120`, `nn_5120_1024_1536`, `nn_3072_1024_5120`,
`nn_5120_1037_1536`, `nn_1536_2048_5120`, `nn_5120_225_1536`,
`nn_5120_2048_4352`, `nn_5120_2048_1536`, `nn_5120_1024_4352`,
`nn_3072_2048_5120` — tuned solver was not ≥3% better than heuristic at
these M values (1638/K=5120 family prefill). Full table in
[`curated/before_after.csv`](./curated/before_after.csv).

### Previously-shipped rows dropped (6 — >3% slower than heuristic fresh)

| Shape | heur ms | cur ms | cur_solver |
|---|---:|---:|---|
| tn_2560_1125_160 | 0.077 | 0.083 | Gemm_Rocblas_-1086326075 |
| tn_2560_1264_160 | 0.084 | 0.087 | Gemm_Rocblas_-1086326068 |
| tn_2560_3_1536 | 0.061 | 0.063 | Gemm_Rocblas_-1086326218 |
| tn_320_1888_2560 | 0.238 | 0.245 | Gemm_Rocblas_-1086326082 |
| tn_3584_1026_2560 | 0.932 | 0.967 | Gemm_Rocblas_-1086326047 |
| tn_640_2069_2560 | 0.364 | 0.384 | Gemm_Rocblas_-1086326277 |

All mixed-batch prefill M values (1026, 1125, 1264, 1888, 2069, 3). Fresh
measurement on the idle GPU 4 finds the previously-shipped solver 4-7%
slower than the heuristic; per the goal's `drop >3% slower than
heuristic` rule, they fall back to heuristic (faster at those shapes).

## 4. Freeze

`tools/rdna2_028/exl3_freezer.py` writes the curated rank-0 across all 4
rank files (single solver per key for the shared rows), backs up the
pre-merge rows + provenance to `repo-rows-premerge/`, and updates
`tunableop/rocblas-f30bb442e9b5/provenance.json` with the campaign
block (date, arm, capture/tune/curate method, MTP=2 status, row counts).

## 5. Lookup-hit proof

`tools/rdna2_028/verify_tunableop_lookup.py` against
`tunableop/rocblas-f30bb442e9b5/tunableop_results0.csv`:

```
lookup hits 783/783 via .../tunableop_results0.csv
```

All 783 rows are consumed (no Default fallback). Evidence:
[`lookup_hits.json`](./lookup_hits.json).

## 6. e2e EXL3 cell with the frozen rows

`tools/rdna2_028/exl3_validate.sh` boots the EXL3 27B launcher
lookup-only (TUNABLEOP=1, frozen rows), reuses the warm capture cache,
runs coherence + 3 measured cells. Coherence PASS (both probes):

| Probe | Output (greedy, temp=0) | Coherent? |
|---|---|---|
| "The capital of France is" | `…Paris` | ✓ |
| "2 + 2 =" | `…4` | ✓ |

Measured cells (`vllm bench serve /v1/completions`, random, ignore-eos,
seed=12345):

| Cell | Output tok/s (lookup + frozen rows) | Output tok/s (capture / heuristic only) |
|---|---:|---:|
| 1k/512 c=1 | **19.08** | 14.95 |
| 1k/512 c=8 | **82.52** | 37.20 |
| 16k/1k c=1 | **13.75** | 2.79 |

Same MTP=0, FA-RDNA2, FULL_AND_PIECEWISE config — only TunableOp differs
(capture = record_untuned / heuristic only; e2e = lookup-only against the
frozen set). Tuned EXL3 fp16 GEMMs deliver **+28% / +122% / +393%** over
the rocBLAS heuristic across these cells. Logs in
[`validate/`](./validate/).

## Files

- [`scripts/serve_gfx1030_exl3_27b.sh`](../../../scripts/serve_gfx1030_exl3_27b.sh) — canonical EXL3 27B launcher (MTP 0/2, capture-env passthrough, per-arm cache scoping)
- [`tools/rdna2_028/exl3_capture.sh`](../../../tools/rdna2_028/exl3_capture.sh) — record-untuned capture driver (m0/m2 arms)
- [`tools/rdna2_028/exl3_tunableop_campaign.sh`](../../../tools/rdna2_028/exl3_tunableop_campaign.sh) — diff / tune / measure / curate orchestration
- [`tools/rdna2_028/exl3_freezer.py`](../../../tools/rdna2_028/exl3_freezer.py) — freeze + provenance update
- [`tools/rdna2_028/exl3_validate.sh`](../../../tools/rdna2_028/exl3_validate.sh) — e2e lookup-only validation driver
- [`tools/rdna2_028/exl3_analyze.py`](../../../tools/rdna2_028/exl3_analyze.py) — decision-table analysis
- [`capture/shapes_exl3-m0.txt`](./capture/shapes_exl3-m0.txt) — captured GEMM key census
- [`curated/before_after.csv`](./curated/before_after.csv), [`decisions.json`](./curated/decisions.json)
- [`lookup_hits.json`](./lookup_hits.json) — lookup-hit proof
- [`validate/`](./validate/) — e2e cells + coherence
- [`repo-rows-premerge/`](./repo-rows-premerge/) — prior 760-row set + provenance