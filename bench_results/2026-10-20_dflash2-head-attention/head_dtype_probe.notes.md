# DFlash2 shared-head dtype diagnostic

Persistent raw results are
`/home/chenco_adm/w4a8_runs/port-v031/dflash2-profile-refresh-tuned1860/head-rank{0,1,2,3}.json`
on `par1-cs25`. Copied local sources are
`/private/tmp/dflash2-2026-10-20-evidence/profile-refresh/head-rank{0,1,2,3}.json`.

[`head_dtype_probe.csv`](head_dtype_probe.csv) contains 32 rows: each of four
TP ranks (physical HIP 6–9) times M=2,3,4,6,8,12,16,24. The rank-specific
TunableOp profile was enabled lookup-only. Each JSON source is retained in
`profile-refresh/head-rank{0..3}.json`.

The target checkpoint stores the 248320×2560 head in BF16, while the FP16
target runtime loads it as FP16. The current BF16 DFlash2 `LogitsProcessor`
converts its shared target head and hidden activations back to BF16 on every
candidate call. The proposed path uses the existing FP16 head directly; the
ROCm unquantized dispatcher chooses `gemv_f16_rdna2` for M≤8 under the recipe's
default skinny-GEMM setting and rocBLAS at larger M. The probe's `current`
timing includes the weight recast and GEMM. `bf16_cached_weight_gemm_ms`
excludes the recast and shows its separate cost. GPU-event values are medians
of 30 repetitions after five warmups; this is isolated head work, **not**
end-to-end draft or serving speed.

Across ranks, mean median current→proposed time is 3.512→0.757 ms at M=2,
3.729→0.773 ms at M=8, 3.721→1.116 ms at M=16, and 3.915→1.298 ms at M=24.
The four rank shards contain no nonfinite or FP16-overflowing weights; each
shard has 158,924,800 weights and 49,553–55,279 BF16 checkpoint values
change when converted to FP16 and back. Synthetic inputs use the checkpoint's
BF16 final RMSNorm weight and seed 12345; they are **not model activations**.
Local top-1 matched in 292/300 synthetic rows across all shapes/ranks. Local
top-16 sets overlap almost completely, but order often changes. FP16 logits
have lower mean absolute error to the FP32 projection of the FP16 runtime
operands. These observations do not establish candidate parity or serving
quality. Use target-only comparisons, normal-EOS/PPL/acceptance gates and a
same-client, same-boot serving A/B before promotion.

## Conservative output range check

For an ideal RMSNorm with finite input `x`, epsilon ≥0, width D and per-channel
scale `g`, `||RMSNorm(x,g)||₂ ≤ sqrt(D) max|g|`. Cauchy–Schwarz then gives
`|h · w_row| ≤ D max|g| max|w_row|`. The actual rank-local head maxima from
the JSON sources yield ideal absolute-logit bounds of 8044.375 (rank 0),
4117.5 (rank 1), 4460.625 (rank 2) and 3240.625 (rank 3). Even after a
conservative 2% arithmetic-rounding allowance, the largest is 8205.263,
about 8.0× below FP16 max finite 65504. The CSV repeats these derived bounds
for each rank/shape. This is an analytical bound conditional on finite RMSNorm
input and the standard RMSNorm formula, with a practical rounding allowance;
it does not replace a runtime finite check or establish candidate parity.
