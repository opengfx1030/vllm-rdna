# MTP FP16 TunableOp rows — rocblas-f30bb442e9b5

Online-tuned rocBLAS rows for the MTP serving stack on 4× V620 (gfx1030), captured
during MTP=2 serving with the 0.28.0 tree (`rdna_extras`). They cover the dense
FP16 projections at the MTP verify batch sizes (M = 16/24/32) plus decode and
chunked-prefill shapes. Without them those shapes land on latency-bound small-K
rocBLAS kernels (`Cijk_MT32x32x8`, K-depth 8 → 320 K-iterations, measured
< 1 TFLOPS), which is the largest single item in the MTP step budget.

## Use

```bash
cp tunableop_results{0..3}.csv ~/.cache/tunableop/
export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1
# PYTORCH_TUNABLEOP_FILENAME defaults to ~/.cache/tunableop/tunableop_results.csv
# in tools/rdna2_028/serve_ours_028.sh; only the two enable flags need setting.
```

`TUNING=1` re-tunes only shapes missing from the table.

Never place these files under `/tmp` (wiped on reboot) or in a run-specific CWD;
`~/.cache/tunableop/` is the canonical location.

## Measured (4× V620, TP4, Qwen3.8-Flash-Next-AWQ-W4A16, MTP=2, c=8, seed 12345)

| cell | MTP-0 | MTP-2 (no rows) | MTP-2 + these rows |
| --- | ---: | ---: | ---: |
| 8×1k/512 | 166.3 tok/s / 40.8 ms | 129.4 / 50.1 | 135.5 / 43.4 |
| 8×16k/1k | 67.3 / 75.6 | 58.7 / 84.7 | **73.8 / 64.9** |

In-protocol A/B (same driver, same protocol, rows the only change): 140.5 vs
114.8 tok/s (+22.4 %). These rows require the draft-decode cudagraph fix
(`30632b2fa`, branch `rdna_extras`) to be present.

## Caveats

- Solution IDs are build-specific: never reuse across a different rocBLAS build
  (compare the library sha256 in `provenance.json`).
- Tuned algorithms may change the floating-point reduction order. Outputs are not
  bit-identical and greedy acceptance shifts slightly (observed 1.81 vs 2.43 at
  8×1k/512). Validation here was 8/8 cells plus coherent probes, not bit equality.
- Cold tuning is impractical online (>20 min of inline tuning per workload);
  ship these rows instead.
- The first requests after a fresh start JIT the QSA Triton kernels; that first
  pass is slow (tens of seconds) — wait for it rather than treating it as a hang.

## Regeneration

```bash
# one cold serving run with tuning enabled writes
# ~/.cache/tunableop/tunableop_results{0..3}.csv at process exit
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 <launch>
# then qualify: run the benchmark cells with and without the rows and compare
```
