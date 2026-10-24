# Native MTP2 path audit on the BF16-core build

## Scope and conclusion

This audit separates functional correctness from performance attribution. The
native `method=mtp` serving route on the c390 BF16-core build passed normal-EOS,
greedy, PPL, prefix-cache, after-load, request-success, and fault checks. It
showed no failed requests or recorded GPU faults. That establishes that native
MTP2 still runs correctly in the tested configuration; it does not establish
that all shared code changes are performance-neutral.

The new `gemv_bf16_rdna2` implementation is guarded by gfx10, BF16 operands,
the existing skinny-GEMM flag, an M of 3/4/24, and a measured shape whitelist.
The bounded native trace observed zero `gemv_bf16_rdna2` launches and 24,383
`gemv_f16_rdna2` launches. The trace covered TP0/TP2/TP3, not TP1, and includes
startup/capture plus tiny c1/c8 diagnostic requests. Therefore it is direct
evidence for the observed workers, not proof that the BF16 kernel can never be
used by native MTP2.

Other shared surfaces changed in the branch:

- FA-RDNA2 gained BF16 and noncausal handling. The native test configuration
  used FP16 causal attention, which stays on the existing dispatch branch.
- BF16 TunableOp rows were appended; existing FP16 rows were retained.
- Qwen4 gained an auxiliary-hidden-state capture hook for DFlash2. It is
  conditional on the requested auxiliary layers and is not intended for native
  MTP2.
- The Mono QSA stride patch widens stride arguments and removes narrowing
  casts; it is a stride-safety change, not a scheduling/performance change.

## Same-boot 16k/1k long-cell results

All runs below used boot `ccd3c480-76df-44a2-a67f-ee1db46a46e5`, the same model
snapshot, TP4/EP4 and HIP6-9. Aggregate throughput is the benchmark's total
output-token rate; per-request decode and acceptance are reported separately.
Acceptance is draft acceptance and affects speculative speed, not output
validity by itself.

| Arm | Cell | Decode tok/s/request | Aggregate output tok/s | Acceptance | Requests |
| --- | --- | ---: | ---: | ---: | ---: |
| Old DFlash2 53a8 | c1 n8 | 84.16 | 51.13 | 64.46% | 8/8 |
| BF16-core c390 DFlash2 | c1 n8 | 91.51 | 53.77 | 58.40% | 8/8 |
| Old DFlash2 53a8 | c8 n32 | 14.39 | 90.26 | 60.84% | 32/32 |
| BF16-core c390 DFlash2 | c8 n32 | 14.29 | 90.88 | 59.32% | 32/32 |
| Native MTP2, first run | c1 n8 | 84.85 | 51.68 | 53.40% | 8/8 |
| Native MTP2, repeat | c1 n8 | 79.90 | 49.82 | 47.45% | 8/8 |
| Native MTP2, first run | c8 n32 | 14.48 | 90.85 | 57.55% | 32/32 |
| Native MTP2, repeat | c8 n32 | 14.74 | 92.55 | 65.25% | 32/32 |

Both same-boot DFlash arms and both native runs passed the recorded sanity and
fault gates. The DFlash c8 aggregate result is effectively tied with native;
the c1 result is higher than the two current native runs, but the requested
margin beyond run variance is not established.

The two native c1 runs used the same CLI seed/settings and had matching token
length arrays, but their saved generated texts were 0/8 exact matches and raw
prompt IDs/hashes were not saved. Acceptance varied from 53.40% to 47.45% on
the repeat. This prevents a strong causal attribution of the historical c1
gap. The current native c8 results span 90.85-92.55 aggregate tok/s, including
the earlier 92.47 baseline.

## Evidence

- Native first-run and repeat summaries/details are retained under
  `/private/tmp/dflash2-2026-10-20-evidence/serve-corebf16-currentboot-native-mtp2-long/`
  and `.../serve-corebf16-currentboot-native-mtp2-repeat-long/`.
- Same-boot DFlash old/new summaries/details are under
  `/private/tmp/dflash2-2026-10-20-evidence/serve-corebf16-old53a8-long/` and
  `.../serve-corebf16-newc390-long/`.
- Kernel trace summary:
  `/private/tmp/dflash2-2026-10-20-evidence/corebf16-native-kerneltrace-driver/native-kerneltrace-summary.json`.
  Raw trace SHA256 is `c7d415d2dc07d7fb6c21d33962b8ec1199efe48ba66f0b96f50615fb5e96f6f2`;
  it contains 244,168 lines, 0 BF16-GEMV matches, and 24,383 FP16-GEMV matches.

No performance-neutrality claim is made for the full shared code diff. The
current evidence says the new BF16 GEMV is not on the observed native MTP2
route; prompt-hashed repeated comparisons and wider worker coverage would be
needed for a definitive no-regression claim.
