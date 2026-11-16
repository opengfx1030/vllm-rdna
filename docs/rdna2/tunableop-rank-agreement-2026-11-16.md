# TunableOp rank files must agree per key (2026-11-16)

Why the rows of `tunableop/rocm7.14-rocblas5.5` (and the derived `rocm7.14-rocblas5.5-hi2-27b-tp2`) were rewritten so that every rank file of a profile names one solution per key, how to check it, and what it cost. Box par1-cs25 (6x V620, one PCIe switch tree), Flash-Next MTP2 at TP4 on HIP 2-5, tree `818fa99830`. The replay and analysis tooling (`tools/rdna/eplb/`) and the long form of this note (section 11) are in `docs/rdna2/eplb-lean-design-2026-11-16.md` once the lean EPLB branch lands.

## Symptom

The EP4 expert-id consensus of the lean EPLB counters found rank 2 hashing different top-k ids than ranks 0, 1 and 3 in forward passes of exactly 1023 prefill rows. Replayed at c=1 with the prefix cache defeated (one `cache_salt` per request), 700 of 48,480 compared records disagree, always 3 against 1 with rank 2 odd. 68 of 68 requests that contain a 1023-row pass are affected (a prompt of k x 1024 tokens ends in one) and 0 of 75 that do not: prompts of 777, 1023, 1025, 1030 and 2047 tokens, and the 1024- and 2048-row chunks of a 16k prompt, are clean.

## Cause

A worker on device ordinal `r` reads `tunableop_results<r>.csv`. 531 of the 3,330 keys named different rocBLAS solutions on different ranks (ranks 4 to 7 are copies of 0 to 3), among them the replicated MoE router GEMM: `tn_512_1023_2560_ld_2560_2560_512` is `...046` on ranks 0, 1 and 3 and `...044` on rank 2, and 28 of the 126 router keys were split. Two solutions of one GEMM accumulate in a different order.

- One process per ordinal, the router GEMM on seeded fp16 inputs: ordinal 2 differs from 0, 1 and 3 in 1,678 of 523,776 logits (one ulp) at 1023 rows; 3, 24, 512, 1022, 1024, 1025, 1036, 1045, 1197, 2047 and 2048 rows are bit-identical on all ordinals, and ordinals 0, 1 and 3 are identical at every shape.
- The same ordinal on another card (`HIP_VISIBLE_DEVICES=5,4,3,2`) gives identical bits, and a boot with that order, rdna_ar off, the hierarchical all-reduce off and the shared-expert stream off still has rank 2 odd in 265 of 265 records: the odd rank follows the rank id, not the card and not the all-reduce.
- With `VLLM_RDNA2_MOE_DETERMINISTIC=1` the same 8 (pass, layer) cells disagree in each of 24 replays of one prompt (with the production epilogue the layer set moves between replays): the disagreement is a deterministic function of the input and the rank's rows.

## Fix

`tools/rdna2_028/tunableop_agree.py write <profile>` keeps, per key, the solution with the lowest recorded time (ties to the lowest rank) and replaces only the lines that name another solution; it adds and drops no key, so `tn_2048_12_2560_ld_2560_2560_2048` stays out of the set (see the profile README). 2,124 lines change in the eight files of `rocm7.14-rocblas5.5` (316, 282, 290 and 174 on ranks 0 to 3, the same on 4 to 7) and 1,062 in the four files of the hi2 profile; `rocm10-rocblas5.6` already agreed. `check` lists the split keys, and `tests/tools/rdna/test_tunableop_rank_agree.py` fails on any profile whose rank files disagree and pins the router key. No production code changed.

## Acceptance (Flash-Next MTP2, TP4, HIP 2-5, agreed rows)

| boot | greedy / PPL | deterministic concurrency probe (failing / near-tie / soft / identical of 48) | cross-rank records, 1023-row passes |
|---|---|---|---|
| shipped rows, deterministic epilogue (baseline) | PASS / 4.82 | PASS 0 / 10 / 2 / 36 | 208 of 60,384 disagree |
| agreed, production epilogue | PASS / 4.82 | PASS 0 / 17 / 1 / 30 | 0 of 79,872; 54 passes |
| agreed, deterministic, flat all-reduce | PASS / 4.86 | PASS 0 / 10 / 2 / 36 (repeats 0 / 11 / 3 / 34, 0 / 12 / 1 / 35) | 0 of 130,656; 14 passes |
| agreed, deterministic, hierarchical all-reduce forced 2 + 2 | PASS / 4.86 | PASS 0 / 14 / 0 / 34 (repeats 0 / 15 / 0 / 33, 0 / 13 / 1 / 34) | 0 of 133,248; 14 passes |

All boots: 0 KFD evicted ms, 0 fault markers. All 3,330 rows of each agreed rank file hit on its own card (`verify_tunableop_lookup.py`, HIP 2-5, 16 runs), so every chosen solution runs on the four cards. TP8 itself cannot run on this six-card box: the hierarchical all-reduce was exercised as a forced 2 + 2 at TP4, and the files of ranks 4 to 7 are covered by `check` and the CPU test.

## Cost

- The router key of the fix is a prefill key: at 1023 rows the measured call takes 148.7, 148.3, 162.0 and 151.5 us on ranks 0 to 3 before and 149.6, 149.6, 154.0 and 151.7 us after (only rank 2 changes, 8 us faster, 0.4 ms per prompt).
- ABBA at c=1 (counters off, two boots per side, 24 requests of 1024 + 128 tokens and 6 of 16,384 + 128): median ITL -0.06 % at 1k and -0.21 % at 16k, TTFT +0.10 % at both (+0.4 ms of 398 ms, +6.0 ms of 5.92 s); same-arm spreads 0 to 0.06 ms of ITL and 0.5 to 3.9 ms of TTFT. TPOT and tok/s move by up to 5 % between boots with the MTP acceptance length (2.04 to 2.21 at 1k), not with the rows.
- Re-measured on the same card, the rewritten rows are -0.1 to +0.7 % (decode-class keys, about 1 us per call), +1.2 to +1.6 % (mid) and +1.6 to +2.5 % (prefill) slower summed over the rewritten keys than each rank's own tuned row: the price of one solution for four cards and of picking the lowest of four noisy tuner times `[INFERENCE]`.
- Every candidate was also run on every card (`vmatrix`: the four shipped rank files on the four cards, median of 40 timed calls per key, all 3,330 rows hit in all 32 runs, so any solution any rank file names for a key runs on all four cards). Summed over the 531 split keys the agreed rows cost +0.14 % (decode-class), +1.16 % (mid) and +1.51 % (prefill) against each rank's own tuned row, and the best measured candidate per key would cost -0.13 %, +0.31 % and +0.16 %. That choice changes numerics again and was not made: it needs the acceptance above and the ABBA repeated, for at most the 0.1 % of TTFT that the ABBA bounds.

## Keeping it true

Run `python tools/rdna2_028/tunableop_agree.py check tunableop/<profile>` after a re-tune or a harvest that touched one rank file, `write` to repair it, and accept the new rows with the deterministic concurrency probe on the flat and the hierarchical all-reduce, the PPL and greedy probes and 0 KFD evictions. A launcher-side boot warning for disagreeing rank files is not implemented (`tools/rdna2_028/tunableop_env.sh` is production code).
