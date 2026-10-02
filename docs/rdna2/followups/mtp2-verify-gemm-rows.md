# Follow-up ticket: TunableOp rows for the MTP=2 verify-batch GEMM keys (M∈{3,6,12,24})

**Status:** ticket (draft PR, no code intended here) · scope: small, self-contained.

## Problem
The shared TunableOp set (`tunableop/rocm7.14-rocblas5.5/`, 783 rows/rank) was captured at MTP=0 (plus later MTP=2 mixed prefill+decode shapes). The **MTP=2 speculative-decode verify batches** issue GEMMs at **M = num_seqs × (1 + num_spec) ∈ {3, 6, 12, 24}**, whose keys are **not in the rows** → those GEMMs run on rocBLAS heuristics.

## Impact
The verify path is on the MTP=2 decode critical path (c=1–4; the cells where MTP=2 wins at c=1). Tuned rows elsewhere delivered 1.3–4× per shape, so the delta here is expected to be modest but real.

## Plan
1. Capture (`record_untuned` census) with MTP=2 on Flash-Next and 27B EXL3, c=1..4, loads 1k/512 and 16k/1k (fire the verify families).
2. Tune the novel keys (scratch copy, warm first, ≥10 iters, 25 ms/solver budget).
3. Measure heuristic/current/new same-process; curate — adopt new only if ≥3 % better; drop rows >3 % slower than the heuristic; fold-to-default where the default already wins.
4. Freeze into the single shared profile (`tunableop/rocm7.14-rocblas5.5/`) + update `provenance.json`; commit.
5. Verify: `verify_tunableop_lookup.py` (row count + 100 % hits) and one MTP=2 A/B at 16k/1k c=1 showing the delta.

## Acceptance
- Verify-batch keys present in the rows; lookup 100 % hits at the new count.
- A measured MTP=2 c=1 delta at 16k/1k; if none, document the no-gain result and close.

## Notes
- Storage policy: rows are keyed by the rocBLAS build hash; never `/tmp` or CWD; profiles live under `tunableop/<profile>/` (see the top-level README).
- Pipeline to reuse: `tools/rdna2_028/{campaign_*|exl3_capture.sh|exl3_tunableop_campaign.sh|exl3_freezer.py|verify_tunableop_lookup.py}`.
