# Long-context BF16 core timing diagnostic

These rows compare the original tuned head/S16 FA graph-internal phase3 run with the new BF16 core build, on the same boot, TP4 HIP 6–9, static DFlash2 depth 2, at 16,384/1,024. Both runs used `VLLM_RDNA_OP_TIMING=3`, which synchronizes sampled graph replays; their benchmark throughput is diagnostic and **must not be presented as unprofiled serving speed**. The old timing records are in `../long_phase/`. New source, profile and boot manifests match before and after the run.

| Pure decode geometry | Graph samples old/new | Body mean old → new | Graph total mean old → new | Outer draft replay old → new | Outer phase total old → new |
| --- | ---: | ---: | ---: | ---: | ---: |
| c1, tokens=3, reqs=1 | 9 / 9 | 4.507 → 1.512 ms (−66.5%) | 5.827 → 2.722 ms (−53.3%) | 5.621 → 2.569 ms (−54.3%) | 26.429 → 23.043 ms (−12.8%) |
| c8, tokens=24, reqs=8 | 9 / 10 | 6.234 → 4.242 ms (−32.0%) | 7.941 → 5.933 ms (−25.3%) | 7.671 → 5.656 ms (−26.3%) | 58.125 → 55.210 ms (−5.0%) |

The corresponding pure outer-phase TP0 step counts were 390/386 at c1 and 303/342 at c8 (old/new). Outer means are weighted by the PhaseTimer `n` count. The first complete flush timestamp in each benchmark cell is excluded because the timer can carry warmup or a previous cell into that flush. `corebf16_outer_phase_long.csv` retains every remaining exact-token-count bucket: `pure_decode` means 3 tokens at c1 or 24 at c8; `partial_decode` is below 1,024 tokens otherwise; `prefill_or_mixed` is at least 1,024 tokens. These mixed/partial rows are not pooled into the pure comparison.

`corebf16_graph_internal_samples.csv` contains each matched TP0 pure graph sample within the timed benchmark windows, with `reqs=1` at c1 or `reqs=8` at c8. `corebf16_graph_internal_long.csv` has their means and medians. `old_vs_corebf16_phase.csv` contains all old/new fields and percentage changes. Internal graph spans describe the replay and cannot be added to outer PhaseTimer spans. The outer `total` is a measured event boundary, not a sum of independently rounded fields. `phase_raw_sha256.csv` points to the complete copied logs, summary, and driver manifests. Unprofiled full 16k/1k DFlash2 versus native MTP2 controls are still required for the user’s speed objective.
