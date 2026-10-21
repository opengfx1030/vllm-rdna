# Long-context DFlash2 phase diagnostics and BF16 GEMV prototype

These are diagnostic runs on the same V620 HIP 6–9 host and boot, AWQ
Flash-Next TP4, BF16 PLE, MonoKernel, tuned 1860-row rocBLAS registry,
DFlash2 depth 2, and the candidate FP16 shared head plus FA RDNA2 S16
attention. Both used 16k/1k with one request at c1 and eight at c8.
The launchers set `VLLM_RDNA_PHASE_TIMING=50`: phase0 set
`VLLM_RDNA_OP_TIMING=0`; phase3 set it to `3` to retain external event
nodes inside DFlash's captured graph. Phase3 synchronizes sampled replays.
The benchmark JSON throughput from these runs is **diagnostic only** and is
not a production serving comparison.

## Files and definitions

- `outer_phase_long.csv`: TP0 PhaseTimer rows, grouped by cell and exact
  `tokens` count. Means are weighted by each log row's `n` measured steps.
  `pure_decode` means `tokens=3` for c1 and `tokens=24` for c8. Other
  `tokens<1024` are `partial_decode`; `tokens>=1024` are
  `prefill_or_mixed`. The first timestamp cluster after each cell begins is
  excluded because its 50-step flush can span prior warmup or the previous
  cell. The `steps` column is the sum of `n` after exclusion.
- `graph_internal_samples.csv`: each TP0 pure-decode captured-graph sample
  inside the phase3 benchmark window, including its timestamp and event
  spans. `graph_internal_long.csv` gives unweighted sample means and medians.
  Nine c1 and nine c8 graph samples remained. `body_fraction` is body time
  divided by graph `total` per sampled replay. Captured-graph spans and
  outer PhaseTimer spans are separate measurements; do not add them.
- `bf16_skinny_micro.csv`: 24 isolated graph-replay GEMV cases from a
  standalone, non-production HIP prototype versus lookup-only tuned rocBLAS
  on rank 0. Inputs are seeded synthetic BF16 tensors; the test checks
  output against BF16 rocBLAS with `rtol=atol=0.04`. Its full-range BF16
  case passed. The largest observed scaled difference was 0.007634 and
  largest absolute difference 0.015625. The repeated graph uses the same
  activations and weights, allowing warm caches; it does not prove serving
  speed or DFlash proposal parity. The prototype requires 16-byte aligned,
  dense operands and wave32.
- `bf16_skinny_rowtile.csv`: isolated M24 rerun of the eight shapes with
  row tiles 0 (original), 4, 8, and 12. All 32 cases passed the same BF16
  closeness check and the full-range case passed. `core_policy_selected`
  marks only the two small-output variants selected for the pending core
  build: N640,K2560 uses row tile 8 (19.24 versus 42.75 microseconds for
  tile 0), and N256,K2560 uses row tile 4 (10.96 versus 42.11 microseconds).
  These small N shapes showed large isolated margins from reusing the input
  rows across output waves. Other tiles show smaller or inconsistent gains
  across shapes and remain on the original path pending serving evidence.
  This selected policy has not yet been qualified in a serving build.
- `raw_sha256.csv`: hashes and local/remote locations of raw logs, summaries,
  boot IDs, and before/after source manifests. Copies of the two driver
  manifests and boot IDs are committed here. Each driver's before/after
  manifest and boot ID match exactly. Raw logs remain on the remote host.

## Pure decode results

| Diagnostic | Cell | Weighted outer forward | Weighted outer draft | Weighted outer total | Internal graph body | Internal graph total | Body share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase0 outer | c1 | 19.33 ms | 5.57 ms | 26.60 ms | — | — | — |
| Phase0 outer | c8 | 47.00 ms | 7.60 ms | 57.18 ms | — | — | — |
| Phase3 graph events | c1 | 19.11 ms | 5.62 ms | 26.43 ms | 4.51 ms | 5.83 ms | 77.3% |
| Phase3 graph events | c8 | 47.87 ms | 7.67 ms | 58.12 ms | 6.23 ms | 7.94 ms | 78.5% |

PhaseTimer segment durations include GPU idle gaps if the host queues late;
its `total` is a marked step interval, not whole server wall time. The
captured body dominates the sampled DFlash graph, but this alone does not
identify which body projection or attention kernel dominates. The BF16
prototype is a candidate for the body; runtime routing and full serving
quality/throughput need separate qualification.
