# ABBA repeat comparison

`compare_repeats.py` consumes the analyzer's `per-cell-analysis.csv` or a
provisional JSON containing its per-cell `rows`, then writes a descriptive CSV
and a definitions JSON into a new output directory. It never overwrites a
nonempty output directory or modifies its input.

```bash
python3 compare_repeats.py \
  --input /path/to/serve-inline-ab/per-cell-analysis.csv \
  --out-dir /path/to/new/abba-repeat-summary
```

For each workload cell it emits candidate-minus-control deltas for both
ABBA comparisons (02 vs 01 and 03 vs 04), within-variant repeat spread
(control 04 vs 01; candidate 03 vs 02), and descriptive two-repeat means when
all four arms are present. On the first pair alone it emits only that pair and
lists missing second-pair metrics. Rates include whole-cell input/output
throughput, arithmetic means of request-level prefill/decode rates, and pooled
ratio-of-sums decode rate as separate rows. The TTFT-derived prefill wave
proxy remains explicitly labeled as a proxy, not throughput. Time deltas are
in milliseconds and percent; acceptance and per-position acceptance deltas
are in percentage points. Acceptance length remains a separate metric.

The exporter also computes a two-repeat count-weighted acceptance rate as
`100 * sum(accepted tokens) / sum(draft tokens)`, alongside the arithmetic
mean of each run's acceptance percentage. It does not average run percentages
and call that token-count weighted.

The script performs no bootstrap or significance test. The serving run/cell is
the experimental unit; two repeats per variant provide only two paired
contrasts and a descriptive repeat spread. Request-level resampling would
artificially inflate precision. Treat results from the recovered first pair
as provisional until arms 03 and 04 are available and validated.
