# Curated Mono v3 serving arm

This package contains one validated 80-request arm with the target MonoKernel enabled at K2 row cap 8 and dense Mono MTP2 drafting enabled. The matched native-MTP2/no-target-MonoKernel arm is still required for comparison.

per-cell-metrics.csv has four cold-prefix cells. per-request-metrics.csv has request timings and token counts. The TTFT-derived prefill wave value is a proxy; whole-run input/output rates are reported separately. ITL uses stream chunks, not TPOT.

raw-file-sha256-receipts.json records hashes and sizes for original top-level files. Response-bearing JSON, text and logs are not included. The cache subtree is excluded. quality-and-gates.txt contains filtered numeric gate results only.

Validation checked boot/core/source, all four TP-rank to physical HIP 6–9 mappings, API-server initial environment, all four benchmark windows, quality probes, after-load probe and fault gates. The native arm and matched comparison remain pending.

## Environment audit scope

worker-environment-audit.json is a whitelist read from each worker's initial process environment. A null entry there does not show that the corresponding Python runtime setting was disabled: the launcher can update os.environ after the worker starts, and /proc/PID/environ does not reflect those later changes. The API server's separate initial-exec audit records the NaN guard enabled and HIP_VISIBLE_DEVICES=6,7,8,9. The launch configuration, source/driver pins, and completed safety gates also passed. Treat the API audit as startup evidence, not as a snapshot of later worker runtime state.

Exact analyzer, test, curation, and metric-extractor source snapshots are included as .py.txt files. Their hashes and copy-back/test commands are in analyzer-source-snapshot-manifest.json. The suffix prevents repository hooks from rewriting evidence snapshots.
