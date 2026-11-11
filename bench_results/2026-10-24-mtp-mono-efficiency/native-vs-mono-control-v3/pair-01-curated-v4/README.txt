Native MTP2 modules without target MonoKernel versus target MonoKernel with dense Mono MTP2 drafting.
Per-request rates are arithmetic per request. Aggregate whole-run rates are summed tokens divided by summed timed-cell duration.
TTFT is kept per request/cell; prefill wave proxy is explicitly not throughput. ITL is streamed chunk timing, not TPOT.
Acceptance columns preserve per-cell JSON values; the pooled field uses summed accepted/draft tokens.
Per-request rate/latency summary reports arithmetic mean and median per request, by cell and across all cells. Whole-run rates divide total tokens by summed measured cell duration. Acceptance length remains the detailed-JSON value per cell; pooled acceptance rate uses total accepted tokens divided by total draft tokens.
The TTFT wave proxy is a proxy, not measured prefill throughput. ITL values are streaming chunk intervals, not TPOT.
Only EXTRA_ARGS (target MonoKernel/drafter flags) and MTP_DRAFTER may differ. A single pair is descriptive only; it does not establish run variance or statistical confidence.

Native worker/API initial-environment audit snapshots were not present in the native raw arm. This does not weaken the startup-config, route, rank, source, or serving gates. The Mono worker environment snapshot reflects initial process exec state only; it cannot prove later runtime os.environ updates. Its paired API initial-exec audit is included and has a narrower scope.

Analyzer v4 preserves the `tests/` path in SHA manifests; v3 normalized it to the basename and could not evaluate the frozen SAME_SOURCE_KEYS check. V4 retains that strict source-equality gate. Exact analyzer, curator, test, and metric-extractor snapshots are included as `.py.txt`; copy back to the listed source filenames to rerun. `same-source-keys-receipt.json` records both raw manifest hashes and the 23 equal checked source hashes. `native-kernel-journal-audit-summary.json` is an offline rerun of the pinned append-only checker; raw journal snapshots were omitted.
