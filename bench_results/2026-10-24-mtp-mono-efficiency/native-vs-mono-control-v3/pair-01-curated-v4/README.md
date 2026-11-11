# Curated native versus Mono pair: v3 runs, v4 analyzer

This package is produced only after both arms pass the strict v4 provenance validator and original v3 serving gates. It contains response-free per-request and per-cell metrics, pair deltas, provenance, and native concurrency counts.

The native concurrency gate passed with all numeric outcomes retained, including near-ties and soft differences. Soft differences are reported and are not silently counted as identical. Review native-concurrency-details.txt and the concurrency gate summary before interpreting the pair.

The output represents one matched pair and does not establish repeatability or statistical confidence. Prefill wave proxy is not true throughput; whole-run throughput is separately defined as total measured tokens divided by summed timed-cell duration. ITL values are streamed chunk intervals, not TPOT.

Raw response-bearing files, raw kernel journals, binaries, and cache trees are excluded. evidence-receipts.json pins input and tool hashes.

## Measured result

One matched pair, 80 successful requests per arm; native MTP2 modules versus target MonoKernel cap8 plus dense Mono MTP2 drafting. Both target and drafter execution change, so this pair does not isolate either component. Native concurrency checks recorded 0 hard failures, 14 near-ties, 2 soft differences and 32 identical results out of 48.

Cell order below is 1k/512 c1, 1k/512 c8, 16k/1k c1, 16k/1k c8. Changes are Mono relative to native.

- Whole-run input/output throughput: +20.18%, +2.01%, +6.04%, +0.31%.
- Mean per-request prefill rate: −2.91%, −4.46%, −0.09%, −2.40%.
- Mean TTFT: +2.98%, +4.40%, +0.09%, +0.73%.
- Median streamed ITL: −8.48%, −4.02%, −7.89%, −4.14%.

Decode benefits coexist with slightly worse prefill/TTFT. This is not an all-metric win, and the long-c8 throughput margin has not been shown to exceed run variance. Source, binary, rank, boot and privileged kernel-journal gates pass; the native journal added 234 records with no matched GPU/PCIe faults. See CSVs for arithmetic per-request decode, acceptance and other latency summaries.
