# P1 trace and cap24 c8 diagnostics — curated evidence

Completed evidence only. No production change, kernel/binary object, raw serving log, prompt/reply/token dump or raw kernel journal is included. Exact tool sources are stored as `.py.txt`, `.sh.txt`, `.cu.txt` and `.cpp.txt`, so code hooks cannot rewrite the executed source snapshots. `source-and-curation-manifest.json` records original and curated SHA256 values and every transformation. `SHA256SUMS` covers the full delivered bundle.

## Actual statuses

P1 trace v1 failed (wrapper exit1). Its result stayed RUNNING/complete=false and is preserved as such. The diagnostic guard wrongly required aligned native onepassK1/320threads; actual native dispatch usesK2/160threads. It is not a kernel correctness failure or a successful trace. The failed result, source/build provenance available locally, guard exception and fault check are retained. The v1 build manifest was not locally recovered; its recorded remote SHA is preserved without claiming its contents.

P1 trace v2 passed (wrapper exit0, complete=true). Eager seed250252, same-input graph seed250252 and genuinely changed graph seed250253 passed the recorded output/XN/route parity and sticky-error checks. Source and boot hashes match before/after; archived privileged journal snapshots are append-only with0new records/faults. The v2 native/global trace source, build manifest and exact runtime tools are retained.

The analyzer-v3 result shows a one-FP32-ULP sum and rrms difference for the failing M128 row104/stream0 input: native rrms0.9044788479804993 versus global0.904478907585144. At XN index1690 these round to0.98828125 versus0.98876953125. Using the measured rrms and FMA reconstructs all2560values of each archived XN vector exactly. This identifies a sufficient P1 rounding boundary for this case; it does not prove a route repair, full-M qualification or serving gain. The changed-input trace lacks an archived full-XN reference. The numeric subset needed for this exact analysis is retained instead of the original35MBsnapshot.

Cap24 c8 v4 is a completed instrumented diagnostic PASS, with8/8requests successful, source/core/boot gates matched and all four rank core/physical queue mappings verified. Target width24 reaches K2 on all ranks; MTP width8 remains K2. Archived journal rechecks show104new records and0faults. Cap8 c8 v2 is retained as the matched diagnostic comparison with the same boot/core/model setup; it also has104new journal records and0faults.

The single cap24 cell reports97.02aggregate output tok/s versus86.42 at cap8, acceptance2.633 versus2.052 and meanTTFT35.11s versus35.88s. Median/meanITL increase to58.28/119.19ms from55.68/102.69ms. These separate single instrumented cells have no variance qualification; they do not establish a causal performance win. The exact priority-authored comparison is `profiles/PROFILE_COMPARISON.md`. Normal-EOS concurrency gates retain their documented near-tie policy; no strict token-identity claim is made.

## Contents and checks

- `p1/v1/`, `p1/v2/`: actual trace outcomes, source/build provenance, boot/source/owner/fault checks.
- `p1/analysis-v3-result.json`: exact completed analyzer result.
- `p1/native-repeat-result.json` and `p1/snapshot-row104-stream0-subset.json`: response-free numeric analysis inputs. Omitted snapshot rows are explicit; the original snapshot SHA remains recorded.
- `profiles/cap8/`, `profiles/cap24/`: full parsed profiling payloads (JSON whitespace compacted only), exact config/source/core/rank evidence, response-free cell timing/acceptance arrays, gate summaries and compact journal checks.
- `tools/`: exact versioned source snapshots. Tool execution SHA comparisons have no mismatches; tools absent from recorded input lists are marked unknown rather than labeled verified-executed.

CPU-only replay (no torch/GPU/SSH):

```bash
python replay_analysis.py.txt
```

It loads the unchanged analyzer/helper snapshots as text and reproduces the stored analyzer result exactly, excluding only its original source-path hash map. The compact numeric subset is sufficient for this replay; it is not represented as the original full snapshot. Curation and validation did not rerun serving, benchmarks or GPU traces.

This bundle intentionally excludes the later native-order experiment and its in-progress repeat. The native-order operator result and wrapper/journal outcome require their own evidence and must not be inferred from these earlier clean P1 trace gates.
