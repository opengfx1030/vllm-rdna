# Mixed prefill/decode qualification

Runtime code: `048904c843`; subsequent commits through `8837180c7e` only
add receipts and documentation. Only HIP GPUs 6–7 are used. Flash-Next AWQ
INT4 uses TP2/EP2, MTP2, FULL_AND_PIECEWISE graphs, native expert copy,
96 slots/layer, overlap off, 4,026,531,840 GPU KV bytes/rank and 16 GiB
total native CPU KV. Active request paging is off to isolate prefill stalls.

Reuse `tools/rdna/port_v031/stall_probe.py`. Four decoders each start with
8,192 prompt tokens and must reach 16 output tokens before a five-second
baseline. Inject token-exact 1,024- and 16,384-token prompts into their ongoing
decode. Two repeats use seed 24683, salt 777 and a common warmup. The tool also
collects solo references, including its default periodic-length 4,096-token
reference. Detailed server iteration logging is enabled for both arms.

These synthetic forced-length streams test scheduling, not semantic quality.
Decoders are cancelled after each scenario; their progress does not establish
completed-request throughput. `NO-GATES` means measurements were collected
without an acceptance threshold, not that performance passed. Cache and expert
history are retained within each launch; paired arms require separate launches.

## Baseline

The controller is disabled. Across two repeats, the 16k injection's median
maximum decode gap is 2,675 ms, TTFT is 21.03 s, and decode rate during prefill
is 7.25% of its preceding baseline. Every decoder makes some progress, but the
worst decoder spends 98.3% of the injection window in gaps above the probe's
stall threshold. The 1k injection has a 1,686 ms maximum gap and 1.96 s TTFT.
Server logs corroborate eight mixed 2,048-token prefill steps for the 16k
injection, with mean mixed-step durations of 2,573 and 2,589 ms across repeats.

`base-raw.tar.gz` retains the JSON, timelines, summary, metadata, server log
and wrapper scripts. `base-manifest.json` records member hashes and exact
configuration. `base-summary.txt` is the readable probe output. The server and
its owned workers were stopped before starting the controller arm.

## Packed row metadata

Both ranks report 28,966,912 bytes per physical row. Defined payload per owner:
each of three GDN groups uses 19,488,768 bytes, PLE uses 225,280 bytes, the raw
indexer group uses 26,816 bytes, and the main/compressed QSA group uses the full
row. Groups overlay the same row; these sizes must not be added as independent
allocations. `kv-layout.json` retains all intervals and layer placements.

This establishes potential bytes to omit, not a measured copy or latency gain.
Actual savings depend on snapshot page counts by owner. Current transfers still
copy full rows, CPU allocation stays unchanged, and any future segmented restore
must account for padding initialization and existing lifetime fences.

## Controller comparison

The candidate uses the existing controller with a 250 ms requested budget,
default 0.25 fixed-overhead floor and no added decode-time-share debt. The floor
may override that budget, so 250 ms is not a hard guarantee. Keep general serving
defaults unchanged until mixed-stream and completed-request trials justify it.
