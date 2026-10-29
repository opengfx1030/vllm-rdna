# Mixed prefill/decode qualification

Baseline runtime code: `048904c843`; subsequent commits through `8837180c7e`
only add receipts and documentation. Fixed-controller runtime: `a733f14e38`.
Only HIP GPUs 6–7 are used. Flash-Next AWQ
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

The first enabled attempt was stopped during warmup: the cap shrank from 512
to 64 tokens and remained at `nofit`. `observe()` ignores chunks below 128
tokens, so the controller never collected another usable bucket and could not
learn the fixed expert-streaming cost. Iterations repeatedly copied experts for
64-token chunks (roughly 0.44 s/step in the retained log). This aborted attempt
is a diagnostic, not a completed comparison. `cap250-unfixed.log.gz` retains
the full log, including shutdown after the deliberate termination.

The fix keeps the cap within the estimator's sample range when hard token
budgets and existing block tiling permit. Eight focused tests cover the expensive
fixed-cost feedback loop, small hard budgets, tiny tails, block tiling and
decode-share debt. The four feedback-loop parameterizations fail before the fix.

## Fixed-controller result

Two matched repeats do not show a meaningful gain. The 16k injection's median
maximum gap is 2,685 ms versus 2,675 ms off; TTFT is 21.01 s versus 21.03 s.
Both arms execute eight mixed 2,048-token steps. Solo references and the
preceding decode baseline are also nearly unchanged. The fitted fixed cost
starts around 530 ms, and the default efficiency floor raises the cap back to
2,048 tokens instead of keeping unproductive tiny chunks.

Keep the controller off in the Flash-Next offloading defaults. A smaller time
target alone does not improve fluidity here. More permissive overhead settings
or decode-time-share debt are separate, unqualified tradeoffs; they were not
enabled by this comparison. No controller-enabled completed-request throughput
or semantic-quality claim follows from these cancelled-stream measurements.

`cap250fixed-raw.tar.gz` and its manifest retain the candidate data and closed
server log. `cap250fixed-summary.txt` is the readable output. To reproduce
`comparison.json`, extract both raw archives to a scratch directory, then run
`python summarize-mixed.py.txt /scratch/mixed`. The helper checks identical
workload configurations, request errors and premature decoder exhaustion before
comparing the full aggregate distributions. Both owned test servers were stopped.
