# Cap-8 versus cap-24 c8 diagnostic profile

## Scope and matched setup

This is a descriptive comparison of two separate instrumented c8 serving
profiles, not an unprofiled A/B or a qualification result. Both completed one
16k/1k, concurrency-8 cell with all eight requests successful and the listed
startup, PPL, prefix, after-load, and fault gates passing. Both use boot
`1e8b5da1-759f-4dbb-ae87-e58c2faaff48`, HIP 6–9, core SHA
`6facdbd60118fc14a5cf1ceb5c2494d38ed1be09f55e4636bb0f0825678e2eed`, Mono CU
SHA `c2f189cf559432bf56fd9443fb52d3e91308c2cb255799d60f897a9d4dabc434`,
engine Python SHA `d68e8983fa59059cdfa34be018aba291b7ffb0970cb4390fb3e6742b1e08228e`,
AWQ snapshot `0939125b929543a783ce700c90e36dd1a575c00c`, PLE int4 snapshot
`4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4`, TP4/EP4, GMEM 0.92,
LPTH 1024, prefill interval 2, temperature 0, and phase timing enabled with
op timing disabled. Prefix cache reset before the cell.

The intended inference-config difference is target `k2_max_rows`: 8 in
`mtp-body-profile-cap8-c8-v2-results/config.env`, 24 in
`mtp-body-profile-cap24-c8-v4-results/config.env`. Both have
`mtp_drafter=true` and `mtp_k2_max_rows=24`. Output/cache paths, port, and tag
are necessarily distinct. Each arm is instrumented and ran once.

## Routes confirmed

At cap 8, the target's c8 width-24 region is module fallback
(`region_n=20160`, `kernel_n=0` in the final profile check). At cap 24, target
width 24 dispatches to K2 (`region_n=20160`, `kernel_n=20304`, all four TP
ranks). MTP width 8 uses K2 in both arms (cap-8 final counts: region/kernel
423/423; cap-24: 395/395, all four ranks). Thus cap 24 enables an additional
large-batch target K2 route while retaining MTP's width-8 route.

## Instrumented cell metrics

The single cap-24 profile cell has higher output throughput and better
acceptance than the single cap-8 profile cell, alongside somewhat higher ITL.
These are descriptive deltas only; neither profile provides variance estimates.

| Metric | cap 8 | cap 24 | cap 24 relative to cap 8 |
| --- | ---: | ---: | ---: |
| Completed / failed requests | 8 / 0 | 8 / 0 | same |
| Output throughput (aggregate tok/s) | 86.42 | 97.02 | +12.3% |
| Total token throughput (tok/s) | 1469.08 | 1649.38 | +12.3% |
| Mean TTFT | 35.875 s | 35.111 s | -2.1% |
| Median TTFT | 34.922 s | 34.126 s | -2.3% |
| Median ITL | 55.68 ms | 58.28 ms | +4.7% |
| Mean ITL | 102.69 ms | 119.19 ms | +16.1% |
| Acceptance length | 2.052 | 2.633 | +28.3% |
| Accepted / drafted tokens | 4197 / 7982 | 5080 / 6222 | counts differ |

Acceptance and throughput can move together because accepted draft tokens
reduce target verification work. A single instrumented cell cannot establish
that cap 24 caused these deltas or that they persist across run order.

## Phase observations and caveats

For the low-gap sampled `PHASE tokens=24, n_steps=16` records, cap-8 forward is
47.09–47.58 ms and total is 53.84–54.42 ms. Cap-24 forward is 50.40–50.65 ms
and total is 57.56–57.83 ms. The cap-24 sample is roughly 3–4 ms slower in this
instrumented region even though aggregate TTFT/output throughput in its single
cell is somewhat better. This is a reason to measure the actual serving tradeoff,
not a contradiction that can be resolved from one diagnostic run.

MTP width-8 timing counters are cumulative across startup, probes, and the
serving cell. Their final snapshots show cap-8 region/kernel means 602.8/508.4
us (n=423) and cap-24 629.4/526.9 us (n=395); P5/P6 are 284.2/130.4 us versus
297.7/135.7 us. These cumulative means are not clean per-cell latencies and are
not suitable for claiming that MTP itself regressed. They confirm that the
MTP width-8 K2 route remains active in each run. Startup rows with multi-second
entry spread and early phase samples with large host gaps are excluded from the
low-gap examples, but excluding displayed outlier rows does not subtract their
contribution from later cumulative counters.

## Recommendation

An unprofiled matched cap-8 versus cap-24 serving comparison is warranted. The
candidate changes a real route at target width 24, and the two diagnostic cells
show a potentially useful output/acceptance improvement with an ITL cost. Use
fresh server/cache per arm, cold-prefix reset, same request corpus, unchanged
model/core/PLE/TP4/EP4/HIP6–9, alternate arm order, and repeat enough cells to
measure run spread. Report per-request and aggregate prefill/decode, TTFT,
median/mean ITL, acceptance, route counts, and all correctness/fault gates.
Do not decide from the diagnostic numbers above. The existing profiles are
`mtp-body-profile-cap8-c8-v2-results/` and
`mtp-body-profile-cap24-c8-v4-results/`; their full cell JSONs are
`bench-16384x1024-c8.json`.
