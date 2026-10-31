# Eager hybrid KV paging qualification

Flash-Next AWQ INT4, HIP 6–7 only, TP2/EP2 MTP2, native expert copy,
K2 gate 8, uniform96, GPU KV 3.75 GiB and CPU KV 16 GiB total.
`EAGER=1`, Model Runner V2; live paging forced at watermark zero for stress.
Source runtime: `5ffc4260ba`. This is correctness qualification, not a normal
threshold performance benchmark or a throughput claim.

- PPL 4.80 passed; 56 continuation/concurrency checks including eight 25k–29k
  contexts passed with zero hard failures (12 near-tie, 44 identical).
- Repeated nonaligned long-context spill/resume is present in the compressed
  server log; `summary.json` pairs every spill with its unchanged frontier.
- Prefix probe: cold 12.13 s, warm 2.38 s TTFT (5.1x).
- Combined eager known-demand overlap plus live paging: PPL 4.86 and 48 checks
  passed with zero hard failures (18 near-tie, 30 identical). Overlap was restored
  off after the gate; useful/unused-byte deltas are retained in `summary.json`.
- Final native GPU+CPU prefix reset succeeded. The model server was stopped
  after all probes completed. A separate graph-mode spill-triggered disconnect
  test is retained in `../live-qualification/`.

`kv-layout.json` reports both ranks' configured/available paging capability and
physical GPU/CPU pool bytes. The 28,966,912-byte full packed pages include every
supported hybrid group; small persistent V2 request slots remain on GPU.
Kernel-ring access was unavailable; compressed logs retain any marker findings.
