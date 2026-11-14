# Decode-growth admission as its own switch (2026-11-14)

The scheduler policy that [PR #168](https://github.com/opengfx1030/vllm-rdna/pull/168)
developed for LeapQuant GDN state now has its own flag, so hybrid models with the
FP32 state can use it. Default off. This note covers the switch, not the policy
design: the mechanism, the allocator accounting and the CPU reproduction are in
[`leapquant-gdn-2026-11-13.md`](leapquant-gdn-2026-11-13.md#scheduling-implementation-2026-11-14).

## What it does

Two things, both for hybrid (Mamba plus attention) models:

- **Bounded-output admission.** A request is admitted only if the pool can hold
  its prompt, its declared `max_tokens`, the speculative verify rows and the
  recurrent-state peak of every concurrent batch. The promise is derived from live
  group tables; no pages are allocated early. A very large `max_tokens` can lower
  admitted concurrency even if EOS comes early. It needs
  `scheduler_reserve_full_isl` (on by default); with that off, only the ordering
  below applies.
- **Decoder-first ordering.** Under FCFS, requests that are decoding are scheduled
  before running prefills, so their target and draft rows are not starved by a long
  prefill. An explicit priority policy is left alone.

## How to turn it on

| Where | Setting |
|---|---|
| vLLM CLI | `--decode-growth-admission` (`--no-decode-growth-admission` to be explicit) |
| Launcher | `DECODE_GROWTH_ADMISSION=1 bash tools/rdna/serve_rdna.sh RECIPE=<name>` |
| LeapQuant | Already on with `VLLM_RDNA_GDN_LEAPQUANT=1`; the flag is not needed |

No shipped recipe sets it. The owner enables it per recipe after measuring.

The scheduler logs `Decode-growth scheduling (selected by --decode-growth-admission
| LeapQuant): bounded output admission=..., decoder-first FCFS` once at start. On a
model with no Mamba layers the flag does nothing and the scheduler logs one info
line saying so; it is not an error. With the flag off and LeapQuant off, admission
and ordering are the upstream code paths.

The startup capacity check for the Mamba state (`MambaSpec.max_memory_usage_bytes`,
align mode) counts one state page per concurrent batch under the same condition, so
the check and the admission bound agree. Without that, a single request could pass
startup and still never fit the admission peak.

## Measured and not measured

Measured (the four-boot matrix,
`bench_results/2026-11-14_leapquant-matrix/`; summary in the LeapQuant note):
at 32k prompt, 2k output, c=8 the FP32 baseline boots preempted 10 and 6 times,
and the LeapQuant boots, which run this policy, preempted 0 and 0. That run also
changed the block size (1024 to 512) and the state format, so the gain is not
attributed to the policy alone.

CPU only (`tests/v1/core/test_leapquant_scheduling.py`, real Scheduler and KV
managers, synthetic outputs, tight pool, cap 8): with an FP32-state hybrid layout,
the flag on gives no preemptions and no repeated prompt rows; the flag off still
preempts. These are allocator and ordering checks, not GPU timings.

**Not measured:** FP32 state with the policy on a GPU. The control run is pending.
Whether it removes the FP32 preemptions at 32k c=8 on hardware, and what it costs or
gains in throughput and TTFT, is [INFERENCE] until then. The policy lowers
admitted concurrency when `max_tokens` is large, so a throughput loss at high
`max_tokens` is possible [INFERENCE].

## Credits

The policy, its accounting and its CPU reproduction were developed in PR #168 for
LeapQuant. This change only moves the gate behind a flag; the code is original work
for this fork, with no outside code copied.
