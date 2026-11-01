# RDNA2 AWQ INT4 expert and KV offloading target

Status: 2026-10-28. TP2 segmented graphs/MTP2 expert offloading and native
CPU KV restore pass generation, normal-EOS concurrency, and strict numerical
parity. The deterministic-dense run has identical cold, GPU-cached, repeated,
and CPU-reloaded token logprobs. Longer-context coverage also passes: fixed-length 16k/1k at c=1/c=8,
32,256/256 at c=1, and coherent 32,256-token normal-EOS marker retrieval.
Forced random outputs are performance evidence only; their quality warnings
are retained separately.
Branch: `codex/rdna2-awq-moe-offloading`.
Base: `afc3735bf93d841397740b07eb3ef7aad6a21855`.

## Confirmed first target

Use the existing `tools/rdna/mono/serve_2a.sh` configuration with `ARM=on`:
Flash-Next Qwen4Exp, TP2/EP2, PP1, FP16, symmetric W4A16, top-10 routing,
V2 runner, FULL_AND_PIECEWISE, and the existing MTP2 recipe. Preserve MTP0
as a correctness/control case. K1, K2, GDN, QSA and larger-batch module fallback
remain in the qualification scope.

The user allocation is HIP 6,7,8 only; use two of those devices for TP2.
The original TP4 resident path remains a regression reference. Offloaded K2
supports 256 logical experts per TP2 rank and shared width 320; physical cache
capacity is independently bounded.

## Implementation and use

`tools/rdna/mono/serve_offload.sh` resolves the existing recipe with TP2,
`--monokernel`, the configured expert slots, and the existing native KV tier.
`PRINT=1` prints the command; `VALIDATE=1` uses the existing serve validation
driver. The default is 96 slots per MoE layer/rank on the V620 TP2 target.
Model and PLE cache paths below have been verified, including all model
shards. Defaults use HIP 6,7; GPUS may select another pair from the allocation.

```bash
PRINT=1 bash tools/rdna/mono/serve_offload.sh
VALIDATE=1 MTP=0 EAGER=1 CELLS=none \
  bash tools/rdna/mono/serve_offload.sh MAXLEN=8192 SEQS=2
```

The config keys under `rocm_mono_decode` are `expert_cache_slots` (0 disables),
`expert_cache_policy` (`lfu_lru` or `lru`), and `expert_cache_decay` (invocations).
The wrapper exposes SLOTS/POLICY/DECAY and KV_OFFLOAD_GB; KV_OFFLOAD_GB=0 disables
the native host KV tier for expert-only comparison. The offloading launcher
also defaults `VLLM_RDNA2_W4A16_DETERMINISTIC_DENSE=1`; standard resident recipes
retain their existing default. This removes FP16 split-K atomic rounding
variation from the target's dense INT4 projections.

Implemented: bounded CPU loading and per-expert HIP packing, immutable pinned
six-tensor banks, event-fenced stable GPU slots, protected demand leases, overflow
passes, frequency decay/LRU, active-pass map upload reuse, native K2 prefix/pass/
finish stages, shape-based stage graph reuse inside explicit host boundaries,
and the modular prefill path. Offloaded K2 has its own workspace so it cannot
alias ordinary K1/GDN scratch or statistics. This feature selects segmented
FULL and PIECEWISE graphs, with host planning outside graph segments. Eligible
K2 compute stages retain their own reusable graphs. Other configurations retain
their existing graph defaults.

The modular fallback uses integer accumulators with 32 fractional bits for W1
and W2, matching MonoKernel's FP32 finalization contract. W2 accumulates across
every cache pass before one conversion to FP16, so cache capacity and history
cannot change the reduction order. Per-row sticky status preserves nonfinite
and unsafe-range failures without poisoning valid neighboring rows. At the
4096-token/top10/H2560 target, transient integer scratch is about 480 MiB plus
the existing activation buffers.

Not yet implemented: predictive cross-layer prefetch, persisted frequency priors,
device-managed miss filling, or dynamic redistribution between expert and KV
allocations. KV transfer ownership remains with the native connector. There is
no new cross-subsystem copy-priority scheduler; contention must be measured.

`tools/rdna/mono/build_offload.sh` builds in an isolated .venv, inheriting read-only
ROCm dependencies from BASE_VENV. `tools/rdna/mono/test_offload.sh` runs policy,
storage, native split/graph and CT resident-parity tests on one allocated GPU.
Logs live under `~/w4a8_runs/moe-offload/`. Local CPU validation has 153 passing
tests (12 GPU tests skipped locally). The current remote focused suite passes
197 tests (49 expected unsupported cases), including native TP2 geometry, eviction, cross-stream fences, CT
packing parity, and strided-input adapters. The production segmented K2 graph
regression passes on HIP 8. A separate real two-rank test on HIP 6,7 passes
20 rank/width/iteration cells at widths 1,3,6,12,24 with forced eviction and
fused all-reduce: maximum difference from independent ordinary HIP is 0.00110.
Six draft graph-manager regressions pass remotely.

The isolated full-model TP2 eager/MTP0 run on HIP 6,7 passed repeated greedy
generation before and after load, with 32 slots per layer and the native KV
tier enabled. The latest-source recheck also passed the explicit KV proof:
320,864,256 bytes stored and reloaded after a GPU-only prefix reset, identical
32 output tokens, and maximum token-logprob difference 0.03569 (limit 0.05).
The initial raw PIECEWISE graph trials produced NaNs or incorrect text. Selecting
segmented FULL/PIECEWISE capture and making the PLE dispatch boundary explicit
resolved those generation failures.

The latest `tp2-qualification-v1` run uses HIP 6,7, TP2, MTP2, 96 expert slots,
16 GiB native CPU KV tier, MAXLEN=32768, MAXBAT=4096 and SEQS=8. It passes
3/3 generation probes before and after load, prefix reuse (12.38 s cold TTFT,
2.33 s warm), and graph-bucket warmup. Per-rank model allocation is 21.39 GiB;
the reported KV capacity is 138,022 tokens. A 1024/128 c=1 cell measures
27.59 decode tok/s/request; a 1024/64 c=8 cell measures 30.95 aggregate output
tok/s. These are initial measurements without a controlled resident A/B.
No GPU or native fault markers were recorded.

Two strict concurrency failures occurred under forced continuation after EOS,
following identical completed haiku text. The separate `tp2-normal-eos` run
passes all 44 normal-EOS comparisons (38 identical, three near ties, three soft
differences, zero hard failures), plus generation probes before/after load and
zero native/GPU faults. The native KV probe demonstrably reloaded 289,669,120 bytes and
preserved all 32 output tokens, but failed the unchanged 0.05 token-logprob
limit (maximum difference 0.09928). Its GPU-cached reference reused 2048 prompt
tokens. The follow-up `tp2-kv-boundary` diagnostic finds 32 identical tokens
between two GPU-resident references but a maximum logprob difference of
0.06360, already above the unchanged 0.05 check. Capping external lookup at
2048 tokens produced no native load and zero cached tokens after GPU reset;
that attempt is a failed diagnostic, not a CPU restore proof. It does not
resolve whether unrestricted restore adds error beyond resident execution.
The corrected `tp2-kv-aligned` diagnostic requests a cap of 2049 to
compensate for native Mamba lookup's subtraction before alignment. It confirms
exactly 2048 cached tokens in both GPU and CPU paths and identical 32 output
tokens. CPU-versus-original-GPU logprob difference is 0.07583; GPU repeat
versus original GPU is 0.06910. The original 0.05 gate still fails.

The new native hybrid DMA regression on HIP 8 passes exact byte equality for
attention, latent attention, FP16 convolution / FP32 GDN state, and replicated
PLE state. Two chunk sizes (1 and 3 GPU blocks) and two changing-pattern rounds
transfer 80,871,424 bytes in total. It checks different source/destination block
IDs, partial chunks, pending producer/destination writes, and untouched padding
and null blocks. This validates transfer integrity for those layouts, without
proving scheduler boundary or full-model numerical parity. Local full-module
collection requires missing vLLM dependencies (`regex`); static validation and
CPU policy tests passed before the remote hardware run.

The `tp2-kv-mtp0` graph control also fails the strict check: both GPU references
and CPU restore report exactly 4096 cached tokens; all 32 generated tokens
match. GPU-versus-GPU logprob delta is 0.08484 and CPU-versus-original-GPU delta
is 0.13086. Thus the variation is not confined to MTP verification. The updated
probe gates both resident-reference stability and CPU parity separately, with
the original 0.05 tolerance, and retains both measured deltas and transfer
counters on failure. It cannot label an unstable reference as a successful
numerical qualification.

Source inspection also finds differing deployed GDN arithmetic contracts:
non-spec HIP rounds sigmoid beta to FP16, whereas speculative HIP/Triton retain
FP32. MonoKernel matches each deployed reference. That can explain cross-mode
sensitivity, but it does not establish the cause of the MTP0 repeat variation;
no production arithmetic was changed during this investigation.

Further repeatability work identifies two concrete sources of variation:

- QSA top-k atomically appends selected block IDs, so the same selected set can
  arrive in different order. RDNA QSA expansion now optionally canonicalizes
  at most 512 selected IDs before token expansion; its selector enables this,
  while standalone defaults preserve input order. Seventy focused GPU tests
  pass, including strict order repeats, independent attention references, and
  permutation/padding/causal-tail cases. Measured expansion cost increases by
  7.4–8.0 microseconds per call in the tested shapes.
- Dense W4A16 decode and prefill previously accumulate split-K partials through
  FP16 CAS atomics. Identical qkvz, BA and output-projection inputs vary across
  eager and captured replays. A guarded Q32 dense reduction is qualified
  behind `VLLM_RDNA2_W4A16_DETERMINISTIC_DENSE=1`; defaults preserve the existing
  resident path. The flag also routes symmetric compressed-tensors large
  prefills through the deterministic RDNA kernel instead of exllama. W4A8 is
  rejected with this diagnostic/qualification setting. Scratch is
  `8 * M * (N + 1)` bytes (224.03 MiB for 4096-by-7168 qkvz), plus FP16 output.

Opt-in bounded traces locate the first GPU-repeat difference before layer 1
expert computation: layer 0 K2 inputs, routes and outputs match exactly; layer
1 attention differs. Layer 1 GDN traces show identical current qkvz, BA and
convolution state, but differing initial recurrent state. This supports tracing
variation from prior computation rather than attributing all of it to expert
cache transfers. The corrected QSA-only control still fails CPU logprob parity
(0.19422) despite an in-tolerance GPU repeat (0.04289); dense stabilization is
therefore a separate required check.

Tracing is disabled by default and intentionally synchronizes when enabled.
`VLLM_RDNA_EXPERT_TRACE_DIR`, an optional `_ARM_FILE`, `_MAX_CALLS`, and
`_MAX_ROWS` control K2 snapshots; `VLLM_RDNA_GDN_TRACE_LAYER`, `_SEQ_LEN`, and
`_MAX_CALLS` select bounded non-spec GDN state receipts. The comparison tool
requires identical generated-token proof and matching sequence/token metadata;
matching graph bucket sizes alone are insufficient alignment evidence.

The later `tp2-deterministic-final` run resolves this numerical gate without
changing its tolerance: all 32 generated tokens and all token logprobs match
exactly across cold prefill, original GPU-cached reference, GPU repeat, and CPU
reload (maximum difference 0.0). It stores 347,602,944 bytes and reloads
289,669,120 bytes after a GPU-only reset. Normal-EOS concurrency passes all
44 comparisons (zero hard failures). The initial long-workload client then
stops before submitting its first benchmark request because ratio 1 is invalid
in the pinned dataset sampler. The corrected client uses ratio 0.0 for fixed
lengths and is tested against the actual sampler implementation.

Dense GPU coverage passes 55 deterministic reference/repeat/guard cases, eight
legacy dense cases, and 90 shared-helper MoE cases (49 expected unsupported
cases). Isolated qkvz prefill cost at M4096 increases from 8.84 to 14.17 ms;
this is a kernel measurement, not an end-to-end slowdown claim. Numerical
stability has an explicit workspace/latency cost; resident defaults remain
available for controlled comparison.

Graph-padding routes are now excluded from K2 host planning and frequency
observations, while GPU route buffers and graph shapes stay fixed. A real TP2
padded-M3 regression passes independent output and per-expert frequency checks
across forced cache churn, with no workspace faults.

The offline real-row decode sweep has 96 rank/layer caches and 100 calls per
cache, starting empty. At 96 slots, LFU and LRU both need 5,647 expert copies
and reach 88.24% expert-demand hits; 128 slots saves no further copies. At 64
slots LFU saves 6.12% copies versus LRU. This excludes initial prefill residency,
physical padding history, transfer overlap and timings; no 128-call frequency
decay occurs. It supports the 96-slot default for this sample, without proving
whole-workload optimality. Receipts are preserved in the bench-results folder.

Final long-context evidence combines `tp2-qualified-long` and `tp2-32k-tail`:
all nine 16k/1024 performance requests complete with exact requested lengths,
zero request failures, and zero preemptions. The c=1 cell measures TTFT 17.31 s,
about 29.7 decode tok/s, and 19.77 aggregate output tok/s including prefill.
The c=8 cell measures 26.72 aggregate output tok/s including prefill, with
mean TTFT 106.26 s. Its KV pool reaches 100% observed occupancy without
preemption. CPU KV stores are 926,941,184 and 7,415,529,472 bytes respectively;
neither cold performance cell needs a CPU restore. One c=1 and five c=8
forced-random output warnings include repetition and whitespace; these remain
recorded and are not counted as semantic validation. The original client
failure receipt is preserved, alongside a SHA256-provenance offline audit of
the completed requests under the corrected performance-only classification.

The remaining 32,256/256 performance cell completes with TTFT 33.70 s and
30.08 decode tok/s (6.07 aggregate output tok/s including prefill), with zero
preemptions or request failures. A separate coherent prompt of exactly 32,256
tokens retrieves `ORBITAL_FENCE_4729` correctly and stops naturally after 73
output tokens. The final run repeats strict KV parity with maximum logprob
difference 0.0 before long requests, passes generation probes after load, and
reports zero native/GPU fault markers. Each run's scope is explicit; no single
continuous three-cell run is claimed. `qualification-summary.json` links the
combined evidence and its limitations. Actual workload
coverage is about 4k prompt tokens, not the configured 32k context limit.
Evidence is in `bench_results/2026-10-28-moe-offload/` and corresponding remote
run directories under `~/w4a8_runs/moe-offload/`.

Verified on `par1-cs25` on 2026-10-27:

- Model: `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`, snapshot
  `0939125b929543a783ce700c90e36dd1a575c00c`.
- Actual checkpoint quantization: `compressed-tensors`, pack-quantized,
  symmetric INT4, group size 128; model type `qwen4_exp`.
- Model directory:
  `/home/chenco_adm/.cache/huggingface/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c`.
- INT4 PLE directory:
  `/home/chenco_adm/.cache/huggingface/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4`.
- Isolated remote checkout:
  `/home/chenco_adm/opengfx1030_vllm-rdna_moe_offloading`, on the same branch
  and base as local. Existing runtime/service checkouts were not changed.

The wrapper's historical `/home/chenco_adm/hfcache/` defaults do not exist on
this host. Pass MODEL and PLE_QUANT_DIR overrides; do not download duplicates.

### MonoKernel execution requirement

K2 fuses HC, routing, shared experts, routed experts and optional all-reduce.
Supported decode/verify bypasses the generic RoutedExperts hook, so a #37-style
hook alone cannot implement this target. The TP2 layout owns 256 contiguous local experts out of 512 (the original
TP4 resident layout owns 128), with hidden size 2560 and packed W13 output
width 1280. Keep logical ownership distinct from cache slot count.

The implemented offload-specific K2 split preserves the full resident K2:

1. Prefix/router/shared contribution once; publish selected IDs and normalized
   weights for every supported row bucket into stable storage.
2. Bounded slot-aware expert passes; change only six payload pointer offsets
   from logical expert to physical slot. Preserve original pair indices and
   fixed-point accumulation across all passes.
3. Finish/conversion/all-reduce once. Initialize accumulator and barrier state
   with explicit stage lifetimes; preserve sticky fault and recovery semantics.

Fixed-shape compute stages use stable buffers. Host miss handling stays
at an explicit orchestration boundary until a device-managed fill mechanism is
qualified. Also cover module fallback for prefill/wider rows. Dense MTP experts
retain their existing resident path initially; MTP remains a model-level gate.

## Objective

Serve models whose routed experts exceed VRAM using an immutable packed INT4
host bank and a bounded GPU expert cache. Optimize end-to-end latency and
throughput through graph reuse, expert reuse, and coordinated expert/KV memory
and transfer budgets. Eager correctness is an intermediate gate, not the
production performance target.

Cold experts remain in RAM; demanded experts execute on the existing FP16 HIP
RDNA2 path. Shared experts, routers, dense layers and recurrent state retain
their existing placement contracts. Scope is initially gfx1030; other devices
and the disabled path retain existing behavior.

## Sources and existing facilities

- Fork draft [#37](https://github.com/opengfx1030/vllm-rdna/pull/37), head
  `142eeaba9b2f28df8326b8cd9eaeae51cc817a83`: residency prototype on v0.28.
- Fork draft [#40](https://github.com/opengfx1030/vllm-rdna/pull/40): heterogeneous
  GPU execution, a separate extension dependent on #37.
- TensorFold [#1113](https://github.com/ashhart/TensorFold/pull/1113) and its split
  foundation [#1441](https://github.com/ashhart/TensorFold/pull/1441): expert pools,
  frequency priors, asynchronous staging and graph boundaries. Reuse concepts;
  do not port CUDA/EXL3 packing or unrelated KV implementations.
- `vllm/model_executor/offloader/`: existing whole-module offloading; reuse
  suitable allocation/event patterns, not its cyclic layer placement policy.
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/` and
  `vllm/v1/kv_offload/`: existing native KV tier, scheduler and transfer handling.

## Execution and ownership contracts

1. Keep router IDs global through prepare/finalize and EP distribution. Perform
   rank-local expert-to-slot lookup after dispatch at the expert execution
   boundary. Do not temporarily override the layer's global expert map.
2. Explicitly bind packed W13/W2, scales, zero points, layout and zero convention.
   GPU payload addresses remain stable. Kernel-held scale/zero references must
   point at those same buffers; no recursive object-graph alias rewriting.
3. Key entries by model/layer identity and rank-local expert. MTP and decoder
   caches have distinct identities. Protect demanded, copying and consumer-live
   slots from replacement.
4. For each slot, establish copy-complete before consumption and consumer-complete
   before overwrite. Hits require no host copy. Expert eviction needs no D2H
   because host weights are immutable.
5. Partition demand exceeding pool capacity into bounded passes. Preserve token
   positions, router weights and every selected pair exactly once. Shared experts
   and final distributed reduction execute once. Pass accumulation must account
   for the existing expert implementation zeroing its output on each invocation.

## Graph efficiency

- Preallocate payload, routing/slot tables, scratch and pass outputs. Change
  contents rather than captured tensor addresses or registered Parameters.
- Establish explicit graph boundaries for host planning and transfers; preserve
  capture of eligible dense/attention and expert compute segments where supported.
  Do not treat Python planning inside a FULL graph as replay-safe.
- Qualify shape buckets for tokens and bounded expert passes. Misses, eviction,
  overflow, MTP verification and changing routing must work across repeated replay.
- First measure host planning (including routing D2H waits). Evaluate a device
  planner only if that synchronization is material. Mapped-host direct expert
  reads are an experimental alternative, not the default assumption for PCIe.
- Cache-policy state must update on real execution/replay, not only during capture.
  A hit-heavy path must not synchronize merely to maintain Python counters.
- Evaluate the existing `eager_break_during_capture` mechanism and MoE partition
  boundary against the V2 runner; neither is assumed sufficient for FULL replay.
  Reuse the existing cudagraph dispatch buckets. Graph keys describe payload
  layout and execution shapes, not which expert IDs occupy the slots.

## Expert policy

Compare plain LRU with decayed access frequency plus last-use recency and a
deterministic tie-break. Count actual routed pairs; distinguish prefill and decode
telemetry so large prefills cannot silently dominate policy decisions forever.
Select policy from measured workload traces, including phase changes and MTP.

Demand outranks speculative prefetch. Prefetch the next actual routed MoE layer,
skip dense/hybrid-only layers, cap speculative bytes, and measure unused evictions.
Optional persistent priors require checkpoint/layout/partition/schema validation,
decay and atomic saves. They are not required for initial correctness.

## Joint expert and KV management

Plan per-rank VRAM as dense/shared weights + expert slots + KV blocks + recurrent
state + graph/workspace allocations + staging + margin. Include postprocessing
peaks and host/pinned RAM. Two full expert banks must be charged as two banks.

Use the existing native KV connector and scheduler; do not introduce a second KV
pager. Verify the selected model's attention groups and state dependencies before
claiming KV offload support. Prefix reuse/preemption offload does not automatically
provide arbitrary active-context paging for every attention backend.

Coordinate PCIe traffic: required KV restores and expert demand copies take
priority over speculative expert prefetch and background KV stores. Measure
contention and scheduling overhead; a second copy stream alone does not establish
bandwidth or overlap. KV data can change and must obey existing store/restore
ownership, unlike immutable expert payloads.

Initially choose fixed, explicit expert/KV budgets from a workload sweep. Compare
the latency saved by additional expert slots against KV reload/recomputation and
admission pressure. Adaptive redistribution is later work and may happen only at
quiescent boundaries with allocator, scheduler and graph validity preserved.
`kv_cache_memory_bytes` is per GPU; `kv_offloading_size` is a total CPU buffer
across TP ranks. Initialize the expert pool before worker memory profiling.
Explicit KV sizing bypasses automatic KV budget inference, so validate total
admission separately. Startup-plan identity must include offload configuration.

## Work packages

- Cache policy/lifecycle: payload and lease API, protected demand sets, overflow
  partitioning, replacement and exact memory accounting.
- AWQ/HIP integration: checkpoint/backend compatibility, bounded loading and
  packing, explicit kernel bindings, local slot mapping and pass accumulation.
- Graph/KV validation: graph boundaries, TP/EP/MTP support matrix, native KV tier
  contracts, joint memory and transfer measurements, regression coverage.
- Integration owner: config propagation, startup validation, shared contracts,
  recipe integration and review of cross-package changes.

Agree interfaces before parallel implementation. Assign disjoint files; keep
loader, RoutedExperts and config integration under one owner.

## Acceptance gates and unresolved inputs

Confirm checkpoint path/format, symmetric versus asymmetric zeros, group size,
first TP/EP backend and target concurrency/context before implementation dispatch.
Checkpoint/format/group size, complete model shards, and TP2/EP2 are now
verified above. Longer context and sustained-load sweeps remain outstanding.
On ROCm, AutoAWQ falls back through the Marlin-support check to MoeWNA16.
The generic RDNA2 WNA16 oracle currently rejects asymmetric checkpoints; the
compressed-tensors RDNA2 method has an explicit raw-zero path. AWQ compatibility
must be demonstrated for the chosen checkpoint, not inferred from INT4 alone.

Extend existing kernel/offloader/connector tests where applicable. Cover forced
churn, oversized demand, all payload tensors, nonlocal/padded IDs, stream hazards,
repeated graph replay with changing routes, MTP, hybrid state and KV restores.
Compare pooled/resident layer and model outputs with a justified tolerance when
multi-pass reduction order differs.

Primary existing suites: `tests/quantization/test_moe_wna16.py`,
`tests/quantization/test_rdna2_moe_asymmetric.py`,
`tests/kernels/quantization/test_rdna2_moe_w4a16.py`, and native KV connector tests.
A new pure expert-policy suite is justified because cyclic dense offloader tests
do not cover demand protection or expert overflow passes.

Measure prefill tok/s, per-request decode tok/s, TTFT and ITL for 1k/512 and
16k/1k at c=1/c=8, plus capacity/long-context cases. Record expert hit rate,
useful-prefetch rate, transferred bytes, exposed copy wait, routing/planning cost,
graph replay/fallback rates, KV reloads/preemptions, pinned RAM and peak VRAM.
No speedup claim before controlled A/B hardware measurements.
