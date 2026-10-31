# Live Flash-Next KV spill into the native CPU pool

Status: experimental implementation present; complete model qualification is
in progress. The initial audit below preceded implementation; its V1 portability
notes are superseded for the target recipe by the final Model Runner V2 section.
This work does not extend deferred PR #23. Only physical HIP devices 6 and 7
are authorized for model validation (unit/transport tests use GPU 7).

The first graph-mode live-paging run passed PPL 4.87 and 48 concurrency checks
with no hard failures and with actual spill/resume at unchanged frontiers.
The eight-long-context probe also passed 56 checks with zero hard failures;
logs confirm repeated spill/resume at nonaligned 25.9k–29k-token frontiers.
A disconnect triggered by a host-ready spill also passed: the eight surviving
requests completed, post-abort PPL passed and reset succeeded. Across the run,
31 spill/resume pairs preserved frontiers; only the intentionally aborted request
had an unpaired spill. Eager PPL 4.80 and 56 continuation checks passed with
28 additional equal-frontier spill/resume pairs; combined eager overlap plus
live paging passed PPL 4.86 and 48 checks. Normal-threshold paging performance
is a separate open measurement. Controls remain disabled by default.

## Boundary and smallest useful slice

Start with an explicit, same-engine suspend/resume operation for one live,
text-only Flash-Next request at a fully processed target/drafter step boundary:
TP2/EP2, MTP2, fixed model/layout, native CPU backend, blocks_per_chunk=1,
worker-private storage, and a verified single packed GPU manager-page backing.
Keep existing prefix offloading and its defaults unchanged when paging is off.

The request retains its CPU token history, sampler/grammar state and worker
request object. Save its allocated GPU pages, release its references only after
both ranks finish storing, then allocate replacement GPU pages and restore
before allowing another model step. This is whole-request residency paging:
the request must fit the GPU pool while running. It is not per-layer demand
paging, does not run arbitrary active contexts directly from RAM, and does not
make a context larger than the GPU pool executable.

Do not start with automatic victim selection, cross-engine migration, CPU-tier
eviction of suspended requests, vision inputs, TP changes, DCP/PCP/PP, canonical
layout conversion, mmap deduplication, or partially in-flight MTP verification.
An explicit manual operation establishes correctness before scheduling policy.

## What existing code does

- `vllm/v1/core/sched/scheduler.py::_preempt_request` immediately frees/detaches
  GPU blocks, resets num_computed_tokens to zero, clears speculative IDs and
  queues PREEMPTED. Admission then finds cached prefixes and recomputes any
  missing suffix. This is not a save-before-release live paging transition.
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` stores
  content-hashed prefix chunks and materialized recurrent checkpoints. Its
  lookup is bounded/aligned; it is not an exact arbitrary live frontier.
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py` selects
  prefix-cacheable groups. `CircularBufferSpec.prefix_cacheable` is false, so
  QSA raw rings are not represented as ordinary offload keys.
- `vllm/v1/kv_offload/cpu/gpu_worker.py` already has pinned host storage, batched
  copies, descriptor lifetime retention, compute-stream dependencies, pooled
  events and completion polling. Real loads block admission until completed;
  `wait_for_layer_load` is intentionally unnecessary for this async contract.
- `offloading/common.py::OffloadingWorkerMetadata` and scheduler job pending
  counts aggregate completion across workers. Existing source-reuse fences and
  jobs_to_flush remain necessary alongside a new paging transaction.
- LRU/ARC in `vllm/v1/kv_offload/cpu/policies/` choose evictable CPU cache chunks.
  They do not choose GPU request victims or guarantee a suspended request keeps
  its only copy. `CPUOffloadingManager.complete_store` normally makes a ready
  chunk evictable; transfer pinning lasts only through prepare/complete_load.

## Save the complete state, not just a token prefix

The snapshot manifest must inventory every allocated group, including sparse
logical block-table positions, null sentinels and any repeated physical IDs:

- QSA main K/V pages and compressed index keys, including the exact partial
  final page and quantization metadata stored in those pages.
- QSA raw-key circular buffer, including cached MRoPE positions if enabled.
  The initial text-only gate excludes MRoPE/vision, but must reject rather than
  silently omit an unrecognized ring format. See
  `vllm/models/qwen4_exp/common/qsa_cache.py::QSAKeyStateCache` and
  `QSACompressedKeyCache`, and `amd/indexer_qsa.py`.
- GDN convolution history and recurrent state, in every non-null live Mamba
  slot needed by align mode, acceptance and the next MTP step. GDN is TP-sharded.
- PLE short/dilated convolution state, including speculative history. This
  state is replicated across TP, but both worker-private copies must restore.
  See `vllm/v1/kv_cache_interface.py::MambaSpec`, Mamba state calculators and
  `vllm/models/qwen4_exp/amd/ple_layer.py`.
- Drafter KV and recurrent state, if its groups are separately allocated.
  Prefix offloading's volatile-tail exclusion must not be reused for a live
  snapshot. Save the exact live bytes or explicitly normalize and regenerate
  pending drafts under a separately validated resume contract.

PLE embedding tables are immutable model storage, not request state. N-gram
context can be reconstructed from retained token history and the exact frontier
by `gpu_model_runner.py::_prepare_ngram_context`; it must not be reconstructed
from prompt length alone. PLE IPC output/doorbell buffers are transient: drain
existing operations before suspension and rebuild normal per-step inputs.

Also retain scheduler and manager state: accepted/computed frontier, token and
output counters, pending input/speculative IDs, request RNG/grammar state, and
per-group bookkeeping. `MambaManager` keeps last_state_block_idx, retired-block
counts, allocation flags and checkpoint/partial-tail metadata beyond its block
table. `pop_blocks_for_free` removes these. A raw table transplant is insufficient.
Do not serialize arbitrary manager __dict__ objects: add typed detach/reattach
hooks and explicitly validate each supported manager/spec.

## Reuse packed physical pages when actually available

`offloading/worker.py::register_kv_caches` already detects interleaved manager
pages when a layer's stride exceeds its page size. It registers one full-page
CanonicalKVCacheTensor and identical whole-page refs for selected groups.

For that layout only, deduplicate all non-null physical block IDs in the
request's manifest and copy each complete manager page once. This preserves
raw rings and partial hybrid state without making them prefix-hashable or
inventing a tensor serializer. Preserve the group's sparse logical tables in
the manifest and replace old physical IDs through a restore map.

Registration must prove a single packed backing, bounds/stride/extent validity,
full-page refs with no fragment/canonical conversion, supported group specs,
and matching TP/layout/dtype/model identities. Actual Flash-Next worker layout
must be inspected and tested; the presence of this code path alone is not proof
that a given serve uses it. Reject unsupported layouts before reserving or
freeing anything. Do not change generic get_offloading_group_ids to include
rings: that would break positional prefix planning.

An internal physical-page transfer spec can translate to existing
GPULoadStoreSpec with the complete page-ID list assigned to one verified
full-page ref slot and zero counts in other slots. CPU chunk allocation and
copy execution stay in the existing native pool. Explicitly validate this
translation; do not rely on an incidental first-group ordering.

## Transaction and ownership

State transitions:

1. RUNNING -> QUIESCING: stop scheduling this request; retain GPU blocks.
2. Drain its in-flight scheduler output and acceptance corrections. Require
   num_in_flight_tokens=0, processed scheduling fences satisfied, and a worker
   event after target, drafter and state postprocessing. Wait on that event
   from the store stream. Host output arrival alone is not a GPU fence.
3. QUIESCING -> STORING: freeze the accepted frontier and manifest. Resolve or
   flush existing prefix stores that reference these blocks. Atomically reserve
   all snapshot CPU chunks across both ranks; admission filtering must not
   silently skip snapshot pages. If capacity is unavailable, retain GPU state
   and refuse/defer suspension without changing request semantics.
4. STORING -> HOST_RESIDENT: only after both ranks acknowledge every page,
   publish the immutable snapshot and detach/release request GPU references.
   Shared prefix pages may remain GPU-resident for other requests; releasing
   this request does not guarantee all those physical pages become free.
5. HOST_RESIDENT -> LOADING: reserve replacement GPU pages atomically, construct
   sparse per-group tables and restore into them. Preserve nulls/aliasing.
   Fence initialization/zeroing before H2D; do not expose tables to computation.
6. LOADING -> RUNNING: after both ranks finish restoration and typed manager
   metadata is reattached, restore the exact computed frontier and request
   state, rebuild batch metadata, and permit the next decode operation. Bypass
   prefix lookup and do not reset the frontier to zero. Preserve or explicitly
   invalidate request-indexed scratch/top-k reuse; batch-slot reuse is not state.

Use request-private keys domain-separated from prefix content keys, containing
engine generation, request generation, snapshot ID and physical-page ordinal.
A snapshot holds a persistent CPU residency lease until successful resume or
cancellation. Transient load refs are additional to that lease. Store completion
must transfer directly into the lease without an evictable interval. LRU/ARC may
evict best-effort prefix entries to make room; neither may discard a suspended
request's only state. Refuse capacity pressure rather than silently recompute.
Do not count a snapshot insertion/pin as a new frequency hit on every poll.

The scheduler needs explicit blocked/live-paging states or an equivalent paging
state plus queues. These must remain unfinished requests and count against
appropriate admission limits. If RequestStatus gains members, place them before
PREEMPTED or update is_finished: the existing enum treats later values as finished.
Existing WAITING_FOR_REMOTE_KVS can represent the loading phase, but cannot
represent durable HOST_RESIDENT state or outbound storage by itself.

## Proposed APIs

Names are a proposal, not existing interfaces:

- Scheduler: request_suspend(req_id) -> operation_id; request_resume(req_id).
  Suspension becomes effective only at a processed step boundary.
- KVCacheManager/coordinator: export_live_state(req_id) -> typed manager/table
  manifest; detach_live_state(req_id, manifest); allocate_live_restore(manifest)
  -> replacement tables; commit_live_restore(req_id, manifest, tables).
- Native connector: prepare_live_snapshot(req_id, frontier, manifest) -> handle
  or backpressure; prepare_live_restore(handle, replacement_tables) -> job_ids;
  release_live_snapshot(handle). Metadata carries a distinct job kind and
  generation so completions cannot enter prefix-store bookkeeping accidentally.
- CPU manager: atomic snapshot reservation plus acquire/release persistent lease.
  Reserve the complete page set before launch. Keep prefix admission/policy
  behavior unchanged for existing calls.

First implementation: explicit suspend/resume of one quiescent request,
worker-private packed pages, whole-request capacity reservation and exact
restore. No background automatic swapping policy until this passes qualification.

## Failure, cancellation and reset

- Before committed storage, failure retains the authoritative GPU state. Drain
  any started copies before releasing CPU reservations; restore RUNNING or fail
  the operation. Never claim HOST_RESIDENT after partial rank completion.
- After GPU release, failed restore keeps the CPU lease; discard failed GPU
  allocations only after their H2D jobs stop. Retry or explicitly fail the
  request. Do not silently fall back to prefix recomputation.
- Cancellation in any phase is idempotent. Retain each source/destination until
  its last GPU event and all-rank completion are accounted for, then free both
  resources. Reject late completions from an older request/snapshot generation.
- CPU manager reset now refuses live leases. The connector must preflight
  this refusal before clearing its own job/request state. Refuse connector
  reset while retained snapshots exist in the first slice, unless an explicit
  abort-all operation drains and releases every snapshot first.
- Process restart recovery is excluded. GPU loss fails affected requests; a
  same-process CPU pool is not durable storage. Bound stuck operations and
  report failure without unsafe reclamation of DMA-visible memory.

## Verification and acceptance gates

Extend nearby tests, rather than replacing the existing prefix suites:

- CPU manager LRU/ARC tests: atomic reservation, lease/pin transitions, full
  capacity backpressure, no eviction window, cancellation and reset rejection.
- Scheduler tests: two-phase free, no extra model step after quiesce, late and
  duplicate rank completion, async stale output, blocked-state counts, sparse
  table remapping and exact non-aligned frontier preservation.
- Native worker test: full packed-page byte parity including QSA raw ring,
  compressed KV, GDN and PLE/MTP slots; different destination IDs; dirty/NaN
  destination memory; writes queued before snapshot and zeroing before restore.
- TP2/MTP2 real model: pause mid-decode at compression/block edges and arbitrary
  acceptance counts, overwrite/reuse released GPU pages with a second request,
  then resume under eager and FULL_AND_PIECEWISE graphs. Compare continuation
  tokens/logprobs against a no-suspend control, repeated cycles and prefix hits.
  Prove no already-computed prefix rows executed again, and report saved bytes,
  reclaimed GPU refs/pages, CPU lease bytes and suspend/resume latency.

Existing reusable tests and commands (not run during this design audit):

```bash
.venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_manager.py \
  tests/v1/kv_offload/cpu/policies -q
.venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py \
  tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py -q
HIP_VISIBLE_DEVICES=6,7 .venv/bin/python -m pytest \
  tests/v1/kv_offload/cpu/test_gpu_worker.py \
  tests/v1/kv_connector/unit/offloading_connector/test_worker.py -q
```

Specifically preserve test_request_preemption, test_reset_cache,
test_async_preempt_readmit_before_transfer_output_is_deferred,
test_native_hybrid_cpu_roundtrip_is_byte_identical, and
test_load_waits_for_pending_compute_stream_writes. Those existing tests do not
qualify live paging: add the transaction, raw-ring and model continuation cases
above. No merge/readiness claim before those new gates pass on the actual layout.

## Integration audit: typed manager and worker state

The packed-page transport and native CPU leases are only foundations. Exact
Flash-Next TP2/MTP2 suspension additionally requires the following typed state.
Do not copy a manager or runner's `__dict__`: it includes shared pools, other
requests, transient queues and device objects with independent ownership.

### Per-group manager manifest

Use immutable records containing explicit optional fields, validated against
an allowlist of the actual concrete manager/spec pairs at startup:

- Common state: group ID and spec/layout fingerprint; the entire logical
  `req_to_blocks[req_id]` table, including null holes and trailing speculative
  slots; presence/value of `num_cached_block[req_id]`; and optional
  `_partial_hit_reqs[req_id] = (logical_index, source_page_handle)`.
  See `vllm/v1/core/single_type_kv_cache_manager.py:128` and `:149`.
- CircularBufferManager: its one-element physical ring table. It has no extra
  request-owned ring cursor; QSA derives ring positions from token positions.
  Preserve the exact token frontier and the ring bytes, including packed
  MRoPE positions when configured. Its normal prefix-cache lookup returns no
  hit and is unsuitable for restoration. See
  `vllm/v1/core/single_type_kv_cache_manager.py:1199` and
  `vllm/models/qwen4_exp/common/qsa_cache.py:113`.
- MambaManager in align mode: presence/value of `last_state_block_idx` and
  `_num_retired_blocks`; membership in `_allocated_block_reqs`; optional
  `_checkpoints = (checkpoint_token_position, logical_block_index)`; and optional
  `_producer_partial_tail_reqs = (source_page_handle, boundary_tokens)`.
  Logical indices remain unchanged when physical IDs change. Every stored
  KVCacheBlock reference must instead resolve through the page remapping.
  See `vllm/v1/core/single_type_kv_cache_manager.py:1474`, `:1800`, `:1862`
  and `:1914`.

Ordinary `pop_blocks_for_free` destroys these records: the base removes table,
cache cursor and partial-hit state (`single_type_kv_cache_manager.py:562`),
while Mamba additionally removes its align bookkeeping and queued boundary
handoffs (`:1971`). The coordinator simply concatenates those destructive pops
(`vllm/v1/core/kv_cache_coordinator.py:420`). Export before detach; do not infer
these records afterward from `num_computed_tokens` or table length.

Pending CoW copies and boundary-state handoffs are not passive manifest fields.
They own additional asynchronous source/destination lifetimes; handoff blocks
can already be outside the request's table. Initial integration must drain and
acknowledge them, plus external prefix stores and their retained endpoints,
before declaring the request quiescent. Reject suspension if any dependency
cannot be drained. Do not serialize a stale block pointer or silently discard
an offer. See `single_type_kv_cache_manager.py:422`, `:430` and
`vllm/v1/core/kv_cache_manager.py:896`. The per-step
`cached_blocks_this_step` set is shared step-local exclusion state, not a
request snapshot; suspend at a completed step boundary (`:2044`).

### Physical ownership, hashes and atomic allocation

A page manifest needs a unique `(pool_identity, old_physical_id)` handle per
non-null source page, plus references from every logical table position.
Transport each unique page once. Preserve table holes without allocating or
transferring null page zero. Ownership accounting must follow table references,
not the deduplicated transfer list; never restore an old absolute `ref_cnt`,
because other requests may have acquired/released the original shared page.

`BlockPool.get_new_blocks` checks total capacity before dequeuing and returns
fresh, unhashed blocks with one owning reference (`block_pool.py:669`). For the
initial single-pool packed layout, validate every group record first and reserve
all required destination pages with one call on the scheduler thread, before
publishing any table. Add references for any validated alias multiplicity;
never call `touch` on null holes. Keep destinations private in a reservation
until all loads succeed. On failure, drain H2D and free the reservation without
installing request tables. Defer multiple-pool restoration until an explicit
all-pool reservation/rollback protocol exists.

Freeing sources releases only this request's references, following the existing
pool's reverse-order semantics. It does not guarantee physical reclamation:
shared pages remain live for other owners, and unhashed/hashed idle pages have
different free-queue behavior (`block_pool.py:755`, `:773`, `:777`). Report actual
reclaimed pages separately from saved bytes. Mutable ring/live state must never
be deduplicated with another request's snapshot merely by prefix hash.

Preserve immutable prefix metadata separately from page bytes: each page's
primary `block_hash`, `block_hash_num_tokens`, and all secondary aliases in
`cached_block_hashes_by_block`. Restoring fresh unhashed pages while reinstating
`num_cached_block` would skip their cache publication; losing partial-tail
hashes also bypasses hash-driven copy-on-write protection. Publish the saved
hashes only after H2D completion. The hash map already supports multiple physical
blocks under the same hash (`block_pool.py:35`, `:85`), so a restored private
copy need not steal entries from a still-shared original. A narrow public
`restore_block_hashes(block, primary, num_tokens, aliases)` should wrap the
existing insertion logic at `block_pool.py:628`; do not use
`move_block_hashes`, which removes the original's entries (`:651`). Validate
request/model/cache generation before publishing any saved hash metadata.

### State outside the packed GPU allocation

The accepted speculative state selector is the most important additional
payload. `GPUModelRunner.mamba_state_idx[req_id]` is a logical source column
(`gpu_model_runner.py:1085`). The prior step's accepted count is held in
`InputBatch.num_accepted_tokens_cpu[row]`, backed by an asynchronous D2H copy and
`num_accepted_tokens_event` (`gpu_model_runner.py:1711`). Align postprocessing
can preserve a count greater than one; it does not always normalize the state
in place (`mamba_utils.py:1805`). The next preprocess uses `accepted_count - 1`
to select the correct conv/temporal state (`mamba_utils.py:1669`).

Synchronize that event, then snapshot the count by request ID, not by the old
batch row. `InputBatch.add_request` resets it to one (`gpu_input_batch.py:481`),
so restore the saved count after the resumed request has been added and rows
condensed, before `_prepare_inputs` copies accepted counts and preprocesses
Mamba. Preserve the logical `mamba_state_idx` at that point too. Existing
`cleanup_mamba_state_idx` removes entries for every resumed request
(`mamba_utils.py:1557`); live restoration needs an explicit exception for a
validated restored request, or a hook restoring its saved selector immediately
after that cleanup. Reusing the preemption-resume path unchanged is incorrect.

Retain each worker's `CachedRequestState` while it is absent from the batch:
its token history/frontier, RNG generator, sampling parameters, prompt-logprob
accumulator, prompt embeddings/masks, multimodal positions and
`prev_num_draft_len` are outside packed KV (`gpu_input_batch.py:35`). Existing
unscheduling already retains this object (`gpu_model_runner.py:1386`); finishing
removes it. Preserve the scheduler's Request object and committed output/token
history, speculative IDs, grammar state and accounting rather than rebuilding
it as a fresh request (`vllm/v1/request.py:160`). Initial suspension must wait for
`num_in_flight_tokens == 0` and all optimistic output/acceptance corrections;
ordinary `_preempt_request` resets the frontier and speculative IDs
(`scheduler.py:1806`) and must not be used for this transition.

PLE's short-conv history, including speculative slots, is allocator-owned
`kv_cache`; it is TP-replicated and both worker-private copies must be stored
(`vllm/models/qwen4_exp/amd/ple_layer.py:777`, `:788`, `:800`). QSA raw/compressed
side caches are also registered cache owners
(`vllm/models/qwen4_exp/amd/indexer_qsa.py:199`). The actual deployed capability
probe must prove all these owners, target GDN, and every MTP owner share the
certified backing. A positive one-tensor transport capability alone cannot
prove owner completeness.

Ngram context is rebuilt from retained token history and the exact frontier
(`gpu_model_runner.py:3646`). Batch-indexed QSA/top-k metadata, MTP per-group
block-table staging (`vllm/v1/spec_decode/qwen4_exp.py:30`), and graph input
buffers must be rebuilt for the new tables and batch slot. They are scratch,
not transferable request state. PLE embedding tables and model weights remain
resident/shared and are not part of the request snapshot. Initial integration
remains text-only; multimodal encoder ownership needs separate qualification.

### Minimal consuming APIs and restore order

Implement explicit records beside the existing managers, not a generic state
serialization framework:

- `SingleTypeKVCacheManager.export_request_state(req_id)` returns a common
  table record; Mamba overrides it with a typed align record. Concrete spec
  validation rejects unsupported managers rather than omitting fields.
- `KVCacheManager.export_live_state(request)` collects group records, unique
  physical pages and prefix-hash metadata without mutation.
- `detach_live_state(request, validated_manifest)` removes request bookkeeping
  and returns precisely owned references only after all-rank store success and
  the native CPU lease exist. It must check the manifest is still current.
- `reserve_live_restore(manifest)` validates all records and reserves the full
  destination set atomically; it returns an unpublished page remapping and
  replacement typed tables. `abort_live_restore(reservation)` frees it only
  after DMA drains.
- `commit_live_restore(request, manifest, reservation)` installs remapped tables,
  cache aliases and typed manager records after all-rank load completion.
- A worker-local typed record holds `mamba_state_idx`, accepted count and the
  retained CachedRequestState generation. `restore_live_worker_state` applies
  those selectors after batch placement and Mamba cleanup, before preprocessing.

The required order is: finish model/MTP and acceptance postprocessing; drain
output, CoW and prefix transfers; export the typed snapshot; reserve CPU pages;
store all ranks; complete stores directly into retained CPU leases; detach and
release source references. On resume: reserve all GPU destinations; finish any
required zeroing before H2D; load all ranks; publish hashes and typed tables;
replace worker block tables through the existing resumed-table path
(`gpu_model_runner.py:1582`); restore worker selectors after row placement and
cleanup; rebuild metadata; then allow the next forward. No zeroing or pending
CoW operation may overwrite restored pages after H2D.

Keep the CPU lease through this complete commit, not merely through DMA. A
failed rank leaves the request suspended with its CPU copy retained. Cancel,
abort and reset must fence the same generation and transfer ownership before
freeing either side. The current CPU manager rejects reset with live leases;
the connector must preflight that refusal before clearing its own job/request
state. Full hybrid MTP qualification remains mandatory before calling this
usable live paging.

Add manager tests for sparse/null tables, shared-prefix owner counts, hash alias
restoration and partial-tail CoW at different IDs; transactional allocation
failure with no partial publication; checkpoint and producer-tail remapping;
and cleanup/reset generation failures. Add worker tests with accepted counts
one, two and three, changed batch rows, same-block versus cross-block MTP state,
and repeated suspend/resume. Existing nearby regression cases include
`test_hybrid_coordinator_skips_non_cacheable_circular_group`,
`test_hybrid_model_mamba_align_with_dynamic_draft_tokens`,
`test_mamba_boundary_handoffs_do_not_pin_obsolete_blocks`
(`tests/v1/core/test_prefix_caching.py`) and
`tests/v1/core/test_deferred_block_free.py`. Only a real TP2/MTP2 continuation
comparison with no prefix recomputation can close the final gate.

## Confirmed target runner: Model Runner V2

The actual MonoKernel recipe and `/server_info` receipt select Model Runner V2
(`VLLM_USE_V2_MODEL_RUNNER=1`, async scheduling enabled). Its persistent
`RequestState` and `MambaHybridState` keep per-request GPU slots while a request
is unscheduled. The V1 accepted-count reset discussion above is a portability
hazard, not the target implementation path.

Live resumes must remain __cached requests__ in V2. Sending a normal resumed
request through `scheduled_new_reqs` calls `_remove_request`/`add_request`, resets
accepted counts and recurrent selectors, and loses RNG/request state. Instead,
retain the original slot and send `block_table_updates` to overwrite every group
row. Keep `resumed_req_ids` for the connector so it replaces its source tables.
See `vllm/v1/worker/gpu/model_runner.py:1135`, `:1180`, `:1237` and
`vllm/v1/worker/gpu/model_states/mamba_hybrid.py:207`.

Count host-resident requests against V2's total request-slot budget. Offloading
large KV pages does not free the small persistent request slot. Fresh admission
must not exceed `max_num_seqs` across running and suspended slots; an existing
suspended request may still restore at that limit. Initial support targets the
requested TP2/MTP2, FCFS, PP1, aligned-recurrent-state configuration. No V1 worker
serialization layer is needed for this implementation.

## Admission and restore-reader fences

After detaching a snapshot, move it out of the GPU-KV-holding priority queue
and behind fresh waiting admission. A ready restored request already owns its
GPU pages and V2 slot; it must not trigger another fresh-admission spill.
Blocked grammar/remote-KV waiters and adapter requests do not trigger proactive
paging. These queue/readiness cases have actual FCFS regression coverage.

Private restore reservations bypass the ordinary allocation callback. Register
their destination pages in the connector's existing allocated-page reuse set:
a finished request's ordinary prefix store can still read a logically freed
page. Store and load handlers have separate event chains, so submission order
alone is insufficient. The normal metadata fence flushes only readers of reused
destinations before H2D; unrelated stores remain asynchronous. The expanded
GPU-7 regression suite passes 801 cases, including both reuse and unrelated-page
checks. Normal-threshold performance is rerun on this fenced implementation.
