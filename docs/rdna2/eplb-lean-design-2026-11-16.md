# Lean EPLB for the RDNA2 expert-parallel MoE: design, phase-0 tooling and the numbers

| Item | Value |
|---|---|
| Date | 2026-11-16 |
| Tree | worktree `vllm-eplb`, branch `feat/eplb-lean`, base `rdna_extras` at `262ff0edd3` (v0.33 line) |
| Target | Flash-Next (Qwen4Exp, 48 routed layers x 512 experts, top-10) on TP4/EP4, six V620 behind one PEX880xx switch, W4A16 `moe_gptq_gemm_rdna2`, resident experts, FULL_AND_PIECEWISE graphs, MTP2 |
| Status | design plus phase-0 code. The phase-0 arms A1, B1 and L1 ran on the box (section 10); P8 and P1 (kernel profile) were still running when section 10 was written. The HIP counters are built and match the torch reference on a V620 (50 tests on HIP 5) |
| Conventions | `[INFERENCE]` marks anything neither measured nor read in code. Every PCIe number must be re-measured on the single-switch box (the old two-root numbers do not apply) |
| Companion | `docs/rdna2/a2a-dbo-feasibility-2026-11-16.md` (all-to-all and DBO: NO-GO at c=1..8; this study is its phase 1) |

## 0. Verdict

1. **Upstream EPLB does not start on our quantized MoE.** On gfx1030 the WNA16 oracle picks `RDNA2_W4A16`, so `supports_eplb` is False and `FusedMoE` raises `NotImplementedError("EPLB is not supported CompressedTensorsWNA16RDNA2MoEMethod.")` as soon as `--enable-eplb` is set (section 1). We do not need its machinery: a **permutation-only** placement changes only `expert_map` and the bytes of the moved experts, and every kernel already reads `expert_map` at run time.
2. **Placement alone buys almost nothing at EP4.** On the real T10 marginals a held-out packed placement removes about 0.8 % of MoE time at 3 rows (c=1, MTP2) and 1.3 % at 24 rows (c=8): about 0.10 % and 0.37 % of ITL. The 1.14 to 1.43 slowest-over-mean of a decode step is the maximum of random touch counts, which no placement changes; DeepInfer's `slowest/mean <= 1.09` is unreachable below about 48 decode rows (section 3).
3. **Redundancy is a bad trade here.** Metro-style routing (needs bit-identical top-k ids on every rank) gains +6.7 % of MoE time at 24 rows for 4 extra slots, which cost 468 MiB per rank (12.2 % of the KV budget); hash routing, the engine's default, makes decode slower (-3 % to -15 %).
4. **Prefill straggling is not a placement problem either.** The persistent expert skew packs away (1.106 to 1.004) but the chunk-to-chunk topic transient (median 1.39, measured) only drops to 1.377.
5. **Recommendation.** Build phase 0 (done, section 8) and phase 1 (static placement at boot with the existing `VLLM_RDNA_MOE_PLACEMENT`, no engine change). Build online rebalancing (phase 2) only if phase-0 data show drift, or a persistent imbalance above the sampling floor, at the batch sizes that matter. If the measured per-layer decode `T` max/mean is already near the floor of section 3.1, stop at phase 0 and say so.
6. **Phase-0 result (section 10).** Measured on the L1 traffic (the `hi2` suite behind the Cline prompt, 8 workers), decode `T` max/mean is 1.435 at 3 rows and 1.164 at 24 rows against sampling floors of 1.416 and 1.154 (median layer), and the held-out placement gain is 0.07 to 0.84 % of the slowest rank at 24 rows (0.02 to 0.19 % of ITL) and under 1 % at 3 rows: the stop rule of section 7 is met for decode, so no phase 1 and no phase 2 for ITL. Prefill placement is a free file worth about 1 % of TTFT, optional. The counters' ITL cost is unresolved (one off boot; +0.6 to +0.9 % at c=1). The ranks disagree on one routed pair, always on rank 2, in pure 1023-row prefill passes; decode is clean.

## 1. Upstream EPLB against the fork's layouts

All of this is a code read of this tree (`[INFERENCE]` until a box run constructs the model).

| # | Finding | Where |
|---|---|---|
| F1 | **Hard refusal.** `CompressedTensorsWNA16RDNA2MoEMethod` inherits `supports_eplb`, which is `wna16_backend == TRITON`; the oracle returns `RDNA2_W4A16` first on gfx10x when `moe_gptq_gemm_rdna2` is built, so `FusedMoE.__init__` raises when `enable_eplb` is on | `routed_experts.py:189-201`, `compressed_tensors_moe_wna16.py:562-563`, `oracle/int_wna16.py:141-150` |
| F2 | **What a move carries.** `get_expert_weights()` returns the parameters as E-leading views: `w13_weight_packed`, `w2_weight_packed` (K-first int32 `[E, K/8, N]`), `w13_weight_scale`, `w2_weight_scale` (fp16 `[E, groups, N]`), `w13_qzeros`, `w2_qzeros`. One expert is six contiguous slices. Symmetric W4A16 qzeros are constant (19,200 B per expert): moving them is wasted | `routed_experts.py:1364-1434`, `compressed_tensors_moe_wna16_rdna2.py:226-290` |
| F3 | **Bytes.** Expert 2,534,400 B (packed int4 2,457,600 + fp16 scales 76,800); parameter slot 2,553,600 B. 128 local experts per layer per rank, 14.5 GiB per rank over 48 layers. One extra slot on every layer = 122,572,800 B per rank = 3.0 % of the KV budget (8,126 of 266,945 tokens). The MTP layer holds 512 BF16 experts of 9,830,400 B and is not moved or counted | `tools/rdna/eplb/geometry.py` (from the checkpoint headers) |
| F4 | **Routing translation.** Upstream maps logical to physical ids with a Triton kernel per layer plus `topk_ids.contiguous().to(int32)` and a `[L, E, 1024]` int64 map (about 192 MiB per rank). Permutation-only needs neither: logical id to (rank, slot) is `expert_map` | `router/base_router.py:129,148` |
| F5 | **Upstream rebalance.** Host sync on the load window; ROCm-only skip below 5 % relative improvement of the cross-layer, token-count imbalance; default window 1000, interval 3000, async on; the default communicator is NIXL if present, else gloo (host staging); `torch_nccl` and `pynccl` are rejected with async; `enable_migration_batching` exists. The default policy balances token counts, so replication is counterproductive for bandwidth-bound decode | `eplb_state.py:876-924`, `config/parallel.py:67-130,1123-1141` |
| F6 | **Exclusions.** The expert cache rejects EPLB; mono K2 needs the linear contiguous block; DSV4 mono decode refuses it; redundancy breaks the tile8 prefill gate (`w13_weight_packed.shape[0] == 128 and global_num_experts == 512`); `num_redundant_experts` is rejected without EPLB | `rdna_expert_cache.py:66-67`, `deepseek_v4/amd/mono_decode.py:153`, `config/parallel.py:592-597` |
| F7 | **Model hooks exist.** `Qwen4ExpMixtureOfExperts` and the MTP model implement the EPLB hooks and the V2 runner has `eplb_utils.py`; they are never exercised because of F1 | `qwen4_exp/amd/model.py:409-455`, `amd/mtp.py:420` |
| F8 | **Static placement already exists.** `VLLM_RDNA_MOE_PLACEMENT=<json>` (v1: `layers["<idx>"][rank] = [global experts]`) changes only the `expert_map`; the loader writes each expert straight into its slot. Non-uniform lists fall back to linear, so EP6 (86/86/85/85/85/85) is not covered | `rdna_moe_placement.py`, `expert_map_manager.py:230-245` |
| F9 | **Shared experts** run through `runner/shared_experts.py`; `_no_fork_in_capture = on_gfx10x()`. A copy must never fork a stream inside a captured graph (a captured aux-stream fork wedges queues about once per 1 to 4 h) | `runner/shared_experts.py` |

Paths a physical move can take (all bandwidths `[INFERENCE]` until `tools/rdna/a2a/p2p_bench.py` runs on the single-switch box; the lease README measured about 13.9 GB/s P2P on the interim x8 box, sysfs says x16 16 GT/s, the user says x8):

| Path | Hops | Notes |
|---|---|---|
| Peer copy over the switch (`hipMemcpyPeerAsync` or a copy kernel) | 1 | needs no host buffer; competes with the rdna_ar all-reduce on the same switch, the ITL risk to measure |
| Host staging (D2H, then H2D; what the gloo or NIXL communicator does) | 2 | host DMA and pinned memory; the pinned-host eviction rules of AGENTS.md apply |
| Offload cache slots (pinned host to slot, `PackedExpertCache._copy`) | 1 | source is the host bank, not a peer; requires `--monokernel` and excludes EPLB: a model for the protocol, not a path |

One packed expert is 2.55 MB: about 182 us of link time at 14 GB/s, 102 us at 25 GB/s `[INFERENCE]`.

## 2. (b) Lean load counting (built)

`VLLM_RDNA_EPLB_STATS_DIR=<dir>` turns the counters on; off, the call sites take the original branch (AST-tested) and the existing launch is the same template instantiation with the original arguments and grid.

- **Where.** In the HIP routing kernel `moe_route_zero_k<kStats>` one extra CTA (the last of the grid) counts the call's top-k ids, so the fused decode launch stays one launch. Calls that take another routing branch (more than 1024 entries, deterministic, resident skinny, prefill) use the standalone single-CTA op `rdna_eplb_stats`. The torch formulation (`LayerStats.count_torch`) is the CPU oracle and the `VLLM_RDNA_EPLB_STATS_KERNEL=torch` fallback.
- **What.** Per call and per EP rank: touched experts `T` and routed pairs `P`, in a device ring (record = 32 int32 words: layer, valid rows, total rows, id hash, clock, phase, ordinal, `T` x8, `P` x8). Cumulative tables per layer: `pairs[2][E]` (decode, prefill) and `touch[band][E]` per decode row band, plus an optional prefill histogram ring (`HIST`, int16 `[H, E]` per layer).
- **Cost model.** No host sync, no atomics on the hot path (one writer per state word), padding rows excluded. Defaults: ring 262,144 records = 32 MiB per rank, 15 row bands. `VLLM_RDNA_EPLB_STATS_EVERY` (power of two) samples. Whether the stats CTA hides behind the zero-fill CTAs is `[INFERENCE]`; gate G0.1 measures it (0.5 % of c=1 ITL is 2.3 us per layer).
- **Dump.** `gpu_worker`'s existing 30 s tick copies the state, writes `eplb_stats-rank<R>-pid<P>.npz` (`eplb-stats/1`), atomically. `tools/rdna/eplb/summary.py` reads, cross-checks the ranks (every rank sees the whole batch's top-k ids under all-gather/reduce-scatter EP, so the tables must agree; the id-hash word is the consensus check) and exports `hetero-moe-expert-stats/1` for the solver.
- **No exchange needed.** At TP4/EP4 with dp=1 every rank already holds all ids, so no count-vector all-gather exists. With DP attention a cheap `[E]` int count-vector exchange (E x 4 B = 2 KiB) would replace any large gather: phase 2 only.
- **Limits.** K2 stands down while the counters are on (it hides the ids); the expert-cache path and the MTP BF16 layers are not counted; one stream, one writer (DBO would need a ring per micro-batch); cumulative tables, so cell data and agentic data need separate arms (done in `phase0.sh`).

Windows (DeepInfer, the starting values of the policy): prefill EMA over 100 steps, decision every 1000; decode EMA over 1000 steps, decision every 10,000; target slowest/mean at most 1.09 `[the 1.09 is unreachable for decode below about 48 rows, section 3.1]`.

## 3. (a) Cost model and simulation

Method: `tools/rdna/eplb/simulate.py`. EP4, 12 layers (`0:48:4`), placements fitted on the T10 W1 window and scored on W2 (held out), MTP2 (3 rows per request), top-10 per row, row correlation fitted on a real id trace. A rank's layer time is `fixed 3.1 us + sum over touched experts of max(6.7 us, 0.68 us x rows)` (decode is a weight read of 2.53 MB at about 380 GB/s; prefill is compute bound). The model gives 58 us per layer at c=1 against 52 us measured, and about 12.0 ms of MoE at c=8 against about 10.1 ms measured (19 % high), so every percentage of ITL below is an upper bound. The Zipf skew model (alpha 0.5857, top-64 share 0.388) reproduces the real-marginal results.

### 3.1 Decode: E[max]/E[mean] of touched experts T, EP4

| Rows (MTP2: 3 x batch) | linear | best permutation (`pack_touch`) | MoE ms per step, linear | gain of the permutation |
|---|---|---|---|---|
| 3 (c=1) | 1.432 | 1.421 | 2.80 | 0.77 % of MoE |
| 6 | 1.309 | 1.294 | 4.64 | 1.11 % |
| 12 | 1.222 | 1.206 | 7.58 | 1.28 % |
| 24 (c=8) | 1.160 | 1.144 | 12.04 | 1.33 % |
| 48 | 1.114 | 1.098 | 18.38 | 1.44 % |
| 96 | 1.080 | 1.066 | 26.61 | 1.27 % |
| 192 (c=64) | 1.057 | 1.044 | 36.65 | 1.05 % |

- ITL translation: 0.10 % of 22.38 ms (c=1) and 0.37 % of 43.97 ms (c=8): not measurable. The oracle placement fitted on the test steps is within 0.001 of the packed result, so the rest is sampling noise, not skew.
- MTP0 (rows = batch): linear 1.670 (1 row), 1.234 (8), 1.081 (64); packed 1.664, 1.217, 1.066.
- Pairs `P` and touched experts `T` disagree: P is 1.236 (linear) at 24 rows against T 1.160. **Which one the measured MoE kernel time follows is the question phase 0 answers** (`objective.py`, section 8); the model assumes T for decode, the bandwidth-bound reading.
- EP-size, persistent touch imbalance (W=24): EP2 1.016 to 1.0045, EP4 1.066 to 1.0096, EP8 1.120 to 1.0195. Real T10, EP4: linear 1.112 (prefill) and 1.120 (decode) against held-out LPT 1.004 and 1.010.

### 3.2 Prefill

Persistent marginal imbalance (pairs): linear 1.106, round robin 1.113, packed 1.0036. The measured busiest-rank MoE GEMM is 1.11x to 2.04x the rank mean per layer (median 1.39x, 2048-token step of the 11-02 profile; 36 of 48 layers have the same slowest rank in 70 % of steps), and the 11-03 static-placement test was within noise. Chunk-transient model (Dirichlet alpha 0.0564, calibrated so the linear median is 1.39 `[INFERENCE]`): linear 1.392, packed 1.377, oracle 1.371. A marginal placement removes about 1 % of the straggler factor; only `pack_empirical` on real per-call histograms (arm L of phase 0) can touch the transient part.

### 3.3 Redundant experts: ITL gain against VRAM and KV cost

One slot on every layer = 117 MiB per rank = 3.0 % of the KV budget. Gain over the permutation-only placement, MoE time, 24 rows:

| Extra slots per rank | MiB per rank | KV lost | `min_touch` routing | `hash` routing | `primary` routing |
|---|---|---|---|---|---|
| 2 | 234 | 6.1 % | +3.9 % | -3.1 % | -0.5 % |
| 4 | 468 | 12.2 % | +6.7 % | -5.2 % | -0.7 % |
| 8 | 935 | 24.4 % | +9.6 % | -8.0 % | -0.7 % |
| 12 | 1403 | 36.5 % | +10.2 % | -10.3 % | -0.8 % |

At 3 rows `min_touch` gives +5.7 / +8.8 / +12.9 / +15.5 % at 2 / 4 / 8 / 12 slots. With experts at 11 % (c=1) and 23 % (c=8) of the step, +6.7 % of MoE is about 1.5 % of ITL at c=8 for 12 % of the KV: poor. `min_touch` (Metro, arXiv:2512.09277) needs every rank to reach the same decision from bit-identical top-k ids; T10 saw ranks disagree on at most 2 rows in 1e7, so it would need the ring's id-hash consensus plus a fallback. Metro's own numbers at 1.5x replication (Qwen3-30B, 8xA100): +30 % activated experts, +14 % decode latency, -10 % throughput. The "2.4 GB per redundant expert" of the vLLM docs is the DeepSeek-V3 figure; for Flash-Next use 117 MiB per slot per rank. Relief (arXiv:2611.11402, abstract only) reports per-step balancing beating static placement by up to 37.3 points at 16 or more GPUs; not verified for 4.

## 4. (c) Rebalancing policy with hysteresis (built as a model)

`tools/rdna/eplb/policy.py`, a pure function of the EMA, the loads the ranks really had and the placement in force, so replay, boot-time solver and a phase-2 balancer share it.

1. **Weight** per expert: EMA of the touched flag (decode) or of the pairs (prefill).
2. **Candidate** = a prefix of the swap sequence of `placement.repair_owner` from the placement in force. On the W1 marginals a full descent from linear moves about 17 experts per layer (8.5 swaps); 50 / 80 / 90 % of the in-sample drop of the heaviest rank takes 2.5 / 3.6 / 4.7 swaps at 24 rows (1.6 / 2.8 / 3.2 at 3 rows); 84 to 86 % of the drop survives on the held-out window. `pack_owner` from scratch reshuffles 380 to 390 of the 512 experts: 23x more bytes.
3. **Gain** = `mean over calls of max_r L - mean max_r (L + delta)` (`shifted_gain`): the realised per-call `T` or `P` of each rank (the ring) with the means moved by the candidate's expected change. The EMA-only predictor (`predictor="expected"`) ignores that the slowest rank is mostly the maximum of noise and overstates the gain:

   | Rows | predicted by the EMA alone (share of slowest-rank expert time) | realised by the packed placement (share of MoE time, section 3.1) |
   |---|---|---|
   | 3 | 9.9 % | 0.77 % |
   | 24 | 6.0 % | 1.33 % |
   | 192 | 2.9 % | 1.05 % |

   The denominators differ slightly (fixed cost), so the ratio is approximate. A deterministic test (`test_decide_swaps_acts_on_a_skew_above_the_noise_and_not_below_it`) shows the same: with a 5 % persistent skew and noise above it, the old predictor rebalances (48 experts) while the new one declines; below the noise it moves one swap.
4. **Move cost** = busiest rank's bytes over the link x 0.25 stalled share + 300 us per layer `[INFERENCE: stall share, fixed cost]`, at 14 GB/s `[INFERENCE; 25 if x16]`. One swap is 2 experts, 5.1 MB over two ranks, about 0.35 ms stalled-equivalent per layer; the reshuffle is 4.9 ms per layer.
5. **Hysteresis.** Rebalance only if (a) the best prefix has `horizon x gain >= safety (2) x move`, (b) the gain is at least 1 % of the slowest rank's time, and (c) each half of the window passes when judged with the other half's weights (cross-validation: the noise that shaped the candidate cannot vouch for it).

At the numbers of section 3.1 (0.4 % of ITL at best) a rebalance is cheap enough to pass (a) and still unmeasurable, which is why phase 2 is conditional.

## 5. (d) Move mechanism (phase 2 design, modelled on the offload cache)

`PackedExpertCache.ensure()` is the template: `ExpertCachePolicy.acquire()` plans a lease, copies run on the stream that runs the pass (stream order is the fence; consumers on other streams are fenced by `_consumer_event`), `commit()` publishes, `release()` records the consumer event, `cancel()` rolls back; host planning raises inside capture (`is_capturing_expert_work`).

1. **Plan** on the CPU at a step boundary from the EMA snapshot: per layer a swap list and an epoch id. Every rank holds identical counters (consensus-checked) and the policy has no randomness, so every rank computes the same plan with **no control-plane message**; ranks commit at the same step number (steps are lockstep across the TP group). A plan-hash mismatch must disable rebalancing and log loudly, never serve with inconsistent maps (an expert owned by nobody is silently dropped).
2. **Lease** per layer: one spare slot per swap in flight on each rank (2.55 MB each, 8 swaps in flight = 20 MB), taken from a staging pool; no per-layer 327 MB intermediate buffer as upstream uses.
3. **Copy** the incoming expert into the spare slot on a side stream between steps, outside graph replay and never forking inside a capture (F9), chunked to a per-step byte budget, ending in an event. The path is the peer copy of section 1, host staging as the fallback; the copy only writes spare slots, which the forward never reads.
4. **Commit** at a step boundary on the forward stream: a local D2D copy spare slot to slot (2.55 MB, about 8 us at 300 GB/s `[INFERENCE]`), then an in-place `copy_` into `expert_map` (the graph holds the pointers of weights and map: **never reallocate**), the counters' owner table, the epoch id; then `release()` records the consumer event. Cancel on failure and keep the old map.
5. **Graphs.** FULL_AND_PIECEWISE replays read the weight and `expert_map` pointers, so in-place updates are all that is allowed. The update is data, not control flow; no new op reads runtime metadata, so the breakable-graph rule (`@eager_break_during_capture`) adds nothing.

## 6. (e) Interactions

| Piece | Interaction |
|---|---|
| TP4 expert-sharded kernels | Resident skinny (M=1..4), fused route-zero, `moe_align_block_size` and deterministic branches all read `expert_map` per call: compatible with an in-place update. The counters hook every routing branch once |
| Mono K2 | needs the linear contiguous block: incompatible with placement, and it stands down under the counters |
| Expert cache | rejects EPLB; incompatible |
| W4A8 SDOT4 | "same weight buffers, different packing" (`compressed_tensors_moe_wna16_rdna2.py:292`): a move of the six parameters covers it; derived per-layer buffers (`rdna2_w1_buf`) are not per expert |
| MTP2 | 3 rows per request set the decode row band; the MTP layer's BF16 experts are neither counted nor moved; MTP acceptance must stay within noise in the gates |
| Prefix cache | KV and GDN state do not depend on the placement. A permutation re-associates the fp16 partial sums of the all-reduce, so outputs are not bitwise identical: the gate is PPL within noise (11-03 P1: 4.87 against 4.85) and greedy identical on the probes |
| TunableOp | independent: the MoE HIP path has no TunableOp GEMM and a permutation changes which experts a rank touches, not any shape |
| Breakable graphs | counters read device tensors only; `layer_stats` skips CUDA capture (`is_current_stream_capturing`); the tick runs outside graphs |
| PLE | unrelated to routed experts; no interaction |
| Quant layouts | W4A16 resident K-first int32 plus fp16 scales plus constant qzeros; AWQ and GPTQ checkpoints load through the same compressed-tensors WNA16 path; asymmetric checkpoints add real zero points that must move (the resident layouts are symmetric) |
| a2a and DBO | `docs/rdna2/a2a-dbo-feasibility-2026-11-16.md`: NO-GO at c=1..8. The interface an a2a would take from this study is one placement table per layer (`expert_map`), swapped at a step boundary with double-buffered slots |

## 7. (f) Phased plan and exact acceptance gates

Thresholds are `[INFERENCE]` choices; an ITL arm is invalid if KFD `evicted_ms` jumps or a quiet window is broken.

| Phase | Deliverable | Gate |
|---|---|---|
| 0 measure | counters, summary, simulator and replay, `phase0.sh`, ITL gate (this branch) | **G0.1** counters on against off, median and p99 ITL per cell (c=1,8 x 1k,16k x MTP0,MTP2): `\|on - off\| <= max(spread of two off arms, 0.5 %)`; 0 KFD evictions; PPL and greedy probes identical (the counters change no output). **G0.2** 4 of 4 rank files, id-hash consensus clean. **G0.3** `objective.py` names T or P for decode kernel time. **G0.4** per-layer measured T and P max/mean at c=1,8, drift between the two agentic halves, `solve.py` held-out gain. **Stop rule:** measured decode T max/mean within 0.02 of the sampling floor, or held-out gain below 1 % of MoE time, ends the programme here |
| 1 static placement | `solve.py` output as `VLLM_RDNA_MOE_PLACEMENT` (no engine change; not with `--monokernel`) | only if G0.4 predicts at least 1 % of MoE time: ITL p50 and p99 within noise of flag off (not slower), PPL within 0.03 of flag off, greedy identical on 3 of 3 probes, 0 KFD evictions |
| 2 online | the move mechanism of section 5 | only if phase 0 shows drift: P2P copy bandwidth measured with concurrent decode, ITL added during a rebalance window within the noise of the gate, 0 KFD evictions, at least 4 h soak with no rdna_ar wedge, no oscillation (at most 1 rebalance per 10,000 steps on stationary traffic) |

## 8. Phase-0 implementation

| File | Role |
|---|---|
| `csrc/rocm/rdna_fused_glue.cu`, `ops.h`, `torch_bindings.cpp`, `vllm/_custom_ops.py` | stats CTA in `moe_route_zero_k<kStats>`, `rdna_moe_route_zero_stats`, `rdna_eplb_stats` |
| `vllm/model_executor/layers/fused_moe/rdna_eplb_stats.py` | env, `LayerStats`, `EplbStats`, npz writer; hooks in `compressed_tensors_moe_wna16_rdna2.py`, `rdna2_moe_resident.py`, `k2.py`, `gpu_worker.py` |
| `tools/rdna/eplb/{geometry,placement,loads,simulate,policy,traces,summary,solve,objective,itl_gate}.py` | CPU tooling |
| `tools/rdna/eplb/phase0.sh`, `load_hook.sh` | box runner (lease and quiet protocol of `t10_run.sh`): arms A1 A2 B1 B2 Z1 Y1 L1 L2 G P8 P1, cells `1024:512:1 1024:512:8 16384:1024:1 16384:1024:8`; `DRY=1` prints the plan, `ANALYZE=1` runs only the analysis |

Env (also in `docs/rdna2/tuning.md` and `RDSE_ENV_KEYS`): `VLLM_RDNA_EPLB_STATS_{DIR,RING,HIST,EVERY,INTERVAL_S,DECODE_MAX,KERNEL}`.

CPU test commands, from the worktree root:

```bash
uv run --no-project --python 3.12 --with numpy --with regex --with pytest python -m pytest --noconftest -p no:cacheprovider tests/tools/rdna/test_eplb_model.py tests/tools/rdna/test_eplb_io.py
uv run --no-project --python 3.12 --with-requirements tools/rdna/eplb/cpu_test_requirements.txt python -m pytest -p no:cacheprovider tests/model_executor/layers/fused_moe/test_rdna_eplb_stats.py tests/tools/rdna/test_eplb_model.py tests/tools/rdna/test_eplb_io.py
```

Before a box run: rebuild `_rocm_C` (the new ops), run the 3 GPU parity tests of `test_rdna_eplb_stats.py` (skipped on CPU), then `phase0.sh` (it refuses lean arms on a tree without `rdna_eplb_stats`).

## 9. Caveats

- The HIP counters are built and match the torch reference on a V620 (HIP 5, 50 tests); the stats CTA's critical path is still unmeasured (section 10.1 infers 2.7 to 4.2 us per layer from the ITL deltas, P8 and P1 settle it).
- `bench serve` cells use random tokens, so their routing is not real traffic; representative routing comes from the agentic arms L1, L2 and G. Cumulative tables are per process.
- The objective join (`objective.py`) is strict: whole-process rocprof trace, unwrapped ring (`RING=1048576`), dispatch count equal to 2 x ring records, else `JoinError`; profile arms use `CG_MODE=NONE`.
- The old collector is the only id-trace source (`VLLM_EXPERT_STATS_TRACE_LAYERS`); the lean ring holds per-rank `T` and `P`, not per-expert sets.
- Link bandwidth, stalled share, per-layer fixed cost, the stats CTA's critical path and the move copy's effect on the all-reduce are all `[INFERENCE]`.
- Moving experts changes fp16 reduction association: PPL and greedy gates, not bitwise equality.

## 10. Phase-0 measured results (arms A1, B1, L1; box par1-cs25, 2026-11-16)

Tree `818fa99830` (this branch plus the gdn fix), recipe `flashnext-mtp2` (TP4/EP4, MTP2, 8 sequences, 2048 batched tokens, mamba `align` with block 1024), HIP 2-5 (buses d9, dc, df, e2), run dir `~/w4a8_runs/eplb-p0b/arms/{A1,B1,L1}`. A1: counters off, the four ITL cells. B1: lean counters on (`EVERY` 1, HIP kernel, 30 s tick), the same cells. L1: counters plus `HIST=512` over two passes (1,245 s) of the `hi2` task suite behind the Cline system prompt (`agentic_bench.py`, 8 workers, 420 requests per pass: gsm8k, humaneval, mbpp, ifeval, toolcall, needle, decision), a synthetic agentic mix of single-turn tasks and **not a recorded multi-turn session**. Arms A2, B2, Z1, Y1, L2 and G did not run; P8 and P1 were still running. The ring holds the newest 262,144 records, the last 5,461 calls of each of the 48 layers (B1 saw 7,324 calls per layer, L1 9,556 of which 1,953 prefill); all four rank files carry the same `ring_total` (351,552 and 458,688). Everything below is CPU analysis of the dumps (`tools/rdna/eplb/summary.py` plus scratch scripts on `placement.py` and `policy.py`).

| Gate | Result |
|---|---|
| G0.1 counters against off | **inconclusive**: one off boot, no spread; c=1 +0.6 to +0.9 %, c=8 -1.8 to +1.0 % (10.1) |
| G0.2 4 of 4 files, consensus clean | 4 of 4 files, **consensus not clean**: rank 2 disagrees on one routed pair in pure 1023-row prefill passes; decode is clean (10.2) |
| G0.3 T or P names the decode kernel time | open, needs P8 and P1 |
| G0.4 floor, drift, held-out gain | decode: stop rule met; prefill: 5.5 % of the slowest rank's pairs held out (10.3, 10.4) |

### 10.1 ITL gate, A1 against B1: inconclusive

| cell | median ITL A1 to B1 (ms) | p99 ITL (ms) | median TTFT (ms) | output tok/s | MTP acceptance length |
|---|---|---|---|---|---|
| 1k c1 | 22.48 to 22.68 (+0.85 %) | 23.07 to 23.20 (+0.59 %) | 397.98 to 403.29 (+1.34 %) | 83.82 to 70.27 (-16.2 %) | 2.017 to 1.685 |
| 1k c8 | 46.24 to 45.40 (-1.80 %) | 379.93 to 383.87 (+1.04 %) | 539.59 to 543.09 (+0.65 %) | 284.75 to 277.53 (-2.5 %) | 2.138 to 2.033 |
| 16k c1 | 23.02 to 23.15 (+0.58 %) | 23.64 to 23.80 (+0.68 %) | 5923.67 to 5971.68 (+0.81 %) | 68.56 to 66.02 (-3.7 %) | 2.607 to 2.483 |
| 16k c8 | 48.89 to 48.41 (-1.00 %) | 431.19 to 432.65 (+0.34 %) | 7488.29 to 7027.98 (-6.15 %) | 127.53 to 124.15 (-2.7 %) | 2.506 to 2.365 |

All cells are valid (0 ms KFD eviction on the four workers in both arms). `itl_gate.py` marks every row `unknown`: its rule is `max(spread of two off arms, 0.5 %)` and there is one off boot.

- **What the gate cannot resolve: boot-to-boot noise.** The same recipe's documented same-arm pairs (slot B ABBA, `hwq-abba-2026-11-06.md`) differ by 0.06 to 0.63 ms (B1 against B2) and 0.88 to 1.59 ms (A1 against A2, 3 to 4 % of the median). The deltas here (+0.20 and +0.13 ms at c=1, -0.84 and -0.48 ms at c=8) sit inside that range, and the c=8 medians are lower with counters on, which no counter can cause: the noise at c=8 is at least that large.
- **What points at a small real cost at c=1.** All four c=1 metrics (median and p99, both contexts) move +0.6 to +0.9 % the same way. 0.13 to 0.20 ms per step is 2.7 to 4.2 us per layer over 48 layers, the order of a stats CTA that does not hide behind the zero-fill CTAs `[INFERENCE]`. The prefill path agrees: the c=8 p99 ITL is a 2048-token chunk step (380 to 430 ms) and moves +3.9 and +1.5 ms, TTFT at c=1 moves +1.3 % (1k, +5.3 ms) and +0.8 % (16k, +48 ms), about 60 to 125 us per layer-call of the single-CTA standalone op (8 to 16 prefill passes of 48 layers) `[INFERENCE]`. The ABBA's TTFT noise is 1.4 %, so none of this is resolved.
- **Throughput is acceptance noise, not step time.** B1's MTP acceptance length is lower in 4 of 4 cells and tok/s follows acceptance over ITL: (1.685 / 2.017) x (22.48 / 22.68) = 0.828 against 0.839 measured at 1k c1. The ABBA saw the same swing between same-arm boots (1k c1 acceptance 1.50 and 1.94 for A1 and A2) and the counters change no output; if A2 and B2 repeat 4 of 4, that stops being an argument.
- B1's cells ran 3 C hotter on average (HIP 2-5: 69.9, 68.1, 71.9, 69.9 C against 66.9, 65.3, 68.6, 67.3 C in A1; peaks 77 to 82 C in both).
- **Next arms.** A2 and B2 (`ARMS="A2 B2"`; the full order `A1 B1 B2 A2` gives two boots per side) measure the spread; then Z1 and Y1 (MTP0, one row per request: the step where a fixed per-layer cost is the largest share). If the c=1 cost is real, the counters are a measurement tool and not an always-on feature: every placement gain in 10.4 is smaller than +0.6 % of ITL.

### 10.2 EP4 consensus: a real, narrow disagreement, not a counter artefact

`summary.py` reports `consensus ok False`: 87 of 262,144 compared records (B1) and 5 (L1). It now prints who deviates (`detail`, built for this study): B1 `odd_rank {2: 87}`, `phase {prefill: 87}`, `rows {1023: 87}`, `loads_differ 18`, `hash_only 69`, `passes 7`.

- **Same rank, same shape.** All 92 records are a 3-against-1 hash split with rank 2 the odd rank. All sit in 8 forward passes (7 in B1 at records 96,240, 113,088, 165,504, 184,944, 204,384, 227,808 and 246,384; 1 in L1 at 198,720), each a **pure prefill pass of exactly 1023 valid rows** with no decode rows (the next pass has 1 row: the block-aligned last chunk of a 1024 k-token prompt under `align`, `[INFERENCE]`). 7 of 7 such passes in B1's window and 1 of 1 in L1's are affected, 7 to 16 of the 48 layers each, spread over all layers (0 to 5 per layer).
- **Nothing else disagrees.** Outside the affected passes all records agree (261,808 in B1, 262,096 in L1), and so do the unaffected layers inside them: 237,136 and 181,456 decode records (rows 1 to 24 and 1 to 191, all MTP verify rows included), 4 passes of 1024 rows, 28 and 382 passes of 2048 rows, and 482 mixed passes of 1026 to 1045 rows in B1 (a 1023-row chunk plus 3 to 22 decode rows, `[INFERENCE]`).
- **One entry.** Of the 87 B1 records, 69 have identical per-rank `T` and `P` (positions swapped inside a row, or an expert replaced by one of the same owner) and 18 differ by exactly one routed pair moving between two owners (`P` +1 and -1, `T` unchanged); L1's 5 are all hash-only. So rank 2 differs in one entry (one token, one slot) of about 10,230 per affected layer-call; in 14 of the 18 load-changing records the pair moves to or from rank 2's own experts.
- **Why it is not an artefact (b).** The layer, rows, total, phase and call-ordinal words agree on all 4 x 262,144 records, so the rings are aligned (no sampling, no wrap shift, equal `ring_total`). A torn read at dump time would hit the ring head (slot of record 89,408 in B1; the first mismatch is 6,835 records away) and corrupt other words of the record. `T` and `P` come from the same CTA and the same ids, so one owner-to-owner pair move in `P` can only come from a different id in rank 2's `topk_ids`. The same kernel gives equal hashes on ranks 0, 1 and 3 for the same ids, so a tail bug of the standalone op at 10,230 entries would hit all ranks alike. Valid rows (word 1) are 1023 on all ranks, so a stale padding mask is out.
- **Not timing or hardware.** Per-layer record spacing and the cross-rank skew in the affected passes are the same on the four ranks; sysfs shows four identical cards (x16 16 GT/s, 144 SIMDs, same clocks, RAS ue/ce 0).
- **Where it enters: the replicated router GEMM, through the per-rank TunableOp rows (isolated in section 11).** Every rank multiplies the same `[rows, 2560]` hidden state by the same `[512, 2560]` router weight, and worker `r` reads `tunableop_results<r>.csv`. The shipped files name `...046` for `tn_512_1023_2560_ld_2560_2560_512` on ranks 0, 1 and 3 and `...044` on rank 2; the two solutions round 1,678 of 523,776 fp16 logits differently on the same inputs, and 1023 is the only router shape of these passes whose rows are split. The three candidates listed here before were wrong: the shared-expert aux stream only forks up to 256 tokens (`VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`), a 1023-row pass reduces over RCCL (5.2 MB, above the rdna_ar gate), and the odd rank follows the rank id, not the card (11.2).
- **E1 to E3 ran (11.1).** 24 replays of one 1024-token prompt give a different layer set every time (24 distinct sets, mean Jaccard 0.13) with rank 2 odd in 700 of 700 records, and only the pure 1023-row passes are affected: 68 of 68 requests that contain one, 0 of 75 that do not. No race and no asymmetric all-reduce is needed to explain it.
- **Does it matter, and what the fix cost.** For a static placement the disagreement did not matter: each rank applies its own ids to its own experts, so one entry in about 49,000 token-layers of an affected pass dropped or added one near-tied expert's contribution; none in decode (0 of 418,592 decode records) `[INFERENCE: far inside W4A16 and fp16-atomic noise]`. A design that needs consensus (Metro `min_touch`, per-step replica routing) cannot assume it, and now does not have to: making the rank files agree per key (11.3) removed it (0 of 79,872 records over 54 pure 1023-row passes, 11.4) at a cost inside the noise of an ABBA at c=1 (ITL -0.06 % and -0.21 %, TTFT +0.10 %, 11.5). The broadcast of rank 0's ids that this bullet proposed (+0.3 to +0.6 % TTFT, +0.6 to +0.9 % of ITL at c=1 at decode `[INFERENCE]`) is not needed. No production code changed.

### 10.3 `T` and `P` against the sampling floor (G0.4)

Per layer and band from rank 0's ring (ratio of sums, the table of `summary.py`), median over 48 layers with the range in brackets. *floor* = the same ratio after removing each rank's mean load (the noise a perfectly balanced placement would still have, with its row and MTP correlation); *persistent* = max over mean of the per-rank mean loads (what a placement can move).

| band | arm | calls per layer | `T` observed | `T` floor | `T` persistent | `P` observed | `P` floor |
|---|---|---|---|---|---|---|---|
| 3 rows (c=1, MTP2) | B1 | 2,339 | 1.432 [1.383, 1.580] | 1.409 | 1.135 | 1.480 | 1.452 |
| 3 rows | L1 | 281 | 1.435 [1.380, 1.760] | 1.416 | 1.146 | 1.482 | 1.456 |
| 24 rows (c=8) | B1 | 2,222 | 1.170 [1.143, 1.245] | 1.144 | 1.093 | 1.224 | 1.160 |
| 24 rows | L1 | 3,319 | 1.164 [1.137, 1.216] | 1.154 | 1.054 | 1.228 | 1.201 |
| prefill | B1 | 521 | | | | 1.403 [1.122, 2.044] | 1.178 (persistent 1.352) |
| prefill | L1 | 1,681 | | | | 1.168 [1.066, 1.351] | 1.127 (persistent 1.108) |

- **The simulator was right.** Section 3.1 predicted 1.432 (3 rows) and 1.160 (24 rows) linear and 1.421 and 1.144 for the best permutation; measured 1.435 and 1.164 (L1), 1.432 and 1.170 (B1), floors 1.416 and 1.154 (L1), 1.409 and 1.144 (B1). The T10 persistent touch imbalance at 24 rows (1.066) matches L1's 1.054; the prefill pair imbalance (1.106) matches L1's 1.108.
- **Decode is the floor.** On L1 the excess over 1 is 94 % floor at 24 rows (0.154 of 0.164, all layers) and 92 % at 3 rows (0.415 of 0.450). Observed minus floor is 0.010 at 24 rows and 0.019 at 3 rows for the median layer (0.035 over all layers at 3 rows: a few skewed layers, persistent up to 1.72, dominate), inside the 0.02 of the stop rule except at 3 rows over all layers. B1's random tokens give 0.026 and 0.023 (median layers).
- **Prefill has real skew on random tokens only.** B1's prefill `P` median 1.403 (range to 2.04, persistent 1.352) equals the 1.39 straggler of the 11-02 profile `[INFERENCE: that profile used random-token prompts]`; L1 is 1.168 with 1.108 persistent. Of L1's 0.168 excess about 0.016 is multinomial noise `[INFERENCE: 3,200 pairs per rank]`, about 0.11 chunk-to-chunk topic transient that no static placement removes, the rest persistent skew.
- `P` and `T` disagree as predicted (24 rows: 1.228 against 1.164); which one the decode kernel time follows is G0.3, open until P8 and P1.

### 10.4 Placement gain and drift

Decode, relative drop of the slowest rank's touched experts under the first-order counterfactual on the realised ring loads (`policy.shifted_gain`; weights are the measured per-band touch frequencies, packed with `pack_owner`), load-weighted mean over 48 layers, then the median layer and the layers above 1 %. Two captures of different traffic stand in for a held-out set, because no second agentic capture exists.

| rows | fit to evaluate | gain | median layer | layers above 1 % |
|---|---|---|---|---|
| 24 | L1 to L1 (in sample) | 0.84 % | 0.69 % | 15 of 48 |
| 24 | B1 to L1 (other workload) | 0.07 % | 0.05 % | 2 |
| 24 | L1 to B1 | 0.60 % | 0.59 % | 16 |
| 3 | L1 to L1 | 0.84 % | 0.21 % (range -3.0 to +11.5 %) | 14 |
| 3 | B1 to L1 | 0.54 % | 0.32 % | 15 |
| 3 | L1 to B1 | 0.39 % | 0.58 % | 14 |

- **Decode gain is under the 1 % rule.** At 24 rows 0.07 to 0.84 % of the slowest rank's touched experts, about the same of decode MoE time: 0.02 to 0.19 % of ITL at the 23 % MoE share of c=8 (section 3.1). At 3 rows the per-layer scatter (-3 to +11 %) is estimation noise on 281 calls. Packing by all-decode pairs with a row-independence model is worse (24 rows +0.60 % in sample, -0.91 % across workloads; 3 rows negative): MTP rows of one request are correlated, so measure touch frequencies per band.
- **An ideal adaptive scheme is not much better.** Re-centring each rank's mean load per window of W calls, minus the same on time-shuffled calls (the part that is noise absorption, not drift), leaves 0.0095 (W=1000), 0.0122 (500) and 0.015 (250) of ratio at 24 rows in L1, on top of the 0.008 persistent part: 1.5 to 2.0 % of decode MoE time, 0.35 to 0.46 % of ITL, free of move cost and counter cost.
- **Drift is real.** Per-rank mean-load deviations in the two halves of the ring window differ 8 times more than noise at 24 rows in L1 (median 1.41 touched experts against 0.17 expected, correlation 0.53; B1 0.71 against 0.23, 0.94) and 2.4 times at 3 rows (0.58 against 0.24, 0.66). Halves are different suites in L1 and different cells in B1, so this is workload-mix drift: a placement fit on one mix serves the next only in part, the reason held-out gains are below in-sample ones.
- **Prefill packs better than decode.** L1's histogram ring (last 512 prefill calls per layer, older 256 against newer 256, 48 layers), per-call ratio median: linear 1.165; fit on the older half 1.1005 (held out); fit on all 1.0996; fit on the newer half itself 1.108. The expert shares of the two halves correlate at 0.963. That is 5.5 % of the slowest rank's pairs, about 1.2 % of a prefill step at the 21.4 % MoE share of a 16k prefill (11-02 profile) `[INFERENCE: GEMM time proportional to pairs; G0.3]`, about 70 ms of the 5.9 s c=1 16k TTFT, under the TTFT noise of one ABBA. Packing can beat the mean-equalising ceiling of 10.3 because it separates co-activating experts and so lowers the per-call variance `[INFERENCE]`. The prefill-fit placement is neutral for decode (24 rows -0.01 % weighted, 3 rows +0.19 %; 6 to 11 layers move by more than 1 % either way).

### 10.5 Recommendation and what is still open

- **Decode (ITL): no-go for phase 1 and for phase 2.** The stop rule of section 7 is met: the held-out gain is below 1 % of MoE time at both widths (0.07 to 0.84 % at 24 rows, 0.4 to 0.8 % at 3 rows) and `T` max/mean is within 0.02 of the floor at 24 rows (0.010; at 3 rows 0.019 for the median layer, 0.035 over all layers). Online rebalancing would chase at most 1.5 to 2.0 % of decode MoE time and pay a P2P move that competes with the rdna_ar all-reduce.
- **Prefill (TTFT): optional and free.** `solve.py --objective pairs` on L1 gives a `VLLM_RDNA_MOE_PLACEMENT` file, no engine change, about 1 % of TTFT, decode-neutral. It cannot be gated on one ABBA (TTFT noise 1.4 %); build it only if G0.3 shows the prefill MoE kernel time is pair-bound, and fit it on agentic-prompt traffic such as L1 (better, a recorded session), never on bench cells: B1's random-token skew (1.35 persistent) would pack a placement that real traffic does not want.
- **Counters stay a measurement tool.** At c=1 a placement gain of at most 0.1 % of ITL cannot pay for a counter cost of +0.6 to +0.9 % (unresolved, 10.1). If they are ever kept on, use `EVERY` 16 or more and split the prefill op over several CTAs.
- **Still open.** P8 and P1: which of `T` and `P` the decode kernel time follows (G0.3, `objective.py`), the durations of `moe_route_zero_k<true>` against `<false>` and of `eplb_stats_k` (the counter cost), the MoE share of the prefill step on this tree. Arms A2 and B2 (gate noise), Z1 and Y1 (MTP0), L2 and G (a second capture, and recorded multi-turn agentic sessions: L1 is the `hi2` suite behind the Cline prompt). The rank 2 disagreement is closed (section 11).

## 11. The cross-rank top-k disagreement: cause, replay and fix (2026-11-16)

Tree `818fa99830`, recipe `flashnext-mtp2`, HIP 2-5 (b4 lease), lean counters on (`EVERY` 1, HIP kernel, 30 s tick). `tools/rdna/eplb/e_run.sh` boots through `serve_validate.sh` and replays prompts at c=1 (`e_replay.py`: greedy, `max_tokens` 4, `ignore_eos`, one `cache_salt` per request, so the prefix cache served 0 tokens in every group), waits for a counter snapshot newer than the last request and dumps the four rank files; `replay_check.py` cuts each ring into forward passes and the passes into requests. Run dir `~/w4a8_runs/eplb-e/<stage>` on par1-cs25.

### 11.1 E1 and E3: reproduced, rank 2, only the pure 1023-row passes

Boot `e1` (shipped rows), 143 requests: 700 of 48,480 compared records (1,010 passes) disagree.

| prompt (tokens) | requests | prefill passes per request (valid rows) | requests with a disagreeing layer | disagreeing records (range per request) | odd rank |
|---|---|---|---|---|---|
| 777 | 3 | 776 | 0 | 0 | |
| 1023 | 24 | 1022 | 0 | 0 | |
| **1024** | 24 | **1023** | **24** | 231 (5 to 15) | 2 (231 of 231) |
| 1025 | 24 | 1024 | 0 | 0 | |
| 1030 | 12 | 1024 | 0 | 0 | |
| 2047 | 12 | 1024, 1022 | 0 | 0 | |
| **2048** | 24 | 1024, **1023** | **24** | 265 (5 to 17) | 2 (265 of 265) |
| **16,384** (E3: aligned last chunk) | 8 | 2048 x 7, 1024, **1023** | **8** | 99 (8 to 20) | 2 (99 of 99) |
| **text 1024** (`docs/rdna2/tuning.md`) | 12 | **1023** | **12** | 80 (5 to 9) | 2 (80 of 80) |

- **Reproduced: yes.** All 700 records are a 3-against-1 hash split with rank 2 odd; 118 also differ in one routed pair between two owners, 582 are hash only. 699 sit in the 70 pure 1023-row passes of the window (68 replayed requests and two probes), one in a 130-row warm-up pass.
- **The shape decides.** Every request that has a pass of exactly 1023 rows is affected (68 of 68, 5 to 20 of the 48 layers each, random and text prompts alike) and no request without one is (0 of 75): the 776-, 1022-, 1024- and 2048-row passes are clean, the seven 2048-row chunks of the 16k prompt too. A prompt of k x 1024 tokens ends in a 1023-row chunk (the last token runs alone, `[INFERENCE]` from the signatures).
- **Not the same layers twice, until the epilogue is deterministic.** With the production MoE epilogue 24 replays of one prompt give 24 different layer sets (mean pairwise Jaccard 0.13; 0.14, 0.17 and 0.07 for the 2048, 16k and text groups); no (pass, layer) cell disagrees in every replay and the most frequent one in 12 of 24. Boot `accbase` (shipped rows, `VLLM_RDNA2_MOE_DETERMINISTIC=1`, otherwise the same) gives the same 8 disagreeing (pass, layer) cells in each of 24 replays of the 1024-token prompt (1 distinct set, Jaccard 1.0) and one greedy output, against 6 distinct outputs in 24 production replays of the 1023-token prompt: the disagreement is a deterministic function of the input and the rank's rows. The layer set only moves in production because the hidden state changes from run to run (the fp16 atomics of the MoE epilogue, `[INFERENCE]`: the flag removes exactly this variation), and a near-tie flips only where the hidden state happens to land on it.

### 11.2 Cause: the router GEMM through per-rank TunableOp rows

A worker on device ordinal `r` reads `tunableop_results<r>.csv` (`tunableop_env.sh`; PyTorch picks the file by the device that is current at the first GEMM of the process). 531 of the 3,330 keys of `rocm7.14-rocblas5.5` name different solutions on different ranks (ranks 4 to 7 are copies of 0 to 3, the same 531). The router GEMM is replicated, so a split key is a different accumulation order on one rank. Router keys `tn_512_<M>_2560_ld_2560_2560_512` at the replayed shapes:

| M | ranks 0, 1, 3 | rank 2 | recorded ms (ranks 0 to 3) |
|---|---|---|---|
| **1023** | `...046` | `...044` | 0.1397, 0.1359, 0.1442, 0.1415 |
| 130 | `...128` | `...139` | 0.0497, 0.0497, 0.0493, 0.0485 |
| 776, 1022 | no row (rocBLAS heuristic) | same | |
| 1024, 1025, 1030, 2047, 2048 | `...046`, `...046`, `...098`, `...047`, `...063` | same | one row for all four ranks |

Decode is not exposed: the router keys at 3, 6, 12 and 24 rows name one solution on every rank, and 9 to 32-row router GEMMs run the custom split-K kernel instead of rocBLAS (0 of 418,592 decode records disagree, 10.2). The table therefore predicts every row of 11.1 and the one record at 130 rows. Direct proof, `tools/rdna/eplb/tunable_bits.py` (seeded fp16 `[rows, 2560] x [512, 2560]`, 16 repetitions, one process per ordinal with `PYTORCH_TUNABLEOP_FILENAME` set as the launcher does, no collective, no second stream; it prints the file TunableOp opened and the solver it holds), run 09:51 to 09:54 on HIP 2-5:

| rows | ordinal 2 against 0, 1, 3 | ordinals 0, 1, 3 among themselves |
|---|---|---|
| **1023** | **1,678 of 523,776 logits differ (0.32 %), max 1.95e-3 (one fp16 ulp), 0 of 16,368 top-10 sets flip** | identical |
| 3, 24, 512, 1022, 1024, 1025, 1036, 1045, 1197, 2047, 2048 | identical | identical |

A split key does not always change the bits: ordinal 3 at 512 rows and ordinal 2 at 1197 rows hold another solution (`...128` against `...082`) with the same result. A flip needs a near-tie between the 10th and 11th logit, about one entry in 49,000 token-layers in the model (10.2), so 16,368 random rows show none.

The shipped files split 28 of the 126 router keys across the ranks, not only the one at 1023 rows: M = 2, 25, 27, 28, 29, 34, 63 to 71, 128, 130, 131, 134, 256, 257, 259, 261, 262, 320, 384, 512, 1023 and 1197. Boot `accbase` met a second one: the 261-token prompt of the perplexity probe is a 261-row prefill pass whose key names `...140` on ranks 0 and 1, `...129` on rank 2 and `...132` on rank 3, and one of its records disagrees with rank 3 odd. Any prompt chunk that lands on a split key can do this; 1023 is the one that block-aligned prompts hit every time.

- **E2: the odd rank follows the rank id, not the card.** With `HIP_VISIBLE_DEVICES=5,4,3,2` the same ordinal gives identical bits on the other physical card at every shape (micro-test, all 12 row counts, 4 ordinals). End to end, boot `e2ctl` (shipped rows, rank 2 on HIP 3 instead of HIP 4, `VLLM_RDNA_AR=0` with no `one-shot all-reduce active` line in the serve log (8 in `e1`), `VLLM_RDNA_AR_HIER=0`, `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`): 265 of 20,112 compared records disagree, all in 26 pure 1023-row prefill passes, rank 2 odd in 265 of 265; 24 of 24 replays of the 1024-token prompt are affected (246 records, 5 to 15 per replay, 24 distinct layer sets, mean Jaccard 0.13), 0 of 12 replays of the 1023-token prompt (a 1022-row pass). The file is chosen by the worker's ordinal, so rank 2 stays the odd rank on whichever card it runs, and neither the all-reduce path, the aux stream nor the card is needed to produce the disagreement.
- **The earlier hypotheses (10.2) are out.** The shared-expert aux stream forks only up to 256 tokens (`VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`), so not in a 1023-row pass. The 1023 x 2560 x 2 B = 5.2 MB all-reduce of that pass is above the 256 KiB rdna_ar gate and goes over RCCL, and the hierarchical all-reduce (`VLLM_RDNA_AR_HIER=auto`) applies only to a world of 8 across two islands, so TP4 stays flat (and all six cards sit behind one root port). A race or an asymmetric collective is not needed: the micro-test shows the different bits in one process per ordinal.

### 11.3 The fix: one solution per key in every rank file

`tools/rdna2_028/tunableop_agree.py write <profile>` keeps, per key, the solution with the lowest recorded time (ties go to the lowest rank) and replaces only the lines that name another solution by the winning line verbatim; it adds and drops no key, so the keys left out on purpose stay out (`tn_2048_12_2560_ld_2560_2560_2048`, see the profile README). `check` lists the split keys and exits 1; the CPU test `tests/tools/rdna/test_tunableop_rank_agree.py` runs it over every shipped profile. The condensed evidence for the data change, without the replay tooling, is `docs/rdna2/tunableop-rank-agreement-2026-11-16.md`; the READMEs, the tool and the test point to it.

| profile | rank files | split keys before | lines rewritten |
|---|---|---|---|
| `rocm7.14-rocblas5.5` | 8 | 531 of 3,330 | 2,124 (316, 282, 290, 174 on ranks 0 to 3, the same on 4 to 7) |
| `rocm7.14-rocblas5.5-hi2-27b-tp2` (canonical rows plus 36) | 4 | 531 of 3,366 | 1,062 |
| `rocm10-rocblas5.6` | 4 | 0 of 139 | 0 |

The 531 keys are 47 `(op, N, K)` families: 146 prefill (M >= 512), 214 mid (33 to 511) and 171 decode (M <= 32) keys, so the change is not confined to the router. The same rows serve TP8 and TP4 x PP2 (`TUNABLEOP_RANKS=TP*PP` reads ranks 0 to 7: before, 2 of 8 ranks were odd at M=1023): every TP group now sees one solution per key. The solutions are valid on every card because all six V620 are gfx1030 under one rocBLAS build hash dir.

### 11.4 Acceptance of the new rows

Gate: the deterministic concurrency probe (`VLLM_RDNA2_MOE_DETERMINISTIC=1`) on the flat and on the hierarchical all-reduce, the greedy and perplexity probes, 0 KFD evicted milliseconds, no fault marker, and the E1 replay on the agreed rows with 0 disagreeing records. All boots below use `TUNABLEOP_ROOT=$RUN/tunable-agree` (the files committed with this change; the log line `TunableOp lookup enabled ... rows:` names the directory and `rows per rank 0=3330 ... 3=3330`).

| boot | flags | greedy probe | PPL | concurrency probe (failing / near-tie / soft / identical of 48) | KFD evicted ms, faults | cross-rank records, 1023-row passes |
|---|---|---|---|---|---|---|
| `accbase` (baseline, **shipped** rows) | `VLLM_RDNA2_MOE_DETERMINISTIC=1`, flat | PASS 3/3 | PASS 4.82 | PASS 0 / 10 / 2 / 36 | +0 on 4 workers, 0 | **208 of 60,384**: 207 in 26 passes of 1023 rows (8 in each of 24 replays), 1 at 261 rows |
| `fix` | production (fp16-atomic MoE epilogue), flat | PASS 3/3 | PASS 4.82 | PASS 0 / 17 / 1 / 30 | +0 on 4 workers, 0 | 0 of 79,872; 54 passes, 52 of them replayed |
| `accflat` | `VLLM_RDNA2_MOE_DETERMINISTIC=1`, flat | PASS 3/3 | PASS 4.86 | PASS 0 / 10 / 2 / 36, repeats 0 / 11 / 3 / 34 and 0 / 12 / 1 / 35 | +0 on 4 workers, 0 | 0 of 130,656; 14 passes |
| `acchier` | deterministic, hierarchical all-reduce forced 2 + 2 (`VLLM_RDNA_AR_HIER=1`, `VLLM_RDNA_AR_HIER_ISLANDS=0,1\|2,3`) | PASS 3/3 | PASS 4.86 | PASS 0 / 14 / 0 / 34, repeats 0 / 15 / 0 / 33 and 0 / 13 / 1 / 34 | +0 on 4 workers, 0 | 0 of 133,248; 14 passes |

The replay of the shipped rows had 699 disagreeing records in 70 passes of 1023 rows with the production epilogue (`e1`) and 207 in 26 such passes with the deterministic one (`accbase`); the agreed rows have 0 in 54 (`fix`), 14 (`accflat`) and 14 (`acchier`), and every other shape stays clean. With the deterministic epilogue a replayed prompt gives one greedy output in 24 of 24 replays (`accflat`, 1023 tokens), against 6 distinct outputs with the production epilogue: the run-to-run variation of 11.1 is not the rows. The rows move the numerics little: the concurrency probe has the same profile with both row sets (0 / 10 / 2 / 36 on the first run of `accbase` and `accflat`) and the PPL goes from 4.82 to 4.86 in the deterministic mode (261 tokens; the placement test of 11-03 moved it 4.87 against 4.85).

The `acchier` serve log shows `hier islands [[0, 1], [2, 3]], algo auto, hier from 20 KB, owners 2` on every rank, so the decode all-reduces from 20 KB up to the 1 MB gate took the two-level path (the concurrency probe runs eight prompts, up to 24 rows per MTP2 step, 123 KB per all-reduce); the 5.2 MB all-reduce of a 1023-row prefill pass is above the gate and stays on RCCL in both modes. The `auto` mode itself (TP8, two islands of four ranks) cannot run on this six-card box, so the forced 2 + 2 at TP4 is the stand-in. In the deterministic mode the probe fails a flip only if the solo run was sure (margin above the 0.2-nat tie band); the near-tie and soft lines are the expected warnings of a batch-shape change.

### 11.5 What the new rows cost

- **The router key of the fix is a prefill key.** The 70 passes of 1023 rows in `e1` are all prefill passes (rows above 192); the 757 passes below 193 rows have at most 189 rows, and the decode steps among them 32 or fewer (3 rows at c=1 in 472 passes, 24 at eight sequences; the larger counts are probe and warm-up prompts). Measured on the cards (`verify`, one synchronised call, event to event) the router GEMM at 1023 rows takes 148.7, 148.3, 162.0 and 151.5 us on ranks 0 to 3 with the shipped rows and 149.6, 149.6, 154.0 and 151.7 us with the agreed ones: only rank 2 changes (`...044` to `...046`), 8 us faster per call, 0.4 ms over 48 layers once per prompt, 0.1 % of a 1k TTFT (398 ms) and 0.007 % of a 16k one (5.92 s). It touches no decode step.
- **Not every rewritten key is a router key.** 171 of the 531 keys are decode-class (M <= 32), 214 mid and 146 prefill, in 47 `(op, N, K)` families of several models. By the recorded tuner times of the rewritten rows each rank gets faster (sum over the rewritten rows, each key once: 103.0 to 101.3 ms on rank 0, 94.9 to 93.3, 149.5 to 143.7, 73.6 to 71.5; decode-class 20.1 to 19.8, 19.0 to 18.6, 20.0 to 19.7, 8.2 to 8.1 ms), but those times come from four tunings run in parallel and they do not survive a re-measurement.
- **Re-measured on the cards** (`verify`: the shipped and the agreed file of each rank on its own card; 3,330 of 3,330 keys hit in all 16 runs; 5 warm-up and 15 timed calls per key): summed over the rewritten keys the agreed rows are slower than each rank's own tuned row by +0.6, +0.7, +0.1 and -0.1 % for the decode-class keys, +1.2, +1.6, +1.5 and +1.5 % for the mid ones and +2.5, +2.2, +1.8 and +1.6 % for the prefill ones (ranks 0 to 3). That is the price of one solution for four cards (a rank's own row was tuned on its own card) and of picking the lowest of four noisy tuner times `[INFERENCE]`. For decode keys it is about 1 us per call: the fp16 rows at 1 to 3 rows that were rewritten (3, 13, 11 and 5 on ranks 0 to 3), each called once, add up to +0.7, +42.0, +9.9 and +1.7 us, and 38 of rank 1's 42 us are two lm_heads of other models (`tn_32320_2_4096` of a 129k-vocab profile, +21.8 us, and `tn_62080_{2,3}_5120` of the 27B, +16.3 us). Flash-Next's own lm_head keys (`tn_62080_{1,2,3}_2560`) were not split, so they did not change.
- **Every candidate on every card** (`vmatrix`: each of the four shipped rank files on each of the four cards, 32 runs, one process per card, the four in parallel, median of 40 timed calls per key after 10 warm-up calls; all 3,330 rows hit in all 32 runs, so every solution any rank file names for a key runs on all four cards). Per split key and candidate the mean over the cards of that median ranks the candidates by measurement. Summed over the 531 split keys (each key once, per call): decode-class (171 keys) own-tuned 35.30 ms, agreed 35.34 ms (+0.14 %), best measured 35.25 ms (-0.13 %); mid (214 keys) 53.08, 53.69 (+1.16 %) and 53.24 ms (+0.31 %); prefill (146 keys) 118.13, 119.91 (+1.51 %) and 118.32 ms (+0.16 %). The best measured candidate differs from the agreed row on 96, 121 and 88 keys, and 6, 23 and 37 of them would gain more than 2 % and 2 us; the largest single gains are 157 us (13.6 %) on `nn_1280_800_10240` and 125 us (13.2 %) on `nn_1280_667_10240`. The committed rows keep the recorded-time rule: choosing by this matrix is the one change that would recover most of the +1.2 and +1.5 %, but it changes numerics again and needs the acceptance of 11.4 and the ABBA below repeated (about 50 minutes of box time) for an end-to-end gain that the ABBA bounds at 0.1 % of TTFT, so it was not made. The matrix is the input for it.
- **ABBA at c=1** (`benchA`, `benchB`, `benchB2`, `benchA2`: counters off, flat rdna_ar, greedy, 24 requests of 1024 + 128 tokens and 6 of 16,384 + 128, two boots per side):

| boot | rows | acceptance length 1k / 16k | 1k TTFT (ms) | 1k ITL median / p99 (ms) | 16k TTFT (ms) | 16k ITL median / p99 (ms) |
|---|---|---|---|---|---|---|
| A1 `benchA` | shipped | 2.041 / 2.045 | 398.0 | 22.55 / 23.43 | 5,922.7 | 23.00 / 23.67 |
| B1 `benchB` | agreed | 2.205 / 2.286 | 398.7 | 22.58 / 23.46 | 5,926.5 | 22.95 / 23.76 |
| B2 `benchB2` | agreed | 2.122 / 2.201 | 399.2 | 22.55 / 23.44 | 5,930.4 | 22.94 / 23.78 |
| A2 `benchA2` | shipped | 2.170 / 2.293 | 399.1 | 22.61 / 23.55 | 5,922.2 | 23.00 / 24.02 |
| **agreed against shipped** (means of two boots) | | | **+0.10 % (+0.4 ms)** | **-0.06 % (-0.01 ms)** / -0.17 % | **+0.10 % (+6.0 ms)** | **-0.21 % (-0.05 ms)** / -0.3 % |
| same-arm spread shipped, agreed | | | 1.0, 0.6 ms | 0.06, 0.03 ms | 0.5, 3.9 ms | 0.00, 0.01 ms |

The step time (the median ITL is the gap between MTP2 token bursts) and the TTFT are the same within 0.1 to 0.2 % on both sides: ITL -0.06 % at 1k and -0.21 % at 16k, TTFT +0.10 % at both (+0.4 ms of 398 ms, +6.0 ms of 5.92 s). The 1k differences sit inside the same-arm spreads; at 16k the 6 ms of TTFT is 1.5 times the larger spread, which two boots per arm cannot call either way, and the 0.05 ms of ITL is 0.2 %. The agreed side is not slower in ITL, as the 1 us per call of the decode-class keys predicts; if the +0.10 % of TTFT is real it is the re-measured +1.6 to +2.5 % of the rewritten prefill keys, which are a small part of the keys of a prefill step `[INFERENCE]`. The same-arm spread of ITL here (0.3 %) is smaller than the 3 to 4 % quoted in 10.1. The acceptance length moves by 0.13 and 0.08 (1k) and 0.25 and 0.09 (16k) between boots of one arm, so TPOT (10.18 to 10.50 ms at 1k) and tok/s move by up to 5 % between boots with no step-time cause: read the step time, not the throughput. HIP 2 to 5 averaged 59.4, 58.5, 59.2 and 59.8 C during the four cells (peaks 71 to 76 C); every boot has 0 KFD evicted ms, 0 fault markers and 0 amdgpu dmesg faults.

### 11.6 What is not covered

- **TP8 and TP4 x PP2 on hardware.** The six-card box cannot run TP8. The files of ranks 4 to 7 are checked (`check` over all eight, the CPU test), every TP group of the launcher reads rank files that agree per key, and the hierarchical all-reduce ran only as a forced 2 + 2 at TP4.
- **HIP 0 and 1.** The agreed rows hit 3,330 of 3,330 on HIP 2 to 5 and were not run on HIP 0 and 1, which are the same gfx1030 under the same rocBLAS build `[INFERENCE]`.
- **The Flash-Next hot-path inventory.** The rewritten decode-class keys of Flash-Next are not enumerated; the ABBA bounds the end-to-end effect and the per-key table the per-call one.
- **A launcher guard.** A boot-time warning when two rank files of the selected profile disagree (a short awk over 3,330 lines per file) would catch a profile tuned per rank on a deployment that does not run the CPU test. Not implemented: `tools/rdna2_028/tunableop_env.sh` is production code.
- **Merge.** The data commit applies to `origin/rdna_extras` (`2c6d5d450d`) with one conflict in `tools/pre_commit/test_tethering_allowlist.txt` (the neighbouring lines are fork-only tests of this study; keep the single `test_tunableop_rank_agree.py` line) and needs a boot of the merged head.
