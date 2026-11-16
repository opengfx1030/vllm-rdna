# All-to-all, DBO and EP on gfx1030: feasibility, go/no-go and plan

| Item | Value |
|---|---|
| Date | 2026-11-16 |
| Tree | worktree `vllm-eplb`, branch `feat/eplb-lean`, base `rdna_extras` at `262ff0edd3` (v0.33 line) |
| Question | Can DeepSeek-style expert parallelism (EPLB, async all-to-all, dual-batch overlap) pay on six V620 behind one PEX880xx switch, and what is the cheapest order to find out |
| Method | code reading in the worktree with `file:line` cites, repo measurements with provenance, a byte and bandwidth-floor model. No kernel was written |
| GPU work | none. Leases `HI2.a2` (HIP 0,1) and `LeapQuant.b4` (HIP 2-5) were held at 2026-11-16 00:45 UTC and a `docker buildx build` was running, so the quiet rules forbid a measured window. The benchmark in section 10 is shipped, not run |
| Conventions | `[INFERENCE]` marks anything not measured and not read in code. "Old topology" marks numbers from the two-root, eight-GPU box of 2026-11-06, which must be re-measured on the single-switch box |
| Companion | `tools/rdna/a2a/p2p_bench.py` (CPU selftest passes; the torch backend ran only against a CPU stand-in) |

## Answers at a glance

| # | Question | Answer |
|---|---|---|
| 1 | Is there token dispatch/combine traffic in the production decode config (TP4/EP4, dp=1)? | No. The only MoE communication is one TP all-reduce of the partial sums per layer. `use_all2all` is False and no all-to-all manager is built (section 1) |
| 2 | Does `--enable-dbo` do anything there? | No. The V2 runner returns before building a microbatch runner when `data_parallel_size <= 1`, so it is a silent no-op (section 1.3) |
| 3 | Can DBO over an a2a backend pay for Flash-Next MTP2 decode on this box? | Not at c=1..8. The communication it could hide is 5-14 % of the step (the all-reduce share), the split re-streams weights and doubles a latency-bound chain, and model and runner reject it (hybrid GDN state, speculative decoding). **NO-GO** (section 7) |
| 4 | Which layout would make a2a and DBO meaningful? | DP attention plus EP (dp >= 2) at 24-48 or more rows per rank. By bandwidth floors that is a high-concurrency throughput mode; it cannot meet the c=1..8 ITL gate (section 4) |
| 5 | Can an async a2a be built on the rdna_ar machinery? | Yes: a single-stream LL protocol (posted pushes, count as flag, device-derived sequence, the recv hook is a wait kernel). It needs no second queue and no cross-queue fork/join. Design and a 2-GPU MVP in section 6 |
| 6 | Where to start? | Measure first: run the P2P benchmark on the single-switch box, then EPLB-lean as a pure permutation. The a2a program is gated on a DP-layout probe with the existing AgRs backend (section 9) |

## 1. What dispatch and combine traffic exists in the production decode config

Production is `flashnext-mtp2`: TP=4, MTP=2, `SEQS=8`, `FULL_AND_PIECEWISE`, `expert-parallel`, V2 runner, `GPU_MAX_HW_QUEUES=4`,
resident experts and no `--monokernel` (an opt-in, `flashnext-mtp2.env:77`; recipe lines `:13,16,20,26,29,32,78`). The answer is none, and the chain of evidence is:

### 1.1 Evidence chain at TP4/EP4, dp=1

| Step | Where | What it says |
|---|---|---|
| SP-MoE flag | `vllm/config/parallel.py:774-791` | `use_sequence_parallel_moe` needs an allowed backend, EP, `tensor_parallel_size > 1` and `data_parallel_size > 1`. False at dp=1 |
| a2a flag | `vllm/config/parallel.py:794-799` | `use_all2all` is `data_parallel_size > 1`, or SP-MoE, or EP with `prefill_context_parallel_size > 1`. False |
| Communicator | `vllm/distributed/parallel_state.py:2339-2356`, `device_communicators/base_device_communicator.py:245`, `cuda_communicator.py:229-235` | the flag reaches the EP communicator, and `AgRsAll2AllManager` is built only `if self.use_all2all`. At dp=1 `all2all_manager` stays None |
| MoE parallel config | `fused_moe/config.py:1035-1037` | `use_all2all_kernels = use_ep and (dp_size > 1 or pcp_size > 1 or is_sequence_parallel)`. False |
| Prepare/finalize | `fused_moe/all2all_utils.py:190-231` | not `use_all2all_kernels` and `dp_size == 1` gives `make_moe_prepare_and_finalize_no_dp_ep` (Standard format, no collective) |
| Sharding | `fused_moe/config.py:1226-1233` | under EP, `ep_size = tp_size` and `tp_size = 1`: experts are sharded over the TP group and every rank owns whole experts |
| What remains | `fused_moe/runner/moe_runner.py:486-520` (the call is at :518), `cuda_communicator.py:395-397` | one `tensor_model_parallel_all_reduce` of the local-expert partial sum, served by the rdna_ar one-shot |
| Model | `vllm/models/qwen4_exp/amd/model.py:209-212` | `use_sequence_parallel = parallel_config.use_sequence_parallel_moe`, False here. The model does carry SP code (`sp_shard`, `sp_all_gather`, `sp_reduce_scatter`, `model.py:75,83,87`; `enable_hc_sp`, `parallel.py:185`) that no RDNA recipe sets and AGENTS.md never mentions (no `data-parallel` hit in either) |

So a decode step has two all-reduces per layer (attention out, MoE out), about 96-98 per step
(`feat/hetero-moe:docs/rdna2/hetero-moe-design-2026-11-07.md` section 1.3: 98 calls, 10.6 us per call in the monokernel path),
each of 15-240 KiB at 3-48 rows. This is the only communication an overlap scheme could hide.

### 1.2 Escape hatches and what they cost

The MoE all-to-all becomes live only through `data_parallel_size > 1` (DP attention, then `use_sequence_parallel_moe` also
turns on when TP > 1) or `prefill_context_parallel_size > 1`. Neither is a decode-latency feature (section 4). The
SP-MoE path at dp=1 would need a code change to `parallel.py:774-791`; it moves the same bytes as two ring all-reduces
(section 3.3) while turning two collectives into four.

### 1.3 DBO is inert at dp=1

| Where | Behaviour |
|---|---|
| `vllm/v1/worker/gpu/ubatch_utils.py:559-560` | V2 runner: `if not use_ubatching or data_parallel_size <= 1: return None`. No microbatch runner, no message |
| `vllm/v1/worker/gpu/dp_utils.py:67` | the DP handshake asserts `dp_size > 1` |
| `vllm/v1/worker/gpu_ubatch_wrapper.py:395` | V1 wrapper asserts `dp_metadata is not None` |
| `vllm/config/vllm.py:756-758` | `use_v2_model_runner` returns the env value first. The recipe sets `VLLM_USE_V2_MODEL_RUNNER=1`, so the DBO blocker list (`_get_dbo_unsupported_features`, `vllm.py:3259-3299`) is never evaluated: it is reached only through `_get_v2_model_runner_unsupported_features` (`:3121-3122`), which `use_v2_model_runner` calls at `:766` when the env var is unset. With the explicit env the blockers are advisory and the failure is a runtime assert (section 5.2) |

## 2. Verification of the second-hand claims

| Claim | Verdict | Evidence |
|---|---|---|
| `use_all2all` is False for TP4/EP4 without DP, and SP-MoE needs DP > 1 | Confirmed. The line cites given in the brief (about 690-695) are off; use the ones in section 1.1 | `parallel.py:774-799` |
| `enable_dbo` recv hooks are registered only by `deepep_ll.py` and `nixl_ep.py` | Confirmed, with nuance. `dbo_register_recv_hook` is called only in the generic wrapper (`fused_moe/modular_kernel.py:1302,1482`) and only when `prepare_async`/`finalize_async` return a hook. Only `deepep_ll.py` (dispatch `return_recv_hook=True` at `:309`, combine hook returned at `:406`) and `nixl_ep.py` (`:261`, `:358`) return one. DeepEP-HT and V2 return a bare receiver (`supports_async` True at `deepep_ht.py:288`, `deepep_v2.py:372`) and switch to a comm stream instead. AgRs has `supports_async` False but `supports_dbo()` True on ROCm (`prepare_finalize/naive_dp_ep.py:145-150`): under DBO its collective moves to the comm stream, but `_comm_region` (`:153-171`) keeps every collective on the capture stream during graph capture, so overlap is limited to eager steps (its docstring says prefill). Net: with AgRs, `--enable-dbo` is not inert in eager steps at dp > 1, and it is inert in FULL-graph decode | the files cited |
| "push all-gather versus ring 0.314 ms versus 0.062 ms at 10.5 MB, from AGENTS.md" | Not supported. The string is not in AGENTS.md. The source is `docs/rdna2/tp4-allreduce-ceiling-2026-10-06.md` (referenced at `docs/rdna2/rdna-ars-allreduce-plan.md:8-14`), which is absent from this tree and is withdrawn by `docs/rdna2/tp4-allreduce-sizes-2026-10-10.md:42` ("RCCL 0.06 ms for 10-21 MB ... not physically possible and was an unsynced timing"). The re-measure gives RCCL TP4 10 MB = 616 us (`:35`). A ring all-reduce of 10.5 MB moves 2(W-1)/W x N = 15.75 MB per rank, at least 0.62 ms at 25 GB/s, so 0.062 ms would be above 250 GB/s. Do not use these numbers | grep of AGENTS.md; the three docs |
| Shared-expert overlap is dead because of `is_cuda()` | Stale in this fork. The gate is `current_platform.is_cuda_alike()` (`fused_moe/runner/shared_experts.py:144`). What kills overlap in decode graphs is the gfx10x capture guard (`:61-72`, `:148-150`): a captured aux-stream fork/join replays as a cross-HSA-queue barrier pair and was seen to hang about once per 1-4 h of serving. Eager steps keep it (token threshold 256, `envs.py:345,2274`) | `shared_experts.py` |
| Do not enable EPLB redundant experts | Agree for decode: replicas raise the distinct experts touched (hetero doc: streamed bytes 1.235-1.365x) and the objective is the maximum bytes streamed per rank. vLLM's `enable_eplb` also demands physical experts divisible by the EP size (`fused_moe/layer.py:271-279`), so EP6 would need 4 redundant replicas (516). The explicit placement path avoids both: `explicit_local_experts` (`fused_moe/expert_map_manager.py:72-82`) and the uneven linear split (`:84-90`; 512 experts over 6 ranks is 86, 86, 85, 85, 85, 85) | section 9, phase 1 |

Two further corrections that matter to the design. First, the "token dedup ratio" for E=512, top-10 is not the 1.10x of the
E=128, top-8 notes: with R=4 the expected distinct ranks per token is 3.78 of 4 (1.06x) and with R=6 it is 5.05 of 6, so
dispatch carries 94.5 % (R=4) and 84.1 % (R=6) of an all-gather's bytes (section 3.2). Second, a rank that
issues a collective from a second stream creates an independent graph branch whose replay order is not guaranteed across
ranks (`shared_experts.py:107-113`, vllm issue 60845). For rdna_ar's device-derived sequence numbers that is a
sequence/parity mismatch, which is why the design in section 6 is single-stream.

## 3. Byte and time model for Flash-Next decode

### 3.1 Constants

| Quantity | Value | Source |
|---|---|---|
| Hidden size | 2560, fp16 row 5,120 B | model config; hetero doc section 2 |
| Experts / top-k | 512 routed, top-10, one shared expert, 48 layers (36 GDN, 12 QSA) plus one MTP layer | hetero doc sections 1.3, 2 |
| Rows per step | 3 per request with MTP2, so c requests give 3c rows | `flashnext-mtp2.env:16` |
| Routing record | 64 B per dispatched token: 10 int16 ids and 10 fp32 weights (60 B, hetero row format) plus a 4 B source index | [INFERENCE] chosen here |
| W4A16 expert | 2,534,400 B (2.417 MiB) | hetero doc section 2 |
| Achieved weight streaming | about 380 GB/s per GPU | hetero doc section 1.3 |
| Dense weights | 7.96 GiB per decode step at TP1; HC (hyper-connection) 25.2 MiB per layer is replicated on every TP rank; TP4 per-rank dense about 3.09 GB (8.1 ms at 380 GB/s), TP2 about 4.9 GB (12.9 ms), DP 8.55 GB (22.5 ms) | hetero doc section 2; own arithmetic |
| GPU link | x8: 15.754 GB/s raw, 14.015 GB/s practical, about 13.9 GB/s measured P2P (every route is limited by the x8 GPU link, not the x16 host uplink). x16: 23-25.6 GB/s practical | `docs/rdna2/pciescope-2026-11-07.md:22-26,64-76`; hetero doc section 11.0; `docs/rdna2/pcie-iommu-audit-2026-11-06.md:112-131` (old topology, x16) |

The x8 reading is the repo's own: the pciescope capture of 2026-11-10 shows GPU ports at x8 (the port's configured
maximum) behind x16 virtual fabric ports, and the lease README measures about 13.9 GB/s for same-leaf and cross-leaf
pairs. A sysfs reading of x16 on a GPU endpoint is [INFERENCE] the on-card bridge link, not the slot link; the benchmark's
`topo` stage prints the narrowest link on each GPU's path to settle it. All time bounds below use 14 GB/s as the primary
case and 25 GB/s as the alternative.

### 3.2 Fraction of an all-gather that a token all-to-all must still send

A token is sent once to each distinct remote rank that hosts one of its top-10 experts (with 64 B of routing record).
For uniform routing the probability that a rank hosts none of a token's 10 distinct experts is C(E - E/R, 10) / C(E, 10):

| EP ranks R | Experts | P(rank unused) | Distinct ranks per token | Remote copies per token | Share of all-gather bytes |
|---|---|---|---|---|---|
| 2 | 512 | 0.0009 | 2.00 | 1.00 | 99.9 % |
| 4 | 512 | 0.0547 | 3.78 | 2.84 of 3 | 94.5 % |
| 6 | 512 split 86, 86, 85, 85, 85, 85 (`fused_moe/expert_map_manager.py:84-90`; with vLLM EPLB 516 = 512 + 4 redundant, `fused_moe/layer.py:271-279`) | 0.1562-0.1599 (0.1587 at 516) | 5.05 | 4.21 of 5 | 84.1 % |

Routing in production is skewed (top 8, 32, 64, 128, 256 experts carry 9.9, 25.1, 38.8, 58.5, 83.1 % of the rows,
`feat/expert-stats:docs/rdna2/expert-stats-t10-2026-11-07.md` section 1) and rows of one MTP group share experts, so the
real ratio is at or below these uniform values [INFERENCE]. Token dedup is therefore a 6-16 % effect at EP <= 6 with
top-10: an a2a here is an all-gather with structure, not a sparse exchange.

### 3.3 Bytes per layer per rank, by system-wide requests

Egress bytes per rank per layer. TP4 pays two all-reduces per layer (one-shot pushes (R-1) x payload; a ring moves
2(R-1)/R x payload). DP layouts pay dispatch (5,180 B per remote copy) plus combine (5,120 B per remote copy), with
requests spread over the ranks; the busiest rank holds `3 x ceil(c / R)` rows. INT8 dispatch sends 2,560 B + 2 B scale +
64 B per copy and keeps an fp16 combine.

| c | Rows | TP4 AR payload KiB | TP4 two one-shot AR KiB | TP4 two ring AR KiB | DP4/EP4 a2a KiB | DP4 INT8 dispatch KiB | DP6/EP6 a2a KiB | DP6 INT8 dispatch KiB |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 15 | 90 | 45 | 85.6 | 64.4 | 127.0 | 95.5 |
| 2 | 6 | 30 | 180 | 90 | 85.6 | 64.4 | 127.0 | 95.5 |
| 4 | 12 | 60 | 360 | 180 | 85.6 | 64.4 | 127.0 | 95.5 |
| 8 | 24 | 120 | 720 | 360 | 171.2 | 128.7 | 254.0 | 190.9 |
| 16 | 48 | 240 | 1440 | 720 | 342.4 | 257.4 | 381.0 | 286.4 |
| 32 | 96 | 480 | 2880 | 1440 | 684.9 | 514.9 | 761.9 | 572.8 |
| 64 | 192 | 960 | 5760 | 2880 | 1369.8 | 1029.7 | 1396.9 | 1050.1 |

For equal total rows a DP layout moves about 2.1x fewer bytes than two ring all-reduces (2.10x at R=4, 2.13x at R=6) and
about 4.2x fewer than two one-shots, and it has no attention all-reduce at all. SP-MoE on TP4 (reduce-scatter, dispatch,
combine, all-gather) moves 43.9 / 87.8 / 175.6 / 351.2 / 702.3 KiB at 3 / 6 / 12 / 24 / 48 rows against 45 / 90 / 180 /
360 / 720 KiB for two ring all-reduces: 2.4 % fewer bytes and twice the collective count.

### 3.4 Link-bound time per decode step (48 layers, bandwidth term only)

Milliseconds per step at x8 (14 GB/s) / x16 (25 GB/s):

| c | TP4 two one-shot | TP4 two ring | DP4/EP4 a2a | DP6/EP6 a2a | DP6 INT8 dispatch (x8) |
|---|---|---|---|---|---|
| 1 | 0.32 / 0.18 | 0.16 / 0.09 | 0.30 / 0.17 | 0.45 / 0.25 | 0.34 |
| 4 | 1.26 / 0.71 | 0.63 / 0.35 | 0.30 / 0.17 | 0.45 / 0.25 | 0.34 |
| 8 | 2.53 / 1.42 | 1.26 / 0.71 | 0.60 / 0.34 | 0.89 / 0.50 | 0.67 |
| 16 | 5.06 / 2.83 | 2.53 / 1.42 | 1.20 / 0.67 | 1.34 / 0.75 | 1.01 |
| 32 | 10.11 / 5.66 | 5.06 / 2.83 | 2.40 / 1.35 | 2.68 / 1.50 | 2.01 |
| 64 | 20.22 / 11.32 | 10.11 / 5.66 | 4.81 / 2.69 | 4.90 / 2.75 | 3.69 |

Measured one-shot all-reduce per call, graph replay, hidden 2560 fp16 (old topology, x16;
`docs/rdna2/tp8-pp2-flashnext-2026-11-06.md:119-142`): TP8 gives 29 us at 5 KiB, 33 us at 15 KiB, 46 / 62 / 107 / 136 /
196 us at 30 / 60 / 120 / 160 / 240 KiB (RCCL 134-178 us up to 240 KiB). TP4 gives 43 us at 120 KiB and 67 us at 240 KiB
(RCCL 65-74 us). The 7 us one-way flag latency and the one-shot all-to-all step (4 / 8 / 16 / 32 / 64 KiB: 10.4 / 11.3 /
10.9 / 14.8 / 22.0 us for four GPUs inside a switch; 15.7 / 13.6 / 20.4 / 36.8 / 62.4 us for eight GPUs flat) are in
`pcie-iommu-audit-2026-11-06.md:222-241`.

### 3.5 What the numbers say

- At c <= 8 communication is latency-bound, not byte-bound. A 120 KiB TP4 all-reduce costs 43 us measured against a
  14.7 us (x16) or 26.3 us (x8) bandwidth term, so 28 us or more is fixed cost. The interim-box slowdown agrees with the
  x8 model: 96 calls x 11.6 us = 1.1 ms predicted against +1.3 ms measured at c8 (22.54 / 43.57 / 45.63 ms at
  c1 / c6 / c8 on HIP 2-5, hetero doc section 11.0).
- The all-reduce share of the step caps any overlap gain: about 14 % at c1 for the production recipe if a standalone one-shot
  costs 33 us (96 calls, 3.2 ms of 22.38 ms; 33 us is the TP8 figure at 15 KiB, taken as an upper bound for TP4
  [INFERENCE]), 4.6 % with the opt-in monokernel's 10.6 us calls (1.04 ms), and 9.4 % at c8 (96 x 43 us = 4.1 ms of
  43.97 ms; TP4 baselines of hetero doc section 1.2).
- An a2a replaces two collectives per layer with two exchanges per layer of similar fixed cost. It is a one-hop push like
  the one-shot, so there is no byte or latency win at c <= 8 [INFERENCE]. The win appears only when the all-reduce payload
  scales with the whole batch (c >= 32: 5-10 ms per step of ring link time at x8) while a2a scales with rows per rank.
- Which option can give a decode gain: none of the a2a options at c=1..8. At c >= 32 the DP layout saves 3-5 ms of
  communication per step but pays 14.4 ms more dense streaming (section 4).

### 3.6 Overlap window and its ceiling

The only compute an exchange can hide behind is the other microbatch (DBO) or the shared expert plus router work of
the same layer (single stream, no DBO). At 6 ranks (DP6/EP6) the per-layer weight-streaming floor and the a2a link time
(x8) are, at 380 GB/s achieved and 14 GB/s link:

| Rows per rank | Total rows | Dense floor us | Expert floor us | Layer floor us | a2a link us | Share of layer | With 30 us latency [INFERENCE] | Cost of a second microbatch re-streaming dense weights us |
|---|---|---|---|---|---|---|---|---|
| 4 | 24 | 399 | 142 | 540 | 12.4 | 2.3 % | 7.8 % | 399 |
| 8 | 48 | 399 | 224 | 623 | 24.8 | 4.0 % | 8.8 % | 399 |
| 24 | 144 | 399 | 386 | 785 | 74.3 | 9.5 % | 13.3 % | 399 |
| 96 | 576 | 399 | 574 | 972 | 297.2 | 30.6 % | 33.7 % | 399 |

(144 rows use 347 distinct experts per layer, the T10 correlated count extended with a 0.72 factor [INFERENCE]; 24 and 48
rows use the measured 127.5 and 201.6; 576 rows touch all 86 experts of a rank.) DBO can hide at most 8-34 % of the
layer floor at these sizes. Its second microbatch re-reads the dense weights (399 us) unless they stay resident in the
128 MB Infinity Cache between the two microbatches' identical layers [INFERENCE: not measured; the cache would have to
hold a 55-145 MiB layer]. At 96 rows per rank each half batch still touches nearly every expert of the rank (0.997 under
independent routing [INFERENCE]), so a split also re-reads about 574 us of experts: about 973 us of extra streaming
against 297 us of communication. By this floor model the split costs more than it hides in every row of the table. The
single-stream overlap that needs no DBO is the shared expert (19 us at c1, hetero doc section 1.3) hiding part of the
dispatch latency, which requires `supports_async()` (`modular_kernel.py:1223-1234`).

## 4. Options for decode

### 4.1 Layout floors

Bandwidth-only lower bounds per step per rank at 380 GB/s (dense + experts, TP4 dense shard with HC replicated). They do
not include the launch chain, all-reduce or a2a latency, or DP synchronisation.

| Layout | Dense ms | Experts ms, c=1 (W=3) | Sum c=1 | Experts ms, c=8 (W=24) | Sum c=8 |
|---|---|---|---|---|---|
| TP4/EP4 (production) | 8.1 | 1.8 | 10.0 | 10.2 | 18.3 |
| TP2 x DP2/EP4 | 12.9 | 1.8 | 14.7 | 10.2 | 23.1 |
| TP2 x DP3/EP6 | 12.9 | 1.2 | 14.1 | 6.8 | 19.7 |
| DP4/EP4 | 22.5 | 1.8 | 24.3 | 10.2 | 32.7 |
| DP6/EP6 | 22.5 | 1.2 | 23.7 | 6.8 | 29.3 |

Measured TP4 steps are 22.4 ms at c1 and 44.0 ms at c8, so floors are 45 % and 42 % of the step; the rest is the
launch chain, glue and 96 all-reduces. Two consequences, both [INFERENCE] from this model:

- DP6 and DP4 have a c=1 floor (23.7 and 24.3 ms) above the measured TP4 step (22.4 ms), so the c=1..8 ITL gate cannot be
  met by any DP-attention layout. DP replicates the dense stack, which is the cost of removing the attention all-reduce.
- No layout beats TP4/EP4's floor at c=8. Expert parallelism across more ranks saves only 3.4 ms (EP6 against EP4) of
  10.2 ms of expert streaming, and replicating or re-sharding the dense side costs more than that.

At high concurrency the picture changes. Once a layer touches most of its 512 experts (c >= 32; expert floor saturates
at 41.0 ms per step for EP4 and 27.5 ms for EP6), TP4 pays all-reduce time that grows with the batch
(5.1 ms ring at c=32 and 10.1 ms at c=64, x8) while DP6 pays 2.7 and 4.9 ms. The floors for serving 576 rows are
8.1 + 41.0 + about 30 = 79 ms on four GPUs for TP4 and 22.5 + 27.5 + about 15 = 65 ms on six GPUs for DP6 [INFERENCE],
about 7.3 and 8.9 rows per ms in total and 1.8 and 1.5 per GPU. DP-attention EP is a throughput mode that spends 50 %
more GPUs for about 20 % more throughput by this model. It is worth measuring, not worth assuming.

### 4.2 The three honest options at dp=1

| Option | Content | Verdict |
|---|---|---|
| (a) DP attention plus EP layouts | evaluated above by bytes and floors | throughput mode only; gate with a probe before any kernel (phase 2) |
| (b) Overlap existing all-reduce and shared-expert compute | shared-expert overlap is eager-only on gfx10x (capture guard, `shared_experts.py:61-72,148-150`), so decode graphs run it inline; the opt-in monokernel can already run the FFN all-reduce inside K2 on the rdna_ar buffers (`k2_ar`, default off, `mono/rdna2/__init__.py:31`; `mono_k2_rdna2_set_ar`, `vllm/models/qwen4_exp/amd/mono/rdna2/layout.py:104`); a second queue in a captured graph is the known hang | nothing to build. One speculative item, flagged [INFERENCE]: warm the next layer's weights into the 128 MB Infinity Cache from spare blocks of the all-reduce kernel while it waits on peers (a 10-43 us window moves 4-16 MB at 380 GB/s, which is 7-30 % of one layer's dense shard). It needs a microbenchmark before any design |
| (c) Nothing | keep rdna_ar as it is; the monokernel stays an opt-in | the default |

EPLB itself needs no a2a: at dp=1 it is a weight permutation across the EP group (section 9, phase 1).

## 5. What DBO requires and where Flash-Next stands

### 5.1 What ubatching asks of the a2a backend

| ID | Requirement | Reference |
|---|---|---|
| A1 | `supports_async()` True, or `supports_dbo()` True for a synchronous P/F | `modular_kernel.py:255-268`; `_prepare` at `:1236-1321` asserts `not dbo_enabled() or supports_dbo()` |
| A2 | `prepare_async` returns `(hook, receiver)`: the send is issued in `prepare_async`, the hook completes the receive, `receiver()` builds the expert inputs; under DBO the hook is registered with `dbo_register_recv_hook` and the other microbatch runs it at its next yield | `modular_kernel.py:1278,1302-1303` and `:1480-1483` for finalize |
| A3 | Per-microbatch state: handles and staging indexed by `dbo_current_ubatch_id()` | `deepep_ll.py:248,382`, `cuda_communicator.py:731-737` |
| A4 | `ExpertTokensMetadata` plus a format the experts accept, `max_num_tokens_per_rank`, `num_dispatchers` | `modular_kernel.py:89-125`; `deepep_ll.py:135-145` |
| A5 | Backend name in the ROCm DBO allowlist | `vllm/config/vllm.py:2210-2234` (AgRs, deepep, nixl only today) |
| A6 | Graph safety: no host sync in prepare or finalize; no collective on a second stream while capturing | `naive_dp_ep.py:153-171`; `shared_experts.py:107-113` |
| A7 | Shared-expert overlap needs `supports_async()` (the MK asserts it) | `modular_kernel.py:1223-1234` |

Only `deepep_ll.py` and `nixl_ep.py` satisfy A2 today; neither runs on RDNA2. AgRs satisfies A1 only (eager).

### 5.2 What the model and runner ask, and Flash-Next's status

| ID | Requirement | Status for Flash-Next MTP2 |
|---|---|---|
| M1 | Model code yields at layer boundaries or lets the MK do it | Yields live only in `modular_kernel.py` and the P/Fs. No `dbo_` or `ubatch` reference exists in `vllm/models/qwen4_exp`, `vllm/model_executor/layers/mamba` or the GDN/linear attention backends (grep empty). Nothing to wire in the model beyond the MK, but also nothing validated |
| M2 | A DP handshake and dp > 1 | production is dp=1 (section 1.3); no RDNA recipe has run dp > 1 |
| M3 | Per-microbatch attention metadata and builders | V2 builds per-ubatch builders (`vllm/v1/worker/gpu/attn_utils.py:299,480,551-575`); the GDN metadata builder and the QSA state are not ubatch-aware (`vllm/v1/worker/gpu/model_states/mamba_hybrid.py:355` asserts `ubatch_idx == 0, "DBO is not supported"`; V1 `vllm/v1/worker/gpu_model_runner.py:2586` asserts "UBatching not supported with GDN or linear attn yet") |
| M4 | Request-aligned microbatch split | `gpu/ubatch_utils.py` allows a request to straddle the boundary and appear in both microbatches with a truncated query. That is invalid for GDN recurrent state and for the three MTP verify rows of a request |
| M5 | Speculative decoding | rejected by `vllm.py:3278-3279` (advisory under the explicit V2 env, see section 1.3) and unimplemented in the ubatch path |
| M6 | Graph capture | DBO captures FULL graphs on one capture stream (`gpu/cudagraph_utils.py:461-466,555-566`; the PIECEWISE blocker is `vllm.py:3294-3295`). Resident Flash-Next decode uses a plain monolithic `torch.cuda.CUDAGraph`: the V2 FULL path builds a `BreakableCUDAGraphCapture` only when `expert_cache_graphs` is set (monokernel with `expert_cache_slots > 0`, `cudagraph_utils.py:190-195,540-547`), Flash-Next is not in the ROCm default breakable set (`vllm.py:113-137`), AGENTS.md:2363-2365 says only the offload launcher forces breakable graphs, and `@eager_break_during_capture` runs the function inline unless a breakable capture is current (`vllm/compilation/breakable_cudagraph.py:103-105`). So graph structure does not by itself forbid DBO on the resident recipe. It does for every breakable configuration (expert offload, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`), where the decorator breaks FULL captures on RDNA (`breakable_cudagraph.py:106-117`, AGENTS.md:2367-2372; GDN core `qwen_gdn_linear_attn.py:3387,3435`, `mono/rdna2/gdn.py:277-301`, QSA transaction `qsa.py:825`): two microbatch threads cannot interleave across host-executed eager segments. What the resident capture still needs is a pair of per-microbatch GDN conv/recurrent state and `spec_sequence_masks` buffers and QSA state indices, which are persistent buffers sized for one batch [INFERENCE: not audited] |
| M7 | PLE and PLE offload | `ple_layer.py` ops are eager-break during piecewise capture and the runner's `_ple_offload_connector` has one `prepare_forward` and one `release_outputs` per step (`gpu/model_runner.py:2124`); two forwards per step is unaudited |
| M8 | HQD and stream budget | `UBatchRunner` adds a comm stream and a capture stream (`gpu/ubatch_utils.py:327-329`); production already holds 5 KFD queues per worker of the 8 user HQDs at `GPU_MAX_HW_QUEUES=4` (AGENTS.md:975-977 for the budget, :994-999 for the measurement); extra streams alias onto the pool queues and can serialize behind compute |

Config blockers in `vllm.py:3259-3299` also list LoRA, PP > 1, DCP/PCP, multimodal and `is_hybrid` (`:3291-3292`);
Flash-Next trips hybrid, speculative decoding and (with vision) multimodal.

## 6. An async a2a on the rdna_ar machinery

### 6.1 What can be reused

| Piece | Where | Use |
|---|---|---|
| Uncached IPC staging with parity slots `stage[(parity * W + src) * max_elems + i]` | `csrc/rocm/rdna_allreduce.cuh:11,40,181-184` | per-peer receive slots |
| 4 KiB flag page in the same allocation, announce = release store at system scope after the payload push | `rdna_allreduce.cuh:198-227` (`:218`) | count-as-flag |
| Local acquire polling with sticky abort, host-mapped abort record, late-peer versus lost-write timeouts | `rdna_allreduce.cuh:70-81,123-129,248-385` | wedge diagnosis, no GPU hang |
| Device-derived sequence (`seqbuf`), never a kernel argument | `rdna_allreduce.cuh:204,256,271,320` | graph replay safe |
| Instance table (up to 8) and `rdna_ar_device()` for in-kernel reuse | `rdna_allreduce.cuh:209`; `cuda_communicator.py:194-216` | communicator and MonoKernel reuse |
| Island-aware 3-phase algorithm, per-(phase, block, src) flags, slot reuse rule c+2 after c+1 | `csrc/rocm/rdna_allreduce_hier.cuh:1-63`; `docs/rdna2/rdna-ar-hier-2026-11-06.md` | phase-2 hierarchical a2a |
| Placement tables | `fused_moe/rdna_moe_placement.py`, `tools/rdna/prof/moe_placement.py` | expert-to-rank map shared with EPLB |

### 6.2 Protocol

Fixed (src, dst) slots, each `T_max x (5,120 + 64)` B: 122 / 243 / 486 KiB for T_max = 24 / 48 / 96, so a 6-rank receive area
with two parities and two microbatches is 3.0 / 6.0 / 11.9 MB per GPU. A token is sent once per distinct remote rank
(dedup), with its routing record. A sender writes the rows, then one 64-bit word `(sequence, count)` with a release store
into the receiver's flag page: the count is the flag. The receiver polls its own page. The same pattern as DeepEP-LL,
over posted PCIe writes, with no host involvement.

| Kernel | Where it runs | Work |
|---|---|---|
| K0 route | before K1 (fusable into `rdna_moe_route_zero`, `csrc/rocm/torch_bindings.cpp:126`) | expert id to (rank, local expert) from the placement table; per-destination slot index (warp ballot plus a local counter) |
| K1 dispatch send | `prepare_async` | push rows and records into the peers' slots, then the `(sequence, count)` word. Local rows stay in place |
| K2 dispatch recv | the hook | wait for the flags of all R-1 sources (sticky abort as rdna_ar), set ids to -1 beyond each count so padding rows are skipped |
| K3 experts | unchanged | the existing `moe_gptq_gemm_rdna2` pair on `[R x T_max, H]` as Standard format: `moe_align_block_size(..., ignore_invalid_experts=True)` over the received local-expert ids (`fused_moe/experts/rdna2_w4a16_moe.py:207-218`), top-k weight reduction fused into the second GEMM, so `TopKWeightAndReduceNoOP` still applies. `rdna_moe_route_zero` fuses align, count, sort and zero-fill for up to 1024 routed entries (R x T_max <= 102); beyond that use `moe_align_block_size`. `batched_moe_align_block_size` (`moe_align_block_size.py:172`) is the alternative if a `BatchedExperts` layout is ever wanted |
| K4 combine send | `finalize_async` | return each valid received row's partial sum to its source slot (fixed by the source token index: no atomics), then the `(sequence, count)` word |
| K5 combine recv | the finalize receiver | wait for the expected counts, sum the local partial and the returned partials in ascending rank order (deterministic) into `output` |

`output_is_reduced` is True for the DP layout (no TP all-reduce remains), `num_dispatchers = R`,
`max_num_tokens_per_rank = T_max`, `topk_indices_dtype` int64 as in `deepep_ll.py`. `moe_gptq_gemm_rdna2` needs sorted,
padded token ids; the padded `M = R x T_max` is constant per captured size, which is what makes the slot scheme
graph-safe, at the price of an over-launched grid whose idle blocks must exit early [INFERENCE: to confirm in the
kernel]. Because the experts see ordinary Standard-format rows with `-1` padding, no change to the GEMM is needed.

### 6.3 Why single stream is async here

The send kernel returns once its stores are posted; the bytes keep draining over PCIe while the next kernels run. The
hook launches the wait kernel later on the same stream. Overlap comes from posted-write semantics and ordering of the
enqueue, not from a second queue, so there is no cross-queue fork/join and no collective on another stream, the two
hazards in `shared_experts.py:61-72,107-113`. Under DBO the two microbatch threads interleave their enqueues on the
capture stream with plain `dbo_yield()` (`ubatching.py:128`); no `switch_to_comm` is used.

### 6.4 Queue and stream budget

| Resource | Production now | The design adds |
|---|---|---|
| Streams | compute, plus rdna_ar and offload helpers | none; DBO's comm and capture streams exist only with `--enable-dbo` and alias onto the pool |
| KFD queues | 5 per worker at `GPU_MAX_HW_QUEUES=4` (AGENTS.md:994-999) | none (same stream) |
| HQD | 8 user HQDs per GPU (AGENTS.md:975-981) | none; a comm stream would alias, which is the serialization risk |
| Staging | rdna_ar staging per instance | 3-12 MB per GPU, uncached |

### 6.5 Python and config registration

| Change | File |
|---|---|
| `RdnaAll2AllManager` next to `AgRsAll2AllManager` (owns the staging, flag page, ubatch-keyed scratch) | `vllm/distributed/device_communicators/all2all.py` (AgRs at `:55-202`) |
| `elif self.all2all_backend == "rdna_a2a"` branch | `vllm/distributed/device_communicators/cuda_communicator.py:229-305` |
| Literal entry (`rdna_a2a`); default stays `allgather_reducescatter` | `vllm/config/parallel.py:45-60,220` |
| Add to the SP-MoE backend list so TP x DP SP works | `vllm/config/parallel.py:774-791` |
| ROCm DBO allowlist entry | `vllm/config/vllm.py:2210-2234` |
| `use_rdna_a2a_kernels` property | `fused_moe/config.py` next to `:1035-1105` |
| `RdnaPrepareAndFinalize` and the branch in `maybe_make_prepare_finalize` | `fused_moe/prepare_finalize/` (new), `fused_moe/all2all_utils.py:190-` |
| HIP kernels `csrc/rocm/rdna_all2all.cu` plus bindings | `csrc/rocm/torch_bindings.cpp`, `CMakeLists.txt` |

Flag-off behaviour is byte-identical: nothing is constructed unless `--all2all-backend rdna_a2a` is given.

### 6.6 Minimum viable step, validated on CPU plus two GPUs

| Step | Content | Where | Acceptance |
|---|---|---|---|
| M0 | pure-torch reference of the whole protocol (slot assignment, records, counts, combine order) with simulated ranks, plus the P/F adapter checked against a dense MoE reference and the `no_dp_ep` path | CPU, any machine | exact match for random routing, empty ranks, full ranks, skew |
| M1 | kernels K1, K2, K4, K5 on a second rdna_ar instance | two GPUs of slot a2 (HIP 0,1, one child switch) when `HI2.a2` is released | bit-exact against M0 for T = 1..64; graph capture and replay soak (pattern of `tools/rdna/rdna_ar_stress.py`); abort record and recovery with one rank paused |
| M2 | EP2 on a real layer with `RDNA2W4A16MoEExperts` at dp=2 | the same two GPUs | output equal to the AgRs path on the same tokens; per-layer latency at 3 / 12 / 24 / 48 rows per rank against AgRs |

Two GPUs prove protocol, flags and graph safety; they cannot show four- or six-way contention, which needs the lease for
b4 or both slots.

## 7. Go/no-go

| Item | Verdict |
|---|---|
| DBO over an a2a backend on production TP4/EP4 dp=1 MTP2 | **NO-GO.** It does not run (inert at dp=1), the hideable communication is at most 5-14 % of the step, the split costs at least as much at c <= 8 (hetero doc section 4.1 option (d): "Not for decode (W <= 48 is latency-bound; splitting doubles launches)"), and the hybrid-GDN and MTP paths of model and runner do not support it |
| An rdna_a2a backend without DBO | **Conditional.** Feasible (section 6) but has no consumer until a DP layout exists; build it only after phase 2 shows a DP layout worth serving |
| DBO on a DP6/EP6 or DP4/EP4 throughput layout | **NO-GO by the floor model** (section 3.6: the split re-streams more than it hides at every size tabulated). It flips only if a measurement shows compute-bound layers or resident weights; preconditions below |
| Quantized dispatch, LL fixed-slot, island-aware a2a | follow the a2a backend; INT8 dispatch is lossy and cannot meet "greedy identical" strictly (section 9) |

Exact preconditions for a DBO GO (all required):

1. Layout: dp >= 2 with EP and a serving target of 24-48 or more rows per rank (a throughput recipe; `flashnext-mtp2` stays
   TP4).
2. Evidence from phase 2/3 profiles that exposed communication is at least 10 % of the step at that size.
3. A single-stream async backend (A1-A7) that passes the M1 soak.
4. GDN, QSA and PLE state per microbatch; request-aligned splits that keep the three MTP rows of a request together;
   `ubatch_idx` support in `mamba_hybrid.py`; the speculative-decoding and hybrid blockers lifted by implementing the
   support, not by bypassing them.
5. Capture of a microbatch pair with per-microbatch GDN and QSA state buffers. Resident decode uses a monolithic FULL graph, so this is buffer work; breakable configurations (expert offload, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`) cannot interleave two threads across eager segments and stay out of scope.
6. A measured two-microbatch microbenchmark with real kernels showing the hidden communication exceeds the re-streamed
   weight cost.
7. The ITL, PPL/greedy and KFD gates of section 9.

The layout change that makes an a2a backend meaningful is DP attention plus EP: DP6/EP6 on all six GPUs (DP4/EP4 on b4
loses to TP4/EP4 by the floors of section 4.1), run as a separate throughput recipe. DBO on top of it also needs the
measurement of precondition 6. Neither can replace `flashnext-mtp2` for c=1..8.

## 8. Ranked risks

| Rank | Risk | Why | Mitigation |
|---|---|---|---|
| 1 | Cross-queue fork/join in a captured graph hangs gfx10x | `shared_experts.py:61-72`: about one hang per 1-4 h, wedging the rank's queues and the next collective | single-stream LL protocol; never a comm stream under capture |
| 2 | Config and model hard blockers | hybrid GDN, speculative decoding, `ubatch_idx` asserts, per-microbatch GDN/QSA state (section 5.2) | treat as design work, not flags; gate DBO last |
| 3 | The split costs more than it hides | second microbatch re-streams dense weights (about 399 us per layer at 380 GB/s) and doubles a latency-bound chain; the ceiling is 8-13 % | measure with real kernels before building the scheduler |
| 4 | Collective ordering under independent branches | `shared_experts.py:107-113`, vllm issue 60845; sequence/parity mismatch for rdna_ar | ubatch-keyed staging parities (`cuda_communicator.py:731-737`), one stream |
| 5 | DP>1 has never run on this stack | no recipe, DP sync (`dp_utils.py`) and AgRs collectives untested on RDNA | phase 2 probe with the existing AgRs backend before any kernel |
| 6 | HQD aliasing | extra streams share the pool queues; production already uses 5 of 8 | one stream; recipe `GPU_MAX_HW_QUEUES` unchanged |
| 7 | rdna_ar failure modes multiply | lost posted writes, wedge records, more flag pages | reuse the abort record and soak tooling; per-instance isolation |
| 8 | Topology unknown until measured | x8 vs x16, intra-leaf vs cross-leaf contention, `HSA_ENABLE_SDMA` | `p2p_bench.py all` once per SDMA setting |
| 9 | EPLB replicas raise streamed bytes | hetero doc: 1.235-1.365x streamed; vLLM EPLB at EP6 needs 4 redundant replicas (`layer.py:271-279`) | permutation only, through explicit placement and the uneven split (`expert_map_manager.py:72-90`) |
| 10 | INT8 dispatch changes numerics | lossy; W4A16 decode would see quantized activations | PPL and MTP-acceptance gate, not greedy equality |

## 9. Phased plan, gates and effort

Efforts are engineer-days, plus or minus 50 % [INFERENCE]. Gates common to every phase that touches serving: decode ITL
p50 and p99 within noise or better at c=1..8 with MTP2 (ABBA, fresh boots), PPL and greedy probes identical, zero KFD
evictions (`evicted_ms` +0), no rdna_ar wedge, AER counters unchanged.

| Phase | Deliverable | Gate | Days | Stop rule |
|---|---|---|---|---|
| 0 | Run `tools/rdna/a2a/p2p_bench.py all` on b4 and on a2 plus b4, once per `HSA_ENABLE_SDMA` value; graph fork/join soak only if DBO continues | x8/x16 settled; per-pair copy and probe push/pull/flag numbers; collapse verdicts for intra-leaf and cross-leaf flows | 3 (mostly lease waits) | none; the numbers feed every later model |
| 1 | EPLB-lean, owned by the `tools/rdna/eplb/` workstream already on this branch (commits `6abab04fb7` and `ff3a0a9e55`: simulator, placement policies, rebalance policy, solver, kernel-time objective fit). Constraints taken from this study: permutation only, no replicas; objective = distinct experts touched per rank at decode (weight bytes), not token counts; EP6 uses the uneven 86/86/85/85/85/85 split; moves are P2P pushes of packed expert blobs; one placement table per layer, shared with the future dispatch kernel | that workstream's gates; the cross-cutting ones: ITL within noise at c=1..8, greedy identical (a permutation is exact), 0 evictions | not counted here | that workstream's stop rule (its offline replay must show a gain before any serving change) |
| 2 | DP-layout probe with the existing AgRs backend, no new kernel: DP6/EP6 on all six GPUs (both leases) and DP4/EP4 on b4, against TP4/EP4 on b4 at c=16/32/64/128, MTP2; needs dp > 1 bring-up on the hybrid GDN/MTP model (no recipe has run it); profile the collective share of each layer | DP6/EP6 total tokens per second at least 1.15x TP4/EP4 at c >= 64 (the floor model predicts about 1.2x [INFERENCE]); otherwise the a2a program stops, because AgRs moves only about 19 % more bytes than a dedup a2a at R=6 (a2a is 84.1 % of an all-gather's bytes) and a2a cannot close a larger gap | 5 | below 1.15x, or DP does not bring up on the hybrid/MTP model |
| 3 | `rdna_a2a` MVP: M0-M2 of section 6.6, manager, P/F, allowlist | M1 soak clean; M2 latency per layer at 24 rows at most half of AgRs | 9 | M2 not at least 2x better than AgRs at 24 rows |
| 4 | LL fixed-slot tuning and INT8 dispatch (dispatch carries int8 rows plus scale, feeding the W4A8 path or a receiver dequantize) | PPL within +0.02 of 4.80 and MTP acceptance within 0.02; greedy identical is not promised and needs a decision | 6 | no measurable per-layer gain |
| 5 | Single-stream DBO scheduler, GDN/QSA/PLE per microbatch, request-aligned split | preconditions 1-7 of section 7 | 12-15 | exposed communication under 10 % or re-stream cost above the hidden time (the floor model says it is, section 3.6; this phase starts only if a measurement overturns that) |
| 6 | Island-aware hierarchical a2a from the `rdna_allreduce_hier.cuh` pattern | only if phase 0 shows intra-leaf LINEAR and cross-leaf COLLAPSE or SATURATED | 5 | phase 0 shows no cross-leaf contention (the lease README measures about 13.9 GB/s same or cross leaf) |

Interfaces the a2a work needs from EPLB (phase 1): one placement table (logical expert to rank and local slot) per layer,
shared by the router epilogue and the dispatch kernel; updates swapped at a step boundary with double-buffered slots; no
dependence on `rdna_expert_cache.py` (which rejects EPLB at `:66-67`; `rdna_expert_stats.py:77` also says keep EPLB off).
The load signal must be cheaper than the existing collector, which costs +0.49 ms at c1 with fused kernels and +24.7 % ITL
with the torch formulation (T10 section 1): sample windows or fuse the histogram into the router epilogue. Moving one
packed expert (2.4 MiB) costs about 180 us of link time at 14 GB/s [INFERENCE], so a 32-experts-per-layer rebalance across
48 layers is about 0.28 s per rank.

## 10. PCIe microbenchmark

`tools/rdna/a2a/p2p_bench.py` is the plan's phase 0. It records the sysfs topology (island key, link speed and width, the
narrowest link on each GPU's path) and measures with two backends.

| Stage | Torch backend (`Tensor.copy_`, no kernels) | Probe backend (existing `tools/rdna/p2p_probe/p2p_probe`, kernel level) |
|---|---|---|
| Per-pair bandwidth and latency, 4 KiB to 64 MiB | copy bandwidth per ordered pair (the write direction) and single-copy completion latency, with verification | `curve push/pull/sdma`: posted stores, peer loads, `hipMemcpyPeerAsync`; flag latency `lat` |
| 1..4 concurrent pairs | `flows` ladders: `intra_A`, `intra_B`, `intra_all`, `cross`, `cross_bidir`, `fanin`, `fanout`, with a COLLAPSE / SATURATED / LINEAR verdict | `bw` with concurrent `p:` transfers |
| All-to-all among 4 and 6 GPUs | flat step for `island1_4`, `mixed_4`, `flat_6`, `pair_2`, and the intra-island concurrent groups (phase 1 of an island-aware a2a) | `a2a` one-shot step as used by rdna_ar |

Limits, stated so results are read correctly. Torch runs a cross-device copy on the source device's stream between a
two-way event barrier, so it measures the write direction of a synchronised copy: small sizes are launch and barrier
bound, flows that share a destination are lock-stepped, and a small-chunk all-to-all is bound by the host enqueue loop.
Torch cannot issue peer loads or flag stores; use the probe for pull bandwidth and flag latency. `HSA_ENABLE_SDMA` changes
the `hipMemcpyPeerAsync` path and `GPU_MAX_HW_QUEUES` below the largest per-GPU fan-out (5 for a flat six-GPU a2a) makes
flows share a queue, so both are recorded and the script warns.

Run it only inside a lease window. The script checks that `<owner>.<slot>` lease files cover every visible GPU's PCI
bus, refuses while a docker build runs, touches `~/w4a8_runs/lanes/quiet/<owner>` (at most 14 minutes), samples AER
counters before and after, and writes JSON.

```text
python3 tools/rdna/a2a/p2p_bench.py selftest
python3 tools/rdna/a2a/p2p_bench.py plan --islands "0,1;2,3,4,5"
HIP_VISIBLE_DEVICES=2,3,4,5 GPU_MAX_HW_QUEUES=6 HSA_ENABLE_SDMA=0 \
  python3 tools/rdna/a2a/p2p_bench.py all --owner <Agent> --out b4_sdma0.json
HIP_VISIBLE_DEVICES=2,3,4,5 GPU_MAX_HW_QUEUES=6 HSA_ENABLE_SDMA=1 \
  python3 tools/rdna/a2a/p2p_bench.py all --owner <Agent> --out b4_sdma1.json
```

The `plan` output estimates about 4-5 minutes for six GPUs (rough). The six-GPU run needs both the a2 and b4 leases.

Decision rules the results feed:

| Result | Consequence |
|---|---|
| narrowest link x8 and single-pair plateau about 13-14 GB/s | use the x8 column of section 3; a2a is latency-bound at c <= 8 |
| plateau at or above 23 GB/s | x16 column; the model's link-bound times halve |
| `cross` or `cross_bidir` COLLAPSE or SATURATED while `intra_*` is LINEAR | contention across child switches: phase 6 is justified |
| all ladders LINEAR | no island structure to exploit; drop phase 6 |
| probe pull far below push | keep the design push-only (it is) |
| probe one-shot a2a step for six GPUs at 16-64 KiB about 15-60 us (old eight-GPU flat 15.7-62.4 us) | the a2a fixed cost is comparable to an all-reduce; no latency win at c <= 8 |
| AER counters move | link errors under load; stop and report |

## Appendix A. Measured numbers and their provenance

| Number | Value | Topology | Source |
|---|---|---|---|
| Push write per direction | 24.5-27 GB/s | old, x16 | `pcie-iommu-audit-2026-11-06.md:112-131` |
| Cross-root pull | 5.7 GB/s (intra 24.4) | old | same |
| Concurrent cross-root pairs 1/2/3/4 | 25.2 / 27.6 / 18.7 / 17.0 GB/s | old | `:180-192` |
| One-way flag latency | about 7 us at 4-32 KiB | old | `:222-241` |
| One-shot a2a step, 4 GPUs inside a switch | 10.4 / 11.3 / 10.9 / 14.8 / 22.0 us at 4-64 KiB | old | same |
| rdna_ar per call, TP8 | 29 / 33 / 46 / 62 / 107 / 136 / 196 us at 5-240 KiB | old | `tp8-pp2-flashnext-2026-11-06.md:119-142` |
| rdna_ar per call, TP4 | 43 us at 120 KiB, 67 us at 240 KiB | old | same |
| RCCL TP4, 10 MB and 20 MB | 616 us and 1228 us | old | `tp4-allreduce-sizes-2026-10-10.md:35-36` |
| GPU link, interim box | x8, 14.015 GB/s practical, about 13.9 measured | single switch | `pciescope-2026-11-07.md:22-26,64-76`; hetero doc section 11.0 |
| TP4 ITL c1 / c6 / c8 | 22.54 / 43.57 / 45.63 ms | single switch (HIP 2-5) | hetero doc section 11.0 |
| TP4 ITL c1 / c8, HWQ=4 | 22.66 / 44.49 ms | old | `hwq-abba-2026-11-06.md` |
| Decode trace c1 (attention 252 us, FFN 181 us) and c8 (490 / 411 us) | see hetero doc section 1.3 | older tree | hetero doc |
| Expert collector overhead | +0.49 ms (+2.2 %) c1 fused; +24.7 % torch | single switch | T10 section 1 |
