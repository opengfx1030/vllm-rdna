# Heterogeneous MoE serving: W7800 hubs and V620 expert servers (design, 2026-11-07)

Date: 2026-11-07. Branch `feat/hetero-moe`. Box: par1-cs25 (two PEX88096 switches, each under one x16 Gen4 root: `c0` = slot A, `00` = slot B; per switch one W7800 and four V620).
Model: `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` (Qwen4Exp, `vllm/models/qwen4_exp/amd/`), W4A16 symmetric g128 routed experts, MTP=2.

Status: **design only**. No GPU, build or service was used for this document; the box was queried read-only (`rocm-smi --showmeminfo vram`, sysfs, file reads).
Every W7800 or hetero-path performance number below is an output of the analytic model `tools/rdna/hetero_moe/perf_model.py` (calibrated to measured V620 TP4 and
cross-checked against measured TP8), not a measurement. Claims without a measurement or a file:line are marked [INFERENCE].

This document owns the plan format, the runtime semantics and the kernel requirement lists. Two sibling branches implement against it:
`feat/rdna3-flashnext` (Flash-Next W4A16 on gfx1100) and `feat/expert-stats` (routing statistics, placement solver; its document is `expert-stats-2026-11-07.md`,
CPU-tested tooling at `tools/rdna/hetero_moe/`, pushed as `origin/feat/expert-stats`).

## Summary and decisions

1. **Shape.** The W7800s ("hubs", gfx1100) run everything except the cold routed-expert GEMMs: HC, GDN, QSA attention, router, shared expert, lm_head, PLE, the MTP layer,
   KV and the sampler, plus a small set of hot experts. The V620s ("expert servers", gfx1030) hold all routed experts of their island and compute the cold ones.
   The hub **ships activations, never weights**: per layer it pushes the MLP input rows plus top-k ids and weights (5,180 B per row per server) into each server's
   inbox with P2P posted stores, the servers push their fp16 partial sums (5,120 B per row) back into hub VRAM, and the hub polls a flag in its own VRAM.
2. **Topology: H2.** One engine, TP2 across both hubs, EP2 across the two switches: island k = hub k + the four V620 of its switch, and each island owns half of the 512
   experts of every layer. One hub per switch as an independent TP1 engine (H1) is dominated: a lone W7800 must stream 7.96 GiB of dense weights per step, 11.5 ms at
   691 GB/s, before any compute. The prerequisite and baseline is W2: the same TP2 engine on the two W7800 alone, with no V620 (29.4 GiB of experts per hub).
3. **Model results** (central assumptions; ms per decode step with MTP2 included; TP4 and TP8 rows are measured, the rest is model):

   | Engine | c1 step ms | c8 step ms | 1k c8 tok/s (one engine) | prefill chunk s (2048) | 16k TTFT s | KV tokens |
   |---|---|---|---|---|---|---|
   | TP4 V620 (today) | 22.4 | 44.0 | 263 | 0.69 | 5.5 | 266,945 |
   | TP8 V620 (today) | 19.6 | 41.6 (37.2 hier) | 290 | 0.66 | 5.3 | 266,945 |
   | W2 (2 W7800, no V620) | 24.7 | 45.7 | 278 | 0.49 | 3.9 | 388k |
   | H1 (hub TP1 per switch) | 32.0 | 52.9 | 232 | 0.66 | 5.3 | 718k |
   | **H2 (TP2 hubs + 8 expert servers)** | **23.3** | **36.1** | **354** | **0.37** | **3.0** | 1,702k |
   | H2 + W8A16 hub dense + lm_head on the servers | 18.9 | 31.8 | 391 | 0.37 | 3.0 | 1,702k |

   H2 versus TP4: c1 +4 %, c8 step -18 %, prefill 1.9x faster, 16k TTFT 3.0 s versus 5.5 s. With the two dense levers c1 is -16 % and c8 -28 %. The uncertainty is
   large and one-sided on the W7800 port quality: pessimistic assumptions give H2 c1 29.1 and c8 46.8 ms (section 5.4).
4. **The hub's hot set is balance-limited, not capacity-limited.** A hub could hold 20+ GiB of experts, but a resident expert only helps while the hub and the servers
   finish together. Per hub in H2 about 7.2 of its 14.6 touched experts per layer run locally at c1, 39 of 87 at c8, 63.5 of 141 at c16 (45-49 % of the touched set; 34-39 % in H1).
   Every expert also lives on a V620 (`cold_complete`), so a touched hub-resident expert beyond the balance point is computed remotely ("spill").
   The rest of the hub VRAM goes to KV. The V620s hold primaries (7.3 GiB per V620 in H2) plus replicas up to the 26 GiB budget, which is what balances the servers.
5. **Servers are separate processes outside vLLM's EP group** and the protocol is the `rdna_ar` one: payload stores, fence, sequence-numbered flag in the receiver's own
   VRAM, device-side wait with timeout and sticky abort. Not NIXL/UCX (28.96 us notification), not upstream all2all/EPLB (lockstep, one arch per process), not weight
   streaming (measured 52.6 to 61.1 ms c1 ITL under the existing TP4 offload).
6. **MTP: policy `hub`.** The MTP layer and its experts live on the hubs (each hub holds its island's half), requantized to W4A16 (1.2 GiB total) with
   `tools/rdna/quant_mtp_experts.py`. Gate: drafter acceptance length at least 97 % of the bf16 drafter, else bf16 on the hub (4.69 GiB).
7. **Machine-level caveat (key user decision).** H2 lowers per-stream latency, it does not raise aggregate throughput. On all ten GPUs the engine split
   D3 = W2 on the W7800s + 2 x TP4 on the V620s is predicted at about 803 tok/s over 24 users (1k/512) against 525 for today's 2 x TP4 and about 494 for H2 at 16
   users, with no new protocol (only the RDNA3 port and a request router). Hetero wins per-stream latency at c8, TTFT, and models whose experts do not fit the
   W7800s (this model's experts are 58 GiB, they do fit 2 x 44.98 GiB). Section 5.5 has the table; section 13 lists the decision.
8. **Order of work.** S1 = W2 first (it is also the numerics and speed baseline for the hub, and its gate decides whether H2 is worth building), S0 microbenchmarks
   in parallel, then the expert-server protocol (S3 to S5), MTP (S6), prefill (S7), placement from real routing statistics (S-1, S8). Slices and gates are in section 10,
   the GPU test plan in section 11.
9. **Not claimed.** Nothing was run on a W7800. The routing skew of this model is unmeasured (no trace exists; collection is S-1). Cross-arch `hipIpcOpenMemHandle`
   and kernel stores into a V620 buffer mapped in a W7800 process are untested (same-arch cross-root IPC with disjoint visible sets is measured:
   `docs/rdna2/disagg-g0-2026-11-06.md:99-101`).

## 1. Inputs

### 1.1 Devices and PCIe

| Role | Cards | Arch | VRAM (rocm-smi, 2026-11-07) | Memory bandwidth | Compute |
|---|---|---|---|---|---|
| hub | 2 x W7800 48GB (HIP 0 bus `cb`, HIP 5 bus `07`; rocm-smi GPU[0], GPU[5]) | gfx1100 | 48,301,604,864 B = 44.98 GiB | 864 GB/s, 96 MB Infinity Cache (datasheet; achieved unmeasured, model assumes 0.80) | 90.4 TFLOPS fp16 WMMA, bf16 dot2 |
| expert | 8 x V620 (HIP 1-4 buses `ce d3 d8 db`, HIP 6-9 buses `0a 0f 14 17`) | gfx1030 | 32,195,477,504 B = 29.98 GiB | 512 GB/s peak, about 380 GB/s achieved on W4A16 experts (`docs/rdna/monokernel/p0-results.md:263`) | 40.5 TFLOPS fp16 vector, V_DOT2, no WMMA |

The brief and the first solver defaults assumed 32 GiB for the W7800; the real value is 44.98 GiB (`rocm-smi --showmeminfo vram`, both W7800). Host: EPYC 7K62 (48 cores), 512 GiB;
the PLE n-gram table is 95.4 GiB of pinned host RAM (or the int4 sidecar).

PCIe, all measured on this box (`docs/rdna2/pcie-iommu-audit-2026-11-06.md`; the 2026-11-07 reboot tests changed no P2P number, the box now runs `pcie_ports=native`
without `amd_iommu=on`):

| Quantity | Value | Source |
|---|---|---|
| P2P push (posted stores), per pair per direction | 24.5-27 GB/s intra-switch, 25.2 GB/s cross-root | audit `:20,22,129-136` |
| Cross-root read (`hipMemcpyPeerAsync`) | 5.65 GB/s (64 KiB: 4.6-6 GB/s); UCX `rocm_ipc` READ 25.5 GB/s | audit `:129-136`; `disagg-g0-2026-11-06.md:17,60-61` |
| Cross-root aggregate by concurrent pairs | 1: 25.2, 2: 27.6, 3: 18.7, 4: 17.0 GB/s | audit `:326-350` |
| One-way flag latency | 6.2 us intra-switch, 6.9 us cross-root; 16 KiB + flag about 7 us; 64 KiB + flag 8.4 / 9.0 us | audit `:151` |
| Intra-switch ring of four, fan-in/fan-out from one GPU | 105 GB/s; 25.6 GB/s | audit `:191` |
| NIXL/UCX notification | median 28.96 us one way | `disagg-g0-2026-11-06.md:68` |
| rdna_ar one-shot vs RCCL at 128 KiB, TP8 | 36 vs 136 us; RCCL 4 KiB TP8 110 us | `rdna-ar-hier-2026-11-06.md:236-276` |

Hub to server traffic stays inside one PEX88096 switch (one x16 link on the hub side, 25.6 GB/s per direction shared by the four servers). The only cross-root flows in H2
are the two hubs' TP2 all-reduces, far below the three-pair collapse.

### 1.2 Measured V620 baselines (the calibration anchors)

From `docs/rdna2/tp8-pp2-flashnext-2026-11-06.md:424-441` (production compiled mode, MTP=2, one fresh boot per arm):

| Layout | c1 ITL p50 | 1k c8 ITL | 1k c8 tok/s | 16k c1 TTFT | Acceptance (1k c1 / c8) | Weights per GPU | KV tokens at 3.75 GiB |
|---|---|---|---|---|---|---|---|
| TP4 | 22.38 ms | 43.97 ms | 262.6 | 5.52 s | 1.84 / 2.01 | 20.56 GiB | 266,945 |
| TP8 | 19.59 ms | 41.56 ms (37.2 with the hierarchical all-reduce) | 290.3 | 5.28 s | 2.00 / 1.97 | 11.23 GiB | 266,945 |
| PP2 (TP4 x PP2) | 23.08 ms | 47.23 ms | 272.7 | 3.40 s | 2.39 / 2.04 | 9.79 / 11.12 GiB | 495,480 |

Prefill aggregate at 1k c8: TP4 2806, TP8 3054, PP2 4110 tok/s; per request at 16k c1: 2966 / 3100 / 4823 tok/s. Non-torch fixed cost per GPU is 1.4-1.7 GiB.

### 1.3 Decode-step trace (TP4, older tree, `docs/rdna/monokernel/p0-results.md:149-279`)

| Item | Value |
|---|---|
| Attention block, c1 | 252 us, of which HC preamble 66 us |
| FFN block, c1 | 181 us: HC 66, shared expert 19, router 39, experts 52, residual 4 |
| All-reduce | 10.6 us per call (98 calls per step) |
| Blocks at c8 (24 rows) | attention 490 us, FFN 411 us |
| Expert streaming rate | about 380 GB/s (18 MB touched at c1) |
| Kernels per step | 2523 for the target graph at width 3 in the trace tree (`:187-203`); later trees measured 1739 then 1595 per M=24 step (`docs/rdna2/v0.31.0-port-notes.md:788,816`) |

The decode step is bandwidth-bound on the dense chain at c1 (7.96 GiB per step) and on the expert set at c8 and above; the glue (router, top-k, align, sort) is latency-bound.

## 2. Byte table

Exact, from the safetensors headers of the 168.285 GiB checkpoint (computed with a throwaway script; identical to `tools/rdna/hetero_moe/data/flashnext_w4a16_bytes.json` on `origin/feat/expert-stats`).
Model: 48 layers = 12 x [GDN, GDN, GDN, QSA]; hidden 2560; 512 routed experts, top-10, one shared expert; `moe_intermediate_size` 640; vocab 248,320.

### 2.1 What lives where

| Component | Per unit | Units | Total GiB | Hub | Server |
|---|---|---|---|---|---|
| Routed expert, W4A16 sym g128 (packed int4 + fp16 scales) | 2,534,400 B (2.417 MiB; V620 repack adds 19,200 B qzeros) | 24,576 | 58.0 (1,297,612,800 B = 1237.5 MiB per layer) | hot subset | all of the island's |
| GDN (in_proj, out_proj, conv, a/b) | 110.55 MiB | 36 | 3.886 | yes | no |
| QSA attention | 95.0 MiB | 12 | 1.113 | yes | no |
| QSA indexer | 3.125 MiB | 12 | 0.037 | yes | no |
| HC (both sublayers; replicated on every TP rank) | 25.2 MiB | 48 | 1.193 | yes | no |
| Router (gate) | 2.5 MiB | 48 | 0.117 | yes | no |
| Shared expert | 9.38 MiB | 48 | 0.440 | yes | no |
| Embedding | | | 1.184 | yes | no |
| lm_head | | | 1.184 | yes | optional vocab shard |
| PLE projections | | | 0.061 | yes | no |
| MTP dense | | | 0.169 | yes | no |
| MTP experts, bf16 | 9,830,400 B (9.375 MiB) x 512 | | 4.688 (W4A16: 1.20) | policy `hub` | no |
| Vision tower | | | 0.836 | excluded (text-only v1) | no |
| PLE n-gram table | | | 95.4 | host pinned RAM (sidecar) | no |

Active part per hub without MTP experts: 9.215 GiB of dense weights plus 0.169 MTP dense = 9.385 GiB (`hub_active_bytes` 10,076,534,296 B in the solver's byte table). At TP2 a hub
holds 5.3 GiB of it (HC replicated), plus 0.7 GiB of W4A16 MTP experts. Routed experts are 86 % of the resident bytes and 29 % of the bytes a c1 step touches (section 2.2).

Per layer: routed experts 1,237.5 MiB (512 x 2.417 MiB); active part 147.6 MiB for a GDN layer (110.55 + HC 25.2 + router 2.5 + shared 9.38) and 135.2 MiB for a QSA layer (95.0 + indexer 3.1 + 25.2 + 2.5 + 9.38),
144.5 MiB averaged over the 36:12 mix, plus 2.43 GiB once for embedding, lm_head and PLE projections. The routed-to-active ratio is 8.6 : 1 in bytes stored, but one token reads only its top-10 experts
(24.2 MiB per layer), so a single row streams 6x more active-part bytes than routed bytes.

### 2.2 Bytes touched per decode step

Distinct experts per layer: N(W) = 512 (1 - (1 - 10/512)^W) for W rows, scaled by 0.9 beyond W = 24 (fit of the TP4 c8 delta, section 5.3); decode reads dense 7.96 GiB per step
(GDN 3.886 + QSA 1.113 + indexer 0.037 + HC 1.193 + router 0.117 + shared 0.440 + lm_head 1.184).

| Rows W (requests x 3 with MTP2) | Distinct experts per layer | Routed bytes per step (48 layers) | Dense bytes per step |
|---|---|---|---|
| 1 | 10 | 1.1 GiB | 7.96 GiB |
| 3 (c1) | 28.8 | 3.3 GiB | 7.96 GiB |
| 24 (c8) | 174 | 19.7 GiB | 7.96 GiB |
| 48 (c16) | 282 | 31.9 GiB | 7.96 GiB |

So at c1 the dense chain, not the experts, is 71 % of the bytes; that is why the dense part decides the hub topology (section 5).

Per-token wire sizes: one MLP-input row x is 5,120 B (2560 fp16). A request row adds the routing, 10 expert ids as int16 plus 10 weights as fp32 = 60 B, so 5,180 B per row per
server; the reply is 5,120 B per row; a residual-stream row is 20,480 B (HC keeps four streams) and never travels.

KV and state: 24 KiB per token at TP1 (12 QSA layers x 2 KV heads x 256 x 2 x 2 B) plus about 2.7 KiB indexer; 14.7 KiB per token measured at TP4 and above. GDN state is about 3.06 MiB per
layer per sequence (about 110 MiB per sequence over 36 layers).

## 3. Existing mechanisms: reuse, adapt, avoid

All paths relative to the repository root, lines verified on `rdna_extras @ 750fff8a64`.

| Mechanism | Where | Verdict |
|---|---|---|
| RDNA2 W4A16 MoE method; `apply` passes `global_num_experts` and `expert_map` | `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py:322-323` | **Reuse as the server's kernel path** and as the model for the hub path |
| Expert-cache slot remap: ids -> slots with `expert_map=None, ignore_invalid_experts=True` | same file `:338-361` | **Reuse the pattern** for the hub's swappable slot pool |
| `_rdna2_fused_moe` body; align with `ignore_invalid_experts`; fused top-k reduce `output_topk=top_k` | same file `:366-379`, `:491-497`, `:619,636` | Reuse: a masked call returns the partial sum over the experts it owns |
| `-1` expert masking | `fused_moe/experts/rdna2_w4a16_moe.py:100,131-132,154-156`; `fused_moe/moe_align_block_size.py:128-129,164-165,228-231`; `csrc/rocm/moe_q_gemm_rdna2.cu:111-116` | Reuse as is; this is what makes partial sums safe |
| Placement JSON for EP | `vllm/model_executor/layers/fused_moe/rdna_moe_placement.py` (`VLLM_RDNA_MOE_PLACEMENT`; uniform capacity per rank; tool `tools/rdna/prof/moe_placement.py`) | **Reuse for the island partition** (island = EP rank, 256/256) |
| Routing histogram | `fused_moe/rdna_routing_hist.py` (prefill-only, host sync, skipped under capture) | Not reused; replaced by the graph-safe collector `rdna_expert_stats.py` on `feat/expert-stats` |
| Expert cache (host-packed experts into device slots) | `fused_moe/rdna_expert_cache.py`, `rdna_expert_cache_policy.py`, `csrc/rocm/rdna_expert_cache.cu` | Avoid: graph capture refused, monokernel + gfx10x only, rejects EPLB. Measured 52.6-61.1 ms c1 ITL at TP4 offload (`docs/rdna2/moe-offload-topology-2026-11-02.md:58-67`) |
| EP group and EPLB | `vllm/distributed/parallel_state.py:1581-1587,1657-1666,2324-2351`; `vllm/distributed/eplb/eplb_state.py:561-693,758-826`; `vllm/models/qwen4_exp/amd/model.py:409-456` | Avoid: lockstep collectives on every EP rank (`eplb_state.py:666-693`); servers outside the group |
| all2all managers, `naive_dp_ep` prepare/finalize | `vllm/distributed/device_communicators/all2all.py`; `fused_moe/prepare_finalize/naive_dp_ep.py:118-127,192-269` | Avoid: collectives within one engine; DeepEP/NixlEP `combine` unimplemented |
| Shared-expert ordering (`NO_OVERLAP`, `MK_INTERNAL_OVERLAPPED`, `MULTI_STREAM_OVERLAPPED`) | `fused_moe/runner/shared_experts.py:26-37`; `runner/moe_runner.py:598-652` | Reuse the `MK_INTERNAL_OVERLAPPED` slot: shared expert runs between dispatch and combine. Aux stream is disabled under capture on gfx10x (`shared_experts.py:61-66`, hang history) |
| MoE block, HC, PLE, MTP | `vllm/models/qwen4_exp/amd/model.py:175-192` (SparseMoeBlock), `:195-406` (decoder layer, `self.mlp(block_input)` at `:383,387`); base `vllm/model_executor/models/qwen3_next.py:131-285` (gate, shared expert, `self.experts(...)`); `amd/hyperconnection.py:92-296`; `amd/mtp.py:190-390`; `vllm/v1/spec_decode/qwen4_exp.py` | Unchanged except the experts' quant method |
| Shared expert fused ops | `vllm/model_executor/layers/rdna_ops.py:215-265` | Hub needs a gfx1100 equivalent (Rdna3Port) |
| `rdna_ar` protocol (VRAM flags, parity staging, payload-before-flag, sticky abort record, 300 s late-peer vs 3 s lost-write rules) | `csrc/rocm/rdna_allreduce.cuh:4-35,40,62-70,88-110,137,181-227,298-301`, `rdna_allreduce.cu:75,99-106,141,188-189`; `ops.h:248-269`; `vllm/distributed/device_communicators/rdna_all_reduce.py`; K2 reuse `csrc/rocm/mono_k2_rdna2.cu:1995-2040` | **Reuse the design** for `hx` (section 6.4). Python gate is `on_gfx10x` at `vllm/distributed/device_communicators/cuda_communicator.py:196-200`: must widen for TP2 on gfx1100 |
| Arch gating | `vllm/platforms/rocm.py:225-260`, `rdna.py`; `CMakeLists.txt:55,1651-1654,1706-1711,1732-1735,1743-1749`; `csrc/rocm/torch_bindings.cpp:99-100,299,1287-1313`; wrong-arch behaviour `docs/rdna2/build-break-2026-10-31.md:281-288` | Constraint (section 8): process-global arch, silent zero output on a wrong-arch kernel |
| Launcher and recipes | `tools/rdna/serve_rdna.sh:170-176,226-240` (`PCI_BUSES`), `tools/rdna/rdna_launcher_common.sh:125-131`; `tools/rdna/recipes/flashnext-mtp2.env`, `flashnext-tp8-mtp2.env`, `flashnext-tp4pp2-mtp2.env`, `flashnext-pp3-mtp2.env` | Reuse; add `flashnext-w2-mtp2.env` and `flashnext-hetero-h2.env` (section 6.10) |
| TunableOp | `tools/rdna/prof/tunableop_missing.py`, `tools/rdna/prof/tune_missing_gemms.sh` (`REFUSED_GPUS` default "0 1": the W7800 are 0 and 5 on this box) | Per-ordinal rows needed for the W7800 |
| PLE sidecar | `docs/rdna2/flash-next-ple-ipc.md` (`ALLOC_CONF=auto`; expandable segments break ROCm IPC) | Same on the hubs; applies to every IPC buffer in `hx` |
| Pinned-RAM wedge | `docs/rdna2/rdna-ar-wedge-2026-11-04.md` | All `hx` mailboxes are `hipMalloc` VRAM, never host-pinned |
| MTP quantizer | `tools/rdna/quant_mtp_experts.py` (RTN; output shard `model_mtp_w4a16.safetensors`) | Reuse for the W4A16 drafter |
| Two-lane engine | branch `feat/two-lane`, `docs/rdna2/two-lane.md:3,74-80,955` (not on `rdna_extras`) | Not combined: its KFD census is already 6-7 queues against the repo budget of 8 (`:74-80`) and the TP4 Flash-Next M4 verdict is no-go (`:3,955`) |
| Prior art `origin/cursor/hetero-moe-rdna3-rdna2-13b1` | branch only | Not reused as code; only the naming `VLLM_HETERO_MOE*` and `vllm/distributed/hetero_moe/` is kept |
| Disaggregation contract (NIXL `rocm_ipc`) | `docs/rdna2/disagg-g0-2026-11-06.md` | Evidence for same-arch IPC with disjoint visible sets and for 25 GB/s cross-root; not the per-layer transport |

## 4. Execution-model options

All options are costed at W = 3 rows (c1 with MTP2), W = 24 (c8), and a 2048-token prefill chunk; h = share of touched experts that is hub-resident.

### 4.1 Comparison

| Option | Data volume | Added latency per layer | Overlap | CUDA graph | MTP2 | Prefill | Verdict |
|---|---|---|---|---|---|---|---|
| (a) Activation shipping, hub-and-spoke (ship x + ids, return partial sums) | 6 MiB per step at W=3, 45 MiB at W=24, 91 MiB at W=48 (both directions, four servers) | about 30-45 us critical path, hidden behind hub-local shared + resident experts; model: MoE 50 us per layer at c1 | Full inside a layer (hub-local work runs while servers compute); none across layers (data dependence) | Hub: yes (plain kernels, no host involvement). Servers: per-width step graphs, device-gated wake | Target rows (3 per request) cross the link; the two draft passes are hub-local under policy `hub` | 41 MB down per layer if broadcast as a star (1.7 ms per layer), 10 MB with a V620-forwarded tree; hub chunk 0.37 s covers it | **Chosen** |
| (b) Stream cold weights into the hub (expert cache over P2P or from host) | Per layer (1 - h) x N(W) x 2.417 MiB: W=3 at h 0.88 / 0.6: 8.4 / 28 MiB, i.e. 17 / 56 ms per step on a 25 GB/s link; a 2048-token chunk touches all 512 experts: 1.2 GiB per layer, 2.3 s per chunk | 350-1170 us per layer at W=3, not hideable: routing is known only after the router of the same layer | None possible inside a layer | Refused by the expert cache (it raises unless the graph is breakable, `compressed_tensors_moe_wna16_rdna2.py:337`) | Rows triple the distinct experts (29 vs 10) and the misses | Hopeless (link time 6x the chunk compute) | Rejected. Measured under TP4 offload: 52.6-61.1 ms c1 ITL vs 22.4 resident |
| (c) One lockstep EP group over hub + V620 ranks (upstream all2all, EPLB) | Same activation volume, but all-gather + reduce-scatter across every rank | At least one RCCL collective per call (110 us at 4 KiB on 8 ranks), 96+ per step: more than 10 ms | `MK_INTERNAL_OVERLAPPED` shared expert only | Possible in principle | n/a | 10 MiB per layer per rank | Rejected: V620 ranks would also run attention/GDN (no specialization), one arch per process (section 8), EPLB lockstep (`eplb_state.py:666-693`) |
| (d) Attention/FFN disaggregation with two micro-batches ping-ponging | As (a) | As (a) per micro-batch | Hub dense of micro-batch B overlaps servers of micro-batch A | As (a) | n/a | Overlap hides the link and keeps both sides busy | Not for decode (W <= 48 is latency-bound; splitting doubles launches); **used inside (a) for prefill** (S7) |
| (e) Inverse split: V620 TP4 core, W7800 as a hot-expert accelerator | Activation round trip to the W7800 per layer | Same protocol cost as (a) | Partial | n/a | n/a | W7800 idle at prefill | Not recommended: the V620 chain still pays 96 all-reduces; expert time is 52 us of 367 us per layer at c1, so at most about 5 % at c1 (1.2 ms of 22.4) and low double digits at c8 before the link cost [INFERENCE from the model terms]. It becomes the better use of the W7800 only if the W7800 dense chain is much slower than modelled (W2 c1 above 28 ms, S1 gate) |
| (f) Cold experts on the host CPU | Rows over PCIe to host | tens of us per expert on AVX2 int4 at best | n/a | no | n/a | n/a | Rejected |
| (g) Prefill/decode disaggregation (W7800 prefill, V620 decode) | 426 MiB per rank at 16k (`disagg-g0-2026-11-06.md:31-33`) | n/a | n/a | n/a | n/a | G0a fails at 16385 tokens (`:36-39`) | Out of scope; W2 can later act as the prefill node of such a split |

### 4.2 Why activation shipping works here

- Routed-expert compute is tiny next to its weight bytes: one row through one expert is 2 x 3 x 2560 x 640 = 9.8 MFLOP against 2.4 MiB of weights, so the V620s (512 GB/s) are the right place for *capacity* and the
  link only has to carry 10 KiB per row per layer, 0.4 percent of the weight bytes a streaming design would move.
- The link round trip (about 15-20 us per layer) is smaller than the shared expert plus the hub's resident experts at c1 (about 18 + 26 us), which is the compute the hub overlaps with the remote work.
- It leaves the vLLM engine, the scheduler, KV management, MTP, the sampler and the all-reduce structure unchanged: the only new object is a quant method that computes the island's local-expert partial sum
  with part of the work executed on other devices.

### 4.3 Wire sizes by rows (four servers per hub)

| Rows T | Down per layer | Up per layer | Push us | Pull us | Per step (48 layers, both ways) |
|---|---|---|---|---|---|
| 3 | 61 KiB | 60 KiB | 8.8 | 8.8 | 6 MiB |
| 6 | 121 KiB | 120 KiB | 11.3 | 11.2 | 11 MiB |
| 12 | 243 KiB | 240 KiB | 16.2 | 16.1 | 23 MiB |
| 24 | 486 KiB | 480 KiB | 26.2 | 26.0 | 45 MiB |
| 48 | 971 KiB | 960 KiB | 46.1 | 45.6 | 91 MiB |
| 2048 | 41,440 KiB | 40,960 KiB | 1,703.7 | 1,684.0 | 3,862 MiB |

Push and pull times are 6.3 us flag latency plus bytes over 25 GB/s on the hub's one link (the four servers share it).

## 5. Topology and performance model

### 5.1 Candidates

| Name | Engines | Dense part | Routed experts | Expert servers |
|---|---|---|---|---|
| W2 | 1 | TP2 on the two W7800 (EP2) | all on the W7800 (256 per layer each, 29.4 GiB per hub) | none; the 8 V620 stay free for other engines |
| H1 | 2, one per switch | hub TP1 (whole active part, 9.2 GiB, on one W7800) | hot subset on the hub, all 512 on the island's four V620 | 4 per hub |
| H2 | 1 | TP2 across both hubs, EP2 | island k = hub k + its four V620 owns half of every layer: a hot subset on the hub, all of the half on the V620 | 8 |
| H2' | 1 | TP2 | as H2 but only one switch's four V620 serve both hubs (the far hub pushes cross-root, two flows); the other four V620 run a TP4 engine (D4) | 4 |
| A (rejected) | 1 | V620 TP4 core, W7800 as expert assist | hot subset on the W7800 | section 4.1 (e) |
| TP5 (rejected) | 1 | 1 W7800 + 4 V620 in one TP group | | TP shards are equal: the V620 would gate every dense GEMM at half the W7800's speed |

### 5.2 Memory fit (GiB)

Hub dense is the active part divided by TP (HC replicated), MTP is the W4A16 drafter, fixed = non-torch 1.6 + workspace 2.5 [INFERENCE on the 2.5], KV takes all remaining VRAM
(the solver reserves a smaller KV, section 7). V620 fixed = non-torch 1.6 + graphs and buffers 1.0 [INFERENCE on the 1.0].

| Device | Dense | MTP | Experts | Fixed | KV | KV tokens |
|---|---|---|---|---|---|---|
| H1 hub | 9.2 | 1.4 | 12.0 | 4.1 | 18.3 | 718k (26.7 KiB per token) |
| H2 hub (each) | 5.3 | 0.7 | 11.0 | 4.1 | 23.9 | 1,702k (14.7 KiB per token) |
| W2 hub (each) | 5.3 | 0.7 | 29.4 | 4.1 | 5.5 | 388k |
| H1 V620, primaries only | | | 14.6 | 2.6 | free 12.5 | none (servers hold no KV) |
| H1 V620, replicas to the 26 GiB budget | | | 25.6 | 2.6 | free 1.5 | |
| H2 V620, primaries only | | | 7.3 | 2.6 | free 19.9 | |
| H2 V620, replicas to the 26 GiB budget | | | 25.6 | 2.6 | free 1.7 | |

The hub's hot-set column is a plan input (11-12 GiB is about 97-106 experts per layer; the solver caps at 130 per layer in H1 and 115 per island half in H2). W2 is the only layout that
needs more than 12 GiB of experts on a hub, and it fits with 5.5 GiB of KV (388k tokens, enough for 16 users at 16k).

### 5.3 Model structure and assumptions

`tools/rdna/hetero_moe/perf_model.py` (stdlib only; `--sens` adds the sensitivity tables, `--json` dumps everything; appendix A). Structure:

- Step = 48 target layers + 0.3 ms (embedding, PLE, final HC) + lm_head + two draft passes (MTP layer + head each) + 0.5 ms host gap; throughput divides by the measured tokens per step (2.01 at 1k c8).
- Layer = two HC preambles + attention core (GDN/QSA mix 36:12) + router + (local experts + shared expert | hub/server MoE) + all-reduce + residual.
- Each term is bytes over the device's achieved bandwidth plus a latency chain taken from the TP4 trace (`p0-results.md:149-279`) minus its own byte time, times a calibrated latency scale
  s_l (c1 0.611, c8 0.484) chosen so that TP4 reproduces 22.38 and 43.97 ms exactly. Per-token work scales with `(phi + (1 - phi) x 4/tp) / speed`.
- Experts cost physical bytes over bandwidth: 3.67 us per expert on a hub (2.53 MB at 0.8 x 864 GB/s), 6.67 us on a V620 (380 GB/s). Distinct touched experts N(W) as in section 2.2;
  imbalance over D servers is `mean + E[max of D standard normals] x sqrt(mean x (1 - 1/D))`.
- Hub MoE layer = max(hub-local, remote). Hub-local = shared expert + n_hub x 3.67 us + fixed. Remote = push (6.3 us + bytes/25 GB/s), then wake 3 us, server chain 3.1 us,
  n_srv x 6.67 us, pull, combine 3 us. n_hub is chosen by the spill rule to equalize the two, capped by the resident touched count (hit x N).
- Prefill, one 2048-token chunk: max(hub, servers, link) x 1.10 with perfect overlap assumed; hub = dense FLOPs over (n_hubs x 0.30 x 90.4 TFLOPS) + 15 % of the expert FLOPs + TP2 all-reduce
  (0.4 ms per 10 MiB call), servers at 10 TFLOP/s all-in per V620, link = one copy of the rows per layer (V620-forwarded tree).

Validation: TP4 is reproduced by construction. Held out: TP8 c1 model 19.1 vs 19.59 ms measured (-2.5 %); TP8 c8 model 37.0 vs 41.56 (flat all-reduce) and 37.2 (hierarchical);
TP8 1k c8 throughput 301 vs 290.3 tok/s (+3.7 %). Nothing in the model is validated on a W7800.

Reading of the measured numbers: the measured ITL p50 is taken as the decode-step time (one streamed chunk per step); the tokens per step (1.84 at c1, 2.01 at 1k c8) enter only the throughput
columns. With that reading TP4's 1k c8 aggregate (262.6 tok/s) is reproduced from the step time, the measured prefill rate and one stall constant (2.14 prefill chunks per run, the prefill
chunks that block the other streams' steps; ITL p99 is 370 ms at c8).

Every unmeasured knob (defaults = central case; `Assume` in the script):

| Knob | Central | Pessimistic | Optimistic | Meaning and basis |
|---|---|---|---|---|
| `eta_h` | 0.80 | 0.65 | 0.95 | W7800 achieved share of 864 GB/s on streaming kernels (V620 reaches 0.74 of peak) |
| `kappa` | 1.0 | 1.4 | 0.8 | W7800 small-kernel chain speed relative to the V620 (port quality x clock) |
| `gamma` | 0.30 | | | share of small-kernel time that grows with per-GPU heads |
| `comp_ratio` | 1.25 | | | per-token head-parallel kernel speed, W7800 over V620 |
| `phi` | 0.60 | 0.25 | 1.0 | share of per-token work that does not shard with TP (TP8 c8 data say at least 0.7) |
| `rho_hc`, `rho_small` | 0.5 | | | bandwidth-bound share of the HC / router / shared GEMV weight-stream time |
| `hit` | 0.60 | | | share of touched experts resident on the hub (skew unmeasured; c1 barely depends on it) |
| `detect_us` | 3 | 30 | | server wake: device-gated 3 us, host-driven about 30 us |
| `eff_hub` | 0.30 | | | W7800 all-in prefill efficiency over 90.4 TFLOPS |
| `eff_v620_moe` | 10 TFLOP/s | | | V620 all-in expert-GEMM rate at prefill |
| `prefill_hub_share` | 0.15 | | | share of prefill expert FLOPs run on a hub |
| `locality` | 0.90 | | | distinct experts at W >= 24 relative to uniform (fit of the TP4 c8 FFN delta) |
| `hub_dense_scale` | 1.0 | | | 0.5 = W8A16 hub dense, 0.25 = W4A16 |
| `lmhead_remote` | off | | | vocab-parallel lm_head on the servers |
| `mtp_w4` | on | | | W4A16 MTP experts on the hub |

### 5.4 Results

Engines (calibration s_l: c1 0.611, c8 0.484, phi 0.60; step ms includes MTP2 draft passes):

| Engine | c1 step ms | c8 step ms | c16 step ms | 1k c8 tok/s | 16 users tok/s | prefill chunk s | 16k TTFT s |
|---|---|---|---|---|---|---|---|
| TP4 V620 | 22.4 | 44.0 | 61.5 | 263 | 357 | 0.69 | 5.5 |
| TP8 V620 | 19.1 | 37.0 | 47.7 | 301 | | 0.66 | 5.3 |
| W2 (2 W7800, TP2, no offload) | 24.7 | 45.7 | 65.0 | 278 | 376 | 0.49 | 3.9 |
| H1 (hub TP1 + 4 V620 per switch) | 32.0 | 52.9 | 75.3 | 232 | 313 | 0.66 | 5.3 |
| H2 (TP2 hubs + 8 V620) | 23.3 | 36.1 | 49.6 | 354 | 494 | 0.37 | 3.0 |
| H2' (TP2 hubs + 4 shared V620) | 23.9 | 38.5 | 53.3 | 336 | 468 | 0.37 | 3.0 |

Per-layer components at c1 (us):

| Engine | HC | attention core | router | shared | experts | MoE (hub + servers) | all-reduce | residual | layer |
|---|---|---|---|---|---|---|---|---|---|
| TP4 V620 | 108 | 144 | 27 | 14 | 52 | 0 | 21 | 2 | 367 |
| W2 | 92 | 178 | 25 | 18 | 57 | 0 | 36 | 2 | 408 |
| H1 | 92 | 309 | 25 | 0 | 0 | 74 | 0 | 2 | 502 |
| H2 | 92 | 178 | 25 | 0 | 0 | 50 | 36 | 2 | 384 |

Per-layer components at c8 (us): TP4 786 (HC 100, attention 267, router 22, shared 13, experts 292, all-reduce 90); W2 813; H1 917 (attention 535, MoE 276); H2 641 (HC 84, attention 313,
router 21, MoE 165, all-reduce 56). H2's MoE balance: hub-local 162 us against remote 161 us at c8; 47 against 47 at c1.

Why H1 loses: its hub carries the whole active part alone, 7.96 GiB per step is 11.5 ms of streaming at 691 GB/s, and the per-GPU attention core is 309 us (c1) against 178 us at TP2.
Why H2 wins: TP2 halves the per-hub dense bytes, the all-reduce is only 36 us per layer (two 18 us calls across roots, [INFERENCE] until measured), and the MoE term drops from
W2's 75 us (shared 18 + experts 57) to 50 us because the V620s absorb half of the expert streaming.

Scenarios (c1 / c8 step ms):

| Scenario | W2 | H1 | H2 | H2 prefill chunk s |
|---|---|---|---|---|
| Central | 24.7 / 45.7 | 32.0 / 52.9 | 23.3 / 36.1 | 0.37 |
| + lm_head on the expert servers (vocab-parallel, argmax only) | 24.7 / 45.7 | 29.1 / 49.9 | 21.9 / 34.7 | 0.37 |
| + W8A16 hub dense | 20.8 / 41.8 | 24.7 / 45.5 | 19.6 / 32.5 | 0.37 |
| + both | 20.8 / 41.8 | 23.3 / 44.0 | 18.9 / 31.8 | 0.37 |
| Pessimistic (eta 0.65, kappa 1.4, phi 0.25, host-driven wake) | 30.3 / 59.0 | 40.2 / 72.3 | 29.1 / 46.8 | 0.37 |
| Optimistic (eta 0.95, kappa 0.8, phi 1.0) | 21.4 / 37.6 | 27.7 / 41.6 | 20.5 / 30.3 | 0.37 |

Sensitivity (c1 / c8 step ms, one knob moved from the central case):

| Knob (low / high) | W2 c1 | W2 c8 | H1 c1 | H1 c8 | H2 c1 | H2 c8 |
|---|---|---|---|---|---|---|
| central | 24.7 | 45.7 | 32.0 | 52.9 | 23.3 | 36.1 |
| `eta_h` 0.65 / 0.95 | 27.1 / 23.0 | 51.7 / 41.6 | 35.5 / 29.6 | 57.7 / 49.4 | 25.2 / 22.0 | 38.8 / 34.2 |
| `kappa` 0.8 / 1.4 | 23.1 / 27.8 | 43.2 / 50.7 | 30.1 / 35.8 | 49.4 / 59.8 | 21.8 / 26.3 | 33.6 / 41.0 |
| `gamma` 0.15 / 0.5 | 24.2 / 25.3 | 45.3 / 46.2 | 30.6 / 34.0 | 51.7 / 54.4 | 22.8 / 24.0 | 35.7 / 36.6 |
| `comp_ratio` 1.0 / 1.7 | 24.7 / 24.7 | 47.3 / 44.0 | 32.0 / 32.0 | 55.4 / 50.2 | 23.3 / 23.3 | 37.7 / 34.4 |
| `rho_hc` 0 / 1 | 25.4 / 23.9 | 46.5 / 44.9 | 32.8 / 31.3 | 53.6 / 52.1 | 24.1 / 22.5 | 36.9 / 35.3 |
| `phi` 0.25 / 1.0 | 24.7 / 24.7 | 47.3 / 43.9 | 32.0 / 32.0 | 57.6 / 47.4 | 23.3 / 23.3 | 37.7 / 34.3 |
| `hit` 0.35 / 0.88 | 24.7 / 24.7 | 45.7 / 45.7 | 32.1 / 32.1 | 53.2 / 52.9 | 23.5 / 23.3 | 36.8 / 36.1 |
| `detect_us` 30 / 3 | 24.7 / 24.7 | 45.7 / 45.7 | 32.9 / 32.0 | 53.7 / 52.9 | 24.4 / 23.3 | 37.0 / 36.1 |
| `hub_dense_scale` 0.5 / 0.25 | 20.8 / 18.9 | 41.8 / 39.9 | 24.7 / 21.0 | 45.5 / 41.9 | 19.6 / 17.8 | 32.5 / 30.6 |

Reading it: the dominant uncertainties are `eta_h` and `kappa` (the W7800 port), then the dense precision; the routing skew (`hit`) hardly matters at c1 and little at c8, because every expert
also lives on a V620 and the hub only computes what balances the servers. The expert-server wake mode costs 1.1 ms at c1 if host-driven (24.4 vs 23.3), which is why S5 moves it into the graph.

Prefill chunk time by W7800 all-in efficiency (TP4 measured 0.69 s):

| `eff_hub` | W2 chunk s | H1 chunk s | H2 chunk s |
|---|---|---|---|
| 0.2 (18 TFLOPS) | 0.72 | 0.99 | 0.54 |
| 0.3 (27 TFLOPS) | 0.49 | 0.66 | 0.37 |
| 0.4 (36 TFLOPS) | 0.38 | 0.50 | 0.29 |
| 0.5 (45 TFLOPS) | 0.31 | 0.40 | 0.24 |

H2 beats TP4's measured chunk time down to `eff_hub` 0.15 (13.8 TFLOPS all-in), W2 down to 0.21 and H1 only down to 0.29 (model). At 0.30 the H2 chunk is hub-bound: dense 0.274 s, hub total 0.339 s
(dense + 15 % of the expert FLOPs + all-reduce), servers 0.103 s, star link 0.081 s, tree link 0.020 s, times the 1.10 overhead factor = 0.373 s.

### 5.5 Machine-level view (all ten GPUs)

Rows marked M are measured (`tp8-pp2-flashnext-2026-11-06.md`), P are model, M+P are a measured V620 engine added to a modelled W7800 engine. `users` is the concurrency of the 1k/512 workload.

| Deployment | GPUs | Users | c1 ITL ms | Aggregate 1k/512 tok/s | Aggregate prefill tok/s | Best 16k TTFT s | Basis |
|---|---|---|---|---|---|---|---|
| D0a 2 x TP4 (today) | 8 V620 | 16 | 22.4 | 525 | 5,932 | 5.5 | M |
| D0b TP8 (today) | 8 V620 | 8 | 19.6 | 290 | 3,100 | 5.3 | M |
| D0c TP4 x PP2 (today) | 8 V620 | 8 | 23.1 | 273 | 4,823 | 3.4 | M |
| D1 2 x H1 | 2 W7800 + 8 V620 | 16 | 32.0 | 463 | 6,195 | 5.3 | P |
| D2 H2 (single engine) | 2 W7800 + 8 V620 | 8 | 23.3 | 354 | 5,493 | 3.0 | P |
| D2b H2, 16 users on the one engine | 2 W7800 + 8 V620 | 16 | 23.3 | 494 | 5,493 | 3.0 | P |
| D3 W2 + 2 x TP4 (engine split, no offload) | 2 W7800 + 8 V620 | 24 | 22.4 | 803 | 10,108 | 3.9 | M+P |
| D4 H2' + 1 x TP4 | 2 W7800 + 8 V620 | 16 | 23.9 | 599 | 8,459 | 3.0 | M+P |

D3 needs no new transport: only gfx1100 support (S1) and a request router in front of three engines. It dominates on aggregate throughput and prefill; D2/D2b win per-stream latency at
c8 and above (step 36.1 vs 44.0 ms), TTFT (3.0 vs 3.9 s for the W2 engine and 5.5 s for TP4) and KV capacity (1.7M tokens against 388k on W2). Hetero also generalizes to models whose
experts exceed what two W7800 hold (this model's 58 GiB of experts fit in 2 x 44.98 GiB, so today it is not needed for capacity).

### 5.6 Recommendation

1. Build **W2 first** (S1): it is required by every other layout (the hub is a W7800 running the dense chain), it is a product on its own, and its measured c1/c8/prefill numbers replace the
   model's W7800 assumptions. Gate: the decision rule in section 10 (S1).
2. If the objective is per-stream latency and TTFT on the single engine, build **H2** on top (S3 to S8). Decision rule from the model: H2 c1 is about W2 c1 minus 1.4 ms and H2 c8 is
   about W2 c8 minus 9.6 ms; H2 matches TP4's c1 when W2 c1 is at most 23.8 ms and stays within +15 % when W2 c1 is at most 27.2 ms. Above that, stop and use D3 or option (e).
3. If the objective is aggregate throughput, deploy **D3** and do not build the hetero protocol.
4. Do not build H1, TP5 or the inverse split A.

## 6. Stack design (H2; H1 and W2 are degenerate cases)

### 6.1 Processes and device visibility

```text
      engine (vLLM EngineCore) - hub worker 0 (W7800 HIP 0) <== TP2 all-reduce ==> hub worker 1 (W7800 HIP 5)
                                      |  P2P stores + flags (intra-switch)             |
            hx-srv 0..3 (V620 HIP 1-4, island 0, switch c0)        hx-srv 4..7 (V620 HIP 6-9, island 1, switch 00)
```

- **Hub workers** are the ordinary vLLM TP2 workers on gfx1100 (`PCI_BUSES="cb 07"`, `tools/rdna/serve_rdna.sh:226-240`). Hub k is EP rank k; the island partition is the EP placement
  (`rdna_moe_placement.py`, 256 experts per rank).
- **Expert servers** are eight separate processes (`python -m vllm.distributed.hetero_moe.server`), one per V620, started by the launcher before the engine and torn down with it.
  They are outside vLLM's `_EP` group by construction (no collectives, no EPLB).
- **Visibility rule.** Preferred: every process sees only its own device(s) (a server sees its V620; the hub workers see the two W7800), which keeps `_GCN_ARCH` (`vllm/platforms/rocm.py:225-260`,
  resolved from device 0) correct and makes a wrong-arch kernel launch impossible. IPC across disjoint visible sets is measured for same-arch cross-root pairs (`disagg-g0-2026-11-06.md:99-101`);
  the cross-arch case and kernel stores into an IPC-mapped buffer of an invisible device are [INFERENCE] and are test T1. Fallback: shared visible sets with the process's own role device first
  (servers: own V620, then the island's hub; hubs: both W7800, then the island's V620).
- **Handshake.** Servers allocate their inbox with `hipMalloc` (never host-pinned, never expandable segments: `flash-next-ple-ipc.md`, `rdna-ar-wedge-2026-11-04.md`) and publish `hipIpcMemHandle_t`
  files under `/dev/shm/hx_<plan_id>/`; hub workers allocate their reply buffers, publish handles, and every side imports the peer's. Liveness both ways by `pidfd`: a dead peer aborts the engine
  (hub) or the server loop (server) within seconds. A boot self-test (known-answer MoE per server, per-role `gcnArchName` assertion) runs before the engine reports ready.

### 6.2 Hub integration (`HeteroMoEMethod`)

The insertion point is the experts' quant method, below `Qwen4ExpSparseMoeBlock` (`amd/model.py:175-192`; `self.mlp(block_input)` at `:383,387`): routing, shared expert placement, the TP/EP
all-reduce and the EP2 island split all stay as they are; only what an EP rank does for "its local experts" changes. A new quant method `HeteroMoEMethod`
(`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_hetero.py`, selected when `VLLM_HETERO_MOE=1` and a plan is loaded)
wraps the gfx1100 W4A16 MoE method (S1/S2) and implements, after `select_experts` (`moe_runner.py:630`):

```text
apply(layer, x[T,2560] fp16, topk_w[T,10] fp32, topk_ids[T,10] int32):
  1. hx_dispatch(x, topk_ids, topk_w, tables, ctx)        # push to the island's 4 servers; returns hub_ids[T,10] (-1 where the pair runs remotely)
  2. shared expert, order MK_INTERNAL_OVERLAPPED, same stream   # overlaps the servers (aux stream is off under capture on gfx10x, shared_experts.py:61-66)
  3. y_hub = hub W4A16 fused MoE(x, hub_ids, topk_w, expert_map=slot_map)   # resident experts only, output_topk reduce, fp16 partial
  4. out = hx_combine(y_hub, ctx)                          # wait on 4 flags, fixed-order fp32 sum, fp16
  return out        # the island's partial sum; the runner's TP/EP all-reduce adds the other island (and the shared expert's partial)
```

- **Weights.** `create_weights` allocates a **slot pool** of `n_slots` experts per layer (not `local_num_experts`): `n_slots` = the plan's hub-expert count for this layer, 256 in W2. The weight loader
  filters checkpoint experts through the plan (hub-resident ids only) and the usual W4A16 repack runs on the slot pool. A device `slot_map[512]` (slot or -1) replaces `expert_map`; the kernel is
  called with `ignore_invalid_experts=True` exactly as the expert-cache path does (`compressed_tensors_moe_wna16_rdna2.py:338-361`). Slots follow the plan's optional `layers[].hub_order`
  (descending benefit) and otherwise ascending expert id; the spill rule uses the slot order.
- **Dispatch tables** (device, built once at load, about 200 KB): per layer and expert, `hub_slot` (or -1), the ordered server list and `n_cold_copies`. The copy for row r is
  `(r + layer) % n_cold_copies`, the runtime rule of the plan format (section 9.2).
- **Spill rule.** A touched hub-resident expert stays on the hub iff its slot ordinal among the touched resident experts is below `K[bucket]`; the rest run on their V620 copy. `K` is a static
  table per capture width (default from the cost model: H2 per hub 7 at W=3, 39 at W=24, 63 at W=48; refined from `simulate` output and tuned in S5). Device-side this is a 512-bit touched bitmap
  (LDS atomics) and a prefix popcount over slot order: no host sync, capturable.
- **MTP layer** (index 48): hub-local only, no `hx` call; its experts are the island's half of the 512, all resident, W4A16 (section 6.7).
- **Rows.** Padded rows carry ids of -1 and are never dispatched. The router, `shared_expert_gate`, HC, PLE and the lm_head are untouched.
- **W2** is this method with `n_slots` = 256, no tables and no `hx` calls (the standard EP2 path). H1 is the same with `n_slots` < 512 per layer, TP1 and `island_mode: replicate`.

### 6.3 Expert server

One process per V620, no vLLM engine. `ExpertServerModel` instantiates only the experts' modules of the 48 layers with the plan's per-device expert set (primaries + replicas), loads them from
the same safetensors through the same repack as the TP4 EP path (V620 repack with synthesized qzeros), and keeps: the inbox (section 6.4), a `local_map[layer][512]` (local slot or -1, a
safety net on top of the hub's masking), per-width workspaces and graphs. Per step:

1. Host: read the step descriptor from a shared-memory ring written by the hub worker before it launches its graph (`epoch_abs`, `bucket`, `T`, mode), launch the graph for `bucket`.
   This costs the server host 30-100 us but starts about 250 us before the first layer-0 request can arrive (HC 46 + attention core 178 + router 25 us in the model, plus embedding and PLE), so it is hidden in steady state [INFERENCE until T5].
2. Device, per layer inside the graph: `hx_srv_wait` (spin on the local flag, timeout, sticky abort) -> `moe_align_block_size` -> W4A16 w13 GEMM + SwiGLU -> w2 GEMM with fused top-k reduce
   (`output_topk`, same kernels as TP4 EP) -> `hx_srv_push` (y into the hub's reply buffer, fence, flag).
3. Prefill (T above the largest decode bucket): the same chain enqueued eagerly for T rows with the W4A8 sdot4 path (`VLLM_RDNA2_W4A8_SDOT4=1`, as the production recipe does from 512 tokens);
   all 48 layers are enqueued up front, each `hx_srv_wait` blocks the stream until its request arrives, so the host is out of the loop.
4. Optional (S10): vocab-parallel lm_head shard (argmax-only for draft passes).

The server is stateless across steps apart from the epoch counter. It needs 1-2 HW queues (`GPU_MAX_HW_QUEUES` default 4 is plenty; the repo rule is at most 8 KFD queues per process, `async-stall-profile-2026-11-02.md:884-887`).

### 6.4 Wire protocol (`hx`)

Modelled on `rdna_ar` (`csrc/rocm/rdna_allreduce.cuh:4-35`): data is pushed with posted stores, never pulled (cross-root pulls run at 5.7 GB/s); flags live in the receiver's own uncached
VRAM so every wait is a local poll with no PCIe reads; the producer orders payload before flag; waits have timeouts and a sticky abort with a record.

| Object | Location | Written by | Layout |
|---|---|---|---|
| `inbox[h][s]` slot j in {0,1} | server s VRAM | hub h | `x[2048][2560]` fp16 (10 MiB), `ids[2048][10]` int16, `w[2048][10]` fp32, flag page (4 KiB); about 10.2 MiB per slot |
| `hdr[h][s]` | server s VRAM | hub h | magic, `epoch_abs` u64, `T`, `bucket`, mode, plan id hash |
| `reply[h][s]` slot j | hub h VRAM | server s | `y[2048][2560]` fp16 (10 MiB) + flag page; 81 MiB per hub for 4 servers x 2 slots |
| abort record | each side's VRAM | whichever side times out | phase, peer, seq (as the `rdna_ar` report word); read by the host with a D2H copy |

- Flag value `seq = epoch_abs x 64 + layer + 1` (64-bit, monotone, compared with the wrap-safe `rdna_ar_before`, `rdna_allreduce.cuh:137`); layer < 64 (48 target layers).
- Slot parity: layer l uses slot `l & 1`. A request for layer l+2 can only be issued after layer l+1's replies, which come after layer l's were consumed, so two slots are sufficient for one
  micro-batch in flight; the prefill pipeline (section 6.8) uses four slots (two per micro-batch).
- Producer: payload stores from all blocks, `__threadfence_system()`, per-block completion counter, the last block stores the flag with release semantics (`rdna_allreduce.cuh:298-301`).
  `hx_dispatch` writes four inboxes with one launch; the epoch comes from an in-graph device counter incremented by `hx_begin_step`, which also writes `hdr` so the servers adopt the hub's epoch (no
  host-supplied epoch inside a replayed graph).
- Consumer: spin with acquire loads and `s_sleep`; `hx_combine` waits on the four reply flags in fixed server order. Timeouts: 2 s per wait (a layer is 0.05-10 ms), then an abort record, the sticky
  abort word, and NaN outputs so the sampler/NaN check fails loudly instead of hanging (`rdna_allreduce.cuh:88-110` shows why a bare spin is not acceptable: the 2026-09 "GPU pinned at 99 %" hangs).
  A late peer is waited for only up to this bound (the `rdna_ar` 300 s engine timeout still bounds the whole engine step).
- Message size at W=3 is 15.2 KiB per server (3 x 5,180 B: x + routing): one-way 7 us measured for 16 KiB + flag (audit `:151`); at W=24, 121 KiB: about 11-12 us.

### 6.5 Decode timeline (H2, per layer, microseconds)

Central model at c1 (W=3 rows): per hub 14.6 touched experts of its 256, 8.7 of them resident, 7.2 computed locally and 7.4 remotely (about 1.85 per server, maximum about 3).

```text
hub    | HC 46 | attention core 178 | router 25 | dispatch 3 | shared 18 | 7.2 experts 26 | wait ~0 | combine 3 | AR 18 | HC 46 | ... next layer
link   |                                         |-- x, ids 8.8 -->|                                   |<-- y 8.8 --|
server |                                                   | wake 3 | align 3.1 | ~3 experts 20 |
remote path 8.8 + 3 + 3.1 + 20 + 8.8 + 3 (combine) = 46.7 us (model 46.9) against hub-local 3 + 18 + 26 = 47 us (model 47.2): balanced by the spill rule
```

At c8 (W=24): remote 160.8 us against hub-local 161.9 us (n_hub 39 of 87 touched per island); link 26 us each way (486 KiB down, 480 KiB up per layer, 4 servers). At c16 (W=48): 251-256 us
(n_hub 63.5 of 141), link 46 us each way.

### 6.6 CUDA graphs and compilation

- Production mode stays: `VLLM_USE_V2_MODEL_RUNNER=1`, `COMPILE_MODE=3`, `FULL_AND_PIECEWISE`, MTP=2; capture widths 3/6/12/24 (SEQS=8) plus 48 for 16 users. `hx_dispatch`, `hx_combine`,
  `hx_begin_step` are `torch.ops._rocm_C` custom ops with fake implementations and `mutates_args`, so they sit inside the piecewise graphs like any other kernel; nothing in them needs the host.
- Hub graph = unchanged structure; replays are launched at the usual cadence. Server graphs are captured per bucket at server boot (the same widths); they contain only device-side waits, so a graph
  launched late is harmless (it just starts waiting).
- Eager steps (prefill chunks, widths above the buckets): hub eager, servers enqueue per layer without a host round trip (6.3, item 3).
- Memory: capture does not allocate IPC buffers. They are raw `hipMalloc` allocations made at init outside the torch allocator, so they do not depend on `expandable_segments` (the PLE sidecar, which exports torch tensors, does: `ALLOC_CONF=auto`).
- Known risks: the RDNA hybrid-GDN prefill faults under capture and breaks to eager (`vllm/config/vllm.py:128-134`); the shared-expert aux stream is disabled under capture on gfx10x and must stay
  disabled on gfx1100 until proven [INFERENCE].

### 6.7 MTP2 and speculative decoding

- Target verify rows are 3 per request (1 + 2 draft tokens) and are the rows that cross the link; rejected draft rows are routed and computed like any other (ExpertStats counts them too).
- The MTP layer and its head run twice per step as drafters, hub-local (policy `hub`): its 512 experts are requantized to W4A16 with `tools/rdna/quant_mtp_experts.py` (1.2 GiB total; each hub holds its
  island's 256, 0.6 GiB). bf16 would cost 9.375 MiB per expert and add 1.8-2.4 ms per draft-pass pair at c8 on a hub (4.69 GiB total). Production TP4 keeps bf16 MTP experts; the W4A16 drafter
  is used in `flashnext-pp3-mtp2`, and no acceptance A/B between the two was found [INFERENCE: none exists].
- **Gate (T8):** mean accepted tokens per step with the W4A16 drafter is at least 97 % of the bf16 drafter on the same prompts at c1 and c8, else bf16 on the hub (and the plan's
  `--mtp-expert-bytes 9830400`).
- Under EP2 both draft passes end in the same island all-reduce as any MoE layer; no remote traffic is added.

### 6.8 Prefill

- A 2048-token chunk touches all 512 experts of a layer; the hub's resident set is therefore not a cache but a share of the FLOPs: `K[prefill]` is set so that hub and servers finish together (model:
  15 % of expert FLOPs on the hubs). Servers run the W4A8 sdot4 chain they run today; hubs run W4A16 WMMA (gfx1100 has no sdot4 path in this repo [INFERENCE: Rdna3Port to confirm]), so the two
  devices round differently by placement: a numerics risk covered by the PPL gate (section 12, R9).
- Link: a star broadcast moves 41 MB down and 41 MB up per layer (1.7 ms each way, 81 ms per chunk against a 0.37 s chunk: not binding). A V620-forwarded tree (hub -> one server -> the other three,
  replies reduced pairwise) cuts the hub link to one copy (20 ms per chunk) and is only needed if the star shows link-bound (T9).
- Two-microbatch software pipeline over layers inside the chunk: hub dense A_l(1) runs while the servers compute M_l(0); needs four inbox slots; implemented in S7 after the star works.
- Cross-root traffic in prefill is the TP2 all-reduce (10 MiB per call, about 0.4 ms, 96 calls = 38 ms per chunk), two flows only.

### 6.9 Failure, determinism and observability

- Failure model: a dead server, a lost write or a timeout turns into an abort record plus NaN output on the hub, which the engine surfaces as a failed step; the launcher restarts hubs and servers together.
  There is no degraded mode: serving a layer without a server's experts is a wrong answer, not a slow one.
- Determinism: partial sums are combined in fixed order (hub, then servers 0..3) in fp32; each server uses the existing deterministic reduction path
  (`compressed_tensors_moe_wna16_rdna2.py:567-592`); no atomics across devices. Results are reproducible for a fixed plan; any placement change (rebalancing, a different replica choice) changes
  rounding, so rebalancing is applied only between requests or at boot, never mid-stream. The concurrency probe (deterministic MoE concurrency test used in the V620 work) is a gate (S4).
- Observability: a device stats buffer per hub (per layer and step: hub-local pairs, remote pairs, spilled pairs, combine wait cycles, per-server wait cycles) drained like `rdna_expert_stats.py`
  (`tick()` once per step, no host sync in the forward) and exported through the vLLM stat logger; `VLLM_EXPERT_STATS_DIR` (ExpertStats) keeps working because the collector hooks the router output
  before dispatch (`moe_runner.py`, right after `select_experts`).

### 6.10 Launcher, environment and resources

- New recipes (S1, S4): `tools/rdna/recipes/flashnext-w2-mtp2.env` (TP=2, `PCI_BUSES="cb 07"`, EP, MTP=2, rdna_ar or RCCL) and `flashnext-hetero-h2.env` (the same plus `VLLM_HETERO_MOE=1`,
  `VLLM_HETERO_MOE_PLAN`, `VLLM_HETERO_MOE_SERVERS`), and a launcher stanza that starts the eight servers first (`serve_rdna.sh` / `rdna_launcher_common.sh`).
- `TUNABLEOP_RANKS` follows TP x PP (`rdna_launcher_common.sh:125-131`): the hub rows are per-ordinal files for ordinals 0 and 5 (`tools/rdna/prof/tunableop_missing.py:6-9`); the servers use no TunableOp.
- Process and queue budget (repo rule: at most 8 KFD queues per process, `async-stall-profile-2026-11-02.md:884-887`): hub worker = graph stream + aux + sidecar IPC + stats, well below 8; server = 1-2 streams.
- Memory: servers 2.6 GiB fixed + experts; hubs per section 5.2; host: the PLE table sharded 2 ways at TP2 (47.7 GiB pinned per hub worker) or the int4 sidecar.
- `HSA_FORCE_FINE_GRAIN_PCIE=1`, `HSA_ENABLE_SDMA=0`, `GPU_MAX_HW_QUEUES=4` as in the production recipes; the PCIe link of an idle W7800 can sit at 2.5 GT/s x1 (runtime PM,
  `pcie-iommu-audit-2026-11-06.md`): T0 checks the link is at 16 GT/s x16 under load before any measurement.

## 7. Placement policy

The solver and the statistics collector are `feat/expert-stats` (`docs/rdna2/expert-stats-2026-11-07.md`); this section fixes the policy and the constants the solver must use.

### 7.1 What a plan decides

Per layer and expert: the hub copies (0 or 1 per hub), and the V620 copies (primary plus replicas). Rules the runtime relies on:

- **Island partition (H2).** `island_mode: "partition"`: experts of each layer are split between the two islands by longest-processing-time on expected touches, 256 and 256; the island is the EP
  rank. The EP-level split is exported to `rdna_moe_placement.py` (`VLLM_RDNA_MOE_PLACEMENT`, uniform capacity per rank). `replicate` (H1) makes every island a full copy.
- **Hub hot set.** Global greedy by marginal benefit per byte (decode weight 1.0, prefill 0.05), clamped to `[--min-hub-per-layer, --max-hub-per-layer]`, MTP experts first (policy `hub`).
  Capacity would allow 27.1 GiB per hub (44.98 - 5.8 active - 12.1 reserve), but the cap of 115 experts per layer (13.0 GiB) is the balance point: with flat routing 115 of 256 residents give
  n_res = 0.45 x N = 6.6 / 39 / 63.5 touched residents at W = 3 / 24 / 48, which is the balance count n_hub = 7.2 / 39 / 63.5. Skewed routing makes n_res larger and the surplus spills to the V620 copies,
  so skew improves server balance and the replica budget, not the mean step time (model: `hit` 0.35 to 0.88 moves H2 c1 by 0.2 ms).
- **V620 primaries.** Exactly E/D experts per device per layer (64 per V620 in H2), chosen by LPT on expected distinct touches (optional co-activation swap refinement from the pairs file).
- **Replicas.** Hottest cold experts first, round by round onto the least-loaded V620 without a copy, until the 26 GiB budget is spent (up to about 3.5 copies in H2). A row's copy is
  `(row + layer) % n_cold_copies`, so dispatch is stateless and deterministic.
- **`cold_complete`** (default): every expert keeps at least one V620 copy, hub copies are an additional cache. This is what allows the spill rule.
- **No statistics:** uniform fallback stripe (hub holds `e % 4 == 0` or an evenly spaced stripe, owner `(e + e // D) % D`, no replicas); also the baseline the solver reports gains against.

### 7.2 Constants and recommended solver command

| Constant | Value | Basis |
|---|---|---|
| `--hub-vram-gib` | 44.98 | `rocm-smi --showmeminfo vram`, both W7800 |
| `--hub-active-gib` | 5.8 (TP2 hub; 9.4 for an H1 hub) | section 2.1 |
| `--hub-reserve-gib` | 12.1 | non-torch 1.6 + workspace 2.5 + 8 GiB KV |
| `--expert-budget-gib` | 26 | 29.98 - non-torch 1.6 - graphs/buffers 1.0 - margin |
| `--max-hub-per-layer` | 115 per island half (130 in H1) | balance point, 7.2 |
| `--hub-us` / `--expert-us` | 3.7 / 6.7 | 2.53 MB over 0.8 x 864 GB/s and over 380 GB/s |
| `--hub-fixed-us` | 20.7 (H2), 30.8 (H1) | model: shared expert, dispatch issue, combine |
| `--expert-fixed-us` | 3.1 | server align and launch chain |
| `--latency-us` | 13.3 per direction, device-gated; about 27 with a host-driven wake (S3) | (2 x 6.3 flag + wake 3 + chain 3.1 + dispatch 3 + combine 3) / 2 + 1 margin |
| `--gbps` | 25.0 | measured push rate |
| `--mtp-policy`, `--mtp-expert-bytes` | `hub`, 2,534,400 (9,830,400 for the bf16 drafter) | section 6.7 |

```bash
python -m tools.rdna.hetero_moe.solve --stats $D/stats.json --island-mode partition \
    --hub-active-gib 5.8 --hub-vram-gib 44.98 --hub-reserve-gib 12.1 --expert-budget-gib 26 \
    --max-hub-per-layer 115 --mtp-policy hub --mtp-expert-bytes 2534400 \
    --hub-us 3.7 --expert-us 6.7 --hub-fixed-us 20.7 --expert-fixed-us 3.1 \
    --latency-us 13.3 --gbps 25 --out $D/plan.json
python -m tools.rdna.hetero_moe.validate $D/plan.json --stats $D/stats.json
python -m tools.rdna.hetero_moe.simulate $D/plan.json --stats $D/stats.json --hub-fixed-us 20.7 --expert-fixed-us 3.1
```

`simulate` reports `hub_spilled_per_step`, from which the spill table `K[bucket]` (section 6.2) is seeded. The recorded hub reserve and the constants are [INFERENCE] until T2/T3/T6 replace them.

### 7.3 Collection, re-planning and application

1. Collect (S-1): `VLLM_EXPERT_STATS_DIR` on the V620 TP4 `flashnext-mtp2` server (HIP 6-9), about one hour of agentic traffic; the exact commands and the collector's own GPU validation list are
   in the ExpertStats document, sections 7 and 8. The router output is the same on any layout, so statistics from TP4 apply to the hetero engine.
2. Plan offline (above); the plan id is a content hash and is stamped into every step header, so a hub and a server on different plans abort at the first step instead of mixing placements.
3. Apply at boot (v1). Hot re-placement (S8): experts are re-read from the host page cache (168 GiB checkpoint; 2.4 MiB per expert at about 10-25 GB/s, 100-250 us each; a whole 11 GiB hub hot set
   in about 0.5-1.2 s), written into a shadow slot set, and the device `slot_map` is flipped between requests. Not EPLB: no collective, no all-rank step counter (`eplb_state.py:666-693`).
4. The hub collector keeps running in production (the same hook, before dispatch), so drift is detected by comparing the live hub-hit and server-load histograms with the plan's `estimate` block.

## 8. Mixed-arch practicalities

1. **One process, one arch.** Every `on_gfx*()` predicate reads a single `_GCN_ARCH` resolved once at import (`vllm/platforms/rocm.py:232-236`). With the `amdsmi` Python module importable it is the arch of amdsmi
   handle 0 (`:194-205`), and amdsmi enumerates all ten GPUs and ignores `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`: `amd-smi static --asic` on par1-cs25 (read-only, 2026-11-07) lists handle 0 = gfx1100 (the W7800) with both set to a V620,
   so every process of this tree would be classified gfx1100, V620-only ones included. The box venv has no `amdsmi` module (`find_spec` is None), so the import falls back to `torch.cuda.get_device_properties("cuda").gcnArchName`,
   the first visible device (`:216-225`), which is correct today; any environment that installs `amdsmi` breaks it. A patch is in progress in the `feat/rdna3-flashnext` worktree (seen uncommitted: raise on mixed archs so the torch path runs).
   `get_device_capability(device_id)` also ignores its argument (`:1096-1107`). A process serving both a W7800 and a V620 would take one arch's kernels for both, so hubs and servers are separate processes with their own visible set (section 6.1).
2. **Wrong-arch kernels fail silently.** One fat `_rocm_C` can hold both arches (`CMakeLists.txt:1732-1735`: one `define_extension_target` for all `VLLM_GPU_ARCHES`), but the device pass compiles
   bodies only for the matching `__gfx*__` macro (`csrc/rocm/q_gemm_rdna2.cu:41`, `moe_q_gemm_rdna2.cu:49`, `moe_q_gemm_rdna3.cu:30`): a gfx1030 entry point launched on gfx1100 runs and writes nothing
   (`build-break-2026-10-31.md:281-288`). Guard: each role asserts `gcnArchName` at start and both run a known-answer MoE self-test before the engine reports ready.
3. **Build.** The shipped build scripts are gfx1030-only (`tools/rdna/*/build_*.sh` set `PYTORCH_ROCM_ARCH=gfx1030`; `AGENTS.md:101-104`); `docker/Dockerfile.rdna:13` lists gfx1100 as well. Hub and servers
   use **one** fat build (`PYTORCH_ROCM_ARCH="gfx1030;gfx1100"`) so that only the environment differs. The box venv (`~/Apps/vllm/venv-7.14.0`, torch 2.12.0+rocm7.14.0) carries both device libraries
   (`rocm_sdk_device_gfx1030`, `rocm_sdk_device_gfx1100`) but its editable `vllm` maps to `/home/chenco_adm/vllm-rdna-mtp-mono`, not to any worktree: verify the imported tree before every run.
4. **Registration is per `.so`, target-wide:** `VLLM_ROCM_GFX1030` and `VLLM_ROCM_GFX1100` are defined for the whole target (`CMakeLists.txt:1743-1749`), `torch_bindings.cpp:99-100,299,1287-1313`
   registers both sets, so both arches' ops exist in either process; dispatch must follow the role, not the registration.
5. **Visible sets.** `serve_rdna.sh` maps `PCI_BUSES` (hex buses in rank order) to `HIP_VISIBLE_DEVICES`; an explicit `HIP_VISIBLE_DEVICES` wins (`serve_rdna.sh:170-176,226-240`). HIP 0 and 5 are the W7800;
   the default recipes (no `PCI_BUSES`) take HIP indices and must never be launched on this box without a visible set. The V620 slots are HIP 1-4 (`ce d3 d8 db`) and 6-9 (`0a 0f 14 17`).
6. **Libraries per ordinal.** TunableOp writes `tunableop_results{N}.csv` per ordinal; ranks above 3 were seeded from `r % 4` and TP8/PP2 ran untuned for rows missing at ordinals 4-7
   (`tunableop_missing.py:6-9,436-444`; `tp8-pp2-flashnext-2026-11-06.md:40`). The W7800 ordinals 0 and 5 need their own rows; `tune_missing_gemms.sh` refuses GPUs "0 1" by default and must be
   told which devices are busy (`:24-25`). hipBLASLt and rocBLAS gfx1100 data files are not verified on the box [INFERENCE].
7. **Comms.** `rdna_ar` is enabled only under `on_gfx10x()` (`cuda_communicator.py:196-200`) and has no arch guard inside; TP2 over the two W7800 needs the gate widened to the RDNA family and a two-rank
   cross-root test (RCCL fallback: 25-29 us at 8 KiB [INFERENCE for the exact figure]). `hx` itself is arch-neutral HIP (no gfx-specific intrinsics beyond `s_sleep`).
8. **IPC and allocator.** `hipIpcGetMemHandle` needs `hipMalloc` memory; `expandable_segments:True` breaks IPC (`flash-next-ple-ipc.md`). All `hx` buffers are allocated outside torch's allocator;
   the PLE sidecar rule `ALLOC_CONF=auto` already forces `expandable_segments:False` when the sidecar is on.
9. **Queues and links.** The repo budget is at most 8 KFD queues per process (`async-stall-profile-2026-11-02.md:884-887`); `GPU_MAX_HW_QUEUES=4` is required for `flashnext-mtp2` (2 oversubscribes the V620 runlist: `hwq-abba-2026-11-06.md:3,23`). The W7800 runtime-PM link
   state (2.5 GT/s x1 when idle) must be woken and verified (T0). One GTT buffer once landed on a W7800 belonging to another job (`rdna-ar-wedge-2026-11-04.md:68-73`): servers never use GTT mailboxes.
10. **Features that do not carry over.** The monokernel K1/K2 (gfx10x only, `mono_decode_rdna2.py`) and the K2 expert cache cannot run on a hub; the hub runs the module path. Servers could later use a K2-style
    persistent expert kernel (outside this spec).
11. **Weight layouts.** The checkpoint is shared; the hub loads the RDNA3 W4A16 layout (upstream `gptq_gemm_rdna3` family) and the servers the RDNA2 repack with synthesized qzeros (19,200 B per expert):
    two arch-specific `process_weights_after_loading` paths, one checkpoint reader.
12. **dtype.** fp16 activations on both roles (no bf16 dot on the V620); dense weights ship bf16 and are converted at load.

## 9. Interfaces

### 9.1 Device roles

| Role | Arch | Process | Owns | Visible set | Notes |
|---|---|---|---|---|---|
| `hub` | gfx1100 (W7800) | vLLM TP worker, rank k | the active part, hub-resident experts (slot pool), MTP layer, KV, reply buffers, sampler | both W7800 (TP group) | EP rank k = island k |
| `expert` | gfx1030 (V620) | `hx-server` | its experts (primaries + replicas), inbox, step graphs | its own V620 (preferred) | outside `_EP`; 1-2 queues |

An island is one hub plus the four V620 of its switch (`islands` in the plan). Roles map to the plan's `devices[]` by `id`; the launcher sets `VLLM_HETERO_MOE_ROLE` and `VLLM_HETERO_MOE_DEVICE_ID` per process.

### 9.2 Placement plan, `hetero-moe-placement/1`

Produced by `python -m tools.rdna.hetero_moe.solve`, checked by `validate`, consumed by the hub loader and the servers. Fields (as implemented on `origin/feat/expert-stats`):

```json
{
  "format": "hetero-moe-placement/1", "version": 1, "plan_id": "85307050c2418c2d",
  "model": {"...": "layers, experts, top_k, hidden, quant"}, "expert_bytes": 2534400,
  "devices": [
    {"id": "hub0", "role": "hub", "arch": "gfx1100", "hip": 0, "switch": "c0", "expert_budget_bytes": 29076928594},
    {"id": "exp0", "role": "expert", "arch": "gfx1030", "hip": 1, "switch": "c0", "expert_budget_bytes": 27917287424}
  ],
  "islands": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
  "cold_complete": true, "hub_mode": "replicate", "island_mode": "partition", "source": "stats",
  "stats_sha256": ["..."],
  "layers": [{"layer": 0, "kind": "target", "owners": [[5, 6], [8, 6, 7, 9], "... 512 entries"]}],
  "estimate": {"plan": {"analytic": {}, "simulated": {}}, "stripe": {}},
  "notes": {"owners": "...", "dispatch": "..."}
}
```

Semantics (frozen v1; section 4 of the ExpertStats document is the producer side):

- `devices[]` index is the global device index used in `owners`; roles are `hub` (W7800) or `expert` (V620); `islands` partition the device indices and share one role order; an island stays on one PCIe switch.
- `owners[e]` lists, for expert `e` of that layer, the hub device indices first (the hub copy), then the expert-device indices with the primary first and replicas ascending.
- `island_mode: "replicate"`: `owners` is written for island 0 and island k maps by position (every island is a full copy, one engine per switch: H1). `"partition"`: `owners` holds **global** device indices,
  every expert lives in exactly one island, each island is solved with its own budgets and is `cold_complete` per island (H2).
- Runtime dispatch: a hub copy wins (subject to the spill rule, section 6.2); otherwise the cold copy is `(row + layer) % n_cold_copies` over the expert-role owners in listed order.
- The MTP layer has index 48 and `kind: "mtp"`; under policy `hub` it carries only hub owners (each hub holds its island's half).
- Optional fields the hub reads if present and defaults otherwise: `layers[].hub_order` (hub-resident ids by descending benefit; default ascending id) and `runtime.spill_k` (`{rows_bucket: K}`; default from the cost
  model, overridable by `VLLM_HETERO_MOE_SPILL_K`). The solver does not emit them yet (ExpertStats informed); nothing depends on them for correctness.
- `plan_id` (a content hash) is verified in the step header by both sides.

### 9.3 Expert-stats records

Raw `expert-stats/1` JSONL (`meta` lines and cumulative `snapshot` lines: `counts[phase][e]` = rows whose top-k contains `e`, `tokens`, `steps`, `rows_hist`, `invalid`, `elapsed_s`) from
`vllm/model_executor/layers/fused_moe/rdna_expert_stats.py` (hook `MoERunner._apply_quant_method` right after `select_experts`, logical pre-EP ids, TP rank 0 only); converted
`hetero-moe-expert-stats/1` JSON (`stats_io convert`) with `layers[{layer, decode{counts}, prefill{counts}}]`, `step_hist.tokens_per_step`, optional `coact` (`.coact.npz` pairs). MTP-rejected rows are
counted; the MTP layer is index 48, kind `mtp`. Phase rule: a call with at most `VLLM_EXPERT_STATS_DECODE_MAX_TOKENS` (64) rows is decode.

### 9.4 Hub MoE call contract and `hx` ops

The contract the gfx1100 MoE path must satisfy (communicated to `feat/rdna3-flashnext`; same entry as `_rdna2_fused_moe`, `compressed_tensors_moe_wna16_rdna2.py:366-379`):

```text
hub_fused_moe(x[T,2560] fp16, topk_ids[T,10] int32 (-1 = masked), topk_weights[T,10] fp32,
              slot_map[512] int32 (slot or -1; device tensor, swappable between steps),
              w13_slots, w2_slots, scales ...) -> y[T,2560] fp16
  y = sum over valid (row,k) of weight * expert(slot_map[id])(x); pairs with id -1 or slot -1 contribute zero;
  no all-reduce; no atomics that change rounding between runs; graph-capturable for T <= 48; T up to 2048 eager or captured.
```

New ops (HIP, `csrc/rocm/hetero_moe/`, registered in `_rocm_C`, compiled for both arches; Python wrappers in `vllm/distributed/hetero_moe/`):

```text
hx_begin_step(ctx, T, bucket)                                   # hub: in-graph epoch++, writes hdr to the 4 inboxes
hx_dispatch(x, topk_ids, topk_w, hub_ids_out, layer, ctx)       # hub: spill rule, per-server masked ids, x + routing pushes, flags; hub_ids_out[T,10] int32
hx_combine(y_hub, out, layer, ctx)                              # hub: wait 4 flags (timeout), fixed-order fp32 sum, fp16 out
hx_srv_begin(ctx) / hx_srv_wait(ctx, layer) / hx_srv_push(ctx, y, layer)   # server side
```

`ctx` is a device-resident struct (peer pointers, flag pointers, epoch counter, spill table, dispatch tables, abort word); all shapes are static per capture width.

### 9.5 Kernel requirements

**gfx1100 hub** (needed for W2 and for the hub role; status from the RDNA3 port scouting: upstream gfx1100 kernels `gptq_gemm_rdna3`, `q_gemm_rdna3_wmma`, `moe_gptq_gemm_rdna3` exist behind `VLLM_ROCM_GFX1100`;
most fork HIP ops are gfx1030-only stubs on gfx1100; the per-op status table is in the `feat/rdna3-flashnext` documentation):

| # | Op | Regime | Slice |
|---|---|---|---|
| 1 | fp16 dense GEMV/GEMM for GDN `in_proj`/`out_proj`, QSA qkv/o + indexer, shared expert, lm_head (bf16 checkpoint converted to fp16) | M = 3-48 decode, up to 2048 prefill; TunableOp rows for ordinals 0, 5 | S1 |
| 2 | W4A16 MoE decode GEMM with slot masking, fused top-k reduce, deterministic mode | M <= 48, 10 experts per row | S1/S2 |
| 3 | W4A16 MoE prefill grouped GEMM (WMMA) | M up to 2048; model assumes 27 TFLOPS all-in | S1/S7 |
| 4 | GDN decode (recurrent) and prefill chain, causal_conv1d update and prefill | all widths; graph-capturable decode | S1 |
| 5 | QSA: indexer, sparse attention decode and prefill (or the stock attention backend if numerically equal) | 12 layers, 2 KV heads | S1 |
| 6 | HC kernels: preamble mix, combine, norm, prefill variants (66 us per preamble on a V620, 24 % of the layer) | W = 3-48, 2048 | S1 |
| 7 | Router (gate GEMV, top-k, renormalize), align/sort glue | W <= 48 and 2048 | S1 |
| 8 | Shared-expert fused ops (`rdna_ops.py:215-265` equivalent) | W <= 48 | S1 |
| 9 | PLE conv and n-gram lookup (sidecar IPC) | decode and prefill | S1 |
| 10 | lm_head GEMV + fused argmax / top-k for draft and verify | W <= 48 | S1/S6 |
| 11 | TP2 all-reduce across roots: `rdna_ar` widened to gfx1100, or RCCL | 8 KiB-120 KiB decode, 10 MiB prefill | S1 |
| 12 | `hx_*` ops (arch-neutral) | section 9.4 | S3 |
| 13 | W8A16 / W4A16 dense GEMV on gfx1100 (lever) | M <= 48 | S10 |

**V620 expert server:** existing kernels only: `moe_gptq_gemm_rdna2` (decode, M <= 48), `moe_w4a8_gemm_rdna2` with `VLLM_RDNA2_W4A8_SDOT4=1` (prefill from 512 rows), `moe_align_block_size`, the fused top-k reduce
(`output_topk`) and the deterministic sum; new: `hx_srv_*`, optional lm_head shard GEMV + argmax (S10). Shapes: M = 3, 6, 12, 24, 48 decode buckets and up to 2048 eager, local expert count up to 25.6 GiB / 2.43 MiB = 10.8k expert slots
across the 48 layers (about 224 per layer incl. replicas in H2).

### 9.6 Environment variables (to be added to `vllm/envs.py` with the implementation)

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_HETERO_MOE` | 0 | enable the hetero quant method on the hub |
| `VLLM_HETERO_MOE_PLAN` | unset | path of the placement plan JSON |
| `VLLM_HETERO_MOE_ROLE` | unset | `hub` or `expert`, set by the launcher per process |
| `VLLM_HETERO_MOE_DEVICE_ID` | unset | plan `devices[].id` of this process |
| `VLLM_HETERO_MOE_SERVERS` | unset | server buses per island, e.g. `ce,d3,d8,db;0a,0f,14,17` |
| `VLLM_HETERO_MOE_SPILL_K` | from the cost model | `bucket:K` pairs, e.g. `3:7,6:14,12:26,24:39,48:63` |
| `VLLM_HETERO_MOE_WAIT_MS` | 2000 | device wait timeout before an abort record |
| `VLLM_HETERO_MOE_STATS` | 0 | per-layer dispatch/combine counters |
| `VLLM_EXPERT_STATS_*` | off | routing collector (ExpertStats) |

## 10. Slices and acceptance gates

Owners: R3 = branch `feat/rdna3-flashnext`, ES = branch `feat/expert-stats`, HM = branch `feat/hetero-moe` and its follow-up implementers. A slice is done when its gate passes on the box; gate numbers cite
the tests of section 11. CPU-only parts can start before the W7800 are free.

| Slice | Content | Owner | GPU needed | Gate |
|---|---|---|---|---|
| S-1 | Routing statistics on V620 TP4 `flashnext-mtp2` (about 1 h agentic), converted stats, solver plans, `simulate` | ES tools, run by the lead | V620 only (HIP 6-9) | T10: collector validation items 1-8 of the ExpertStats document; two disjoint windows give the same plan quality (held-out `simulate` hub hit within 0.05) |
| S0 | Microbenchmarks: cross-arch IPC and P2P, W7800 bandwidth, WMMA, expert GEMV | HM follow-up | 1 W7800 + 1 V620 per switch | T1, T2: handle opens and stores byte-exact; 16 KiB push + flag at most 9 us p50; W7800 streaming at least 0.70 x 864 GB/s; expert GEMV at most 5.0 us |
| S1 | **W2**: Flash-Next W4A16 + MTP2 on gfx1100, TP2 over both W7800, EP2, no V620; `flashnext-w2-mtp2.env`; TP2 all-reduce across roots | R3 | both W7800 | T4: boots in production compiled mode, probes 3/3, PPL 4.75-4.85, concurrency probe PASS, c1 at most 28 ms (go; 26 ms good), c8 at most 52 ms, chunk at most 0.60 s. Measured values replace the model's W7800 knobs; the decision rule of section 5.6 is applied |
| S2 | Hub MoE masked partial sums and swappable slot pool (contract of section 9.4) | R3 + HM | CPU first, then W7800 | CPU: partial-sum identity (hub residents + the other experts equals the full MoE; masked ids and `slot_map` swap behave; deterministic) with a CPU reference. GPU: bit-equal to the same-order masked reference, per-expert time at most 5.0 us (T2) |
| S3 | `hx` kernels and the expert server, host-driven per-layer, eager; loopback with a replaying hub | HM | 1 W7800 + 4 V620 | T5: server output byte-equal to the local TP4-EP kernel for the same ids; per-layer MoE latency at most 1.3 x model at W = 3, 24, 48; kill tests clean within 5 s; 10^6 iterations without a mismatch |
| S4 | Integration, eager: `HeteroMoEMethod`, plan loader, server launch, boot self-test, EP2 islands | HM + R3 | both W7800 + 8 V620 | T6: PPL equal to W2 within 0.02, probes 3/3, concurrency probe PASS, 1 h c8 soak without a hang; c1 eager ITL recorded |
| S5 | CUDA graphs: hub FULL_AND_PIECEWISE with `hx` ops, server step graphs with in-graph waits, spill table `K` tuned | HM | same | T7: c1 at most 26 ms, c8 at most 40 ms, MoE per layer at most 60 us at c1, no capture fault; ABBA against TP4 with fresh boots |
| S6 | MTP2 on the hub: W4A16 drafter, acceptance A/B | HM + R3 | same | T8: accepted tokens per step with W4A16 at least 97 % of bf16 at c1 and c8, else bf16 on the hub |
| S7 | Prefill: star then V620-forwarded tree, two-microbatch pipeline, servers on W4A8 sdot4, `K[prefill]` | HM | same | T9: 16k c1 TTFT at most 3.5 s, chunk at most 0.45 s, star link at most 50 % busy |
| S8 | Placement runtime: plans from real stats, replicas, partition islands, hot re-placement at idle | ES + HM | same | T11: hub hit within 0.05 of the estimate on a held-out window, server max/mean touched at most 1.35, step time at most 1.15 x `simulate` |
| S9 | Hardening: failure matrix, soak, thermals, `flashnext-hetero-h2.env`, docs | HM | same | T12: 4 h agentic soak, injected kills and timeouts surface as failed steps, no hang, no wedge marker, restart works |
| S10 | Optional levers: W8A16 (or W4A16) hub dense, vocab-parallel lm_head on the servers | R3 + HM | same | PPL inside the 4.75-4.85 band, probes 3/3; c1 gain measured (model: -3.7 ms and -1.4 ms) |

Dependencies: S1 gates S4 and the H2 go/no-go; S0 gates S3; S2 can proceed in parallel with S1; S-1 gates S8 only (S4 to S7 run on the stripe plan); S6 and S7 are independent of each other.

CPU-testable work that does not wait for any GPU: the plan loader and validator wrapper (`vllm/distributed/hetero_moe/plan.py`, reusing `tools/rdna/hetero_moe/validate.py`), a Python/numpy reference of the
`hx` protocol (slot layout, sequence arithmetic with the wrap-safe compare, timeout and abort state machine), a reference of the spill rule (bitmap + prefix popcount) against a plain loop, the dispatch/combine
reference with the partial-sum identity test over a CPU W4A16 MoE reference. (`perf_model.py` is already covered by `tests/tools/rdna/test_hetero_perf_model.py`.)

## 11. GPU test plan (when the W7800 are free)

Nothing here may run before the user frees HIP 0 and 5 (T10 needs only HIP 6-9). Rules for every test: one change per boot; fresh boot per arm and ABBA ordering for every performance claim (repo practice,
`tp8-pp2-flashnext-2026-11-06.md`); protected window as in the disagg G0 runs (quiet box, no concurrent build, KFD `evicted_ms` sampler, junction-temperature watcher `tools/rdna/tp8pp2/tempwatch.py`,
`dmesg` diff; a cell is invalid when `evicted_ms` jumps: `disagg-g0-2026-11-06.md:84-89`); clear the `rdna_ar_wedged` marker per arm (`rdna-ar-twoshot.md:74-75`); confirm the imported vllm tree
(the venv's editable path points at `vllm-rdna-mtp-mono`); never start a server on a device another job owns.

| ID | Test | Setup | Procedure | Pass | Feeds |
|---|---|---|---|---|---|
| T0 | Preflight | idle box, W7800 released | `rocm-smi --showmeminfo vram` and `--showpids`; `lspci -vv` link status of buses `cb` and `07` under a trivial load; kernel command line (`pcie_ports=native`); imported vllm path; `_GCN_ARCH` printed by one hub process (gfx1100) and one server process (gfx1030), with and without `amdsmi` importable | both W7800 44.98 GiB total and unused; link 16 GT/s x16 under load (not the 2.5 GT/s x1 runtime-PM state); path as intended; arch per role as stated | all |
| T1 | Cross-arch IPC and P2P | HIP 0 + 1 (intra-switch), 0 + 6 (cross-root) | single process: `tools/rdna/p2p_probe/p2p_probe.cpp` rebuilt with `--offload-arch=gfx1030 --offload-arch=gfx1100`, modes `lat`, `latc`, `bw`, `a2a`; then a two-process IPC probe (hub visible {0,5}, server visible {1}) in four visibility configurations (disjoint, shared, own-first both ways), kernel stores into the IPC-mapped buffer, fan-in of 4 servers into one hub | `hipIpcOpenMemHandle` works in at least one configuration (disjoint preferred); stores byte-exact over 10^6 iterations with a rotating pattern; 16 KiB + flag one-way at most 9 us p50 and 20 us p99 (model 7); 4 x 15.2 KiB fan-in at most 15 us p50; 1 MiB fan-in at least 24 GB/s aggregate; no amdgpu error; KFD queue counts as expected | G0, `PCIE_LAT_US`, risk R2 |
| T2 | W7800 microbenchmarks | HIP 0 | streaming read kernel (achieved fraction); W4A16 MoE GEMV per expert at M = 1, 3, 8 through the gfx1100 path; fp16 GEMV/GEMM at the dense shapes (rocBLAS, TunableOp); WMMA GEMM at M = 2048 on the dense and expert shapes | achieved at least 0.70 x 864 GB/s; per expert at most 5.0 us (model 3.67); WMMA all-in at least 27 TFLOPS (model), floor 13.8 TFLOPS (break-even against TP4 prefill) | `eta_h`, `--hub-us`, `eff_hub` |
| T3 | Hub dense chain per block | W2 boot (S1) | `rocprofv3` trace of a decode step at W = 3, 24, 48 and of a 2048 chunk, as in `p0-results.md`: HC preamble, GDN and QSA blocks, router, shared expert, lm_head, all-reduce | per-op table; refit `kappa`, `gamma`, `comp_ratio`; re-run `perf_model.py` with the measured values | model refit, S1 decision |
| T4 | W2 boot, quality, speed | HIP 0 + 5, TP2, EP2, MTP2 | `serve_rdna.sh RECIPE=flashnext-w2-mtp2 PCI_BUSES="cb 07"`; smoke probes, PPL harness (`w4a8-quality-2026-11-04.md`), concurrency probe; cells c1, 1k c8, 16k c1 with the `tools/rdna/tp8pp2` harness against a V620 TP4 boot (ABBA) | probes 3/3; PPL 4.75-4.85; c1 at most 28 ms; c8 at most 52 ms; chunk at most 0.60 s; 16k TTFT at most 4.5 s; TP2 all-reduce at most 30 us per call at 8 KiB | S1 gate |
| T5 | Expert server alone | 1 W7800 process replaying recorded layer calls + 4 V620 servers | recorded `x`, ids and weights from T4; compare server output with the local TP4-EP kernel; round-trip at W = 3, 24, 48, 2048; kill the hub mid-step, kill a server mid-step, wrong `plan_id`, forced timeouts | byte-exact outputs; MoE layer time at most 1.3 x model (47, 161, 256 us at W = 3, 24, 48); abort within 5 s, no wedge, buffers released, peers exit | S3 gate |
| T6 | Integrated eager | H2, `--enforce-eager` | `VLLM_HETERO_MOE=1` with the stripe plan; PPL, probes, concurrency probe, 1 h c8 soak, per-layer trace from the stats buffer | PPL within 0.02 of T4; probes 3/3; no hang; trace shows hub-local / remote / spill counts | S4 gate |
| T7 | Graph mode | H2, production compiled mode | capture widths 3/6/12/24/48; cells c1, 1k c8, 16 users; spill table `K` sweep; ABBA against TP4 | c1 at most 26 ms; c8 at most 40 ms; MoE per layer at most 60 us at c1; spill counts match `K`; no capture fault; ITL p99 not worse than TP4 | S5 gate, `--hub-fixed-us` |
| T8 | MTP acceptance | W2 is enough | W4A16 versus bf16 drafter on identical prompts at c1 and c8, at least 3 runs each, mean accepted tokens per step | W4A16 at least 97 % of bf16 | MTP policy |
| T9 | Prefill | H2 | 16k c1 TTFT; 1k c8 prefill aggregate; per-layer hub, server and link occupancy from device clock stamps in the stats buffer | TTFT at most 3.5 s; chunk at most 0.45 s; star link at most 50 % busy (else build the tree) | S7 gate, `eff_hub`, `eff_v620_moe` |
| T10 | Routing statistics | V620 TP4 (HIP 6-9), runs before any W7800 time | ExpertStats document, sections 7 and 8 | validation items 1-8; plan validates | S-1, S8 |
| T11 | Placement quality | H2 | plan from T10; run the held-out workload; compare hub hit, server max/mean touched and step time with `simulate` | hub hit within 0.05; max/mean at most 1.35; step time at most 1.15 x `simulate` | S8 gate |
| T12 | Soak and failure injection | H2 | 4 h agentic load; kill each process kind; forced timeouts; thermal and `dmesg` watchers (pattern of `tools/rdna/ar_soak/`) | no hang, no wedge marker, aborts surface as failed steps, restart works, junction temperatures inside limits | S9 gate |
| T13 | TunableOp rows for the W7800 | HIP 0, 5 | `tools/rdna/prof/tune_missing_gemms.sh` with `REFUSED_GPUS` set to the devices that must not be touched | rows for ordinals 0 and 5 present; ABBA gain recorded | S1, S5 |

Model parameters each test replaces: T1 `PCIE_LAT_US`, `PCIE_BW`; T2 `eta_h`, `eff_hub`, hub per-expert time; T3 `kappa`, `gamma`, `comp_ratio`, `rho_*`; T5/T7 `detect_us`, fixed terms; T9 `eff_hub`,
`eff_v620_moe`, `prefill_hub_share`; T10 `hit`, `locality`. After T3 and T7 the model is re-run and section 5 of this document is updated with measured rows.

## 12. Risks

| ID | Risk | Mitigation or test |
|---|---|---|
| R1 | The W7800 dense chain is slower than modelled (`kappa`, `eta_h`); most fork HIP ops are stubs on gfx1100, so port quality decides. Pessimistic H2 c1 is 29.1 ms (+30 % vs TP4) | S1 gate (W2 c1 at most 28 ms), T3 refit; fallback D3 or option (e) |
| R2 | Cross-arch IPC, or kernel stores into a buffer of an invisible device, fail or are slow | T1 first; ladder: shared visible sets; UCX `rocm_ipc` pushes (28.96 us notification against the 6.3 us flag: +22.7 us per crossing, two crossings per layer, about +2.2 ms at c1, arithmetic only); last resort `hipMemcpyPeerAsync` pulls (5.7 GB/s) |
| R3 | Spin-wait kernels wedge a queue or pin a GPU (history: `rdna-ar-wedge-2026-11-04.md`, T44b in `rdna_allreduce.cuh:88-110`) | device timeouts, sticky abort and record, VRAM-only flags, one waiting kernel per stream, T5 and T12 |
| R4 | Server wake host-driven until in-graph waits work: +1.1 ms at c1 (model), more if the host thread is descheduled | S5; pin the server host thread; `rdna_ar` already captures spin kernels in graphs on gfx10x |
| R5 | Process-global arch (amdsmi handle 0 ignores the visible set, section 8 item 1), fat-build mistakes, silent zero output from wrong-arch kernels | role assertion, known-answer self-test, disjoint visible sets, `_GCN_ARCH` check in T0 |
| R6 | KFD queue budget (rule: at most 8 per process, a ninth slowed the full-GPU decode graph by 30 %: `async-stall-profile-2026-11-02.md:884-887`) and `GPU_MAX_HW_QUEUES` interplay on the hub (graph stream, aux stream, IPC, stats) | queue census in T6; no two-lane |
| R7 | The routing skew is unmeasured (no trace exists) | Mean step time does not depend on it (section 7.1), balance and replica value do; S-1 measures; stripe plan as fallback |
| R8 | Expectation mismatch: H2 does not raise aggregate throughput (D3 does) | section 5.5 and decision D1 |
| R9 | Numerics: V620 prefill uses W4A8 sdot4, hub prefill W4A16 WMMA; rounding depends on placement; partial sums are combined over 5 devices instead of 4 ranks | PPL within 0.02 of W2, concurrency probe, fixed placement per run (section 6.9) |
| R10 | W4A16 drafter acceptance drops | T8 gate; bf16 fallback costs 4.69 GiB (2.3 per hub under partition) and 1.8-2.4 ms per draft pair at c8 |
| R11 | Memory interplay: 95.4 GiB pinned PLE table (47.7 GiB per hub worker at TP2) with IPC buffers and allocator flags; pinned-host eviction history (`pinned-host-evictions-2026-11-05.md`) | `ALLOC_CONF=auto`, int4 sidecar as fallback, `hx` buffers outside torch |
| R12 | Upstream drift of the quant-method API across fork rebases | `HeteroMoEMethod` is the only integration point; CPU reference tests |
| R13 | W7800 link power state and thermals in a server chassis [INFERENCE] | T0 link check, `tempwatch.py`, T12 |
| R14 | Hot re-placement changes rounding and complicates determinism | v1 applies plans at boot only; S8 re-places at idle |
| R15 | TP2 all-reduce across roots slower than the assumed 18 us per call [INFERENCE]: at 40 us H2 c1 gains +1.7 ms | T4 measures; widen `rdna_ar` or use RCCL |
| R16 | No monokernel on the hub: launch gaps (0.85 us per kernel on V620, `p0-results.md:237`) and graph replay behaviour on gfx1100 are unmeasured | folded into `kappa`; T3 |
| R17 | One failure domain: any of the ten GPUs dying fails the single H2 engine (two independent TP4 engines fail separately) | launcher restart; D3 as the resilient alternative |

## 13. Open decisions for the user

1. **Objective.** Per-stream latency and TTFT on one engine (H2: c8 step -18 %, TTFT 3.0 s) or aggregate throughput (D3: about 803 tok/s against 494 for H2 at 16 users, no new protocol)? Recommendation: build W2 either way;
   build H2 only if single-engine latency matters more than the roughly 60 % aggregate gap.
2. **W2 as a product now.** S1 alone yields an engine that runs next to two TP4 engines (D3) and is the baseline for everything else. Go/no-go after the T4 numbers.
3. **Hub dense precision.** W8A16 (or W4A16) for GDN/QSA/shared/lm_head on the hub: -3.7 ms c1 / -3.6 ms c8 in the model, needs a calibration set and the quality gate (PPL band, probes). Who owns the calibration?
4. **MTP experts.** W4A16 on the hub (1.2 GiB, gate: at least 97 % of the bf16 acceptance) or bf16 (4.69 GiB, +1.8-2.4 ms per draft pair at c8).
5. **Text-only v1.** The vision tower (0.84 GiB) is excluded and image requests rejected; acceptable?
6. **Hub KV reserve.** The solver reserves 8 GiB of KV per hub (12.1 GiB reserve in total); the model could give 23.9 GiB (1.7M tokens) because more hot experts bring no gain beyond the balance point. Larger KV for prefix caching and long contexts, or keep the VRAM free?
7. **Request router.** D3 needs a router in front of three engines (different speeds, prefix-cache locality). In scope?
8. **Mono K2 statistics extension** (counting ids inside the persistent kernel): not needed for this plan; only if statistics are wanted from the monokernel recipes.
9. **Schedule.** Exclusive use of both W7800 (and one V620 island for S0/S3) is required from T0 on; T10 only needs the V620 and can run first.
10. **Server lifecycle.** Launcher-managed processes (this design) or servers owned by the vLLM engine lifecycle (a larger change, but one command line)?

## Appendix A. Reproducing the model

```bash
uv run --no-project python tools/rdna/hetero_moe/perf_model.py          # engines, components, machine level, memory, wire sizes, prefill table, scenarios, solver constants
uv run --no-project python tools/rdna/hetero_moe/perf_model.py --sens   # the same plus the one-knob sensitivity table
uv run --no-project python tools/rdna/hetero_moe/perf_model.py --json    # everything machine-readable
```

The script is stdlib-only and CPU-only. Measured anchors live in its `TODAY` and `OLD_*` constants with their document references; every unmeasured knob is a field of `Assume`. To fold in W7800
measurements, edit `Assume` (or add a field) and rerun: `eta_h`, `kappa`, `gamma`, `comp_ratio`, `eff_hub`, `detect_us` are the ones tests T2, T3, T7 and T9 produce.

`tests/tools/rdna/test_hetero_perf_model.py` (11 CPU tests) pins the TP4 calibration (22.38 / 43.97 ms, 262.6 tok/s, 5.52 s), the held-out TP8 prediction, the H1/H2 conclusions of section 5, the spill balance,
the byte-table identities and the wire sizes: `uv run --no-project --with pytest python -m pytest --noconftest tests/tools/rdna/test_hetero_perf_model.py`.

## Appendix B. Source documents for the measured numbers

| Number | Document |
|---|---|
| TP4, TP8, PP2 ITL, throughput, TTFT, KV, weights per GPU | `docs/rdna2/tp8-pp2-flashnext-2026-11-06.md:424-441` |
| Decode-step trace, 380 GB/s expert streaming, HC and FFN block times | `docs/rdna/monokernel/p0-results.md:149-279` |
| PCIe push, pull, latency, fan-in, concurrent cross-root pairs | `docs/rdna2/pcie-iommu-audit-2026-11-06.md` |
| Cross-root IPC with disjoint visible sets, NIXL 25.3 GB/s, 28.96 us notification, G0a failure at 16385 tokens | `docs/rdna2/disagg-g0-2026-11-06.md` |
| Offload ITL (52.6-61.1 ms), K2 offload split | `docs/rdna2/moe-offload-topology-2026-11-02.md`, `docs/rdna2/moe-offloading-target.md` |
| `rdna_ar` one-shot and hierarchical all-reduce | `docs/rdna2/rdna-ar-hier-2026-11-06.md`, `csrc/rocm/rdna_allreduce.cuh` |
| W7800 and V620 VRAM | `rocm-smi --showmeminfo vram` on par1-cs25, 2026-11-07 |
