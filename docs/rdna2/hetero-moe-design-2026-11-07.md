# Heterogeneous MoE serving: W7800 hubs and V620 expert servers (design, 2026-11-07)

Date: 2026-11-07. Branch `feat/hetero-moe`. Box: par1-cs25 (two PEX88096 switches, each under one x16 Gen4 root: `c0` = slot A, `00` = slot B; per switch one W7800 and four V620).
Model: `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` (Qwen4Exp, `vllm/models/qwen4_exp/amd/`), W4A16 symmetric g128 routed experts, MTP=2.

Status: **design only**. No GPU, build or service was used for this document; the box was queried read-only (`rocm-smi --showmeminfo vram`, `amd-smi static --asic`, sysfs, file reads, one `find_spec("amdsmi")` check in the venv).
Every W7800 or hetero-path performance number below is an output of the analytic model `tools/rdna/hetero_moe/perf_model.py` (calibrated to measured V620 TP4 and
cross-checked against measured TP8), not a measurement. Claims without a measurement or a file:line are marked [INFERENCE].

This document owns the plan format, the runtime semantics, the interface contract (Appendix C) and the kernel requirement lists. Four sibling branches implement against it:
`feat/rdna3-flashnext` (Flash-Next W4A16 on gfx1100: `origin/feat/rdna3-flashnext @ 19c506cba6`: the port as of `30b6a47eca`, the amdsmi stand-in that `venv-7.14.0` needs (`b8b99e3b73`), the WMMA MoE prefill kernel of S1b (`b3e6129c61`, opt-in, built and checked against the untouched gfx1030 base, never run on a GPU), the G0 to G8 and G16 runs on one W7800 (11.3), the G13 runner step, the shared-expert tiles to 64 rows and the QSA and MoE decode benches (`c283fbffa6`); its document `docs/rdna2/rdna3-flashnext-port-2026-11-07.md` is cited below as the "R3 document", with the per-op table in its section 3, the GPU plan G0-G16 in its section 12 and the W7800 results in its section 16),
`feat/expert-stats` (routing statistics, placement solver: `origin/feat/expert-stats @ 58889261df`; its document is `expert-stats-2026-11-07.md`, CPU-tested tooling at `tools/rdna/hetero_moe/`, and the T10 windows, plans and simulations of the real V620 collection in `bench_results/2026-11-07_expert-stats-t10/`, section 11.4),
`feat/hetero-hx` (HxServer: the `hx` kernels, the expert server and the launcher: `origin/feat/hetero-hx @ 7361027340`, document `hetero-hx-2026-11-07.md`) and
`feat/hetero-hub` (HubMethod: plan loader, dispatch tables, `HeteroMoEMethod`, envs, runner hook, recipe: `origin/feat/hetero-hub @ b36d8cd30b`, which merges the hx, R3 and expert-stats branches).
Stage 1 GPU work is approved by the user on W7800 HIP 0 (bus `cb`) and the slot A V620; HIP 5 is not (sections 10 and 11).

## Summary and decisions

**Stage plan (user decision, 2026-11-07).**

| Stage | Hardware | Layout | Goal |
|---|---|---|---|
| 1 | island 0: W7800 HIP 0 (bus `cb`) + V620 HIP 1-4 (slot A) | H1: hub TP1 with a slot pool of hot experts, cold experts on four V620 servers over `hx`; one engine | correct first (production compiled mode, probes 3/3, PPL 4.75-4.85, MTP2, concurrency probe), then fast; gates against today's TP4 slot A in 5.4a |
| 2 | all ten GPUs (HIP 5 and slot B join) | chosen in 5.7: W2 plus a request router, the V620 as 2 x TP4 (D3, throughput) or TP8 (D5, latency); H2 only if the rule of 5.7.5 fires | the fastest use of ten GPUs per objective |

1. **Shape.** The W7800s ("hubs", gfx1100) run everything except the cold routed-expert GEMMs: HC, GDN, QSA attention, router, shared expert, lm_head, PLE, the MTP layer,
   KV and the sampler, plus a small set of hot experts. The V620s ("expert servers", gfx1030) hold all routed experts of their island and compute the cold ones.
   The hub **ships activations, never weights**: per layer it pushes the MLP input rows plus top-k ids and weights (5,180 B per row per server) into each server's
   inbox with P2P posted stores, the servers push their fp16 partial sums (5,120 B per row) back into hub VRAM, and the hub polls a flag in its own VRAM.
2. **Topology.** Stage 1 is H1: one island, hub TP1 (the whole active part on one W7800) and four expert servers. H2 is the single-engine candidate for all ten GPUs: TP2 across both hubs, EP2 across the two switches, island k = hub k + the four V620 of its switch, each island owning half of the 512
   experts of every layer; H1 is a degenerate H2 (one island, TP1) and runs the same code. As an engine H1 is slower than the TP4 slot A it replaces (a lone W7800 must stream 7.96 GiB of dense weights per step, 11.5 ms at 691 GB/s, before any compute: 32.0 against 22.4 ms at c1 in the model, parity only with both levers, 5.4a).
   The prerequisite and baseline for everything on the W7800 is W2: the same TP2 engine on the two W7800 alone, with no V620 (29.4 GiB of experts per hub).
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
   The prefill columns assume a WMMA W4A16 MoE kernel (`eff_hub` 0.30, 27 TFLOPS all-in). The RDNA3 port reports that kernel missing (section 9.5); with the scalar `moe_gptq_gemm_rdna3` at `eff_hub` 0.14 to 0.10 the
   model gives a W2 chunk of 1.0 to 1.4 s and an H2 chunk of 0.75 to 1.03 s against 0.69 s on TP4 [INFERENCE: the scalar-kernel rate is unmeasured].
4. **The hub's hot set is balance-limited, not capacity-limited.** A hub could hold 20+ GiB of experts, but a resident expert only helps while the hub and the servers
   finish together. Per hub in H2 about 7.2 of its 14.6 touched experts per layer run locally at c1, 39 of 87 at c8, 63.5 of 141 at c16 (45-49 % of the touched set; 34-39 % in H1); with the measured routing of T10 (11.4) 7.0 of 14.4, 37 of 82 and 58 of 125 (45-49 %), 36-39 % in H1.
   Every expert also lives on a V620 (`cold_complete`), so a touched hub-resident expert beyond the balance point is computed remotely ("spill").
   The rest of the hub VRAM goes to KV. The V620s hold primaries (7.3 GiB per V620 in H2) plus replicas up to the 26 GiB budget, which is what balances the servers.
5. **Servers are separate processes outside vLLM's EP group** and the protocol is the `rdna_ar` one: payload stores, fence, sequence-numbered flag in the receiver's own
   VRAM, device-side wait with timeout and sticky abort. Not NIXL/UCX (28.96 us notification), not upstream all2all/EPLB (lockstep, one arch per process), not weight
   streaming (measured 52.6 to 61.1 ms c1 ITL under the existing TP4 offload).
6. **MTP: policy `hub`.** The MTP layer and its experts live on the hubs (each hub holds its island's half). The checkpoint ships the whole drafter in BF16 (`model_mtp.safetensors` 4.86 GiB, experts 4.69 GiB, 2.34 GiB per hub at EP2);
   the design assumes the W4A16 shard that `tools/rdna/quant_mtp_experts.py <src> <dst>` derives offline on a CPU (1.2 GiB total, 0.6 GiB per hub). Gate: drafter acceptance length at least 97 % of the bf16 drafter, else bf16 on the hub.
7. **Machine-level result (stage 2, section 5.7).** On all ten GPUs the hetero engine does not beat splitting the GPUs into independent engines behind a router. At equal total users D3 = W2 on the W7800s + 2 x TP4 on the V620s gives about 803 tok/s over 24 users (822 with W8A16 on W2) against 525 for today's 2 x TP4 and about 494 for H2 at 16 users,
   and the lowest per-user ITL from four users on; D5 = W2 + TP8 on the eight V620 has the lowest ITL at two to four users (10.8 ms per token at two users against 11.9 for today's TP8). H2 with both levers is ahead of both only at one user, by 1 % in the central model (7.7 % with the W7800 per-op numbers of G7, 11.3: just short of the 8 % that would justify building it, 5.7.5) and behind from two users on.
   Neither D3 nor D5 needs a new protocol (the RDNA3 port and a request router only). Hetero wins TTFT (3.0 s against W2's 3.9 s), KV capacity, and models whose experts do not fit the W7800s (this model's experts are 58 GiB, they do fit 2 x 44.98 GiB). Section 13 lists the decisions.
8. **Order of work.** Stage 1: S-1 statistics (V620 only) and the S0-H1 microbenchmarks, S1-H1 (the RDNA3 port at the TP1 shapes on one W7800), S2 and S3 (slot pool, `hx` kernels and servers), S4-H1 (eager integration), S5-H1 (production compiled mode), S6-H1 (MTP), S10-H1 (the two levers). Stage 2: S1 = W2 (also the numerics baseline for the hub; its code is on `origin/feat/rdna3-flashnext`, its G0 to G8 and G16 have run on one W7800 and G9 to G15 wait for HIP 5)
   and S1b (the WMMA MoE prefill kernel: written and CPU-tested, never run on a GPU), the router S11 and D3/D5 bring-up S12; S4-H2 to S9 only if the rule of 5.7.5 fires. Slices and gates are in section 10, the GPU test plan in section 11 (11.1 maps the T-steps to the G-steps), the implementer interface contract in Appendix C.
9. **Not claimed.** Of the hetero stack only the per-op kernels of the RDNA3 port have run on a W7800 (G0 to G8 and G16, 11.3); `hx`, the hub method and the integrated stack have not, and the `hx` and T5 results of 11.2 are V620-only. The routing skew was measured on the V620 TP4 server (T10, 11.4: two agentic-bench windows, 79-83 % of the decode steps are 24 rows wide); the tables there treat the rows of one sequence as independent, and no other workload, width or drift is measured. Cross-arch `hipIpcOpenMemHandle`
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
holds 5.3 GiB of it (HC replicated), plus 0.6 GiB of W4A16 MTP experts (1.2 GiB over both hubs; the checkpoint's own drafter experts are bf16, 4.69 GiB, 2.34 GiB per hub). Routed experts are 86 % of the resident bytes and 29 % of the bytes a c1 step touches (section 2.2).

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
| Expert-cache slot remap: ids -> slots with `expert_map=None, ignore_invalid_experts=True` | same file `:338-361` | **Reuse the slot semantics** for the hub's swappable slot pool; the cache path itself is eager-only (`@eager_break_during_capture`, host `.tolist()`: `:329,339`), so the hub passes a device `slot_map` as `expert_map` with the global count instead (section 9.4) |
| `_rdna2_fused_moe` body; align with `ignore_invalid_experts`; fused top-k reduce `output_topk=top_k` | same file `:366-379`, `:491-497`, `:619,636` | Reuse: a masked call returns the partial sum over the experts it owns |
| `-1` expert masking | `fused_moe/experts/rdna2_w4a16_moe.py:100,131-132,154-156`; `fused_moe/moe_align_block_size.py:128-129,164-165,228-231`; `csrc/rocm/moe_q_gemm_rdna2.cu:111-116` | Reuse as is; this is what makes partial sums safe. The modular class passes `local_num_experts` together with `expert_map` (`rdna2_w4a16_moe.py:213-219`; R3 document section 7 flags that under EP it would drop global ids at or above the local count), so hub and servers use the method path above, which passes the global count (`:491-497`) |
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
| Launcher and recipes | `tools/rdna/serve_rdna.sh:170-176,226-240` (`PCI_BUSES`), `tools/rdna/rdna_launcher_common.sh:125-131`; `tools/rdna/recipes/flashnext-mtp2.env`, `flashnext-tp8-mtp2.env`, `flashnext-tp4pp2-mtp2.env`, `flashnext-pp3-mtp2.env` | Reuse; `flashnext-w2-mtp2.env` already exists on `origin/feat/rdna3-flashnext` (safe baseline, section 6.10); add `flashnext-hetero-h2.env` |
| TunableOp | `tools/rdna/prof/tunableop_missing.py`, `tools/rdna/prof/tune_missing_gemms.sh` (`REFUSED_GPUS` default "0 1": the W7800 are 0 and 5 on this box) | Per-ordinal rows needed for the W7800 |
| PLE sidecar | `docs/rdna2/flash-next-ple-ipc.md` (`ALLOC_CONF=auto`; expandable segments break ROCm IPC) | Same on the hubs; applies to every IPC buffer in `hx` |
| Pinned-RAM wedge | `docs/rdna2/rdna-ar-wedge-2026-11-04.md` | All `hx` mailboxes are device VRAM in the uncached pool (`hx_alloc`, section 6.4), never host-pinned |
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
| H1 | 1 in stage 1 (island 0), 2 as D1 (one per switch) | hub TP1 (whole active part, 9.2 GiB, on one W7800) | hot subset on the hub, all 512 on the island's four V620 | 4 per hub |
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

### 5.2a H1 on one W7800 (stage 1)

H1 is TP1: the whole active part and a slot pool of hot experts per layer on one 44.98 GiB W7800. `h1_fit` in `perf_model.py` gives the trade between slots and KV (dense 9.2 GiB, fixed 4.1 GiB; MTP 1.4 GiB as the W4A16 shard or 4.9 GiB with the checkpoint's bf16 experts):

| H1 hub (one W7800, TP1) | Slots per layer | Slot pool GiB | Dense | MTP | Fixed | KV GiB | KV tokens |
|---|---|---|---|---|---|---|---|
| 130 slots per layer (solver cap), W4A16 MTP | 130 | 14.7 | 9.2 | 1.4 | 4.1 | 15.6 | 611k |
| 130 slots per layer, bf16 MTP experts | 130 | 14.7 | 9.2 | 4.9 | 4.1 | 12.1 | 474k |
| 106 slots per layer (the 12.0 GiB row above; 718k there with a rounded per-token size) | 106 | 12.0 | 9.2 | 1.4 | 4.1 | 18.3 | 717k |
| 8 GiB KV reserve (the solver default), W4A16 MTP | 197 | 22.3 | 9.2 | 1.4 | 4.1 | 8.0 | 314k |
| 8 GiB KV reserve, bf16 MTP experts | 166 | 18.8 | 9.2 | 4.9 | 4.1 | 8.0 | 314k |

TP4 slot A has 267k KV tokens, so H1 at the solver cap has 1.8 to 2.3 times its capacity. The cap of 130 slots per layer, not memory, limits the hub's hot set (6.2); the bf16 drafter costs 3.5 GiB of KV (137k tokens) against the W4A16 shard. The full model cannot run on one W7800 (experts 58 GiB against 44.98 GiB):
the first end-to-end run is H1 with its four servers (S4-H1), and S1-H1 proves per-op parity and a layer-truncated check only (R23).

### 5.3 Model structure and assumptions

`tools/rdna/hetero_moe/perf_model.py` (stdlib only; `--sens` adds the sensitivity tables, `--json` dumps everything; appendix A). Structure:

- Step = 48 target layers + 0.3 ms (embedding, PLE, final HC) + lm_head + two draft passes (MTP layer + head each) + 0.5 ms host gap; throughput divides by the measured tokens per step (2.01 at 1k c8).
- Layer = two HC preambles + attention core (GDN/QSA mix 36:12) + router + (local experts + shared expert | hub/server MoE) + all-reduce + residual.
- Each term is bytes over the device's achieved bandwidth plus a latency chain taken from the TP4 trace (`p0-results.md:149-279`) minus its own byte time, times a calibrated latency scale
  s_l (c1 0.611, c8 0.484) chosen so that TP4 reproduces 22.38 and 43.97 ms exactly. Per-token work scales with `(phi + (1 - phi) x 4/tp) / speed`.
- Experts cost physical bytes over bandwidth: 3.67 us per expert on a hub (2.53 MB at 0.8 x 864 GB/s), 6.67 us on a V620 (380 GB/s). Distinct touched experts N(W) as in section 2.2;
  imbalance over D servers is `mean + E[max of D standard normals] x sqrt(mean x (1 - 1/D))`.
- Hub MoE layer = max(hub-local, remote). Hub-local = shared expert + n_hub x 3.67 us + fixed. Remote = push (6.3 us + bytes/25 GB/s), then wake 3 us, server chain 3.1 us,
  n_srv x 6.67 us, pull, combine 3 us. n_hub is chosen by the spill rule to equalize the two, capped by the resident touched count (hit x N; `hit` is a flat 0.60 in the central case and the width-dependent share measured by T10 with `routing="t10"`, 11.4).
- Prefill, one 2048-token chunk: max(hub, servers, link) x 1.10 with perfect overlap assumed; hub = dense FLOPs over (n_hubs x 0.30 x 90.4 TFLOPS) + 15 % of the expert FLOPs + TP2 all-reduce
  (0.4 ms per 10 MiB call), servers at 10 TFLOP/s all-in per V620, link = one copy of the rows per layer (V620-forwarded tree).

Validation: TP4 is reproduced by construction. Held out: TP8 c1 model 19.1 vs 19.59 ms measured (-2.5 %); TP8 c8 model 37.0 vs 41.56 (flat all-reduce) and 37.2 (hierarchical);
TP8 1k c8 throughput 301 vs 290.3 tok/s (+3.7 %). Nothing in the model is validated on a W7800.

Reading of the measured numbers: the measured ITL p50 is taken as the decode-step time (one streamed chunk per step); the tokens per step (1.84 at c1, 2.01 at 1k c8) enter only the throughput
columns. With that reading TP4's 1k c8 aggregate (262.6 tok/s) is reproduced from the step time, the measured prefill rate and one stall constant (2.14 prefill chunks per run, the prefill
chunks that block the other streams' steps; ITL p99 is 370 ms at c8).

Cross-check by the RDNA3 port (R3 document section 10): `tools/rdna/rdna3/perop_expectations.py` on `origin/feat/rdna3-flashnext @ 30b6a47eca` rebuilds the W2 target forward from per-op bytes and the same V620 trace anchor with the same constants
(eta 0.8, kappa 1.0, `s_l` 0.611, 691 GB/s), so it decomposes this model and is not an independent estimate (the R3 document says the same). Its sum before all-reduce, drafter and gaps is 18.5 / 33.1 / 46.9 ms at W = 3 / 24 / 48
(pessimistic 23.9 / 41.9 / 61.8, optimistic 15.3 / 27.6 / 38.2) against 18.1 / 36.7 / 54.6 ms here: equal at c1 (+2 %), lower by 10 % at W = 24 and 14 % at W = 48. The MoE rows agree (58 / 322 / 521 us per layer
against 57 / 321 / 519). The gap is the attention-core term: its per-op rows give 150 / 150 / 168 us per layer where this model, which carries the measured TP4 c8 slope in that term, has 178 / 313 / 489 (its
docstring says the width-dependent terms are listed as separate floors instead). T3 decides; if the per-op reading holds, W2 is 3.6 ms (c8) and 7.7 ms (c16) faster than tabulated, and H2 gains the same attention-core term.
Both assume the HIP kernel chain of the V620 anchors, while gfx1100 defaults to the Triton/torch route until `VLLM_RDNA3_HIP_OPS` families are enabled (section 9.5); that chain is unmeasured.

Every unmeasured knob (defaults = central case; `Assume` in the script):

| Knob | Central | Pessimistic | Optimistic | Meaning and basis |
|---|---|---|---|---|
| `eta_h` | 0.80 | 0.65 | 0.95 | W7800 achieved share of 864 GB/s on streaming kernels (V620 reaches 0.74 of peak) |
| `kappa` | 1.0 | 1.4 | 0.8 | W7800 small-kernel chain speed relative to the V620 (port quality x clock) |
| `gamma` | 0.30 | | | share of small-kernel time that grows with per-GPU heads |
| `comp_ratio` | 1.25 | | | per-token head-parallel kernel speed, W7800 over V620 |
| `phi` | 0.60 | 0.25 | 1.0 | share of per-token work that does not shard with TP (TP8 c8 data say at least 0.7) |
| `rho_hc`, `rho_small` | 0.5 | | | bandwidth-bound share of the HC / router / shared GEMV weight-stream time |
| `hit` | 0.60 | | | share of touched experts resident on the hub (central guess; T10 measured 0.57 / 0.49 / 0.44 for H1's 130 slots per layer and 0.78 / 0.73 / 0.68 for H2's 115 per hub at W = 3 / 24 / 48, 11.4) |
| `detect_us` | 3 | 30 | | server wake: device-gated 3 us, host-driven about 30 us |
| `eff_hub` | 0.30 | | | W7800 all-in prefill efficiency over 90.4 TFLOPS |
| `eff_v620_moe` | 10 TFLOP/s | | | V620 all-in expert-GEMM rate at prefill |
| `prefill_hub_share` | 0.15 | | | share of prefill expert FLOPs run on a hub |
| `locality` | 0.90 | | | distinct experts at W >= 24 relative to uniform (fit of the TP4 c8 FFN delta; T10 measured 0.85 at W = 24 and 0.80 at W = 48, i.e. 0.946 and 0.889 of the central count) |
| `routing` | `""` | | | `"t10"` replaces `locality` and `hit` by the measured tables of 11.4 (`T10_N`, `T10_HIT`) and redoes the TP4 calibration |
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

Why H1 is slower than H2 and than TP4 slot A: its hub carries the whole active part alone, 7.96 GiB per step is 11.5 ms of streaming at 691 GB/s, and the per-GPU attention core is 309 us (c1) against 178 us at TP2 (5.4a compares it with TP4 slot A; it is the stage-1 engine regardless).
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
also lives on a V620 and the hub only computes what balances the servers. The expert-server wake mode costs 1.1 ms at c1 if host-driven (24.4 vs 23.3), which is why S5 moves it into the graph. The measured routing of T10 (11.4) moves no engine row by more than 2.4 % (W2 at c16 -2.4 %, the hub engines +1.6 % at most).

Prefill chunk time by W7800 all-in efficiency (TP4 measured 0.69 s):

| `eff_hub` | W2 chunk s | H1 chunk s | H2 chunk s |
|---|---|---|---|
| 0.2 (18 TFLOPS) | 0.72 | 0.99 | 0.54 |
| 0.3 (27 TFLOPS) | 0.49 | 0.66 | 0.37 |
| 0.4 (36 TFLOPS) | 0.38 | 0.50 | 0.29 |
| 0.5 (45 TFLOPS) | 0.31 | 0.40 | 0.24 |

H2 beats TP4's measured chunk time down to `eff_hub` 0.15 (13.8 TFLOPS all-in), W2 down to 0.21 and H1 only down to 0.29 (model). At 0.30 the H2 chunk is hub-bound: dense 0.274 s, hub total 0.339 s
(dense + 15 % of the expert FLOPs + all-reduce), servers 0.103 s, star link 0.081 s, tree link 0.020 s, times the 1.10 overhead factor = 0.373 s.

With a scalar MoE prefill kernel (`eff_hub` 0.14 and 0.10, about 12.7 and 9.0 TFLOPS all-in, [INFERENCE]) the chunk is 1.01 and 1.39 s for W2, 1.42 and 1.98 s for H1 and 0.75 and 1.03 s for H2: W2 beats TP4's 0.69 s only
with a WMMA MoE kernel (S1b), while H2 with a scalar kernel at 0.14 stays within 9 % of TP4.

### 5.4a H1 against TP4 slot A (the stage-1 gates)

H1 is the stage-1 engine (W7800 HIP 0 and the four V620 of slot A), and those four V620 serve TP4 slot A today (measured: 22.4 ms per step at c1, 44.0 ms at c8, 262.6 tok/s at 1k c8, 267k KV tokens). The model says H1 as designed is slower than that engine and has 1.8 to 2.3 times its KV; it reaches parity only with both levers of S10-H1 (`h1_ladder`, one lever at a time on the central assumptions):

| H1 (stage 1) | c1 step ms | c8 step ms | c16 step ms | 1k c8 tok/s | c1 vs TP4 slot A | c8 vs TP4 slot A |
|---|---|---|---|---|---|---|
| central | 32.0 | 52.9 | 75.3 | 232 | 1.43x | 1.20x |
| + lm_head on the servers | 29.1 | 49.9 | 72.4 | 242 | 1.30x | 1.14x |
| + W8A16 hub dense | 24.7 | 45.5 | 67.9 | 259 | 1.10x | 1.03x |
| + both levers | 23.3 | 44.0 | 66.4 | 266 | 1.04x | 1.00x |
| pessimistic | 40.2 | 72.3 | 108.1 | 181 | 1.80x | 1.64x |
| optimistic | 27.7 | 41.6 | 55.6 | 277 | 1.24x | 0.95x |

Gates (`h1_gates`; T14 and T15 apply them in production compiled mode):

- green: c1 at most 36.9 ms and c8 at most 60.8 ms (central x 1.15) with the correctness gates met: go on to the levers;
- red: c1 above 40.2 ms or c8 above 72.3 ms (the pessimistic scenario): stop, trace one step (`rocprofv3`, T3 style) and fix the port before building more;
- amber: between the two: go on with T15 and report;
- parity with both levers: c1 at most 24.6 ms and c8 at most 48.4 ms (TP4 slot A x 1.10).

The levers are worth 8.7 ms at c1 and 8.9 ms at c8 (W8A16 alone 7.3 ms at c1, because the single hub streams the whole active part), and the quality of the gfx1100 port moves the c1 step by +8.2 ms (pessimistic) or -4.3 ms (optimistic). H1 is not a speed deliverable (R20): it is the smallest complete hetero engine (hub, slot pool, servers, compiled graphs, plan and statistics on one switch),
and its measured step refits `kappa`, `eta_h`, `gamma` and `detect_us`, which decide stage 2 (5.7.5).

G7 refit (the W7800 per-op measurements of 11.3; `g7_refit`): with HC, the shared expert and lm_head replaced by the measured numbers on the best routes (`VLLM_RDNA3_HIP_OPS=hc,se`) the H1 central row becomes 30.5 / 54.3 / 80.2 ms at c1 / c8 / c16 (1.36x and 1.23x TP4 slot A at c1 and c8); on the default torch routes it is 31.4 / 59.5 / 83.3 ms (c8 1.3 ms under the green limit), and with both levers on the best routes 21.9 / 44.2 / 68.8 ms (0.98x and 1.01x). The lm_head lever is worth 2.8 / 4.2 / 5.2 ms there instead of 2.9 / 3.0 / 2.9 because the measured pass grows with the rows. The gates above stay as they are (they bracket the central model); a T14 result is read against both rows. The attention core, the router and the routed experts are still modelled (R27).

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
| D5 TP8 + W2 (engine split, no offload) | 2 W7800 + 8 V620 | 16 | 19.6 | 568 | 7,276 | 3.9 | M+P |
| D4 H2' + 1 x TP4 | 2 W7800 + 8 V620 | 16 | 23.9 | 599 | 8,459 | 3.0 | M+P |

D3 and D5 need no new transport: only gfx1100 support (S1, S1b) and a request router in front of two or three engines. D3 dominates aggregate throughput and prefill; D5 (TP8 on all eight V620 plus W2) gives the lowest per-user ITL at two to four users; D2/D2b win TTFT (3.0 vs 3.9 s for the W2 engine and 5.5 s for TP4) and KV capacity (1.7M tokens against 388k on W2),
and their c8 step (36.1 vs 44.0 ms) beats a single TP4 engine, but not the engine splits at equal users (5.7). Hetero also generalizes to models whose experts exceed what two W7800 hold (this model's 58 GiB of experts fit in 2 x 44.98 GiB, so today it is not needed for capacity).

### 5.6 Recommendation

1. **Stage 1 (decided): H1 on island 0** (W7800 HIP 0 + V620 HIP 1-4), correct first and then fast, gated against TP4 slot A (5.4a, T14, T15). H1 is the smallest complete hetero engine; the slot pool, the `hx` protocol, the servers, the compiled graphs, the plan and the statistics it needs are all stage 2's if H2 is built, the gfx1100 port it needs (S1-H1) is W2's, and its measurements replace the model's W7800 knobs (`kappa`, `eta_h`, `gamma`, `detect_us`).
2. **Stage 2 (recommended): W2 first (S1, S1b), then a router (S11), with the V620 grouped for the objective: D3 (W2 + 2 x TP4) for aggregate throughput, D5 (W2 + TP8) for per-user latency at one to four users** (5.7). Both need only the RDNA3 port and a router. H2 is built only if the rule of 5.7.5 fires (refit ITL at least 8 % below the best deployment without `hx`, or TTFT and KV decide, or a model whose experts do not fit).
3. Not built: TP5 or uneven TP, the inverse split A, PP2 across the islands, V620-side attention or dense shards, D4 (5.7.4).

### 5.7 Stage 2: the fastest use of all ten GPUs

Stage 2 starts once stage 1 has measured H1 (section 10). The question is how to use both switches (2 W7800 + 8 V620) for the fastest serving of this model, for per-user latency and for aggregate throughput. Everything below comes from `perf_model.py` (`stage2`, `itl_curve`, `md_itl_scenarios`, `h2_ratio`, `dense_offload`, `pd_split_tps`; appendix A) on the assumptions of 5.3:
the TP4 and TP8 rows are measured anchors, every W7800 row is model, and a claim with no measurement behind it is marked [INFERENCE]. The "levers" are the two S10 options: vocab-parallel lm_head on the servers (hub engines only) and W8A16 hub dense (every W7800 engine, W2 included).

#### 5.7.1 Method

The candidates differ in how many engines run and where, so comparing one engine at 8 users with three engines at 24 users says nothing about speed. The comparison is at equal total users behind a router: for each total concurrency of 1, 2, 4, 8, 16 and 24 users the model adds the users one at a time to the engine whose per-user ITL after the placement is lowest (at most 16 users per engine, the 48-row capture bucket)
and reports the mean ITL over users. ITL here is ms per output token: the engine's step time at its user count divided by the accepted tokens per step (MTP2: 1.84 at one user, rising linearly to 2.01 from eight), decode only; prefill stalls are in the throughput numbers, not in ITL. Aggregate throughput is the 1k/512 workload of 5.3 anchored on the measured 262.6 tok/s of one TP4.

#### 5.7.2 Candidates

| Candidate | GPUs | Users | c1 step ms | c8 step ms | tok/s | tok/s per GPU | tok/s per user | Prefill tok/s | 16k TTFT s | Basis |
|---|---|---|---|---|---|---|---|---|---|---|
| D0a 2 x TP4 (today) | 8 | 16 | 22.4 | 44.0 | 525 | 66 | 32.8 | 5932 | 5.5 | measured |
| D0b TP8 (today) | 8 | 8 | 19.6 | 41.6 | 290 | 36 | 36.3 | 3100 | 5.3 | measured |
| D1 2 x H1 (one hub per switch) | 10 | 16 | 32.0 | 52.9 | 463 | 46 | 29.0 | 6195 | 5.3 | model |
| D1 + both levers | 10 | 16 | 23.3 | 44.0 | 531 | 53 | 33.2 | 6195 | 5.3 | model |
| P2 PP2 across the islands (hubs are the stage heads) | 10 | 16 | 32.1 | 52.9 | 463 | 46 | 29.0 | 6195 | 3.0 | model |
| H2, 16 users | 10 | 16 | 23.3 | 36.1 | 494 | 49 | 30.9 | 5493 | 3.0 | model |
| H2 + both levers, 16 users | 10 | 16 | 18.9 | 31.8 | 530 | 53 | 33.1 | 5493 | 3.0 | model |
| D3 W2 + 2 x TP4 | 10 | 24 | 22.4 | 45.7 | 803 | 80 | 33.4 | 10108 | 3.9 | measured + model |
| D3 with W8A16 on W2 | 10 | 24 | 20.8 | 44.0 | 822 | 82 | 34.3 | 10108 | 3.9 | measured + model |
| D5 TP8 + W2 | 10 | 16 | 19.6 | 45.7 | 568 | 57 | 35.5 | 7276 | 3.9 | measured + model |
| D5 with W8A16 on W2 | 10 | 16 | 19.6 | 41.8 | 588 | 59 | 36.7 | 7276 | 3.9 | measured + model |
| D4 H2' + 1 x TP4 | 10 | 16 | 23.9 | 38.5 | 599 | 60 | 37.4 | 8459 | 3.0 | measured + model |
| D3-PD (W2 prefills, 2 x TP4 decode) | 10 | 24 | 22.4 | 45.7 | 960 | 96 | 40.0 | 4176 | 3.9 | model, handoff blocked |

D0a and D0b are today's V620-only deployments (the W7800 idle). D1 is two H1 engines, one per switch; P2 is pipeline parallelism across the two islands; H2 is one TP2/EP2 engine over all ten GPUs; D3 is W2 on the W7800 plus TP4 on slot A and TP4 on slot B behind a router; D5 is W2 plus the eight V620 as one TP8 engine; D4 is H2' (TP2 hubs sharing one switch's four V620) plus one TP4; D3-PD sends every prefill to W2 and decodes on the two TP4.

Aggregate throughput is highest for D3 (803 tok/s, 822 with W8A16 on W2; today's 2 x TP4: 525), 80 to 82 tok/s per GPU against 66 today and 46 to 53 for the hetero engines. No hetero candidate beats today's V620 pair per GPU: a W7800 adds more as an independent engine (W2: 278 tok/s at 8 users) than as a hub (H2: 494 tok/s at 16 users on ten GPUs),
because a hub still pays the dense chain and the link on every layer while an engine split pays neither. D1 with both levers (531) only equals today's 2 x TP4 (525). D3-PD would reach 960 tok/s (+20 % over D3, KV handoff cost neglected [INFERENCE]), but prefill/decode handoff fails at 16k (`docs/rdna2/disagg-g0-2026-11-06.md`, G0a), so it is not a candidate.

#### 5.7.3 ITL against users

| Deployment (ITL ms per token, mean over users) | 1 user | 2 users | 4 users | 8 users | 16 users | 24 users |
|---|---|---|---|---|---|---|
| TP8 (today) | 10.4 | 11.9 | 14.5 | 18.4 | 23.7 | - |
| D0a 2 x TP4 (today) | 12.2 | 12.2 | 14.1 | 17.3 | 21.9 | 26.5 |
| D1 2 x H1 + both levers | 12.6 | 12.6 | 14.5 | 17.6 | 21.9 | 27.7 |
| H2 + both levers | 10.3 | 11.4 | 13.2 | 15.8 | 22.5 | - |
| D3 W2 + 2 x TP4 | 12.2 | 12.2 | 13.4 | 15.6 | 19.4 | 22.2 |
| D3 with W8A16 on W2 | 11.3 | 11.7 | 12.7 | 15.0 | 18.7 | 21.5 |
| D5 TP8 + W2 + W8A16 | 10.4 | 10.8 | 12.7 | 15.3 | 19.5 | 22.8 |
| D4 H2' + 1 x TP4 | 12.2 | 12.6 | 14.2 | 16.6 | 20.5 | 24.6 |

Central assumptions. At one user H2 + levers is lowest at 10.3 ms per token, 1 % below today's TP8 and D5 (10.4; the first user of D5 runs on the TP8 engine): nothing built on the W7800 improves a single stream by more than 1 % over the V620 TP8 engine, which is already at 19.6 ms per step. The TP8 rows use the model's TP8 (19.1 ms at c1; the measured 19.59 ms gives 10.7 ms per token), so the one-user rows of TP8, D5 and H2 are equal within the model's error. At two users D5 is lowest (10.8; TP8 alone 11.9, H2 + levers 11.4).
At four users D5 and D3 + W8A16 tie at 12.7. From eight users D3 + W8A16 is lowest (15.0, 18.7 and 21.5 ms at 8, 16 and 24 users: 2, 4 and 6 % below D5 and 13 to 19 % below today's 2 x TP4), while one H2 engine falls behind (4 %, 5 % and 20 % above the best at 4, 8 and 16 users) and cannot hold 24 users at all.

The W7800 port quality decides the low-concurrency end (lowest ITL per scenario, then H2 + both levers divided by the best deployment without the `hx` protocol; H2 is built only below 0.92, 5.7.5):

| W7800 assumptions | 1 user | 2 users | 4 users | 8 users | 16 users | 24 users |
|---|---|---|---|---|---|---|
| central | H2L 10.3 | D5L 10.8 | D3L 12.7 | D3L 15.0 | D3L 18.7 | D3L 21.5 |
| optimistic (eta 0.95, kappa 0.8, phi 1.0) | H2L 9.2 | H2L 10.0 | H2L 11.3 | H2L 13.3 | D1L 17.0 | D3L 20.4 |
| pessimistic (eta 0.65, kappa 1.4, phi 0.25, host-driven servers) | TP8 10.4 | TP8 11.8 | D5L 13.3 | D3L 15.9 | D3L 20.1 | D3L 23.5 |

| H2L / best without hx | 1 user | 2 users | 4 users | 8 users | 16 users | 24 users |
|---|---|---|---|---|---|---|
| central | 0.99 | 1.05 | 1.04 | 1.05 | 1.20 | - |
| optimistic | 0.93 | 0.99 | 0.97 | 0.96 | 1.06 | - |
| pessimistic | 1.25 | 1.22 | 1.28 | 1.29 | 1.51 | - |

(H2L = H2 + both levers, D3L = D3 with W8A16 on W2, D5L = D5 with W8A16 on W2, D1L = D1 + both levers.) With optimistic W7800 knobs H2 wins up to eight users, by 1 to 7 % against the best engine split; with the pessimistic ones TP8 (today) is best at one and two users and H2 is 22 to 51 % behind.

With the G7 per-op numbers (11.3, `g7_stage2`) the central rows move to: H2L ITL 9.6 / 10.8 / 12.8 / 15.8 / 23.6 ms at 1 / 2 / 4 / 8 / 16 users and H2L / best 0.92 / 1.03 / 1.05 / 1.07 / 1.26; D3 796 tok/s over 24 users (818 with W8A16 on W2) and D5 + W8A16 583 tok/s over 16, about 1 % below the rows above.

#### 5.7.4 Variants considered and rejected

- **Dense or attention shards on the V620 (TP across hub and servers).** Upper bound (`dense_offload`): moving a share f of the attention-projection weight stream of H2's hub to the servers turns each moved phase into a round trip (push, wake, chain, pull) that must overlap the hub's remaining share; it wins only at c1, by at most 1.15 ms of 23.3 ms, and nothing at c8 or with W8A16 hub dense:

| H2 hub dense chain | Step | Share f moved | us per layer | ms per step | % of the step |
|---|---|---|---|---|---|
| H2 | c1 | 0.30 | 23.9 | 1.15 | 4.9 |
| H2 | c8 | 0.00 | 0.0 | 0.00 | 0.0 |
| H2 + W8A16 hub dense | c1 | 0.00 | 0.0 | 0.00 | 0.0 |
| H2 + W8A16 hub dense | c8 | 0.00 | 0.0 | 0.00 | 0.0 |

- **PP2 across the islands** (each hub the head of a pipeline stage): decode throughput equals 2 x H1 by construction (463 tok/s, per-stream step unchanged at 32.1 ms c1) and the 16k TTFT halves to 3.0 s (4.5 chunks); no better than D1 for throughput and behind D3 and D5 for ITL. Rejected.
- **TP5 or uneven TP** (one W7800 and four V620 in one group): the attention heads (24, 2 KV heads) do not divide by five and TP shards are equal, so a mixed group runs every dense GEMM at the V620's speed. Rejected.
- **V620 hosting attention or KV for hub layers**: moves KV and GDN state across the link on every layer and is an engine split in disguise [INFERENCE, not modelled]. Rejected.
- **D4** (H2' + TP4): 599 tok/s at 16 users and needs the hetero stack; behind D3 on throughput. **D1** (2 x H1): 463 tok/s, 531 with both levers, which only equals today's 2 x TP4. **D3-PD**: blocked (5.7.2).

#### 5.7.5 Recommendation and decision rule

Stage 2 delivers W2 (S1, S1b: HIP 0 + 5, TP2, EP2, no V620) and a request router (S11), then groups the eight V620 for the objective:

- **Aggregate throughput, and any load of eight or more concurrent users: D3** = W2 + TP4 on slot A + TP4 on slot B. Model: 803 tok/s over 24 users (822 with W8A16 on W2) against 525 for today's 2 x TP4 (+53 % to +57 %), prefill 10.1k tok/s, lowest ITL from four users on. The 16k TTFT is 3.9 s on the W2 engine, so long prompts are routed there.
- **Per-user latency at one to four users: D5** = W2 + TP8 on all eight V620. The first user runs on the TP8 engine (10.4 ms per token, today's TP8), the second on W2 (10.8 ms mean at two users against 11.9 on TP8 alone), 12.7 ms at four users against 14.5; aggregate 568 to 588 tok/s over 16 users, 16k TTFT 3.9 s on W2.

D3 and D5 are the same W2 engine behind the same router; they differ in how the V620 are grouped (two TP4 engines or one TP8), which is a recipe choice and a relaunch, not new code. Both need HIP 5. Neither runs a line of `hx` code, so stage 1's H1 is justified by what it makes possible (a model whose experts do not fit the W7800s, the H2 option) and by the measurements it yields, not by this model's numbers (R22).
Unmeasured: interference between engines on the root complexes (the TP8 all-reduce and W2's TP2 all-reduce both cross them) and the thermals of ten loaded GPUs [INFERENCE]; T16 measures them.

**H2 decision rule.** After S5-H1, W2 (T4) and T14 the model is refit with the measured `kappa`, `eta_h`, `gamma`, `detect_us` and `eff_hub`, and `perf_model.py` reprints `h2_ratio`. H2 (S4-H2 onwards) is built only if (a) its ITL with both levers is at least 8 % below the best deployment that needs no `hx` protocol (`H2_GO_RATIO` 0.92) at the concurrency the user cares about,
or (b) TTFT at long context or KV capacity is the objective (H2: 3.0 s and 1.7M tokens against W2's 3.9 s and 388k), or (c) the model's experts no longer fit the W7800 pair. The model as it stands fails (a) in every scenario: the lowest ratio is 0.93 (optimistic W7800, one user), 0.99 central and 1.22 pessimistic (a CPU test pins this). With the measured W7800 per-op numbers of G7 (11.3) the central one-user ratio is 0.92 (H2 + both levers 9.6 ms per token against the model's TP8 of 10.4; 0.90 against the measured TP8 step of 19.59 ms) and 1.03 to 1.26 from two users on, while D3 and D5 move by about 1 %: the rule is still not met on the model, but one user is now at its edge, and the attention core, the largest unmeasured term, decides it (R27).

#### 5.7.6 Request routing

No router exists today: the engines are reached directly by port, and "two-lane" is an in-engine scheduler on `feat/two-lane`, not a request router (`docs/rdna2/two-lane.md`, on that branch only). D3 and D5 (and 2 x H1) need one. Requirements from the model and the engines' limits:

- Least-ITL placement: send each request to the engine whose per-user ITL after the placement is lowest, with the live load from each engine's `/metrics` (`vllm:num_requests_running`, `vllm:num_requests_waiting`); the offline form of the rule is `place_users` (5.7.1).
- Long prompts (above about 4k tokens) go to W2 while it has room: modelled chunk 0.49 s against 0.69 s on TP4 and 0.66 s on TP8 [INFERENCE until S1b].
- Session or prefix affinity: each engine has its own prefix cache, so follow-up requests of a conversation return to the engine that holds the prefix (hash of the first prompt blocks or an explicit session header).
- Health: remove an engine that fails its probe or shows a wedge marker and return it after the restart; a failed engine fails its in-flight streams (no replay: tokens would repeat).
- Streaming: pass SSE chunks through unbuffered and turn a client disconnect into an upstream abort.
- Capacity: `max_num_seqs` per engine (8 in the production recipes; 16 users per engine need the 48-row capture bucket) and different KV capacities (W2 388k tokens, each TP4 and the TP8 267k).

Recommendation (decision 7): a small asyncio proxy in `tools/rdna/` (S11), CPU-tested against fake engines (placement, affinity, health, disconnects) before T16 runs it against the real engines.

#### 5.7.7 What stage 1 must expose

For D3 and D5 stage 1 owes the port and the numbers: the gfx1100 ops proven at the TP1 shapes in S1-H1 (the TP2 shapes of W2 are the same ops, G2 to G8 and G16 at TP2 are R3's S1), the leases and benchmark harness of section 11, and the refit of the model (T3, T14); the router and the V620 groupings do not touch the hetero code.
For H2, if the rule of 5.7.5 fires, Appendix C.8 lists what the stage-1 code must keep general so that H2 is a plan, a recipe and a second hub worker instead of a rewrite: no hard-coded island, hub or TP size (the hub index is the TP rank, the island is the EP rank, `owners` carries global device indices), per-hub objects, the server opcode registry, a plan-driven launcher and in-place table updates.

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
- **Handshake.** Servers allocate their inbox in uncached VRAM (`hx_alloc`; never host-pinned, never expandable segments: `flash-next-ple-ipc.md`, `rdna-ar-wedge-2026-11-04.md`) and publish `hipIpcMemHandle_t`
  files `srv_<id>.json` under `/dev/shm/hx_<plan_id>/` (`<id>` is the plan's device id; written only after the weights are loaded, the self-test passed and the graphs are captured); hub workers allocate their reply buffers, publish `hub_<id>.json`, and every side imports the peer's (`ack_<srv>_<hub>.json`). Liveness both ways by `pidfd`: a dead peer aborts the engine
  (hub) or the server loop (server) within seconds (61 to 126 ms in the V620 loopback). A boot self-test (known-answer MoE per server, per-role `gcnArchName` assertion) runs before the engine reports ready.

### 6.2 Hub integration (`HeteroMoEMethod`)

The insertion point is the experts' quant method, below `Qwen4ExpSparseMoeBlock` (`amd/model.py:175-192`; `self.mlp(block_input)` at `:383,387`): routing, shared expert placement, the TP/EP
all-reduce and the EP2 island split all stay as they are; only what an EP rank does for "its local experts" changes. A new quant method `HeteroMoEMethod`
(`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_hetero.py`, selected when `VLLM_HETERO_MOE=1` and a plan is loaded)
wraps the gfx1100 W4A16 MoE method (S1/S2) and implements, after `select_experts` (`moe_runner.py:630`):

```text
apply(layer, x[T,2560] fp16, topk_w[T,10] fp32, topk_ids[T,10] int32):
  1. hx_dispatch(x, topk_ids, topk_w, hub_ids_out, ctx, layer)   # push to the island's 4 servers; fills hub_ids_out[T,10] (-1 where the pair runs remotely or is not owned)
  2. shared expert, order MK_INTERNAL_OVERLAPPED, same stream   # overlaps the servers (aux stream is off under capture on gfx10x, shared_experts.py:61-66)
  3. y_hub = hub W4A16 fused MoE(x, hub_ids, topk_w, expert_map=slot_map)   # resident experts only, output_topk reduce, fp16 partial
  4. out = hx_combine(y_hub, out, ctx, layer)              # wait on 4 flags, fixed-order fp32 sum, fp16 (out may alias y_hub)
  return out        # the island's partial sum; the runner's TP/EP all-reduce adds the other island (and the shared expert's partial)
```

- **Weights.** `create_weights` allocates a **slot pool** of `n_slots` experts per layer (not `local_num_experts`): `n_slots` = the plan's hub-expert count for this layer, 256 in W2. The weight loader
  filters checkpoint experts through the plan (hub-resident ids only) and the usual W4A16 repack runs on the slot pool. A device `slot_map[512]` (slot or -1) replaces `expert_map`; the kernel is
  called with `ignore_invalid_experts=True` exactly as the expert-cache path does (`compressed_tensors_moe_wna16_rdna2.py:338-361`). Slots follow the plan's optional `layers[].hub_order`
  (descending benefit) and otherwise ascending expert id; the spill rule uses the slot order.
- **Dispatch tables** (device, built once at load, about 200 KB): per layer and expert, `hub_slot` (or -1), the ordered server list and `n_cold_copies`. The copy for row r is
  `(r + layer) % n_cold_copies`, the runtime rule of the plan format (section 9.2; decision 16 proposes the primary copy only at the decode widths).
- **Spill rule.** A touched hub-resident expert stays on the hub iff its slot ordinal among the touched resident experts is below `K[bucket]`; the rest run on their V620 copy. `K` is a static
  table per capture width, seeded from the cost model (`perf_model.py spill_k`, the balance point of hub and servers; per hub at W = 3 / 6 / 12 / 24 / 48: H2 7 / 13 / 23 / 39 / 63, H1 10 / 21 / 39 / 65 / 110 because the H1 hub touches twice the experts of an H2 hub; with the measured routing of T10 H2 7 / 13 / 22 / 37 / 58 and H1 10 / 21 / 37 / 63 / 98, 11.4), refined from `simulate` output and tuned in S5. Device-side this is a 512-bit touched bitmap
  (LDS atomics) and a prefix popcount over slot order: no host sync, capturable. Prefill uses no spill in stage 1 (`K[prefill]` is tuned in S7).
- **MTP layer** (plan layer `num_target_layers + N` for the draft module `mtp.layers.N`: 96 in the T10 stats and plans; target layers are 0 to 47): hub-local only, no `hx` call; its experts are the island's half of the 512 (all 512 in H1), all resident, W4A16 or bf16 (section 6.7).
- **Rows.** Padded rows carry ids of -1 and are never dispatched. The router, `shared_expert_gate`, HC, PLE and the lm_head are untouched.
- **W2, H1 and H2 are one method with different parameters.** `apply()` returns the island's sum; the runner's TP/EP all-reduce (if any) does the rest. W2 has no tables and no `hx` calls (the standard EP2 path).

| Parameter | W2 | H1 (stage 1) | H2 (stage 2) |
|---|---|---|---|
| Hubs (W7800) | 2 | 1 | 2 |
| Islands | none | 1: HIP 0 + 1-4 | 2: HIP 0 + 1-4 and HIP 5 + 6-9 |
| `island_mode` | n/a | `replicate` | `partition` |
| Hub TP / EP | TP2 / EP2 | TP1 / EP1 | TP2 / EP2 (island = EP rank) |
| Slots per layer per hub | 256 | up to 130 (5.2a) | up to 115 |
| Expert servers | 0 | 4 | 8 |
| All-reduce | TP2, RCCL or `rdna_ar` | none | TP2 across the roots |
| `apply()` returns | the EP2 half | the complete MoE sum | the island's partial sum |
| MTP experts on the hub | 256 per hub | 512 | 256 per hub |

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
| `inbox[s]`, one per server (written by the hub of the server's island) | server s VRAM | hub | control page of 4096 B (`hdr`, `req_flag[j]`), then 2 payload slots j in {0,1}, each `x[2048][2560]` fp16 (10 MiB), `ids[2048][10]` int16, `w[2048][10]` fp32; slot stride 10,608,640 B (10.12 MiB), 21.2 MB per inbox (Appendix C.3) |
| `hdr` | first 64 B of `inbox[s]` | hub | magic, version, opcode, plan id hash, `epoch_abs` u64, `T`, `bucket` (Appendix C.3) |
| `reply`, one allocation per hub | hub VRAM | server s | control page with `reply_flag[s][j]`, then `y[s][j][2048][2560]` fp16 (10 MiB each); 83,890,176 B (80 MiB) for 4 servers x 2 slots (Appendix C.3) |
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
- gfx1100 unknowns shared with `rdna_ar` (R3 document section 8, read from the disassembly of `rdna_allreduce.hip.o` built for both arches): `__hip_atomic_*` loads and stores compile to `glc dlc` on gfx1030 but to `glc` on gfx1100, so poll visibility in practice is UNKNOWN;
  `wall_clock64` is `s_sendmsg_rtn_b64 REALTIME` on gfx1100 against `s_memrealtime`, with an unknown tick rate (`rdna_allreduce.cu:54` hard-codes 100000 ticks per ms and the abort record depends on it); it is unknown whether the uncached pool behind
  `hipExtMallocWithFlags(hipDeviceMallocUncached)` (`rdna_allreduce.cu:150`) exists on gfx1100 and whether peer stores into it are coherent; cross-root P2P was measured on V620 pairs only. Consequences for `hx`: the 2 s timeout is converted to ticks from a rate measured at init
  (device clock against the host clock, refused outside 5 % of the expected rate) instead of a constant; mailboxes and flags are always uncached VRAM with no fallback (T1 on four V620, section 11.2: a polled flag in coarse or fine-grained memory is never seen, in uncached memory always; `wall_clock64` runs at 99.98 MHz on gfx1030); T1 repeats all three on the W7800 before any other W7800 `hx` work.

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
- Memory: capture does not allocate IPC buffers. They are raw `hx_alloc` allocations (uncached VRAM) made at init outside the torch allocator, so they do not depend on `expandable_segments` (the PLE sidecar, which exports torch tensors, does: `ALLOC_CONF=auto`).
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
  15 % of expert FLOPs on the hubs). Servers run the W4A8 sdot4 chain they run today; hubs run W4A16 (the scalar `moe_gptq_gemm_rdna3` until the WMMA MoE kernel of S1b exists; gfx1100 has no sdot4 path in this repo [INFERENCE: Rdna3Port to confirm]), so the two
  devices round differently by placement: a numerics risk covered by the PPL gate (section 12, R9).
- Link: a star broadcast moves 41 MB down and 41 MB up per layer (1.7 ms each way, 81 ms per chunk against a 0.37 s chunk: not binding; at the 22.6 GB/s that the V620 loopback measured for a 1 MiB fan-in it is 1.9 ms and 89 ms, still not binding, R25). A V620-forwarded tree (hub -> one server -> the other three,
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

- Recipes. `tools/rdna/recipes/flashnext-w2-mtp2.env` (S1) already exists on `origin/feat/rdna3-flashnext`: TP=2, `PCI_BUSES="cb 07"`, EP, MTP=2, `ATTN=none` (the QSA layers bind their own backend), `KV=4026531840`, `TUNED_CONFIG=0` and `TUNABLEOP=0` (the tuned MoE configs and TunableOp rows are gfx1030 artefacts),
  `VLLM_RDNA3_HIP_OPS` empty, `VLLM_RDNA3_MOE_DETERMINISTIC=0`, `VLLM_RDNA_AR=0` (RCCL), `GPU_MAX_HW_QUEUES=4`, its own `VLLM_CACHE_ROOT`. `flashnext-hetero-h2.env` (S4, S9; ours) is that file plus `VLLM_HETERO_MOE=1`, `VLLM_HETERO_MOE_PLAN`, `VLLM_HETERO_MOE_SERVERS`,
  `VLLM_RDNA3_MOE_DETERMINISTIC=1`, the HIP families that G7 and G11 promoted (G7: `hc,se` on and `gdn` off from three sequences, 11.3) and, once G9 has passed, `ar` with `VLLM_RDNA_AR=1` and `VLLM_RDNA_AR_MAX_KB` = `VLLM_RDNA_AR_ONESHOT_KB` = 256 (c8 needs 128, c16 needs 256: section 8 item 7).
  A launcher stanza starts the servers first (four in H1, eight in H2; `serve_rdna.sh` / `rdna_launcher_common.sh`). Stage 1 uses `flashnext-hetero-h1.env` (HubMethod, S4-H1): the same keys with `TP=1`, `PCI_BUSES="cb"` for the engine, the H1 plan, four servers on the buses `ce d3 d8 db` and `KV` sized from 5.2a.
- `TUNABLEOP_RANKS` follows TP x PP (`rdna_launcher_common.sh:125-131`): the hub rows are per-ordinal files for ordinals 0 and 5 (`tools/rdna/prof/tunableop_missing.py:6-9`); the servers use no TunableOp.
- Process and queue budget (repo rule: at most 8 KFD queues per process, `async-stall-profile-2026-11-02.md:884-887`): hub worker = graph stream + aux + sidecar IPC + stats, well below 8; server = 1-2 streams.
- Memory: servers 2.6 GiB fixed + experts; hubs per section 5.2; host: the PLE table sharded 2 ways at TP2 (47.7 GiB pinned per hub worker) or the int4 sidecar; the H1 worker owns the whole 95.4 GiB, as the one worker of today's TP4 does.
- `HSA_FORCE_FINE_GRAIN_PCIE=1`, `HSA_ENABLE_SDMA=0`, `GPU_MAX_HW_QUEUES=4` as in the production recipes; the PCIe link of an idle W7800 can sit at 2.5 GT/s x1 (runtime PM,
  `pcie-iommu-audit-2026-11-06.md`): T0 checks the link is at 16 GT/s x16 under load before any measurement.

### 6.11 H1 (stage 1) specifics

- **Layout.** One engine (TP1, EP1, `island_mode: replicate`) on the W7800 HIP 0 (bus `cb`, switch `c0`) and four server processes on the V620 of slot A (HIP 1-4, buses `ce d3 d8 db`), recipe `flashnext-hetero-h1.env`, production mode (`VLLM_USE_V2_MODEL_RUNNER=1`, `COMPILE_MODE=3`, `FULL_AND_PIECEWISE`, MTP2, capture widths 3/6/12/24 and 48). The engine sees only HIP 0 and each server only its own V620 (disjoint visibility, Appendix C.2). The H1 servers are the V620 that TP4 slot A uses today, so the two engines never run together.
- **No collectives.** TP1 has no all-reduce; stage 1 needs neither RCCL nor `rdna_ar` (G9 and the section 8 item 7 gates are W2 and stage-2 items).
- **TP1 shapes** (the checkpoint's `config.json`, read on the box): `num_attention_heads` 24, `num_key_value_heads` 2, `head_dim` 256, so the QSA group size is 12 (the same as TP2; 6 at TP4) with two KV-head blocks on the one rank; GDN `linear_num_key_heads` 16 and `linear_num_value_heads` 48 (key and value dim 128), all on the one rank; `I_SE` 640 (TP2 320, TP4 160); lm_head N = 248,320, the whole vocabulary. The TP2 constants of R3's G7 benches (`I_SE` 320, 8 and 24 heads) do not apply; S1-H1 reruns G2 to G8 and G16 at these shapes.
- **Memory** (5.2a): 9.2 GiB of dense weights and 4.1 GiB fixed on a 44.98 GiB W7800; with 130 slots per layer (14.7 GiB) the KV is 12.1 GiB (474k tokens) with the checkpoint's bf16 MTP shard and 15.6 GiB (611k tokens) with the W4A16 shard, against 267k tokens on TP4 slot A.
- **Host.** The PLE n-gram table (95.4 GiB, pinned) belongs to the one worker, as in today's TP4 slot A; a TP1 worker is not expected to behave differently [INFERENCE until S4-H1 boots].
- **Plan.** The ExpertStats solver, first run without `--stats` (uniform stripe, `--stripe 4` keeps 128 experts per layer on the hub) and then with S-1 statistics:

```bash
python -m tools.rdna.hetero_moe.solve --stats $D/stats.json --islands 1 \
    --hub-hips 0 --expert-hips 1,2,3,4 --switches c0 --island-mode replicate \
    --hub-active-gib 9.4 --hub-vram-gib 44.98 --hub-reserve-gib 12.1 --expert-budget-gib 26 \
    --max-hub-per-layer 130 --mtp-policy hub --mtp-expert-bytes 9830400 \
    --hub-us 3.7 --expert-us 6.7 --hub-fixed-us 30.8 --expert-fixed-us 3.1 \
    --latency-us 13.3 --gbps 25 --out $D/plan.json
```

`--mtp-expert-bytes` is 9,830,400 while the drafter is the checkpoint's bf16 shard (S6-H1 first run) and 2,534,400 with the W4A16 shard.

## 7. Placement policy

The solver and the statistics collector are `feat/expert-stats` (`docs/rdna2/expert-stats-2026-11-07.md`); this section fixes the policy and the constants the solver must use.

### 7.1 What a plan decides

Per layer and expert: the hub copies (0 or 1 per hub), and the V620 copies (primary plus replicas). Rules the runtime relies on:

- **Island partition (H2).** `island_mode: "partition"`: experts of each layer are split between the two islands by longest-processing-time on expected touches, 256 and 256; the island is the EP
  rank. The EP-level split is exported to `rdna_moe_placement.py` (`VLLM_RDNA_MOE_PLACEMENT`, uniform capacity per rank). `replicate` (H1) makes every island a full copy.
- **Hub hot set.** Global greedy by marginal benefit per byte (decode weight 1.0, prefill 0.05), clamped to `[--min-hub-per-layer, --max-hub-per-layer]`, MTP experts first (policy `hub`).
  Capacity would allow 27.1 GiB per hub (44.98 - 5.8 active - 12.1 reserve), but the cap of 115 experts per layer (13.0 GiB) is the balance point: with flat routing 115 of 256 residents give
  n_res = 0.45 x N = 6.6 / 39 / 63.5 touched residents at W = 3 / 24 / 48, which is the balance count n_hub = 7.2 / 39 / 63.5. Skewed routing makes n_res larger and the surplus spills to the V620 copies,
  so skew improves server balance and the replica budget, not the mean step time (model: `hit` 0.35 to 0.88 moves H2 c1 by 0.2 ms). Measured (T10, 11.4): the 115 residents of an island carry 0.78 / 0.73 / 0.68 of the touched experts, n_res = 11.2 / 59.7 / 85.5 per hub against the balance counts 7.0 / 37.3 / 57.7, so the hub holds more residents than it keeps and 4 to 28 per layer spill to their V620 copies, as the spill rule intends; H1's 130 residents carry 0.57 / 0.49 / 0.44 and the resident cap binds only at 48 rows (K 98 against 109 residents).
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
   the first visible device (`:216-225`), which is correct today; any environment that installs `amdsmi` breaks it. R3's fix (`30b6a47eca`, `vllm/platforms/rocm.py:195` on its branch; R3 document sections 3 row 13 and 5) collects every amdsmi handle and raises on mixed archs, so the torch path runs and honours `HIP_VISIBLE_DEVICES`: the dependence is gone once that branch is the one built. A second fix (`b8b99e3b73`) defines stand-ins for `amdsmi_init` and the handle query when the module is missing, because `venv-7.14.0` ships no `amdsmi` and the first `get_device_name()` of an engine start (the startup plan, the Triton MoE config lookup) raised `NameError` without it (found by R3's G4).
   CPU-only runs of vLLM on the box go through `tools/rdna/rdna3/cpu_pytest.sh` (it blocks `amdsmi`, unsets `ROCR_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`, sets `HIP_VISIBLE_DEVICES=`), because importing `vllm.platforms` calls amdsmi, which can also wake idle GPUs.
   `get_device_capability(device_id)` also ignores its argument (`:1096-1107`). A process serving both a W7800 and a V620 would take one arch's kernels for both, so hubs and servers are separate processes with their own visible set (section 6.1).
2. **Wrong-arch kernels fail silently.** One fat `_rocm_C` can hold both arches (`CMakeLists.txt:1732-1735`: one `define_extension_target` for all `VLLM_GPU_ARCHES`), but the device pass compiles
   bodies only for the matching `__gfx*__` macro (`csrc/rocm/q_gemm_rdna2.cu:41`, `moe_q_gemm_rdna2.cu:49`, `moe_q_gemm_rdna3.cu:30`): a gfx1030 entry point launched on gfx1100 runs and writes nothing
   (`build-break-2026-10-31.md:281-288`). Guard: each role asserts `gcnArchName` at start and both run a known-answer MoE self-test before the engine reports ready.
3. **Build.** The shipped build scripts are gfx1030-only (`tools/rdna/*/build_*.sh` set `PYTORCH_ROCM_ARCH=gfx1030`; `AGENTS.md:101-104`); `docker/Dockerfile.rdna:13` lists gfx1100 as well. Hub and servers
   use **one** fat build (`PYTORCH_ROCM_ARCH="gfx1030;gfx1100"`) so that only the environment differs. The box venv (`~/Apps/vllm/venv-7.14.0`, torch 2.12.0+rocm7.14.0) carries both device libraries
   (`rocm_sdk_device_gfx1030`, `rocm_sdk_device_gfx1100`) but its editable `vllm` maps to `/home/chenco_adm/vllm-rdna-mtp-mono`, not to any worktree: verify the imported tree before every run.
   R3 measured the fat build at 18.7 min with `MAX_JOBS=8` on the box (116 MB `.so`, 177 `_rocm_C` ops, loads: R3 document section 4, `tools/rdna/rdna3/build_rocm_c.sh`). A gfx1100-only build of the base tree does not load (4 undefined project symbols, the registration bug R3 fixed);
   R3's gfx1100-only build (9.6 min, 44 ops) loads but lacks every op of the gfx1030-only registration block, and the HIP families raise a clear error when named. 69 of the 70 gfx1030 device objects are byte-identical to the base build (one stub churn): evidence about code objects, not a V620 run.
4. **Registration is per `.so`, target-wide:** `VLLM_ROCM_GFX1030` and `VLLM_ROCM_GFX1100` are defined for the whole target (`CMakeLists.txt:1743-1749`), `torch_bindings.cpp:99-100,299,1287-1313`
   registers both sets, so both arches' ops exist in either process; dispatch must follow the role, not the registration.
5. **Visible sets.** `serve_rdna.sh` maps `PCI_BUSES` (hex buses in rank order) to `HIP_VISIBLE_DEVICES`; an explicit `HIP_VISIBLE_DEVICES` wins (`serve_rdna.sh:170-176,226-240`). HIP 0 and 5 are the W7800;
   the default recipes (no `PCI_BUSES`) take HIP indices and must never be launched on this box without a visible set. The V620 slots are HIP 1-4 (`ce d3 d8 db`) and 6-9 (`0a 0f 14 17`).
6. **Libraries per ordinal.** TunableOp writes `tunableop_results{N}.csv` per ordinal; ranks above 3 were seeded from `r % 4` and TP8/PP2 ran untuned for rows missing at ordinals 4-7
   (`tunableop_missing.py:6-9,436-444`; `tp8-pp2-flashnext-2026-11-06.md:40`). The W7800 ordinals 0 and 5 need their own rows; `tune_missing_gemms.sh` refuses GPUs "0 1" by default and must be
   told which devices are busy (`:24-25`). hipBLASLt and rocBLAS gfx1100 data files are not verified on the box [INFERENCE].
7. **Comms.** On `rdna_extras` `rdna_ar` is enabled only under `on_gfx10x()` (`cuda_communicator.py:196-200`). R3 widens the gate to `on_gfx10x() or on_rdna3_hip("ar")` (`VLLM_RDNA_AR=1` still required, so RCCL stays the default), adds a group-wide agreement on arch, opt-in and wedge marker to the existing
   first `all_gather_object`, and gives gfx1100 its own marker (R3 document section 8; no `.cu` edit); the two-rank cross-root test is G9 (`tools/rdna/rdna3/ar_probe.py`). **One-shot size gate:** `rdna_ar` serves only messages up to min(`VLLM_RDNA_AR_MAX_KB`, `VLLM_RDNA_AR_ONESHOT_KB`),
   both 64 KiB by default; larger messages go to RCCL, a MAX_KB above the gate is dropped with a warning and the two-shot kernel no longer exists (`rdna_all_reduce.py:18-23,123-147`). A TP2 message is 5,120 B x W (W x 2560 fp16): 15 KiB at W = 3 (c1), 120 KiB at W = 24 (c8), 240 KiB at W = 48 (c16)
   and 10 MiB for a 2048-row prefill chunk. So c1 fits the default gate, c8 needs both variables at least 128, c16 at least 256, and prefill chunks always take RCCL (RCCL fallback: 25-29 us at 8 KiB [INFERENCE for the exact figure]).
   `hx` itself is arch-neutral HIP (no gfx-specific intrinsics beyond `s_sleep`), with the gfx1100 unknowns of section 6.4.
8. **IPC and allocator.** `hipIpcGetMemHandle` needs memory from the HIP allocator and `expandable_segments:True` breaks IPC (`flash-next-ple-ipc.md`). All `hx` buffers are `hx_alloc` allocations (uncached VRAM; T1 exported and opened them in all four visibility configurations on V620 pairs) outside torch's allocator;
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
- Stage 1 (H1) is `"islands": [[0, 1, 2, 3, 4]]` with `devices[0]` the W7800 on bus `cb` (HIP 0, switch `c0`), `devices[1..4]` the V620 of slot A (HIP 1-4) and `island_mode: "replicate"`: `owners[e]` starts with hub 0 for the up to 130 hub-resident experts of a layer, the rest are served by the four V620 (solver command in 6.11).
- Runtime dispatch: a hub copy wins (subject to the spill rule, section 6.2); otherwise the cold copy is `(row + layer) % n_cold_copies` over the expert-role owners in listed order (decision 16 proposes the primary copy only at the decode capture widths).
- The MTP layer is numbered `num_target_layers + N` for the draft module `mtp.layers.N` (the stats collector's rule, which the hub's plan loader applies): 96 for this checkpoint, as in the T10 plans, with `kind: "mtp"`; under policy `hub` it carries only hub owners (each hub holds its island's half; the H1 hub holds all 512).
- Optional fields the hub reads if present and defaults otherwise: `layers[].hub_order` (hub-resident ids by descending benefit; default ascending id) and `runtime.spill_k` (`{rows_bucket: K}`; default from the cost model, overridable by `VLLM_HETERO_MOE_SPILL_K`). The solver does not emit them yet (ExpertStats informed); nothing depends on them for correctness. The built-in defaults are the central routing's; the T10 seeds (11.4) are 2 to 11 % lower from 3 to 48 rows, so a plan solved on real statistics should carry `runtime.spill_k` (requested from ExpertStats).
- `plan_id` (a content hash) is verified in the step header by both sides.

### 9.3 Expert-stats records

Raw `expert-stats/1` JSONL (`meta` lines and cumulative `snapshot` lines: `counts[phase][e]` = rows whose top-k contains `e`, `tokens`, `steps`, `rows_hist`, `invalid`, `elapsed_s`) from
`vllm/model_executor/layers/fused_moe/rdna_expert_stats.py` (hook `MoERunner._apply_quant_method` right after `select_experts`, logical pre-EP ids, TP rank 0 only); converted
`hetero-moe-expert-stats/1` JSON (`stats_io convert`) with `layers[{layer, decode{counts}, prefill{counts}}]`, `step_hist.tokens_per_step`, optional `coact` (`.coact.npz` pairs). MTP-rejected rows are
counted; the MTP layer is numbered `num_target_layers + N` (96 here), kind `mtp`. Phase rule: a call with at most `VLLM_EXPERT_STATS_DECODE_MAX_TOKENS` (64) rows is decode.

### 9.4 Hub MoE call contract and `hx` ops

The contract the gfx1100 MoE path must satisfy (communicated to `feat/rdna3-flashnext`, which implements and audits it: R3 document section 7; same entry as `_rdna2_fused_moe`, `compressed_tensors_moe_wna16_rdna2.py:366-379`):

```text
hub_fused_moe(x[T,2560] fp16, topk_ids[T,10] int32 (-1 = masked), topk_weights[T,10] fp32,
              slot_map[512] int32 (slot or -1; device tensor passed as expert_map with global_num_experts = 512, updated in place between steps),
              w13_slots, w2_slots, scales ...) -> y[T,2560] fp16
  y = sum over valid (row,k) of weight * expert(slot_map[id])(x); pairs with id -1 or slot -1 contribute zero;
  no all-reduce; no atomics that change rounding between runs; graph-capturable for T <= 48; T up to 2048 eager or captured.
```

On gfx1100 the default accumulation of the W4A16 MoE kernel is a packed fp16 CAS-64 atomic add (order-dependent rounding). The hetero hub therefore requires `VLLM_RDNA3_MOE_DETERMINISTIC=1` (Q32: int64 fixed-point atomics, then
`moe_fixedpoint_to_half_rdna3`; R3 document section 7). Q32 is bit-stable across runs and routing order for a fixed `VLLM_RDNA3_MOE_BLOCK_M` (automatic: 1 for M <= 4, else 4, so results are not bit-equal across capture widths), fp16 only, returns NaN at or above 65520,
and needs scratch of 5.6 MiB at M = 48 and 240 MiB at M = 2048. R3's audit of this contract: `-1` ids are skipped end to end (`moe_align_sum_kernels.cu`; `moe_q_gemm_rdna3.cu:179,181-182,472` on R3's branch); `num_experts` must be the global count when a map is passed; `expert_map` is read
through a raw device pointer, so an in-place update is seen on graph replay and **rebinding a new tensor is not** (the slot-map flip of S8 writes into the existing tensor); no `.item()` or host sync and no data-dependent shape (checked under `FakeTensorMode` at M = 1, 3, 5, 24, 48).
Q32 speed is unmeasured (int64 atomics, scratch zeroing per call): T2 and G12.

New ops (HIP, `csrc/rocm/hetero_moe/`, registered in `_rocm_C`, compiled for both arches; Python wrappers in `vllm/distributed/hetero_moe/`):

```text
hx_begin_step(ctx, T, bucket)                                   # hub: in-graph epoch++, writes hdr to the 4 inboxes (bucket 0 = eager step)
hx_dispatch(x, topk_ids, topk_w, hub_ids_out, ctx, layer)       # hub: ownership and spill rule, per-server masked ids, x + routing pushes, flags; hub_ids_out[T,10] int32
hx_combine(y_hub, out, ctx, layer)                              # hub: wait 4 flags (timeout), fixed-order fp32 sum, fp16 out
hx_status(ctx)                                                  # CPU int64[8]: abort code, epoch, steps, counters
hx_srv_begin(ctx) / hx_srv_wait(ctx, layer, T) / hx_srv_push(y, ctx, layer, T)   # server side (T explicit: static per captured bucket)
```

`ctx` is an int64 device tensor holding a device-resident struct (peer pointers, flag pointers, epoch counter, spill and dispatch tables, abort word); all shapes are static per capture width. The schemas with their mutation annotations, the wire layout and the failure behaviour are Appendix C.3, C.4 and C.7.

### 9.5 Kernel requirements

**gfx1100 hub** (needed for W2 and for the hub role; the upstream gfx1100 kernels `gptq_gemm_rdna3`, `q_gemm_rdna3_wmma`, `moe_gptq_gemm_rdna3` exist behind `VLLM_ROCM_GFX1100`;
most fork HIP ops are gfx1030-only stubs on gfx1100; the status of each row after the RDNA3 port audit follows the table):

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
| 11 | TP2 all-reduce across roots: `rdna_ar` widened to gfx1100, or RCCL | 15 KiB (W = 3), 120 KiB (W = 24), 240 KiB (W = 48) decode; 10 MiB prefill (RCCL, above any one-shot gate) | S1 |
| 12 | `hx_*` ops (arch-neutral) | section 9.4 | S3 |
| 13 | W8A16 / W4A16 dense GEMV on gfx1100 (lever; the checkpoint's dense weights are bf16, so they must be quantized first) | M <= 48 | S10 |

Status on gfx1100. The R3 document (`origin/feat/rdna3-flashnext @ 30b6a47eca`) is the source: its section 2 is the dispatch policy, section 3 the per-op table (14 rows with before, after, default route and opt-in family), section 10 the per-op W7800 expectations and section 12 the GPU plan G0-G16.
This section keeps only the map from the rows above to its section 3 rows and the facts the design relies on. R3 ran 640 tests on the CPU and nothing on a GPU, so every HIP kernel on gfx1100 beyond the native MoE kernel is unvalidated (its sections 11 and 14).

| # | R3 row | Default route on gfx1100 | Opt-in family of `VLLM_RDNA3_HIP_OPS`; gaps |
|---|---|---|---|
| 1 | 6, 14 | upstream `wvSplitK` (n <= 5), `LLMM1`, hipBLASLt/rocBLAS above; no W7800 TunableOp rows (recipe `TUNABLEOP=0`; T13 = G15) | `dense`: `gemv_f16_rdna2`, int8 shadow |
| 2 | 7 | native `moe_gptq_gemm_rdna3`; its default mode is bit-identical to the old kernel | `VLLM_RDNA3_MOE_DETERMINISTIC=1` (Q32), `VLLM_RDNA3_MOE_BLOCK_M`; `-1` masking audited (section 9.4) |
| 3 | 7 | the same scalar kernel at decode and verify; the WMMA MoE kernel `moe_gptq_gemm_rdna3_wmma` (slice S1b, `b3e6129c61`) is written and CPU-tested only (`q_gemm_rdna3_wmma.cu` stays dense-only) | `VLLM_RDNA3_MOE_WMMA=1` (default off), batches of at least `VLLM_RDNA3_MOE_WMMA_MIN_M` = 256 tokens, tile `VLLM_RDNA3_MOE_WMMA_BLOCK_M` (0 = automatic), fp16 only |
| 4 | 3 | Triton | `gdn`: GDN decode and prefill chain, `causal_conv1d` update, sigmoid gating; real in the fat build, but the two kernels with inline `v_dot2c_f32_f16` are tested first (G2) [INFERENCE: whether gfx11 needs `v_dot2acc_f32_f16`] |
| 5 | 4, 9 | Triton / upstream metadata; `top_k_per_row_decode` is radix by default on gfx11 [INFERENCE], `VLLM_TOPK_DECODE_ALGO` | `qsa` |
| 6 | 1, 2 | Triton | `hc` |
| 7 | 7, 14 | router GEMM, top-k and align/sort as upstream (align gained `ignore_invalid_experts`) | none |
| 8 | 5 | torch | `se` |
| 9 | 12 | Triton `ple_conv` (arch-neutral); the n-gram sidecar is arch-neutral | none (the HIP `ple_short_conv_*` ops have no Python caller) |
| 10 | 14 | upstream torch lm_head, no fused argmax (three passes are 2.8 ms of weight streaming per step at TP2) | none; levers in S10 |
| 11 | 10 | RCCL | `ar` plus `VLLM_RDNA_AR=1`; size gate in section 8 item 7, unknowns in section 6.4 |
| 12 | none | not applicable (new ops of this spec) | none |
| 13 | 6 | none | `dense` (`rdna_dense_int8`); `gptq_gemm_rdna3` takes uint4b8 only and rejects AWQ zero points; nothing to feed it yet because the dense weights are bf16 in the checkpoint |
| + | 8 | MTP drafter experts: bf16 in the checkpoint, Triton `fused_moe` | `moeu` (fp16 HIP kernels); the derived W4A16 shard runs on row 2 |

What the design takes from the R3 document:

- Route policy (its section 2): gfx1100 defaults to the Triton/torch/upstream route plus the native W4A16 MoE kernel. The fork's HIP families (`hc,gdn,qsa,se,dense,moeu,ar,v1`, or `all`) are opt-in one at a time through `VLLM_RDNA3_HIP_OPS` and become default only after their per-op test and serve gates
  (G7, G11; decision 12). An unknown family name, or a named family whose probe op is missing from `_rocm_C`, raises at startup. gfx10x behaviour is unchanged by construction: each gate keeps its old term and adds `or on_rdna3_hip(fam)`, which is false off gfx1100.
- Build (its section 4): one fat `_rocm_C` (`tools/rdna/rdna3/build_rocm_c.sh <tree> 'gfx1030;gfx1100' 8`); details in section 8 item 3.
- MoE (its section 7): the hub contract of 9.4 is implemented in `csrc/rocm/moe_q_gemm_rdna3.cu` and `experts/rdna3_moe.py`; G4 holds the 10 new GPU tests (hub shapes, `-1` masking, Q32 identity over 50 repeats, split-expert partial sums, graph capture and replay).
- All-reduce (its section 8): the size gate and the gfx1100 unknowns are in section 8 item 7 and section 6.4; the probe is G9.
- Checkpoint (its section 9): only the routed experts are W4A16 (the `ignore` list keeps `lm_head`, `embed_tokens`, `mtp.*`, PLE, vision, the router `.gate`, `hyper_connection`, `indexer`, `.linear_attn.`, `.self_attn.` and `shared_expert` unquantized), so every dense weight is bf16 in the file and fp16 at run time;
  the W8A16/W4A16 hub dense lever (row 13, S10) is therefore a new quantization with its own calibration, not a conversion of AWQ weights.
- MTP drafter (its section 9): the drafter is entirely BF16 (`model_mtp.safetensors` 4.86 GiB; experts 4.69 GiB, 2.34 GiB per W7800 at EP2). `tools/rdna/quant_mtp_experts.py <src> <dst>` derives the W4A16 shard this spec assumes: 512 x 2,534,400 B, 0.60 GiB per hub
  (the R3 document and recipe write 0.65 GiB per rank for the same shard; the byte count is 0.649 GB). Open decision 4 here, decision 2 there.
- Expected numbers (its section 10): central target forward without all-reduce, drafter and gaps 18.5 / 33.1 / 46.9 ms at W = 3 / 24 / 48 (pessimistic 23.9 / 41.9 / 61.8, optimistic 15.3 / 27.6 / 38.2) with the per-op microseconds there; the 4.64 GiB per rank of dense bytes are a 7.2 ms floor of that (its section 13).
  They are compared with this model in 5.3. Its memory estimate per W7800 at TP2 ([INFERENCE] from sizes): experts 29.4 GiB, dense 5.3 GiB, MTP 0.65 GiB (W4A16) or 2.34 GiB (bf16), KV 3.75 GiB, against 44.98 GiB (the byte table here has 29.0 GiB of experts per hub, section 2.1).
- Recipe and plan: `flashnext-w2-mtp2.env` in 6.10, the GPU steps G0-G16 against the T-steps in 11.1.

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
| `VLLM_RDNA3_HIP_OPS` | empty | R3 (`30b6a47eca`): fork HIP families on gfx1100, `hc,gdn,qsa,se,dense,moeu,ar,v1` or `all`; empty keeps the Triton/torch route (the W2 recipe on R3's branch still leaves it empty); G7 (11.3): `hc,se` go into the H1 and W2 recipes, `gdn` stays out (HIP wins only at one or two sequences) |
| `VLLM_RDNA3_MOE_DETERMINISTIC` | 0 | R3: gfx1100 W4A16 MoE accumulates in Q32 fixed point (int64 atomics, fp16 only, run-to-run identical for a fixed block size); required on the hub, 0 in the W2 recipe until G12 |
| `VLLM_RDNA3_MOE_BLOCK_M` | 0 | R3: gfx1100 W4A16 MoE token-block size 1, 2, 4 or 8; 0 = automatic (1 for M <= 4, else 4) |
| `VLLM_RDNA3_MOE_WMMA` | 0 | R3 (`b3e6129c61`): prefill-sized batches of the gfx1100 W4A16 MoE on the WMMA grouped GEMM (S1b), bit-deterministic, fp16 only; off until it has run on a W7800 |
| `VLLM_RDNA3_MOE_WMMA_MIN_M` | 256 | R3: smallest token batch that takes the WMMA path (decode and verify keep the scalar kernel) |
| `VLLM_RDNA3_MOE_WMMA_BLOCK_M` | 0 | R3: WMMA token-block size 16, 32 or 64; 0 = automatic from the mean pair rows per expert |
| `VLLM_RDNA_SE_DECODE_MAX` | 32 | R3 (`c283fbffa6`): largest decode batch of the fused shared-expert op; 33 to 64 rows run on the same tiles (host checks only, gfx1030 and gfx1100 code objects identical), need fp16 weights and at most 128 tickets (I = 640 and 320 pass up to 64 rows), else torch; the H1 and W2 recipes set 48, the hub capture width |
| `VLLM_TOPK_DECODE_ALGO` | auto | R3: `top_k_per_row_decode` radix or insertion (`auto` is radix on gfx11; G8: insertion is 1.3 to 2 times faster at 4k to 8k live columns and equal at 1k and from 16k, 45 to 110 us per step over the 12 QSA layers; R3 sets `insertion` in the recipes after a serve A/B) |
| `VLLM_RDNA_AR` | 0 | existing (`vllm/envs.py:198`): 1 enables `rdna_ar`; on gfx1100 it also needs the `ar` family |
| `VLLM_RDNA_AR_MAX_KB`, `VLLM_RDNA_AR_ONESHOT_KB` | 64, 64 | existing (`rdna_all_reduce.py:123-147`): the one-shot gate is the smaller of the two and larger messages go to RCCL; TP2 messages are 15 KiB at W = 3, 120 KiB at W = 24, 240 KiB at W = 48, so set both to at least 128 for c8 and 256 for c16 |

## 10. Slices and acceptance gates

Owners: R3 = Rdna3Port (`feat/rdna3-flashnext`), ES = ExpertStats (`feat/expert-stats`), HX = HxServer (`feat/hetero-hx`), HUB = HubMethod (`feat/hetero-hub`), HM = this branch `feat/hetero-moe` (HeteroDesign, spec, model, integration). A slice is done when its gate passes on the box; gate numbers cite
the tests of section 11. CPU-only parts can start before the W7800 are free. The R3 document uses the slice names S1 and S1b for W2; its G0-G16 are the GPU steps of S1, S1b and S2. IDs are never shared: S-1 to S12 and T0-T16 are this document's, G0-G16 are R3's (11.1 maps them).

**The work is staged (user decision, 2026-11-07).** Stage 1 brings one island up end to end: W7800 HIP 0 (bus `cb`) + V620 HIP 1-4 (slot A, switch `c0`), layout H1 (hub TP1, slot pool, four expert servers). The goal is correct first (production compiled mode: probes 3/3, PPL 4.75-4.85, MTP2, concurrency probe), then fast.
HIP 5 stays untouched in stage 1. Stage 2 uses all ten GPUs the fastest way, as decided by section 5.7; its slices are in the second table. W7800 time follows the order Rdna3Port (S1-H1), HxServer (S0-H1 and S3), HubMethod (S4-H1 onwards), one owner at a time under the lease rules of section 11.

**Stage 1 (H1, island 0)**; S-1, S2 and S3 of the second table are stage-1 slices as written there:

| Slice | Content | Owner | GPU needed | Gate |
|---|---|---|---|---|
| S0-H1 | Cross-arch IPC and microbenchmarks on island 0: T1 on HIP 0 + 1 (intra-switch; the cross-root pairs 0 + 6 and 0 + 5 are stage 2) including the three gfx1100 unknowns of 6.4 on both arches; T2 on HIP 0 | HX (T1), R3 (T2) | HIP 0 + 1 | the T1 and T2 numbers of S0 below |
| S1-H1 | The RDNA3 port at the TP1 hub shapes on **one** W7800: G2 to G8 and G16 at GDN 16 k-heads / 48 v-heads, QSA 24 query / 2 KV heads (group 12), `I_SE` 640, lm_head N = 248,320, `VLLM_RDNA3_MOE_DETERMINISTIC=1`, plus a layer-truncated end-to-end check (R3 G13 method: the first layers of Flash-Next with all their experts on the W7800, hidden states against a V620 process). Flash-Next does not fit one W7800 (experts 58 GiB against 44.98 GiB), so the first full-model run is S4-H1 | R3 | HIP 0 (+ one V620 for the G13 reference) | per-op parity as G2 to G6; G7 timings at the TP1 shapes inside the pessimistic-to-optimistic band of `perop_expectations.py` (a family is enabled only if at least as fast as the default); the truncated run matches the V620 hidden states at the R3 G13 tolerance; the first TP1-only failure is reported at once |
| S4-H1 | Integration, eager: `HeteroMoEMethod` at TP1 (`island_mode: replicate`, 130 slots per layer), plan loader and tables, server launch, boot self-test | HUB (+ R3, HX) | HIP 0 + 1-4 | T6-H1: boots within 44.98 GiB, probes 3/3, PPL 4.75-4.85 and within 0.02 of a TP4 slot A run on the same text, concurrency probe PASS, 1 h c8 soak without a hang or an abort record, per-layer hub-local / remote / spilled counters recorded; eager ITL is a diagnostic only |
| S5-H1 | Production compiled mode: hub `FULL_AND_PIECEWISE` with the `hx` ops, server step graphs, spill table K tuned (default 10 / 21 / 39 / 65 / 110, 6.2) | HUB + HX | same | T7-H1 and T14 (`COMPILE_MODE=3`, MTP2, ABBA against TP4 slot A with fresh boots): the correctness gates of T6-H1 again (probes, PPL, concurrency probe, and a 10-min c8 compiled soak in place of the 1 h one), MTP acceptance within 3 % of TP4 slot A (1.84 / 2.01), no capture fault; first-build speed not red (5.4a: green is c1 at most 36.9 ms and c8 at most 60.8 ms, red is above 40.2 or 72.3 ms: stop and trace); 2048-token chunk at most 1.5 s with the scalar MoE kernel (at most 0.8 s with WMMA; model 0.66 s at `eff_hub` 0.30 and 1.32 s at 0.15); KV at least 470k tokens (TP4 slot A: 267k; the model gives 474k with the bf16 drafter and 611k with the W4A16 drafter at 130 slots, 5.2a) |
| S6-H1 | MTP2 on the hub at TP1: the checkpoint's bf16 drafter first (it fits, 5.2a), the W4A16 shard second | HUB + R3 | same | T8: W4A16 acceptance at least 97 % of bf16 at c1 and c8, else bf16 stays |
| S10-H1 | "Then fast": the two levers of 5.4a, vocab-parallel lm_head on the servers (opcode 2) and W8A16 hub dense | R3 + HX | same | T15: PPL 4.75-4.85 and probes 3/3; ABBA against TP4 slot A with both levers: c1 at most 24.6 ms and c8 at most 48.4 ms (TP4 slot A x 1.10; model 23.3 / 44.0), each lever's gain within 30 % of the model (-2.9 ms lm_head, -7.3 ms W8A16 at c1) |

**Stage-1 exit:** S4-H1 passes (correct, eager); S5-H1 passes its correctness gates in production compiled mode with the speed gate not red (eager numbers are diagnostics, never verdicts); S6-H1 has chosen the drafter precision; the S10-H1 numbers are recorded; and the T3 and T7 measurements are folded into `perf_model.py` (refit `kappa`, `eta_h`, `gamma`, `detect_us`), which reprints the stage-2 tables of 5.7 and the decision rule of 5.7.5.

**Stage 2 and general slices.** S1 and S1b (W2) are the unconditional start of stage 2 (they gate D3 as well as H2) and need HIP 5; S4-S9 apply if H2 is built (5.7.5); S11 and S12 are D3's.

| Slice | Content | Owner | GPU needed | Gate |
|---|---|---|---|---|
| S-1 | Routing statistics on V620 TP4 `flashnext-mtp2` (about 1 h agentic), converted stats, solver plans, `simulate` | ES tools, run by the lead | V620 only (HIP 6-9) | T10: collector validation items 1-8 of the ExpertStats document; two disjoint windows give the same plan quality (held-out `simulate` hub hit within 0.05). Status: collected, two windows (11.4); the plan-quality gate is met (held-out hub hit within 0.01 of in-sample) |
| S0 | Microbenchmarks: cross-arch IPC and P2P, W7800 bandwidth, WMMA, expert GEMV (stage 1 runs the intra-switch part as S0-H1; the cross-root pairs 0 + 6, 0 + 5 and the W7800 pair are stage 2) | HX | 1 W7800 + 1 V620 per switch | T1, T2: handle opens and stores byte-exact; 16 KiB push + flag at most 9 us p50; W7800 streaming at least 0.70 x 864 GB/s; expert GEMV at most 5.0 us |
| S1 | **W2** (stage 2, the engine of D3 and D5): Flash-Next W4A16 + MTP2 on gfx1100, TP2 over both W7800, EP2, no V620; `flashnext-w2-mtp2.env`; TP2 all-reduce across roots; default Triton/torch route first, then the `VLLM_RDNA3_HIP_OPS` families one at a time | R3 | both W7800 | Code, recipe and CPU tests are on `origin/feat/rdna3-flashnext @ 30b6a47eca` (640 CPU tests); the GPU steps are its G0-G16. T4 (= G9 to G12): boots in production compiled mode, probes 3/3, PPL 4.75-4.85, MTP acceptance 1.84 / 2.01, concurrency probe PASS, memory within 44.98 GiB, c1 at most 28 ms (go; 26 ms good), c8 at most 52 ms on the better route. The prefill gates (chunk at most 0.60 s, 16k TTFT at most 4.5 s) apply with the WMMA MoE kernel (S1b) and are only recorded with the scalar kernel. Measured values replace the model's W7800 knobs and the model is refit (rule 5.7.5) |
| S1b | WMMA W4A16 MoE prefill grouped GEMM on gfx1100 (the scalar `moe_gptq_gemm_rdna3` is the fallback) | R3 | 1 W7800 | Written and CPU-tested on `origin/feat/rdna3-flashnext @ b3e6129c61` (R3 document 7.1; opt-in `VLLM_RDNA3_MOE_WMMA=1`), built into the fat tree (178 `_rocm_C` ops; 9 gfx1100 kernels with 0 spills and 148-233 VGPR) with the gfx1030 code objects identical to the untouched base (R3 document 16.5); never run on a GPU: `gpu_g0_g7.sh --only G4 --labels G4-wmma-1,G4-wmma-2,G4-wmma-3` (the smoke subset, without `--keep-going` the first time), then a timing of the three tiles at M = 256 to 2048. T2: at least 27 TFLOPS all-in on the expert shapes (floor 13.8); T4 prefill gates (chunk at most 0.60 s, 16k TTFT at most 4.5 s) |
| S2 | Hub MoE masked partial sums and swappable slot pool (contract of section 9.4) | R3 + HM | CPU first, then W7800 | CPU: partial-sum identity (hub residents + the other experts equals the full MoE; masked ids and `slot_map` swap behave; deterministic) with a CPU reference. GPU: G4 (hub shapes, `-1` masking, Q32 identity over 50 repeats including M = 2048, split-expert partial sums, graph capture and replay) with `VLLM_RDNA3_MOE_DETERMINISTIC=1` and a fixed `VLLM_RDNA3_MOE_BLOCK_M`, G12 (run-to-run identical logprobs, Q32 cost), per-expert time at most 5.0 us (T2) |
| S3 | `hx` kernels and the expert server, host-driven per-layer, eager; loopback with a replaying hub | HM | 1 W7800 + 4 V620 | T5 (gates as amended in 11.2): server output byte-equal to the local TP4-EP kernel for the same ids; the server chain (real minus synthetic servers) within 1.15 x the model's expert time of the busiest server; kill tests clean within 5 s; 10^6 iterations without a mismatch; the exchange and the layer time recorded against the loopback and the plan-specific limits |
| S4-H2 | Integration, eager, stage 2: the S4-H1 code with `island_mode: partition`, TP2, EP2 islands and the second hub worker | HUB + R3 | both W7800 + 8 V620 | T6: PPL equal to W2 within 0.02, probes 3/3, concurrency probe PASS, 1 h c8 soak without a hang; c1 eager ITL recorded |
| S5-H2 | CUDA graphs, stage 2: hub FULL_AND_PIECEWISE with `hx` ops on two hubs, spill table K tuned | HUB + HX | same | T7: c1 at most 26 ms, c8 at most 40 ms, MoE per layer at most 60 us at c1, no capture fault; ABBA against TP4 with fresh boots |
| S6 | MTP2 on the hub, H2 (stage 1 does it at TP1 as S6-H1): W4A16 drafter, acceptance A/B | HUB + R3 | same | T8: accepted tokens per step with W4A16 at least 97 % of bf16 at c1 and c8, else bf16 on the hub |
| S7 | Prefill: star then V620-forwarded tree, two-microbatch pipeline, servers on W4A8 sdot4, `K[prefill]` | HM | same | T9: 16k c1 TTFT at most 3.5 s, chunk at most 0.45 s, star link at most 50 % busy |
| S8 | Placement runtime: plans from real stats, replicas, partition islands, hot re-placement at idle | ES + HM | same | T11: hub hit within 0.05 of the estimate on a held-out window, server max/mean touched at most 1.35, step time at most 1.15 x `simulate` |
| S9 | Hardening: failure matrix, soak, thermals, `flashnext-hetero-h2.env`, docs | HM | same | T12: 4 h agentic soak, injected kills and timeouts surface as failed steps, no hang, no wedge marker, restart works |
| S10 | Optional levers, H2 (stage 1 does them as S10-H1): W8A16 (`rdna_dense_int8`) or W4A16 (`gptq_gemm_rdna3`, uint4b8 only; the checkpoint's dense layers are bf16, so this is a new symmetric quantization with its own calibration and quality gate) hub dense, vocab-parallel lm_head on the servers | R3 + HX | same | PPL inside the 4.75-4.85 band, probes 3/3; c1 gain measured (model: -3.7 ms and -1.4 ms) |
| S11 | Request router for D3 and D5 (and 2 x H1): least-ITL placement with session affinity, health checks, SSE passthrough (5.7.6); CPU tests first | HM follow-up | none, then all engines | T16: the engines under the 1k/512 load, aggregate within 5 % of the sum of the engines run alone, no engine starved, mean ITL within 10 % of the model's placement |
| S12 | D3 and D5 bring-up: W2 + 2 x TP4 or W2 + TP8 concurrently (ports, `PCI_BUSES`, KFD queue and thermal budget, host RAM for the PLE tables) | HM follow-up | all ten | T16: D3 at least 780 tok/s at 24 users on the 1k/512 load (model 803 central), D5 at least 540 tok/s at 16 users (model 568); 16k TTFT at most 4.5 s on the W2 engine, no evicted_ms jump, junction temperatures inside limits |

Dependencies: stage 1 runs S-1 (V620 only) and S0-H1 in parallel, S1-H1 and S2 in parallel, S3 after S0-H1, S4-H1 after the per-op part of S1-H1 (G2 to G8, G16), S2 and S3, then S5-H1, S6-H1 (independent of S5-H1 until the compiled gates) and S10-H1. S1-H1's layer-truncated end-to-end check (G13 method) does not block the start of S4-H1 (R3 plans it after the HxServer and HubMethod windows, unless the lead reorders),
but it is the localization tool for a failed S4-H1 correctness gate (then it jumps the queue) and must have passed before the stage-1 exit. Stage 2: S1 gates S4-H2 and the H2 go/no-go; S1b gates only the prefill gates of S1 and the hub prefill rate in S7
(H2 tolerates a scalar kernel down to `eff_hub` 0.15); S0 gates S3 for the cross-root pairs; S-1 gates S8 only (S4-H1 to S7 run on the stripe plan); S6 and S7 are independent of each other; S11 and S12 need S1 only.

CPU-testable work that does not wait for any GPU: the plan loader and validator wrapper (`vllm/distributed/hetero_moe/plan.py`, HUB, reusing the semantics of `tools/rdna/hetero_moe/validate.py`), the Python/numpy reference of the `hx` wire (slot layout, sequence arithmetic with the wrap-safe compare, abort record encode and decode, HX),
the plan-to-table builder and the numpy reference of the dispatch, spill and combine rule (`ref_hx.py`, HUB) with the spill rule (bitmap + prefix popcount) against a plain loop and the partial-sum identity over a CPU W4A16 MoE reference (Appendix C.9). (`perf_model.py` is already covered by `tests/tools/rdna/test_hetero_perf_model.py`.)

## 11. GPU test plan (when the W7800 are free)

Stage 1 uses HIP 0 (approved by the user for stage-1 work) and the slot A V620s HIP 1-4; HIP 5 (bus `07`) stays untouched until the lead says otherwise, so T4 (W2), the cross-root pairs of T1, the two-hub tests and T16 are stage 2 (T10 needs only HIP 6-9). Rules for every test: one change per boot; fresh boot per arm and ABBA ordering for every performance claim (repo practice,
`tp8-pp2-flashnext-2026-11-06.md`); protected window as in the 2026-11-06 disagg experiment (whose first step was also called G0; it is unrelated to the port steps below): quiet box, no concurrent build, KFD `evicted_ms` sampler, junction-temperature watcher `tools/rdna/tp8pp2/tempwatch.py`,
`dmesg` diff; a cell is invalid when `evicted_ms` jumps: `disagg-g0-2026-11-06.md:84-89`); clear the `rdna_ar_wedged` marker per arm (`rdna-ar-twoshot.md:74-75`; gfx1100 has its own marker `rdna_ar_wedged_gfx1100`, R3 document section 8); confirm the imported vllm tree
(the venv's editable path points at `vllm-rdna-mtp-mono`); never start a server on a device another job owns. The rules of the R3 document section 12 apply to every W7800 step: box lease and lock rules as it cites them, never a V620 in a W7800 process, `ROCR_VISIBLE_DEVICES` unset,
`HIP_VISIBLE_DEVICES=0` for single-GPU steps and `0,5` for TP2, stop at the first failing step.

Box rules that apply to every stage-1 step (the lead's rules, abridged): resolve HIP to PCI bus before every launch (`rocm-smi` lists KFD order `07 0a 0f 14 17 cb ce d3 d8 db`, not HIP order; HIP 0 is bus `cb`, slot A = HIP 1-4 is `ce d3 d8 db`); every GPU command, bench clients included, sets `HIP_VISIBLE_DEVICES` on the same line;
a W7800 process holds the lease file `<owner>.w7800cb` in `~/w4a8_runs/lanes/lease/` (one owner at a time), and a full-island run (T5 with real servers, T6-H1, T7-H1, T14, T15) also holds the slot lease `<owner>.slotA` (at least 28 GiB free per V620 before the boot); a boot takes `~/w4a8_runs/boot.lock` and waits until no other quiet file is newer than 6 min;
a measured window touches `~/w4a8_runs/lanes/quiet/<owner>` (at most 15 min, re-touched every 5), waits until `flock -n ~/w4a8_runs/boot.lock true` succeeds, re-checks before each cell and never overlaps a `docker buildx build`; junction temperature is logged on every sustained run, which aborts at a 5 s mean of 90 °C or one sample of 95 °C; a KFD `evicted_ms` jump invalidates the window;
verdicts come from production mode only (`COMPILE_MODE=3`, `FULL_AND_PIECEWISE`, MAXBAT 2048; eager is a debug step and dropping graphs counts as eager); benches run on exact-SHA trees (`git archive` of the SHA, the matching `.so`, a `.sha` file, a per-tree `VLLM_CACHE_ROOT`; never the venv's installed `_rocm_C`);
every process that touches the W7800 uses `~/Apps/vllm/venv-7.14.0` (the only venv with the gfx1100 torch and rocm-sdk device packages; the `0.31.0-final`, `0.32.0` and `0.32.0-fix` venvs are gfx1030-only and fail the first torch kernel on a W7800 with `hipErrorInvalidImage`) with `PYTHONPATH` set to the exact-SHA tree (otherwise `import torch` autoloads vllm from the venv's editable tree and a second `_rocm_C` registration aborts the process),
and the V620 expert servers of a mixed setup use the same venv and the same fat gfx1030 + gfx1100 build (reference tree `~/vllm-rdna3-flashnext` @ `30b6a47eca`, 177 `_rocm_C` ops).
Order of use of the W7800 (HIP 0): Rdna3Port (S1-H1: G0 to G8), then HxServer (T1 on HIP 0 + 1, then T5 with the W7800 as hub), then HubMethod (S4-H1: eager, then compiled); a later owner starts when the earlier one has released the lease. Status 2026-11-07: Rdna3Port's G0 to G8 and G16 are done (11.3), HxServer's W7800 steps (runbook 3a to 3d) are next, and S1-H1's layer-truncated end-to-end check follows the HxServer and HubMethod windows.

| ID | Test | Setup | Procedure | Pass | Feeds |
|---|---|---|---|---|---|
| T0 | Preflight | idle box, W7800 released | `rocm-smi --showmeminfo vram` and `--showpids`; `lspci -vv` link status of buses `cb` and `07` under a trivial load; kernel command line (`pcie_ports=native`); imported vllm path; G0 on a hub process (`_GCN_ARCH` gfx1100; the mixed-arch fallback is logged once) and the same print from one server process (gfx1030), with and without `amdsmi` importable; G16 (QSA route and mrope print, gfx1100 rocBLAS and hipBLASLt libraries) | both W7800 44.98 GiB total and unused; link 16 GT/s x16 under load (not the 2.5 GT/s x1 runtime-PM state); path as intended; arch per role as stated; G0 and G16 pass | all |
| T1 | Cross-arch IPC and P2P | HIP 0 + 1 (intra-switch), 0 + 6 (cross-root), 0 + 5 (the W7800 pair, cross-root) | single process: `tools/rdna/p2p_probe/p2p_probe.cpp` rebuilt with `--offload-arch=gfx1030 --offload-arch=gfx1100`, modes `lat`, `latc`, `bw`, `a2a`; then a two-process IPC probe (hub visible {0,5}, server visible {1}) in four visibility configurations (disjoint, shared, own-first both ways), kernel stores into the IPC-mapped buffer, fan-in of 4 servers into one hub; on both arches the three gfx1100 unknowns of section 6.4 (flag poll visibility, `wall_clock64` tick rate against the host clock, uncached allocation) | `hipIpcOpenMemHandle` works in at least one configuration (disjoint preferred); stores byte-exact over 10^6 iterations with a rotating pattern; 16 KiB + flag one-way at most 9 us p50 and 20 us p99 (model 7); 4 x 15.2 KiB fan-in at most 15 us p50; 1 MiB fan-in at least 24 GB/s aggregate; a polled flag written by the peer is seen on both arches within the same p99 (never stale); the tick rate is known to 5 % (else the measured rate is used); no amdgpu error; KFD queue counts as expected | S0 gate, `PCIE_LAT_US`, risks R2 and R3 |
| T2 | W7800 microbenchmarks | HIP 0 | streaming read kernel (achieved fraction); W4A16 MoE GEMV per expert at M = 1, 3, 8 through `moe_gptq_gemm_rdna3` (default and `VLLM_RDNA3_MOE_DETERMINISTIC=1` with a fixed `VLLM_RDNA3_MOE_BLOCK_M`; correctness is G4, dense GEMV/GEMM timing is G7); WMMA GEMM at M = 2048 on the dense shapes (`gptq_gemm_rdna3_wmma`) and the scalar MoE kernel at M = 2048 on the expert shapes (the WMMA MoE kernel is S1b) | achieved at least 0.70 x 864 GB/s; per expert at most 5.0 us (model 3.67); WMMA all-in at least 27 TFLOPS (model), floor 13.8 TFLOPS (break-even against TP4 prefill); the scalar MoE prefill rate is recorded | `eta_h`, `--hub-us`, `eff_hub` |
| T3 | Hub dense chain per block | HIP 0 for G7, the W2 boot (S1) for the trace | G7 (graph-timed per-op benches at W = 3, 24, 48 on the default route and with the family set, against `perop_expectations.py --scenario central`), then a `rocprofv3` trace of a decode step at W = 3, 24, 48 and of a 2048 chunk, as in `p0-results.md`: HC preamble, GDN and QSA blocks, router, shared expert, lm_head, all-reduce | per-op table against the R3 per-op model (R3 document section 10); all-reduce per call at 15 / 120 / 240 KiB recorded for RCCL (model 18 us); refit `kappa`, `gamma`, `comp_ratio`; re-run `perf_model.py` with the measured values | model refit, S1 decision |
| T4 | W2 boot, quality, speed | HIP 0 + 5, TP2, EP2, MTP2 | G9 (`rdna_ar` probe), G10 (`serve_rdna.sh RECIPE=flashnext-w2-mtp2 PCI_BUSES="cb 07"` on the default route, `serve_validate.sh`), G11 (family ladder, `ar` last with `VLLM_RDNA_AR=1` and `VLLM_RDNA_AR_MAX_KB` = `VLLM_RDNA_AR_ONESHOT_KB` at least 128 for c8, 256 for c16) and G12 (Q32 cost); in addition the PPL harness (`w4a8-quality-2026-11-04.md`), the concurrency probe, and the cells c1, 1k c8, 16k c1 of the `tools/rdna/tp8pp2` harness against a V620 TP4 boot (ABBA) | G10: probes 3/3; PPL 4.75-4.85; MTP acceptance 1.84 / 2.01; memory within 44.98 GiB; c1 at most 28 ms; c8 at most 52 ms (better route). G9: zero mismatches at 8, 64, 120, 256 KiB and 10 MiB, p50 at most 20 us per call at 8 KiB in graph mode, timeout phase aborts within 0.8-1.5 x 1500 ms. With the WMMA MoE kernel (S1b) also chunk at most 0.60 s and 16k TTFT at most 4.5 s, recorded only with the scalar kernel | S1 gate (decode), S1b gate (prefill) |
| T5 | Expert server alone | 1 W7800 process replaying layer calls + 4 V620 servers | recorded `x`, ids and weights from T4 (stage 1: the synthetic Zipf family of `t5_common.make_inputs`, since T4 needs HIP 5); compare server output with the local TP4-EP kernel; round-trip at W = 3, 24, 48, 2048; kill the hub, kill a server, freeze a server, freeze the hub (3 s begin wait), wrong `plan_id`; with synthetic servers (exchange alone) and with real weights | byte-exact outputs; server chain (real minus synthetic servers) at most 1.15 x the model's expert time of the busiest server; exchange and layer time recorded against the loopback (35.9 / 64.2 / 98.6 us) and the plan-specific limits of 1.3 x `perf_model.py t5_budget` (11.2; the 47 / 161 / 256 us of the first draft are the H2 remote path); aborts within 5 s with no wedge, buffers released, peers exit; a wrong `plan_id` ends the hub with exit 4 after the 20 s connect timeout and leaves the servers up | S3 gate |
| T6 | Integrated eager | H2, `--enforce-eager` | `VLLM_HETERO_MOE=1` with the stripe plan; PPL, probes, concurrency probe, 1 h c8 soak, per-layer trace from the stats buffer | PPL within 0.02 of T4; probes 3/3; no hang; trace shows hub-local / remote / spill counts | S4 gate |
| T7 | Graph mode | H2, production compiled mode | capture widths 3/6/12/24/48; cells c1, 1k c8, 16 users; spill table `K` sweep; ABBA against TP4 | c1 at most 26 ms; c8 at most 40 ms; MoE per layer at most 60 us at c1; spill counts match `K`; no capture fault; ITL p99 not worse than TP4 | S5 gate, `--hub-fixed-us` |
| T8 | MTP acceptance | W2 is enough | `tools/rdna/quant_mtp_experts.py` first (the checkpoint's drafter is bf16); W4A16 versus bf16 drafter on identical prompts at c1 and c8, at least 3 runs each, mean accepted tokens per step | W4A16 at least 97 % of bf16 | MTP policy |
| T9 | Prefill | H2 | 16k c1 TTFT; 1k c8 prefill aggregate; per-layer hub, server and link occupancy from device clock stamps in the stats buffer | TTFT at most 3.5 s; chunk at most 0.45 s; star link at most 50 % busy (else build the tree) | S7 gate, `eff_hub`, `eff_v620_moe` |
| T10 | Routing statistics | V620 TP4 (HIP 6-9), runs before any W7800 time | ExpertStats document, sections 7 and 8 | validation items 1-8; plan validates | S-1, S8. Ran (11.4); ExpertStats reports items 1-8, the count identity was re-checked from the converted stats of both windows |
| T11 | Placement quality | H2 | plan from T10; run the held-out workload; compare hub hit, server max/mean touched and step time with `simulate` | hub hit within 0.05; max/mean at most 1.35; step time at most 1.15 x `simulate` | S8 gate |
| T12 | Soak and failure injection | H2 | 4 h agentic load; kill each process kind; forced timeouts; thermal and `dmesg` watchers (pattern of `tools/rdna/ar_soak/`) | no hang, no wedge marker, aborts surface as failed steps, restart works, junction temperatures inside limits | S9 gate |
| T13 | TunableOp rows for the W7800 | HIP 0, 5 | the TunableOp item of G15, after G10 because the W2 recipe starts with `TUNABLEOP=0` (a row is an arch-specific solver index, gfx1030 rows must not be used on gfx1100): `tools/rdna/prof/tune_missing_gemms.sh` with `REFUSED_GPUS` set to the devices that must not be touched | rows for ordinals 0 and 5 present; ABBA gain recorded | S1, S5 |
| T14 | H1 against TP4 slot A | HIP 0 + 1-4 (H1) and HIP 1-4 (TP4 slot A) in alternation (the H1 servers are the TP4 V620s, so the arms never overlap), both in production compiled mode with MTP2; leases `w7800cb` and `slotA` | ABBA with fresh boots (TP4 slot A, H1, H1, TP4 slot A) on the `tools/rdna/tp8pp2` harness: cells c1, 1k c8 (8 users), 16 users, 16k c1 TTFT and the 2048-token chunk; per arm the probes, the PPL harness, MTP acceptance at c1 and c8, the boot log's KV tokens, the hub counters (hub-local / remote / spilled per layer), junction temperatures and `evicted_ms` | correctness: probes 3/3, PPL 4.75-4.85, MTP acceptance within 3 % of TP4 slot A (1.84 / 2.01), KV at least 470k tokens (TP4 slot A: 267k). Speed, first build: green is c1 at most 36.9 ms and c8 at most 60.8 ms (model central x 1.15), red is above 40.2 ms or 72.3 ms (pessimistic: stop and trace one step with `rocprofv3` before any further slice), between is amber (go on with T15). The parity target after T15 is in 5.4a | stage-1 exit, refit of `kappa`, `eta_h`, `gamma`, `detect_us`, rule 5.7.5 |
| T15 | Levers on H1 | H1 as T14 | four H1 arms in ABBA order: base, + vocab-parallel lm_head on the servers (opcode 2), + W8A16 hub dense, + both; per arm the PPL harness, probes, MTP acceptance, cells c1 and 1k c8 | PPL inside 4.75-4.85 and probes 3/3 in every arm; c1 gain per lever within 30 % of the model (-2.9 ms lm_head, -7.3 ms W8A16, -8.7 ms both); both levers: c1 at most 24.6 ms and c8 at most 48.4 ms (TP4 slot A x 1.10) | S10-H1, rule 5.7.5 |
| T16 | D3 and D5 concurrent (stage 2) | W2 (HIP 0 + 5) with either TP4 slot A + TP4 slot B (D3) or TP8 on all eight V620 (D5), each engine in production compiled mode; leases on both W7800 and both slots | each engine alone first (the load of T14, ABBA), then all engines together on the 1k/512 mixed load with clients pinned per engine, then through the router (S11); aggregate tok/s, per-engine ITL and TTFT, junction temperatures, `evicted_ms`, KFD queue count per process, host RAM and pinned-memory totals (D3: three PLE tables, D5: two, each 95.4 GiB of the 512 GiB, R11) | D3: aggregate within 5 % of the sum of the engines alone and at least 780 tok/s at 24 users (model 803); D5: at least 540 tok/s at 16 users (model 568) and ITL at two users at most 11.4 ms (model 10.8, TP8 alone 11.9); 16k TTFT at most 4.5 s on W2; no `evicted_ms` jump; temperatures inside limits; host RAM within budget | S11, S12, rule 5.7.5 |

Model parameters each test replaces: T1 `PCIE_LAT_US`, `PCIE_BW`; T2 `eta_h`, `eff_hub`, hub per-expert time; T3 `kappa`, `gamma`, `comp_ratio`, `rho_*`; T5/T7 `detect_us`, fixed terms; T14 the whole H1 step (`kappa`, `eta_h`, `gamma`, `detect_us`, spill `K`); T9 `eff_hub`,
`eff_v620_moe`, `prefill_hub_share`; T10 `routing` (its tables replace `hit` and `locality`; ran, 11.4). After T3, T7 and T14 the model is re-run and sections 5.4a and 5.7 of this document are updated with measured rows.

Stage-1 variants of the tests above (island 0, HIP 0 + 1-4, no HIP 5):

| Test | Stage-1 variant |
|---|---|
| T0 | bus `cb` only (link and VRAM of the W7800); `gcnArchName` print on the hub process and on one server; G16 at the TP1 shapes |
| T1 | pair HIP 0 + 1 only (intra-switch); the cross-root pairs and the W7800 pair are stage 2; the three gfx1100 unknowns of 6.4 are tested here on both arches |
| T2 | as written, on HIP 0 |
| T3 | no W2 boot: G7 at the TP1 shapes replaces the W2 trace; the `rocprofv3` trace of a decode step at W = 3, 24, 48 and of a 2048 chunk is taken from the S5-H1 hub |
| T4 | stage 2 (needs HIP 5); S1-H1 replaces its G9 to G12 |
| T5 | as written: replaying hub on HIP 0, four servers on HIP 1-4 |
| T6, T7 | T6-H1 and T7-H1 (S4-H1, S5-H1): one hub at TP1, `island_mode: replicate`, 130 slots per layer; T7-H1 is run as T14 |
| T8 | on the H1 hub (S6-H1); W2 is not needed |
| T9 to T12 | stage 2 (T9 and T11 need two hubs); the stage-1 soaks are the 1 h c8 soak of T6-H1 and a 10-min compiled soak in T7-H1; T10 needs only HIP 6-9 and may run at any time |
| T13 | HIP 0 only |

### 11.1 Numbering against the R3 port plan (G0-G16)

S-1 to S12 (slices) and T0-T16 (tests) are this document's; G0-G16 are the GPU steps of the R3 document section 12, which R3 owns. A check that exists as a G-step is run once, as that G-step; the T-step names it and adds only what is hetero-specific.

| This document | R3 step | Division of work |
|---|---|---|
| T0 preflight | G0, G16 | G0 (arch print on a hub process, mixed-arch fallback log) is part of T0, which adds the same print from a server process and the VRAM and link checks; G16 (QSA route and mrope print, gfx1100 Tensile libraries) runs inside T0 |
| none | G1 | build (fat `_rocm_C`, 177 ops loaded): prerequisite of every GPU step |
| none | G2 to G6 | per-op parity on gfx1100 (GDN with the two inline-asm kernels first, dense/QSA/HC/SE, MoE, skinny GEMMs and top-k, V1 ops): R3 only, all must pass before T4; G4 is also the GPU gate of S2 |
| S1-H1 | G2 to G8, G16, G13 (truncated) | the same G-steps at the TP1 hub shapes on one W7800 (the W2 shapes are TP2's); G9 to G12 and G14 are W2-only and are not run in stage 1; the truncated check follows the G13 method with all experts of the first layers on the W7800 |
| T1 | none | cross-arch IPC, poll visibility, tick rate, cross-root P2P (section 6.4); G9 covers the W7800 pair for `rdna_ar` only |
| T2 | G4, G7 | G4 is correctness (shapes, masking, Q32); T2 adds the timing G7 does not take: streaming read, per-expert MoE GEMV (default and Q32), WMMA GEMM and scalar MoE at M = 2048 |
| T3 | G7, G8 | T3 is G7 (per-op latency against `perop_expectations.py`, default route and family set) plus the `rocprofv3` trace of a W2 step; G8 (top-k algorithm timing) stays R3's |
| T4 | G9 to G14 | T4 = G9 (`rdna_ar` probe), G10 (W2 serve, default route), G11 (family ladder), G12 (determinism, Q32 cost); G13 (cross-arch numerics, needs a V620 process) and G14 (prefix-cache NaN repro) follow; T4 adds the ABBA cells against a V620 TP4 boot |
| T5 to T12, T14 to T16 | none | hetero only (the stage-1 variants of T6 to T8 run on the H1 hub) |
| T13 | G15 | the TunableOp rows for ordinals 0 and 5 are the TunableOp item of G15; its other A/B items (`rms_norm` IR priority, `LLMM1`, `gemv_f16_rdna2`, int8 shadow) stay R3's |

Suggested order on the W7800 pair (W2, stage 2): G1, T0 (with G0 and G16), G2 to G6, T2 with G7 and G8 on one W7800, G9, G10, G11, G12, G13, G14, then the T3 trace and T13; T1 needs one V620 per switch and can run in the same window.
Suggested order in stage 1 (one W7800, HIP 0): Rdna3Port G1, T0, G2 to G8 and G16 at the TP1 shapes with T2, then T13; HxServer T1 (HIP 0 + 1) and T5; HubMethod T6-H1, T7-H1 with T14 (T8, then T15); Rdna3Port's truncated G13 after these windows, or at once when the S4-H1 correctness gate fails.

### 11.2 Results received so far (V620 only; nothing of `hx` has run on gfx1100)

HxServer, `origin/feat/hetero-hx @ 7361027340` (`docs/rdna2/hetero-hx-2026-11-07.md` sections 3, 8 and 10): slot A with a V620 standing in for the hub (hub HIP 1, servers HIP 2 to 4); T1 and the loopback ran at box load 17 to 90, T5 in measured quiet windows. The model rows come from `perf_model.py` (`loopback_rows`, `loopback_refit`, `t5_rows`, `t5_budget`).

| What | Result | Effect on this document |
|---|---|---|
| CPU tests | 743 pass on the box for `tests/distributed/hetero_moe/` (proto, model, ring, ctx, rdv, handshake, server, launcher, T5 harness); 577 on the merged tree with the hub's plan, runtime and method tests | the hx part of the C.9 CPU list is met |
| T1 IPC open | works in all four visibility configurations, the disjoint one included | the disjoint rule of 6.1 stands |
| T1 stores | 1e6 x 16 KiB and 1e4 x 1 MiB byte-exact | gate met |
| T1 16 KiB + flag, one way | p50 6.9 to 7.2 us, p99 7.5 us (flag alone p50 6.4 us) | gate (p50 9 us, p99 20 us) met; model flag 6.3 us |
| T1 poll | round trip p50 12.4 us, p99 24.4 us, never stale or out of order; per-poll acquire loads or fences show a rare 0.5 to 1.8 ms stall (about one per 200k round trips, `evicted_ms` 0), relaxed polling never exceeded 57 us | R24 |
| T1 fan-in 3 x 15.5 KB | p50 13.8 us, p99 14.2 us | gate (4 x 15.2 KiB, p50 15 us) met on three senders; the model's pull term (8.8 us at W = 3) is optimistic |
| T1 fan-in 1 MiB | 22.6 GB/s with three senders in every configuration, gate 24 GB/s | R25 |
| T1 allocation kinds | uncached `coherent=yes`; coarse and fine-grained `coherent=no`, a polled flag there is never seen (8 timeouts) | uncached VRAM only, no fallback (C.2) |
| `wall_clock64` | 99.98 MHz on gfx1030 | the 5 % calibration check stays for gfx1100 |
| Loopback, real kernels | 1e6 exchanges byte-exact; a killed peer is seen in 61 to 126 ms, a hung server ends at the device timeout (3.0 s with a 3 s limit), every process out within 4.7 s | the T5 abort gate (5 s) is met in the loopback |
| Loopback, exchange alone, graph mode, three servers | 35.9 / 64.2 / 98.6 us per layer at T = 3 / 24 / 48 (stream mode p50 44.8 / 72.6 / 104.4 us; T = 2048: 2.68 ms) | the model's exchange for three servers is 27.3 / 53.3 / 82.9 us, so the measured one is +8.6 / +10.9 / +15.7 us (+31 / +21 / +19 %); raising `detect_us` to match moves H1 from 32.0 to 32.3 ms (c1) and from 52.9 to 53.2 ms (c8), inside the green band |
| T5 verify and graph, real weights (Python stack, hub stand-in on HIP 1, three servers, `plan_coldonly`, goldens of the local TP4-EP kernel; `hx_t5_g1`, `hx_t5_r2`) | byte-exact PASS at W = 3 / 24 / 48 for 4 patterns x 48 layers (verify 0.2 / 0.9 / 1.7 s; graph, 8 replays each, 0.4 / 1.6 / 2.9 s); the servers exit 0 after the hub closes and nothing is left in `/dev/shm` | the correctness half of the S3 gate is met on the V620 stand-in |
| T5 fault cases (`hx_t5_f3`, `hx_t5_f4`) | all five PASS: kill-hub, kill-server (hub aborts in 0.38 s), freeze-server (hub aborts in 2.17 s), freeze-hub with a 3 s begin wait (abort phase 3 on the servers), wrong `plan_id`; every process is out within 3.4 s of the fault, except wrong-plan: the hub exits 4 after the 20 s connect timeout (27 s wall) and the servers stay up | the 5 s gate is met; wrong-plan is a bounded handshake timeout, not a step abort (T5 row amended) |
| T5 latency, real weights, graph-mode mean per layer (`hx_t5_r2`) | 152.8 / 379.0 / 569.5 us at W = 3 / 24 / 48 (p50 153.3 / 379.6 / 569.6; eager mean 181.1 / 395.7 / 586.6), 3.25 / 2.35 / 2.22 x the 47 / 161 / 256 us of the first gate | the budget belonged to another plan: see the paragraph below |
| T5 replay, synthetic servers (`hx_t5_s1`) | 86.1 / 99.3 / 134.0 us at W = 3 / 24 / 48 (eager 138.0 / 138.7 / 154.7; step 4.13 / 4.77 / 6.43 ms) | the exchange through the Python stack: +50 / +35 / +35 us per layer over the loopback (R4) |

**T5 latency: the first gate used the wrong model.** The budget of 1.3 x 47 / 161 / 256 us is the model's H2 remote path (the hub holds the hot experts, so the busiest server runs about three experts at W = 3). `plan_coldonly` sends every pair to the three servers, so under the T5 inputs (Zipf 1.1 popularity, Gumbel top-10 per row) the busiest server runs 10.5 / 41.3 / 62.4 experts per layer at W = 3 / 24 / 48 (16 pattern draws x 48 layers counted with `t5_common.make_inputs` and the plan's stripe; `T5_NMAX` in `perf_model.py`). The model of that plan (`t5_rows`: the three-server exchange plus those experts at 6.67 us each) is 97 / 329 / 499 us and the measurement is 1.57 / 1.15 / 1.14 x that. The server chain matches the model: real minus synthetic servers is 67 / 280 / 436 us against 70 / 275 / 416 us (0.96 / 1.02 / 1.05 x), so the V620 expert rate behind `expert_us` (380 GB/s effective, 6.67 us per expert) holds from 10 to 62 experts per layer and the chain needs neither a refit nor an optimisation for this model. HxServer's document (sections 8.5 and 10) reads the result the other way, as a server compute of 67 / 280 / 435 us against a model value of 18 / 97 / 157 us; those model figures are this document's earlier budget-minus-loopback arithmetic (marked [INFERENCE until T5] in the previous revision of this paragraph) for the balanced H2 plan, not for the cold-only plan that T5 ran.

What is left is the exchange: 86.1 / 99.3 / 134.0 us with synthetic servers against 35.9 / 64.2 / 98.6 us in the loopback, +50 / +35 / +35 us per layer or +2.4 / +1.7 / +1.7 ms per step. A candidate is the host-driven wake of S3's servers (C.6, R4: the server host thread polls the ring and launches the step graph; the model had +1.1 ms at c1), a per-step cost that fits all three differences [INFERENCE: HxServer fits `step_us` over 12, 24 and 48 layers per step to split a per-step from a per-layer cost, and checks how the loopback's step loop wakes its servers]. The model plus the measured exchange gives 156 / 375 / 550 us, within 2 to 4 % of the measurement. In the H1 model the larger exchange moves the c1 step from 32.0 to 34.2 ms and the c8 step from 52.9 to 54.4 ms, both inside the green band; with both levers and the G7 routes (11.3) the steps are 24.3 / 45.6 ms against the parity band of 24.6 / 48.4 ms.

T5 gates as amended (S3 and the T5 row): (a) byte-exact verify and graph replays and the five fault cases (met); (b) the server chain, real minus synthetic servers, at most 1.15 x the model's expert time of the busiest server (met: 0.96 / 1.02 / 1.05); (c) recorded, not gated at S3: the exchange with synthetic servers, and the layer time against the plan-specific limit of 1.3 x model (`t5_budget`: 127 / 427 / 649 us for three servers, 113 / 363 / 553 us for four), met at W = 24 and 48 and exceeded at W = 3 (153 us) by the larger exchange. S5 holds the exchange to 1.3 x the loopback (47 / 83 / 128 us) once the wake is device-gated (R4, T14). Not yet run: everything on the W7800 (HxServer runbook steps 3a to 3d: single-GPU gfx1100 loopback, T1 with the W7800 as hub and four V620 senders, loopback soak and kill tests, T5).

HxServer's independent review of `hx_core.cuh` (nothing built or run) and the contract's position on each finding:

- Plan error in `hx_k_dispatch` (finding 1): after the first claim nothing is published (C.7), as in the kernel; the servers end on the host abort poke once the failed hub exits (the engine fails the step and the launcher restarts hubs and servers together) or on their own device timeout. The CPU model `HxSim`, which always publishes, follows the kernel.
- LIVE and RECORD in coarse ctx memory (2): move both into the uncached control page (observability only).
- `ktab` indexed by T (3): C.5 follows the code.
- T argument against the header's T in dispatch and combine (4): add the comparison (abort phase 5, peer 4) if it costs no extra kernel launch.
- Combine on abort (5): the failure signal is the record, which the engine reads before any token is emitted (C.6); NaN poisoning is per element block and best effort, so "every element is NaN" is relaxed to "every element written after the abort is seen".
- Table values not bounds-checked in the kernel (6): the plan loader validates them (`n_cold` at most 8, slot below 512).
- Pinned step ring reused without an event (7): safe while the host runs at most a few steps ahead of the stream (async scheduling keeps it at one or two); keep the depth at least 8.
- Phases 1 and 6 and record bit 2 (8): reserved in C.7, never produced.

Decisions taken here: no heartbeat for a stopped hub in stage 1 (a stopped but live hub is an operator action; `--begin-wait-ms` stays 600 s because the first forward at a new shape may JIT-compile between the announce and `begin_step`; the T5 `freeze-hub` case stops the hub between two layers of a step, which the servers' request wait catches (abort phase 3, every process out in 3.4 s), while a hub stopped between steps is the begin-wait gap of R26 (phase 4), so the harness runs its servers with `--begin-wait-ms 3000`); polling relaxed with one system-scope acquire fence after the flag is seen becomes the default only after T1 on the W7800 (poll variants) and the soak (step 3c) show it ordered and stall-free (R24); the 1 MiB fan-in gate stays at 24 GB/s until the four-sender W7800 measurement (R25).

### 11.3 W7800 per-op results (Rdna3Port, G0 to G8 and G16 on HIP 0: the first W7800 numbers)

Rdna3Port, `origin/feat/rdna3-flashnext @ b964e29f2d` (R3 document section 16; logs `~/w4a8_runs/rdna3_port/gpu/20261107-{143512,154440,155515,160008}/G7-*.log` on par1-cs25): one W7800 (HIP 0, bus `cb`; HIP 5 was never opened), `venv-7.14.0` with the amdsmi stand-in, junction temperature never above 57 C, per-op tests widened to the TP1 and TP2 head counts. G0, G1, G3 to G8 and G16 pass after the test fixes and the runner's family variable; G2 passes except one delta-h element (1 of 3.9 million, the same fp64 error for HIP and Triton) whose widened tolerance has not been rerun yet (R3 document 16.3 and 16.5: the failures were stale tests and runner environment, no gfx1100 kernel defect). Graph-timed microseconds per call at W = 3 / 24 / 48 rows (c1 / c8 / c16); the model column is `perf_model.py` central (`g7_ops`):

| Op (per call) | Route | Measured at W = 3 / 24 / 48 | Model |
|---|---|---|---|
| HC preamble | torch | 32.9 / 88.5 / 90.4 | 46.0 / 42.0 / 42.0 |
| HC preamble | HIP `hc` | 33.2 / 41.7 / 59.8 | same |
| Shared expert, TP1 (intermediate 640) | fused op | 22.8 / 35.3 / 77.2 (torch fallback above 32 rows) | 27.7 / 26.1 / 26.1 |
| Shared expert, TP1 | torch composition | 74.2 / 73.8 / 77.6 | same |
| Shared expert, TP2 (320) | fused op | 15.3 / 23.6 / 61.6 (fallback above 32 rows) | 17.7 / 16.1 / 16.1 |
| Shared expert, TP2 | torch composition | 59.1 / 62.2 / 62.0 | same |
| lm_head, TP1 (N 248,320), ms | `F.linear` | 1.769 / 2.429 / 2.823 | 1.839 |
| lm_head, TP2 (N 124,160), ms | `F.linear` | 0.898 / 1.466 / 1.208 | 0.920 |
| lm_head, ms | `gemv_f16_rdna2` | TP1 1.757 / 6.907 / n.a.; TP2 0.888 / 3.474 / n.a. (32 rows at most) | |
| GDN decode recurrent step per layer, TP1, at 1 / 2 / 3 / 4 / 6 / 8 / 12 / 16 sequences | Triton | 19.3 / 23.5 / 28.9 / 33.3 / 41.1 / 50.7 / 65.5 / 114.1 | inside the attention-core lump |
| the same | HIP `gdn` | 11.6 / 23.6 / 32.9 / 40.4 / 62.6 / 88.3 / 126.1 / 162.6 | |
| Top-k decode (k 2048), 4,096 live columns, us | radix (`auto`) / insertion | 15.1 / 15.2 / 18.6 against 8.9 / 9.0 / 9.3 | equal at 1k and from 16k columns |

What the data says (R3 document 16.4, with the model comparison added here):

- **Routes.** `hc` and `se` are on: HIP HC equals torch at W = 3 and is 2.1x faster at 24 and 1.5x at 48; the fused shared-expert op is 3.3 to 3.9x faster than the torch composition at W = 3 and 2.1 to 2.6x at 24. `gdn` is off from three sequences on: HIP wins at one sequence (0.21 ms per step at TP1, 0.62 ms at TP2), ties at two on TP1 and loses from three or four on, by 1.65 ms (TP1) or 0.45 ms (TP2) per step at c8 and 2.3 or 1.5 ms at c16. `VLLM_TOPK_DECODE_ALGO=insertion` is never slower in this range.
- **Against this document's model.** At W = 3 the measured ops are faster (HC -28 %, shared expert -18 %, lm_head -4 % at TP1); at W = 24 and 48 they are slower: the shared expert +35 % and +196 % (the torch fallback above 32 rows), lm_head +32 % and +53 % (`F.linear` streams 720 GB/s at 3 rows, 523 GB/s at 24 and 450 GB/s at 48), HIP HC +42 % at 48 rows.
- **No TP1-only failure.** Every failure was a stale test or the runner's environment (R3 document 16.3), so R19 is closed for the ops that ran; the layer-truncated end-to-end check stays open.

Model effect (`g7_refit`, `g7_stage2`; HC, shared expert and lm_head replaced, the attention core, router and routed experts still modelled; ms at c1 / c8 / c16):

| Engine | Model | G7, best routes (`hc,se`) | G7, default routes (torch HC and shared expert) | G7, best routes + both levers |
|---|---|---|---|---|
| H1 | 32.0 / 52.9 / 75.3 | 30.5 / 54.3 / 80.2 | 31.4 / 59.5 / 83.3 | 21.9 / 44.2 / 68.8 |
| W2 | 24.7 / 45.7 / 65.0 | 23.2 / 47.2 / 69.9 | 25.4 / 53.7 / 73.0 | 19.4 / 42.7 / 64.9 |
| H2 | 23.3 / 36.1 / 49.6 | 21.9 / 37.4 / 53.0 | 22.7 / 42.7 / 56.1 | 17.6 / 31.8 / 47.4 |

- **H1 gates.** The central H1 is 30.5 / 54.3 ms against the green limit of 36.9 / 60.8 ms; on the default torch routes c8 is 59.5 ms, 1.3 ms under the limit, so `hc,se` are part of the H1 recipe, not an option (the torch HC costs 5 ms of the c8 step). With both levers H1 is 0.98x / 1.01x TP4 slot A (parity needs 1.10x): the levers are worth 8.6 / 10.1 / 11.4 ms because the measured lm_head grows with the rows (the vocabulary-parallel lever alone: 2.8 / 4.2 / 5.2 ms).
- **Stage 2.** D3 moves from 803 to 796 tok/s over 24 users (822 to 818 with W8A16 on W2), D5 + W8A16 from 588 to 583 over 16, so the recommendation of 5.7.5 stands. H2 with both levers is 9.6 ms per token at one user (10.3 in the central model), 7.7 % below the model's TP8 where the rule asks for 8 %, and 3 to 26 % behind the best deployment without `hx` from two users on (5.7.3).
- **Recipe.** H1 and W2 recipes set `VLLM_RDNA3_HIP_OPS=hc,se` and leave `gdn` out (a one-sequence service would gain 0.2 to 0.6 ms per step, not worth a recipe of its own); `insertion` for the top-k after a serve A/B (R3's recipe note).
- **Asks to Rdna3Port** (sent by message): the fused shared-expert op up to the 48-row capture width (about 1.3 ms of the H1 c16 step [INFERENCE: its 22.8 and 35.3 us extrapolated to about 50 us at 48 rows against 77 us]); a look at lm_head at 24 and 48 rows; and the rows the model still lacks: the attention core (GDN `in_proj` and `out_proj`, QSA dense GEMVs, QSA decode at S sequences, indexer), the router and the per-expert W4A16 MoE GEMV (T2). The attention core is 15 / 26 / 40 ms of the modelled H1 step at c1 / c8 / c16 and the largest unmeasured term (R27).
- **R3's window plan** (reply to those asks, `origin/feat/rdna3-flashnext @ 19c506cba6`; written, none of it run: HIP 0 is free for R3 after HxServer's window): the cold `dense` GEMV cells (TP1 and TP2 shapes of `in_proj_qkvz` 16384 / 8192, `in_proj_ba` 96 / 48, the QSA qkv 7168 / 3584, the out and o projections 2560 x 6144 / 3072, the shared-expert gate-up and down and the router 512 x 2560 at M = 3 / 24 / 48, rocBLAS against `gemv_f16` against the small-M kernel); `qsa_decode_bench.py` (store, compress, paged MQA score, top-k, expand and sparse attention at 1 / 8 / 16 sequences x 3 rows, 8k and 32k contexts, TP1 24q/2kv and TP2 12q/1kv heads, cold banks, per-layer microseconds and 12 layers per step, default against `VLLM_RDNA3_HIP_OPS=qsa` and the insertion top-k); `moe_decode_bench.py` (gate GEMM and `topk_softmax`, then the real `Rdna3WNA16Experts.apply` at M = 1, 3, 8, 24, 48, default and Q32, 8 cold banks of 192 experts, also 192 of 512 resident through the expert map, a `VLLM_RDNA3_MOE_BLOCK_M` sweep; it prints distinct experts, microseconds per call and per expert and GB/s); the shared-expert tiles at 24, 32 and 48 rows (`VLLM_RDNA_SE_DECODE_MAX`, 9.6); lm_head at 24 and 48 rows by a TunableOp search (cells `G7-lm_head-tp1-tune` and `-tp2-tune`, then `-tuned` reading the csv back), with a WMMA skinny GEMM only if that finds nothing. The TP4 anchors of the model were measured on the 1k/512 load (contexts of about 1-1.5k tokens), so a 1k cell next to the 8k and 32k ones is what ties the QSA numbers to the model's attention core.
- **Caveat** [INFERENCE]: graph-timed micro benchmarks repeat one layer's weights, so HC (25 MB per layer) and the shared expert (9 MB) can stay in the 96 MB Infinity Cache between calls and read low in a full step, which lm_head (1.2 GB) cannot; T3's `rocprofv3` trace of a whole step is the check.
- **S1-H1 status.** G2 to G8 and G16 are done at the TP1 and TP2 shapes. Open: the layer-truncated end-to-end check (G13: `gpu_g0_g7.sh --only G13`, three engine starts on the first four layers of the real checkpoint, base, all HIP families and WMMA, six prompts and the PPL probe; the step refuses to start below 130 GiB of available host memory because of the pinned 95 GiB PLE table, and the V620 side needs a slot), the delta-h group rerun after the tolerance change, the cold `dense` GEMV cells, and S1b's first GPU run.

### 11.4 Routing statistics (T10: the first real routing numbers)

ExpertStats ran the collector on the V620 TP4 `flashnext-mtp2` server (arm B1 of `tools/rdna/hetero_moe/t10_run.sh`, engine tree `eeec2b0d38`) under the agentic bench (hi2 suite behind `cline_act`, seeds 1-7, 8 workers) and cut two windows at whole passes: W1 (1,851 s, 9,099 decode steps, 207,534 decode rows; 2,690 prefill steps, 3.30 M rows) and W2 (2,304 s, 11,815 decode steps, 278,058 rows; 3,569 prefill steps, 4.41 M rows). Files, plans and simulations are `bench_results/2026-11-07_expert-stats-t10/` on `origin/feat/expert-stats @ 58889261df`: per window a solved H1 plan (130 slots per layer on the stage-1 island: `hub0` = HIP 0, `exp0` to `exp3` = HIP 1-4) and a solved H2 plan (two islands, 115 per hub), each simulated on both windows. The rows below come from those files, from the converted stats (re-derived with `perf_model.py --t10-tables`), from the model, and from fixed-width runs of ExpertStats' `simulate` that I made on a scratch copy of it.

| What | Measured | Reading |
|---|---|---|
| Counts | for all 48 target layers of both windows, decode and prefill, the counts sum to exactly 10 x rows | the count identity of the collector (its validation item 2) holds; its other items are ExpertStats' to report |
| Step widths | 24 rows in 79 % (W1) and 83 % (W2) of the decode steps, 21-22 rows in 14-15 %; widths 3 and 48 do not occur (8 workers x 3 verify rows) | the c1 and c16 rows of the model come from the marginals, not from measured steps |
| Skew | the 64 / 130 / 230 most popular experts of a layer carry 39 / 59 / 79 % of the decode selections (uniform: 12.5 / 25.4 / 44.9 %); the top-130 share runs from 45 to 74 % across layers | a strong skew, stable between the windows |
| Held out | pair hub hit of a plan solved on one window and scored on the other: H1 0.5833 / 0.5863 against 0.5893 / 0.5857 in sample; H2 0.7864 / 0.7876 against 0.7913 / 0.7902 | within 0.006: the S-1 plan-quality gate (within 0.05) is met by a wide margin |
| Prefill | the same plans' hub hit on the prefill selections: 0.375 (H1), 0.611 (H2) | the model's `prefill_hub_share` 0.15 is a choice (the prefill spill K, S7), not a residency limit |
| Against the stripe | `simulate` against the plan's own stripe baseline (the hub holds one expert in four): decode hub hit +0.34 (H1) and +0.55 (H2), busiest-server touches x0.77 and x0.48, step x0.87 and x0.77 (its own exchange term, see below) | stage 1 should start from a solved plan; the model puts the stripe plan 1.8 / 3.0 ms (c8 / c16) above the solved H1 plan |
| Distinct experts per layer | W = 1 / 3 / 6 / 12 / 24 / 48: 10.0 / 28.8 / 54.2 / 97.8 / 164.4 / 250.7 for independent rows (the windows differ by less than 0.3 %); central 10.0 / 29.2 / 55.9 / 102.8 / 173.8 / 282.0 | 0.85 of uniform at 24 rows and 0.80 at 48: the central `locality` 0.90 was 5 % high at c8 and 11 % at c16 |
| Hub's resident share of the touched experts, held out | H1 0.574 / 0.559 / 0.533 / 0.492 / 0.435 and H2 0.781 / 0.771 / 0.755 / 0.726 / 0.682 at W = 3 / 6 / 12 / 24 / 48; the solved plans give the same within 0.002 | the central `hit` 0.60 fits H1 at c1, is 0.11 and 0.17 too high at c8 and c16, and 0.13 and 0.08 too low for H2 |

**What the model does with it** (`Assume(routing="t10")`; `perf_model.py` prints the tables). The TP4 calibration is redone (`s_l` at c8 0.484 to 0.518, the c1 anchor and the c8 anchor hold), so fewer touched experts move cost into the attention scale instead of out of the step. Engine steps in ms at c1 / c8 / c16, central routing then T10: TP8 19.1 / 37.0 / 47.7 then 19.1 / 37.3 / 47.4 (measured 19.59 and 37.2 hierarchical, 41.56 flat); W2 24.7 / 45.7 / 65.0 then 24.7 / 45.7 / 63.5; H1 32.0 / 52.9 / 75.3 then 32.1 / 53.5 / 75.3; H2 23.3 / 36.1 / 49.6 then 23.3 / 36.7 / 50.1. No row moves by more than 2.4 %, the H1 gates stay as set (green 36.9 / 60.8 ms), D3 stays at 803 tok/s and D5 + W8A16 at 587 (588), and H2 + both levers + G7 at one user stays at 0.923 of the model's TP8 (5.7.5). The spill K seeds change: H1 10.3 / 20.5 / 36.5 / 62.7 / 98.1 and H2 7.0 / 12.6 / 22.1 / 37.3 / 57.7 at W = 3 / 6 / 12 / 24 / 48, against 10.5 / 21.0 / 38.5 / 65.2 / 110.0 and 7.2 / 13.0 / 23.1 / 39.1 / 63.5 (up to 11 % lower at 48 rows).

**Cross-check against the simulator, per layer** (MoE microseconds of one hub, plans solved on W1 and scored on W2 at a fixed width, hub 3.7 us and V620 6.7 us per expert, fixed 30.8 / 20.7 us on the hub and 3.1 us on a server; the simulator draws real routing marginals, dispatches with the plan's copies, takes the maximum over the servers per step and water-fills the spill). `simulate` charges an exchange of `2 x (13.3 us + rows to the busiest device x 5120 B / 25 GB/s)` with only the rows that have a pair on a device, while the `hx` protocol pushes all T rows to every server through the hub's single link and pulls T rows back (C.5), `2 x (13.3 us + D x T x 5120 B / 25 GB/s)`. The runs with `hx` exchange use that rule (three lines on the scratch copy, not pushed); the dispatch column changes only how a row picks among an expert's V620 copies:

| Variant | H1 W = 3 | H1 W = 24 | H1 W = 48 | H2 W = 3 | H2 W = 24 | H2 W = 48 |
|---|---|---|---|---|---|---|
| model, T10 routing | 73.1 | 263.4 | 394.3 | 49.9 | 158.9 | 240.5 |
| simulator, its own exchange, `(row + layer) % n_cold` | 68.5 | 240.8 | 376.0 | 50.2 | 133.5 | 192.3 |
| simulator, `hx` exchange, `(row + layer) % n_cold` (the plan format's rule) | 71.6 | 265.8 | 433.5 | 53.1 | 158.7 | 243.3 |
| simulator, `hx` exchange, `(expert + layer) % n_cold` | 71.5 | 254.1 | 390.9 | 52.8 | 155.6 | 232.4 |
| simulator, `hx` exchange, primary copy only | 71.3 | 252.2 | 384.0 | 52.7 | 155.5 | 232.2 |

At W = 6 and 12 the H1 simulator with the `hx` exchange gives 105.6 and 165.1 against the model's 108.6 and 169.7, H2 71.4 and 104.1 against 70.1 and 104.3. Spill (kept) counts of H1 in the simulator with the `hx` exchange: 10.5 / 19.6 / 35.7 / 62.8 / 102.2 at W = 3 / 6 / 12 / 24 / 48 against the model's K of 10.3 / 20.5 / 36.5 / 62.7 / 98.1.

- **The model and the independent implementation agree** once the exchange is the protocol's: with the plan format's dispatch rule the simulator is 2-3 % under the model for H1 up to 12 rows, 1 % over at 24 rows and 10 % over at 48; H2 is within 1.2 % from 6 rows (6 % over at 3 rows, where the busiest-server count is small and fluctuates). With only the primary copy the model is 2-4 % above the simulator for H1 from 24 rows and within +6 / -2 / -4 % for H2. The spill seeds agree within 5 % (H1, up to 24 rows) and about 8 % (H2).
- **The simulator's own exchange term is 3-21 % optimistic** from 6 rows on (H1 at 48 rows 376 against 433 us, H2 192 against 243): its `step_us` and spill counts should not be quoted as hetero numbers until `simulate` charges the broadcast (request to ExpertStats).
- **Replicas cost time in decode under the plan format's rule.** With `(row + layer) % n_cold_copies` a replicated cold expert is streamed on every copy that receives one of its rows: server touches exceed the distinct cold experts by 8 % at 24 rows and 20 % at 48 rows (H1: 23.1 against 20.9 per server and layer at 24 rows, 42.4 against 35.5 at 48). Choosing the copy by expert (no duplicates) lowers H1 by 4 % / 10 % at 24 / 48 rows and the primary copy only by 5 % / 11 % (H2: 2 / 4.5 %), i.e. 0.6 / 2.4 ms of the H1 step at c8 / c16 and 0.15 / 0.5 ms of the H2 step, because the primaries are already balanced by LPT (max/mean touches 1.12-1.19 at H1) and a decode GEMV streams the weights once per copy whatever the row count. Replicas stay useful for prefill, where an expert has hundreds of rows and the work is compute bound. The model has no duplicates, so it matches the primary-only column; decision 15 proposes that the decode tables use the primary copy only.
- **Not said by T10.** The tables treat the rows of one sequence as independent (only marginals and pair counts were recorded, no ids per step), the workload is one agentic bench at 8 workers, and the router ran in V620 numerics (the W7800 hub's router may round differently [INFERENCE: not expected to change the skew]). R7 keeps those.

## 12. Risks

| ID | Risk | Mitigation or test |
|---|---|---|
| R1 | The W7800 dense chain is slower than modelled (`kappa`, `eta_h`); gfx1100 defaults to the Triton/torch route (the fork's HIP families are opt-in, `VLLM_RDNA3_HIP_OPS`) whose launch chain is unmeasured, and the model's `kappa` assumes the HIP chain; port quality decides. Pessimistic H2 c1 is 29.1 ms (+30 % vs TP4) | S1 gate (W2 c1 at most 28 ms on the better route), G7 and G11 per-family A/B (T3 refit); fallback D3 or option (e) |
| R2 | Cross-arch IPC, or kernel stores into a buffer of an invisible device, fail or are slow | T1 first; ladder: shared visible sets; UCX `rocm_ipc` pushes (28.96 us notification against the 6.3 us flag: +22.7 us per crossing, two crossings per layer, about +2.2 ms at c1, arithmetic only); last resort `hipMemcpyPeerAsync` pulls (5.7 GB/s) |
| R3 | Spin-wait kernels wedge a queue or pin a GPU (history: `rdna-ar-wedge-2026-11-04.md`, T44b in `rdna_allreduce.cuh:88-110`); on gfx1100 the poll visibility, the `wall_clock64` tick rate and the uncached pool are unknown (section 6.4), on gfx1030 they are measured (section 11.2) | device timeouts from a measured tick rate, sticky abort and record, uncached VRAM-only flags with no fallback, one waiting kernel per stream, T1, T5 and T12 |
| R4 | Server wake host-driven until in-graph waits work: +1.1 ms at c1 (model); the T5 replay on V620s costs +2.4 / +1.7 / +1.7 ms per step over the loopback at W = 3 / 24 / 48 (11.2), more if the host thread is descheduled | S5 (device-gated wake: T14 holds the exchange to 1.3 x the loopback, 47 / 83 / 128 us); pin the server host thread; HxServer fits `step_us` over 12, 24 and 48 layers per step to split a per-step cost from a per-layer one; `rdna_ar` already captures spin kernels in graphs on gfx10x |
| R5 | Process-global arch (amdsmi handle 0 ignores the visible set, section 8 item 1; fixed on R3's branch, which also needs the `b8b99e3b73` stand-in to start an engine from `venv-7.14.0`), fat-build mistakes, silent zero output from wrong-arch kernels | role assertion, known-answer self-test, disjoint visible sets, `_GCN_ARCH` check in T0 (G0) |
| R6 | KFD queue budget (rule: at most 8 per process, a ninth slowed the full-GPU decode graph by 30 %: `async-stall-profile-2026-11-02.md:884-887`) and `GPU_MAX_HW_QUEUES` interplay on the hub (graph stream, aux stream, IPC, stats) | queue census in T6; no two-lane |
| R7 | The routing skew was unmeasured; T10 measured it on one workload (11.4). Still open: the tables treat the rows of one sequence as independent (the collector records marginals and pair counts, no trace), the only workload is the agentic bench at 8 workers (79-83 % of the steps are 24 rows wide) and nothing is known about drift | The measured skew moves the engines by at most 2.4 % and the K seeds by up to 11 % (11.4), so no decision depends on the open parts; the plan's `estimate` block against the live hub-hit and server-load histograms detects drift (7.3); a routing trace of a few thousand steps (`evaluate_ids`) would settle the independence question |
| R8 | Expectation mismatch: H2 does not raise aggregate throughput (D3 does) and gives no per-user latency gain beyond one user (D5 does) | sections 5.5 and 5.7, decision 1 |
| R9 | Numerics: V620 prefill uses W4A8 sdot4, hub prefill W4A16 (scalar kernel, WMMA after S1b); rounding depends on placement; partial sums are combined over 5 devices instead of 4 ranks | PPL within 0.02 of W2, concurrency probe, fixed placement per run (section 6.9), per-layer hidden-state comparison V620 against W7800 (G13) |
| R10 | W4A16 drafter acceptance drops | T8 gate; the checkpoint's drafter is bf16 (4.69 GiB, 2.34 GiB per hub under partition) until `quant_mtp_experts.py` has produced the W4A16 shard (bf16 route on gfx1100: Triton `fused_moe`, or HIP `moeu`); bf16 costs 1.8-2.4 ms per draft pair at c8 and leaves W2 3.7 GiB of KV (264k tokens, TP4 has 267k) and H2 22.2 GiB |
| R11 | Memory interplay: 95.4 GiB pinned PLE table (47.7 GiB per hub worker at TP2) with IPC buffers and allocator flags; pinned-host eviction history (`pinned-host-evictions-2026-11-05.md`) | `ALLOC_CONF=auto`, int4 sidecar as fallback, `hx` buffers outside torch |
| R12 | Upstream drift of the quant-method API across fork rebases | `HeteroMoEMethod` is the only integration point; CPU reference tests |
| R13 | W7800 link power state and thermals in a server chassis [INFERENCE] | T0 link check, `tempwatch.py`, T12 |
| R14 | Hot re-placement changes rounding and complicates determinism | v1 applies plans at boot only; S8 re-places at idle |
| R15 | TP2 all-reduce across roots slower than the assumed 18 us per call (36 us per layer for two calls) [INFERENCE]; the W2 baseline runs on RCCL because `rdna_ar` is opt-in and its default gate (64 KiB) does not cover c8: at 36 us per call W2 c1 is 26.5 ms and H2 c1 25.1 ms (+1.8 ms), at 40 us +2.2 ms (the G9 note of the R3 document reads 36 us as one call) | G9 measures `rdna_ar` (p50 at most 20 us per call at 8 KiB); T3 records the RCCL time per call; `rdna_ar` widened to gfx11 (`ar` family, `VLLM_RDNA_AR=1`, `VLLM_RDNA_AR_MAX_KB` and `VLLM_RDNA_AR_ONESHOT_KB` at least 128 for c8, 256 for c16) or RCCL |
| R16 | No monokernel on the hub: launch gaps (0.85 us per kernel on V620, `p0-results.md:237`) and graph replay behaviour on gfx1100 are unmeasured | folded into `kappa`; T3 |
| R17 | One failure domain: any of the ten GPUs dying fails the single H2 engine (two independent TP4 engines fail separately) | launcher restart; D3 as the resilient alternative |
| R18 | gfx11 correctness of kernels nothing has run yet: the GDN prefill kernels `gdn_prefill_prep_rdna2.cu:75-79` and `gdn_prefill_kkt_rdna2.cu:78-82` use inline `v_dot2c_f32_f16` on every arch but gfx1200/1201 (whether gfx11 needs `v_dot2acc_f32_f16` is unknown [INFERENCE]); Q32 speed is unmeasured and it returns NaN at or above 65520; the radix top-k default on gfx11 is [INFERENCE] | G2 (these two kernels first), G4, G8, G12 |
| R19 | TP1 hub shapes: GDN with 16 k-heads and 48 v-heads per rank (TP2: 8 and 24), QSA with 24 query heads and 2 KV heads per rank (group size 12 as at TP2, but two KV-head blocks per rank), shared-expert intermediate 640 (TP2: 320, TP4: 160), lm_head N = 248,320. Status 2026-11-07: G2 to G8 and G16 ran on one W7800 with the per-op tests widened to the TP1 and TP2 head counts and no TP1-only failure (11.3); open: the layer-truncated end-to-end check (G13 method) | S1-H1: the truncated check before the stage-1 exit; report any TP1-only failure (message `Rdna3Port`) |
| R20 | Expectation: stage 1 is not a speed deliverable. H1 is modelled 43 % slower than TP4 slot A at c1 (1.20x at c8) and reaches parity only with both levers (5.4a); a result inside the green band is a pass | gates of section 10 (S5-H1), model refit decides stage 2 |
| R21 | Stage 1 takes island 0 (HIP 0 and slot A) from today's production TP4 lane for the duration of the work; capacity falls to slot B's single TP4 engine | leases (section 11), windows scheduled by the lead, T10 and the CPU work run on slot B or on no GPU |
| R22 | The stage-2 premise may fail: the model says engine splits beat H2, D3 on aggregate throughput and from four users on per-user latency, D5 at two to four users (5.7), so the hetero stack may never become the product for this model | 5.7.5 decision rule after S5-H1 and W2; D3 and D5 need W2 and a router only; H1 and H2 stay the way to use these ten GPUs for a model whose experts do not fit the W7800s |
| R23 | Verification gap on one W7800: the full model cannot run there (experts 58 GiB against 44.98 GiB), so S1-H1 proves per-op parity and a layer-truncated end-to-end check only | the first full-model run is S4-H1 with the four servers; the truncated check follows the R3 G13 method |
| R24 | Per-poll acquire loads or fences show a rare 0.5 to 1.8 ms round-trip stall on V620 pairs (about one per 200k round trips, not KFD evictions; relaxed polling never exceeded 57 us); the production wait loop polls with acquire loads | poll relaxed and execute one system-scope acquire fence after the flag is seen (same ordering), proven by the T1 poll variants on the W7800 and the soak before it is the default; until then a rare outlier of about 1 ms [INFERENCE: one step in about 4,000 at 48 layers] |
| R25 | 1 MiB fan-in into a V620 hub reaches 22.6 GB/s with three senders in every visibility configuration (gate 24 GB/s; every link Gen4 x16 with 256 B MaxPayload); a hub-ingress limit is [INFERENCE] | decode rows are KiB-sized (no effect); the prefill star costs 89 ms instead of 81 ms per chunk, not binding (6.8); re-measure with four senders into the W7800 (T1 step 3b) before any change of the gate |
| R26 | A hub that is stopped but alive (SIGSTOP, debugger, hung host thread) is detected by the servers only through the 600 s begin wait when it stops between steps (mid-step the request wait ends the servers in about 3 s: abort phase 3); a dead hub is seen in under 100 ms by `pidfd` | no heartbeat in stage 1; the begin wait must also cover a first-use JIT compile between the announce and `begin_step`; the T5 `freeze-hub` case stops the hub between two layers (phase 3, servers out in 3.4 s) and the harness runs its servers with a 3 s begin wait |
| R27 | The W7800 ops that G7 measured scale worse with rows than the model: `F.linear` for lm_head loses 27 to 37 % of its 3-row bandwidth at 24 and 48 rows, the fused shared-expert op falls back to torch above 32 rows (capture width 48), HIP `hc` is 59.8 us at 48 rows against 42 modelled; the attention core (15 / 26 / 40 ms of the H1 step at c1 / c8 / c16), the router and the dense GEMVs have no W7800 measurement, and the numbers are warm-cache micro benchmarks | `hc,se` in the H1 and W2 recipes; the vocabulary-parallel lever (2.8 / 4.2 / 5.2 ms in the G7 refit); R3 extends the shared-expert op to 48 rows and times the attention core, router and per-expert MoE GEMV; T3's `rocprofv3` trace of a full step checks the warm-cache bias |

## 13. Open decisions for the user

1. **Objective (stage 2).** The model's stage-2 evaluation (5.7) answers by objective: D3 (W2 + 2 x TP4) for aggregate throughput (822 tok/s with W8A16 on W2 at 24 users against 525 for today's 2 x TP4) and for per-user latency from about four concurrent users; D5 (W2 + TP8 on all eight V620) for per-user latency at one to four users (10.8 ms per token at two users against 11.9 for today's TP8);
   H2 with both levers is ahead of both only at one user, by 1 % in the central model and 7.7 % with the G7 per-op numbers (the rule asks for 8 %), and falls behind if the W7800 port is at the pessimistic end or from two users on. Recommendation: after stage 1, build W2 and a router and deploy D3 or D5 per objective; build H2 only if the rule of 5.7.5 fires. Which objective matters, and at what concurrency?
2. **W2 as a product now.** S1 is stage 2's first slice (both W7800, so HIP 5 must be approved) and yields the engine of D3 and D5 and the baseline for everything else. Go/no-go after the T4 numbers.
3. **Hub dense precision.** W8A16 (or W4A16) for GDN/QSA/shared/lm_head on the hub: -3.7 ms c1 / -3.6 ms c8 for H2 in the model (H1: -7.3 / -7.4 ms; W2: -3.9 / -3.9 ms), needs a calibration set and the quality gate (PPL band, probes). Who owns the calibration?
4. **MTP experts.** The checkpoint's drafter is bf16 (experts 4.69 GiB, 2.34 GiB per hub at EP2). Derive the W4A16 shard with `quant_mtp_experts.py` (1.2 GiB; gate: at least 97 % of the bf16 acceptance) or keep bf16 (+1.8-2.4 ms per draft pair at c8; W2 KV 3.7 GiB; the H1 hub fits either, 5.2a: 15.6 or 12.1 GiB of KV); R3's open decision 2. Stage 1 starts with the bf16 drafter if the shard is not ready.
5. **Text-only v1.** The vision tower (0.84 GiB) is excluded and image requests rejected; acceptable?
6. **Hub KV reserve.** The solver reserves 8 GiB of KV per hub (12.1 GiB reserve in total); the model could give 23.9 GiB (1.7M tokens) because more hot experts bring no gain beyond the balance point. Larger KV for prefix caching and long contexts, or keep the VRAM free?
7. **Request router.** D3, D5 (and 2 x H1) need a router in front of two or three engines (different speeds, prefix-cache locality, section 5.7.6); not written. In scope? Recommended: yes, a small proxy in `tools/rdna/` with least-ITL placement and session affinity.
8. **Mono K2 statistics extension** (counting ids inside the persistent kernel): not needed for this plan; only if statistics are wanted from the monokernel recipes.
9. **Schedule.** Stage 1 needs HIP 0 (approved 2026-11-07) and slot A for island 0, with leases and one owner at a time; T10 only needed the V620 and has run (11.4). Stage 2 needs HIP 5 (not approved).
10. **Server lifecycle.** Launcher-managed processes (this design) or servers owned by the vLLM engine lifecycle (a larger change, but one command line)?
11. **S1b first GPU run.** The WMMA W4A16 MoE prefill kernel now exists (`b3e6129c61`, opt-in, CPU-tested, built and oracle-checked, never run on a GPU); W2's prefill gates (chunk at most 0.60 s, 16k TTFT at most 4.5 s) need it, and H2 prefill tolerates a scalar kernel only down to `eff_hub` 0.15 (R3's open decision 3). Run its smoke subset in Rdna3Port's next W7800 window on HIP 0 (no new approval needed), last in the window and in its own process after the cells that feed the model (a first run of a new kernel can hang a queue [INFERENCE]), T2 at M = 2048 after it?
12. **HIP-family defaults and Q32.** After G7 and G11, flip the fork's HIP families to default on gfx1100 per family or keep them opt-in; accept `VLLM_RDNA3_MOE_DETERMINISTIC=1` as the hub default if the G12 cost is acceptable, since the hetero contract requires it (R3's open decisions 1 and 4).
13. **HIP 5 for stage 2.** W2 (S1) and everything after it need both W7800; approve HIP 5 when stage 1 reaches S5-H1 (or earlier for W2 alone, which also gives D3 and D5)?
14. **Stage-2 go/no-go on H2.** The rule of 5.7.5 (refit the model with the stage-1 and W2 measurements; build H2 only if its ITL is at least 8 % below the best deployment without `hx` (D3, D5, TP8) at the concurrency that matters, or if TTFT/KV decide) is proposed; accept, or build H2 regardless for the stack's sake? With the G7 numbers (11.3) the one-user case is 7.7 % below TP8 and every other concurrency is behind, so the rule can only fire for a single-stream latency objective.
15. **Stage-1 end state.** After S10-H1 the island runs H1 as a second product engine next to slot B's TP4 only if the gates of section 10 are met (parity with TP4 slot A, 2.3x its KV capacity); otherwise it is retired and slot A returns to production. Accept?
16. **Decode dispatch uses the primary copy only.** The plan format picks a cold copy by `(row + layer) % n_cold_copies`; with replicas that streams an expert on every copy that receives a row, and the simulator (11.4) puts the cost at 5 % of the H1 MoE layer at 24 rows and 11 % at 48 (0.6 / 2.4 ms of the H1 step at c8 / c16), 2 / 4.5 % for H2, while the primaries are already balanced; replicas stay useful for prefill. Proposal: decode capture widths dispatch to the primary (host side: a second table set with `n_cold` = 1, or a per-call table pointer), eager and prefill keep the row rule; not before S4-H1 and S5-H1 pass on the current rule (T5's plans, `stripe` and `coldonly`, give every expert one cold copy, so the replica rule itself is untested on a GPU). Accept as an S5 item, or change the rule now in C.5?

## Appendix A. Reproducing the model

```bash
uv run --no-project python tools/rdna/hetero_moe/perf_model.py          # engines, components, machine level, memory, wire sizes, hx loopback and T5 comparison, G7 per-op comparison and refit, T10 routing tables and refit, prefill table, scenarios, solver constants, H1 ladder and gates, H1 fit, spill K, stage-2 candidates, ITL against users, H2 ratio, offload bound
uv run --no-project python tools/rdna/hetero_moe/perf_model.py --sens   # the same plus the one-knob sensitivity table
uv run --no-project python tools/rdna/hetero_moe/perf_model.py --json    # everything machine-readable
uv run --no-project python tools/rdna/hetero_moe/perf_model.py --t10-tables W1.stats.json W2.stats.json   # re-derive T10_N and T10_HIT from the converted stats (git show origin/feat/expert-stats:bench_results/2026-11-07_expert-stats-t10/W1.stats.json)
```

The script is stdlib-only and CPU-only. Measured anchors live in its `TODAY` and `OLD_*` constants with their document references; every unmeasured knob is a field of `Assume`. To fold in W7800
measurements, edit `Assume` (or add a field) and rerun: `eta_h`, `kappa`, `gamma`, `comp_ratio`, `eff_hub`, `detect_us` are the ones tests T2, T3, T7, T9 and T14 produce; `LOOPBACK_US` holds HxServer's V620 loopback (11.2) and is replaced by the W7800 numbers of T1 step 3c and T5; `T5_*` hold the T5 measurements and the routing family of its inputs; `G7` holds Rdna3Port's W7800 per-op numbers (11.3), applied to H1, W2 and H2 by `Assume(g7="best")` or `"default"`; `T10_N` and `T10_HIT` hold the measured routing marginals (11.4), applied by `Assume(routing="t10")`.

`tests/tools/rdna/test_hetero_perf_model.py` (40 CPU tests) pins the TP4 calibration (22.38 / 43.97 ms, 262.6 tok/s, 5.52 s), the held-out TP8 prediction, the H1/H2 conclusions of section 5, the spill balance and the spill K defaults,
the byte-table identities, the wire sizes, the scalar-MoE-kernel prefill range, the bf16-drafter KV capacity, the TP2 all-reduce sensitivity, the exchange term and its comparison with the hx loopback, the T5 budget for the plan under test, the G7 per-op refit and its effect on stage 2, the T10 routing tables (their derivation from converted stats, held-out scoring, the shape of the measured skew, the effect on the engine rows, the K seeds and the stage-2 choice), and the stage plan (H1 ladder and gates, H1 memory fit, stage-2 candidates, ITL against concurrency under three W7800 scenarios, D3 against D5, the H2 go ratio, the dense-offload bound, the P/D split bound): `uv run --no-project --with pytest python -m pytest --noconftest tests/tools/rdna/test_hetero_perf_model.py`.

## Appendix B. Source documents for the measured numbers

| Number | Document |
|---|---|
| TP4, TP8, PP2 ITL, throughput, TTFT, KV, weights per GPU | `docs/rdna2/tp8-pp2-flashnext-2026-11-06.md:424-441` |
| Decode-step trace, 380 GB/s expert streaming, HC and FFN block times | `docs/rdna/monokernel/p0-results.md:149-279` |
| PCIe push, pull, latency, fan-in, concurrent cross-root pairs | `docs/rdna2/pcie-iommu-audit-2026-11-06.md` |
| Cross-root IPC with disjoint visible sets, NIXL 25.3 GB/s, 28.96 us notification, the disagg experiment's G0a failure at 16385 tokens | `docs/rdna2/disagg-g0-2026-11-06.md` |
| Offload ITL (52.6-61.1 ms), K2 offload split | `docs/rdna2/moe-offload-topology-2026-11-02.md`, `docs/rdna2/moe-offloading-target.md` |
| `rdna_ar` one-shot and hierarchical all-reduce, one-shot size gate | `docs/rdna2/rdna-ar-hier-2026-11-06.md`, `csrc/rocm/rdna_allreduce.cuh`, `vllm/distributed/device_communicators/rdna_all_reduce.py:18-23,123-147` |
| W7800 and V620 VRAM | `rocm-smi --showmeminfo vram` on par1-cs25, 2026-11-07 |
| gfx1100 per-op status, dispatch policy, build, `rdna_ar` on gfx11, checkpoint and memory facts, per-op W7800 model figures, GPU plan G0-G16 | `docs/rdna2/rdna3-flashnext-port-2026-11-07.md` and `tools/rdna/rdna3/perop_expectations.py` on `origin/feat/rdna3-flashnext @ 30b6a47eca` (sections 2-4, 7-10, 12) |
| W7800 per-op latencies (HC, shared expert, lm_head, GDN decode), top-k algorithm timing, G0 to G8 and G16 results | `docs/rdna2/rdna3-flashnext-port-2026-11-07.md` section 16 on `origin/feat/rdna3-flashnext @ b964e29f2d`; logs `~/w4a8_runs/rdna3_port/gpu/20261107-{143512,154440,155515,160008}/G7-*.log` on par1-cs25 |
| T5 on V620s (verify, graph, fault cases, latency with real and synthetic servers) | `docs/rdna2/hetero-hx-2026-11-07.md` sections 8.4 and 8.5 on `origin/feat/hetero-hx @ 7361027340`; `~/w4a8_runs/hx_t5_{g1,r2,s1,f3,f4}/` on par1-cs25 |
| Routing marginals (T10): windows, plans, simulations | `bench_results/2026-11-07_expert-stats-t10/` on `origin/feat/expert-stats @ 58889261df` (arm B1 of `tools/rdna/hetero_moe/t10_run.sh`, engine tree `eeec2b0d38`) |
| amdsmi handle order on the mixed box | `amd-smi static --asic` on par1-cs25, 2026-11-07 (read-only) |

## Appendix C. H1 interface contract (S2, S3, S4-H1)

Owner: HeteroDesign (this document). Implementers: HxServer (branch `feat/hetero-hx`: `hx` ops, ctx, server, launcher), HubMethod (plan tables, quant method, runner hook, recipe), Rdna3Port (`hub_fused_moe`, section 9.4), ExpertStats (plan producer, `simulate`).
The byte-level source of truth for the wire is `vllm/distributed/hetero_moe/hx_proto.py` (a test keeps it equal to `csrc/rocm/hetero_moe/hx_core.cuh`); the tables below mirror it as of 2026-11-07 and a mismatch is a bug in this document.
Everything here is the H1 case of the general H2 design: one island, one hub (TP1), `ep_size` 1, `island_mode: replicate`. C.8 lists what that choice must not hard-code.

### C.1 Files and ownership

| Path | Owner | Content |
|---|---|---|
| `csrc/rocm/hetero_moe/hx_*.cu`, `hx_core.cuh`; `_rocm_C` registration (`csrc/rocm/ops.h`, `torch_bindings.cpp`, CMake); fakes in `vllm/_custom_ops.py` | HxServer | hub ops, server ops, allocation and IPC helpers |
| `vllm/distributed/hetero_moe/hx.py`, `hx_proto.py`, `ctx.py`, `ring.py`, `server.py`, `launcher.py` | HxServer | public module, constants, ctx packing and handshake, step ring, `ExpertServerModel` and the server loop, launcher |
| `vllm/distributed/hetero_moe/__init__.py`, `plan.py`, `tables.py`, `ref_hx.py` | HubMethod | plan loader (wraps `tools/rdna/hetero_moe/validate.py` semantics), plan to table builder, numpy reference of dispatch, spill and combine |
| `.../compressed_tensors_moe/compressed_tensors_moe_wna16_hetero.py`, `vllm/envs.py` additions, the runner hook (C.6), `tools/rdna/recipes/flashnext-hetero-h1.env`, the `serve_rdna.sh` stanza | HubMethod | `HeteroMoEMethod`, slot-pool weights and loader filter, the env vars of section 9.6, recipe |
| gfx1100 W4A16 MoE (`hub_fused_moe`: Q32, `-1` masking, in-place `slot_map`) | Rdna3Port | contract of section 9.4 |
| `tools/rdna/hetero_moe/solve.py`, `validate.py`, `simulate.py`, `stats_io.py` | ExpertStats | plan producer; `simulate` uses the dispatch rule of C.5 |
| `tools/rdna/hetero_moe/perf_model.py`, this document | HeteroDesign | model and spec |

Tests live next to each owner's code (`tests/distributed/hetero_moe/test_hx_*.py`, `tests/quantization/test_hetero_moe_*.py`). A new test file needs a line in `tools/pre_commit/test_tethering_allowlist.txt` only where `check_test_tethering` says so (directories that a Buildkite job already tethers, such as `tests/quantization/`, do not; `tests/tools/rdna/` does).

### C.2 Processes, visibility and boot (H1)

| Process | Count | Visible set | Role environment |
|---|---|---|---|
| vLLM engine with one TP1 hub worker | 1 | `HIP_VISIBLE_DEVICES=0` (W7800, bus `cb`) | `VLLM_HETERO_MOE_ROLE=hub`, `VLLM_HETERO_MOE_DEVICE_ID=hub0` |
| expert server | 4 | its own V620 (HIP 1 to 4, buses `ce d3 d8 db`) | `VLLM_HETERO_MOE_ROLE=expert`, `VLLM_HETERO_MOE_DEVICE_ID=exp0` to `exp3` |

1. The launcher reads the plan's `islands` and starts one server per V620. Each allocates its inbox in **uncached VRAM** (`hx_alloc`: kind 0 coarse, 1 uncached, 2 fine-grained; default uncached through `VLLM_HX_ALLOC`; no silent fallback, because on V620 pairs a polled flag in coarse or fine-grained memory is never seen while the uncached one always is; never host-pinned, never expandable segments), loads its experts through the V620 repack, runs its self-test, captures its per-bucket graphs and only then writes
   `/dev/shm/hx_<plan_id>/srv_<id>.json` (`<id>` = the plan's device id, e.g. `exp0`; inbox IPC handle, byte size, `gcnArchName`, pid, geometry, allocation kind) and waits.
2. The engine starts; the hub worker allocates its reply buffer (uncached VRAM as well), writes `hub_<id>.json` (e.g. `hub0`), imports every `srv_<id>.json` handle, answers each with `ack_<srv>_<hub>.json` and opens the ring `ring_<hub>.bin`; the servers import `hub_<id>.json`. A server under another plan id never finds its files (a bounded timeout); a plan hash mismatch on a live link is caught by the header check.
3. Self-test before the engine reports ready: a known-answer MoE on every server (per-role `gcnArchName` assertion), the three gfx1100 checks of 6.4 recorded in the log, one dummy step at every capture width.
4. Liveness by `pidfd` both ways; a dead peer fails the hub step or ends the server loop within seconds. `plan_hash = int(plan_id, 16)` (16 hex characters) is in every hdr; a hub and a server on different plans abort at the first step.
5. Every process of the island (engine, hub worker and the four servers) starts from `~/Apps/vllm/venv-7.14.0` with `PYTHONPATH` at the same exact-SHA fat tree (gfx1030 + gfx1100 build that contains R3's amdsmi stand-in, `b8b99e3b73` or later: `venv-7.14.0` ships no `amdsmi` and an engine start dies in the first `get_device_name()` with a `NameError` without it): the other venvs have no gfx1100 device libraries (`hipErrorInvalidImage` on the first W7800 kernel), and an unset `PYTHONPATH` lets `import torch` autoload vllm from the venv's editable tree and abort on a second `_rocm_C` registration. The launcher sets both and checks the imported path in the self-test.

### C.3 Wire layout (`hx` v1)

Constants: `HX_VERSION` 1, `HX_MAGIC` 0x4858_5F50_524F_5431, `HX_MAX_LAYERS` 64, `HX_MAX_SERVERS` 8, `HX_MAX_COLD` 8, `HX_MAX_SLOTS` 4, `HX_MAX_EXPERTS` 512, `HX_MAX_TOPK` 16, `HX_PAGE` 4096; defaults `rows_cap` 2048, `topk` 10, `hidden` 2560, `nslots` 2,
wait timeout 2000 ms (`VLLM_HETERO_MOE_WAIT_MS`), server begin wait 600 s. Sequence `seq(epoch, layer) = epoch x 64 + layer + 1` (u64, epoch starts at 1); a flag value f satisfies a wait for s iff `int64(f - s) >= 0`; flags start at 0; the slot of layer l is `l % nslots`.

| Object | Location and writer | Layout |
|---|---|---|
| inbox | server VRAM (uncached), one per server, written by the hub | offset 0: control page of 4096 B (hdr at 0x000, `req_flag[j]` u64 at 0x100 + 64 j, host abort word at 0xF00); then `nslots` payload slots at 4096 + j x slot_stride, each `x[rows_cap][hidden]` fp16, `ids[rows_cap][topk]` int16 (-1 = masked), `w[rows_cap][topk]` fp32 (16 B aligned); slot_stride 10,608,640 B (10.12 MiB), 21,221,376 B for two slots |
| reply | hub VRAM (uncached), one allocation per hub, written by the servers | control page with `reply_flag[s][j]` u64 at 64 (s x nslots + j) and the host abort word at 0xF00; then `y[s][j][rows_cap][hidden]` fp16 at 4096 + (s x nslots + j) x 10,485,760 B; 83,890,176 B (80 MiB) for 4 servers x 2 slots |
| abort word and record | each process, device memory | C.7 |
| step ring | `/dev/shm/hx_<plan_id>/ring_<hub>.bin`, hub host writes, server host polls | C.6 |

```text
hdr (64 B): u64 magic @0x00 | u64 epoch_abs @0x08 | u32 T @0x10 | u32 bucket @0x14 | u16 opcode @0x18 | u16 version @0x1A | u32 mode @0x1C | u64 plan_hash @0x20 | u32 hub_index @0x28 | u32 rsv @0x2C | u64 rsv @0x30 | u64 hdr_seq @0x38
  opcode 1 = MOE (stage 1); 2 = LMHEAD and 3 = DENSE are reserved for stage 2; mode 0 = decode graph, 1 = eager (bucket 0); hdr_seq is stored last, with release semantics, and equals epoch_abs.
  Unknown magic, version, opcode, plan, a skipped epoch or a T/bucket that differs from the ring descriptor is an abort (phase 5), never a silent skip.
```

### C.4 Torch ops and Python API

```text
torch.ops._rocm_C (fakes in vllm/_custom_ops.py; ctx = the int64 device tensor of HxHub.ctx or of the server)
hx_begin_step(Tensor! ctx, int T, int bucket) -> ()           # hub, once per forward: epoch += 1, hdr into every inbox; bucket 0 = eager step
hx_dispatch(Tensor x, Tensor topk_ids, Tensor topk_w, Tensor! hub_ids_out, Tensor! ctx, int layer) -> ()
hx_combine(Tensor y_hub, Tensor! out, Tensor! ctx, int layer) -> ()           # out may alias y_hub
hx_status(Tensor ctx) -> Tensor                                               # CPU int64[8]: abort code, epoch, live word, steps, dispatches, combines, hub pairs, remote pairs
hx_srv_begin(Tensor! ctx) -> ()
hx_srv_wait(Tensor! ctx, int layer, int T) -> ()                              # waits for the request; widens ids int16 -> int32 and copies x and w into cached workspaces
hx_srv_push(Tensor y, Tensor! ctx, int layer, int T) -> ()
x, y_hub, out: fp16 [T, 2560] contiguous; topk_ids, hub_ids_out: int32 [T, 10]; topk_w: fp32 [T, 10]; T = x.size(0) on the hub, explicit (static per captured bucket) on the server
```

Python (`vllm/distributed/hetero_moe/hx.py`): `HxHub.create(plan, hub_index, widths)` allocates the reply buffer, imports the servers' handles and builds `ctx` (setup ops used by `ctx.py` only: `hx_alloc`, `hx_open`, `hx_view`, `hx_calibrate`, `hx_ctx_new`, `hx_ctx_set`, `hx_ctx_get`, `hx_ctx_register`); `HxHub.ctx`; module functions `hx.begin_step`, `hx.dispatch`, `hx.combine`, `hx.status` wrap the ops;
`HxHub.post_step(T, bucket)` appends the step to the ring (host, before the graph launch, bucket 0 = eager); `HxHub.check()` reads the status and raises `HxAbort` (with the record) on a non-zero abort code; `HxHub.layer_stats()` returns int64 `[nlayers, 3]` (hub, cold, spill pairs; only with `create(..., layer_stats=True)`, else None) and `reset_layer_stats()`; `HxHub.close()`. Servers run as `python -m vllm.distributed.hetero_moe.server --plan P --device-id exp<k>`, started by `launcher.py`.
The layer index passed to the ops is the target layer l in 0 to 47, taken from the module prefix `layers.<l>.`; the MTP layer (plan layer 96, section 9.2) never calls `hx`.

### C.5 Plan to tables, and the dispatch rule

`tables.py` builds, per hub h, from the plan (numpy, then device tensors that `ctx` points at): `slot_map[48][512]` int16 (hub slot or -1; the live per-layer int32 `[512]` tensors handed to `hub_fused_moe` are updated in place and agree with it);
`owned[48][512]` uint8 (1 = the expert belongs to this hub's island; all ones in H1); `n_cold[48][512]` uint8; `cold_srv[48][512][8]` int8 (server index within the island in the plan's listed owner order, -1 padding);
`ktab[rows_cap + 1]` int16, built by `ctx.py` from `IslandTables.k_for` and indexed by the row count (`ktab[T]`, entry 0 unused): the spill count K of the smallest capture width that holds T rows, the prefill K above the largest width (stage 1 sets it to no spill); the per-width values come from `runtime.spill_k` or `VLLM_HETERO_MOE_SPILL_K`. Defaults are the cost model's balance points (`perf_model.py spill_k`, 6.2): width 3 / 6 / 12 / 24 / 48 gives
K = 10 / 21 / 39 / 65 / 110 for the H1 hub (it covers all 512 experts) and 7 / 13 / 23 / 39 / 63 for each H2 hub (central routing; a plan solved on the T10 statistics should carry the measured-routing seeds 10 / 21 / 37 / 63 / 98 and 7 / 13 / 22 / 37 / 58 as `runtime.spill_k`, 11.4). Layer l of every table is the target layer index.

Per layer l and pair (row r, slot k) with id e, exactly (the reference is `ref_hx.py`; ExpertStats' `simulate` and the kernels implement the same rule):

- valid: `0 <= e < 512` and `owned[l][e]`; any other pair is dropped on this hub (`hub_ids_out = -1`, nothing pushed; in H2 the other island's hub computes it).
- resident: valid and `slot_map[l][e] >= 0`. `touched` = the distinct ids among resident pairs; `rank(e)` = the number of touched ids with a smaller slot. A resident pair stays on the hub iff `rank(e) < K`.
- otherwise (valid, not kept) the pair runs on server `cold_srv[l][e][(r + l) % n_cold[l][e]]`; `n_cold == 0` there is a plan error (abort phase 7: sticky abort, NaN output, never a dropped pair).
- `hub_ids_out[r][k] = e` if kept else -1. Per server s: `ids16[s][r][k] = e` if the pair goes to s else -1; `w` is copied unchanged and all T rows of `x` go to every server.
- every server replies every layer with the full `y[T][2560]` (zeros for rows without a pair).
- combine: `out = fp16(((f32(y_hub) + f32(y_0)) + f32(y_1)) + ... + f32(y_{S-1}))`, left to right in fp32; on timeout or abort every element is NaN (fp16 0x7E00).

### C.6 Step protocol and hooks

- Hub per forward: `hx_begin_step(T, bucket)` once at layer 0 (epoch + 1, hdr into every inbox, then `hdr_seq`), then per layer `hx_dispatch` (payload from all blocks, fence, last block stores `req_flag[l % nslots] = seq(epoch, l)` in every server) and `hx_combine` (waits `reply_flag[s][l % nslots]`
  for s = 0 to S-1 in order). Padded rows carry ids -1 and are not dispatched, but all T rows of `x` are pushed so the shapes stay static under capture.
- Server per step (S3, host-driven wake): the server host thread polls the ring, launches the captured graph for `bucket` (decode) or enqueues the eager 48-layer chain (`bucket` 0, T rows); in the graph `hx_srv_begin` waits for the hdr (checks magic, version, opcode, plan hash, epoch, T and bucket against the ring descriptor),
  `hx_srv_wait(layer, T)` waits for `req_flag`, then the MoE chain of the V620 TP4 kernels runs on the inbox slot, `hx_srv_push(y, layer, T)` stores y into the hub reply slot and the flag. S5 replaces the ring by a device-gated wake without changing any layout.
- Ring (`ring_<hub>.bin`, hub host writes, server hosts poll): a 64 B header (magic, depth 64, closed, head = steps posted) and 32 B entries `{step_no u64, epoch_abs u64, T u32, bucket u16, opcode u16, check u64}` at slot `(step_no - 1) % depth`. The writer stores the entry, then `head`; a reader that sees `head >= n` verifies `check` (a torn read is retried) and a reader more than `depth` steps behind gets an overrun error, never a misread entry. The hub host mirrors the epoch (one step = one `begin_step`).
- Step announcement (HubMethod, a no-op when `VLLM_HETERO_MOE=0`), in two places so that every real execution is announced exactly once: the runner hook in `vllm/v1/worker/gpu/model_runner.py` `execute_model`, next to `self._ple_offload_connector.prepare_forward(...)` (line 2123 at `9b222fed61`, before `runtime_mode = batch_desc.cg_mode`),
  calls `post_step(input_batch.num_tokens_after_padding, bucket)` with `bucket` the padded width for a graph replay (FULL or PIECEWISE) and 0 for an eager step, dummy runs included; the `begin_step` wrapper posts for eager launches that do not go through `execute_model` (the warm-up passes of `CudaGraphManager.capture`, profile runs) and never during capture itself.
  Without the second place a capture warm-up would hang the hub (no server wake). After the step, `hx_status` (a side-stream copy, never a sync of the compute stream) is read once the sampler has synchronized; a non-zero abort code raises and the engine fails the step.

### C.7 Failure and abort record

A waiting block claims the per-process abort word (`atomicCAS` 0 to 1) on the first timeout and writes the record; after that every `hx` kernel of the ctx is a no-op (dispatch and push do nothing, `hub_ids_out = -1`, combine writes NaN per element block, best effort: the record is the failure signal, 11.2; `hx_srv_wait` writes ids32 = -1). Timeouts are device wall-clock ticks from a rate calibrated at init against the host clock (refused outside 5 % of the expected rate).
A host abort word at offset 0xF00 of the process's own control page (`HX_CTRL_HOST_ABORT`) is poked by the pidfd watcher when a peer dies: a spinning wait sees it within about 100 us and ends with phase 8 without a second claim (61 to 126 ms from the death in the loopback).

| Record field (u64) | Bits |
|---|---|
| valid, live (live word only), peer seen (reserved, never set) | 0, 1, 2 |
| phase | 4 to 7: 1 dispatch arrival barrier and 6 push arrival barrier (reserved, never produced: v1 has no inter-block barriers), 2 combine wait for a server (peer = s), 3 server request wait, 4 server begin (hdr) wait, 5 hdr mismatch (peer: 1 magic, 2 plan, 3 epoch, 4 T, 5 bucket, 6 opcode, 7 version, 8 skipped epoch), 7 plan error, 8 host abort poke |
| peer index | 8 to 15 |
| measured wait in ms (saturating) | 16 to 31 |
| low 32 bits of seq (low 6 bits = layer + 1; the raw epoch for phases 4 and 5) | 32 to 63 |

A second word `live` (same layout, bit 1 set) is refreshed by a waiting block about once a second; it is not cleared after a successful wait, so it shows the last slow wait and is informational (HxServer's review proposes moving it and the record into the uncached control page so that a hung wait is visible before its kernel exits).

### C.8 What stage 1 must expose so that stage 2 needs no rewrite

1. **No hard-coded island, hub or TP size.** The plan loader and tables accept any number of islands and hubs per island, `island_mode` replicate or partition, and global device indices in `owners`; the hub index is the TP rank, the island is the EP rank; `apply()` returns the island's sum
   (the complete MoE output in H1, a partial sum in H2) and never calls an all-reduce. H2 is then `islands = 2`, `island_mode = partition`, TP2 and a second hub worker, with the same code.
2. **Per-hub objects.** Each TP rank creates its own `HxHub(hub_index = rank)` with its own ring, reply buffer and server list; servers are island-local (hub k and its own four V620). Servers shared by two hubs (H2' and D4) need an inbox region per hub and are not part of any recommended stage-2 mode, so they are not built.
3. **Opcode registry.** The server loop dispatches on `hdr.opcode` through `server.register_op(opcode, handler)`; `ExpertServerModel` keeps a name-to-tensor weight registry per opcode. Stage 1 registers MOE only; the vocab-parallel lm_head lever (opcode 2) and any dense shard (opcode 3) are later handlers. An unknown opcode stays abort phase 5.
4. **Launcher driven by the plan.** The island list, the server buses and the hub bus come from the plan (`VLLM_HETERO_MOE_SERVERS` is only the override); `PCI_BUSES` and the recipe come from the island, so adding a second island is a recipe change.
5. **Static limits are constants, not assumptions:** servers per hub up to `HX_MAX_SERVERS` 8, `rows_cap` 2048, capture widths from the recipe (3, 6, 12, 24 and 48).
6. **Slot-map flip and plan swap** stay in place updates of existing device tensors (S8), and the plan hash is the only identity.
7. **If stage 2 is D3 or D5** (section 5.7.5) none of the `hx` code runs in production: both need W2 (the R3 port at TP2), the V620 engines (two TP4 or one TP8) and a router. Stage 1 then leaves the stack, the measurements and a model refit.

### C.9 Tests by owner

- **CPU, before any GPU.** HxServer: `hx_proto.py` equals `hx_core.cuh` (constants and offsets), the wrap-safe sequence compare, ring torn-read handling, abort record encode and decode, hdr validation cases. HubMethod: plan loader and negative cases (a server on another switch than its hub, `n_cold == 0` for an owned
  expert that is not resident, unknown format version), table shapes and dtypes, the dispatch reference against a plain loop and against `tools/rdna/hetero_moe/simulate.py`, the spill rule (bitmap and prefix popcount against a sorted-rank loop), the partial-sum identity (kept plus remote equals the full MoE) over a CPU W4A16 MoE reference, `owned` dropping,
  the in-place `slot_map` flip. One shared golden-vector file written by `ref_hx.py` (plan, `x`, ids, weights, expected `hub_ids_out`, per-server `ids16`, combine result) is consumed by the GPU tests.
- **GPU.** T1 and T5 (HxServer), G4 and G12 (Rdna3Port), T6-H1 and T7-H1 (HubMethod); section 11 has the procedures, the lease rules and the order of W7800 use (Rdna3Port, then HxServer, then HubMethod).
