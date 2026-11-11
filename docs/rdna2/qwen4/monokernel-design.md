# Qwen4 monokernel design for gfx1030

Status: **design**, written before any Qwen4 weight exists. Qwen4 is
unreleased: no checkpoint, no `config.json`, no model card. Everything about
its geometry is either the published Qwen3.8-Flash-Next (`Qwen4Exp`) config,
which the live monokernel already serves, or a leaked, unconfirmed kernel
description. Items that cannot be settled before release are marked
**[UNCONFIRMED]**; items this page derives rather than observes are marked
**[INFERENCE]**.

Scope: a persistent fused decode path for the Qwen4 family on 8× Radeon PRO
V620 (gfx1030), opt-in, with the existing `Qwen4Exp*` monokernel untouched.
Code that exists today lives on `feat/qwen4-prep` (planner plus CPU tests);
the kernel work is specified here and starts after the release.

## 1. The monokernel that ships today

The Flash-Next (`Qwen4Exp`) monokernel is selected by `--monokernel` (or
`VLLM_ROCM_MONO_DECODE=1`) and described for users in
[`vllm/models/qwen4_exp/amd/mono/rdna2/README.md`](../../../vllm/models/qwen4_exp/amd/mono/rdna2/README.md).
It follows the upstream MonoKernel contract: decode only, default off,
default-deny, bounded cross-workgroup waits with a sticky error word, fail-stop
instead of silent fallback
([RFC](https://github.com/vllm-project/vllm/issues/60904),
[switch PR](https://github.com/vllm-project/vllm/pull/61082),
[templates #60397](https://github.com/vllm-project/vllm/pull/60397) and
[#61325](https://github.com/vllm-project/vllm/pull/61325)).

### 1.1 Switch, refusals, parts

`vllm/models/qwen4_exp/amd/mono_decode_rdna2.py` owns the switch. Startup
refuses with `ValueError("--monokernel <reason>.")` when the GPU is not
gfx10x, the model is not `Qwen4Exp*`, Model Runner V2 is off, PP > 1, the KV
cache is not FP16/BF16, the `_rocm_C` ops are missing, or the MoE is not
symmetric W4A16 on `CompressedTensorsWNA16RDNA2MoEMethod`. It also raises the
workers to `GPU_MAX_HW_QUEUES=4` and forces `VLLM_RAISE_ON_LOGIT_NANS=1` so a
poisoned output fails loudly.

Four parts, each with an independent width gate and its own fallback:

| Part | Driver | Kernel | What runs | Rows |
|---|---|---|---|---|
| `k1` | `mono/rdna2/k1.py` | `mono_k2_rdna2.cu` (`hc_only`) | attention-side HC combine, grouped RMSNorm, down+inject, up-mix | ≤ 24 |
| `k2` | `mono/rdna2/k2.py` | `mono_k2_rdna2.cu` | the whole FFN block: HC, router, shared expert, routed W4A16 experts | ≤ 8 (MTP ≤ 24) |
| `gdn` | `mono/rdna2/gdn.py` | `mono_gdn_rdna2.cu` | GDN decode core: conv update, gated delta rule, gated RMSNorm | ≤ 64 |
| `qsa` | `mono/rdna2/qsa.py` | `mono_qsa_decode_rdna2.cu`, Triton fusions | indexer epilogue, indexer scoring, sparse decode, norm+RoPE | 3–64 |

`layout.py` resolves the per-layer weight views, allocates one workspace per
device and scope, and attaches the all-reduce.

### 1.2 Stage graph per decoder layer

The decoder layer (`Qwen4ExpDecoderLayer.forward`) calls a part or a module at
each step; K1 replaces `attn_hyper_connection.combine_and_mix`, K2 replaces
everything after attention.

GDN layer (36 of 48):

```text
[ple add] -> k1 -> gdn core -> attn all-reduce -> k2 -> [ffn all-reduce]
```

QSA layer (12 of 48, every fourth layer):

```text
[ple add] -> k1 -> qsa prepare -> indexer epilogue -> indexer scoring
          -> top-k select -> sparse decode -> output gate/o_proj
          -> attn all-reduce -> k2 -> [ffn all-reduce]
```

The PLE add fires only where `ple_layer_ids` says so (layer 2, 1-based, on
Flash-Next) and stays outside the kernels: it adds to the multi-stream state
before the pending HC combine is materialized.

K2's internal phases, one persistent launch:

| Phase | Work | Replaces |
|---|---|---|
| P1 | HC combine + grouped Gemma RMSNorm | 2 fills + `_hc_combine_norm` |
| P2 | HC down+inject GEMV (`[336, 10240]`) | `gemv_act_k` / split-K + Tensile |
| P3 | HC up GEMV + sigmoid + gated mean | `hc_up_gate_mix_k` / prefill variant |
| P4 | router GEMV, shared-expert gate_up + SiLU, expert gate | router GEMV, `se_gate_up_silu_k` |
| P5 | top-k (ballot radix select), expert lists, routed gate_up + SwiGLU, shared down | topk, align, count/sort, MoE GEMM 1, act_and_mul, se down |
| P6 | routed down × top-k weight | MoE GEMM 2 |
| P7 | `out = fp16(shared + routed)` (rank partial), optional all-reduce | Triton add |

That is ~21 launches at M ≤ 8 and ~28 at M > 8 replaced by one, which is the
part's main value; the phase timings are in
[`../monokernel/2a-results.md`](../monokernel/2a-results.md).

### 1.3 Persistent kernel and barrier scheme

- One plain (non-cooperative) launch, one stream, graph-capturable. Grid =
  residency cap from the code object: `min(by_waves, by_lds) * 2 * WGPs`,
  where `by_waves = 2 * min(16, 1024/VGPR) / 8`. For the K2 build (128 VGPR,
  29 KiB LDS) that is 2 workgroups per CU, **144 workgroups × 256 threads**
  on 72 CUs, i.e. 16 waves per CU. `hipOccupancy*` under-reports by 2× on
  gfx10 and is not used.
- Cross-workgroup phases are separated by a grid barrier on a generation
  counter in global memory (`count`, `gen`), agent-scope atomics, relaxed
  polling with `s_sleep`. Not `grid.sync()`: it is unbounded and not
  capturable with a partial grid.
- Every wait is bounded by `timeout_ticks = timeout_ms × wall-clock kHz`,
  with `timeout_ms = 2000` from the K1/K2 call sites. On expiry the first
  workgroup claims a sticky `{code, phase, wg}` record, every workgroup fills
  the caller-visible outputs with NaN and leaves. Later launches return at
  entry until the engine restarts (fail-stop).
- Cross-workgroup sums use 64-bit fixed point (`Fx`, 2^-32 resolution) with
  native integer atomics, because gfx1030 has no native global fp32 atomic
  add; the result is deterministic and independent of arrival order.
- Production MTP launches a same-stream async memset of only `count`/`gen`
  before the kernel; the error record stays sticky.

### 1.4 All-reduce integration

Two collectives sit in the layer, both `RowParallelLinear` results:

1. attention output all-reduce, always outside the monokernel (the GDN/QSA
   parts see the reduced output);
2. FFN all-reduce, outside by default.

`rdna_ar` is a push-based one-shot all-reduce for ≤ 64 KiB messages: per-rank
uncached staging slots (peer stores are visible, coarse-grained memory is
not), per-rank uncached flag pages, the sequence number derived on-device
from a local counter (a kernel argument would freeze at graph capture), a
fixed rank-order fp32 sum so every rank is bit-identical, bounded spins, and
a host-mapped abort record with phase/peer/sequence and the measured wait.
A peer that has not reached the collective is waited for up to 600 s, past
the engine's 300 s execute timeout, because a rank can legitimately arrive
seconds or minutes late (host stall, JIT); a peer that announced but whose
write never landed aborts after 10 s. Waits over 2 s are logged as late peers
([`rdna_allreduce.cuh`](../../../csrc/rocm/rdna_allreduce.cuh),
[wedge investigation](../rdna-ar-wedge-2026-11-04.md)).

With `k2_ar` on, P7 does the FFN all-reduce itself: push the fp16 partial into
every peer's slot, one system fence per workgroup, publish the flag after a
grid barrier, WG0 alone polls the peers, then a grid barrier releases the grid
to sum the slots in rank order. Its wait uses the same 2 s budget and writes
phase 8 into the error record. The option is off by default and was not
required for the measured wins.

### 1.5 MoE path and expert offload

Routed experts are symmetric W4A16, exllama-shuffled int4 (`w13 [E, H/8, 2I]`,
`w2 [E, I/8, H]`) with per-group fp16 scales and synthesized symmetric zeros.
The kernel holds 128 local experts resident (`kLocalE`); the router is
`[512, 2560]` fp16, top-10 with renormalization, shared expert 160 local rows
at TP4, and local experts must be one contiguous `expert_map` block.

The expert cache (`--additional-config rocm_mono_decode.expert_cache_slots`)
moves routed weights to host and runs the FFN in three captured stages: 1
(prefix, reads the captured inputs), 2 (resident or miss pass), 3 (finish),
with the host planning between them through a breakable CUDA graph. A demand
that cannot outgrow the cache runs as one pass whose two launches each count
the whole demand against a zero-weight dummy slot, so the sum is one full
pass to the bit; a larger demand is split once the host knows the routing.
Measured: +15.7 % decode tok/s and −13.6 % ITL at TP2/MTP2 with 96 slots
([`../moe-offloading-improvements.md`](../moe-offloading-improvements.md),
[`../moe-offload-topology-2026-11-02.md`](../moe-offload-topology-2026-11-02.md)).

### 1.6 QSA handling

The QSA layer's indexer selects at most `indexer_budget` (2048) positions over
the compressed cache (`indexer_compress_ratio` 4), with 4 indexer query heads,
1 KV head and head_dim 128; the main attention is 24 query / 2 KV heads at
head_dim 256 on Flash-Next. The monokernel contributes:

- a bounded indexer: scoring is bounded to the live context (QSA runs as an
  eager break) and tiles past it exit early, valid in graphs;
- one HIP epilogue launch for q norm+RoPE, group pooling, k norm+RoPE and the
  three cache stores;
- HIP indexer scoring (`mono_qsa_mqa_rdna2`) with a per-call shape policy;
- one HIP sparse-decode launch (split + merge) from 3 rows, lane owning 8 head
  dims, one wave walking every 8th selected token, K/V read once per KV head
  group, splits merged in LDS by the last arrival;
- fused norm+RoPE for the main q/k.

The HIP sparse decode requires head_dim 256, `group = Q_heads/KV_heads ≤ 8`,
FP16 K/V, 3–64 rows. At TP4 Flash-Next has group 3; at TP2 the same
checkpoint gives 12 query heads and **1** KV head per rank, group 12, so the
Triton kernel runs there instead [INFERENCE from the sharding rule; the
HIP gate refuses it].

### 1.7 GDN handling

One launch per linear-attention layer for conv update + gated delta rule +
gated RMSNorm. One workgroup per (sequence, value head, 32-row value tile),
256 threads: the 256 q|k channels of the key head and the tile's 32 value
channels are rolled through LDS; the fp32 recurrent state tile lives in
registers; per-token state slots and the `num_accepted - 1` initial state
handle MTP verify. Head dims are 128/128, conv width 4, ≤ 64 rows, ≤ 8 tokens
per sequence. State is exact (conv) and ≤ 7e-8 (SSM) against the modular
path ([`../monokernel/2a-results.md`](../monokernel/2a-results.md)).

### 1.8 MTP

MTP=k changes the capture ladder, not the kernel: widths `(k+1) × {1,2,4,8}`
(k=2 → 3, 6, 12, 24). The drafter is a separate one-layer model with its own
`VllmConfig` and its own workspace, so its sticky error cannot latch the
target's. K1/K2 serve it only behind `mtp_drafter`; K2 then needs either a
supported symmetric W4A16 drafter or the experimental dense FP16 route, which
is hard-coded to `w13[128,1280,2560]`/`w2[128,2560,640]`, the unquantized
Flash-Next drafter geometry. Enabling the route requires
`VLLM_RAISE_ON_LOGIT_NANS=1` and a same-stream control reset; the measured
value was not established (see
[`../monokernel/README.md`](../monokernel/README.md) and the MTP section of
[`MTP-MONO-HANDOVER-2026-10-24.md`](../../../MTP-MONO-HANDOVER-2026-10-24.md)).

### 1.9 Hardware queues

`GPU_MAX_HW_QUEUES=4` is required with `--monokernel`. With two HSA queues the
CLR shares queues between streams ("lowest load" at stream creation), the
attention kernels of one to three TP ranks ran ~10 % slower next to the
persistent kernel, and which ranks lose changes per boot. The engine raises
the budget to 4 for the workers; recipes that set 2 for prefill reasons
(`flashnext-mtp2`) are overridden
([`../async-stall-profile-2026-11-02.md`](../async-stall-profile-2026-11-02.md)).

### 1.10 Known limits

| Limit | Today | Consequence |
|---|---|---|
| K1 rows | 24 | above the cap the layer runs the modules |
| K2 rows | 8 target / 24 MTP | the FFN block falls back above it |
| GDN rows | 64 | prefill and wide batches use the modular core |
| QSA rows | 3–64, group ≤ 8, head_dim 256 | TP1 and TP2 Flash-Next lose the HIP decode |
| Local experts | 128 resident, 256 in the offload stages | TP2 needs the expert cache (256 × 2.55 MB × 48 ≈ 31 GB) |
| Barrier timeout | 2 s | a host stall longer than that poisons logits: a boot that pinned ~90 GB held a rank 22 s at a collective and one K2 split measurement returned NaN ([`../moe-offload-topology-2026-11-02.md`](../moe-offload-topology-2026-11-02.md)) |
| Graph modes | FULL / PIECEWISE / eager (configurable) | off-graph shapes take the modules |
| Deployment | Qwen4Exp only, MRv2, PP=1, FP16/BF16 KV | everything else refuses at startup |
| MoE format | symmetric W4A16; W4A8 prefill (`VLLM_RDNA2_W4A8_SDOT4=1`) runs next to it, from 512 tokens, on the same packed weights | asymmetric zeros, FP8, MXFP4 fall back or refuse; the expert cache refuses W4A8 |

Performance context, so the Qwen4 design does not over-promise: the FFN-only
kernel was neutral-to-negative on its own ([`../monokernel/2a-results.md`](../monokernel/2a-results.md));
the all-on monokernel (K1 + GDN + QSA + K2) reached −1.4 % to −12.2 % ITL
across eight MTP0/MTP2 cells, and the merged tree measured 85.02 → 86.16
decode tok/s at 16k/1k c=8. MoE offload, not the kernel, is the large win at
TP2. On TP4 `flashnext-mtp2` with the W4A8 prefill, mono + W4A8 beats the shipped
recipe by 0.6-3.2 % ITL p50, but it is 1-6 % slower than production at the same
4 queues ([`../mono-w4a8-2026-11-05.md`](../mono-w4a8-2026-11-05.md)).

## 2. Gap analysis

Inputs: the published Flash-Next config (`model_type: qwen4_exp`, the only
`qwen4*` config that exists —
[HF config](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16/blob/main/config.json),
[transformers model doc](https://huggingface.co/docs/transformers/main/model_doc/qwen4_exp),
[transformers PR 48337](https://github.com/huggingface/transformers/pull/48337))
and the leaked Qwen4-27B description from the Intel XPU kernel work
([llama.cpp issue 31072](https://github.com/ggml-org/llama.cpp/issues/31072))
**[UNCONFIRMED]**. Upstream vLLM has no Qwen4 model class: everything named
"Qwen4" there is `qwen4_exp`.

### 2.1 What carries over unchanged

| Item | Why |
|---|---|
| Barrier, fixed-point reductions, sticky errors, poison + fail-stop | model-independent |
| Workspace/offset scheme, per-scope workspaces, graph-mode gates | model-independent once sizes come from the plan |
| `rdna_ar` one-shot and the P7 fused FFN all-reduce | message size ≤ 64 KiB holds for H ≤ 8192 at M ≤ 48 |
| K1 phase set (combine, norm, down+inject, up-mix) | HC is the same 4-stream, rank-320 residual on both known geometries |
| K2 HC phases P1–P3 and shared-expert phases | same layout, only row counts and widths change |
| Expert cache stages 1/2/3, dummy slot, one-pass accounting | layout-driven, not model-specific |
| GDN kernel | already parameterised by heads/dims; Flash-Next and the leaked 27B are both d128, conv 4 |
| QSA HIP sparse decode and indexer kernels | same semantics per the leak (budget 2048, compress 4, head_dim 256) |
| MTP worker/workspace separation and NaN gate | structural |

### 2.2 What needs parameterisation

| Constant today | Where | Qwen4 need |
|---|---|---|
| `kHC = 4`, `kMaxM = 24`, `kTopK = 10`, `kLocalE = 128` | `mono_k2_rdna2.cu` | from config: HC count, row cap, top-k, local experts (templates already exist for M buckets 4/8/12/24 and local 256) |
| `ND = 336`, `R = 320` | launch arguments | `hc_lowrank + hc_count²`, `hc_lowrank` |
| `H`, `I`, `Ise`, `E` | launch arguments, already dynamic | no change; the workspace and phase code already read them |
| `kHD = 256`, `kMaxG = 8` | QSA kernel | template on head_dim, or keep 256 and gate (both known geometries are 256); group gate stays |
| `kMaxT = 8` tokens/sequence | GDN kernel | MTP depth > 7 needs a bigger `kMaxT` (leak says 4 draft tokens, still ≤ 8) |
| `k1_max_rows 24`, `k2_max_rows 8` | `Options` | planner input; MTP4 at c=8 needs 40 |
| `kLocalE` stage capacity 256 | offload specialisations | Qwen4 MoE with > 256 local experts needs a new specialisation |
| GDN head constants `kGdnH=4, kGdnHV=12` | K2's experimental GDN mode | must come from the layer or stay off (the separate GDN kernel is the production path) |
| QSA group ≥ 1 | indexer/scoring | already dynamic; only the sparse decode has the ≤ 8 gate |

### 2.3 What is new

1. **Dense FFN variant of K2, and a dense MLP module.** The leaked
   Qwen4-27B is dense (no routed experts). Today's K2 requires a router and
   W4A16 experts; its only dense route is the MTP-only FP16 specialisation
   pinned to `[128,1280,2560]`/`[128,2560,640]`. Qwen4 needs a production
   dense FFN block: gate/up + SwiGLU + down at `intermediate_size` (leak:
   ~14336), sharded by TP, with the same P1–P3 and P7 epilogue. The saving is
   bigger than for Flash-Next because the routers, top-k and expert-list LDS
   disappear. Note that the model side is missing too: `Qwen4ExpDecoderLayer`
   builds `Qwen4ExpSparseMoeBlock` for every layer unconditionally
   (`vllm/models/qwen4_exp/amd/model.py`, around line 265, and
   `layer_hook(..., is_moe_layer=True)`), so a dense Qwen4 also needs a
   decoder-layer/MLP implementation before the kernel matters.
2. **A config-driven stage graph.** Today the layer hook decides K1/K2 from
   `is_moe_layer` and the plan is implicit. Qwen4 needs the plan computed
   from the config (layer pattern, layer types, dense vs MoE, indexer
   presence) and asserted against the loaded module geometry at startup.
3. **Larger row caps.** A dense model at MTP4 with c=8 needs 40 rows for
   K1/K2. For the dense K2 that is LDS-cheap; for the MoE K2 the per-expert
   lists (`s_urow`, `s_lora`) push static LDS past 32 KiB, which halves the
   residency cap to 72 workgroups. Either shrink those buffers or accept the
   smaller grid (section 3.4).
4. **Architecture dispatch that does not touch `Qwen4Exp`.** A new opt-in
   switch for non-`Qwen4Exp` Qwen4 names; the existing `--monokernel` path
   keeps matching `Qwen4Exp*` only.
5. **A dense-model QSA fallback.** The leak says QSA kernels exist but no QSA
   weights were available, so a Qwen4 variant may ship without a usable
   indexer. The planner must emit a full-attention stage instead of an
   indexer stage, not refuse the model.
6. **New quant story if Qwen4 ships unquantized-only.** The monokernel
   requires symmetric W4A16 experts and fp16 dense weights; a BF16/FP8
   checkpoint would need the existing AWQ/GPTQ conversion before the kernel
   applies.

## 3. Design

### 3.1 Principles

1. **Plan first, then kernels.** The stage graph, caps, workspace and grid are
   computed from the config. The kernel keeps reading a flat parameter block;
   nothing in the new path reads Flash-Next constants.
2. **Opt-in and default-deny.** A new switch accepts only architecture names
   that start with `Qwen4` and are not `Qwen4Exp`; `--monokernel` behaviour is
   unchanged, and enabling both is an error.
3. **One fallback per part.** A part either covers a shape or hands that
   shape to the modules for the whole step; no silent per-phase downgrade.
   The plan lists the fallbacks at startup, one line per rank.
4. **Same safety contract.** Bounded waits, sticky error, NaN poison,
   fail-stop, MRv2, decode-only, no PP.
5. **No new numerics without a gate.** The dense FFN keeps fp32 accumulation
   and fixed-point cross-workgroup sums; fp16 rounding of the published
   weights stays in the loaders.

### 3.2 The planner (exists on `feat/qwen4-prep`)

`vllm/models/qwen4_prep/mono_plan.py` (no torch, no vLLM import):

- `Geometry.from_hf(cfg, tp)` normalises an HF config: hidden, layer count and
  pattern (`layer_types` or `full_attention_interval`), Q/KV heads and
  head_dim, GDN heads and dims, experts/top-k/intermediate/shared, HC count
  and low-rank, indexer fields, PLE layer ids, MTP depth; derives the
  per-rank shards (`hc_width = H·hc`, `hc_down_rows = lowrank + hc²`,
  `local_experts = E/tp`, `group = Q/KV`).
- `plan_for(geo, spec_tokens, ...)` emits, per layer, the ordered `Stage`s
  (`ple`, `k1`, `gdn` | `qsa_prepare`/`qsa_indexer`/`qsa_mqa`/`qsa_select`/
  `qsa_sparse`, `attn_ar`, `k2`, `ffn_ar`), the row cap per part, the capture
  widths, the grid, per-part warnings and deployment refusals.
- `estimate_traffic(...)` gives the roofline bytes per step per rank:
  replicated HC mixes and router, TP-sharded dense weights, the expected
  number of distinct experts a batch of `rows × top_k` pairs touches, and the
  QSA selected-KV and indexer-scoring traffic.
- `flash_next_plan(tp=4, spec_tokens=2)` reproduces today's plan exactly; the
  CPU test asserts the literal part sequences, the 36/12 split, the single PLE
  layer, the caps (24/8/64), the grid (144) and the widths (3, 6, 12, 24), and
  cross-checks every `Limits` field against the C++ and Python sources, so the
  planner cannot drift from the kernels.

Design intent: the serving path calls the planner once per deployment, stores
the plan in the mono options, and the kernel launcher selects template
instantiations from it. `vllm/models/qwen4_prep/__init__.py` holds the only
"registry" for now: `is_qwen4_architecture()` (prefix `Qwen4`, excluding
`Qwen4Exp`), used by nothing in serving until the release.

### 3.3 Stage graph rules

| Condition | Stage choice |
|---|---|
| layer index in `ple_layer_ids` | PLE module before K1; it materialises the pending HC state |
| `linear_attention` | fused GDN core (≤ 64 rows, ≤ 8 tokens/sequence) |
| `full_attention` + indexer fields | QSA prepare, indexer epilogue, scoring, select, sparse decode |
| `full_attention`, no indexer | FA-RDNA2 decode as a module stage; no QSA part |
| MoE (`num_experts > 0`) | K2 with router/top-k/expert phases; experts resident ≤ 128 or cached ≤ 256 |
| dense FFN | new dense K2 (P1–P3, gate_up, SwiGLU, down, P7); no router/expert stages |
| FFN all-reduce | in P7 when `k2_ar`, else the `RowParallelLinear` collective |
| attention all-reduce | always between the attention core and K2 |

### 3.4 Wave32 tiles, LDS and VGPR budgets on gfx1030

Fixed choices that carry over: 256 threads = 8 waves of 32 lanes; fp16 pairs
with `v_dot2_f32_f16`; int4 with `sdot4`-style dot products and fp32
accumulate; 64 KiB LDS per workgroup; 1024 VGPR per SIMD (wave32), so 128
VGPR per lane buys two workgroups per CU.

| Stage tile | Shape | Note |
|---|---|---|
| HC/router/shared GEMV, M ≥ 5 | LDS tiles 32 rows × 64 half2, K split across WGs, fp32 fixed-point reduction | the M ≤ 8 path uses per-row loops with ≤ 128 VGPR |
| routed gate_up | WG task = (expert × 64 gate columns), activation rows staged in LDS, LDS reduction, SwiGLU in place | P5 |
| routed down | wave tasks with a K split, fixed-point atomics, × top-k weight | P6 |
| GDN | WG = (sequence, value head, 32-row V tile); 256 q\|k channels + 32 v channels in LDS; state in registers | 4 tiles per value head |
| QSA sparse decode | lane owns 8 head dims of a 256-dim head; 8 waves walk every 8th selected token; LDS merge | 1 K/V read per KV-head group |
| dense FFN (new) | WG tasks of 64 output rows over the intermediate, K split; LDS staging as the shared-expert phases; no per-expert lists | new work |

Residency, from the launch-time formula
(`min(by_waves, by_lds) × 2 × WGPs`, 36 WGPs):

| Build | VGPR | Static LDS | Workgroups | Waves/CU |
|---|---|---|---|---|
| K2 today (M ≤ 8) | 128 | 29 KiB | 144 | 16 |
| K2 today (M = 24, MoE, tiled) | ≤ 128 | 29 KiB | 144 | 16 |
| K2 at M = 40, MoE | 128 | ~41 KiB [INFERENCE: `s_lora` 25.6 KiB + `s_urow` 10.2 KiB + lists] | 72 | 8 |
| K2 at M = 40, dense (new) | 128 | ~27 KiB [INFERENCE: `s_lora` only] | 144 | 16 |
| K2 if any stage needed 68 KiB (the gfx950 template) | 128 | 68 KiB | 72 | 8 |

Consequence: the dense Qwen4 path can carry MTP4 at c=8 (40 rows) with the
full grid if the dense K2 drops the expert-list buffers; the MoE path either
tiles `s_lora`/`s_urow` (a 40-row scratch in the workspace) or accepts 72
workgroups, which the Flash-Next measurements say costs (two WGs per CU was
slower only when it spilled, but halving the grid removes the latency hiding
the 16-wave build relies on). Measure both before choosing.

VGPR/LDS for the GDN and QSA decode kernels are not recorded in the tree; print
them with `MONO_FFN_VERBOSE=1` (K2) and `hipFuncGetAttributes` in a probe
before sizing a Qwen4 grid [UNCONFIRMED for those two kernels].

### 3.5 Memory and bandwidth budgets per decode step

Roofline from the planner (bytes per rank per step, 512 GB/s V620 HBM
**[INFERENCE: spec sheet bandwidth, not measured in-tree]**). Context 16k.
"floor" is DRAM time only: no compute, latency, all-reduce or sampling.

Flash-Next, TP4 (48 layers, 512 experts top-10, 128 local):

| rows | dense | experts | QSA KV | total | floor |
|---:|---:|---:|---:|---:|---:|
| 3 | 2.89 GB | 3.29 GB | 0.19 GB | 6.37 GB | 12.4 ms |
| 6 | 2.89 GB | 5.89 GB | 0.38 GB | 9.16 GB | 17.9 ms |
| 12 | 2.89 GB | 9.57 GB | 0.75 GB | 13.21 GB | 25.8 ms |
| 24 | 2.89 GB | 13.30 GB | 1.51 GB | 17.70 GB | 34.6 ms |

Flash-Next, TP2 (256 local experts, expert cache required):

| rows | dense | experts | QSA KV | total | floor |
|---:|---:|---:|---:|---:|---:|
| 3 | 4.21 GB | 3.48 GB | 0.11 GB | 7.80 GB | 15.2 ms |
| 6 | 4.21 GB | 6.57 GB | 0.23 GB | 11.00 GB | 21.5 ms |
| 12 | 4.21 GB | 11.76 GB | 0.45 GB | 16.42 GB | 32.1 ms |
| 24 | 4.21 GB | 19.11 GB | 0.91 GB | 24.23 GB | 47.3 ms |

Qwen4-27B as leaked (dense, H 5120, 64 layers, I 14336, HC 4/320):

| rows | TP4 total | TP4 floor | TP2 total | TP2 floor |
|---:|---:|---:|---:|---:|
| 5 (MTP4, c=1) | 14.27 GB | 27.9 ms | 24.85 GB | 48.5 ms |
| 10 | 14.52 GB | 28.4 ms | 25.27 GB | 49.4 ms |
| 20 | 15.03 GB | 29.4 ms | 26.11 GB | 51.0 ms |
| 40 (MTP4, c=8) | 16.03 GB | 31.3 ms | 27.79 GB | 54.3 ms |

Reading of the table:

- For a **dense** model every step reads the whole per-rank weight set, so the
  batch barely moves the floor (14.0 → 16.0 GB from 5 to 40 rows). The
  monokernel's value there is the ~21–28 launches per layer it removes
  (≈1,000–1,400 launches per step at 48 layers), and the dense K2 is much
  cheaper to build than the MoE one. Qwen4-27B at TP4 lands near the
  Flash-Next width-24 floor; at TP2 it is ~1.7× worse, so TP4 is the recipe
  target if memory allows (14 GB weights/rank + KV + PLE).
- For an **MoE** model the expert union grows with rows: 4.0× from 3 to 24
  rows at TP4. That is why the FFN-only kernel lost at width 24 in the 2a
  work and why the design keeps K2's width gate (8 for the target model)
  instead of assuming mono wins everywhere.
- TP2 Flash-Next cannot be resident: 256 local experts × 2.55 MB × 48 layers
  ≈ 31.3 GB of expert weights alone, which is why the expert cache exists and
  why the planner refuses TP2 without `offload=True`.
- HC mixes are replicated per rank: 26.9 MB per layer at H=2560 and 53.7 MB
  at H=5120, i.e. 1.3 GB and 3.4 GB per rank per step. Sharding the HC
  low-rank or the down projection is the only place a Qwen4 step can drop
  bytes without changing the model; anything else is latency work.

KV per token per rank (16 QSA layers at 27B, 12 at Flash-Next): Flash-Next
TP4 ≈ 24 KB/token (384 MB at 16k), 27B TP4 ≈ 16 KB/token (256 MB), 27B TP2
≈ 32 KB/token (512 MB).

### 3.6 Fallback and gating

The deployment gate mirrors today's, keyed on the new architecture name:

| Check | Action |
|---|---|
| not gfx10x | refuse |
| architecture starts with `Qwen4Exp` | not this switch (today's path) |
| not MRv2, PP > 1, KV not FP16/BF16 | refuse |
| `hc_count != 4`, `H % 128`, low-rank > 320 | refuse (planner refusal) |
| MoE not symmetric W4A16, top-k ≠ 10, local experts over the build | refuse for K2 only; K1/GDN/QSA keep running |
| dense checkpoint | dense K2; if it is unsupported, K2 falls back and the rest stays |
| rows over a part's cap | that part falls back for the whole step; plan logs it |
| QSA group > 8, head_dim ≠ 256, no indexer | QSA sparse falls back to Triton/FA |
| drafter | separate plan and workspace; opt-in as today |

Switch: `--monokernel` keeps its current meaning (`Qwen4Exp*`). For Qwen4 the
proposal is a new opt-in, `VLLM_ROCM_MONO_QWEN4=1` (read where the existing
alias is read), because it needs no upstream `KernelConfig` field and cannot
change an existing model's behaviour; a `--monokernel-qwen4` CLI flag is the
alternative if the branch later wants parity. Both switches on is an error.

### 3.7 Work items

| # | Work | Size | Depends on |
|---|---|---|---|
| W1 | Planner → options → launcher plumbing, plan assertion at startup | S | release config |
| W2 | Parameterise K2 (`ND`, `R`, caps, experts) and raise M buckets | M | W1 |
| W3 | Dense K2 stage graph (no router/expert lists, wide intermediate) plus the dense MLP module the model class lacks | L | W1 |
| W4 | M = 40 support: shrink `s_lora`/`s_urow` or accept 72 WGs | M | W2/W3 |
| W5 | QSA head_dim/group templates and the no-indexer FA stage | M | W1 |
| W6 | GDN `kMaxT`, head parameterisation for TP2/TP4 | S | W1 |
| W7 | Expert cache sizing for E/tp > 256 | M | W2 |
| W8 | Recipe `qwen4*.env`: TP, fp16 forcing, queues, MTP depth, PLE | S | release |

## 4. Test plan

### 4.1 CPU, now (no GPU, no weights)

On `feat/qwen4-prep`: `tools/rdna/qwen4/test_mono_plan.py`, 14 tests, green on
CPU. It asserts the Flash-Next config reproduces today's plan literally (part
order, 36/12 split, one PLE layer, caps 24/8/64, grid 144, widths 3/6/12/24),
that every `Limits` field still equals the C++ constexpr or Python default it
came from, the residency formula (144 at 128 VGPR/29 KiB, 72 at 64 KiB), the
TP4/TP2 sharding rules, the TP2-without-offload refusal, TP1 refusal, the
MTP4 ladder and its row-cap warning, the dense plan (no expert bytes, constant
dense sweep), QSA fallbacks, traffic monotonicity, determinism and the
architecture gate. Command:
`pytest tools/rdna/qwen4/test_mono_plan.py -q`.

### 4.2 Synthetic weights, once GPUs are scheduled (pre-release)

1. **Reference first.** Build the modular block in fp32 on CPU from
   unshuffled random weights, as the 2a work did; keep it as the golden.
2. **Part-vs-module on GPU.** Random weights in the checkpoint layouts
   (symmetric int4 g128 exllama-shuffled, fp16 HC/router/shared; fp16 dense
   FFN), compare `mono_k1`, `mono_k2`, `mono_gdn_rdna2`,
   `mono_qsa_decode_rdna2` against the exact op sequence production runs, at
   rows 1, 3, 8, 12, 24 and 40. Acceptance: residual bit-identical, injection
   ≤ today's error, expert output within the 2a bounds (max abs ≤ 1.5e-3 vs
   the module path; mono closer to the fp32 golden than the modules), routing
   identical at every width, no sticky error.
3. **Dense K2** (new): vs `UnquantizedFusedMoEMethod`-free dense reference at
   I = 14336, rows 1…40, TP4 and TP2 shards; assert TP sharding is exact
   against the unsharded fp32 block.
4. **Rows > cap**: 40-row MoE with today's build must fall back for the whole
   step (assert no partial kernel call), and the dense build must run.
5. **Graph modes**: eager, piecewise, FULL for every supported width; the
   breakable-graph path only with the expert cache.
6. **Fail-stop**: force a barrier timeout (a stalled peer on a second stream)
   and assert NaN outputs plus a sticky record, then restart-only recovery.
7. **Planner/service agreement**: boot with a synthetic `Qwen4*` config and
   assert the runtime plan equals `plan_for()` (no fallen-back part at a
   supported shape).

### 4.3 Post-release gates

| Gate | Criterion |
|---|---|
| Boot probe | plan logged once per rank, no part silently on modules at the target widths |
| Greedy probes | 3/3 prompts, byte-identical to the modular path at temperature 0 where the numerics allow, otherwise divergence only at top-2 gap < 0.3 nats |
| PPL | paired run, mono vs modular, ≤ +0.03 or inside run noise |
| Logprob parity | same-prefix tokens within the off-vs-off noise band |
| ITL | c=1 and c=8, 1k/16k, MTP on/off, mono vs modular, ABBA; require a win outside run variance at the target cell before enabling by default |
| GSM8K | full 1,319-question run, mono vs modular |
| Long context | KL vs BF16 reference at 16k–64k; residual/HC activations must not exceed fp16 range |
| Faults | zero sticky errors, zero new dmesg faults, no boot during measurement (a boot can poison logits through the 2 s barrier) |

## 5. Open questions that need the release

1. **Config schema.** Field names, defaults and the actual `layer_types`
   pattern; whether `hc_count`/`hc_lowrank` carry over; whether PLE and its
   n-gram table exist on Qwen4 (the Flash-Next table is 32.6 GB at ng5).
2. **Dense or MoE per size.** The leaked 27B is dense; a Qwen4 family may
   include both. The dense K2 (W3) is the largest new kernel and is only
   justified if a dense release exists; the current model class has no dense
   MLP path at all, so that release also needs a model-side module.
3. **FFN form.** Activation (SwiGLU? gated?), `intermediate_size`, whether a
   shared expert exists, and whether the attention output gate is sigmoid
   (Flash-Next is).
4. **QSA.** Whether Qwen4 ships an indexer at all, its budget/compress and
   head geometry; the leak had QSA kernels but no weights to test.
5. **MTP depth and drafter layout.** 4 draft tokens were reportedly exercised;
   that sets the width ladder (5/10/20/40) and the row-cap decision in W4, and
   it needs a drafter weight layout for K2.
6. **Quantisation of the release.** If only BF16/FP8 ships, the monokernel
   needs a converted symmetric W4A16 (or a new dense int4 path) before any of
   this applies.
7. **KV cache dtype.** Upstream's QSA path grew an `fp8_e4m3` main KV cache
   ([PR 55557](https://github.com/vllm-project/vllm/pull/55557)); the mono
   kernels read FP16/BF16 only.
8. **Numerics.** Whether Qwen4 needs Flash-Next's "dense native rounding" at
   the P7 epilogue, and whether expert routing near-ties flip at the same
   rate (the 2a data says fp32 accumulation halves the error).
9. **TP target.** The 29 GiB budget on a V620 decides TP2 vs TP4; TP4 is
   ~1.7× better on the roofline for the dense 27B, TP2 needs the expert cache
   for any 512-expert model.

## Sources

- Qwen3.8-Flash-Next config and code:
  [HF config.json](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16/blob/main/config.json),
  [transformers model doc](https://huggingface.co/docs/transformers/main/model_doc/qwen4_exp),
  [transformers PR 48337](https://github.com/huggingface/transformers/pull/48337),
  [Qwen blog](https://qwen.ai/blog?id=qwen3.8-flash-next).
- Qwen4 status and the 27B leak:
  [llama.cpp issue 31072](https://github.com/ggml-org/llama.cpp/issues/31072),
  [huxiu, 2026-10-26](https://www.huxiu.com/article/4900440.html). The leak is
  **[UNCONFIRMED]**: no weights, no config, no model card.
- MonoKernel contract and templates:
  [RFC 60904](https://github.com/vllm-project/vllm/issues/60904),
  [switch 61082](https://github.com/vllm-project/vllm/pull/61082),
  [DSv4.1 60397](https://github.com/vllm-project/vllm/pull/60397),
  [gfx942 port 60731](https://github.com/vllm-project/vllm/pull/60731),
  [Flash-Next draft 61325](https://github.com/vllm-project/vllm/pull/61325).
- This fork: [`../monokernel/README.md`](../monokernel/README.md),
  [`../monokernel/2a-results.md`](../monokernel/2a-results.md),
  [`../monokernel/p0-results.md`](../monokernel/p0-results.md),
  [`../async-stall-profile-2026-11-02.md`](../async-stall-profile-2026-11-02.md),
  [`../moe-offloading-improvements.md`](../moe-offloading-improvements.md),
  [`../moe-offload-topology-2026-11-02.md`](../moe-offload-topology-2026-11-02.md),
  [`../qwen4_exp_hip_path.md`](../qwen4_exp_hip_path.md).
- Kernel sources: `csrc/rocm/mono_k2_rdna2.cu`, `mono_gdn_rdna2.cu`,
  `mono_qsa_decode_rdna2.cu`, `rdna_allreduce.cuh`.
- Planner and tests (branch `feat/qwen4-prep`):
  `vllm/models/qwen4_prep/mono_plan.py`,
  `tools/rdna/qwen4/test_mono_plan.py`.
