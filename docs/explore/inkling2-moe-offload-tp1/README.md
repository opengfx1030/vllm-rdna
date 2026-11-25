# Inkling-2-Mini TP=1 MoE Expert Offloading (16 GiB & 32 GiB Unified Recipe)

**Status**: Shipped in recipe [`inkling2-mini-tp1-offload`](../../../tools/rdna/recipes/inkling2-mini-tp1-offload.env), qualified on gfx1030 (Radeon PRO V620, ROCm 7.14).

## Overview

`thinkingmachines/Inkling-2-Mini` (`Inkling2LoopForConditionalGeneration`) carries `18.68 GiB` of TP=1 weights (including the vision tower), of which **`14.86 GiB` (`79.5%`)** are routed MoE experts (`128` experts/layer across `20` NVFP4 layers and `4` MXFP8 layers). Resident TP=1 does not fit on 16 GiB RDNA2/RDNA3 GPUs and leaves only `196k` KV tokens on a 32 GiB V620.

TP=1 MoE expert offloading reuses this fork's `PackedExpertCache` + `csrc/rocm/rdna_expert_cache.cu` infrastructure:

- **Generalized 4/6-payload SoA cache**: NVFP4 layers (`0..19`) store 6 SoA tensors (`w13_qweight`, `w13_scales`, `w13_globals`, `w2_qweight`, `w2_scales`, `w2_globals`); MXFP8 layers (`20..23`) store 4 SoA tensors (`w13_qweight`, `w13_scales`, `w2_qweight`, `w2_scales`). Both `"native"` (`hipMemcpyBatchAsync` via `rdna2_expert_cache_copy`, default) and `"tensor"` copy backends are supported.
- **Async layer-ahead speculative expert prefetching**: During decode (`tc <= 16`), each offloaded MoE layer runs `next_mlp`'s lightweight fused router (`inkling2_glue_moe_route`) on the current layer's MoE input `hq` and piggybacks its D2H readback onto the single existing `route_event` sync (**zero extra host syncs**). Immediately after enqueuing the current layer's GEMMs on the compute stream, `PackedExpertCache.prefetch()` queues up to `MAX=4` predicted missing experts from each token's top `TOPK=4` ranks onto a dedicated `copy_stream`, overlapping PCIe H2D DMA with both the current layer's MoE compute and the next layer's entire attention block (`attn_in + self_attn + mlp_input`). Predicted resident hits are protected from eviction, `policy.observe()` is never inflated by speculative predictions, and `ensure()` fences via a GPU-side `wait_event` before use. Controlled via `VLLM_INKLING2_MOE_OFFLOAD_PREFETCH=1|0` (`EXPERT_CACHE_PREFETCH=1|0` in the recipe).
- **Streaming CPU weight load**: When `VLLM_INKLING2_MOE_OFFLOAD_SLOTS > 0`, expert parameters are staged on CPU during model construction and repacked in 16-expert chunks directly into pinned host RAM (`14.59 GiB`), allocating only `S` GPU slots per layer (`5.47 GiB` at `S = 48`). Peak VRAM never spikes to `18.68 GiB`, so 16 GiB GPUs never OOM during load.
- **Zero inner-GEMM rewrite**: `Inkling2LoopMoE` binds its non-persistent weight buffers directly to the GPU slot pool `[S, ...]`, plans bounded passes (`unique_experts > S` overflow handled transparently), remaps logical `expert_ids` / prefill `cta_info` to slot indices, and calls the unchanged `moe_nvfp4_gemm_rdna2`, `moe_mxfp8_gemm_rdna2`, and `moe_prefill_bm_rdna2` kernels.
- **Breakable `FULL_AND_PIECEWISE` HIP graphs**: Dense attention, norms, message gates, and short convolutions stay captured in HIP graph segments while offloaded MoE layers break out cleanly via `@eager_break_during_capture`.
- **TP=1 guard**: Explicitly gated to `tp_size == 1` (`ValueError` if `tp_size > 1`). When `VLLM_INKLING2_MOE_OFFLOAD_SLOTS=0` (default), the resident path is 100% untouched.

## Usage

One unified recipe serves both **16 GiB** and **32 GiB** gfx1030 cards:

```bash
# Default (16 GiB & 32 GiB out of the box: EXPERT_SLOTS=48, EXPERT_CACHE_COPY=native,
# EXPERT_CACHE_PREFETCH=1, MAXLEN=65536)
bash tools/rdna/serve_rdna.sh RECIPE=inkling2-mini-tp1-offload \
  MODEL=/path/to/inkling2-mini VENV=/path/to/venv HIP_VISIBLE_DEVICES=4

# 32 GiB V620 with ultra-long single-sequence context
bash tools/rdna/serve_rdna.sh RECIPE=inkling2-mini-tp1-offload \
  MODEL=/path/to/inkling2-mini VENV=/path/to/venv HIP_VISIBLE_DEVICES=3 \
  GMEM=0.92 MAXLEN=430000
```

## Measured Silicon Qualification (gfx1030 / V620, ROCm 7.14) **[measured]**

### 1. Memory & KV Capacity (16 GiB Cap vs 32 GiB V620)

| Configuration | `GMEM` | Model Load VRAM | Peak Device VRAM | Graph Capture | KV Pool (tokens) | `MAXLEN` Verified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Resident `inkling2-mini-tp1` (reference) | `0.92` | `18.68 GiB` | `25.59 GiB` | `0.70 GiB` | `196,065` | `194,000` |
| **Offload `S=48`, 16 GiB sim (`HIP 4`)** | `0.48` (`14.39 GiB`) | **`9.36 GiB`** (`29.0 s`) | **`12.94 GiB`** | `1.72 GiB` (`5 s`) | **`71,099`** (`1.08x`) | **`65,536`** |
| **Offload `S=48`, 32 GiB V620 (`HIP 3`)** | `0.92` | **`9.36 GiB`** (`28.8 s`) | — | `1.72 GiB` (`5 s`) | **`432,484`** (`6.60x`) | **`65,536`** |
| **Offload `S=48`, 32 GiB long ctx (`HIP 3`)** | `0.92` | **`9.36 GiB`** (`29.4 s`) | — | — | **`502,329`** (`1.17x`) | **`430,000`** |

Pinned host expert bank: `14.59 GiB` (`24/24` layers, `128` experts/layer); GPU expert slot cache: `5.47 GiB` (`48/128` slots/layer).

### 2. Correctness, Parity & Looped-Transformer Locality

- **Unit & GPU parity suite** (`tests/models/inkling2_loop/test_moe_offload.py`, `HIP 4`): **`13/13` passed** (NVFP4 6-pool & MXFP8 4-pool, `tensor` & `native` `hipMemcpyBatchAsync`, decode + prefill BM + overflow `16 > 6`, `BreakableCUDAGraphCapture` capture & replay, `tp_size > 1` guard).
- **Bit-exact greedy parity** (`HIP 2`, `VLLM_RDNA2_MOE_DETERMINISTIC=1`, 32 generated tokens on real checkpoint): **`32/32` identical token IDs** vs resident `inkling2-mini-tp1`.
- **Looped-transformer Pass-0 vs Pass-1 hit rate** (`HIP 2`, `S=48`, 32-token cold-start generation):
    - Overall: **`73.06%`** (`20,709` demands, `5,580` misses)
    - Looped core (`layers 6..17`) **Pass 0**: **`75.48%`** (`4,964` demands, `1,217` misses)
    - Looped core (`layers 6..17`) **Pass 1**: **`83.04%`** (`10,078` demands, `1,709` misses, **+7.56 pp** locality gain)
- **Serving gates**:
    - `HIP 4` (16 GiB sim, `GMEM=0.48`): `serve_smoke.py` **`RESULT PASS`** (text, image, prefix cold `4.56 s` -> warm `0.57 s`, stream `23.2 chunks/s`, 4-way concurrent `44.6 tok/s`); `concurrency_probe.py` **`RESULT PASS`** (`0` failing, `3` near-tie, `25/28` identical).
    - `HIP 3` (32 GiB, `GMEM=0.92`): `serve_smoke.py` **`RESULT PASS`** (`tensor` & `native`); `concurrency_probe.py` **`RESULT PASS`** (`0` failing, `5` near-tie, `33/38` identical); `stall_probe.py` (`4` decoders, `4096` & `16384` injections) **`0` starved decoders**, Jain fairness **`1.00`**.

### 3. Throughput A/B: Shipped `tensor` vs `native` vs `native` + Layer-Ahead Prefetch (`HIP 4`, `S=48`, `FULL_AND_PIECEWISE` graphs)

| Workload | `tensor` + `prefetch=0` (prev default) | `native` + `prefetch=0` | **`native` + `prefetch=1` (`TOPK=4, MAX=4`, new default)** | Delta vs prev default |
| --- | ---: | ---: | ---: | ---: |
| Decode `b=1` (64 tok) | `14.95 tok/s` | `16.19 tok/s` | **`16.18–17.81 tok/s`** | **`+8.2% to +19.1%`** |
| Decode `b=4` (64 tok) | `27.04 tok/s` (`6.76/req`) | `29.17 tok/s` (`7.29/req`) | **`33.85 tok/s`** (`8.46/req`) | **`+25.2%`** |
| Decode `b=8` (64 tok) | `31.73 tok/s` (`3.97/req`) | `35.50 tok/s` (`4.44/req`) | **`37.03 tok/s`** (`4.63/req`) | **`+16.7%`** |
| Prefill `1k` (`1,350 tok`) | `282.5 tok/s` (`4.78 s`) | `330.9 tok/s` (`4.08 s`) | **`322.4 tok/s`** (`4.19 s`) | **`+14.1%`** |
| Prefill `16k` (`22,600 tok`) | `364.5 tok/s` (`62.01 s`) | `431.1 tok/s` (`52.43 s`) | **`422.0 tok/s`** (`53.56 s`) | **`+15.8%`** |

- **Prefetch accuracy**: `85.3%` of prefetched experts (`14,352 / 16,833`) were consumed as cache hits on the immediately following layer; `100%` of native transfers (`22,424 / 22,424`) used `hipMemcpyBatchAsync`.
- **Bit-exact deterministic parity** (`VLLM_RDNA2_MOE_DETERMINISTIC=1`, `HIP 4`): **`32/32` identical token IDs** vs resident `inkling2-mini-tp1` with `copy=native` + `prefetch=1`.
- **Extended soak (`VLLM_RAISE_ON_LOGIT_NANS=1`, `copy=native` + `prefetch=1`, `HIP 4`)**:
    - **16 GiB sim (`GMEM=0.48`)**: Model load `9.37 GiB`, peak VRAM **`12.94 GiB`** (`< 15.0 GiB`), KV pool `71,046` tokens (`1.08x` at 65k); `serve_smoke.py` **`RESULT PASS`**; `concurrency_probe.py` **`RESULT PASS`** (`0` failing, `4` near-tie, `44/48` identical); `0` faults, `0` NaNs.
    - **32 GiB (`GMEM=0.92`)**: KV pool **`432,430` tokens** (`6.60x` at 65k); `serve_smoke.py` **`RESULT PASS`**; `concurrency_probe.py` **`RESULT PASS`** (`0` failing, `8` near-tie, `40/48` identical); `stall_probe.py` (`4` decoders, `4096` & `16384` injections) **`0` starved decoders**, Jain fairness **`1.00`**; `0` faults, `0` NaNs.
- `EXPERT_CACHE_COPY=native` and `EXPERT_CACHE_PREFETCH=1` are promoted to default in `inkling2-mini-tp1-offload.env` (`EXPERT_CACHE_COPY=tensor` and `EXPERT_CACHE_PREFETCH=0` remain available as fallback switches).
