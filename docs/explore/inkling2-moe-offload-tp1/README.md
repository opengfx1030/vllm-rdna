# Inkling-2-Mini TP=1 MoE Expert Offloading (16 GiB & 32 GiB Unified Recipe)

**Status**: Shipped in recipe [`inkling2-mini-tp1-offload`](../../../tools/rdna/recipes/inkling2-mini-tp1-offload.env), qualified on gfx1030 (Radeon PRO V620, ROCm 7.14).

## Overview

`thinkingmachines/Inkling-2-Mini` (`Inkling2LoopForConditionalGeneration`) carries `18.68 GiB` of TP=1 weights (including the vision tower), of which **`14.86 GiB` (`79.5%`)** are routed MoE experts (`128` experts/layer across `20` NVFP4 layers and `4` MXFP8 layers). Resident TP=1 does not fit on 16 GiB RDNA2/RDNA3 GPUs and leaves only `196k` KV tokens on a 32 GiB V620.

TP=1 MoE expert offloading reuses this fork's `PackedExpertCache` + `csrc/rocm/rdna_expert_cache.cu` infrastructure:

- **Generalized 4/6-payload SoA cache**: NVFP4 layers (`0..19`) store 6 SoA tensors (`w13_qweight`, `w13_scales`, `w13_globals`, `w2_qweight`, `w2_scales`, `w2_globals`); MXFP8 layers (`20..23`) store 4 SoA tensors (`w13_qweight`, `w13_scales`, `w2_qweight`, `w2_scales`). Both `"tensor"` (default) and `"native"` (`hipMemcpyBatchAsync` via `rdna2_expert_cache_copy`) copy backends are supported.
- **Streaming CPU weight load**: When `VLLM_INKLING2_MOE_OFFLOAD_SLOTS > 0`, expert parameters are staged on CPU during model construction and repacked in 16-expert chunks directly into pinned host RAM (`14.59 GiB`), allocating only `S` GPU slots per layer (`5.47 GiB` at `S = 48`). Peak VRAM never spikes to `18.68 GiB`, so 16 GiB GPUs never OOM during load.
- **Zero inner-GEMM rewrite**: `Inkling2LoopMoE` binds its non-persistent weight buffers directly to the GPU slot pool `[S, ...]`, plans bounded passes (`unique_experts > S` overflow handled transparently), remaps logical `expert_ids` / prefill `cta_info` to slot indices, and calls the unchanged `moe_nvfp4_gemm_rdna2`, `moe_mxfp8_gemm_rdna2`, and `moe_prefill_bm_rdna2` kernels.
- **Breakable `FULL_AND_PIECEWISE` HIP graphs**: Dense attention, norms, message gates, and short convolutions stay captured in HIP graph segments while offloaded MoE layers break out cleanly via `@eager_break_during_capture`.
- **TP=1 guard**: Explicitly gated to `tp_size == 1` (`ValueError` if `tp_size > 1`). When `VLLM_INKLING2_MOE_OFFLOAD_SLOTS=0` (default), the resident path is 100% untouched.

## Usage

One unified recipe serves both **16 GiB** and **32 GiB** gfx1030 cards:

```bash
# Default (16 GiB & 32 GiB out of the box: EXPERT_SLOTS=48, MAXLEN=65536)
bash tools/rdna/serve_rdna.sh RECIPE=inkling2-mini-tp1-offload \
  MODEL=/path/to/inkling2-mini VENV=/path/to/venv HIP_VISIBLE_DEVICES=4

# 32 GiB V620 with ultra-long single-sequence context + native batch H2D copy
bash tools/rdna/serve_rdna.sh RECIPE=inkling2-mini-tp1-offload \
  MODEL=/path/to/inkling2-mini VENV=/path/to/venv HIP_VISIBLE_DEVICES=3 \
  GMEM=0.92 MAXLEN=430000 EXPERT_CACHE_COPY=native
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

### 3. Throughput A/B: `copy=tensor` vs `copy=native` (`HIP 3`, `S=48`, 32 GiB V620)

| Workload | `EXPERT_CACHE_COPY=tensor` | `EXPERT_CACHE_COPY=native` | Delta (`native` vs `tensor`) |
| --- | ---: | ---: | ---: |
| Decode `b=1` (64 tok) | `19.14 tok/s` (`19.14/req`) | `19.03 tok/s` (`19.03/req`) | `-0.6%` |
| Decode `b=4` (64 tok) | `39.70 tok/s` (`9.93/req`) | **`40.61 tok/s`** (`10.15/req`) | **`+2.3%`** |
| Decode `b=8` (64 tok) | `64.50 tok/s` (`8.06/req`) | `64.51 tok/s` (`8.06/req`) | `+0.0%` |
| Prefill `1k` (`1,008 tok`) | `448.6 tok/s` (`2.25 s`) | **`512.4 tok/s`** (`1.97 s`) | **`+14.2%`** |
| Prefill `16k` (`16,286 tok`) | `578.3 tok/s` (`28.16 s`) | **`644.9 tok/s`** (`25.25 s`) | **`+11.5%`** |

`tensor` stays the default for portability; `EXPERT_CACHE_COPY=native` is recommended on HIP 7.14+ for `+11.5–14.2%` faster prefill.
