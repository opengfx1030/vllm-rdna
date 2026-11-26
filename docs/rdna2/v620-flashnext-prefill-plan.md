# 4× Radeon PRO V620 Prefill Optimization Plan (Qwen3.8-Flash-Next)

> Living progress tracker for pushing `Qwen3.8-Flash-Next-Uncensored-W4A16` prefill throughput past the current ~2,000–2,400 tok/s baseline on 4× Radeon PRO V620 (`gfx1030`, TP4/EP4).

## Hardware & Current Production Baseline

- **Host**: Gigabyte MC62-G40, AMD Threadripper PRO 3945WX (all 4 GPUs on direct CPU PCIe 4.0 x16 lanes, P2P enabled, no x8 PLX bottleneck).
- **GPUs**: 4× Radeon PRO V620 (`gfx1030`, 72 CUs, 32 GiB GDDR6, 512 GB/s/card, 128 MiB Infinity Cache, 200 W / 2449 MHz OD via `tools/rdna2/apply_v620_power.sh`).
- **ISA Reference**: `~/work/amd-rdna2-isa.txt` (RDNA 2 ISA Reference Guide):
    - `V_DOT2_F32_F16` / `V_DOT2C_F32_F16`: ~20.6 GMAC/s (current Dense, HC, QSA, GDN path)
    - `V_DOT4_I32_I8` / `V_DOT4C_I32_I8`: ~39.2 GMAC/s (**1.9× FP16**; currently MoE W4A8 only)
    - `V_DOT8_I32_I4` / `V_DOT8_U32_U4`: ~78.3 GMAC/s (**3.8× FP16, 2× dot4**; currently unused in `csrc/rocm/`)
    - `V_PERM_B32`, `V_PERMLANE16_B32`, DPP, `S_CLAUSE`, Wave32, 64 KiB LDS/WGP
- **Served Model & Config** (`tools/rdna2/serve_v620_stable.sh`, `tools/rdna/recipes/flashnext-mtp2.env`):
    - `Qwen3.8-Flash-Next-Uncensored-W4A16` (48 layers: 36 GDN + 12 QSA + 1 MTP; hidden 2560; 512 experts / 128 per EP rank, top-10, `compressed-tensors` W4A16 `g=32`; PLE `int4-disk`)
    - `TP=4, EP=4`, `--dtype float16`, `MAXBAT=2048`, `KV=6743000000` (~2.7 GiB/card transient slack), `GPU_MAX_HW_QUEUES=4`, `MTP=4`
    - **Already ON**: `VLLM_RDNA2_W4A8_SDOT4=1`, `VLLM_RDNA2_W4A8_BLOCK_M=8`, `VLLM_RDNA_W4A8_KBLOCK=-1`, `VLLM_RDNA_ALL_HIP=1`, `VLLM_GDN_HIP_PREFILL=1`, `VLLM_RDNA_QSA_PIPE=1`, `VLLM_RDNA_FUSED_SE=1`, `VLLM_RDNA_DENSE_INT8=1`, `VLLM_RDNA_DENSE_INT8_ONLY=1`, `PYTORCH_TUNABLEOP_ENABLED=1` (`rocm10-rocblas5.6`), `VLLM_RDNA_MOE_PLACEMENT=placement-coding-humaneval.json`
    - **Currently OFF / Diagnostic**: `VLLM_RDNA_PHASE_TIMING=10`, `VLLM_RDNA_GDN_DH_PIPE=0`, `VLLM_CAUSAL_CONV1D_RDNA2_FWD=0`, `VLLM_RDNA_AR_MAX_KB=256` (prefill ARs use FP16 RCCL ring)

### Measured 2048-Token Prefill Step Breakdown (`rocprofv3`, 2026-11-26)

Measured on 4× V620 (200 W / 2449 MHz, `serve_v620_stable.sh`, `CG_MODE=NONE`, `--no-enable-prefix-caching`, no kvfs offload, `VLLM_RDNA_PHASE_TIMING=` off; 4092-tok cold prefill = 2×2048 chunks, Rank 0 `840.3 ms` wall / `769.0 ms` kernel per 2048-tok chunk):

| Subsystem | Rank 0 ms / 2048-tok chunk | % GPU | Calls / chunk | Notes |
| --- | ---: | ---: | ---: | --- |
| Dense GEMM (rocBLAS FP16) | **152.4 ms** | 19.8% | 588 | `Cijk_Alik_Bljk_HHS` (`v_dot2` FP16); weights already INT8 |
| Dequant / Elementwise mul | **29.1 ms** | 3.8% | 819 | `24.7 ms` (`462x`) is `rdna_dense_int8.dequant` alone! |
| Hyper-Connections (HC) | **147.6 ms** | 19.2% | 404 | `gemm_tn_splitk` `79.5 ms` + `hc_up_mix` `46.7 ms` + `hc_combine_norm` `20.3 ms` |
| RCCL / All-Reduce | **156.5 ms** | 20.3% | 157 | `ncclDevKernel` `108.0 ms` + `rdna_ar_oneshot` `48.4 ms`; Rank 3 only `97.6 ms` (68 ms rank skew) |
| QSA Sparse Attention | **107.9 ms** | 14.0% | 164 | `qsa_sparse_paged_gqa_splitk_kernel` `100.9 ms` (`21x`) |
| MoE Routed Experts (W4A8 v2) | **79.8 ms** | 10.4% | 243 | `moe_w4a8_gemm_kernel_v2<8,32>` `74.0 ms` (`96x`) + `w4a8_act_quant` `5.1 ms` |
| GDN / Conv / PLE | **55.7 ms** | 7.2% | 252 | `delta_h_packed` `32.7 ms` (`36x`), `prefill_o` `9.4 ms`, `solve_wy` `8.7 ms` |
| Other + MTP unquant MoE | **21.2 ms** | 2.8% | 980 | Includes `15.0 ms` unquantized FP16 MTP MoE (`10.8 + 4.2 ms`) |
| Norm / RoPE / Act | **18.8 ms** | 2.4% | 261 | `act_and_mul` `4.5 ms`, RMSNorm, RoPE |

> **Key takeaway**: Dense GEMM + Dequant (`181.5 ms`, 23.6%) + HC (`147.6 ms`, 19.2%) = **`329.1 ms` (42.8% of GPU time)** — over **4× larger** than MoE (`79.8 ms`). Moving Dense and HC from FP16 `v_dot2` to native INT8 `v_dot4_i32_i8` (Phase 1.1 & 1.2) is overwhelmingly the highest-leverage win.

---

## Phase 0: Fresh Baseline, Profiling & Zero-Code Quick Wins

- [x] **0.1 Verify host power/clock operating point**
    - Verified 2026-11-26 via `bash tools/rdna2/apply_v620_power.sh --check`: all 4 cards (`card1..card4`) at **200 W, 2449 MHz OD SCLK**.
    - Verified via `rocm-smi`: all 4 GPUs at **PCIe 4.0 x16 (`16.0GT/s x16`)**, symmetric 2-hop P2P topology (`weight 40`), idle temps 32–33°C edge / 34–35°C junction / 34–38°C mem (plenty of thermal headroom if testing 225–250 W later).
- [x] **0.2 Run MoE dot-rate probe (gates Phase 2.1 `V_DOT8_I32_I4`)**
    - Ran 2026-11-26 on idle GPU 0 (`gfx1030`, 200 W / 2449 MHz):
        - **MoE W4A8 v2 (`KBLOCK=-1`, `bm=8`, production)** at `M=2048`: `w13` = `1.313 ms` (**16.48 TMAC/s = 42.0%** of 39.19 TMAC/s `v_dot4` ceiling; 2.36× faster than W4A16), `w2` = `0.675 ms` (**16.11 TMAC/s = 41.1%** of `v_dot4` ceiling; 2.30× faster than W4A16).
        - **MoE W4A8 v2 `bm=16`** at `M=2048`: `w13` = `1.801 ms` (13.01 TMAC/s), `w2` = `0.793 ms` (14.67 TMAC/s) — **slower** than `bm=8` due to 170 VGPRs (1 wave/SIMD) + padding (`7152` vs `6600`).
        - **Decision**: W4A8 v2 sits at **41–46% of the `v_dot4` ceiling** (not ALU-saturated at >70%). Phase 2.1 takes **Path B** (VGPR/LDS optimization, not lossy W4A4 `v_dot8`), while **Phase 1.1 (W8A8 `v_dot4` Dense GEMM)** is confirmed as the #1 bottleneck: microbench of `rdna_dense_int8.dequant` + rocBLAS FP16 at `M=2048` measured only **21–24 TFLOP/s (27–31% of `v_dot4` ceiling)** (`gdn_in_proj_qkvz` 1.793 ms, `qsa_qkv_proj` 1.022 ms, `qsa_o_proj` 0.715 ms, `se_gate_up` 0.619 ms, `gdn_out_proj` 0.441 ms, `se_down` 0.279 ms), with `dequant()` alone burning 0.04–0.13 ms/call.
- [x] **0.3 Capture fresh `rocprofv3` 2048-token prefill trace on current prod config**
    - Captured 2026-11-26 on all 4 ranks (`/tmp/opencode/prof_p03/c4/*_kernel_trace.csv`, 31 MiB/rank) and updated the measured step breakdown table above.
- [x] **0.4 Zero-code config A/Bs** (2026-11-26, 4k/8k/16k cold `c=1`, 2 reps each, `--no-enable-prefix-caching`, no kvfs offload, `CG_MODE=NONE`)
    - [x] Confirm overhead of `VLLM_RDNA_PHASE_TIMING=10` vs off: `A_base` (off) = `2460 / 2543 / 2586 tok/s` (4k/8k/16k), small/clean improvement at 4k (`2460` vs `2370`).
    - [x] A/B `VLLM_RDNA_GDN_DH_PIPE=1` + `VLLM_CAUSAL_CONV1D_RDNA2_FWD=1` (`B_gdn_conv`): **`2488.8 / 2602.3 / 2611.2 tok/s`** (**+1.2% / +2.3% / +1.0%** across all lengths, zero extra VRAM). **KEEP** (`GDN_DH_PIPE=1` is bitwise identical; confirm `CONV1D_FWD=1` quality in final gate).
    - [x] A/B `MAXBAT=4096` vs `2048` (`C_maxbat4096`): **`2187.7 / 2295.9 / 2269.8 tok/s`** (**-11.8% to -13.1%**, +830 MiB VRAM/card — misses `rocm10-rocblas5.6` TunableOp rows past 2048). **REJECT — keep `MAXBAT=2048`.**

---

## Phase 1: Highest-ROI ISA Kernel Wins (`V_DOT4_I32_I8` on Dense & HC)

### 1.1 Native W8A8 `v_dot4_i32_i8` Dense Prefill GEMM

*Target: save ~50–70 ms / 2048-tok chunk (~1.5–1.7× faster dense GEMMs, zero extra VRAM).*

- [x] **1.1.1 Audit dense prefill shapes & callers**
    - Audited `vllm/model_executor/layers/rdna_ops.py` and `rdna_dense_int8.py`: all dense/shared-expert prefill projections funnel through `_rdna_dense_gemm` and `_rdna_shared_expert` with per-channel symmetric INT8 `weight_i8` + fp16 `weight_i8_scale` already in VRAM (`K % 64 == 0` on all Flash-Next dense projections).
- [x] **1.1.2 Implement `gemm_w8a8_i8_rdna2` HIP kernel**
    - Implemented `csrc/rocm/gemm_w8a8_i8_rdna2.cu` (`w8a8_act_quant_per_token_k` + Wave32 register-tiled `gemm_w8a8_i8_kernel` with `BK=64` int8, `BUF=1`, `SWZ=1`, `BROT=1`, `NBUF=2`, `SKIP=1`, `v_dot4c_i32_i8`, designs `D1_I8`/`D4_I8`/`D6_I8`, cached per-device scratch).
    - Registered in `csrc/rocm/ops.h`, `csrc/rocm/torch_bindings.cpp` (`VLLM_ROCM_GFX1030`), `CMakeLists.txt` (with `-O3`), and `vllm/_custom_ops.py` (+ `register_fake`).
- [x] **1.1.3 Wire into `rdna_ops.py` behind opt-in flag**
    - Wired into `_rdna_dense_gemm` and `_rdna_shared_expert` behind `VLLM_RDNA_DENSE_W8A8_PREFILL=1` (default `0`, `M >= 64`, `K % 64 == 0`, `on_gfx10x()`).
- [x] **1.1.4 Correctness & microbenchmark**
    - Verified 2026-11-26 on idle GPU 0 (`gfx1030`, 200 W / 2449 MHz) vs `rdna_dense_int8.dequant + rocBLAS FP16`:
        - `gdn_in_proj_qkvz [2048×2560×4096]`: `1.818 → 0.715 ms` (**2.54×**, **60.1 TFLOP/s = 76.6% `v_dot4` ceiling**), `relL2=0.0084`
        - `qsa_qkv_proj [2048×2560×2048]`: `1.059 → 0.424 ms` (**2.50×**, 50.6 TFLOP/s), `relL2=0.0084`
        - `qsa_o_proj [2048×1536×2560]`: `0.760 → 0.310 ms` (**2.45×**, 52.0 TFLOP/s), `relL2=0.0081`
        - `se_gate_up [2048×2560×1280]`: `0.650 → 0.285 ms` (**2.28×**, 47.0 TFLOP/s), `relL2=0.0084`
        - `se_down [2048×640×2560]`: `0.323 → 0.146 ms` (**2.21×**, 46.0 TFLOP/s), `relL2=0.0076`
        - `gdn_out_proj [2048×1024×2560]`: `0.504 → 0.253 ms` (**2.00×**, 42.5 TFLOP/s), `relL2=0.0079`
        - `hc_down [2048×10240×336]`: `0.444 ms` (`D6_I8`, **1.79×** faster than FP16 `gemm_tn_splitk` `0.795 ms`), `relL2=0.0092`
        - `tail_M1031 [1031×2560×2048]`: `0.690 → 0.224 ms` (**3.08×**); `small_M128`: `0.227 → 0.064 ms` (**3.56×**)
    - ISA audit: `1,280` `v_dot4c_i32_i8`, `40` `buffer_load_dwordx4`, **0 VGPR/SGPR spills**, **0 scratch bytes** on all 4 kernels.
- [x] **1.1.5 End-to-end ABBA serving validation**
    - Verified 2026-11-26 on 4× V620 (`FULL_AND_PIECEWISE`, `VLLM_TREE=$WT`, 4/4 workers confirmed `rdna_dense_gemm W8A8 sdot4 prefill: active`):
        - `1_base` (GDN_DH_PIPE=1, CONV1D_FWD=1): 4k/8k/16k = `2,681.4 / 2,698.5 / 2,656.3 tok/s`, 16k TTFT `6.166 s`, greedy `391`, PPL `1.5956`.
        - `2_dense_w8a8` (`VLLM_RDNA_DENSE_W8A8_PREFILL=1`): 4k/8k/16k = **`3,009.3 / 3,031.2 / 2,978.1 tok/s`** (**+12.1–12.3%**), 16k TTFT **`5.499 s`** (-667 ms), greedy `391`, PPL `1.4030`.
        - (Full non-repetitive PPL + c=8 concurrency probe deferred to final Phase 1 gate.)

### 1.2 Hyper-Connection (HC) Prefill → `v_dot4_i32_i8` & Grid Tuning

*Target: save ~40–60 ms / 2048-tok chunk.*

- [x] **1.2.1 Sweep `VLLM_RDNA_HC_DOWN_TARGET` / `VLLM_RDNA_HC_DOWN_SPLITS`**
    - Superseded by 1.2.2: `gemm_w8a8_i8_rdna2` (`D6_I8`) already beats `gemm_tn_splitk_k` by **1.79×** (`0.444 ms` vs `0.795 ms`).
- [x] **1.2.2 Port `gemm_tn_splitk_k` (HC down) to INT8 `v_dot4_i32_i8`**
    - Wired `VLLM_RDNA_HC_W8A8_DOWN=1` in `_rdna_hc_mix` (`vllm/model_executor/layers/rdna_ops.py`) using `gemm_w8a8_i8_rdna2(xn, w_down_i8, s_down)` + `rdna_hc_up_gate_mix_prefill`.
    - Measured 2026-11-26 (`3_dense_hc_w8a8`, 4× V620 `FULL_AND_PIECEWISE`): 4k/8k/16k = **`3,120.1 / 3,212.6 / 3,169.4 tok/s`** (**+19.3% over `1_base`**, **+21–32% over original live prod**), 16k TTFT **`5.167 s`** (-1.00 s vs base, ~646 ms/chunk), greedy `391`, PPL `1.5598`.
- [x] **1.2.3 Port `hc_up_mix_prefill_k` (HC up + sigmoid + mix) to INT8 `v_dot4_i32_i8`**
    - Added `hc_silu_quant_k` + `hc_up_mix_prefill_i8_k` (`rdna_hc_up_gate_mix_prefill_i8`) to `csrc/rocm/gemm_w8a8_i8_rdna2.cu`, wired behind `VLLM_RDNA_HC_W8A8_UP=1` in `vllm/model_executor/layers/rdna_ops.py`.
    - Microbench on idle GPU 0 (`M=2048, H=2560, R=320`): `0.478 ms → 0.234 ms` (**2.04× faster** vs FP16 + dequant, `relL2=0.00193`).
- [x] **1.2.4 Microbench & end-to-end ABBA serving validation**
    - Verified 2026-11-26 (`P1_full`: `GDN_DH_PIPE=1` + `CONV1D_FWD=1` + `DENSE_W8A8_PREFILL=1` + `HC_W8A8_DOWN=1` + `HC_W8A8_UP=1`, 4× V620 `FULL_AND_PIECEWISE`, 4/4 workers active):
        - **4k cold**: **`3,370.4 tok/s`** (`1.215 s`, **+42.2%** vs initial prod / **+25.7%** vs `1_base`)
        - **8k cold**: **`3,370.0 tok/s`** (`2.429 s`, **+33.1%** vs initial prod / **+24.9%** vs `1_base`)
        - **16k cold**: **`3,262.6 tok/s`** (`5.023 s`, **+24.4%** vs initial prod / **+22.8%** vs `1_base`, ~628 ms / 2048-tok chunk)
        - **Realistic prose decode** (1103 → 256): prefill **`3,083.3 tok/s`**, decode **`128.51 tok/s`**
        - **Quality**: greedy math `391` PASS, greedy code `def fib(n)` PASS, real non-repetitive prose PPL (2,113 tok) = `8.7111`.

---

## Phase 2: Second-Wave ISA & Direct-CPU PCIe 4.0 x16 P2P Wins

### 2.1 MoE Prefill: `V_DOT8_I32_I4` (W4A4) vs. `BLOCK_M=16` VGPR Optimization

*Gated on Phase 0.2 `moe_dot_rate_probe.py` result.*

- [ ] **2.1.1 Decision gate from `moe_dot_rate_probe.py`**
    - [ ] **Path A (if W4A8 > 70% of 39.2 GMAC/s `v_dot4` ceiling)**: Prototype W4A4 prefill-only MoE kernel using `V_DOT8_I32_I4` (`__builtin_amdgcn_sdot8`, 78.3 GMAC/s ceiling) with per-(token, group=32) INT4 activation quantization. Gate strictly on PPL + KL + coding evals.
    - [ ] **Path B (if W4A8 < 60% of `v_dot4` ceiling — LDS/global-bound)**: Optimize `moe_w4a8_gemm_kernel_v2` (`csrc/rocm/moe_w4a8_rdna2.cu`) at `BLOCK_M=16` to bring VGPRs from 170 → ≤128 (doubling occupancy to 2 waves/SIMD on RDNA2) and use `V_PERM_B32` for nibble unpack.
- [ ] **2.1.2 Microbench (`tools/rdna/bench_moe_w4a8.py`) & end-to-end ABBA**

### 2.2 INT8-Compressed P2P Reduce-Scatter + All-Gather (`rdna_ars`) for Prefill

*Target: save ~30–35 ms / 2048-tok chunk by halving PCIe 4.0 x16 wire bytes (~15.75 MB → ~7.9 MB per 10.5 MB call).*

- [x] **2.2.0 Probe RCCL `all_to_all + all_gather` on INT8**
    - Measured 2026-11-26 on 4× V620 PCIe 4.0 x16 P2P (`[2048×2560]` = 10.49 MB FP16): FP16 RCCL ring = `667.6 µs`, RCCL `all_to_all + all_gather` INT8 = `703.7 µs` (`0.95×`, slower — RCCL doesn't pipeline the two). Any INT8 prefill AR must be a single fused custom P2P kernel (`rdna_ars`).
- [x] **2.2.1–2.2.3 Custom P2P `rdna_ars_i8` feasibility benchmark (MC62-G40 + 3945WX)**
    - Prototyped and benchmarked fused per-block INT8 P2P RS+AG (`ars_i8_bench.cu`) + raw P2P bandwidth (`p2p_bw.cu`) on 4× V620:
        - Raw shader P2P push (`GPU0 → GPU1`, 4 MB, 16–64 blocks): **`15.1–15.3 GB/s`** (both uncached and cached).
        - FP16 RCCL ring (`[2048×2560]` = 10.49 MB): **`662.7 µs`** (**23.8 GB/s** bus BW).
        - INT8 P2P wire floor (`7.92 MB / 15.1 GB/s` = `524 µs`) + quant/reduce/dequant (~100–150 µs) ≈ `630–670 µs` (zero net gain over RCCL FP16 ring on this CPU root complex).
    - **Verdict: NO-GO** on MC62-G40 + 3945WX. Keep RCCL FP16 ring for prefill AR (`> 256 KiB`) and `rdna_ar` one-shot for decode (`<= 256 KiB`).

### 2.4 Post-Trace Targeted Micro-Optimizations (Single-Pass Quant, `D7_I8` HC Down, Grouped MTP MoE)

- [x] **2.4.1 Single-pass register-cached `w8a8_act_quant_per_token_k` (`K <= 4096`) + `D7_I8` (`64×64×64`) for skinny `N <= 384`**
    - `gdn_in_proj_qkvz [2048×2560×4096]`: `0.715 → 0.653 ms` (**65.7 TFLOP/s = 83.8% of `v_dot4` ceiling**).
    - `hc_down [2048×10240×336]`: `0.439 → 0.414 ms` (`D7_I8`, +6%, bit-identical `max_diff=0.0`).
- [x] **2.4.2 Pure HIP grouped MTP Unquantized MoE prefill (`num_tokens >= 64`)**
    - Replaced the temporary Triton `fused_experts` fallback with a **100% pure HIP** grouped FP16 `v_dot2c_f32_f16` kernel (`moe_gemm_unquantized_prefill_rdna2` in `csrc/rocm/moe_unquantized_rdna2.cu`: `BM=16, BN=64, BK=32, TM=4, TN=4`, 64 threads, 68 VGPRs, 0 spills, 11.6 KiB LDS).
    - Microbench (`M=2048, E=128, N=640, K=2560, topk=3`): **`29.14 ms → 11.13 ms`** (**2.62× faster than old HIP GEMV, 1.10× faster than Triton `12.21 ms`**, `relL2=0.000618`, **zero Triton / zero JIT pause**).

### 2.3 QSA Pipelined KV-Range Fix, EP Straggler Balance & QSA Indexer

- [x] **2.3.0 Fix `qsa_sparse_paged_gqa_pipe_kernel` `int32` offset overflow on >4.29 GiB KV pools**
    - `rocprofv3` revealed `KV=6743000000` (6.74 GB > `INT_MAX` fp16 elements) silently disabled `VLLM_RDNA_QSA_PIPE=1` and fell back to the 2× slower legacy kernel (`100.9 ms/chunk`).
    - Fixed `csrc/rocm/qsa_rdna2.cu` to store `s_off[i]` in units of 8 fp16 elements (16-byte `uint4` vectors, guaranteed by `vec8`), extending range to **34.36 GiB/rank** with zero extra LDS and bitwise-identical math.
    - Measured 2026-11-26 (`P2_qsa`, 4× V620 `FULL_AND_PIECEWISE`): 4k/8k/16k cold prefill = **`3,649.1 / 3,679.2 / 3,652.6 tok/s`** (16k TTFT **`4.486 s`**, ~560 ms/chunk, **+39–54% vs initial prod**), prose decode **`134.60 tok/s`** (+4.7% vs `128.51`), greedy `391` + `def fib(n)` PASS, 2,113-tok prose PPL `8.7985`.
- [ ] **2.3.1 Re-capture EP routing histogram on real workload**
    - Run `VLLM_RDNA_EPLB_STATS_DIR` + `tools/rdna/prof/moe_placement.py` on actual production prompts (not just HumanEval) to check if a refreshed placement cuts the ~90 ms EP straggler wait inside RCCL.
- [ ] **2.3.2 QSA Indexer (`qsa_mqa_paged_rdna2`) INT8 `v_dot4_i32_i8` scoring**
    - Quantize indexer Q/K to INT8 for top-k block selection (biggest payoff at 64k–128k context).

---

## Explicitly Out of Scope (Already Disproven on This Hardware)

- [x] ~~`HSA_ENABLE_SDMA=1` / `PEER_SDMA=1`~~ — tested ±0.5%, no-op for P2P (`serve_v620_v031.sh:32-45`).
- [x] ~~Plain FP16 custom prefill all-reduce~~ — RCCL ring already saturates PCIe 4.0 x16 (~25 GB/s; `tp4-allreduce-sizes-2026-10-10.md`).
- [x] ~~`--kv-cache-dtype fp8`~~ — measured -40% prefill (`1527` vs `2500 tok/s`; `serve_v620_stable.sh:65-66`).
- [x] ~~`--enable-dbo`~~ — inert at `dp=1`, incompatible with hybrid GDN + MTP (`a2a-dbo-feasibility-2026-11-16.md`).
- [x] ~~`VLLM_RDNA2_MOE_TILE8_MIN_TOKENS=1024`~~ — measured -11% to -14%; W4A8 v2 already uses `BLOCK_M=8`.

---

## Validation Gate (Required Before Keeping Any Change)

Per `docs/rdna/agent-rules.md` and `docs/rdna2/tuning.md`:

- [x] **Paired ABBA serving benchmark** (`A1_base → B1_opt → B2_opt → A2_base`, 2026-11-26, 4× V620 200 W / 2449 MHz, `FULL_AND_PIECEWISE`, `KV=6743000000`, `MTP=4`):
    - **4k / 8k / 16k cold `c=1` prefill**: `2,437 / 2,519 / 2,552` → **`3,324 / 3,551 / 3,627 tok/s`** (**+36.4% / +40.9% / +42.2%**); 16k TTFT `6.420 → 4.518 s` (**-29.6%**)
    - **`c=8` 4k→256 concurrent**: mean TTFT `11.78 → 8.25 s` (**-30.0%**), agg prefill `1,519 → 1,956 tok/s` (**+28.8%**), agg output `76.55 → 101.10 tok/s` (**+32.1%**), wall `26.77 → 20.28 s` (**-24.2%**)
    - **Real non-repetitive prose PPL (2,113 tok)**: `A = 8.6287 / 8.6869` (mean `8.6578`) vs `B = 8.6995 / 8.6998` (mean `8.6997`, Δ `+0.042`, within baseline boot spread `0.058`)
    - **Probes & stability**: 4/4 greedy math (`391`) + code (`def fib(n)`) PASS, 4/4 prefix-cache PASS (`warm TTFT 0.537 → 0.409 s`), `dmesg` GPU/PCIe AER fault delta = **0** on all 4 boots.

---

## Progress & Results Log

| Date | Item | Prefill 4k / 8k / 16k c=1 cold (tok/s) | Prefill 16k TTFT (s) | Decode ITL p50 c1/c8 (ms) | PPL / Quality | Verdict |
| --- | --- | --- | ---: | --- | --- | --- |
| 2026-11-26 | Baseline (live prod PID 2985973, 200 W / 2449 MHz, `cached=0`) | **2,369.5 / 2,531.8 / 2,622.8** | **6.247** (~781 ms / 2048-tok chunk) | ~27.5 / — | Baseline | Reference |
| 2026-11-26 | P0.4 `A_base` (`PHASE_TIMING=` off, no kvfs, `MAXBAT=2048`) | **2,460.0 / 2,543.3 / 2,585.8** | **6.337** | — | — | Clean cold baseline |
| 2026-11-26 | P0.4 `B_gdn_conv` (`GDN_DH_PIPE=1`, `CONV1D_FWD=1`, `MAXBAT=2048`) | **2,488.8 / 2,602.3 / 2,611.2** | **6.273** | — | Bitwise (GDN) | **KEEP (+1.0–2.3%)** |
| 2026-11-26 | P0.4 `C_maxbat4096` (`MAXBAT=4096`) | 2,187.7 / 2,295.9 / 2,269.8 | 7.216 | — | — | **REJECT (-12–13%)** |
| 2026-11-26 | P1.1 `1_base` (`FULL_AND_PIECEWISE`, `GDN_DH_PIPE=1`, `CONV1D_FWD=1`) | 2,681.4 / 2,698.5 / 2,656.3 | 6.166 | 173.2 tok/s c1 | greedy `391`, PPL 1.5956 | Phase 1 baseline |
| 2026-11-26 | P1.1 `2_dense_w8a8` (`VLLM_RDNA_DENSE_W8A8_PREFILL=1`) | **3,009.3 / 3,031.2 / 2,978.1** | **5.499** | 125.3 tok/s c1 | greedy `391`, PPL 1.4030 | **KEEP (+12.1–12.3%)** |
| 2026-11-26 | P1.2 `3_dense_hc_w8a8` (`+ VLLM_RDNA_HC_W8A8_DOWN=1`) | **3,120.1 / 3,212.6 / 3,169.4** | **5.167** | — | greedy `391`, PPL 1.5598 | **KEEP (+19.3% vs base)** |
| 2026-11-26 | **P1 Full** (`+ VLLM_RDNA_HC_W8A8_UP=1`) | **3,370.4 / 3,370.0 / 3,262.6** | **5.023** (~628 ms / 2048-tok chunk) | **128.5 tok/s** (prose c1) | greedy `391` + `fib(n)` PASS, 2113-tok prose PPL 8.7111 | **KEEP (+22.8–25.7% vs base, +24–42% vs prod)** |
| 2026-11-26 | **P2.3.0 QSA pipe fix** (`s_off >> 3` for >4.29 GiB KV) | **3,649.1 / 3,679.2 / 3,652.6** | **4.486** (~560 ms / 2048-tok chunk) | **134.6 tok/s** (prose c1) | greedy `391` + `fib(n)` PASS, 2113-tok prose PPL 8.7985 | **KEEP (+37.5% vs base, +39–54% vs prod!)** |
| 2026-11-26 | **P2.4 Opt** (`single-pass quant` + `D7_I8` + `grouped MTP MoE`) | **3,770.5* / 3,766.2 / 3,773.2** (peak **3,826**) | **4.343** (~543 ms / 2048-tok chunk) | **138.6 tok/s** (prose c1) | greedy `391` + `fib(n)` PASS, 2113-tok prose PPL 8.6515 | **KEEP (+43.9–59.1% vs prod!)** |
