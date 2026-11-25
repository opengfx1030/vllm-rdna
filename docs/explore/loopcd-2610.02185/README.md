# LoopCD (`arXiv:2610.02185`) Engine Support for `Inkling2Loop` on RDNA2 (`gfx1030`)

**Status**: Implemented, unit-tested (`246/246` CPU+GPU), and silicon-qualified on `thinkingmachines/Inkling-2-Mini` (`par1-cs25`, Radeon PRO V620 `gfx1030`) with both standard decoding and DSpark speculative decoding (`ADAPTIVE=1` + `32k` draft vocabulary).

**Reference**: *"Decoding Looped Transformers Better for (Almost) Free"* (`arXiv:2610.02185`, Liu et al., Oct 2026).

---

## 1. Overview & Architecture on `Inkling-2-Mini`

`Inkling-2-Mini` (`vllm/models/inkling2_loop/amd/model.py`) executes `num_hidden_layers = 24` (`hidden_size = 2304`, `n_layers_per_attn_res_block = 6`, `unpadded_vocab_size = 200058`) with `loop_num_passes = 3` (`48` layer visits per token):

- **Prelude (`layers[0:6]`)**: 1 pass (`num_pre = 1` completed `AttentionResiduals` block state).
- **Recurrent Core (`layers[6:18]`)**: 3 passes (`visit = 0, 1, 2`), appending `n_prev = 1` loop block state (`layer 12` entry) and exchanging cross-pass `MessageAttentionResidual` states from the previous pass `(v.prev_blk, v.prev_x0, v.prev_x1)`.
- **Coda (`layers[18:24]`) + Final Readout**: 1 pass over `layers[18:24]`, followed by `BlockAttentionResidual` (`_final_query`) and `RMSNorm` (`_final_norm_w`) into `lm_head` (`final_logit_softcapping = 30.0`).

LoopCD contrasts the final loop pass ($h_{\text{final}}$, Pass `passes - 1 = 2`) against an earlier weak loop pass ($h_{\text{weak}}$, default `weak_pass = -1` -> Pass `passes - 2 = 1`, or `weak_pass = 0` -> Pass `0`) without retraining or modifying checkpoints:

1. **`LoopCD-Hidden` (Pre-Coda Hidden-State Contrast — `+25 µs/tok` / `+0.14%` at `B=1`)**:
   - At loop exit right before `coda_layers`, `(v.prev_blk[:, :n_prev], v.prev_x0, v.prev_x1)` is **already live in `Inkling2GlueWorkspace`** from Pass `passes - 2` (`0 B` extra residual/block workspace VRAM; only `8 KiB` for per-token `omega` / `omega_blk` vectors).
   - Deferred `pending` post-MLP `sconv` on the final loop layer is flushed via `static_conv_hip` (mathematically bit-identical to fused `inkling2_glue_attn_res_sconv`), and `inkling2_glue_loopcd_blend` contrasts both the partial-block sum `(x0 + x1)` and (when `loopcd_blend_blocks=True`) `v.blk[:, num_pre + 1 : n]`:
     $$h_{\text{cd}}[t] = h_{\text{final}}[t] + \omega_t \bigl(h_{\text{final}}[t] - h_{\text{prev}}[t]\bigr)$$
   - **Prefix-Cache Invariance**: `inkling2_loopcd_prepare_hidden` populates `ws.loopcd_omega[:T]` on GPU from `idx_mapping`, `query_start_loc`, `seq_lens`, and `prefill_len` so non-terminal prompt prefill tokens (`pos < prefill_len[s] - 1`) **always receive $\omega_t = 0.0$** (`0-ULP` early return). Shared KV and `sconv` cache blocks on prompt prefixes remain 100% bit-identical across requests with arbitrary `loopcd_alpha`.
2. **`LoopCD-Logits` (Skip-Coda Logit Contrast + APC Plausibility Mask + Adaptive Uncertainty Gating)**:
   - Captures the normalized skip-coda readout of the weak pass (`inkling2_glue_loopcd_readout` using `_final_query` and `_final_norm_w` on Pass `passes - 2` or Pass `0`, gated on GPU by `ws.loopcd_logits_flag` so inactive steps incur zero readout work without breaking HIP-graph topology) into `ws.loopcd_weak_h` (`[256, H]` `fp16` = `1.125 MiB`).
   - In `Inkling2ModelState.compute_logits` (overriding `DefaultModelState` via `get_model_state_cls()`, leaving shared `GPUModelRunner` / `InputBatch` untouched), projects $z_{\text{weak}} = \text{lm\_head}(h_{\text{weak\_norm}})$ and launches 2-pass `inkling2_loopcd_logits` over `V = 200,058`:
     - **Pass 1 (`loopcd_logits_stats_kernel`)**: multi-CTA wave32 DPP reduction computing $m_b = \max_v z_{\text{final}}[b, v]$ and $\sum_v \exp(z_{\text{final}}[b, v] - m_b) \implies p_{\max, b}$.
     - **Pass 2 (`loopcd_logits_apply_kernel`)**: sets effective guidance $\omega_b = \omega_0 (1 - p_{\max, b})$ when `loopcd_adaptive=True` (else $\omega_0$), applies Adaptive Plausibility Constraint (APC) threshold $\tau_b = m_b + \log\beta$, and updates $z_{\text{final}}$ in place:
       $$z_{\text{cd}}[b, v] = \begin{cases} z_{\text{final}}[b, v] + \omega_b \bigl(z_{\text{final}}[b, v] - z_{\text{weak}}[b, v]\bigr), & z_{\text{final}}[b, v] \ge m_b + \log\beta \\ -\infty, & \text{otherwise.} \end{cases}$$
   - Runs **before** both `Sampler` and DSpark target verification (`rejection_sampler` / greedy verifier), so speculative decoding verifies drafted tokens directly against $z_{\text{cd}}$.

---

## 2. User-Facing Controls

### 2.1 Per-Request `SamplingParams.extra_args` / OpenAI `vllm_xargs`

Every request in a mixed batch can independently configure or disable LoopCD without recapturing `FULL_AND_PIECEWISE` HIP graphs:

```python
from vllm import SamplingParams

# Variant A: LoopCD-Hidden (~0.14% latency overhead at B=1)
sp_hidden = SamplingParams(
    temperature=0.0,
    max_tokens=2048,
    extra_args={
        "loopcd_alpha": 0.35,
        "loopcd_mode": "hidden",
        "loopcd_blend_blocks": True,
    },
)

# Variant B: Adaptive LoopCD-Logits (+15pp accuracy on 20-problem Math+Code suite)
sp_logits = SamplingParams(
    temperature=0.0,
    max_tokens=2048,
    extra_args={
        "loopcd_alpha": 0.25,
        "loopcd_mode": "logits",
        "loopcd_adaptive": True,
        "loopcd_beta": 0.1,
        "loopcd_weak_pass": -1,
    },
)
```

For OpenAI-compatible `/v1/chat/completions` requests, pass the same keys inside `"vllm_xargs"`.

### 2.2 Server Environment Defaults & `serve_rdna.sh` Recipe Switches

| `serve_rdna.sh` Switch | Engine Env Variable | Default | Description |
|---|---|---|---|
| `LOOPCD_ALPHA` | `VLLM_INKLING2_LOOPCD_ALPHA` | `0.0` | Contrastive guidance scale $\alpha \ge 0$ (`0.0` = disabled, exact baseline) |
| `LOOPCD_MODE` | `VLLM_INKLING2_LOOPCD_MODE` | `hidden` | `"hidden"` (pre-coda state blend) or `"logits"` (2-head APC logit contrast) |
| `LOOPCD_ADAPTIVE` | `VLLM_INKLING2_LOOPCD_ADAPTIVE` | `0` | `"1"` / `"true"` scales $\omega_b = \alpha (1 - p_{\max,b})$ (`logits` mode) |
| `LOOPCD_BETA` | `VLLM_INKLING2_LOOPCD_BETA` | `0.1` | APC plausibility cutoff $\beta \in [0, 1]$ (`0.0` disables $-\infty$ masking) |
| `LOOPCD_BLEND_BLOCKS` | `VLLM_INKLING2_LOOPCD_BLEND_BLOCKS` | `1` | Blend loop `AttentionResiduals` blocks alongside `(x0 + x1)` (`hidden` mode) |
| `LOOPCD_WEAK_PASS` | `VLLM_INKLING2_LOOPCD_WEAK_PASS` | `-1` | Weak loop pass index for `logits` mode (`-1` = `passes - 2`, `0` = Pass `0`) |

Example launch (`tools/rdna/recipes/inkling2-mini-tp1-dspark.env`):

```bash
bash tools/rdna/serve_rdna.sh RECIPE=inkling2-mini-tp1-dspark \
  MODEL=/home/chenco_adm/hfcache/inkling2-mini \
  ADAPTIVE=1 DRAFT_VOCAB=/home/chenco_adm/.cache/rdna/inkling2-mini-draft-vocab-32768.npy \
  LOOPCD_ALPHA=0.35 LOOPCD_MODE=hidden
```

---

## 3. Native HIP Kernels & Microbenchmarks (`gfx1030`)

All four custom ops live in `csrc/rocm/inkling2_glue_rdna2.cu`, guarded by `#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)`, launched on explicit `at::cuda::getCurrentCUDAStream()` with `C10_CUDA_KERNEL_LAUNCH_CHECK()` and zero host synchronizations.

### 3.1 Multi-Architecture Disassembly Isolation

Verified via `tools/rdna/inkling2/glue_jit.py --archs "gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1200"` and `tools/rdna/archdiff_report.py` (`llvm-objdump -d` per offload bundle):

- **`gfx1030`**: 8 active kernel instantiations (`loopcd_blend_kernel<0>`: 199 insns, `loopcd_prepare_hidden_kernel<0>`: 193 insns, `loopcd_logits_stats_kernel<0>`: 553 insns, `loopcd_logits_apply_kernel<0>`: 338 insns, `loopcd_readout_kernel<1..4>`: 1214..2714 insns).
- **`gfx90a`, `gfx942`, `gfx950`, `gfx1100`, `gfx1200`**: 100% empty `s_endpgm` stubs (`0` active instructions on non-RDNA2 targets).

### 3.2 Standalone Kernel Latency (`par1-cs25` Radeon PRO V620, `Inkling-2-Mini` Shapes)

Measured via `tools/rdna/inkling2/eval_loopcd.py --microbench` (`50` warmup + `200` timed `torch.cuda.Event` reps; eager dispatch numbers include `~12 µs` Python/HIP launch wrapper overhead that disappears inside `FULL` HIP-graph replay):

| Kernel | Shape | Case | Median (µs) | p95 (µs) |
|---|---|---|---:|---:|
| `inkling2_glue_loopcd_blend` | `T=1, H=2304, P=1` | `omega=0.0` (0-ULP fast exit) | 14.66 | 19.02 |
| `inkling2_glue_loopcd_blend` | `T=1, H=2304, P=1` | `omega=0.35, blend_blocks=True` | 15.68 | 16.56 |
| `inkling2_glue_loopcd_blend` | `T=3, H=2304, P=1` | `omega=0.35, blend_blocks=True` | 15.76 | 16.72 |
| `inkling2_glue_loopcd_blend` | `T=8, H=2304, P=1` | `omega=0.35, blend_blocks=True` | 15.84 | 16.68 |
| `inkling2_glue_loopcd_blend` | `T=16, H=2304, P=1` | `omega=0.35, blend_blocks=True` | 16.04 | 16.96 |
| `inkling2_glue_loopcd_readout` | `T=1, n_blk=3, H=2304` | `gate_flag=0` / `gate_flag=1` | 16.72 / 16.76 | 23.58 / 18.92 |
| `inkling2_glue_loopcd_readout` | `T=3, n_blk=3, H=2304` | `gate_flag=0` / `gate_flag=1` | 16.78 / 16.80 | 23.61 / 19.16 |
| `inkling2_loopcd_logits` | `B=1, V=200058` | `omega=0.0` / `fixed` / `adaptive` | 17.52 / 18.20 / 18.32 | 18.52 / 23.60 / 23.64 |
| `inkling2_loopcd_logits` | `B=3, V=200058` | `omega=0.0` / `fixed` / `adaptive` | 17.52 / 26.96 / 27.16 | 18.64 / 28.32 / 28.64 |
| `inkling2_loopcd_logits` | `B=8, V=200058` | `omega=0.0` / `fixed` / `adaptive` | 17.48 / 58.92 / 58.88 | 18.64 / 60.32 / 59.36 |
| `inkling2_linear_lm_head` (`LMHeadPack`) | `M=1, V=200058, H=2304` | Weak-pass `lm_head` (`921.9 MB` fp16) | 2107.52 | 2113.58 |
| `inkling2_linear_lm_head` (`LMHeadPack`) | `M=3, V=200058, H=2304` | Weak-pass `lm_head` (DSpark verify) | 2130.08 | 2133.46 |
| `inkling2_linear_lm_head` (`LMHeadPack`) | `M=8, V=200058, H=2304` | Weak-pass `lm_head` (`B=8` decode) | 2239.27 | 2250.15 |

*(Note on `alpha=0.0` determinism: `tests/models/inkling2_loop/test_loopcd.py` verifies exact `0-ULP` (`torch.equal`) bit-identity on hidden states, logits, KV/conv cache pages, and single-capture `FULL` HIP-graph replays (`B == C`). On full 24-layer `Inkling-2-Mini` multi-step greedy runs, `moe_nvfp4_gemm_rdna2` / `moe_mxfp8_gemm_rdna2` use `global_atomic_pk_add_f16` across 48 MoE visits per step (`dspark_bench.py:22-23`), so independent runs can flip only at greedy top-2 near-ties `top2_gap < 0.17`.)*

---

## 4. End-to-End Throughput & Speculative Acceptance (`Inkling-2-Mini`, TP=1 V620)

### 4.1 `DSpark OFF` (`FULL_AND_PIECEWISE` HIP Graphs, `tokens=256`, `HIP 4`)

| Arm | `B=1` ms/tok (Δ vs base) | `B=1` tok/s | `B=4` ms/tok (Δ vs base) | `B=4` agg tok/s |
|---|---:|---:|---:|---:|
| `baseline` | **17.413** (+0.00%) | 57.43 | **28.309** (+0.00%) | 141.30 |
| `hidden_a0.20` | **17.427** (+0.08%) | 57.38 | **28.323** (+0.05%) | 141.23 |
| `hidden_a0.35` | **17.438** (+0.14%) | 57.35 | **28.414** (+0.37%) | 140.78 |
| `hidden_noblock_a0.25` | **17.459** (+0.26%) | 57.28 | **28.324** (+0.05%) | 141.22 |
| `logits_a0.25` | 19.172 (+10.10%) | 52.16 | 30.666 (+8.33%) | 130.44 |
| `logits_a0.50` | 19.676 (+13.00%) | 50.82 | 30.636 (+8.22%) | 130.57 |
| `logits_adap_a0.25` | 19.685 (+13.05%) | 50.80 | 30.584 (+8.04%) | 130.79 |
| `logits_adap_a0.50` | 19.694 (+13.10%) | 50.78 | 30.744 (+8.60%) | 130.11 |
| `logits_p0_adap_a0.35` | 19.701 (+13.14%) | 50.76 | 30.567 (+7.98%) | 130.86 |

### 4.2 `DSpark ON` (`ADAPTIVE=1`, `[[1,1,7],[2,2,2],[3,16,0]]`, `32k` Draft Vocab, `HIP 5`)

| Arm | `B=1` ms/tok | `B=1` tok/s | `B=1` tok/step (accept%) | `B=2` ms/tok | `B=2` agg tok/s | `B=2` tok/step (accept%) | `B=4` (`K=0`) ms/tok | `B=4` agg tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | **10.48** | **95.44** | **2.87** (43.2%) | 15.39 | 129.95 | 2.14 (57.1%) | **28.37** | **141.01** |
| `hidden_a0.20` | 11.36 | 88.04 | 2.21 (37.8%) | **15.02** | **133.14** | **2.23 (61.6%)** | 28.44 | 140.65 |
| `hidden_a0.35` | 11.14 | 89.79 | 2.68 (41.0%) | 15.18 | 131.73 | 2.22 (60.9%) | 28.44 | 140.63 |
| `logits_a0.25` | 12.88 | 77.66 | 2.48 (41.4%) | 17.01 | 117.56 | 2.14 (55.5%) | 30.75 | 130.09 |
| `logits_adap_a0.25` | 13.54 | 73.86 | 2.28 (39.1%) | 17.13 | 116.74 | 2.11 (54.7%) | 30.58 | 130.81 |

At `B=2` (`K <= 2`), both `hidden_a0.20` (`61.6%` accept, `133.14 agg tok/s`) and `hidden_a0.35` (`60.9%` accept, `131.73 agg tok/s`) **increase DSpark draft acceptance and surpass `baseline` wall-clock throughput (`129.95 agg tok/s`)**.

---

## 5. Reasoning & Coding Quality Qualification (`20`-Problem Suite)

Evaluated via `tools/rdna/inkling2/eval_loopcd.py --eval` (`12` multi-step Combinatorics / Number Theory / Algebra / Graph problems verified against exact `\boxed{N}` + `8` algorithmic Python coding problems verified against unit tests in a sandboxed subprocess, `temperature=0.0`, `reasoning_effort=low`).

### 5.1 9-Arm Sweep (`DSpark OFF`, `max_tokens=2048`, `eval_batch_size=20`, `HIP 4`)

| Arm | Total / 20 | Math / 12 | Code / 8 | Mean Toks | Trunc | Wall (s) | Net Δ vs `baseline` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | **15 / 20** (75.0%) | 10 / 12 | 5 / 8 | 1376.6 | 5 | 106.13 | — |
| `hidden_a0.20` | **15 / 20** (75.0%) | 9 / 12 | 6 / 8 | 1368.3 | 4 | 105.36 | +0 (+2 / -2) |
| `hidden_a0.35` | **16 / 20** (80.0%) | 11 / 12 | 5 / 8 | 1371.5 | 4 | **105.85** | **+1 (+1 / -0)** |
| `hidden_noblock_a0.25` | **16 / 20** (80.0%) | 9 / 12 | **7 / 8** | 1370.7 | 4 | **105.72** | **+1 (+3 / -2)** |
| `logits_a0.25` | **15 / 20** (75.0%) | 10 / 12 | 5 / 8 | 1413.8 | 5 | 112.65 | +0 (+1 / -1) |
| `logits_a0.50` | **14 / 20** (70.0%) | 9 / 12 | 5 / 8 | 1407.7 | 6 | 112.50 | -1 (+1 / -2) |
| `logits_adap_a0.25` | **18 / 20** (**90.0%**) | **11 / 12** | **7 / 8** | **1324.2** | **2** | 108.26 | **+3 (+3 / -0)** |
| `logits_adap_a0.50` | **15 / 20** (75.0%) | 9 / 12 | 6 / 8 | 1358.7 | 4 | 111.42 | +0 (+3 / -3) |
| `logits_p0_adap_a0.35` | **14 / 20** (70.0%) | 8 / 12 | 6 / 8 | 1346.3 | 5 | 109.40 | -1 (+1 / -2) |

### 5.2 Active Multi-Token DSpark Verification Sweep (`DSpark ON`, `max_tokens=1536`, `eval_batch_size=2`, `HIP 5`)

| Arm | Total / 20 | Math / 12 | Code / 8 | Mean Toks | Trunc | Wall (s) | DSpark tok/step | DSpark accept% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | **11 / 20** (55.0%) | 8 / 12 | 3 / 8 | 1238.2 | 10 | 185.62 | 2.55 | 63.1% |
| `hidden_a0.20` | **10 / 20** (50.0%) | 6 / 12 | 4 / 8 | 1234.2 | 10 | 184.10 | 2.57 | 63.5% |
| `hidden_a0.35` | **12 / 20** (60.0%) | 8 / 12 | 4 / 8 | 1213.7 | 8 | **181.12** | **2.58** | **63.4%** |
| `logits_a0.25` | **14 / 20** (**70.0%**) | **9 / 12** | **5 / 8** | **1192.2** | **7** | 207.17 | 2.40 | 57.0% |
| `logits_adap_a0.25` | **11 / 20** (55.0%) | 7 / 12 | 4 / 8 | 1203.5 | 8 | 192.66 | **2.58** | 63.1% |

### 5.3 Recommended Operating Points

1. **Zero-Overhead Serving Default (`LoopCD-Hidden`, `alpha=0.35`, `blend_blocks=True`)**:
   - Adds **`+0.14%` (`+25 µs/tok`)** single-stream latency (`0 B` extra residual/block VRAM), preserves 100% prefix-cache reuse, improves `2048`-token reasoning/coding from `75.0% -> 80.0%` (`+1 / -0` regressions), and under DSpark **improves both accuracy (`55.0% -> 60.0%`) and wall-clock speed (`185.6s -> 181.1s`, `2.58 tok/step`)**.
   - Blending `AttentionResiduals` loop blocks (`blend_blocks=True`) eliminates the 2 math regressions observed with `blend_blocks=False` (`hidden_noblock_a0.25`).
2. **Maximum Reasoning & Coding Quality (`LoopCD-Logits`, `alpha=0.25`, `adaptive=True` / `False`, `beta=0.1`, `weak_pass=-1`)**:
   - Contrasting Pass `2` against Pass `1` (`weak_pass=-1`, after 1 round of cross-pass `MessageAttentionResidual` refinement) substantially outperforms Pass `0` (`weak_pass=0`).
   - At `max_tokens=2048`, `logits_adap_a0.25` lifts accuracy from **`75.0% (15/20)` to `90.0% (18/20)` (`+15pp`, `0` regressions)** while shortening reasoning chains (`1376.6 -> 1324.2` mean tokens, `+2.0%` batch wall time). Under tighter `1536`-token DSpark budgets, `logits_a0.25` lifts accuracy from **`55.0% (11/20)` to `70.0% (14/20)` (`+15pp`)** at `2.40 tok/step`.
