# Recipe: Ornith-1.5-9b-exl3-3bpw on gfx1030 / Radeon PRO V620

**Target**: vLLM inference on AMD RDNA2 (gfx1030 / gfx1031 / gfx1032) and
RDNA3 (gfx1100), focused on the EXL3 3bpw quantization path.
**Status**: validated end-to-end on .176 / venv-7.14.0 / 4× V620.
**Date**: 2026-09-05.

This recipe captures the exact production setup: which commits,
which env vars, what kernel paths are active, what performance you
should expect, what memory looks like. Use it as a baseline when
testing on other RDNA hardware (e.g., Steam Deck's RDNA2 APU).

## Hardware

| Component | Detail |
|---|---|
| GPU | AMD Radeon PRO V620 (gfx1030, 32 GB VRAM, Wave32, V_DOT2) |
| Test setup | 4× V620 on .176 (chenco_adm@192.168.1.176) |
| ROCm | 7.14.0 at `/opt/rocm/core-7.14` |
| Python | 3.12 (venv-7.14.0 at `/home/chenco_adm/Apps/vllm/venv-7.14.0`) |
| PyTorch | 2.12.0+rocm7.14.0 |
| Triton | 3.6.0 ROCm fork |

For **Steam Deck** (RDNA2 APU, ~16 GB shared RAM): scale the recipe to
a lighter model (e.g., 3B-4B EXL3 3bpw). The kernel work is identical,
the VRAM budget shrinks proportionally. See "Fitment notes for Steam
Deck" at the end.

## Repository / commits

```
opengfx1030/vllm-rdna branch rdna_extras
HEAD: c7053d482  perf(exl3): drop CG-PATH duplicate trellis/svh buffers (-2.45 GiB, 0% perf)
PREV: c3e525859  feat(exl3): VLLM_EXL3_MEMORY_MODE env var (full/packed), stub dispatcher
PREV: 63584d6bc  perf(attn): GQA-aware D=256 paged decode kernel (1.5-2.0x FA decode)
PREV: abc77aa31  perf(attn): fa_rdna2 decode kv_splits 8 -> 16
PREV: 6acee1bfc  perf(exl3): extend grain-based v2 dispatch to full decode batches (M<=8)
```

All commits are pushed to `opengfx1030/vllm-rdna` `rdna_extras`.

## Env vars (set in `/tmp/start_cg.sh`)

```bash
# Required for RDNA GPUs
export VLLM_ROCM_USE_AITER=0                    # disable CDNA-only backends
export VLLM_ROCM_USE_AITER_MOE=0                # disable AITER MoE
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE   # enable Triton FA
export VLLM_RDNA_FORCE_FP16=1                   # no BF16 emulation
export VLLM_USE_RDNA2_FA=1                      # enable FA-RDNA2 dispatch
export VLLM_GDN_HIP_KERNELS=1                   # GDN prefill/decode on HIP

# Performance
export TORCH_BLAS_PREFER_HIPBLASLT=0            # force rocBLAS on gfx1030
export PYTORCH_TUNABLEOP_ENABLED=1              # GEMM autotuning
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0    # disable hipBLASLt tuning (broken on gfx1030)
export PYTORCH_TUNABLEOP_FILENAME=$HOME/.cache/tunableop/tunableop_results.csv  # stable cache
export VLLM_BATCH_INVARIANT=0                   # don't force cublasLt
export GPU_MAX_HW_QUEUES=2                      # RDNA2 8-HQD budget
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# v2 model runner + compile cache control
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_AOT_COMPILE=0
export VLLM_DISABLE_COMPILE_CACHE=1

# Misc
export HIP_VISIBLE_DEVICES=0
```

**Option C was tested and dropped** — see the dedicated section below.

## Launch command

```bash
bash /tmp/start_cg.sh
```

The script does:
```
source /home/chenco_adm/Apps/vllm/venv-7.14.0/bin/activate
export LD_LIBRARY_PATH=...rocm_sdk_libraries/lib:...rocm_sdk_core/lib/host-math/lib:...
# All env vars above
cd /tmp
python -m vllm.entrypoints.cli.main serve /home/chenco_adm/models/Ornith-1.5-9b-exl3-3bpw \
  --port 18005 --tensor-parallel-size 1 \
  --max-model-len 200000 --max-num-seqs 8 \
  --dtype float16 --gpu-memory-utilization 0.91 \
  --block-size 16 \
  --language-model-only --skip-mm-profiling --trust-remote-code
```

The `_rocm_C.abi3.so` is pre-built and ships with the editable install
in venv-7.14.0. To rebuild after .cu changes:

```bash
ssh chenco_adm@192.168.1.176
cd /home/chenco_adm/opengfx1030_vllm-rdna
rm -rf build/ .deps/ vllm/*.abi3.so
/tmp/rebuild_so.sh  # uses CCACHE_DISABLE=1, PYTORCH_ROCM_ARCH=gfx1030, etc.
```

## Model & VRAM

| Spec | Value |
|---|---|
| Model | `Ornith-1.5-9b-exl3-3bpw` (Qwen3.5-style hybrid: 64 layers, 48 GDN / 16 full-attn) |
| Disk (active checkpoint) | **6.4 GB** safetensors + ~20 MB tokenizer/config |
| Disk (full dir incl. .bak) | 12 GB (5.2 GB `.bak` from prior exl3lmhead re-quant — leftover, not loaded) |

**VRAM partition (32 GB V620, gpu-memory-utilization=0.91 → 27.29 GiB budget):**

| Bucket | Size | Notes |
|---|---|---|
| Weights (post-Option-A) | **~6.65 GiB** | EXL3 trellis ~2.527 GiB + dense fp16 ~1.908 GiB + scales ~0.005 GiB + bf16→fp16 cast buffer (peak during load) |
| KV cache | **18.36 GiB** | 591,688 tokens (2.96× the 200k max-model-len) |
| Cudagraph | 0.66 GiB | FULL_AND_PIECEWISE capture at sizes [1, 2, 4, 8, 16] |
| Peak activation | 0.37 GiB | |
| Conserved overhead | ~0.66 GiB | |
| **Total resident** | **~25.5 GiB** | |
| Budget | 27.29 GiB | 0.91 × 29.98 GiB free |

**Loading peak**: 7.34 GiB "Model loading took" (post-Option-A). Was 9.8 GiB
before Option A — the 2.46 GiB drop is the staging-copy elimination.

## Performance (Ornith-1.5-9b-exl3-3bpw, TP=1, seed 4242)

### 1k in / 512 out (matrix)

| Conc | TTFT | Prefill tok/s | Median TPOT | Decode tok/s/req | Aggregate output tok/s |
|---|---|---|---|---|---|
| c=1 | 1.14s | ~900 | 25.7ms | 38.9 | 35.9 |
| c=4 | 2.84s | ~1440 | 38.5ms | 26.0 | 89.9 |
| c=8 | 4.53s | ~1810 | 52.0ms | 19.2 | 129.7 |

### 16k in / 1k out (matrix, cold cache)

| Conc | TTFT | Prefill tok/s | Median TPOT | Decode tok/s/req | Aggregate output tok/s |
|---|---|---|---|---|---|
| c=1 | 30.6s | ~535 | 29.1ms | 34.2 | 17.1 |
| c=4 | 33.5s | ~1960 agg | 95.6ms (outlier; warm 49.2ms) | 10.5 (warm 20.3) | 35.0 (warm 79.2) |
| c=8 | 44.1s | bursts @1588 | 134.5ms | 7.4 | 44.8 |

### 16k warm-cache (post-prefix-cache, what production sees after warmup)

| Conc | TTFT (warm) | Median TPOT | Aggregate output tok/s |
|---|---|---|---|
| c=1 | 0.38s | 28.6ms | 34.6 |
| c=4 | 1.11s | 49.2ms | 79.2 |
| c=8 | 1.36-1.92s | 69.1ms | 113.0 |

**Comparison vs the pre-Option-A baseline** (also seed 4242, fresh server):
- 1k cells: identical to within 0.2% noise (no perf change as expected).
- c=8 16k matrix cold: TPOT 154 → 134.5ms (~−13% on the same matrix sweep).
- 16k warm-cache rerun: TPOT 134.93 → 69.1ms (apples-to-warm-oranges vs
  baseline cold; the freed KV cache gives the scheduler more headroom for
  chunked-prefill/decode interleaving at high concurrency, which is the
  real production benefit).

## What each commit contributes (debt ledger)

| Commit | Purpose | Effect |
|---|---|---|
| `6acee1bfc` EXL3 v2 dispatch | Grain-based decode kernel for M ≤ 8 | EXL3 decode 168 → 25.7ms TPOT at c=1 (5.9 → 38.5 tok/s/req) |
| `abc77aa31` FA kv_splits 8→16 | More partials = finer combine | FA decode 4-19% of BW (vs 17%); small at every cell |
| `63584d6bc` GQA-aware FA decode | One CTA per (token, kv-head, split) reuses KV tile | c=4 16k: TPOT 97.9 → 76.8ms (−22%); c=1 16k: 33.6 → 29.3ms (−13%) |
| `c3e525859` MEMORY_MODE stub | Env var hook for future packed-trellis work | No-op today (both modes call same kernel); dispatcher wired for Option D when/if it lands |
| `c7053d482` Drop CG-PATH duplicates | Stride-view layer.trellis in CG-PATH (no int16 staging copy) | −2.45 GiB VRAM; +15.4% KV tokens (591k vs 512k); ~−13% c8 16k cold TPOT; −36% to −49% warm |

## What does NOT run on this stack

- **Triton attention decode** (except for 1 Triton JIT compile during warmup of `fused_sigmoid_gating_delta_rule_update_kernel` — GDN update path). Total Triton JIT warnings during the entire matrix run: 1.
- **AITER / CK** — `VLLM_ROCM_USE_AITER=0`, CDNA-only backends disabled.
- **hipBLASLt GEMM** — `TORCH_BLAS_PREFER_HIPBLASLT=0`, rocBLAS forced on gfx1030.
- **cudagraph FP8 / INT8 GEMM** — not used in this path (EXL3 is the quantization).

## Option C — tested, dropped

**Hypothesis**: PyTorch's caching allocator doubles peak memory during
safetensors load (destination GPU tensor + temporary GPU staging buffer
for the CPU→GPU copy). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
should reduce fragmentation.

**Tested**: ran with `expandable_segments:True` in `/tmp/start_cg.sh`,
restarted, measured:

| Metric | Default | expandable_segments=True |
|---|---|---|
| Loading peak | 7.34 GiB | 7.31 GiB (no change) |
| Consumed (steady) | 8.55 GiB | **9.39 GiB** (+0.84 GiB) |
| KV tokens | 591,688 | **564,675** (−4.6%) |

**Conclusion**: **Option C as configured doesn't help and slightly
hurts steady-state memory.** The `expandable_segments:True` allocator
trades fragmentation reduction for slightly higher per-block overhead.
Our workload (single contiguous model load, no fragmented allocations)
doesn't trigger the fragmentation pattern this flag is designed for.

**Also tried**: `PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:False`.
This avoids the power-of-2 bucket rounding that wastes memory on
small allocations. Result: no measurable improvement on this workload.

**Why "loading peak vs steady state" looked like 2×**: pre-Option-A
the loading peak was 9.8 GiB and the steady-state was 11.0 GiB
(non-torch consumed 1.2 GiB included CG-PATH staging copies, plus
cudagraph + activation working set). Option A's −2.45 GiB reduction
already cut the peak to 7.34 GiB. The remaining "gap" between
7.34 (peak) and 8.55 (steady) is just cudagraph capture + activation
working set, not allocator waste.

**Verdict**: Option C is **dropped**. No env var, no env tweak. The
loading peak is already well-controlled by Option A.

**What WOULD need Option C**: if you see loading peak >> steady by
>2× on a different workload (e.g., many small tensors, fragmented
checkpoint shards), retry with `expandable_segments:True` then.

## Fitment notes for Steam Deck (RDNA2 APU, ~16 GB shared RAM)

**Hardware profile** (Steam Deck OLED): Van Gogh APU, RDNA2, 16 GB
LPDDR5 unified memory shared between CPU + GPU. The APU exposes
~8-12 GB as GPU-visible VRAM (depending on system load and driver
settings). At 12 GB usable VRAM, the recipe above won't fit.

**Recommendations for Steam Deck**:

1. **Use a smaller EXL3 model.** The 9B model needs ~25.5 GiB resident.
   Try a 3B or 4B EXL3 3bpw model (~9-11 GiB resident at 200k context).
   Verify cache slot count with the gpu_worker log: `GPU KV cache size:`
   line.

2. **Reduce max-model-len.** At max-model-len=32768 (instead of 200000),
   the KV cache drops from 18.36 GiB to ~3 GiB. Total resident ≈ 10 GiB.

3. **Reduce gpu-memory-utilization.** Default 0.91 → try 0.85 (leave more
   headroom for system RAM sharing).

4. **Force FP16 everywhere.** Already done via `VLLM_RDNA_FORCE_FP16=1`.

5. **Skip MTP / speculative decoding.** Don't pass `--speculative-config`;
   MTP models need extra activation memory that competes with KV cache.

6. **Lower max-num-seqs.** Default 8 → 4. Fewer concurrent requests =
   smaller activations, smaller KV working set.

**What you CANNOT skip**: the EXL3 kernel commits (`63584d6bc`,
`abc77aa31`, `6acee1bfc`, `c7053d482`) — those are required for
correctness and perf on gfx1030. Option A's −2.45 GiB saving is even
more valuable on Steam Deck's tight memory budget (it lets the KV
cache hold ~79k more tokens = +15% concurrency headroom).

**Test plan on Steam Deck** (suggested):
1. Build a minimal vLLM container (Layer 1: ROCm base + PyTorch +
   Triton). The vllm-rdna fork's Layer 2 Dockerfile is a starting
   point, but the APU may need different `HSA_*` env vars.
2. Apply the 5 commits listed above to the vLLM source.
3. Build with `PYTORCH_ROCM_ARCH="gfx1035"` (Steam Deck's APU is
   gfx1035 — same RDNA2 ISA, just clocked lower).
4. Run with the env-var recipe above.
5. Smoke test: `vllm bench throughput --input-len 256 --output-len 64
   --num-prompts 4 --max-model-len 4096`. Check `GPU KV cache size`
   line in worker log to confirm ~40-60% KV cache vs total resident.
6. If OOM at load: drop gpu-memory-utilization to 0.75, or lower
   max-model-len further.

**Expected throughput on Steam Deck** (rough estimate, ~half the V620
memory BW + lower clocks): 0.3-0.5x the V620 numbers above for decode,
similar for prefill. The kernel is the same — only the hardware clocks
and memory BW differ.

## Reproducibility checklist

- [ ] Pull `opengfx1030/vllm-rdna` `rdna_extras` at HEAD `c7053d482`
- [ ] venv-7.14.0 active (`source /home/chenco_adm/Apps/vllm/venv-7.14.0/bin/activate`)
- [ ] `_rocm_C.abi3.so` present in `vllm/` (rebuild via `/tmp/rebuild_so.sh` if missing)
- [ ] All env vars in `/tmp/start_cg.sh` set (no `PYTORCH_CUDA_ALLOC_CONF`)
- [ ] `.cache/tunableop/tunableop_results.csv` exists for warm GEMM tuning
- [ ] Server starts on port 18005; `curl localhost:18005/v1/models` returns 200
- [ ] `GPU KV cache size: 591,688 tokens` in worker log (confirms Option A active)
- [ ] Golden probes: `Paris.`, `2\n1+1=2`, `Once upon a time` for the Ornith model

## Validation commands

```bash
# Server health
curl -s http://localhost:18005/v1/models | python3 -m json.tool

# ITL stream
source /home/chenco_adm/Apps/vllm/venv-7.14.0/bin/activate
python /tmp/stream_tokens.py  # expect ITL mean=25.1ms

# Golden probes
python /tmp/validate_prod.py  # expect "Paris." for "The capital of France is"

# Full matrix (1k + 16k cells, seed 4242)
rm -rf /tmp/bench_run_optA && /tmp/bench_fresh.sh
# Expect (per /tmp/bench_optA.log):
#   c1 1k512:   TPOT ~25.7ms, agg ~36 tok/s
#   c4 1k512:   TPOT ~38.5ms, agg ~90 tok/s
#   c8 1k512:   TPOT ~52ms,   agg ~130 tok/s
#   c1 16k1k:  TPOT ~29ms,   agg ~17 tok/s
#   c4 16k1k:  TPOT matrix ~95ms (outlier; warm rerun 49ms), agg warm ~79 tok/s
#   c8 16k1k:  TPOT matrix ~134ms (cold); warm rerun ~69ms, agg warm ~113 tok/s
```
