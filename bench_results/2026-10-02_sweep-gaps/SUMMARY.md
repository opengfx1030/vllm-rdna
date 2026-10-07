# Sweep: remaining "not FA-RDNA2 / not HIP" gaps + PR#33 recovery

**Date:** 2026-10-02
**Branch:** `rdna_extras` (tip `bb40498cc`) on the 0.28.0 fork clone
**Box:** `par1-cs25`, GPUs 4–7, venv-7.14.0_0.28.0
**Verdict summary:** 3 items measured **no-gain**, 1 backend item **confirmed
correct (no fix needed)**, 1 item **root-caused + patch prepared (unvalidated,
stays off)**, and PR#33 **not merged** (kernel faster in isolation, but the
scored model never dispatches it; the 27B A/B that would show a win was
blocked by a chassis PCI SERR).

---

## 1. STEP-0 profile — the full-engine kernel table is tooling-blocked

The in-model GPU kernel table was **not obtainable** with any available
profiler on this stack:

| Tool | Attempt | Result |
|---|---|---|
| `rocprofv3 --run` (spawn) | wraps `vllm bench throughput` | **No CSV.** The launcher reports the wrapped engine as a `0.000000 sec` child and never merges the per-worker rocprofiler buffers. vLLM's `multiprocessing.spawn` tree breaks the spawn-mode tool (a minimal spawn probe produced nested/confused invocations). |
| `rocprofv3 --attach` | attach to a live process | **rocattach returns status 1.** `_rocm_sdk_core/bin/rocprof-attach` is missing from the venv wheel; the system `/opt/rocm/core-7.14` attach helper fails with `librocprofiler-sdk-rocattach.so ... non-zero status 1` even with `ptrace_scope=0`. |
| torch profiler (`--profile --profiler-config`) | vLLM built-in, runs in each worker | **Only CPU ops** (1.66 M `cpu_op` events). No `kernel`/CUDA events — this ROCm/PyTorch build's kineto does not surface HIP kernels. |

**What worked:** single-process `rocprofv3 --run` around a standalone probe.
Used for the MTP sampling kernels (item 1) and attempted for conv1d (item 2).
The CPU-side op table (torch profiler `profiler_out_*.txt`) is captured as
supplementary evidence but is CPU time, not GPU share.

### MTP sampling trio (item 1) — standalone probe, real geometry

`tools/rdna2_028/probe_mtp_sampling.py`, V=248320, num_spec=2, num_reqs=1,
200 iters, rocprofv3 kernel trace:

| kernel | GPU busy per step |
|---|---:|
| `_compute_local_logits_stats_kernel` | 3.9 µs |
| `_rejection_kernel` | 4.4 µs |
| `_resample_kernel` | 2.0 µs |
| `_insert_resampled_kernel` | 2.0 µs |
| **trio + insert total** | **~12.3 µs / step** |

At a c=1 decode step of ~23–30 ms, that is **~0.05 %** of the step. **NO-GAIN.**
The trio scans the 248 k vocab (3 logits × 31 blocks for stats, 243 blocks for
resample) but is memory-bound and tiny next to the GEMM/attention work.

---

## 2. Per-item verdicts

### Item 1 — MTP sampling kernels: **NO-GAIN**
~12.3 µs/step GPU busy, ~0.05 % of a decode step. Even fully eliminated, no
measurable serving win. No fused/HIP implementation warranted. (Evidence above.)

### Item 2 — conv1d HIP re-enable: **NO-GAIN (parity)**
A/B on Flash-Next MTP=0, c=1, TP=4+F&P+prefix caching (`wt_ab.sh`,
`CONV1D=1` HIP vs `CONV1D=0` Triton):

| cell | HIP (`CONV1D=1`) | Triton baseline (clean 3-seed retest) | Δ |
|---|---:|---:|---:|
| c=1 16384 | TPOT 23.38 ms | 23.336 ms | +0.19 % |
| c=1 1024 | TPOT 23.12 ms | 23.082 ms | +0.16 % |

Coherence 2/2 on the HIP arm. The HIP path engages at MTP=0 (the
`state_len == width-1` gate is satisfied) but is **parity** with the tuned
Triton kernel — conv1d is a small memory-bound op already well-served.
Verdict: **no-gain; leave the recipes at `VLLM_CAUSAL_CONV1D_RDNA2_*=0`**
(or re-enable — no measurable difference).

### Item 3 — `flashnext`/`full` attention backend: **confirmed correct, no fix**
Not silently on Triton. The boot log shows `attention_backend: 'RDNA_ATTN'`
and `rocm.py:1034 VLLM_USE_RDNA2_FA=1: pinning the attention backend to
RDNA_ATTN (FA-RDNA2)`.
- `flashnext-mtp0/mtp2` pass `--attention-backend RDNA_ATTN` explicitly.
- `flashnext.env` / `full.env` (`ATTN=none` + `VLLM_USE_RDNA2_FA=1`) get pinned
  to `RDNA_ATTN` by `rocm.py:1030-1036` (the `check_and_update_config` pin).
- `27b-awq` also passes `RDNA_ATTN`.

**But it is moot for Flash-Next**: all 12 `full_attention` layers route to
`Qwen4ExpQSAAttention` (`config.indexer_n_heads=4` → `use_qsa=True`),
confirmed by `qsa.py:622 QSA launch params` in the boot log and by the absence
of `fa_rdna2_decode_paged` in the path. So `ATTN=fa` vs `ATTN=triton` is
structurally parity for Flash-Next — there is no standard-attention layer to
switch. No recipe change needed.

### Item 4 — QSA-HIP fault: **root-caused; patch prepared, stays off**
**Root cause:** `csrc/rocm/qsa_rdna2.cu` launches both
`qsa_store_cache_rows_kernel` and `qsa_compress_groups_kernel` on the **null
stream** (`<<<grid, block>>>` with no stream arg) — the exact race class
documented in AGENTS.md ("capture `getCurrentCUDAStream()` and launch with
`<<<..., 0, stream>>>`"). The sibling 0.29/0.30 trees
(`opengfx1030_vllm-v030`, `opengfx1030_vllm-rdna`) **already carry the fix**
(`const cudaStream_t stream = at::cuda::getCurrentCUDAStream();` +
`<<<..., 0, stream>>>`); the 0.28.0 `rdna_extras` branch never received it.

Two further facts:
- The 0.28.0 HIP path also has the **old dtype contract** (`qsa_store_cache_rows_rdna2`
  requires `slots` int32, fp16-only rows/cache), while the real QSA metadata is
  int64 slots / int64 rope rows (per the AGENTS.md dtype/stride contract). So
  the path is **dtype-incompatible** and would fail loudly (TORCH_CHECK), not
  page-fault, in this tree. The intermittent page fault is a 0.29-tree
  phenomenon, not reproduced here.
- Prepared a minimal stream-fix patch (revertible; see `git diff` before
  revert) but **did not build or validate it** — a build + `VLLM_RDNA_QSA_HIP=1`
  smoke boot was abandoned because the box hit a **new PCI SERR** (below) and
  the task rule is to stop on a new PCI SERR.

Verdict: **keep the QSA-HIP path off** (default-off). The safe next step is to
port the full 0.29-tree fix (stream + dtype-generic + int64 slots) and validate
on the 0.29 tree, not here.

### Item 5 — PR#33 recovery: **kernel faster, but no serving win; NOT merged**
Prior retest (2026-10-02, `pr33-retest`) established:
- The #33 FA-RDNA2 decode change is **real and faster in isolation**: at the
  6/1 D=256 in-model geometry, −18 % (b1 1024) to **−31 %** (b1 16384) on
  `fa_rdna2_decode_paged`; consistent −21…−31 % across D=128/256 shapes.
- But **Flash-Next never dispatches it** (QSA), so the clean 3-seed MTP=0
  in-model A/B is parity (every cell within run-to-run noise; c=1 spread ≤0.16 %).

**Recovery attempt this session:** the model that DOES dispatch FA-RDNA2 is the
27B AWQ (`indexer_n_heads=None` → `Qwen3NextAttention` → `RDNA_ATTN`). Started a
baseline-vs-patched `.so` A/B on `27b-awq` (TP=4). **The box hit a NEW PCI SERR
and rebooted mid-boot** (below), so the 27B in-model A/B could not be completed.

Verdict: **do not merge #33 into `rdna_extras`.** It has no validated serving
win (Flash-Next unaffected), and the 27B A/B that would demonstrate one is
blocked by the chassis PCI-SERR (which the 27B boot at TP=4 with `RDNA_AR=1`
one-shot allreduce likely triggers — the documented "one-shot kernel
PCI-SERRs the host" risk). Recommended future step: validate #33 on 27B with
`VLLM_RDNA_AR=0` (fall back to RCCL) in a clean session.

---

## 3. PCI-SERR stop (mandate)

**NEW PCI SERR at 2026-10-02 12:23:54 CEST** — IPMI `e85` "Critical Interrupt |
PCI SERR | Asserted"; dmesg shows `aer_uncor_status: 0x00004000` on
`0000:80:01.1` (the same AER signature as the documented chassis SERR), and the
box rebooted at 12:24:20 (journal boot `0`). This occurred during the 27B boot.
Per the task rule ("stop on new PCI SERR"), all active benchmarking was halted
and the box left clean (all engines reaped, GPUs 4–7 idle, SERR count recorded).

---

## 4. Tooling added (uncommitted, `tools/rdna2_028/`)

- `step0_profile.sh` / `step0_parse.py` — rocprofv3-spawn in-model driver + parser (works only for single-process probes).
- `step0_torchprof.sh` / `step0_tp_parse.py` — vLLM torch-profiler driver + parser (CPU-only events on this build).
- `wt_ab.sh` — generic recipe-parameterized in-model A/B driver (server + coherence + cells + PCI-SERR guard).
- `probe_mtp_sampling.py` — single-process MTP sampling kernel probe.
- `probe_conv1d.py` — conv1d HIP/Triton probe (needs `has_initial_state`; not fully debugged).

## 5. Merge decisions

- **No merges to `rdna_extras`.** No item produced a validated, measured serving
  win. Items 1 and 2 are measured no-gain; item 3 requires no fix; item 4's fix
  is unvalidated and the path stays off; item 5 is not merged (no serving win).
- The prepared item-4 stream fix was **reverted** (not shipped unvalidated).
