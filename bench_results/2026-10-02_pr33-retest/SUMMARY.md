# PR #33 re-test — FA-RDNA2 decode (all-wave scores, 4 barriers/tile)

**Date:** 2026-10-02
**Branch:** `pr33-retest` (bb40498cc + #33, re-authored `BlivionIaG <kev29lt@gmail.com>`)
**Model:** Qwen3.8-Flash-Next AWQ-W4A16, TP=4 (GPUs 4–7), FULL_AND_PIECEWISE,
prefix caching, `flashnext-mtp0` recipe
**Verdict:** **DO NOT MERGE.** The kernel win is real in isolation but the target
model never dispatches the kernel (its full-attention layers are QSA, not
FA-RDNA2). Clean 3-seed in-model A/B is parity.

---

## 1. What was measured

### 1a. Kernel-level (in-tree benchmark, exact in-model geometry)

`benchmarks/kernels/benchmark_fa_rdna2_decode.py`, D=256, H_q/H_kv=6/1,
kv_splits=16, cudagraph. Baseline vs #33 (µs/op, lower is better):

| batch | ctx | baseline | patched | Δ |
|---|---:|---:|---:|---:|
| 1 | 1024 | 19.2 | 15.8 | −18 % |
| 1 | 16384 | 183.9 | 126.9 | **−31 %** |
| 2 | 16384 | 350.9 | 242.2 | −31 % |
| 3 | 16384 | 505.8 | 360.5 | −29 % |
| 8 | 16384 | 1373.6 | 956.3 | −30 % |

(D=128 / D=256 at 32/8, 28/4, 16/4, 12/2 are the same −21 … −31 %.) The #33
kernel change is real and reproduces.

### 1b. In-model, clean MTP=0 A/B (low-variance acceptance)

Two boots, identical compile cache, only `_rocm_C.abi3.so` swapped (baseline
`a52cbf46`, patched `a373e447`), 3 seeds for every c=1 cell. `mean_tpot_ms`:

| cell | baseline | patched | Δ | spread (3 seeds) |
|---|---:|---:|---:|---:|
| c1 16384 | 23.336 | 23.333 | **−0.01 %** | 0.16 % |
| c3 16384 | 47.086 | 47.182 | +0.20 % | — |
| c1 1024 | 23.082 | 23.118 | +0.16 % | 0.02 % |
| c3 1024 | 30.257 | 30.366 | +0.36 % | — |
| c8 16384 | 69.209 | 69.869 | +0.95 % | — |
| c8 1024 | 40.842 | 40.696 | −0.36 % | — |

No affected cell wins; every cell is within run-to-run noise (and the c=1 cells
are the most stable at ≤0.16 %). Coherence 2/3 on both arms (the "gpu" probe
fails identically on both — pre-existing, not a #33 regression).

Raw: `remote/summary_baseline_mtp0.csv`, `remote/summary_patched_mtp0.csv`.

**MTP=2** (requested 8-cell matrix): not re-run. The draft path (`Mtf verify`)
also runs QSA, so the kernel is equally absent; the prior session's MTP2 A/B
(`bench_results/2026-10-01_pr33-fa-decode/raw_mtp2_im2b_*`, 8 cells × 3 seeds)
already showed parity inside a 5–55 % acceptance swing. With the structural
proof in §2 the MTP2 cells cannot move; spending two more 27B boots on them
would only re-measure spec-acceptance noise.

## 2. Root cause — the target model does not use the kernel

`vllm/models/qwen4_exp/amd/model.py:214-237`: for `full_attention` layers,

```python
use_qsa = getattr(config, "indexer_n_heads", None) is not None
if not use_qsa:
    self.self_attn = Qwen3NextAttention(...)     # -> RDNA_ATTN -> fa_rdna2_decode_paged
else:
    self.self_attn = Qwen4ExpQSAAttention(...)   # -> QSA Triton kernels
```

Flash-Next sets `indexer_n_heads = 4`, so **all 12 full-attention layers use
`Qwen4ExpQSAAttention`**; no standard attention layer is instantiated and
`RDNA_ATTN` / `fa_rdna2_decode_paged` is never dispatched. The serve log shows
the QSA kernels (`_qsa_mqa_paged_kernel`, `_qsa_sparse_paged_gqa_splitk_kernel`,
`_compress_qsa_groups_kernel`, `_expand_qsa_indices_kernel`), and there is no
`fa_rdna2_decode_paged` call anywhere under `vllm/models/qwen4_exp/`.

Corroborating arithmetic: the c=1 TPOT difference between 16k and 1k ctx is
23.336 − 23.082 = **0.254 ms/step**. Twelve per-head FA calls at 166–184 µs
would be ~2.0–2.2 ms/step. The kernel is not on the path.

## 3. rocprofv3 in-model kernel trace — attempted, tooling-blocked

Wrapped `serve_rdna.sh` from birth with venv `rocprofv3 --kernel-trace --stats`
(three attempts, `tools/rdna2_028/pr33_rocprof.sh`). The injected
`librocprofiler-sdk-tool` runs in-process and writes per-worker `.dat`, but under
this stack the finalizer defers on children and never flushes: SIGINT does not
exit vLLM, and KILL aborts the flush. Result: no CSV after a full boot +
measurement. `.dat` (641 MB × 4 workers) has no offline converter in the venv
(`rocpd` needs a `.rocpd` DB; the wheel's `rocpd` also refuses the host's
Python 3.14). The kernel-level benchmark (1a) plus the dispatch-path proof (2)
and the multi-seed A/B (1b) are used instead; none of them depend on acceptance
noise.

## 4. Improvements — measured, none pays on this model

**2a GQA-256 all-wave scoring.** Not implementable as the per-head layout: GQA
already spends one wave per head (`G` waves of 8), so there is no spare wave to
put 8 lanes on a key without serialising heads. Implemented the achievable
variant — four independent `fdot2` accumulators in the GQA S-phase (branch
`_rocm_C.gqa_ilp.so`, sha `aaa59006`). Measured (6/1, block 1024, GQA on):

| batch | ctx | #33 patched | +ILP | Δ |
|---|---:|---:|---:|---:|
| 4 | 16384 | 418.5 | 420.6 | +0.5 % |
| 8 | 16384 | 931.8 | 932.4 | +0.1 % |
| 4 | 1024 | 36.9 | 37.0 | +0.3 % |

Parity — the GQA kernel is not latency-bound on the score chain. Reverted.

**2b GQA row-threshold.** Measured the per-head vs GQA crossover at the
in-model geometry (6/1, D=256, block 1024, env `VLLM_FA_RDNA2_GQA_DECODE`):

| batch | per-head 1k / 16k (µs) | GQA 1k / 16k (µs) |
|---|---:|---:|
| 1 | 16.1 / 165.0 | 16.1 / 167.5 (gate off) |
| 2 | 24.0 / 273.6 | 24.0 / 273.0 (gate off) |
| 3 | 39.6 / 426.5 | 39.6 / 427.1 (gate off) |
| 4 | 48.1 / 514.2 | **36.9 / 418.5** |
| 8 | 97.0 / 1031.8 | **72.9 / 931.8** |

GQA only wins once it has ≥4 CTAs (batch×H_kv ≥ 4). The existing gate
`num_tokens * H_kv >= 4` sits exactly at the crossover, so lowering it to cover
c=1–3 would trade a 20–25 % kernel win at batch≥4 for a low-occupancy loss at
batch<4. **The gate is an occupancy guard and is correctly placed — do not
lower it.** (It is also moot for Flash-Next, which runs neither kernel.)

**2c Leaner #33 (dedup D=128/D=256 scoring helper, VGPR trim).** Not done: it is
a pure refactor with no measured perf payoff, and the branch is not landing.

## 5. What would make #33 land

1. Validate the in-model A/B on a model that actually routes decode through
   FA-RDNA2: an AWQ/GPTQ checkpoint whose `full_attention` layers use
   `Qwen3NextAttention` (i.e. `indexer_n_heads` absent), e.g. a non-QSA
   Qwen3.8-27B AWQ. The 6/1 kernel bench predicts 12 × (184→127) ≈ 0.68 ms/token
   at 16k — ~4 % TPOT if the share holds.
2. Or route QSA's main attention through FA-RDNA2 — a different, larger change.

## 6. Artifacts

- `remote/summary_baseline_mtp0.csv`, `remote/summary_patched_mtp0.csv`
- `remote/kb_patched_gqaoff.txt`, `remote/kb_patched_gqaon.txt` (2b crossover)
- `tools/rdna2_028/pr33_ab.sh` (clean A/B driver, no /tmp, PCI-SERR guard)
- `tools/rdna2_028/pr33_rocprof.sh`, `tools/rdna2_028/pr33_rocprof_parse.py`
  (rocprof driver + parser; parser is usable once a trace is obtained)
- `staging/fa_rdna2.{baseline,patched}.cu`
- Correctness: `tests/kernels/attention/test_fa_rdna2_writer_layout.py`
  → **56 passed** on the #33 build. No PCI SERR in any boot; all engines reaped.

## 7. Merge decision

**No merge to `rdna_extras`.** The change is correct and the kernel is faster,
but the scored model dispatches QSA kernels, not FA-RDNA2, so no affected cell
shows a win. `rdna_extras` is untouched; `pr33-retest` holds the evidence.
