# Explore: unintegrated PR #5 wvSplitK + rdna_ar pieces

Draft explore branch. Original Flash-Next / V620 candidate work is from
[PR #5](https://github.com/opengfx1030/vllm-rdna/pull/5)
(George Muravei-Alkhavoi / GeorgeMA-Strong and co-authors).

## In this PR

| Item | Why |
| --- | --- |
| Independent `wvSplitK` output buffers | Shared `Rdna2PersistBuf` lets a later GEMV overwrite a retained projection. Restore per-call `torch::empty`. |
| Qualified gfx1030 FP16/BF16 `wvSplitK` for **n = 1..5** | Revisit vs blanket `gemv_f16_rdna2` for all gfx10x n≤8. GEMV remains for other gfx10x, for n=6..8 on gfx1030, and when `VLLM_RDNA_DENSE_GEMV=1`. |
| `rdna_ar` device-index normalization | Backend is deprecated as a default path but still opt-in. Integer device indices avoid Torch API footguns that previously disabled init/self-test. |

## Explicitly not in this PR

These touch areas where `rdna_extras` already has different work, or are
larger framework ports still under separate investigation:

- TP4 conversation / prefix-cache recovery and state-boundary retention
- Circular-cache exclusions from prefix caching
- GDN gate batch-stride / LDS output-race fixes in `gdn_prefill_o_rdna2`
- FULL-graph capture/replay policy with compilation disabled
- CPU PLE meta-discovery / BF16 table path (already via PR #8)

## How to A/B the decode path

```bash
# Default on this branch (gfx1030): wvSplitK for 1..5 tokens
unset VLLM_RDNA_DENSE_GEMV

# Force donor gemv_f16_rdna2 even for 1..5 (previous rdna_extras behavior)
VLLM_RDNA_DENSE_GEMV=1
```

## Validation notes

- CPU/dispatch: `test_gfx1030_decode_dispatch` (monkeypatched, no GPU kernel needed for selection).
- GPU: `test_wvsplitk_retained_output_survives_later_call` requires ROCm + rebuilt native ops.
- `rdna_ar` remains default-off (`VLLM_RDNA_AR=0`).
