# Inkling-2-Mini: which 27B recipe switches reach the model

Measured 2026-11-26 on `a2` (2× V620, TP=2) against the two recipes:
`tools/rdna/recipes/27b-hi2-tp2.env` (eight switches on) and
`tools/rdna/recipes/inkling2-hi2-tp2.env` (none of them set).

The 27B recipe turns on eight switches at once (`f7edb8b1e7` and the HI2 port).
Most of them are inert here, but two are not, and one that looks applicable is
inert for a reason worth writing down.

## On the Inkling path

| switch | 27B | Inkling | reader | note |
|---|---|---|---|---|
| `VLLM_FA_RDNA2_UNFUSED_PREFILL` | `1` | unset | `vllm/v1/attention/backends/rdna_attn.py:343` | the shared FA backend, so Inkling's full-attention prefill reaches it. Unmeasured on this model. |
| `VLLM_RDNA_HINT_FLEX_COMPUTE_MODE` | `f16` | unset | `vllm/model_executor/layers/quantization/rdna_hint_flex.py:151` | read when the expert quant config is built, i.e. on the path. Unmeasured here. |

Both are speed/numerics-of-quant settings, not quality: an unset value is not a
defect, it is an untested default. Measure them as an A/B before adopting.

## Not on the Inkling path

`RDNA_HINT_FLEX_GEMV`, `RDNA_HINT_FLEX_FUSE_SILU`, `RDNA_HINT_FLEX_PREFILL_GEMM`,
`VLLM_RDNA_GEMMA_RESIDUAL_DIRECT`, `VLLM_RDNA_GDN_DECODE_INPLACE` are never read
on `inkling2_loop` (checked in the code, not inferred from the recipe): Inkling
has no GatedDeltaNet, no Gemma-style residual, and its experts take the HI2
container path rather than the rdna_hint_flex `f16` compute mode.

## The lm_head threshold is a no-op for this model

`VLLM_RDNA_LM_HEAD_GEMV_MAX_M=32` (27B) versus the default 8 does nothing for
Inkling: `_gemv_rows_for_layer` returns 8 for a plain `nn.Module` under both
values, and Inkling's lm_head is one (`vllm/models/inkling2_loop/model.py:1264`)
served through `LMHeadPack`. Only a layer whose class is named
`ParallelLMHead` takes the 32-row kernel (`gemv_f16_rdna2_<8,16,true>` at n=16).

The A/B agreed: solo decode 33.00 ms (8) versus 33.25 / 33.30 ms (32) on `b4`,
inside the ≤1 % boot spread; the c=8 arm that boot produced 0 completions, so it
carries no measurement and is not quoted. Do not set the switch for this model.

## Rejected on measurement

`HSA_FORCE_FINE_GRAIN_PCIE=1`: 1k TPOT +6.5 %, b4 +1.1 % (one boot each).
Not adopted; it is a HIP-runtime knob with no reader in the fork.

## TunableOp

`inkling2-hi2-tp2.env` ships `TUNABLEOP=0`, from the recipe's first commit
(`43bb3e46ba`). The 27B's tuned rows (`~/.cache` / the fork's
`tunableop/rocblas-<hash>/`) are 27B shapes and do not transfer. Leaving it off
is honest only while it is unmeasured for Inkling shapes; retuning is a row
sweep of its own, not a recipe line to copy.
