# EXL3 27B mul1 + MTP=2 — TP=4 on gfx1030 (2026-10-01)

**Model**: `Qwen3.8-27B-exl3-3.00bpw` (codebook `mul1`, body bits 3.0, `mtp_bits` 4, head_bits 6).
**Box**: `par1-cs25`, venv-7.14.0_0.28.0, tree `/home/chenco_adm/vllm-rdna-0.28.0`, GPUs 4-7
(4x Radeon PRO V620, intra-PLX). **Tree**: `vllm-rdna-0.28.0`, branch `rdna_extras`.
**Config**: TP=4, `RDNA_ATTN` (FA-RDNA2), `FULL_AND_PIECEWISE`, prefix caching,
`--language-model-only --skip-mm-profiling`, KV cap 6e9 @ 0.90, `mtp_num_hidden_layers=1`,
`num_speculative_tokens=2`, `use_local_argmax_reduction=true`.

## Root cause (why MTP=2 did not load)

The checkpoint ships a **quantized MTP draft**: 39 `mtp.*` tensors incl.
`mtp.fc.{suh,svh,trellis,mul1}` and `mtp.layers.0.*.{suh,svh,trellis,mul1}`. Three gaps:

1. **`Exl3Config.get_quant_method` declined `mtp.fc`.** `fc` was missing from the
   module-name allowlist and `tensor_storage` (707 entries) enumerates only the
   target model, so `_exl3_suffixes` had no `fc` entry. The fc was built with
   `UnquantizedLinearMethod` (only `fc.weight`) → loader aborted with
   `no module or parameter named 'fc.suh'`.
2. **`_mtp_weights_unquantized()` misread EXL3 as unquantized.**
   `Quwen3_5MultiTokenPredictor.__init__` nulls `vllm_config.quant_config` for the
   MTP layers when the heuristic thinks the draft is unquantized. The heuristic
   only knew GPTQ/AWQ packed suffixes, so EXL3 `.trellis/.suh/.svh` looked like
   plain weights → MTP decoder layers built with only `.weight` →
   `no module or parameter named 'layers.0.mlp.down_proj.suh'`.
3. **RDNA2 HIP conv1d is incompatible with spec-sized conv state.**
   `causal_conv1d_fwd_rdna2` / `causal_conv1d_update_rdna2` assert
   `width == state_len + 1`, but speculative decoding sizes the conv state as
   `conv_kernel - 1 + num_spec` (`mamba_utils.py:202`), i.e. `state_len=5` for
   `num_spec=2`. The warmup crashed at `kernel_warmup`.

The dense `Exl3LinearMethod` also never received the checkpoint `codebook`, so its
`self.cb` stayed `0` (3inst); only `layers.N.*` markers were consulted. A `mul1`
draft head would have decoded with the wrong codebook.

**v0.30.0 reference**: `opengfx1030_vllm-v030` (`rdna_extra/v0.30.0`) has **no**
MTP+EXL3 wiring (its `qwen3_5_mtp.py` has no exl3 handling and its `exl3.py` is
byte-identical to the v0.28 HEAD); the `codebook=self.codebook` there is the MoE
path only. This is a minimal implementation, not a port.

## Fix (3 files, all EXL3/RDNA2-scoped)

- `vllm/model_executor/layers/quantization/exl3.py`: accept `fc` on the MTP draft
  (explicit `mtp.fc` exception to the storage-suffix heuristic); forward the
  config `codebook` into `Exl3LinearMethod` so `self.cb` defaults to the
  checkpoint codebook (markers still override per block). Non-RDNA is unaffected
  because `_rdna_exl3_available()` returns first.
- `vllm/model_executor/models/qwen3_5_mtp.py`: add EXL3's `.trellis/.suh/.svh` to
  `_QUANT_PARAM_SUFFIXES` so `_mtp_weights_unquantized()` does not strip the quant
  config from a quantized EXL3 draft.
- `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`: gate the RDNA2 HIP
  conv1d dispatch on `state_len == width - 1`, falling back to Triton for the
  spec-sized state. This replaces the manual
  `VLLM_CAUSAL_CONV1D_RDNA2_FWD=0 UPDATE=0` workaround that
  `serve_gfx1030_flashnext_mtp.sh` applies (the guard makes it automatic and
  launcher-independent, and keeps HIP for non-spec prefill).

## Load + coherence

- Weights load with **no** `fc.suh` / `down_proj.suh` error; model 8.35 GiB/worker.
- **Eager** (MTP=2): coherence PASS 2/2 (France→"Paris", 2+2→"4"), init 11.8 s;
  SpecDecoding mean acceptance length **2.68-2.84**, draft acceptance **84-92%**.
- **F&P** (MTP=2): coherence PASS 2/2; init 90.2 s (compilation 72.8 s); FULL
  cudagraphs captured for `[3,6,12,24]` (12 s, ~1.2 GiB).
- **Guard smoke** (MTP=2 eager, **no** conv env): booted init 11.5 s, coherence
  PASS 2/2 → the dispatch guard alone is sufficient.

## MTP=2 cells (F&P, TP=4, GPUs 4-7)

`vllm bench serve` `/v1/completions`, random, `--ignore-eos --temperature 0`,
seed 301-304, warm Triton cache. Per-req decode = output tok/s / concurrency;
prefill = input / TTFT at c=1. ITL median is burst-quantized under spec decode
(tokens arrive in groups of 3) — TPOT is the meaningful per-token latency.

| Cell | c | Prefill tok/s | Decode per-req | Agg out tok/s | TTFT (s) | ITL med (ms) | TPOT (ms) | Acc len | Acc rate |
|------|---|--------------:|---------------:|--------------:|---------:|-------------:|----------:|--------:|---------:|
| 1k/512  | 1 | 980  | 22.05 | 22.05 | 1.04 | 117.94 | 43.40 | 3.00 | 100% |
| 1k/512  | 8 | —    | 5.66  | 45.28 | 5.79 | 473.45 | 164.03 | 3.00 | 100% |
| 16k/1k  | 1 | 820  | 15.98 | 15.98 | 19.97 | 128.07 | 43.13 | 3.00 | 100% |
| 16k/1k  | 8 | —    | 3.07  | 24.52 | 94.17 | 510.21 | 227.19 | 3.00 | 100% |

**MTP=2 vs MTP=0** (same model/config, MTP=0 from the 2026-10-01 mul1 run):

| Cell | MTP=0 agg | MTP=2 agg | Δ |
|------|----------:|----------:|---:|
| 1k/512 c=1 | 18.91 | **22.05** | **+16.6%** |
| 1k/512 c=8 | **81.77** | 45.28 | −44.6% |
| 16k/1k c=1 | 13.68 | **15.98** | **+16.8%** |
| 16k/1k c=8 | **30.58** | 24.52 | −19.8% |

MTP=2 wins low-concurrency latency (+17% at c=1, TPOT 50.8→43.4 ms) and loses at
c=8 where the batched MTP=0 path is already efficient; the draft itself is perfect
(acceptance length 3.00 = all spec tokens accepted in every cell). The c=8 loss is
verify-path overhead, not draft quality — the EXL3 draft `fc`/verify GEMMs at
M=24 are the likely lever.

## TunableOp

Verify batches use `M = num_seqs x (1 + num_spec)`, so new GEMM keys at
**M ∈ {3, 6, 12, 24}** (per c=1/2/4/8) for the draft and target shapes. The shared
27B EXL3 set (783 rows) was tuned for MTP=0 M ∈ {1,2,4,8}; the MTP=2 M values are
not covered, so those GEMMs fall back to rocBLAS heuristics (lookup-only boot, no
tuning). Extending the set is a follow-up — the TunableOp unification is mid-flight
and owns `tunableop/`; the capture/tune/curate/freeze pipeline applies unchanged
(`tools/rdna2_028/exl3_tunableop_campaign.sh`).

## Artifacts

- `eager/` — eager rung: serve log, driver, coherence, acceptance, cell.
- `fp/` — F&P rung: serve log, driver, coherence, acceptance, 4 cell logs + JSON.
- `tools/rdna2_028/exl3_mtp2_validate.sh` — the driver (boot, coherence, 4 cells,
  SpecDecoding acceptance, PCI-SERR gate, per-run cache, own-PID hygiene).
- `guard_smoke/` — the no-conv-env guard verification (on the box).
