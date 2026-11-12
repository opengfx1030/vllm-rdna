# LeapQuant INT8 GDN recurrent state (opt-in, 2026-11-13)

Port of LeapQuant ([arXiv 2609.38166](https://arxiv.org/abs/2609.38166),
requested in [vllm#64725](https://github.com/vllm-project/vllm/issues/64725))
to the RDNA2 Gated DeltaNet path. Opt-in with `VLLM_RDNA_GDN_LEAPQUANT=1`
(gfx10x only; the flag raises elsewhere). Default off: the change is lossy.

## What it stores

The FP32 state of one value head (`[V, K] = [128, 128]`, 64 KiB) becomes one
28,768-byte blob (2.28x smaller):

| section | dtype | shape | bytes | offset |
|---|---|---|---:|---:|
| `z` INT8 residual | int8 | `[V, K]` | 16384 | 0 |
| `bscale` per-row scale | fp32 | `[V]` | 512 | 16384 |
| `csmooth` per-column smoothing | fp32 | `[K]` | 512 | 16896 |
| `comp_k` compensator keys | fp16 | `[r, K]` | 1024 | 17408 |
| `comp_u` compensator values | fp16 | `[r, V]` | 1024 | 18432 |
| `rec_k` token records | fp16 | `[cap, K]` | 4608 | 19456 |
| `rec_u` token records | fp16 | `[cap, V]` | 4608 | 24064 |
| `rec_g` decays | fp32 | `[cap]` | 72 | 28672 |
| header `[cursor, pending, 0, 0]` | int32 | `[4]` | 16 | 28752 |

`r = 4` compensator tokens, window 16, `cap = window + num_spec = 18` for MTP2.
The state is `replay(cursor records, Z*B*C + comp_u^T comp_k)`: the boundary
state (INT8 + rank-4 FP16 compensator of its top subspace, found by 2 power
iterations) plus the exact FP16 records `S_j = exp(g_j) S_{j-1} + u_j k_j^T`
since the boundary. Every 16 committed tokens the records are folded into a
new boundary state, so quantization error enters once per window instead of
once per token. The FP32 oracle is
`vllm/model_executor/layers/mamba/ops/leapquant_gdn.py`; the kernels are
`csrc/rocm/gdn_leapquant_rdna2.cu`.

## How it fits the serving path (MTP2, cudagraphs, prefix caching)

* **Slots are record counts.** The FP32 layout keeps one state per speculative
  token (`spec_state_indices[n, t]`). The blob lives at the home block
  `spec_state_indices[n, 0]`; "slot `t`" is "commit `t + 1` pending records".
  A decode/verify step appends its records and sets `pending`;
  `gdn_lq_normalize_rdna2` (first op of the next step) commits
  `num_accepted` of them (rejected drafts are dropped by never committing) and
  folds once `cursor >= window`. MTP verify needs no per-token state copies.
* **Readers of slot 0** (chunked prefill, non-spec decodes) commit
  `min(1, pending)`, the same state the FP32 path reads.
* **Prefill** materializes the blob to an FP32 `[N, HV, V, K]` initial state,
  runs the unchanged HIP/Triton chunk kernels, and stores the final state
  back (`gdn_lq_store_rdna2`). Inside a chunk nothing changes; LeapQuant error
  enters at chunk boundaries (MAXBAT 2048).
* **Prefix caching (`mamba-align`, V2 runner):** the align precopy/postprocess
  HIP kernels (`v1_mamba_rdna2.cu`, `copy_mamba_state_block`) copy the whole
  blob from the home block and write `cursor + min(bias + 1, pending)` into
  the copy's header, which is exactly the FP32 "copy slot `bias`" rule. The V1
  runner's memcpy specs and the Triton precopy fallback raise instead of
  silently copying wrong bytes.
* **Cudagraphs:** every op is a fixed-shape launch on the current stream that
  reads its metadata from device tensors, so FULL_AND_PIECEWISE captures it
  like the FP32 kernels.
* **Refused combinations** (raise at init): mono decode op, fused CUDA spec
  decode, AITER/FlashInfer GDN, align mode on the V1 runner, head dims other
  than 128.

## Use

```bash
bash tools/rdna/serve_rdna.sh RECIPE=flashnext-mtp2 VLLM_RDNA_GDN_LEAPQUANT=1
```

or uncomment `VLLM_RDNA_GDN_LEAPQUANT=1` in
`tools/rdna/recipes/flashnext-mtp2.env`. The worker log prints
`GDN recurrent state: LeapQuant INT8 (...)`.
