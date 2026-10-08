# DeepSeek-V4-Flash: fp16 attention on RDNA (plan)

Status: assessment, 2026-10-08. DeepSeek-V4-Flash (K128, MXFP4 experts) does not boot on
gfx1030 on either v0.31 or 0.28. The checkpoint loads after the fused-expert loader fix,
and fp16 mHC works. The attention path is the next and largest blocker, because upstream's
AMD DeepSeek-V4 attention assumes bf16 end to end (it was built for MI300).

Why fp16 and not bf16:
- gfx1030 has no bf16 math.
- `moe_mxfp4_gemm_rdna2` (`v_dot2_f32_f16`) only takes fp16 activations.

## Attention stage map (AMD path: `vllm/models/deepseek_v4/amd/rocm.py`)

| # | Stage | Today | bf16 coupling | RDNA status | Work |
|---|---|---|---|---|---|
| 1 | q/kv projections | `TritonFp8BlockScaledMMKernel` | none; follows activation dtype | works | – |
| 2 | q-norm + RoPE + KV insert | `_C.fused_deepseek_v4_qnorm_rope_kv_*_insert` (`csrc/libtorch_stable`, CUDA-style) | `STD_TORCH_CHECK` bf16 q/kv; cache row = 448 fp8 NoPE + 128 **bf16** RoPE | blocked | **new RDNA HIP kernel**: fp16 q/kv in, same cache row format out (RoPE part written as bf16) |
| 3 | Compressor | Triton | comments only; check the kernel's dtype casts | probably works | verify |
| 4a | Indexer q RoPE + fp8 quant | Triton `fused_indexer_q_rope_quant` | `assert q.dtype == bf16`; `.to(tl.bfloat16)` rounding inside | blocked | dtype-parametrize (small) or RDNA HIP |
| 4b | Indexer fp8 paged MQA logits | AITER (CDNA) / Triton / torch | fp8 in, fp32 out | `indexer_paged_mqa_rdna2.cu` exists | wire up + test |
| 4c | Indexer top-k | AITER top-k (CDNA) → fallback | none | fallback | verify fallback; HIP later |
| 5 | Sparse MLA **decode** | Triton ragged/partial kernels (bf16) | bf16 in the kernels | `sparse_mla_decode_rdna2` already selected (`_rdna2_mla_available`), fp16 + bf16 q | ✓ (check the fp16 output path) |
| 6 | Sparse MLA **prefill** | `dequantize_and_gather_k_cache` → bf16 workspace → Triton ragged prefill | workspace hard-coded bf16 (`rocm.py` x2); Triton kernel bf16 | `sparse_mla_prefill_rdna2` exists in `_rocm_C`, **not wired** | wire up the HIP prefill; workspace dtype = `q.dtype` |
| 7 | Inverse RoPE + `wo_a` bmm + `wo_b` | Triton `_inverse_rope_gptj` + einsum with cached **bf16** `wo_a` | assert + bf16 weight cache + bf16 casts | blocked | dtype-parametrize (cache `wo_a` in the activation dtype), or an fp8 `wo_a` bmm with fp16 output |
| 8 | mHC pre/post | torch fallback | fixed (`mhc/torch.py` accepts fp16) | works | – |
| – | Attention-output and workspace buffers | `torch.empty(..., bfloat16)` in a few places | dtype hard-coding | blocked | use the activation dtype |

KV cache format: keep the existing fp8 DS-MLA row (448 fp8 NoPE + 128 bf16 RoPE). It is a
**storage** format. The decode/prefill HIP kernels already read the RoPE part as bf16 and
convert to fp32, so fp16 compute does not need a new cache layout. Only the writer
(stage 2) has to accept fp16 input.

## Effort

| Item | Size | Notes |
|---|---|---|
| Stage 2 RDNA HIP KV insert | M (port ~400 lines) | Byte-identical cache rows vs the CUDA kernel's output for bf16-rounded inputs (test) |
| Stage 6 wire `sparse_mla_prefill_rdna2` | S–M | Kernel exists; needs contract check vs upstream's ragged prefill (indices, sink, SWA combine) |
| Stage 4a / 7 dtype parametrization | S | Triton; RDNA-gated where behaviour changes |
| Stage 4b/4c wiring | S | Kernel exists for 4b |
| Workspace/output dtype plumbing | S | |
| Tests + bring-up (TP=4, then TP=8) | M | Correctness vs the torch reference per op, then serve probes |

Overall: **medium**. Most heavy kernels (decode, prefill, MQA logits) already exist as RDNA2
HIP in the fork. The new kernel is the KV insert; the rest is plumbing and
dtype-parametrizing upstream Triton.

## Proposed layout (clean multi-RDNA support)

C++/HIP: one folder per model feature, one translation unit per kernel, shared arch traits.

```
csrc/rocm/rdna/
  common/
    arch.cuh          # RdnaArch traits: wave32, has_dot2_f16, has_wmma (gfx11/12),
                      # has_fp8_wmma (gfx12), LDS size; compile-time from
                      # __gfx10*/__gfx11*/__gfx115*/__gfx12* macros
    convert.cuh       # fp16/bf16/fp8(e4m3, fnuz)/e8m0 helpers
  dsv4/
    kv_insert.cu      # stage 2 (fp16/bf16 in, fp8+bf16-rope cache row out)
    sparse_mla_decode.cu   # moved from csrc/rocm/sparse_mla_rdna2.cu
    sparse_mla_prefill.cu
    indexer_mqa_logits.cu  # moved from indexer_paged_mqa_rdna2.cu
    torch_bindings.cpp     # _rocm_C::dsv4_*_rdna ops (one schema per op)
```

- Kernels are templated on `RdnaArch`; the host wrapper picks the instantiation from the
  device arch (gfx10.3 / gfx11 / gfx11.5 / gfx12). RDNA3+ variants can then use WMMA
  without forking the file.
- CMake builds `csrc/rocm/rdna/**` only for RDNA targets in `PYTORCH_ROCM_ARCH`.

Python: one selection point, no scattered `if on_rdna`.

```
vllm/models/deepseek_v4/amd/rdna/
  __init__.py
  ops.py         # thin wrappers + register_fake for the _rocm_C dsv4 ops
  attention.py   # DeepseekV4RDNAAttention(DeepseekV4ROCMAiterMLAAttention):
                 #   overrides _fused_qnorm_rope_kv_insert, forward_mqa (prefill/decode),
                 #   _o_proj, workspace/output dtypes; fp16 throughout
```

- `amd/model.py` instantiates `DeepseekV4RDNAAttention` when `on_rdna_family()`; the CDNA
  path (AITER, bf16) stays untouched.
- Tests: `tests/kernels/attention/rdna/dsv4/` (per-op vs torch reference, fp16 + bf16
  inputs) and a serve probe recipe (`tools/rdna/recipes/dsv4-flash.env`).
- Over time, move the existing flat `csrc/rocm/*_rdna2.cu` into `csrc/rocm/rdna/<feature>/`
  the same way (separate PRs).

## Order of work

1. Layout skeleton plus moving the two existing DSV4 kernels (no behaviour change; tests pass).
2. Stage 2 KV-insert HIP kernel + test.
3. Output projection (stage 7) and indexer q-quant (4a) dtype work; workspace dtypes.
4. Wire `sparse_mla_prefill_rdna2` (6) and the MQA logits kernel (4b).
5. `DeepseekV4RDNAAttention` + boot TP=4 (fp16), probes, then FULL_AND_PIECEWISE
   cudagraphs (required), then TP=8.
