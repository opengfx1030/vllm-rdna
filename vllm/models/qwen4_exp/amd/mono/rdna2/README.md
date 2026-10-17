# Qwen4Exp MonoKernel on gfx1030 (RDNA2)

Decode-step kernels for Flash-Next (Qwen4Exp, symmetric W4A16 MoE) on
Radeon PRO V620, behind the generic `--monokernel` switch (upstream
vllm-project/vllm#61082).

```bash
vllm serve ... --monokernel            # or VLLM_ROCM_MONO_DECODE=1
```

`GPU_MAX_HW_QUEUES=4` is required (the Flash-Next recipes in
`tools/rdna/recipes/` set it): with 2 HW queues CLR shares HSA queues between
streams and one to three TP ranks run their attention kernels ~10 % slower
next to the persistent kernel. If the launcher sets less, the engine raises it
to 4 for the workers (`hw_queues` knob).

## Parts (all on with the flag)

| Part | File | What runs | Width gate |
| --- | --- | --- | --- |
| `k1` | `k1.py` | attention-side HC preamble (combine, norm, down+inject, up-mix) on the persistent kernel | <= 24 rows |
| `k2` | `k2.py` | FFN block (HC combine+mix, router, shared expert, routed W4A16 experts) in one persistent launch | <= 8 rows |
| `gdn` | `gdn.py` | GDN decode core (conv update, delta rule, gated RMSNorm) in one launch | <= 64 rows, pure decode / MTP verify |
| `qsa` | `qsa.py` | bounded decode indexer, HIP sparse decode (>= 3 rows), fused norm+RoPE, one-launch indexer stores | decode / verify |

Overlap with the generic HIP QSA path (`VLLM_RDNA_QSA_HIP=1`, default on):
the fused sparse decode is HIP and takes precedence over
`qsa_sparse_paged_attention_rdna2`; the one-launch indexer store and the fused
norm+RoPE are Triton, so `store_rows3` stands down while the HIP gate is on (the
caller's three `qsa_store_cache_rows` are HIP) but `norm_rope` does not -- a
provably zero-Triton run omits `--monokernel`.

`layout.py` holds the per-layer weight views, the workspace and the rdna_ar
attach; `debug.py` the in-serve timing. Kernels: `csrc/rocm/mono_k2_rdna2.cu`
(K1 + K2), `mono_gdn_rdna2.cu`, `mono_qsa_decode_rdna2.cu`. Torch ops:
`vllm::qwen4_exp_mono_{k1,k2,gdn,mark}_rdna2`. Wider batches, prefill and the
MTP drafter keep today's modules.

## Knobs (only read with `--monokernel`)

`--additional-config '{"rocm_mono_decode": {...}}'`, keys of `Options` in
`__init__.py`: `k1`, `k1_max_rows`, `k2`, `k2_max_rows`, `k2_modes`,
`k2_num_wg`, `k2_sleep`, `k2_flags`, `k2_ar` (FFN all-reduce in the kernel),
`gdn`, `gdn_max_rows`, `gdn_graph` (record the GDN core in FULL graphs),
`qsa`, `timing`, `modules` (timing reference), `hw_queues`. Unknown keys are an
error.

## Refusals

With `--monokernel`, startup fails with `ValueError("--monokernel <reason>.")`
when the GPU is not gfx10x, the model is not Qwen4Exp, Model Runner V2 is off,
PP > 1, the `_rocm_C` MonoKernel ops are missing, or the MoE is not symmetric
W4A16 on `CompressedTensorsWNA16RDNA2MoEMethod`. Each rank logs one
`mono rdna2: --monokernel on TP rank N: ...` line with the active parts.

Tests: `tests/models/qwen4_exp/test_amd_mono_rdna2*.py`. Tools and serve
drivers: `tools/rdna/mono/`. History: `docs/rdna/monokernel/`.
