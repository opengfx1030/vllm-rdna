# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Glm5Next kernels for AMD RDNA (gfx1030).

Thin wrappers over ``_rocm_C`` ops built from ``csrc/rocm/rdna/glm5next/``.
Plain decode may call ``glm5_kda_recurrent`` only when
``VLLM_GLM5_KDA_HIP=1``. That kernel is opt-in; Triton
``fused_recurrent_kda`` stays the default until the HIP path has run on
hardware. The other ops are registered for the same ISA and are not
substituted for upstream Triton KDA, the indexer cache, or latent MLA.

Cache dtypes actually served on gfx1030:

* DSA / MLA KV (``cache_config.cache_dtype``, default ``auto``) resolves
  to the checkpoint dtype. GLM-5.3-Flash is bfloat16, and the public
  W4A16 checkpoint sets ``kv_cache_scheme`` to null, so the main KV is
  bf16. Nothing on RDNA defaults that cache to fp8. ``RocmPlatform``
  does not report an NVIDIA capability family, so SM100 ``fp8_ds_mla``
  selectors stay off. There is no HIP fp8 encode of this cache.
* Indexer K cache (``Glm5NextIndexerCache``) is the upstream naive fp8
  layout: uint8 e4m3 values plus a packed fp32 scale
  (``head_dim + head_dim // 128 * 4``). The live encode is the Triton
  kpool compress kernel (software absmax), not ``<hip/hip_fp8.h>`` and
  not ``convert.cuh``. The registered HIP DSA indexer is not called, so
  it does not encode or decode this cache. A future HIP fp8 path must
  use ``csrc/rocm/rdna/common/convert.cuh``.
* Indexer tail cache (``Glm5NextTailCache`` / ``KpoolTailSpec``) is bf16.
* KDA recurrent state is fp32 (``mamba_ssm_cache_dtype="auto"``) in the
  Mamba state group, a separate buffer from the DSA KV cache. Conv state
  follows ``mamba_cache_dtype`` (auto → model dtype, bf16).

``glm5_kda_prefill_kkt_rdna2`` uses 48 KB of LDS per 256-thread workgroup
(two workgroups per WGP). That prefill chain stays off the default path.
"""
