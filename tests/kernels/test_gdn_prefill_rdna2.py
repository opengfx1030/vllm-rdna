# SPDX-License-Identifier: Apache-2.0
"""Differential test: HIP GDN prefill chain vs the FLA Triton reference.

Guards the chunk-boundary state propagation in gdn_prefill_delta_h_rdna2
(regression: the kernel once indexed k/w/u/v_new with chunk-local token
indices, so chunks after the first read chunk 0's rows and left v_new
regions unwritten — garbage output for any prefill longer than one
64-token FLA chunk).
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx103" not in getattr(
        torch.cuda.get_device_properties(0), "gcnArchName", ""),
    reason="RDNA2 (gfx103x) GPU required",
)


def _inputs(T, FLA_CHUNK_SIZE, seed=0):
    torch.manual_seed(seed)
    B, H, K, V, Hg = 1, 4, 128, 128, 4
    q = torch.nn.functional.normalize(
        torch.randn(B, T, Hg, K, device="cuda"), dim=-1).half()
    k = torch.nn.functional.normalize(
        torch.randn(B, T, Hg, K, device="cuda"), dim=-1).half()
    v = (torch.randn(B, T, H, V, device="cuda") * 0.5).half()
    g_raw = (-0.05 * torch.rand(B, T, H, device="cuda")).float()
    beta = (torch.rand(B, T, H, device="cuda") * 0.9 + 0.05).float()
    initial_state = torch.zeros(1, H, V, K, dtype=torch.float32,
                                device="cuda")
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    return q, k, v, g_raw, beta, initial_state, cu_seqlens


@pytest.mark.parametrize("T", [64, 128, 192, 256])
def test_gdn_prefill_chain_matches_fla_reference(T):
    from vllm.third_party.flash_linear_attention.ops import (
        chunk_gated_delta_rule as ref_fn,
    )
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.third_party.flash_linear_attention.ops.utils import (
        FLA_CHUNK_SIZE,
    )
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        _gdn_prefill_chain_rdna2,
    )

    q, k, v, g_raw, beta, initial_state, cu_seqlens = _inputs(
        T, FLA_CHUNK_SIZE)
    chunk_indices = prepare_chunk_indices(cu_seqlens, FLA_CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, FLA_CHUNK_SIZE)

    o_ref, fs_ref = ref_fn(
        q=q, k=k, v=v, g=g_raw, beta=beta, initial_state=initial_state,
        output_final_state=True, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
        use_qk_l2norm_in_kernel=False)

    # The HIP chain consumes the per-chunk inclusive cumsum (what
    # gdn_prefill_prep_rdna2 produces); the reference cumsums raw g itself.
    g_hip = g_raw.clone()
    for c0 in range(0, T, FLA_CHUNK_SIZE):
        g_hip[:, c0:c0 + FLA_CHUNK_SIZE] = \
            g_raw[:, c0:c0 + FLA_CHUNK_SIZE].cumsum(dim=1)
    o_hip, fs_hip = _gdn_prefill_chain_rdna2(
        q=q, k=k, v=v, g_cumsum=g_hip.float(), beta=beta,
        initial_state=initial_state, scale=q.shape[-1] ** -0.5,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets)

    o_ref = o_ref.reshape(o_hip.shape)
    torch.testing.assert_close(o_hip.float(), o_ref.float(), atol=1e-2,
                               rtol=1e-2)
    torch.testing.assert_close(fs_hip.float(), fs_ref.float(), atol=1e-2,
                               rtol=1e-2)
