# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA chunked prefill vs the slot-indexed recurrence.

The layer runs ``chunk_kda_with_fused_gate`` for prefill and
``fused_recurrent_kda`` for decode. The Triton recurrent kernel hangs on
gfx1030 for a 1-D slot tensor with more than one prompt token, so this test
uses a float32 torch recurrence as the prefill reference. It also covers the
gather/scatter state round-trip the layer performs around the chunk kernel.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm.models.glm5next.amd.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
    )
else:
    from vllm.models.glm5next.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
    )

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires CUDA or ROCm"
)

D = 128
LOWER_BOUND = -5.0


def _inputs(H: int, T: int, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(0)
    q = torch.randn(1, T, H, D, dtype=dtype, device=device) * 0.5
    k = torch.randn(1, T, H, D, dtype=dtype, device=device) * 0.5
    v = torch.randn(1, T, H, D, dtype=dtype, device=device) * 0.5
    raw_g = torch.randn(1, T, H, D, dtype=dtype, device=device)
    raw_beta = torch.randn(1, T, H, dtype=dtype, device=device)
    a_log = (torch.randn(H, dtype=torch.float32, device=device) * 0.5).contiguous()
    g_bias = (
        torch.randn(H * D, dtype=torch.float32, device=device) * 0.1
    ).contiguous()
    return q, k, v, raw_g, raw_beta, a_log, g_bias


def _naive_recurrence(
    q, k, v, raw_g, raw_beta, a_log, g_bias, initial_state, cu
):
    """fp32 reference for ``[1, T, H, D]`` inputs and ``[N, H, V, K]`` state."""
    H, K = q.shape[2], q.shape[3]
    qf, kf, vf = q[0].float(), k[0].float(), v[0].float()
    gf, bf = raw_g[0].float(), raw_beta[0].float()
    qf = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
    qf = qf * D**-0.5
    kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    gate = LOWER_BOUND * torch.sigmoid(
        a_log.exp()[None, :, None]
        * (gf + g_bias.view(H, K)[None])
    )
    beta = torch.sigmoid(bf)
    out = torch.empty_like(vf)
    final_state = initial_state.clone()
    for n in range(cu.numel() - 1):
        start, end = int(cu[n]), int(cu[n + 1])
        state = final_state[n].clone()
        for t in range(start, end):
            state = state * gate[t].exp()[:, None, :]
            u = beta[t][:, None] * (
                vf[t] - (state * kf[t][:, None, :]).sum(-1)
            )
            state = state + u[:, :, None] * kf[t][:, None, :]
            out[t] = (state * qf[t][:, None, :]).sum(-1)
        final_state[n] = state
    return out.unsqueeze(0), final_state


@pytest.mark.parametrize(
    ("H", "T", "dtype"),
    [
        (8, 64, torch.float16),
        (8, 128, torch.float16),
        (8, 512, torch.float16),
        (8, 1024, torch.float16),
        (64, 512, torch.float16),
        (8, 512, torch.bfloat16),
    ],
)
@torch.inference_mode()
def test_chunk_kda_prefill_matches_recurrent(H, T, dtype):
    device = torch.device("cuda")
    q, k, v, raw_g, raw_beta, a_log, g_bias = _inputs(H, T, dtype, device)

    cu = torch.tensor([0, T // 2, T], dtype=torch.int32, device=device)
    num_slots = 4
    indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True, False], device=device)
    state = torch.randn(num_slots, H, D, D, dtype=torch.float32, device=device) * 0.1
    state[indices[1]] = 0

    initial_state = gather_initial_states(state, indices, has_initial_state)
    out_chunk, final_chunk = chunk_kda_with_fused_gate(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        raw_g=raw_g,
        beta=raw_beta.float().sigmoid(),
        A_log=a_log,
        g_bias=g_bias,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
    )
    out_ref, final_ref = _naive_recurrence(
        q, k, v, raw_g, raw_beta, a_log, g_bias, initial_state, cu
    )

    torch.testing.assert_close(
        out_chunk.float(), out_ref.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(final_chunk, final_ref, rtol=1e-2, atol=1e-2)

    state_chunk = state.clone()
    scatter_states(state_chunk, final_chunk, indices)
    state_ref = state.clone()
    state_ref[indices.long()] = final_ref
    torch.testing.assert_close(state_chunk, state_ref, rtol=1e-2, atol=1e-2)
