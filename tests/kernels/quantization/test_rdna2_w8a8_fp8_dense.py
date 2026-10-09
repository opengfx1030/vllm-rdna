# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gemm_w8a8_fp8_dense (RDNA2) over the full E4M3 range, vs torch."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 kernel", allow_module_level=True)
if not (
    hasattr(torch.ops, "_rocm_C") and hasattr(torch.ops._rocm_C, "gemm_w8a8_fp8_dense")
):
    pytest.skip("gemm_w8a8_fp8_dense not built", allow_module_level=True)

FP8 = torch.float8_e4m3fn


def _codes(shape, g):
    """Every finite E4M3 code (incl. subnormals and the 256..448 binade)."""
    c = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
    c[(c & 0x7F) == 0x7F] = 0x7E  # no NaN
    return c


@pytest.mark.parametrize("m", [1, 4, 33])
@pytest.mark.parametrize("per_block_act", [False, True])
def test_w8a8_fp8_dense_full_range(m, per_block_act):
    g = torch.Generator().manual_seed(m)
    k, n, gs = 1024, 512, 128
    a_q = _codes((m, k), g)
    b_q = _codes((k, n), g)
    b_s = torch.exp2(-torch.randint(8, 12, (k // gs, n), generator=g).float())
    if per_block_act:
        a_s = torch.exp2(-torch.randint(6, 10, (m, k // gs), generator=g).float())
        a_full = a_q.view(FP8).float() * a_s.repeat_interleave(gs, dim=1)
        groups = k // gs
    else:
        a_s = torch.exp2(-torch.randint(6, 10, (m,), generator=g).float())
        a_full = a_q.view(FP8).float() * a_s[:, None]
        groups = 1
    w_full = b_q.view(FP8).float() * b_s.repeat_interleave(gs, dim=0)
    ref = a_full @ w_full

    c = torch.zeros(m, n, dtype=torch.float16, device="cuda")
    torch.ops._rocm_C.gemm_w8a8_fp8_dense(
        a_q.cuda(), a_s.cuda(), b_q.cuda(), b_s.half().cuda(), c, gs, groups
    )
    torch.testing.assert_close(c.float().cpu(), ref, atol=5e-2, rtol=2e-2)
