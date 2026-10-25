# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1030 post-conv KDA recurrence (``glm5_kda_recurrent_rdna2``).

The decode path hands the op ``[1, T, H, D]`` column slices of the merged
``q|k|v`` conv output and the ``[1, T, H]`` beta slice of the fused
projection, exactly what ``fused_recurrent_kda`` takes. The Python wrapper
squeezes those to the ``[T, H, D]`` / ``[T, H]`` the C++ entry checks, and
passes the token and head strides through.

Without the HIP build (any CPU box) a stand-in op with the C++ entry's shape
and dtype checks runs the fp32 reference, so the wrapper contract is tested
everywhere. With ``vllm._rocm_C`` on gfx1030 the same tests run the HIP
kernel, plus a HIP-vs-Triton comparison on the real decode layout.
"""

import pytest
import torch

from vllm.models.glm5next.amd.rdna import ops as rdna_ops
from vllm.models.glm5next.amd.rdna.ops import glm5_kda_recurrent
from vllm.platforms import current_platform

H, D = 4, 128
LOWER_BOUND = -5.0
QK_SCALE = D**-0.5

HIP_OP = rdna_ops.has_glm5_kda_recurrent()
_STAND_IN: torch.library.Library | None = None


def reference_kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    ssm_state: torch.Tensor,
    lower_bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 reference of ``fused_recurrent_kda``'s decode step per token.

    ``[T, H, D]`` activations, ``[T, H]`` raw beta, ``[slots, H, V, K]`` state.
    Token ``t`` reads and writes slot ``state_indices[t]``; slot 0
    (NULL_BLOCK_ID) and negative slots (PAD_SLOT_ID) leave the state alone
    and zero the output row, which is what the HIP kernel does.

    Returns:
        The output in ``q``'s dtype and the updated state (a copy).

    """
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * QK_SCALE
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    dt_bias = dt_bias.reshape(H, D)
    gate = lower_bound * torch.sigmoid(a_log.reshape(-1).exp()[:, None] * (g + dt_bias))
    beta = torch.sigmoid(beta)
    state = ssm_state.clone()
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for t in range(q.shape[0]):
        slot = int(state_indices[t])
        if slot <= 0 or slot >= state.shape[0]:
            continue
        s = state[slot] * gate[t].exp()[:, None, :]
        u = beta[t][:, None] * (v[t] - torch.einsum("hvk,hk->hv", s, k[t]))
        s = s + u[:, :, None] * k[t][:, None, :]
        out[t] = torch.einsum("hvk,hk->hv", s, q[t])
        state[slot] = s
    return out.to(ssm_state.dtype if False else q.dtype), state


def _check_op_contract(q, k, v, g, beta, a_log, dt_bias, state_indices, ssm_state, out):
    """The TORCH_CHECK calls of ``glm5_kda_recurrent_rdna2`` (after its squeeze)."""
    dt = q.dtype
    assert dt in (torch.float16, torch.bfloat16), "q must be fp16 or bf16"
    for name, t in (("k", k), ("v", v), ("g", g), ("beta", beta), ("out", out)):
        assert t.dtype == dt, f"{name} must match q dtype"
    assert q.dim() == 3 and q.shape[2] == D and q.stride(2) == 1, "q [B, H, 128]"
    for name, t in (("k", k), ("v", v), ("g", g), ("out", out)):
        assert t.shape == q.shape and t.stride(2) == 1, f"{name} shape mismatch"
    assert beta.dim() == 2 and beta.shape == q.shape[:2], "beta must be [B, H]"
    assert a_log.is_contiguous() and a_log.dtype == torch.float32
    assert a_log.numel() == q.shape[1], "A_log must be [H]"
    assert dt_bias.is_contiguous() and dt_bias.dtype == torch.float32
    assert dt_bias.numel() == q.shape[1] * D, "dt_bias must be [H*128]"
    assert state_indices.dim() == 1 and state_indices.shape[0] == q.shape[0]
    assert state_indices.dtype == torch.int32, "state_indices must be int32 [B]"
    assert ssm_state.dim() == 4 and ssm_state.dtype == torch.float32
    assert ssm_state.shape[1:] == (q.shape[1], D, D)
    assert ssm_state.stride(3) == 1 and ssm_state.stride(2) == D


def _stand_in_op(
    q, k, v, g, beta, a_log, dt_bias, state_indices, ssm_state, out, lower_bound
):
    _check_op_contract(q, k, v, g, beta, a_log, dt_bias, state_indices, ssm_state, out)
    ref_out, ref_state = reference_kda_recurrent(
        q, k, v, g, beta, a_log, dt_bias, state_indices, ssm_state, lower_bound
    )
    out.copy_(ref_out.to(out.dtype))
    ssm_state.copy_(ref_state)


def _install_stand_in() -> None:
    """Register a CPU ``_rocm_C::glm5_kda_recurrent_rdna2`` when HIP is absent."""
    global _STAND_IN
    if HIP_OP or _STAND_IN is not None:
        return
    try:
        lib = torch.library.Library("_rocm_C", "DEF")
    except RuntimeError:
        lib = torch.library.Library("_rocm_C", "FRAGMENT")
    lib.define(
        "glm5_kda_recurrent_rdna2(Tensor q, Tensor k, Tensor v, Tensor g, "
        "Tensor beta, Tensor A_log, Tensor dt_bias, Tensor state_indices, "
        "Tensor! ssm_state, Tensor! out, float lower_bound) -> ()"
    )
    lib.impl("glm5_kda_recurrent_rdna2", _stand_in_op, "CPU")
    _STAND_IN = lib


@pytest.fixture(autouse=True)
def _op_available():
    _install_stand_in()


def _device() -> torch.device:
    return torch.device("cuda" if HIP_OP else "cpu")


def make_decode_inputs(T: int, dtype: torch.dtype, device: torch.device, slots):
    """The decode layout: column slices, not contiguous tensors.

    q/k/v are ``[1, T, H, D]`` views into one ``[T, 3*H*D]`` conv output and
    beta is the ``[1, T, H]`` slice of a wider ``[T, 3*H*D + H + 2*D]``
    projection, as in ``Glm5NextLinearAttention.forward``.
    """
    proj = H * D
    qkv = torch.randn(T, 3 * proj, dtype=dtype, device=device)
    projected = torch.randn(T, 3 * proj + H + 2 * D, dtype=dtype, device=device)
    q, k, v = (x.reshape(1, T, H, D) for x in qkv.split(proj, dim=-1))
    beta = projected[:, 3 * proj : 3 * proj + H].unsqueeze(0)
    if T > 1:
        assert not q.is_contiguous() and not beta.is_contiguous()
    return dict(
        q=q,
        k=k,
        v=v,
        g=torch.randn(1, T, H, D, dtype=dtype, device=device),
        beta=beta,
        a_log=0.5 * torch.randn(H, dtype=torch.float32, device=device),
        dt_bias=0.1 * torch.randn(H * D, dtype=torch.float32, device=device),
        state_indices=torch.tensor(slots, dtype=torch.int32, device=device),
        ssm_state=torch.randn(
            max(slots) + 2, H, D, D, dtype=torch.float32, device=device
        ),
        out=torch.empty(1, T, H, D, dtype=dtype, device=device),
    )


def _run_reference(inp: dict) -> tuple[torch.Tensor, torch.Tensor]:
    return reference_kda_recurrent(
        inp["q"][0],
        inp["k"][0],
        inp["v"][0],
        inp["g"][0],
        inp["beta"][0],
        inp["a_log"],
        inp["dt_bias"],
        inp["state_indices"],
        inp["ssm_state"],
        LOWER_BOUND,
    )


def _run_op(inp: dict) -> None:
    glm5_kda_recurrent(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["g"],
        inp["beta"],
        inp["a_log"],
        inp["dt_bias"],
        inp["state_indices"],
        inp["ssm_state"],
        inp["out"],
        LOWER_BOUND,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_decode_views_match_reference(dtype: torch.dtype):
    """``[1, T, H, D]`` / ``[1, T, H]`` column slices go through unchanged."""
    torch.manual_seed(0)
    inp = make_decode_inputs(5, dtype, _device(), slots=[3, 1, 5, 2, 4])
    expected_out, expected_state = _run_reference(inp)
    _run_op(inp)
    torch.testing.assert_close(
        inp["out"][0].float(), expected_out.float(), rtol=1e-2, atol=2e-3
    )
    torch.testing.assert_close(inp["ssm_state"], expected_state, rtol=1e-4, atol=1e-4)


@torch.inference_mode()
def test_null_and_pad_slots_zero_output_and_keep_state():
    """Slot 0 (NULL_BLOCK_ID) and -1 (PAD_SLOT_ID) are padding rows."""
    torch.manual_seed(0)
    inp = make_decode_inputs(3, torch.float16, _device(), slots=[0, -1, 2])
    inp["out"].fill_(7.0)
    before = inp["ssm_state"].clone()
    _run_op(inp)
    assert torch.all(inp["out"][0, :2] == 0)
    torch.testing.assert_close(
        inp["ssm_state"][[0, 1, 3]], before[[0, 1, 3]], rtol=0, atol=0
    )
    assert not torch.equal(inp["ssm_state"][2], before[2])


@torch.inference_mode()
def test_squeezed_shapes_are_accepted_too():
    torch.manual_seed(0)
    inp = make_decode_inputs(2, torch.float16, _device(), slots=[1, 2])
    expected_out, expected_state = _run_reference(inp)
    glm5_kda_recurrent(
        inp["q"][0],
        inp["k"][0],
        inp["v"][0],
        inp["g"][0],
        inp["beta"][0],
        inp["a_log"],
        inp["dt_bias"],
        inp["state_indices"],
        inp["ssm_state"],
        inp["out"][0],
        LOWER_BOUND,
    )
    torch.testing.assert_close(
        inp["out"][0].float(), expected_out.float(), rtol=1e-2, atol=2e-3
    )
    torch.testing.assert_close(inp["ssm_state"], expected_state, rtol=1e-4, atol=1e-4)


def test_wrapper_rejects_wrong_rank_and_strided_last_dim():
    inp = make_decode_inputs(2, torch.float16, _device(), slots=[1, 2])
    bad_rank = dict(inp, q=inp["q"][0, :, :, 0])
    with pytest.raises(ValueError, match="q must have 3 dims"):
        _run_op(bad_rank)
    transposed = dict(inp, g=inp["g"][0].transpose(1, 2).unsqueeze(0))
    with pytest.raises(ValueError, match="contiguous on dim -1"):
        _run_op(transposed)


@pytest.mark.skipif(
    not (HIP_OP and current_platform.is_rocm()),
    reason="needs the gfx1030 HIP op",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_hip_matches_triton_fused_recurrent_kda(dtype: torch.dtype):
    """Same decode batch through the HIP op and ``fused_recurrent_kda``."""
    from vllm.models.glm5next.amd.ops.third_party.kda import fused_recurrent_kda

    torch.manual_seed(0)
    T = 7
    inp = make_decode_inputs(T, dtype, _device(), slots=list(range(1, T + 1)))
    triton_state = inp["ssm_state"].clone()
    triton_out, _ = fused_recurrent_kda(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        initial_state=triton_state,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.arange(0, T + 1, dtype=torch.int32, device=inp["q"].device),
        ssm_state_indices=inp["state_indices"],
        sigmoid_beta=True,
        a_log=inp["a_log"],
        g_bias=inp["dt_bias"],
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )
    _run_op(inp)
    torch.testing.assert_close(
        inp["out"].float(), triton_out.float(), rtol=1e-2, atol=2e-3
    )
    torch.testing.assert_close(inp["ssm_state"], triton_state, rtol=1e-4, atol=1e-4)
