# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stride tests for the gfx1030 HIP ``causal_conv1d_update_rdna2`` kernel.

The GDN layer passes ``kv_cache[0].transpose(-1, -2)`` as the conv state with
the default SD layout: storage is ``[lines, state_len, dim]`` and the view is
``[lines, dim, state_len]``, so time is not contiguous. The kernel used to
index the state as packed ``slot*dim*state_len + c*state_len + k`` and read
another channel (NaN tokens in eager mode and in oversize decode batches,
which bypass the contiguous cudagraph state arenas).

Run ``pytest tests/kernels/mamba/test_causal_conv1d_update_rdna2.py``.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 conv1d update kernel is ROCm-only", allow_module_level=True)

from vllm.platforms.rdna import on_rdna2  # noqa: E402

if not on_rdna2():
    pytest.skip("RDNA2 conv1d update kernel is gfx10x-only", allow_module_level=True)

import vllm._custom_ops  # noqa: E402, F401
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_update,
)

if not hasattr(torch.ops._rocm_C, "causal_conv1d_update_rdna2"):
    pytest.skip("causal_conv1d_update_rdna2 not built", allow_module_level=True)

DEVICE = "cuda"


def _reference(x, state, weight, bias, indices, silu):
    """fp32 reference. ``state`` is a [lines, dim, state_len] view (any
    strides); returns (out [batch, dim], new_state contiguous fp16)."""
    state = state.float().clone()
    w = weight.float()
    state_len = w.size(1) - 1
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for b, slot in enumerate(indices.tolist()):
        if slot < 0 or slot >= state.size(0):
            continue
        s = state[slot]
        acc = bias.float().clone() if bias is not None else torch.zeros_like(w[:, 0])
        acc = (
            acc
            + (w[:, :state_len] * s[:, :state_len]).sum(-1)
            + w[:, state_len] * x[b].float()
        )
        if silu:
            acc = acc * torch.sigmoid(acc)
        out[b] = acc
        # Only the first width - 1 columns shift; any spec-sized capacity past
        # them is left untouched, exactly like the HIP and Triton kernels.
        s[:, :state_len] = torch.cat(
            [s[:, 1:state_len], x[b].float().unsqueeze(-1)], dim=-1
        )
    return out.half(), state.half()


def _make(batch, dim, width, lines, seed, transposed, state_capacity=None):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    # MTP sizes the conv state as width - 1 + num_spec; default to width - 1.
    state_len = width - 1 if state_capacity is None else state_capacity
    if transposed:
        # Paged SD layout: storage [lines, state_len, dim], view transposed.
        storage = torch.randn(
            lines, state_len, dim, device=DEVICE, dtype=torch.float16, generator=g
        )
        state = storage.transpose(-1, -2)
        assert not state.is_contiguous()
    else:
        state = torch.randn(
            lines, dim, state_len, device=DEVICE, dtype=torch.float16, generator=g
        )
    x = torch.randn(batch, dim, device=DEVICE, dtype=torch.float16, generator=g)
    weight = (
        torch.randn(dim, width, device=DEVICE, dtype=torch.float16, generator=g) * 0.5
    )
    bias = torch.randn(dim, device=DEVICE, dtype=torch.float16, generator=g)
    # Skip line 0: it is the null block the Triton path ignores.
    perm = torch.randperm(lines - 1, device="cpu")[:batch] + 1
    indices = perm.to(device=DEVICE, dtype=torch.int32)
    return x, state, weight, bias, indices


def _run_hip(x, state, weight, bias, indices, silu, monkeypatch):
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "1")
    return causal_conv1d_update(
        x.clone(),
        state,
        weight,
        bias,
        "silu" if silu else None,
        conv_state_indices=indices,
    )


@pytest.mark.parametrize("batch", [1, 7, 64])
@pytest.mark.parametrize("dim", [256, 2048])
@pytest.mark.parametrize("silu", [True, False])
@pytest.mark.parametrize("transposed", [False, True])
def test_update_matches_reference(batch, dim, silu, transposed, monkeypatch):
    width, lines = 4, 96
    x, state, weight, bias, indices = _make(batch, dim, width, lines, 0, transposed)
    ref_out, ref_state = _reference(x, state, weight, bias, indices, silu)
    out = _run_hip(x, state, weight, bias, indices, silu, monkeypatch)
    torch.accelerator.synchronize()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref_out, atol=2e-3, rtol=2e-3)
    # The state shift is a pure fp16 copy: exact.
    torch.testing.assert_close(state, ref_state, atol=0, rtol=0)


@pytest.mark.parametrize("silu", [True, False])
def test_transposed_matches_contiguous_bitwise(silu, monkeypatch):
    """Same values in both layouts must give bit-identical outputs/states:
    the arithmetic is layout-independent."""
    batch, dim, width, lines = 32, 2048, 4, 64
    x, state_c, weight, bias, indices = _make(batch, dim, width, lines, 1, False)
    state_t = state_c.contiguous().transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not state_t.is_contiguous()
    assert torch.equal(state_t, state_c)
    out_c = _run_hip(x, state_c, weight, bias, indices, silu, monkeypatch)
    out_t = _run_hip(x, state_t, weight, bias, indices, silu, monkeypatch)
    torch.accelerator.synchronize()
    assert torch.equal(out_c, out_t)
    assert torch.equal(state_c, state_t)


def test_matches_triton_path_transposed(monkeypatch):
    batch, dim, width, lines = 16, 1024, 4, 32
    x, state, weight, bias, indices = _make(batch, dim, width, lines, 2, True)
    state_tri = state.clone()  # clone keeps the transposed strides
    out_hip = _run_hip(x, state, weight, bias, indices, True, monkeypatch)
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "0")
    out_tri = causal_conv1d_update(
        x.clone(), state_tri, weight, bias, "silu", conv_state_indices=indices
    )
    torch.accelerator.synchronize()
    # Bit-for-bit: Triton rounds each w * x product to fp16 before the fp32
    # accumulate, and the HIP kernel reproduces that exactly.
    assert torch.equal(out_hip, out_tri)
    assert torch.equal(state, state_tri)


def test_strided_x_inplace_and_padded_slot(monkeypatch):
    """X is a column slice of a wider buffer (non-contiguous rows); the
    default out=x is written in place; slot -1 leaves its row and state
    untouched."""
    batch, dim, width, lines = 8, 512, 4, 16
    _, state, weight, bias, indices = _make(batch, dim, width, lines, 3, True)
    indices[3] = -1
    wide = torch.randn(batch, 3 * dim, device=DEVICE, dtype=torch.float16)
    x = wide[:, dim : 2 * dim]
    assert not x.is_contiguous()
    x_before = x.clone()
    ref_out, ref_state = _reference(x_before, state, weight, bias, indices, True)
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "1")
    out = causal_conv1d_update(
        x, state, weight, bias, "silu", conv_state_indices=indices
    )
    torch.accelerator.synchronize()
    assert out.data_ptr() == x.data_ptr()
    keep = torch.ones(batch, dtype=torch.bool)
    keep[3] = False
    torch.testing.assert_close(out[keep], ref_out[keep], atol=2e-3, rtol=2e-3)
    assert torch.equal(out[3], x_before[3])
    torch.testing.assert_close(state, ref_state, atol=0, rtol=0)


@pytest.mark.parametrize("capacity", [5, 7])
def test_update_kernel_ignores_spec_state_capacity(capacity):
    """MTP sizes the conv state as width - 1 + num_spec. The HIP kernel must
    consume only the first width - 1 columns (like the Triton kernel's
    effective state_len) and leave the extra capacity untouched.

    The op is called directly: the dispatch guard requires
    ``conv_state.size(2) == width - 1``, so an oversized cache never reaches it
    from the dispatcher.
    """
    batch, dim, width, lines = 16, 1024, 4, 32
    assert capacity > width - 1
    x, state, weight, bias, indices = _make(
        batch, dim, width, lines, 4, True, state_capacity=capacity
    )
    assert state.size(2) == capacity
    ref_out, ref_state = _reference(x, state, weight, bias, indices, True)
    x3 = x.unsqueeze(-1)
    out = torch.empty_like(x3)
    torch.ops._rocm_C.causal_conv1d_update_rdna2(
        x3, state, weight, bias, out, indices, True
    )
    torch.accelerator.synchronize()
    torch.testing.assert_close(out.squeeze(-1), ref_out, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(state, ref_state, atol=0, rtol=0)


def test_hip_conv_state_gate_is_opt_in(monkeypatch):
    """An over-capacity (MTP-sized) conv state only takes HIP when opted in.

    MTP sizes the conv state as width - 1 + num_spec; the HIP kernels ignore the
    extra columns, but that is only validated for this model, so the dispatcher
    must keep it behind VLLM_CAUSAL_CONV1D_RDNA2_SPEC_STATE (default off).
    """
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        _hip_conv_state_ok,
    )

    width = 4
    exact = torch.zeros(4, 8, width - 1)
    over = torch.zeros(4, 8, width - 1 + 2)
    under = torch.zeros(4, 8, width - 2)
    monkeypatch.delenv("VLLM_CAUSAL_CONV1D_RDNA2_SPEC_STATE", raising=False)
    assert _hip_conv_state_ok(exact, width)
    assert not _hip_conv_state_ok(over, width)
    assert not _hip_conv_state_ok(under, width)
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_SPEC_STATE", "1")
    assert _hip_conv_state_ok(exact, width)
    assert _hip_conv_state_ok(over, width)
    assert not _hip_conv_state_ok(under, width)


def test_update_dispatch_admits_over_capacity_state_only_when_opted_in(monkeypatch):
    """The opt-in is what lets the HIP update kernel see an MTP-sized state."""
    batch, dim, width, lines = 4, 256, 4, 32
    x, state, weight, bias, indices = _make(
        batch, dim, width, lines, 7, True, state_capacity=width - 1 + 2
    )
    assert state.size(2) > width - 1
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "1")
    calls: list[int] = []
    monkeypatch.setattr(
        torch.ops._rocm_C,
        "causal_conv1d_update_rdna2",
        lambda *a, **k: calls.append(1),
    )

    def run():
        return causal_conv1d_update(
            x.clone(), state, weight, bias, "silu", conv_state_indices=indices
        )

    monkeypatch.delenv("VLLM_CAUSAL_CONV1D_RDNA2_SPEC_STATE", raising=False)
    run()
    assert calls == []
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_SPEC_STATE", "1")
    run()
    assert calls == [1]


def test_fwd_kernel_ignores_spec_state_capacity():
    """The varlen HIP conv op must ignore conv-state capacity past width - 1.

    MTP sizes the conv state as width - 1 + num_spec. The op is called directly
    because the dispatch guard requires ``conv_states.size(2) == width - 1``, so
    an oversized cache never reaches it from the dispatcher. ``x`` is
    channel-last, as the serving path passes it.
    """
    batch, dim, width, seqlen, capacity = 3, 256, 4, 5, 6
    lines = 8
    state_len = width - 1
    g = torch.Generator(device=DEVICE).manual_seed(11)
    total = batch * seqlen
    x = torch.randn(total, dim, device=DEVICE, dtype=torch.float16, generator=g)
    x = x.transpose(0, 1)
    weight = torch.randn(dim, width, device=DEVICE, dtype=torch.float16, generator=g)
    bias = torch.randn(dim, device=DEVICE, dtype=torch.float16, generator=g)
    # Both caches hold the same history; the oversized one carries extra
    # capacity that must be neither read nor written.
    history = torch.randn(
        lines, state_len, dim, device=DEVICE, dtype=torch.float16, generator=g
    )
    storage = torch.randn(
        lines, capacity, dim, device=DEVICE, dtype=torch.float16, generator=g
    )
    storage[:, :state_len] = history
    query_start_loc = torch.arange(
        0, total + 1, seqlen, dtype=torch.int32, device=DEVICE
    )
    cache_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    has_initial_state = torch.ones(batch, dtype=torch.bool, device=DEVICE)

    def run(state):
        out = torch.empty_like(x)
        torch.ops._rocm_C.causal_conv1d_fwd_rdna2(
            x,
            weight,
            bias,
            state,
            query_start_loc,
            cache_indices,
            has_initial_state,
            out,
            True,
            0,
        )
        return out

    state_cap = storage.transpose(-1, -2)
    state_min = history.clone().transpose(-1, -2)
    spare = state_cap[..., state_len:].clone()
    out_cap = run(state_cap)
    out_min = run(state_min)
    torch.accelerator.synchronize()
    assert torch.equal(out_cap, out_min)
    assert torch.equal(state_cap[..., :state_len], state_min)
    assert torch.equal(state_cap[..., state_len:], spare)


# =============================================================================
# Varlen / speculative-decoding update (causal_conv1d_update_spec_rdna2).
#
# MTP-2 passes num_accepted_tokens and query_start_loc on every spec-verify
# step, which used to skip the single-token HIP dispatch entirely and fall
# through to the Triton kernel. These tests pin the HIP kernel bit-for-bit
# against that Triton kernel on the deployed geometry (8 reqs x 3 tokens,
# conv_dim 2560/TP4, conv_state capacity width - 1 + num_spec = 5).
# =============================================================================

SPEC_WIDTH = 4
SPEC_NUM_SPEC = 2
SPEC_CAPACITY = SPEC_WIDTH - 1 + SPEC_NUM_SPEC  # 5, from num_spec=2
SERVING_DIM = 2560  # (2 * 128 * 16 + 128 * 48) / 4


def _make_spec(n_req, dim, lens, accepted, transposed=True, seed=0, capacity=None):
    """Build one varlen spec-verify step. ``conv_state_indices`` is a
    ``[:, 0]`` slice of the [n_req, 1 + num_spec] state-index tensor, like the
    GDN layer passes it, so ``stride_state_indices`` is exercised too."""
    capacity = SPEC_CAPACITY if capacity is None else capacity
    lines = 1 + n_req * (1 + SPEC_NUM_SPEC)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(sum(lens), dim, device=DEVICE, dtype=torch.float16, generator=g)
    weight = (
        torch.randn(dim, SPEC_WIDTH, device=DEVICE, dtype=torch.float16, generator=g)
        * 0.5
    )
    bias = torch.randn(dim, device=DEVICE, dtype=torch.float16, generator=g)
    if transposed:  # SD layout: storage [lines, capacity, dim], view transposed
        storage = torch.randn(
            lines, capacity, dim, device=DEVICE, dtype=torch.float16, generator=g
        )
        state = storage.transpose(-1, -2)
        assert not state.is_contiguous()
    else:  # DS layout: [lines, dim, capacity]
        state = torch.randn(
            lines, dim, capacity, device=DEVICE, dtype=torch.float16, generator=g
        )
    indices = (
        torch.arange(
            n_req * (1 + SPEC_NUM_SPEC), device=DEVICE, dtype=torch.int32
        ).reshape(n_req, 1 + SPEC_NUM_SPEC)
        + 1
    )
    query_start_loc = torch.zeros(n_req + 1, device=DEVICE, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(lens, device=DEVICE, dtype=torch.int32)
    query_start_loc = query_start_loc.cumsum(0).to(torch.int32)
    return x, state, weight, bias, indices, query_start_loc, accepted


def _run_update(
    x,
    state,
    weight,
    bias,
    indices,
    monkeypatch,
    hip,
    accepted=None,
    query_start_loc=None,
    max_query_len=1 + SPEC_NUM_SPEC,
    silu=True,
):
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "1" if hip else "0")
    return causal_conv1d_update(
        x.clone(),
        state,
        weight,
        bias,
        "silu" if silu else None,
        conv_state_indices=indices,
        num_accepted_tokens=accepted,
        query_start_loc=query_start_loc,
        max_query_len=max_query_len,
    )


def _assert_spec_bitwise(n_req, dim, lens, accepted, monkeypatch, transposed=True):
    x, state, weight, bias, indices, qsl, _ = _make_spec(
        n_req, dim, lens, accepted, transposed=transposed
    )
    accepted = torch.tensor(accepted, device=DEVICE, dtype=torch.int32)
    state_hip = state.clone()
    out_hip = _run_update(
        x, state_hip, weight, bias, indices[:, 0], monkeypatch, True, accepted, qsl
    )
    state_tri = state.clone()
    out_tri = _run_update(
        x, state_tri, weight, bias, indices[:, 0], monkeypatch, False, accepted, qsl
    )
    torch.accelerator.synchronize()
    assert torch.equal(out_hip, out_tri)
    assert torch.equal(state_hip, state_tri)


@pytest.mark.parametrize(
    "n_req,dim", [(1, SERVING_DIM), (2, SERVING_DIM), (8, 256), (1, 32)]
)
def test_spec_update_matches_triton_bitwise(n_req, dim, monkeypatch):
    """Deployed MTP-2 shape: n_req requests x (1 + num_spec) tokens, accepted."""
    spec_len = 1 + SPEC_NUM_SPEC
    _assert_spec_bitwise(
        n_req, dim, [spec_len] * n_req, [spec_len] * n_req, monkeypatch
    )


@pytest.mark.parametrize("accepted", [[1], [2], [3]])
def test_spec_update_partial_acceptance_bitwise(accepted, monkeypatch):
    """num_accepted_tokens - 1 shifts the history window; only the carried
    prefix of the new state changes, so a wrong offset still looks plausible."""
    _assert_spec_bitwise(1, 512, [3], accepted, monkeypatch)
    _assert_spec_bitwise(4, 512, [3] * 4, accepted * 4, monkeypatch)


@pytest.mark.parametrize("lens", [[3, 1, 2, 3], [1, 1, 1], [2, 3, 1], [1, 3]])
def test_spec_update_ragged_bitwise(lens, monkeypatch):
    """Ragged spec batches revise state_len per request
    (state_len -= max_query_len - seqlen), shortening both the write-back and
    the `idx_tokens + seqlen < state_len` carry mask."""
    accepted = [min(ln, 3) for ln in lens]
    _assert_spec_bitwise(len(lens), 256, lens, accepted, monkeypatch)


def test_spec_update_ds_layout_bitwise(monkeypatch):
    _assert_spec_bitwise(
        4, 256, [3, 3, 3, 3], [3, 3, 3, 3], monkeypatch, transposed=False
    )


def test_spec_update_no_silu_bitwise(monkeypatch):
    x, state, weight, bias, indices, qsl, _ = _make_spec(4, 256, [3] * 4, [3] * 4)
    accepted = torch.full((4,), 3, device=DEVICE, dtype=torch.int32)
    state_hip = state.clone()
    out_hip = _run_update(
        x,
        state_hip,
        weight,
        bias,
        indices[:, 0],
        monkeypatch,
        True,
        accepted,
        qsl,
        silu=False,
    )
    state_tri = state.clone()
    out_tri = _run_update(
        x,
        state_tri,
        weight,
        bias,
        indices[:, 0],
        monkeypatch,
        False,
        accepted,
        qsl,
        silu=False,
    )
    torch.accelerator.synchronize()
    assert torch.equal(out_hip, out_tri)
    assert torch.equal(state_hip, state_tri)


def test_varlen_update_without_spec_matches_triton(monkeypatch):
    """IS_VARLEN without IS_SPEC_DECODING: no num_accepted_tokens, offset 0,
    state_len revised down to width - 1 - (max_query_len - seqlen)."""
    x, state, weight, bias, indices, qsl, _ = _make_spec(3, 512, [3, 2, 1], [0, 0, 0])
    state_hip = state.clone()
    out_hip = _run_update(
        x, state_hip, weight, bias, indices[:, 0], monkeypatch, True, None, qsl
    )
    state_tri = state.clone()
    out_tri = _run_update(
        x, state_tri, weight, bias, indices[:, 0], monkeypatch, False, None, qsl
    )
    torch.accelerator.synchronize()
    assert torch.equal(out_hip, out_tri)
    assert torch.equal(state_hip, state_tri)


def test_spec_update_skips_null_block(monkeypatch):
    """Slot == NULL_BLOCK_ID: the Triton kernel returns before writing `out`,
    leaving the caller's buffer untouched and the state intact."""
    x, state, weight, bias, indices, qsl, _ = _make_spec(3, 256, [3, 3, 3], [3, 3, 3])
    bad_indices = indices[:, 0].clone()
    bad_indices[1] = 0  # NULL_BLOCK_ID
    accepted = torch.full((3,), 3, device=DEVICE, dtype=torch.int32)
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "1")
    out = _run_update(
        x, state, weight, bias, bad_indices, monkeypatch, True, accepted, qsl
    )
    torch.accelerator.synchronize()
    # rows 3..5 of `out` are the untouched input rows (out is x in place).
    assert torch.equal(out[3:6], x[3:6])


class _NoLaunch:
    def __getitem__(self, _key):
        raise AssertionError("Triton kernel launched on the RDNA2 path")


def test_update_does_not_launch_triton(monkeypatch):
    """Both HIP update ops must be reached with the Triton kernel poisoned:
    single-token decode and the MTP-2 spec-verify shape."""
    import vllm.model_executor.layers.mamba.ops.causal_conv1d as cc

    monkeypatch.setattr(cc, "_causal_conv1d_update_kernel", _NoLaunch())
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "1")

    x, state, weight, bias, indices = _make(4, 512, 4, 16, 5, True)
    out = causal_conv1d_update(
        x.clone(), state, weight, bias, "silu", conv_state_indices=indices
    )
    torch.accelerator.synchronize()
    assert torch.isfinite(out).all()

    sx, sstate, sw, sb, sidx, sqsl, _ = _make_spec(4, 512, [3] * 4, [3] * 4)
    accepted = torch.full((4,), 3, device=DEVICE, dtype=torch.int32)
    sout = causal_conv1d_update(
        sx.clone(),
        sstate,
        sw,
        sb,
        "silu",
        conv_state_indices=sidx[:, 0],
        num_accepted_tokens=accepted,
        query_start_loc=sqsl,
        max_query_len=3,
    )
    torch.accelerator.synchronize()
    assert torch.isfinite(sout).all()
