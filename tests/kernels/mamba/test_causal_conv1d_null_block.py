# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output rows of a null-block (padding) sequence must be defined.

A sequence whose state index is ``NULL_BLOCK_ID`` neither reads nor writes the
conv state. The conv1d kernels used to return before storing its output rows,
so ``causal_conv1d_fn`` (``out = torch.empty_like(x)``) and
``causal_conv1d_update`` with a caller ``out`` handed back whatever the caching
allocator left in them. Those rows must now be zero, and every other row and
the conv state must be unchanged by the fix.

The ``causal_conv1d_fn`` tests leave a NaN-filled block in the caching
allocator for its ``torch.empty_like`` to reuse, and check that it is reused.

Run ``pytest tests/kernels/mamba/test_causal_conv1d_null_block.py``.
"""

import types

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
)

if not current_platform.is_cuda_alike():
    pytest.skip("causal_conv1d GPU kernels", allow_module_level=True)

DEVICE = "cuda"
DIM = 2560  # GDN conv_dim per rank on the TP4 serving config
WIDTH = 4
STATE_LEN = WIDTH - 1
LINES = 8
# Prefill batch: 256 tokens, two sequences on the null block. 2560 * 256 fp16
# is 1.25 MiB, so `out` comes from the large pool and the wrapper's small
# allocations cannot take the poisoned block.
LENS = [37, 64, 19, 100, 36]
SLOTS = [3, NULL_BLOCK_ID, 5, NULL_BLOCK_ID, 7]
HAS_INIT = [True, True, False, False, True]


def _hip_op_available(name: str) -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rdna import on_rdna2

    if not on_rdna2():
        return False
    import vllm._custom_ops  # noqa: F401

    return hasattr(torch.ops._rocm_C, name)


def _bounds(lens):
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + n)
    return starts


def _poison_like(x: torch.Tensor) -> None:
    """Free a NaN-filled block shaped like ``x`` so the next
    ``torch.empty_like(x)`` gets it back, and check that it does."""
    torch.accelerator.synchronize()
    poison = torch.full_like(x, float("nan"))
    del poison
    probe = torch.empty_like(x)
    assert probe.isnan().all(), "caching allocator did not reuse the block"
    del probe


def _fwd_inputs(seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    total = sum(LENS)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=DEVICE, dtype=torch.float16)

    x = randn(total, DIM).t()  # [dim, cu_seqlen], channel-last
    weight = randn(DIM, WIDTH) * 0.5
    bias = randn(DIM)
    # SD layout: storage [lines, state_len, dim], passed transposed.
    conv_states = randn(LINES, STATE_LEN, DIM).transpose(-1, -2)
    qsl = torch.tensor(_bounds(LENS), dtype=torch.int32, device=DEVICE)
    cache_indices = torch.tensor(SLOTS, dtype=torch.int32, device=DEVICE)
    has_init = torch.tensor(HAS_INIT, dtype=torch.bool, device=DEVICE)
    return x, weight, bias, conv_states, qsl, cache_indices, has_init


def _ref_seq(x_seq, init, weight, bias):
    """fp32 reference for one sequence: x_seq [dim, L], init [dim, state_len]
    or None. Returns (silu(conv) [dim, L], final state [dim, state_len])."""
    hist = (
        init.float()
        if init is not None
        else torch.zeros(DIM, STATE_LEN, device=DEVICE, dtype=torch.float32)
    )
    full = torch.cat([hist, x_seq.float()], dim=-1)
    out = F.conv1d(
        full.unsqueeze(0), weight.float().unsqueeze(1), bias.float(), groups=DIM
    )[0]
    return F.silu(out).half(), full[:, -STATE_LEN:].half()


@pytest.mark.parametrize("path", ["triton_metadata", "triton", "hip"])
def test_fwd_null_block_rows_are_zero(path, monkeypatch):
    """``causal_conv1d_fn``: the GDN layer passes ``metadata`` (Triton,
    block-aligned scheduling); ``metadata=None`` callers take the HIP kernel
    on gfx1030 unless VLLM_CAUSAL_CONV1D_RDNA2_FWD=0."""
    if path == "hip" and not _hip_op_available("causal_conv1d_fwd_rdna2"):
        pytest.skip("causal_conv1d_fwd_rdna2 needs a gfx1030 ROCm build")
    import vllm.model_executor.layers.mamba.ops.causal_conv1d as cc

    if path == "hip":

        class _NoLaunch:
            def __getitem__(self, _key):
                raise AssertionError("Triton kernel launched on the HIP path")

        monkeypatch.setattr(cc, "_causal_conv1d_fwd_kernel", _NoLaunch())
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_FWD", "0" if path == "triton" else "1")

    x, weight, bias, conv_states, qsl, cache_indices, has_init = _fwd_inputs()
    states_before = conv_states.clone()
    metadata = None
    if path == "triton_metadata":
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            qsl.cpu(), device=x.device
        )
        metadata = types.SimpleNamespace(
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )

    _poison_like(x)
    out = causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        qsl,
        cache_indices=cache_indices,
        has_initial_state=has_init,
        activation="silu",
        metadata=metadata,
    )
    torch.accelerator.synchronize()

    starts = _bounds(LENS)
    for i, slot in enumerate(SLOTS):
        rows = out[:, starts[i] : starts[i + 1]]
        if slot == NULL_BLOCK_ID:
            assert torch.equal(rows, torch.zeros_like(rows)), (
                f"seq {i} (null block): {rows.isnan().sum().item()} NaN of "
                f"{rows.numel()}"
            )
            continue
        init = states_before[slot] if HAS_INIT[i] else None
        ref_out, ref_state = _ref_seq(
            x[:, starts[i] : starts[i + 1]], init, weight, bias
        )
        torch.testing.assert_close(rows, ref_out, atol=2e-2, rtol=1e-2)
        assert torch.equal(conv_states[slot], ref_state)

    untouched = [n for n in range(LINES) if n not in SLOTS or n == NULL_BLOCK_ID]
    assert torch.equal(conv_states[untouched], states_before[untouched])


# -----------------------------------------------------------------------------
# causal_conv1d_update with a caller `out` that does not alias `x`.
# -----------------------------------------------------------------------------

SPEC_LENS = [3, 3, 3, 3]  # MTP-2 verify: 1 + num_spec tokens per request
SPEC_SLOTS = [2, NULL_BLOCK_ID, 5, NULL_BLOCK_ID]
SPEC_ACCEPTED = [3, 1, 2, 3]
SPEC_CAPACITY = STATE_LEN + 2  # width - 1 + num_spec


def _update_both(x, state, call):
    """Run ``call`` into a NaN-filled ``out`` and in place (``out=None``) on
    cloned inputs. Returns (out, state after out-run, in-place out, state
    after in-place run)."""
    state_out = state.clone()
    out = torch.full_like(x, float("nan"))
    ret = call(x.clone(), state_out, out)
    assert ret.data_ptr() == out.data_ptr()
    state_inplace = state.clone()
    ref = call(x.clone(), state_inplace, None)
    torch.accelerator.synchronize()
    return out, state_out, ref, state_inplace


@pytest.mark.parametrize("path", ["triton", "hip"])
def test_spec_update_null_block_rows_are_zero(path, monkeypatch):
    """Varlen spec-verify update (``num_accepted_tokens`` + ``query_start_loc``),
    the MTP-2 decode shape. HIP: ``causal_conv1d_update_spec_rdna2``."""
    if path == "hip" and not _hip_op_available("causal_conv1d_update_spec_rdna2"):
        pytest.skip("causal_conv1d_update_spec_rdna2 needs a gfx1030 ROCm build")
    import vllm.model_executor.layers.mamba.ops.causal_conv1d as cc

    if path == "hip":

        class _NoLaunch:
            def __getitem__(self, _key):
                raise AssertionError("Triton kernel launched on the HIP path")

        monkeypatch.setattr(cc, "_causal_conv1d_update_kernel", _NoLaunch())
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "1" if path == "hip" else "0")

    g = torch.Generator(device=DEVICE).manual_seed(1)
    x = torch.randn(
        sum(SPEC_LENS), DIM, generator=g, device=DEVICE, dtype=torch.float16
    )
    weight = (
        torch.randn(DIM, WIDTH, generator=g, device=DEVICE, dtype=torch.float16) * 0.5
    )
    bias = torch.randn(DIM, generator=g, device=DEVICE, dtype=torch.float16)
    state = torch.randn(
        LINES, SPEC_CAPACITY, DIM, generator=g, device=DEVICE, dtype=torch.float16
    ).transpose(-1, -2)
    indices = torch.tensor(SPEC_SLOTS, dtype=torch.int32, device=DEVICE)
    accepted = torch.tensor(SPEC_ACCEPTED, dtype=torch.int32, device=DEVICE)
    qsl = torch.tensor(_bounds(SPEC_LENS), dtype=torch.int32, device=DEVICE)

    def call(x_, state_, out_):
        return causal_conv1d_update(
            x_,
            state_,
            weight,
            bias,
            "silu",
            conv_state_indices=indices,
            num_accepted_tokens=accepted,
            query_start_loc=qsl,
            max_query_len=max(SPEC_LENS),
            out=out_,
        )

    out, state_out, ref, state_inplace = _update_both(x, state, call)
    starts = _bounds(SPEC_LENS)
    for i, slot in enumerate(SPEC_SLOTS):
        rows = out[starts[i] : starts[i + 1]]
        if slot == NULL_BLOCK_ID:
            assert torch.equal(rows, torch.zeros_like(rows)), (
                f"request {i} (null block): {rows.isnan().sum().item()} NaN"
            )
        else:
            assert torch.equal(rows, ref[starts[i] : starts[i + 1]])
    assert torch.equal(state_out, state_inplace)
    assert torch.equal(state_out[NULL_BLOCK_ID], state[NULL_BLOCK_ID])


def test_decode_update_null_block_rows_are_zero(monkeypatch):
    """Single-token batch-gather decode on the Triton kernel. (The gfx1030
    HIP single-token kernel treats the null block as a scratch line and
    writes every row, so it is not exercised here.)"""
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "0")
    slots = [2, NULL_BLOCK_ID, 5, NULL_BLOCK_ID, 1]
    g = torch.Generator(device=DEVICE).manual_seed(2)
    x = torch.randn(len(slots), DIM, generator=g, device=DEVICE, dtype=torch.float16)
    weight = (
        torch.randn(DIM, WIDTH, generator=g, device=DEVICE, dtype=torch.float16) * 0.5
    )
    bias = torch.randn(DIM, generator=g, device=DEVICE, dtype=torch.float16)
    state = torch.randn(
        LINES, STATE_LEN, DIM, generator=g, device=DEVICE, dtype=torch.float16
    ).transpose(-1, -2)
    indices = torch.tensor(slots, dtype=torch.int32, device=DEVICE)

    def call(x_, state_, out_):
        return causal_conv1d_update(
            x_, state_, weight, bias, "silu", conv_state_indices=indices, out=out_
        )

    out, state_out, ref, state_inplace = _update_both(x, state, call)
    for i, slot in enumerate(slots):
        if slot == NULL_BLOCK_ID:
            assert torch.equal(out[i], torch.zeros_like(out[i])), (
                f"row {i} (null block): {out[i].isnan().sum().item()} NaN"
            )
        else:
            assert torch.equal(out[i], ref[i])
    assert torch.equal(state_out, state_inplace)
    assert torch.equal(state_out[NULL_BLOCK_ID], state[NULL_BLOCK_ID])
