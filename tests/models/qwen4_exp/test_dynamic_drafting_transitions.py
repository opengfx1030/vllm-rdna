# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transition fuzz for dynamic speculative drafting on the hybrid Qwen4Exp.

Dynamic SD changes the number of draft tokens K per step (K = 0 included), so
a verify's accepted count ``a`` (``num_accepted - 1`` = the state offset the
next step reads) must survive any K change, including a K>0 -> K=0 step and
batches whose rows have different widths. Each test drives the speculative
state path of one stateful op through random K / acceptance sequences and
checks every accepted output, and the final state, against a non-speculative
reference that only ever sees the accepted tokens (greedy decoding):

- GDN conv: ``causal_conv1d_update`` with the extended (width-1+K_max) window.
- GDN recurrent: ``fused_sigmoid_gating_delta_rule_update`` with per-token
  speculative state columns.
- PLE dilated short-conv: the AMD spec path vs its one-token decode path.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs a CUDA/HIP device"
)

K_MAX = 4
NUM_REQS = 5
NUM_STEPS = 16


def _schedule(gen: torch.Generator) -> list[tuple[list[int], list[int]]]:
    """Per step: (query widths, accepted counts) for every request.

    Most steps are uniform (all rows 1 + K_step, K_step in [0, K_MAX] with
    K=0 frequent), some get mixed tail widths. Acceptance is any prefix of the
    verified row, so every (K_prev accepted -> K_next) transition shows up.
    """
    steps = []
    for _ in range(NUM_STEPS):
        k_step = int(torch.randint(0, K_MAX + 1, (1,), generator=gen))
        if torch.rand(1, generator=gen) < 0.3:
            k_step = 0
        widths = [k_step + 1] * NUM_REQS
        if torch.rand(1, generator=gen) < 0.35:
            for r in range(NUM_REQS):
                if torch.rand(1, generator=gen) < 0.5:
                    widths[r] = int(torch.randint(1, K_MAX + 2, (1,), generator=gen))
        accepted = [int(torch.randint(1, w + 1, (1,), generator=gen)) for w in widths]
        steps.append((widths, accepted))
    # Force the transitions dynamic SD hits most: full accept, then K=0 twice.
    steps.append(([K_MAX + 1] * NUM_REQS, [K_MAX + 1] * NUM_REQS))
    steps.append(([1] * NUM_REQS, [1] * NUM_REQS))
    steps.append(([1] * NUM_REQS, [1] * NUM_REQS))
    steps.append(([K_MAX + 1] * NUM_REQS, [2] * NUM_REQS))
    return steps


def _cu(lengths: list[int], device) -> torch.Tensor:
    out = [0]
    for n in lengths:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32, device=device)


def _greedy_token_steps(widths: list[int], accepted: list[int], device):
    """Yield (live requests, their input rows) one accepted token at a time.

    The non-speculative reference decodes only accepted tokens, one per call
    (the plain decode path), never seeing a rejected draft.
    """
    starts = _cu(widths, "cpu").tolist()
    for t in range(max(accepted)):
        live = [r for r in range(len(widths)) if accepted[r] > t]
        rows = torch.tensor([starts[r] + t for r in live], device=device)
        yield torch.tensor(live, device=device), rows


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gdn_conv_state_survives_k_transitions(seed):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    device = torch.device(current_platform.device_type)
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    dim, width = 64, 4
    weight = torch.randn(dim, width, device=device)
    bias = torch.randn(dim, device=device)
    # Slot 0 is NULL_BLOCK_ID; each request owns slot r + 1.
    slots = torch.arange(1, NUM_REQS + 1, dtype=torch.int32, device=device)
    init = torch.randn(NUM_REQS, dim, width - 1, device=device)
    spec_state = torch.randn(NUM_REQS + 1, dim, width - 1 + K_MAX, device=device)
    spec_state[1:, :, : width - 1] = init
    ref_state = torch.zeros(NUM_REQS + 1, dim, width - 1, device=device)
    ref_state[1:] = init
    num_accepted = torch.ones(NUM_REQS, dtype=torch.int32, device=device)

    for widths, accepted in _schedule(gen):
        x = torch.randn(sum(widths), dim, device=device)
        spec_out = causal_conv1d_update(
            x.clone(),
            spec_state,
            weight,
            bias,
            "silu",
            conv_state_indices=slots,
            num_accepted_tokens=num_accepted,
            query_start_loc=_cu(widths, device),
            max_query_len=K_MAX + 1,
        )
        for live, rows in _greedy_token_steps(widths, accepted, device):
            ref_out = causal_conv1d_update(
                x[rows].clone(),
                ref_state,
                weight,
                bias,
                "silu",
                conv_state_indices=slots[live],
            )
            torch.testing.assert_close(spec_out[rows], ref_out, atol=1e-4, rtol=1e-4)
        num_accepted = torch.tensor(accepted, dtype=torch.int32, device=device)

    # The window the next step would read equals the reference state.
    for r in range(NUM_REQS):
        off = int(num_accepted[r]) - 1
        torch.testing.assert_close(
            spec_state[r + 1, :, off : off + width - 1], ref_state[r + 1]
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gdn_recurrent_state_survives_k_transitions(seed):
    from vllm.third_party.flash_linear_attention.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )

    device = torch.device(current_platform.device_type)
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    num_k_heads, num_v_heads, head_k, head_v = 2, 4, 64, 64
    A_log = torch.rand(num_v_heads, device=device)
    dt_bias = torch.rand(num_v_heads, device=device)
    cols = K_MAX + 1
    # Spec layout: request r owns columns [1 + r*cols, 1 + (r+1)*cols).
    spec_indices = (
        torch.arange(NUM_REQS * cols, dtype=torch.int32, device=device).view(
            NUM_REQS, cols
        )
        + 1
    )
    init = 0.1 * torch.randn(NUM_REQS, num_v_heads, head_k, head_v, device=device)
    spec_state = torch.randn(
        NUM_REQS * cols + 1, num_v_heads, head_k, head_v, device=device
    )
    spec_state[spec_indices[:, 0].long()] = init
    ref_slots = torch.arange(1, NUM_REQS + 1, dtype=torch.int32, device=device)
    ref_state = torch.zeros(NUM_REQS + 1, num_v_heads, head_k, head_v, device=device)
    ref_state[1:] = init
    num_accepted = torch.ones(NUM_REQS, dtype=torch.int32, device=device)

    def run(q, k, v, a, b, state, indices, cu, accepted_tokens):
        out, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=cu,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted_tokens,
            use_qk_l2norm_in_kernel=True,
        )
        return out

    for widths, accepted in _schedule(gen):
        n = sum(widths)
        q = torch.randn(1, n, num_k_heads, head_k, device=device)
        k = torch.randn(1, n, num_k_heads, head_k, device=device)
        v = torch.randn(1, n, num_v_heads, head_v, device=device)
        a = torch.randn(n, num_v_heads, device=device)
        b = torch.randn(n, num_v_heads, device=device)
        spec_out = run(
            q, k, v, a, b, spec_state, spec_indices, _cu(widths, device), num_accepted
        )
        for live, rows in _greedy_token_steps(widths, accepted, device):
            ref_out = run(
                q[:, rows].contiguous(),
                k[:, rows].contiguous(),
                v[:, rows].contiguous(),
                a[rows].contiguous(),
                b[rows].contiguous(),
                ref_state,
                ref_slots[live],
                _cu([1] * len(live), device),
                None,
            )
            torch.testing.assert_close(spec_out[:, rows], ref_out, atol=2e-3, rtol=2e-3)
        num_accepted = torch.tensor(accepted, dtype=torch.int32, device=device)

    for r in range(NUM_REQS):
        col = int(spec_indices[r, int(num_accepted[r]) - 1])
        torch.testing.assert_close(
            spec_state[col], ref_state[r + 1], atol=2e-3, rtol=2e-3
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ple_short_conv_state_survives_k_transitions(seed):
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLELayer

    device = torch.device(current_platform.device_type)
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    hidden, kernel, dilation = 64, 4, 3
    state_len = (kernel - 1) * dilation
    layer = SimpleNamespace(conv_state_len=state_len, short_conv_dilation=dilation)
    weights = torch.randn(hidden, kernel, device=device, dtype=torch.float16)
    slots = torch.arange(1, NUM_REQS + 1, dtype=torch.int32, device=device)
    init = torch.randn(NUM_REQS, hidden, state_len, device=device, dtype=torch.float16)
    spec_state = torch.randn(
        NUM_REQS + 1, hidden, state_len + K_MAX, device=device, dtype=torch.float16
    )
    spec_state[1:, :, :state_len] = init
    ref_state = torch.zeros(
        NUM_REQS + 1, hidden, state_len, device=device, dtype=torch.float16
    )
    ref_state[1:] = init
    num_accepted = torch.ones(NUM_REQS, dtype=torch.int32, device=device)
    has_init = torch.ones(NUM_REQS, dtype=torch.bool, device=device)

    for widths, accepted in _schedule(gen):
        x = torch.randn(sum(widths), hidden, device=device, dtype=torch.float16)
        spec_out = Qwen4ExpPLELayer._short_conv_dilated_spec_batched(
            layer,
            x,
            spec_state,
            weights,
            spec_state_indices_tensor=slots,
            spec_query_start_loc=_cu(widths, device),
            num_accepted_tokens=num_accepted,
            spec_query_len=K_MAX + 1,
        )
        for live, rows in _greedy_token_steps(widths, accepted, device):
            ref_out = Qwen4ExpPLELayer._short_conv_dilated_decode_batched(
                layer,
                x[rows],
                ref_state,
                weights,
                slots[live],
                has_init[live],
            )
            torch.testing.assert_close(spec_out[rows], ref_out, atol=2e-3, rtol=2e-3)
        num_accepted = torch.tensor(accepted, dtype=torch.int32, device=device)

    for r in range(NUM_REQS):
        off = int(num_accepted[r]) - 1
        torch.testing.assert_close(
            spec_state[r + 1, :, off : off + state_len], ref_state[r + 1]
        )
