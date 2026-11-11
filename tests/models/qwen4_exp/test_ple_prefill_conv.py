# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""short_conv_flat_prefill matches a per-request causal dilated conv."""

import pytest
import torch
import torch.nn.functional as F

from vllm.models.qwen4_exp.amd.ple_layer import short_conv_flat_prefill


def _reference(x_p, lengths, initial_state, weights, dilation, state_len):
    outs, states = [], []
    start = 0
    for i, n in enumerate(lengths):
        tokens = x_p[start : start + n].t()  # [hidden, n]
        start += n
        hist = torch.cat((initial_state[i], tokens), dim=-1) if state_len else tokens
        if n > 0:
            y = F.conv1d(
                hist.unsqueeze(0),
                weights.unsqueeze(1),
                groups=hist.size(0),
                dilation=dilation,
            )[0]
            outs.append(F.silu(y).t())
        if state_len:
            states.append(hist[:, n : n + state_len])
    out = torch.cat(outs, dim=0)
    return out, (torch.stack(states) if state_len else None)


@pytest.mark.parametrize("dilation,kernel", [(1, 4), (3, 4), (4, 2)])
@pytest.mark.parametrize("block", [5, 64, 2048])
@pytest.mark.parametrize("lengths", [[1], [7, 0, 13, 2], [40, 3, 1, 25, 9]])
def test_flat_prefill_matches_per_request_reference(dilation, kernel, block, lengths):
    torch.manual_seed(0)
    hidden = 6
    state_len = (kernel - 1) * dilation
    n = sum(lengths)
    x_p = torch.randn(n, hidden, dtype=torch.float64)
    weights = torch.randn(hidden, kernel, dtype=torch.float64)
    initial_state = torch.randn(len(lengths), hidden, state_len, dtype=torch.float64)
    q_starts = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0)
    lens = torch.tensor(lengths, dtype=torch.int64)

    out, next_state = short_conv_flat_prefill(
        x_p,
        q_starts,
        lens,
        initial_state,
        weights,
        dilation,
        state_len,
        len(lengths),
        n,
        block=block,
    )
    ref_out, ref_state = _reference(
        x_p, lengths, initial_state, weights, dilation, state_len
    )
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(next_state, ref_state)


def test_flat_prefill_without_state():
    torch.manual_seed(1)
    lengths = [5, 3]
    x_p = torch.randn(8, 4, dtype=torch.float64)
    weights = torch.randn(4, 1, dtype=torch.float64)
    q_starts = torch.tensor([0, 5, 8], dtype=torch.int64)
    out, next_state = short_conv_flat_prefill(
        x_p, q_starts, torch.tensor(lengths), None, weights, 1, 0, 2, 8
    )
    assert next_state is None
    torch.testing.assert_close(out, F.silu(x_p * weights[:, 0]))
