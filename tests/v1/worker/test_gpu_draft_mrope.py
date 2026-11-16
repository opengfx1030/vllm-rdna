# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms.rdna import on_rdna2_hip
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.mrope import MropeInputBuffers


@pytest.mark.parametrize("num_dims", [3, 4])
def test_prefill_axes_and_reordered_request_decode_positions(num_dims):
    buffers = MropeInputBuffers(num_dims, 4, 12, torch.device("cpu"))
    target = torch.arange(num_dims * 7).view(num_dims, 7)
    expected_prefill = target.clone()
    logical_prefill = torch.tensor([99, 100, 101, 299, 300, 301, 302])
    last_indices = torch.tensor([2, 6])
    buffers.prepare_prefill(target, last_indices, logical_prefill)
    target.fill_(-1)
    torch.testing.assert_close(buffers.get_prefill_positions(7), expected_prefill)

    # Logical positions include accepted-token rollback; preserve every axis.
    logical = torch.tensor([102, 303, 0, 0])
    original_logical = logical.clone()
    expected = torch.cat(
        (expected_prefill[:, last_indices] + 1,
         torch.zeros(num_dims, 2, dtype=torch.int64)), dim=1
    )
    torch.testing.assert_close(buffers.get_decode_positions(logical), expected)
    torch.testing.assert_close(logical, original_logical)
    logical[:2].add_(1)
    expected[:, :2].add_(1)
    torch.testing.assert_close(buffers.get_decode_positions(logical), expected)


def test_recycled_slots_and_smaller_padded_batch_drop_old_deltas():
    buffers = MropeInputBuffers(3, 4, 12, torch.device("cpu"))
    target = torch.tensor([5, 17, 26, 38]).expand(3, -1)
    buffers.prepare_prefill(target, torch.arange(4), torch.tensor([10, 20, 30, 40]))
    torch.testing.assert_close(
        buffers.get_decode_positions(torch.tensor([20, 30, 40, 50])),
        torch.tensor([15, 27, 36, 48]).expand(3, -1),
    )
    # A new text-only request replaces the batch; other captured rows pad it.
    target = torch.arange(2).expand(3, -1)
    buffers.prepare_prefill(target, torch.tensor([1]), torch.arange(2))
    torch.testing.assert_close(
        buffers.get_decode_positions(torch.tensor([9, 0, 0, 0])),
        torch.tensor([9, 0, 0, 0]).expand(3, -1),
    )


def test_partial_multimodal_prefill_uses_current_sample_coordinates():
    buffers = MropeInputBuffers(3, 4, 1001, torch.device("cpu"))
    # One prompt finished with delta -5957; the other is before its first image.
    target = torch.cat(
        (torch.full((3, 1), 29064), torch.arange(1000).expand(3, -1)), dim=1
    )
    logical_prefill = torch.cat((torch.tensor([35021]), torch.arange(1000)))
    buffers.prepare_prefill(target, torch.tensor([0, 1000]), logical_prefill)
    torch.testing.assert_close(
        buffers.get_decode_positions(torch.tensor([35022, 1000, 0, 0])),
        torch.tensor([29065, 1000, 0, 0]).expand(3, -1),
    )


@pytest.mark.skipif(not on_rdna2_hip(), reason="RDNA2 HIP QSA kernel required")
@torch.inference_mode()
def test_native_qsa_mrope_and_changed_input_graph_replay(default_vllm_config):
    import vllm._rocm_C  # noqa: F401
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
    from vllm.models.qwen4_exp.amd.indexer_qsa import apply_qsa_rope

    device = torch.device("cuda")
    rotary = MRotaryEmbedding(
        head_size=128,
        rotary_dim=64,
        max_position_embeddings=1024,
        base=10000000,
        is_neox_style=True,
        dtype=torch.float16,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
        positions_factor=1,
    ).to(device)
    buffers = MropeInputBuffers(3, 4, 12, device)
    target = torch.tensor(
        [[10, 11, 11, 11, 12, 13, 14],
         [10, 11, 11, 12, 12, 13, 14],
         [10, 11, 12, 11, 12, 13, 14]],
        device=device,
    )
    logical_prefill = torch.tensor([14, 15, 16, 18, 19, 20, 21], device=device)
    last_indices = torch.tensor([0, 6], device=device)
    prefill_input = torch.linspace(-0.7, 0.7, 7 * 2 * 128, device=device)
    prefill_input = prefill_input.half().view(7, 2, 128)
    decode_input = prefill_input[:4].clone()
    logical = torch.tensor([100, 300, 0, 0], device=device)
    buffers.prepare_prefill(target, last_indices, logical_prefill)

    def prefill():
        return apply_qsa_rope(
            rotary, buffers.get_prefill_positions(7), prefill_input.clone()
        )

    def decode():
        return apply_qsa_rope(
            rotary, buffers.get_decode_positions(logical), decode_input.clone()
        )

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        prefill()
        decode()
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    prefill_graph, decode_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(prefill_graph):
        prefill_output = prefill()
    with torch.cuda.graph(decode_graph):
        decode_output = decode()

    for offset, indices, expected_positions in (
        (0, [0, 6], [96, 293, 0, 0]),
        (17, [6, 0], [110, 313, 0, 0]),
        (0, [3, 6], [[93, 293, 0, 0], [94, 293, 0, 0], [93, 293, 0, 0]]),
    ):
        current_target = target + offset
        buffers.prepare_prefill(
            current_target, torch.tensor(indices, device=device), logical_prefill
        )
        prefill_graph.replay()
        decode_graph.replay()
        expected_decode = torch.tensor(expected_positions, device=device).expand(3, -1)
        torch.testing.assert_close(
            prefill_output,
            apply_qsa_rope(rotary, current_target, prefill_input.clone()),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            decode_output,
            apply_qsa_rope(rotary, expected_decode, decode_input.clone()),
            rtol=0, atol=0,
        )
    legacy = apply_qsa_rope(rotary, logical, decode_input.clone())
    assert not torch.equal(decode_output, legacy)

    # Replay both graphs with a smaller partial-prefill batch and position 1000.
    partial_target = torch.arange(993, 1000, device=device).expand(3, -1)
    logical.copy_(torch.tensor([1000, 0, 0, 0], device=device))
    buffers.prepare_prefill(
        partial_target,
        torch.tensor([6], device=device),
        torch.arange(993, 1000, device=device),
    )
    prefill_graph.replay()
    decode_graph.replay()
    torch.testing.assert_close(
        prefill_output,
        apply_qsa_rope(rotary, partial_target, prefill_input.clone()),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        decode_output,
        apply_qsa_rope(rotary, logical.expand(3, -1), decode_input.clone()),
        rtol=0, atol=0,
    )
