# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
    alloc_gdn_state_arenas,
    gather_gdn_state_arenas,
    gdn_arenas_ready_for_capture,
    gdn_decode_arena_max_bs,
    scatter_gdn_state_arenas,
    static_gdn_cache_slots,
)
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")

_DUMMY_MODEL_DIR = Path(tempfile.mkdtemp(prefix="gdn_test_model_"))
(_DUMMY_MODEL_DIR / "config.json").write_text(
    json.dumps(
        {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "vocab_size": 128,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "model_type": "llama",
            "torch_dtype": "float16",
        }
    )
)


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
    piecewise_cuda_graph: bool = False,
    max_num_seqs: int = 256,
    max_cudagraph_capture_size: int | None = None,
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name=str(_DUMMY_MODEL_DIR),
        block_size=BLOCK_SIZE,
        max_num_seqs=max_num_seqs,
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    elif piecewise_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    if max_cudagraph_capture_size is not None:
        vllm_config.compilation_config.max_cudagraph_capture_size = (
            max_cudagraph_capture_size
        )
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)


def test_decode_arena_max_bs_covers_capture_size():
    """Capture sizes [1,2,4,8] must not be clipped by max_num_seqs=4."""
    builder = _create_gdn_builder(
        piecewise_cuda_graph=True,
        max_num_seqs=4,
        max_cudagraph_capture_size=8,
    )
    assert builder.decode_cudagraph_max_bs == 8
    assert builder.non_spec_state_indices_tensor.shape[0] == 8
    assert builder.cache_slot_indices_buf.shape[0] == 8
    assert gdn_decode_arena_max_bs(builder.vllm_config) == 8


def test_piecewise_decode_copies_indices_into_static_buffers():
    """PIECEWISE decode must sync block ids and 1-based arena rows."""
    builder = _create_gdn_builder(
        piecewise_cuda_graph=True,
        max_num_seqs=4,
        max_cudagraph_capture_size=8,
    )
    batch = BatchSpec(seq_lens=[40, 30], query_lens=[1, 1])
    meta = _build(builder, batch)

    assert meta.use_state_arenas
    assert meta.cache_slot_indices_is_static
    assert meta.cache_slot_indices is not None
    assert meta.non_spec_state_indices_tensor is not None
    assert (
        meta.cache_slot_indices.data_ptr() == builder.cache_slot_indices_buf.data_ptr()
    )
    assert torch.equal(
        meta.cache_slot_indices[:2].cpu(), builder.block_table_buf[:2, 0].cpu()
    )
    assert (
        meta.non_spec_state_indices_tensor.data_ptr()
        == builder.arena_state_indices.data_ptr()
    )
    assert torch.equal(
        meta.non_spec_state_indices_tensor[:2].cpu(),
        torch.tensor([1, 2], dtype=torch.int32),
    )
    # Replay a second batch: pointers stay put, contents update.
    batch2 = BatchSpec(seq_lens=[16], query_lens=[1])
    meta2 = _build(builder, batch2)
    assert (
        meta2.non_spec_state_indices_tensor is not None
        and meta2.non_spec_state_indices_tensor.data_ptr()
        == builder.arena_state_indices.data_ptr()
    )
    assert torch.equal(
        meta2.non_spec_state_indices_tensor[:1].cpu(),
        torch.tensor([1], dtype=torch.int32),
    )


def test_state_arena_data_ptr_stable_capture_vs_replay():
    """Arena storage must not move between gather (capture) and scatter (replay)."""
    max_bs = 4
    conv_cache = torch.randn(16, 8, 3)
    ssm_cache = torch.randn(16, 2, 4, 4)
    conv_arena, ssm_arena = alloc_gdn_state_arenas(
        max_bs,
        (8, 3),
        (2, 4, 4),
        conv_cache.dtype,
        ssm_cache.dtype,
        DEVICE,
    )
    conv_ptr = conv_arena.data_ptr()
    ssm_ptr = ssm_arena.data_ptr()
    slots = torch.tensor([3, 7], dtype=torch.int32)

    gather_gdn_state_arenas(conv_cache, ssm_cache, conv_arena, ssm_arena, slots, 2)
    assert conv_arena.data_ptr() == conv_ptr
    assert ssm_arena.data_ptr() == ssm_ptr
    torch.testing.assert_close(conv_arena[1], conv_cache[3])
    torch.testing.assert_close(ssm_arena[2], ssm_cache[7])

    conv_arena[1:3] += 1
    ssm_arena[1:3] += 1
    scatter_gdn_state_arenas(conv_cache, ssm_cache, conv_arena, ssm_arena, slots, 2)
    assert conv_arena.data_ptr() == conv_ptr
    assert ssm_arena.data_ptr() == ssm_ptr
    torch.testing.assert_close(conv_cache[3], conv_arena[1])
    torch.testing.assert_close(ssm_cache[7], ssm_arena[2])


def test_arenas_must_exist_before_capture():
    """Lazy first-decode alloc is illegal once BeginCapture has started."""
    conv, ssm = alloc_gdn_state_arenas(
        2, (4, 2), (1, 2, 2), torch.float32, torch.float32, DEVICE
    )
    gdn_arenas_ready_for_capture(conv, ssm, capturing=True)
    gdn_arenas_ready_for_capture(None, None, capturing=False)
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        gdn_arenas_ready_for_capture(None, None, capturing=True)


def test_gather_rejects_ephemeral_block_table_view():
    """A fresh block_table[:, 0] view must not be used as gather indices."""
    with pytest.raises(RuntimeError, match="static buffer"):
        static_gdn_cache_slots(torch.tensor([1, 2], dtype=torch.int32), is_static=False)
    slots = torch.tensor([3, 7], dtype=torch.int32)
    assert static_gdn_cache_slots(slots, is_static=True) is slots
