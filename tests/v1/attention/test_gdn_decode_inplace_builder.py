# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDNAttentionMetadataBuilder with VLLM_RDNA_GDN_DECODE_INPLACE=1.

The in-place decode (``gdn_decode_inplace_enabled``) keeps the paged slots in
the builder's own static index buffer instead of routing them through the
per-layer state arenas: a FULL/PIECEWISE graph replays the address it was
captured with, and the block table is a fresh tensor on every step. The real
builder runs on CPU tensors with the RDNA platform probes patched, so these
tests need no GPU.
"""

import dataclasses

import pytest
import torch

from tests.v1.attention.test_gdn_metadata_builder import (
    _DUMMY_MODEL_DIR,
    BLOCK_SIZE,
    DEVICE,
    _create_gdn_builder,
)
from tests.v1.attention.test_gdn_metadata_builder import _build as _build_batch
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends import gdn_attn
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
    gdn_decode_inplace_enabled,
)
from vllm.v1.kv_cache_interface import MambaSpec

NULL = 0


@pytest.fixture
def rdna2(monkeypatch):
    """An RDNA2 platform with the recipe's flags: in-place decode on, Triton
    conv update."""
    monkeypatch.setattr(gdn_attn, "on_rdna_family", lambda: True)
    monkeypatch.setattr(gdn_attn, "on_rdna2_hip", lambda: True)
    monkeypatch.setattr(gdn_attn, "_gdn_decode_rdna2_loaded", lambda: True)
    monkeypatch.setattr(gdn_attn, "leapquant_gdn_enabled", lambda: False)
    monkeypatch.setenv("VLLM_RDNA_GDN_DECODE_INPLACE", "1")
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "0")


def _builder(
    *,
    mode: CUDAGraphMode = CUDAGraphMode.FULL_AND_PIECEWISE,
    dtype="auto",
    cache_mode: str = "none",
    max_num_seqs: int = 4,
) -> GDNAttentionMetadataBuilder:
    config = create_vllm_config(
        model_name=str(_DUMMY_MODEL_DIR),
        block_size=BLOCK_SIZE,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
    )
    config.compilation_config.cudagraph_mode = mode
    config.cache_config.mamba_cache_mode = cache_mode
    spec = MambaSpec(block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,))
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["layer.0"],
        vllm_config=config,
        device=DEVICE,
    )


def _table(*slots: int, cols: int = 2) -> torch.Tensor:
    table = torch.zeros(len(slots), cols, dtype=torch.int32)
    table[:, 0] = torch.tensor(slots, dtype=torch.int32)
    return table


def _build(
    builder: GDNAttentionMetadataBuilder,
    seq_lens: list[int],
    query_lens: list[int],
    table: torch.Tensor,
) -> GDNAttentionMetadata:
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens), BLOCK_SIZE, DEVICE
    )
    return builder.build(0, common.replace(block_table_tensor=table))


def _indices(meta: GDNAttentionMetadata) -> torch.Tensor:
    indices = meta.non_spec_state_indices_tensor
    assert indices is not None
    return indices


def test_flag_off_keeps_the_arena_path(rdna2, monkeypatch):
    monkeypatch.delenv("VLLM_RDNA_GDN_DECODE_INPLACE")
    builder = _builder()
    assert builder._rdna_state_arenas and not builder._decode_inplace
    meta = _build(builder, [17, 33], [1, 1], _table(5, 9))
    assert meta.use_state_arenas
    assert meta.cache_slot_indices is not None
    assert _indices(meta).data_ptr() == builder.arena_state_indices.data_ptr()


def test_decode_slots_are_staged_in_the_static_buffer(rdna2):
    """Pure decode: the paged slots are the state indices, in the builder's own
    buffer (stable address, NULL tail); no arena rows, no cache-slot copy."""
    builder = _builder()
    assert builder._decode_inplace
    meta = _build(builder, [17, 33, 0, 0], [1, 1, 0, 0], _table(5, 9, NULL, NULL))
    assert not meta.use_state_arenas and meta.cache_slot_indices is None
    ptr = builder.non_spec_state_indices_tensor.data_ptr()
    assert _indices(meta).data_ptr() == ptr
    assert _indices(meta).tolist() == [5, 9, NULL, NULL]
    # The next step: another table, the same address, the new slots.
    meta2 = _build(builder, [20, 0, 0, 0], [1, 0, 0, 0], _table(11, NULL, NULL, NULL))
    assert _indices(meta2).data_ptr() == ptr
    assert _indices(meta2).tolist() == [11, NULL, NULL, NULL]
    assert _indices(meta).tolist() == [11, NULL, NULL, NULL]  # same buffer


def test_padded_requests_read_null_not_the_previous_slots(rdna2):
    """A padded graph batch: the zero-length requests' block-table rows are
    NULL, so rows that held live slots one step earlier read NULL now."""
    builder = _builder(max_num_seqs=8)
    meta = _build(builder, [17] * 6, [1] * 6, _table(1, 2, 3, 4, 5, 6))
    assert _indices(meta).tolist() == [1, 2, 3, 4, 5, 6]
    meta = _build(
        builder, [17, 18, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], _table(7, 8, *[NULL] * 4)
    )
    assert _indices(meta).tolist() == [7, 8, NULL, NULL, NULL, NULL]


def test_prefill_batches_are_not_staged(rdna2):
    builder = _builder()
    meta = _build(builder, [17, 40], [1, 5], _table(5, 9))
    assert meta.num_prefills == 1
    assert not meta.use_state_arenas and meta.cache_slot_indices is None
    assert _indices(meta).data_ptr() != builder.non_spec_state_indices_tensor.data_ptr()


@pytest.mark.parametrize(
    "mode", [CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.PIECEWISE]
)
def test_other_kv_groups_stage_their_own_buffer(rdna2, mode):
    """update_block_table re-gathers another KV group's slots: they land in
    that group's builder, whose graphs read its buffer, not an ephemeral
    block-table view."""
    first, second = _builder(mode=mode), _builder(mode=mode)
    meta = _build(first, [17, 33], [1, 1], _table(5, 9))
    other = second.update_block_table(
        meta, _table(15, 19), torch.zeros(2, dtype=torch.int64)
    )
    assert _indices(other).data_ptr() == second.non_spec_state_indices_tensor.data_ptr()
    assert _indices(other).data_ptr() != first.non_spec_state_indices_tensor.data_ptr()
    assert _indices(other).tolist() == [15, 19]
    assert not other.use_state_arenas
    assert _indices(meta).tolist() == [5, 9]  # the first group is unchanged


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_same(a[k], b[k]) for k in a)
        )
    return a == b


# (num speculative tokens, seq_lens, query_lens, draft tokens per request or None)
_NOT_PURE_DECODE = {
    "mtp_verify": (2, [80, 96], [3, 3], [2, 2]),
    "mtp_verify_with_a_row_that_did_not_draft": (
        2,
        [80, 96, 70],
        [3, 3, 1],
        [2, 2, -1],
    ),
    "mtp_verify_and_a_prefill": (2, [80, 50], [3, 50], [2, -1]),
    "decode_and_a_prefill": (0, [64, 50], [1, 50], None),
}


@pytest.mark.parametrize("case", _NOT_PURE_DECODE.values(), ids=_NOT_PURE_DECODE.keys())
def test_other_steps_keep_their_metadata(rdna2, monkeypatch, case):
    """MTP verify and mixed prefill + decode steps never went through the
    arenas: the flag leaves their metadata field for field what it was."""
    num_spec, seq_lens, query_lens, draft = case
    inplace = _create_gdn_builder(num_spec, full_cuda_graph=True, max_num_seqs=8)
    monkeypatch.setenv("VLLM_RDNA_GDN_DECODE_INPLACE", "0")
    arena = _create_gdn_builder(num_spec, full_cuda_graph=True, max_num_seqs=8)
    assert inplace._decode_inplace and not arena._decode_inplace
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    table = torch.arange(1, 1 + 8 * len(seq_lens), dtype=torch.int32).view(-1, 8)
    got = _build_batch(inplace, batch, draft, table)
    want = _build_batch(arena, batch, draft, table)
    assert not got.use_state_arenas and not want.use_state_arenas
    for field in dataclasses.fields(GDNAttentionMetadata):
        assert _same(getattr(got, field.name), getattr(want, field.name)), field.name


@pytest.mark.parametrize("cache_mode", ["none", "align"])
def test_cache_modes(rdna2, cache_mode):
    builder = _builder(cache_mode=cache_mode)
    # seq_len <= block_size: the state block is column 0 in align mode too
    meta = _build(builder, [9, 12], [1, 1], _table(5, 9))
    assert not meta.use_state_arenas
    assert _indices(meta).data_ptr() == builder.non_spec_state_indices_tensor.data_ptr()
    assert _indices(meta).tolist() == [5, 9]


def test_align_mode_stages_the_last_block_of_each_request(rdna2, monkeypatch):
    """Prefix caching (mamba align mode) keeps a request's state in the last
    block of its table, a different column per request. The staged slots are
    what the arena gather would have read."""
    table = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.int32)
    seq_lens = [BLOCK_SIZE // 2, 2 * BLOCK_SIZE + 5]  # state in block 0 and block 2
    inplace = _builder(cache_mode="align")
    monkeypatch.setenv("VLLM_RDNA_GDN_DECODE_INPLACE", "0")
    arena = _builder(cache_mode="align")
    assert inplace._decode_inplace and not arena._decode_inplace
    staged = _build(inplace, seq_lens, [1, 1], table)
    reference = _build(arena, seq_lens, [1, 1], table)
    assert reference.cache_slot_indices is not None
    assert reference.cache_slot_indices.tolist() == [5, 11]
    assert _indices(staged).tolist() == [5, 11]
    assert (
        _indices(staged).data_ptr() == inplace.non_spec_state_indices_tensor.data_ptr()
    )


def test_enable_predicate(rdna2, monkeypatch):
    config = _builder().vllm_config
    assert gdn_decode_inplace_enabled(config)
    # the HIP conv update runs slot 0 as a real line: it would write the
    # paged cache's null block
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "1")
    assert not gdn_decode_inplace_enabled(config)
    monkeypatch.delenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE")
    assert not gdn_decode_inplace_enabled(config)
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "0")
    assert gdn_decode_inplace_enabled(config)
    monkeypatch.setattr(gdn_attn, "on_rdna2_hip", lambda: False)
    assert not gdn_decode_inplace_enabled(config)
    monkeypatch.setattr(gdn_attn, "on_rdna2_hip", lambda: True)
    # the layer's strided HIP decode is what was validated
    for name in (
        "VLLM_RDNA_GDN_OPAQUE",
        "VLLM_GDN_DECODE_RDNA2",
        "VLLM_GDN_STRIDED_INPUTS",
    ):
        monkeypatch.setenv(name, "0")
        assert not gdn_decode_inplace_enabled(config), name
        monkeypatch.setenv(name, "1")
        assert gdn_decode_inplace_enabled(config), name
        monkeypatch.delenv(name)
    monkeypatch.setattr(gdn_attn, "_gdn_decode_rdna2_loaded", lambda: False)
    assert not gdn_decode_inplace_enabled(config)
    monkeypatch.setattr(gdn_attn, "_gdn_decode_rdna2_loaded", lambda: True)
    monkeypatch.setattr(gdn_attn, "leapquant_gdn_enabled", lambda: True)
    assert not gdn_decode_inplace_enabled(config)
    monkeypatch.setattr(gdn_attn, "leapquant_gdn_enabled", lambda: False)
    assert not gdn_decode_inplace_enabled(_builder(dtype=torch.bfloat16).vllm_config)
    monkeypatch.delenv("VLLM_RDNA_GDN_DECODE_INPLACE")
    assert not gdn_decode_inplace_enabled(config)
    monkeypatch.setenv("VLLM_RDNA_GDN_DECODE_INPLACE", "0")
    assert not gdn_decode_inplace_enabled(config)


def test_ignored_flag_keeps_the_arenas(rdna2, monkeypatch):
    """Flag on with the HIP conv update: the arena path, not a half-way mix."""
    monkeypatch.setenv("VLLM_CAUSAL_CONV1D_RDNA2_UPDATE", "1")
    builder = _builder()
    assert not builder._decode_inplace
    meta = _build(builder, [17, 33], [1, 1], _table(5, 9))
    assert meta.use_state_arenas
