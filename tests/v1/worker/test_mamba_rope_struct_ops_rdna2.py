# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-for-bit parity between the gfx1030 HIP ports and the Triton kernels for

  preprocess_mamba_align_fused_kernel  (model_states/mamba_hybrid.py)
  _scatter_num_accepted_kernel         (model_states/mamba_hybrid.py)
  _fill_num_accepted_kernel            (model_states/mamba_hybrid.py)
  precopy_mamba_align_fused_kernel     (mamba_utils.py)
  postprocess_mamba_fused_kernel       (mamba_utils.py)
  batch_memcpy_kernel                  (mamba_utils.py)
  _prepare_rope_positions_kernel       (gpu/mm/rope.py)
  _apply_grammar_bitmask_kernel        (gpu/structured_outputs.py)

All of these run once per decode step inside a FULL cudagraph capture, so the
only acceptable difference between the two paths is none. Every assertion is
`torch.equal` (bit-for-bit), not a tolerance.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms.rdna import on_rdna2
from vllm.utils.platform_utils import is_uva_available
from vllm.v1.worker import mamba_utils as mu
from vllm.v1.worker.gpu import structured_outputs as so
from vllm.v1.worker.gpu.mm import rope as rope_mod
from vllm.v1.worker.gpu.model_states import mamba_hybrid as mh

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_rdna2()),
    reason="gfx10x (RDNA2) ROCm required",
)

DEVICE = torch.device("cuda")
OP = torch.ops._rocm_C

# Real serving geometry (Qwen3.8-Flash-Next W4A16, TP4, MTP2, block_size 1024).
SERVE_BLOCK_SIZE = 1024
SERVE_NUM_REQS = 8
SERVE_CONV_WIDTH = 4
SERVE_CONV_DIM = 2560  # SD inner dim / DS dim rows (GDN, TP4)
SERVE_TEMPORAL = 196608  # value_head_dim * value_heads (GDN, TP4)
TEMPORAL_TILES = 16
TABLE_COLS = 16


def sync():
    torch.cuda.synchronize()


def cdiv(a, b):
    return (a + b - 1) // b


def rand(shape, dtype, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=gen).to(dtype).to(DEVICE)


def randint(low, high, shape, dtype, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(low, high, shape, generator=gen).to(dtype).to(DEVICE)


# ---------------------------------------------------------------------------
# Mamba copy rig: state tensors + per-group block tables + flattened metadata
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class StateSpec:
    tensor: torch.Tensor
    block_stride: int
    elem_size: int
    inner_size: int
    conv_width: int
    group_idx: int
    dim_row_count: int = 0
    dim_row_stride: int = 0
    storage: torch.Tensor | None = None


def _conv_state(layout, num_blocks, conv_width, conv_dim, dtype, seed):
    if layout == "SD":
        base = rand((num_blocks, conv_width, conv_dim), dtype, seed)
        return StateSpec(
            base,
            base.stride(0) * base.element_size(),
            base.element_size(),
            base.stride(1),
            conv_width,
            0,
        )
    base = rand((num_blocks, conv_dim, conv_width), dtype, seed)
    return StateSpec(
        base,
        base.stride(0) * base.element_size(),
        base.element_size(),
        1,
        conv_width,
        0,
        dim_row_count=conv_dim,
        dim_row_stride=base.stride(1) * base.element_size(),
    )


def _temporal_state(num_blocks, inner, dtype, seed, page_pad=0):
    """Temporal state, optionally with padded (page-rounded) block stride."""
    stride0 = inner + page_pad
    flat = rand((num_blocks * stride0,), dtype, seed)
    view = flat.as_strided((num_blocks, inner), (stride0, 1))
    return StateSpec(
        view,
        stride0 * view.element_size(),
        view.element_size(),
        inner,
        0,
        0,
        storage=flat,
    )


@dataclasses.dataclass
class Rig:
    specs: list[StateSpec]
    block_tables: list[torch.Tensor]
    block_table_stride_req: int

    def clone(self):
        """A structurally identical rig with freshly allocated state memory."""
        specs = []
        for spec in self.specs:
            if spec.storage is not None:
                storage = spec.storage.clone()
                view = storage.as_strided(spec.tensor.shape, spec.tensor.stride())
                specs.append(
                    dataclasses.replace(spec, tensor=view, storage=storage)
                )
            else:
                specs.append(dataclasses.replace(spec, tensor=spec.tensor.clone()))
        return Rig(specs, self.block_tables, self.block_table_stride_req)

    def payload(self):
        return [
            s.storage if s.storage is not None else s.tensor for s in self.specs
        ]

    def meta(self):
        return _build_meta(self.specs, self.block_tables,
                           self.block_table_stride_req)


def _raw_ptr(tensor):
    ptr = tensor.data_ptr()
    return ptr if ptr < (1 << 63) else ptr - (1 << 64)


def _build_meta(specs, block_tables, stride):
    return {
        "state_base_addrs": torch.tensor(
            [_raw_ptr(s.tensor) for s in specs], dtype=torch.int64, device=DEVICE
        ),
        "state_block_strides": torch.tensor(
            [s.block_stride for s in specs], dtype=torch.int64, device=DEVICE
        ),
        "state_elem_sizes": torch.tensor(
            [s.elem_size for s in specs], dtype=torch.int32, device=DEVICE
        ),
        "state_inner_sizes": torch.tensor(
            [s.inner_size for s in specs], dtype=torch.int64, device=DEVICE
        ),
        "state_conv_widths": torch.tensor(
            [s.conv_width for s in specs], dtype=torch.int32, device=DEVICE
        ),
        "state_group_indices": torch.tensor(
            [s.group_idx for s in specs], dtype=torch.int32, device=DEVICE
        ),
        "state_dim_row_count": torch.tensor(
            [s.dim_row_count for s in specs], dtype=torch.int32, device=DEVICE
        ),
        "state_dim_row_stride": torch.tensor(
            [s.dim_row_stride for s in specs], dtype=torch.int64, device=DEVICE
        ),
        "block_table_ptrs": torch.tensor(
            [_raw_ptr(bt) for bt in block_tables],
            dtype=torch.int64,
            device=DEVICE,
        ),
        "block_table_stride_req": stride,
    }


def make_rig(
    *,
    layout="SD",
    max_reqs=8,
    table_cols=None,
    conv_width=4,
    conv_dim=64,
    temporal_inner=512,
    dtype=torch.float16,
    num_groups=2,
    page_pad=0,
    seed=0,
    same_block=False,
    bias_max=4,
):
    """Two mamba groups; each group has a conv state and a temporal state.

    The block table is built per row so that every row's *destination* block is
    unique (row i writes block i through every column that is not a reserved
    source column). That keeps the concurrent per-request copies race-free, as
    they are in serving. Row i's conv source column carries block ``max_reqs+i``
    (distinct, never a destination block) unless ``same_block``, in which case
    it carries block ``i`` so the same-block shift path is exercised.
    """
    table_cols = table_cols if table_cols is not None else 2 * max_reqs + 8
    num_blocks = 2 * max_reqs
    src_cols = max_reqs + torch.arange(max_reqs)

    row = torch.arange(max_reqs, dtype=torch.int32).unsqueeze(1)
    bt = row.expand(max_reqs, table_cols).contiguous()
    for i in range(max_reqs):
        base = int(src_cols[i])
        for k in range(bias_max):
            if same_block and k == 0:
                continue
            bt[i, base + k] = max_reqs + i
    block_tables = [bt.clone().to(DEVICE) for _ in range(num_groups)]

    specs = []
    for g in range(num_groups):
        conv = _conv_state(layout, num_blocks, conv_width, conv_dim, dtype,
                           seed + 10 * g)
        conv.group_idx = g
        temporal = _temporal_state(num_blocks, temporal_inner, dtype,
                                   seed + 10 * g + 1, page_pad)
        temporal.group_idx = g
        specs.extend([conv, temporal])

    stride = block_tables[0].stride(0)
    return Rig(specs, block_tables, stride), src_cols.to(DEVICE).to(torch.int32)

# ---------------------------------------------------------------------------
# _prepare_rope_positions_kernel
# ---------------------------------------------------------------------------
def triton_rope(positions, prefill_positions, prefill_delta, idx_mapping,
                query_start_loc, prefill_lens, num_computed, num_dims,
                max_model_len):
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    rope_mod._prepare_rope_positions_kernel[(num_reqs,)](
        positions,
        positions.stride(0),
        prefill_positions,
        num_dims * max_model_len,
        max_model_len,
        prefill_delta,
        idx_mapping,
        query_start_loc,
        prefill_lens,
        num_computed,
        BLOCK_SIZE=1024,
        NUM_DIMS=num_dims,
    )


def hip_rope(positions, prefill_positions, prefill_delta, idx_mapping,
             query_start_loc, prefill_lens, num_computed, num_dims,
             max_model_len):
    OP.prepare_rope_positions_rdna2(
        positions,
        positions.stride(0),
        prefill_positions,
        num_dims * max_model_len,
        max_model_len,
        prefill_delta,
        idx_mapping,
        query_start_loc,
        prefill_lens,
        num_computed,
        num_dims,
    )


# ---------------------------------------------------------------------------
# _apply_grammar_bitmask_kernel
# ---------------------------------------------------------------------------
def triton_bitmask(logits, logits_indices, cu_num_logits, bitmask, mask_stride):
    num_masks = logits_indices.shape[0]
    if num_masks == 0:
        return
    vocab_size = logits.shape[-1]
    block_size = 8192
    grid = (num_masks, cdiv(vocab_size, block_size))
    so._apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        logits_indices,
        cu_num_logits,
        bitmask,
        bitmask.stride(0),
        vocab_size,
        MASK_STRIDE=mask_stride,
        BLOCK_SIZE=block_size,
    )


def hip_bitmask(logits, logits_indices, cu_num_logits, bitmask, mask_stride):
    OP.apply_grammar_bitmask_rdna2(
        logits,
        logits.stride(0),
        logits_indices,
        cu_num_logits,
        bitmask,
        bitmask.stride(0),
        logits.shape[-1],
        mask_stride,
    )


# ===========================================================================
# preprocess_mamba_align_fused_kernel
# ===========================================================================
def triton_preprocess(idx_mapping, state_idx, num_computed, query_start_loc,
                      num_accepted, src_col, src_off, num_reqs, mamba_block):
    if num_reqs == 0:
        return
    block = 256
    grid = (cdiv(num_reqs, block),)
    mu.preprocess_mamba_align_fused_kernel[grid](
        idx_mapping,
        state_idx,
        num_computed,
        query_start_loc,
        num_accepted,
        src_col,
        src_off,
        num_reqs,
        BLOCK_SIZE=block,
        MAMBA_BLOCK_SIZE=mamba_block,
    )


@pytest.mark.parametrize("num_reqs", [0, 1, 7, 255, 256, 257, 1000])
@pytest.mark.parametrize("mamba_block", [4, 1024])
def test_preprocess_mamba_align_parity(num_reqs, mamba_block):
    max_reqs = max(num_reqs, 1) + 4
    seed = num_reqs + mamba_block
    idx_mapping = torch.randperm(
        max_reqs, generator=torch.Generator().manual_seed(seed)
    )[:num_reqs].to(torch.int32).to(DEVICE)
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
    gen = torch.Generator().manual_seed(seed + 1)
    qsl[1:] = torch.randint(0, 5, (num_reqs,), generator=gen).cumsum(0)
    query_start_loc = qsl.to(DEVICE)

    inputs = {
        "state_idx": randint(-1, 6, (max_reqs,), torch.int32, seed + 2),
        "num_computed": randint(0, 4 * mamba_block, (max_reqs,), torch.int32,
                                seed + 3),
        "num_accepted": randint(-1, 5, (max_reqs,), torch.int32, seed + 4),
    }

    def run(fn):
        args = {k: v.clone() for k, v in inputs.items()}
        src_col = torch.full((max_reqs,), -777, dtype=torch.int32, device=DEVICE)
        src_off = torch.full((max_reqs,), -777, dtype=torch.int32, device=DEVICE)
        fn(
            idx_mapping,
            args["state_idx"],
            args["num_computed"],
            query_start_loc,
            args["num_accepted"],
            src_col,
            src_off,
            num_reqs,
            mamba_block,
        )
        return args["state_idx"], args["num_accepted"], src_col, src_off

    ref = run(triton_preprocess)
    got = run(mh._preprocess_mamba_align_fused)
    sync()
    for a, b in zip(ref, got):
        assert torch.equal(a, b)


# ===========================================================================
# _scatter_num_accepted_kernel / _fill_num_accepted_kernel
# ===========================================================================
@pytest.mark.parametrize("num_reqs", [0, 1, 8, 256, 257])
def test_scatter_and_fill_num_accepted_parity(num_reqs):
    max_reqs = max(num_reqs, 1) + 4
    idx_mapping = torch.randperm(
        max_reqs, generator=torch.Generator().manual_seed(num_reqs)
    )[:num_reqs].to(torch.int32).to(DEVICE)
    # Include a -1 sentinel row (filtered request under PP).
    if num_reqs >= 3:
        idx_mapping[1] = -1
    num_sampled = randint(-2, 5, (num_reqs,), torch.int32, num_reqs + 7)
    base = torch.full((max_reqs,), -999, dtype=torch.int32, device=DEVICE)

    ref = base.clone()
    if num_reqs:
        mh._scatter_num_accepted_kernel[(num_reqs,)](idx_mapping, num_sampled, ref)
    got = base.clone()
    mh._update_num_accepted_tokens(idx_mapping, num_sampled, got, num_reqs)
    sync()
    assert torch.equal(ref, got)

    for value in (1, 2, 0):
        ref = base.clone()
        if num_reqs:
            mh._fill_num_accepted_kernel[(num_reqs,)](
                idx_mapping, ref, max(value, 1)
            )
        got = base.clone()
        mh._update_num_accepted_tokens(idx_mapping, value, got, num_reqs)
        sync()
        assert torch.equal(ref, got)


# ===========================================================================
# precopy_mamba_align_fused_kernel / postprocess_mamba_fused_kernel
# ===========================================================================
def _triton_precopy(rig, num_reqs, idx_mapping, src_col, dst_col, token_bias,
                    conv_state_dim_first):
    mu.precopy_mamba_align_fused_kernel[(num_reqs, len(rig.specs),
                                         TEMPORAL_TILES)](
        dst_col,
        src_col,
        token_bias,
        rig.meta()["block_table_ptrs"],
        rig.meta()["block_table_stride_req"],
        rig.meta()["state_base_addrs"],
        rig.meta()["state_block_strides"],
        rig.meta()["state_elem_sizes"],
        rig.meta()["state_inner_sizes"],
        rig.meta()["state_conv_widths"],
        rig.meta()["state_group_indices"],
        rig.meta()["state_dim_row_count"],
        rig.meta()["state_dim_row_stride"],
        idx_mapping,
        num_reqs,
        COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=conv_state_dim_first,
        HAS_IDX_MAPPING=idx_mapping is not None,
        TEMPORAL_TILES=TEMPORAL_TILES,
    )


def _hip_precopy(rig, num_reqs, idx_mapping, src_col, dst_col, token_bias,
                 conv_state_dim_first):
    OP.precopy_mamba_align_fused_rdna2(
        dst_col,
        src_col,
        token_bias,
        idx_mapping=idx_mapping,
        num_reqs=num_reqs,
        conv_state_dim_first=conv_state_dim_first,
        temporal_tiles=TEMPORAL_TILES,
        **rig.meta(),
    )


def _triton_postprocess(rig, num_reqs, idx_mapping, num_accepted, src_col,
                        num_scheduled, num_computed, num_draft, block_size,
                        conv_state_dim_first, precomputed):
    accepted_out = torch.full(
        (num_accepted.shape[0],), -321, dtype=torch.int32, device=DEVICE
    )
    mu.postprocess_mamba_fused_kernel[(num_reqs, len(rig.specs),
                                       TEMPORAL_TILES)](
        num_accepted,
        src_col,
        num_scheduled if not precomputed else None,
        num_computed,
        num_draft if not precomputed else None,
        rig.meta()["block_table_ptrs"],
        rig.meta()["block_table_stride_req"],
        rig.meta()["state_base_addrs"],
        rig.meta()["state_block_strides"],
        rig.meta()["state_elem_sizes"],
        rig.meta()["state_inner_sizes"],
        rig.meta()["state_conv_widths"],
        rig.meta()["state_group_indices"],
        rig.meta()["state_dim_row_count"],
        rig.meta()["state_dim_row_stride"],
        accepted_out,
        idx_mapping,
        num_reqs,
        block_size=block_size,
        COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=conv_state_dim_first,
        HAS_IDX_MAPPING=idx_mapping is not None,
        PRECOMPUTED_NEW_COMPUTED=precomputed,
        TEMPORAL_TILES=TEMPORAL_TILES,
    )
    return accepted_out


def _hip_postprocess(rig, num_reqs, idx_mapping, num_accepted, src_col,
                     num_scheduled, num_computed, num_draft, block_size,
                     conv_state_dim_first, precomputed):
    accepted_out = torch.full(
        (num_accepted.shape[0],), -321, dtype=torch.int32, device=DEVICE
    )
    OP.postprocess_mamba_fused_rdna2(
        num_accepted,
        src_col,
        None if precomputed else num_scheduled,
        num_computed,
        None if precomputed else num_draft,
        num_accepted_tokens_out=accepted_out,
        idx_mapping=idx_mapping,
        num_reqs=num_reqs,
        block_size=block_size,
        conv_state_dim_first=conv_state_dim_first,
        precomputed_new_computed=precomputed,
        temporal_tiles=TEMPORAL_TILES,
        **rig.meta(),
    )
    return accepted_out


@dataclasses.dataclass
class CopyCase:
    layout: str = "SD"
    num_reqs: int = SERVE_NUM_REQS
    conv_state_dim_first: bool = False
    has_idx_mapping: bool = False
    precomputed: bool = False
    block_size: int = SERVE_BLOCK_SIZE
    conv_width: int = SERVE_CONV_WIDTH
    conv_dim: int = SERVE_CONV_DIM
    temporal_inner: int = SERVE_TEMPORAL
    page_pad: int = 0
    seed: int = 7
    same_block: bool = False


def run_copy_case(mode, case: CopyCase):
    """Run `mode` on the Triton and HIP paths over two identical rigs."""
    num_reqs = case.num_reqs
    max_reqs = num_reqs + 4
    seed = case.seed
    rig, src_col = make_rig(
        layout=case.layout,
        max_reqs=max_reqs,
        conv_width=case.conv_width,
        conv_dim=case.conv_dim,
        temporal_inner=case.temporal_inner,
        num_groups=2,
        page_pad=case.page_pad,
        seed=seed,
        same_block=case.same_block,
    )
    hip_rig = rig.clone()
    sync()

    if case.has_idx_mapping:
        idx_mapping = torch.randperm(
            max_reqs, generator=torch.Generator().manual_seed(seed)
        )[:num_reqs].to(torch.int32).to(DEVICE)
    else:
        idx_mapping = None

    if case.same_block:
        token_bias = torch.full((max_reqs,), 2, dtype=torch.int32, device=DEVICE)
    else:
        token_bias = randint(0, 4, (max_reqs,), torch.int32, seed + 3)
    dst_col = torch.arange(max_reqs, dtype=torch.int32, device=DEVICE)
    if mode == "precopy":
        _triton_precopy(rig, num_reqs, idx_mapping, src_col, dst_col, token_bias,
                        case.conv_state_dim_first)
        sync()
        _hip_precopy(hip_rig, num_reqs, idx_mapping, src_col, dst_col, token_bias,
                     case.conv_state_dim_first)
        sync()
        return rig.payload(), hip_rig.payload(), None

    num_accepted = token_bias + 1
    num_scheduled = randint(0, 4, (max_reqs,), torch.int32, seed + 4)
    num_draft = randint(0, 3, (max_reqs,), torch.int32, seed + 5)
    # The destination column is derived by the kernel; every block-table
    # row defaults to its own block, so the per-row destination is unique.
    num_computed = randint(
        case.block_size + 8, 4 * case.block_size, (max_reqs,), torch.int32,
        seed + 6,
    )
    ref_out = _triton_postprocess(
        rig, num_reqs, idx_mapping, num_accepted, src_col, num_scheduled,
        num_computed, num_draft, case.block_size, case.conv_state_dim_first,
        case.precomputed,
    )
    sync()
    got_out = _hip_postprocess(
        hip_rig, num_reqs, idx_mapping, num_accepted, src_col, num_scheduled,
        num_computed, num_draft, case.block_size, case.conv_state_dim_first,
        case.precomputed,
    )
    sync()
    return rig.payload(), hip_rig.payload(), (ref_out, got_out)


def assert_copy_parity(mode, case: CopyCase):
    ref, got, outs = run_copy_case(mode, case)
    for i, (a, b) in enumerate(zip(ref, got)):
        assert torch.equal(a, b), f"state {i} differs"
    if outs is not None:
        assert torch.equal(outs[0], outs[1]), "num_accepted_tokens_out differs"


def test_precopy_sd_serving_shapes():
    assert_copy_parity("precopy", CopyCase())


def test_precopy_ds_serving_shapes():
    assert_copy_parity(
        "precopy", CopyCase(layout="DS", conv_state_dim_first=True)
    )


def test_precopy_padded_pages_and_idx_mapping():
    assert_copy_parity(
        "precopy", CopyCase(has_idx_mapping=True, page_pad=64, seed=11)
    )


def test_precopy_small_block_size():
    assert_copy_parity(
        "precopy", CopyCase(block_size=4, temporal_inner=512, conv_dim=32, seed=13)
    )


def test_precopy_same_block_shift():
    """src_col and dst_col resolving to the same physical block with
    token_bias > 0 -> the per-token same-block shift path."""
    assert_copy_parity("precopy", CopyCase(same_block=True, temporal_inner=1024))


def test_precopy_single_request():
    assert_copy_parity(
        "precopy", CopyCase(num_reqs=1, conv_dim=16, temporal_inner=256, seed=19)
    )


@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("has_idx_mapping", [False, True])
def test_postprocess_parity(precomputed, has_idx_mapping):
    assert_copy_parity(
        "postprocess",
        CopyCase(
            precomputed=precomputed,
            has_idx_mapping=has_idx_mapping,
            seed=17 + int(precomputed) + 2 * int(has_idx_mapping),
        ),
    )


def test_postprocess_ds_layout():
    assert_copy_parity(
        "postprocess",
        CopyCase(layout="DS", conv_state_dim_first=True, seed=23),
    )


def test_postprocess_same_block_shift():
    assert_copy_parity(
        "postprocess",
        CopyCase(same_block=True, temporal_inner=1024, seed=27),
    )


def test_postprocess_single_request():
    assert_copy_parity(
        "postprocess",
        CopyCase(num_reqs=1, block_size=1024, conv_dim=16, temporal_inner=256,
                 seed=29),
    )


def test_postprocess_padded_pages():
    assert_copy_parity(
        "postprocess",
        CopyCase(page_pad=64, conv_dim=128, temporal_inner=2048, seed=33),
    )


# ===========================================================================
# batch_memcpy_kernel
# ===========================================================================
def _ptr_tensor(ptrs):
    return torch.tensor(ptrs, dtype=torch.uint64, device=DEVICE)


def _batch_memcpy_run(fn, total, src_off, dst_off, sizes, seed):
    buf = randint(0, 256, (total,), torch.uint8, seed)
    sp = _ptr_tensor([buf.data_ptr() + o for o in src_off])
    dp = _ptr_tensor([buf.data_ptr() + o for o in dst_off])
    sz = torch.tensor(sizes, dtype=torch.int32, device=DEVICE)
    fn(sp, dp, sz)
    return buf


def _triton_batch_memcpy(sp, dp, sz):
    mu.batch_memcpy_kernel[(sz.shape[0],)](sp, dp, sz, BLOCK_SIZE=1024)


def _hip_batch_memcpy(sp, dp, sz):
    OP.batch_memcpy_rdna2(sp, dp, sz)


def test_batch_memcpy_parity_disjoint():
    sizes = [0, 1, 7, 8, 9, 1023, 1024, 1025, 8192, 3 * 1024 * 1024 + 5]
    src_off, dst_off = [], []
    off = 32
    for s in sizes:
        src_off.append(off)
        dst_off.append(off)
        off += max(s, 1)
    total = off + 64
    ref = _batch_memcpy_run(_triton_batch_memcpy, total, src_off, dst_off, sizes, 41)
    got = _batch_memcpy_run(_hip_batch_memcpy, total, src_off, dst_off, sizes, 41)
    sync()
    assert torch.equal(ref, got)
    # Sizes above 1024 must actually have been copied.
    assert not torch.equal(ref[:total], torch.zeros(total, dtype=torch.uint8,
                                                    device=DEVICE))


def test_batch_memcpy_parity_left_overlap():
    """dst < src with overlapping ranges: the in-kernel barrier matters.

    The entries must not overlap each other: the kernel makes no ordering
    guarantee across entries, so two entries sharing bytes are undefined and
    Triton and HIP legitimately disagree there.
    """
    sizes = [4096, 1024]
    src_off = [64, 8192 + 64]
    dst_off = [0, 8192]
    total = max(s + o for s, o in zip(sizes, src_off)) + 64
    ref = _batch_memcpy_run(_triton_batch_memcpy, total, src_off, dst_off, sizes, 43)
    got = _batch_memcpy_run(_hip_batch_memcpy, total, src_off, dst_off, sizes, 43)
    sync()
    assert torch.equal(ref, got)


def test_batch_memcpy_parity_misaligned():
    """Mismatched sub-8B alignment takes the byte-wide slow path."""
    sizes = [1023, 512, 8]
    # Non-overlapping ranges with mismatched sub-8B alignment: a right overlap
    # (dst > src) is unsafe on both paths (no barrier), as upstream documents.
    src_off = [3, 11, 1]
    dst_off = [2048 + 5, 1024 + 2, 3072 + 7]
    total = max(s + o for s, o in zip(sizes, src_off)) + 4096
    ref = _batch_memcpy_run(_triton_batch_memcpy, total, src_off, dst_off, sizes, 47)
    got = _batch_memcpy_run(_hip_batch_memcpy, total, src_off, dst_off, sizes, 47)
    sync()
    assert torch.equal(ref, got)


def test_batch_memcpy_empty_batch():
    OP.batch_memcpy_rdna2(
        torch.zeros(0, dtype=torch.uint64, device=DEVICE),
        torch.zeros(0, dtype=torch.uint64, device=DEVICE),
        torch.zeros(0, dtype=torch.int32, device=DEVICE),
    )


# ===========================================================================
# _prepare_rope_positions_kernel
# ===========================================================================
@pytest.mark.parametrize("num_dims", [2, 3, 4])
@pytest.mark.parametrize("num_reqs", [0, 1, 8, 33])
def test_prepare_rope_positions_parity(num_dims, num_reqs):
    max_reqs = max(num_reqs, 1) + 4
    max_num_tokens = 2048
    max_model_len = 4096
    seed = num_reqs * 10 + num_dims
    idx_mapping = torch.randperm(
        max_reqs, generator=torch.Generator().manual_seed(seed)
    )[:num_reqs].to(torch.int32).to(DEVICE)
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
    gen = torch.Generator().manual_seed(seed + 1)
    qsl[1:] = torch.randint(1, 40, (num_reqs,), generator=gen).cumsum(0)
    query_start_loc = qsl.to(DEVICE)
    # Keep num_computed + query_len inside one prefill_positions row so both
    # paths read the same in-bounds data.
    prefill_lens = randint(0, max_model_len, (max_reqs,), torch.int32, seed + 2)
    num_computed = randint(0, max_model_len - 2048, (max_reqs,), torch.int32,
                           seed + 3)
    prefill_delta = randint(-5, 5, (max_reqs,), torch.int32, seed + 4)
    prefill_positions = randint(
        -1000, 100000, (max_reqs * num_dims, max_model_len), torch.int32, seed + 5
    )

    ref = torch.full(
        (num_dims, max_num_tokens + 1), -12345, dtype=torch.int64, device=DEVICE
    )
    got = ref.clone()
    triton_rope(ref, prefill_positions, prefill_delta, idx_mapping,
                query_start_loc, prefill_lens, num_computed, num_dims,
                max_model_len)
    hip_rope(got, prefill_positions, prefill_delta, idx_mapping,
             query_start_loc, prefill_lens, num_computed, num_dims, max_model_len)
    sync()
    assert torch.equal(ref, got)


def test_prepare_rope_positions_serving_boundaries():
    """Real serving max_model_len, both prefill and decode rows, and query
    lengths that straddle the 1024-element Triton tile edge."""
    num_dims, num_reqs, max_model_len = 4, 8, 262144
    max_num_tokens = 8192
    max_reqs = num_reqs + 4
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    qsl[1:] = torch.tensor(
        [1, 1, 1, 1, 1023, 1024, 1025, 2048], dtype=torch.int32, device=DEVICE
    ).cumsum(0)
    # Rows 0..3 are decode (num_computed >= prefill_len), rows 4..7 prefill.
    prefill_lens = torch.full((max_reqs,), max_model_len, dtype=torch.int32,
                              device=DEVICE)
    prefill_lens[:4] = 1
    num_computed = torch.tensor(
        [0, 5, 500, 1023, 0, 0, 0, 1000, 0, 0, 0, 0],
        dtype=torch.int32,
        device=DEVICE,
    )
    prefill_delta = randint(-3, 3, (max_reqs,), torch.int32, 51)
    prefill_positions = randint(
        -1000, 100000, (max_reqs * num_dims, max_model_len), torch.int32, 53
    )
    ref = torch.full((num_dims, max_num_tokens + 1), -12345, dtype=torch.int64,
                     device=DEVICE)
    got = ref.clone()
    triton_rope(ref, prefill_positions, prefill_delta, idx_mapping, qsl,
                prefill_lens, num_computed, num_dims, max_model_len)
    hip_rope(got, prefill_positions, prefill_delta, idx_mapping, qsl,
             prefill_lens, num_computed, num_dims, max_model_len)
    sync()
    assert torch.equal(ref, got)


# ===========================================================================
# _apply_grammar_bitmask_kernel
# ===========================================================================
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab_size", [32, 33, 1024, 8192, 8193, 151936])
def test_apply_grammar_bitmask_parity(dtype, vocab_size):
    mask_stride = 4
    cu = [0, 3, 5, 9]
    mapping = [0, 1, 2, 4, 5, 3, 2 * mask_stride, 2 * mask_stride + 1,
               2 * mask_stride + 5]
    logits = rand((len(mapping), vocab_size), dtype, 5)
    bitmask = randint(
        0, 2**31 - 1, (len(mapping), cdiv(vocab_size, 32)), torch.int32, 6
    )
    logits_indices = torch.tensor(mapping, dtype=torch.int32, device=DEVICE)
    cu_num_logits = torch.tensor(cu, dtype=torch.int32, device=DEVICE)

    ref = logits.clone()
    got = logits.clone()
    triton_bitmask(ref, logits_indices, cu_num_logits, bitmask, mask_stride)
    hip_bitmask(got, logits_indices, cu_num_logits, bitmask, mask_stride)
    sync()
    assert torch.equal(ref, got)
    assert torch.isinf(ref).any()


def test_apply_grammar_bitmask_single_and_empty():
    logits = rand((1, 64), torch.float16, 6)
    cu = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    for mapping in ([], [0]):
        idx = torch.tensor(mapping, dtype=torch.int32, device=DEVICE)
        bm = torch.zeros((len(mapping), 2), dtype=torch.int32, device=DEVICE)
        ref, got = logits.clone(), logits.clone()
        triton_bitmask(ref, idx, cu, bm, 4)
        hip_bitmask(got, idx, cu, bm, 4)
        sync()
        assert torch.equal(ref, got)


# ===========================================================================
# The HIP path is taken (poison every Triton kernel)
# ===========================================================================
class _NoLaunch:
    def __getitem__(self, _grid):
        raise AssertionError("Triton kernel launched on the gfx1030 HIP path")

    def __call__(self, *args, **kwargs):
        raise AssertionError("Triton kernel launched on the gfx1030 HIP path")


@pytest.mark.skipif(not is_uva_available(), reason="UVA required")
def test_dispatch_does_not_fall_back_to_triton(monkeypatch):
    for mod, names in (
        (mu, ("postprocess_mamba_fused_kernel",
              "precopy_mamba_align_fused_kernel",
              "batch_memcpy_kernel")),
        (so, ("_apply_grammar_bitmask_kernel",)),
        (rope_mod, ("_prepare_rope_positions_kernel",)),
        (mh, ("_scatter_num_accepted_kernel", "_fill_num_accepted_kernel",
              "preprocess_mamba_align_fused_kernel")),
    ):
        for name in names:
            monkeypatch.setattr(mod, name, _NoLaunch())

    num_reqs = SERVE_NUM_REQS
    max_reqs = num_reqs + 4

    # --- preprocess + scatter/fill (gpu/model_states/mamba_hybrid.py)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    qsl[1:] = 3
    mh._preprocess_mamba_align_fused(
        idx_mapping,
        torch.zeros(max_reqs, dtype=torch.int32, device=DEVICE),
        torch.zeros(max_reqs, dtype=torch.int32, device=DEVICE),
        qsl,
        torch.ones(max_reqs, dtype=torch.int32, device=DEVICE),
        torch.full((max_reqs,), -1, dtype=torch.int32, device=DEVICE),
        torch.zeros(max_reqs, dtype=torch.int32, device=DEVICE),
        num_reqs,
        SERVE_BLOCK_SIZE,
    )
    mh._update_num_accepted_tokens(
        idx_mapping,
        torch.ones(num_reqs, dtype=torch.int32, device=DEVICE),
        torch.ones(max_reqs, dtype=torch.int32, device=DEVICE),
        num_reqs,
    )
    mh._update_num_accepted_tokens(
        idx_mapping,
        num_reqs * 2,  # int -> the fill path
        torch.ones(max_reqs, dtype=torch.int32, device=DEVICE),
        num_reqs,
    )

    # --- precopy / postprocess / batch_memcpy (v1/worker/mamba_utils.py)
    rig, _src_cols = make_rig(
        max_reqs=max_reqs, conv_dim=128, temporal_inner=2048, seed=3
    )
    shim = SimpleNamespace(
        is_initialized=True,
        num_states=len(rig.specs),
        block_size=4,
        num_accepted_tokens_out=torch.zeros(max_reqs, dtype=torch.int32,
                                            device=DEVICE),
        **rig.meta(),
    )
    dst_col = torch.zeros(max_reqs, dtype=torch.int32, device=DEVICE)
    src_col = torch.full((max_reqs,), -1, dtype=torch.int32, device=DEVICE)
    bias = torch.zeros(max_reqs, dtype=torch.int32, device=DEVICE)
    mu.MambaSpecDecodeGPUContext.run_fused_precopy(
        shim, num_reqs, dst_col, src_col, bias, idx_mapping
    )
    accepted = torch.ones(max_reqs, dtype=torch.int32, device=DEVICE)
    zeros = torch.zeros(max_reqs, dtype=torch.int32, device=DEVICE)
    decode_computed = torch.full((max_reqs,), 1025, dtype=torch.int32,
                                 device=DEVICE)
    mu.MambaSpecDecodeGPUContext.run_fused_postprocess_align(
        shim, num_reqs, accepted, dst_col, decode_computed, idx_mapping
    )
    mu.MambaSpecDecodeGPUContext.run_fused_postprocess(
        shim, num_reqs, accepted, dst_col, zeros, decode_computed, zeros
    )
    buf = torch.zeros(4096, dtype=torch.uint8, device=DEVICE)
    mu.batch_memcpy(
        _ptr_tensor([buf.data_ptr()]),
        _ptr_tensor([buf.data_ptr() + 64]),
        torch.tensor([1024], dtype=torch.int32, device=DEVICE),
    )

    # --- rope (gpu/mm/rope.py)
    state = rope_mod.RopeState(4, max_reqs, 2048, 4096, DEVICE)
    state.prepare_positions(idx_mapping, qsl, torch.ones(
        max_reqs, dtype=torch.int32, device=DEVICE), zeros)

    # --- grammar bitmask (gpu/structured_outputs.py)
    worker = so.StructuredOutputsWorker(16, 256, DEVICE, 4, 1)
    batch = SimpleNamespace(
        req_ids=["r0", "r1"],
        cu_num_logits_np=[0, 2, 4],
        num_draft_tokens_per_req=None,
        cu_num_logits=torch.tensor([0, 2, 4], dtype=torch.int32, device=DEVICE),
    )
    worker.apply_grammar_bitmask(
        rand((4, 256), torch.float16, 8),
        batch,
        ["r0", "r1"],
        torch.zeros((4, 8), dtype=torch.int32).numpy(),
    )
    sync()

# ===========================================================================
# Every op runs inside a FULL cudagraph capture: nothing may sync or allocate.
# ===========================================================================
@pytest.mark.skipif(not is_uva_available(), reason="UVA required")
def test_ops_are_cudagraph_capturable():
    n, mr = SERVE_NUM_REQS, SERVE_NUM_REQS + 4
    rig, src_col_full = make_rig(max_reqs=mr, conv_dim=128, temporal_inner=2048,
                                 seed=61)
    shim = SimpleNamespace(
        is_initialized=True,
        num_states=len(rig.specs),
        block_size=1024,
        num_accepted_tokens_out=torch.zeros(mr, dtype=torch.int32, device=DEVICE),
        **rig.meta(),
    )
    idx = torch.arange(n, dtype=torch.int32, device=DEVICE)
    qsl = torch.zeros(n + 1, dtype=torch.int32, device=DEVICE)
    qsl[1:] = 3
    state_idx = torch.zeros(mr, dtype=torch.int32, device=DEVICE)
    computed = torch.zeros(mr, dtype=torch.int32, device=DEVICE)
    accepted = torch.full((mr,), 3, dtype=torch.int32, device=DEVICE)
    src_col = torch.full((mr,), -1, dtype=torch.int32, device=DEVICE)
    src_off = torch.zeros(mr, dtype=torch.int32, device=DEVICE)
    zeros = torch.zeros(mr, dtype=torch.int32, device=DEVICE)
    src_col_full = src_col_full.to(DEVICE)
    copied = torch.zeros(mr, dtype=torch.int32, device=DEVICE)
    decode_computed = torch.full((mr,), 1025, dtype=torch.int32, device=DEVICE)

    buf = randint(0, 256, (4096,), torch.uint8, 63)
    sp = _ptr_tensor([buf.data_ptr()])
    dp = _ptr_tensor([buf.data_ptr() + 64])
    sz = torch.tensor([1024], dtype=torch.int32, device=DEVICE)

    positions = torch.zeros((4, 2049), dtype=torch.int64, device=DEVICE)
    pp = randint(-100, 100, (mr * 4, 4096), torch.int32, 65)
    pd = torch.zeros(mr, dtype=torch.int32, device=DEVICE)
    pl = torch.full((mr,), 4096, dtype=torch.int32, device=DEVICE)
    # The bitmask kernel only writes -inf, so zero-filled logits make the
    # eager and replayed results comparable element for element.
    logits = torch.zeros((4, 1024), dtype=torch.float16, device=DEVICE)
    li = torch.tensor([0, 1, 4, 5], dtype=torch.int32, device=DEVICE)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32, device=DEVICE)
    bm = randint(0, 2**31 - 1, (4, 32), torch.int32, 69)

    def launch_all():
        mh._preprocess_mamba_align_fused(
            idx, state_idx, computed, qsl, accepted, src_col, src_off, n, 1024
        )
        mh._update_num_accepted_tokens(idx, accepted, copied, n)
        mu.MambaSpecDecodeGPUContext.run_fused_precopy(
            shim, n, src_col, src_col_full, zeros, None
        )
        mu.MambaSpecDecodeGPUContext.run_fused_postprocess_align(
            shim, n, accepted, copied, decode_computed, idx
        )
        mu.MambaSpecDecodeGPUContext.run_fused_postprocess(
            shim, n, accepted, copied, zeros, decode_computed, zeros
        )
        mu.batch_memcpy(sp, dp, sz)
        hip_rope(positions, pp, pd, idx, qsl, pl, computed, 4, 4096)
        hip_bitmask(logits, li, cu, bm, 4)

    launch_all()
    sync()
    outputs = (state_idx, copied, positions, logits)
    eager = tuple(o.clone() for o in outputs)
    # Reset the outputs so a replayed graph has to rewrite them.
    for o in outputs:
        o.zero_()
    sync()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_all()
    graph.replay()
    sync()
    for a, b in zip(eager, outputs):
        assert torch.equal(a, b)
