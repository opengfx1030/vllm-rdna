# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the plumbing around the HIP serving path of Inkling-2.

``test_serving_path_gpu.py`` runs the kernels.  This file pins what needs no GPU: the
guard that refuses shapes the native kernels were not written for, the selection of the
path, the bound of the split-K workspace over every batch shape the split policy can
produce, the aliasing of the persistent workspace views and the launch count of the
same-step prefix-cache fix.

    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        PYTHONPATH=$TREE:$TFM_MAIN pytest \
        tests/models/inkling2_loop/test_serving_path_cpu.py -q
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.models.inkling2_loop.amd.glue")
pytest.importorskip("transformers")

from paged_harness import text_config  # noqa: E402

from vllm import _custom_ops as ops  # noqa: E402
from vllm.models.inkling2_loop.amd import attention as attention_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import glue as glue_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import state as state_mod  # noqa: E402
from vllm.models.inkling2_loop.configs import Inkling2LoopTextConfig  # noqa: E402

# attention shapes of the native kernels (head_dim 128, d_rel 16, GQA group 3)
HIP_SHAPE = {"head_dim": 128, "d_rel": 16, "num_heads": 6, "num_kv_heads": 2}


def test_inkling2_mini_shapes_are_supported():
    glue_mod.check_kernel_shapes(Inkling2LoopTextConfig())  # the real model's defaults
    glue_mod.check_kernel_shapes(text_config(**HIP_SHAPE))


def test_the_default_tiny_checkpoint_is_refused_with_every_reason():
    with pytest.raises(ValueError, match="VLLM_INKLING2_PATH=reference") as err:
        glue_mod.check_kernel_shapes(text_config())  # head_dim 32, d_rel 8, 4 / 2 heads
    for needle in ("head_dim=32", "swa_head_dim=32", "d_rel=8"):
        assert needle in str(err.value)


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"head_dim": 64}, "head_dim=64"),
        ({"d_rel": 8}, "d_rel=8"),
        ({"conv_kernel": 3}, "sconv_kernel_size=3"),
        ({"num_heads": 10, "num_kv_heads": 2}, "full heads 10/2"),  # group 5
        ({"num_heads": 9, "num_kv_heads": 3}, "full heads 9/3"),  # odd q heads
        ({"hidden_size": 4128}, "hidden_size=4128"),
        ({"loop_gate_rank": 15}, "loop_gate_rank=15"),
        ({"n_routed_experts": 256}, "256 experts"),
        ({"num_experts_per_tok": 17}, "top-17"),
    ],
)
def test_each_unsupported_shape_is_named(overrides, needle):
    with pytest.raises(ValueError, match="does not support this config") as err:
        glue_mod.check_kernel_shapes(text_config(**{**HIP_SHAPE, **overrides}))
    assert needle in str(err.value)


def test_a_full_attention_layer_zero_is_refused():
    cfg = Inkling2LoopTextConfig(local_layer_ids=[1, 2, 3, 4])
    assert not cfg.is_sliding_layer(0)
    with pytest.raises(ValueError, match="does not support this config") as err:
        glue_mod.check_kernel_shapes(cfg)
    assert "layer 0 is a full-attention layer" in str(err.value)


def test_path_selection(monkeypatch):
    env = state_mod.PATH_ENV
    monkeypatch.delenv(env, raising=False)
    assert not state_mod.use_hip_ops("cpu")  # the CPU harness runs the reference
    assert state_mod.use_hip_ops("cuda")  # serving on a GPU runs the kernels
    monkeypatch.setenv(env, "reference")
    assert not state_mod.use_hip_ops("cuda")
    monkeypatch.setenv(env, "hip")
    assert state_mod.use_hip_ops("cuda")
    with pytest.raises(RuntimeError, match="needs a ROCm device"):
        state_mod.use_hip_ops("cpu")  # no silent fallback to torch
    monkeypatch.setenv(env, "triton")
    with pytest.raises(ValueError, match="must be 'hip' or 'reference'"):
        state_mod.use_hip_ops("cuda")


def test_conv_ops_write_the_state_themselves_only_for_one_token_per_request():
    assert attention_mod.conv_writes_in_op(SimpleNamespace(max_query_len=1))
    for max_q in (2, 3, 16, 4096):  # a chunk: ``inkling2_conv_state_write`` after
        assert not attention_mod.conv_writes_in_op(SimpleNamespace(max_query_len=max_q))


# -- the split-K workspace bound ----------------------------------------------------
_REQS = (*range(1, 40), 47, 63, 64, 100, 127, 128, 200, 255, 256, 257, 512, 1024)
_WINDOWS = (0, 1, 16, 64, 1024, 5000)
_MAX_SEQ = (1, 16, 17, 100, 1024, 4096, 1 << 17, 1 << 20)
_MAX_Q = (1, 2, 15, 16, 17, 31, 32, 33, 64, 100, 128, 256, 512, 1024, 8192)


def _rows_used(nkv: int, reqs: int, window: int, max_seq: int, max_q: int) -> int:
    """Split-K partial rows the kernels use in the worst batch of that shape: every
    request decodes (``splits_decode`` rows each) or carries ``max_q`` tokens
    (``splits_tile`` rows per token); a split factor of 1 needs no rows."""
    sd, st = ops.inkling2_attention_splits(reqs, nkv, window, max_seq, max_q)
    tokens = reqs * max_q
    return (sd * reqs if sd > 1 else 0) + (st * tokens if st > 1 and max_q > 1 else 0)


@pytest.mark.parametrize("group", [1, 2, 3, 4])
@pytest.mark.parametrize("nkv", [1, 2, 3, 8, 16])
def test_attention_workspace_bound_covers_every_split_configuration(nkv, group):
    nh = nkv * group
    if nh % 2:
        pytest.skip("odd q head counts are refused by check_kernel_shapes")
    bound_rows = glue_mod.attention_workspace_numel(nh, nkv) // (nh * 130)
    worst = 0
    for reqs, window, max_seq, max_q in itertools.product(
        _REQS, _WINDOWS, _MAX_SEQ, _MAX_Q
    ):
        worst = max(worst, _rows_used(nkv, reqs, window, max_seq, max_q))
    assert worst <= bound_rows, (worst, bound_rows)
    # not a wildly loose bound either (the real model's is 8 MB of fp32)
    assert bound_rows <= 1.1 * worst + 64, (worst, bound_rows)


def test_attention_workspace_of_the_real_model_is_small():
    numel = glue_mod.attention_workspace_numel(24, 8)
    assert numel * 4 < 9 * 2**20


# -- persistent workspace --------------------------------------------------------------
def _extent(t: torch.Tensor) -> tuple[int, int]:
    lo = t.data_ptr()
    last = sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
    return lo, lo + (last + 1) * t.element_size()


_ALIAS_GROUPS = [
    {"x16", "hq", "hk", "hv", "x16_g1", "norm3", "norm1"},
    {"msgs", "msgs_qkv", "msgs_mlp"},
    {"raw", "raw3", "raw1"},
    {"d", "d1"},
]


def test_workspace_views_alias_only_what_they_should():
    cfg = text_config(
        **HIP_SHAPE,
        num_hidden_layers=8,
        loop_start=2,
        loop_end=6,
        loop_num_passes=3,
    )
    glue_mod.check_kernel_shapes(cfg)
    ws = glue_mod.Inkling2GlueWorkspace(cfg, 48, "cpu")
    v = ws.bind(37)
    views = {k: t for k, t in vars(v).items() if isinstance(t, torch.Tensor)}
    group_of = {n: i for i, g in enumerate(_ALIAS_GROUPS) for n in g}
    names = sorted(views)
    for a, b in itertools.combinations(names, 2):
        if a in group_of and group_of.get(b) == group_of[a]:
            continue
        (lo_a, hi_a), (lo_b, hi_b) = _extent(views[a]), _extent(views[b])
        assert hi_a <= lo_b or hi_b <= lo_a, f"{a} and {b} overlap"
    # the three layernorm outputs are three different rows of one buffer
    assert views["hq"].data_ptr() < views["hk"].data_ptr() < views["hv"].data_ptr()
    for name in [
        "hid",
        "res",
        "attn_c",
        "mlp_c",
        "moe_out",
        "o_out",
        "hq",
        "hk",
        "hv",
        "x16",
        "raw",
        "msgs",
        "d",
        "g_scratch",
        "g_part",
    ]:
        assert views[name].is_contiguous(), name
    for name, t in views.items():
        assert t.data_ptr() % 16 == 0, f"{name} is not 16-byte aligned"
    assert tuple(views["blk"].shape) == (37, ws.S, cfg.hidden_size)
    with pytest.raises(ValueError, match="exceed"):
        ws.bind(49)


def test_workspace_moe_scratch_is_per_chunk():
    cfg = text_config(**HIP_SHAPE)
    ws = glue_mod.Inkling2GlueWorkspace(cfg, 4096, "cpu")
    assert ws.moe_chunk == glue_mod.MAX_MOE_TOKENS
    views = ws.moe_views(glue_mod.MAX_MOE_TOKENS, 8)
    n_pad, n_blk, _ = ops.inkling2_glue_moe_route_sizes(
        glue_mod.MAX_MOE_TOKENS, ws.E, ws.K, 8
    )
    assert views.sorted_ids.numel() == n_pad and views.expert_ids.numel() == n_blk
    assert views.gu.shape == (glue_mod.MAX_MOE_TOKENS * ws.K, 2 * ws.I)
    assert ws.moe_views(glue_mod.MAX_MOE_TOKENS, 8) is views  # memoised
    # a step of 4096 tokens is routed as four chunks of the same scratch
    assert ws.f32["logits"].numel() == glue_mod.MAX_MOE_TOKENS * ws.E
