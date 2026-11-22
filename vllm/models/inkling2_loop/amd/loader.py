# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert storage format and per-rank slicing of an Inkling-2-Mini checkpoint.

The name mapping and the tensor copies live in ``model.py``
(``Inkling2LoopForConditionalGeneration.load_weights``); the only thing that has to
be known before the modules are built is which layers carry NVFP4 and which MXFP8
routed experts, because the two formats have different parameter shapes.

Tensor parallelism: every rank reads the whole checkpoint and keeps its slice
(:func:`local_slice`).  Each rule is an even, contiguous split of one dimension in
the checkpoint's own (head-major / unit-major) layout, so the slice is a view and a
rank never materialises another rank's share.  The one non-obvious rule is the fused
``w13`` of the experts, whose rows are stored interleaved (``[g0, u0, g1, u1, ...]``):
the rows of intermediate units ``[lo, hi)`` are the contiguous rows ``[2 * lo,
2 * hi)``, which the de-interleaving copy of ``model.py`` then turns into the rank's
``[gate; up]`` layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import regex as re

if TYPE_CHECKING:
    import torch

    from .parallel import TensorParallel

_EXPERT_W13_RE = re.compile(r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.w13_weight$")

ExpertFormat = Literal["nvfp4", "mxfp8"]


def expert_formats(quant_config: dict, num_layers: int) -> list[ExpertFormat]:
    """Per-layer expert storage (``"nvfp4"`` / ``"mxfp8"``) from the
    ``model.parameter_groups`` of the checkpoint's ``hf_quant_config.json``.

    A group's ``compute_dtype`` names the format of every ``layers.N.mlp.experts
    .w13_weight`` it lists (the real checkpoint is NVFP4 for layers 0-19 and MXFP8
    for 20-23, the tiny checkpoints put MXFP8 on the last layer only).
    """
    formats: dict[int, ExpertFormat] = {}
    for group in quant_config["model"]["parameter_groups"]:
        fmt = group["compute_dtype"]
        if fmt not in ("nvfp4", "mxfp8"):
            continue
        for name in group["tensors"]:
            m = _EXPERT_W13_RE.match(name)
            if m:
                formats[int(m.group(1))] = fmt
    missing = [i for i in range(num_layers) if i not in formats]
    if missing:
        raise ValueError(
            f"hf_quant_config.json lists no nvfp4/mxfp8 experts for layers {missing}"
        )
    return [formats[i] for i in range(num_layers)]


def read_expert_formats(
    model: str, revision: str | None, num_layers: int
) -> list[ExpertFormat]:
    """:func:`expert_formats` of the ``hf_quant_config.json`` next to ``model``."""
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    cfg = get_hf_file_to_dict("hf_quant_config.json", model, revision)
    if cfg is None:
        raise FileNotFoundError(
            f"{model}: hf_quant_config.json is required to tell the NVFP4 and "
            "MXFP8 expert layers apart"
        )
    return expert_formats(cfg, num_layers)


# ---------------------------------------------------------------------------
# tensor-parallel slicing
# ---------------------------------------------------------------------------

_LAYER_PARAM_RE = re.compile(r"^model\.layers\.\d+\.(.+)$")

#: dimension of the checkpoint tensor that follows the attention heads, by the name of
#: the destination parameter inside ``model.layers.N.``; everything not listed here
#: (norms, residual queries, message gates, the router, the post-attention / post-MLP
#: convolutions, the relative-position basis ``relative_position_bias.proj`` shared by
#: the heads) is replicated.
ATTENTION_SPLIT_DIM: dict[str, int] = {
    "self_attn.q_proj.linear.weight": 0,  # [nh * D, H]
    "self_attn.q_proj.norm.scale": 0,  # [nh, D]
    "self_attn.o_proj.gate_proj.weight": 0,  # [nh * D, H]
    "self_attn.o_proj.linear.weight": 1,  # [H, nh * D]  (row-parallel)
    "self_attn.sink_proj.weight": 0,  # [nh, H]
    "self_attn.relative_position_bias.linear.weight": 0,  # [nh * d_rel, H]
    "self_attn.relative_position_bias.norm.scale": 0,  # [nh, d_rel]
    "self_attn.k_proj.linear.weight": 0,  # [nkv * D, H]
    "self_attn.v_proj.linear.weight": 0,
    "self_attn.k_proj.sconv.conv1d.weight": 0,  # [nkv * D, 1, k]  (sliding layers)
    "self_attn.v_proj.sconv.conv1d.weight": 0,
    "self_attn.k_proj.sconv.static_weight": 1,  # [k, nkv * D]  (full layers)
    "self_attn.v_proj.sconv.static_weight": 1,
    "self_attn.k_proj.sconv.weight_proj.weight": 0,  # [nkv * k, H]  (full layers)
    "self_attn.v_proj.sconv.weight_proj.weight": 0,
}

#: expert tensors whose last dimension is the intermediate size (packed bytes or scale
#: blocks of it): ``w2`` is ``[E, H, I...]``
_EXPERT_W2 = ("mlp.w2_weight", "mlp.w2_weight_scale")
#: fused gate / up tensors ``[E, 2 I, ...]`` with interleaved rows
_EXPERT_W13 = ("mlp.w13_weight", "mlp.w13_weight_scale")


def local_slice(dst: str, tensor: torch.Tensor, tp: TensorParallel) -> torch.Tensor:
    """The part of checkpoint ``tensor`` (destination parameter ``dst``) that rank
    ``tp.rank`` keeps; a view of ``tensor`` (the tensor itself when replicated).

    The caller checks the result against the shape of the rank's parameter, so a rule
    that does not match the model's own sharding fails loudly at load time.
    """
    if not tp.enabled:
        return tensor
    m = _LAYER_PARAM_RE.match(dst)
    if m is None:
        return tensor  # embedding, norms, final queries: replicated
    tail = m.group(1)
    dim = ATTENTION_SPLIT_DIM.get(tail)
    if dim is not None:
        return tp.narrow(tensor, dim, dst)
    if tail in _EXPERT_W13:
        # [E, 2I, ...] interleaved rows: units [lo, hi) are rows [2 lo, 2 hi)
        units = tensor.shape[1] // 2
        lo, hi = tp.span(units, dst)
        return tensor.narrow(1, 2 * lo, 2 * (hi - lo))
    if tail in _EXPERT_W2:
        return tp.narrow(tensor, 2, dst)
    return tensor  # router, norms, message gates, per-expert amax / denominator
