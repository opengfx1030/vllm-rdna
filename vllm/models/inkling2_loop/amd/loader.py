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

ExpertFormat = Literal["nvfp4", "mxfp8", "hi2"]

#: ``quantization_config.quant_method`` of the quantizer's ``hi2`` output
HI2_QUANT_METHOD = "rdna_hint_flex"
HI2_SPEC_VERSIONS = ("3.2",)
#: the stacked expert groups of an HI2 Inkling directory and their part order
HI2_STACKS: dict[str, tuple[str, ...]] = {
    "gate_up": ("gate_proj", "up_proj"),
    "down": ("down_proj",),
}


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


def hi2_expert_formats(quant_config: dict, num_layers: int) -> list[ExpertFormat]:
    """All layers carry the quantizer's stacked HI2 experts.

    The ``quantization_config`` of the output's ``config.json`` names the method,
    the spec version and the stacked groups (``hi2/checkpoint.py`` of the
    quantizer).  A directory whose experts are per-expert HI2 tensors (the other
    ``--moe-layout``) is refused: the Inkling runtime reads the stacked form.
    """
    version = str(quant_config.get("version", ""))
    if version not in HI2_SPEC_VERSIONS:
        raise ValueError(
            f"HI2 quantization_config version {version!r} is not supported "
            f"(supported: {HI2_SPEC_VERSIONS})"
        )
    if str(quant_config.get("mode", "flex")) != "flex":
        raise ValueError(f"HI2 mode {quant_config.get('mode')!r} is not 'flex'")
    if str(quant_config.get("moe_layout", "")) != "stacked":
        raise ValueError(
            "the Inkling HI2 runtime needs moe_layout='stacked' "
            f"(got {quant_config.get('moe_layout')!r})"
        )
    stacks = quant_config.get("moe_stacks") or {}
    for group, parts in HI2_STACKS.items():
        got = tuple(stacks.get(group) or ())
        if got != parts:
            raise ValueError(f"HI2 moe_stacks[{group!r}] is {got}, expected {parts}")
    return ["hi2"] * num_layers


def read_expert_formats(
    model: str, revision: str | None, num_layers: int
) -> list[ExpertFormat]:
    """Expert storage of ``model``: HI2 from the ``quantization_config`` of
    ``config.json`` when the quantizer wrote it, else the legacy
    ``hf_quant_config.json`` of the NVFP4 / MXFP8 stock checkpoint."""
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    cfg = get_hf_file_to_dict("config.json", model, revision)
    quant_config = (cfg or {}).get("quantization_config")
    if (
        isinstance(quant_config, dict)
        and quant_config.get("quant_method") == HI2_QUANT_METHOD
    ):
        return hi2_expert_formats(quant_config, num_layers)

    cfg = get_hf_file_to_dict("hf_quant_config.json", model, revision)
    if cfg is None:
        raise FileNotFoundError(
            f"{model}: hf_quant_config.json is required to tell the NVFP4 and "
            "MXFP8 expert layers apart"
        )
    return expert_formats(cfg, num_layers)


# ---------------------------------------------------------------------------
# Hugging Face names of an HI2 output -> serving names
# ---------------------------------------------------------------------------

# The quantizer's exporter reads the stock directory through the transformers
# model and writes the dense text tensors under the Hugging Face names of the
# reference model (``model.language_model.*``); the serving checkpoint and this
# loader use ``model.llm.*``.  This is the inverse of the reference name map
# (``tools/rdna/inkling2/ref_checkpoint.py::build_name_map``); ``lm_head.weight``
# and ``model.visual.*`` are already identical in both.
_HF_TEXT_PREFIX = "model.language_model."
_SERVING_TEXT_PREFIX = "model.llm"

_HF_LAYER_FIELDS: tuple[tuple[str, str], ...] = (
    ("input_layernorm.weight", "attn_norm.weight"),
    ("q_input_layernorm.weight", "q_input_norm.weight"),
    ("k_input_layernorm.weight", "k_input_norm.weight"),
    ("v_input_layernorm.weight", "v_input_norm.weight"),
    ("post_attention_layernorm.weight", "mlp_norm.weight"),
    ("post_attention_convolution.conv1d.weight", "attn_sconv.weight"),
    ("post_mlp_convolution.conv1d.weight", "mlp_sconv.weight"),
    ("self_attn.q_proj.linear.weight", "attn.wq_du.weight"),
    ("self_attn.q_proj.norm.scale", "attn.q_l2_clip.scale"),
    ("self_attn.k_proj.linear.weight", "attn.wk_dv.weight"),
    ("self_attn.v_proj.linear.weight", "attn.wv_dv.weight"),
    ("self_attn.o_proj.linear.weight", "attn.wo_ud.weight"),
    ("self_attn.o_proj.gate_proj.weight", "attn.gate_attn.weight"),
    ("self_attn.sink_proj.weight", "attn.attn_sink_proj.weight"),
    ("self_attn.relative_position_bias.linear.weight", "attn.wr_du.weight"),
    ("self_attn.relative_position_bias.norm.scale", "attn.rel_l2_clip.scale"),
    ("self_attn.relative_position_bias.proj", "attn.rel_logits_proj.proj"),
    ("self_attn.k_proj.sconv.conv1d.weight", "attn.k_sconv.weight"),
    ("self_attn.v_proj.sconv.conv1d.weight", "attn.v_sconv.weight"),
    ("self_attn.k_proj.sconv.static_weight", "attn.k_sconv.static_w"),
    ("self_attn.v_proj.sconv.static_weight", "attn.v_sconv.static_w"),
    ("self_attn.k_proj.sconv.weight_proj.weight", "attn.k_sconv.weight_proj"),
    ("self_attn.v_proj.sconv.weight_proj.weight", "attn.v_sconv.weight_proj"),
    ("mlp.gate.e_score_correction_bias", "mlp.gate.bias"),
    ("mlp.gate.weight", "mlp.gate.weight"),
)
_HF_GATES: dict[str, str] = {
    "q_message_gate": "attn",
    "k_message_gate": "k",
    "v_message_gate": "v",
    "mlp_message_gate": "mlp",
}
_HF_GATE_PARTS: dict[str, str] = {
    "down_proj": "down",
    "forget_gate_proj": "forget",
    "message_gate_proj": "gain",
}
_HF_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
_HF_GATE_RE = re.compile(r"^(\w+_message_gate)\.(\w+)\.weight$")


def hf_to_serving(name: str) -> str | None:
    """Serving name of a dense Hugging Face text tensor, else ``None``.

    Expert groups are not handled here: the loader collects them per layer
    (``hi2.parse_tensor_name``).
    """
    if not name.startswith(_HF_TEXT_PREFIX):
        return None
    rest = name[len(_HF_TEXT_PREFIX) :]
    if rest == "embed_tokens.embed_norm.weight":
        return f"{_SERVING_TEXT_PREFIX}.embed_norm.weight"
    if rest == "embed_tokens.weight":
        return f"{_SERVING_TEXT_PREFIX}.embed.weight"
    if rest == "norm.weight":
        return f"{_SERVING_TEXT_PREFIX}.norm.weight"
    if rest == "final_input.query":
        return f"{_SERVING_TEXT_PREFIX}.attn_res.output_query"
    if rest == "message_input.query":
        return f"{_SERVING_TEXT_PREFIX}.message_input.query"
    m = _HF_LAYER_RE.match(rest)
    if m is None:
        return None
    layer, tail = int(m.group(1)), m.group(2)
    if tail == "attention_input.query":
        if layer == 0:
            return None
        return f"{_SERVING_TEXT_PREFIX}.attn_res.attn_queries.{layer - 1}"
    if tail == "mlp_input.query":
        return f"{_SERVING_TEXT_PREFIX}.attn_res.mlp_queries.{layer}"
    for src, dst in _HF_LAYER_FIELDS:
        if tail == src:
            return f"{_SERVING_TEXT_PREFIX}.layers.{layer}.{dst}"
    m = _HF_GATE_RE.match(tail)
    if m is not None and m.group(1) in _HF_GATES:
        part = _HF_GATE_PARTS.get(m.group(2))
        if part is not None:
            ctl = _HF_GATES[m.group(1)]
            return (
                f"{_SERVING_TEXT_PREFIX}.layers.{layer}.controllers.{ctl}.{part}_weight"
            )
    return None


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
