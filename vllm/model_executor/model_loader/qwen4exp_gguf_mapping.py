# SPDX-License-Identifier: Apache-2.0
"""Fail-closed Qwen4Exp GGUF logical parameter and slice planning.

Validate names and shapes against the supplied native parameter catalogue.
Mapped entries are not loaded weights, packed GPU execution or serving admission.
"""

import re
from dataclasses import dataclass


class MappingError(ValueError):
    pass


@dataclass(frozen=True)
class Entry:
    source: str
    shape: tuple[int, ...]
    ggml_type: int
    destination: str | None
    transform: str
    reason: str
    destination_shape: tuple[int, ...] | None = None
    slice_axis: int | None = None
    slice_start: int = 0
    slice_size: int | None = None
    delivery: str = "dense"


@dataclass(frozen=True)
class Plan:
    entries: tuple[Entry, ...]

    @property
    def by_source(self):
        return {e.source: e for e in self.entries}

    @property
    def unsupported(self):
        return tuple(e for e in self.entries if e.destination is None)


def inverse_zero_centered_norm(weight):
    """Invert converter gamma+1 for native zero-centered norm parameters."""
    return weight - 1


_HEAD_DESTINATIONS = {
    "output.weight": "lm_head.weight",
    "token_embd.weight": "model.embed_tokens.weight",
    "output_hc_down.weight": (
        "model.hyper_connection_mixer.input_mix_weight_down.weight"
    ),
    "output_hc_up.weight": "model.hyper_connection_mixer.input_mix_weight_up.weight",
    "output_hc_norm.weight": "model.hyper_connection_mixer.hc_norm.weight",
}


_DIRECT_LAYER = {
    "ffn_gate_inp.weight": ("mlp.gate.weight", "identity"),
    "ffn_gate_inp_shexp.weight": ("mlp.shared_expert_gate.weight", "reshape"),
    "ffn_down_shexp.weight": ("mlp.shared_expert.down_proj.weight", "identity"),
    "hc_attn_norm.weight": ("attn_hyper_connection.hc_norm.weight", "subtract_one"),
    "hc_ffn_norm.weight": ("mlp_hyper_connection.hc_norm.weight", "subtract_one"),
    "hc_attn_up.weight": (
        "attn_hyper_connection.input_mix_weight_up.weight",
        "identity",
    ),
    "hc_ffn_up.weight": ("mlp_hyper_connection.input_mix_weight_up.weight", "identity"),
    "attn_output.weight": ("self_attn.o_proj.weight", "identity"),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", "subtract_one"),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", "subtract_one"),
    "indexer.q_norm.weight": ("self_attn.indexer.q_layernorm.weight", "subtract_one"),
    "indexer.k_norm.weight": ("self_attn.indexer.k_layernorm.weight", "subtract_one"),
    "ple_norm_key.weight": ("ple.norm_key.weight", "subtract_one"),
    "ple_norm_query.weight": ("ple.norm_query.weight", "subtract_one"),
    "ple_norm_conv.weight": ("ple.norm_conv.weight", "subtract_one"),
    "ple_conv1d.weight": ("ple.conv1d.weight", "unsqueeze_conv"),
    "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "gdn_conv_inverse"),
    "ssm_dt.bias": ("linear_attn.dt_bias", "gdn_inverse"),
    "ssm_a": ("linear_attn.A_log", "gdn_inverse"),
    "ssm_norm.weight": ("linear_attn.norm.weight", "identity"),
    "ssm_out.weight": ("linear_attn.out_proj.weight", "gdn_inverse"),
}


# Logical dense shapes verified against the exact native meta construction.
# These declarations do not materialize expert/PLE tensors as dense weights.
_MERGED_LAYER = {
    "hc_attn_down.weight": (
        "attn_hyper_connection.input_mix_weight_down_block_inject.weight",
        0,
        0,
        320,
    ),
    "hc_attn_inject.weight": (
        "attn_hyper_connection.input_mix_weight_down_block_inject.weight",
        0,
        320,
        4,
    ),
    "hc_ffn_down.weight": (
        "mlp_hyper_connection.input_mix_weight_down_block_inject.weight",
        0,
        0,
        320,
    ),
    "hc_ffn_inject.weight": (
        "mlp_hyper_connection.input_mix_weight_down_block_inject.weight",
        0,
        320,
        4,
    ),
    "ffn_gate_shexp.weight": ("mlp.shared_expert.gate_up_proj.weight", 0, 0, 640),
    "ffn_up_shexp.weight": ("mlp.shared_expert.gate_up_proj.weight", 0, 640, 640),
    "ffn_gate_exps.weight": ("mlp.experts.routed_experts.w13_weight", 1, 0, 640),
    "ffn_up_exps.weight": ("mlp.experts.routed_experts.w13_weight", 1, 640, 640),
    "attn_q.weight": ("self_attn.qkv_proj.weight", 0, 0, 12288),
    "attn_k.weight": ("self_attn.qkv_proj.weight", 0, 12288, 512),
    "attn_v.weight": ("self_attn.qkv_proj.weight", 0, 12800, 512),
    "indexer.q_proj.weight": ("self_attn.indexer.index_qk_proj.weight", 0, 0, 512),
    "indexer.k_proj.weight": ("self_attn.indexer.index_qk_proj.weight", 0, 512, 128),
    "ple_key.weight": ("ple.kv_proj.weight", 0, 0, 10240),
    "ple_value.weight": ("ple.kv_proj.weight", 0, 10240, 2560),
    "attn_qkv.weight": ("linear_attn.in_proj_qkvz.weight", 0, 0, 10240),
    "attn_gate.weight": ("linear_attn.in_proj_qkvz.weight", 0, 10240, 6144),
    "ssm_beta.weight": ("linear_attn.in_proj_ba.weight", 0, 0, 48),
    "ssm_alpha.weight": ("linear_attn.in_proj_ba.weight", 0, 48, 48),
}


def plan_tensors(tensors, *, native_parameters=None):
    """Plan every header tensor or raise; unknowns are never discarded."""
    entries = []
    seen = set()
    for tensor in tensors:
        try:
            name = tensor["name"]
            dims = tuple(tensor["dims"])
            typ = tensor["type"]
        except (KeyError, TypeError, ValueError) as exc:
            raise MappingError(
                f"invalid tensor header contract: {tensor!r}: {exc}"
            ) from exc
        if (
            not isinstance(name, str)
            or not name
            or not dims
            or any(
                not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in dims
            )
            or not isinstance(typ, int)
            or isinstance(typ, bool)
        ):
            raise MappingError(f"invalid tensor name/shape contract: {tensor!r}")
        if name in seen:
            raise MappingError(f"duplicate source tensor {name}")
        seen.add(name)
        destination = _HEAD_DESTINATIONS.get(name)
        transform = "subtract_one" if name == "output_hc_norm.weight" else "identity"
        axis, start, count = None, 0, None
        delivery = "dense"
        if name == "per_layer_token_embd.weight":
            destination = "model.layers.1.ple.ple_embedding.ngram_embedding.weight"
            delivery = "packed_ple"
        match = re.fullmatch(r"blk\.(0|[1-9][0-9]*)\.(.+)", name)
        if match and match[2] in _DIRECT_LAYER:
            suffix, transform = _DIRECT_LAYER[match[2]]
            destination = f"model.layers.{match[1]}.{suffix}"
        if match and match[2] in _MERGED_LAYER:
            suffix, axis, start, count = _MERGED_LAYER[match[2]]
            destination = f"model.layers.{match[1]}.{suffix}"
            if match[2] in (
                "attn_qkv.weight",
                "attn_gate.weight",
                "ssm_beta.weight",
                "ssm_alpha.weight",
            ):
                transform = "gdn_inverse"
        if match and match[2] in (
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ):
            delivery = "packed_expert"
            if match[2] == "ffn_down_exps.weight":
                destination = (
                    f"model.layers.{match[1]}.mlp.experts.routed_experts.w2_weight"
                )
        if native_parameters is not None and destination is not None:
            if destination not in native_parameters:
                raise MappingError(f"native parameter missing: {destination}")
            shape = tuple(native_parameters[destination]["shape"])
            logical = tuple(reversed(dims))
            if transform == "reshape":
                logical = (1,) + logical
            elif transform in ("unsqueeze_conv", "gdn_conv_inverse"):
                logical = (logical[0], 1, logical[1])
            expected = list(shape)
            if axis is not None:
                if (
                    count is None
                    or start < 0
                    or count <= 0
                    or start + count > shape[axis]
                ):
                    raise MappingError(f"invalid native shard range: {name}")
                expected[axis] = count
            if logical != tuple(expected):
                raise MappingError(
                    f"native parameter shape mismatch: {name} -> {destination}"
                )
            allowed = (
                (7, 8, 12, 13)
                if delivery == "packed_expert"
                else (20,)
                if delivery == "packed_ple"
                else (0, 8, 30)
            )
            if typ not in allowed:
                raise MappingError(f"unsupported {delivery} tensor type: {typ}")
            entries.append(
                Entry(
                    name,
                    dims,
                    typ,
                    destination,
                    transform,
                    "",
                    shape,
                    axis,
                    start,
                    count,
                    delivery,
                )
            )
        else:
            entries.append(
                Entry(
                    name,
                    dims,
                    typ,
                    None,
                    "none",
                    "native destination/shape contract not yet verified",
                )
            )
    if not entries and tensors:
        raise MappingError("no tensors planned")
    return Plan(tuple(entries))
