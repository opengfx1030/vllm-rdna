# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native vLLM loader for mixed-quant Qwen4Exp GGUF shard sets.

Routes:
- Dense projections, embeddings, routers, and norms: bounded row-chunk GGUF
  decode -> exact converter inverses (zero-centered norm -1, GDN V-head/A_log
  inverses, conv1d unsqueeze, indexer Q/K merge) -> native ``load_weights``
  into FP16/FP32 parameters so existing RDNA2 HIP kernels execute unchanged.
- Routed MoE experts: kept packed in GGML blocks (`Q4_K`/`Q5_K`/`Q5_1`/`Q8_0`)
  and loaded into ``GGUFMoEMethod`` parameters with exact TP sharding.
- Per-layer token embeddings (PLE): kept packed as `IQ4_NL` in rank-local
  pinned host RAM and decoded on demand via UVA.

Fails closed on unmapped/missing tensors, mismatched gate/up quant types,
conflicting PLE env vars, or unconfigured MTP/vision sidecar requests.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import regex as re
import torch
from torch import nn

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.gguf_gdn_transforms import inverse_gdn_tensor
from vllm.model_executor.model_loader.gguf_manifest import (
    Manifest,
    TensorInfo,
    validate_shards,
)
from vllm.model_executor.model_loader.gguf_packed import dequantize as dequantize_packed
from vllm.model_executor.model_loader.gguf_plugin_compat import (
    get_loader_moe_method_cls,
    store_gguf_weight_type,
)
from vllm.model_executor.model_loader.qwen4exp_gguf_mapping import (
    inverse_zero_centered_norm,
    plan_tensors,
)
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)

_FORMAT_BY_GGML_TYPE = {
    0: "F32",
    7: "Q5_1",
    8: "Q8_0",
    12: "Q4_K",
    13: "Q5_K",
    20: "IQ4_NL",
    30: "BF16",
}
_BLOCK_GEOMETRY = {
    0: (1, 4),
    7: (32, 24),
    8: (32, 34),
    12: (256, 144),
    13: (256, 176),
    20: (32, 18),
    30: (1, 2),
}
_DENSE_CHUNK_FLOAT32_BUDGET = 256 << 20  # 256 MiB bounded float32 decode window

_HEAD_CHECKPOINT_NAMES = {
    "output.weight": ("lm_head.weight", "identity"),
    "token_embd.weight": ("model.embed_tokens.weight", "identity"),
    "output_hc_down.weight": (
        "model.hyper_connection_mixer.input_mix_weight_down.weight",
        "identity",
    ),
    "output_hc_up.weight": (
        "model.hyper_connection_mixer.input_mix_weight_up.weight",
        "identity",
    ),
    "output_hc_norm.weight": (
        "model.hyper_connection_mixer.hc_norm.weight",
        "subtract_one",
    ),
}

_LAYER_CHECKPOINT_NAMES = {
    "ffn_gate_inp.weight": ("mlp.gate.weight", "identity"),
    "ffn_gate_inp_shexp.weight": ("mlp.shared_expert_gate.weight", "reshape"),
    "ffn_gate_shexp.weight": ("mlp.shared_expert.gate_proj.weight", "identity"),
    "ffn_up_shexp.weight": ("mlp.shared_expert.up_proj.weight", "identity"),
    "ffn_down_shexp.weight": ("mlp.shared_expert.down_proj.weight", "identity"),
    "hc_attn_norm.weight": ("attn_hyper_connection.hc_norm.weight", "subtract_one"),
    "hc_ffn_norm.weight": ("mlp_hyper_connection.hc_norm.weight", "subtract_one"),
    "hc_attn_up.weight": (
        "attn_hyper_connection.input_mix_weight_up.weight",
        "identity",
    ),
    "hc_ffn_up.weight": ("mlp_hyper_connection.input_mix_weight_up.weight", "identity"),
    "hc_attn_down.weight": (
        "attn_hyper_connection.input_mix_weight_down.weight",
        "identity",
    ),
    "hc_attn_inject.weight": (
        "attn_hyper_connection.block_inject_weight.weight",
        "identity",
    ),
    "hc_ffn_down.weight": (
        "mlp_hyper_connection.input_mix_weight_down.weight",
        "identity",
    ),
    "hc_ffn_inject.weight": (
        "mlp_hyper_connection.block_inject_weight.weight",
        "identity",
    ),
    "attn_q.weight": ("self_attn.q_proj.weight", "identity"),
    "attn_k.weight": ("self_attn.k_proj.weight", "identity"),
    "attn_v.weight": ("self_attn.v_proj.weight", "identity"),
    "attn_output.weight": ("self_attn.o_proj.weight", "identity"),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", "subtract_one"),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", "subtract_one"),
    "indexer.q_norm.weight": ("self_attn.indexer.q_layernorm.weight", "subtract_one"),
    "indexer.k_norm.weight": ("self_attn.indexer.k_layernorm.weight", "subtract_one"),
    "ple_norm_key.weight": ("ple.norm_key.weight", "subtract_one"),
    "ple_norm_query.weight": ("ple.norm_query.weight", "subtract_one"),
    "ple_norm_conv.weight": ("ple.norm_conv.weight", "subtract_one"),
    "ple_conv1d.weight": ("ple.conv1d.weight", "unsqueeze_conv"),
    "ple_key.weight": ("ple.key_proj.weight", "identity"),
    "ple_value.weight": ("ple.value_proj.weight", "identity"),
    "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", "gdn_inverse"),
    "attn_gate.weight": ("linear_attn.in_proj_z.weight", "gdn_inverse"),
    "ssm_beta.weight": ("linear_attn.in_proj_b.weight", "gdn_inverse"),
    "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", "gdn_inverse"),
    "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "gdn_inverse"),
    "ssm_dt.bias": ("linear_attn.dt_bias", "gdn_inverse"),
    "ssm_a": ("linear_attn.A_log", "gdn_inverse"),
    "ssm_norm.weight": ("linear_attn.norm.weight", "identity"),
    "ssm_out.weight": ("linear_attn.out_proj.weight", "gdn_inverse"),
}


class Qwen4ExpGGUFQuantConfig(QuantizationConfig):
    """Quantization config keeping dense weights native FP16 and experts/PLE packed."""

    def __init__(self, *, gguf_ple_payload: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.gguf_ple_payload = gguf_ple_payload

    def __repr__(self) -> str:
        return "Qwen4ExpGGUFQuantConfig()"

    def get_name(self) -> str:  # type: ignore[override]
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Qwen4ExpGGUFQuantConfig:
        del config
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        del prefix
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
            VocabParallelEmbedding,
        )

        if isinstance(layer, RoutedExperts):
            return get_loader_moe_method_cls()(self, layer.moe_config)
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, VocabParallelEmbedding):
            return UnquantizedEmbeddingMethod()
        return None


def _decode_dense_gguf_tensor(
    info: TensorInfo, *, out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Bounded row-chunk decode of a 1D/2D dense GGUF tensor (`F32`/`Q8_0`/`BF16`)."""
    if info.type not in (0, 8, 30):
        raise ValueError(
            f"dense GGUF decode refuses type {info.type} on tensor {info.name}"
        )
    if len(info.dims) not in (1, 2):
        raise ValueError(
            f"dense GGUF decode expects 1D/2D tensor, got {info.dims} for {info.name}"
        )
    fmt = _FORMAT_BY_GGML_TYPE[info.type]
    block_vals, block_bytes = _BLOCK_GEOMETRY[info.type]
    row_width = info.dims[0]
    num_rows = 1 if len(info.dims) == 1 else info.dims[1]
    row_bytes = (row_width // block_vals) * block_bytes
    if row_bytes * num_rows != info.nbytes:
        raise ValueError(f"inconsistent byte extent for {info.name}")

    rows_per_chunk = max(1, _DENSE_CHUNK_FLOAT32_BUDGET // max(4, row_width * 4))
    target_dtype = torch.float32 if out_dtype is None else out_dtype
    logical_shape = tuple(reversed(info.dims))
    out = torch.empty((num_rows, row_width), dtype=target_dtype, device="cpu")
    with open(info.path, "rb", buffering=0) as fp:
        for row_start in range(0, num_rows, rows_per_chunk):
            chunk_rows = min(rows_per_chunk, num_rows - row_start)
            byte_len = chunk_rows * row_bytes
            fp.seek(info.offset + row_start * row_bytes)
            raw = fp.read(byte_len)
            if len(raw) != byte_len:
                raise ValueError(f"truncated read on {info.name}")
            decoded = dequantize_packed(
                raw,
                fmt,
                row_width=row_width,
                budget_bytes=chunk_rows * row_width * 4,
            ).reshape(chunk_rows, row_width)
            out[row_start : row_start + chunk_rows].copy_(decoded.to(target_dtype))
    return out.reshape(logical_shape)


def _read_packed_expert_3d(
    info: TensorInfo, *, expected_dims: tuple[int, int, int]
) -> torch.Tensor:
    """Read a packed 3D routed-expert tensor into `[num_experts, rows, packed_cols]`."""
    if info.dims != expected_dims:
        raise ValueError(
            f"expert shape mismatch for {info.name}: "
            f"expected {expected_dims}, got {info.dims}"
        )
    if info.type not in (7, 8, 12, 13):
        raise ValueError(f"unsupported packed expert type {info.type} on {info.name}")
    block_vals, block_bytes = _BLOCK_GEOMETRY[info.type]
    in_elems, out_rows, num_experts = info.dims
    if in_elems % block_vals != 0:
        raise ValueError(f"unaligned expert input width on {info.name}")
    packed_cols = (in_elems // block_vals) * block_bytes
    expected_bytes = num_experts * out_rows * packed_cols
    if info.nbytes != expected_bytes:
        raise ValueError(f"expert byte count mismatch on {info.name}")
    payload = torch.empty((expected_bytes,), dtype=torch.uint8, device="cpu")
    with open(info.path, "rb", buffering=0) as fp:
        fp.seek(info.offset)
        pos = 0
        chunk_cap = 32 << 20
        while pos < expected_bytes:
            want = min(chunk_cap, expected_bytes - pos)
            buf = fp.read(want)
            if len(buf) != want:
                raise ValueError(f"truncated expert read on {info.name}")
            payload[pos : pos + want].copy_(
                torch.frombuffer(bytearray(buf), dtype=torch.uint8)
            )
            pos += want
    return payload.reshape(num_experts, out_rows, packed_cols)


def _zero_hc_alignment_padding(module: nn.Module) -> None:
    """Zero MergedColumnParallelLinear HC buffers so alignment padding rows are 0."""
    for name, param in module.named_parameters():
        if name.endswith("input_mix_weight_down_block_inject.weight"):
            param.data.zero_()


def _iter_filtered_safetensors(
    directory: str | os.PathLike[str], prefix_filter: str
) -> Iterator[tuple[str, torch.Tensor]]:
    base = Path(directory)
    index_path = base / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shard_names = sorted(
            {
                fname
                for key, fname in weight_map.items()
                if key.startswith(prefix_filter) or f".{prefix_filter}" in key
            }
        )
        files = [str(base / fname) for fname in shard_names]
    else:
        files = [str(p) for p in sorted(base.glob("*.safetensors"))]
    if not files:
        raise ValueError(
            f"no safetensors files matching {prefix_filter!r} under {base}"
        )
    for name, tensor in safetensors_weights_iterator(files, False):
        if name.startswith(prefix_filter) or f".{prefix_filter}" in name:
            yield name, tensor


class Qwen4ExpGGUFModelLoader(BaseModelLoader):
    """Fail-closed native model loader for Qwen4Exp GGUF shard sets."""

    _ALLOWED_EXTRA_KEYS = frozenset(
        {"gguf_dir", "mtp_safetensors_dir", "mm_safetensors_dir"}
    )

    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config or {}
        if not isinstance(extra, dict):
            raise ValueError("model_loader_extra_config must be a dict")
        unknown = set(extra) - self._ALLOWED_EXTRA_KEYS
        if unknown:
            raise ValueError(
                f"Unsupported qwen4exp_gguf extra config keys: {sorted(unknown)}"
            )
        gguf_dir = extra.get("gguf_dir")
        if not isinstance(gguf_dir, str) or not gguf_dir:
            raise ValueError(
                "qwen4exp_gguf requires model_loader_extra_config['gguf_dir']"
            )
        self.gguf_dir = Path(gguf_dir)
        self.mtp_safetensors_dir = extra.get("mtp_safetensors_dir")
        self.mm_safetensors_dir = extra.get("mm_safetensors_dir")

    def _resolve_manifest(self) -> Manifest:
        if not self.gguf_dir.is_dir():
            raise ValueError(f"GGUF directory does not exist: {self.gguf_dir}")
        shards = sorted(self.gguf_dir.glob("*.gguf"))
        if not shards:
            raise ValueError(f"No .gguf files found in {self.gguf_dir}")
        return validate_shards(shards, architecture="qwen4exp")

    def download_model(self, model_config: ModelConfig) -> None:
        del model_config
        self._resolve_manifest()

    def _param_target_dtype(
        self,
        checkpoint_name: str,
        params: dict[str, nn.Parameter],
        default_dtype: torch.dtype,
    ) -> torch.dtype:
        if checkpoint_name in params:
            return params[checkpoint_name].dtype
        # Stacked names (`in_proj_qkv` -> `in_proj_qkvz`, etc.) use model dtype
        return default_dtype

    def _iter_dense_weights(
        self,
        by_name: dict[str, TensorInfo],
        consumed: set[str],
        text_config: Any,
        params: dict[str, nn.Parameter],
        default_dtype: torch.dtype,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        num_layers = int(text_config.num_hidden_layers)
        num_k_heads = int(text_config.linear_num_key_heads)
        num_v_heads = int(text_config.linear_num_value_heads)
        head_k_dim = int(text_config.linear_key_head_dim)
        head_v_dim = int(text_config.linear_value_head_dim)
        conv_kernel = int(text_config.linear_conv_kernel_dim)
        full_attn_interval = int(getattr(text_config, "full_attention_interval", 4))

        def transform_dense(
            source_name: str, transform: str, checkpoint_name: str
        ) -> torch.Tensor:
            info = by_name[source_name]
            consumed.add(source_name)
            target_dtype = self._param_target_dtype(
                checkpoint_name, params, default_dtype
            )
            if transform == "identity":
                return _decode_dense_gguf_tensor(info, out_dtype=target_dtype)
            # Transforms that do float math decode in float32 first before casting.
            tensor = _decode_dense_gguf_tensor(info, out_dtype=torch.float32)
            if transform == "subtract_one":
                tensor = inverse_zero_centered_norm(tensor)
            elif transform == "reshape":
                tensor = tensor.unsqueeze(0)
            elif transform == "unsqueeze_conv":
                tensor = tensor.unsqueeze(1)
            elif transform == "gdn_inverse":
                tensor = inverse_gdn_tensor(
                    source_name,
                    tensor,
                    num_k_heads=num_k_heads,
                    num_v_heads=num_v_heads,
                    head_k_dim=head_k_dim,
                    head_v_dim=head_v_dim,
                    conv_kernel=conv_kernel,
                    num_layers=num_layers,
                    full_attention_interval=full_attn_interval,
                )
            else:
                raise ValueError(f"unknown dense transform {transform!r}")
            return tensor.to(target_dtype)

        for src, (ckpt_name, transform) in _HEAD_CHECKPOINT_NAMES.items():
            if src not in by_name:
                raise ValueError(f"missing required head/embedding GGUF tensor: {src}")
            yield ckpt_name, transform_dense(src, transform, ckpt_name)

        pending_indexer_q: dict[int, torch.Tensor] = {}
        pending_indexer_k: dict[int, torch.Tensor] = {}

        for name in by_name:
            if name in _HEAD_CHECKPOINT_NAMES or name == "per_layer_token_embd.weight":
                continue
            match = re.fullmatch(r"blk\.(0|[1-9][0-9]*)\.(.+)", name)
            if match is None:
                raise ValueError(f"unrecognized GGUF tensor name: {name}")
            layer_idx = int(match[1])
            role = match[2]
            if layer_idx >= num_layers:
                raise ValueError(f"GGUF tensor layer out of range: {name}")
            if role in (
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ):
                continue
            if role in ("indexer.q_proj.weight", "indexer.k_proj.weight"):
                ckpt_name = (
                    f"model.layers.{layer_idx}.self_attn.indexer.index_qk_proj.weight"
                )
                target_dtype = self._param_target_dtype(
                    ckpt_name, params, default_dtype
                )
                part = _decode_dense_gguf_tensor(by_name[name], out_dtype=target_dtype)
                consumed.add(name)
                if role == "indexer.q_proj.weight":
                    pending_indexer_q[layer_idx] = part
                else:
                    pending_indexer_k[layer_idx] = part
                if layer_idx in pending_indexer_q and layer_idx in pending_indexer_k:
                    merged = torch.cat(
                        [
                            pending_indexer_q.pop(layer_idx),
                            pending_indexer_k.pop(layer_idx),
                        ],
                        dim=0,
                    )
                    yield ckpt_name, merged
                continue
            if role not in _LAYER_CHECKPOINT_NAMES:
                raise ValueError(f"unsupported GGUF layer tensor: {name}")
            suffix, transform = _LAYER_CHECKPOINT_NAMES[role]
            ckpt_name = f"model.layers.{layer_idx}.{suffix}"
            yield ckpt_name, transform_dense(name, transform, ckpt_name)

        if pending_indexer_q or pending_indexer_k:
            raise ValueError("incomplete indexer q_proj/k_proj pair in GGUF shard set")

    def _load_packed_experts(
        self,
        lm_module: nn.Module,
        by_name: dict[str, TensorInfo],
        consumed: set[str],
        text_config: Any,
    ) -> set[str]:
        num_layers = int(text_config.num_hidden_layers)
        hidden_size = int(text_config.hidden_size)
        inter_size = int(text_config.moe_intermediate_size)
        num_experts = int(text_config.num_experts)
        loaded: set[str] = set()

        for layer_idx in range(num_layers):
            gate_key = f"blk.{layer_idx}.ffn_gate_exps.weight"
            up_key = f"blk.{layer_idx}.ffn_up_exps.weight"
            down_key = f"blk.{layer_idx}.ffn_down_exps.weight"
            for req in (gate_key, up_key, down_key):
                if req not in by_name:
                    raise ValueError(f"missing routed expert GGUF tensor: {req}")
            gate_info = by_name[gate_key]
            up_info = by_name[up_key]
            down_info = by_name[down_key]
            if gate_info.type != up_info.type:
                raise ValueError(
                    f"gate/up quant mismatch on layer {layer_idx}: "
                    f"{gate_info.type} != {up_info.type}"
                )

            routed = lm_module.model.layers[layer_idx].mlp.experts.routed_experts
            store_gguf_weight_type(routed.w13_weight_type, torch.tensor(gate_info.type))
            store_gguf_weight_type(routed.w2_weight_type, torch.tensor(down_info.type))

            gate_3d = _read_packed_expert_3d(
                gate_info, expected_dims=(hidden_size, inter_size, num_experts)
            )
            routed.w13_weight.weight_loader(
                routed.w13_weight, gate_3d, "w13_weight", "w1", 0
            )
            del gate_3d

            up_3d = _read_packed_expert_3d(
                up_info, expected_dims=(hidden_size, inter_size, num_experts)
            )
            routed.w13_weight.weight_loader(
                routed.w13_weight, up_3d, "w13_weight", "w3", 0
            )
            del up_3d

            down_3d = _read_packed_expert_3d(
                down_info, expected_dims=(inter_size, hidden_size, num_experts)
            )
            routed.w2_weight.weight_loader(
                routed.w2_weight, down_3d, "w2_weight", "w2", 0
            )
            del down_3d

            consumed.update((gate_key, up_key, down_key))
            base = f"model.layers.{layer_idx}.mlp.experts.routed_experts"
            loaded.update(
                {
                    f"{base}.w13_weight",
                    f"{base}.w13_weight_type",
                    f"{base}.w2_weight",
                    f"{base}.w2_weight_type",
                }
            )
        return loaded

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        manifest = self._resolve_manifest()
        by_name = {t.name: t for t in manifest.tensors}
        text_config = model_config.hf_text_config

        if (
            int(text_config.num_hidden_layers) == 48
            and int(text_config.hidden_size) == 2560
            and int(text_config.num_experts) == 512
        ):
            # Name-only admission: real parameter shapes are rank-local under TP,
            # so full shape equality is enforced separately on TP1.
            plan = plan_tensors(
                [
                    {"name": t.name, "dims": t.dims, "type": t.type}
                    for t in manifest.tensors
                ],
                names_only=True,
            )
            if len(plan.entries) != 1224 or plan.unsupported:
                raise ValueError(
                    f"Qwen3.8 GGUF contract check failed: entries={len(plan.entries)}, "
                    f"unsupported={len(plan.unsupported)}"
                )

        lm_module = model.language_model if hasattr(model, "language_model") else model
        _zero_hc_alignment_padding(lm_module)
        lm_params = dict(lm_module.named_parameters())
        consumed: set[str] = set()

        loaded_dense = lm_module.load_weights(
            self._iter_dense_weights(
                by_name,
                consumed,
                text_config,
                lm_params,
                model_config.dtype,
            )
        )
        loaded_experts = self._load_packed_experts(
            lm_module, by_name, consumed, text_config
        )

        ple_params: set[str] = set()
        if getattr(text_config, "ple_layer_ids", None):
            if "per_layer_token_embd.weight" not in by_name:
                raise ValueError("missing required per_layer_token_embd.weight")
            consumed.add("per_layer_token_embd.weight")
            for ple_one_based in text_config.ple_layer_ids:
                ple_base = (
                    f"model.layers.{int(ple_one_based) - 1}"
                    ".ple.ple_embedding.ngram_embedding"
                )
                ple_params.update({f"{ple_base}.weight", f"{ple_base}.weight_scale"})

        expected_params = set(lm_params)
        accounted = loaded_dense | loaded_experts | ple_params
        unloaded = sorted(expected_params - accounted)
        if unloaded:
            raise ValueError(
                f"unloaded native Qwen4Exp parameters ({len(unloaded)}): {unloaded[:8]}"
            )
        unconsumed = sorted(set(by_name) - consumed)
        if unconsumed:
            raise ValueError(
                f"unconsumed GGUF tensors ({len(unconsumed)}): {unconsumed[:8]}"
            )

        if hasattr(model, "visual") and not getattr(model, "language_model_only", True):
            if not self.mm_safetensors_dir:
                raise ValueError(
                    "GGUF artifact contains no vision tower tensors; pass "
                    "--language-model-only or set "
                    "model_loader_extra_config.mm_safetensors_dir"
                )
            loaded_mm = model.load_weights(
                _iter_filtered_safetensors(self.mm_safetensors_dir, "visual.")
            )
            vis_params = {
                k for k in dict(model.named_parameters()) if k.startswith("visual.")
            }
            missing_vis = sorted(vis_params - loaded_mm)
            if missing_vis:
                raise ValueError(
                    f"unloaded visual sidecar parameters ({len(missing_vis)}): "
                    f"{missing_vis[:8]}"
                )

    def _load_mtp_draft_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        target_device: torch.device,
        prefix: str,
    ) -> nn.Module:
        if not self.mtp_safetensors_dir:
            raise ValueError(
                "GGUF artifact contains no MTP tensors; disable speculative_config "
                "or explicitly set model_loader_extra_config['mtp_safetensors_dir']"
            )
        prev_quant = vllm_config.quant_config
        vllm_config.quant_config = None
        try:
            with set_default_torch_dtype(model_config.dtype), target_device:
                draft_model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
            _zero_hc_alignment_padding(draft_model)
            with set_current_vllm_config(vllm_config):
                loaded = draft_model.load_weights(
                    _iter_filtered_safetensors(self.mtp_safetensors_dir, "mtp.")
                )
            # embed_tokens and lm_head are shared from target_model by EagleProposer
            required = {
                name
                for name in dict(draft_model.named_parameters())
                if not name.startswith(("lm_head.", "model.embed_tokens."))
            }
            missing = sorted(required - loaded)
            if missing:
                raise ValueError(
                    f"unloaded MTP draft parameters ({len(missing)}): {missing[:8]}"
                )
            process_weights_after_loading(draft_model, model_config, target_device)
            return draft_model
        finally:
            vllm_config.quant_config = prev_quant

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        target_device = torch.device(vllm_config.device_config.device)
        archs = getattr(model_config, "architectures", None) or []
        if "Qwen4ExpMTP" in archs:
            return self._load_mtp_draft_model(
                vllm_config, model_config, target_device, prefix
            )

        if envs.VLLM_PLE_QUANT_DIR:
            raise ValueError(
                "qwen4exp_gguf conflicts with VLLM_PLE_QUANT_DIR; "
                "unset it so exact GGUF IQ4_NL PLE runs"
            )
        if envs.VLLM_PLE_CPU_OFFLOAD:
            raise ValueError(
                "qwen4exp_gguf uses pinned-host UVA IQ4_NL PLE and "
                "conflicts with VLLM_PLE_CPU_OFFLOAD"
            )
        mm_cfg = getattr(model_config, "multimodal_config", None)
        if (
            "Qwen4ExpForConditionalGeneration" in archs
            and mm_cfg is not None
            and not mm_cfg.language_model_only
            and not self.mm_safetensors_dir
        ):
            raise ValueError(
                "GGUF artifact contains no vision tower tensors; "
                "pass --language-model-only "
                "or set model_loader_extra_config['mm_safetensors_dir']"
            )

        manifest = self._resolve_manifest()
        by_name = {t.name: t for t in manifest.tensors}
        ple_payload: dict[str, Any] | None = None
        if getattr(model_config.hf_text_config, "ple_layer_ids", None):
            if "per_layer_token_embd.weight" not in by_name:
                raise ValueError("missing per_layer_token_embd.weight in GGUF manifest")
            ple_info = by_name["per_layer_token_embd.weight"]
            ple_payload = {
                "path": ple_info.path,
                "data_offset": ple_info.offset,
                "rows": ple_info.dims[1],
                "width": ple_info.dims[0],
                "type": ple_info.type,
            }

        vllm_config.quant_config = Qwen4ExpGGUFQuantConfig(gguf_ple_payload=ple_payload)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
            with set_current_vllm_config(vllm_config):
                self.load_weights(model, model_config)
                process_weights_after_loading(model, model_config, target_device)
        return model
