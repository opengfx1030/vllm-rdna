# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Classify a Qwen4-family HF config against what this fork can serve.

Opt-in release-day triage. Nothing imports this from the serving path and it
never changes engine behaviour: it reads a ``config.json`` and reports which
Qwen4 features the fork implements, which ones block a load, and which
warnings apply on gfx1030. Run it on the first released checkpoint before
writing a recipe::

    python -m vllm.models.qwen4_prep.config_probe /path/to/config.json

The module is stdlib-only so it runs on a CPU box without torch or vLLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Model types with a live implementation in this fork. Everything else in the
# Qwen4 line needs a new model class; the probe says so instead of letting the
# registry fail with a generic "architecture not supported" message.
SUPPORTED_MODEL_TYPES = frozenset({"qwen4_exp", "qwen4_exp_text"})

# Layer types the Qwen4Exp decoder understands. ``qwen_sparse_attention`` is
# the alias SGLang and older checkpoints use for QSA.
KNOWN_LAYER_TYPES = frozenset(
    {"linear_attention", "full_attention", "qwen_sparse_attention"}
)

# Quantization formats that cannot run on gfx1030: no FP8, no FP4, no MX.
UNSUPPORTED_QUANT_FORMATS = frozenset(
    {
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "modelopt_fp4",
        "mxfp4",
        "mxfp8",
        "nvfp4",
        "w4a8_fp8",
    }
)

_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
}


@dataclass(frozen=True)
class Qwen4ConfigReport:
    """Verdict for one Qwen4-family config."""

    model_type: str
    text_model_type: str
    supported: bool
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    geometry: dict[str, Any] = field(default_factory=dict)
    kv_bytes_per_token: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "text_model_type": self.text_model_type,
            "supported": self.supported,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "geometry": self.geometry,
            "kv_bytes_per_token": self.kv_bytes_per_token,
        }


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    text = config.get("text_config")
    return text if isinstance(text, Mapping) else config


def _quant_format(config: Mapping[str, Any], text: Mapping[str, Any]) -> str | None:
    for source in (config, text):
        quant = source.get("quantization_config")
        if isinstance(quant, Mapping):
            for key in ("quant_method", "format", "quant_format"):
                value = quant.get(key)
                if isinstance(value, str):
                    return value.lower()
    return None


def _dtype(config: Mapping[str, Any], text: Mapping[str, Any]) -> str | None:
    for source in (text, config):
        for key in ("dtype", "torch_dtype"):
            value = source.get(key)
            if isinstance(value, str):
                return value
    return None


def _mtp_depth(text: Mapping[str, Any]) -> int:
    mtp = text.get("mtp")
    if isinstance(mtp, Mapping):
        depth = mtp.get("num_hidden_layers")
        if isinstance(depth, int):
            return depth
    depth = text.get("mtp_num_hidden_layers")
    return depth if isinstance(depth, int) else 0


def _full_attention_layers(text: Mapping[str, Any]) -> int:
    layer_types = text.get("layer_types")
    if isinstance(layer_types, Sequence) and not isinstance(layer_types, str):
        return sum(
            1
            for layer in layer_types
            if layer in ("full_attention", "qwen_sparse_attention")
        )
    layers = text.get("num_hidden_layers")
    interval = text.get("full_attention_interval")
    if isinstance(layers, int) and isinstance(interval, int) and interval > 0:
        return layers // interval
    return 0


def probe_config(config: Mapping[str, Any]) -> Qwen4ConfigReport:
    """Classify one HF config mapping.

    Raises ``ValueError`` when the config is not a Qwen4-family config, so a
    caller cannot silently apply the verdict to another model.
    """
    text = _text_config(config)
    model_type = config.get("model_type") or text.get("model_type")
    text_model_type = text.get("model_type") or model_type
    if not isinstance(model_type, str) or not model_type.startswith("qwen4"):
        raise ValueError(f"not a Qwen4-family config: model_type={model_type!r}")

    blockers: list[str] = []
    warnings: list[str] = []

    if model_type not in SUPPORTED_MODEL_TYPES:
        blockers.append(
            f"no model class for model_type {model_type!r}; this fork implements "
            f"{sorted(SUPPORTED_MODEL_TYPES)}"
        )

    num_experts = text.get("num_experts")
    if not isinstance(num_experts, int) or num_experts <= 0:
        blockers.append(
            "dense MLP is not implemented: Qwen4ExpDecoderLayer always builds a "
            "sparse MoE block, so a dense Qwen4 checkpoint needs a new MLP class"
        )

    layer_types = text.get("layer_types")
    if isinstance(layer_types, Sequence) and not isinstance(layer_types, str):
        unknown = sorted({str(layer) for layer in layer_types} - KNOWN_LAYER_TYPES)
        if unknown:
            blockers.append(f"unknown layer_types: {unknown}")

    indexer_heads = text.get("indexer_n_heads")
    if indexer_heads is None:
        warnings.append(
            "no indexer_* fields: QSA is disabled and full-attention layers fall "
            "back to dense attention"
        )
    else:
        indexer_kv_heads = text.get("indexer_kv_heads")
        if indexer_kv_heads != 1:
            blockers.append(
                f"QSA requires indexer_kv_heads == 1 (got {indexer_kv_heads!r})"
            )
        if text.get("indexer_head_dim") is None:
            warnings.append("indexer_head_dim is missing; QSA geometry is incomplete")

    if text.get("hc_count") is None:
        warnings.append("no hc_count: the checkpoint has no hyper-connections")

    ple_layer_ids = text.get("ple_layer_ids")
    if ple_layer_ids and text.get("ngram_size") is None:
        warnings.append("PLE layers present but ngram_size is missing")

    dtype = _dtype(config, text)
    if dtype == "bfloat16":
        warnings.append(
            "dtype is bfloat16; gfx1030 has no bf16 dot product, serve --dtype float16"
        )

    quant_format = _quant_format(config, text)
    if quant_format in UNSUPPORTED_QUANT_FORMATS:
        blockers.append(
            f"quantization format {quant_format!r} cannot run on gfx1030 "
            "(no FP8/FP4/MX support)"
        )

    kv_heads = text.get("num_key_value_heads")
    head_dim = text.get("head_dim")
    n_full = _full_attention_layers(text)
    kv_bytes = None
    if isinstance(kv_heads, int) and isinstance(head_dim, int) and n_full:
        element = _DTYPE_BYTES.get(dtype or "float16", 2)
        kv_bytes = n_full * kv_heads * head_dim * 2 * element

    geometry = {
        "num_hidden_layers": text.get("num_hidden_layers"),
        "hidden_size": text.get("hidden_size"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "full_attention_layers": n_full,
        "num_experts": num_experts,
        "num_experts_per_tok": text.get("num_experts_per_tok"),
        "moe_intermediate_size": text.get("moe_intermediate_size"),
        "shared_expert_intermediate_size": text.get("shared_expert_intermediate_size"),
        "hc_count": text.get("hc_count"),
        "hc_lowrank": text.get("hc_lowrank"),
        "ple_layer_ids": list(ple_layer_ids) if ple_layer_ids else [],
        "mtp_depth": _mtp_depth(text),
        "linear_num_value_heads": text.get("linear_num_value_heads"),
        "dtype": dtype,
        "quant_format": quant_format,
    }

    return Qwen4ConfigReport(
        model_type=model_type,
        text_model_type=str(text_model_type),
        supported=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        geometry=geometry,
        kv_bytes_per_token=kv_bytes,
    )


def load_config(path: str | Path) -> dict[str, Any]:
    """Read ``config.json`` from a file or a checkpoint directory."""
    target = Path(path)
    if target.is_dir():
        target = target / "config.json"
    with target.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{target} does not contain a JSON object")
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="config.json or checkpoint directory")
    args = parser.parse_args(argv)
    report = probe_config(load_config(args.config))
    json.dump(report.as_dict(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report.supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
