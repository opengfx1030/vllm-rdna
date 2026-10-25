# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refuse a dense load of Glm5Next leftovers that the checkpoint quantized."""

from collections.abc import Mapping

from vllm.model_executor.layers.quantization.utils.config_utils import (
    is_equal_or_regex_match,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)


def glm5_leftover_is_quantized(
    quant_config: object | None,
    prefix: str,
    module_cls_name: str,
) -> bool:
    """Return whether ``prefix`` would be weight-quantized.

    Compressed-tensors is decided the same way as ``get_scheme_dict``:
    an ignore/regex hit is dense, otherwise a config-group target that
    matches the prefix or the module class name (``Linear`` matches
    ``MergedColumnParallelLinear``) is quantized. AWQ quantizes ``lm_head``
    only when ``lm_head_quantized`` is set, and other linears only when
    they are absent from ``modules_to_not_convert``. Embeddings stay dense
    unless a target actually names them.

    Args:
        quant_config: The checkpoint quant config, or None.
        prefix: vLLM module prefix.
        module_cls_name: Class name used for compressed-tensors class targets.

    Returns:
        True when the checkpoint quantizes this module.
    """
    if quant_config is None:
        return False

    ignore = getattr(quant_config, "ignore", None)
    scheme_map = getattr(quant_config, "target_scheme_map", None)
    if ignore is not None or scheme_map:
        from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
            should_ignore_layer,
        )

        fused: Mapping[str, list[str]] = (
            getattr(quant_config, "packed_modules_mapping", None) or {}
        )
        if should_ignore_layer(prefix, ignore=ignore or [], fused_mapping=fused):
            return False
        if not scheme_map:
            return False
        for target, scheme in scheme_map.items():
            if not isinstance(scheme, dict) or scheme.get("weights") is None:
                continue
            if is_equal_or_regex_match(prefix, target) or is_equal_or_regex_match(
                module_cls_name, target, check_contains=True
            ):
                return True
        return False

    if prefix.endswith("lm_head"):
        quantized = bool(getattr(quant_config, "lm_head_quantized", False))
        return quantized and not _awq_skipped(quant_config, prefix)
    if "embed_tokens" in prefix:
        return False

    modules = getattr(quant_config, "modules_to_not_convert", None)
    if modules is None:
        return False
    return not _awq_skipped(quant_config, prefix)


def require_glm5_dense_leftover(
    quant_config: object | None,
    prefix: str,
    kind: str,
    module_cls_name: str,
) -> None:
    """Raise if a leftover that this model builds dense is quantized.

    Args:
        quant_config: The checkpoint quant config, or None.
        prefix: vLLM module prefix.
        kind: Short name used in the error (for example ``shared expert``).
        module_cls_name: Class name used for compressed-tensors class targets.

    Raises:
        ValueError: The checkpoint quantizes ``prefix``.
    """
    if not glm5_leftover_is_quantized(quant_config, prefix, module_cls_name):
        return
    raise ValueError(
        f"GLM-5.3-Flash {kind} ({prefix}) is quantized in this checkpoint. "
        "Shared experts, lm_head, embeddings, the MoE router, KDA "
        "projections, and the DSA indexer are dense bf16/fp32 leftovers. "
        "Refusing to load them through an unquantized path."
    )


def _awq_skipped(quant_config: object, prefix: str) -> bool:
    modules = getattr(quant_config, "modules_to_not_convert", None) or []
    fused = getattr(quant_config, "packed_modules_mapping", None) or {}
    return is_layer_skipped(
        prefix,
        list(modules),
        fused,
        match_mode="substring",
    )
