# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one place the Qwen4Exp GGUF loader touches ``vllm-gguf-plugin``.

GGUF support moved out of vLLM into the out-of-tree ``vllm-gguf-plugin``
package. The packed-expert MoE method (``GGUFMoEMethod``) and the helper that
stores a tensor's GGUF quant type on a parameter live there and are not part of
vLLM. Keeping every plugin import in this module means a plugin change is a
one-file fix, and a missing or mismatched plugin raises one clear error at the
point of use instead of a bare ``ModuleNotFoundError`` deep in model
construction.

Two failure modes are told apart:

* the plugin is not installed;
* the plugin is installed but does not import against this vLLM (for example
  plugin 0.0.5 imports ``vllm.transformers_utils.configs.qwen3_5``, which vLLM
  removed in v0.32; plugin PR #148 fixes that).

Nothing is checked beyond what the loader calls: the symbols are looked up by
name and a missing one is reported, with no version floor guessed.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch

_PLUGIN = "vllm_gguf_plugin"
_MOE_MODULE = f"{_PLUGIN}.quantization.fused_moe"
_PARAMS_MODULE = f"{_PLUGIN}.quantization.params"


def _import(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        missing = (getattr(exc, "name", None) or "").split(".")[0]
        if isinstance(exc, ModuleNotFoundError) and missing == _PLUGIN:
            raise ImportError(
                "load_format 'qwen4exp_gguf' needs the out-of-tree "
                "vllm-gguf-plugin package (pip install vllm-gguf-plugin; on "
                "ROCm build it from source for your gfx target)"
            ) from exc
        raise ImportError(
            f"vllm-gguf-plugin is installed but '{module}' failed to import "
            f"against this vLLM ({type(exc).__name__}: {exc}); the plugin "
            "version does not match this vLLM"
        ) from exc


def _symbol(module: str, name: str) -> Any:
    mod = _import(module)
    try:
        return getattr(mod, name)
    except AttributeError as exc:
        raise ImportError(
            f"vllm-gguf-plugin has no '{name}' in '{module}'; the plugin "
            "version does not match this loader"
        ) from exc


def get_gguf_moe_method_cls() -> type:
    """``GGUFMoEMethod``: the packed-expert fused-MoE quant method."""
    return _symbol(_MOE_MODULE, "GGUFMoEMethod")


def store_gguf_weight_type(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: int | str | None = None,
) -> None:
    """Record a tensor's GGUF quant type on a plugin-created parameter."""
    _symbol(_PARAMS_MODULE, "_store_gguf_weight_type")(param, loaded_weight, shard_id)
