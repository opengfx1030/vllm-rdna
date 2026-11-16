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

import functools
import importlib
import os
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


# vllm-gguf-plugin runs the packed-expert GEMM with the MMQ kernel (`moe_q`) for
# batches of more than this many tokens and with the per-slot vector kernel
# (`moe_vec_q`) otherwise (fused_moe.py in the plugin).
_PLUGIN_MMQ_MIN_TOKENS = 65

# `moe_q` returns early for expert ids above 255 (hard-coded in the plugin's
# moe.cuh), so it silently drops the contribution of experts 256+ . Models with
# more experts than this are fed to the vector kernel in slices instead. A plugin
# built without that bound can lift the limit with the variable below.
_MMQ_MAX_EXPERTS_ENV = "VLLM_QWEN4EXP_GGUF_MMQ_MAX_EXPERTS"
_PLUGIN_MMQ_MAX_EXPERTS = 256


def _mmq_max_experts() -> int:
    return int(os.environ.get(_MMQ_MAX_EXPERTS_ENV, _PLUGIN_MMQ_MAX_EXPERTS))


@functools.cache
def _harden_moe_method(base: type) -> type:
    class HardenedGGUFMoEMethod(base):  # type: ignore[misc, valid-type]
        """The plugin's MoE method with two kernel limits worked around.

        * ``topk_ids < 0`` means "no expert" in vLLM; the plugin kernels read
          ``expert * stride`` unchecked, so -1 is an out-of-bounds weight read
          (GPU memory fault seen on gfx1030 during CUDA-graph warmup, where the
          dummy batch routes every slot to -1). Those slots go to expert 0 with
          weight 0: they add exactly zero to the output.
        * Batches above the plugin's MMQ threshold are split into slices that
          stay on the vector kernel when the model has more experts than the
          MMQ kernel supports (see ``_MMQ_MAX_EXPERTS_ENV``).
        """

        def apply(self, layer, x, topk_weights, topk_ids, *args, **kwargs):
            topk_weights = topk_weights * (topk_ids >= 0)
            topk_ids = topk_ids.clamp(min=0)
            rows = x.shape[0]
            step = _PLUGIN_MMQ_MIN_TOKENS - 1
            if (
                rows <= step
                or getattr(layer, "global_num_experts", 0) <= _mmq_max_experts()
            ):
                return super().apply(layer, x, topk_weights, topk_ids, *args, **kwargs)
            return torch.cat(
                [
                    super(HardenedGGUFMoEMethod, self).apply(
                        layer,
                        x[i : i + step],
                        topk_weights[i : i + step],
                        topk_ids[i : i + step],
                        *args,
                        **kwargs,
                    )
                    for i in range(0, rows, step)
                ]
            )

    HardenedGGUFMoEMethod.__name__ = f"Hardened{base.__name__}"
    return HardenedGGUFMoEMethod


def get_loader_moe_method_cls() -> type:
    """The plugin's MoE method with its kernel limits worked around."""
    return _harden_moe_method(get_gguf_moe_method_cls())


def store_gguf_weight_type(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: int | str | None = None,
) -> None:
    """Record a tensor's GGUF quant type on a plugin-created parameter."""
    _symbol(_PARAMS_MODULE, "_store_gguf_weight_type")(param, loaded_weight, shard_id)
