# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind the draft to a Flash-Next module. Called only when the flag is on.

The forward hook records that layer ``split`` was entered and checks the
carried residual width. It does not replace activations, so a boot with
the flag on is still numerically exact. Skipping the late MoE and writing
the cache from the prediction is the GPU follow-up in ENGINE.md.
"""

from __future__ import annotations

from vllm.model_executor.layers.ridgefill.config import (
    RidgeConfig,
    RidgeFillConfigError,
    enabled,
    load_config,
)
from vllm.model_executor.layers.ridgefill.loader import require_bundle
from vllm.model_executor.layers.ridgefill.policy import (
    RowPlan,
    apply_v1_guards,
    plan_prompt,
)


class Session:
    """Per-model draft state. ``observe_enter`` does not write tensors."""

    def __init__(self, cfg: RidgeConfig, block_size: int) -> None:
        self.cfg = cfg
        self.block_size = block_size
        self.saw_split = False
        self.boundary_width: int | None = None
        self.plan: RowPlan | None = None

    def observe_enter(self, layer_idx: int, num_tokens: int, width: int) -> None:
        """Note the residual carried into ``layer_idx``.

        The map's input is the four raw HC streams entering layer ``split``,
        before that layer's norm. In this model that tensor is materialized
        inside the layer, by ``combine`` / ``combine_and_mix``, and this hook
        only checks the carried width. A GPU port captures the combined
        tensor there and does not run a second combine.
        """
        if layer_idx != self.cfg.split:
            return
        self.saw_split = True
        self.boundary_width = width
        if width != self.cfg.din:
            raise RidgeFillConfigError(
                f"split boundary width {width} != map din {self.cfg.din}. "
                "Flash-Next's four streams are 10240 wide on every TP rank."
            )
        align = self.cfg.apc == "v2" and self.block_size > 1
        self.plan = plan_prompt(
            num_tokens,
            split=self.cfg.split,
            tail=self.cfg.tail,
            block_size=self.block_size,
            align=align,
        )


def _block_size(vllm_config) -> int:
    cache = vllm_config.cache_config
    hashed = getattr(cache, "hash_block_size", None)
    if hashed:
        return int(hashed)
    block = getattr(cache, "block_size", None)
    if block:
        return int(block)
    return 0


def bind_flashnext(causal_lm) -> None:
    """Install guards and a session. Raises if MTP or v1 APC is on.

    A map path without a matching pin file is refused. No map path means
    the session is installed but no bundle is applied: the forward stays
    exact. That is deliberate. This draft does not approximate.
    """
    if not enabled():
        return
    cfg = load_config()
    if cfg.mode != "plain":
        raise RidgeFillConfigError(
            "VLLM_RIDGEFILL_MODE=quality is not in this draft. "
            "Terminal-state correction is ENGINE.md, later."
        )
    if cfg.apc == "v2" and not cfg.map_sha:
        raise RidgeFillConfigError(
            "VLLM_RIDGEFILL_APC=v2 needs VLLM_RIDGEFILL_MAP_SHA, the "
            "sha256 of the refit bundle, so filled blocks are not shared "
            "across different maps."
        )
    vllm_config = causal_lm.vllm_config
    spec = vllm_config.speculative_config
    apply_v1_guards(
        flag_on=True,
        speculative=spec is not None,
        prefix_caching=bool(vllm_config.cache_config.enable_prefix_caching),
        apc_mode=cfg.apc,
    )
    if cfg.map_path or cfg.pins_path:
        require_bundle(cfg.map_path, cfg.pins_path)
    block = cfg.block_size or _block_size(vllm_config)
    causal_lm.model._ridgefill = Session(cfg, block)
