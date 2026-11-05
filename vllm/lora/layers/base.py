# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import TYPE_CHECKING, overload

import torch
import torch.nn as nn
from transformers import PreTrainedConfig

from vllm.config.lora import LoRAConfig

if TYPE_CHECKING:
    from vllm.lora.punica_wrapper import PunicaWrapperBase


class LanePunicaWrapperView:
    """A punica wrapper binding that resolves per calling lane.

    A layer binds one wrapper per lane; some paths stash the wrapper object
    once (the fused-MoE kernel's LoRA context) and call it during forwards
    from either lane. This view forwards every attribute access to the
    calling thread's lane wrapper, so those paths stay lane-correct without
    rebuilding the stashed object.
    """

    __slots__ = ("_wrappers",)

    def __init__(self, wrappers: dict[int, "PunicaWrapperBase"]):
        object.__setattr__(self, "_wrappers", wrappers)

    def _resolve(self) -> "PunicaWrapperBase":
        from vllm.v1.lane import current_lane, lane_mode_enabled

        wrappers: dict[int, PunicaWrapperBase] = object.__getattribute__(
            self, "_wrappers"
        )
        if not lane_mode_enabled():
            return next(iter(wrappers.values()))
        return wrappers[int(current_lane())]

    def __getattr__(self, name):
        return getattr(self._resolve(), name)


class BaseLayerWithLoRA(nn.Module):
    def __getattr__(self, name):
        d = self.__dict__
        if name in d.get("_parameters", ()):
            return d["_parameters"][name]
        if name in d.get("_buffers", ()):
            return d["_buffers"][name]
        if name in d.get("_modules", ()):
            return d["_modules"][name]
        # Forward public misses to ``base_layer``; private names are framework
        # bookkeeping and must stay local.
        if not name.startswith("_"):
            base_layer = d.get("_modules", {}).get("base_layer")
            if base_layer is not None:
                return getattr(base_layer, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        """Load checkpoint weights into the wrapped base layer."""
        base_load_weights = getattr(self.base_layer, "load_weights", None)
        if callable(base_load_weights):
            return base_load_weights(weights)

        from vllm.model_executor.models.utils import AutoWeightsLoader

        return AutoWeightsLoader(self.base_layer).load_weights(weights)

    @overload
    def slice_lora_a(
        self, lora_a: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]: ...
    @overload
    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor: ...
    def slice_lora_a(
        self, lora_a: torch.Tensor | list[torch.Tensor | None]
    ) -> torch.Tensor | list[torch.Tensor | None]:
        """Slice lora a if splitting for tensor parallelism."""
        ...

    @overload
    def slice_lora_b(
        self, lora_b: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]: ...
    @overload
    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor: ...
    def slice_lora_b(
        self, lora_b: torch.Tensor | list[torch.Tensor | None]
    ) -> torch.Tensor | list[torch.Tensor | None]:
        """Slice lora b if splitting with tensor parallelism."""
        ...

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PreTrainedConfig | None = None,
    ) -> None:
        """Initializes lora matrices."""
        ...

    def reset_lora(self, index: int):
        """Resets the lora weights at index back to 0."""
        ...

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ):
        """Overwrites lora tensors at index."""
        ...

    def set_mapping(
        self,
        punica_wrapper,
    ):
        """Bind the punica wrapper(s) this layer's forward reads.

        Single-lane mode passes one wrapper. In lane mode the LoRA manager
        passes one wrapper per lane: both lanes share this layer (and its
        adapter weights), so each lane's forward must read its own mapping
        buffers. A single shared mapping would let one lane's per-step copy
        overwrite the mapping the other lane's in-flight kernels read.
        """
        self._lane_punica_wrappers: dict[int, PunicaWrapperBase] | None
        if isinstance(punica_wrapper, dict):
            lane_wrappers = punica_wrapper
            self._lane_punica_wrappers = lane_wrappers
            self._single_punica_wrapper = next(iter(lane_wrappers.values()))
        else:
            self._lane_punica_wrappers = None
            self._single_punica_wrapper = punica_wrapper

    def punica_wrapper_view(self) -> "PunicaWrapperBase | LanePunicaWrapperView":
        """The binding to stash for later use from either lane."""
        lane_wrappers = self.__dict__.get("_lane_punica_wrappers")
        if lane_wrappers is None:
            return self.__dict__["_single_punica_wrapper"]
        return LanePunicaWrapperView(lane_wrappers)

    @property
    def punica_wrapper(self) -> "PunicaWrapperBase":
        """The punica wrapper for the calling thread's lane."""
        lane_wrappers = self.__dict__.get("_lane_punica_wrappers")
        single = self.__dict__["_single_punica_wrapper"]
        if lane_wrappers is None:
            return single
        from vllm.v1.lane import current_lane, lane_mode_enabled

        if not lane_mode_enabled():
            return single
        return lane_wrappers[int(current_lane())]

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PreTrainedConfig | None = None,
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        raise NotImplementedError
