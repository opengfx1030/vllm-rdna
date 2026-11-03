# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind the draft to a Flash-Next module. Called only when the flag is on.

The session records the split boundary, builds the row plan, and loads the
fitted fp16 maps. The model's forward captures the four-stream residual at
layer ``split`` and, for each late layer, applies that layer's map to the
split state. Filled rows run attention and GDN from the prediction and skip
the MoE. Exact rows, including the tail, run the layer as they do now. Decode
never fills: its row plan has an empty filled region.
"""

from __future__ import annotations

from typing import Any

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
    """Per-model draft state and the fitted late-layer maps."""

    def __init__(self, cfg: RidgeConfig, block_size: int) -> None:
        self.cfg = cfg
        self.block_size = block_size
        self.saw_split = False
        self.boundary_width: int | None = None
        self.plan: RowPlan | None = None
        self.maps: dict[int, tuple] = {}
        # ``torch`` is lazy in this package; the state is a CPU tensor.
        self.split_state: Any = None
        self.overflow = False
        # Per-sequence lengths of this forward, from the attention metadata.
        # With chunked prefill ``num_tokens`` is the chunk, not the prompt.
        self.seq_lens = None
        self.query_start_loc = None
        self.num_prefills = 0

    def _seq_metadata(self):
        """The forward's ``seq_lens`` and ``query_start_loc``, or (None, None).

        Chunked prefill gives each layer the chunk's rows. The tail is the
        last ``tail`` tokens of the full prompt, so the cut needs the full
        sequence length. QSA metadata carries it; GDN metadata does not, so
        any QSA entry in this forward serves.
        """
        try:
            from vllm.forward_context import get_forward_context

            metadata = get_forward_context().attn_metadata
        except Exception:
            return None, None, 0
        entries = metadata if isinstance(metadata, list) else [metadata]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for value in entry.values():
                lens = getattr(value, "seq_lens", None)
                loc = getattr(value, "query_start_loc", None)
                if lens is not None and loc is not None:
                    prefills = int(getattr(value, "num_prefills", 0))
                    return lens.detach().cpu(), loc.detach().cpu(), prefills
        return None, None, 0

    def observe_enter(self, layer_idx: int, num_tokens: int, width: int) -> None:
        """Note the residual carried into ``layer_idx`` and build the row plan.

        The map's input is the four raw HC streams entering layer ``split``,
        before that layer's norm. In this model that tensor is materialized
        inside the layer, by ``combine`` / ``combine_and_mix``. The model's
        forward captures it with ``capture_split`` and does not run a second
        combine.
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
        self.seq_lens, self.query_start_loc, self.num_prefills = self._seq_metadata()
        # Decode never fills. Without metadata, a forward of one token row is
        # a decode step; every longer forward is the prefill we want.
        prefill = (
            self.num_prefills > 0 if self.seq_lens is not None else (num_tokens > 1)
        )
        full_tokens = num_tokens
        if prefill and self.seq_lens is not None and len(self.seq_lens) > 0:
            full_tokens = max(full_tokens, int(self.seq_lens.max().item()))
        if not prefill:
            self.plan = plan_prompt(0, split=self.cfg.split, tail=self.cfg.tail)
            return
        self.plan = plan_prompt(
            full_tokens,
            split=self.cfg.split,
            tail=self.cfg.tail,
            block_size=self.block_size,
            align=align,
        )

    def load_maps(self) -> None:
        """Read the fitted fp16 maps. ``VLLM_RIDGEFILL_MAP`` is a dir or file."""
        if not self.cfg.map_path:
            return
        from pathlib import Path

        import torch

        path = Path(self.cfg.map_path)
        if path.is_dir():
            path = path / "maps.pt"
        if not path.exists():
            raise RidgeFillConfigError(f"RidgeFill map not found: {path}")
        maps = torch.load(path, map_location="cpu")
        if not isinstance(maps, dict) or not maps:
            raise RidgeFillConfigError(f"{path} is not a non-empty map dict")
        self.maps = {int(k): v for k, v in maps.items()}

    def capture_split(self, hidden_states) -> None:
        """Keep the four-stream residual entering the split layer."""
        self.split_state = hidden_states.detach()
        self.overflow = False
        if self.plan is not None and self.plan.active:
            # gfx1030 has no fp8 and bf16 is the overflow fallback. One check
            # per prefill forward; decode never activates the plan.
            self.overflow = bool((self.split_state.abs() > 65504).any())

    def predict(self, layer_idx: int):
        """Apply layer ``layer_idx``'s map to the split state, or None."""
        if not self.maps or self.split_state is None:
            return None
        pair = self.maps.get(layer_idx)
        if pair is None:
            return None
        import torch

        a, b = pair
        x_s = self.split_state
        # Path A: the existing fp16 GEMM with fp32 accumulate. The map is
        # replicated, so there is no all-gather of x_S and no new kernel.
        # bf16 is the overflow fallback (no bf16 dot on gfx1030, slower).
        overflow = self.overflow or self.cfg.map_dtype == "bf16"
        dtype = torch.bfloat16 if overflow else torch.float16
        if a.dtype != dtype:
            a = a.to(dtype)
            b = b.to(dtype)
        if a.device != x_s.device:
            a = a.to(x_s.device)
            b = b.to(x_s.device)
            self.maps[layer_idx] = (a, b)
        # fp16 GEMM, fp32 accumulate: the half map times the half split state
        # (the device accumulates in fp32), returned fp32 so ``torch.where``
        # can mix it with the exact fp32 rows.
        return torch.nn.functional.linear(x_s.to(dtype), a, b).float()

    def filled_mask(self, num_tokens: int, positions=None):
        """Boolean [num_tokens]; True for filled rows of this forward.

        With chunked prefill the forward holds one chunk of each prompt. A
        row is filled iff its absolute position is before the full prompt's
        cut. ``seq_lens`` (full length) and ``query_start_loc`` (this
        forward's packed rows) give that per sequence, so a chunk that
        crosses the cut is mixed. Without metadata, fall back to the plan's
        prefix, which is exact only when the forward is the whole prompt.
        """
        if self.plan is None or not self.plan.active:
            return None
        import torch

        device = self.split_state.device if self.split_state is not None else "cpu"
        mask = torch.zeros(int(num_tokens), dtype=torch.bool, device=device)
        lens, loc = self.seq_lens, self.query_start_loc
        if (
            lens is not None
            and loc is not None
            and positions is not None
            and len(loc) == len(lens) + 1
            and int(loc[-1]) == int(num_tokens)
        ):
            # ``positions`` are absolute; a row is filled iff its position is
            # before its sequence's cut. A chunk that crosses the cut is mixed,
            # and a decode row (position = length - 1) is never filled.
            tail = self.cfg.tail
            cut = [int(length) - tail for length in lens.tolist()]
            pos = positions.detach().cpu().tolist()
            for seq in range(len(lens)):
                start, end = int(loc[seq]), int(loc[seq + 1])
                for row in range(start, end):
                    if pos[row] < cut[seq]:
                        mask[row] = True
            return mask
        end = min(int(self.plan.filled_end), int(num_tokens))
        mask[:end] = True
        return mask


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
    exact. That is deliberate. This draft does not approximate without a
    fitted map.
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
    session = Session(cfg, block)
    session.load_maps()
    causal_lm.model._ridgefill = session
