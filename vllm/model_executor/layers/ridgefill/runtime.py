# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind the draft to a Flash-Next module. Called only when the flag is on.

The session records the split boundary, builds the row plan, and loads the
fitted fp16 maps. The model's forward captures the four-stream residual at
layer ``split`` and, for each late layer, applies that layer's map to the
split state. Exact rows, including the tail, run the layer as they do now.

The row plan is built **once per prefill forward**, not once per layer, and
it is built *outside* the compiled region: ``Qwen4ExpForCausalLM.forward``
calls ``begin_forward`` and passes the two index tensors down. A forward with
no filled rows passes ``None``, so its compiled graph is the flag-off graph.
Decode never fills and never syncs: ``num_prefills`` is a plain int on the
metadata, so the two ``.cpu()`` reads only happen on a prefill.

The fill steps themselves are opaque custom ops (``ops.py``), the PLE
pattern: ``fullgraph`` refuses a ``torch._dynamo.disable`` split point, so
each step is one node in the graph and its body runs outside Inductor.

Two filled-row paths exist. The legacy path runs attention and GDN from the
prediction and skips the MoE. ``VLLM_RIDGEFILL_CACHE_ONLY=1`` instead gives
filled rows only their cache side effects (QSA K/V plus indexer keys, GDN
state) and runs the layer's output work on the compacted exact rows. The
cache-write ledger in ``rows`` proves the two paths write the same rows to the
same slots in the same order. Default off until the compiled A/B runs.
"""

from __future__ import annotations

from typing import Any

from vllm.model_executor.layers.ridgefill.config import (
    RidgeConfig,
    RidgeFillConfigError,
    cache_only,
    enabled,
    load_config,
)
from vllm.model_executor.layers.ridgefill.loader import require_bundle
from vllm.model_executor.layers.ridgefill.policy import (
    RowPlan,
    apply_v1_guards,
    plan_prompt,
)
from vllm.model_executor.layers.ridgefill.rows import (
    RowSplit,
    RowSplitError,
    assert_partition,
    build_row_split,
    build_row_split_from_positions,
    cuts_from_lengths,
    plan_cache_stream,
    prefix_split,
)


class Session:
    """Per-model draft state and the fitted late-layer maps."""

    def __init__(
        self, cfg: RidgeConfig, block_size: int, compress_ratio: int = 0
    ) -> None:
        self.cfg = cfg
        self.block_size = block_size
        # The QSA indexer's compression ratio. Zero means unknown, which only
        # disables the compressed-group cut alignment, never a fill.
        self.compress_ratio = max(0, int(compress_ratio))
        # Read once per session: the flag cannot change under a serve.
        self.cache_only = cache_only()
        self.saw_split = False
        self.boundary_width: int | None = None
        self.plan: RowPlan | None = None
        # Built once per forward in ``begin_forward``; reused by every layer.
        self.rows: RowSplit | None = None
        self.maps: dict[int, tuple] = {}
        # ``torch`` is lazy in this package; the state is a CPU tensor.
        self.split_state: Any = None
        self.overflow = False
        # Device tensors materialized once in ``capture_split``: the filled row
        # mask and the two index tensors the layers select with.
        self.fill_mask_t = None
        self.filled_index = None
        self.exact_index = None
        # The split state's filled rows in map dtype, gathered once per forward
        # so each layer's map GEMM runs on F rows instead of T.
        self.fill_state_mm = None
        self._fill_state_dtype = None
        # Per-sequence lengths of this forward, from the attention metadata.
        # With chunked prefill ``num_tokens`` is the chunk, not the prompt.
        self.seq_lens = None
        self.query_start_loc = None
        self.num_prefills = 0

    def _seq_metadata(self):
        """``(seq_lens, query_start_loc, num_prefills)`` for this forward.

        Chunked prefill gives each layer the chunk's rows. The tail is the
        last ``tail`` tokens of the full prompt, so the cut needs the full
        sequence length. QSA metadata carries it; GDN metadata does not, so
        any QSA entry in this forward serves.

        ``num_prefills`` is None when no attention metadata is reachable. A
        decode forward returns ``(None, None, 0)`` without reading a device
        tensor: ``num_prefills`` is a plain int on the metadata, so the two
        ``.cpu()`` syncs only happen on a prefill.
        """
        try:
            from vllm.forward_context import get_forward_context

            metadata = get_forward_context().attn_metadata
        except Exception:
            return None, None, None
        entries = metadata if isinstance(metadata, list) else [metadata]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for value in entry.values():
                lens = getattr(value, "seq_lens", None)
                loc = getattr(value, "query_start_loc", None)
                if lens is None or loc is None:
                    continue
                prefills = getattr(value, "num_prefills", None)
                if prefills is not None and int(prefills) == 0:
                    # Decode: no row can be filled and no host read is needed.
                    return None, None, 0
                if prefills is None:
                    # QSA metadata may not carry ``num_prefills``; a query
                    # longer than one row is a prefill row either way.
                    steps = loc[1:] - loc[:-1]
                    prefills = int((steps > 1).sum().item())
                return lens.detach().cpu(), loc.detach().cpu(), int(prefills)
        return None, None, None

    def begin_forward(self, positions, num_tokens: int, metadata=None):
        """Build this forward's row plan once, before the compiled model.

        The fill decision must be made outside the compiled region: a forward
        with no filled rows passes ``None`` down and takes the untouched
        compiled path, so its graph is the flag-off graph. ``positions``,
        when given, decides each row's absolute position; without it the
        split uses the scheduler metadata.

        ``metadata`` overrides the forward context's ``(seq_lens,
        query_start_loc, num_prefills)``; the CPU proofs inject it.
        """
        self.saw_split = True
        align = self.cfg.apc == "v2" and self.block_size > 1
        if metadata is None:
            lens, loc, prefills = self._seq_metadata()
        else:
            lens, loc, prefills = metadata
        if prefills is None:
            # No metadata this forward: fall back to the token count. A
            # forward of one token row is a decode step; every longer forward
            # is the prefill we want.
            self.seq_lens = self.query_start_loc = None
            self.num_prefills = 0
            prefill = num_tokens > 1
        else:
            self.seq_lens, self.query_start_loc = lens, loc
            self.num_prefills = prefills
            prefill = prefills > 0
        full_tokens = num_tokens
        if prefill and self.seq_lens is not None and len(self.seq_lens) > 0:
            full_tokens = max(full_tokens, int(self.seq_lens.max().item()))
        if not prefill:
            self.plan = plan_prompt(0, split=self.cfg.split, tail=self.cfg.tail)
            self.rows = prefix_split(num_tokens, 0)
            self._reset_fill_tensors()
            return None, None
        self.plan = plan_prompt(
            full_tokens,
            split=self.cfg.split,
            tail=self.cfg.tail,
            block_size=self.block_size,
            align=align,
        )
        self.rows = self._build_rows(num_tokens, positions)
        # A fill whose row set cannot be partitioned is a bug, not a fallback:
        # the cache-write-only path would drop or double-write a row.
        assert_partition(self.rows)
        self._reset_fill_tensors()
        return self._build_index_tensors(positions.device)

    def _build_index_tensors(self, device):
        """Upload this forward's row indices once, on the host.

        The plan is contiguous ranges per request, so the two index tensors
        are built from Python lists here and never touched again: no per-row
        device write, no per-layer host read, no boolean gather.
        """
        rows = self.rows
        if rows is None or not rows.active:
            return None, None
        import torch

        self.filled_index = torch.tensor(
            list(rows.filled_idx), dtype=torch.long, device=device
        )
        self.exact_index = torch.tensor(
            list(rows.exact_idx), dtype=torch.long, device=device
        )
        return self.filled_index, self.exact_index

    def _reset_fill_tensors(self) -> None:
        """Drop the previous forward's device tensors before rebuilding."""
        self.fill_mask_t = None
        self.filled_index = None
        self.exact_index = None
        self.fill_state_mm = None
        self._fill_state_dtype = None

    def _build_rows(self, num_tokens: int, positions=None) -> RowSplit:
        """The row split for this forward. Positions are the authority.

        ``build_row_split`` derives the same answer from ``seq_lens`` and
        ``query_start_loc`` without reading a device tensor; the CPU proof
        compares the two. Positions stay the authority because Flash-Next can
        carry 3-D M-RoPE positions for vision tokens. When neither is usable,
        fall back to the plan's prefix, which is exact only when the forward is
        the whole prompt.
        """
        lens = self.seq_lens
        loc = self.query_start_loc
        if lens is not None and loc is not None and len(loc) == len(lens) + 1:
            lengths = [int(v) for v in lens.tolist()]
            starts = [int(v) for v in loc.tolist()]
            if starts and starts[0] == 0 and starts[-1] == int(num_tokens):
                cuts = cuts_from_lengths(
                    lengths,
                    tail=self.cfg.tail,
                    block_size=self.block_size,
                    align=(self.cfg.apc == "v2" or self.cache_only)
                    and self.block_size > 1,
                    # A compressed group must not straddle the two passes.
                    compress_ratio=self.compress_ratio if self.cache_only else 0,
                )
                try:
                    if positions is not None:
                        # Positions are the authority: Flash-Next can carry
                        # 3-D M-RoPE positions for vision tokens, which the
                        # text arithmetic does not reproduce. The arithmetic
                        # path below is the fallback and the CPU proof shows
                        # the two agree on plain text positions.
                        host = positions.detach()
                        if host.dim() > 1:
                            # M-RoPE sends one row per axis; text positions
                            # repeat, so the first axis is the absolute one.
                            host = host[0]
                        return build_row_split_from_positions(
                            seq_lens=lengths,
                            query_start_loc=starts,
                            cuts=cuts,
                            positions=[int(v) for v in host.tolist()],
                        )
                    return build_row_split(
                        seq_lens=lengths, query_start_loc=starts, cuts=cuts
                    )
                except RowSplitError:
                    pass
        end = 0 if self.plan is None else int(self.plan.filled_end)
        return prefix_split(num_tokens, end)

    def fill_tensors(self):
        """``(mask, filled_index, exact_index)`` or ``(None, None, None)``.

        The mask is a test and debug convenience: the model forward selects
        with the two index tensors only, so the hot path never builds a
        per-row mask. A None triple means this forward has no filled rows.
        """
        if self.rows is None or not self.rows.active:
            return None, None, None
        if self.fill_mask_t is None:
            import torch

            device = (
                self.filled_index.device if self.filled_index is not None else "cpu"
            )
            self.fill_mask_t = torch.tensor(
                self.rows.filled_mask(), dtype=torch.bool, device=device
            )
        return self.fill_mask_t, self.filled_index, self.exact_index

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
        """Keep the four-stream residual and gather the filled rows once.

        The row indices were uploaded in ``begin_forward``; this only stores
        the split state, runs the one overflow check, and gathers the filled
        rows in map dtype so each late layer's map GEMM runs on F rows rather
        than T with the cast paid once per forward.
        """
        import torch

        width = int(hidden_states.shape[-1])
        self.boundary_width = width
        if width != self.cfg.din:
            raise RidgeFillConfigError(
                f"split boundary width {width} != map din {self.cfg.din}. "
                "Flash-Next's four streams are 10240 wide on every TP rank."
            )
        self.split_state = hidden_states.detach()
        self.overflow = False
        if self.plan is not None and self.plan.active:
            # gfx1030 has no fp8 and bf16 is the overflow fallback. One check
            # per prefill forward; decode never activates the plan.
            self.overflow = bool((self.split_state.abs() > 65504).any())
        rows = self.rows
        if rows is None or not rows.active or self.filled_index is None:
            return
        dtype = torch.bfloat16 if self.map_dtype_is_bf16() else torch.float16
        self.fill_state_mm = self.split_state.index_select(0, self.filled_index).to(
            dtype
        )
        self._fill_state_dtype = dtype

    def map_dtype_is_bf16(self) -> bool:
        """True when the map runs in bf16: explicit request or fp16 overflow."""
        return self.overflow or self.cfg.map_dtype == "bf16"

    def predict(self, layer_idx: int, *, filled_only: bool = False, out_dtype=None):
        """Apply layer ``layer_idx``'s map to the split state, or None.

        ``filled_only`` runs the GEMM on the gathered filled rows only, so a
        late layer costs F rows instead of T. The result is in the map dtype;
        the caller casts only when the destination differs.
        """
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
        dtype = torch.bfloat16 if self.map_dtype_is_bf16() else torch.float16
        if a.dtype != dtype:
            a = a.to(dtype)
            b = b.to(dtype)
        if a.device != x_s.device:
            a = a.to(x_s.device)
            b = b.to(x_s.device)
            self.maps[layer_idx] = (a, b)
        if filled_only and self.rows is not None and self.rows.active:
            if self.fill_state_mm is None:
                self.fill_state_mm = x_s.index_select(0, self.filled_index).to(dtype)
            # One gather + cast per forward, shared by every late layer.
            out = torch.nn.functional.linear(self.fill_state_mm, a, b)
        else:
            # fp16 GEMM, fp32 accumulate. Returning the GEMM's own dtype lets
            # the caller mix it with block_input without a full-width cast.
            out = torch.nn.functional.linear(x_s.to(dtype), a, b)
        return out if out_dtype is None else out.to(out_dtype)

    def predict_rows(self, x_s, filled_index, layer_idx: int, dout: int):
        """The map GEMM on the filled rows, for the opaque predict op.

        Returns ``[F, dout]`` in ``x_s``'s dtype, or an empty ``[0, dout]``
        when the layer has no fitted map. ``capture_split`` gathered and cast
        the filled rows once per forward, so a late layer pays the GEMM only.
        """
        pair = self.maps.get(layer_idx)
        if pair is None:
            return x_s.new_empty((0, dout))
        import torch

        a, b = pair
        dtype = torch.bfloat16 if self.map_dtype_is_bf16() else torch.float16
        if a.dtype != dtype:
            a = a.to(dtype)
            b = b.to(dtype)
        if a.device != x_s.device:
            a = a.to(x_s.device)
            b = b.to(x_s.device)
            self.maps[layer_idx] = (a, b)
        rows = self.fill_state_mm
        if rows is None:
            rows = x_s.index_select(0, filled_index).to(dtype)
            self.fill_state_mm = rows
        out = torch.nn.functional.linear(rows, a, b)
        return out.to(x_s.dtype)

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
        # The hot path: ``capture_split`` already built this forward's mask.
        # Returning it costs no host read and no per-row device write.
        cached = self.fill_mask_t
        if cached is not None and cached.shape[0] == int(num_tokens):
            return cached
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
            pos = positions.detach().cpu()
            if pos.dim() > 1:
                # Qwen-Flash-Next passes M-RoPE positions, one channel row per
                # token axis (3 x num_tokens). Text positions repeat across the
                # rows, so the first channel is the absolute position.
                pos = pos[0]
            pos = pos.tolist()
            for seq in range(len(lens)):
                start, end = int(loc[seq]), int(loc[seq + 1])
                for row in range(start, end):
                    if int(pos[row]) < int(cut[seq]):
                        mask[row] = True
            return mask
        end = min(int(self.plan.filled_end), int(num_tokens))
        mask[:end] = True
        return mask

    def cache_streams(self, *, slots, positions):
        """Host ledger of this forward's cache writes, split by pass.

        ``slots`` and ``positions`` are per-row host sequences. The returned
        streams name which pass writes each row for each cache the model keeps:
        the QSA main K/V cache and the indexer's raw and compressed key caches.
        The compressed cache is a ring: a group's store lands in whichever pass
        holds the group's last token. This is the object the CPU proof compares
        between the full path and the split path.
        """
        if self.rows is None:
            return ()
        return (
            plan_cache_stream("main_kv", self.rows, slots=slots, positions=positions),
            plan_cache_stream(
                "indexer_raw", self.rows, slots=slots, positions=positions
            ),
            plan_cache_stream(
                "indexer_compressed",
                self.rows,
                slots=slots,
                positions=positions,
                ring=True,
            ),
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
    # The QSA indexer's group size drives the compressed-cache cut alignment.
    # A missing attribute leaves it unknown (0), which only widens the exact
    # tail's alignment unit to the block size.
    text_config = getattr(vllm_config.model_config, "hf_text_config", None)
    ratio = int(getattr(text_config, "indexer_compress_ratio", 0) or 0)
    session = Session(cfg, block, compress_ratio=ratio)
    session.load_maps()
    causal_lm.model._ridgefill = session
    causal_lm.model._ridgefill_map_layers = frozenset(session.maps)
    # The fill steps are opaque custom ops so the compiled forward keeps one
    # node per step instead of a Dynamo graph break. The ops read the session
    # and the decoder layers from these registries.
    from vllm.model_executor.layers.ridgefill import ops

    ops.set_session(session)
    for layer in causal_lm.model.layers:
        ops.register_layer(layer.layer_name, layer)
