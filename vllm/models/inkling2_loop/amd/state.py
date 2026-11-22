# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-(layer, loop-pass) paged state of Inkling-2: attention KV and conv taps.

Every decoder layer visit owns two cache owners registered in the static
forward context, so the loop passes of layers 6-17 get independent caches
(48 owners for 24 layers x 3 passes of the core):

* :class:`Inkling2KVOwner`   - K|V of one attention visit
  (``FullAttentionSpec`` or ``SlidingWindowSpec``), page ``[nb, H, bs, 2*D]``.
* :class:`Inkling2ConvOwner` - raw conv inputs of the four short convolutions
  (k, v, attn-output, mlp-output) packed head-major as a position-addressed ring
  of ``S`` rows per cache block (``Inkling2ConvSpec`` in :mod:`.conv_state`, one
  state per block, model dtype).  Storing raw *inputs* of the last ``k-1`` tokens
  (not a rolling state) keeps the cache valid for chunked prefill, prefix-cache
  hits and rejected draft tokens.

Both owners share one metadata builder; model code reads the metadata with
``attn_metadata[owner.prefix]``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from .conv_state import Inkling2ConvSpec, conv_ring_rows

# Stream order inside the packed per-head conv width.
STREAM_K, STREAM_V, STREAM_ATTN, STREAM_MLP = 0, 1, 2, 3

#: ``VLLM_INKLING2_PATH=reference`` runs the eager torch reference path (the oracle of
#: the tests) on a GPU, ``hip`` insists on the HIP kernels; unset: HIP kernels on a
#: GPU and the reference path on the CPU.  There is never a silent fallback from one
#: to the other.
PATH_ENV = "VLLM_INKLING2_PATH"


def use_hip_ops(device: torch.device | str | None = None) -> bool:
    """Whether the model serves with the native HIP kernels (``csrc/rocm``)."""
    want = os.environ.get(PATH_ENV, "").strip().lower()
    if want not in ("", "hip", "reference"):
        raise ValueError(f"{PATH_ENV} must be 'hip' or 'reference', got {want!r}")
    dev = torch.device(device if device is not None else torch.get_default_device())
    if want == "reference":
        return False
    if dev.type != "cuda":
        if want == "hip":
            raise RuntimeError(f"{PATH_ENV}=hip needs a ROCm device, got {dev}")
        return False
    return True


@dataclass
class Inkling2Metadata(AttentionMetadata):
    block_table: torch.Tensor  # [R, max_blocks] int32
    slot_mapping: torch.Tensor  # [T] int64, -1 => padding
    seq_lens: torch.Tensor  # [R] int32 total context incl. this step
    query_start_loc: torch.Tensor  # [R + 1] int32
    seq_idx: torch.Tensor  # [T] int32 request of each token
    num_actual_tokens: int
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    # split-K factors of ``inkling2_attention`` per (kv heads, window); the layers of
    # a step share the metadata, so the host-side helper runs once per distinct shape
    splits: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)


class SeqIdxScratch:
    """Persistent buffers of the HIP ``token -> request`` map (no per-step allocation).

    ``torch.ops._rocm_C.build_qsa_metadata_rdna2`` is the engine's device-side
    request-of-token search; with ``compress_ratio=1`` and no ring it only writes
    ``token_to_req`` (tokens past the mapped ones read as request 0, which is harmless:
    padded tokens carry ``slot_mapping = -1`` and the conv / KV ops skip them) plus two
    per-token scratch arrays that Inkling-2 does not use.
    """

    def __init__(self, max_tokens: int, device: torch.device | str) -> None:
        from vllm import _custom_ops  # noqa: F401  (registers torch.ops._rocm_C)

        if not hasattr(torch.ops._rocm_C, "build_qsa_metadata_rdna2"):
            raise RuntimeError(
                "Inkling-2 serving needs torch.ops._rocm_C.build_qsa_metadata_rdna2 "
                "(csrc/rocm/qsa_metadata_rdna2.cu is not in this build)"
            )
        self.max_tokens = max_tokens
        self.token_to_req = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        self._positions = torch.empty(max_tokens, dtype=torch.int64, device=device)
        self._visible = torch.empty(max_tokens, dtype=torch.int32, device=device)
        self._slots = torch.empty(max_tokens, dtype=torch.int64, device=device)

    def fill(self, c: CommonAttentionMetadata) -> torch.Tensor:
        num_tokens = c.num_actual_tokens
        if num_tokens > self.max_tokens:
            raise ValueError(
                f"{num_tokens} tokens exceed the {self.max_tokens} the metadata "
                "scratch was sized for (scheduler max_num_batched_tokens)"
            )
        num_reqs = c.num_reqs
        if num_tokens and num_reqs:
            torch.ops._rocm_C.build_qsa_metadata_rdna2(
                c.query_start_loc[: num_reqs + 1],
                c.seq_lens[:num_reqs],
                c.slot_mapping,
                c.block_table_tensor,
                self.token_to_req,
                self._positions,
                self._visible,
                self._slots,
                None,
                num_reqs,
                int(c.query_start_loc_cpu[num_reqs]),
                num_tokens,
                0,
                math.ceil(math.log2(num_reqs)),
                0,
                1,
                1,
                1,
                0,
            )
        return self.token_to_req[:num_tokens]


def make_metadata(
    c: CommonAttentionMetadata, scratch: SeqIdxScratch | None
) -> Inkling2Metadata:
    """Per-group metadata of one step.  ``scratch`` (HIP path) fills the persistent
    ``seq_idx`` buffer on the device; without it (reference path) torch builds it."""
    num_reqs = c.num_reqs
    num_tokens = c.num_actual_tokens
    qsl = c.query_start_loc[: num_reqs + 1]
    if scratch is None:
        q_lens = (qsl[1:] - qsl[:-1]).to(torch.int64)
        seq_idx = torch.repeat_interleave(
            torch.arange(num_reqs, device=qsl.device, dtype=torch.int32),
            q_lens,
            output_size=num_tokens,
        )
    else:
        seq_idx = scratch.fill(c)
    return Inkling2Metadata(
        block_table=c.block_table_tensor[:num_reqs],
        slot_mapping=c.slot_mapping[:num_tokens],
        seq_lens=c.seq_lens[:num_reqs],
        query_start_loc=qsl,
        seq_idx=seq_idx,
        num_actual_tokens=num_tokens,
        num_reqs=num_reqs,
        max_query_len=c.max_query_len,
        max_seq_len=c.max_seq_len,
    )


class Inkling2MetadataBuilder(AttentionMetadataBuilder[Inkling2Metadata]):
    # No graph capture yet: the HIP ops are capturable but the metadata tensors the
    # runner hands over are not persistent (see docs/rdna2/inkling2-port-spec.md).
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._scratch: SeqIdxScratch | None = None
        if use_hip_ops(device):
            self._scratch = SeqIdxScratch(
                vllm_config.scheduler_config.max_num_batched_tokens, device
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Inkling2Metadata:
        return make_metadata(common_attn_metadata, self._scratch)


class Inkling2Backend(AttentionBackend):
    """Metadata-only backend: attention and conv run in the model."""

    @staticmethod
    def get_name() -> str:
        return "INKLING2_LOOP"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("Inkling-2 attention runs inside the model.")

    @staticmethod
    def get_builder_cls() -> type[Inkling2MetadataBuilder]:
        return Inkling2MetadataBuilder

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Block-outermost layouts let the planner pack the conv-state pages (a
        # fraction of a KV page) next to the KV pages of a block instead of padding
        # every page to the largest one.
        return (KVCacheLayout.BLNHC, KVCacheLayout.BLHNC)


def _register(owner: nn.Module, prefix: str, vllm_config: VllmConfig) -> None:
    ctx = vllm_config.compilation_config.static_forward_context
    if prefix in ctx:
        raise ValueError(f"Duplicate layer name: {prefix}")
    ctx[prefix] = owner


class Inkling2KVOwner(nn.Module, AttentionLayerBase):
    """K|V cache of one attention visit.  Page layout ``[nb, H, bs, 2*D]``."""

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        vllm_config = get_current_vllm_config()
        self._dtype = vllm_config.model_config.dtype
        self.kv_cache = torch.tensor([])  # replaced by bind_kv_cache
        _register(self, prefix, vllm_config)

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2Backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = vllm_config.cache_config.block_size
        if self.sliding_window is not None:
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                dtype=self._dtype,
                sliding_window=self.sliding_window,
            )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=self._dtype,
        )

    @property
    def block_size(self) -> int:
        return self.kv_cache.shape[2]

    def views(self) -> tuple[torch.Tensor, torch.Tensor]:
        """K and V as ``[nb, bs, H, D]`` views of the bound page."""
        k, v = self.kv_cache.transpose(1, 2).split(self.head_dim, dim=-1)
        return k, v


class Inkling2ConvOwner(nn.Module, AttentionLayerBase):
    """Conv-input state of one layer visit: a ring of raw-input rows per block.

    ``H = num_kv_heads`` heads; each of the four streams owns a column range of the
    ``W = row_width`` columns of every head (a stream of ``C`` channels is viewed
    ``[H, C/H]``).  ``state`` is the bound cache as ``[nb, H, S, W]`` in the model dtype
    (fp16).  At TP=1 every conv input is an fp16 GEMM output, so the state is exact; at
    TP>1 the post-attention stream is the fp32 all-reduce of the o_proj partials: the
    conv reads the current chunk's taps exactly and stores them rounded to fp16, so the
    history of a later chunk or step comes back rounded to nearest even.  Position
    ``p`` of a request lives in the block ``block_table[req, p // block_size]`` at
    ring row ``p % S``.
    """

    def __init__(
        self,
        *,
        num_kv_heads: int,
        stream_channels: tuple[int, int, int, int],
        kernel_size: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.num_kv_heads = num_kv_heads
        self.kernel_size = kernel_size
        widths = []
        for c in stream_channels:
            assert c % num_kv_heads == 0, (c, num_kv_heads)
            widths.append(c // num_kv_heads)
        offs = [0]
        for w in widths[:-1]:
            offs.append(offs[-1] + w)
        self.stream_ranges = tuple(zip(offs, widths))
        self.row_width = sum(widths)
        vllm_config = get_current_vllm_config()
        state_dtype = vllm_config.model_config.dtype
        assert isinstance(state_dtype, torch.dtype), state_dtype  # config-resolved
        self.state_dtype = state_dtype
        self.block_size = vllm_config.cache_config.block_size
        self.ring_rows = conv_ring_rows(vllm_config, kernel_size)
        self.state = torch.tensor([])
        self.kv_cache = torch.tensor([])  # replaced by bind_kv_cache
        _register(self, prefix, vllm_config)

    @property
    def kv_cache(self) -> torch.Tensor:
        return self._kv_cache

    @kv_cache.setter
    def kv_cache(self, cache: torch.Tensor) -> None:
        # The runner binds (and clears) the page view ``[nb, H, 1, S * W]``; the
        # ops address the ring rows, so keep the ``[nb, H, S, W]`` view of it.
        self._kv_cache = cache
        self.state = (
            cache
            if cache.numel() == 0
            else cache.view(
                cache.shape[0], self.num_kv_heads, self.ring_rows, self.row_width
            )
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2Backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # the spec fixes the geometry the cache is bound with
        self.block_size = block_size = vllm_config.cache_config.block_size
        self.ring_rows = conv_ring_rows(vllm_config, self.kernel_size)
        return Inkling2ConvSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.row_width,
            head_size_v=0,  # the page is one packed state per block
            dtype=self.state_dtype,
            sliding_window=self.kernel_size,
            tokens_per_state=block_size,
            state_content_bytes=self.ring_rows
            * self.row_width
            * self.state_dtype.itemsize,
        )


def _ring_rows(
    owner: Inkling2ConvOwner,
    md: Inkling2Metadata,
    req: torch.Tensor,
    pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(physical block, ring row) of position ``pos`` of request ``req``."""
    blk = md.block_table[req, (pos // owner.block_size).clamp(min=0)].long()
    return blk, pos.clamp(min=0) % owner.ring_rows


def conv_taps(
    owner: Inkling2ConvOwner,
    stream: int,
    x: torch.Tensor,
    positions: torch.Tensor,
    md: Inkling2Metadata,
) -> torch.Tensor:
    """Previous ``k-1`` inputs of every token and the cache write of ``x``.

    ``x`` is ``[T, C]`` fp32 (this step's raw conv inputs).  Returns
    ``taps [k, T, C]`` with ``taps[lag]`` = input ``lag`` tokens back (``lag=0``
    is ``x``); positions before the sequence start are zero.  The history is read
    before this step's inputs are written (:func:`conv_write`), as the HIP ops do.
    """
    T, C = x.shape
    k = owner.kernel_size
    o0, w = owner.stream_ranges[stream]
    cache = owner.state  # [nb, H, S, W]
    seq_idx = md.seq_idx.long()
    first = md.query_start_loc[:-1].long()[seq_idx]  # [T] first row of the request
    row = torch.arange(T, device=x.device)
    p0 = positions[first]  # position of the request's first token this step
    taps = [x]
    for lag in range(1, k):
        src_pos = positions - lag
        in_chunk = src_pos >= p0
        valid = src_pos >= 0
        blk, r = _ring_rows(owner, md, seq_idx, src_pos)
        # In-chunk history comes from x, older history from the cache.
        hist_cache = cache[blk, :, r, o0 : o0 + w].reshape(T, C).to(x.dtype)
        hist_x = x[(row - lag).clamp(min=0)]
        t = torch.where(in_chunk[:, None], hist_x, hist_cache)
        taps.append(torch.where(valid[:, None], t, torch.zeros_like(t)))
    conv_write(owner, stream, x, md)
    return torch.stack(taps, dim=0)


def conv_write(
    owner: Inkling2ConvOwner,
    stream: int,
    x: torch.Tensor,
    md: Inkling2Metadata,
) -> None:
    """Write the raw inputs ``x [T, C]`` of ``stream`` into the ring: the last ``S``
    positions of every (block, chunk) intersection (one writer per ring row)."""
    T = x.shape[0]
    H = owner.num_kv_heads
    o0, w = owner.stream_ranges[stream]
    cache = owner.state
    B, S = owner.block_size, owner.ring_rows
    seq_idx = md.seq_idx.long()
    qs = md.query_start_loc[:-1].long()[seq_idx]
    end = md.seq_lens.long()[seq_idx]  # context length after this step
    pos = (
        end
        - (md.query_start_loc[1:].long()[seq_idx] - qs)
        + (torch.arange(T, device=x.device) - qs)
    )
    keep = (pos + S >= torch.minimum(end, (pos // B + 1) * B)) & (md.seq_idx >= 0)
    blk, r = _ring_rows(owner, md, seq_idx, pos)
    cache[blk[keep], :, r[keep], o0 : o0 + w] = x[keep].view(-1, H, w).to(cache.dtype)


def kv_write(
    owner: Inkling2KVOwner,
    k: torch.Tensor,
    v: torch.Tensor,
    md: Inkling2Metadata,
) -> None:
    """Scatter this step's K/V ``[T, H, D]`` into the paged cache."""
    cache = owner.kv_cache  # [nb, H, bs, 2D]
    bs = cache.shape[2]
    D = owner.head_dim
    slot = md.slot_mapping
    ok = slot >= 0
    s = slot.clamp(min=0)
    blk, off = s // bs, s % bs
    old = cache[blk, :, off]
    new = torch.cat([k, v], dim=-1).to(cache.dtype)
    cache[blk, :, off] = torch.where(ok[:, None, None], new, old)
    del D


def kv_read(
    owner: Inkling2KVOwner,
    md: Inkling2Metadata,
    req: int,
    lo: int,
    hi: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K and V of positions ``[lo, hi)`` of request ``req``: ``[n, H, D]``."""
    cache = owner.kv_cache
    bs = cache.shape[2]
    D = owner.head_dim
    pos = torch.arange(lo, hi, device=cache.device)
    blk = md.block_table[req, pos // bs].long()
    kv = cache[blk, :, pos % bs]  # [n, H, 2D]
    return kv[..., :D], kv[..., D:]
