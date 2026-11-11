# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side construction of the ``hx`` device contexts (hub and server).

The ``hx`` kernels read every parameter of a step from a small device tensor, the
*ctx* (``int64[HX_CTX_WORDS]``, layout owned by ``hx_core.cuh``). Python never
packs it: ``ops.hx_ctx_new`` makes a host image, ``ops.hx_ctx_set`` fills named
fields, ``ops.hx_ctx_register`` uploads it and records the launch geometry in
the C++ host registry. This module wraps that sequence and the allocations the
protocol needs (exported VRAM, IPC import, clock calibration, plan tables).
"""

import os
import struct
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.hetero_moe import hx_proto as P

# hx_alloc kinds (hx_ops.cu)
ALLOC_KINDS = {"coarse": 0, "uncached": 1, "finegrained": 2}
ALLOC_NAMES = {0: "coarse", 1: "uncached", 2: "finegrained"}

ROLE_HUB = 1
ROLE_SERVER = 2

_OPS = None


def ops():
    """``vllm._custom_ops`` (imported on first use so CPU-only tools can import
    this module; tests set ``_OPS`` to a fake).

    ``VLLM_HX_DEV_LIB`` names the ``hx_dev.so`` of ``tools/rdna/hetero_moe/
    build_hx_dev.sh``, loaded before the first op call: a development shortcut
    for trees whose ``_rocm_C`` was built without the hx ops. A ``_rocm_C`` that
    has them must not be combined with it (the registrations would collide)."""
    global _OPS
    if _OPS is None:
        dev_lib = os.environ.get("VLLM_HX_DEV_LIB")
        if dev_lib:
            torch.ops.load_library(dev_lib)
        from vllm import _custom_ops

        _OPS = _custom_ops
    return _OPS


def _i64(value: int) -> int:
    """A u64 as the signed int64 the torch op schema takes (same bits)."""
    value &= P.U64_MAX
    return value - (1 << 64) if value >= (1 << 63) else value


@dataclass(frozen=True)
class Exported:
    """VRAM allocated by ``hx_alloc``: zeroed, IPC-exportable."""

    ptr: int
    nbytes: int
    kind: int  # ALLOC_NAMES
    handle: bytes  # hipIpcMemHandle_t, 64 bytes

    @property
    def kind_name(self) -> str:
        return ALLOC_NAMES[self.kind]

    def handle_hex(self) -> str:
        return self.handle.hex()


def alloc_exported(nbytes: int, kind: str = "uncached") -> Exported:
    """Zeroed exported device memory. Receive buffers must be ``uncached`` (the
    other kinds are for T1 measurements and never substituted silently)."""
    raw = ops().hx_alloc(nbytes, ALLOC_KINDS[kind]).numpy().tobytes()
    ptr, size, used = struct.unpack_from("<QQQ", raw, 0)
    return Exported(ptr, size, used, bytes(raw[24 : 24 + 64]))


def free_exported(buf: Exported) -> None:
    ops().hx_free(buf.ptr)


def open_handle(handle: bytes) -> int:
    """Map a peer's exported allocation; returns the local device pointer."""
    if len(handle) != 64:
        raise ValueError(f"IPC handle must be 64 bytes, got {len(handle)}")
    tensor = torch.frombuffer(bytearray(handle), dtype=torch.uint8)
    return int(ops().hx_open(tensor))


def close_handle(ptr: int) -> None:
    ops().hx_close(ptr)


def calibrate_clock(interval_ms: int = 150) -> tuple[int, float]:
    """``(ticks per ms measured against the host clock, device attribute kHz)``.
    The measured rate is the one used for timeouts; the attribute is logged."""
    res = ops().hx_calibrate(interval_ms)
    return int(round(float(res[0]))), float(res[1])


@dataclass(frozen=True)
class Geometry:
    """Shapes the transport is sized for (identical on hub and servers)."""

    rows_cap: int = P.HX_DEFAULT_ROWS_CAP
    topk: int = P.HX_DEFAULT_TOPK
    hidden: int = P.HX_DEFAULT_HIDDEN
    nslots: int = P.HX_DEFAULT_NSLOTS
    num_experts: int = P.HX_MAX_EXPERTS
    nlayers: int = 48

    def inbox(self) -> P.InboxLayout:
        return P.inbox_layout(self.rows_cap, self.topk, self.hidden, self.nslots)

    def reply(self, nsrv: int) -> P.ReplyLayout:
        return P.reply_layout(self.rows_cap, self.hidden, self.nslots, nsrv)

    def as_dict(self) -> dict[str, int]:
        return {
            "rows_cap": self.rows_cap,
            "topk": self.topk,
            "hidden": self.hidden,
            "nslots": self.nslots,
            "num_experts": self.num_experts,
            "nlayers": self.nlayers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Geometry":
        return cls(**{k: int(d[k]) for k in cls.__dataclass_fields__})


@dataclass
class HubTables:
    """The plan tables of one hub on its device (kept alive with the ctx)."""

    slot_map: torch.Tensor  # int16 [L, E]
    cold_srv: torch.Tensor  # int8 [L, E, MAXC]
    n_cold: torch.Tensor  # int8 [L, E]
    owned: torch.Tensor  # uint8 [L, E]
    ktab: torch.Tensor  # int16 [rows_cap + 1]
    nlayers: int
    num_experts: int


def upload_tables(tables, rows_cap: int, device) -> HubTables:
    """``tables``: ``tables.IslandTables``. Element types match the ctx fields
    (``hx_core.cuh``); ``ktab[T] = tables.k_for(T)`` (``T = 0`` unused)."""
    st = tables.stacked()
    n_layers, n_experts = st.slot_map.shape
    if n_layers > P.HX_MAX_LAYERS or n_experts > P.HX_MAX_EXPERTS:
        raise ValueError(
            f"tables have {n_layers} layers x {n_experts} experts; "
            f"hx supports {P.HX_MAX_LAYERS} x {P.HX_MAX_EXPERTS}"
        )
    if st.cold_srv.shape[2] != P.HX_MAX_COLD:
        raise ValueError(f"cold_srv has {st.cold_srv.shape[2]} copies, not 8")
    if st.cold_srv.max(initial=-1) >= P.HX_MAX_SERVERS:
        raise ValueError("cold_srv names a server index >= HX_MAX_SERVERS")
    if st.slot_map.max(initial=-1) >= P.HX_MAX_EXPERTS:
        raise ValueError("slot_map has a slot >= HX_MAX_EXPERTS")
    kvals = np.array(
        [tables.k_for(max(1, t)) for t in range(rows_cap + 1)], dtype=np.int64
    )
    if kvals.min() < 0 or kvals.max() > P.HX_MAX_EXPERTS:
        raise ValueError(f"spill K outside [0, {P.HX_MAX_EXPERTS}]")

    def up(arr: np.ndarray, dtype) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=dtype)).to(device)

    return HubTables(
        slot_map=up(st.slot_map, np.int16),
        cold_srv=up(st.cold_srv, np.int8),
        n_cold=up(st.n_cold, np.int8),
        owned=up(st.owned, np.uint8),
        ktab=up(kvals, np.int16),
        nlayers=n_layers,
        num_experts=n_experts,
    )


def _fill(image: torch.Tensor, **fields: int) -> None:
    for name, value in fields.items():
        ops().hx_ctx_set(image, name, _i64(value))


def hub_image(
    geom: Geometry,
    *,
    plan_id: str,
    hub_index: int,
    nsrv: int,
    reply: Exported,
    inbox_ptrs: list[int],
    tables: HubTables,
    ticks_per_ms: int,
    wait_ms: int,
    flush_mode: int = 0,
    layer_stats: torch.Tensor | None = None,
) -> torch.Tensor:
    """Host image of a hub ctx. ``layer_stats``: optional int64
    ``[HX_MAX_LAYERS, HX_LSTAT_COLS]`` device tensor the dispatch kernel adds its
    per-layer pair counts to (None: counters off, zero cost)."""
    if layer_stats is not None and (
        layer_stats.dtype != torch.int64
        or tuple(layer_stats.shape) != (P.HX_MAX_LAYERS, P.HX_LSTAT_COLS)
        or not layer_stats.is_contiguous()
    ):
        raise ValueError(
            f"layer_stats must be a contiguous int64 [{P.HX_MAX_LAYERS}, "
            f"{P.HX_LSTAT_COLS}] tensor"
        )
    if len(inbox_ptrs) != nsrv:
        raise ValueError(f"need {nsrv} server inbox pointers, got {len(inbox_ptrs)}")
    inbox, rep = geom.inbox(), geom.reply(nsrv)
    if reply.nbytes < rep.total_bytes:
        raise ValueError(f"reply region {reply.nbytes} B < layout {rep.total_bytes} B")
    image = ops().hx_ctx_new()
    _fill(
        image,
        role=ROLE_HUB,
        plan_hash=P.plan_hash(plan_id),
        index=hub_index,
        nsrv=nsrv,
        rows_cap=geom.rows_cap,
        topk=geom.topk,
        hidden=geom.hidden,
        nslots=geom.nslots,
        num_experts=tables.num_experts,
        nlayers=tables.nlayers,
        flush_mode=flush_mode,
        wait_ticks=wait_ms * ticks_per_ms,
        begin_wait_ticks=wait_ms * ticks_per_ms,
        ticks_per_ms=ticks_per_ms,
        slot_stride=inbox.slot_stride,
        off_ids=inbox.ids_off,
        off_w=inbox.w_off,
        y_stride=rep.y_stride,
        own_base=reply.ptr,
        hub_slot=tables.slot_map.data_ptr(),
        cold_srv=tables.cold_srv.data_ptr(),
        n_cold=tables.n_cold.data_ptr(),
        owned=tables.owned.data_ptr(),
        ktab=tables.ktab.data_ptr(),
        layer_stats=0 if layer_stats is None else layer_stats.data_ptr(),
    )
    for s, ptr in enumerate(inbox_ptrs):
        _fill(image, **{f"peer_base{s}": ptr})
    return image


def server_image(
    geom: Geometry,
    *,
    plan_id: str,
    server_index: int,
    nsrv: int,
    inbox: Exported,
    hub_reply_ptr: int,
    ids32: torch.Tensor,
    x_ws: torch.Tensor,
    w_ws: torch.Tensor,
    ticks_per_ms: int,
    wait_ms: int,
    begin_wait_ms: int,
    hub_nsrv: int,
    flush_mode: int = 0,
) -> torch.Tensor:
    """Host image of a server ctx. Workspaces in normal (cached) device memory,
    filled by ``hx_srv_wait`` from the uncached inbox slot of the request:
    ``ids32`` int32 ``[rows_cap * topk]`` (widened ids), ``x_ws`` fp16
    ``[rows_cap, hidden]``, ``w_ws`` fp32 ``[rows_cap, topk]``.
    ``hub_reply_ptr``: the hub's mapped reply region (own inbox when the hub is
    not known yet; ``register`` again once it is)."""
    inbox_l, rep = geom.inbox(), geom.reply(hub_nsrv)
    if inbox.nbytes < inbox_l.total_bytes:
        raise ValueError(f"inbox {inbox.nbytes} B < layout {inbox_l.total_bytes} B")
    if ids32.dtype != torch.int32 or ids32.numel() < geom.rows_cap * geom.topk:
        raise ValueError("ids32 must be int32[rows_cap * topk]")
    if x_ws.dtype != torch.float16 or x_ws.numel() < geom.rows_cap * geom.hidden:
        raise ValueError("x_ws must be fp16[rows_cap * hidden]")
    if w_ws.dtype != torch.float32 or w_ws.numel() < geom.rows_cap * geom.topk:
        raise ValueError("w_ws must be fp32[rows_cap * topk]")
    image = ops().hx_ctx_new()
    _fill(
        image,
        role=ROLE_SERVER,
        plan_hash=P.plan_hash(plan_id),
        index=server_index,
        nsrv=nsrv,
        rows_cap=geom.rows_cap,
        topk=geom.topk,
        hidden=geom.hidden,
        nslots=geom.nslots,
        num_experts=geom.num_experts,
        nlayers=geom.nlayers,
        flush_mode=flush_mode,
        wait_ticks=wait_ms * ticks_per_ms,
        begin_wait_ticks=begin_wait_ms * ticks_per_ms,
        ticks_per_ms=ticks_per_ms,
        slot_stride=inbox_l.slot_stride,
        off_ids=inbox_l.ids_off,
        off_w=inbox_l.w_off,
        y_stride=rep.y_stride,
        own_base=inbox.ptr,
        peer_base0=hub_reply_ptr,
        ids32=ids32.data_ptr(),
        x_ws=x_ws.data_ptr(),
        w_ws=w_ws.data_ptr(),
    )
    return image


def register_ctx(image: torch.Tensor, device) -> torch.Tensor:
    """The device ctx for ``image`` (uploaded, geometry recorded host-side)."""
    ctx = torch.zeros(P.HX_CTX_WORDS, dtype=torch.int64, device=device)
    ops().hx_ctx_register(ctx, image)
    return ctx


def reregister_ctx(ctx: torch.Tensor, image: torch.Tensor) -> None:
    """Upload a changed image into an existing ctx (state words restart at the
    image's values: call before the first step only)."""
    ops().hx_ctx_register(ctx, image)
