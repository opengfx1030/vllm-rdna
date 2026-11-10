# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Support module of the hx host tests (not a test module): a CPU stand-in for the
``hx_*`` functions of ``vllm._custom_ops`` and builders for tiny plans.

``FakeOps`` is installed by ``monkeypatch.setattr(ctx_module, "_OPS", fake)``. It
keeps the contracts of ``csrc/rocm/hetero_moe/hx_ops.cu`` that the host modules
rely on, strictly enough that a host module which forgets a field or leaks a
resource fails here:

* ``hx_alloc`` (kinds 0/1/2 only) needs a page multiple, returns ``uint8[88]``
  (ptr | nbytes | kind used | 64 B handle); ``hx_free`` of an unknown pointer
  raises.
* ``hx_open`` maps a handle to a NEW pointer (a peer process sees other
  addresses); an open of a freed allocation raises; ``hx_close`` of an unknown
  mapping raises. ``assert_clean`` reports leaked allocations and mappings.
* the ctx image is the real field table of ``HxCtx`` (names, offsets, sizes), so
  a test reads a field back with ``hx_ctx_get``; ``hx_ctx_register`` applies the
  validation of the C++ op and copies the image into the (CPU) ctx tensor;
  ``hx_status`` / ``hx_host_abort`` need a registered ctx.
"""

import os
import struct
import threading

import numpy as np
import torch

from vllm.distributed.hetero_moe import hx_proto as P
from vllm.distributed.hetero_moe.ctx import (
    Exported,
    alloc_exported,
    register_ctx,
    server_image,
)

FAKE_PTR_BASE = 0x7F00_0000_0000
FAKE_MAP_BASE = 0x7E00_0000_0000
FAKE_TICKS_PER_MS = 100_000


def _field_table() -> dict[str, tuple[int, int]]:
    """Name -> (byte offset, size) of ``struct HxCtx`` (hx_core.cuh), natural
    alignment, in declaration order."""
    layout = [
        ("magic", 8),
        ("role", 4),
        ("version", 4),
        ("plan_hash", 8),
        ("index", 4),
        ("nsrv", 4),
        ("rows_cap", 4),
        ("topk", 4),
        ("hidden", 4),
        ("nslots", 4),
        ("num_experts", 4),
        ("nlayers", 4),
        ("flush_mode", 4),
        ("pad0", 4),
        ("wait_ticks", 8),
        ("begin_wait_ticks", 8),
        ("ticks_per_ms", 8),
        ("slot_stride", 8),
        ("off_ids", 8),
        ("off_w", 8),
        ("y_stride", 8),
        ("last_epoch", 8),
        ("desc_epoch", 8),
        ("desc_T", 4),
        ("desc_bucket", 4),
        ("cur_T", 4),
        ("cur_bucket", 4),
        ("cur_K", 4),
        ("cur_mode", 4),
        ("arrive_a", 4),
        ("abort_word", 4),
        ("status", 64),
        ("own_base", 8),
        *[(f"peer_base{i}", 8) for i in range(P.HX_MAX_SERVERS)],
        ("hub_slot", 8),
        ("cold_srv", 8),
        ("n_cold", 8),
        ("owned", 8),
        ("ktab", 8),
        ("ids32", 8),
        ("x_ws", 8),
        ("w_ws", 8),
        ("layer_stats", 8),
        ("arrive_b", 4),
        ("dig_pad", 4),
        ("dig_acc", 4 * (P.HX_MAX_SERVERS + 2)),
        ("dig_stat", 8 * P.HX_DS_WORDS),
        ("dbg", 32),
    ]
    out: dict[str, tuple[int, int]] = {}
    off = 0
    for name, size in layout:
        out[name] = (off, size)
        off += size
    assert off <= P.HX_CTX_WORDS * 8
    return out


CTX_FIELDS = _field_table()
# the C++ op only exposes these (HXF in hx_ops.cu): the step-state fields are
# written by kernels
_SETTABLE = {
    n
    for n in CTX_FIELDS
    if n
    not in (
        "magic",
        "pad0",
        "desc_epoch",
        "desc_T",
        "desc_bucket",
        "cur_T",
        "cur_bucket",
        "cur_K",
        "cur_mode",
        "arrive_a",
        "abort_word",
        "status",
        "arrive_b",
        "dig_pad",
        "dig_acc",
        "dig_stat",
        "dbg",
    )
}


class FakeOps:
    """Drop-in for the ``hx_*`` part of ``vllm._custom_ops`` (CPU)."""

    def __init__(self, calibration: tuple[float, float] | None = None):
        self._lock = threading.Lock()
        self._next_ptr = FAKE_PTR_BASE
        self._next_map = FAKE_MAP_BASE
        self.calibration = calibration or (
            float(FAKE_TICKS_PER_MS),
            float(FAKE_TICKS_PER_MS),
        )
        self.allocs: dict[int, dict] = {}  # ptr -> {nbytes, kind, handle}
        self.freed: list[int] = []
        self.opened: dict[int, int] = {}  # mapped ptr -> allocation ptr
        self.closed_maps: list[int] = []
        self.registered: dict[int, int] = {}  # ctx data_ptr -> registrations
        self.unregistered: list[int] = []
        self.records: dict[int, int] = {}  # ctx data_ptr -> abort record word
        self.dig_words: dict[int, list[int]] = {}  # ctx data_ptr -> hx_dig_stats
        self.aborts: list[int] = []  # ctx data_ptr of each hx_host_abort
        self.abort_event = threading.Event()
        self.calls: list[tuple] = []  # (name, args...) of the recorded ops
        self.views: list[tuple] = []
        self.pushed: list[tuple[int, torch.Tensor]] = []  # (layer, y) of hx_srv_push

    # ---- allocation and IPC --------------------------------------------------

    def hx_alloc(self, nbytes: int, kind: int) -> torch.Tensor:
        if nbytes <= 0 or nbytes % P.HX_PAGE:
            raise RuntimeError(
                f"hx_alloc: nbytes must be a positive multiple of {P.HX_PAGE}"
            )
        if kind not in (0, 1, 2):
            raise RuntimeError(f"hx_alloc: bad kind {kind}")
        used = kind
        with self._lock:
            ptr = self._next_ptr
            self._next_ptr += nbytes + P.HX_PAGE
            # the handle bytes: the allocation pointer, then a recognisable tail
            handle = struct.pack("<Q", ptr) + bytes(range(56))
            self.allocs[ptr] = {"nbytes": nbytes, "kind": used, "handle": handle}
        raw = struct.pack("<QQQ", ptr, nbytes, used) + handle
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8)

    def hx_free(self, ptr: int) -> None:
        with self._lock:
            if ptr not in self.allocs:
                raise RuntimeError(f"hx_free: {ptr:#x} is not a live allocation")
            del self.allocs[ptr]
            self.freed.append(ptr)

    def hx_open(self, handle: torch.Tensor) -> int:
        if handle.numel() != 64 or handle.dtype != torch.uint8:
            raise RuntimeError("hx_open: handle must be a uint8[64] tensor")
        raw = handle.numpy().tobytes()
        (ptr,) = struct.unpack_from("<Q", raw, 0)
        with self._lock:
            entry = self.allocs.get(ptr)
            if entry is None or entry["handle"] != raw:
                raise RuntimeError("hx_open: invalid IPC handle")
            mapped = self._next_map
            self._next_map += entry["nbytes"] + P.HX_PAGE
            self.opened[mapped] = ptr
        return mapped

    def hx_close(self, ptr: int) -> None:
        with self._lock:
            if ptr not in self.opened:
                raise RuntimeError(f"hx_close: {ptr:#x} is not an open mapping")
            del self.opened[ptr]
            self.closed_maps.append(ptr)

    def hx_view(self, ptr: int, sizes, dtype, device: int) -> torch.Tensor:
        self.views.append((ptr, list(sizes), dtype, device))
        return torch.zeros(list(sizes), dtype=dtype)

    def hx_calibrate(self, interval_ms: int) -> torch.Tensor:
        self.calls.append(("hx_calibrate", interval_ms))
        return torch.tensor(self.calibration, dtype=torch.float64)

    def assert_clean(self) -> None:
        """Every allocation freed and every mapping closed."""
        assert not self.allocs, f"leaked allocations: {sorted(self.allocs)}"
        assert not self.opened, f"leaked mappings: {sorted(self.opened)}"

    # ---- ctx image -----------------------------------------------------------

    @staticmethod
    def _check_image(t: torch.Tensor, who: str) -> None:
        if (
            t.device.type != "cpu"
            or t.dtype != torch.int64
            or t.numel() != P.HX_CTX_WORDS
            or not t.is_contiguous()
        ):
            raise RuntimeError(f"{who}: ctx image must be a contiguous CPU int64[128]")

    def hx_ctx_new(self) -> torch.Tensor:
        image = torch.zeros(P.HX_CTX_WORDS, dtype=torch.int64)
        raw = image.numpy().view("uint8")
        raw[0:8] = np.frombuffer(struct.pack("<Q", P.HX_MAGIC), dtype=np.uint8)
        off, _ = CTX_FIELDS["version"]
        raw[off : off + 4] = np.frombuffer(
            struct.pack("<I", P.HX_VERSION), dtype=np.uint8
        )
        return image

    def hx_ctx_set(self, image: torch.Tensor, name: str, value: int) -> None:
        self._check_image(image, "hx_ctx_set")
        if name not in _SETTABLE:
            raise RuntimeError(f"hx_ctx_set: unknown field '{name}'")
        off, size = CTX_FIELDS[name]
        raw = image.numpy().view("uint8")
        raw[off : off + size] = np.frombuffer(
            struct.pack("<q", value)[:size], dtype=np.uint8
        )

    def hx_ctx_get(self, image: torch.Tensor, name: str) -> int:
        self._check_image(image, "hx_ctx_get")
        if name not in _SETTABLE:
            raise RuntimeError(f"hx_ctx_get: unknown field '{name}'")
        off, size = CTX_FIELDS[name]
        raw = image.numpy().view("uint8")[off : off + size].tobytes()
        value = int.from_bytes(raw + bytes(8 - size), "little")
        # the C++ op returns the 8 copied bytes as int64
        return value - (1 << 64) if value >= (1 << 63) else value

    def hx_ctx_register(self, ctx: torch.Tensor, image: torch.Tensor) -> None:
        self._check_image(image, "hx_ctx_register")
        if ctx.dtype != torch.int64 or ctx.numel() != P.HX_CTX_WORDS:
            raise RuntimeError("hx_ctx_register: ctx must be int64[128]")

        def u(name: str) -> int:
            return self.hx_ctx_get(image, name) & ((1 << (8 * CTX_FIELDS[name][1])) - 1)

        raw = image.numpy().view("uint8")
        if (
            struct.unpack_from("<Q", raw, 0)[0] != P.HX_MAGIC
            or u("version") != P.HX_VERSION
        ):
            raise RuntimeError("hx_ctx_register: image was not made by hx_ctx_new")
        role = u("role")
        if role not in (1, 2):
            raise RuntimeError("hx_ctx_register: bad role")
        if not (
            u("rows_cap") >= 1
            and 1 <= u("topk") <= P.HX_MAX_TOPK
            and u("hidden") >= 8
            and u("hidden") % 8 == 0
            and 1 <= u("nslots") <= P.HX_MAX_SLOTS
            and 1 <= u("num_experts") <= P.HX_MAX_EXPERTS
            and 1 <= u("nlayers") <= P.HX_MAX_LAYERS
        ):
            raise RuntimeError("hx_ctx_register: geometry out of range")
        if not (
            u("own_base") != 0
            and u("ticks_per_ms") > 0
            and u("wait_ticks") > 0
            and u("slot_stride") > 0
            and u("y_stride") > 0
        ):
            raise RuntimeError("hx_ctx_register: own_base/ticks/wait/strides not set")
        if role == 1:
            nsrv = u("nsrv")
            if not (
                1 <= nsrv <= P.HX_MAX_SERVERS
                and u("hub_slot")
                and u("cold_srv")
                and u("n_cold")
                and u("ktab")
            ):
                raise RuntimeError("hx_ctx_register: hub needs nsrv and the tables")
            for s in range(nsrv):
                if u(f"peer_base{s}") == 0:
                    raise RuntimeError(f"hx_ctx_register: peer_base{s} not set")
        elif not (
            u("index") < P.HX_MAX_SERVERS
            and u("ids32")
            and u("peer_base0")
            and u("begin_wait_ticks") > 0
        ):
            raise RuntimeError(
                "hx_ctx_register: server needs index, ids32, peer_base0, "
                "begin_wait_ticks"
            )
        ctx.copy_(image)
        with self._lock:
            key = ctx.data_ptr()
            self.registered[key] = self.registered.get(key, 0) + 1

    def hx_ctx_unregister(self, ctx: torch.Tensor) -> None:
        with self._lock:
            key = ctx.data_ptr()
            if self.registered.pop(key, None) is not None:
                self.unregistered.append(key)

    def is_registered(self, ctx: torch.Tensor) -> bool:
        return ctx.data_ptr() in self.registered

    def _registered_key(self, ctx: torch.Tensor) -> int:
        key = ctx.data_ptr()
        if key not in self.registered:
            raise RuntimeError("hx: ctx is not registered")
        return key

    # ---- host side channel ---------------------------------------------------

    def set_record(self, ctx: torch.Tensor, value: int) -> None:
        """What ``hx_status(ctx)[HX_ST_RECORD]`` returns from now on."""
        self.records[ctx.data_ptr()] = value

    def hx_status(self, ctx: torch.Tensor) -> torch.Tensor:
        key = self._registered_key(ctx)
        out = torch.zeros(P.HX_STATUS_WORDS, dtype=torch.int64)
        value = self.records.get(key, 0)
        out[P.HX_ST_RECORD] = value - (1 << 64) if value >= (1 << 63) else value
        return out

    def set_dig_stats(self, ctx: torch.Tensor, words: list[int]) -> None:
        """What ``hx_dig_stats(ctx)`` returns from now on (HX_DS_* words)."""
        assert len(words) == P.HX_DS_WORDS
        self.dig_words[ctx.data_ptr()] = list(words)

    def hx_dig_stats(self, ctx: torch.Tensor) -> torch.Tensor:
        key = self._registered_key(ctx)
        words = self.dig_words.get(key, [0] * P.HX_DS_WORDS)
        return torch.tensor(
            [w - (1 << 64) if w >= (1 << 63) else w for w in words], dtype=torch.int64
        )

    def hx_host_abort(self, ctx: torch.Tensor) -> None:
        key = self._registered_key(ctx)
        self.aborts.append(key)
        self.abort_event.set()

    def aborted(self, ctx: torch.Tensor) -> bool:
        return ctx.data_ptr() in self.aborts

    # ---- recorded compute ops ------------------------------------------------

    def hx_srv_post_desc(self, ctx, epoch: int, T: int, bucket: int) -> None:
        self.calls.append(("hx_srv_post_desc", epoch, T, bucket))

    def hx_begin_step(self, ctx, T: int, bucket: int) -> None:
        self.calls.append(("hx_begin_step", T, bucket))

    def hx_dispatch(self, x, topk_ids, topk_w, hub_ids_out, ctx, layer: int) -> None:
        self.calls.append(("hx_dispatch", layer))

    def hx_combine(self, y_hub, out, ctx, layer: int) -> None:
        self.calls.append(("hx_combine", layer))

    def hx_srv_begin(self, ctx) -> None:
        self.calls.append(("hx_srv_begin",))

    def hx_srv_wait(self, ctx, layer: int, T: int) -> None:
        self.calls.append(("hx_srv_wait", layer, T))

    def hx_srv_push(self, y, ctx, layer: int, T: int) -> None:
        self.calls.append(("hx_srv_push", layer, T, tuple(y.shape), y.dtype))
        self.pushed.append((layer, y.clone()))


# ---- tiny plans ----------------------------------------------------------------

TINY_EXPERTS = 8
TINY_TOP_K = 2
TINY_LAYERS = 4


def tiny_plan_doc(nservers: int = 2, hubs: int = 1) -> dict:
    """A valid ``hetero-moe-placement/1`` document: ``hubs`` hub(s) (W7800) and
    ``nservers`` expert servers (V620) in one island, 8 experts top-2, layers 0-3.
    Experts 0-3 are hub resident; every expert has a server primary
    (``e % nservers``), the experts divisible by 4 a replica as well."""
    if not 1 <= nservers <= P.HX_MAX_SERVERS:
        raise ValueError("nservers out of range")
    devices = [
        {
            "id": f"hub{h}",
            "role": "hub",
            "arch": "gfx1100",
            "hip": h,
            "switch": "c0",
            "expert_budget_bytes": 10**9,
        }
        for h in range(hubs)
    ] + [
        {
            "id": f"exp{s}",
            "role": "expert",
            "arch": "gfx1030",
            "hip": hubs + s,
            "switch": "c0",
            "expert_budget_bytes": 10**9,
        }
        for s in range(nservers)
    ]
    first_expert = hubs
    layers = []
    for layer in range(TINY_LAYERS):
        owners = []
        for e in range(TINY_EXPERTS):
            row = [e % hubs] if e < 4 else []
            cold = [first_expert + e % nservers]
            if nservers > 1 and e % 4 == 0:
                other = first_expert + (e + 1) % nservers
                cold += [other] if other > cold[0] else []
            owners.append(row + cold)
        layers.append({"layer": layer, "kind": "target", "owners": owners})
    return {
        "format": "hetero-moe-placement/1",
        "version": 1,
        "model": {
            "name": "tiny",
            "num_layers": TINY_LAYERS,
            "num_experts": TINY_EXPERTS,
            "top_k": TINY_TOP_K,
            "mtp_layers": 0,
        },
        "expert_bytes": 1000,
        "cold_complete": True,
        "devices": devices,
        "islands": [list(range(len(devices)))],
        "layers": layers,
    }


def tiny_plan(nservers: int = 2, hubs: int = 1):
    from vllm.distributed.hetero_moe.plan import plan_from_doc

    return plan_from_doc(tiny_plan_doc(nservers, hubs))


def tiny_tables(plan, spill_k: dict[int, int] | None = None):
    from vllm.distributed.hetero_moe.tables import build_island_tables

    hub_dev = plan.hubs(0)[0]
    return build_island_tables(plan, hub_dev, spill_k=spill_k or {3: 2, 6: 3})


# ---- what ExpertServer.prepare() would have set ----------------------------------


def inject_prepared(
    server,
    *,
    ticks_per_ms: int = FAKE_TICKS_PER_MS,
    arch: str = "gfx1030",
    bus: str = "ce",
    dir=None,
    device: str = "cpu",
) -> None:
    """Set the attributes ``ExpertServer.prepare()`` sets on a GPU (allocation,
    ctx, views, model) from the fake ops, so ``publish``/``connect_hub``/``serve``
    run on the CPU. ``prepare`` itself needs CUDA and is not exercised."""
    from vllm.distributed.hetero_moe import rdv
    from vllm.distributed.hetero_moe.server import SyntheticModel

    geom = server.geometry
    server.dir = dir if dir is not None else rdv.hx_dir(server.plan.plan_id)
    server.device = torch.device(device)
    server.arch = arch
    server.bus = bus
    server.ticks_per_ms = ticks_per_ms
    server._inbox_layout = geom.inbox()
    server._inbox = alloc_exported(server._inbox_layout.total_bytes, server.alloc)
    server.ids32 = torch.zeros(geom.rows_cap * geom.topk, dtype=torch.int32)
    server.x_ws = torch.zeros(geom.rows_cap, geom.hidden, dtype=torch.float16)
    server.w_ws = torch.zeros(geom.rows_cap, geom.topk, dtype=torch.float32)
    server._hub_nsrv = len(server.plan.experts(server.island))
    server._image = server_image(
        geom,
        plan_id=server.plan.plan_id,
        server_index=server.server_index,
        nsrv=server._hub_nsrv,
        inbox=server._inbox,
        hub_reply_ptr=server._inbox.ptr,
        ids32=server.ids32,
        x_ws=server.x_ws,
        w_ws=server.w_ws,
        ticks_per_ms=ticks_per_ms,
        wait_ms=server.wait_ms,
        begin_wait_ms=server.begin_wait_ms,
        hub_nsrv=server._hub_nsrv,
        flush_mode=server.flush,
    )
    server.ctx = register_ctx(server._image, server.device)
    assert server._inbox_layout.total_bytes == server._inbox.nbytes
    server.model = SyntheticModel(server.nlayers, geom.hidden, geom.topk)


def server_doc(
    plan,
    device_id: str,
    inbox: Exported,
    pid: int | None = None,
    **overrides,
) -> dict:
    """A server's rendezvous document as ``ExpertServer.publish`` writes it (for
    tests that play the server by hand). ``overrides`` replace top-level keys."""
    from vllm.distributed.hetero_moe.ctx import Geometry
    from vllm.distributed.hetero_moe.hx import SRV_FORMAT

    dev = plan.device_index(device_id)
    geom = Geometry(
        rows_cap=8,
        topk=plan.top_k,
        hidden=P.HX_DEFAULT_HIDDEN,
        nslots=2,
        num_experts=plan.num_experts,
        nlayers=plan.target_layers[-1] + 1,
    )
    doc = {
        "format": SRV_FORMAT,
        "plan_id": plan.plan_id,
        "device_id": device_id,
        "server_index": plan.server_index(dev),
        "island": plan.island_of(dev),
        "pid": os.getpid() if pid is None else pid,
        "arch": "gfx1030",
        "bus": "ce",
        "geometry": geom.as_dict(),
        "buckets": [3],
        "ticks_per_ms": FAKE_TICKS_PER_MS,
        "inbox": {
            "handle": inbox.handle_hex(),
            "nbytes": inbox.nbytes,
            "kind": inbox.kind_name,
        },
        "state": "ready",
    }
    doc.update(overrides)
    return doc


class FakeRing:
    """A ring stand-in for ``ExpertServer.serve``: ``posted`` descriptors are
    returned by ``get``, later ones are ``None``."""

    def __init__(self, posted=(), closed: bool = False, raises: dict | None = None):
        self.posted = list(posted)
        self.closed = closed
        self.raises = raises or {}
        self.gets: list[int] = []
        self.close_calls = 0

    def get(self, n: int):
        self.gets.append(n)
        if n in self.raises:
            raise self.raises[n]
        return self.posted[n - 1] if n <= len(self.posted) else None

    def close(self) -> None:
        self.close_calls += 1
