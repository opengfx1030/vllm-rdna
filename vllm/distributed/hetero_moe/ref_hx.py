# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process reference of the ``hx`` ops: the same module-level API as
``hetero_moe.hx`` (``begin_step`` / ``dispatch`` / ``combine`` on a hub object),
with torch ops instead of the HIP kernels and with in-process "servers".

Use: unit tests (CPU) and a functional fallback while the HIP transport is
being brought up. It implements exactly the dispatch rule of ``tables.py``
(``split_pairs`` is the vectorised, capture-safe statement of it: static
shapes, no host sync) and the combine order of the wire protocol (fp16 hub
partial, then the servers' fp16 partials in server order, fp32 sum, one fp16
rounding).
"""

from collections.abc import Callable

import torch

from .tables import IslandTables

# A server computes the partial MoE of the pairs sent to it: (layer, x, ids, w) with
# ids [T, top_k] int32 (-1 where the pair is not this server's) -> fp32 [T, H].
ServerFn = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def split_pairs(
    topk_ids: torch.Tensor,
    slot_map: torch.Tensor,
    n_cold: torch.Tensor,
    cold_srv: torch.Tensor,
    layer: int,
    k_keep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised dispatch rule; returns int32 ``(hub_ids, dest)`` ``[T, top_k]``
    (see ``tables.split_pairs_loop``). Ids outside ``[0, E)`` are invalid."""
    t_n = topk_ids.shape[0]
    e_n = slot_map.shape[0]
    ids = topk_ids.long()
    valid = (ids >= 0) & (ids < e_n)
    idc = ids.clamp(0, e_n - 1)
    slot = torch.where(valid, slot_map[idc].long(), torch.full_like(ids, -1))
    resident = slot >= 0
    # touched-slot bitmap (slots are below E; index E is the "none" sink)
    touched = torch.zeros(e_n + 1, dtype=torch.int32, device=ids.device)
    sink = torch.full_like(slot, e_n)
    where = torch.where(resident, slot, sink)
    touched.scatter_(
        0, where.reshape(-1), torch.ones_like(where, dtype=torch.int32).reshape(-1)
    )
    touched[e_n] = 0
    rank = torch.cumsum(touched, 0) - touched  # prefix popcount in slot order
    keep = resident & (rank[where] < k_keep)
    hub_ids = torch.where(keep, topk_ids.to(torch.int32), torch.full_like(topk_ids, -1))
    ncold = n_cold[idc].long()
    row = torch.arange(t_n, device=ids.device, dtype=torch.long).unsqueeze(1)
    copy = (row + layer) % ncold.clamp(min=1)
    srv = cold_srv[idc, copy]
    remote = valid & ~keep & (ncold > 0)
    dest = torch.where(remote, srv, torch.full_like(srv, -1))
    return hub_ids.to(torch.int32), dest.to(torch.int32)


class RefHub:
    """Stand-in of ``HxHub``. ``servers[s]`` computes server ``s``' partial."""

    def __init__(
        self,
        tables: IslandTables,
        servers: dict[int, ServerFn],
        *,
        device: torch.device | str = "cpu",
        exact: bool = False,
    ):
        self.tables = tables
        self.servers = servers
        # exact: float64 partials without the fp16 wire rounding (identity tests)
        self.exact = exact
        self.device = torch.device(device)
        self._dev = {
            li: (
                torch.from_numpy(lt.slot_map).to(self.device),
                torch.from_numpy(lt.n_cold).to(self.device),
                torch.from_numpy(lt.cold_srv).to(self.device),
            )
            for li, lt in tables.layers.items()
        }
        self.ctx = torch.zeros(8, dtype=torch.int64, device=self.device)
        self.rows = 0
        self.k_keep = 0
        self.posted: list[tuple[int, int]] = []
        self.began: list[tuple[int, int]] = []
        self.checks = 0
        self.closed = False
        self.abort: Exception | None = None
        self.calls: list[tuple[str, int]] = []
        self._pending: dict[int, tuple] = {}
        self.dispatched_pairs = {"hub": 0, "remote": 0, "dropped": 0}

    # ---- host side (as HxHub) ---------------------------------------------------

    def post_step(self, rows: int, bucket: int) -> None:
        self.posted.append((rows, bucket))

    def check(self) -> None:
        self.checks += 1
        if self.abort is not None:
            raise self.abort

    def close(self) -> None:
        self.closed = True


def begin_step(hub: RefHub, rows: int, bucket: int) -> None:
    hub.calls.append(("begin", -1))
    hub.rows = rows
    hub.k_keep = hub.tables.k_for(rows)
    hub.began.append((rows, bucket))
    hub._pending.clear()


def dispatch(
    hub: RefHub,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    hub_ids_out: torch.Tensor,
    layer: int,
) -> None:
    hub.calls.append(("dispatch", layer))
    assert x.shape[0] == hub.rows, "dispatch rows differ from begin_step rows"
    slot_map, n_cold, cold_srv = hub._dev[layer]
    hub_ids, dest = split_pairs(topk_ids, slot_map, n_cold, cold_srv, layer, hub.k_keep)
    hub_ids_out.copy_(hub_ids)
    hub._pending[layer] = (x, topk_ids, topk_w, dest)
    valid = (topk_ids >= 0) & (topk_ids < slot_map.shape[0])
    n_hub = int((hub_ids >= 0).sum())
    n_remote = int((dest >= 0).sum())
    hub.dispatched_pairs["hub"] += n_hub
    hub.dispatched_pairs["remote"] += n_remote
    hub.dispatched_pairs["dropped"] += int(valid.sum()) - n_hub - n_remote


def combine(
    hub: RefHub, y_hub: torch.Tensor, out: torch.Tensor, layer: int
) -> torch.Tensor:
    hub.calls.append(("combine", layer))
    x, topk_ids, topk_w, dest = hub._pending.pop(layer)
    acc_dtype = torch.float64 if hub.exact else torch.float32
    wire_dtype = torch.float64 if hub.exact else torch.float16
    acc = y_hub.to(acc_dtype).clone()
    for s in range(hub.tables.num_servers):
        sel = dest == s
        if not bool(sel.any()):
            continue
        ids_s = torch.where(
            sel, topk_ids.to(torch.int32), torch.full_like(topk_ids, -1)
        )
        y_s = hub.servers[s](layer, x, ids_s, topk_w)
        acc += y_s.to(wire_dtype).to(acc_dtype)  # the wire carries fp16
    out.copy_(acc.to(out.dtype))
    return out


def status(hub: RefHub) -> torch.Tensor:
    return torch.zeros(8, dtype=torch.int64)
