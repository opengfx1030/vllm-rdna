# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""File-backed stand-in for ``nixl_rocm._api`` (G0a spike, branch spike/disagg-g0).

The real vLLM NIXL connector (scheduler, worker, descriptor planning, handshake)
runs unchanged on top of this module. Only the transport library is replaced:

* the producer's ``G0aWorkerExtension.g0a_dump`` copies, for each finished
  request, exactly the memory windows a decode-side read would address
  (``tools/rdna/disagg/g0a_ext.py``) into ``$G0A_STORE_DIR/rank<r>/<tag>.bin``
  and indexes them by absolute producer address;
* on the consumer a READ transfer resolves every (local, remote) descriptor
  pair, looks the remote window up in that index, copies it into the local
  device address with hipMemcpy and reads it back to check its digest.

A remote window that was not dumped fails the transfer. Per-transfer statistics
are appended to ``$G0A_STATS_DIR/xfer_rank<r>.jsonl``.

Environment:
  G0A_STORE_DIR    dump directory (consumer)
  G0A_STATS_DIR    statistics directory (consumer; default $G0A_STORE_DIR)
  G0A_VERIFY       0 skips the read-back digest check (default 1)
  G0A_SKIP_GROUPS  comma list of transfer-group ids whose windows are not
                   copied (ablation: leaves the destination as allocated)
"""

from __future__ import annotations

import glob
import json
import os
import threading
import time
from types import SimpleNamespace
from typing import Any

import g0_hip
import numpy as np
from g0_desc import digest, resolve, row_starts

STORE_ENV = "G0A_STORE_DIR"
STATS_ENV = "G0A_STATS_DIR"
VERIFY_ENV = "G0A_VERIFY"
SKIP_ENV = "G0A_SKIP_GROUPS"
LOCAL_AGENT = "NIXL_INIT_AGENT"


def tp_rank() -> int:
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    return int(get_tensor_model_parallel_rank())


class _Store:
    """Memory-mapped dump files of one producer rank."""

    def __init__(self, rank_dir: str):
        self.windows: dict[tuple[int, int], tuple[Any, int, str, int, str]] = {}
        self.tags: list[str] = []
        for meta_path in sorted(glob.glob(os.path.join(rank_dir, "*.json"))):
            with open(meta_path) as f:
                meta = json.load(f)
            data = None
            if meta["windows"]:
                data = np.memmap(meta_path[:-5] + ".bin", dtype=np.uint8, mode="r")
            for w in meta["windows"]:
                key = (int(w["addr"]), int(w["len"]))
                if key in self.windows:
                    raise ValueError(f"window {key} dumped twice ({meta_path})")
                self.windows[key] = (
                    data,
                    int(w["off"]),
                    w["h"],
                    int(w["g"]),
                    meta["tag"],
                )
            self.tags.append(meta["tag"])


class nixl_agent_config:  # noqa: N801 - mirrors the NIXL name
    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs


class nixl_agent:  # noqa: N801 - mirrors the NIXL name
    """The subset of ``nixl_agent`` the vLLM NIXL connector uses."""

    def __init__(self, agent_name: str, config: Any = None):
        self.agent_name = agent_name
        self._lock = threading.Lock()
        self._next = 1
        self._dlists: dict[int, SimpleNamespace] = {}
        self._xfers: dict[int, dict[str, Any]] = {}
        self._remote_rank: dict[str, int] = {}
        self._stores: dict[int, _Store] = {}
        self.registered: list[tuple[int, int, int, str]] = []
        self.notifs_sent: list[tuple[str, bytes]] = []

    # -- registration -------------------------------------------------------
    def get_reg_descs(self, caches_data: Any, memory_type: str = "VRAM") -> Any:
        return SimpleNamespace(descs=list(caches_data), mem_type=memory_type)

    def get_xfer_descs(self, blocks_data: Any, memory_type: str = "VRAM") -> Any:
        return SimpleNamespace(descs=list(blocks_data), mem_type=memory_type)

    def register_memory(self, descs: Any, backends: Any = None) -> None:
        for start, nbytes, device_id, _name in descs.descs:
            self.registered.append(
                (int(start), int(nbytes), int(device_id), descs.mem_type)
            )

    def deregister_memory(self, descs: Any) -> None:
        pass

    def create_backend(self, *args: Any, **kwargs: Any) -> None:
        pass

    # -- agents -------------------------------------------------------------
    def get_agent_metadata(self) -> bytes:
        return json.dumps({"name": self.agent_name, "rank": tp_rank()}).encode()

    def add_remote_agent(self, agent_metadata: bytes) -> str:
        meta = json.loads(bytes(agent_metadata).decode())
        self._remote_rank[meta["name"]] = int(meta["rank"])
        return str(meta["name"])

    def remove_remote_agent(self, agent: str) -> None:
        self._remote_rank.pop(agent, None)

    def check_remote_metadata(self, agent: str) -> bool:
        return True

    # -- descriptor lists and transfers --------------------------------------
    def _new_id(self) -> int:
        with self._lock:
            handle = self._next
            self._next += 1
        return handle

    def prep_xfer_dlist(
        self, agent_name: str, descs: Any, mem_type: str | None = None, **_: Any
    ) -> int:
        rows = np.asarray(descs, dtype=np.uint64).reshape(-1, 5)
        handle = self._new_id()
        self._dlists[handle] = SimpleNamespace(
            agent=agent_name, rows=rows, starts=row_starts(rows), mem_type=mem_type
        )
        return handle

    def release_dlist_handle(self, handle: int) -> None:
        self._dlists.pop(handle, None)

    def make_prepped_xfer(
        self,
        xfer_type: str,
        local_xfer_side_handle: int,
        local_block_descs_ids: Any,
        remote_xfer_side_handle: int,
        remote_block_descs_ids: Any,
        notif_msg: bytes | None = None,
        **_: Any,
    ) -> int:
        handle = self._new_id()
        self._xfers[handle] = {
            "op": xfer_type,
            "lh": local_xfer_side_handle,
            "lids": np.asarray(local_block_descs_ids, dtype=np.int64),
            "rh": remote_xfer_side_handle,
            "rids": np.asarray(remote_block_descs_ids, dtype=np.int64),
            "notif": notif_msg,
            "state": "INIT",
            "telemetry": SimpleNamespace(
                xferDuration=0.0, postDuration=0.0, totalBytes=0, descCount=0
            ),
        }
        return handle

    def transfer(self, handle: int, notif_msg: bytes | None = None) -> str:
        xfer = self._xfers[handle]
        if xfer["op"] != "READ":
            raise NotImplementedError("G0a fake NIXL implements READ only")
        try:
            self._read(xfer)
        except BaseException:
            xfer["state"] = "ERR"
            raise
        xfer["state"] = "DONE"
        return "PROC"

    def check_xfer_state(self, handle: int) -> str:
        return self._xfers[handle]["state"]

    def get_xfer_telemetry(self, handle: int) -> Any:
        return self._xfers[handle]["telemetry"]

    def release_xfer_handle(self, handle: int) -> None:
        self._xfers.pop(handle, None)

    # -- notifications -------------------------------------------------------
    def send_notif(self, agent_name: str, notif_msg: bytes = b"") -> None:
        self.notifs_sent.append((agent_name, bytes(notif_msg)))

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        return {}

    # -- the transfer itself -------------------------------------------------
    def _store(self, rank: int) -> _Store:
        store = self._stores.get(rank)
        if store is None:
            root = os.environ[STORE_ENV]
            store = _Store(os.path.join(root, f"rank{rank}"))
            self._stores[rank] = store
        return store

    def _read(self, xfer: dict[str, Any]) -> None:
        t0 = time.perf_counter()
        local = self._dlists[xfer["lh"]]
        remote = self._dlists[xfer["rh"]]
        laddr, llen, lrow = resolve(local.rows, local.starts, xfer["lids"])
        raddr, rlen, _ = resolve(remote.rows, remote.starts, xfer["rids"])
        if laddr.shape != raddr.shape or not np.array_equal(llen, rlen):
            raise ValueError(
                f"descriptor mismatch: {laddr.size} local vs {raddr.size} remote "
                "entries or differing lengths"
            )
        remote_rank = self._remote_rank[remote.agent]
        store = self._store(remote_rank)
        skip = {int(g) for g in os.environ.get(SKIP_ENV, "").split(",") if g}
        verify = os.environ.get(VERIFY_ENV, "1") != "0"
        g0_hip.set_device(int(local.rows[0, 2]))
        scratch = np.empty(int(llen.max()) if llen.size else 0, dtype=np.uint8)
        copied = skipped = 0
        missing: list[list[int]] = []
        bad: list[list[int]] = []
        per_group: dict[int, dict[str, int]] = {}
        per_row: dict[int, int] = {}
        for i in range(laddr.size):
            n = int(rlen[i])
            hit = store.windows.get((int(raddr[i]), n))
            if hit is None:
                missing.append([int(raddr[i]), n, int(lrow[i])])
                continue
            data, off, want, group, _tag = hit
            stats = per_group.setdefault(
                group, {"windows": 0, "bytes": 0, "skipped_bytes": 0}
            )
            if group in skip:
                stats["skipped_bytes"] += n
                skipped += n
                continue
            g0_hip.h2d(int(laddr[i]), data[off : off + n], n)
            if verify:
                g0_hip.d2h(scratch, int(laddr[i]), n)
                if digest(scratch[:n]) != want:
                    bad.append([int(raddr[i]), n, int(lrow[i])])
            stats["windows"] += 1
            stats["bytes"] += n
            per_row[int(lrow[i])] = per_row.get(int(lrow[i]), 0) + n
            copied += n
        elapsed = time.perf_counter() - t0
        telemetry = xfer["telemetry"]
        telemetry.xferDuration = elapsed * 1e6
        telemetry.totalBytes = copied
        telemetry.descCount = int(laddr.size)
        record = {
            "time": time.strftime("%F %T"),
            "rank": tp_rank(),
            "remote_rank": remote_rank,
            "notif": (xfer["notif"] or b"").decode(errors="replace"),
            "pairs": int(laddr.size),
            "bytes_copied": copied,
            "bytes_skipped": skipped,
            "seconds": round(elapsed, 4),
            "verify": verify,
            "missing": len(missing),
            "missing_first": missing[:8],
            "digest_mismatch": len(bad),
            "digest_mismatch_first": bad[:8],
            "per_group": {str(g): v for g, v in sorted(per_group.items())},
            "per_local_row_bytes": {str(r): b for r, b in sorted(per_row.items())},
        }
        out_dir = os.environ.get(STATS_ENV) or os.environ[STORE_ENV]
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"xfer_rank{record['rank']}.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        if missing or bad:
            raise RuntimeError(
                f"G0a fake NIXL: {len(missing)} remote windows were not dumped, "
                f"{len(bad)} read-back digests differ (rank {record['rank']})"
            )
