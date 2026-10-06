# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transports for cold-expert pairs.

1. ``loopback`` — in-process, for tests.
2. ``host_staged`` — pinned host buffers. Default. ``local`` stays on
   this host, ``shm`` crosses a shared-memory segment, ``tcp`` uses the
   length-prefixed frame for a second host.
3. ``peer`` — direct peer copy. UNVERIFIED. Constructed only when a
   probe file records both-way ``hipDeviceCanAccessPeer`` and a real
   copy. Mixed gfx1100/gfx1030 RCCL collectives are not used.
"""

import json
import socket
from dataclasses import dataclass

import torch

from vllm.distributed.hetero_moe.wire import (
    pack_tensors,
    read_frame,
    shm_write_read,
    stage_pinned,
    unpack_tensors,
    write_frame,
)


class UnverifiedTransport(RuntimeError):
    """Peer copy is gated off until a real probe passes."""


@dataclass
class ColdPayload:
    """One layer's cold pairs.

    Attributes:
        hidden: ``[P, H]`` token rows.
        expert_ids: ``[P]`` expert ids.
        router_weights: ``[P]`` router weights.
        token_index: ``[P]`` positions in the original token batch.
    """

    hidden: torch.Tensor
    expert_ids: torch.Tensor
    router_weights: torch.Tensor
    token_index: torch.Tensor

    @property
    def cold_count(self) -> int:
        return int(self.expert_ids.numel())


@dataclass
class PeerProbe:
    """What ``probe_hetero_peer.py`` is allowed to claim.

    Attributes:
        passed: The probe process set this after a real run.
        measured: A copy was timed. False means no bandwidth exists.
        access: ``(src, dst) -> hipDeviceCanAccessPeer``.
        nbytes: Bytes actually copied. Zero when nothing was timed.
        seconds: Measured elapsed seconds. Zero when nothing was timed.
    """

    passed: bool
    measured: bool
    access: dict[tuple[int, int], bool]
    nbytes: int
    seconds: float
    note: str = ""

    def allows(self, src: int, dst: int) -> bool:
        """True when both directions were probed and a copy was timed."""
        if not self.passed or not self.measured:
            return False
        if self.nbytes <= 0 or self.seconds <= 0:
            return False
        return bool(self.access.get((src, dst)) and self.access.get((dst, src)))


def load_probe(path: str) -> PeerProbe | None:
    """Load a probe JSON file. Missing path yields None.

    Args:
        path: File written by the probe script. Empty skips the load.

    Returns:
        The probe, or None when ``path`` is empty.
    """
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    access = {
        (int(item["src"]), int(item["dst"])): bool(item["can"])
        for item in raw.get("access", [])
    }
    return PeerProbe(
        passed=bool(raw.get("passed", False)),
        measured=bool(raw.get("measured", False)),
        access=access,
        nbytes=int(raw.get("nbytes") or 0),
        seconds=float(raw.get("seconds") or 0.0),
        note=str(raw.get("note") or ""),
    )


class LoopbackTransport:
    """In-process handoff. ``send`` does not run the expert; ``recv`` does."""

    name = "loopback"

    def __init__(self, cold_fn) -> None:
        self.cold_fn = cold_fn
        self._pending: ColdPayload | None = None
        self.sends = 0
        self.recvs = 0

    def send(self, payload: ColdPayload) -> None:
        if payload.cold_count == 0:
            return
        self._pending = payload
        self.sends += 1

    def recv(self) -> tuple[torch.Tensor, torch.Tensor]:
        pending = self._pending
        if pending is None:
            raise RuntimeError("recv without a cold send")
        self._pending = None
        self.recvs += 1
        out = self.cold_fn(
            pending.hidden,
            pending.expert_ids,
            pending.router_weights,
        )
        return out, pending.token_index


class HostStagedTransport:
    """Pinned host buffers, then local, shm, or TCP.

    ``send`` copies onto the side-stream staging buffer and returns.
    ``recv`` is the wait. The default link is ``local``.
    """

    name = "host_staged"

    def __init__(self, cold_fn, link: str = "local", addr: str = "") -> None:
        if link not in ("local", "shm", "tcp"):
            raise ValueError(f"unknown host link {link}")
        self.cold_fn = cold_fn
        self.link = link
        self.addr = addr
        self._pending: bytes | None = None
        self._conn: socket.socket | None = None
        self.sends = 0
        self.recvs = 0
        self.side_stream_ops = 0
        self.last_pinned = False

    def send(self, payload: ColdPayload) -> None:
        if payload.cold_count == 0:
            return
        if self.link == "tcp" and not self.addr:
            raise RuntimeError(
                "host-staged TCP needs VLLM_HETERO_MOE_HOST_ADDR; "
                "refusing to open a socket with an empty address"
            )
        staged, pinned = stage_pinned(payload.hidden.to(torch.float16))
        self.last_pinned = pinned
        blob = pack_tensors(
            {
                "hidden": staged,
                "expert_ids": payload.expert_ids,
                "router_weights": payload.router_weights,
                "token_index": payload.token_index,
            }
        )
        # The copy is recorded as a side-stream op. No host synchronize.
        self.side_stream_ops += 1
        if self.link == "shm":
            blob = shm_write_read(blob)
        elif self.link == "tcp":
            host, _, port_s = self.addr.rpartition(":")
            conn = socket.create_connection((host, int(port_s)), timeout=5.0)
            write_frame(conn, blob)
            # The expert runs on the far side. recv() reads the reply.
            self._conn = conn
            self._pending = b""
            self.sends += 1
            return
        self._pending = blob
        self.sends += 1

    def recv(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.link == "tcp":
            if self._conn is None:
                raise RuntimeError("recv without a cold send")
            try:
                blob = read_frame(self._conn)
            finally:
                self._conn.close()
                self._conn = None
            self._pending = None
            self.recvs += 1
            data = unpack_tensors(blob)
            return data["hidden"], data["token_index"]
        blob = self._pending
        if blob is None:
            raise RuntimeError("recv without a cold send")
        self._pending = None
        self.recvs += 1
        data = unpack_tensors(blob)
        out = self.cold_fn(
            data["hidden"],
            data["expert_ids"],
            data["router_weights"],
        )
        return out, data["token_index"]


class PeerCopyTransport:
    """Direct peer memcpy. Status stays UNVERIFIED.

    RCCL send/recv across gfx1100 and gfx1030 is intentionally absent.
    """

    name = "peer"
    status = "UNVERIFIED"

    def __init__(
        self,
        probe: PeerProbe | None,
        cold_fn,
        src: int = 0,
        dst: int = 1,
    ) -> None:
        if probe is None or not probe.allows(src, dst):
            raise UnverifiedTransport(
                "direct peer copy is UNVERIFIED and stays off until "
                "hipDeviceCanAccessPeer is true both ways and the "
                "bandwidth probe records a passing measurement. "
                "Mixed-arch RCCL collectives are not used."
            )
        self.cold_fn = cold_fn
        self.probe = probe
        self._pending: ColdPayload | None = None
        self.sends = 0
        self.recvs = 0

    def send(self, payload: ColdPayload) -> None:
        if payload.cold_count == 0:
            return
        # The probe gate passed. This object still does not claim a
        # measured peer memcpy on the mixed-arch box.
        self._pending = payload
        self.sends += 1

    def recv(self) -> tuple[torch.Tensor, torch.Tensor]:
        pending = self._pending
        if pending is None:
            raise RuntimeError("recv without a cold send")
        self._pending = None
        self.recvs += 1
        out = self.cold_fn(
            pending.hidden,
            pending.expert_ids,
            pending.router_weights,
        )
        return out, pending.token_index


def build_transport(config, cold_fn):
    """Build the configured transport.

    Args:
        config: :class:`HeteroMoEConfig`.
        cold_fn: ``(hidden, expert_ids, router_weights) -> [P, H]``.

    Returns:
        A transport whose ``send`` is non-blocking relative to ``recv``.

    Raises:
        UnverifiedTransport: ``peer`` without a passing probe.
        ValueError: Unknown transport name.
    """
    if config.transport == "loopback":
        return LoopbackTransport(cold_fn)
    if config.transport == "host_staged":
        return HostStagedTransport(
            cold_fn,
            link=config.host_link,
            addr=config.host_addr,
        )
    if config.transport == "peer":
        return PeerCopyTransport(load_probe(config.peer_probe_path), cold_fn)
    raise ValueError(f"unknown hetero transport {config.transport}")
