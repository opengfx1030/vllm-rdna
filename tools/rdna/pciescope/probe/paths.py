# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fabric flow model of the probe: capacities and max-min fair flow rates.

The PCI tree, the links and the path between two entities come from the shared
``pciescope.model.Topology`` (``Topology.path``). ``FabricModel`` turns the
physical links of a path into directed resources, ``"<link entity>:<up|down>"``
(``up`` = toward the root complex), each with the raw line rate of the link per
direction after line coding (``Link.max_gbs``), and computes the rate each of a
set of concurrent, saturating flows gets: the theoretical aggregate a measured
bandwidth is compared with.

Not resources (unlimited by design): the root-complex to root-complex data
fabric (the ``link=None`` pseudo hop) and the links inside a switch (no lanes).
The audit measured the real limit of the cross-root fabric (3 or more
concurrent flows collapse); the model deliberately does not claim to know it
[INFERENCE: docs/rdna2/pcie-iommu-audit-2026-11-06.md, "Cross-root P2P is
direct"].
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pciescope.model import (
    DEFAULT_MPS,
    DLLP_OVERHEAD,
    HOST,
    TLP_OVERHEAD_BYTES,
    Topology,
    gpu_entity,
)

HOST_READ = "host:read"
HOST_WRITE = "host:write"

# A resource counts as full when less than this fraction of its capacity is left.
_SATURATION_EPS = 1e-12


def _entity(token: str) -> str:
    """Return the entity id of ``token`` (``0a`` and ``gpu:0a`` -> ``gpu:0a``)."""
    t = token.strip().lower()
    if t == HOST or t.startswith(("gpu:", "port:")):
        return t
    return gpu_entity(t)


class FabricModel:
    """Directed capacities of a ``Topology`` and max-min fair flow rates.

    Resources are ``"<link entity>:<up|down>"`` for every physical link with a
    known ``max_gbs`` on a path (GB/s per direction, raw rate), plus
    ``host:read`` (host to GPU) and ``host:write`` (GPU to host) when
    ``host_gbps`` is not None: the DRAM side of host transfers.
    """

    def __init__(self, topo: Topology, *, host_gbps: float | None = None) -> None:
        self.topo = topo
        self.host_gbps = host_gbps

    def _limits(self, link_id: str) -> bool:
        link = self.topo.links.get(link_id)
        return link is not None and link.physical and bool(link.max_gbs)

    def resources(self, src: str, dst: str) -> tuple[str, ...]:
        """Return the directed resources of a ``src`` to ``dst`` flow, in path order.

        ``src``/``dst`` are buses (``0a``), entity ids (``gpu:0a``) or
        ``'host'``.

        Raises:
            ValueError: when both ends are the same entity.
            KeyError: when a GPU is not in the topology.

        """
        s, d = _entity(src), _entity(dst)
        if s == d:
            raise ValueError(f"a flow needs two different endpoints, got {src!r}")
        hops = self.topo.path(s, d).hops
        out = [
            f"{h.link}:{h.direction}"
            for h in hops
            if h.link is not None and self._limits(h.link)
        ]
        if self.host_gbps is not None:
            if s == HOST:
                out.insert(0, HOST_READ)
            elif d == HOST:
                out.append(HOST_WRITE)
        return tuple(out)

    def capacity(self, resource: str) -> float:
        """Return the capacity of ``resource`` in GB/s per direction.

        Raises:
            ValueError: for a resource the model does not know.

        """
        if resource in (HOST_READ, HOST_WRITE):
            if self.host_gbps is None:
                raise ValueError(f"{resource} is unlimited (no host_gbps given)")
            return self.host_gbps
        link_id, _, direction = resource.rpartition(":")
        if direction in ("up", "down") and self._limits(link_id):
            gbs = self.topo.links[link_id].max_gbs
            assert gbs is not None  # _limits() checked it
            return gbs
        raise ValueError(f"unknown resource {resource!r}")

    def bottleneck(self, src: str, dst: str) -> tuple[str, float]:
        """Return ``(resource, GB/s)`` with the smallest capacity on the path.

        The first of equal resources in path order wins.

        Raises:
            ValueError: when no resource limits the path.

        """
        path = self.resources(src, dst)
        if not path:
            raise ValueError(f"no limiting resource on the path {src!r} to {dst!r}")
        best = path[0]
        for res in path[1:]:
            if self.capacity(res) < self.capacity(best):
                best = res
        return best, self.capacity(best)

    def ideal_rates(self, flows: Sequence[tuple[str, str]]) -> list[float]:
        """Return the max-min fair rate in GB/s of each flow (progressive filling).

        Flows are independent and saturating: every active flow grows at the
        same pace until a resource is full, the flows through a full resource
        stop, the others keep growing. Each round freezes at least the flows
        through the tightest resource, so the loop ends after at most
        ``len(flows)`` rounds. Ties are broken by resource name: the result is
        deterministic. ``'host'`` may be either end of a flow. A flow without
        any limiting resource gets ``inf``.
        """
        paths = [self.resources(s, d) for s, d in flows]
        remaining = {r: self.capacity(r) for p in paths for r in p}
        total = dict(remaining)
        rates = [0.0] * len(paths)
        active = list(range(len(paths)))
        while active:
            users: dict[str, int] = {}
            for i in active:
                for res in paths[i]:
                    users[res] = users.get(res, 0) + 1
            if not users:
                for i in active:
                    rates[i] = math.inf
                break
            share, tight = min((remaining[r] / n, r) for r, n in users.items())
            for i in active:
                rates[i] += share
            full = {tight}
            for res, n in users.items():
                remaining[res] = max(0.0, remaining[res] - share * n)
                if remaining[res] <= _SATURATION_EPS * total[res]:
                    full.add(res)
            active = [i for i in active if full.isdisjoint(paths[i])]
        return rates

    def theoretical_aggregate(self, flows: Sequence[tuple[str, str]]) -> float:
        """Return the sum of ``ideal_rates`` in GB/s."""
        return math.fsum(self.ideal_rates(flows))

    @staticmethod
    def ceiling(raw_gbps: float, mps: int = DEFAULT_MPS) -> float:
        """Return the practical posted-write ceiling of a raw rate [INFERENCE].

        ``raw * mps / (mps + TLP_OVERHEAD_BYTES) * (1 - DLLP_OVERHEAD)``: header
        and framing bytes per TLP and the DLLP share of the wire, with the
        constants of ``pciescope.model``.
        """
        return raw_gbps * mps / (mps + TLP_OVERHEAD_BYTES) * (1.0 - DLLP_OVERHEAD)
