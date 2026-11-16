# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metric rows of the ``plx.*`` namespace (PEX88096 switch telemetry).

Plain tuples ``(name, unit, kind, entity, source, doc)`` so ``pciescope.model`` can
build ``MetricDef`` objects from them without importing the collector (no import
cycle); ``pciescope.collectors.plx.METRICS`` exposes the same rows as ``MetricDef``.

Entities: ``port:<bdf>`` is the switch port function (a leaf downstream port is
the link of a GPU, the upstream port ``01:00.0`` / ``c5:00.0`` the host uplink,
the internal ports are the virtual fan-out); ``switch:<A|B>``.  Every sample of a
port carries the tags ``switch``, ``plx_port`` (PEX port number), ``role`` and, for
leaf and upstream ports, ``peer`` (``gpu:<bus>`` or ``host``).

``rx`` / ``tx`` follow ``port.rx_bytes`` of the core model: ``rx`` = received by the
switch port from its link partner (GPU -> switch on a downstream port, host ->
switch on the upstream port), ``tx`` = transmitted to the partner.  The two
generic names ``port.rx_bytes`` / ``port.tx_bytes`` (and ``*_gbs``) carry the
traffic counters so the core report can use them; the PEX-specific detail
(payload estimate, TLP and DLLP counts, utilisation) lives in ``plx.*``.
"""

from __future__ import annotations

NAMESPACE = "plx"
SOURCE = "plx"

Row = tuple[str, str, str, str, str, str]

_STICKY = "Sticky: only the OS error handler clears it, pciescope never writes."


def _port(name: str, unit: str, doc: str, kind: str = "gauge") -> Row:
    return (f"plx.{name}", unit, kind, "port", SOURCE, doc)


def _core(name: str, unit: str, doc: str, kind: str = "gauge") -> Row:
    """A generic ``port.*`` metric produced by this source."""
    return (f"port.{name}", unit, kind, "port", SOURCE, doc)


def _switch(name: str, unit: str, doc: str) -> Row:
    return (f"plx.{name}", unit, "gauge", "switch", SOURCE, doc)


ROWS: tuple[Row, ...] = (
    _port(
        "link_speed_gts",
        "GT/s",
        "Current link speed of the switch port (LnkSta read through the PEX BAR0 "
        "window). Virtual fan-out ports are not sampled for it.",
    ),
    _port(
        "link_width",
        "lanes",
        "Negotiated link width of the switch port (LnkSta); 0 = no link.",
    ),
    _port(
        "link_up",
        "bool",
        "1 if the port has a link: DL_Active for downstream ports, negotiated "
        "width > 0 for ports that cannot report DL_Active.",
    ),
    _port(
        "link_training",
        "bool",
        "LnkSta Link Training bit: 1 while the port is retraining.",
    ),
    _port(
        "link_degraded",
        "bool",
        "1 if the link is up but below the port's own LnkCap maximum speed or "
        "width (a GPU may legitimately drop to a lower speed when idle).",
    ),
    _port(
        "eq_complete",
        "bool",
        "Equalization complete at the current speed (16 GT/s status register, "
        "LnkSta2 at 8 GT/s); only emitted at 8 GT/s and above.",
    ),
    _port(
        "link_bw_status",
        "mask",
        "LnkSta bandwidth-management flags: 1 = LBMS (the link changed speed or "
        "width), 2 = LABS (hardware-autonomous change). " + _STICKY,
    ),
    _port(
        "dev_status",
        "mask",
        "DevSta error flags: 1 = correctable, 2 = non-fatal, 4 = fatal, "
        "8 = unsupported request seen. " + _STICKY,
    ),
    _port(
        "uncorr_status",
        "mask",
        "AER uncorrectable error status register (raw bits, kernel bit names in "
        "aer_dev_nonfatal). " + _STICKY,
    ),
    _port(
        "corr_status",
        "mask",
        "AER correctable error status register (bit 0 RxErr, 6 BadTLP, 7 BadDLLP, "
        "8 Rollover, 12 Timeout, 13 NonFatalErr). " + _STICKY,
    ),
    _port(
        "lane_err",
        "mask",
        "Secondary PCIe capability lane error status (one bit per lane that saw "
        "an error during equalization). " + _STICKY,
    ),
    _port(
        "dpc_status",
        "mask",
        "DPC status register of downstream ports (bit 0 = containment triggered, "
        "bits 2:1 reason); 0 = never triggered.",
    ),
    _port(
        "target_speed_gts",
        "GT/s",
        "LnkCtl2 target link speed the port tries to train to.",
    ),
    _port(
        "mps_bytes",
        "B",
        "Max_Payload_Size programmed in DevCtl of the switch port.",
    ),
    _port(
        "mrrs_bytes",
        "B",
        "Max_Read_Request_Size programmed in DevCtl of the switch port.",
    ),
    _port(
        "aspm",
        "mask",
        "LnkCtl ASPM control: 1 = L0s, 2 = L1, 3 = both, 0 = disabled.",
    ),
    _port(
        "acs_ctl",
        "mask",
        "ACS control bits (1 source validation, 2 translation blocking, 4 P2P "
        "request redirect, 8 completion redirect, 16 upstream forwarding, 32 "
        "egress control, 64 direct translated P2P); 0 = direct peer-to-peer.",
    ),
    _core(
        "rx_bytes",
        "B",
        "TLP + DLLP bytes received by the switch on this port from its link "
        "partner (PerformancePAK counters; need --allow-switch-writes). 64-bit, "
        "the 32-bit chip counters are unwrapped by the helper.",
        "counter",
    ),
    _core(
        "tx_bytes",
        "B",
        "TLP + DLLP bytes transmitted by the switch on this port to its link "
        "partner (PerformancePAK counters).",
        "counter",
    ),
    _core(
        "rx_gbs",
        "GB/s",
        "Rate of port.rx_bytes over the last counter interval (TLP + DLLP bytes, "
        "comparable with link.capacity_gbs).",
    ),
    _core(
        "tx_gbs",
        "GB/s",
        "Rate of port.tx_bytes over the last counter interval.",
    ),
    _port(
        "rx_payload_bytes",
        "B",
        "Estimated data payload received (posted writes + completions, headers "
        "removed); a model, see the switch section of the doc.",
        "counter",
    ),
    _port(
        "tx_payload_bytes",
        "B",
        "Estimated data payload transmitted.",
        "counter",
    ),
    _port(
        "rx_payload_gbs",
        "GB/s",
        "Rate of plx.rx_payload_bytes over the last counter interval.",
    ),
    _port(
        "tx_payload_gbs",
        "GB/s",
        "Rate of plx.tx_payload_bytes over the last counter interval.",
    ),
    _port(
        "rx_tlps",
        "count",
        "TLPs received (posted + non-posted + completion headers).",
        "counter",
    ),
    _port("tx_tlps", "count", "TLPs transmitted.", "counter"),
    _port(
        "rx_dllps",
        "count",
        "DLLPs received (ACK/NAK and flow-control updates, so non-zero when idle).",
        "counter",
    ),
    _port("tx_dllps", "count", "DLLPs transmitted.", "counter"),
    _port(
        "rx_util",
        "%",
        "TLP + DLLP bytes received over the raw capacity of the link at its "
        "current speed and width (clamped to 100).",
    ),
    _port(
        "tx_util",
        "%",
        "TLP + DLLP bytes transmitted over the raw link capacity.",
    ),
    _switch(
        "perf_enabled",
        "bool",
        "1 if the PerformancePAK traffic counters of this switch are running "
        "(they are off at boot and need --allow-switch-writes).",
    ),
    _switch(
        "perf_ports",
        "count",
        "Number of ports whose traffic counters pciescope enabled.",
    ),
    _switch(
        "ports_up",
        "count",
        "Physical ports of the switch (upstream + downstream) with a link up.",
    ),
)
