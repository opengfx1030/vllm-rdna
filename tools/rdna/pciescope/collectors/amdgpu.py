# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""amdgpu telemetry per GPU (sysfs only): load, memory, thermals, GPU-side link.

Primary source is the binary ``gpu_metrics`` table (format 1, content revisions
1.0-1.3, the PMFW view the driver itself serves ``gpu_busy_percent``,
``mem_busy_percent`` and the hwmon files from), read once per GPU; when the table
is absent or of another revision the individual sysfs/hwmon files are read
instead.  Added per GPU: ``mem_info_vram_used`` / ``mem_info_gtt_used`` and the
active ``pp_dpm_pcie`` level.  ``pcie_bw`` is only read when requested
(``pcie_bw=True``): on the ASICs that expose it the kernel blocks the read for
about a second, and RDNA2/RDNA3 (V620, W7800) do not expose it at all.  A GPU that
is runtime-suspended is not read (``gpu.asleep`` = 1): its attributes would resume it.

Helpers used by the active probe's safety guard: ``junction_temp_c`` and
``read_temps_c`` read the hwmon files only.
"""

from __future__ import annotations

import struct

from pciescope.collectors import BaseCollector
from pciescope.model import Gpu, Sample, Topology
from pciescope.model import now_ns as _now_ns
from pciescope.sysfs import Sysfs, is_awake

NAME = "amdgpu"

_INVALID_U16 = 0xFFFF
_MIN_METRICS_V1 = 0x50  # size of gpu_metrics_v1_0
_HWMON_TEMPS = (
    ("edge", "temp1_input", "gpu.temp_edge_c"),
    ("junction", "temp2_input", "gpu.temp_junction_c"),
    ("mem", "temp3_input", "gpu.temp_mem_c"),
)


def parse_gpu_metrics(raw: bytes | None) -> dict[str, float] | None:
    """Decode the common prefix of ``gpu_metrics`` v1.0-v1.3 (None otherwise).

    Offsets (bytes): 0 header (u16 size, u8 format, u8 content), 4 temperature
    edge/hotspot/mem (u16 C), 0x10 average gfx/umc/mm activity (u16 %), 0x16 average
    socket power (u16 W), 0x44 throttle_status (u32), 0x48 fan speed (u16),
    0x4a pcie_link_width (u16), 0x4c pcie_link_speed (u16, 0.1 GT/s), and for
    v1.3 0x70 indep_throttle_status (u64).  0xFFFF marks an unavailable field.
    """
    if raw is None or len(raw) < 4:
        return None
    size, fmt, content = struct.unpack_from("<HBB", raw, 0)
    if fmt != 1 or content > 3 or size < _MIN_METRICS_V1 or len(raw) < size:
        return None
    t_edge, t_hot, t_mem = struct.unpack_from("<HHH", raw, 4)
    gfx, umc, _mm, power = struct.unpack_from("<HHHH", raw, 0x10)
    (throttle,) = struct.unpack_from("<I", raw, 0x44)
    _fan, width, speed = struct.unpack_from("<HHH", raw, 0x48)
    out: dict[str, float] = {
        "format_revision": float(fmt),
        "content_revision": float(content),
        "throttle_status": float(throttle),
    }
    if content == 3 and size >= 0x78:
        (indep,) = struct.unpack_from("<Q", raw, 0x70)
        out["throttle_status"] = float(indep)
    for key, val in (
        ("temp_edge_c", t_edge),
        ("temp_junction_c", t_hot),
        ("temp_mem_c", t_mem),
        ("busy_pct", gfx),
        ("mem_busy_pct", umc),
        ("power_w", power),
    ):
        if val != _INVALID_U16:
            out[key] = float(val)
    if width not in (0, _INVALID_U16) and speed not in (0, _INVALID_U16):
        out["link_width"] = float(width)
        out["link_speed_gts"] = speed / 10.0
    return out


def parse_pp_dpm_pcie(text: str | None) -> tuple[float, int] | None:
    """Active level of ``pp_dpm_pcie`` as (GT/s, lanes).

    Lines look like ``1: 16.0GT/s, x16 619Mhz *``; the active one ends in ``*``.
    """
    if not text:
        return None
    for line in text.splitlines():
        if not line.rstrip().endswith("*"):
            continue
        try:
            _, rest = line.split(":", 1)
            speed_txt, width_txt = rest.split(",", 1)
            speed = float(speed_txt.strip().lower().replace("gt/s", ""))
            width = int(width_txt.split()[0].lstrip("xX"))
        except (ValueError, IndexError):
            return None
        return speed, width
    return None


def parse_pcie_bw(text: str | None) -> tuple[int, int, int] | None:
    """``pcie_bw`` file: ``<received packets> <sent packets> <max payload B>``."""
    if not text:
        return None
    try:
        rx, tx, mps = (int(x) for x in text.split()[:3])
    except ValueError:
        return None
    return rx, tx, mps


def read_temps_c(sysfs: Sysfs, gpu: Gpu) -> dict[str, float]:
    """Hwmon temperatures of ``gpu`` in degrees C (keys edge/junction/mem)."""
    out: dict[str, float] = {}
    if not gpu.hwmon:
        return out
    for key, fname, _metric in _HWMON_TEMPS:
        milli = sysfs.read_int(f"{gpu.hwmon}/{fname}")
        if milli is not None:
            out[key] = milli / 1000.0
    return out


def junction_temp_c(sysfs: Sysfs, gpu: Gpu) -> float | None:
    """Junction (hotspot) temperature of ``gpu`` in degrees C, None if unreadable."""
    if not gpu.hwmon:
        return None
    milli = sysfs.read_int(f"{gpu.hwmon}/temp2_input")
    return None if milli is None else milli / 1000.0


class AmdgpuCollector(BaseCollector):
    """Per-GPU load, memory, thermal and GPU-side link telemetry."""

    name = NAME

    def __init__(
        self,
        topo: Topology,
        sysfs: Sysfs | None = None,
        *,
        gpu_metrics: bool = True,
        dpm: bool = True,
        pcie_bw: bool = False,
    ) -> None:
        super().__init__(topo, sysfs)
        self.use_metrics = gpu_metrics
        self.use_dpm = dpm
        self.use_pcie_bw = pcie_bw
        self._gpus = topo.gpu_list("bus")
        self._base = {g.id: f"bus/pci/devices/{_sysname(g.bdf)}" for g in self._gpus}
        self._pcie_bw_prev: dict[str, tuple[int, int, int]] = {}

    def available(self) -> tuple[bool, str]:
        if not self._gpus:
            return False, "topology has no GPU"
        base = self._base[self._gpus[0].id]
        # exists(), not a read: a read would resume a runtime-suspended GPU
        if not (
            self.sysfs.exists(f"{base}/gpu_busy_percent")
            or self.sysfs.exists(f"{base}/gpu_metrics")
        ):
            return False, "amdgpu sysfs attributes are missing"
        return True, ""

    def pcie_bw_available(self) -> bool:
        """True if the first GPU exposes ``pcie_bw`` (not the case on RDNA2/3)."""
        return bool(self._gpus) and self.sysfs.exists(
            f"{self._base[self._gpus[0].id]}/pcie_bw"
        )

    def collect(self) -> list[Sample]:
        ts = _now_ns()
        out: list[Sample] = []
        for g in self._gpus:
            base = self._base[g.id]
            ent = g.id
            awake = is_awake(self.sysfs.runtime_status(g.bdf))
            out.append(Sample(ts, ent, "gpu.asleep", float(not awake), "bool"))
            if not awake:  # reading its attributes would resume the GPU
                continue
            met = (
                parse_gpu_metrics(self.sysfs.read_bytes(f"{base}/gpu_metrics"))
                if self.use_metrics
                else None
            )
            if met is not None:
                self._from_metrics(out, ts, ent, met)
            else:
                self._from_sysfs(out, ts, ent, base, g)
            vram = self.sysfs.read_int(f"{base}/mem_info_vram_used")
            if vram is not None:
                out.append(Sample(ts, ent, "gpu.vram_used_bytes", float(vram), "B"))
            if g.vram_total_bytes:
                out.append(
                    Sample(
                        ts, ent, "gpu.vram_total_bytes", float(g.vram_total_bytes), "B"
                    )
                )
            gtt = self.sysfs.read_int(f"{base}/mem_info_gtt_used")
            if gtt is not None:
                out.append(Sample(ts, ent, "gpu.gtt_used_bytes", float(gtt), "B"))
            if self.use_dpm:
                lvl = parse_pp_dpm_pcie(self.sysfs.read(f"{base}/pp_dpm_pcie"))
                if lvl is not None:
                    out.append(
                        Sample(ts, ent, "gpu.pcie_dpm_speed_gts", lvl[0], "GT/s")
                    )
                    out.append(
                        Sample(ts, ent, "gpu.pcie_dpm_width", float(lvl[1]), "lanes")
                    )
            if self.use_pcie_bw:
                self._pcie_bw(out, ts, ent, base)
        return out

    @staticmethod
    def _from_metrics(
        out: list[Sample], ts: int, ent: str, met: dict[str, float]
    ) -> None:
        for key, metric, unit in (
            ("busy_pct", "gpu.busy_pct", "%"),
            ("mem_busy_pct", "gpu.mem_busy_pct", "%"),
            ("temp_edge_c", "gpu.temp_edge_c", "C"),
            ("temp_junction_c", "gpu.temp_junction_c", "C"),
            ("temp_mem_c", "gpu.temp_mem_c", "C"),
            ("power_w", "gpu.power_w", "W"),
            ("link_speed_gts", "gpu.link_speed_gts", "GT/s"),
            ("link_width", "gpu.link_width", "lanes"),
            ("throttle_status", "gpu.throttle_status", "mask"),
        ):
            if key in met:
                out.append(Sample(ts, ent, metric, met[key], unit))

    def _from_sysfs(
        self, out: list[Sample], ts: int, ent: str, base: str, g: Gpu
    ) -> None:
        busy = self.sysfs.read_int(f"{base}/gpu_busy_percent")
        if busy is not None:
            out.append(Sample(ts, ent, "gpu.busy_pct", float(busy), "%"))
        mbusy = self.sysfs.read_int(f"{base}/mem_busy_percent")
        if mbusy is not None:
            out.append(Sample(ts, ent, "gpu.mem_busy_pct", float(mbusy), "%"))
        if g.hwmon:
            for _key, fname, metric in _HWMON_TEMPS:
                milli = self.sysfs.read_int(f"{g.hwmon}/{fname}")
                if milli is not None:
                    out.append(Sample(ts, ent, metric, milli / 1000.0, "C"))
            micro = self.sysfs.read_int(f"{g.hwmon}/power1_average")
            if micro is not None:
                out.append(Sample(ts, ent, "gpu.power_w", micro / 1e6, "W"))

    def _pcie_bw(self, out: list[Sample], ts: int, ent: str, base: str) -> None:
        """Bytes/s from the ``pcie_bw`` packet counts (one second per read)."""
        got = parse_pcie_bw(self.sysfs.read(f"{base}/pcie_bw"))
        if got is None:
            return
        rx, tx, mps = got
        out.append(Sample(ts, ent, "gpu.pcie_rx_bytes_per_s", float(rx * mps), "B/s"))
        out.append(Sample(ts, ent, "gpu.pcie_tx_bytes_per_s", float(tx * mps), "B/s"))


def _sysname(bdf: str) -> str:
    return bdf if bdf.count(":") == 2 else f"0000:{bdf}"


def make(
    topo: Topology,
    *,
    sysfs: Sysfs | None = None,
    gpu_metrics: bool = True,
    dpm: bool = True,
    pcie_bw: bool = False,
    **_opts: object,
) -> AmdgpuCollector:
    return AmdgpuCollector(
        topo, sysfs, gpu_metrics=gpu_metrics, dpm=dpm, pcie_bw=pcie_bw
    )
