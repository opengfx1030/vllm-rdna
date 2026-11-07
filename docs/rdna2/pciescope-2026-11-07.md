# pciescope: PCIe fabric monitor and probe for par1-cs25 (2026-11-07)

Status: in progress on branch `feat/pciescope`. Core owns this document; the
Switch, Probe and TUI sections are written by their owners.

## Goal

Read and monitor the two Broadcom PEX88096 switches and the GPU links of
par1-cs25 while we work, and measure latency, bandwidth and effectiveness
(host to device, slot to slot inside a switch, between the switches). Surfaces:
a CLI with `--json` plus JSONL and Perfetto traces for agents, a TUI for
humans, and an MCP server.

## Fabric as discovered from sysfs

Filled from the live box by `python -m pciescope topo`.

## Usage

Filled when the CLI lands.

## Data model and metric definitions

Filled from `pciescope/model.py` (`METRICS`).

## Storage and traces

Filled when the store lands.

## MCP server

Filled when the MCP server lands.

## Switch registers (PEX88096)

Section owner: PlxSwitch.

Each root complex of par1-cs25 has one Broadcom PEX88096 (`1000:c010`,
ExpressFabric "Atlas", rev B0): switch A behind root `c0` (upstream port
`c5:00.0`), switch B behind root `00` (upstream port `01:00.0`). This section is
what those chips tell us without writing to them, how we read it, what it
costs, and the one thing that needs register writes (the traffic counters).
Code: `tools/rdna/pciescope/plx/` (decoders, root helper, CLI `python -m
pciescope.plx`) and `tools/rdna/pciescope/collectors/plx.py` (collector `plx`).
Everything marked [BOX] was measured on par1-cs25 (kernel
`7.1.7-100.p2p_post7.fc43`) with read-only BAR0 mappings; no switch or PCI
register was written. [SDK] marks facts taken from the Broadcom PLX SDK sources,
[INFERENCE] anything not observed.

### Verdict: read-only switch telemetry

| Telemetry | Read-only? | Source and evidence |
| --- | --- | --- |
| Link speed, width, up/down, training, max and target speed, link bandwidth flags (LBMS/LABS), degraded vs LnkCap | yes | LnkSta/LnkCap/LnkCtl2 in the port window. All 36 PEX functions decode identically to `lspci -vvv` (`test_pciescope_plx_decode.py`) [BOX] |
| Equalization complete at the current speed | yes | Phy16 status (16 GT/s) or LnkSta2 (8 GT/s); `ok` on all 12 GPU-facing and uplink links [BOX] |
| Sticky error flags: DevSta CE/NF/F/UR, AER uncorrectable and correctable status, lane error status, DPC trigger status | yes, but sticky | Same windows. Only the OS error handler clears them, pciescope never writes. They show that something happened, not when; the kernel `aer` collector has the counters [BOX] |
| Port settings: MPS, MRRS, ASPM, ACS control | yes | DevCtl, LnkCtl, ACS capability. MPS 256 on all 36 functions (they support 256 to 2048), MRRS 128 (512 on the management endpoints), ASPM off everywhere, ACS control 0 wherever ACS exists (peer-to-peer goes straight through) [BOX] |
| Port number, BDF, GPU map; captured bus per port; clock-enabled port masks; virtual-switch upstream port | yes | Port window header, LnkCap[31:24], vendor registers `0x30C..0x318`, `0x360`, `0x97C` [BOX] |
| Chip serial number | yes | Device Serial Number capability through BAR0 (`serial_number` in sysfs needs root) [BOX] |
| Per-port TLP/DLLP counters (bytes, TLPs, payload estimate, utilisation) | no, needs writes | PerformancePAK is off at boot (`0x3E0` = 0, `0x768[29]` = 0, `0xF30` = 0). Three register groups must be written, listed below; gated by `--allow-switch-writes` [BOX, SDK] |
| LTSSM state, recovery counts | no | No documented register. Undecoded vendor dwords exist (`0x7C8..0x7DC`, 6 per link-up port) and are not interpreted [BOX] |
| Flow-control credits, buffer occupancy, congestion | no | Only static, undecoded credit-like registers (`0x9D0..0x9E8`, `0xA00..0xA08`); no live gauge in the SDK [BOX, SDK] |
| Replay, bad TLP/DLLP, receiver-error counts | no | Not in the switch registers we can read. The kernel AER counters (`aer_dev_*`, collector `aer`) are the only counters [BOX] |

### Access paths and risk

| Path | Verdict |
| --- | --- |
| Configuration space (`/sys/bus/pci/devices/*/config`, `lspci`, `setpci`) | Standard capabilities only; every PLX-specific register reads 0 there. Used by the kernel, nothing to gain [BOX] |
| BAR0 of the upstream port (`resource0`, 16 MiB) | The path used. Port `N` has a 4 KiB window at `0x800000 + N * 0x1000` (SDK `ATLAS_REGS_AXI_BASE_ADDR` `0x60800000` = BAR0 `0x800000`). The window mirrors the port's configuration space (every non-zero config dword matched for ports 0, 48, 116) and adds 130 to 139 vendor dwords in `0x200..0xFFF`. Needs root; kernel lockdown is `none`; `flock` on `resource0` works [BOX] |
| Management endpoint (`1c:00.0` / `e0:00.0`, class `0x0180`, PEX port 247) | Not used. Memory decoding is disabled and there is no in-tree Linux driver for it (DeviceHunt lists `1000:00B2` as "PCIe Switch management endpoint"); reading it is pointless and unsafe to guess [BOX] |
| Broadcom PLX SDK service driver (`PlxSvc`) | Out of tree, not installed, kernel module with its own write paths. Used only as a reference for the register sequences |
| SMBus / I2C / UART / EEPROM | Not wired to the host, not touched |

Risks of the read path: a read of a vendor window register can have a side
effect (below). The helper therefore refuses any offset in `READ_SENSITIVE`
(`0x3E4`, `0x3F4`, `0x3F8`, `0x3FC`), non-aligned offsets, offsets outside the
window and ports that are neither clock-enabled nor a known virtual port. The
mapping is `PROT_READ` unless writes were explicitly enabled, and the helper has
no generic write operation: the only writes it can ever issue are the
PerformancePAK sequence below, and only when started with `--allow-writes`.

Process model: the registers need root, so the collector starts one long-lived
`sudo -n python -m pciescope.plx.helper` child (in-process when already root)
and sends it JSON lines. One batched read per switch gives all ports.

### Port map (port number, BDF, GPU)

The PEX port number is `LnkCap[31:24]` of the function; the map is derived from
the topology (rule below) and checked at start-up against both `LnkCap` and the
bus numbers in the window header, so a wrong map fails `available()` instead of
mislabelling data. A "station" is `port / 16`; the six stations 0..5 carry the
PerformancePAK counters.

| Port | Station | Switch A | GPU | Switch B | GPU |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0 | `c8:00.0` | `cb`: W7800, HIP 0, smi 5, no slot | `04:00.0` | `07`: W7800, HIP 5, smi 0, no slot, never touched |
| 16 | 1.0 | `c8:10.0` | `ce`: V620, HIP 1, smi 6, slot A | `04:10.0` | `0a`: V620, HIP 6, smi 1, slot B |
| 32 | 2.0 | `d0:00.0` | `d3`: V620, HIP 2, smi 7, slot A | `0c:00.0` | `0f`: V620, HIP 7, smi 2, slot B |
| 48 | 3.0 | `c5:00.0` (uplink, root port `c0:03.1`) | host | `01:00.0` (uplink, root port `00:01.1`) | host |
| 64 | 4.0 | `d5:00.0` | `d8`: V620, HIP 3, smi 8, slot A | `11:00.0` | `14`: V620, HIP 8, smi 3, slot B |
| 80 | 5.0 | `d5:10.0` | `db`: V620, HIP 4, smi 9, slot A | `11:10.0` | `17`: V620, HIP 9, smi 4, slot B |
| 116, 117 | 7.4, 7.5 | `dd:14.0`, `dd:15.0` | empty x1 slots | `19:14.0`, `19:15.0` | empty x1 slots |
| 224..227 | none | `c6:00.0`, `c6:04.0`, `c6:08.0`, `c6:0c.0` | internal downstream ports | `02:00.0`, `02:04.0`, `02:08.0`, `02:0c.0` | internal downstream ports |
| 232..235 | none | `c7:00.0`, `cf:00.0`, `d4:00.0`, `dc:00.0` | internal upstream ports | `03:00.0`, `0b:00.0`, `10:00.0`, `18:00.0` | internal upstream ports |
| 255 | none | `c6:1c.0` | GEP, parent of the management endpoint | `02:1c.0` | GEP, parent of the management endpoint |
| 247 | none | `e0:00.0` | management endpoint (never read) | `1c:00.0` | management endpoint (never read) |

HIP index, smi index and lease slot come from the topology (`Gpu.hip`, `.smi`,
`.slot`); the `ports` verb prints them in the columns `hip smi slot gpu`,
`ports --json` and `status --json` carry `hip`, `smi`, `slot`, `gpu_name`, and every
per-port sample is tagged `hip` (absent on non-GPU ports).

Rule: uplink = 48; the top-level downstream port by device number `0, 4, 8, 0xC`
is port `224..227` and device `0x1C` is the GEP `255`; a bridge below it is
internal upstream port `232 + device / 4`; a leaf downstream port is
`32 * j + device` for the `j`-th internal switch (`j` = 0 to 3) and its device
number on the bus. SDK special ports: `INT_DS` 224..227, `INT_UP` 232..235,
`MPT0` 247, `GEP_DS` 255.

Hierarchy inside each chip: root port, uplink (48), four internal downstream
ports, four internal upstream ports, then the leaf ports, then the on-card AMD
bridges (`1478`/`1479`) and the GPU. Consequence for peer-to-peer measurements:
ports 0 and 16 share one internal switch, ports 64 and 80 share another, port 32
sits alone, so slot-to-slot pairs (0,16) and (64,80) cross fewer virtual hops
than every other pair. All of it is inside one chip; none of it goes through
the host root port [BOX topology, INFERENCE for the latency consequence].

### Registers used

Per port function, read every sample through the window (offsets inside the
capability, capability offsets differ per role and are discovered by walking the
chain in the window):

| Register | Capability and offset | Used for |
| --- | --- | --- |
| DevCtl, DevSta | PCIe `+0x08` | MPS, MRRS, sticky CE/NF/F/UR |
| LnkCtl, LnkSta | PCIe `+0x10` | ASPM, speed, width, training, DL_Active, LBMS/LABS |
| LnkCtl2, LnkSta2 | PCIe `+0x30` | target speed, equalization (8 GT/s) |
| Uncorrectable status, correctable status | AER (`0xFB4`) `+0x04`, `+0x10` | sticky error bits |
| DPC status | DPC (`0xB60`) `+0x08` | containment triggered and reason |
| Lane error status | Secondary PCIe (`0xDB4`) `+0x08` | per-lane equalization/receiver errors |
| Phy 16 GT/s status | `0xD00` `+0x0C` | equalization complete at 16 GT/s |
| ACS capability and control | ACS (`0xF24`) `+0x04` | peer-to-peer redirection |

Vendor registers in the port window (the chip-level ones are read at the station
base ports `0, 16, 32, 48, 64, 80`):

| Offset | Content | Value on the box |
| --- | --- | --- |
| `0x30C`, `0x310`, `0x314`, `0x318` | clock-enabled port masks, 32 ports each | `00010001`, `00010001`, `00010001`, `00000003` (ports 0, 16, 32, 48, 64, 80 and 116, 117) |
| `0x350` | probe mode (bit 8 must be 0 for the counters) | `00040800` |
| `0x360` | virtual switch 0 upstream port | `00000030` (48) |
| `0x3E0` | PerformancePAK control (bit 31 enable) | 0 |
| `0x3E4` | counter FIFO, every read advances it | never read outside the counter sequence |
| `0x3F0` | counter RAM control | `00000021` |
| `0x3F4`, `0x3F8`, `0x3FC` | RAM read pointers, `+ (1 << 22)` per read | never read |
| `0x768` | bit 29 ingress counter enable (station base ports) | `09000000` |
| `0x97C` | bus `[7:0]` and device `[12:8]` the port captured for itself | `c8`, `10` for `c8:10.0` |
| `0xF30` | bit 6 egress counter enable, bit 21 egress credit enable (reads 0) | 0 |

### Traffic counters (PerformancePAK): the write gate

Per-port ingress and egress TLP and DLLP counters exist on the chip but are off
at boot, and the SDK enables them with register writes
(`PlxPciPerformanceMonitorControl`, `Driver/Source.PlxSvc/ApiFunc.c` line 2045;
`PlxPciPerformanceGetCounters` line 2842; statistics in
`PlxApi/PlxApi.c` `PlxPci_PerformanceCalcStatistics` line 5502 of
[xiallc/broadcom_pci_pcie_sdk](https://github.com/xiallc/broadcom_pci_pcie_sdk),
GPL-2.0; the same sources are mirrored in
[DaiZhiyuan/PlxSdk](https://github.com/DaiZhiyuan/PlxSdk)) [SDK]. There is no
read-only way to get traffic from the switch.

Off by default: the collector, the CLI and the helper never write. Nothing
below runs without `--allow-switch-writes` (collector option `allow_writes`);
`python -m pciescope.plx perf plan` prints the exact writes with the current and
the new value of every register and writes nothing.

Writes per switch (BAR0 of the upstream port, window of the named port),
restricted to the ports being measured (default: every physical port of stations
0..5 except the port of `gpu:07`, HIP 5, whose switch port is never written):

| Step | Register | Change | Observed before, after |
| --- | --- | --- | --- |
| 1 | `0x768` of each station base port | read-modify-write, set bit 29 | `09000000`, `29000000` |
| 2 | `0xF30` of each measured port | read-modify-write, set bits 6 and 21 | `00000000`, `00200040` |
| 3 | `0x3E0` of each station base port | write `0xD8000000` (enable, reset, infinite sampling, start) | `00000000`, `D8000000` |
| per poll, 10 Hz | `0x3F0` of each station | write `0x25` (reset RAM read pointer), then 224 dword reads of `0x3E4` | - |

Undo (on stop, EOF of the helper, SIGTERM, collector `close()`): `0x768` clear
bit 29, `0xF30` clear bit 6 and keep bit 21 set (as the SDK does), `0x3E0` write
`0x40000000`. If the monitor is already running (`0x3E0` bit 31 set) the helper
writes nothing but the per-poll capture and leaves the monitor on when it
stops. A second perf owner is refused by a `flock` on `resource0`.

Hazards to approve: `0xF30[21]` (egress credit enable) reads 0 but must stay
set by every write, a wrong value could stall egress credits of a GPU link and
possibly hang the host; the writes land in registers of live GPU links; a
`SIGKILL` of the helper leaves the monitor enabled (harmless, still counting);
SDK tools running at the same time would corrupt the RAM read pointers.

Counter layout per station (224 dwords, 32-bit, wraps in about 0.5 s at line
rate, so the helper polls at 10 Hz and publishes 64-bit unwrapped totals): per
port `p` of the station, ingress `[PH, PDW, NPH, NPDW, CPLH, CPLDW]` at `6p`,
egress the same at `96 + 6p`, ingress DLLPs at `192 + p`, egress DLLPs at
`208 + p`. Total bytes = `(PDW + NPDW + CPLDW) * 4 + 8 * DLLPs`. The payload
estimate subtracts 24 bytes per posted TLP (4 DW header with 64-bit address plus
2 DW framing) and 20 per completion; the SDK uses 20 for both [INFERENCE: not
calibrated against known traffic until the counters may run]. "rx" is traffic
received by the switch port from its link partner (GPU to switch on a leaf port,
host to switch on the uplink).

Validation plan once approved: idle run first on switch A (DLLPs tick, TLPs
about 0), then the probe's known-size transfers on one slot to calibrate the
payload model and the util figures.

### Using it

```bash
cd ~/Apps/pciescope-src   # PYTHONPATH=tools/rdna, venv ~/Apps/pciescope-venv
python -m pciescope.plx ports            # port <-> BDF <-> GPU map, no root
python -m pciescope.plx status [--json]  # one sample of every port (sudo -n)
python -m pciescope.plx dump 16 --switch B --decode
python -m pciescope.plx perf plan        # the writes above, nothing written
python -m pciescope.plx capture --out fixtures/plx_par1_cs25.json.gz
python -m pciescope.plx perf run --allow-switch-writes --duration 10   # gated
```

Every verb takes `--json`, `--switch A|B` and `--fixture FILE` (replay a
capture instead of touching hardware). The collector is `plx` in the registry
(`--collectors plx`); `--allow-switch-writes` is the collector option
`allow_writes`. Without root or passwordless `sudo` the collector is skipped with
the reason.

Metrics (entity `port:<short BDF>` unless noted; every sample is tagged `switch`,
`plx_port`, `role` and, for leaf and uplink ports, `peer`): `plx.link_speed_gts`,
`plx.link_width`, `plx.link_up`, `plx.link_training`, `plx.link_degraded`,
`plx.link_bw_status`, `plx.target_speed_gts`, `plx.eq_complete`, `plx.lane_err`,
`plx.dpc_status`, `plx.dev_status`, `plx.uncorr_status`, `plx.corr_status`,
`plx.mps_bytes`, `plx.mrrs_bytes`, `plx.aspm`, `plx.acs_ctl`; per `switch:<A|B>`
`plx.ports_up`, `plx.perf_enabled`, `plx.perf_ports`; only with the counters on:
`port.rx_bytes`, `port.tx_bytes` (counters), `port.rx_gbs`, `port.tx_gbs`,
`plx.rx_payload_bytes`, `plx.tx_payload_bytes`, `plx.rx_payload_gbs`,
`plx.tx_payload_gbs`, `plx.rx_tlps`, `plx.tx_tlps`, `plx.rx_dllps`,
`plx.tx_dllps`, `plx.rx_util`, `plx.tx_util`. The definitions (unit, kind, doc)
are in `pciescope/plx/metrics.py`. Events: `link_change`, `link_degraded`,
`link_restored`, `dpc_triggered`, `plx_error_flags`, `plx_perf_start`,
`plx_perf_error`. Virtual fan-out ports only report the error masks and
settings; the traffic counters of ports 116 and 117 (station 7) do not exist.

### Measured on par1-cs25 (read-only)

```text
$ python -m pciescope.plx ports --switch B      # excerpt
switch B  root 00  upstream 01:00.0 (port 48)  serial ?
  port stn bdf      kind     peer     hip smi slot  gpu
     0 0.0 04:00.0  dsp      gpu:07     5   0    -  Radeon Pro W7800 48GB
    16 1.0 04:10.0  dsp      gpu:0a     6   1    B  AMD RADEON PRO V620 Azure
    48 3.0 01:00.0  usp      host       -   -    -
   116 7.4 19:14.0  dsp                 -   -    -
   224   - 02:00.0  int_dsp             -   -    -

$ python -m pciescope.plx status --switch B     # excerpt
switch B  upstream 01:00.0  serial 00-80-5e-10-f9-57-20-61  traffic counters off
  port bdf      kind     peer     hip link            eq  flags
     0 04:00.0  dsp      gpu:07     5 16 GT/s x16     ok  CorrErr,UnsupReq
    16 04:10.0  dsp      gpu:0a     6 16 GT/s x16     ok  CorrErr,UnsupReq
    48 01:00.0  usp      host       - 16 GT/s x16     ok  -
   116 19:14.0  dsp                 - down            -   -
   224 02:00.0  int_dsp             - virtual         -   CorrErr,UnsupReq
382 samples in 2.99 ms
```

Cost of the collector (`bench_collect.py`, helper through `sudo -n`): one
`collect()` is 382 samples; 300 collects took mean 1.92 ms, p50 1.84, p95 2.38,
p99 3.21, max 4.37 ms; one batched read of 142 registers is 0.45 ms round trip
through the helper; starting the helper, dumping all windows and verifying the
map takes 126 ms once. (`ports` prints `serial ?` because the serial lives in
extended config space, which only root can read; `status` reads it through BAR0.)

What the numbers say: all 10 GPU-facing links and both uplinks run 16 GT/s x16
with equalization complete and no degradation; ports 116 and 117 are empty. The
sticky DevSta bits `CorrErr` and `UnsupReq` are set on nine functions of switch
B (the five GPU-facing ports, three internal downstream ports and the GEP) and
on none of switch A, while the AER status registers are clear everywhere. The
kernel counters since boot (read at 14:57, uptime 12 h 32 min) show
`NonFatalErr` (advisory non-fatal, in `aer_dev_correctable`) once on ten
functions of switch A (twice on `c6:0c.0`), on four on-card bridges and on
`gpu:cb`, and once on `02:0c.0` of switch B; each host root port received 10
correctable messages. An advisory non-fatal error is what a completer logs for
an Unsupported Request, so the pattern (DevSta cleared where the kernel
handled the event, still set on switch B) looks like enumeration-time probing
[INFERENCE]; it is not a live signal.

The BAR0 path is not a source of errors or wake-ups: three `status` runs, 300
collects and a `dump` changed none of the 327 `aer_dev_*` counter files, and a
300-collect run on a saved topology (no sysfs access at all) wrote no kernel
log line. The `pcieport ... Configured ACS to 0x0000` debug lines that this
kernel logs for `02:0c.0` and `c6:0c.0` (the runtime-suspended internal
downstream port above the two empty slots; 296 each in the current dmesg
buffer) appear about 2 s after a sysfs-based discovery (`pciescope topo`) and
not after BAR0 reads [two runs each; other tools on the box also trigger them].

### Tests, fixtures and disclosures

`tests/tools/rdna/pciescope/test_pciescope_plx_*.py` need no GPU and no root:
decoders against `lspci` output of all 36 functions, the port map, the helper's
read policy and write refusal on a fake `resource0` and through the real helper
process, the exact write sequence and the 32-bit unwrap on a stateful fake of
the counter RAM, the collector's metrics, events and gating, and the CLI.
Fixtures: `fixtures/plx_par1_cs25.json.gz` (topology plus the non-zero dwords of
every port window of both chips, read-only capture), `fixtures/plx_lspci_expect.json`.
What no test can show: real counter values, which need the write approval.

Disclosure: while exploring I read the window registers `0x3F4`, `0x3F8` and
`0x3FC` through BAR0 (a full-window read of a few ports); each read advanced the
register by `1 << 22` (switch A ports about 9 reads, switch B 1 to 2). They look
like the counter RAM read pointers, which the SDK resets with its `0x3F0` write
before every capture, so the effect should be nil [INFERENCE]. `0x3E4` was never
read. Since then no code path reads any of the four offsets.

References: [PEX88000 product brief BC-0484EN](https://docs.broadcom.com/doc/BC-0484EN);
[billauer.se on PLX switch registers outside configuration space](https://billauer.se/blog/2015/10/linux-plx-avago-pcie-switch-eeprom);
[mithro/plxtools](https://github.com/mithro/plxtools);
[DeviceHunt 1000:00B2](https://devicehunt.com/view/type/pci/vendor/1000/device/00B2).

## Active probe

Section owner: PcieProbe.

Code: `tools/rdna/pciescope/probe/`; the measuring binary is
`tools/rdna/p2p_probe/p2p_probe.cpp`, the audit tool of
[pcie-iommu-audit-2026-11-06.md](pcie-iommu-audit-2026-11-06.md) extended in
place (the legacy modes `info`, `lat`, `latc`, `bw` and `a2a` print
byte-identical output). Entry points: `python -m pciescope measure` (CLI,
`--json`), the MCP tool `measure`, the TUI Measure panel, and
`python -m pciescope.probe build|status` for the binary.

### What it measures

| Scenario | Probe command | Records (`Measurement`) |
|---|---|---|
| host to device and back: pinned, pageable, zero-copy kernel; one GPU, a whole switch, one GPU per switch; H2D and D2H at once | `hbw` | kind `bw`, direction `h2d` / `d2h`, tag `method` = `pinned` / `pageable` / `zero-copy`; several flows give one `concurrency` record with the aggregate |
| GPU to GPU: push (kernel stores), pull (kernel loads), SDMA (`hipMemcpyPeerAsync`); single pair, ring of 4, fan-in, fan-out, 2..4 concurrent cross-root pairs, bidirectional | `bw` | `bw` (one per flow; direction `d2d-push` / `d2d-pull` / `d2d-sdma`) and `concurrency` (the pattern aggregate, tag `pattern`) |
| one-way latency (flag ping-pong RTT/2, optional payload): pair by pair (`lat`) or disjoint pairs at once (`latc`) | `lat`, `latc` | `lat` in us |
| latency versus size 4 B .. 64 MiB for push, pull and SDMA (p50/p99 per size) | `curve` | `lat`, one record per size |
| host-timed copy latency per size; CPU to GPU doorbell round trip (CPU store, GPU poll and answer, CPU poll; not halved) | `hlat`, `hflag` | `lat` (`hflag`: direction `h2d`, size 0, `method=doorbell`) |
| one-shot all-to-all step as `rdna_ar` does it | `a2a` | `a2a` in us |

Every record carries `path_class` (`intra-switch`, `cross-root`, `host`),
`theoretical` (the fair share of the narrowest physical link of the route,
from `Topology.path` and the link capacities; shared links are split
max-min fairly between concurrent flows) and `efficiency` = value /
theoretical. Repeats of a cell are separate records (tags `rep`, `reps`);
`aggregate()` folds them into the median. One probe process runs a whole plan
(`batch`: one command per line, `READY`, a `###` echo, the rows and `END`
per command), so GPU initialisation is paid once.

### Suites

| Suite | Time | Content |
|---|---|---|
| `quick` | about 30 s | one pair per path class (latency, push/pull/SDMA at 1 MiB, SDMA at 64 MiB), host transfers on one GPU per switch, doorbell round trip, ring / fan-in / 3 cross-root pairs, whole-switch H2D and D2H |
| `standard` | about 5 min | quick plus latency and bandwidth curves, concurrency sweeps (1..4 cross-root pairs both ways), host concurrency, all-to-all; key cells run 3 times |
| `full` | about 15 min | standard plus every GPU pair (latency, push, pull, SDMA, bidirectional) and the doorbell of every GPU |

On the captured topology (steps, estimated time): quick on slot B 16 steps,
17 s, on both slots 28 steps, 28 s; standard 101 steps, 144 s on slot B and
266 steps, 333 s on both; full 235 steps, 245 s on slot B and 854 steps, 775 s
on both (repeats included). Selections that lack a path class say what was
skipped (`plan.skipped`).

### Safety

* **Lease.** A GPU is only touched if the caller holds its lease
  (`~/w4a8_runs/lanes/lease/<owner>.slotA|slotB|w7800cb`, owner from `--owner`
  or `$PCIESCOPE_OWNER`). Another holder of the same key refuses as well. HIP 5
  (W7800 bus 07) is never runnable without a lead-created `<owner>.w7800-07`.
  The refusal names the file that is needed. Nothing is launched before the
  lease check passes.
* **Quiet box.** No run while `docker buildx build` runs, `boot.lock` is held,
  or another agent's quiet file is newer than 6 min. `--wait-quiet SECONDS`
  waits, `--force-noisy` runs anyway and marks every result INVALID (exit 4).
  Lease and VRAM checks are never bypassed.
* **VRAM.** 3 GiB free per GPU (1 GiB for host-only plans); a GPU shared with
  a running server is refused. A dry run does not read the counters of a
  runtime-suspended GPU (reading wakes it and its bridge): such GPUs are
  listed in `asleep`.
* **Junction temperature** (hwmon `temp2_input` of the amdgpu device), sampled
  during the run: abort when the 5 s mean is at least 90 C or any sample at
  least 95 C (status `aborted`, exit 3, partial results kept).
* **Time.** Per run `3 x estimate + 60 s` (at least 120 s, at most 1 h, or
  `--max-seconds`), per step a limit from its estimate; the device spin loops
  of the new kernels have their own deadline (`P2P_KERNEL_TIMEOUT_S`, 10 s).
* **Evictions.** KFD `evicted_ms` of the probed GPUs is watched; a jump marks
  the window INVALID and records the process that caused it.
* **Links.** The best PCIe link state seen on the probed links during the run
  is kept; a downtrained link is reported in `guard.degraded_links` and as the
  finding `link-downtrained`.

Exit codes: 0 ok or dry run, 1 failed or partial, 2 refused, 3 aborted, 4
ran but INVALID. Events in the trace: `measure_start`, `measurement`,
`thermal`, `evicted_jump`, `measure_end`, `measure_refused`, `insitu_end`.

### Using it

```bash
cd ~/Apps/pciescope-src   # PYTHONPATH=tools/rdna, venv ~/Apps/pciescope-venv
python -m pciescope.probe status                      # binary, version, modes
python -m pciescope.probe build                       # fat gfx1030+gfx1100, under the build lock
python -m pciescope measure --list-suites
python -m pciescope measure --suite quick --gpus B --dry-run --json   # plan and blockers, no GPU work
python -m pciescope measure --suite quick --gpus B --owner PcieProbe --json   # needs PcieProbe.slotB
python -m pciescope measure --in-situ 60 --json       # passive utilisation of a running workload
```

`--gpus` takes buses (`0a`, `gpu:0f`), HIP indices (`hip:6`), `A` / `B`
(lease slots) or `w7800cb`. The JSON result has `status`, `exit_code`,
`valid`, `refusal` and `needed`, `blockers` (dry run), `plan`, `measurements`,
`summary` (repeats folded), `effectiveness` (`rows`, `by_class`, `findings`),
`guard` and `lease`. The same records are in the session trace, so
`pciescope report <session>` and the Matrix view show them.

### Effectiveness

Measured against the theoretical rate of the route and against the audit
numbers of `pcie-iommu-audit-2026-11-06.md`, per path class: single-flow push
below 70 % of the practical ceiling (28 GB/s for 16 GT/s x16) warns;
cross-root pull under half of the intra-switch pull, 3 or more concurrent
cross-root pairs under 75 % of one pair, a result 25 % above the raw capacity
(wrong model), latency above 1.25 x the audit baseline, host transfers under
60 % of the raw link and downtrained links are findings. Known audit
behaviour (pull collapse across roots, uplink sharing) is reported as `info`
when it matches the audit and as `bad` when it is worse.

### In-situ mode

`--in-situ SECONDS` injects nothing and needs no lease: it runs the passive
collectors and turns traffic counters into per-link, per-direction
utilisation (switch port counters of `plx`, or `gpu.pcie_rx/tx_bytes_per_s`
of amdgpu). On par1-cs25 neither source reports today: the PEX88096 counters
are not free-running (enabling them writes switch registers and needs lead
approval) and amdgpu has no `pcie_bw` on RDNA2/3 [INFERENCE: not tested on
every kernel]. The report then says `available: false` and why; the GPU
load in the window is still shown as context.

### Binary

`python -m pciescope.probe build` compiles `p2p_probe.cpp` with
`hipcc -x hip -O3 -std=c++17 --offload-arch=gfx1030 --offload-arch=gfx1100`
under `flock ~/w4a8_runs/audit/build.lock nice -n 19` and installs it
atomically as `~/Apps/pciescope/bin/p2p_probe` (override with `--binary` or
`$PCIESCOPE_PROBE_BIN`). `measure` runs `version` first and refuses a binary
that lacks a mode of the plan.

### Tests and what is not verified

`tests/tools/rdna/pciescope/test_pciescope_probe_*.py` need no GPU: the parser
and the result conversion run on real audit logs
(`fixtures/probe/audit/`), the planner, the lease and guard rules, the
runner and the end-to-end path (trace and report) on the captured sysfs of
par1-cs25 with a probe double. The double and its rows are labelled SYNTHETIC;
no number in these tests is a measurement of the new modes. The GPU code of
`hbw`, `hlat`, `hflag`, `curve` and `batch` has been compiled for both
architectures but not yet run: its numbers, the clock calibration
(`tick_mhz`) and the multi-block `curve` protocol need a leased GPU window.

## TUI

Section owner: PcieTui.

`pciescope tui` is a live terminal dashboard (Textual) over the passive
collectors, the recorded sessions and the active probe. It is built for an ssh
pty at 80x24 and uses the extra room of a larger terminal for wider
sparklines. Only one view is shown at a time (keys `1`-`5`).

```bash
python -m pciescope tui                       # live, records a session
python -m pciescope tui --no-record           # live, nothing written
python -m pciescope tui --replay latest       # replay (dir, name or latest)
python -m pciescope tui --replay <dir> --speed 8
ssh -tt par1-cs25 'cd ~/Apps/pciescope-src && PYTHONPATH=tools/rdna \
  ~/Apps/pciescope-venv/bin/python -m pciescope tui'
```

Flags: `--replay SESSION`, `--interval S` (sampling period, default 1.0),
`--collectors a,b` (default: the passive set), `--session-dir DIR`,
`--sysfs-root DIR` (a sysfs tree for dry runs), `--owner NAME` (lease owner for
the measure panel, default `$PCIESCOPE_OWNER`), `--no-record`, `--dry-run`
(measure panel plans and validates but launches nothing), `--speed X` (replay).
The TUI is passive: it reads sysfs/procfs only and writes nothing to the
switches or devices; the only thing that can touch a GPU is the measure panel,
and only under a lease held by `--owner`.

### Views

| Key | View | Content |
| --- | --- | --- |
| `1` | Topology | root complexes, switches (slot A/B, link), ports, GPUs (HIP, KFD node, arch, link speed x width, state colour), management endpoints, unused ports |
| `2` | Live | per GPU RX/TX GB/s, sparkline, busy, temperature, VRAM; per switch uplink h2d/d2h and utilisation; the collectors that run, their per-tick cost and those that are off |
| `3` | Matrix | latest probe results as a heatmap with efficiency, intra-switch versus cross-root summaries |
| `4` | Errors | AER counter deltas, PLX error registers, DPC and link events with timestamps, and the number of standing non-zero counters |
| `5` | Measure | starts a probe suite on leased GPUs and streams its progress |

The header shows the mode (`LIVE` or `REPLAY`), the recording indicator
(`● REC <session dir>`, or `not recording`), the clock, and `frozen` while the
display is paused. The last line lists the keys of the current view.

### Keys

| Key | Action |
| --- | --- |
| `1`-`5`, `tab`, `shift+tab` | switch view |
| `space` | live: freeze the display (recording continues); replay: play/pause |
| `r` | live: stop/resume recording (a new session is created when started without one) |
| `e` | export the session as a Perfetto trace, `<session>/trace.perfetto.json` |
| `left`, `right` | replay: seek -/+ 10 s (`shift`: 60 s); `home`, `end`: start/end |
| `+`, `-` | replay: speed x2 / x0.5 |
| `up`, `down` | scroll; on the measure view, select the suite |
| `d` | topology: detail chain (root, switch, port, GPU) |
| `m`, `[`, `]`, `v` | matrix: metric (bandwidth, efficiency, latency), transfer size, direction |
| `g` | measure: choose the GPU set |
| `s`, `x` | measure: start / cancel the suite |
| `c` | errors: clear the list (counters keep their baseline) |
| `?` | help; `q` quit |

### What "n/a" means

* RX/TX GB/s is `n/a` when no throughput counter exists. The amdgpu `pcie_bw`
  sysfs file is not exposed on RDNA2/RDNA3 (neither the V620 nor the W7800 on
  this box has it), and the PLX per-port counters are off until the switch
  register writes are approved (collector `plx`, see the switch section). The
  Live view says so and falls back to the GPU busy percentage for the history.
* The Matrix is empty until a session holds probe measurements: run the
  measure panel or load a session that has them.
* Errors are changes, not levels: a counter that is non-zero but constant is
  counted as "standing" and listed, a counter that moves becomes a timestamped row.

### Replay

Sessions are deduplicated (a series is stored when it changes, plus a 30 s
heartbeat); the TUI holds each series at its last value between records, so a
replayed sparkline looks like the live one. Seeking back rebuilds the view from
the start of the session. `--replay` needs `topology.json` in the session.

### Measure panel

The suites and the lease check come from the probe package
(`list_suites`, `check_leases`, `run_suite`). The panel checks the leases for
the chosen GPUs first, shows who holds them when they are missing, and refuses
to start (nothing is launched); HIP 5 (bus 07) is never offered. Progress
events (plan, step, guard, abort) stream into the panel and every measurement
is written to the recording and lands in the Matrix view as it completes. `x`
cancels between steps.

### Tests

`tests/tools/rdna/pciescope/test_pciescope_tui_*.py` need no GPU. The render
functions are pure and checked against golden screens in `golden_tui/`
(80x24 screens, plus 80/120/160 column renders), the app with Textual's pilot, and the replay against a real
34 s passive session recorded on the box (`fixtures/tui_real_session_*`).
`PCIESCOPE_UPDATE_GOLDEN=1` rewrites the goldens.

## Evidence

Filled with real output from the box.
