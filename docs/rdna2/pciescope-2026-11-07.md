# pciescope: PCIe fabric monitor and probe for par1-cs25 (2026-11-07)

Status: implemented on branch `feat/pciescope` (not merged), passive parts verified on
par1-cs25 (see "Evidence"); active probe runs wait for a GPU window ("Risks and
approvals"). Core owns this document; the Switch, Probe and TUI sections are written
by their owners.

## Goal

Read and monitor the two Broadcom PEX88096 switches and the GPU links of
par1-cs25 while we work, and measure latency, bandwidth and effectiveness
(host to device, slot to slot inside a switch, between the switches). Surfaces:
a CLI with `--json` plus JSONL and Perfetto traces for agents, a TUI for
humans, and an MCP server.

## Fabric as discovered from sysfs

`python -m pciescope topo` needs no root and writes nothing. It builds the model from
`/sys/bus/pci/devices`, the KFD topology (`/sys/class/kfd/kfd/topology/nodes`) and the
amdgpu/DRM links.

The box changed on 2026-11-08 (case move): the second switch and both W7800 cards are
gone, and six V620 sit behind ONE PEX88096 with x8 links ("Switch registers" covers what
the register reader makes of this chip). The box rules call this layout interim until the
replacement switch arrives. Real output on par1-cs25 on 2026-11-10 (EPYC 7K62, kernel
`7.1.7-100.p2p_post7.fc43`):

```text
par1-cs25: 1 root complex(es), 1 switch(es), 6 GPU(s); HIP order from kfd-node-order
root complex c0 (switch A)
  └─ c0:03.1 root port [16 GT/s x16]
     └─ c5:00.0 switch A upstream port [16 GT/s x16]
        ├─ c6:00.0 switch A fabric port [16 GT/s x16]
        │  └─ c7:00.0 switch A sub-switch [16 GT/s x16]
        │     ├─ c8:10.0 downstream port [unused: down]
        │     └─ c8:18.0 downstream port [unused: down]
        ├─ c6:04.0 switch A fabric port [16 GT/s x16]
        │  └─ cb:00.0 switch A sub-switch [16 GT/s x16]
        │     ├─ cc:00.0 downstream port [unused: down]
        │     ├─ cc:08.0 downstream port [unused: down]
        │     ├─ cc:10.0 port -> gpu:d1 AMD RADEON PRO V620 Azure gfx1030 HIP 0 smi 0 kfd-node 1 slot A [16 GT/s x8 15.8 GB/s raw]
        │     └─ cc:18.0 port -> gpu:d4 AMD RADEON PRO V620 Azure gfx1030 HIP 1 smi 1 kfd-node 2 slot A [16 GT/s x8 15.8 GB/s raw]
        ├─ c6:08.0 switch A fabric port [16 GT/s x16]
        │  └─ d5:00.0 switch A sub-switch [16 GT/s x16]
        │     ├─ d6:00.0 port -> gpu:d9 AMD RADEON PRO V620 Azure gfx1030 HIP 2 smi 2 kfd-node 3 slot B [16 GT/s x8 15.8 GB/s raw]
        │     ├─ d6:08.0 port -> gpu:dc AMD RADEON PRO V620 Azure gfx1030 HIP 3 smi 3 kfd-node 4 slot B [16 GT/s x8 15.8 GB/s raw]
        │     ├─ d6:10.0 port -> gpu:df AMD RADEON PRO V620 Azure gfx1030 HIP 4 smi 4 kfd-node 5 slot B [16 GT/s x8 15.8 GB/s raw]
        │     └─ d6:18.0 port -> gpu:e2 AMD RADEON PRO V620 Azure gfx1030 HIP 5 smi 5 kfd-node 6 slot B [16 GT/s x8 15.8 GB/s raw]
        ├─ c6:0c.0 switch A fabric port [16 GT/s x16]
        │  └─ e3:00.0 switch A sub-switch [16 GT/s x16]
        │     ├─ e4:14.0 downstream port [unused: down]
        │     └─ e4:15.0 downstream port [unused: down]
        └─ c6:1c.0 switch A fabric port [16 GT/s x16]
           └─ e7:00.0 switch A management endpoint [16 GT/s x16]
```

Reading the tree:

- One root complex (`c0`), one root port (`c0:03.1`, 16 GT/s x16) and one switch with its
  upstream port at `c5:00.0`. Inside it: five virtual *fabric* ports (`c6:xx.0`), four
  sub-switches below them and the management endpoint (`e7:00.0`). Fabric ports and
  sub-switches carry no lanes, so the five links to them are `virtual`; `report` hides
  them (`--virtual` shows them).
- The box rules call the two groups of cards *leaves*: leaf A is the sub-switch
  `cb:00.0` (V620 `d1` on `cc:10.0`, `d4` on `cc:18.0`), leaf B is `d5:00.0` (`d9`,
  `dc`, `df`, `e2` on `d6:00.0`, `d6:08.0`, `d6:10.0`, `d6:18.0`). The sub-switches
  `c7:00.0` and `e3:00.0` have no card; their ports (and `cc:00.0`, `cc:08.0`) are
  unused.
- 13 physical links: the host uplink (x16), six GPU links (`gpu_link`, x8: 15.754 GB/s
  raw and 14.015 GB/s practical per direction) and six on-card bridge links (`onboard`,
  x16). The x8 GPU links are the port's configured maximum (`links: 13 physical, 13 at
  their maximum`), not a degradation.
- HIP order = KFD node order = PCI order: HIP 0 to 5 are `d1 d4 d9 dc df e2`.
- `Topology.path`: every GPU pair is `intra-switch` (one chip). A same-leaf pair
  (`d9`-`dc`) stays below `d5:00.0`; a cross-leaf pair (`d1`-`d9`) crosses the chip's
  virtual fabric; neither touches the host uplink. Every route is limited by an x8 GPU
  link (15.754 GB/s raw); host traffic of all six GPUs shares the one x16 uplink.
- `c7:00.0`, `c8:10.0`, `c8:18.0`, `e3:00.0`, `e4:14.0`, `e4:15.0` were runtime suspended
  when this was recorded (see "Never wake what sleeps").

### Layouts and lease slots

`Gpu.slot` is the lease slot letter of a V620: the unit a lease holder file covers. It is
a policy of the box rules, not a property of the wiring (it was a switch until
2026-11-08, it is a leaf of the single switch since), so `Gpu.switch` (the chip, `A`
here for all six) and `Gpu.slot` differ. `topology.DOCUMENTED_LAYOUTS` holds the layouts
the box rules have documented:

| Layout | GPU buses, HIP order | Slot A | Slot B | Holder files |
| --- | --- | --- | --- | --- |
| six V620 behind one switch (interim, since 2026-11-08) | `d1 d4 d9 dc df e2` | `d1 d4` (leaf A, HIP 0-1) | `d9 dc df e2` (leaf B, HIP 2-5) | `<owner>.a2`, `<owner>.b4` |
| ten GPUs behind two switches (2026-11-05) | `cb ce d3 d8 db 07 0a 0f 14 17` | `ce d3 d8 db` (switch A) | `0a 0f 14 17` (switch B) | `<owner>.slotA`, `<owner>.slotB`; W7800: `w7800cb`, `w7800-07` |

The layout is chosen by the set of GPU buses. If the set matches but the HIP order
differs, `topo` adds the note `HIP order (KFD node order) differs from the box rules
table (<layout>): ...`; a set that matches no layout gets the switch letter as slot and
the note `GPU buses ... match no layout of the box rules table`. Add an entry to the
table in `topology.py` when the hardware changes again (the replacement switch). The
probe's `--gpus A` / `B` and the TUI's slot presets read `Topology.slot_gpus`, so they
follow the layout; the holder-file names themselves live in the probe's lease table.

### Before the case move (2026-11-07)

The same command on 2026-11-07 (ten GPUs, two switches). The Evidence section records
its first group of runs on this layout.

```text
par1-cs25: 2 root complex(es), 2 switch(es), 10 GPU(s); HIP order from kfd-node-order
root complex c0 (switch A)
  └─ c0:03.1 root port [16 GT/s x16]
     └─ c5:00.0 switch A upstream port [16 GT/s x16]
        ├─ c6:00.0 switch A fabric port [16 GT/s x16]
        │  └─ c7:00.0 switch A sub-switch [16 GT/s x16]
        │     ├─ c8:00.0 port -> gpu:cb Radeon Pro W7800 48GB gfx1100 HIP 0 smi 5 kfd-node 1 [16 GT/s x16 31.5 GB/s raw]
        │     └─ c8:10.0 port -> gpu:ce AMD RADEON PRO V620 Azure gfx1030 HIP 1 smi 6 kfd-node 2 slot A [16 GT/s x16 31.5 GB/s raw]
        ├─ c6:04.0 switch A fabric port [16 GT/s x16]
        │  └─ cf:00.0 switch A sub-switch [16 GT/s x16]
        │     └─ d0:00.0 port -> gpu:d3 AMD RADEON PRO V620 Azure gfx1030 HIP 2 smi 7 kfd-node 3 slot A [16 GT/s x16 31.5 GB/s raw]
        ├─ c6:08.0 switch A fabric port [16 GT/s x16]
        │  └─ d4:00.0 switch A sub-switch [16 GT/s x16]
        │     ├─ d5:00.0 port -> gpu:d8 AMD RADEON PRO V620 Azure gfx1030 HIP 3 smi 8 kfd-node 4 slot A [16 GT/s x16 31.5 GB/s raw]
        │     └─ d5:10.0 port -> gpu:db AMD RADEON PRO V620 Azure gfx1030 HIP 4 smi 9 kfd-node 5 slot A [16 GT/s x16 31.5 GB/s raw]
        ├─ c6:0c.0 switch A fabric port [16 GT/s x16]
        │  └─ dc:00.0 switch A sub-switch [16 GT/s x16]
        │     ├─ dd:14.0 downstream port [unused: down]
        │     └─ dd:15.0 downstream port [unused: down]
        └─ c6:1c.0 switch A fabric port [16 GT/s x16]
           └─ e0:00.0 switch A management endpoint [16 GT/s x16]
root complex 00 (switch B)
  └─ 00:01.1 root port [16 GT/s x16]
     └─ 01:00.0 switch B upstream port [16 GT/s x16]
        ├─ 02:00.0 switch B fabric port [16 GT/s x16]
        │  └─ 03:00.0 switch B sub-switch [16 GT/s x16]
        │     ├─ 04:00.0 port -> gpu:07 Radeon Pro W7800 48GB gfx1100 HIP 5 smi 0 kfd-node 6 [16 GT/s x16 31.5 GB/s raw]
        │     └─ 04:10.0 port -> gpu:0a AMD RADEON PRO V620 Azure gfx1030 HIP 6 smi 1 kfd-node 7 slot B [16 GT/s x16 31.5 GB/s raw]
        ├─ 02:04.0 switch B fabric port [16 GT/s x16]
        │  └─ 0b:00.0 switch B sub-switch [16 GT/s x16]
        │     └─ 0c:00.0 port -> gpu:0f AMD RADEON PRO V620 Azure gfx1030 HIP 7 smi 2 kfd-node 8 slot B [16 GT/s x16 31.5 GB/s raw]
        ├─ 02:08.0 switch B fabric port [16 GT/s x16]
        │  └─ 10:00.0 switch B sub-switch [16 GT/s x16]
        │     ├─ 11:00.0 port -> gpu:14 AMD RADEON PRO V620 Azure gfx1030 HIP 8 smi 3 kfd-node 9 slot B [16 GT/s x16 31.5 GB/s raw]
        │     └─ 11:10.0 port -> gpu:17 AMD RADEON PRO V620 Azure gfx1030 HIP 9 smi 4 kfd-node 10 slot B [16 GT/s x16 31.5 GB/s raw]
        ├─ 02:0c.0 switch B fabric port [16 GT/s x16]
        │  └─ 18:00.0 switch B sub-switch [16 GT/s x16]
        │     ├─ 19:14.0 downstream port [unused: down]
        │     └─ 19:15.0 downstream port [unused: down]
        └─ 02:1c.0 switch B fabric port [16 GT/s x16]
           └─ 1c:00.0 switch B management endpoint [16 GT/s x16]
```

Reading the tree:

- Two root complexes: `c0` carries switch A, `00` carries switch B. Each switch is one
  PEX88096 (`1000:c010`) behind a root port at 16 GT/s x16.
- Inside a switch: an upstream port, five *fabric* ports (virtual internal downstream
  ports), four sub-switches (virtual internal upstream ports) with the physical
  downstream ports below them, and the management endpoint behind the fifth fabric
  port (`e0:00.0`, `1c:00.0`). Fabric ports and sub-switches carry no lanes: the model
  marks those 10 links `virtual`; `report` hides them (`--virtual` shows them).
- 22 physical links: 2 host uplinks, and per GPU the switch-to-card link (`gpu_link`)
  and the on-card bridge link (`onboard`). All are 16 GT/s x16, 31.5 GB/s raw per
  direction, MPS 256 B.
- HIP order is the KFD node order: HIP 0 is the W7800 on bus `cb` (switch A), HIP 1 to 4
  the V620s of switch A (slot A), HIP 5 the W7800 on bus `07` (switch B), HIP 6 to 9
  the V620s of switch B (slot B).
- `dd:14.0`, `dd:15.0`, `19:14.0` and `19:15.0` have no device. They and the sub-switches
  `dc:00.0` and `18:00.0` stay in D3hot (runtime suspended); see "Never wake what
  sleeps" below.
- `Topology.path(a, b)` classifies a route `intra-switch` (one switch), `cross-root`
  (the GPUs hang off different root complexes) or `host`.

## Usage

The core is stdlib-only; `mcp` and `textual` are optional extras of
`tools/rdna/pyproject.toml`. On the box the editable install is `~/Apps/pciescope-venv`,
tracking the sparse clone `~/Apps/pciescope-src` (`git -C ~/Apps/pciescope-src pull
--ff-only` updates it). From a checkout: `PYTHONPATH=tools/rdna python -m pciescope ...`.

| Verb | What it does |
| --- | --- |
| `topo` | discover the fabric and print the tree; `--json` prints the whole model |
| `snap` | one pass of every passive collector: GPU and link tables, AER, KFD queues, cost per collector |
| `watch` | sample every `--interval` s (default 1.0) for `--duration` s or `--ticks`, record a session |
| `measure` | active probe suites (needs a GPU lease; `--dry-run` plans only, `--list-suites` lists) |
| `report` | summarise a session (`latest` by default): findings, links, GPUs, errors, events, probes, collector cost |
| `trace export` | session to a Perfetto / Chrome trace |
| `sessions`, `metrics` | list recorded sessions; print the metric table |
| `mcp`, `tui` | MCP stdio server; Textual UI (extras) |
| `plx` | PEX88096 register reader (own parser, see "Switch registers") |
| `capture` | developer: record sysfs for offline fixtures |

Every verb takes `--json` (machine output on stdout; logs and progress go to stderr),
`--sysfs-root`, `--proc-root` and `--host` (replay a captured tree), `--root` (session
store) and `-v`/`-q`. Exit codes: 0 ok, 1 failure, 2 usage error; `measure` passes the
probe's codes through (2 refused, e.g. no lease; 3 aborted by a guard; 4 ran but
invalid, see "Active probe"). Environment: `PCIESCOPE_HOME` (session store, default
`~/w4a8_runs/pciescope`), `PCIESCOPE_SYSFS`, `PCIESCOPE_PROC`, `PCIESCOPE_OWNER`
(lease owner for `measure`).

`watch` options: `--collectors` (comma list, `all` or `default` =
`sysfs_link,aer,amdgpu,kfd,plx`; a collector that cannot work here is skipped and the
reason logged, at INFO for the default set and WARNING when named), `--full` (disable
change-only dedup), `--heartbeat`, `--label`, `--no-record`, `--stream` (print the
records as JSONL), `--topology-from`, `--allow-switch-writes` (the only switch write path,
see "Risks and approvals").

```bash
ssh par1-cs25 '~/Apps/pciescope-venv/bin/python -m pciescope snap --json'
python -m pciescope watch --duration 60 --label myrun --json
python -m pciescope report latest            # or --json
python -m pciescope trace export myrun       # writes trace.perfetto.json
```

## Data model and metric definitions

Entity ids: `gpu:<bus>` (the GPU function, e.g. `gpu:ce`), `port:<BDF>` (any bridge, BDF
without the `0000:` domain), `link:<upper>-<lower>` (e.g. `link:c8:10.0-cc:00.0`, the
switch downstream port to the on-card bridge), `switch:A|B`, `root:<bus>`, `host`.
Timestamps are wall-clock nanoseconds.

`pciescope.model` holds the shared records: `Sample(ts_ns, entity, metric, value, unit,
tags)`, `Event(ts_ns, entity, name, severity, message, data)` and `Measurement(kind,
src, dst, direction, size_bytes, value, unit, stats, path_class, theoretical,
efficiency, ...)`, plus the `Collector` protocol every passive source implements:

```python
class Collector(Protocol):
    name: str
    def available(self) -> tuple[bool, str]: ...  # (usable here, reason when not)
    def collect(self) -> list[Sample]: ...        # one pass; target < 5 ms
```

Metrics of the core collectors (`pciescope metrics` prints the full table, including the
`plx.*` and `port.*` rows that the switch reader defines in "Switch registers"):

| Metric | Unit | Kind | Entity | Source | Meaning |
| --- | --- | --- | --- | --- | --- |
| `pcie.link_speed_gts` | GT/s | gauge | port or gpu | sysfs_link | current link speed of the device's own link end (`current_link_speed`) |
| `pcie.link_width` | lanes | gauge | port or gpu | sysfs_link | current negotiated width of the device's own link end |
| `pcie.asleep` | bool | gauge | port or gpu | sysfs_link | 1 while the parent bridge is runtime-suspended: link attributes are not read |
| `link.speed_gts` | GT/s | gauge | link | sysfs_link | link speed, minimum of the two ends |
| `link.width` | lanes | gauge | link | sysfs_link | link width, minimum of the two ends |
| `link.capacity_gbs` | GB/s | gauge | link | sysfs_link | raw payload capacity per direction at the current speed and width |
| `link.degraded` | bool | gauge | link | sysfs_link | 1 if below the link maximum or the two ends disagree |
| `aer.correctable` | count | counter | port or gpu | aer | `aer_dev_correctable` TOTAL_ERR_COR |
| `aer.nonfatal` | count | counter | port or gpu | aer | `aer_dev_nonfatal` total |
| `aer.fatal` | count | counter | port or gpu | aer | `aer_dev_fatal` total |
| `aer.error` | count | counter | port or gpu | aer | one non-zero AER error type; tags `class`, `type` |
| `aer.rootport_correctable` | count | counter | port | aer | root-port total of correctable errors from below |
| `aer.rootport_nonfatal` | count | counter | port | aer | root-port total of non-fatal errors from below |
| `aer.rootport_fatal` | count | counter | port | aer | root-port total of fatal errors from below |
| `gpu.asleep` | bool | gauge | gpu | amdgpu | 1 while the GPU is runtime-suspended: amdgpu sysfs is not read |
| `gpu.busy_pct` | % | gauge | gpu | amdgpu | `gpu_busy_percent` (graphics engine) |
| `gpu.mem_busy_pct` | % | gauge | gpu | amdgpu | `mem_busy_percent` (memory controller) |
| `gpu.power_w` | W | gauge | gpu | amdgpu | hwmon `power1_average` |
| `gpu.temp_edge_c` | C | gauge | gpu | amdgpu | hwmon `temp1` |
| `gpu.temp_junction_c` | C | gauge | gpu | amdgpu | hwmon `temp2`; the safety limit of the box rules |
| `gpu.temp_mem_c` | C | gauge | gpu | amdgpu | hwmon `temp3` |
| `gpu.vram_total_bytes`, `gpu.vram_used_bytes` | B | gauge | gpu | amdgpu | `mem_info_vram_*` |
| `gpu.gtt_used_bytes` | B | gauge | gpu | amdgpu | `mem_info_gtt_used` |
| `gpu.link_speed_gts`, `gpu.link_width` | GT/s, lanes | gauge | gpu | amdgpu | link state as the SMU (`gpu_metrics`) reports it |
| `gpu.pcie_dpm_speed_gts`, `gpu.pcie_dpm_width` | GT/s, lanes | gauge | gpu | amdgpu | active `pp_dpm_pcie` level |
| `gpu.throttle_status` | mask | gauge | gpu | amdgpu | SMU `throttle_status` bit mask (0 = not throttled); bit meaning is firmware specific |
| `gpu.pcie_rx_bytes_per_s`, `gpu.pcie_tx_bytes_per_s` | B/s | gauge | gpu | amdgpu | `pcie_bw`: opt-in, blocks about 1 s per read, not exposed by RDNA2/RDNA3 here |
| `kfd.evicted_ms` | ms | counter | gpu | kfd | per-process queue eviction time; tags `pid`, `comm`; a jump invalidates a measured window |
| `collector.cost_ms` | ms | gauge | host | runner | wall time of one `collect()`; tag `collector` |
| `collector.samples` | count | gauge | host | runner | samples one `collect()` produced; tag `collector` |

## Storage and traces

Every `watch` (and every `measure`) writes a session directory under
`$PCIESCOPE_HOME` (default `~/w4a8_runs/pciescope`):

```text
20261107-151404-core-60s-nowake/
  session.json          host, cmdline, interval, collectors, skipped, counts, status
  topology.json         the fabric model the run used (--topology-from replays it)
  trace.jsonl           one JSON record per line: sample | event | measurement
  trace.perfetto.json   written by `trace export`
```

Real records of that session:

```json
{"type":"event","ts_ns":1794060844867517850,"entity":"host","name":"watch_start","severity":"info","data":{"interval_s":1.0,"collectors":["sysfs_link","aer","amdgpu","kfd","plx"]}}
{"type":"sample","ts_ns":1794060844867567983,"entity":"port:00:01.1","metric":"pcie.link_speed_gts","value":16.0,"unit":"GT/s"}
```

Policy: a sample is written when its value changes, or when `--heartbeat` (30 s)
elapsed since the last write of that series; readers forward-fill (`report` weights its
statistics by time). The 60 s watch below saw 62,820 samples and wrote 3,426 (5.5 %).
`--full` writes everything. Events are derived in one place (`EventDetector`) so every
surface agrees: `link_change`, `link_degraded` / `link_restored`, `aer_delta`,
`thermal`, `evicted_jump`, `throttle`, `collector_error`, plus `watch_start` /
`watch_stop` and the probe's own events.

`trace export` writes Chrome trace-event JSON that Perfetto (<https://ui.perfetto.dev>,
"Open trace file") and `chrome://tracing` load: metadata (`M`), one counter track per
metric and entity (`C`), instants for events (`i`) and complete slices for probe
measurements (`X`). The 60 s session above exports to 140,559 bytes.

## MCP server

`python -m pciescope mcp` is a stdio MCP server (extra `mcp`; SDK 2.x `MCPServer`, with a
fallback to the 1.x `FastMCP`). Stdout carries only the protocol; logs go to stderr; a
tool error comes back as an `isError` result with the message (for example
`FileNotFoundError: session not found: ...`).

| Tool | Purpose |
| --- | --- |
| `topology` | the fabric model: summary, GPUs, links |
| `snapshot` | one passive pass, same document as `snap --json` |
| `start_watch`, `stop_watch`, `watch_status` | background recording; `start_watch` returns the session name at once, `stop_watch` finalises it and returns the findings; running watches are stopped when the server exits |
| `measure` | active probe (dry run by default; a real run needs `dry_run=false`, an `owner` and the GPU leases of the box rules); `list_suites`, `gpus`, `in_situ_s`, `interval_s`, `wait_quiet_s` |
| `report` | the `report` document, whole or by `sections`; `format="text"` for the readable form |
| `query_trace` | `kind` = `stats`, `samples` or `events` for a session, filtered by `metric` / `entity`, with `limit` and `tail` |
| `export_trace`, `sessions`, `metrics` | Perfetto export, session list, metric table |

From the Mac, the server runs on the box over ssh (nothing is installed locally):

```bash
ssh par1-cs25 ~/Apps/pciescope-venv/bin/python -m pciescope mcp
claude mcp add pciescope -- ssh par1-cs25 ~/Apps/pciescope-venv/bin/python -m pciescope mcp
```

```json
{"mcpServers": {"pciescope": {"command": "ssh", "args": ["-o", "BatchMode=yes", "par1-cs25", "~/Apps/pciescope-venv/bin/python", "-m", "pciescope", "mcp"]}}}
```

The ssh target is whatever reaches the box (key and user from `~/.ssh/config`). The
transcript of a Python SDK client doing exactly this is under "Evidence".

## Passive collectors and their cost

| Collector | Reads | Mean / p95 per pass: ten GPUs, 2026-11-07 | six GPUs, 2026-11-10 |
| --- | --- | --- | --- |
| `sysfs_link` | `current_link_speed` / `current_link_width` of every port and GPU, `power/runtime_status` of their parents | 4.3 / 6.6 ms | 4.0 / 4.3 ms |
| `aer` | `aer_dev_*` of every bridge and GPU, root-port totals (66 devices, 220 samples per pass; 41 devices, 143 since the move) | 5.0 / 6.8 ms | 5.5 / 6.0 ms |
| `amdgpu` | `gpu_metrics` (SMU), hwmon, `mem_info_*`, `gpu_busy_percent`, `pp_dpm_pcie` of every GPU | 21.9 / 28.3 ms (10 GPUs) | 7.8 / 10.8 ms (6 GPUs) |
| `kfd` | `/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms` | 0.7 / 1.0 ms | 1.2 / 1.4 ms |
| `plx` | PEX BAR0 registers of the switch through a long-lived `sudo -n` helper | 2.5 / 3.3 ms | 2.4 / 2.8 ms (272 samples; from `0a33892`, see below) |

The target was under 5 ms per pass; `report` raises an info finding `collector_cost` for
each collector above it. `amdgpu` misses it for a reason the collector cannot remove: a
`gpu_metrics` read asks the SMU firmware for a fresh table (median 0.6 ms, p90 6.7 ms,
max 16 ms per GPU; 3 to 6 ms for a busy V620, 0.3 to 0.8 ms for an idle GPU); the other
attributes are cheap only because the driver caches the metrics for 1 ms right after
that read. Ten GPUs add up to about 22 ms. If that matters, use a longer `--interval`
or `--collectors sysfs_link,aer,kfd,plx`. `aer` and `sysfs_link` sit at the edge of the
budget (mean 4.3 to 5.0 ms, p95 about 6.7 ms). Everything is one pass per `--interval`;
the 60 s watches had 0 overruns at 1 s. After the move `amdgpu` costs 1.3 ms per GPU
(2.2 ms before). The `plx` collector declined to run on the new chip at `fe57c33`: its
header check failed with `topology and BAR0 windows disagree: port 48 (c5:00.0)`,
because its port map was the old chips'; it refuses rather than guess, and `snap` /
`watch` log that at INFO and carry on with the sysfs collectors. Since PlxSwitch derived
the interim chip (`8b8b9c7`) it runs there: 272 samples per pass, 2.4 / 2.8 ms (60 s
watch of 2026-11-10 at `0a33892`).

### Never wake what sleeps

The unused branches (2026-11-07: `c6:0c.0`, `dc:00.0`, `dd:14.0`, `dd:15.0` and the same
on switch B; since the move: the cardless sub-switches `c7:00.0` and `e3:00.0` with their
ports `c8:10.0`, `c8:18.0`, `e4:14.0`, `e4:15.0`) are in D3hot (`power/runtime_status` =
`suspended`, `control` = `auto`). Reading a
config-backed sysfs attribute (`current_link_speed`, `secondary_bus_number`, ...) of a
device behind such a bridge makes the kernel resume the parent bridge first
(the kernel's `pci_config_pm_runtime_get` path [INFERENCE: from memory of the source;
the effect itself is measured]): each read then cost 11 to 35 ms, and polling at 1 Hz
kept the branch awake and flapping. First
60 s watch (session `20261107-145744-core-60s`): `sysfs_link` mean 50.3 ms, p95 51.8 ms.
After the fix (`20261107-151404-core-60s-nowake`): mean 4.3 ms, p95 6.6 ms.

The rule, for every collector and probe: read `power/runtime_status` first (reading it
does not wake anything) and skip a device that is not `active`/`unsupported`, or whose
parent is not. The skip is visible as `pcie.asleep` (ports behind a suspended bridge) and
`gpu.asleep` (a suspended GPU) = 1 and in `snap` (`asleep (runtime-suspended, attributes
not read): ports ...`). AER attribute reads (0.03 to 0.07 ms, `runtime_active_time`
unchanged) and the plx BAR0 reads do not wake anything. Discovery (`topo`, the start of
`snap`, `watch`, `tui`) still reads the link attributes once per process, which resumes
the idle sub-switch branches once; they suspend again by themselves, and the kernel logs
`pcieport ...: Configured ACS to 0x0000` for each resume (PlxSwitch observed this as
well). On 2026-11-10 all six V620 are in D0 and `snap` lists exactly those six
branch devices as asleep; the two W7800 of the 2026-11-07 layout (`cb`, `07`) had
`power/control` = `auto` and could have suspended when idle, which `gpu.asleep` shows.

## Report and the bandwidth ceilings

`pciescope report` (text, or `--json`) has: findings (severity `error` > `warn` >
`info`: `link_degraded`, `link_retrain`, `aer_increase`, `aer_nonzero_since_boot`,
`thermal`, `evicted_ms`, `probe_failed`, `collector_cost`, `collector_error`), the
physical links (speed and width min/max/last, time degraded, retrain count), GPUs
(busy, junction temperature, power, VRAM, eviction deltas), AER (counters that grew
during the session, non-zero totals since boot), notable events, collector cost and the
probe analysis.

Two ceilings are shown next to every rate (the report header and the JSON
`ceilings.link_classes` list them for each width of physical link in the topology: x16
and x8 on the interim box):

- raw: `speed x width x 128/130 / 8` = 16 GT/s x 16 x 128/130 / 8 = 31.508 GB/s per
  direction for Gen4 x16;
- practical: `raw x MPS / (MPS + 26) x 0.98` with the measured MPS 256 B = 28.03 GB/s
  (29.4 GB/s at MPS 512). 26 B is the per-TLP overhead at Gen3+ (4DW header 16, sequence
  2, LCRC 4, framing 4); 2 % is DLLP traffic (ACK/NAK, flow control). It is the best a
  stream of max-size posted writes can reach; reads sit lower (completion headers and
  the request rate).
  The switch counters (PlxSwitch, calibration on switch B) show one DW (4 B) more
  overhead on a TLP a GPU sends than on other TLPs (7 DW against 6 DW around a 256 B
  write), which fits ECRC generation [INFERENCE: the ECRC settings were read on the
  switch ports and bridges, not on the GPU functions]. ECRC adds 4 B to the 26 B above,
  so the ceiling would be about 27.6 GB/s at MPS 256: read 28.03 as an upper bound.
  For an x8 link the same formulas give 15.754 GB/s raw and 14.015 GB/s practical. The
  box rules quote about 13.9 GB/s measured (P2P and host to GPU, any x8 GPU link) and
  about 25 GB/s for all six GPUs through the one x16 root port: 99 % and 89 % of the
  practical ceilings, by the box rules' own figures, which this tool did not measure.

Per-link utilisation comes from the switch traffic counters when they run and
otherwise from probe measurements.

- Counters (`port.rx_gbs` / `port.tx_gbs`, or the rate of `port.rx_bytes` /
  `port.tx_bytes`; they need `--allow-switch-writes`): the `TRAFFIC` table of the
  report has one row per switch port end and direction (`rx` = received from the link
  partner, `tx` = sent) with the mean and peak GB/s of wire bytes (TLP + DLLP) as a
  share of the link's raw capacity, and the peak of the estimated payload
  (`plx.rx_payload_gbs` / `plx.tx_payload_gbs`) as a share of the practical ceiling.
  In JSON these are the `traffic_counters` rows of each link. The counters themselves
  were exercised on hardware by the switch reader (see "Switch registers"); the path
  through `watch --allow-switch-writes` and `report` is covered by unit tests on
  synthetic series only, not yet run on the box (it needs the approval below).
- Probes: the best single-flow bandwidth through a link and direction, as a share of
  raw and of practical. Patterns and concurrent flows are never charged to a link;
  their aggregate is not one flow's route.

Probe results are summarised per kind, direction, path class, method and pattern at one
headline size (the smallest for latency, the largest for bandwidth, concurrency and
all-to-all; `sizes` lists the others), and as a source-to-destination matrix of
kernel-store GPU pairs and host links (largest size, median of the repeats, one flow
per cell, pageable host memory left out; `size_bytes` gives the size behind each cell).

## AER on par1-cs25

Counters restart at every boot, so each figure below belongs to one boot.

Before the move (2026-11-07): 18 devices had a non-zero correctable counter since boot:
17 show one `NonFatalErr` (`c6:0c.0` two), the two root ports (`c0:03.1`, `00:01.1`)
show 10 each (`aer_rootport_total_err_cor`); every other counter, including all
uncorrectable and fatal ones, is zero. None increased during the 60 s watch (`report`:
"no counter increased during the session"). The switch registers agree (one correctable
event on `c5:00.0` and `c8:00.0`, see "Switch registers"); the `DevSta` correctable and
unsupported-request flags the plx collector reports on switch B's ports are sticky
status bits that only the OS error handler clears. Cause [INFERENCE]: a one-off at boot
or driver load; `aer_delta` events will show if it recurs.

After the move (2026-11-10, the box had been up for under an hour): again 18 non-zero
counters, all correctable. One `NonFatalErr` each on `c5:00.0`, `c6:04.0`, `c6:08.0`,
the leaf A sub-switch `cb:00.0` with its ports `cc:00.0`, `cc:08.0`, `cc:10.0` and
`cc:18.0`, the cards `d1` and `d4` with their on-card bridges `cf:00.0` and `d2:00.0`,
and the cardless `c7:00.0`, `c8:10.0` and `c8:18.0`; three on `c6:00.0` and five on
`c6:0c.0`; the root port `c0:03.1` shows 75 (10 before the move). Nothing under leaf B
(`d5:00.0`, `d6:xx.0`, the cards `d9`, `dc`, `df`, `e2`) or under `e3:00.0` is non-zero,
and no uncorrectable or fatal counter is. None increased during the 60 s watch.
Cause [INFERENCE]: events of the freshly re-cabled branches around boot; `aer_delta`
events show if it recurs. The pattern (leaf A and the cardless branch only) is worth a
glance if the counters start to move.

## Switch registers (PEX88096)

Section owner: PlxSwitch.

The GPUs of par1-cs25 hang off Broadcom PEX88096 switches (`1000:c010`,
ExpressFabric "Atlas", rev B0, subsystem `1000:a096`). Two arrangements are
described; the code derives the installed one from the topology and from the
serial number inside the chip, nothing is hardwired to a bus number:

- Interim, since the case move of 2026-11-08 (derived and read on 2026-11-10): ONE
  chip on a SlimSAS card, serial `00-80-5e-10-b7-3d-3b-62`, behind root port
  `c0:03.1` (upstream port `c5:00.0`, Gen4 x16), carrying the six V620 on Gen4 x8
  links; no W7800. The topology calls it switch A. The "top switch plus two leaf
  switches" of the box rules are the chip's own internal virtual switches (below).
- Pre-move (2026-11-07): two chips of the same model, A behind root `c0` (serial
  `00-80-5e-10-fa-53-6b-2a`, upstream `c5:00.0`) and B behind root `00` (serial
  `00-80-5e-10-f9-57-20-61`, upstream `01:00.0`), ten GPUs on x16 links. Not
  installed now; its measurements, its fixture and its approval history stay as
  the baseline.

This section is what the chips tell us without writing to them, how we read it,
what it costs, and the one thing that needs register writes (the traffic
counters). Code: `tools/rdna/pciescope/plx/` (decoders, root helper, CLI `python -m
pciescope.plx`) and `tools/rdna/pciescope/collectors/plx.py` (collector `plx`).
Everything marked [BOX] was measured on par1-cs25 (kernel
`7.1.7-100.p2p_post7.fc43`) with read-only BAR0 mappings; no switch or PCI
register was written. Facts without a date are from the pre-move chips
(2026-11-07); the interim chip decodes through the same code. [SDK] marks facts
taken from the Broadcom PLX SDK sources, [INFERENCE] anything not observed.

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
| Management endpoint (`1c:00.0` / `e0:00.0` pre-move, `e7:00.0` now; class `0x0180`, PEX port 247) | Not used. Memory decoding is disabled and there is no in-tree Linux driver for it (DeviceHunt lists `1000:00B2` as "PCIe Switch management endpoint"); reading it is pointless and unsafe to guess [BOX] |
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
mislabelling data (`capture` refuses to record one either). A "station" is
`port / 16`; the six stations 0..5 carry the PerformancePAK counters.

Interim chip (switch A, serial `00-80-5e-10-b7-3d-3b-62`, 2026-11-10; every GPU
link is Gen4 x8, the uplink x16) [BOX]:

| Port | Station | BDF | Device |
| --- | --- | --- | --- |
| 0 | 0.0 | `c5:00.0` (uplink, root port `c0:03.1`) | host |
| 16, 24 | 1.0, 1.8 | `c8:10.0`, `c8:18.0` | empty x8 slots (internal switch 0) |
| 32, 40 | 2.0, 2.8 | `cc:00.0`, `cc:08.0` | empty x8 slots (internal switch 1) |
| 48 | 3.0 | `cc:10.0` | `d1`: V620, HIP 0, smi 0, KFD node 1, slot A, lease `a2` |
| 56 | 3.8 | `cc:18.0` | `d4`: V620, HIP 1, smi 1, KFD node 2, slot A, lease `a2` |
| 64 | 4.0 | `d6:00.0` | `d9`: V620, HIP 2, smi 2, KFD node 3, slot B, lease `b4` |
| 72 | 4.8 | `d6:08.0` | `dc`: V620, HIP 3, smi 3, KFD node 4, slot B, lease `b4` |
| 80 | 5.0 | `d6:10.0` | `df`: V620, HIP 4, smi 4, KFD node 5, slot B, lease `b4` |
| 88 | 5.8 | `d6:18.0` | `e2`: V620, HIP 5, smi 5, KFD node 6, slot B, lease `b4` |
| 116, 117 | 7.4, 7.5 | `e4:14.0`, `e4:15.0` | empty x1 slots |
| 224..227 | none | `c6:00.0`, `c6:04.0`, `c6:08.0`, `c6:0c.0` | internal downstream ports |
| 232..235 | none | `c7:00.0`, `cb:00.0`, `d5:00.0`, `e3:00.0` | internal upstream ports |
| 255 | none | `c6:1c.0` | GEP, parent of the management endpoint |
| 247 | none | `e7:00.0` | management endpoint (never read) |

Pre-move chips (2026-11-07, two chips; ten x16 GPU links):

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
`.slot`; on the interim chip the slot follows the lease unit, leaf A or leaf B,
not the chip) and the lease key from the probe's table (`probe/hw.py`); the
`ports` verb prints them in the columns `hip smi slot lease gpu`, `ports --json`
and `status --json` carry `hip`, `smi`, `slot`, `lease`, `gpu_name`, and every
per-port sample is tagged `hip` (absent on non-GPU ports).

Rule: the uplink is the base port of the one station no leaf port lives in (48
on the two original cards, 0 on the SlimSAS card; if several stations are free
the original 48 wins when it is free, else the lowest; a wrong guess fails the
window check); the top-level downstream port by device number `0, 4, 8, 0xC` is
port `224..227` and device `0x1C` is the GEP `255`; a bridge below it is
internal upstream port `232 + device / 4`; a leaf downstream port is
`32 * j + device` for the `j`-th internal switch (`j` = 0 to 3) and its device
number on the bus. SDK special ports: `INT_DS` 224..227, `INT_UP` 232..235,
`MPT0` 247, `GEP_DS` 255.

Hierarchy inside each chip: root port, uplink, four internal downstream ports,
four internal upstream ports, then the leaf ports (the `j`-th internal switch
carries the leaf ports `32 * j` to `32 * j + 31`), then the on-card AMD bridges
(`1478`/`1479`) and the GPU. On the interim chip internal switch 1 is the "leaf A"
of the box rules (ports 48 and 56: HIP 0 and 1, lease `a2`; ports 32 and 40 are
empty) and internal switch 2 is "leaf B" (ports 64 to 88: HIP 2 to 5, lease `b4`);
internal switches 0 and 3 only carry empty slots. Consequence for peer-to-peer
measurements: HIP 0 and 1 share one internal switch, HIP 2 to 5 share another,
and a pair across the leaves has to go up through the top level of the chip. All
of it is inside one chip; none of it goes through the host root port [BOX
topology, INFERENCE for the latency consequence]. On the pre-move chips ports 0
and 16 shared one internal switch, ports 64 and 80 another and port 32 sat alone.

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
| `0x30C`, `0x310`, `0x314`, `0x318` | clock-enabled port masks, 32 ports each | pre-move `00010001`, `00010001`, `00010001`, `00000003` (ports 0, 16, 32, 48, 64, 80 and 116, 117); interim `01010001`, `01010101`, `01010101`, `00000003` (ports 0, 16, 24, 32, 40, 48 to 88 in steps of 8, 116, 117) |
| `0x350` | probe mode (bit 8 must be 0 for the counters) | `00040800` |
| `0x360` | virtual switch 0 upstream port | pre-move `00000030` (48), interim `00000000` (0) |
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

Approval is per chip. The user approved switch B of the pre-move layout
(2026-11-07, relayed by the lead): while it is idle (no process on HIP 5 to 9,
slot B lease held by the caller), never port 0 (HIP 5), health and register diff
after every run; it ran clean (below). That approval does not carry over: the
SlimSAS chip of 2026-11-10 is another chip (serial, port layout, x8 links) and
has none. `pciescope/plx/policy.py` therefore holds `DEFAULT_WRITE_SWITCHES =
()`; once the user approves, the environment variable
`PCIESCOPE_PLX_WRITE_SWITCHES=A` names the switch for one invocation. The
collector skips every switch that is not approved with a `plx_perf_skipped`
event, `perf run` refuses it (exit 2), and the helper maps the BAR0 of a chip
that was not named with `--write-chip BDF` `PROT_READ`, so a bug cannot write it
either.

What no write may touch is data of the chip, not code. `regs.KNOWN_CHIPS` lists
the chips by Device Serial Number (read from the port window), the BDF of their
upstream port where the entry was derived, and per chip the stations whose
windows no write may land in (`protected`: station to reason). The per-station
registers (`0x3E0`, `0x3F0`, `0x768`) live in the window of the station's base
port, so a protected station is not written at all. The helper maps a chip
writable only if its serial is listed AND it sits at the listed BDF: a replaced
card or a chip that moved (bus numbers change with the hardware) stays read-only
until someone re-derives its entry, instead of being written unprotected. Being
listed approves nothing; the policy does.

| Serial | Where | Letter | Protected stations |
| --- | --- | --- | --- |
| `00-80-5e-10-b7-3d-3b-62` | `0000:c5:00.0`, interim layout | A | none |
| `00-80-5e-10-fa-53-6b-2a` | `0000:c5:00.0`, pre-move | A | none |
| `00-80-5e-10-f9-57-20-61` | `0000:01:00.0`, pre-move | B | 0: the W7800 of HIP 5 (`gpu:07`, the lead only) hung on port 0 |

Nothing is protected on the interim chip. The only reason a station was ever
protected was the W7800 of HIP 5, which is not installed any more, and no device
behind the interim chip has a known hang. The fence for this chip is the policy
(no chip approved), the identity check (serial and BDF) and the idle gate: one
chip hosts all six GPUs and the host link, so the gate wants all six idle and
BOTH leases `a2` and `b4` held by the caller, and a hang of the chip would hit
every GPU at once. The protection itself stays generic and is enforced in three
places that do not trust each other: the client defaults (`NEVER_WRITTEN_GPUS` in
`plx/policy.py`, empty now, is merged into every exclusion list, so `--exclude`
and `perf_exclude` can only add to it); the collector, which does not start a
chip whose helper does not report the topology's ports of a never-written GPU
among its `protected_ports` (a stale table gives a `plx_perf_error`); and the
root helper itself, where the chip's `protected` stations make `perf_plan`,
`perf_start` and the single store path (`ChipAccess._write`) refuse every port
window of the station, whoever asks, before one dword is written. If a GPU of the
interim chip ever hangs on a port, its station goes into `KnownChip.protected` and
its `gpu:<bus>` into `NEVER_WRITTEN_GPUS`.

Writes per switch (BAR0 of the upstream port, window of the named port),
restricted to the ports being measured (default: every physical port of stations
0..5 with a device behind it, a GPU or the host, without the ports of
`NEVER_WRITTEN_GPUS` and `--exclude`; empty slots keep their stations untouched):

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

Bit 21 of `0xF30` can never be observed: it reads 0 before, during and after.
`perf run` therefore verifies the undo on what can be read: `0x768` and `0xF30`
must read as before, `0x3E0` as before or at the SDK stop value `0x40000000`,
and `0x3F0` (the capture strobe that every poll rewrites) is only listed.

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
estimate subtracts 28 bytes per posted TLP and 20 per completion (calibrated,
below; the SDK subtracts 20 for both). "rx" is traffic received by the switch
port from its link partner (GPU to switch on a leaf port, host to switch on the
uplink).

### Validation on the pre-move switch B and calibration (2026-11-07)

Run under the approval of that day on the idle pre-move switch B (no process on
HIP 5 to 9), slot B lease `PlxSwitch.slotB` held from 16:37 to 17:12 box time (CET),
counters on ports 16, 32, 48, 64 and 80, never port 0. A read-only snapshot (`lspci`
DevSta, the AER error status registers and LnkSta of the 35 functions below the
root port `00:01.1`, the kernel AER counters, the `dmesg` position) was taken
before and after every stage, next to the register readback of `perf run`.

| Stage | Ports | Time (box clock) | Result |
| --- | --- | --- | --- |
| 1a | 16 | 16:39, 60 s | clean |
| 1b | 16, 32, 64, 80 | 16:41, 60 s | clean |
| 1c, idle | 16, 32, 48, 64, 80 | 16:44, 300 s | clean |
| 2 | same five | 16:49:57, ended by the monitor after 19 s (exit 4) | `link_change` of port 0 (HIP 5): 2.5 GT/s x1 to 16 GT/s x16 |
| 2b | same five | 17:02:01 to 17:11:53 (SIGTERM, 1,169 samples) | clean; the probe's `quick` and `standard` suites ran inside |

Clean: exit 0, `undo_mismatch: []`, `health_changes: []`, `lspci` and both AER
counters unchanged, and no new `dmesg` line but `pcieport ...: Configured ACS to
0x0000` (a bridge of an idle branch that our own config reads resumed from runtime
suspend) and the `veth` lines of Docker. Stage 2 was not caused by the writes: a
foreign process (test workers of another agent without `HIP_VISIBLE_DEVICES`,
started 16:50:14) woke both W7800 (`dmesg`: PSP and SMU resume of `07:00.0` and
`cb:00.0`); the station of port 0 was never written and the links of ports 16 to
80 and AER did not change. A runtime-PM change of the HIP 5 port is since logged
by `perf run` and no longer ends the run ("Using it"); any other link change
does.

Registers, the same in every stage on ports 16 to 80, before, during, after:
`0x768` `09000000`, `29000000`, `09000000`; `0xF30` `00000000`, `00000040`,
`00000000`; `0x3E0` `00000000`, `98000000`, `00000000` (the reset bit of the
written `D8000000` clears itself); `0x3F0` `00000021` throughout. Bit 21 of
`0xF30` was written every time, as the SDK does, and reads 0 in every state, so a
readback cannot confirm it; nothing stalled with it set.

Idle (1c): every port counts 92.2 k DLLPs per second in each direction (flow
control updates, 0.74 MB/s, `port.rx_gbs` 0.0007). The GPU ports see only the
host's background: a posted 4 x 6 DW heartbeat every 5.06 s and bursts of reads
(0.01 to 0.03 MB/s), plus one 2.5 s burst of 13.9 k posted TLPs on each GPU port
at 16:41:37 of unknown origin. The uplink shows the helper's own polling: 11,200
one-DW reads per second (`nph` received = `cplh` sent), 0.35 MB/s of payload.

Calibration (stage 2b, the probe's `quick` suite, steps 1 to 5: copies of 64 MiB
between host and `gpu:0a`; uplink counters, 17:03:47.5 to 17:03:51.5).
`p2p_probe hbw` copies 3 times to warm up, then in batches of 16 timed copies,
and reads the buffer of a host-to-device transfer back once, so the bytes that
crossed the link are known: 505 host-to-GPU and 396 GPU-to-host copies of 64 MiB.

| Direction | Known bytes | Counters (model) | Difference |
| --- | --- | --- | --- |
| host to GPU: uplink ingress, completions | 33,889,976,320 | 33,958,399,568 (20 B per TLP) | +0.20 % |
| GPU to host: uplink egress, posted writes | 26,575,110,144 | 26,575,289,404 (28 B per TLP) | +0.0007 % |

The DW counters include the TLP header, the LCRC, one framing DW and the ECRC
digest when the sender generates one. Measured DW per TLP:

| TLP | Payload | Counted DW | Overhead |
| --- | --- | --- | --- |
| posted write of a GPU to host memory (peer pushes too) | 256 B | 71 | 7 DW |
| non-posted read of a GPU | none | 7 | 7 DW |
| completion of the host to a GPU (copies to the GPU) | up to 256 B | 60.7 mean (pinned), 67.8 mean (pageable) | 5 DW |
| completion of a GPU to a GPU (peer reads) | 256 B | 70 | 6 DW |
| register read of the helper (host to the switch BAR0) | none | 5 | 5 DW |
| its completion, and a one-DW posted write of the host | 4 B | 6 | 5 DW |

A GPU's TLPs carry one DW more than the host's, which fits ECRC generation
(`ECRCGenEn+` on the downstream ports of the switch and on the bridges of the
cards, `ECRCGenEn-` on the root port and the uplink; the GPU functions were not
read [INFERENCE]). Every read of a GPU gets exactly one completion (`nph` =
`cplh` on all ports). The model is therefore 28 B for posted writes and 20 B for
completions; a completion of a GPU counts 24 B, so the payload estimate of peer
reads is 1.6 % high at 256 B per TLP, and the payload of tiny TLPs is not
meaningful. With the previous 24 B the GPU-to-host figure was 1.6 % high. Under
load the uplink ingress carried 1.6 DLLPs per TLP (acks and flow-control
updates): 225 M DLLPs, 1.80 GB, 4.9 % of the 36.8 GB of TLP bytes of the window,
which `plx.*_util` (total bytes including DLLPs over the raw rate) includes.

Not validated (pre-move): peer-to-peer bytes against the counters of ports 16 and
32 (the probe does not report the copy count of those steps); counters sampled
faster than 0.5 s for per-step bytes; the live `watch --collectors plx
--allow-switch-writes` and the TUI GB/s on the box.

### After the hardware move: the interim chip (2026-11-10)

Derived and checked read-only on 2026-11-10 (no lease needed for reads; the
CPU-only environment `HIP_VISIBLE_DEVICES=` kept every tool off the GPUs; HxServer
took both leases during the session). What changed against the pre-move chips:

- One chip, not two: the serial is identical on all 23 functions
  (`00-80-5e-10-b7-3d-3b-62`; the old chips were `...fa-53-6b-2a` and
  `...f9-57-20-61`); the management endpoint is `e7:00.0` (PEX port 247).
- The uplink is PEX port 0 (LnkCap "Port #0", `0x360` reads 0), not 48. The code
  derives it as the free station (`ports.uplink_port`) and verifies it against the
  window headers; `capture` and every root verb refuse a map that disagrees.
- The GPUs are on x8 ports (48, 56 | 64, 72, 80, 88), two per station. The second
  port of a station is "station port 8", which no counter test could exercise on
  the pre-move chips. The empty x8 slots (16, 24, 32, 40) and the empty x1 slots
  (116, 117) are left out of the counter plan.
- Nothing is approved and nothing is protected for this chip (`regs.KNOWN_CHIPS`,
  `plx/policy.py`, reasons above). A chip with another serial or at another BDF
  stays read-only.
- Runtime PM: all six V620 have `power/control` = `on` and `runtime_status` =
  `active`; the switch's downstream ports are `auto` but active. `perf run`
  excuses the link changes of any port whose GPU is `auto` (box rules: monitors
  must treat runtime-PM link transitions as non-errors).

Read-only results on the real chip (`ports`, `status`, `dump 0 --decode`, `perf
plan --switch A` and `capture`): `status` samples 272 registers and ports in
1.35 ms; the link speed and width of all 22 bridge functions equal sysfs
`current_link_speed` and `current_link_width`; no sticky error bit is set on any
function (DevSta, AER status, lane errors), MPS is 256, MRRS 128, ASPM off and ACS
control 0 everywhere, equalization is complete on every linked port; the vendor
registers read `0x30C..0x318` as in the table above, `0x360` = 0, and every
counter control is off at the station base ports 0, 16, 32, 48, 64 and 80 (`0x3E0`
= 0, `0x3F0` = `0x21`, `0x768` = `09000000`, `0xF30` = 0); `perf plan` lists
ports 0, 48, 56, 64, 72, 80, 88 (stations 0, 3, 4, 5; the stations of the empty
slots stay untouched). The capture is `fixtures/plx_par1_cs25_interim.json.gz`.

Staged request (not run; needs a new user approval for serial
`00-80-5e-10-b7-3d-3b-62`): `PCIESCOPE_PLX_WRITE_SWITCHES=A` for the invocation
only; `perf plan` shown first; `perf run --require-idle --owner PlxSwitch` with
BOTH leases `PlxSwitch.a2` and `PlxSwitch.b4` held (all six GPUs idle); the
read-only health diff after every stage; at the first AER, DPC, link retrain, GPU
fault or stalled transfer: stop, undo, report.

| Stage | Ports | Time | What it checks |
| --- | --- | --- | --- |
| 1a | 48, 56 (leaf A, lease `a2`) | 60 s | undo registers read back; port 56 (station port 8) counts on its own slot; idle DLLP rate about 92 k per second per direction and port, as on the pre-move chip |
| 1b | 48, 56, 64, 72, 80, 88 (every GPU port) | 60 s | the same on all three stations |
| 1c | the six GPU ports, then the uplink (0) last | 300 s idle | the uplink shows the helper's own polling [INFERENCE: about 11 k one-DW reads per second, as before]; the GPU ports only the host's background |
| 2 | the same | the probe's `quick`, then `standard`, inside the window | calibration: known bytes against the counters |

Undo values per station: `0x768` = `09000000`, `0xF30` = 0, `0x3E0` = 0. To be
verified in stage 1a, not assumed: that the counters of the second port of a
station (56, 72, 88) sit at counter RAM slot `port % 16` as the SDK layout says
[SDK], and the TLP overhead (28 B posted, 20 B completion, ECRC dependent) at MPS
256 on x8 links.

### Using it

```bash
cd ~/Apps/pciescope-src   # PYTHONPATH=tools/rdna, venv ~/Apps/pciescope-venv
python -m pciescope.plx ports            # port <-> BDF <-> GPU <-> lease map, no root
python -m pciescope.plx status [--json]  # one sample of every port (sudo -n)
python -m pciescope.plx dump 48 --decode
python -m pciescope.plx perf plan --ports 48,56   # the writes above, nothing written
python -m pciescope.plx perf preflight --switch A --owner PlxSwitch   # reads only
python -m pciescope.plx capture --out fixtures/plx_par1_cs25_interim.json.gz
# gated: one switch (an approved one; none is for the interim chip yet), optional
# staged --ports, idle gate, raw log
PCIESCOPE_PLX_WRITE_SWITCHES=A python -m pciescope.plx perf run --switch A \
    --ports 48,56 --allow-switch-writes --require-idle --owner PlxSwitch \
    --duration 60 --raw raw.jsonl
```

`perf run` takes one switch and writes only the approved ones. It runs the idle
gate (`--require-idle`, exit 3), records a read-only health baseline, enables the
counters, prints per-port rates, ends on `--duration`, Ctrl-C or SIGTERM or on
the first anomaly (a `link_change`, `link_degraded`, `dpc_triggered`,
`plx_perf_error` or a new `plx_error_flags` at warn or error: exit 4), stops the
counters and prints the registers before, during and after plus every difference
of the link, error and kernel AER state (exit 5 if the undo or the health
differs). A link change of a port whose GPU may be runtime-suspended
(`power/control` is `auto`; the W7800 of HIP 5 did it on the pre-move box: the
link drops to 2.5 GT/s x1 or returns to 16 GT/s x16 and stays up) is printed,
kept under `runtime_pm` in the verdict and does not end the run; every other link
change does. `--raw FILE` writes one JSON line per
capture with the cumulative counters of every port, for calibration.

Every verb takes `--json`, `--switch A|B` and `--fixture FILE` (replay a
capture instead of touching hardware). The collector is `plx` in the registry
(`--collectors plx`); `--allow-switch-writes` is the collector option
`allow_writes`, confined to the approved switches (`write_switches`; options
`perf_ports` and `switches` narrow the ports and the chips). Without root or
passwordless `sudo` the collector is skipped with the reason.

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
`plx_perf_skipped`, `plx_perf_error`. Virtual fan-out ports only report the
error masks and settings; the traffic counters of ports 116 and 117 (station 7)
do not exist.

### Measured on par1-cs25 (read-only)

Interim chip, 2026-11-10 (box up since 08:38):

```text
$ python -m pciescope.plx ports      # excerpt
switch A  root c0  upstream c5:00.0 (port 0)  serial ?
  port stn bdf      kind     peer     hip smi slot lease    gpu
     0 0.0 c5:00.0  usp      host       -   -    - -
    16 1.0 c8:10.0  dsp                 -   -    - -
    48 3.0 cc:10.0  dsp      gpu:d1     0   0    A a2       AMD RADEON PRO V620 Azure
    64 4.0 d6:00.0  dsp      gpu:d9     2   2    B b4       AMD RADEON PRO V620 Azure
    88 5.8 d6:18.0  dsp      gpu:e2     5   5    B b4       AMD RADEON PRO V620 Azure
   116 7.4 e4:14.0  dsp                 -   -    - -
   224   - c6:00.0  int_dsp             -   -    - -

$ python -m pciescope.plx status     # excerpt
switch A  upstream c5:00.0  serial 00-80-5e-10-b7-3d-3b-62  traffic counters off
  port bdf      kind     peer     hip link            eq  flags
     0 c5:00.0  usp      host       - 16 GT/s x16     ok  -
    16 c8:10.0  dsp                 - down            -   -
    48 cc:10.0  dsp      gpu:d1     0 16 GT/s x8      ok  -
    88 d6:18.0  dsp      gpu:e2     5 16 GT/s x8      ok  -
   255 c6:1c.0  gep_dsp             - virtual         -   -
272 samples in 1.35 ms
```

All six GPU-facing links run 16 GT/s x8 and the uplink 16 GT/s x16, equalization
complete, no degradation and no sticky error bit anywhere; ports 16, 24, 32, 40,
116 and 117 have no link. (`ports` prints `serial ?` because the serial lives in
extended config space, which only root can read; `status` reads it through BAR0.)

Pre-move chips, 2026-11-07:

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

What the numbers say (pre-move chips): all 10 GPU-facing links and both uplinks run
16 GT/s x16 with equalization complete and no degradation; ports 116 and 117 are
empty. The
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
decoders against `lspci` output of all 36 functions of the pre-move chips, the
port map, the helper's read policy and write refusal on a fake `resource0` and
through the real helper process, the exact write sequence and the 32-bit unwrap
on a stateful fake of the counter RAM, the collector's metrics, events and
gating, and the CLI. `test_pciescope_plx_interim.py` covers the interim chip: the
port map and the uplink rule, the chip registry and the policy data, the verbs
over its capture, the counters over the fake (only the stations of ports with a
device are written, the second port of a station has its own slot, the x8 link
capacity), `perf run` refusal and staging, and the idle gate and lease keys for
the six GPUs; the older tests run with the policy data of 2026-11-07
(`plx_fakes.use_pre_move_policy`). Fixtures: `fixtures/plx_par1_cs25.json.gz`
(pre-move) and `fixtures/plx_par1_cs25_interim.json.gz` (topology plus the
non-zero dwords of every port window, read-only captures),
`fixtures/plx_lspci_expect.json`. What no test can show: real counter values,
which need the write approval, and the counters of the second port of a station.

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
place (the legacy modes `info`, `lat`, `latc`, `bw` and `a2a` keep their source;
what is verified about their output is in "Tests and what is not verified").
Entry points: `python -m pciescope measure` (CLI,
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

- **Lease.** A GPU is only touched if the caller holds its lease
  (`~/w4a8_runs/lanes/lease/<owner>.slotA|slotB|w7800cb`, owner from `--owner`
  or `$PCIESCOPE_OWNER`). Another holder of the same key refuses as well. HIP 5
  (W7800 bus 07) is never runnable without a lead-created `<owner>.w7800-07`.
  The refusal names the file that is needed. Nothing is launched before the
  lease check passes.
- **Quiet box.** No run while `docker buildx build` runs, `boot.lock` is held,
  or another agent's quiet file is newer than 6 min. `--wait-quiet SECONDS`
  waits, `--force-noisy` runs anyway and marks every result INVALID (exit 4).
  Lease and VRAM checks are never bypassed.
- **VRAM.** 3 GiB free per GPU (1 GiB for host-only plans); a GPU shared with
  a running server is refused. A dry run does not read the counters of a
  runtime-suspended GPU (reading wakes it and its bridge): such GPUs are
  listed in `asleep`.
- **Junction temperature** (hwmon `temp2_input` of the amdgpu device), sampled
  during the run: abort when the 5 s mean is at least 90 C or any sample at
  least 95 C (status `aborted`, exit 3, partial results kept).
- **Time.** Per run `3 x estimate + 60 s` (at least 120 s, at most 1 h, or
  `--max-seconds`), per step a limit from its estimate; the device spin loops
  of the new kernels have their own deadline (`P2P_KERNEL_TIMEOUT_S`, 10 s).
- **Evictions.** KFD `evicted_ms` of the probed GPUs is watched; a jump marks
  the window INVALID and records the process that caused it.
- **Links.** The best PCIe link state seen on the probed links during the run
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

### Measured on par1-cs25 (slot B, 2026-11-07, before the hardware move)

Run on slot B (HIP 6 to 9, buses `0a 0f 14 17`) inside the 16:50 to 17:05 window of
the PlxSwitch lease, owner `PlxSwitch`, binary `~/Apps/pciescope/bin/p2p_probe`
(fat gfx1030 + gfx1100, `pciescope-probe/1`, `tick_mhz` 99.985 on all four):

```bash
python -m pciescope measure --suite quick    --gpus B --owner PlxSwitch --wait-quiet 240 --json
python -m pciescope measure --suite standard --gpus B --owner PlxSwitch --wait-quiet 240 --json
```

Clean runs (exit 0, no flagged step): session `20261107-165742-measure-quick`, 16 steps in
10.5 s (planned 17.3 s), and `20261107-165801-measure-standard`, 101 steps, 342 records
in 70.6 s (planned 144 s). Standard, median of three repeats, `verify=ok` on every row;
audit values from `pcie-iommu-audit-2026-11-06.md`:

| Measurement (0a to 0f unless noted) | Audit | Measured |
|---|---|---|
| push 1 MiB | 24.6 GB/s | 24.55 GB/s |
| pull 1 MiB | 24.4 GB/s | 24.40 GB/s |
| SDMA 1 MiB / 64 MiB | ~20 / 25.5 to 26 GB/s | 19.90 / 25.47 GB/s |
| push / pull 64 MiB | - | 25.45 / 25.42 GB/s |
| ring of 4 (1 MiB) | 105.1 GB/s | 104.65 GB/s |
| fan-in, 3 GPUs into 0a (1 MiB) | - | 25.60 GB/s |
| one-way flag latency, 0 B (p50) | 6.1 to 6.3 us | 6.14 us |
| push 4 KiB then flag (p50) | - | 6.52 us |
| H2D / D2H pinned 64 MiB, GPU 0a | - | 26.54 / 28.05 GB/s |
| H2D / D2H pageable 64 MiB, GPU 0a | - | 27.14 / 7.25 GB/s |
| 4 GPUs H2D at once / D2H at once | - | 28.03 / 28.20 GB/s aggregate |
| H2D + D2H pinned at once, GPU 0a | - | 49.70 GB/s |
| doorbell round trip CPU to GPU 0a | - | 2.73 us |

The four GPUs of a switch share one x16 uplink: 28 GB/s aggregate for four GPUs is the
same 28 GB/s as for one. Pageable D2H is four times slower than every other host copy
[INFERENCE: the staging copy of `hipMemcpy` into pageable memory; not isolated].
Cross-root and slot A rows need the slot A lease and were not run.

The first two runs of the window (16:50, same traffic, sessions
`20261107-165018-measure-quick` and `20261107-165029-measure-standard`) ended INVALID
(exit 4) for two reasons that were guard faults, now fixed: the probe's own pid books
10 to 17 ms of `evicted_ms` on every GPU in its first step (first use of devices and
buffers), now tagged `setup_evicted_ms` and not invalidating; and python workers of
another job counted as "new KFD process on the probed GPUs" although every KFD process
has counters for every GPU, now only a process with a queue on a probed GPU counts. A
jump of any other process's `evicted_ms`, or of the probe's own after step 1, still
invalidates the step.

### Resume after the hardware move

State when the box was drained (2026-11-07 17:1x): nothing of the probe runs on the
box; no lease, quiet file or request file of PcieProbe exists (every run used the
PlxSwitch `slotB` lease); no build is running; `~/Apps/pciescope/bin/p2p_probe` and
its stamp stay (fat gfx1030 + gfx1100, independent of bus numbers; rebuild only if
`p2p_probe.cpp` changes); the checkout `~/Apps/pciescope-src` is clean at `b007218`;
everything is pushed to `feat/pciescope`.

Everything measured above is before the move and belongs to the old wiring. Keep it as
the baseline and redo it: slot B quick and standard; slot A and the cross-root rows
(never run, need the slot A lease); `--in-situ` inside a PlxSwitch counter window
(never run with counters on); the legacy-mode output diff against the audit binary on
a GPU (about 10 s; the note in "Tests and what is not verified" stays [INFERENCE]
until then). Verdicts against the audit baselines only mean something if the new
wiring keeps the audit's paths.

Pre-move sessions under `~/w4a8_runs/pciescope/` (JSON next to them, `probe_*.json`):
`20261107-165742-measure-quick` and `20261107-165801-measure-standard` (clean, the
table above); `20261107-170255-measure-quick` and `20261107-170358-measure-standard`
(clean, same plan, PLX counters on, PlxSwitch window 17:02:01 to 17:17:01);
`20261107-165018-measure-quick` and `20261107-165029-measure-standard` (flagged
INVALID by the two guard faults described above, same traffic, keep only as a record).

Steps, in order, once the lead announces the re-map:

1. `git pull --ff-only` in `~/Apps/pciescope-src`; `python -m pciescope topo` shows the
   new buses, switches and HIP order.
2. Edit `probe/hw.py`, the one place with bus literals: the `GpuSpec` table (bus, arch,
   root, switch, lease key, `hip_hint`), `LEASE_BUSES`, `LEAD_ONLY_KEYS` (today
   `w7800-07`, bus 07 = HIP 5) and `ROOTS`; then `gpus.py` (the `w7800cb` token
   resolves bus `cb`) and the bus 07 texts in `lease.py`, following the lead's new
   lease recipe. Re-capture the fixtures with Core's `capture` if the tests are to
   follow the new layout.
3. `python -m pciescope.probe status`, then `measure --suite quick --gpus <slot>
   --dry-run --json`: no GPU is woken, the blockers are listed. A real run verifies the
   HIP-to-bus order with `p2p_probe buses` and refuses on a mismatch.
4. With a lease and a window: quick, then standard, per slot; compare with the table
   above.

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
no number in these tests is a measurement of the new modes.

Evidence from the box (2026-11-07, build 1046 s, almost all of it waiting for
the build lock): the installed `~/Apps/pciescope/bin/p2p_probe` was built from
the pushed `p2p_probe.cpp` (stamp `p2p_probe.stamp.json`: source sha256
`b9f27e29...`, equal to the file in git) and `llvm-objdump --offloading` lists
the bundles `hipv4-amdgcn-amd-amdhsa--gfx1030` and `--gfx1100` next to the host
code, so the binary is fat. `p2p_probe version` prints
`pciescope-probe/1` with all twelve modes. CPU-only comparison with the audit
binary (`~/w4a8_runs/pcie_audit/p2p/p2p_probe`, no GPU visible): `info`, `lat`,
`bw` and `a2a` fail identically at `hipGetDeviceCount` ("no ROCm-capable device
is detected", the `file:line` prefix differs because the audit binary was built
from `p2p_probe.hip`), the bare call is identical, and an unknown command now
exits 1 without initialising HIP first. [INFERENCE] The legacy modes keep their
output format: the legacy source region is unchanged in the diff and was
reviewed, but their output has not been compared with the audit binary on a
GPU (needs a leased window).

Not yet run on a GPU: the new modes `hbw`, `hlat`, `hflag`, `curve` and
`batch`, the clock calibration (`tick_mhz`) and the multi-block `curve`
protocol.

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
The TUI is passive: its collectors are the default passive set, that is sysfs and
procfs reads plus the `plx` collector's read-only BAR0 register reads through the
`sudo -n` helper (skipped where unreadable; `--collectors` narrows it). It writes
nothing to the switches or devices; the only thing that can touch a GPU is the
measure panel, and only under a lease held by `--owner`.

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

- RX/TX GB/s is `n/a` when no throughput counter exists. The amdgpu `pcie_bw`
  sysfs file is not exposed on RDNA2/RDNA3 (neither the V620 nor the W7800 on
  this box has it), and the PLX per-port counters are off until the switch
  register writes are approved (collector `plx`, see the switch section). The
  Live view says so and falls back to the GPU busy percentage for the history.
- The Matrix is empty until a session holds probe measurements: run the
  measure panel or load a session that has them.
- Errors are changes, not levels: a counter that is non-zero but constant is
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

All of it ran on par1-cs25, passive only (sysfs reads and the read-only switch register
reads; no GPU work, no register written), from the sparse clone `~/Apps/pciescope-src`
at `feat/pciescope`. The first group (up to "After the case move") ran on 2026-11-07 on
the ten-GPU layout (`6011071`); the last group ran on 2026-11-10 on the interim layout
(`bee4bfd`).

### `snap`

```text
par1-cs25 snapshot 2026-11-07 15:14:03  collectors: sysfs_link aer amdgpu kfd plx  (1047 samples; cost sysfs_link 4.971 ms, aer 4.723 ms, amdgpu 43.296 ms, kfd 0.841 ms, plx 2.496 ms)
gpu     hip  smi  arch     sw  slot  busy%  Tj C  W   VRAM GiB  GPU link    card link
------  ---  ---  -------  --  ----  -----  ----  --  --------  ----------  ----------
gpu:cb  0    5    gfx1100  A   -     3      49    20  4.0/45    16GT/s x16  16GT/s x16
gpu:ce  1    6    gfx1030  A   A     0      43    7   0.0/30    16GT/s x16  16GT/s x16
gpu:d3  2    7    gfx1030  A   A     0      41    6   0.0/30    16GT/s x16  16GT/s x16
gpu:d8  3    8    gfx1030  A   A     0      40    9   0.0/30    16GT/s x16  16GT/s x16
gpu:db  4    9    gfx1030  A   A     0      41    7   0.0/30    16GT/s x16  16GT/s x16
gpu:07  5    0    gfx1100  B   -     0      47    20  0.0/45    16GT/s x16  16GT/s x16
gpu:0a  6    1    gfx1030  B   B     0      61    8   26.3/30   16GT/s x16  16GT/s x16
gpu:0f  7    2    gfx1030  B   B     0      60    8   26.3/30   16GT/s x16  16GT/s x16
gpu:14  8    3    gfx1030  B   B     0      56    9   26.3/30   16GT/s x16  16GT/s x16
gpu:17  9    4    gfx1030  B   B     0      58    7   26.3/30   16GT/s x16  16GT/s x16

links: 22 physical, 22 at their maximum
asleep (runtime-suspended, attributes not read): ports 18:00.0 19:14.0 19:15.0 dc:00.0 dd:14.0 dd:15.0; GPUs -
AER: 18 non-zero counter(s) since boot (66 devices checked)
```

The 43 ms `amdgpu` cost is the first, cold pass of a new process; repeated passes cost
17 to 24 ms. The `snap --json` document has the same data plus `cost_ms`, `asleep`,
`aer`, `links`, `events` and the raw `other_metrics` counts (the 20 `plx.*` metrics).

### 60 s `watch`

```text
$ python -m pciescope watch --duration 60 --label core-60s-nowake --json
{"ticks":60,"samples_seen":62820,"samples_written":3426,"events":14,"duration_s":59.035,"overruns":0,"collector_cost_ms":{...}}
pciescope: WARNING: [warn] throttle gpu:cb mask 0x1000000000
```

60 ticks at 1 s, 0 overruns; 62,820 samples seen, 3,426 written by the change-only
store. The session is 524 KB (`trace.jsonl` 472,806 bytes).

### `report` (same session)

```text
pciescope report: 20261107-151404-core-60s-nowake  host par1-cs25  59.056 s  status ok
collectors: sysfs_link, aer, amdgpu, kfd, plx
records: {'samples': 3426, 'series': 1047, 'measurements': 0, 'events': 16}
ceilings: Gen4 x16 raw 31.508 GB/s, practical 28.031 GB/s (MPS 256 B)

FINDINGS
  [info] aer_nonzero_since_boot: 18 AER counter(s) are non-zero since boot (correctable, rootport_correctable); none of them increased during this session
  [info] collector_cost: collector aer: p95 6.826 ms per collect() exceeds the 5 ms budget
  [info] collector_cost: collector amdgpu: p95 28.256 ms per collect() exceeds the 5 ms budget
  [info] collector_cost: collector sysfs_link: p95 6.613 ms per collect() exceeds the 5 ms budget

LINKS (physical)
link                  kind         owner   max    GT/s  width  cap GB/s  chg  degr s
--------------------  -----------  ------  -----  ----  -----  --------  ---  ------
link:c0:03.1-c5:00.0  host_uplink  A       16x16  16    16     31.508    0    0
link:00:01.1-01:00.0  host_uplink  B       16x16  16    16     31.508    0    0
link:c8:00.0-c9:00.0  gpu_link     gpu:cb  16x16  16    16     31.508    0    0
link:ca:00.0-cb:00.0  onboard      gpu:cb  16x16  16    16     31.508    0    0
  ... 18 more rows, all 16x16, 0 changes, 0 s degraded
  (no traffic counters in this session: utilisation below comes from probe measurements only)

GPUS
gpu     hip  arch     slot  busy% mean/max  Tj C mean/max  W mean/max  VRAM GiB  GPU-side link
------  ---  -------  ----  --------------  -------------  ----------  --------  -------------
gpu:cb  0    gfx1100  -     3.73/8          48.3/49        22.27/26    4.00      16x16
gpu:ce  1    gfx1030  A     0/0             42.41/43       7/7         0.02      16x16
  ...
gpu:0a  6    gfx1030  B     97.32/99        75.29/79       171.33/178  26.31     16x16
gpu:0f  7    gfx1030  B     97.32/99        75.19/80       171.44/180  26.32     16x16
gpu:14  8    gfx1030  B     97.32/99        71.66/76       171.46/177  26.32     16x16
gpu:17  9    gfx1030  B     97.32/99        73.73/77       171.39/177  26.29     16x16

ERRORS (AER)
  no counter increased during the session
  non-zero totals since boot:
    gpu:cb         correctable          1  {'NonFatalErr': 1}  Radeon Pro W7800 48GB
    port:00:01.1   rootport_correctable 10  root port
    port:c6:0c.0   correctable          2  {'NonFatalErr': 2}  switch A fabric port
    ... (18 rows in all)

COLLECTOR COST (per collect())
collector   calls  samples  mean ms  p95 ms  max ms
----------  -----  -------  -------  ------  ------
aer         60     220      4.975    6.826   6.976
amdgpu      60     150      21.87    28.256  40.72
kfd         58     5        0.73     1.01    1.064
plx         60     382      2.489    3.332   8.516
sysfs_link  60     280      4.346    6.613   7.146

EVENTS {'watch_start': 1, 'plx_error_flags': 5, 'throttle': 9, 'watch_stop': 1}
  +    1.01s [warn] throttle gpu:0a mask 0x1
  +   21.01s [warn] throttle gpu:cb mask 0x1000000000
```

During the window the four V620s of switch B (`0a`, `0f`, `14`, `17`) ran a vLLM serve
at 97 % busy and about 171 W each (other agents' work, not ours); every link stayed at
16 GT/s x16 and no AER counter moved. The `throttle` events are the SMU throttle masks
turning non-zero under that load (`0x1` on the V620s) and on the idle W7800 `cb`
(`0x1000000000`); pciescope reports the mask only, bit meaning is firmware specific.

### `trace export`

```text
$ python -m pciescope trace export 20261107-151404-core-60s-nowake --json
{"session":"20261107-151404-core-60s-nowake","path":"/home/chenco_adm/w4a8_runs/pciescope/20261107-151404-core-60s-nowake/trace.perfetto.json","bytes":140559,"viewer":"https://ui.perfetto.dev (Open trace file)"}
```

### Switch reader through the CLI

`python -m pciescope plx status` (read-only BAR0 reads through the sudo helper, 382
samples in 3.0 ms):

```text
switch A  upstream c5:00.0  serial 00-80-5e-10-fa-53-6b-2a  traffic counters off
  port bdf      kind     peer     hip link            eq  flags
     0 c8:00.0  dsp      gpu:cb     0 16 GT/s x16     ok  -
    16 c8:10.0  dsp      gpu:ce     1 16 GT/s x16     ok  -
    ...
   116 dd:14.0  dsp                 - down            -   -
   117 dd:15.0  dsp                 - down            -   -
```

### MCP over ssh, from the Mac

A Python SDK client (`mcp` 2.3.0, `stdio_client` with `command="ssh"`) talking to
`ssh chenco_adm@par1-cs25 ~/Apps/pciescope-venv/bin/python -m pciescope mcp`:

```text
server: pciescope
tools: ['topology', 'snapshot', 'start_watch', 'stop_watch', 'watch_status', 'measure', 'report', 'query_trace', 'sessions', 'export_trace', 'metrics']
topology: isError False gpus 10 first {'id': 'gpu:cb', 'hip': 0, 'smi': 5, 'kfd_node': 1, 'arch': 'gfx1100', 'switch': 'A', 'slot': None, 'name': 'Radeon Pro W7800 48GB'}
snapshot: gpus 10 links 22 aer nonzero 18 asleep ['18:00.0', '19:14.0', '19:15.0', 'dc:00.0', 'dd:14.0', 'dd:15.0'] cost_ms {'sysfs_link': 3.983, 'aer': 4.829, 'amdgpu': 18.966, 'kfd': 0.872, 'plx': 2.434}
start_watch: 20261107-151531-mcp-ssh 20261107-151531-mcp-ssh
watch_status: {"watches": [{"collectors": ["sysfs_link", "aer", "amdgpu", "kfd", "plx"], "skipped": {}, "session": "20261107-151531-mcp-ssh", ..., "running": true, "ticks": 8, "samples_written": 1200, "events": 5, "interval_s": 0.5}]}
stop_watch: ticks 9 samples_written 1222 findings ['aer_nonzero_since_boot', 'collector_cost', 'collector_cost', 'collector_cost']
report keys: ['session', 'ceilings', 'collector_cost']
report text head: pciescope report: 20261107-151531-mcp-ssh  host par1-cs25  4.044 s  status ok
query stats: 1 [{"entity": "gpu:0a", "metric": "gpu.temp_junction_c", "n": 8, "first": 77.0, "last": 77.0, "min": 76.0, "max": 78.0, "mean": 76.8753}]
events: ['watch_start', 'plx_error_flags', 'plx_error_flags', 'plx_error_flags', 'plx_error_flags', 'plx_error_flags', 'watch_stop']
measure list_suites: ['quick', 'standard', 'full']
measure dry-run: dry-run 0 ok or dry-run ['gpu:ce'] ['GPU lease check REFUSED (owner: none): ...
sessions: ['20261107-151531-mcp-ssh', '20261107-151404-core-60s-nowake', '20261107-145744-core-60s', '20261107-144343-tui']
bad session isError: True Error executing tool report: FileNotFoundError: session not found: no-such-session (looked in /home/chenco_adm/w4a8_runs/pciescope)
```

The `measure` dry run plans `quick` on `gpu:ce` and reports the lease refusal as a
blocker (no owner given, slot A held by another agent's file); it touches no GPU.

### After the case move (2026-11-10, six V620 behind one switch)

Same rules, from `bee4bfd`, while HxServer's soak kept HIP 1 to 5 busy. `snap` (the `plx`
collector declines on the new chip, see "Passive collectors and their cost"):

```text
par1-cs25 snapshot 2026-11-10 09:18:03  collectors: sysfs_link aer amdgpu kfd  (412 samples; cost sysfs_link 4.38 ms, aer 6.036 ms, amdgpu 5.715 ms, kfd 1.47 ms)
  skipped plx: AccessError: switch A: topology and BAR0 windows disagree: port 48 (c5:00.0): buses (pri,sec,sub) (204, 207, 209) != topology (197, 198, 231)
gpu     hip  smi  arch     sw  slot  busy%  Tj C  W   VRAM GiB  GPU link   card link
------  ---  ---  -------  --  ----  -----  ----  --  --------  ---------  ---------
gpu:d1  0    0    gfx1030  A   A     0      31    7   0.0/30    16GT/s x8  16GT/s x8
gpu:d4  1    1    gfx1030  A   A     65     50    52  0.4/30    16GT/s x8  16GT/s x8
gpu:d9  2    2    gfx1030  A   B     97     50    47  0.3/30    16GT/s x8  16GT/s x8
gpu:dc  3    3    gfx1030  A   B     97     50    55  0.3/30    16GT/s x8  16GT/s x8
gpu:df  4    4    gfx1030  A   B     97     51    47  0.3/30    16GT/s x8  16GT/s x8
gpu:e2  5    5    gfx1030  A   B     96     50    46  0.3/30    16GT/s x8  16GT/s x8

links: 13 physical, 13 at their maximum
asleep (runtime-suspended, attributes not read): ports c7:00.0 c8:10.0 c8:18.0 e3:00.0 e4:14.0 e4:15.0; GPUs -
AER: 18 non-zero counter(s) since boot (41 devices checked)
```

`sw` is the chip, `slot` the lease slot of the documented layout (A = HIP 0 and 1, B = HIP
2 to 5). `watch --duration 60 --label core-interim-60s --json` ended with 60 ticks,
24,720 samples seen, 1,649 written, 0 events and 0 overruns in 59.0 s. `report` on that
session:

```text
pciescope report: 20261110-091820-core-interim-60s  host par1-cs25  59.022 s  status ok
collectors: sysfs_link, aer, amdgpu, kfd  skipped: {'plx': 'AccessError: switch A: topology and BAR0 windows disagree: ...'}
records: {'samples': 1649, 'series': 412, 'measurements': 0, 'events': 2}
ceilings per direction (MPS 256 B): 16 GT/s x16 raw 31.508 / practical 28.031 GB/s (7 links); 16 GT/s x8 raw 15.754 / practical 14.015 GB/s (6 links)

FINDINGS
  [info] aer_nonzero_since_boot: 18 AER counter(s) are non-zero since boot (correctable, rootport_correctable); none of them increased during this session
  [info] collector_cost: collector aer: p95 6.036 ms per collect() exceeds the 5 ms budget
  [info] collector_cost: collector amdgpu: p95 10.78 ms per collect() exceeds the 5 ms budget

LINKS (physical)
link                  kind         owner   max    GT/s  width  cap GB/s  chg  degr s
--------------------  -----------  ------  -----  ----  -----  --------  ---  ------
link:c0:03.1-c5:00.0  host_uplink  A       16x16  16    16     31.508    0    0
link:cc:10.0-cf:00.0  gpu_link     gpu:d1  16x8   16    8      15.754    0    0
link:d0:00.0-d1:00.0  onboard      gpu:d1  16x16  16    16     31.508    0    0
...                                                      (the other five cards alike)
  (no traffic counters in this session: utilisation below comes from probe measurements only)

GPUS
gpu     hip  arch     slot  busy% mean/max  Tj C mean/max  W mean/max  VRAM GiB  GPU-side link
------  ---  -------  ----  --------------  -------------  ----------  --------  -------------
gpu:d1  0    gfx1030  A     0/0             30.97/31       6.47/7      0.02      16x8
gpu:d4  1    gfx1030  A     63.57/78        49.8/51        52/52       0.35      16x8
gpu:d9  2    gfx1030  B     96.8/98         50.27/51       47.14/48    0.30      16x8
gpu:dc  3    gfx1030  B     96.78/98        49.37/50       55/55       0.30      16x8
gpu:df  4    gfx1030  B     96.75/98        51.24/52       47.49/48    0.30      16x8
gpu:e2  5    gfx1030  B     96.76/98        50.03/51       45.19/46    0.30      16x8

COLLECTOR COST (per collect())
collector   calls  samples  mean ms  p95 ms  max ms
----------  -----  -------  -------  ------  ------
aer         60     143      5.547    6.036   6.119
amdgpu      60     90       7.832    10.78   11.047
kfd         59     5        1.206    1.359   1.482
sysfs_link  60     166      3.98     4.289   4.478
```

(The ERRORS section lists the same 18 counters as the AER section above.) `trace export`
wrote `trace.perfetto.json` (62,321 bytes) next to the session. MCP over ssh, same
client as above (11 tools; the `topology` and `snapshot` calls shown):

```text
topology: isError False gpus 6 first {'id': 'gpu:d1', 'hip': 0, 'smi': 0, 'kfd_node': 1, 'arch': 'gfx1030', 'switch': 'A', 'slot': 'A', 'name': 'AMD RADEON PRO V620 Azure'}
snapshot: gpus 6 links 13 aer nonzero 18 asleep ['c7:00.0', 'c8:10.0', 'c8:18.0', 'e3:00.0', 'e4:14.0', 'e4:15.0'] cost_ms {'sysfs_link': 4.387, 'aer': 6.03, 'amdgpu': 7.992, 'kfd': 1.482}
stop_watch: ticks 9 samples_written 565 findings ['aer_nonzero_since_boot', 'collector_cost', 'collector_cost']
```

At `bee4bfd` the `measure` dry run on the interim box was refused for every GPU, with a
lease or not: `measure --suite quick --gpus hip:2 hip:3 --owner nobody --dry-run` gave
`unknown bus 'd9': not a GPU of this box (known buses: 07 0a 0f 14 17 cb ce d3 d8 db),
refused`, because the probe's lease table (`probe/hw.py`) still listed the ten old buses.

At `0a33892` (PlxSwitch's interim derivation and PcieTui's preset change on top) the
table knows the six buses, and the same dry run reaches the lease check and is refused
there, as it should be without a window: `b4 (buses d9,dc): held by HxServer (file
HxServer.b4, 08:56:11, pid 23277 dead, stale?); you need nobody.b4; ask the lead for a
window on b4; do not create the holder file yourself unless the lead granted it`. The
`plx` collector runs: `snap` lists it among the collectors (686 samples) and `plx ports`
maps the six V620 to ports 48 to 88 with the leases `a2` / `b4`. A 60 s `watch`
(`20261110-093024-core-interim-plx-60s`) ended with 60 ticks, 41,160 samples seen, 2,341
written, 0 events and 0 overruns; cost per pass (mean / p95 ms): `plx` 2.43 / 2.76 (272
samples), `sysfs_link` 4.05 / 4.45, `aer` 5.69 / 6.00, `amdgpu` 8.27 / 12.47, `kfd`
1.24 / 1.31. No traffic counters ran: they need `--allow-switch-writes`.

### Tests

```bash
cd ~/Apps/pciescope-src
~/Apps/pciescope-venv/bin/python -m pytest tests/tools/rdna/pciescope
```

on the box (`0a33892`, 2026-11-10): 852 passed in 90 s (core, switch reader, probe and
TUI; 732 passed at `ccd9e6c` on 2026-11-07). The
`pytest.ini` in that directory makes it the rootdir and the conftest cut-off, so the
repository's `tests/conftest.py` (it imports vLLM and `tblib`) is not loaded and the
repository's pytest options do not apply; it also turns the cache provider off. The
`conftest.py` there puts pytest's temp directories under `~/w4a8_runs/pciescope/pytest`
(`PCIESCOPE_PYTEST_TMP` overrides; the box rules forbid `/tmp`). No GPU is needed; the
topology tests run on tarballs of the real sysfs of this box
(`fixtures/sysfs_par1_cs25.tar.gz` of 2026-11-07, `fixtures/sysfs_par1_cs25_interim.tar.gz`
of 2026-11-10; both made with `pciescope capture`).

A `measure` run that is refused (no lease, bad GPU list) or fails before it starts
writes nothing: the session directory is created by the first record of a run that
really starts (`store.LazySession`). On the box, `measure --suite quick --gpus hip:2
--owner nobody` prints the lease refusal, exits 2 and leaves the session store as it was
(41 directories before and after).

## Risks and approvals

- `measure` touches the GPUs, so it needs the leases of the box rules. Since the
  2026-11-08 move the lease slots are `a2` (HIP 0 and 1) and `b4` (HIP 2 to 5), holder
  files `~/w4a8_runs/lanes/lease/<owner>.a2` / `<owner>.b4`. When checked (2026-11-10
  about 09:30) the only holder files were `HxServer.a2` and `HxServer.b4`, written at
  08:56; the probe reports their pid as dead (stale?), which only the lease owners can
  settle. No W7800 is installed. Nothing in this document used a GPU; a real run needs a
  window granted by the lead (owner name, slot, suite).
- Two tables outside the core described the old box, and the box rules say to re-derive
  them before switch writes or probes: the probe's lease table (`probe/hw.py`) and the
  register reader's `KNOWN_CHIPS` / `PROTECTED_STATIONS`. Both know the interim box at
  `0a33892` (the dry run reaches the lease check, `plx` runs read-only; owners PcieProbe
  and PlxSwitch, see their sections). Whether switch counter writes are approved for the
  interim chip is PlxSwitch's gate ("Switch registers"). The core needs no such table
  beyond `topology.DOCUMENTED_LAYOUTS`; when the replacement switch arrives, add its
  layout there, or `topo` falls back to the switch letter as slot and says so.
- Per-link traffic utilisation needs the switch PerformancePAK counters, which are off
  at boot. Enabling them is a switch register write, so it is gated behind
  `watch --allow-switch-writes` (registers and gate: "Switch registers"). It was not
  used here, hence "no traffic counters in this session".
- The `amdgpu` collector costs about 22 ms per pass with ten GPUs and 8 ms with six (SMU
  latency) and `aer` / `sysfs_link` about 5 ms; keep `--interval` at 1 s or more, and
  leave `amdgpu` out (`--collectors sysfs_link,aer,kfd,plx`) while a latency probe runs
  on the same box.
- Discovery wakes the idle sub-switch branches once per process (see "Never wake what
  sleeps"); the collectors after it never do.
- `plx` needs passwordless `sudo` for its helper; without it the collector is skipped
  with the reason, and `topo`, `snap` and `watch` work on sysfs alone.
